"""Persisted approval and agent-routing coordinator for the local bridge."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Protocol
from uuid import UUID

from agent_bridge.adapters.base import AgentRunResult, FableAdapter, SolAdapter
from agent_bridge.adapters.claude_cli import ClaudeRunError
from agent_bridge.adapters.codex_cli import CodexRunError
from agent_bridge.contracts import (
    FableClarification,
    ReviewVerdict,
    SolOutcome,
    TaskBrief,
)
from agent_bridge.process import ProcessRunner
from agent_bridge.repository import (
    RepositoryTracker,
    WorkspaceBaseline,
    WorkspaceDelta,
    validate_allowed_path,
)
from agent_bridge.state_machine import TaskState
from agent_bridge.store import SQLiteStore, TaskRecord


class IdFactory(Protocol):
    def new_task_id(self) -> str:
        raise NotImplementedError

    def new_run_id(self) -> str:
        raise NotImplementedError


_ACTIVE_STATES = frozenset({
    TaskState.FABLE_PLANNING,
    TaskState.SOL_RUNNING,
    TaskState.FABLE_CLARIFYING,
    TaskState.FABLE_REVIEWING,
    TaskState.SOL_CORRECTING,
})
_SOL_STATES = frozenset({TaskState.SOL_RUNNING, TaskState.SOL_CORRECTING})
_SOL_EVENT_TYPES = frozenset({
    "item.started", "item.updated", "item.completed", "thread.started",
})
_SOL_ITEM_TYPES = frozenset({
    "agent_message", "command_execution", "file_change", "plan", "todo_list",
})
_SOL_ITEM_STATUSES = frozenset({
    "completed", "declined", "failed", "in_progress", "interrupted",
})
_FABLE_EVENT_TYPES = frozenset({
    "assistant", "result", "stream_event", "system", "user",
})
_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_MAX_AUDIT_COUNT = 2**63 - 1
_MIN_EXIT_CODE = -(2**31)
_MAX_EXIT_CODE = 2**31 - 1


class Coordinator:
    """Own task transitions, exact continuations, and completion evidence."""

    def __init__(
        self,
        *,
        store: SQLiteStore,
        repository: RepositoryTracker,
        runner: ProcessRunner,
        fable: FableAdapter,
        sol: SolAdapter,
        ids: IdFactory,
        repo_root: str | Path,
        repo_context: str,
        trusted_shells: Mapping[str, str | Path],
    ) -> None:
        self._store = store
        self._repository = repository
        self._runner = runner
        self._fable = fable
        self._sol = sol
        self._ids = ids
        self._repo_root = Path(repo_root).resolve()
        if not self._repo_root.is_dir():
            raise ValueError("repo_root must be an existing directory")
        if not isinstance(repo_context, str) or not repo_context.strip():
            raise ValueError("repo_context must be non-empty")
        self._repo_context_text = repo_context
        self._python_executable = Path(sys.executable).resolve(strict=True)
        if not isinstance(trusted_shells, Mapping) or not trusted_shells:
            raise ValueError("trusted_shells must be a non-empty mapping")
        resolved_shells: dict[str, Path] = {}
        for name, configured in trusted_shells.items():
            if name not in {"bash", "sh"}:
                raise ValueError("trusted shell names must be bash or sh")
            path = Path(configured)
            if not path.is_absolute() or not path.is_file():
                raise ValueError("trusted shell paths must be absolute files")
            resolved_shells[name] = path.resolve(strict=True)
        self._trusted_shells = resolved_shells
        self._writing_lock = asyncio.Lock()
        self._run_completions: dict[str, asyncio.Event] = {}

    async def handle_user_request(self, session_id: str, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        task_id = self._ids.new_task_id()
        self._store.create_planning_task(session_id, task_id)
        self._store.append_event(
            session_id, task_id, "user", "message", {"text": text}
        )
        await self._run_planning(
            self._store.get_task(task_id, 0), text, resume_session_id=None
        )
        return task_id

    async def approve_task(self, task_id: str, revision: int) -> None:
        task = self._latest_required(task_id)
        if revision != task.revision:
            raise ValueError("approval revision must match the latest exact revision")
        if task.state not in {
            TaskState.AWAITING_USER_APPROVAL,
            TaskState.AWAITING_SCOPE_APPROVAL,
        }:
            raise ValueError("task is not awaiting revision approval")
        if task.brief is None:
            raise RuntimeError("approved task revision is missing its brief")
        if task.brief.open_questions:
            raise ValueError("open_questions must be resolved before approval")

        async with self._writing_lock:
            task = self._latest_required(task_id)
            if task.revision != revision or task.state not in {
                TaskState.AWAITING_USER_APPROVAL,
                TaskState.AWAITING_SCOPE_APPROVAL,
            }:
                raise RuntimeError("task approval changed while waiting for the writer lock")
            if task.brief is None or task.brief.open_questions:
                raise RuntimeError("task is no longer eligible for approval")
            if task.baseline_id is None:
                baseline = self._repository.capture(task.brief)
                try:
                    setting = (
                        self._baseline_key(task.task_id, task.revision),
                        self._baseline_setting_value(
                            task.task_id, task.revision, baseline
                        ),
                    )
                    task = self._store.approve_task_with_setting(
                        task.task_id,
                        task.revision,
                        brief=task.brief,
                        baseline_id=baseline.baseline_id,
                        expected=task.state,
                        setting=setting,
                    )
                except BaseException:
                    self._repository.discard_baseline(baseline)
                    raise
            else:
                baseline = self._load_baseline(task)
                if baseline.allowed_paths != task.brief.allowed_paths:
                    raise RuntimeError(
                        "approved baseline scope does not match the task revision"
                    )
                task = self._store.approve_task(
                    task.task_id,
                    task.revision,
                    baseline_id=baseline.baseline_id,
                    expected=task.state,
                )
            if task.continuation_state is not None:
                pending = task.pending
                if pending is None or not isinstance(pending.get("answer"), str):
                    raise RuntimeError("scope revision is missing its continuation answer")
                task = self._store.resume_continuation(
                    task.task_id,
                    task.revision,
                    expected=task.state,
                )
            else:
                pending = None
                task = self._store.transition_task(
                    task.task_id,
                    task.revision,
                    expected=TaskState.AWAITING_USER_APPROVAL,
                    target=TaskState.SOL_RUNNING,
                )
            self._emit_state(task)
            if pending is not None:
                routed = await self._invoke_sol_locked(
                    task,
                    resume_prompt=self._scope_resume_prompt(
                        task, str(pending["answer"])
                    ),
                    context=None,
                )
            else:
                routed = await self._invoke_sol_locked(
                    task,
                    resume_prompt=None,
                    context=(
                        f"{self._repo_context()}\n"
                        f"Approved baseline: {task.baseline_id}"
                    ),
                )
        if routed is not None:
            await self._route_sol_outcome(
                routed[0], routed[1], routed[2]
            )

    async def edit_task(
        self, task_id: str, brief: TaskBrief | Mapping[str, object],
    ) -> None:
        task = self._latest_required(task_id)
        if task.state not in {
            TaskState.AWAITING_USER_APPROVAL,
            TaskState.AWAITING_SCOPE_APPROVAL,
        }:
            raise ValueError("only an unapproved task may be edited")
        edited = brief if isinstance(brief, TaskBrief) else TaskBrief.from_dict(brief)
        if edited.task_id != task.task_id or edited.revision != task.revision + 1:
            raise ValueError("edited task must use the same task_id and next revision")
        for path in edited.allowed_paths:
            validate_allowed_path(path)
        if task.fable_session_id is None:
            raise RuntimeError("edited task is missing its exact Fable session")
        original_baseline: WorkspaceBaseline | None = None
        widened: WorkspaceBaseline | None = None
        if task.baseline_id is not None:
            original_baseline = self._load_baseline(task)
            widened = self._repository.widen_baseline(
                original_baseline, edited
            )
        try:
            setting = None if widened is None else (
                self._baseline_key(task.task_id, edited.revision),
                self._baseline_setting_value(
                    task.task_id, edited.revision, widened
                ),
            )
            saved = self._store.save_edited_revision(
                task.session_id,
                edited,
                fable_session_id=task.fable_session_id,
                sol_thread_id=task.sol_thread_id,
                baseline_id=task.baseline_id,
                correction_count=task.correction_count,
                continuation_state=task.continuation_state,
                pending=task.pending,
                setting=setting,
            )
        except BaseException:
            if original_baseline is not None and widened is not None:
                self._repository.discard_widening(original_baseline, widened)
            raise
        self._store.append_event(
            saved.session_id,
            saved.task_id,
            "coordinator",
            "task_brief",
            {"brief": edited.to_dict()},
        )

    async def reject_task(self, task_id: str) -> None:
        task = self._latest_required(task_id)
        if task.state not in {
            TaskState.AWAITING_USER_APPROVAL,
            TaskState.AWAITING_SCOPE_APPROVAL,
        }:
            raise ValueError("task is not awaiting approval")
        task = self._store.transition_task(
            task.task_id,
            task.revision,
            expected=task.state,
            target=TaskState.FAILED,
        )
        self._store.append_event(
            task.session_id,
            task.task_id,
            "user",
            "task_rejected",
            {"revision": task.revision},
        )

    async def answer_user_question(self, task_id: str, answer: str) -> None:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be non-empty")
        task = self._latest_required(task_id)
        if task.state is not TaskState.AWAITING_USER_INPUT:
            raise ValueError("task is not awaiting user input")
        continuation = task.continuation_state
        pending = task.pending
        if continuation is None or pending is None:
            raise RuntimeError("awaiting-user task has no persisted continuation")
        task = self._store.resume_continuation(
            task.task_id, task.revision, expected=TaskState.AWAITING_USER_INPUT
        )
        self._store.append_event(
            task.session_id, task.task_id, "user", "message", {"text": answer}
        )
        self._emit_state(task)
        if continuation in _SOL_STATES:
            await self._resume_sol(task, answer)
            return
        if continuation is TaskState.FABLE_REVIEWING:
            review_prompt = pending.get("review_prompt")
            completion_allowed = pending.get("completion_allowed")
            if not isinstance(review_prompt, str) or not isinstance(completion_allowed, bool):
                raise RuntimeError("review continuation context is invalid")
            await self._call_fable_review(
                task,
                f"{review_prompt}\nUser answer: {answer}",
                completion_allowed=completion_allowed,
            )
            return
        raise RuntimeError("unsupported user-answer continuation")

    async def stop_task(self, task_id: str) -> None:
        task = self._latest_required(task_id)
        if task.state not in _ACTIVE_STATES:
            raise ValueError("task has no active phase to stop")
        active = self._store.active_run_for_task(task.task_id, task.revision)
        if active is None:
            interrupted = self._store.mark_interrupted(
                task.task_id,
                task.revision,
                continuation=task.state,
            )
            self._emit_state(interrupted)
            return
        if active.task_id != task.task_id or active.revision != task.revision:
            raise RuntimeError("active run does not belong to the exact task revision")
        completion = self._run_completions.get(active.run_id)
        if completion is None:
            raise RuntimeError("active run has no local completion tracker")
        interrupted = self._store.mark_interrupted(
            task.task_id,
            task.revision,
            continuation=task.state,
        )
        self._emit_state(interrupted)
        try:
            await self._runner.stop(active.run_id)
        except BaseException:
            self._finish_interrupted_run(active.run_id, exit_code=-1)
            self._store.append_event(
                interrupted.session_id,
                interrupted.task_id,
                "coordinator",
                "stop_error",
                {"run_id": active.run_id},
            )
            raise
        await completion.wait()

    async def resume_task(self, task_id: str) -> None:
        task = self._latest_required(task_id)
        if task.state is not TaskState.INTERRUPTED:
            raise ValueError("only an interrupted task may be resumed")
        continuation = task.continuation_state
        pending = task.pending
        if continuation is None:
            raise RuntimeError("interrupted task has no persisted continuation")
        if continuation is not TaskState.FABLE_PLANNING:
            baseline = self._load_baseline(task)
            delta = self._repository.compare(baseline)
            self._store.append_event(
                task.session_id,
                task.task_id,
                "coordinator",
                "resume_drift",
                self._delta_summary(delta),
            )
            if delta.unexpected_paths or delta.protected_changed_paths:
                failed = self._store.transition_task(
                    task.task_id,
                    task.revision,
                    expected=TaskState.INTERRUPTED,
                    target=TaskState.FAILED,
                )
                self._emit_state(failed)
                return
        if continuation is TaskState.FABLE_CLARIFYING:
            if pending is None:
                raise RuntimeError("clarification continuation context is missing")
            underlying_raw = pending.get("underlying_continuation")
            try:
                underlying = TaskState(str(underlying_raw))
            except ValueError as error:
                raise RuntimeError(
                    "clarification underlying continuation is invalid"
                ) from error
            task = self._store.resume_nested_continuation(
                task.task_id,
                task.revision,
                active_state=TaskState.FABLE_CLARIFYING,
                continuation_state=underlying,
                pending=pending,
            )
        else:
            task = self._store.resume_continuation(
                task.task_id, task.revision, expected=TaskState.INTERRUPTED
            )
        self._emit_state(task)
        if continuation is TaskState.FABLE_PLANNING:
            original = self._original_user_message(task.session_id, task.task_id)
            await self._run_planning(
                task,
                original,
                resume_session_id=task.fable_session_id,
            )
        elif continuation in _SOL_STATES:
            if task.sol_thread_id is None and continuation is TaskState.SOL_RUNNING:
                await self._start_sol(task)
            else:
                if task.sol_thread_id is None:
                    raise RuntimeError("Sol correction cannot resume without its exact thread")
                prompt = "Resume the explicitly user-approved interrupted task."
                if pending is not None and isinstance(pending.get("prompt"), str):
                    prompt = str(pending["prompt"])
                await self._resume_sol(task, prompt)
        elif continuation is TaskState.FABLE_CLARIFYING:
            if pending is None or not isinstance(pending.get("clarification_prompt"), str):
                raise RuntimeError("clarification continuation context is invalid")
            await self._call_fable_clarification(
                task, str(pending["clarification_prompt"])
            )
        elif continuation is TaskState.FABLE_REVIEWING:
            if pending is None or not isinstance(pending.get("review_prompt"), str):
                raise RuntimeError("review continuation context is invalid")
            allowed = pending.get("completion_allowed")
            if not isinstance(allowed, bool):
                raise RuntimeError("review completion guard is invalid")
            await self._call_fable_review(
                task, str(pending["review_prompt"]), completion_allowed=allowed
            )
        else:
            raise RuntimeError("unsupported interrupted continuation")

    async def _run_planning(
        self,
        task: TaskRecord,
        prompt: str,
        *,
        resume_session_id: str | None,
    ) -> None:
        run_id = self._ids.new_run_id()
        self._store.start_agent_run(run_id, task.task_id, task.revision, "fable")
        completion = self._track_run(run_id)
        try:
            if resume_session_id is None:
                result = await self._fable.plan(
                    run_id=run_id,
                    task_id=task.task_id,
                    prompt=prompt,
                    context=self._repo_context(),
                )
            else:
                result = await self._fable.resume_plan(
                    run_id=run_id,
                    session_id=resume_session_id,
                    task_id=task.task_id,
                    prompt=prompt,
                    context=self._repo_context(),
                )
            self._persist_agent_result(
                task,
                "fable",
                result,
                run_id=run_id,
                expected_cli_session_id=resume_session_id,
            )
            if (
                resume_session_id is not None
                and result.cli_session_id is not None
                and result.cli_session_id != resume_session_id
            ):
                raise RuntimeError("Fable resumed a different session than requested")
            if result.cli_session_id is not None:
                task = self._store.set_fable_session(
                    task.task_id, task.revision, result.cli_session_id
                )
            if result.interrupted:
                self._finish_interrupted_run(run_id, exit_code=result.exit_code)
                current = self._store.get_task(task.task_id, task.revision)
                if current.state is not TaskState.INTERRUPTED:
                    current = self._store.mark_interrupted(
                        current.task_id,
                        current.revision,
                        continuation=TaskState.FABLE_PLANNING,
                        cli_session_id=result.cli_session_id,
                    )
                    self._emit_state(current)
                return
            if result.payload is None or result.cli_session_id is None:
                raise RuntimeError("Fable planning completed without a contract or session ID")
            brief = TaskBrief.from_dict(result.payload)
            if brief.task_id != task.task_id or brief.revision != 1:
                raise ValueError("Fable returned the wrong task identity")
            self._store.finish_agent_run(run_id, status="completed", exit_code=result.exit_code)
            saved = self._store.save_task(
                task.session_id, brief, TaskState.AWAITING_USER_APPROVAL
            )
            saved = self._store.set_fable_session(
                saved.task_id, saved.revision, result.cli_session_id
            )
            self._store.append_event(
                saved.session_id,
                saved.task_id,
                "fable",
                "task_brief",
                {"brief": brief.to_dict()},
            )
        except asyncio.CancelledError:
            self._interrupt_if_active(
                run_id, task.task_id, task.revision, TaskState.FABLE_PLANNING
            )
            raise
        except BaseException as error:
            self._record_agent_failure(
                task,
                "fable",
                run_id,
                error,
                expected_cli_session_id=resume_session_id,
            )
            raise
        finally:
            self._complete_run(run_id, completion)

    async def _start_sol(self, task: TaskRecord) -> None:
        if task.state is not TaskState.SOL_RUNNING:
            raise RuntimeError("Sol start requires persisted SOL_RUNNING state")
        if task.approved_at is None or task.baseline_id is None or task.brief is None:
            raise RuntimeError("Sol start requires persisted approval and baseline")
        await self._invoke_sol(
            task,
            resume_prompt=None,
            context=f"{self._repo_context()}\nApproved baseline: {task.baseline_id}",
        )

    async def _resume_sol(self, task: TaskRecord, prompt: str) -> None:
        if task.state not in _SOL_STATES:
            raise RuntimeError("Sol resume requires a persisted Sol state")
        if task.sol_thread_id is None:
            raise RuntimeError("Sol resume requires the exact persisted thread ID")
        await self._invoke_sol(task, resume_prompt=prompt, context=None)

    async def _invoke_sol(
        self,
        task: TaskRecord,
        *,
        resume_prompt: str | None,
        context: str | None,
    ) -> None:
        async with self._writing_lock:
            routed = await self._invoke_sol_locked(
                task, resume_prompt=resume_prompt, context=context
            )
        if routed is not None:
            await self._route_sol_outcome(routed[0], routed[1], routed[2])

    async def _invoke_sol_locked(
        self,
        task: TaskRecord,
        *,
        resume_prompt: str | None,
        context: str | None,
    ) -> tuple[
        TaskRecord, SolOutcome, tuple[Mapping[str, object], ...]
    ] | None:
        if not self._writing_lock.locked():
            raise RuntimeError("Sol invocation requires the shared writer lock")
        run_id = self._ids.new_run_id()
        completion: asyncio.Event | None = None
        try:
            current = self._store.get_task(task.task_id, task.revision)
            if current.state is not task.state:
                return None
            task = current
            self._store.start_agent_run(
                run_id, task.task_id, task.revision, "sol"
            )
            completion = self._track_run(run_id)
            if resume_prompt is None:
                if task.brief is None or context is None:
                    raise RuntimeError("Sol start context is incomplete")
                result = await self._sol.start(
                    run_id=run_id, brief=task.brief, context=context
                )
            else:
                if task.sol_thread_id is None:
                    raise RuntimeError("Sol resume requires an exact thread")
                result = await self._sol.resume(
                    run_id=run_id,
                    thread_id=task.sol_thread_id,
                    prompt=resume_prompt,
                )
            observed_events = self._persist_agent_result(
                task,
                "sol",
                result,
                run_id=run_id,
                expected_cli_session_id=(
                    task.sol_thread_id if resume_prompt is not None else None
                ),
            )
            if (
                resume_prompt is not None
                and result.cli_session_id is not None
                and result.cli_session_id != task.sol_thread_id
            ):
                raise RuntimeError("Sol resumed a different thread than requested")
            if result.cli_session_id is not None:
                task = self._store.set_sol_thread(
                    task.task_id, task.revision, result.cli_session_id
                )
            if result.interrupted:
                self._finish_interrupted_run(run_id, exit_code=result.exit_code)
                current = self._store.get_task(task.task_id, task.revision)
                if current.state is not TaskState.INTERRUPTED:
                    current = self._store.mark_interrupted(
                        current.task_id,
                        current.revision,
                        continuation=task.state,
                    )
                    self._emit_state(current)
                return None
            if result.payload is None or result.cli_session_id is None:
                raise RuntimeError("Sol completed without an outcome or thread ID")
            outcome = SolOutcome.from_dict(result.payload)
            self._store.finish_agent_run(run_id, status="completed", exit_code=result.exit_code)
        except asyncio.CancelledError:
            if completion is not None:
                self._interrupt_if_active(
                    run_id, task.task_id, task.revision, task.state
                )
            raise
        except BaseException as error:
            if completion is not None:
                self._record_agent_failure(
                    task,
                    "sol",
                    run_id,
                    error,
                    expected_cli_session_id=(
                        task.sol_thread_id if resume_prompt is not None else None
                    ),
                )
            raise
        finally:
            if completion is not None:
                self._complete_run(run_id, completion)
        return (
            self._store.get_task(task.task_id, task.revision),
            outcome,
            observed_events,
        )

    async def _route_sol_outcome(
        self,
        task: TaskRecord,
        outcome: SolOutcome,
        observed_events: tuple[Mapping[str, object], ...],
    ) -> None:
        current = self._store.get_task(task.task_id, task.revision)
        if current.state is not task.state or current.state not in _SOL_STATES:
            return
        task = current
        if outcome.status == "question":
            if outcome.question is None:
                raise RuntimeError("question outcome is missing its question")
            clarification_prompt = self._clarification_prompt(task, outcome)
            task = self._store.pause_for_continuation(
                task.task_id,
                task.revision,
                expected=task.state,
                target=TaskState.FABLE_CLARIFYING,
                continuation_state=task.state,
                pending={
                    "clarification_prompt": clarification_prompt,
                    "underlying_continuation": task.state.value,
                },
            )
            self._emit_state(task)
            self._emit_sol_outcome(task, outcome)
            await self._call_fable_clarification(task, clarification_prompt)
            return
        if outcome.status in {"blocked", "failed"}:
            target = (
                TaskState.AWAITING_USER_INPUT
                if outcome.status == "blocked"
                else TaskState.FAILED
            )
            if target is TaskState.AWAITING_USER_INPUT:
                task = self._store.pause_for_continuation(
                    task.task_id,
                    task.revision,
                    expected=task.state,
                    target=target,
                    continuation_state=task.state,
                    pending={"sol_status": outcome.status},
                )
            else:
                task = self._store.transition_task(
                    task.task_id,
                    task.revision,
                    expected=task.state,
                    target=target,
                )
            self._emit_state(task)
            self._emit_sol_outcome(task, outcome)
            return
        task = self._store.transition_task(
            task.task_id,
            task.revision,
            expected=task.state,
            target=TaskState.FABLE_REVIEWING,
        )
        self._emit_state(task)
        self._emit_sol_outcome(task, outcome)
        await self._review(task, outcome, observed_events)

    async def _call_fable_clarification(
        self, task: TaskRecord, prompt: str,
    ) -> None:
        if task.fable_session_id is None:
            raise RuntimeError("clarification requires the exact Fable session")
        run_id = self._ids.new_run_id()
        self._store.start_agent_run(run_id, task.task_id, task.revision, "fable")
        completion = self._track_run(run_id)
        try:
            result = await self._fable.clarify(
                run_id=run_id, session_id=task.fable_session_id, prompt=prompt
            )
            self._persist_agent_result(
                task,
                "fable",
                result,
                run_id=run_id,
                expected_cli_session_id=task.fable_session_id,
            )
            if result.cli_session_id not in {None, task.fable_session_id}:
                raise RuntimeError("Fable resumed a different session than requested")
            if result.interrupted:
                self._finish_interrupted_run(run_id, exit_code=result.exit_code)
                current = self._store.get_task(task.task_id, task.revision)
                if current.state is not TaskState.INTERRUPTED:
                    current = self._store.mark_interrupted(
                        current.task_id,
                        current.revision,
                        continuation=TaskState.FABLE_CLARIFYING,
                        cli_session_id=result.cli_session_id,
                    )
                    self._emit_state(current)
                return
            if result.payload is None:
                raise RuntimeError("Fable clarification completed without a contract")
            clarification = FableClarification.from_dict(result.payload)
            self._store.finish_agent_run(run_id, status="completed", exit_code=result.exit_code)
        except asyncio.CancelledError:
            self._interrupt_if_active(
                run_id,
                task.task_id,
                task.revision,
                TaskState.FABLE_CLARIFYING,
            )
            raise
        except BaseException as error:
            self._record_agent_failure(
                task,
                "fable",
                run_id,
                error,
                expected_cli_session_id=task.fable_session_id,
            )
            raise
        finally:
            self._complete_run(run_id, completion)
        await self._route_clarification(
            self._store.get_task(task.task_id, task.revision), clarification
        )

    async def _route_clarification(
        self, task: TaskRecord, clarification: FableClarification,
    ) -> None:
        current = self._store.get_task(task.task_id, task.revision)
        if current.state is not TaskState.FABLE_CLARIFYING:
            return
        task = current
        if clarification.status == "escalate_to_user":
            question = clarification.question_for_user
            if question is None:
                raise RuntimeError("Fable escalation is missing its user question")
            task = self._store.retarget_continuation(
                task.task_id,
                task.revision,
                expected=TaskState.FABLE_CLARIFYING,
                target=TaskState.AWAITING_USER_INPUT,
                pending={"question_for_user": question},
            )
            self._emit_state(task)
            self._emit_clarification(task, clarification)
            return
        if clarification.answer is None:
            raise RuntimeError("answered clarification is missing its answer")
        if clarification.scope_changed:
            revised = clarification.revised_brief
            if revised is None:
                raise RuntimeError("scope change is missing its revised brief")
            if (
                revised.task_id != task.task_id
                or revised.revision != task.revision + 1
            ):
                raise ValueError("scope change must create the exact next task revision")
            if task.fable_session_id is None or task.sol_thread_id is None:
                raise RuntimeError("scope change is missing exact agent continuation IDs")
            original_baseline = self._load_baseline(task)
            widened_baseline = self._repository.widen_baseline(
                original_baseline, revised
            )
            try:
                saved = self._store.save_scope_revision(
                    task.session_id,
                    revised,
                    fable_session_id=task.fable_session_id,
                    sol_thread_id=task.sol_thread_id,
                    correction_count=task.correction_count,
                    continuation_state=(
                        task.continuation_state or TaskState.SOL_RUNNING
                    ),
                    pending={"answer": clarification.answer},
                    baseline_id=original_baseline.baseline_id,
                    setting=(
                        self._baseline_key(task.task_id, revised.revision),
                        self._baseline_setting_value(
                            task.task_id, revised.revision, widened_baseline
                        ),
                    ),
                )
            except BaseException:
                self._repository.discard_widening(
                    original_baseline, widened_baseline
                )
                raise
            self._store.append_event(
                saved.session_id,
                saved.task_id,
                "fable",
                "task_brief",
                {"brief": revised.to_dict()},
            )
            self._emit_state(saved)
            self._emit_clarification(saved, clarification)
            return
        resumed = self._store.resume_continuation(
            task.task_id, task.revision, expected=TaskState.FABLE_CLARIFYING
        )
        self._emit_state(resumed)
        self._emit_clarification(resumed, clarification)
        await self._resume_sol(resumed, clarification.answer)

    async def _review(
        self,
        task: TaskRecord,
        outcome: SolOutcome,
        observed_events: tuple[Mapping[str, object], ...],
    ) -> None:
        baseline = self._load_baseline(task)
        delta = self._repository.compare(baseline)
        reconciliation = self._reconcile(task, outcome, observed_events)
        completion_allowed = not (
            delta.unexpected_paths
            or delta.protected_changed_paths
            or not reconciliation["claims_ok"]
            or not reconciliation["required_tests_ok"]
        )
        prompt = json.dumps(
            {
                "brief": None if task.brief is None else task.brief.to_dict(),
                "delta": self._delta_payload(delta),
                "observed_evidence": [dict(event) for event in observed_events],
                "reconciliation": reconciliation,
                "sol_claims": self._sol_claims_projection(outcome),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        await self._call_fable_review(
            task, prompt, completion_allowed=completion_allowed
        )

    async def _call_fable_review(
        self,
        task: TaskRecord,
        prompt: str,
        *,
        completion_allowed: bool,
    ) -> None:
        if task.fable_session_id is None:
            raise RuntimeError("review requires the exact Fable session")
        if task.pending is None:
            task = self._store.set_pending_context(
                task.task_id,
                task.revision,
                expected=TaskState.FABLE_REVIEWING,
                pending={
                    "review_prompt": prompt,
                    "completion_allowed": completion_allowed,
                },
            )
        run_id = self._ids.new_run_id()
        self._store.start_agent_run(run_id, task.task_id, task.revision, "fable")
        completion = self._track_run(run_id)
        try:
            result = await self._fable.review(
                run_id=run_id, session_id=task.fable_session_id, prompt=prompt
            )
            self._persist_agent_result(
                task,
                "fable",
                result,
                run_id=run_id,
                expected_cli_session_id=task.fable_session_id,
            )
            if result.cli_session_id not in {None, task.fable_session_id}:
                raise RuntimeError("Fable resumed a different session than requested")
            if result.interrupted:
                self._finish_interrupted_run(run_id, exit_code=result.exit_code)
                current = self._store.get_task(task.task_id, task.revision)
                if current.state is not TaskState.INTERRUPTED:
                    current = self._store.mark_interrupted(
                        current.task_id,
                        current.revision,
                        continuation=TaskState.FABLE_REVIEWING,
                        cli_session_id=result.cli_session_id,
                    )
                    self._emit_state(current)
                return
            if result.payload is None:
                raise RuntimeError("Fable review completed without a contract")
            verdict = ReviewVerdict.from_dict(result.payload)
            self._store.finish_agent_run(run_id, status="completed", exit_code=result.exit_code)
        except asyncio.CancelledError:
            self._interrupt_if_active(
                run_id,
                task.task_id,
                task.revision,
                TaskState.FABLE_REVIEWING,
            )
            raise
        except BaseException as error:
            self._record_agent_failure(
                task,
                "fable",
                run_id,
                error,
                expected_cli_session_id=task.fable_session_id,
            )
            raise
        finally:
            self._complete_run(run_id, completion)
        await self._route_review(
            self._store.get_task(task.task_id, task.revision),
            verdict,
            prompt,
            completion_allowed=completion_allowed,
        )

    async def _route_review(
        self,
        task: TaskRecord,
        verdict: ReviewVerdict,
        review_prompt: str,
        *,
        completion_allowed: bool,
    ) -> None:
        current = self._store.get_task(task.task_id, task.revision)
        if current.state is not TaskState.FABLE_REVIEWING:
            return
        task = current
        if verdict.status == "approved" and completion_allowed:
            task = self._store.transition_task_clearing_pending(
                task.task_id,
                task.revision,
                expected=TaskState.FABLE_REVIEWING,
                target=TaskState.COMPLETED,
            )
            self._emit_state(task)
            self._emit_review(task, verdict)
            return
        if verdict.status == "corrections_required" and task.correction_count < 3:
            task = self._store.increment_correction_count(task.task_id, task.revision)
            task = self._store.transition_task_clearing_pending(
                task.task_id,
                task.revision,
                expected=TaskState.FABLE_REVIEWING,
                target=TaskState.SOL_CORRECTING,
            )
            self._emit_state(task)
            self._emit_review(task, verdict)
            await self._resume_sol(task, "\n".join(verdict.corrections))
            return

        if verdict.status == "escalate_to_user":
            question = verdict.question_for_user or "Fable requires user direction."
        elif verdict.status == "corrections_required":
            question = "Fable requested a fourth correction cycle; user approval is required."
        else:
            question = "Completion evidence is missing or failed; user review is required."
        paused = self._store.replace_pending_for_continuation(
            task.task_id,
            task.revision,
            expected=TaskState.FABLE_REVIEWING,
            target=TaskState.AWAITING_USER_INPUT,
            continuation_state=TaskState.FABLE_REVIEWING,
            pending={
                "question_for_user": question,
                "review_prompt": review_prompt,
                "completion_allowed": completion_allowed,
            },
        )
        self._emit_state(paused)
        self._emit_review(paused, verdict)

    def _reconcile(
        self,
        task: TaskRecord,
        outcome: SolOutcome,
        observed_events: tuple[Mapping[str, object], ...],
    ) -> dict[str, object]:
        observed = {
            (event.get("command_sha256"), event.get("exit_code"))
            for event in observed_events
            if event.get("type") == "item.completed"
            and event.get("item_type") == "command_execution"
            and isinstance(event.get("command_sha256"), str)
            and isinstance(event.get("exit_code"), int)
            and (
                event.get("status") == "completed"
                or (
                    event.get("status") == "failed"
                    and event.get("exit_code") != 0
                )
            )
        }
        claims: list[dict[str, object]] = []
        for report in outcome.commands_run:
            digest = hashlib.sha256(report.command.encode()).hexdigest()
            claims.append({
                "command_sha256": digest,
                "exit_code": report.exit_code,
                "observed": (digest, report.exit_code) in observed,
            })
        claims_ok = all(bool(claim["observed"]) for claim in claims)
        required: list[dict[str, object]] = []
        brief = task.brief
        if brief is None:
            raise RuntimeError("review task is missing its brief")
        for required_test in brief.required_tests:
            matching = [
                report for report in outcome.commands_run
                if self._is_pytest_execution(report.command, required_test)
                and report.exit_code == 0
            ]
            matched_hash = next(
                (
                    hashlib.sha256(report.command.encode()).hexdigest()
                    for report in matching
                    if (
                        hashlib.sha256(report.command.encode()).hexdigest(), 0
                    ) in observed
                ),
                None,
            )
            required.append({
                "required_test": required_test,
                "command_sha256": matched_hash,
                "observed_zero_exit": matched_hash is not None,
            })
        return {
            "claims": claims,
            "claims_ok": claims_ok,
            "required_tests": required,
            "required_tests_ok": all(
                bool(item["observed_zero_exit"]) for item in required
            ),
        }

    def _is_pytest_execution(
        self,
        command: str,
        required_test: str,
        *,
        _allow_shell_wrapper: bool = True,
    ) -> bool:
        if any(
            marker in command
            for marker in ("&", "||", ";", "|", ">", "<", "`", "$(", "\n")
        ):
            return False
        try:
            tokens = shlex.split(command)
        except ValueError:
            return False
        if any(token.startswith("@") for token in tokens):
            return False
        def consume_assignments() -> bool:
            while tokens and "=" in tokens[0] and not tokens[0].startswith(("/", "./")):
                name, _, value = tokens[0].partition("=")
                if not name or not name.replace("_", "a").isalnum():
                    break
                if name != "PYTHONPATH" or not Path(value).is_absolute():
                    return False
                try:
                    configured_path = Path(value).resolve(strict=True)
                except OSError:
                    return False
                if configured_path != self._repo_root:
                    return False
                tokens.pop(0)
            return True

        if not consume_assignments():
            return False
        if tokens and tokens[0] == "env":
            tokens.pop(0)
            if not consume_assignments():
                return False
        if not tokens:
            return False
        executable = Path(tokens[0]).name
        if executable in {"bash", "sh"}:
            if (
                not _allow_shell_wrapper
                or not self._is_trusted_shell(tokens[0], executable)
                or len(tokens) != 3
                or tokens[1] != "-lc"
            ):
                return False
            return self._is_pytest_execution(
                tokens[2], required_test, _allow_shell_wrapper=False
            )
        non_execution_flags = {
            "--collect-only",
            "--collectonly",
            "--co",
            "--setup-only",
            "--setup-plan",
            "--fixtures",
            "--fixtures-per-test",
            "--help",
            "-h",
            "--markers",
            "--version",
        }
        if any(token in non_execution_flags for token in tokens):
            return False
        if any(
            token.startswith("-")
            and not token.startswith("--")
            and any(flag in token[1:] for flag in ("h", "V"))
            for token in tokens
        ):
            return False
        candidate = Path(tokens[0])
        if not candidate.is_absolute():
            return False
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            return False
        if (
            resolved != self._python_executable
            or len(tokens) < 3
            or tokens[1:3] != ["-m", "pytest"]
        ):
            return False
        arguments = tokens[3:]
        blocked_long_options = {
            "--deselect",
            "--ignore",
            "--ignore-glob",
        }
        safe_flags = {
            "-q",
            "--quiet",
            "-v",
            "--verbose",
            "-s",
            "-x",
            "--exitfirst",
            "--disable-warnings",
            "--strict-config",
            "--strict-markers",
            "--no-header",
            "--no-summary",
        }
        safe_value_options = {
            "--capture",
            "--tb",
            "--color",
            "--code-highlight",
            "--timeout",
            "--durations",
            "--maxfail",
            "-n",
        }
        targets: list[str] = []
        index = 0
        while index < len(arguments):
            token = arguments[index]
            if token == "--":
                targets.extend(arguments[index + 1:])
                break
            if (
                token in blocked_long_options
                or any(token.startswith(f"{option}=") for option in blocked_long_options)
                or token in {"-k", "-m"}
                or token.startswith("-k")
                or token.startswith("-m")
            ):
                return False
            if token in safe_flags:
                index += 1
                continue
            if token.startswith("-n") and len(token) > 2:
                index += 1
                continue
            if (
                token.startswith("-")
                and not token.startswith("--")
                and len(token) > 2
                and all(flag in "qvsx" for flag in token[1:])
            ):
                index += 1
                continue
            matched_value_option = next(
                (
                    option
                    for option in safe_value_options
                    if token == option or token.startswith(f"{option}=")
                ),
                None,
            )
            if matched_value_option is not None:
                if token == matched_value_option:
                    index += 1
                    if index >= len(arguments):
                        return False
                index += 1
                continue
            if token.startswith("-"):
                return False
            targets.append(token)
            index += 1
        try:
            normalized_required = validate_allowed_path(required_test).as_posix()
        except ValueError:
            return False
        for target in targets:
            raw_path, separator, node = target.partition("::")
            try:
                normalized_target = validate_allowed_path(raw_path).as_posix()
            except ValueError:
                continue
            if normalized_target == normalized_required and (not separator or node):
                return True
        return False

    def _is_trusted_shell(self, token: str, name: str) -> bool:
        trusted = self._trusted_shells.get(name)
        if trusted is None:
            return False
        candidate = Path(token)
        if not candidate.is_absolute():
            return token == name
        try:
            return candidate.resolve(strict=True) == trusted
        except OSError:
            return False

    def _baseline_setting_value(
        self, task_id: str, revision: int, baseline: WorkspaceBaseline,
    ) -> Mapping[str, object]:
        return {
            "task_id": task_id,
            "revision": revision,
            "baseline_id": baseline.baseline_id,
            "manifest": self._repository.baseline_manifest(baseline),
        }

    def _load_baseline(self, task: TaskRecord) -> WorkspaceBaseline:
        if task.baseline_id is None:
            raise RuntimeError("task has no approved baseline")
        persisted = self._store.get_setting(
            self._baseline_key(task.task_id, task.revision)
        )
        if not isinstance(persisted, Mapping):
            raise RuntimeError("approved baseline manifest is missing")
        if (
            persisted.get("task_id") != task.task_id
            or persisted.get("revision") != task.revision
            or persisted.get("baseline_id") != task.baseline_id
        ):
            raise RuntimeError("approved baseline manifest identity mismatch")
        return self._repository.restore_baseline(
            persisted.get("manifest"), expected_baseline_id=task.baseline_id
        )

    @staticmethod
    def _baseline_key(task_id: str, revision: int) -> str:
        return f"agent_bridge.baseline.{task_id}.{revision}"

    def _persist_agent_result(
        self,
        task: TaskRecord,
        actor: str,
        result: AgentRunResult,
        *,
        run_id: str,
        expected_cli_session_id: str | None,
    ) -> tuple[Mapping[str, object], ...]:
        if result.run_id != run_id:
            raise RuntimeError("adapter result does not belong to the coordinator-owned run")
        self._validate_cli_session_id(
            actor,
            result.cli_session_id,
            expected=expected_cli_session_id,
        )
        run = self._store.agent_run(run_id)
        if result.cli_session_id is not None and run.status == "running":
            self._store.set_agent_run_session(run_id, result.cli_session_id)
        normalized = tuple(
            safe_event
            for event in result.events
            if (safe_event := self._structural_agent_event(actor, event)) is not None
        )
        for safe_event in normalized:
            self._store.append_event(
                task.session_id,
                task.task_id,
                actor,
                "agent_event",
                safe_event,
            )
        return normalized

    @staticmethod
    def _validate_cli_session_id(
        actor: str,
        session_id: str | None,
        *,
        expected: str | None,
    ) -> None:
        if session_id is None:
            return
        if actor == "sol":
            try:
                canonical = str(UUID(session_id))
            except (ValueError, AttributeError):
                raise RuntimeError("Sol returned an invalid session ID") from None
            if canonical != session_id:
                raise RuntimeError("Sol returned a non-canonical session ID")
        elif actor == "fable":
            if _SAFE_ID.fullmatch(session_id) is None:
                raise RuntimeError("Fable returned an invalid session ID")
        else:
            raise ValueError("agent result actor must be fable or sol")
        if expected is not None and session_id != expected:
            raise RuntimeError(f"{actor.title()} resumed a different session than requested")

    @staticmethod
    def _structural_agent_event(
        actor: str, event: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        if actor == "sol":
            return Coordinator._sol_structural_event(event)
        if actor == "fable":
            return Coordinator._fable_structural_event(event)
        raise ValueError("agent event actor must be fable or sol")

    @staticmethod
    def _sol_structural_event(
        event: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        event_type = event.get("type")
        if not isinstance(event_type, str) or event_type not in _SOL_EVENT_TYPES:
            return None
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
            if not isinstance(thread_id, str):
                return None
            try:
                canonical = str(UUID(thread_id))
            except ValueError:
                return None
            if canonical != thread_id:
                return None
            return {"type": event_type, "thread_id": canonical}

        item_type = event.get("item_type")
        if not isinstance(item_type, str) or item_type not in _SOL_ITEM_TYPES:
            return None
        safe: dict[str, object] = {
            "type": event_type,
            "item_type": item_type,
        }
        if event_type == "item.completed":
            status = event.get("status")
            if isinstance(status, str) and status in _SOL_ITEM_STATUSES:
                safe["status"] = status
        if item_type != "command_execution":
            return safe
        command_digest = event.get("command_sha256")
        if isinstance(command_digest, str) and _HEX_256.fullmatch(command_digest):
            safe["command_sha256"] = command_digest
        if event_type != "item.completed":
            return safe
        exit_code = event.get("exit_code")
        if (
            isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
            and _MIN_EXIT_CODE <= exit_code <= _MAX_EXIT_CODE
        ):
            safe["exit_code"] = exit_code
        output_digest = event.get("output_sha256")
        if isinstance(output_digest, str) and _HEX_256.fullmatch(output_digest):
            safe["output_sha256"] = output_digest
        for field in ("output_bytes", "output_lines"):
            value = event.get(field)
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= _MAX_AUDIT_COUNT
            ):
                safe[field] = value
        return safe

    @staticmethod
    def _fable_structural_event(
        event: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        event_type = event.get("type")
        if not isinstance(event_type, str) or event_type not in _FABLE_EVENT_TYPES:
            return None
        if event_type in {"assistant", "stream_event", "user"}:
            return {"type": event_type}
        if event_type == "result":
            structured = event.get("has_structured_output")
            if not isinstance(structured, bool):
                return None
            return {"type": event_type, "has_structured_output": structured}
        if event.get("subtype") != "init":
            return None
        safe: dict[str, object] = {"type": "system", "subtype": "init"}
        session_id = event.get("session_id")
        if session_id is None:
            return safe
        if not isinstance(session_id, str) or _SAFE_ID.fullmatch(session_id) is None:
            return None
        safe["session_id"] = session_id
        return safe

    def _record_agent_failure(
        self,
        task: TaskRecord,
        actor: str,
        run_id: str,
        error: BaseException,
        *,
        expected_cli_session_id: str | None,
    ) -> None:
        result = (
            error.result
            if isinstance(error, (ClaudeRunError, CodexRunError))
            else None
        )
        if result is not None:
            try:
                self._persist_agent_result(
                    task,
                    actor,
                    result,
                    run_id=run_id,
                    expected_cli_session_id=expected_cli_session_id,
                )
            except BaseException:
                self._finish_failed_run(run_id)
                self._fail_if_active(task.task_id, task.revision)
                raise
            self._finish_failed_run(run_id, exit_code=result.exit_code)
        else:
            self._finish_failed_run(run_id)
        self._fail_if_active(task.task_id, task.revision)

    def _finish_failed_run(self, run_id: str, *, exit_code: int = 1) -> None:
        try:
            run = self._store.agent_run(run_id)
        except RuntimeError:
            return
        if run.status == "running":
            self._store.finish_agent_run(
                run_id, status="failed", exit_code=exit_code
            )

    def _finish_interrupted_run(self, run_id: str, *, exit_code: int) -> None:
        run = self._store.agent_run(run_id)
        if run.status == "running":
            self._store.finish_agent_run(
                run_id, status="interrupted", exit_code=exit_code
            )

    def _track_run(self, run_id: str) -> asyncio.Event:
        completion = asyncio.Event()
        if run_id in self._run_completions:
            raise RuntimeError("run completion is already tracked")
        self._run_completions[run_id] = completion
        return completion

    def _complete_run(self, run_id: str, completion: asyncio.Event) -> None:
        completion.set()
        if self._run_completions.get(run_id) is completion:
            del self._run_completions[run_id]

    def _interrupt_if_active(
        self,
        run_id: str,
        task_id: str,
        revision: int,
        continuation: TaskState,
    ) -> None:
        run = self._store.agent_run(run_id)
        if run.status == "running":
            self._finish_interrupted_run(run_id, exit_code=-1)
        task = self._store.get_task(task_id, revision)
        if task.state in _ACTIVE_STATES:
            task = self._store.mark_interrupted(
                task.task_id, task.revision, continuation=continuation
            )
            self._emit_state(task)

    def _fail_if_active(self, task_id: str, revision: int) -> None:
        task = self._store.get_task(task_id, revision)
        if task.state in _ACTIVE_STATES:
            failed = self._store.transition_task(
                task.task_id,
                task.revision,
                expected=task.state,
                target=TaskState.FAILED,
            )
            self._emit_state(failed)

    def _emit_state(self, task: TaskRecord) -> None:
        self._store.append_event(
            task.session_id,
            task.task_id,
            "coordinator",
            "task_state",
            {"state": task.state.value, "revision": task.revision},
        )

    def _emit_clarification(
        self, task: TaskRecord, clarification: FableClarification,
    ) -> None:
        self._store.append_event(
            task.session_id,
            task.task_id,
            "fable",
            "clarification",
            clarification.to_dict(),
        )

    def _emit_review(self, task: TaskRecord, verdict: ReviewVerdict) -> None:
        self._store.append_event(
            task.session_id, task.task_id, "fable", "review", verdict.to_dict()
        )

    def _emit_sol_outcome(self, task: TaskRecord, outcome: SolOutcome) -> None:
        self._store.append_event(
            task.session_id,
            task.task_id,
            "sol",
            "outcome",
            self._sol_claims_projection(outcome),
        )

    @staticmethod
    def _sol_claims_projection(outcome: SolOutcome) -> dict[str, object]:
        return {
            "status": outcome.status,
            "summary": outcome.summary,
            "changed_files": list(outcome.changed_files),
            "known_failures": list(outcome.known_failures),
            "remaining_risks": list(outcome.remaining_risks),
            "architecture_docs": outcome.architecture_docs,
            "question": None if outcome.question is None else outcome.question.to_dict(),
            "command_claims": [
                {
                    "command_sha256": hashlib.sha256(
                        report.command.encode()
                    ).hexdigest(),
                    "exit_code": report.exit_code,
                }
                for report in outcome.commands_run
            ],
        }

    @staticmethod
    def _scope_resume_prompt(task: TaskRecord, answer: str) -> str:
        if task.brief is None:
            raise RuntimeError("scope continuation task is missing its revised brief")
        serialized = json.dumps(
            task.brief.to_dict(), separators=(",", ":"), sort_keys=True
        )
        return (
            f"The user approved this exact revised TaskBrief: {serialized}\n"
            f"Fable clarification: {answer}\n"
            "Continue only within this exact approved revision."
        )

    def _latest_required(self, task_id: str) -> TaskRecord:
        task = self._store.latest_task(task_id)
        if task is None:
            raise RuntimeError("task record not found")
        return task

    def _repo_context(self) -> str:
        return f"Repository: {self._repo_root}\n{self._repo_context_text}"

    def _original_user_message(self, session_id: str, task_id: str) -> str:
        for event in self._store.events_after(session_id, 0):
            if (
                event.task_id == task_id
                and event.actor == "user"
                and event.kind == "message"
                and isinstance(event.payload.get("text"), str)
            ):
                return str(event.payload["text"])
        raise RuntimeError("interrupted planning task has no original user message")

    @staticmethod
    def _clarification_prompt(task: TaskRecord, outcome: SolOutcome) -> str:
        if outcome.question is None:
            raise RuntimeError("question outcome has no question")
        return json.dumps(
            {
                "approved_revision": task.revision,
                "question": outcome.question.to_dict(),
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _delta_summary(delta: WorkspaceDelta) -> dict[str, object]:
        return {
            "changed_paths": list(delta.changed_paths),
            "unexpected_paths": list(delta.unexpected_paths),
            "protected_changed_paths": list(delta.protected_changed_paths),
        }

    @classmethod
    def _delta_payload(cls, delta: WorkspaceDelta) -> dict[str, object]:
        return {
            **cls._delta_summary(delta),
            "preexisting_unchanged_paths": list(delta.preexisting_unchanged_paths),
            "text_diffs": dict(delta.text_diffs),
            "binary_changed_paths": list(delta.binary_changed_paths),
        }
