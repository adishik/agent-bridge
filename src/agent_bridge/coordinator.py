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
from dataclasses import dataclass
from enum import Enum
from typing import Protocol
from uuid import UUID

from agent_bridge.adapters.base import AgentRunResult, FableAdapter, SolAdapter
from agent_bridge.adapters.claude_cli import ClaudeCLI, ClaudeRunError, SubscriptionAuthError
from agent_bridge.adapters.codex_cli import CodexCLI, CodexRunError
from agent_bridge.contracts import (
    ConversationActor,
    ConversationEnvelope,
    ConversationMessageType,
    ConversationTarget,
    DirectedAgentQuestion,
    FableClarification,
    ReviewVerdict,
    SolOutcome,
    TaskBrief,
    UserConversationInput,
)
from agent_bridge.process import ProcessRunner
from agent_bridge.projects import project_id_for_root
from agent_bridge.repository import (
    RepositoryTracker,
    WorkspaceBaseline,
    WorkspaceDelta,
    validate_allowed_path,
)
from agent_bridge.state_machine import TaskState
from agent_bridge.store import (
    AnswerContext,
    AnswerPayload,
    BaselineSetting,
    ClarificationContext,
    COMPATIBILITY_PREPARATION_GENERATION,
    ContinuationMessagePayload,
    ExchangeGrantPayload,
    NewRequestPayload,
    PreparedActionInterruptionReason,
    PreparedActionOutcome,
    PreparedActionRecord,
    QuestionAnswerPayload,
    QuestionRecord,
    ReviewContext,
    ResumeDriftProjection,
    ResumePayload,
    RecoverySummary,
    ScopeApprovalContext,
    SolResumeContext,
    SQLiteStore,
    TaskRecord,
    ApprovalPayload,
)


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
MAX_AGENT_STRUCTURAL_EVENTS = 1_024
_MIN_EXIT_CODE = -(2**31)
_MAX_EXIT_CODE = 2**31 - 1
_TERMINAL_ROUTING_STATES = frozenset({TaskState.COMPLETED, TaskState.FAILED})
_CONVERSATION_PREPARED_ACTIONS = frozenset({
    "continuation_message", "question_answer", "exchange_grant",
})
_EXCHANGE_PERMISSION_TEXT = (
    "Automatic exchange limit reached. Allow three more internal exchanges to continue."
)


class RoutingMode(str, Enum):
    """The only two user-message routing outcomes."""

    NEW_FABLE_TASK = "new_fable_task"
    BOUND_CONTINUATION = "bound_continuation"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """A deterministic route derived solely from authenticated task state."""

    addressed_to: ConversationTarget
    routed_to: ConversationTarget
    mode: RoutingMode
    task_id: str | None
    revision: int | None
    continuation_generation: int | None


class RoutingError(RuntimeError):
    """One bounded failure category for invalid or stale conversation routes."""

    def __init__(self) -> None:
        super().__init__("conversation routing is unavailable")


def route_user_intent(
    authenticated_task: TaskRecord | None,
    intent: UserConversationInput,
) -> RoutingDecision:
    """Route an authenticated user intent without touching a lease or adapter.

    An unbound message is always a new Fable-planned task.  A bound message
    must match every persisted identity coordinate; stale or unknown bindings
    fail closed rather than selecting another task.  Before exact approval,
    an apparent Sol address remains visible but is routed to Fable.
    """
    if not isinstance(intent, UserConversationInput):
        raise RoutingError()
    if intent.task_id is None:
        return RoutingDecision(
            addressed_to=intent.addressed_to,
            routed_to=ConversationTarget.FABLE,
            mode=RoutingMode.NEW_FABLE_TASK,
            task_id=None,
            revision=None,
            continuation_generation=None,
        )
    if not isinstance(authenticated_task, TaskRecord):
        raise RoutingError()
    if (
        intent.task_id != authenticated_task.task_id
        or intent.revision != authenticated_task.revision
        or intent.continuation_generation != authenticated_task.continuation_generation
    ):
        raise RoutingError()
    if authenticated_task.state in _TERMINAL_ROUTING_STATES:
        return RoutingDecision(
            addressed_to=intent.addressed_to,
            routed_to=ConversationTarget.FABLE,
            mode=RoutingMode.NEW_FABLE_TASK,
            task_id=None,
            revision=None,
            continuation_generation=None,
        )
    routed_to = intent.addressed_to
    if routed_to in {ConversationTarget.TEAM, ConversationTarget.USER}:
        routed_to = ConversationTarget.FABLE
    if routed_to is ConversationTarget.SOL and authenticated_task.approved_at is None:
        routed_to = ConversationTarget.FABLE
    return RoutingDecision(
        addressed_to=intent.addressed_to,
        routed_to=routed_to,
        mode=RoutingMode.BOUND_CONTINUATION,
        task_id=authenticated_task.task_id,
        revision=authenticated_task.revision,
        continuation_generation=authenticated_task.continuation_generation,
    )


class ResumeDriftBlocked(RuntimeError):
    """A persisted terminal Resume result that must not schedule a child."""

    def __init__(self, task: TaskRecord) -> None:
        self.task = task
        super().__init__("resume blocked by repository drift")


class PreparedActionFailed(RuntimeError):
    """A bounded public failure after a prepared action was terminalized."""

    def __init__(self) -> None:
        super().__init__("prepared action failed")


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
        self._terminal_retries: dict[str, PreparedActionOutcome] = {}
        self._compatibility_errors: dict[str, RuntimeError] = {}
        # A prepared row is never proof that a child is still alive after a
        # process restart.  Recover before this coordinator can be admitted to
        # a runtime registry or start another workflow.
        self.recover_unfinished_prepared_actions()

    def close(self) -> None:
        """Release coordinator-local tracking; the runtime owns its store."""
        self._run_completions.clear()
        self._terminal_retries.clear()
        self._compatibility_errors.clear()

    @property
    def project_id(self) -> str:
        return project_id_for_root(self._repo_root)

    async def handle_user_request(self, session_id: str, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        task_id = self._ids.new_task_id()
        prepared = self.prepare_new_request(
            session_id=session_id,
            task_id=task_id,
            text=text,
            generation=COMPATIBILITY_PREPARATION_GENERATION,
        )
        await self._run_compatibility_prepared(prepared.preparation_id)
        return task_id

    def prepare_new_request(
        self, *, session_id: str, task_id: str, text: str, generation: int,
    ) -> PreparedActionRecord:
        return self._store.prepare_new_request_action(
            project_id=self.project_id,
            session_id=session_id,
            task_id=task_id,
            generation=generation,
            payload=NewRequestPayload(text=text),
        )

    def prepare_user_request(
        self, session_id: str, text: str, task_id: str,
    ) -> TaskRecord:
        """Compatibility wrapper for the pre-hub synchronous preparation seam."""
        prepared = self.prepare_new_request(
            session_id=session_id,
            task_id=task_id,
            text=text,
            generation=COMPATIBILITY_PREPARATION_GENERATION,
        )
        return self._store.get_task(prepared.task_id, prepared.revision)

    def prepare_approval(
        self, *, session_id: str, task_id: str, revision: int, generation: int,
    ) -> PreparedActionRecord:
        task = self._store.get_task(task_id, revision)
        if task.session_id != session_id:
            raise RuntimeError("task record not found")
        if task.state not in {
            TaskState.AWAITING_USER_APPROVAL,
            TaskState.AWAITING_SCOPE_APPROVAL,
        } or task.brief is None or task.brief.open_questions:
            raise ValueError("task is not awaiting revision approval")
        captured: WorkspaceBaseline | None = None
        if task.baseline_id is None:
            captured = self._repository.capture(task.brief)
            baseline_id = captured.baseline_id
            setting = BaselineSetting(
                key=self._baseline_key(task.task_id, task.revision),
                value_json=json.dumps(
                    self._baseline_setting_value(task.task_id, task.revision, captured),
                    separators=(",", ":"), sort_keys=True,
                ),
            )
        else:
            baseline = self._load_baseline(task)
            if baseline.allowed_paths != task.brief.allowed_paths:
                raise RuntimeError("approved baseline scope does not match the task revision")
            baseline_id = baseline.baseline_id
            setting = None
        scope = None
        if (
            task.state is TaskState.AWAITING_SCOPE_APPROVAL
            or task.continuation_state in _SOL_STATES
        ):
            answer = (task.pending or {}).get("answer")
            if task.continuation_state not in _SOL_STATES or not isinstance(answer, str):
                raise RuntimeError("scope approval is missing its exact continuation")
            scope = ScopeApprovalContext(
                baseline_id=baseline_id,
                approved_revision=task.revision,
                underlying_continuation=self._sol_context(task, self._scope_resume_prompt(
                    task, answer
                )),
            )
        try:
            return self._store.prepare_approval_action(
                project_id=self.project_id,
                session_id=session_id,
                task_id=task_id,
                revision=revision,
                generation=generation,
                payload=ApprovalPayload(
                    baseline_id=baseline_id, baseline_setting=setting, scope=scope,
                ),
            )
        except BaseException:
            if captured is not None:
                self._repository.discard_baseline(captured)
            raise

    def prepare_answer(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        answer: str,
        generation: int,
    ) -> PreparedActionRecord:
        task = self._store.get_task(task_id, revision)
        if task.session_id != session_id or task.state is not TaskState.AWAITING_USER_INPUT:
            raise ValueError("task is not awaiting user input")
        if task.continuation_state is None or task.pending is None:
            raise RuntimeError("awaiting-user task has no persisted continuation")
        return self._store.prepare_answer_action(
            project_id=self.project_id,
            session_id=session_id,
            task_id=task_id,
            revision=revision,
            generation=generation,
            payload=AnswerPayload(answer=answer, continuation=self._context_from_task(task)),
        )

    def prepare_continuation_message(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        continuation_generation: int,
        text: str,
        addressed_to: ConversationTarget,
    ) -> TaskRecord:
        """Prepare one bound user statement through its distinct durable kind."""
        record = self._prepare_continuation_message_action(
            session_id=session_id,
            task_id=task_id,
            revision=revision,
            continuation_generation=continuation_generation,
            text=text,
            addressed_to=addressed_to,
            generation=COMPATIBILITY_PREPARATION_GENERATION,
        )
        return self._store.get_task(record.task_id, record.revision)

    def _prepare_continuation_message_action(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        continuation_generation: int,
        text: str,
        addressed_to: ConversationTarget,
        generation: int,
    ) -> PreparedActionRecord:
        task = self._store.get_task(task_id, revision)
        if task.session_id != session_id:
            raise RuntimeError("task record not found")
        intent = UserConversationInput(
            addressed_to=addressed_to,
            message_type=ConversationMessageType.STATEMENT,
            text=text,
            task_id=task_id,
            revision=revision,
            continuation_generation=continuation_generation,
        )
        decision = route_user_intent(task, intent)
        if decision.mode is not RoutingMode.BOUND_CONTINUATION:
            raise RoutingError()
        context = self._context_from_task(task)
        return self._store.prepare_continuation_message_action(
            project_id=self.project_id,
            session_id=session_id,
            task_id=task_id,
            revision=revision,
            generation=generation,
            payload=ContinuationMessagePayload(
                text=text,
                addressed_to=decision.addressed_to,
                routed_to=decision.routed_to,
                continuation_generation=continuation_generation,
                continuation=context,
            ),
        )

    def prepare_question_answer(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        continuation_generation: int,
        question_id: str,
        answer: str,
    ) -> TaskRecord:
        """Prepare an answer only for the exact user-routed question."""
        record = self._prepare_question_answer_action(
            session_id=session_id,
            task_id=task_id,
            revision=revision,
            continuation_generation=continuation_generation,
            question_id=question_id,
            answer=answer,
            generation=COMPATIBILITY_PREPARATION_GENERATION,
        )
        return self._store.get_task(record.task_id, record.revision)

    def _prepare_question_answer_action(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        continuation_generation: int,
        question_id: str,
        answer: str,
        generation: int,
    ) -> PreparedActionRecord:
        task = self._store.get_task(task_id, revision)
        if task.session_id != session_id:
            raise RuntimeError("task record not found")
        return self._store.prepare_question_answer_action(
            project_id=self.project_id,
            session_id=session_id,
            task_id=task_id,
            revision=revision,
            generation=generation,
            payload=QuestionAnswerPayload(
                question_id=question_id,
                answer=answer,
                continuation_generation=continuation_generation,
                continuation=self._context_from_task(task),
            ),
        )

    def prepare_exchange_grant(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        continuation_generation: int,
        request_id: str,
    ) -> TaskRecord:
        """Prepare one fixed grant through its distinct durable kind."""
        record = self._prepare_exchange_grant_action(
            session_id=session_id,
            task_id=task_id,
            revision=revision,
            continuation_generation=continuation_generation,
            request_id=request_id,
            generation=COMPATIBILITY_PREPARATION_GENERATION,
        )
        return self._store.get_task(record.task_id, record.revision)

    def _prepare_exchange_grant_action(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        continuation_generation: int,
        request_id: str,
        generation: int,
    ) -> PreparedActionRecord:
        task = self._store.get_task(task_id, revision)
        if task.session_id != session_id:
            raise RuntimeError("task record not found")
        attempted = (task.pending or {}).get("attempted_question")
        if not isinstance(attempted, Mapping):
            raise RuntimeError("exchange permission is missing its attempted question")
        try:
            directed = DirectedAgentQuestion.from_dict(attempted)
        except ValueError as error:
            raise RuntimeError("exchange permission is missing its attempted question") from error
        outer = self._store.unanswered_question_for_task(task_id, revision)
        outer_question_id = (
            outer.question_id
            if task.continuation_state is TaskState.FABLE_CLARIFYING and outer is not None
            else None
        )
        continuation = (
            self._stored_sol_context(task) if outer_question_id is not None
            else self._context_from_task(task)
        )
        return self._store.prepare_exchange_grant_action(
            project_id=self.project_id,
            session_id=session_id,
            task_id=task_id,
            revision=revision,
            generation=generation,
            payload=ExchangeGrantPayload(
                request_id=request_id,
                continuation_generation=continuation_generation,
                attempted_question=directed,
                continuation=continuation,
                parent_mode=(
                    "question" if outer_question_id is not None
                    else "clarification" if isinstance(continuation, ClarificationContext)
                    else "top_level"
                ),
                outer_question_id=outer_question_id,
            ),
        )

    def prepare_resume(
        self, *, session_id: str, task_id: str, revision: int, generation: int,
    ) -> PreparedActionRecord:
        task = self._store.get_task(task_id, revision)
        if task.session_id != session_id:
            raise ValueError("only an interrupted task may be resumed")
        predecessor = self._store.latest_prepared_action_for_task(
            project_id=self.project_id,
            session_id=session_id,
            task_id=task_id,
            revision=revision,
        )
        if (
            predecessor is not None
            and predecessor.action == "exchange_grant"
            and predecessor.status == "RECOVERED"
            and isinstance(predecessor.payload, ExchangeGrantPayload)
            and predecessor.payload.parent_mode in {"clarification", "question"}
            and task.state is TaskState.AWAITING_USER_INPUT
        ):
            return self._store.rebind_recovered_exchange_grant_checkpoint(
                predecessor.preparation_id,
                old_generation=predecessor.generation,
                generation=generation,
                project_id=self.project_id,
                session_id=session_id,
                task_id=task_id,
                revision=revision,
            )
        if task.state is not TaskState.INTERRUPTED:
            raise ValueError("only an interrupted task may be resumed")
        if task.continuation_state is None:
            raise RuntimeError("interrupted task has no persisted continuation")
        if (
            predecessor is not None
            and predecessor.action == "exchange_grant"
            and predecessor.status == "RECOVERED"
        ):
            return self._store.resume_recovered_exchange_grant(
                predecessor.preparation_id, generation=predecessor.generation,
            )
        context = self._initial_approval_resume_context(task, predecessor)
        if context is None:
            context = self._context_from_task(task)
        if predecessor is not None and (
            predecessor.pending_context is not None
            or task.continuation_state is TaskState.FABLE_PLANNING
        ):
            context = predecessor.pending_context
        drift = ResumeDriftProjection(
            status="unchanged", summary="Repository drift was checked.", evidence_hashes=(),
        )
        if task.continuation_state is not TaskState.FABLE_PLANNING:
            baseline = self._load_baseline(task)
            delta = self._repository.compare(baseline)
            if delta.unexpected_paths or delta.protected_changed_paths:
                drift = ResumeDriftProjection(
                    status="drifted", summary="Repository drift prevented automatic resume.", evidence_hashes=(),
                )
                failed = self._store.fail_resume_for_drift(
                    project_id=self.project_id,
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    drift_event=drift,
                )
                raise ResumeDriftBlocked(failed)
        return self._store.prepare_resume_action(
            project_id=self.project_id,
            session_id=session_id,
            task_id=task_id,
            revision=revision,
            generation=generation,
            payload=ResumePayload(continuation=context, drift_event=drift),
            previous_preparation_id=None if predecessor is None else predecessor.preparation_id,
        )

    async def run_prepared_action(self, preparation_id: str) -> PreparedActionOutcome:
        record = self._store.prepared_action(preparation_id)
        if record is None:
            raise RuntimeError("prepared action not found")
        if record.project_id != self.project_id:
            raise RuntimeError("prepared action belongs to a different project")
        if record.status == "CLAIMED":
            outcome = self._terminal_retries.get(record.preparation_id)
            if outcome is None and isinstance(record.payload, ExchangeGrantPayload):
                child = self._store.resume_claimed_exchange_grant_checkpoint(
                    record.preparation_id, generation=record.generation,
                )
                await self.answer_directed_question(child)
                return self._persist_terminal_prepared_outcome(
                    record, self._terminal_outcome_after_child(record),
                )
            if outcome is None:
                raise PreparedActionFailed()
            return self._persist_terminal_prepared_outcome(record, outcome)
        if record.status != "PREPARED":
            raise RuntimeError("prepared action is not runnable")
        claimed = self._store.claim_prepared_action(
            record.preparation_id, generation=record.generation,
        )
        task = self._store.get_task(claimed.task_id, claimed.revision)
        try:
            if isinstance(claimed.payload, NewRequestPayload):
                await self._run_planning(task, claimed.payload.text, resume_session_id=None)
            elif isinstance(claimed.payload, ApprovalPayload):
                if claimed.payload.scope is None:
                    await self._start_sol(task)
                else:
                    continuation = claimed.payload.scope.underlying_continuation
                    if continuation is None:
                        raise RuntimeError("scope approval has no exact Sol continuation")
                    await self._resume_sol(task, continuation.prompt)
            elif isinstance(claimed.payload, AnswerPayload):
                await self._run_context(task, claimed.payload.continuation, claimed.payload.answer)
            elif isinstance(claimed.payload, ResumePayload):
                await self._run_context(task, claimed.payload.continuation, None)
            elif isinstance(claimed.payload, ContinuationMessagePayload):
                self._validate_claimed_conversation_action(task, claimed)
                await self._run_context(
                    task, claimed.payload.continuation, claimed.payload.text,
                )
            elif isinstance(claimed.payload, QuestionAnswerPayload):
                self._validate_claimed_conversation_action(task, claimed)
                question = self._store.question(claimed.payload.question_id)
                if (
                    question is None
                    or question.session_id != claimed.session_id
                    or question.task_id != claimed.task_id
                    or question.revision != claimed.revision
                    or question.continuation_generation
                    != claimed.payload.continuation_generation
                    or question.routed_to is not ConversationTarget.USER
                    or question.answer_text != claimed.payload.answer
                    or question.answered_by is not ConversationActor.USER
                    or task.continuation_generation
                    != claimed.payload.continuation_generation + 1
                ):
                    raise RuntimeError("prepared question answer changed")
                await self._run_context(
                    task, claimed.payload.continuation, claimed.payload.answer,
                )
            elif isinstance(claimed.payload, ExchangeGrantPayload):
                self._validate_claimed_conversation_action(task, claimed)
                if task.continuation_generation != claimed.payload.continuation_generation:
                    raise RuntimeError("prepared exchange grant changed")
                if claimed.payload.outer_question_id is not None:
                    outer = self._store.restore_fable_answer_parent_for_retry(
                        session_id=task.session_id, task_id=task.task_id,
                        revision=task.revision,
                        expected_generation=claimed.payload.continuation_generation,
                        outer_question_id=claimed.payload.outer_question_id,
                    )
                    restored = self._store.get_task(task.task_id, task.revision)
                    await self._answer_sol_question_with_fable(
                        restored, outer, claimed.payload.continuation,
                    )
                    return self._persist_terminal_prepared_outcome(
                        claimed, self._terminal_outcome_after_child(claimed),
                    )
                if claimed.payload.parent_mode == "clarification":
                    question_id, request_key = self._directed_question_identifiers(
                        task, ConversationActor.FABLE, claimed.payload.attempted_question,
                    )
                    _, question = self._store.reserve_fable_clarification_evidence_question(
                        session_id=task.session_id, task_id=task.task_id,
                        revision=task.revision,
                        expected_generation=claimed.payload.continuation_generation,
                        question_id=question_id, request_key=request_key,
                        text=claimed.payload.attempted_question.text,
                        event=ConversationEnvelope(
                            sender=ConversationActor.FABLE, addressed_to=ConversationTarget.SOL,
                            routed_to=ConversationTarget.SOL,
                            message_type=ConversationMessageType.QUESTION,
                            text=claimed.payload.attempted_question.text,
                            task_id=task.task_id, revision=task.revision,
                            continuation_generation=claimed.payload.continuation_generation,
                            question_id=question_id,
                        ),
                    )
                    await self.answer_directed_question(question)
                    return self._persist_terminal_prepared_outcome(
                        claimed, self._terminal_outcome_after_child(claimed),
                    )
                await self._route_directed_question(
                    task,
                    self._actor_for_context(claimed.payload.continuation),
                    claimed.payload.attempted_question,
                    continuation=claimed.payload.continuation,
                )
            else:
                raise RuntimeError("prepared action payload is invalid")
        except asyncio.CancelledError:
            outcome = self._terminal_outcome_after_error(claimed)
            self._persist_terminal_prepared_outcome(claimed, outcome)
            raise asyncio.CancelledError() from None
        except BaseException as error:
            outcome = self._terminal_outcome_after_error(claimed)
            self._persist_terminal_prepared_outcome(claimed, outcome)
            if claimed.generation == COMPATIBILITY_PREPARATION_GENERATION:
                self._compatibility_errors[claimed.preparation_id] = (
                    self._bounded_compatibility_error(sys.exception())
                )
            raise PreparedActionFailed() from None
        return self._persist_terminal_prepared_outcome(
            claimed, self._terminal_outcome_after_child(claimed),
        )

    async def run_prepared_conversation_action(self, task_id: str, action: str) -> None:
        """Reload and run the latest exact typed conversation preparation only."""
        if action not in _CONVERSATION_PREPARED_ACTIONS:
            raise ValueError("conversation prepared action is invalid")
        task = self._latest_required(task_id)
        record = self._store.latest_prepared_action_for_task(
            project_id=self.project_id,
            session_id=task.session_id,
            task_id=task.task_id,
            revision=task.revision,
        )
        if record is None or record.action != action:
            raise RuntimeError("prepared conversation action not found")
        await self.run_prepared_action(record.preparation_id)

    def _validate_claimed_conversation_action(
        self, task: TaskRecord, record: PreparedActionRecord,
    ) -> None:
        """Recheck persisted state after claim before another agent may start."""
        if (
            record.action not in _CONVERSATION_PREPARED_ACTIONS
            or task.task_id != record.task_id
            or task.revision != record.revision
            or task.session_id != record.session_id
            or task.state is not record.active_state
            or self._latest_required(task.task_id).revision != task.revision
        ):
            raise RuntimeError("prepared conversation action changed")
        payload = record.payload
        context = (
            payload.continuation
            if isinstance(
                payload,
                (ContinuationMessagePayload, QuestionAnswerPayload, ExchangeGrantPayload),
            )
            else None
        )
        if not self._context_matches_active_task(task, context):
            if not (
                isinstance(payload, ExchangeGrantPayload)
                and payload.outer_question_id is not None
                and task.state is TaskState.FABLE_CLARIFYING
                and isinstance(context, SolResumeContext)
            ):
                raise RuntimeError("prepared conversation continuation changed")
        if isinstance(payload, ContinuationMessagePayload):
            if (
                task.continuation_generation != payload.continuation_generation
                or self._target_for_context(payload.continuation) is not payload.routed_to
            ):
                raise RuntimeError("prepared continuation message changed")

    @staticmethod
    def _target_for_context(context: object) -> ConversationTarget:
        while isinstance(context, AnswerContext):
            context = context.underlying_continuation
        if isinstance(context, (ScopeApprovalContext, SolResumeContext)):
            return ConversationTarget.SOL
        if isinstance(context, (ReviewContext, ClarificationContext)):
            return ConversationTarget.FABLE
        raise RuntimeError("prepared continuation has no exact route")

    @classmethod
    def _actor_for_context(cls, context: object) -> ConversationActor:
        target = cls._target_for_context(context)
        if target is ConversationTarget.FABLE:
            return ConversationActor.FABLE
        if target is ConversationTarget.SOL:
            return ConversationActor.SOL
        raise RuntimeError("prepared continuation has no answering agent")

    @staticmethod
    def _context_matches_active_task(task: TaskRecord, context: object) -> bool:
        while isinstance(context, AnswerContext):
            context = context.underlying_continuation
        if isinstance(context, SolResumeContext):
            return (
                task.state in _SOL_STATES
                and task.sol_thread_id == context.sol_thread_id
            )
        if isinstance(context, ScopeApprovalContext):
            if task.state not in _SOL_STATES:
                return False
            if task.baseline_id != context.baseline_id or task.revision != context.approved_revision:
                return False
            return (
                context.underlying_continuation is None
                or task.sol_thread_id == context.underlying_continuation.sol_thread_id
            )
        if isinstance(context, ClarificationContext):
            return (
                task.state is TaskState.FABLE_CLARIFYING
                and task.fable_session_id == context.fable_session_id
            )
        if isinstance(context, ReviewContext):
            return (
                task.state is TaskState.FABLE_REVIEWING
                and task.fable_session_id == context.fable_session_id
            )
        return False

    def _terminal_outcome_after_child(
        self, claimed: PreparedActionRecord,
    ) -> PreparedActionOutcome:
        current = self._store.get_task(claimed.task_id, claimed.revision)
        if current.state is TaskState.INTERRUPTED:
            terminal = self._store.prepared_action(claimed.preparation_id)
            if terminal is not None and terminal.status == "INTERRUPTED":
                if terminal.reason == "stop":
                    return PreparedActionOutcome("stop")
                return PreparedActionOutcome("adapter_interrupted")
            return PreparedActionOutcome("adapter_interrupted")
        return PreparedActionOutcome("completed")

    def _terminal_outcome_after_error(
        self, claimed: PreparedActionRecord,
    ) -> PreparedActionOutcome:
        current = self._store.get_task(claimed.task_id, claimed.revision)
        if current.state is TaskState.INTERRUPTED:
            terminal = self._store.prepared_action(claimed.preparation_id)
            if terminal is not None and terminal.status == "INTERRUPTED":
                if terminal.reason == "stop":
                    return PreparedActionOutcome("stop")
                return PreparedActionOutcome("adapter_interrupted")
            return PreparedActionOutcome("adapter_interrupted")
        return PreparedActionOutcome("nonresumable_failure")

    def _persist_terminal_prepared_outcome(
        self,
        claimed: PreparedActionRecord,
        outcome: PreparedActionOutcome,
    ) -> PreparedActionOutcome:
        """Persist a fixed terminal outcome without invoking a child again."""
        try:
            current = self._store.prepared_action(claimed.preparation_id)
            if current is None:
                raise RuntimeError("prepared action not found")
            if current.status in {"COMPLETED", "FAILED", "INTERRUPTED"}:
                if current.status == "INTERRUPTED":
                    if current.reason == "stop":
                        return PreparedActionOutcome("stop")
                    return PreparedActionOutcome("adapter_interrupted")
                if current.status == "FAILED":
                    return PreparedActionOutcome("nonresumable_failure")
                return PreparedActionOutcome("completed")
            if current.status != "CLAIMED":
                raise RuntimeError("prepared action terminal state changed")
            if outcome.category == "completed":
                self._store.complete_prepared_action(
                    current.preparation_id, generation=current.generation,
                )
            elif outcome.category == "nonresumable_failure":
                self._store.fail_prepared_action(
                    current.preparation_id,
                    generation=current.generation,
                    reason="nonresumable_failure",
                )
            else:
                self._store.interrupt_claimed_prepared_action(
                    current.preparation_id,
                    generation=current.generation,
                    reason=(
                        "stop" if outcome.category == "stop" else "adapter_interrupted"
                    ),
                )
        except BaseException:
            self._terminal_retries[claimed.preparation_id] = outcome
            raise PreparedActionFailed() from None
        self._terminal_retries.pop(claimed.preparation_id, None)
        return outcome

    def _bounded_compatibility_error(self, error: BaseException) -> RuntimeError:
        """Keep old wrapper exception types without exposing provider text."""
        if isinstance(error, SubscriptionAuthError) and isinstance(self._fable, ClaudeCLI):
            return SubscriptionAuthError("subscription authentication is unavailable")
        if isinstance(error, ClaudeRunError) and isinstance(self._fable, ClaudeCLI):
            return ClaudeRunError("Fable did not return a valid contract")
        if isinstance(error, CodexRunError) and isinstance(self._sol, CodexCLI):
            return CodexRunError("Sol run ended with a non-zero or invalid result")
        return PreparedActionFailed()

    async def _run_compatibility_prepared(self, preparation_id: str) -> None:
        try:
            await self.run_prepared_action(preparation_id)
        except PreparedActionFailed:
            error = self._compatibility_errors.pop(preparation_id, None)
            if error is not None:
                raise error from None
            raise

    async def run_prepared_request(self, task_id: str) -> None:
        """Compatibility wrapper that runs the local generation-zero record."""
        task = self._latest_required(task_id)
        record = self._store.latest_prepared_action_for_task(
            project_id=self.project_id,
            session_id=task.session_id,
            task_id=task.task_id,
            revision=task.revision,
        )
        if record is None or record.generation != COMPATIBILITY_PREPARATION_GENERATION:
            raise ValueError("task is not a prepared initial request")
        await self._run_compatibility_prepared(record.preparation_id)

    def interrupt_claimed_prepared_action(
        self, preparation_id: str, *, generation: int,
        reason: PreparedActionInterruptionReason,
    ) -> PreparedActionRecord:
        return self._store.interrupt_claimed_prepared_action(
            preparation_id, generation=generation, reason=reason,
        )

    def abort_prepared_action(
        self,
        preparation_id: str,
        *legacy: object,
        generation: int | None = None,
        reason: str | None = None,
    ) -> PreparedActionRecord | TaskRecord:
        if legacy:
            if len(legacy) != 3:
                raise TypeError("legacy abort requires revision, action, and reason")
            revision, action, legacy_reason = legacy
            if not isinstance(revision, int) or not isinstance(action, str) or not isinstance(legacy_reason, str):
                raise ValueError("legacy abort arguments are invalid")
            task = self._store.get_task(preparation_id, revision)
            record = self._store.latest_prepared_action_for_task(
                project_id=self.project_id,
                session_id=task.session_id,
                task_id=task.task_id,
                revision=task.revision,
            )
            if (
                record is None
                or record.generation != COMPATIBILITY_PREPARATION_GENERATION
                or record.action != action
            ):
                raise RuntimeError("task is not a compatible prepared action")
            self._store.abort_prepared_action(
                record.preparation_id,
                generation=COMPATIBILITY_PREPARATION_GENERATION,
                reason=legacy_reason,
            )
            return self._store.get_task(task.task_id, task.revision)
        if generation is None or reason is None:
            raise TypeError("prepared abort requires generation and reason")
        return self._store.abort_prepared_action(
            preparation_id, generation=generation, reason=reason,
        )

    def recover_unfinished_prepared_actions(self) -> RecoverySummary:
        return self._store.recover_unfinished_prepared_actions()

    def _sol_context(self, task: TaskRecord, prompt: str) -> SolResumeContext:
        if not isinstance(task.sol_thread_id, str):
            raise RuntimeError("Sol continuation is missing the exact thread ID")
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError("Sol continuation is missing the exact prompt")
        run_id = (task.pending or {}).get("sol_run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("Sol continuation is missing the exact Sol run ID")
        return SolResumeContext(
            sol_thread_id=task.sol_thread_id,
            sol_run_id=run_id,
            prompt=prompt,
        )

    def _stored_sol_context(self, task: TaskRecord) -> SolResumeContext:
        """Return the exact Sol continuation retained with a paused Fable step."""
        return self._sol_context(task, (task.pending or {}).get("prompt"))

    @staticmethod
    def _initial_approval_resume_context(
        task: TaskRecord,
        predecessor: PreparedActionRecord | None,
    ) -> ScopeApprovalContext | None:
        """Rebuild a pre-thread initial approval from its durable action payload."""
        if (
            predecessor is None
            or predecessor.action != "approval"
            or not isinstance(predecessor.payload, ApprovalPayload)
            or predecessor.payload.scope is not None
            or task.continuation_state is not TaskState.SOL_RUNNING
            or task.sol_thread_id is not None
            or task.baseline_id != predecessor.payload.baseline_id
        ):
            return None
        projection = (task.pending or {}).get("prepared_action")
        if (
            not isinstance(projection, Mapping)
            or set(projection) != {"preparation_id", "action", "reason", "context"}
            or projection.get("preparation_id") != predecessor.preparation_id
            or projection.get("action") != "approval"
            or projection.get("context") is not None
            or not isinstance(projection.get("reason"), str)
        ):
            return None
        return ScopeApprovalContext(
            baseline_id=predecessor.payload.baseline_id,
            approved_revision=task.revision,
            underlying_continuation=None,
        )

    @staticmethod
    def _scope_context(task: TaskRecord) -> ScopeApprovalContext:
        if task.baseline_id is None or task.brief is None:
            raise RuntimeError("continuation is missing the exact approved baseline")
        return ScopeApprovalContext(
            baseline_id=task.baseline_id,
            approved_revision=task.revision,
            underlying_continuation=None,
        )

    def _context_from_task(self, task: TaskRecord):
        continuation = task.continuation_state
        pending = task.pending or {}
        if continuation is TaskState.FABLE_PLANNING:
            return None
        if continuation is TaskState.FABLE_REVIEWING:
            if not isinstance(task.fable_session_id, str):
                raise RuntimeError("review continuation is missing the exact Fable session")
            prompt = pending.get("review_prompt")
            completion_allowed = pending.get("completion_allowed")
            if not isinstance(prompt, str) or not isinstance(completion_allowed, bool):
                raise RuntimeError("review continuation is missing exact context")
            return ReviewContext(
                fable_session_id=task.fable_session_id,
                review_prompt=prompt,
                completion_allowed=completion_allowed,
                underlying_continuation=self._scope_context(task),
            )
        if continuation is TaskState.FABLE_CLARIFYING:
            if not isinstance(task.fable_session_id, str):
                raise RuntimeError("clarification continuation is missing the exact Fable session")
            prompt = pending.get("clarification_prompt")
            if not isinstance(prompt, str):
                raise RuntimeError("clarification continuation is missing exact context")
            return ClarificationContext(
                fable_session_id=task.fable_session_id,
                clarification_prompt=prompt,
                underlying_continuation=self._stored_sol_context(task),
            )
        prompt = pending.get("prompt")
        if task.sol_thread_id is None and not pending:
            return self._scope_context(task)
        return self._sol_context(task, prompt)

    async def _run_context(
        self,
        task: TaskRecord,
        context: object,
        answer: str | None,
        *,
        answer_source: Literal["user", "sol"] = "user",
    ) -> None:
        if isinstance(context, AnswerContext):
            await self._run_context(
                task, context.underlying_continuation, context.answer,
                answer_source=answer_source,
            )
            return
        if isinstance(context, SolResumeContext):
            await self._resume_sol(task, answer if answer is not None else context.prompt)
            return
        if isinstance(context, ScopeApprovalContext):
            if context.underlying_continuation is None:
                await self._start_sol(task)
            else:
                await self._resume_sol(
                    task,
                    answer if answer is not None else context.underlying_continuation.prompt,
                )
            return
        if isinstance(context, ReviewContext):
            prompt = context.review_prompt
            if answer is not None:
                prompt = f"{prompt}\n{'User answer' if answer_source == 'user' else 'Sol evidence'}: {answer}"
            await self._call_fable_review(
                task, prompt, completion_allowed=context.completion_allowed,
            )
            return
        if isinstance(context, ClarificationContext):
            prompt = context.clarification_prompt
            if answer is not None:
                prompt = f"{prompt}\n{'User answer' if answer_source == 'user' else 'Sol evidence'}: {answer}"
            underlying = context.underlying_continuation
            if not isinstance(underlying, SolResumeContext):
                raise RuntimeError("clarification has no exact Sol continuation")
            await self._call_fable_clarification(
                task, prompt, underlying_continuation=underlying,
            )
            return
        if context is None:
            await self._run_planning(
                task,
                self._original_user_message(task.session_id, task.task_id),
                resume_session_id=task.fable_session_id,
            )
            return
        raise RuntimeError("prepared continuation is invalid")

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
            prepared = self.prepare_approval(
                session_id=task.session_id,
                task_id=task_id,
                revision=revision,
                generation=COMPATIBILITY_PREPARATION_GENERATION,
            )
        await self._run_compatibility_prepared(prepared.preparation_id)

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
            pending = task.pending
            if task.continuation_state in _SOL_STATES and pending is not None:
                answer = pending.get("answer")
                if isinstance(answer, str):
                    pending = {
                        **pending,
                        "prompt": self._scope_resume_prompt_for_brief(edited, answer),
                    }
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
                pending=pending,
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
        if self._store.unanswered_question_for_task(task.task_id, task.revision) is not None:
            raise ValueError("exact directed question answer is required")
        prepared = self.prepare_answer(
            session_id=task.session_id,
            task_id=task.task_id,
            revision=task.revision,
            answer=answer,
            generation=COMPATIBILITY_PREPARATION_GENERATION,
        )
        await self._run_compatibility_prepared(prepared.preparation_id)

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
        try:
            prepared = self.prepare_resume(
                session_id=task.session_id,
                task_id=task.task_id,
                revision=task.revision,
                generation=COMPATIBILITY_PREPARATION_GENERATION,
            )
        except ResumeDriftBlocked:
            return
        await self._run_compatibility_prepared(prepared.preparation_id)

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
            persisted_prompt = (
                resume_prompt
                if resume_prompt is not None
                else self._original_user_message(task.session_id, task.task_id)
            )
            task = self._store.set_pending_context(
                task.task_id,
                task.revision,
                expected=task.state,
                pending={"sol_run_id": run_id, "prompt": persisted_prompt},
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

    async def _route_directed_question(
        self,
        task: TaskRecord,
        asked_by: ConversationActor,
        directed: DirectedAgentQuestion,
        *,
        continuation: object | None = None,
    ) -> None:
        """Persist an exact visible directed question before any answering call."""
        if not isinstance(asked_by, ConversationActor) or asked_by not in {
            ConversationActor.FABLE,
            ConversationActor.SOL,
        }:
            raise ValueError("directed questions must be asked by an agent")
        if not isinstance(directed, DirectedAgentQuestion):
            raise ValueError("directed question is invalid")
        current = self._store.get_task(task.task_id, task.revision)
        if (
            current.session_id != task.session_id
            or current.state not in _ACTIVE_STATES
            or current.continuation_state is not None
            or current.approved_at is None
            or current.baseline_id is None
        ):
            raise RuntimeError("directed question continuation changed")
        if continuation is None:
            continuation = self._active_context_from_task(current)
        if not self._context_matches_active_task(current, continuation):
            raise RuntimeError("directed question continuation changed")
        target = ConversationTarget(directed.addressed_to)
        if target is ConversationTarget.USER:
            question_id, _ = self._directed_question_identifiers(
                current, asked_by, directed,
            )
            question = self._store.pause_for_question(
                session_id=current.session_id,
                task_id=current.task_id,
                revision=current.revision,
                expected_generation=current.continuation_generation,
                question_id=question_id,
                asked_by=asked_by,
                addressed_to=ConversationTarget.USER,
                routed_to=ConversationTarget.USER,
                text=directed.text,
                continuation_state=current.state,
                pending_action=self._pending_for_directed_context(
                    current, continuation,
                ),
                event=ConversationEnvelope(
                    sender=asked_by,
                    addressed_to=ConversationTarget.USER,
                    routed_to=ConversationTarget.USER,
                    message_type=ConversationMessageType.QUESTION,
                    text=directed.text,
                    task_id=current.task_id,
                    revision=current.revision,
                    continuation_generation=current.continuation_generation,
                    question_id=question_id,
                ),
            )
            self._emit_state(self._store.get_task(current.task_id, current.revision))
            if question.routed_to is not ConversationTarget.USER:
                raise RuntimeError("directed user question route changed")
            return
        if target is ConversationTarget.TEAM:
            raise RuntimeError("directed question has no exact agent recipient")
        if target.value == asked_by.value:
            raise RuntimeError("directed question cannot route to its asking agent")
        if current.exchange_allowance <= 0:
            paused = self._store.pause_for_exchange_permission(
                session_id=current.session_id,
                task_id=current.task_id,
                revision=current.revision,
                expected_generation=current.continuation_generation,
                attempted_question=directed,
                continuation_state=current.state,
                pending_action=self._pending_for_directed_context(
                    current, continuation,
                ),
                event=ConversationEnvelope(
                    sender=ConversationActor.SYSTEM,
                    addressed_to=ConversationTarget.USER,
                    routed_to=ConversationTarget.USER,
                    message_type=ConversationMessageType.STATUS,
                    text=_EXCHANGE_PERMISSION_TEXT,
                    task_id=current.task_id,
                    revision=current.revision,
                    continuation_generation=current.continuation_generation,
                ),
            )
            self._emit_state(paused)
            return
        question_id, request_key = self._directed_question_identifiers(
            current, asked_by, directed,
        )
        _, question = self._store.reserve_internal_question(
            session_id=current.session_id,
            task_id=current.task_id,
            revision=current.revision,
            expected_generation=current.continuation_generation,
            question_id=question_id,
            request_key=request_key,
            asked_by=asked_by,
            addressed_to=target,
            routed_to=target,
            text=directed.text,
            continuation_state=current.state,
            pending_action=self._pending_for_directed_context(current, continuation),
            event=ConversationEnvelope(
                sender=asked_by,
                addressed_to=target,
                routed_to=target,
                message_type=ConversationMessageType.QUESTION,
                text=directed.text,
                task_id=current.task_id,
                revision=current.revision,
                continuation_generation=current.continuation_generation,
                question_id=question_id,
            ),
        )
        self._emit_state(self._store.get_task(current.task_id, current.revision))
        await self.answer_directed_question(question)

    def _active_context_from_task(self, task: TaskRecord) -> object:
        pending = task.pending or {}
        if task.state in _SOL_STATES:
            prompt = pending.get("prompt")
            return self._sol_context(task, prompt)
        if task.state is TaskState.FABLE_CLARIFYING:
            if not isinstance(task.fable_session_id, str):
                raise RuntimeError("clarification is missing the exact Fable session")
            prompt = pending.get("clarification_prompt")
            if not isinstance(prompt, str):
                raise RuntimeError("clarification is missing its exact prompt")
            return ClarificationContext(
                fable_session_id=task.fable_session_id,
                clarification_prompt=prompt,
                underlying_continuation=self._stored_sol_context(task),
            )
        if task.state is TaskState.FABLE_REVIEWING:
            if not isinstance(task.fable_session_id, str):
                raise RuntimeError("review is missing the exact Fable session")
            prompt = pending.get("review_prompt")
            completion_allowed = pending.get("completion_allowed")
            if not isinstance(prompt, str) or not isinstance(completion_allowed, bool):
                raise RuntimeError("review is missing its exact context")
            return ReviewContext(
                fable_session_id=task.fable_session_id,
                review_prompt=prompt,
                completion_allowed=completion_allowed,
                underlying_continuation=self._scope_context(task),
            )
        raise RuntimeError("directed question has no active continuation")

    @staticmethod
    def _directed_question_identifiers(
        task: TaskRecord,
        asked_by: ConversationActor,
        directed: DirectedAgentQuestion,
    ) -> tuple[str, str]:
        """Make reservation retries reuse one opaque ID without exposing text."""
        seed = json.dumps(
            {
                "task_id": task.task_id,
                "revision": task.revision,
                "continuation_generation": task.continuation_generation,
                "exchange_ordinal": task.exchange_consumed + 1,
                "asked_by": asked_by.value,
                "addressed_to": directed.addressed_to,
                "text": directed.text,
                "reason": directed.reason,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        return f"question-{digest[:48]}", f"exchange-{digest[:48]}"

    @staticmethod
    def _pending_for_directed_context(
        task: TaskRecord, context: object,
    ) -> Mapping[str, object]:
        """Reuse the persisted pending snapshot, rebuilding only grant resumes."""
        if task.pending is not None:
            return dict(task.pending)
        while isinstance(context, AnswerContext):
            context = context.underlying_continuation
        if isinstance(context, SolResumeContext):
            return {"sol_run_id": context.sol_run_id, "prompt": context.prompt}
        if isinstance(context, ScopeApprovalContext):
            if context.underlying_continuation is None:
                return {}
            return {
                "sol_run_id": context.underlying_continuation.sol_run_id,
                "prompt": context.underlying_continuation.prompt,
            }
        if isinstance(context, ClarificationContext):
            underlying = context.underlying_continuation
            if not isinstance(underlying, SolResumeContext):
                raise RuntimeError("clarification has no exact Sol continuation")
            return {
                "clarification_prompt": context.clarification_prompt,
                "sol_run_id": underlying.sol_run_id,
                "prompt": underlying.prompt,
            }
        if isinstance(context, ReviewContext):
            return {
                "review_prompt": context.review_prompt,
                "completion_allowed": context.completion_allowed,
            }
        raise RuntimeError("directed question has no persisted continuation")

    async def answer_directed_question(self, question: QuestionRecord) -> None:
        """Answer one persisted agent-routed question; user questions stay human-only."""
        if not isinstance(question, QuestionRecord):
            raise ValueError("question must be a QuestionRecord")
        persisted = self._store.question(question.question_id)
        if persisted is None or persisted != question:
            raise RuntimeError("directed question changed")
        if persisted.routed_to is ConversationTarget.USER:
            raise RuntimeError("user-routed questions require an exact user answer")
        if persisted.answer_text is not None or persisted.answered_by is not None:
            raise RuntimeError("directed question was already answered")
        task = self._store.get_task(persisted.task_id, persisted.revision)
        if (
            task.session_id != persisted.session_id
            or task.state is not TaskState.AWAITING_USER_INPUT
            or task.continuation_state is None
            or task.continuation_generation != persisted.continuation_generation
            or task.approved_at is None
            or task.baseline_id is None
        ):
            raise RuntimeError("directed question continuation changed")
        continuation = (
            None
            if persisted.nested_parent_kind == "question"
            else self._context_from_task(task)
        )
        if persisted.routed_to is ConversationTarget.FABLE:
            if persisted.asked_by is not ConversationActor.SOL:
                raise RuntimeError("Fable may answer only Sol's exact question")
            await self._answer_sol_question_with_fable(task, persisted, continuation)
            return
        if persisted.routed_to is ConversationTarget.SOL:
            if persisted.asked_by is not ConversationActor.FABLE:
                raise RuntimeError("Sol may answer only Fable's exact question")
            await self._answer_fable_question_with_sol(task, persisted, continuation)
            return
        raise RuntimeError("directed question has no exact answering agent")

    async def _answer_sol_question_with_fable(
        self,
        task: TaskRecord,
        question: QuestionRecord,
        continuation: object,
    ) -> None:
        if task.fable_session_id is None:
            raise RuntimeError("Fable answer requires the exact Fable session")
        evidence = self._store.nested_evidence_for_parent(question.question_id)
        prompt = question.text if evidence is None else f"{question.text}\nSol evidence: {evidence}"
        run_id = self._ids.new_run_id()
        self._store.start_agent_run(run_id, task.task_id, task.revision, "fable")
        completion = self._track_run(run_id)
        try:
            result = await self._fable.answer_sol_question(
                run_id=run_id,
                session_id=task.fable_session_id,
                task_id=task.task_id,
                prompt=prompt,
                context=self._repo_context(),
            )
            self._persist_agent_result(
                task,
                "fable",
                result,
                run_id=run_id,
                expected_cli_session_id=task.fable_session_id,
            )
            if result.interrupted:
                self._finish_interrupted_run(run_id, exit_code=result.exit_code)
                return
            if result.payload is None:
                raise RuntimeError("Fable answer completed without a contract")
            clarification = FableClarification.from_dict(result.payload)
            if clarification.status != "answered" or clarification.answer is None:
                raise RuntimeError("Fable did not answer Sol's exact question")
            self._store.finish_agent_run(
                run_id, status="completed", exit_code=result.exit_code,
            )
        except asyncio.CancelledError:
            self._finish_interrupted_run(run_id, exit_code=-1)
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
        if clarification.directed_question is not None and not clarification.scope_changed:
            task = self._store.get_task(task.task_id, task.revision)
            directed = clarification.directed_question
            if directed.addressed_to != ConversationTarget.SOL.value:
                raise RuntimeError("Fable evidence must target exact Sol")
            if task.exchange_allowance <= 0:
                paused = self._store.pause_fable_answer_evidence_permission(
                    session_id=task.session_id, task_id=task.task_id, revision=task.revision,
                    expected_generation=question.continuation_generation,
                    outer_question_id=question.question_id, attempted_question=directed,
                    event=ConversationEnvelope(
                        sender=ConversationActor.SYSTEM, addressed_to=ConversationTarget.USER,
                        routed_to=ConversationTarget.USER, message_type=ConversationMessageType.STATUS,
                        text=_EXCHANGE_PERMISSION_TEXT, task_id=task.task_id,
                        revision=task.revision, continuation_generation=question.continuation_generation,
                    ),
                )
                self._emit_state(paused)
                return
            question_id, request_key = self._directed_question_identifiers(
                task, ConversationActor.FABLE, directed,
            )
            _, nested = self._store.reserve_fable_answer_evidence_question(
                session_id=task.session_id, task_id=task.task_id, revision=task.revision,
                expected_generation=question.continuation_generation,
                outer_question_id=question.question_id, question_id=question_id,
                request_key=request_key, text=directed.text,
                event=ConversationEnvelope(
                    sender=ConversationActor.FABLE, addressed_to=ConversationTarget.SOL,
                    routed_to=ConversationTarget.SOL,
                    message_type=ConversationMessageType.QUESTION, text=directed.text,
                    task_id=task.task_id, revision=task.revision,
                    continuation_generation=question.continuation_generation,
                    question_id=question_id,
                ),
            )
            self._emit_state(self._store.get_task(task.task_id, task.revision))
            await self.answer_directed_question(nested)
            return
        answered = self._store.answer_question_and_prepare_resume(
            session_id=task.session_id,
            task_id=task.task_id,
            revision=task.revision,
            question_id=question.question_id,
            expected_generation=question.continuation_generation,
            answer_text=clarification.answer,
            answered_by=ConversationActor.FABLE,
            pending_action=self._pending_for_directed_context(task, continuation),
            event=ConversationEnvelope(
                sender=ConversationActor.FABLE,
                addressed_to=ConversationTarget.SOL,
                routed_to=ConversationTarget.SOL,
                message_type=ConversationMessageType.ANSWER,
                text=clarification.answer,
                task_id=task.task_id,
                revision=task.revision,
                continuation_generation=question.continuation_generation,
                reply_to_question_id=question.question_id,
            ),
        )
        if answered.answered_by is not ConversationActor.FABLE:
            raise RuntimeError("directed Fable answer changed")
        resumed = self._store.get_task(task.task_id, task.revision)
        await self._route_directed_fable_answer(resumed, clarification)

    async def _route_directed_fable_answer(
        self, task: TaskRecord, clarification: FableClarification,
    ) -> None:
        current = self._store.get_task(task.task_id, task.revision)
        if current.state not in _SOL_STATES or current.approved_at is None:
            raise RuntimeError("directed Fable answer continuation changed")
        if clarification.scope_changed:
            revised = clarification.revised_brief
            if revised is None or (
                revised.task_id != current.task_id
                or revised.revision != current.revision + 1
            ):
                raise ValueError("scope change must create the exact next task revision")
            if current.fable_session_id is None or current.sol_thread_id is None:
                raise RuntimeError("scope change is missing exact agent continuation IDs")
            sol_run_id = (current.pending or {}).get("sol_run_id")
            if not isinstance(sol_run_id, str):
                raise RuntimeError("scope change is missing the exact Sol run ID")
            original_baseline = self._load_baseline(current)
            widened_baseline = self._repository.widen_baseline(original_baseline, revised)
            try:
                saved = self._store.save_scope_revision(
                    current.session_id,
                    revised,
                    fable_session_id=current.fable_session_id,
                    sol_thread_id=current.sol_thread_id,
                    correction_count=current.correction_count,
                    continuation_state=current.state,
                    pending={
                        "answer": clarification.answer,
                        "sol_run_id": sol_run_id,
                        "prompt": self._scope_resume_prompt_for_brief(
                            revised, clarification.answer or "",
                        ),
                    },
                    baseline_id=original_baseline.baseline_id,
                    setting=(
                        self._baseline_key(current.task_id, revised.revision),
                        self._baseline_setting_value(
                            current.task_id, revised.revision, widened_baseline,
                        ),
                    ),
                )
            except BaseException:
                self._repository.discard_widening(original_baseline, widened_baseline)
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
        resumed = self._store.clear_pending_context(
            current.task_id,
            current.revision,
            expected=current.state,
        )
        self._emit_clarification(resumed, clarification)
        await self._resume_sol(resumed, clarification.answer or "")

    async def _answer_fable_question_with_sol(
        self,
        task: TaskRecord,
        question: QuestionRecord,
        continuation: object,
    ) -> None:
        if task.sol_thread_id is None or task.brief is None:
            raise RuntimeError("Sol answer requires the exact approved thread and brief")
        run_id = self._ids.new_run_id()
        self._store.start_agent_run(run_id, task.task_id, task.revision, "sol")
        completion = self._track_run(run_id)
        try:
            result = await self._sol.answer_fable_question(
                run_id=run_id,
                thread_id=task.sol_thread_id,
                brief=task.brief,
                prompt=question.text,
            )
            self._persist_agent_result(
                task,
                "sol",
                result,
                run_id=run_id,
                expected_cli_session_id=task.sol_thread_id,
            )
            if result.interrupted:
                self._finish_interrupted_run(run_id, exit_code=result.exit_code)
                return
            if result.payload is None:
                raise RuntimeError("Sol answer completed without a contract")
            outcome = SolOutcome.from_dict(result.payload)
            if outcome.status != "completed" or outcome.question is not None:
                raise RuntimeError("Sol did not answer Fable's exact question")
            self._store.finish_agent_run(
                run_id, status="completed", exit_code=result.exit_code,
            )
        except asyncio.CancelledError:
            self._finish_interrupted_run(run_id, exit_code=-1)
            raise
        except BaseException as error:
            self._record_agent_failure(
                task,
                "sol",
                run_id,
                error,
                expected_cli_session_id=task.sol_thread_id,
            )
            raise
        finally:
            self._complete_run(run_id, completion)
        if question.nested_parent_kind == "clarification":
            answered = self._store.answer_fable_clarification_evidence_question_and_resume(
                session_id=task.session_id, task_id=task.task_id, revision=task.revision,
                question_id=question.question_id,
                expected_generation=question.continuation_generation,
                answer_text=outcome.summary,
                event=ConversationEnvelope(
                    sender=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
                    routed_to=ConversationTarget.FABLE,
                    message_type=ConversationMessageType.ANSWER, text=outcome.summary,
                    task_id=task.task_id, revision=task.revision,
                    continuation_generation=question.continuation_generation,
                    reply_to_question_id=question.question_id,
                ),
            )
            if answered.answered_by is not ConversationActor.SOL:
                raise RuntimeError("nested directed Sol answer changed")
            resumed = self._store.get_task(task.task_id, task.revision)
            await self._run_context(
                resumed, continuation, outcome.summary, answer_source="sol",
            )
            return
        if question.nested_parent_kind == "question":
            if question.parent_question_id is None:
                raise RuntimeError("nested Fable evidence parent is missing")
            answered = self._store.answer_fable_answer_evidence_question(
                session_id=task.session_id, task_id=task.task_id, revision=task.revision,
                outer_question_id=question.parent_question_id, question_id=question.question_id,
                expected_generation=question.continuation_generation,
                answer_text=outcome.summary,
                event=ConversationEnvelope(
                    sender=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
                    routed_to=ConversationTarget.FABLE,
                    message_type=ConversationMessageType.ANSWER, text=outcome.summary,
                    task_id=task.task_id, revision=task.revision,
                    continuation_generation=question.continuation_generation,
                    reply_to_question_id=question.question_id,
                ),
            )
            if answered.answered_by is not ConversationActor.SOL:
                raise RuntimeError("nested directed Sol answer changed")
            outer = self._store.question(question.parent_question_id)
            if outer is None:
                raise RuntimeError("nested Fable evidence parent changed")
            resumed = self._store.get_task(task.task_id, task.revision)
            await self._answer_sol_question_with_fable(
                resumed, outer, self._context_from_task(resumed),
            )
            return
        answered = self._store.answer_question_and_prepare_resume(
            session_id=task.session_id,
            task_id=task.task_id,
            revision=task.revision,
            question_id=question.question_id,
            expected_generation=question.continuation_generation,
            answer_text=outcome.summary,
            answered_by=ConversationActor.SOL,
            pending_action=self._pending_for_directed_context(task, continuation),
            event=ConversationEnvelope(
                sender=ConversationActor.SOL,
                addressed_to=ConversationTarget.FABLE,
                routed_to=ConversationTarget.FABLE,
                message_type=ConversationMessageType.ANSWER,
                text=outcome.summary,
                task_id=task.task_id,
                revision=task.revision,
                continuation_generation=question.continuation_generation,
                reply_to_question_id=question.question_id,
            ),
        )
        if answered.answered_by is not ConversationActor.SOL:
            raise RuntimeError("directed Sol answer changed")
        resumed = self._store.get_task(task.task_id, task.revision)
        await self._run_context(
            resumed, continuation, outcome.summary, answer_source="sol",
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
            if outcome.question.directed_question is not None:
                self._emit_sol_outcome(task, outcome)
                await self._route_directed_question(
                    task,
                    ConversationActor.SOL,
                    outcome.question.directed_question,
                )
                return
            clarification_prompt = self._clarification_prompt(task, outcome)
            task = self._store.replace_pending_for_continuation(
                task.task_id,
                task.revision,
                expected=task.state,
                target=TaskState.FABLE_CLARIFYING,
                continuation_state=task.state,
                pending={
                    "clarification_prompt": clarification_prompt,
                    "sol_run_id": (task.pending or {}).get("sol_run_id"),
                    "prompt": (task.pending or {}).get("prompt"),
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
                task = self._store.replace_pending_for_continuation(
                    task.task_id,
                    task.revision,
                    expected=task.state,
                    target=target,
                    continuation_state=task.state,
                    pending={
                        "sol_status": outcome.status,
                        "sol_run_id": (task.pending or {}).get("sol_run_id"),
                        "prompt": (task.pending or {}).get("prompt"),
                    },
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
        task = self._store.transition_task_clearing_pending(
            task.task_id,
            task.revision,
            expected=task.state,
            target=TaskState.FABLE_REVIEWING,
        )
        self._emit_state(task)
        self._emit_sol_outcome(task, outcome)
        await self._review(task, outcome, observed_events)

    async def _call_fable_clarification(
        self,
        task: TaskRecord,
        prompt: str,
        *,
        underlying_continuation: SolResumeContext | None = None,
    ) -> None:
        if task.fable_session_id is None:
            raise RuntimeError("clarification requires the exact Fable session")
        if underlying_continuation is not None and task.pending is None:
            task = self._store.set_pending_context(
                task.task_id,
                task.revision,
                expected=TaskState.FABLE_CLARIFYING,
                pending={
                    "clarification_prompt": prompt,
                    "sol_run_id": underlying_continuation.sol_run_id,
                    "prompt": underlying_continuation.prompt,
                },
            )
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
            self._store.get_task(task.task_id, task.revision),
            clarification,
            underlying_continuation=underlying_continuation,
        )

    async def _route_clarification(
        self,
        task: TaskRecord,
        clarification: FableClarification,
        *,
        underlying_continuation: SolResumeContext | None = None,
    ) -> None:
        current = self._store.get_task(task.task_id, task.revision)
        if current.state is not TaskState.FABLE_CLARIFYING:
            return
        task = current
        if clarification.directed_question is not None and not clarification.scope_changed:
            if task.approved_at is None or task.baseline_id is None:
                raise RuntimeError("Fable evidence request requires exact approval")
            self._emit_clarification(task, clarification)
            directed = clarification.directed_question
            if directed.addressed_to != ConversationTarget.SOL.value:
                raise RuntimeError("Fable clarification evidence must target exact Sol")
            question_id, request_key = self._directed_question_identifiers(
                task, ConversationActor.FABLE, directed,
            )
            if task.exchange_allowance <= 0:
                paused = self._store.pause_fable_clarification_evidence_permission(
                    session_id=task.session_id, task_id=task.task_id, revision=task.revision,
                    expected_generation=task.continuation_generation,
                    attempted_question=directed,
                    event=ConversationEnvelope(
                        sender=ConversationActor.SYSTEM, addressed_to=ConversationTarget.USER,
                        routed_to=ConversationTarget.USER,
                        message_type=ConversationMessageType.STATUS,
                        text=_EXCHANGE_PERMISSION_TEXT, task_id=task.task_id,
                        revision=task.revision,
                        continuation_generation=task.continuation_generation,
                    ),
                )
                self._emit_state(paused)
                return
            _, question = self._store.reserve_fable_clarification_evidence_question(
                session_id=task.session_id, task_id=task.task_id, revision=task.revision,
                expected_generation=task.continuation_generation, question_id=question_id,
                request_key=request_key, text=directed.text,
                event=ConversationEnvelope(
                    sender=ConversationActor.FABLE, addressed_to=ConversationTarget.SOL,
                    routed_to=ConversationTarget.SOL,
                    message_type=ConversationMessageType.QUESTION, text=directed.text,
                    task_id=task.task_id, revision=task.revision,
                    continuation_generation=task.continuation_generation,
                    question_id=question_id,
                ),
            )
            self._emit_state(self._store.get_task(task.task_id, task.revision))
            await self.answer_directed_question(question)
            return
        if clarification.status == "escalate_to_user":
            question = clarification.question_for_user
            if question is None:
                raise RuntimeError("Fable escalation is missing its user question")
            pending = {
                "question_for_user": question,
                "clarification_prompt": (task.pending or {}).get(
                    "clarification_prompt"
                ),
                "sol_run_id": (task.pending or {}).get("sol_run_id"),
                "prompt": (task.pending or {}).get("prompt"),
            }
            if underlying_continuation is None:
                task = self._store.retarget_continuation(
                    task.task_id,
                    task.revision,
                    expected=TaskState.FABLE_CLARIFYING,
                    target=TaskState.AWAITING_USER_INPUT,
                    pending=pending,
                )
            else:
                task = self._store.replace_pending_for_continuation(
                    task.task_id,
                    task.revision,
                    expected=TaskState.FABLE_CLARIFYING,
                    target=TaskState.AWAITING_USER_INPUT,
                    continuation_state=TaskState.FABLE_CLARIFYING,
                    pending=pending,
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
            sol_run_id = (task.pending or {}).get("sol_run_id")
            if not isinstance(sol_run_id, str):
                raise RuntimeError("scope change is missing the exact Sol run ID")
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
                    pending={
                        "answer": clarification.answer,
                        "sol_run_id": sol_run_id,
                        "prompt": self._scope_resume_prompt_for_brief(
                            revised, clarification.answer,
                        ),
                    },
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
        if underlying_continuation is None:
            resumed = self._store.resume_continuation(
                task.task_id, task.revision, expected=TaskState.FABLE_CLARIFYING
            )
        else:
            resumed = self._store.transition_task_clearing_pending(
                task.task_id,
                task.revision,
                expected=TaskState.FABLE_CLARIFYING,
                target=TaskState.SOL_RUNNING,
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
        if verdict.directed_question is not None:
            if task.approved_at is None or task.baseline_id is None:
                raise RuntimeError("Fable evidence request requires exact approval")
            self._emit_review(task, verdict)
            await self._route_directed_question(
                task,
                ConversationActor.FABLE,
                verdict.directed_question,
            )
            return
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
        normalized_events: list[Mapping[str, object]] = []
        for event in result.events:
            if len(normalized_events) == MAX_AGENT_STRUCTURAL_EVENTS:
                break
            safe_event = self._structural_agent_event(actor, event)
            if safe_event is not None:
                normalized_events.append(safe_event)
        normalized = tuple(normalized_events)
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
        return Coordinator._scope_resume_prompt_for_brief(task.brief, answer)

    @staticmethod
    def _scope_resume_prompt_for_brief(brief: TaskBrief, answer: str) -> str:
        serialized = json.dumps(brief.to_dict(), separators=(",", ":"), sort_keys=True)
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
