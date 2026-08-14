from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_bridge.adapters.claude_cli import (
    ClaudeAuthFailureCategory,
    SubscriptionAuthError,
)
import agent_bridge.store as store_module
from agent_bridge.adapters.base import AgentRunResult
from agent_bridge.adapters.claude_cli import ClaudeRunError
from agent_bridge.adapters.codex_cli import CodexRunError
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
from agent_bridge.coordinator import (
    Coordinator,
    InterventionIntent,
    PreparedActionFailed,
    ResumeDriftBlocked,
    RoutingDecision,
    RoutingError,
    RoutingMode,
    route_user_intent,
)
from agent_bridge.process import ProcessRunner, StopReceipt
from agent_bridge.hub import (
    ActiveAgentLease,
    HubWorkflowOrchestrator,
    PreparedWorkflow,
    ProjectRegistry,
    RuntimeReadiness,
    RuntimeStatus,
)
from agent_bridge.repository import RepositoryTracker
from agent_bridge.state_machine import TaskState
from agent_bridge.store import (
    NewRequestPayload,
    PreparedActionOutcome,
    ResumeDriftProjection,
    ResumePayload,
    ScopeApprovalContext,
    SQLiteStore,
    TaskRecord,
)


THREAD_ID = "0199a213-81c0-7800-8aa1-bbab2a035a53"
REQUIRED_TEST = "tests/agent_bridge/test_contracts.py"
TRUSTED_PYTHON = str(Path(sys.executable).resolve())
GIT_EXECUTABLE = Path("/usr/bin/git")
TEST_COMMAND = (
    f"{TRUSTED_PYTHON} -m pytest -q tests/agent_bridge/test_contracts.py"
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    )
    return result.stdout


def _result(
    run_id: str,
    payload: Mapping[str, object] | None,
    *,
    session_id: str | None,
    events: tuple[Mapping[str, object], ...] = (),
    interrupted: bool = False,
    exit_code: int = 0,
) -> AgentRunResult:
    return AgentRunResult(
        run_id=run_id,
        cli_session_id=session_id,
        payload=payload,
        events=events,
        stderr=(),
        exit_code=exit_code,
        interrupted=interrupted,
    )


def _command_event(command: str, exit_code: int = 0) -> Mapping[str, object]:
    return {
        "type": "item.completed",
        "item_type": "command_execution",
        "status": "completed" if exit_code == 0 else "failed",
        "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "exit_code": exit_code,
        "output_sha256": hashlib.sha256(b"bounded structural output").hexdigest(),
        "output_bytes": 25,
        "output_lines": 1,
    }


def _completed(
    *,
    command: str = TEST_COMMAND,
    claimed_exit_code: int = 0,
    summary: str = "Implemented and verified the approved change.",
) -> SolOutcome:
    return SolOutcome.from_dict({
        "status": "completed",
        "summary": summary,
        "changed_files": [],
        "commands_run": [{
            "command": command,
            "exit_code": claimed_exit_code,
            "result": "passed" if claimed_exit_code == 0 else "failed",
        }],
        "known_failures": [],
        "remaining_risks": [],
        "architecture_docs": "No durable architecture change.",
        "question": None,
    })


def _question(text: str) -> SolOutcome:
    return SolOutcome.from_dict({
        "status": "question",
        "summary": "Implementation needs one clarification.",
        "changed_files": [],
        "commands_run": [],
        "known_failures": [],
        "remaining_risks": [],
        "architecture_docs": "No durable architecture change.",
        "question": {
            "ambiguity": text,
            "why_it_matters": "The answer controls the approved implementation.",
            "options": ["Keep the current scope", "Revise the scope"],
            "recommendation": "Keep the current scope",
            "can_continue_safely": False,
        },
    })


def _directed_sol_question(
    text: str,
    *,
    addressed_to: str = "fable",
) -> SolOutcome:
    return SolOutcome.from_dict({
        "status": "question",
        "summary": "Implementation needs one directed clarification.",
        "changed_files": [],
        "commands_run": [],
        "known_failures": [],
        "remaining_risks": [],
        "architecture_docs": "No durable architecture change.",
        "question": {
            "ambiguity": text,
            "why_it_matters": "The answer controls the approved implementation.",
            "options": ["Keep the current scope", "Revise the scope"],
            "recommendation": "Keep the current scope",
            "can_continue_safely": False,
            "directed_question": {
                "addressed_to": addressed_to,
                "text": text,
                "reason": "The approved implementation needs an exact answer.",
            },
        },
    })


def _answer(
    text: str,
    scope_changed: bool,
    *,
    revised_brief: TaskBrief | None = None,
    directed_question: DirectedAgentQuestion | None = None,
) -> FableClarification:
    payload: dict[str, object] = {
        "status": "answered",
        "answer": text,
        "reasoning": "The repository rules resolve the ambiguity.",
        "confidence": 0.95,
        "scope_changed": scope_changed,
        "revised_brief": None if revised_brief is None else revised_brief.to_dict(),
        "question_for_user": None,
    }
    if directed_question is not None:
        payload["directed_question"] = directed_question.to_dict()
    return FableClarification.from_dict(payload)


def _escalation(question: str, *, reasoning: str) -> FableClarification:
    return FableClarification.from_dict({
        "status": "escalate_to_user",
        "answer": None,
        "reasoning": reasoning,
        "confidence": 0.25,
        "scope_changed": False,
        "revised_brief": None,
        "question_for_user": question,
    })


def _verdict(
    brief: TaskBrief,
    status: str = "approved",
    *,
    correction: str = "Add the missing focused evidence.",
    question: str = "Should Sol continue despite the disagreement?",
    directed_question: DirectedAgentQuestion | None = None,
) -> ReviewVerdict:
    payload: dict[str, object] = {
        "status": status,
        "summary": "Review completed.",
        "criteria": [{
            "criterion": criterion,
            "evidence": ["Repository delta and structural command evidence."],
            "satisfied": status == "approved",
        } for criterion in brief.acceptance_criteria],
        "test_assessment": "Observed required test evidence was considered.",
        "scope_violations": [],
        "remaining_risks": [],
        "corrections": [correction] if status == "corrections_required" else [],
        "question_for_user": question if status == "escalate_to_user" else None,
    }
    if directed_question is not None:
        payload["directed_question"] = directed_question.to_dict()
    return ReviewVerdict.from_dict(payload)


@dataclass
class FakeFable:
    brief: TaskBrief
    clarification_prompts: list[str] = field(default_factory=list)
    clarification_sessions: list[str] = field(default_factory=list)
    clarification_run_ids: list[str] = field(default_factory=list)
    review_prompts: list[str] = field(default_factory=list)
    review_sessions: list[str] = field(default_factory=list)
    answer_sol_question_prompts: list[tuple[str, str, str]] = field(default_factory=list)
    answer_sol_question_run_ids: list[str] = field(default_factory=list)
    plan_calls: list[tuple[str, str, str]] = field(default_factory=list)
    resume_plan_sessions: list[str] = field(default_factory=list)
    resume_plan_prompts: list[str] = field(default_factory=list)
    next_clarifications: deque[FableClarification] = field(default_factory=deque)
    next_verdicts: deque[ReviewVerdict] = field(default_factory=deque)
    hold_plan: asyncio.Event | None = None
    hold_clarification: asyncio.Event | None = None
    hold_review: asyncio.Event | None = None
    hold_answer_sol_question: asyncio.Event | None = None
    on_answer_sol_question: Callable[[], None] | None = None
    plan_events: tuple[Mapping[str, object], ...] = ()
    plan_error_result: AgentRunResult | None = None

    async def plan(
        self, *, run_id: str, task_id: str, prompt: str, context: str,
    ) -> AgentRunResult:
        self.plan_calls.append((task_id, prompt, context))
        if self.plan_error_result is not None:
            raise ClaudeRunError("controlled Fable failure", result=self.plan_error_result)
        if self.hold_plan is not None:
            await self.hold_plan.wait()
            return _result(
                run_id, None, session_id="fable-session-1", interrupted=True, exit_code=-15
            )
        return _result(
            run_id,
            self.brief.to_dict(),
            session_id="fable-session-1",
            events=self.plan_events,
        )

    async def resume_plan(
        self,
        *,
        run_id: str,
        session_id: str,
        task_id: str,
        prompt: str,
        context: str,
    ) -> AgentRunResult:
        self.resume_plan_sessions.append(session_id)
        self.resume_plan_prompts.append(prompt)
        return _result(run_id, self.brief.to_dict(), session_id=session_id)

    async def clarify(
        self, *, run_id: str, session_id: str, prompt: str,
    ) -> AgentRunResult:
        self.clarification_run_ids.append(run_id)
        self.clarification_prompts.append(prompt)
        self.clarification_sessions.append(session_id)
        if self.hold_clarification is not None:
            await self.hold_clarification.wait()
            return _result(
                run_id, None, session_id=session_id, interrupted=True, exit_code=-15
            )
        clarification = (
            self.next_clarifications.popleft()
            if self.next_clarifications
            else _answer("Use the existing approved scope.", False)
        )
        return _result(run_id, clarification.to_dict(), session_id=session_id)

    async def answer_sol_question(
        self,
        *,
        run_id: str,
        session_id: str,
        task_id: str,
        prompt: str,
        context: str,
    ) -> AgentRunResult:
        self.answer_sol_question_run_ids.append(run_id)
        self.answer_sol_question_prompts.append((task_id, prompt, context))
        if self.on_answer_sol_question is not None:
            self.on_answer_sol_question()
        if self.hold_answer_sol_question is not None:
            await self.hold_answer_sol_question.wait()
            return _result(
                run_id, None, session_id=session_id, interrupted=True, exit_code=-15,
            )
        clarification = (
            self.next_clarifications.popleft()
            if self.next_clarifications
            else _answer("Use the existing approved scope.", False)
        )
        return _result(run_id, clarification.to_dict(), session_id=session_id)

    async def review(
        self, *, run_id: str, session_id: str, prompt: str,
    ) -> AgentRunResult:
        self.review_prompts.append(prompt)
        self.review_sessions.append(session_id)
        if self.hold_review is not None:
            await self.hold_review.wait()
            return _result(
                run_id, None, session_id=session_id, interrupted=True, exit_code=-15
            )
        verdict = (
            self.next_verdicts.popleft()
            if self.next_verdicts
            else _verdict(self.brief)
        )
        return _result(run_id, verdict.to_dict(), session_id=session_id)


@dataclass
class FakeSol:
    starts: list[TaskBrief] = field(default_factory=list)
    resume_prompts: list[str] = field(default_factory=list)
    resume_threads: list[str] = field(default_factory=list)
    resume_run_ids: list[str] = field(default_factory=list)
    answer_fable_question_calls: list[tuple[str, TaskBrief, str]] = field(default_factory=list)
    answer_fable_question_run_ids: list[str] = field(default_factory=list)
    next_outcomes: deque[tuple[SolOutcome, tuple[Mapping[str, object], ...]]] = field(
        default_factory=deque
    )
    hold_start: asyncio.Event | None = None
    hold_resume: asyncio.Event | None = None
    hold_answer_fable_question: asyncio.Event | None = None
    on_start: Callable[[], None] | None = None
    on_resume: Callable[[], None] | None = None
    returned_run_id: str | None = None
    returned_session_id: str = THREAD_ID
    start_error_result: AgentRunResult | None = None

    def queue(
        self,
        outcome: SolOutcome,
        *,
        events: tuple[Mapping[str, object], ...] | None = None,
    ) -> None:
        if events is None:
            events = (
                _command_event(TEST_COMMAND),
            ) if outcome.status == "completed" else ()
        self.next_outcomes.append((outcome, events))

    def _next(self) -> tuple[SolOutcome, tuple[Mapping[str, object], ...]]:
        if self.next_outcomes:
            return self.next_outcomes.popleft()
        return _completed(), (_command_event(TEST_COMMAND),)

    async def start(
        self, *, run_id: str, brief: TaskBrief, context: str,
    ) -> AgentRunResult:
        self.starts.append(brief)
        if self.start_error_result is not None:
            raise CodexRunError("controlled Sol failure", result=self.start_error_result)
        if self.on_start is not None:
            self.on_start()
        if self.hold_start is not None:
            await self.hold_start.wait()
            return _result(
                run_id, None, session_id=THREAD_ID, interrupted=True, exit_code=-15
            )
        outcome, events = self._next()
        return _result(
            self.returned_run_id or run_id,
            outcome.to_dict(),
            session_id=self.returned_session_id,
            events=events,
        )

    async def resume(
        self, *, run_id: str, thread_id: str, prompt: str,
    ) -> AgentRunResult:
        self.resume_run_ids.append(run_id)
        self.resume_threads.append(thread_id)
        self.resume_prompts.append(prompt)
        if self.on_resume is not None:
            self.on_resume()
        if self.hold_resume is not None:
            await self.hold_resume.wait()
            return _result(
                run_id, None, session_id=thread_id, interrupted=True, exit_code=-15,
            )
        outcome, events = self._next()
        return _result(run_id, outcome.to_dict(), session_id=thread_id, events=events)

    async def answer_fable_question(
        self,
        *,
        run_id: str,
        thread_id: str,
        brief: TaskBrief,
        prompt: str,
    ) -> AgentRunResult:
        self.answer_fable_question_run_ids.append(run_id)
        self.answer_fable_question_calls.append((thread_id, brief, prompt))
        if self.hold_answer_fable_question is not None:
            await self.hold_answer_fable_question.wait()
            return _result(
                run_id, None, session_id=thread_id, interrupted=True, exit_code=-15,
            )
        outcome, events = self._next()
        return _result(run_id, outcome.to_dict(), session_id=thread_id, events=events)


class RecordingRunner(ProcessRunner):
    def __init__(self) -> None:
        super().__init__(stop_grace_seconds=0)
        self.stops: list[str] = []
        self.release_on_stop: asyncio.Event | None = None
        self.on_stop: Callable[[str], None] | None = None
        self.stop_error: BaseException | None = None

    async def stop(self, run_id: str, *, timeout_seconds: float) -> StopReceipt:
        assert timeout_seconds > 0
        self.stops.append(run_id)
        if self.on_stop is not None:
            self.on_stop(run_id)
        if self.release_on_stop is not None:
            self.release_on_stop.set()
        if self.stop_error is not None:
            raise self.stop_error
        return StopReceipt(run_id=run_id, was_running=True, process_exited=True)


@dataclass
class DeterministicIds:
    task_number: int = 0
    run_number: int = 0

    def new_task_id(self) -> str:
        self.task_number += 1
        return f"task-{self.task_number}"

    def new_run_id(self) -> str:
        self.run_number += 1
        return f"run-{self.run_number}"


@dataclass(frozen=True)
class CoordinatorHarness:
    repo: Path
    database: Path
    artifacts: Path
    store: SQLiteStore
    tracker: RepositoryTracker
    runner: RecordingRunner
    fable: FakeFable
    sol: FakeSol
    ids: DeterministicIds
    coordinator: Coordinator

    async def run_approved_task(self) -> None:
        await self.coordinator.handle_user_request("session-1", "Build the bridge")
        task = self.store.latest_task("task-1")
        if task is None:
            raise AssertionError("planning did not persist task-1")
        await self.coordinator.approve_task("task-1", revision=task.revision)


@pytest.fixture
def harness(tmp_path: Path, valid_brief: TaskBrief) -> CoordinatorHarness:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Bridge Test")
    (repo / "fixture.txt").write_text("fixture\n", encoding="utf-8")
    _git(repo, "add", "fixture.txt")
    _git(repo, "commit", "-m", "fixture")

    brief = replace(
        valid_brief,
        allowed_paths=("bridge-output.txt",),
        required_tests=(REQUIRED_TEST,),
    )
    database = tmp_path / "bridge.sqlite3"
    store = SQLiteStore(database, clock=lambda: "2026-08-10T12:00:00Z")
    store.create_session("session-1", str(repo))
    tracker = RepositoryTracker(
        repo, tmp_path / "artifacts", git_executable=GIT_EXECUTABLE
    )
    runner = RecordingRunner()
    fable = FakeFable(brief)
    sol = FakeSol()
    ids = DeterministicIds()
    coordinator = Coordinator(
        store=store,
        repository=tracker,
        runner=runner,
        fable=fable,
        sol=sol,
        ids=ids,
        repo_root=repo,
        repo_context="Binding AGENTS instructions.",
        trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
    )
    return CoordinatorHarness(
        repo=repo,
        database=database,
        artifacts=tmp_path / "artifacts",
        store=store,
        tracker=tracker,
        runner=runner,
        fable=fable,
        sol=sol,
        ids=ids,
        coordinator=coordinator,
    )


def test_coordinator_forwards_recovery_summary(harness: CoordinatorHarness) -> None:
    prepared = harness.store.prepare_new_request_action(
        project_id=harness.coordinator.project_id,
        session_id="session-1",
        task_id="recover-through-coordinator",
        generation=1,
        payload=NewRequestPayload("recover this prepared action"),
    )

    assert harness.coordinator.recover_unfinished_prepared_actions() == store_module.RecoverySummary(
        prepared_actions_recovered=1,
        tasks_interrupted=1,
        agent_runs_interrupted=0,
    )
    assert harness.store.prepared_action(prepared.preparation_id).status == "RECOVERED"
    assert harness.store.get_task(prepared.task_id, prepared.revision).state is TaskState.INTERRUPTED


def test_original_user_message_accepts_only_the_first_valid_unbound_user_source(
    harness: CoordinatorHarness,
) -> None:
    """Directed planning must retain its exact user request without a duplicate event."""
    task_id = "directed-original-message"
    valid = ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.FABLE,
        message_type=ConversationMessageType.STATEMENT,
        text="The first exact directed request.",
    )
    harness.store.append_event(
        "session-1", task_id, "user", "conversation", valid.to_dict(),
    )
    harness.store.append_event(
        "session-1", task_id, "user", "conversation", ConversationEnvelope(
            sender=ConversationActor.USER,
            addressed_to=ConversationTarget.SOL,
            routed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.STATEMENT,
            text="A later request must not substitute the original.",
        ).to_dict(),
    )
    harness.store.append_event(
        "session-1", task_id, "user", "message", {"text": "Later legacy text."},
    )

    assert harness.coordinator._original_user_message(  # noqa: SLF001 - source boundary
        "session-1", task_id,
    ) == "The first exact directed request."


@pytest.mark.parametrize("event", (
    ConversationEnvelope(
        sender=ConversationActor.FABLE,
        addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.FABLE,
        message_type=ConversationMessageType.STATEMENT,
        text="An agent must not supply a user request.",
    ),
    ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.SOL,
        message_type=ConversationMessageType.STATEMENT,
        text="Browser routing cannot select Sol.",
    ),
    ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.USER,
        routed_to=ConversationTarget.FABLE,
        message_type=ConversationMessageType.STATEMENT,
        text="A user destination is not an ordinary planner request.",
    ),
    ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        message_type=ConversationMessageType.APPROVAL,
        text="An approval is not the original request.",
        task_id="other-task",
        revision=1,
    ),
))
def test_original_user_message_rejects_nonexact_or_bound_conversation_sources(
    harness: CoordinatorHarness,
    event: ConversationEnvelope,
) -> None:
    """A forged, bound, or nonstatement envelope cannot become Sol's prompt."""
    task_id = "directed-invalid-source"
    harness.store.append_event(
        "session-1", task_id, "user", "conversation", {"sender": "user"},
    )
    harness.store.append_event(
        "session-1", task_id, "user", "conversation", event.to_dict(),
    )
    harness.store.append_event(
        "session-1", "another-task", "user", "conversation", ConversationEnvelope(
            sender=ConversationActor.USER,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.STATEMENT,
            text="A correct envelope with another task association is unavailable.",
        ).to_dict(),
    )

    with pytest.raises(RuntimeError, match="original user message"):
        harness.coordinator._original_user_message("session-1", task_id)  # noqa: SLF001


def test_expired_fable_login_preserves_early_planning_and_emits_fixed_guidance(
    harness: CoordinatorHarness,
) -> None:
    async def scenario() -> None:
        async def expired_plan(**_: object) -> AgentRunResult:
            raise SubscriptionAuthError(ClaudeAuthFailureCategory.LOGIN_REQUIRED)

        harness.fable.plan = expired_plan  # type: ignore[method-assign]
        with pytest.raises(SubscriptionAuthError) as raised:
            await harness.coordinator.handle_user_request("session-1", "Plan work")

        assert raised.value.category is ClaudeAuthFailureCategory.LOGIN_REQUIRED
        task = harness.store.get_task("task-1", 0)
        assert task.state is TaskState.INTERRUPTED
        assert task.continuation_state is TaskState.FABLE_PLANNING
        assert task.pending is None
        prepared = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id,
            session_id="session-1",
            task_id="task-1",
            revision=0,
        )
        assert prepared is not None
        assert prepared.status == "INTERRUPTED"
        assert prepared.reason == "adapter_interrupted"
        run = harness.store.agent_run("run-1")
        assert run.agent == "fable"
        assert run.status == "interrupted"
        event = harness.store.events_after("session-1", 0)[-1]
        assert event.kind == "conversation"
        assert event.payload == ConversationEnvelope(
            sender=ConversationActor.SYSTEM,
            addressed_to=ConversationTarget.USER,
            routed_to=ConversationTarget.USER,
            message_type=ConversationMessageType.STATUS,
            text="Fable login expired. Run claude auth login on the host, then Resume.",
        ).to_dict()

    asyncio.run(scenario())


def test_expired_fable_login_preserves_clarification_continuation(
    harness: CoordinatorHarness,
) -> None:
    async def scenario() -> None:
        async def expired_clarification(**_: object) -> AgentRunResult:
            raise SubscriptionAuthError(ClaudeAuthFailureCategory.LOGIN_REQUIRED)

        harness.sol.queue(_question("Which bounded option?"))
        harness.fable.clarify = expired_clarification  # type: ignore[method-assign]
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")

        with pytest.raises(SubscriptionAuthError) as raised:
            await harness.coordinator.approve_task("task-1", revision=1)

        assert raised.value.category is ClaudeAuthFailureCategory.LOGIN_REQUIRED
        task = harness.store.get_task("task-1", 1)
        assert task.state is TaskState.INTERRUPTED
        assert task.continuation_state is TaskState.FABLE_CLARIFYING
        assert task.pending is not None
        assert isinstance(task.pending.get("clarification_prompt"), str)
        assert harness.store.agent_run("run-3").status == "interrupted"
        prepared = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id,
            session_id="session-1",
            task_id="task-1",
            revision=1,
        )
        assert prepared is not None
        assert prepared.status == "INTERRUPTED"
        assert prepared.reason == "adapter_interrupted"

    asyncio.run(scenario())


def test_expired_fable_login_preserves_review_continuation(
    harness: CoordinatorHarness,
) -> None:
    async def scenario() -> None:
        async def expired_review(**_: object) -> AgentRunResult:
            raise SubscriptionAuthError(ClaudeAuthFailureCategory.LOGIN_REQUIRED)

        harness.fable.review = expired_review  # type: ignore[method-assign]
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")

        with pytest.raises(SubscriptionAuthError) as raised:
            await harness.coordinator.approve_task("task-1", revision=1)

        assert raised.value.category is ClaudeAuthFailureCategory.LOGIN_REQUIRED
        task = harness.store.get_task("task-1", 1)
        assert task.state is TaskState.INTERRUPTED
        assert task.continuation_state is TaskState.FABLE_REVIEWING
        assert task.pending is not None
        assert isinstance(task.pending.get("review_prompt"), str)
        assert harness.store.agent_run("run-3").status == "interrupted"
        prepared = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id,
            session_id="session-1",
            task_id="task-1",
            revision=1,
        )
        assert prepared is not None
        assert prepared.status == "INTERRUPTED"
        assert prepared.reason == "adapter_interrupted"

    asyncio.run(scenario())


def test_legacy_audit_blocks_corrupt_scope_before_recovery_can_route_a_fresh_sol_start(
    harness: CoordinatorHarness,
) -> None:
    """An exact-thread Answer must not recover into the initial Sol-start path."""
    async def scenario() -> None:
        blocked = SolOutcome.from_dict({
            "status": "blocked",
            "summary": "The exact user decision is required.",
            "changed_files": [],
            "commands_run": [],
            "known_failures": [],
            "remaining_risks": [],
            "architecture_docs": "No durable architecture change.",
            "question": None,
        })
        harness.store.set_setting("agent_bridge.active_session_id", "session-1")
        harness.sol.queue(blocked)
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        await harness.coordinator.approve_task("task-1", revision=1)
        waiting = harness.store.get_task("task-1", 1)
        assert waiting.state is TaskState.AWAITING_USER_INPUT
        assert waiting.sol_thread_id == THREAD_ID
        assert waiting.baseline_id is not None
        prepared = harness.coordinator.prepare_answer(
            session_id="session-1",
            task_id="task-1",
            revision=1,
            answer="continue",
            generation=7,
        )
        corrupt_context = ScopeApprovalContext(
            baseline_id=waiting.baseline_id,
            approved_revision=1,
            underlying_continuation=None,
        )
        payload = json.loads(harness.store._connection.execute(
            "SELECT payload_json FROM prepared_actions WHERE preparation_id = ?",
            (prepared.preparation_id,),
        ).fetchone()["payload_json"])
        context_data = store_module._context_to_data(corrupt_context)
        payload["continuation"] = context_data
        harness.store._connection.execute(
            """
            UPDATE prepared_actions SET payload_json = ?, pending_context_json = ?
            WHERE preparation_id = ?
            """,
            (
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
                json.dumps(context_data, separators=(",", ":"), sort_keys=True),
                prepared.preparation_id,
            ),
        )
        starts_before_startup = len(harness.sol.starts)

        try:
            harness.store.audit_legacy_project_ownership(str(harness.repo))
        except RuntimeError:
            pass
        else:
            assert harness.store.recover_active_tasks() == store_module.RecoverySummary(
                prepared_actions_recovered=0,
                tasks_interrupted=1,
                agent_runs_interrupted=0,
            )
            assert harness.store.recover_unfinished_prepared_actions() == store_module.RecoverySummary(
                prepared_actions_recovered=1,
                tasks_interrupted=0,
                agent_runs_interrupted=0,
            )
            resumed = harness.store.prepare_resume_action(
                project_id=harness.coordinator.project_id,
                session_id="session-1",
                task_id="task-1",
                revision=1,
                generation=8,
                payload=ResumePayload(
                    continuation=corrupt_context,
                    drift_event=ResumeDriftProjection(
                        status="unchanged",
                        summary="Repository drift was checked.",
                        evidence_hashes=(),
                    ),
                ),
                previous_preparation_id=prepared.preparation_id,
            )
            await harness.coordinator.run_prepared_action(resumed.preparation_id)
            assert len(harness.sol.starts) == starts_before_startup + 1
            assert harness.sol.resume_threads == []
            pytest.fail("legacy audit admitted a corrupted exact-thread Answer")

        recovered = harness.store.prepared_action(prepared.preparation_id)
        assert recovered is not None
        assert recovered.status == "PREPARED"
        assert harness.store.get_task("task-1", 1).state is TaskState.SOL_RUNNING
        assert len(harness.sol.starts) == starts_before_startup
        assert harness.sol.resume_threads == []

    asyncio.run(scenario())


def test_sol_never_starts_before_exact_revision_approval(harness) -> None:
    async def scenario() -> None:
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_APPROVAL
        assert harness.sol.starts == []
        initial = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id,
            session_id="session-1",
            task_id="task-1",
            revision=0,
        )
        assert initial is not None
        assert (initial.action, initial.generation, initial.status) == (
            "new_request", 0, "COMPLETED",
        )

        with pytest.raises(ValueError, match="revision"):
            await harness.coordinator.approve_task("task-1", revision=task.revision - 1)
        assert harness.sol.starts == []

        await harness.coordinator.approve_task("task-1", revision=task.revision)
        assert [brief.revision for brief in harness.sol.starts] == [task.revision]
        approved = harness.store.get_task("task-1", task.revision)
        assert approved.approved_at == "2026-08-10T12:00:00Z"
        assert approved.baseline_id is not None
        assert approved.state is TaskState.COMPLETED
        approval = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id,
            session_id="session-1",
            task_id="task-1",
            revision=task.revision,
        )
        assert approval is not None
        assert (approval.action, approval.generation, approval.status) == (
            "approval", 0, "COMPLETED",
        )

    asyncio.run(scenario())


def test_shell_wrapper_trust_is_supplied_by_explicit_absolute_configuration(
    harness,
) -> None:
    assert harness.coordinator._trusted_shells == {
        "bash": Path("/bin/bash").resolve(strict=True),
        "sh": Path("/bin/sh").resolve(strict=True),
    }


def test_unresolved_open_questions_prevent_sol_start(harness) -> None:
    async def scenario() -> None:
        harness.fable.brief = replace(
            harness.fable.brief, open_questions=("Which directory?",)
        )
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")

        with pytest.raises(ValueError, match="open_questions"):
            await harness.coordinator.approve_task("task-1", revision=1)

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_APPROVAL
        assert task.approved_at is None
        assert task.baseline_id is None
        assert harness.sol.starts == []

    asyncio.run(scenario())


def test_sol_question_goes_to_fable_before_user(harness) -> None:
    async def scenario() -> None:
        harness.sol.queue(_question("Should the state live outside the repo?"))
        harness.fable.next_clarifications.append(
            _answer("Use the XDG state directory.", scope_changed=False)
        )
        await harness.run_approved_task()

        assert len(harness.fable.clarification_prompts) == 1
        assert "Should the state live outside the repo?" in harness.fable.clarification_prompts[0]
        assert harness.sol.resume_prompts == ["Use the XDG state directory."]
        assert harness.sol.resume_threads == [THREAD_ID]
        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.COMPLETED

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("reasoning", "question"),
    [
        ("I am uncertain about the repository policy.", "Please choose the state location."),
        ("Sol and the approved brief disagree.", "Should the brief or Sol control?"),
    ],
)
def test_fable_uncertainty_or_disagreement_escalates_without_sol_resume(
    harness, reasoning: str, question: str,
) -> None:
    async def scenario() -> None:
        harness.sol.queue(_question("Can I broaden the implementation?"))
        harness.fable.next_clarifications.append(
            _escalation(question, reasoning=reasoning)
        )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT
        assert task.continuation_state is TaskState.SOL_RUNNING
        assert task.pending is not None
        assert task.pending["question_for_user"] == question
        assert harness.sol.resume_prompts == []

    asyncio.run(scenario())


def test_scope_change_requires_new_exact_revision_approval(harness) -> None:
    async def scenario() -> None:
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", "bridge-extra.txt"),
        )
        harness.sol.queue(_question("May I add bridge-extra.txt?"))
        harness.fable.next_clarifications.append(
            _answer(
                "Add the explicitly scoped file.", True, revised_brief=revised,
                directed_question=DirectedAgentQuestion(
                    addressed_to="sol", text="This must not spend an exchange.",
                    reason="Scope approval takes precedence.",
                ),
            )
        )
        await harness.run_approved_task()

        latest = harness.store.latest_task("task-1")
        assert latest is not None
        assert latest.revision == 2
        assert latest.state is TaskState.AWAITING_SCOPE_APPROVAL
        assert latest.approved_at is None
        assert latest.sol_thread_id == THREAD_ID
        assert latest.continuation_state is TaskState.SOL_RUNNING
        assert harness.sol.resume_prompts == []

        with pytest.raises(ValueError, match="revision"):
            await harness.coordinator.approve_task("task-1", revision=1)
        assert harness.sol.resume_prompts == []

        await harness.coordinator.approve_task("task-1", revision=2)
        assert harness.sol.resume_threads == [THREAD_ID]
        assert len(harness.sol.resume_prompts) == 1
        assert "Add the explicitly scoped file." in harness.sol.resume_prompts[0]
        assert '"revision":2' in harness.sol.resume_prompts[0]
        assert "bridge-extra.txt" in harness.sol.resume_prompts[0]
        completed = harness.store.latest_task("task-1")
        assert completed is not None
        assert completed.revision == 2
        assert completed.state is TaskState.COMPLETED

    asyncio.run(scenario())


def test_scope_approval_preserves_original_baseline_and_prior_sol_edits(harness) -> None:
    async def scenario() -> None:
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", "bridge-extra.txt"),
        )
        harness.sol.on_start = lambda: (harness.repo / "bridge-output.txt").write_text(
            "revision-one work\n", encoding="utf-8"
        )
        harness.sol.queue(_question("May I add bridge-extra.txt?"))
        harness.fable.next_clarifications.append(
            _answer("Add the explicitly scoped file.", True, revised_brief=revised)
        )
        await harness.run_approved_task()

        revision_one = harness.store.get_task("task-1", 1)
        awaiting = harness.store.latest_task("task-1")
        assert awaiting is not None
        assert awaiting.revision == 2
        assert awaiting.baseline_id == revision_one.baseline_id

        await harness.coordinator.approve_task("task-1", revision=2)

        assert harness.store.latest_task("task-1").state is TaskState.COMPLETED  # type: ignore[union-attr]
        review_prompt = harness.fable.review_prompts[-1]
        assert '"changed_paths":["bridge-output.txt"]' in review_prompt
        assert "+revision-one work" in review_prompt

    asyncio.run(scenario())


def test_three_corrections_reuse_exact_sol_thread_and_fourth_goes_to_user(harness) -> None:
    async def scenario() -> None:
        for number in range(4):
            harness.fable.next_verdicts.append(
                _verdict(
                    harness.fable.brief,
                    "corrections_required",
                    correction=f"Correction {number + 1}",
                )
            )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT
        assert task.correction_count == 3
        assert task.continuation_state is TaskState.FABLE_REVIEWING
        assert harness.sol.resume_threads == [THREAD_ID, THREAD_ID, THREAD_ID]
        assert harness.sol.resume_prompts == ["Correction 1", "Correction 2", "Correction 3"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("claimed_command", "observed_command", "observed_exit_code"),
    [
        (TEST_COMMAND, TEST_COMMAND, 1),
        (TEST_COMMAND, f"{TRUSTED_PYTHON} -m pytest -q tests/other.py", 0),
    ],
)
def test_required_test_without_matching_zero_exit_evidence_cannot_complete(
    harness,
    claimed_command: str,
    observed_command: str,
    observed_exit_code: int,
) -> None:
    async def scenario() -> None:
        harness.sol.queue(
            _completed(command=claimed_command),
            events=(_command_event(observed_command, observed_exit_code),),
        )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT
        assert task.continuation_state is TaskState.FABLE_REVIEWING
        assert harness.fable.review_prompts
        assert '"required_tests_ok":false' in harness.fable.review_prompts[-1]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "collection_flag",
    ("--collect-only", "--co", "-h", "-qh", "-qV", "--markers"),
)
def test_collection_only_command_is_not_required_test_execution(
    harness, collection_flag: str,
) -> None:
    async def scenario() -> None:
        command = f"{TRUSTED_PYTHON} -m pytest {collection_flag} {REQUIRED_TEST}"
        harness.sol.queue(
            _completed(command=command), events=(_command_event(command),)
        )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT
        assert '"required_tests_ok":false' in harness.fable.review_prompts[-1]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "selection_arguments",
    (
        f"--ignore {REQUIRED_TEST} tests/other.py",
        f"--ignore-glob {REQUIRED_TEST} tests/other.py",
        f"--deselect {REQUIRED_TEST} tests/other.py",
        f"-k {REQUIRED_TEST} tests/other.py",
        f"-m {REQUIRED_TEST} tests/other.py",
    ),
)
def test_pytest_selection_or_ignore_options_cannot_forge_required_test_evidence(
    harness, selection_arguments: str,
) -> None:
    async def scenario() -> None:
        command = f"{TRUSTED_PYTHON} -m pytest {selection_arguments}"
        harness.sol.queue(
            _completed(command=command), events=(_command_event(command),)
        )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT
        assert '"required_tests_ok":false' in harness.fable.review_prompts[-1]

    asyncio.run(scenario())


def test_pytest_argfile_cannot_hide_collection_only_execution(
    harness,
) -> None:
    async def scenario() -> None:
        required = "required_test.py"
        (harness.repo / required).write_text(
            "def test_required():\n    assert True\n", encoding="utf-8"
        )
        (harness.repo / "pytest.args").write_text(
            "--collect-only\n", encoding="utf-8"
        )
        real_python = Path(sys.prefix) / "bin" / "python"
        command = f"{real_python} -m pytest {required} @pytest.args"
        completed = subprocess.run(
            command.split(),
            cwd=harness.repo,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0
        assert "collected 1 item" in completed.stdout
        harness.fable.brief = replace(
            harness.fable.brief, required_tests=(required,)
        )
        harness.sol.queue(
            _completed(command=command), events=(_command_event(command),)
        )

        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT
        assert '"required_tests_ok":false' in harness.fable.review_prompts[-1]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "command_template",
    (
        f"{TRUSTED_PYTHON} -m pytest -n4 {{required}}",
        f"{TRUSTED_PYTHON} -m pytest -nauto {{required}}",
        f"{TRUSTED_PYTHON} -m pytest ./{{required}}",
    ),
)
def test_normalized_target_and_attached_parallel_options_authenticate_required_test(
    harness, command_template: str,
) -> None:
    async def scenario() -> None:
        command = command_template.format(required=REQUIRED_TEST)
        harness.sol.queue(
            _completed(command=command), events=(_command_event(command),)
        )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.COMPLETED

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "masking_suffix",
    ("|| true", "; true", "| cat", "&& echo masked", "& true"),
)
def test_shell_control_operators_cannot_mask_required_pytest_execution(
    harness, masking_suffix: str,
) -> None:
    async def scenario() -> None:
        command = (
            f"{TRUSTED_PYTHON} -m pytest {REQUIRED_TEST} "
            "-k definitely_no_test_matches "
            f"{masking_suffix}"
        )
        harness.sol.queue(
            _completed(command=command), events=(_command_event(command),)
        )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT
        assert '"required_tests_ok":false' in harness.fable.review_prompts[-1]

    asyncio.run(scenario())


@pytest.mark.parametrize("shell", ("bash", "/bin/bash", "sh", "/bin/sh"))
def test_exact_single_shell_wrapped_pytest_execution_is_accepted(
    harness, shell: str,
) -> None:
    async def scenario() -> None:
        command = f"{shell} -lc '{TEST_COMMAND}'"
        harness.sol.queue(
            _completed(command=command), events=(_command_event(command),)
        )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.COMPLETED

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "masked_inner",
    (
        f"{TEST_COMMAND} || true",
        f"{TEST_COMMAND} && echo extra",
        f"{TEST_COMMAND}; true",
        f"{TEST_COMMAND} | cat",
        f"{TEST_COMMAND} > result.txt",
        f"{TEST_COMMAND} $(echo extra)",
        f"{TEST_COMMAND} `echo extra`",
    ),
)
def test_shell_wrapper_rejects_masking_or_extra_inner_commands(
    harness, masked_inner: str,
) -> None:
    async def scenario() -> None:
        command = f'bash -lc "{masked_inner}"'
        harness.sol.queue(
            _completed(command=command), events=(_command_event(command),)
        )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "command",
    (
        f"PYTEST_ADDOPTS=--collect-only {TEST_COMMAND}",
        f"env PYTEST_ADDOPTS=--markers {TEST_COMMAND}",
        f"bash -lc 'PYTEST_ADDOPTS=--collect-only {TEST_COMMAND}'",
    ),
)
def test_pytest_control_environment_cannot_fake_required_execution(
    harness, command: str,
) -> None:
    async def scenario() -> None:
        harness.sol.queue(
            _completed(command=command), events=(_command_event(command),)
        )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT

    asyncio.run(scenario())


def test_approved_pythonpath_assignment_preserves_exact_pytest_evidence(harness) -> None:
    async def scenario() -> None:
        command = f"PYTHONPATH={harness.repo} {TEST_COMMAND}"
        harness.sol.queue(
            _completed(command=command), events=(_command_event(command),)
        )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.COMPLETED

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "command",
    (
        f"./python-fake -m pytest {REQUIRED_TEST}",
        f"python-not-an-interpreter -m pytest {REQUIRED_TEST}",
        f"PATH=/tmp/fake {TRUSTED_PYTHON} -m pytest {REQUIRED_TEST}",
        f"PYTHONPATH=/tmp/fake {TRUSTED_PYTHON} -m pytest {REQUIRED_TEST}",
    ),
)
def test_untrusted_interpreter_or_environment_cannot_supply_test_evidence(
    harness, command: str,
) -> None:
    async def scenario() -> None:
        harness.sol.queue(
            _completed(command=command), events=(_command_event(command),)
        )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT

    asyncio.run(scenario())


def test_review_pause_persists_hash_only_sol_claims(harness) -> None:
    async def scenario() -> None:
        raw_command = f"echo {REQUIRED_TEST} SECRET_PENDING_COMMAND"
        outcome = SolOutcome.from_dict({
            "status": "completed",
            "summary": "Safe structured summary.",
            "changed_files": [],
            "commands_run": [{
                "command": raw_command,
                "exit_code": 0,
                "result": "SECRET_PENDING_RESULT",
            }],
            "known_failures": [],
            "remaining_risks": [],
            "architecture_docs": "No architecture change.",
            "question": None,
        })
        harness.sol.queue(outcome, events=(_command_event(raw_command),))
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT
        persisted = repr(task.pending)
        assert "SECRET_PENDING_COMMAND" not in persisted
        assert "SECRET_PENDING_RESULT" not in persisted
        assert hashlib.sha256(raw_command.encode()).hexdigest() in persisted

    asyncio.run(scenario())


def test_required_test_name_in_a_non_pytest_command_is_not_execution_evidence(
    harness,
) -> None:
    async def scenario() -> None:
        non_test_command = f"echo {REQUIRED_TEST}"
        harness.sol.queue(
            _completed(command=non_test_command),
            events=(_command_event(non_test_command),),
        )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT
        assert task.continuation_state is TaskState.FABLE_REVIEWING
        assert '"required_tests_ok":false' in harness.fable.review_prompts[-1]

    asyncio.run(scenario())


def test_contradictory_failed_status_with_zero_exit_is_not_completion_evidence(
    harness,
) -> None:
    async def scenario() -> None:
        contradictory = {
            **_command_event(TEST_COMMAND),
            "status": "failed",
        }
        harness.sol.queue(_completed(), events=(contradictory,))
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT
        assert '"required_tests_ok":false' in harness.fable.review_prompts[-1]

    asyncio.run(scenario())


def test_sol_claims_are_reconciled_by_hash_without_raw_command_audit_text(harness) -> None:
    async def scenario() -> None:
        secret_command = f"{TEST_COMMAND} tests/SECRET_COMMAND_VALUE.py"
        unsafe_adapter_event = {
            **_command_event(secret_command),
            "command": secret_command,
            "aggregated_output": "SECRET_RAW_OUTPUT_VALUE",
            "text": "SECRET_RAW_MODEL_TEXT",
        }
        harness.sol.queue(
            _completed(command=secret_command),
            events=(unsafe_adapter_event,),
        )
        harness.fable.brief = replace(
            harness.fable.brief,
            required_tests=(REQUIRED_TEST,),
        )
        await harness.run_approved_task()

        events = harness.store.events_after("session-1", 0)
        persisted = repr([event.to_dict() for event in events])
        assert "SECRET_COMMAND_VALUE" not in persisted
        assert "SECRET_RAW_OUTPUT_VALUE" not in persisted
        assert "SECRET_RAW_MODEL_TEXT" not in persisted
        assert hashlib.sha256(secret_command.encode()).hexdigest() in persisted
        assert harness.store.latest_task("task-1").state is TaskState.COMPLETED  # type: ignore[union-attr]

    asyncio.run(scenario())


def test_coordinator_rebuilds_structural_events_without_forwarding_allowed_key_secrets(
    harness,
) -> None:
    async def scenario() -> None:
        malicious = {
            "type": "item.completed",
            "item_type": "command_execution",
            "status": "SECRET_ALLOWED_STATUS",
            "command_sha256": "SECRET_ALLOWED_COMMAND_HASH",
            "exit_code": True,
            "output_sha256": "SECRET_ALLOWED_OUTPUT_HASH",
            "output_bytes": -1,
            "output_lines": 2**80,
            "thread_id": "SECRET_ALLOWED_THREAD_ID",
        }
        harness.sol.queue(_completed(), events=(malicious, {"type": "SECRET_TYPE"}))
        await harness.run_approved_task()

        events = harness.store.events_after("session-1", 0)
        persisted = repr([event.to_dict() for event in events])
        assert "SECRET_" not in persisted
        structural = [event for event in events if event.kind == "agent_event"]
        assert structural[-1].payload == {
            "type": "item.completed",
            "item_type": "command_execution",
        }
        assert harness.store.latest_task("task-1").state is TaskState.AWAITING_USER_INPUT  # type: ignore[union-attr]

    asyncio.run(scenario())


def test_coordinator_bounds_structural_events_from_a_fake_adapter(harness) -> None:
    """An adapter seam cannot turn one run into unbounded persisted events."""
    async def scenario() -> None:
        harness.sol.queue(
            _completed(),
            events=(_command_event(TEST_COMMAND),) * 1_300,
        )

        await harness.run_approved_task()

        structural = [
            event for event in harness.store.events_after("session-1", 0)
            if event.actor == "sol" and event.kind == "agent_event"
        ]
        assert len(structural) <= 1_024

    asyncio.run(scenario())


def test_coordinator_validates_fable_structural_values_not_only_field_names(
    harness,
) -> None:
    async def scenario() -> None:
        harness.fable.plan_events = (
            {
                "type": "system",
                "subtype": "init SECRET_SUBTYPE",
                "session_id": "SECRET SESSION",
                "has_structured_output": "SECRET_BOOL",
            },
            {"type": "result", "has_structured_output": "SECRET_RESULT_BOOL"},
        )
        await harness.coordinator.handle_user_request(
            "session-1", "Build the bridge"
        )

        events = harness.store.events_after("session-1", 0)
        persisted = repr([event.to_dict() for event in events])
        assert "SECRET_" not in persisted
        assert not [event for event in events if event.kind == "agent_event"]

    asyncio.run(scenario())


def test_adapter_result_run_id_must_match_coordinator_owned_run_before_persistence(
    harness,
) -> None:
    async def scenario() -> None:
        harness.sol.returned_run_id = "run-attacker"
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")

        with pytest.raises(PreparedActionFailed, match="prepared action failed"):
            await harness.coordinator.approve_task("task-1", revision=1)

        assert harness.store.agent_run("run-2").status == "failed"
        assert not [
            event for event in harness.store.events_after("session-1", 0)
            if event.actor == "sol" and event.kind == "agent_event"
        ]

    asyncio.run(scenario())


def test_adapter_session_id_is_validated_before_any_secret_can_be_persisted(
    harness,
) -> None:
    async def scenario() -> None:
        harness.sol.returned_session_id = "SECRET MALICIOUS SESSION"
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")

        with pytest.raises(PreparedActionFailed, match="prepared action failed"):
            await harness.coordinator.approve_task("task-1", revision=1)

        persisted = repr([
            event.to_dict()
            for event in harness.store.events_after("session-1", 0)
        ]) + repr(harness.store.agent_run("run-2"))
        assert "SECRET MALICIOUS SESSION" not in persisted

    asyncio.run(scenario())


def test_sol_outcome_event_keeps_safe_contract_fields_but_hashes_raw_command_claims(
    harness,
) -> None:
    async def scenario() -> None:
        raw_command = f"{TEST_COMMAND} --token SECRET_RAW_COMMAND"
        outcome = SolOutcome.from_dict({
            "status": "completed",
            "summary": "SAFE_SUMMARY",
            "changed_files": ["bridge-output.txt"],
            "commands_run": [{
                "command": raw_command,
                "exit_code": 0,
                "result": "SECRET_RAW_RESULT",
            }],
            "known_failures": ["SAFE_KNOWN_FAILURE"],
            "remaining_risks": ["SAFE_REMAINING_RISK"],
            "architecture_docs": "SAFE_ARCHITECTURE_NOTE",
            "question": None,
        })
        harness.sol.queue(outcome, events=(_command_event(raw_command),))
        await harness.run_approved_task()

        outcome_event = next(
            event for event in harness.store.events_after("session-1", 0)
            if event.actor == "sol" and event.kind == "outcome"
        )
        assert outcome_event.payload["summary"] == "SAFE_SUMMARY"
        assert outcome_event.payload["known_failures"] == ("SAFE_KNOWN_FAILURE",)
        assert outcome_event.payload["remaining_risks"] == ("SAFE_REMAINING_RISK",)
        assert outcome_event.payload["architecture_docs"] == "SAFE_ARCHITECTURE_NOTE"
        assert outcome_event.payload["question"] is None
        persisted = repr(outcome_event.to_dict())
        assert "SECRET_RAW_COMMAND" not in persisted
        assert "SECRET_RAW_RESULT" not in persisted
        assert hashlib.sha256(raw_command.encode()).hexdigest() in persisted

    asyncio.run(scenario())


def test_question_outcome_event_contains_validated_question_for_the_browser(harness) -> None:
    async def scenario() -> None:
        harness.sol.queue(_question("Which safe directory should hold state?"))
        harness.fable.next_clarifications.append(
            _escalation("Please choose the directory.", reasoning="Policy is ambiguous.")
        )
        await harness.run_approved_task()

        outcome_event = next(
            event for event in harness.store.events_after("session-1", 0)
            if event.actor == "sol" and event.kind == "outcome"
        )
        question = outcome_event.payload["question"]
        assert isinstance(question, Mapping)
        assert question["ambiguity"] == "Which safe directory should hold state?"

    asyncio.run(scenario())


def test_unexpected_repository_change_blocks_completion_even_if_fable_approves(
    harness,
) -> None:
    async def scenario() -> None:
        harness.sol.on_start = lambda: (harness.repo / "outside-scope.txt").write_text(
            "unexpected\n", encoding="utf-8"
        )
        await harness.run_approved_task()

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_INPUT
        assert task.continuation_state is TaskState.FABLE_REVIEWING
        assert '"unexpected_paths":["outside-scope.txt"]' in harness.fable.review_prompts[-1]

    asyncio.run(scenario())


def test_stop_uses_persisted_exact_run_and_only_explicit_resume_continues(harness) -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        harness.sol.hold_start = release
        harness.runner.release_on_stop = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await asyncio.sleep(0)
        active = harness.store.active_run_for_task("task-1", 1)
        assert active is not None

        def assert_interrupted_before_stop(run_id: str) -> None:
            persisted = harness.store.latest_task("task-1")
            assert persisted is not None
            assert persisted.state is TaskState.INTERRUPTED
            assert persisted.continuation_state is TaskState.SOL_RUNNING
            assert run_id == active.run_id

        harness.runner.on_stop = assert_interrupted_before_stop

        await harness.coordinator.stop_task("task-1")
        interrupted = harness.store.latest_task("task-1")
        assert interrupted is not None
        assert interrupted.state is TaskState.INTERRUPTED
        assert interrupted.continuation_state is TaskState.SOL_RUNNING
        assert harness.runner.stops == [active.run_id]
        assert harness.sol.resume_prompts == []
        assert harness.store.agent_run(active.run_id).status == "interrupted"
        assert interrupted.sol_thread_id == THREAD_ID
        assert approval.done()
        await approval
        assert harness.coordinator._run_completions == {}

        harness.sol.hold_start = None
        await harness.coordinator.resume_task("task-1")
        completed = harness.store.latest_task("task-1")
        assert completed is not None
        assert completed.state is TaskState.COMPLETED
        assert harness.sol.resume_threads == [THREAD_ID]

    asyncio.run(scenario())


def test_prepare_intervention_commits_exact_stop_intent_without_signaling(harness) -> None:
    """Preparation must persist the source binding before any runner Stop call."""
    async def scenario() -> None:
        release = asyncio.Event()
        harness.sol.hold_start = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await asyncio.sleep(0)
        active = harness.store.active_run_for_task("task-1", 1)
        assert active is not None
        harness.store.set_sol_thread("task-1", 1, THREAD_ID)
        harness.store.set_agent_run_session(active.run_id, THREAD_ID)

        record = harness.coordinator.prepare_intervention(
            "task-1",
            InterventionIntent(
                intervention_id="intervention-1",
                message="Pause and review this exact guidance.",
                addressed_to=ConversationTarget.FABLE,
                revision=1,
                continuation_generation=1,
            ),
        )

        assert record.run_id == active.run_id
        assert record.status.value == "pending_stop"
        assert harness.runner.stops == []
        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.INTERRUPTED
        release.set()
        await approval

    asyncio.run(scenario())


def test_prepare_intervention_same_id_retry_uses_its_authenticated_binding(harness) -> None:
    """Removing the existing-record check would reject a valid retry after Stop mutates task state."""
    async def scenario() -> None:
        release = asyncio.Event()
        harness.sol.hold_start = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await asyncio.sleep(0)
        active = harness.store.active_run_for_task("task-1", 1)
        assert active is not None
        harness.store.set_sol_thread("task-1", 1, THREAD_ID)
        harness.store.set_agent_run_session(active.run_id, THREAD_ID)
        intent = InterventionIntent(
            intervention_id="intervention-1", message="Pause here.",
            addressed_to=ConversationTarget.FABLE, revision=1,
            continuation_generation=1,
        )

        created = harness.coordinator.prepare_intervention("task-1", intent)
        assert harness.coordinator.prepare_intervention("task-1", intent) == created

        release.set()
        await approval

    asyncio.run(scenario())


def test_continue_intervention_marks_ready_only_after_exact_source_finalization(harness) -> None:
    """A durable intervention remains pending until its own source completion ends."""
    async def scenario() -> None:
        release = asyncio.Event()
        harness.sol.hold_start = release
        harness.runner.release_on_stop = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await asyncio.sleep(0)
        active = harness.store.active_run_for_task("task-1", 1)
        assert active is not None
        harness.store.set_sol_thread("task-1", 1, THREAD_ID)
        harness.store.set_agent_run_session(active.run_id, THREAD_ID)
        prepared = harness.coordinator.prepare_intervention(
            "task-1",
            InterventionIntent(
                intervention_id="intervention-1", message="Pause here.",
                addressed_to=ConversationTarget.FABLE, revision=1,
                continuation_generation=1,
            ),
        )

        await harness.coordinator.continue_intervention(prepared.intervention_id)
        await approval

        ready = harness.store.intervention(prepared.intervention_id)
        assert ready is not None
        assert ready.status.value == "ready"
        assert harness.runner.stops == [active.run_id]

    asyncio.run(scenario())


def test_concurrent_intervention_continuations_share_one_source_stop(harness) -> None:
    """Removing coordinator-local ownership would signal the committed source twice."""
    async def scenario() -> None:
        release = asyncio.Event()
        harness.sol.hold_start = release
        harness.runner.release_on_stop = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await asyncio.sleep(0)
        active = harness.store.active_run_for_task("task-1", 1)
        assert active is not None
        harness.store.set_sol_thread("task-1", 1, THREAD_ID)
        harness.store.set_agent_run_session(active.run_id, THREAD_ID)
        record = harness.coordinator.prepare_intervention(
            "task-1",
            InterventionIntent(
                intervention_id="intervention-1", message="Pause here.",
                addressed_to=ConversationTarget.FABLE, revision=1,
                continuation_generation=1,
            ),
        )

        await asyncio.gather(
            harness.coordinator.continue_intervention(record.intervention_id),
            harness.coordinator.continue_intervention(record.intervention_id),
        )
        await approval

        assert harness.runner.stops == [active.run_id]

    asyncio.run(scenario())


def test_continue_intervention_freshly_gates_claims_dispatches_and_releases_lease(harness) -> None:
    """Removing the durable continuation path would leave a ready intervention and retain its lease."""
    async def scenario() -> None:
        release = asyncio.Event()
        readiness_calls: list[str] = []

        async def fable_probe() -> tuple[bool, str]:
            readiness_calls.append("fable")
            return True, "subscription_ready"

        async def sol_probe() -> str:
            readiness_calls.append("sol")
            return "ready"

        harness.sol.hold_start = release
        harness.runner.release_on_stop = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await asyncio.sleep(0)
        active = harness.store.active_run_for_task("task-1", 1)
        assert active is not None
        harness.store.set_sol_thread("task-1", 1, THREAD_ID)
        harness.store.set_agent_run_session(active.run_id, THREAD_ID)
        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id,
            store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
        )
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="intervention-1", message="Continue with this constraint.",
                addressed_to=ConversationTarget.SOL, revision=1,
                continuation_generation=1,
            ),
        )

        stopped = harness.store.get_task("task-1", 1)
        assert stopped.pending == {
            "intervention": {
                "intervention_id": "intervention-1", "source_generation": 1,
                "source_run_id": active.run_id,
                "continuation": {"sol_run_id": active.run_id, "prompt": "Build the bridge"},
            },
        }

        await workflows.continue_intervention(prepared)
        await approval

        resumed = harness.store.intervention("intervention-1")
        assert resumed is not None
        assert resumed.status.value == "resumed"
        assert resumed.resume_attempt_id is not None
        assert resumed.resume_run_id is not None
        assert readiness_calls == ["fable", "sol"]
        assert harness.sol.resume_threads == [THREAD_ID]
        assert harness.sol.resume_prompts == ["Continue with this constraint."]
        assert lease.snapshot() is None

    asyncio.run(scenario())


def test_fable_early_planning_intervention_dispatches_its_persisted_guidance(harness) -> None:
    """An early Fable continuation must receive durable guidance before a new plan starts."""
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        prompts: list[str] = []

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        async def plan(*, run_id: str, task_id: str, prompt: str, context: str) -> AgentRunResult:
            prompts.append(prompt)
            if len(prompts) == 1:
                started.set()
                await release.wait()
                return _result(run_id, None, session_id=None, interrupted=True, exit_code=-15)
            return _result(run_id, harness.fable.brief.to_dict(), session_id="fable-session-1")

        harness.fable.plan = plan  # type: ignore[method-assign]
        harness.runner.release_on_stop = release
        initial = asyncio.create_task(
            harness.coordinator.handle_user_request("session-1", "Build the bridge")
        )
        await started.wait()
        planning = harness.store.get_task("task-1", 0)
        assert planning.state is TaskState.FABLE_PLANNING

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id,
            store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
        )
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="early-fable", message="Keep the deployment bounded.",
                addressed_to=ConversationTarget.FABLE, revision=0,
                continuation_generation=planning.continuation_generation,
            ),
        )
        await workflows.continue_intervention(prepared)
        await initial

        assert prompts == ["Build the bridge", "Build the bridge\n\nIntervention guidance:\nKeep the deployment bounded."]
        assert harness.store.intervention("early-fable").status.value == "resumed"  # type: ignore[union-attr]
        assert lease.snapshot() is None

    asyncio.run(scenario())


def test_interrupted_intervention_adapter_result_remains_unknown_with_exact_owner(
    harness,
) -> None:
    """A normal adapter return marked interrupted must never complete the intervention."""
    async def scenario() -> None:
        release = asyncio.Event()

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        async def interrupted_resume(*, run_id: str, thread_id: str, prompt: str) -> AgentRunResult:
            harness.sol.resume_threads.append(thread_id)
            harness.sol.resume_prompts.append(prompt)
            return _result(run_id, None, session_id=thread_id, interrupted=True, exit_code=-15)

        harness.sol.resume = interrupted_resume  # type: ignore[method-assign]
        harness.sol.hold_start = release
        harness.runner.release_on_stop = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await asyncio.sleep(0)
        active = harness.store.active_run_for_task("task-1", 1)
        assert active is not None
        harness.store.set_sol_thread("task-1", 1, THREAD_ID)
        harness.store.set_agent_run_session(active.run_id, THREAD_ID)
        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(project_id=runtime.project_id, session_id="session-1", task_id="task-1")
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="interrupted-result", message="Preserve this direction.",
                addressed_to=ConversationTarget.SOL, revision=1, continuation_generation=1,
            ),
        )
        await workflows.continue_intervention(prepared)
        await approval

        record = harness.store.intervention("interrupted-result")
        task = harness.store.get_task("task-1", 1)
        assert record is not None
        assert record.status.value == "resume_outcome_unknown"
        assert record.resume_attempt_id == "intervention-run-3"
        assert record.resume_run_id == "run-3"
        assert task.state is TaskState.INTERRUPTED
        assert harness.store.agent_run(record.resume_run_id).status == "interrupted"
        assert lease.snapshot() is None

    asyncio.run(scenario())


def test_fable_clarification_intervention_resumes_exact_session_with_durable_guidance(
    harness: CoordinatorHarness,
) -> None:
    """A persisted Fable clarification resumes its validated session, not a new route."""
    async def scenario() -> None:
        source_released = asyncio.Event()
        harness.sol.queue(_question("Which approved constraint applies?"))
        harness.fable.hold_clarification = source_released
        harness.runner.release_on_stop = source_released
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        while len(harness.fable.clarification_sessions) != 1:
            await asyncio.sleep(0)
        active = harness.store.active_run_for_task("task-1", 1)
        task = harness.store.get_task("task-1", 1)
        assert active is not None
        assert task.state is TaskState.FABLE_CLARIFYING
        assert task.fable_session_id is not None
        harness.store.set_agent_run_session(active.run_id, task.fable_session_id)

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(project_id=runtime.project_id, session_id="session-1", task_id="task-1")
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="fable-clarification", message="Keep this clarification narrow.",
                addressed_to=ConversationTarget.FABLE, revision=1,
                continuation_generation=task.continuation_generation,
            ),
        )
        harness.fable.hold_clarification = None

        await workflows.continue_intervention(prepared)
        await approval

        record = harness.store.intervention("fable-clarification")
        assert record is not None
        assert record.routed_to is ConversationTarget.FABLE
        assert record.status.value == "resumed"
        assert harness.fable.clarification_sessions == [task.fable_session_id] * 2
        assert harness.fable.clarification_prompts[-1] == (
            f"{harness.fable.clarification_prompts[0]}\n"
            "User answer: Keep this clarification narrow."
        )
        assert lease.snapshot() is None

    asyncio.run(scenario())


def test_fable_review_intervention_resumes_exact_session_with_durable_guidance(
    harness: CoordinatorHarness,
) -> None:
    """A persisted Fable review uses ReviewVerdict continuation and releases the Hub lease."""
    async def scenario() -> None:
        source_released = asyncio.Event()
        harness.sol.queue(_completed())
        harness.fable.hold_review = source_released
        harness.runner.release_on_stop = source_released
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        while len(harness.fable.review_sessions) != 1:
            await asyncio.sleep(0)
        active = harness.store.active_run_for_task("task-1", 1)
        task = harness.store.get_task("task-1", 1)
        assert active is not None
        assert task.state is TaskState.FABLE_REVIEWING
        assert task.fable_session_id is not None
        harness.store.set_agent_run_session(active.run_id, task.fable_session_id)

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(project_id=runtime.project_id, session_id="session-1", task_id="task-1")
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="fable-review", message="Review the evidence boundary.",
                addressed_to=ConversationTarget.FABLE, revision=1,
                continuation_generation=task.continuation_generation,
            ),
        )
        harness.fable.hold_review = None

        await workflows.continue_intervention(prepared)
        await approval

        record = harness.store.intervention("fable-review")
        assert record is not None
        assert record.routed_to is ConversationTarget.FABLE
        assert record.status.value == "resumed"
        assert harness.fable.review_sessions == [task.fable_session_id] * 2
        assert harness.fable.review_prompts[-1] == (
            f"{harness.fable.review_prompts[0]}\n"
            "User answer: Review the evidence boundary."
        )
        assert harness.store.get_task("task-1", 1).state is TaskState.COMPLETED
        assert lease.snapshot() is None

    asyncio.run(scenario())


def test_fable_addressed_correction_intervention_uses_exact_fable_clarification(
    harness: CoordinatorHarness,
) -> None:
    """Fable guidance in correction uses FableClarification before the exact Sol resume."""
    async def scenario() -> None:
        source_released = asyncio.Event()
        correction_started = asyncio.Event()
        harness.sol.queue(_completed())
        harness.fable.next_verdicts.append(
            _verdict(harness.fable.brief, status="corrections_required")
        )
        harness.sol.hold_resume = source_released
        harness.sol.on_resume = correction_started.set
        harness.runner.release_on_stop = source_released
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await correction_started.wait()
        active = harness.store.active_run_for_task("task-1", 1)
        task = harness.store.get_task("task-1", 1)
        assert active is not None
        assert task.state is TaskState.SOL_CORRECTING
        assert task.sol_thread_id == THREAD_ID
        harness.store.set_agent_run_session(active.run_id, THREAD_ID)

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(project_id=runtime.project_id, session_id="session-1", task_id="task-1")
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="fable-correction", message="Correct only the recorded gap.",
                addressed_to=ConversationTarget.FABLE, revision=1,
                continuation_generation=task.continuation_generation,
            ),
        )
        harness.sol.hold_resume = None

        await workflows.continue_intervention(prepared)
        await approval

        record = harness.store.intervention("fable-correction")
        assert record is not None
        assert record.routed_to is ConversationTarget.FABLE
        assert record.status.value == "resumed"
        assert harness.fable.clarification_sessions[-1] == task.fable_session_id
        assert harness.fable.clarification_prompts[-1] == "Correct only the recorded gap."
        assert harness.sol.resume_threads[-1] == THREAD_ID
        assert harness.sol.resume_prompts[-1] == "Use the existing approved scope."
        assert lease.snapshot() is None

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ("preapproval", "no_thread", "terminal", "scope_widened"))
def test_intervention_sol_recipient_rejects_ineligible_phase_before_dispatch(
    harness, phase: str,
) -> None:
    """Sol is eligible only for an approved, live exact-thread continuation."""
    async def scenario() -> None:
        if phase == "preapproval":
            release = asyncio.Event()
            harness.fable.hold_plan = release
            initial = asyncio.create_task(
                harness.coordinator.handle_user_request("session-1", "Build the bridge")
            )
            while harness.store.active_run_for_task("task-1", 0) is None:
                await asyncio.sleep(0)
            task = harness.store.get_task("task-1", 0)
            with pytest.raises(RuntimeError, match="Sol recipient is not eligible"):
                harness.coordinator.prepare_intervention(
                    "task-1", InterventionIntent("ineligible-preapproval", "Do not route.",
                    ConversationTarget.SOL, 0, task.continuation_generation),
                )
            release.set()
            await initial
            assert harness.sol.starts == []
            assert harness.sol.resume_threads == []
            return

        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        task = harness.store.get_task("task-1", 1)
        if phase == "terminal":
            harness.store._connection.execute(
                "UPDATE tasks SET state = ? WHERE task_id = ? AND revision = ?",
                (TaskState.COMPLETED.value, "task-1", 1),
            )
        elif phase == "scope_widened":
            harness.store._connection.execute(
                "UPDATE tasks SET state = ? WHERE task_id = ? AND revision = ?",
                (TaskState.AWAITING_SCOPE_APPROVAL.value, "task-1", 1),
            )
        with pytest.raises(RuntimeError):
            harness.coordinator.prepare_intervention(
                "task-1", InterventionIntent(f"ineligible-{phase}", "Do not route.",
                ConversationTarget.SOL, 1, task.continuation_generation),
            )
        assert harness.sol.resume_threads == []

    asyncio.run(scenario())


def test_later_stop_cancels_a_pending_intervention_and_its_exact_source(harness) -> None:
    """A later Stop must win over a pending intervention before it can become READY."""
    async def scenario() -> None:
        release = asyncio.Event()
        harness.sol.hold_start = release
        harness.runner.release_on_stop = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await asyncio.sleep(0)
        active = harness.store.active_run_for_task("task-1", 1)
        assert active is not None
        harness.store.set_sol_thread("task-1", 1, THREAD_ID)
        harness.store.set_agent_run_session(active.run_id, THREAD_ID)
        record = harness.coordinator.prepare_intervention(
            "task-1",
            InterventionIntent(
                intervention_id="intervention-1", message="Pause here.",
                addressed_to=ConversationTarget.FABLE, revision=1,
                continuation_generation=1,
            ),
        )

        await harness.coordinator.stop_task("task-1")
        await approval

        canceled = harness.store.intervention(record.intervention_id)
        assert canceled is not None
        assert canceled.status.value == "canceled_by_stop"
        assert harness.runner.stops == [active.run_id]

    asyncio.run(scenario())


def test_later_stop_cancels_a_claimed_intervention_before_its_resume_run(harness) -> None:
    """Removing the durable intervention lookup would let an active restored task escape Stop."""
    async def scenario() -> None:
        release = asyncio.Event()
        harness.sol.hold_start = release
        harness.runner.release_on_stop = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await asyncio.sleep(0)
        active = harness.store.active_run_for_task("task-1", 1)
        assert active is not None
        harness.store.set_sol_thread("task-1", 1, THREAD_ID)
        harness.store.set_agent_run_session(active.run_id, THREAD_ID)
        record = harness.coordinator.prepare_intervention(
            "task-1",
            InterventionIntent(
                intervention_id="intervention-1", message="Pause here.",
                addressed_to=ConversationTarget.SOL, revision=1,
                continuation_generation=1,
            ),
        )
        await harness.coordinator.continue_intervention(record.intervention_id)
        await approval
        await harness.coordinator.resume_intervention(
            record.intervention_id,
            resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
        )

        await harness.coordinator.stop_task("task-1")

        canceled = harness.store.intervention(record.intervention_id)
        assert canceled is not None
        assert canceled.status.value == "canceled_by_stop"
        stopped = harness.store.get_task("task-1", 1)
        assert stopped.state is TaskState.INTERRUPTED
        assert stopped.continuation_state is TaskState.SOL_RUNNING
        with pytest.raises(RuntimeError, match="resuming"):
            harness.store.complete_intervention(
                record.intervention_id, expected_resume_generation=record.resume_generation,
                resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
            )

    asyncio.run(scenario())


def test_stop_wins_after_sol_run_finishes_before_outcome_route(
    harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        route_reached = asyncio.Event()
        release_route = asyncio.Event()
        original = harness.coordinator._route_sol_outcome

        async def gated(*args, **kwargs) -> None:
            route_reached.set()
            await release_route.wait()
            await original(*args, **kwargs)

        monkeypatch.setattr(harness.coordinator, "_route_sol_outcome", gated)
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        await route_reached.wait()

        assert harness.store.active_run_for_task("task-1", 1) is None
        await harness.coordinator.stop_task("task-1")
        release_route.set()
        await approval

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.INTERRUPTED
        assert task.continuation_state is TaskState.SOL_RUNNING
        assert harness.fable.review_prompts == []

    asyncio.run(scenario())


def test_stop_wins_after_clarification_run_finishes_before_route(
    harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        harness.sol.queue(_question("Which safe option?"))
        route_reached = asyncio.Event()
        release_route = asyncio.Event()
        original = harness.coordinator._route_clarification

        async def gated(*args, **kwargs) -> None:
            route_reached.set()
            await release_route.wait()
            await original(*args, **kwargs)

        monkeypatch.setattr(harness.coordinator, "_route_clarification", gated)
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        await route_reached.wait()

        assert harness.store.active_run_for_task("task-1", 1) is None
        await harness.coordinator.stop_task("task-1")
        release_route.set()
        await approval

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.INTERRUPTED
        assert task.continuation_state is TaskState.FABLE_CLARIFYING
        assert harness.sol.resume_prompts == []

    asyncio.run(scenario())


def test_stop_wins_after_review_run_finishes_before_verdict_route(
    harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        route_reached = asyncio.Event()
        release_route = asyncio.Event()
        original = harness.coordinator._route_review

        async def gated(*args, **kwargs) -> None:
            route_reached.set()
            await release_route.wait()
            await original(*args, **kwargs)

        monkeypatch.setattr(harness.coordinator, "_route_review", gated)
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        await route_reached.wait()

        assert harness.store.active_run_for_task("task-1", 1) is None
        await harness.coordinator.stop_task("task-1")
        release_route.set()
        await approval

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.INTERRUPTED
        assert task.continuation_state is TaskState.FABLE_REVIEWING
        assert task.pending is not None
        assert isinstance(task.pending.get("review_prompt"), str)
        assert task.pending.get("completion_allowed") is True

        await harness.coordinator.resume_task("task-1")

        resumed = harness.store.latest_task("task-1")
        assert resumed is not None
        assert resumed.state is TaskState.COMPLETED
        assert resumed.pending is None

    asyncio.run(scenario())


def test_interrupted_fable_clarification_restores_underlying_sol_continuation(
    harness,
) -> None:
    async def scenario() -> None:
        harness.sol.queue(_question("Which safe option?"))
        release = asyncio.Event()
        harness.fable.hold_clarification = release
        harness.runner.release_on_stop = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        while harness.store.latest_task("task-1").state is not TaskState.FABLE_CLARIFYING:  # type: ignore[union-attr]
            await asyncio.sleep(0)
        await harness.coordinator.stop_task("task-1")
        await approval

        interrupted = harness.store.latest_task("task-1")
        assert interrupted is not None
        assert interrupted.state is TaskState.INTERRUPTED
        assert interrupted.continuation_state is TaskState.FABLE_CLARIFYING

        harness.fable.hold_clarification = None
        await harness.coordinator.resume_task("task-1")

        completed = harness.store.latest_task("task-1")
        assert completed is not None
        assert completed.state is TaskState.COMPLETED
        assert harness.sol.resume_threads == [THREAD_ID]

    asyncio.run(scenario())


def test_interrupted_fable_review_restores_persisted_review_context(harness) -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        harness.fable.hold_review = release
        harness.runner.release_on_stop = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        while harness.store.latest_task("task-1").state is not TaskState.FABLE_REVIEWING:  # type: ignore[union-attr]
            await asyncio.sleep(0)
        await harness.coordinator.stop_task("task-1")
        await approval

        interrupted = harness.store.latest_task("task-1")
        assert interrupted is not None
        assert interrupted.state is TaskState.INTERRUPTED
        assert interrupted.pending is not None
        assert isinstance(interrupted.pending.get("review_prompt"), str)

        harness.fable.hold_review = None
        await harness.coordinator.resume_task("task-1")

        completed = harness.store.latest_task("task-1")
        assert completed is not None
        assert completed.state is TaskState.COMPLETED

    asyncio.run(scenario())


def test_writer_lock_covers_baseline_capture_before_second_task_approval(harness) -> None:
    async def scenario() -> None:
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        second_brief = replace(harness.fable.brief, task_id="task-2")
        harness.store.append_event(
            "session-1", "task-2", "user", "message", {"text": "Build task two"},
        )
        harness.store.save_task(
            "session-1", second_brief, TaskState.AWAITING_USER_APPROVAL
        )
        harness.store.set_fable_session("task-2", 1, "fable-session-1")
        release = asyncio.Event()
        harness.sol.hold_start = release

        first = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        while len(harness.sol.starts) != 1:
            await asyncio.sleep(0)
        second = asyncio.create_task(
            harness.coordinator.approve_task("task-2", revision=1)
        )
        await asyncio.sleep(0)

        queued = harness.store.latest_task("task-2")
        assert queued is not None
        assert queued.state is TaskState.AWAITING_USER_APPROVAL
        assert queued.baseline_id is None
        assert harness.store.active_run_for_task("task-2", 1) is None

        release.set()
        await asyncio.gather(first, second)

    asyncio.run(scenario())


def test_initial_baseline_persistence_failure_discards_capture_artifacts(
    harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        (harness.repo / "bridge-output.txt").write_text(
            "preexisting\n", encoding="utf-8"
        )
        await harness.coordinator.handle_user_request(
            "session-1", "Build the bridge"
        )
        def fail_baseline_setting(*args: object, **kwargs: object) -> None:
            raise RuntimeError("injected initial baseline persistence failure")

        monkeypatch.setattr(harness.store, "prepare_approval_action", fail_baseline_setting)
        with pytest.raises(RuntimeError, match="initial baseline persistence"):
            await harness.coordinator.approve_task("task-1", revision=1)

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.AWAITING_USER_APPROVAL
        assert task.baseline_id is None
        assert harness.store.get_setting(
            "agent_bridge.baseline.task-1.1"
        ) is None
        assert list(harness.artifacts.iterdir()) == []
        assert harness.sol.starts == []

    asyncio.run(scenario())


def test_initial_approval_update_failure_rolls_back_setting_and_artifacts_for_retry(
    harness,
) -> None:
    async def scenario() -> None:
        (harness.repo / "bridge-output.txt").write_text(
            "preexisting\n", encoding="utf-8"
        )
        await harness.coordinator.handle_user_request(
            "session-1", "Build the bridge"
        )
        key = "agent_bridge.baseline.task-1.1"
        harness.store._connection.execute(
            """
            CREATE TRIGGER fail_initial_approval
            BEFORE UPDATE OF baseline_id ON tasks
            WHEN OLD.task_id = 'task-1' AND OLD.revision = 1
            BEGIN
                SELECT RAISE(FAIL, 'injected approval update failure');
            END
            """
        )

        with pytest.raises(
            sqlite3.IntegrityError, match="injected approval update failure"
        ):
            await harness.coordinator.approve_task("task-1", revision=1)

        failed = harness.store.latest_task("task-1")
        assert failed is not None
        assert failed.state is TaskState.AWAITING_USER_APPROVAL
        assert failed.approved_at is None
        assert failed.baseline_id is None
        assert harness.store.get_setting(key) is None
        assert list(harness.artifacts.iterdir()) == []
        assert harness.sol.starts == []

        harness.store._connection.execute("DROP TRIGGER fail_initial_approval")
        await harness.coordinator.approve_task("task-1", revision=1)

        approved = harness.store.latest_task("task-1")
        assert approved is not None and approved.baseline_id is not None
        persisted = harness.store.get_setting(key)
        assert isinstance(persisted, Mapping)
        assert persisted.get("baseline_id") == approved.baseline_id
        artifact_roots = [
            path for path in harness.artifacts.iterdir() if path.is_dir()
        ]
        assert artifact_roots == [harness.artifacts / approved.baseline_id]

    asyncio.run(scenario())


def test_newer_revision_inserted_during_capture_blocks_stale_initial_approval(
    harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        (harness.repo / "bridge-output.txt").write_text(
            "preexisting\n", encoding="utf-8"
        )
        await harness.coordinator.handle_user_request(
            "session-1", "Build the bridge"
        )
        original_capture = harness.tracker.capture

        def capture_then_insert_newer(brief: TaskBrief):
            baseline = original_capture(brief)
            harness.store.save_task(
                "session-1",
                replace(brief, revision=2, title="Concurrent newer revision"),
                TaskState.AWAITING_USER_APPROVAL,
            )
            return baseline

        monkeypatch.setattr(
            harness.tracker, "capture", capture_then_insert_newer
        )
        with pytest.raises(RuntimeError, match="concurrently|latest"):
            await harness.coordinator.approve_task("task-1", revision=1)

        stale = harness.store.get_task("task-1", 1)
        assert stale.state is TaskState.AWAITING_USER_APPROVAL
        assert stale.approved_at is None and stale.baseline_id is None
        assert harness.store.get_setting(
            "agent_bridge.baseline.task-1.1"
        ) is None
        assert list(harness.artifacts.iterdir()) == []
        assert harness.store.latest_task("task-1").revision == 2  # type: ignore[union-attr]
        assert harness.sol.starts == []

    asyncio.run(scenario())


def test_newer_revision_inserted_during_baseline_load_blocks_scope_approval(
    harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", "bridge-extra.txt"),
        )
        harness.sol.queue(_question("May I add bridge-extra.txt?"))
        harness.fable.next_clarifications.append(
            _answer("Add it.", True, revised_brief=revised)
        )
        await harness.run_approved_task()
        awaiting = harness.store.latest_task("task-1")
        assert awaiting is not None
        assert awaiting.state is TaskState.AWAITING_SCOPE_APPROVAL
        key = "agent_bridge.baseline.task-1.2"
        persisted_before = harness.store.get_setting(key)
        starts_before = len(harness.sol.starts)
        original_load = harness.coordinator._load_baseline
        inserted = False

        def load_then_insert_newer(task):
            nonlocal inserted
            baseline = original_load(task)
            if not inserted:
                inserted = True
                harness.store.save_task(
                    "session-1",
                    replace(revised, revision=3, title="Concurrent newer revision"),
                    TaskState.AWAITING_USER_APPROVAL,
                )
            return baseline

        monkeypatch.setattr(
            harness.coordinator, "_load_baseline", load_then_insert_newer
        )
        with pytest.raises(RuntimeError, match="concurrently|latest"):
            await harness.coordinator.approve_task("task-1", revision=2)

        stale = harness.store.get_task("task-1", 2)
        assert stale.state is TaskState.AWAITING_SCOPE_APPROVAL
        assert stale.approved_at is None
        assert stale.baseline_id == awaiting.baseline_id
        assert harness.store.get_setting(key) == persisted_before
        assert harness.store.latest_task("task-1").revision == 3  # type: ignore[union-attr]
        assert len(harness.sol.starts) == starts_before

    asyncio.run(scenario())


def test_each_approval_releases_the_preparation_lock_before_its_sol_run(
    harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingLock:
        def __init__(self) -> None:
            self.inner = asyncio.Lock()
            self.epoch = 0

        async def __aenter__(self):
            await self.inner.acquire()
            self.epoch += 1
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            self.inner.release()

        def locked(self) -> bool:
            return self.inner.locked()

    async def scenario() -> None:
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        second_brief = replace(harness.fable.brief, task_id="task-2")
        harness.store.append_event(
            "session-1", "task-2", "user", "message", {"text": "Build task two"},
        )
        harness.store.save_task(
            "session-1", second_brief, TaskState.AWAITING_USER_APPROVAL
        )
        harness.store.set_fable_session("task-2", 1, "fable-session-1")
        lock = RecordingLock()
        harness.coordinator._writing_lock = lock
        sequence: list[tuple[str, str, int]] = []
        original_capture = harness.tracker.capture

        def capture(brief: TaskBrief):
            sequence.append(("capture", brief.task_id, lock.epoch))
            return original_capture(brief)

        monkeypatch.setattr(harness.tracker, "capture", capture)
        release = asyncio.Event()
        harness.sol.hold_start = release
        harness.sol.on_start = lambda: sequence.append(
            ("sol", harness.sol.starts[-1].task_id, lock.epoch)
        )

        first = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        while len(harness.sol.starts) != 1:
            await asyncio.sleep(0)
        second = asyncio.create_task(
            harness.coordinator.approve_task("task-2", revision=1)
        )
        await asyncio.sleep(0)

        assert sequence == [("capture", "task-1", 1), ("sol", "task-1", 2)]
        release.set()
        await asyncio.gather(first, second)
        assert sequence == [
            ("capture", "task-1", 1),
            ("sol", "task-1", 2),
            ("capture", "task-2", 3),
            ("sol", "task-2", 4),
        ]

    asyncio.run(scenario())


def test_result_events_follow_their_persisted_branch_states(harness) -> None:
    async def scenario() -> None:
        await harness.run_approved_task()
        events = harness.store.events_after("session-1", 0)

        reviewing = next(
            event for event in events
            if event.kind == "task_state"
            and event.payload["state"] == TaskState.FABLE_REVIEWING.value
        )
        outcome = next(event for event in events if event.kind == "outcome")
        completed = next(
            event for event in events
            if event.kind == "task_state"
            and event.payload["state"] == TaskState.COMPLETED.value
        )
        review = next(event for event in events if event.kind == "review")
        assert reviewing.sequence < outcome.sequence
        assert completed.sequence < review.sequence

    asyncio.run(scenario())


def test_stop_error_leaves_durable_interruption_and_finishes_run(harness) -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        harness.sol.hold_start = release
        harness.runner.release_on_stop = release
        harness.runner.stop_error = RuntimeError("controlled stop failure")
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        await asyncio.sleep(0)
        active = harness.store.active_run_for_task("task-1", 1)
        assert active is not None

        with pytest.raises(RuntimeError, match="controlled stop failure"):
            await harness.coordinator.stop_task("task-1")
        await approval

        interrupted = harness.store.latest_task("task-1")
        assert interrupted is not None
        assert interrupted.state is TaskState.INTERRUPTED
        assert interrupted.continuation_state is TaskState.SOL_RUNNING
        assert harness.store.agent_run(active.run_id).status == "interrupted"

    asyncio.run(scenario())


def test_resume_reconstructs_persisted_baseline_after_coordinator_recreation(harness) -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        harness.sol.hold_start = release
        harness.runner.release_on_stop = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await asyncio.sleep(0)
        await harness.coordinator.stop_task("task-1")
        await approval

        baseline_id = harness.store.latest_task("task-1").baseline_id  # type: ignore[union-attr]
        harness.tracker.close()
        harness.store.close()
        reopened_store = SQLiteStore(
            harness.database, clock=lambda: "2026-08-10T12:00:00Z"
        )
        reopened_tracker = RepositoryTracker(
            harness.repo,
            harness.artifacts,
            git_executable=GIT_EXECUTABLE,
        )
        resumed_sol = FakeSol()
        recreated = Coordinator(
            store=reopened_store,
            repository=reopened_tracker,
            runner=RecordingRunner(),
            fable=harness.fable,
            sol=resumed_sol,
            ids=DeterministicIds(task_number=1, run_number=20),
            repo_root=harness.repo,
            repo_context="Binding AGENTS instructions.",
            trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
        )

        await recreated.resume_task("task-1")

        task = reopened_store.latest_task("task-1")
        assert task is not None
        assert task.baseline_id == baseline_id
        assert task.state is TaskState.COMPLETED
        assert resumed_sol.resume_threads == [THREAD_ID]
        reopened_tracker.close()
        reopened_store.close()

    asyncio.run(scenario())


def test_cancelled_planning_finishes_interrupted_and_explicit_resume_restarts_without_guessing(
    harness,
) -> None:
    async def scenario() -> None:
        harness.fable.hold_plan = asyncio.Event()
        planning = asyncio.create_task(
            harness.coordinator.handle_user_request("session-1", "Build the bridge")
        )
        await asyncio.sleep(0)
        planning.cancel()
        with pytest.raises(asyncio.CancelledError):
            await planning

        interrupted = harness.store.get_task("task-1", 0)
        assert interrupted.state is TaskState.INTERRUPTED
        assert interrupted.continuation_state is TaskState.FABLE_PLANNING
        assert interrupted.fable_session_id is None
        assert harness.store.agent_run("run-1").status == "interrupted"

        harness.fable.hold_plan = None
        await harness.coordinator.resume_task("task-1")

        planned = harness.store.latest_task("task-1")
        assert planned is not None
        assert planned.revision == 1
        assert planned.state is TaskState.AWAITING_USER_APPROVAL
        assert len(harness.fable.plan_calls) == 2
        assert harness.fable.resume_plan_sessions == []
        resumed = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id,
            session_id="session-1",
            task_id="task-1",
            revision=0,
        )
        assert resumed is not None
        assert (resumed.action, resumed.generation, resumed.status) == (
            "resume", 0, "COMPLETED",
        )

    asyncio.run(scenario())


def test_interrupted_planning_with_session_resumes_that_exact_session(harness) -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        harness.fable.hold_plan = release
        harness.runner.release_on_stop = release
        planning = asyncio.create_task(
            harness.coordinator.handle_user_request("session-1", "Build the bridge")
        )
        await asyncio.sleep(0)
        await harness.coordinator.stop_task("task-1")
        await planning

        interrupted = harness.store.get_task("task-1", 0)
        assert interrupted.state is TaskState.INTERRUPTED
        assert interrupted.fable_session_id == "fable-session-1"

        harness.fable.hold_plan = None
        await harness.coordinator.resume_task("task-1")

        assert harness.fable.resume_plan_sessions == ["fable-session-1"]
        planned = harness.store.latest_task("task-1")
        assert planned is not None
        assert planned.state is TaskState.AWAITING_USER_APPROVAL

    asyncio.run(scenario())


def test_answer_user_question_resumes_only_persisted_continuation(harness) -> None:
    async def scenario() -> None:
        harness.sol.queue(_question("Which safe option should I use?"))
        harness.fable.next_clarifications.append(
            _escalation("Choose A or B.", reasoning="The evidence is ambiguous.")
        )
        await harness.run_approved_task()

        await harness.coordinator.answer_user_question("task-1", "Use option A.")

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.COMPLETED
        assert task.pending is None
        assert task.continuation_state is None
        assert harness.sol.resume_prompts == ["Use option A."]
        assert harness.sol.resume_threads == [THREAD_ID]
        answer = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id,
            session_id="session-1",
            task_id="task-1",
            revision=task.revision,
        )
        assert answer is not None
        assert (answer.action, answer.generation, answer.status) == (
            "answer", 0, "COMPLETED",
        )

    asyncio.run(scenario())


def test_edit_creates_unapproved_next_revision_and_reject_never_starts_sol(harness) -> None:
    async def scenario() -> None:
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        edited = replace(harness.fable.brief, revision=2, title="Edited bridge task")
        await harness.coordinator.edit_task("task-1", edited)

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.revision == 2
        assert task.state is TaskState.AWAITING_USER_APPROVAL
        assert task.approved_at is None
        await harness.coordinator.reject_task("task-1")
        assert harness.store.latest_task("task-1").state is TaskState.FAILED  # type: ignore[union-attr]
        assert harness.sol.starts == []

    asyncio.run(scenario())


def test_edited_revision_inherits_exact_fable_session_and_can_complete_review(
    harness,
) -> None:
    async def scenario() -> None:
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        edited = replace(harness.fable.brief, revision=2, title="Edited bridge task")
        await harness.coordinator.edit_task("task-1", edited)

        unapproved = harness.store.latest_task("task-1")
        assert unapproved is not None
        assert unapproved.state is TaskState.AWAITING_USER_APPROVAL
        assert unapproved.fable_session_id == "fable-session-1"
        await harness.coordinator.approve_task("task-1", revision=2)

        completed = harness.store.latest_task("task-1")
        assert completed is not None
        assert completed.state is TaskState.COMPLETED
        assert completed.revision == 2
        assert completed.fable_session_id == "fable-session-1"

    asyncio.run(scenario())


def test_edit_while_awaiting_scope_approval_preserves_exact_sol_continuation(
    harness,
) -> None:
    async def scenario() -> None:
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", "bridge-extra.txt"),
        )
        harness.sol.queue(_question("May I add bridge-extra.txt?"))
        harness.fable.next_clarifications.append(
            _answer("Add the explicitly scoped file.", True, revised_brief=revised)
        )
        await harness.run_approved_task()

        edited = replace(revised, revision=3, title="User-edited scoped task")
        await harness.coordinator.edit_task("task-1", edited)
        awaiting = harness.store.latest_task("task-1")
        assert awaiting is not None
        assert awaiting.state is TaskState.AWAITING_USER_APPROVAL
        assert awaiting.fable_session_id == "fable-session-1"
        assert awaiting.sol_thread_id == THREAD_ID
        assert awaiting.continuation_state is TaskState.SOL_RUNNING

        await harness.coordinator.approve_task("task-1", revision=3)

        completed = harness.store.latest_task("task-1")
        assert completed is not None
        assert completed.state is TaskState.COMPLETED
        assert harness.sol.resume_threads == [THREAD_ID]
        assert len(harness.sol.resume_prompts) == 1
        assert '"revision":3' in harness.sol.resume_prompts[0]
        assert "Add the explicitly scoped file." in harness.sol.resume_prompts[0]

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_kind", ("symlink", "oversize", "changed"))
def test_failed_scope_edit_does_not_persist_an_invalid_newest_revision(
    harness, tmp_path: Path, invalid_kind: str,
) -> None:
    async def scenario() -> None:
        candidate = harness.repo / f"{invalid_kind}.py"
        if invalid_kind == "symlink":
            external = tmp_path / "external-secret.txt"
            external.write_text("must not be read\n", encoding="utf-8")
            candidate.symlink_to(external)
        elif invalid_kind == "oversize":
            candidate.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
        else:
            candidate.write_text("before\n", encoding="utf-8")

        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", "bridge-extra.txt"),
        )
        harness.sol.queue(_question("May I add bridge-extra.txt?"))
        harness.fable.next_clarifications.append(
            _answer("Add the explicitly scoped file.", True, revised_brief=revised)
        )
        if invalid_kind == "changed":
            harness.sol.on_start = lambda: candidate.write_text(
                "changed after baseline\n", encoding="utf-8"
            )
        await harness.run_approved_task()

        before = harness.store.latest_task("task-1")
        assert before is not None
        assert before.revision == 2
        assert before.state is TaskState.AWAITING_SCOPE_APPROVAL
        edited = replace(
            revised,
            revision=3,
            allowed_paths=(*revised.allowed_paths, candidate.name),
        )

        with pytest.raises(RuntimeError):
            await harness.coordinator.edit_task("task-1", edited)

        latest = harness.store.latest_task("task-1")
        assert latest is not None
        assert latest.revision == 2
        assert latest.baseline_id == before.baseline_id
        baseline = harness.coordinator._load_baseline(latest)
        assert baseline.baseline_id == before.baseline_id
        harness.tracker.compare(baseline)

    asyncio.run(scenario())


def test_scope_edit_baseline_setting_failure_is_atomic_and_retryable(
    harness,
) -> None:
    async def scenario() -> None:
        candidate = harness.repo / "candidate.py"
        candidate.write_text("before\n", encoding="utf-8")
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", "bridge-extra.txt"),
        )
        harness.sol.queue(_question("May I add bridge-extra.txt?"))
        harness.fable.next_clarifications.append(
            _answer("Add the explicitly scoped file.", True, revised_brief=revised)
        )
        await harness.run_approved_task()
        edited = replace(
            revised,
            revision=3,
            allowed_paths=(*revised.allowed_paths, candidate.name),
        )
        baseline_key = "agent_bridge.baseline.task-1.3"
        harness.store._connection.execute(
            f"""
            CREATE TRIGGER fail_task7_baseline_insert
            BEFORE INSERT ON settings
            WHEN NEW.key = '{baseline_key}'
            BEGIN
                SELECT RAISE(ABORT, 'injected baseline setting failure');
            END
            """
        )

        with pytest.raises(sqlite3.IntegrityError, match="injected baseline"):
            await harness.coordinator.edit_task("task-1", edited)

        failed = harness.store.latest_task("task-1")
        assert failed is not None
        assert failed.revision == 2
        assert harness.store.get_setting(baseline_key) is None
        harness.store._connection.execute("DROP TRIGGER fail_task7_baseline_insert")

        await harness.coordinator.edit_task("task-1", edited)

        retried = harness.store.latest_task("task-1")
        assert retried is not None
        assert retried.revision == 3
        assert retried.baseline_id == failed.baseline_id
        assert harness.store.get_setting(baseline_key) is not None

    asyncio.run(scenario())


def test_direct_edit_manifest_serialization_failure_cleans_widening_for_identical_retry(
    harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        candidate = harness.repo / "candidate.py"
        candidate.write_text("before\n", encoding="utf-8")
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", "bridge-extra.txt"),
        )
        harness.sol.queue(_question("May I add bridge-extra.txt?"))
        harness.fable.next_clarifications.append(
            _answer("Add it.", True, revised_brief=revised)
        )
        await harness.run_approved_task()
        edited = replace(
            revised,
            revision=3,
            allowed_paths=(*revised.allowed_paths, candidate.name),
        )
        original_manifest = harness.tracker.baseline_manifest
        failed_once = False

        def fail_serialization(baseline):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise RuntimeError("injected manifest serialization failure")
            return original_manifest(baseline)

        monkeypatch.setattr(harness.tracker, "baseline_manifest", fail_serialization)
        with pytest.raises(RuntimeError, match="injected manifest serialization"):
            await harness.coordinator.edit_task("task-1", edited)

        assert harness.store.latest_task("task-1").revision == 2  # type: ignore[union-attr]
        failed_artifact = (
            harness.artifacts
            / harness.store.latest_task("task-1").baseline_id  # type: ignore[union-attr,operator]
            / "before"
            / candidate.name
        )
        assert not failed_artifact.exists()
        await harness.coordinator.edit_task("task-1", edited)

        retried = harness.store.latest_task("task-1")
        assert retried is not None
        assert retried.revision == 3
        restored = harness.coordinator._load_baseline(retried)
        candidate_record = next(
            record for record in restored.paths if record.path == candidate.name
        )
        assert candidate_record.before_image is not None
        assert candidate_record.before_image.read_text(encoding="utf-8") == "before\n"

    asyncio.run(scenario())


def test_clarification_manifest_serialization_failure_cleans_widening_for_retry(
    harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        candidate = harness.repo / "candidate.py"
        candidate.write_text("before\n", encoding="utf-8")
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", candidate.name),
        )
        clarification = _answer("Add it.", True, revised_brief=revised)
        harness.sol.queue(_question("May I add candidate.py?"))
        harness.fable.next_clarifications.append(clarification)
        original_manifest = harness.tracker.baseline_manifest
        failed_once = False

        def fail_widened_serialization(baseline):
            nonlocal failed_once
            if candidate.name in baseline.allowed_paths and not failed_once:
                failed_once = True
                raise RuntimeError("injected clarification serialization failure")
            return original_manifest(baseline)

        monkeypatch.setattr(
            harness.tracker, "baseline_manifest", fail_widened_serialization
        )
        with pytest.raises(PreparedActionFailed, match="prepared action failed"):
            await harness.run_approved_task()

        failed = harness.store.latest_task("task-1")
        assert failed is not None
        assert failed.revision == 1
        assert failed.baseline_id is not None
        artifact = (
            harness.artifacts
            / failed.baseline_id
            / "before"
            / candidate.name
        )
        assert not artifact.exists()

        await harness.coordinator._route_clarification(failed, clarification)

        retried = harness.store.latest_task("task-1")
        assert retried is not None
        assert retried.revision == 2
        assert retried.state is TaskState.AWAITING_SCOPE_APPROVAL
        assert artifact.exists()

    asyncio.run(scenario())


def test_clarification_scope_revision_and_baseline_setting_are_atomic_and_retryable(
    harness,
) -> None:
    async def scenario() -> None:
        candidate = harness.repo / "candidate.py"
        candidate.write_text("before\n", encoding="utf-8")
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", candidate.name),
        )
        clarification = _answer(
            "Add the explicitly scoped file.", True, revised_brief=revised
        )
        harness.sol.queue(_question("May I add candidate.py?"))
        harness.fable.next_clarifications.append(clarification)
        baseline_key = "agent_bridge.baseline.task-1.2"
        harness.store._connection.execute(
            f"""
            CREATE TRIGGER fail_task7_clarification_baseline_insert
            BEFORE INSERT ON settings
            WHEN NEW.key = '{baseline_key}'
            BEGIN
                SELECT RAISE(ABORT, 'injected clarification baseline failure');
            END
            """
        )

        with pytest.raises(PreparedActionFailed, match="prepared action failed"):
            await harness.run_approved_task()

        failed = harness.store.latest_task("task-1")
        assert failed is not None
        assert failed.revision == 1
        assert failed.state is TaskState.FABLE_CLARIFYING
        original = harness.coordinator._load_baseline(failed)
        assert original.allowed_paths == ("bridge-output.txt",)
        assert harness.store.get_setting(baseline_key) is None
        harness.store._connection.execute(
            "DROP TRIGGER fail_task7_clarification_baseline_insert"
        )

        await harness.coordinator._route_clarification(failed, clarification)

        retried = harness.store.latest_task("task-1")
        assert retried is not None
        assert retried.revision == 2
        assert retried.state is TaskState.AWAITING_SCOPE_APPROVAL
        widened = harness.coordinator._load_baseline(retried)
        assert widened.allowed_paths == revised.allowed_paths
        candidate_record = next(
            record for record in widened.paths if record.path == candidate.name
        )
        assert candidate_record.before_image is not None
        assert candidate_record.before_image.read_text(encoding="utf-8") == "before\n"

    asyncio.run(scenario())


def test_scope_approval_rejects_a_tampered_widened_before_image_before_sol_resume(
    harness,
) -> None:
    async def scenario() -> None:
        candidate = harness.repo / "candidate.py"
        candidate.write_text("before\n", encoding="utf-8")
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", candidate.name),
        )
        harness.sol.queue(_question("May I add candidate.py?"))
        harness.fable.next_clarifications.append(
            _answer("Add it.", True, revised_brief=revised)
        )
        await harness.run_approved_task()
        awaiting = harness.store.latest_task("task-1")
        assert awaiting is not None
        widened = harness.coordinator._load_baseline(awaiting)
        candidate_record = next(
            record for record in widened.paths if record.path == candidate.name
        )
        assert candidate_record.before_image is not None
        candidate_record.before_image.write_text("tamper\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="before-image integrity"):
            await harness.coordinator.approve_task("task-1", revision=2)

        unchanged = harness.store.latest_task("task-1")
        assert unchanged is not None
        assert unchanged.state is TaskState.AWAITING_SCOPE_APPROVAL
        assert harness.sol.resume_prompts == []

    asyncio.run(scenario())


def test_scope_approval_rejects_missing_before_image_reference_before_sol_resume(
    harness,
) -> None:
    async def scenario() -> None:
        candidate = harness.repo / "candidate.py"
        candidate.write_text("before\n", encoding="utf-8")
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", candidate.name),
        )
        harness.sol.queue(_question("May I add candidate.py?"))
        harness.fable.next_clarifications.append(
            _answer("Add it.", True, revised_brief=revised)
        )
        await harness.run_approved_task()
        key = "agent_bridge.baseline.task-1.2"
        persisted = dict(harness.store.get_setting(key))  # type: ignore[arg-type]
        manifest = dict(persisted["manifest"])  # type: ignore[arg-type]
        manifest["paths"] = [
            {**raw, "before_image": None}
            if raw["path"] == candidate.name
            else raw
            for raw in manifest["paths"]  # type: ignore[union-attr]
        ]
        persisted["manifest"] = manifest
        harness.store.set_setting(key, persisted)

        with pytest.raises(RuntimeError, match="before-image.*required"):
            await harness.coordinator.approve_task("task-1", revision=2)

        unchanged = harness.store.latest_task("task-1")
        assert unchanged is not None
        assert unchanged.state is TaskState.AWAITING_SCOPE_APPROVAL
        assert harness.sol.resume_prompts == []

    asyncio.run(scenario())


def test_resume_rejects_a_tampered_before_image_before_sol_invocation(harness) -> None:
    async def scenario() -> None:
        harness.fable.brief = replace(
            harness.fable.brief, allowed_paths=("fixture.txt",)
        )
        release = asyncio.Event()
        harness.sol.hold_start = release
        harness.runner.release_on_stop = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        await asyncio.sleep(0)
        await harness.coordinator.stop_task("task-1")
        await approval
        interrupted = harness.store.latest_task("task-1")
        assert interrupted is not None
        baseline = harness.coordinator._load_baseline(interrupted)
        fixture_record = next(
            record for record in baseline.paths if record.path == "fixture.txt"
        )
        assert fixture_record.before_image is not None
        fixture_record.before_image.write_text("tamper\n", encoding="utf-8")
        harness.sol.hold_start = None

        with pytest.raises(RuntimeError, match="before-image integrity"):
            await harness.coordinator.resume_task("task-1")

        still_interrupted = harness.store.latest_task("task-1")
        assert still_interrupted is not None
        assert still_interrupted.state is TaskState.INTERRUPTED
        assert harness.sol.resume_prompts == []
        assert len(harness.sol.starts) == 1

    asyncio.run(scenario())


def test_resume_rejects_missing_before_image_reference_before_sol_invocation(
    harness,
) -> None:
    async def scenario() -> None:
        harness.fable.brief = replace(
            harness.fable.brief, allowed_paths=("fixture.txt",)
        )
        release = asyncio.Event()
        harness.sol.hold_start = release
        harness.runner.release_on_stop = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        await asyncio.sleep(0)
        await harness.coordinator.stop_task("task-1")
        await approval
        key = "agent_bridge.baseline.task-1.1"
        persisted = dict(harness.store.get_setting(key))  # type: ignore[arg-type]
        manifest = dict(persisted["manifest"])  # type: ignore[arg-type]
        manifest["paths"] = [
            {**raw, "before_image": None}
            if raw["path"] == "fixture.txt"
            else raw
            for raw in manifest["paths"]  # type: ignore[union-attr]
        ]
        persisted["manifest"] = manifest
        harness.store.set_setting(key, persisted)
        harness.sol.hold_start = None

        with pytest.raises(RuntimeError, match="before-image.*required"):
            await harness.coordinator.resume_task("task-1")

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.INTERRUPTED
        assert harness.sol.resume_prompts == []
        assert len(harness.sol.starts) == 1

    asyncio.run(scenario())


def test_agent_run_is_finished_when_fable_returns_wrong_task_identity(harness) -> None:
    async def scenario() -> None:
        harness.fable.brief = replace(harness.fable.brief, task_id="wrong-task")
        with pytest.raises(PreparedActionFailed, match="prepared action failed"):
            await harness.coordinator.handle_user_request("session-1", "Build the bridge")

        run = harness.store.agent_run("run-1")
        assert run.status == "failed"
        assert run.ended_at == "2026-08-10T12:00:00Z"
        assert harness.store.get_task("task-1", 0).state is TaskState.FAILED

    asyncio.run(scenario())


@pytest.mark.parametrize("actor", ("fable", "sol"))
def test_adapter_contract_error_persists_safe_result_evidence_and_actual_exit(
    harness, actor: str,
) -> None:
    async def fable_scenario() -> None:
        harness.fable.plan_error_result = _result(
            "run-1",
            None,
            session_id="fable-session-1",
            events=({"type": "result", "has_structured_output": False},),
            exit_code=42,
        )
        with pytest.raises(PreparedActionFailed, match="prepared action failed"):
            await harness.coordinator.handle_user_request(
                "session-1", "Build the bridge"
            )
        run_id = "run-1"
        expected_session = "fable-session-1"

        run = harness.store.agent_run(run_id)
        assert run.status == "failed"
        assert run.exit_code == 42
        assert run.cli_session_id == expected_session
        events = harness.store.events_after("session-1", 0)
        assert any(
            event.actor == "fable"
            and event.kind == "agent_event"
            and event.payload == {
                "type": "result", "has_structured_output": False
            }
            for event in events
        )

    async def sol_scenario() -> None:
        command = f"{TEST_COMMAND} --safe-error"
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        harness.sol.start_error_result = _result(
            "run-2",
            None,
            session_id=THREAD_ID,
            events=(_command_event(command, exit_code=42),),
            exit_code=42,
        )
        with pytest.raises(PreparedActionFailed, match="prepared action failed"):
            await harness.coordinator.approve_task("task-1", revision=1)

        run = harness.store.agent_run("run-2")
        assert run.status == "failed"
        assert run.exit_code == 42
        assert run.cli_session_id == THREAD_ID
        digest = hashlib.sha256(command.encode()).hexdigest()
        events = harness.store.events_after("session-1", 0)
        assert any(
            event.actor == "sol"
            and event.kind == "agent_event"
            and event.payload.get("command_sha256") == digest
            for event in events
        )

    asyncio.run(fable_scenario() if actor == "fable" else sol_scenario())


def test_prepare_user_request_persists_a_planning_task_without_starting_a_child(
    harness,
) -> None:
    task = harness.coordinator.prepare_user_request(
        "session-1", "Build the bridge", "prepared-task",
    )

    assert task.task_id == "prepared-task"
    assert task.session_id == "session-1"
    assert task.revision == 0
    assert task.state is TaskState.FABLE_PLANNING
    assert harness.fable.plan_calls == []
    assert [
        event.payload for event in harness.store.events_after("session-1", 0)
        if event.task_id == "prepared-task" and event.kind == "message"
    ] == [{"text": "Build the bridge"}]


def test_run_prepared_request_starts_only_the_task_that_was_already_prepared(
    harness,
) -> None:
    async def scenario() -> None:
        harness.fable.brief = replace(harness.fable.brief, task_id="prepared-task")
        harness.coordinator.prepare_user_request(
            "session-1", "Build the bridge", "prepared-task",
        )

        await harness.coordinator.run_prepared_request("prepared-task")

        assert [task_id for task_id, _, _ in harness.fable.plan_calls] == [
            "prepared-task"
        ]
        task = harness.store.get_task("prepared-task", 1)
        assert task.state is TaskState.AWAITING_USER_APPROVAL

    asyncio.run(scenario())


def test_abort_prepared_action_persists_the_exact_resumable_scheduler_failure(
    harness,
) -> None:
    harness.coordinator.prepare_user_request(
        "session-1", "Build the bridge", "prepared-task",
    )

    interrupted = harness.coordinator.abort_prepared_action(
        "prepared-task", 0, "new_request", "scheduler_unavailable",
    )
    repeated = harness.coordinator.abort_prepared_action(
        "prepared-task", 0, "new_request", "scheduler_unavailable",
    )

    assert interrupted.state is TaskState.INTERRUPTED
    assert interrupted.continuation_state is TaskState.FABLE_PLANNING
    prepared = harness.store.latest_prepared_action_for_task(
        project_id=harness.coordinator.project_id,
        session_id="session-1",
        task_id="prepared-task",
        revision=0,
    )
    assert prepared is not None
    assert prepared.status == "ABORTED"
    assert dict(interrupted.pending or {}) == {
        "prepared_action": {
            "preparation_id": prepared.preparation_id,
            "action": "new_request",
            "reason": "scheduler_unavailable",
            "context": None,
        },
    }
    assert repeated == interrupted
    assert harness.fable.plan_calls == []


def test_prepare_new_request_creates_a_durable_action_with_derived_project_identity(
    harness,
) -> None:
    prepared = harness.coordinator.prepare_new_request(
        session_id="session-1",
        task_id="prepared-task",
        text="Build the bridge",
        generation=1,
    )

    assert prepared.payload == NewRequestPayload(text="Build the bridge")
    assert prepared.project_id == harness.coordinator.project_id
    assert harness.fable.plan_calls == []


def test_prepare_new_request_retains_the_validated_recipient_intent(
    harness,
) -> None:
    prepared = harness.coordinator.prepare_new_request(
        session_id="session-1",
        task_id="prepared-team-task",
        text="Build the bridge with the team",
        generation=1,
        addressed_to=ConversationTarget.TEAM,
    )

    assert prepared.payload == NewRequestPayload(
        text="Build the bridge with the team", addressed_to=ConversationTarget.TEAM,
    )
    assert harness.fable.plan_calls == []


def test_run_prepared_action_persists_only_a_fixed_failure_category(
    harness, monkeypatch,
) -> None:
    async def scenario() -> None:
        prepared = harness.coordinator.prepare_new_request(
            session_id="session-1",
            task_id="prepared-task",
            text="Build the bridge",
            generation=9,
        )
        secret = "raw-provider-output:/private/command --token never-persist-this"

        async def fail_after_claim(*args: object, **kwargs: object) -> None:
            raise RuntimeError(secret)

        monkeypatch.setattr(harness.coordinator, "_run_planning", fail_after_claim)
        with pytest.raises(PreparedActionFailed) as surfaced:
            await harness.coordinator.run_prepared_action(prepared.preparation_id)

        persisted = harness.store.prepared_action(prepared.preparation_id)
        assert str(surfaced.value) == "prepared action failed"
        assert surfaced.value.__cause__ is None
        assert persisted is not None
        assert persisted.status == "FAILED"
        assert persisted.reason == "nonresumable_failure"
        assert secret not in repr(persisted)
        assert secret not in repr(harness.store.events_after("session-1", 0))
        assert secret not in repr(harness.store.get_task("prepared-task", 0).pending)

    asyncio.run(scenario())


def test_run_prepared_action_persists_before_reraising_a_clean_cancellation(
    harness, monkeypatch,
) -> None:
    async def scenario() -> None:
        prepared = harness.coordinator.prepare_new_request(
            session_id="session-1",
            task_id="prepared-task",
            text="Build the bridge",
            generation=12,
        )

        async def cancel_after_claim(*args: object, **kwargs: object) -> None:
            raise asyncio.CancelledError("raw cancellation sentinel")

        monkeypatch.setattr(harness.coordinator, "_run_planning", cancel_after_claim)
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await harness.coordinator.run_prepared_action(prepared.preparation_id)

        persisted = harness.store.prepared_action(prepared.preparation_id)
        assert str(cancelled.value) == ""
        assert persisted is not None
        assert persisted.status == "FAILED"
        assert persisted.reason == "nonresumable_failure"
        assert "raw cancellation sentinel" not in repr(persisted)

    asyncio.run(scenario())


def test_interrupt_claimed_prepared_action_never_completes_an_interrupted_child(harness) -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        harness.fable.hold_plan = release
        harness.runner.release_on_stop = release
        prepared = harness.coordinator.prepare_new_request(
            session_id="session-1",
            task_id="prepared-task",
            text="Build the bridge",
            generation=10,
        )
        running = asyncio.create_task(
            harness.coordinator.run_prepared_action(prepared.preparation_id)
        )
        while harness.store.active_run_for_task("prepared-task", 0) is None:
            await asyncio.sleep(0)

        await harness.coordinator.stop_task("prepared-task")
        outcome = await running
        terminal = harness.store.prepared_action(prepared.preparation_id)

        assert outcome.category == "adapter_interrupted"
        assert terminal is not None
        assert terminal.status == "INTERRUPTED"
        assert terminal.reason == "adapter_interrupted"
        interrupted = harness.store.get_task("prepared-task", 0)
        assert interrupted.state is TaskState.INTERRUPTED
        assert interrupted.pending is not None
        assert interrupted.pending["prepared_action"]["preparation_id"] == prepared.preparation_id

    asyncio.run(scenario())


def test_prepare_resume_persists_drift_failure_without_creating_a_child_action(
    harness,
) -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        harness.sol.hold_start = release
        harness.runner.release_on_stop = release
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        while harness.store.active_run_for_task("task-1", 1) is None:
            await asyncio.sleep(0)
        await harness.coordinator.stop_task("task-1")
        await approval
        (harness.repo / "unapproved-drift.txt").write_text("drift\n", encoding="utf-8")

        with pytest.raises(ResumeDriftBlocked) as blocked:
            harness.coordinator.prepare_resume(
                session_id="session-1", task_id="task-1", revision=1, generation=7,
            )

        assert blocked.value.task.state is TaskState.FAILED
        prior = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id,
            session_id="session-1",
            task_id="task-1",
            revision=1,
        )
        assert prior is not None
        assert prior.status == "INTERRUPTED"
        assert harness.sol.resume_prompts == []
        drift_events = [
            event for event in harness.store.events_after("session-1", 0)
            if event.task_id == "task-1" and event.kind == "resume_drift"
        ]
        assert len(drift_events) == 1
        assert "unapproved-drift.txt" not in repr(drift_events[0])

    asyncio.run(scenario())


@pytest.mark.parametrize("interruption", ("scheduler", "stop", "restart"))
def test_initial_approval_resume_reconstructs_the_durable_start_context_before_a_sol_thread(
    harness: CoordinatorHarness,
    interruption: str,
) -> None:
    """An interrupted initial approval must restart Sol, not require a thread to resume."""
    async def scenario() -> None:
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        prepared = harness.coordinator.prepare_approval(
            session_id="session-1", task_id="task-1", revision=1, generation=41,
        )
        approved = harness.store.get_task("task-1", 1)
        assert approved.state is TaskState.SOL_RUNNING
        assert approved.sol_thread_id is None

        coordinator = harness.coordinator
        if interruption == "scheduler":
            coordinator.abort_prepared_action(
                prepared.preparation_id,
                generation=41,
                reason="scheduler_unavailable",
            )
        elif interruption == "stop":
            harness.store.mark_interrupted(
                "task-1", 1, continuation=TaskState.SOL_RUNNING,
            )
            coordinator.interrupt_claimed_prepared_action(
                prepared.preparation_id, generation=41, reason="stop",
            )
        else:
            coordinator.close()
            coordinator = Coordinator(
                store=harness.store,
                repository=harness.tracker,
                runner=harness.runner,
                fable=harness.fable,
                sol=harness.sol,
                ids=harness.ids,
                repo_root=harness.repo,
                repo_context="Binding AGENTS instructions.",
                trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
            )

        resumed = coordinator.prepare_resume(
            session_id="session-1", task_id="task-1", revision=1, generation=42,
        )

        assert isinstance(resumed.payload, ResumePayload)
        assert resumed.payload.continuation == ScopeApprovalContext(
            baseline_id=prepared.payload.baseline_id,  # type: ignore[union-attr]
            approved_revision=1,
            underlying_continuation=None,
        )
        await coordinator.run_prepared_action(resumed.preparation_id)
        assert len(harness.sol.starts) == 1
        assert harness.sol.resume_threads == []
        assert harness.sol.answer_fable_question_calls == []

    asyncio.run(scenario())


def test_compatibility_initial_approval_does_not_hold_the_writer_lock_across_sol(
    harness,
) -> None:
    async def scenario() -> None:
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")

        await asyncio.wait_for(
            harness.coordinator.approve_task("task-1", revision=1), timeout=0.2,
        )

        task = harness.store.latest_task("task-1")
        assert task is not None
        assert task.state is TaskState.COMPLETED
        assert len(harness.sol.starts) == 1

    asyncio.run(scenario())


def test_compatibility_scope_approval_does_not_hold_the_writer_lock_across_sol(
    harness,
) -> None:
    async def scenario() -> None:
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", "bridge-extra.txt"),
        )
        harness.sol.queue(_question("May I add bridge-extra.txt?"))
        harness.fable.next_clarifications.append(
            _answer("Add the explicitly scoped file.", True, revised_brief=revised)
        )
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        initial = harness.coordinator.prepare_approval(
            session_id="session-1", task_id="task-1", revision=1, generation=7,
        )
        await harness.coordinator.run_prepared_action(initial.preparation_id)
        scope = harness.store.latest_task("task-1")
        assert scope is not None
        assert scope.state is TaskState.AWAITING_SCOPE_APPROVAL

        await asyncio.wait_for(
            harness.coordinator.approve_task("task-1", revision=2), timeout=0.2,
        )

        completed = harness.store.latest_task("task-1")
        assert completed is not None
        assert completed.state is TaskState.COMPLETED
        assert harness.sol.resume_threads == [THREAD_ID]

    asyncio.run(scenario())


def test_terminal_cas_retry_does_not_run_the_prepared_child_twice(
    harness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        prepared = harness.coordinator.prepare_new_request(
            session_id="session-1",
            task_id="task-1",
            text="Build the bridge",
            generation=9,
        )
        original_complete = harness.store.complete_prepared_action
        attempts = 0

        def fail_terminal_once(*args: object, **kwargs: object):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("injected terminal CAS failure")
            return original_complete(*args, **kwargs)

        monkeypatch.setattr(harness.store, "complete_prepared_action", fail_terminal_once)
        with pytest.raises(PreparedActionFailed, match="prepared action failed"):
            await harness.coordinator.run_prepared_action(prepared.preparation_id)

        claimed = harness.store.prepared_action(prepared.preparation_id)
        assert claimed is not None
        assert claimed.status == "CLAIMED"
        assert len(harness.fable.plan_calls) == 1

        outcome = await harness.coordinator.run_prepared_action(prepared.preparation_id)

        terminal = harness.store.prepared_action(prepared.preparation_id)
        assert outcome == PreparedActionOutcome("completed")
        assert terminal is not None
        assert terminal.status == "COMPLETED"
        assert len(harness.fable.plan_calls) == 1
        assert attempts == 2

    asyncio.run(scenario())


def test_compatibility_answer_rejects_missing_legacy_sol_run_without_starting_sol(
    harness,
) -> None:
    async def scenario() -> None:
        task = harness.fable.brief
        harness.store.save_task("session-1", task, TaskState.SOL_RUNNING)
        harness.store.set_sol_thread(task.task_id, task.revision, THREAD_ID)
        waiting = harness.store.pause_for_continuation(
            task.task_id,
            task.revision,
            expected=TaskState.SOL_RUNNING,
            target=TaskState.AWAITING_USER_INPUT,
            continuation_state=TaskState.SOL_RUNNING,
            pending={"prompt": "Use the exact persisted continuation."},
        )

        with pytest.raises(RuntimeError, match="exact Sol run"):
            await harness.coordinator.answer_user_question(
                waiting.task_id, "Use option A.",
            )

        unchanged = harness.store.get_task(waiting.task_id, waiting.revision)
        assert unchanged.state is TaskState.AWAITING_USER_INPUT
        assert unchanged.pending == {"prompt": "Use the exact persisted continuation."}
        assert harness.sol.resume_threads == []

    asyncio.run(scenario())


def test_directed_sol_question_is_visible_before_fable_starts_and_answer_before_sol_resumes(
    harness: CoordinatorHarness,
) -> None:
    """A missing transactional event boundary would let a child run ahead of chat."""
    async def scenario() -> None:
        seen = []
        harness.store.add_event_listener(seen.append)
        harness.sol.queue(
            _directed_sol_question("Which exact interpretation should be used?"),
            events=(_command_event(TEST_COMMAND),),
        )

        def assert_question_is_committed() -> None:
            questions = [
                event for event in seen
                if event.kind == "conversation"
                and event.payload["message_type"] == "question"
            ]
            assert len(questions) == 1
            question = questions[0]
            assert question.payload["sender"] == "sol"
            assert question.payload["addressed_to"] == "fable"
            assert question.payload["routed_to"] == "fable"
            assert harness.store.question(question.payload["question_id"]) is not None

        def assert_answer_is_committed() -> None:
            answers = [
                event for event in seen
                if event.kind == "conversation"
                and event.payload["message_type"] == "answer"
            ]
            assert len(answers) == 1
            assert answers[0].payload["sender"] == "fable"
            assert answers[0].payload["routed_to"] == "sol"

        harness.fable.on_answer_sol_question = assert_question_is_committed
        harness.sol.on_resume = assert_answer_is_committed

        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        await harness.coordinator.approve_task("task-1", revision=1)

        assert len(harness.fable.answer_sol_question_prompts) == 1
        assert harness.sol.resume_threads == [THREAD_ID]
        conversation = [event for event in seen if event.kind == "conversation"]
        assert [event.payload["message_type"] for event in conversation] == [
            "question", "answer",
        ]
        assert any(event.kind == "agent_event" for event in seen)

    asyncio.run(scenario())


def test_directed_answer_post_provider_state_change_cannot_resume_sol(
    harness: CoordinatorHarness,
) -> None:
    """The exact Store CAS rechecks state after Fable returns, before Sol resumes."""
    async def scenario() -> None:
        harness.sol.queue(_directed_sol_question("Which exact rule applies?"))
        original = harness.fable.answer_sol_question

        async def return_after_terminalizing(**kwargs: object) -> AgentRunResult:
            result = await original(**kwargs)  # type: ignore[arg-type]
            waiting = harness.store.get_task("task-1", 1)
            harness.store.transition_task(
                waiting.task_id,
                waiting.revision,
                expected=TaskState.AWAITING_USER_INPUT,
                target=TaskState.FAILED,
            )
            return result

        harness.fable.answer_sol_question = return_after_terminalizing  # type: ignore[method-assign]
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")

        with pytest.raises(PreparedActionFailed):
            await harness.coordinator.approve_task("task-1", revision=1)

        terminal = harness.store.get_task("task-1", 1)
        assert terminal.state is TaskState.FAILED
        assert harness.sol.resume_threads == []
        assert not [
            event for event in harness.store.events_after("session-1", 0)
            if event.kind == "conversation"
            and event.payload["message_type"] == "answer"
        ]

    asyncio.run(scenario())


def test_stop_wins_while_a_reserved_directed_agent_answer_is_in_flight(harness) -> None:
    """Stop must retain, rather than detach, the exact unanswered directed question."""
    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        harness.sol.queue(_directed_sol_question("Which exact rule applies?"))
        harness.fable.on_answer_sol_question = entered.set
        harness.fable.hold_answer_sol_question = release
        harness.runner.release_on_stop = release

        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await entered.wait()
        waiting = harness.store.get_task("task-1", 1)
        question = harness.store.unanswered_question_for_task("task-1", 1)
        active = harness.store.active_run_for_task("task-1", 1)
        assert waiting.state is TaskState.AWAITING_USER_INPUT
        assert question is not None
        assert active is not None

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id,
            store=harness.store,
            coordinator=harness.coordinator,
        )
        lease = ActiveAgentLease()
        lease.acquire(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
        )
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        reservation = workflows.reserve_stop(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
        )
        await workflows.stop(reservation=reservation)
        await approval

        preserved = harness.store.get_task("task-1", 1)
        assert preserved.state is TaskState.INTERRUPTED
        assert preserved.continuation_state is waiting.continuation_state
        unanswered = harness.store.unanswered_question_for_task("task-1", 1)
        assert unanswered is not None
        assert unanswered.question_id == question.question_id
        assert harness.sol.resume_threads == []

    asyncio.run(scenario())


def test_hub_prepared_directed_answer_stop_preserves_the_exact_pause_through_finalization(
    harness,
) -> None:
    """Removing finalizer pause preservation detaches this claimed directed answer."""
    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        harness.sol.queue(_directed_sol_question("Which exact rule applies?"))
        harness.fable.on_answer_sol_question = entered.set
        harness.fable.hold_answer_sol_question = release
        harness.runner.release_on_stop = release

        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await entered.wait()
        waiting = harness.store.get_task("task-1", 1)
        question = harness.store.unanswered_question_for_task("task-1", 1)
        active = harness.store.active_run_for_task("task-1", 1)
        pause_id = harness.store._connection.execute(
            "SELECT continuation_pause_id FROM tasks WHERE task_id = ? AND revision = ?",
            ("task-1", 1),
        ).fetchone()["continuation_pause_id"]
        assert waiting.state is TaskState.AWAITING_USER_INPUT
        assert question is not None
        assert active is not None
        assert pause_id is not None
        harness.store.set_agent_run_session(active.run_id, "fable-session-1")

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id,
            store=harness.store,
            coordinator=harness.coordinator,
        )
        lease = ActiveAgentLease()
        lease.acquire(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
        )
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="directed-answer-stop", message="Pause this answer.",
                addressed_to=ConversationTarget.FABLE, revision=1,
                continuation_generation=waiting.continuation_generation,
            ),
        )
        assert prepared.lease_token == lease.snapshot()

        reservation = workflows.reserve_stop(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
        )
        await workflows.stop(reservation=reservation)
        await approval

        interrupted = harness.store.get_task("task-1", 1)
        persisted_pause_id = harness.store._connection.execute(
            "SELECT continuation_pause_id FROM tasks WHERE task_id = ? AND revision = ?",
            ("task-1", 1),
        ).fetchone()["continuation_pause_id"]
        preserved_question = harness.store.unanswered_question_for_task("task-1", 1)
        assert harness.store.intervention("directed-answer-stop").status.value == "canceled_by_stop"  # type: ignore[union-attr]
        assert interrupted.state is TaskState.INTERRUPTED
        assert interrupted.continuation_state is waiting.continuation_state
        assert interrupted.pending == waiting.pending
        assert persisted_pause_id == pause_id
        assert preserved_question is not None
        assert preserved_question.question_id == question.question_id
        assert lease.snapshot() is None

    asyncio.run(scenario())


def test_hub_directed_fable_intervention_uses_its_claimed_run_and_guidance(
    harness: CoordinatorHarness,
) -> None:
    """The claimed intervention run must be the Fable answer run and carry guidance once."""
    async def scenario() -> None:
        entered = asyncio.Event()
        source_released = asyncio.Event()
        harness.sol.queue(_directed_sol_question("Which exact rule applies?"))
        harness.fable.on_answer_sol_question = entered.set
        harness.fable.hold_answer_sol_question = source_released
        harness.runner.release_on_stop = source_released
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await entered.wait()
        waiting = harness.store.get_task("task-1", 1)
        active = harness.store.active_run_for_task("task-1", 1)
        assert active is not None
        harness.store.set_agent_run_session(active.run_id, "fable-session-1")

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(project_id=runtime.project_id, session_id="session-1", task_id="task-1")
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="directed-fable", message="Use only the approved rule.",
                addressed_to=ConversationTarget.FABLE, revision=1,
                continuation_generation=waiting.continuation_generation,
            ),
        )
        harness.fable.hold_answer_sol_question = None

        await workflows.continue_intervention(prepared)
        await approval

        record = harness.store.intervention("directed-fable")
        assert record is not None
        assert record.status.value == "resumed"
        assert record.resume_run_id is not None
        assert harness.fable.answer_sol_question_run_ids == [active.run_id, record.resume_run_id]
        assert harness.fable.answer_sol_question_prompts[-1][1] == (
            "Which exact rule applies?\n\nIntervention guidance:\nUse only the approved rule."
        )
        assert lease.snapshot() is None
        harness.store.set_setting("agent_bridge.active_session_id", "session-1")
        harness.store.close()
        reopened = SQLiteStore(harness.database, clock=lambda: "2026-08-10T12:00:00Z")
        assert reopened.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)
        assert reopened.authenticated_intervention("directed-fable") is not None
        reopened.audit_legacy_project_ownership(str(harness.repo))
        reopened.close()

    asyncio.run(scenario())


def test_hub_directed_sol_intervention_resumes_the_exact_sol_continuation(
    harness: CoordinatorHarness,
) -> None:
    """Sol-directed guidance bypasses the paused Fable answer and owns the claimed Sol run."""
    async def scenario() -> None:
        entered = asyncio.Event()
        source_released = asyncio.Event()
        harness.sol.queue(_directed_sol_question("Which exact rule applies?"))
        harness.fable.on_answer_sol_question = entered.set
        harness.fable.hold_answer_sol_question = source_released
        harness.runner.release_on_stop = source_released
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await entered.wait()
        waiting = harness.store.get_task("task-1", 1)
        active = harness.store.active_run_for_task("task-1", 1)
        assert active is not None
        assert waiting.sol_thread_id == THREAD_ID
        harness.store.set_agent_run_session(active.run_id, "fable-session-1")

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(project_id=runtime.project_id, session_id="session-1", task_id="task-1")
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="directed-sol", message="Resume only the approved work.",
                addressed_to=ConversationTarget.SOL, revision=1,
                continuation_generation=waiting.continuation_generation,
            ),
        )
        harness.fable.hold_answer_sol_question = None

        await workflows.continue_intervention(prepared)
        await approval

        record = harness.store.intervention("directed-sol")
        assert record is not None
        assert record.status.value == "resumed"
        assert record.resume_run_id is not None
        assert harness.sol.resume_run_ids[-1] == record.resume_run_id
        assert harness.sol.resume_threads[-1] == THREAD_ID
        assert harness.sol.resume_prompts[-1] == "Resume only the approved work."
        assert len(harness.fable.answer_sol_question_prompts) == 1
        assert lease.snapshot() is None
        harness.store.set_setting("agent_bridge.active_session_id", "session-1")
        harness.store.close()
        reopened = SQLiteStore(harness.database, clock=lambda: "2026-08-10T12:00:00Z")
        assert reopened.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)
        assert reopened.authenticated_intervention("directed-sol") is not None
        reopened.audit_legacy_project_ownership(str(harness.repo))
        reopened.close()

    asyncio.run(scenario())


def test_hub_fable_planning_intervention_resumes_existing_session_with_guidance(
    harness: CoordinatorHarness,
) -> None:
    """A planning interruption with a validated session uses resume_plan and its durable message."""
    async def scenario() -> None:
        source_released = asyncio.Event()
        harness.fable.hold_plan = source_released
        harness.runner.release_on_stop = source_released
        planning = asyncio.create_task(
            harness.coordinator.handle_user_request("session-1", "Build the bridge")
        )
        while harness.store.active_run_for_task("task-1", 0) is None:
            await asyncio.sleep(0)
        active = harness.store.active_run_for_task("task-1", 0)
        assert active is not None
        harness.store.set_fable_session("task-1", 0, "fable-session-1")
        harness.store.set_agent_run_session(active.run_id, "fable-session-1")
        task = harness.store.get_task("task-1", 0)

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(project_id=runtime.project_id, session_id="session-1", task_id="task-1")
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="resume-planning", message="Keep the original scope.",
                addressed_to=ConversationTarget.FABLE, revision=0,
                continuation_generation=task.continuation_generation,
            ),
        )
        harness.fable.hold_plan = None

        await workflows.continue_intervention(prepared)
        await planning

        record = harness.store.intervention("resume-planning")
        assert record is not None
        assert record.status.value == "resumed"
        assert harness.fable.resume_plan_sessions == ["fable-session-1"]
        assert harness.fable.resume_plan_prompts == [
            "Build the bridge\n\nIntervention guidance:\nKeep the original scope."
        ]
        assert lease.snapshot() is None

    asyncio.run(scenario())


def test_later_stop_cancels_cross_route_resuming_fable_and_survives_reopen(
    harness: CoordinatorHarness,
) -> None:
    """A later Stop must authenticate the restored Fable continuation, not its source state."""
    async def scenario() -> None:
        source_released = asyncio.Event()
        resumed_fable_released = asyncio.Event()
        correction_started = asyncio.Event()
        harness.sol.queue(_completed())
        harness.fable.next_verdicts.append(
            _verdict(harness.fable.brief, status="corrections_required")
        )
        harness.sol.hold_resume = source_released
        harness.sol.on_resume = correction_started.set
        harness.runner.release_on_stop = source_released
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await correction_started.wait()
        source = harness.store.active_run_for_task("task-1", 1)
        task = harness.store.get_task("task-1", 1)
        assert source is not None
        harness.store.set_agent_run_session(source.run_id, THREAD_ID)

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(project_id=runtime.project_id, session_id="session-1", task_id="task-1")
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="cross-route-stop", message="Keep correction constrained.",
                addressed_to=ConversationTarget.FABLE, revision=1,
                continuation_generation=task.continuation_generation,
            ),
        )
        harness.sol.hold_resume = None
        harness.fable.hold_clarification = resumed_fable_released
        continuing = asyncio.create_task(workflows.continue_intervention(prepared))
        while len(harness.fable.clarification_sessions) != 1:
            await asyncio.sleep(0)
        current = harness.store.active_run_for_task("task-1", 1)
        assert current is not None
        assert current.run_id != source.run_id
        harness.runner.release_on_stop = resumed_fable_released

        reservation = workflows.reserve_stop(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
        )
        await workflows.stop(reservation=reservation)
        await continuing
        await approval

        record = harness.store.intervention("cross-route-stop")
        stopped = harness.store.get_task("task-1", 1)
        assert record is not None
        assert record.status.value == "canceled_by_stop"
        assert stopped.state is TaskState.INTERRUPTED
        assert stopped.continuation_state is TaskState.FABLE_CLARIFYING
        assert harness.runner.stops[-1] == current.run_id
        harness.store.set_setting("agent_bridge.active_session_id", "session-1")
        harness.store.close()
        reopened = SQLiteStore(harness.database, clock=lambda: "2026-08-10T12:00:00Z")
        assert reopened.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)
        assert reopened.authenticated_intervention("cross-route-stop") is not None
        reopened.audit_legacy_project_ownership(str(harness.repo))
        reopened.close()

    asyncio.run(scenario())


def test_later_stop_cancels_cross_route_resuming_nested_sol_child_and_survives_reopen(
    harness: CoordinatorHarness,
) -> None:
    """A later Stop must signal the active nested Sol child, never the original resume run."""
    async def scenario() -> None:
        source_released = asyncio.Event()
        nested_sol_released = asyncio.Event()
        correction_started = asyncio.Event()
        directed = DirectedAgentQuestion(
            addressed_to="sol",
            text="Which exact correction fact is verified?",
            reason="Fable needs the bounded correction evidence.",
        )
        harness.sol.queue(_completed())
        harness.fable.next_verdicts.append(
            _verdict(harness.fable.brief, status="corrections_required")
        )
        harness.sol.hold_resume = source_released
        harness.sol.on_resume = correction_started.set
        harness.runner.release_on_stop = source_released
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await correction_started.wait()
        source = harness.store.active_run_for_task("task-1", 1)
        task = harness.store.get_task("task-1", 1)
        assert source is not None
        harness.store.set_agent_run_session(source.run_id, THREAD_ID)

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(project_id=runtime.project_id, session_id="session-1", task_id="task-1")
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="cross-route-nested-stop", message="Do not widen the correction.",
                addressed_to=ConversationTarget.FABLE, revision=1,
                continuation_generation=task.continuation_generation,
            ),
        )
        harness.sol.hold_resume = None
        harness.sol.hold_answer_fable_question = nested_sol_released
        harness.fable.next_clarifications.append(
            _answer("I need one exact correction fact.", False, directed_question=directed)
        )
        continuing = asyncio.create_task(workflows.continue_intervention(prepared))
        while True:
            current = harness.store.active_run_for_task("task-1", 1)
            if current is not None and current.agent == "sol" and current.run_id != source.run_id:
                break
            await asyncio.sleep(0)
        harness.runner.release_on_stop = nested_sol_released

        reservation = workflows.reserve_stop(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
        )
        await workflows.stop(reservation=reservation)
        await continuing
        await approval

        record = harness.store.intervention("cross-route-nested-stop")
        stopped = harness.store.get_task("task-1", 1)
        assert record is not None
        assert record.status.value == "canceled_by_stop"
        assert stopped.state is TaskState.INTERRUPTED
        assert stopped.continuation_state is TaskState.FABLE_CLARIFYING
        assert harness.runner.stops[-1] == current.run_id
        harness.store.set_setting("agent_bridge.active_session_id", "session-1")
        harness.store.close()
        reopened = SQLiteStore(harness.database, clock=lambda: "2026-08-10T12:00:00Z")
        assert reopened.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)
        assert reopened.authenticated_intervention("cross-route-nested-stop") is not None
        reopened.audit_legacy_project_ownership(str(harness.repo))
        reopened.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "crash_boundary",
    (
        "before_provider_spawn",
        "during_provider",
        "after_child_completion",
        "after_child_answer_before_next_fable",
        "during_next_fable",
    ),
)
def test_cross_route_nested_unknown_recovers_twice_then_retries_one_bound_question(
    harness: CoordinatorHarness,
    crash_boundary: str,
) -> None:
    """Every durable nested provider boundary must reopen idempotently."""
    async def scenario() -> None:
        source_released = asyncio.Event()
        nested_provider_blocked = asyncio.Event()
        correction_started = asyncio.Event()
        crash_captured = asyncio.Event()
        crash_database = harness.database.with_name(f"nested-{crash_boundary}.sqlite3")
        crash_state: dict[str, object] = {}

        def capture_crash_image() -> None:
            crash_state["task"] = harness.store.get_task("task-1", 1)
            nested_row = harness.store._connection.execute(
                """
                SELECT question_id, exchange_id, continuation_pause_id,
                       pending_action_json, continuation_generation
                FROM questions
                WHERE task_id = ? AND revision = ?
                  AND nested_parent_kind IS NOT NULL
                """,
                ("task-1", 1),
            ).fetchone()
            assert nested_row is not None
            crash_state["nested_row"] = dict(nested_row)
            crash_state["question"] = harness.store.question(nested_row["question_id"])
            crash_state["intervention"] = harness.store.intervention("nested-unknown")
            crash_state["run"] = harness.store._connection.execute(
                """
                SELECT * FROM agent_runs
                WHERE task_id = ? AND revision = ? AND agent = 'sol'
                  AND run_id != ?
                ORDER BY rowid DESC LIMIT 1
                """,
                ("task-1", 1, source.run_id),
            ).fetchone()
            crash_state["reservation"] = harness.store._connection.execute(
                "SELECT * FROM exchange_reservations WHERE question_id = ?",
                (nested_row["question_id"],),
            ).fetchone()
            harness.store.set_setting("agent_bridge.active_session_id", "session-1")
            crash_state["events"] = tuple(
                event for event in harness.store.events_after("session-1", 0)
                if event.kind == "conversation"
                and event.payload.get("question_id") == nested_row["question_id"]
            )
            crash_connection = sqlite3.connect(crash_database)
            harness.store._connection.backup(crash_connection)
            crash_connection.close()
            crash_captured.set()
        directed = DirectedAgentQuestion(
            addressed_to="sol",
            text="Which exact correction fact is verified?",
            reason="Fable needs one bounded correction fact.",
        )
        harness.sol.queue(_completed())
        harness.fable.next_verdicts.append(
            _verdict(harness.fable.brief, status="corrections_required")
        )
        harness.sol.hold_resume = source_released
        harness.sol.on_resume = correction_started.set
        harness.runner.release_on_stop = source_released
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        await correction_started.wait()
        source = harness.store.active_run_for_task("task-1", 1)
        source_task = harness.store.get_task("task-1", 1)
        assert source is not None
        harness.store.set_agent_run_session(source.run_id, THREAD_ID)

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
        )
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="nested-unknown", message="Keep the correction bounded.",
                addressed_to=ConversationTarget.FABLE, revision=1,
                continuation_generation=source_task.continuation_generation,
            ),
        )
        harness.sol.hold_resume = None
        if crash_boundary == "during_provider":
            harness.sol.hold_answer_fable_question = nested_provider_blocked
        elif crash_boundary == "before_provider_spawn":
            async def crash_before_provider_spawn(
                question: object, **_: object,
            ) -> None:
                assert isinstance(question, store_module.QuestionRecord)
                capture_crash_image()
                await nested_provider_blocked.wait()

            harness.coordinator.answer_directed_question = crash_before_provider_spawn  # type: ignore[method-assign]
        elif crash_boundary == "after_child_completion":
            def crash_after_child_completion(**_: object) -> object:
                capture_crash_image()
                raise RuntimeError("controlled crash before nested answer CAS")

            harness.store.answer_fable_clarification_evidence_question_and_resume = crash_after_child_completion  # type: ignore[method-assign]
        elif crash_boundary == "after_child_answer_before_next_fable":
            answer_child = harness.store.answer_fable_clarification_evidence_question_and_resume

            def crash_after_child_answer(**kwargs: object) -> object:
                answered = answer_child(**kwargs)
                capture_crash_image()
                raise RuntimeError("controlled crash after nested answer CAS")

            harness.store.answer_fable_clarification_evidence_question_and_resume = crash_after_child_answer  # type: ignore[method-assign]
        else:
            clarify = harness.fable.clarify
            clarification_calls = 0

            async def crash_during_next_fable(**kwargs: object) -> AgentRunResult:
                nonlocal clarification_calls
                clarification_calls += 1
                if clarification_calls == 2:
                    capture_crash_image()
                    raise RuntimeError("controlled crash during next Fable provider")
                return await clarify(**kwargs)

            harness.fable.clarify = crash_during_next_fable  # type: ignore[method-assign]
        harness.fable.next_clarifications.append(
            _answer("I need one exact correction fact.", False, directed_question=directed)
        )
        continuing = asyncio.create_task(workflows.continue_intervention(prepared))
        if crash_boundary == "during_provider":
            while not harness.sol.answer_fable_question_calls:
                await asyncio.sleep(0)
            capture_crash_image()
        else:
            await crash_captured.wait()

        if not continuing.done():
            continuing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await continuing
        else:
            with pytest.raises(RuntimeError, match="controlled crash"):
                await continuing
        await approval
        harness.tracker.close()
        harness.store.close()

        crash_task = crash_state["task"]
        nested_row = crash_state["nested_row"]
        nested_question = crash_state["question"]
        crash_intervention = crash_state["intervention"]
        raw_nested_run = crash_state["run"]
        raw_reservation = crash_state["reservation"]
        question_events_before = crash_state["events"]
        assert isinstance(crash_task, TaskRecord)
        assert isinstance(nested_row, dict)
        assert isinstance(nested_question, store_module.QuestionRecord)
        assert isinstance(crash_intervention, store_module.InterventionRecord)
        assert isinstance(raw_nested_run, sqlite3.Row)
        assert isinstance(raw_reservation, sqlite3.Row)
        assert isinstance(question_events_before, tuple)
        nested_question_id = nested_row["question_id"]
        assert raw_nested_run["agent"] == "sol"
        assert crash_intervention.status.value == "resuming"
        assert crash_intervention.resume_run_id != raw_nested_run["run_id"]
        assert crash_intervention.directed_binding is not None
        if crash_boundary in {
            "after_child_answer_before_next_fable", "during_next_fable",
        }:
            # The child answer CAS must consume the Sol-child stage and bind the
            # exact preallocated Fable continuation before that provider starts.
            assert crash_intervention.directed_binding.stage == "next_fable"
            assert crash_intervention.directed_binding.next_run_id is not None
            assert crash_intervention.directed_binding.next_provider_id == "fable-session-1"
            assert crash_intervention.directed_binding.source_run_id == raw_nested_run["run_id"]
            assert nested_question.answer_text is not None
        else:
            assert crash_intervention.directed_binding.source_run_id == raw_nested_run["run_id"]
            assert nested_question.answer_text is None
        assert crash_intervention.directed_binding.question_id == nested_question_id
        assert crash_intervention.directed_binding.exchange_id == raw_reservation["exchange_id"]
        if crash_boundary in {
            "after_child_answer_before_next_fable", "during_next_fable",
        }:
            assert crash_task.state is TaskState.FABLE_CLARIFYING
            assert crash_task.continuation_state is None
        else:
            assert crash_task.state is TaskState.AWAITING_USER_INPUT
            assert crash_task.continuation_state is TaskState.FABLE_CLARIFYING
        assert nested_row["continuation_pause_id"] == crash_intervention.directed_binding.continuation_pause_id
        assert raw_reservation["question_id"] == nested_question_id
        assert raw_reservation["continuation_generation"] == nested_row["continuation_generation"]
        assert len(question_events_before) == 1

        first = SQLiteStore(
            crash_database, clock=lambda: "2026-08-10T12:00:00Z",
        )
        first_summary = first.recover_active_tasks()
        unknown = first.authenticated_intervention("nested-unknown")
        assert unknown is not None
        assert unknown.status.value == "resume_outcome_unknown"
        assert first_summary == store_module.RecoverySummary(
            0,
            1,
            0 if crash_boundary == "after_child_completion" else 1,
        )
        first_task = first.get_task("task-1", 1)
        assert first_task.state is TaskState.INTERRUPTED
        assert first_task.continuation_state is TaskState.FABLE_CLARIFYING
        assert first.question(nested_question_id) == nested_question
        first.audit_legacy_project_ownership(str(harness.repo))
        first.close()

        second = SQLiteStore(
            crash_database, clock=lambda: "2026-08-10T12:00:00Z",
        )
        assert second.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)
        second_unknown = second.authenticated_intervention("nested-unknown")
        assert second_unknown == unknown
        second_task = second.get_task("task-1", 1)
        assert second_task.pending == first_task.pending
        assert second._connection.execute(
            """
            SELECT continuation_pause_id FROM tasks
            WHERE task_id = ? AND revision = ?
            """,
            ("task-1", 1),
        ).fetchone()["continuation_pause_id"] == (
            None
            if crash_boundary in {
                "after_child_answer_before_next_fable", "during_next_fable",
            }
            else nested_row["continuation_pause_id"]
        )
        assert tuple(
            event for event in second.events_after("session-1", 0)
            if event.kind == "conversation"
            and event.payload.get("question_id") == nested_question_id
        ) == question_events_before
        second.audit_legacy_project_ownership(str(harness.repo))

        acknowledged = second.authorize_retry_after_unknown(
            "nested-unknown", expected_resume_generation=unknown.resume_generation,
            acknowledgment_id="nested-acknowledgment-1",
        )
        assert second.authorize_retry_after_unknown(
            "nested-unknown", expected_resume_generation=unknown.resume_generation,
            acknowledgment_id="nested-acknowledgment-1",
        ) == acknowledged
        with pytest.raises(RuntimeError, match="generation changed"):
            second.authorize_retry_after_unknown(
                "nested-unknown", expected_resume_generation=unknown.resume_generation,
                acknowledgment_id="nested-acknowledgment-2",
            )
        acknowledged_task = second.get_task("task-1", 1)
        acknowledged_question = second.question(nested_question_id)
        assert acknowledged_question is not None
        assert acknowledged.resume_generation == unknown.resume_generation + 1
        assert acknowledged_task.continuation_generation == acknowledged.resume_generation
        assert acknowledged_question.continuation_generation == (
            unknown.resume_generation
            if crash_boundary in {"after_child_answer_before_next_fable", "during_next_fable"}
            else acknowledged.resume_generation
        )
        acknowledged_reservation = second._connection.execute(
            """
            SELECT * FROM exchange_reservations
            WHERE exchange_id = ? AND question_id = ?
            """,
            (raw_reservation["exchange_id"], nested_question_id),
        ).fetchone()
        assert acknowledged_reservation is not None
        assert acknowledged_reservation["continuation_generation"] == (
            unknown.resume_generation
            if crash_boundary in {"after_child_answer_before_next_fable", "during_next_fable"}
            else acknowledged.resume_generation
        )
        assert second._connection.execute(
            """
            SELECT continuation_pause_id FROM tasks
            WHERE task_id = ? AND revision = ?
            """,
            ("task-1", 1),
        ).fetchone()["continuation_pause_id"] == (
            None
            if crash_boundary in {
                "after_child_answer_before_next_fable", "during_next_fable",
            }
            else nested_row["continuation_pause_id"]
        )
        second.close()

        retry_store = SQLiteStore(
            crash_database, clock=lambda: "2026-08-10T12:00:00Z",
        )
        retry_tracker = RepositoryTracker(
            harness.repo, harness.artifacts, git_executable=GIT_EXECUTABLE,
        )
        retry_fable = FakeFable(harness.fable.brief)
        retry_sol = FakeSol()
        retry_sol.queue(_completed(summary="The exact correction fact is verified."))
        recreated = Coordinator(
            store=retry_store, repository=retry_tracker, runner=RecordingRunner(),
            fable=retry_fable, sol=retry_sol,
            ids=DeterministicIds(task_number=1, run_number=80), repo_root=harness.repo,
            repo_context="Binding AGENTS instructions.",
            trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
        )
        retry_runtime = SimpleNamespace(
            project_id=recreated.project_id, store=retry_store, coordinator=recreated,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        retry_lease = ActiveAgentLease()
        retry_workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((retry_runtime,)), lease=retry_lease,
            usage_credits_acknowledged=lambda: True,
        )
        retry = retry_workflows.prepare_recovery_resume(
            project_id=recreated.project_id, session_id="session-1",
            intervention_id="nested-unknown",
            expected_resume_generation=acknowledged.resume_generation,
        )
        await retry_workflows.continue_intervention(retry)

        resumed = retry_store.authenticated_intervention("nested-unknown")
        assert resumed is not None
        assert resumed.status.value == "resumed"
        assert resumed.resume_generation == acknowledged.resume_generation
        assert resumed.resume_run_id is not None
        assert retry_store.agent_run(resumed.resume_run_id).agent == (
            "fable"
            if crash_boundary in {"after_child_answer_before_next_fable", "during_next_fable"}
            else "sol"
        )
        assert retry_store.agent_run(resumed.resume_run_id).status == "completed"
        assert len(retry_sol.answer_fable_question_calls) == (
            0 if crash_boundary in {"after_child_answer_before_next_fable", "during_next_fable"} else 1
        )
        assert len(harness.sol.answer_fable_question_calls) == (
            0 if crash_boundary == "before_provider_spawn" else 1
        )
        if crash_boundary not in {
            "after_child_answer_before_next_fable", "during_next_fable",
        }:
            assert retry_sol.answer_fable_question_calls[0][2] == (
                "Which exact correction fact is verified?"
                "\n\nIntervention guidance:\nKeep the correction bounded."
            )
        answered_question = retry_store.question(nested_question_id)
        assert answered_question is not None
        assert answered_question.answer_text == (
            nested_question.answer_text
            if crash_boundary in {"after_child_answer_before_next_fable", "during_next_fable"}
            else "The exact correction fact is verified."
        )
        question_events_after = tuple(
            event for event in retry_store.events_after("session-1", 0)
            if event.kind == "conversation"
            and event.payload.get("question_id") == nested_question_id
        )
        assert len(question_events_after) == 1
        assert len(tuple(
            event for event in retry_store.events_after("session-1", 0)
            if event.kind == "conversation"
            and event.payload.get("reply_to_question_id") == nested_question_id
        )) == 1
        assert retry_lease.snapshot() is None
        retry_store.audit_legacy_project_ownership(str(harness.repo))
        retry_tracker.close()
        retry_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "crash_boundary", ("after_next_fable_handoff", "during_downstream_sol"),
)
def test_next_fable_handoff_persists_one_normal_sol_continuation_before_spawn(
    harness: CoordinatorHarness,
    crash_boundary: str,
) -> None:
    """A completed staged Fable result must hand off to normal Sol work durably."""
    async def scenario() -> None:
        source_released = asyncio.Event()
        correction_started = asyncio.Event()
        downstream_started = asyncio.Event()
        downstream_blocked = asyncio.Event()
        crash_captured = asyncio.Event()
        crash_database = harness.database.with_name(f"next-handoff-{crash_boundary}.sqlite3")
        crash_state: dict[str, object] = {}

        def capture() -> None:
            harness.store.set_setting("agent_bridge.active_session_id", "session-1")
            crash_state["task"] = harness.store.get_task("task-1", 1)
            crash_state["intervention"] = harness.store.intervention("next-handoff")
            crash_state["active_run"] = harness.store.active_run_for_task("task-1", 1)
            crash_state["fable_calls"] = len(harness.fable.clarification_run_ids)
            crash_state["clarifications"] = tuple(
                event for event in harness.store.events_after("session-1", 0)
                if event.kind == "clarification"
                and event.actor == "fable"
                and event.payload.get("answer") == "The exact correction fact is verified."
            )
            copied = sqlite3.connect(crash_database)
            harness.store._connection.backup(copied)
            copied.close()
            crash_captured.set()

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        directed = DirectedAgentQuestion(
            addressed_to="sol",
            text="Which exact correction fact is verified?",
            reason="Fable needs one bounded correction fact.",
        )
        harness.sol.queue(_completed())
        harness.fable.next_verdicts.append(
            _verdict(harness.fable.brief, status="corrections_required")
        )
        harness.sol.hold_resume = source_released
        harness.sol.on_resume = correction_started.set
        harness.runner.release_on_stop = source_released
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        await correction_started.wait()
        source = harness.store.active_run_for_task("task-1", 1)
        source_task = harness.store.get_task("task-1", 1)
        assert source is not None
        harness.store.set_agent_run_session(source.run_id, THREAD_ID)
        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
        )
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="next-handoff", message="Keep the correction bounded.",
                addressed_to=ConversationTarget.FABLE, revision=1,
                continuation_generation=source_task.continuation_generation,
            ),
        )
        harness.sol.hold_resume = None
        original_handoff = harness.store.handoff_next_fable_intervention_clarification_to_sol

        def handoff_then_crash(*args: object, **kwargs: object):
            handoff = original_handoff(*args, **kwargs)
            if crash_boundary == "after_next_fable_handoff":
                capture()
                raise RuntimeError("controlled crash after next Fable handoff")
            harness.sol.hold_resume = downstream_blocked
            harness.sol.on_resume = downstream_started.set
            return handoff

        harness.store.handoff_next_fable_intervention_clarification_to_sol = handoff_then_crash  # type: ignore[method-assign]
        harness.fable.next_clarifications.extend((
            _answer("I need one exact correction fact.", False, directed_question=directed),
            _answer("The exact correction fact is verified.", False),
        ))
        continuing = asyncio.create_task(workflows.continue_intervention(prepared))
        if crash_boundary == "after_next_fable_handoff":
            await crash_captured.wait()
        else:
            await downstream_started.wait()
            capture()
        if crash_boundary == "after_next_fable_handoff":
            continuing.cancel()
            with pytest.raises(RuntimeError, match="controlled crash"):
                await continuing
        else:
            current = harness.store.active_run_for_task("task-1", 1)
            assert current is not None and current.agent == "sol"
            harness.runner.release_on_stop = downstream_blocked
            await harness.coordinator.stop_task("task-1")
            await continuing
            assert harness.runner.stops[-1] == current.run_id
        await approval
        harness.tracker.close()
        harness.store.close()

        task = crash_state["task"]
        intervention = crash_state["intervention"]
        active_run = crash_state["active_run"]
        clarifications = crash_state["clarifications"]
        assert isinstance(task, TaskRecord)
        assert isinstance(intervention, store_module.InterventionRecord)
        assert isinstance(clarifications, tuple)
        assert crash_state["fable_calls"] == 2
        assert len(clarifications) == 1
        assert clarifications[0].payload["answer"] == "The exact correction fact is verified."
        assert intervention.status is store_module.InterventionStatus.RESUMED
        assert intervention.directed_binding is not None
        assert intervention.directed_binding.stage == "next_fable"
        assert task.state is TaskState.SOL_RUNNING
        assert task.continuation_state is None
        assert task.sol_thread_id == THREAD_ID
        assert task.pending is not None
        assert task.pending["prompt"] == "The exact correction fact is verified."
        if crash_boundary == "after_next_fable_handoff":
            assert active_run is None
            assert task.pending["sol_run_id"] == source.run_id
        else:
            assert isinstance(active_run, store_module.AgentRunRecord)
            assert active_run.agent == "sol"
            assert active_run.status == "running"
            assert task.pending["sol_run_id"] == active_run.run_id

        recovered = SQLiteStore(crash_database, clock=lambda: "2026-08-10T12:00:00Z")
        assert recovered.recover_active_tasks() == store_module.RecoverySummary(
            0, 1, 0 if crash_boundary == "after_next_fable_handoff" else 1,
        )
        terminal = recovered.authenticated_intervention("next-handoff")
        assert terminal is not None
        assert terminal.status is store_module.InterventionStatus.RESUMED
        recovered_task = recovered.get_task("task-1", 1)
        assert recovered_task.state is TaskState.INTERRUPTED
        assert recovered_task.continuation_state is TaskState.SOL_RUNNING
        assert recovered_task.pending == task.pending
        assert len(tuple(
            event for event in recovered.events_after("session-1", 0)
            if event.kind == "clarification"
            and event.actor == "fable"
            and event.payload.get("answer") == "The exact correction fact is verified."
        )) == 1
        recovered.audit_legacy_project_ownership(str(harness.repo))

        retry_tracker = RepositoryTracker(
            harness.repo, harness.artifacts, git_executable=GIT_EXECUTABLE,
        )
        retry_fable = FakeFable(harness.fable.brief)
        retry_sol = FakeSol()
        retry_sol.queue(_completed())
        retry_coordinator = Coordinator(
            store=recovered, repository=retry_tracker, runner=RecordingRunner(),
            fable=retry_fable, sol=retry_sol,
            ids=DeterministicIds(task_number=1, run_number=80), repo_root=harness.repo,
            repo_context="Binding AGENTS instructions.",
            trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
        )
        await retry_coordinator.resume_task("task-1")
        assert retry_sol.resume_threads == [THREAD_ID]
        assert retry_sol.resume_prompts == ["The exact correction fact is verified."]
        assert retry_fable.clarification_run_ids == []
        assert len(tuple(
            event for event in recovered.events_after("session-1", 0)
            if event.kind == "clarification"
            and event.actor == "fable"
            and event.payload.get("answer") == "The exact correction fact is verified."
        )) == 1
        assert recovered.authenticated_intervention("next-handoff") == terminal
        retry_tracker.close()
        recovered.close()

    asyncio.run(scenario())


async def _recover_next_fable_reserved_child_once(
    *,
    harness: CoordinatorHarness,
    crash_database: Path,
    intervention_id: str,
    question_text: str,
    child_run_id: str,
    exchange_consumed: int,
) -> None:
    """Reopen one staged handoff and invoke only its acknowledged Sol retry owner."""
    recovered = SQLiteStore(crash_database, clock=lambda: "2026-08-10T12:00:00Z")
    assert recovered.recover_active_tasks() == store_module.RecoverySummary(0, 1, 1)
    unknown = recovered.authenticated_intervention(intervention_id)
    assert unknown is not None
    assert unknown.status is store_module.InterventionStatus.RESUME_OUTCOME_UNKNOWN
    assert unknown.directed_binding is not None
    assert unknown.directed_binding.stage == "active_question"
    assert unknown.directed_binding.source_run_id == child_run_id
    assert recovered.agent_run(child_run_id).status == "interrupted"
    questions = recovered._connection.execute(
        "SELECT question_id FROM questions WHERE task_id = ? AND revision = ? AND text = ?",
        ("task-1", 1, question_text),
    ).fetchall()
    assert len(questions) == 1
    question_id = questions[0]["question_id"]
    assert len(tuple(
        event for event in recovered.events_after("session-1", 0)
        if event.kind == "conversation"
        and event.payload.get("question_id") == question_id
    )) == 1
    assert recovered.get_task("task-1", 1).exchange_consumed == exchange_consumed
    acknowledged = recovered.authorize_retry_after_unknown(
        intervention_id,
        expected_resume_generation=unknown.resume_generation,
        acknowledgment_id=f"{intervention_id}-acknowledgment",
    )
    recovered.close()

    retry_store = SQLiteStore(crash_database, clock=lambda: "2026-08-10T12:00:00Z")
    retry_tracker = RepositoryTracker(
        harness.repo, harness.artifacts, git_executable=GIT_EXECUTABLE,
    )
    retry_fable = FakeFable(harness.fable.brief)
    retry_sol = FakeSol()
    retry_sol.queue(_completed(summary="The exact downstream fact is verified."))
    retry_sol.queue(_completed())
    recreated = Coordinator(
        store=retry_store, repository=retry_tracker, runner=RecordingRunner(),
        fable=retry_fable, sol=retry_sol,
        ids=DeterministicIds(task_number=1, run_number=90), repo_root=harness.repo,
        repo_context="Binding AGENTS instructions.",
        trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
    )

    async def fable_probe() -> tuple[bool, str]:
        return True, "subscription_ready"

    async def sol_probe() -> str:
        return "ready"

    retry_lease = ActiveAgentLease()
    workflows = HubWorkflowOrchestrator(
        registry=ProjectRegistry((SimpleNamespace(
            project_id=recreated.project_id,
            store=retry_store,
            coordinator=recreated,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe,
                sol_probe=sol_probe,
            ),
        ),)),
        lease=retry_lease,
        usage_credits_acknowledged=lambda: True,
    )
    retry = workflows.prepare_recovery_resume(
        project_id=recreated.project_id,
        session_id="session-1",
        intervention_id=intervention_id,
        expected_resume_generation=acknowledged.resume_generation,
    )
    await workflows.continue_intervention(retry)

    resumed = retry_store.authenticated_intervention(intervention_id)
    assert resumed is not None
    assert resumed.status is store_module.InterventionStatus.RESUMED
    assert resumed.resume_run_id is not None
    assert retry_sol.answer_fable_question_run_ids == [resumed.resume_run_id]
    assert len(retry_store._connection.execute(
        "SELECT 1 FROM questions WHERE task_id = ? AND revision = ? AND text = ?",
        ("task-1", 1, question_text),
    ).fetchall()) == 1
    assert retry_store.get_task("task-1", 1).exchange_consumed == exchange_consumed
    assert len(tuple(
        event for event in retry_store.events_after("session-1", 0)
        if event.kind == "conversation"
        and event.payload.get("question_id") == question_id
    )) == 1
    assert retry_lease.snapshot() is None
    retry_store.audit_legacy_project_ownership(str(harness.repo))
    retry_tracker.close()
    retry_store.close()


async def _recover_accepted_next_fable_scope_once(
    *,
    harness: CoordinatorHarness,
    crash_database: Path,
    intervention_id: str,
    parent_text: str,
) -> None:
    """Resume one accepted staged scope result without replaying its Fable owner."""
    recovered = SQLiteStore(crash_database, clock=lambda: "2026-08-10T12:00:00Z")
    recovered.audit_legacy_project_ownership(str(harness.repo))
    assert recovered.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)
    staged = recovered.authenticated_intervention(intervention_id)
    assert staged is not None
    assert staged.status is store_module.InterventionStatus.RESUMING
    assert staged.directed_binding is not None
    assert staged.directed_binding.stage == "next_fable"
    staged_run_id = staged.directed_binding.next_run_id
    assert staged_run_id is not None
    assert recovered.agent_run(staged_run_id).status == "running"
    parent = recovered._connection.execute(
        "SELECT question_id, answer_text, answered_by FROM questions "
        "WHERE task_id = ? AND revision = ? AND text = ?",
        ("task-1", 1, parent_text),
    ).fetchone()
    assert parent is not None
    assert tuple(parent)[1:] == (None, None)

    tracker = RepositoryTracker(
        harness.repo, harness.artifacts, git_executable=GIT_EXECUTABLE,
    )
    fable = FakeFable(harness.fable.brief)
    sol = FakeSol()
    recreated = Coordinator(
        store=recovered, repository=tracker, runner=RecordingRunner(),
        fable=fable, sol=sol,
        ids=DeterministicIds(task_number=1, run_number=95), repo_root=harness.repo,
        repo_context="Binding AGENTS instructions.",
        trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
    )

    async def fable_probe() -> tuple[bool, str]:
        return True, "subscription_ready"

    async def sol_probe() -> str:
        return "ready"

    workflows = HubWorkflowOrchestrator(
        registry=ProjectRegistry((SimpleNamespace(
            project_id=recreated.project_id,
            store=recovered,
            coordinator=recreated,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe,
                sol_probe=sol_probe,
            ),
        ),)),
        lease=ActiveAgentLease(),
        usage_credits_acknowledged=lambda: True,
    )
    resumed = await workflows.prepare_resume(
        project_id=recreated.project_id,
        session_id="session-1",
        task_id="task-1",
        revision=1,
    )
    await workflows.run(resumed)

    assert fable.answer_sol_question_prompts == []
    assert recovered.get_task("task-1", 1).state is TaskState.SOL_RUNNING
    assert recovered.get_task("task-1", 2).state is TaskState.AWAITING_SCOPE_APPROVAL
    completed = recovered.authenticated_intervention(intervention_id)
    assert completed is not None
    assert completed.status is store_module.InterventionStatus.RESUMED
    assert recovered.agent_run(staged_run_id).status == "completed"
    answered = recovered.question(parent["question_id"])
    assert answered is not None
    assert answered.answer_text == "Add the explicitly bounded scope path."
    assert answered.answered_by is ConversationActor.FABLE
    assert recovered.get_setting("agent_bridge.baseline.task-1.2") is not None
    recovered.audit_legacy_project_ownership(str(harness.repo))
    tracker.close()
    recovered.close()


@pytest.mark.parametrize(
    "branch", ("permission", "reservation_before_spawn", "reservation_during_sol", "scope"),
)
def test_next_fable_outer_branch_consumes_its_stage_before_crash(
    harness: CoordinatorHarness,
    branch: str,
) -> None:
    """Each outer next-Fable branch commits its result with the staged terminalization."""
    async def scenario() -> None:
        reservation_branch = branch.startswith("reservation_")
        crash_database = harness.database.with_name(f"outer-{branch}-branch.sqlite3")
        accepted_scope_database = harness.database.with_name(
            "outer-scope-accepted-before-commit.sqlite3"
        )
        source_started = asyncio.Event()
        source_released = asyncio.Event()
        child_released = asyncio.Event()
        crash_captured = asyncio.Event()
        crash_state: dict[str, object] = {}
        outer_text = "Which exact approved constraint applies?"
        child_text = "Which focused test proves that constraint?"
        followup_text = "Which exact downstream fact is still needed?"
        followup = DirectedAgentQuestion(
            addressed_to="sol",
            text=followup_text,
            reason="Fable needs one exact downstream fact.",
        )
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", "scope-extra.txt"),
        )
        (harness.repo / "scope-extra.txt").write_text("scope fixture\n", encoding="utf-8")
        final = (
            _answer(
                "I need one exact downstream fact.",
                False,
                directed_question=followup,
            )
            if branch == "permission" or reservation_branch
            else _answer("Add the explicitly bounded scope path.", True, revised_brief=revised)
        )
        harness.sol.queue(_directed_sol_question(outer_text))
        harness.sol.queue(_completed(summary="The focused test proves it."))
        harness.fable.hold_answer_sol_question = source_released
        harness.fable.on_answer_sol_question = source_started.set
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        await source_started.wait()
        task = harness.store.get_task("task-1", 1)
        source = harness.store.active_run_for_task("task-1", 1)
        assert source is not None and source.agent == "fable"

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id,
            store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe,
                sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(
            project_id=runtime.project_id,
            session_id="session-1",
            task_id="task-1",
        )
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)),
            lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id,
            session_id="session-1",
            task_id="task-1",
            intent=InterventionIntent(
                intervention_id=f"outer-{branch}-branch",
                message="Keep the original question bounded.",
                addressed_to=ConversationTarget.FABLE,
                revision=1,
                continuation_generation=task.continuation_generation,
            ),
        )
        harness.runner.release_on_stop = source_released
        harness.fable.hold_answer_sol_question = None

        def exhaust_on_final_fable_answer() -> None:
            if len(harness.fable.answer_sol_question_run_ids) == 3:
                staged = harness.store.intervention(prepared.record.intervention_id)
                assert staged is not None
                assert staged.status is store_module.InterventionStatus.RESUMING
                assert staged.directed_binding is not None
                assert staged.directed_binding.stage == "next_fable"
                assert staged.directed_binding.next_run_id == (
                    harness.fable.answer_sol_question_run_ids[-1]
                )
            if branch == "permission" and len(harness.fable.answer_sol_question_run_ids) == 3:
                harness.store._connection.execute(
                    "UPDATE tasks SET exchange_allowance = 0 WHERE task_id = ? AND revision = ?",
                    ("task-1", 1),
                )

        harness.fable.on_answer_sol_question = exhaust_on_final_fable_answer
        harness.fable.next_clarifications.extend((
            _answer(
                "I need one focused test before answering.",
                False,
                directed_question=DirectedAgentQuestion(
                    addressed_to="sol",
                    text=child_text,
                    reason="Fable needs one exact evidence fact.",
                ),
            ),
            final,
        ))

        def capture_after_branch(*, raise_after: bool = True) -> None:
            terminal = harness.store.intervention(prepared.record.intervention_id)
            assert terminal is not None
            assert terminal.status is (
                store_module.InterventionStatus.RESUMING
                if reservation_branch
                else store_module.InterventionStatus.RESUMED
            )
            events = tuple(
                event for event in harness.store.events_after("session-1", 0)
                if event.kind == "clarification" and event.actor == "fable"
            )
            assert len(events) == 1
            if branch == "permission":
                paused = harness.store.get_task("task-1", 1)
                assert paused.state is TaskState.AWAITING_USER_INPUT
                assert paused.pending is not None
                assert paused.pending["attempted_question"] == followup.to_dict()
            elif reservation_branch:
                nested_rows = harness.store._connection.execute(
                    "SELECT question_id FROM questions WHERE task_id = ? AND revision = ? AND text = ?",
                    ("task-1", 1, followup_text),
                ).fetchall()
                assert len(nested_rows) == 1
                assert terminal.directed_binding is not None
                assert terminal.directed_binding.stage == "active_question"
                assert terminal.directed_binding.question_id == nested_rows[0]["question_id"]
                child = harness.store.agent_run(terminal.directed_binding.source_run_id)
                assert child.agent == "sol" and child.status == "running"
                crash_state["child_run_id"] = child.run_id
                crash_state["exchange_consumed"] = harness.store.get_task(
                    "task-1", 1,
                ).exchange_consumed
            else:
                saved = harness.store.latest_task("task-1")
                assert saved is not None
                assert saved.revision == 2
                assert saved.state is TaskState.AWAITING_SCOPE_APPROVAL
            harness.store.set_setting("agent_bridge.active_session_id", "session-1")
            copied = sqlite3.connect(crash_database)
            harness.store._connection.backup(copied)
            copied.close()
            if raise_after:
                raise RuntimeError("controlled crash after outer next-Fable branch CAS")

        if branch == "permission":
            original_pause = harness.store.pause_fable_answer_evidence_permission

            def crash_after_pause(**kwargs: object) -> object:
                result = original_pause(**kwargs)
                capture_after_branch()
                return result

            harness.store.pause_fable_answer_evidence_permission = crash_after_pause  # type: ignore[method-assign]
        elif reservation_branch:
            original_reserve = harness.store.reserve_fable_answer_evidence_question

            def crash_after_reservation(**kwargs: object) -> object:
                result = original_reserve(**kwargs)
                if (
                    kwargs["completed_next_fable_intervention_id"] is not None
                ):
                    capture_after_branch(raise_after=False)
                    if branch == "reservation_during_sol":
                        harness.sol.hold_answer_fable_question = child_released
                    crash_captured.set()
                return result

            harness.store.reserve_fable_answer_evidence_question = crash_after_reservation  # type: ignore[method-assign]
            if branch == "reservation_before_spawn":
                original_answer_directed = harness.coordinator.answer_directed_question

                async def hold_before_child_spawn(
                    question: store_module.QuestionRecord,
                    **kwargs: object,
                ) -> None:
                    if question.text != followup_text:
                        await original_answer_directed(question, **kwargs)
                        return
                    await child_released.wait()

                harness.coordinator.answer_directed_question = hold_before_child_spawn  # type: ignore[method-assign]
        else:
            original_scope = harness.store.save_scope_revision

            def crash_after_scope(*args: object, **kwargs: object) -> object:
                harness.store.set_setting(
                    "agent_bridge.active_session_id", "session-1",
                )
                accepted_copy = sqlite3.connect(accepted_scope_database)
                harness.store._connection.backup(accepted_copy)
                accepted_copy.close()
                outer = harness.store._connection.execute(
                    "SELECT answer_text FROM questions WHERE task_id = ? AND text = ?",
                    ("task-1", outer_text),
                ).fetchone()
                assert outer is not None
                assert outer["answer_text"] is None
                staged = harness.store.intervention(prepared.record.intervention_id)
                assert staged is not None
                assert staged.status is store_module.InterventionStatus.RESUMING
                tables = (
                    "tasks", "interventions", "agent_runs", "questions", "events", "settings",
                )
                before = {
                    table: tuple(tuple(row) for row in harness.store._connection.execute(
                        f"SELECT * FROM {table} ORDER BY rowid"
                    ))
                    for table in tables
                }
                insert_event = harness.store._insert_event_in_transaction

                def fail_scope_event(
                    session_id: str,
                    task_id: str,
                    actor: str,
                    kind: str,
                    payload: Mapping[str, object],
                ) -> object:
                    if kind == "task_brief":
                        raise RuntimeError("injected scope transaction failure")
                    return insert_event(session_id, task_id, actor, kind, payload)

                harness.store._insert_event_in_transaction = fail_scope_event  # type: ignore[method-assign]
                try:
                    with pytest.raises(RuntimeError, match="injected scope transaction failure"):
                        original_scope(*args, **kwargs)
                finally:
                    harness.store._insert_event_in_transaction = insert_event  # type: ignore[method-assign]
                assert {
                    table: tuple(tuple(row) for row in harness.store._connection.execute(
                        f"SELECT * FROM {table} ORDER BY rowid"
                    ))
                    for table in tables
                } == before
                rolled_back_outer = harness.store._connection.execute(
                    "SELECT answer_text FROM questions WHERE task_id = ? AND text = ?",
                    ("task-1", outer_text),
                ).fetchone()
                assert rolled_back_outer is not None
                assert rolled_back_outer["answer_text"] is None
                rolled_back_stage = harness.store.intervention(prepared.record.intervention_id)
                assert rolled_back_stage is not None
                assert rolled_back_stage.status is store_module.InterventionStatus.RESUMING
                result = original_scope(*args, **kwargs)
                capture_after_branch()
                return result

            harness.store.save_scope_revision = crash_after_scope  # type: ignore[method-assign]

        if reservation_branch:
            continuing = asyncio.create_task(workflows.continue_intervention(prepared))
            await crash_captured.wait()
            if branch == "reservation_during_sol":
                while len(harness.sol.answer_fable_question_run_ids) < 2:
                    await asyncio.sleep(0)
            child_run_id = crash_state["child_run_id"]
            assert isinstance(child_run_id, str)
            harness.runner.release_on_stop = child_released
            await harness.coordinator.stop_task("task-1")
            assert harness.runner.stops[-1] == child_run_id
            await continuing
        else:
            with pytest.raises(RuntimeError, match="controlled crash after outer next-Fable branch CAS"):
                await workflows.continue_intervention(prepared)
        await approval
        assert lease.snapshot() is None
        harness.tracker.close()
        harness.store.close()

        if reservation_branch:
            child_run_id = crash_state["child_run_id"]
            exchange_consumed = crash_state["exchange_consumed"]
            assert isinstance(child_run_id, str)
            assert isinstance(exchange_consumed, int)
            await _recover_next_fable_reserved_child_once(
                harness=harness,
                crash_database=crash_database,
                intervention_id=prepared.record.intervention_id,
                question_text=followup_text,
                child_run_id=child_run_id,
                exchange_consumed=exchange_consumed,
            )
        else:
            recovered = SQLiteStore(crash_database, clock=lambda: "2026-08-10T12:00:00Z")
            assert recovered.recover_active_tasks().tasks_interrupted == 0
            terminal = recovered.authenticated_intervention(prepared.record.intervention_id)
            assert terminal is not None
            assert terminal.status is store_module.InterventionStatus.RESUMED
            assert len(tuple(
                event for event in recovered.events_after("session-1", 0)
                if event.kind == "clarification" and event.actor == "fable"
            )) == 1
            if branch == "scope":
                parent = recovered._connection.execute(
                    "SELECT answer_text, answered_by FROM questions "
                    "WHERE task_id = ? AND revision = ? AND text = ?",
                    ("task-1", 1, outer_text),
                ).fetchone()
                assert parent is not None
                assert tuple(parent) == (
                    "Add the explicitly bounded scope path.", "fable",
                )
                assert recovered.get_task("task-1", 1).state is TaskState.SOL_RUNNING
                scoped = recovered.get_task("task-1", 2)
                assert scoped.state is TaskState.AWAITING_SCOPE_APPROVAL
                assert recovered.get_setting("agent_bridge.baseline.task-1.2") is not None
                assert terminal.directed_binding is not None
                assert terminal.directed_binding.next_run_id is not None
                assert recovered.agent_run(
                    terminal.directed_binding.next_run_id,
                ).status == "completed"
                assert len(tuple(
                    event for event in recovered.events_after("session-1", 0)
                    if event.kind == "task_brief"
                    and event.payload.get("brief", {}).get("revision") == 2
                )) == 1
                assert len(tuple(
                    event for event in recovered.events_after("session-1", 0)
                    if event.kind == "conversation"
                    and event.payload.get("reply_to_question_id") is not None
                    and event.payload.get("text")
                    == "Add the explicitly bounded scope path."
                )) == 1
            recovered.audit_legacy_project_ownership(str(harness.repo))
            recovered.close()
            if branch == "scope":
                await _recover_accepted_next_fable_scope_once(
                    harness=harness,
                    crash_database=accepted_scope_database,
                    intervention_id=prepared.record.intervention_id,
                    parent_text=outer_text,
                )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "crash_boundary", ("before_outer_answer_cas", "after_outer_answer_cas"),
)
def test_next_fable_outer_answer_is_not_terminal_before_its_answer_cas(
    harness: CoordinatorHarness,
    crash_boundary: str,
) -> None:
    """An unknown pre-CAS Fable result cannot replay until one exact acknowledgment."""
    async def scenario() -> None:
        crash_database = harness.database.with_name(
            f"outer-answer-{crash_boundary}.sqlite3"
        )
        source_started = asyncio.Event()
        source_released = asyncio.Event()
        outer_text = "Which exact approved constraint applies?"
        child_text = "Which focused test proves that constraint?"
        harness.sol.queue(_directed_sol_question(outer_text))
        harness.sol.queue(_completed(summary="The focused test proves it."))
        harness.sol.queue(_completed())
        harness.fable.hold_answer_sol_question = source_released
        harness.fable.on_answer_sol_question = source_started.set
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        await source_started.wait()
        source = harness.store.active_run_for_task("task-1", 1)
        task = harness.store.get_task("task-1", 1)
        assert source is not None and source.agent == "fable"

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id,
            store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe,
                sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(
            project_id=runtime.project_id,
            session_id="session-1",
            task_id="task-1",
        )
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)),
            lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id,
            session_id="session-1",
            task_id="task-1",
            intent=InterventionIntent(
                intervention_id=f"outer-answer-{crash_boundary}",
                message="Keep the original question bounded.",
                addressed_to=ConversationTarget.FABLE,
                revision=1,
                continuation_generation=task.continuation_generation,
            ),
        )
        harness.runner.release_on_stop = source_released
        harness.fable.hold_answer_sol_question = None
        harness.fable.next_clarifications.extend((
            _answer(
                "I need one focused test before answering.",
                False,
                directed_question=DirectedAgentQuestion(
                    addressed_to="sol",
                    text=child_text,
                    reason="Fable needs one exact evidence fact.",
                ),
            ),
            _answer("The approved constraint is already covered.", False),
        ))
        answer_outer = harness.store.answer_question_and_prepare_resume

        def capture_crash_image() -> None:
            harness.store.set_setting("agent_bridge.active_session_id", "session-1")
            copied = sqlite3.connect(crash_database)
            harness.store._connection.backup(copied)
            copied.close()

        def crash_at_outer_answer(**kwargs: object) -> object:
            assert kwargs["question_id"] != ""
            outer = harness.store.question(kwargs["question_id"])
            assert outer is not None and outer.text == outer_text
            intervention = harness.store.intervention(prepared.record.intervention_id)
            assert intervention is not None
            if crash_boundary == "before_outer_answer_cas":
                assert intervention.status is store_module.InterventionStatus.RESUMING
                assert outer.answer_text is None
                capture_crash_image()
                raise RuntimeError("controlled crash before outer answer CAS")
            answered = answer_outer(**kwargs)
            terminal = harness.store.intervention(prepared.record.intervention_id)
            assert terminal is not None
            assert terminal.status is store_module.InterventionStatus.RESUMED
            assert harness.store.question(outer.question_id).answer_text == (  # type: ignore[union-attr]
                "The approved constraint is already covered."
            )
            capture_crash_image()
            raise RuntimeError("controlled crash after outer answer CAS")

        harness.store.answer_question_and_prepare_resume = crash_at_outer_answer  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="controlled crash"):
            await workflows.continue_intervention(prepared)
        await approval
        assert len(harness.fable.answer_sol_question_run_ids) == 3
        assert lease.snapshot() is None
        harness.tracker.close()
        harness.store.close()

        recovered = SQLiteStore(
            crash_database, clock=lambda: "2026-08-10T12:00:00Z",
        )
        expected_runs = 1 if crash_boundary == "before_outer_answer_cas" else 0
        assert recovered.recover_active_tasks() == store_module.RecoverySummary(
            0, 1, expected_runs,
        )
        recovered_task = recovered.get_task("task-1", 1)
        assert recovered_task.state is TaskState.INTERRUPTED
        assert recovered_task.continuation_state is TaskState.SOL_RUNNING
        terminal = recovered.authenticated_intervention(prepared.record.intervention_id)
        assert terminal is not None
        if crash_boundary == "before_outer_answer_cas":
            assert terminal.status is store_module.InterventionStatus.RESUME_OUTCOME_UNKNOWN
            outer = recovered.question(terminal.directed_binding.parent_question_id)  # type: ignore[union-attr]
            assert outer is not None and outer.answer_text is None
            unavailable_runtime = SimpleNamespace(
                project_id=harness.coordinator.project_id,
                store=recovered,
                coordinator=SimpleNamespace(),
            )
            unavailable_workflows = HubWorkflowOrchestrator(
                registry=ProjectRegistry((unavailable_runtime,)),
                lease=ActiveAgentLease(),
                usage_credits_acknowledged=lambda: True,
            )
            with pytest.raises(RuntimeError, match="recovery is unavailable"):
                unavailable_workflows.prepare_recovery_resume(
                    project_id=unavailable_runtime.project_id,
                    session_id="session-1",
                    intervention_id=terminal.intervention_id,
                    expected_resume_generation=terminal.resume_generation,
                )
            acknowledged = recovered.authorize_retry_after_unknown(
                terminal.intervention_id,
                expected_resume_generation=terminal.resume_generation,
                acknowledgment_id="outer-answer-acknowledgment",
            )
            assert acknowledged.status is store_module.InterventionStatus.READY
            assert recovered.authorize_retry_after_unknown(
                terminal.intervention_id,
                expected_resume_generation=terminal.resume_generation,
                acknowledgment_id="outer-answer-acknowledgment",
            ) == acknowledged
            with pytest.raises(RuntimeError, match="generation changed"):
                recovered.authorize_retry_after_unknown(
                    terminal.intervention_id,
                    expected_resume_generation=terminal.resume_generation,
                    acknowledgment_id="outer-answer-second-acknowledgment",
                )
            retry_fable = FakeFable(harness.fable.brief)
            retry_fable.next_clarifications.append(
                _answer("The approved constraint is already covered.", False)
            )
            retry_sol = FakeSol()
            retry_sol.queue(_completed())
            retry_tracker = RepositoryTracker(
                harness.repo, harness.artifacts, git_executable=GIT_EXECUTABLE,
            )
            retry_coordinator = Coordinator(
                store=recovered, repository=retry_tracker, runner=RecordingRunner(),
                fable=retry_fable, sol=retry_sol,
                ids=DeterministicIds(task_number=1, run_number=80),
                repo_root=harness.repo, repo_context="Binding AGENTS instructions.",
                trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
            )
            retry_runtime = SimpleNamespace(
                project_id=retry_coordinator.project_id,
                store=recovered,
                coordinator=retry_coordinator,
                readiness=RuntimeReadiness(
                    initial=RuntimeStatus(True, "subscription_ready", "ready"),
                    fable_probe=fable_probe,
                    sol_probe=sol_probe,
                ),
            )
            retry_lease = ActiveAgentLease()
            retry_workflows = HubWorkflowOrchestrator(
                registry=ProjectRegistry((retry_runtime,)),
                lease=retry_lease,
                usage_credits_acknowledged=lambda: True,
            )
            retry = retry_workflows.prepare_recovery_resume(
                project_id=retry_coordinator.project_id,
                session_id="session-1",
                intervention_id=acknowledged.intervention_id,
                expected_resume_generation=acknowledged.resume_generation,
            )
            await retry_workflows.continue_intervention(retry)
            final = recovered.authenticated_intervention(acknowledged.intervention_id)
            assert final is not None
            assert final.status is store_module.InterventionStatus.RESUMED
            assert len(retry_fable.answer_sol_question_run_ids) == 1
            assert len(tuple(
                event for event in recovered.events_after("session-1", 0)
                if event.kind == "conversation"
                and event.payload.get("reply_to_question_id") == outer.question_id
            )) == 1
            assert retry_lease.snapshot() is None
            retry_tracker.close()
        else:
            assert terminal.status is store_module.InterventionStatus.RESUMED
            outer = recovered.question(terminal.directed_binding.parent_question_id)  # type: ignore[union-attr]
            assert outer is not None
            assert outer.answer_text == "The approved constraint is already covered."
            retry_tracker = RepositoryTracker(
                harness.repo, harness.artifacts, git_executable=GIT_EXECUTABLE,
            )
            retry_fable = FakeFable(harness.fable.brief)
            retry_sol = FakeSol()
            retry_sol.queue(_completed())
            retry_coordinator = Coordinator(
                store=recovered, repository=retry_tracker, runner=RecordingRunner(),
                fable=retry_fable, sol=retry_sol,
                ids=DeterministicIds(task_number=1, run_number=80),
                repo_root=harness.repo, repo_context="Binding AGENTS instructions.",
                trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
            )
            await retry_coordinator.resume_task("task-1")
            assert retry_fable.answer_sol_question_run_ids == []
            assert retry_sol.resume_threads == [THREAD_ID]
            retry_tracker.close()
        recovered.audit_legacy_project_ownership(str(harness.repo))
        recovered.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "outcome",
    ("directed_before_spawn", "directed_during_sol", "escalation", "scope", "permission"),
)
def test_next_fable_result_stays_resuming_until_its_outcome_and_event_are_durable(
    harness: CoordinatorHarness,
    outcome: str,
) -> None:
    """A staged accepted Fable result cannot terminalize before its branch CAS."""
    async def scenario() -> None:
        directed_outcome = outcome.startswith("directed_")
        source_released = asyncio.Event()
        child_released = asyncio.Event()
        crash_captured = asyncio.Event()
        crash_state: dict[str, object] = {}
        correction_started = asyncio.Event()
        crash_database = harness.database.with_name("next-directed-outcome.sqlite3")
        first_directed = DirectedAgentQuestion(
            addressed_to="sol",
            text="Which exact correction fact is verified?",
            reason="Fable needs one bounded correction fact.",
        )
        second_directed = DirectedAgentQuestion(
            addressed_to="sol",
            text="Which exact downstream fact is still needed?",
            reason="Fable needs one further bounded fact.",
        )
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", "scope-extra.txt"),
        )
        (harness.repo / "scope-extra.txt").write_text("scope fixture\n", encoding="utf-8")
        if directed_outcome or outcome == "permission":
            second_clarification = _answer(
                "I need one exact downstream fact.",
                False,
                directed_question=second_directed,
            )
        elif outcome == "escalation":
            second_clarification = _escalation(
                "Please choose the exact downstream option.",
                reasoning="The bounded evidence remains ambiguous.",
            )
        elif outcome == "scope":
            second_clarification = _answer(
                "Add the explicitly bounded scope path.",
                True,
                revised_brief=revised,
            )
        else:
            raise AssertionError(f"unknown next-Fable outcome {outcome}")
        harness.sol.queue(_completed())
        harness.fable.next_verdicts.append(
            _verdict(harness.fable.brief, status="corrections_required")
        )
        harness.sol.hold_resume = source_released
        harness.sol.on_resume = correction_started.set
        harness.runner.release_on_stop = source_released
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1)
        )
        await correction_started.wait()
        source = harness.store.active_run_for_task("task-1", 1)
        source_task = harness.store.get_task("task-1", 1)
        assert source is not None
        harness.store.set_agent_run_session(source.run_id, THREAD_ID)

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id,
            store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe,
                sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
        )
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id,
            session_id="session-1",
            task_id="task-1",
            intent=InterventionIntent(
                intervention_id="next-directed-outcome",
                message="Keep the correction bounded.",
                addressed_to=ConversationTarget.FABLE,
                revision=1,
                continuation_generation=source_task.continuation_generation,
            ),
        )
        harness.sol.hold_resume = None
        if outcome == "permission":
            original_start_next_fable_stage = (
                harness.store.start_next_fable_intervention_stage
            )

            def exhaust_after_starting_next_fable_stage(
                intervention_id: str, *, run_id: str,
            ) -> store_module.AgentRunRecord:
                started = original_start_next_fable_stage(
                    intervention_id, run_id=run_id,
                )
                if intervention_id == "next-directed-outcome":
                    harness.store._connection.execute(
                        "UPDATE tasks SET exchange_allowance = 0 "
                        "WHERE task_id = ? AND revision = ?",
                        ("task-1", 1),
                    )
                return started

            harness.store.start_next_fable_intervention_stage = exhaust_after_starting_next_fable_stage  # type: ignore[method-assign]
        if directed_outcome:
            original_reserve = harness.store.reserve_fable_clarification_evidence_question

            def capture_directed_reservation(**kwargs: object) -> object:
                result = original_reserve(**kwargs)
                if kwargs["completed_next_fable_intervention_id"] is None:
                    return result
                current = harness.store.authenticated_intervention(
                    "next-directed-outcome",
                )
                assert current is not None
                assert current.status is store_module.InterventionStatus.RESUMING
                assert current.directed_binding is not None
                assert current.directed_binding.stage == "active_question"
                assert current.directed_binding.question_id == result[1].question_id
                child = harness.store.agent_run(current.directed_binding.source_run_id)
                assert child.agent == "sol" and child.status == "running"
                crash_state["child_run_id"] = child.run_id
                crash_state["exchange_consumed"] = harness.store.get_task(
                    "task-1", 1,
                ).exchange_consumed
                harness.store.set_setting(
                    "agent_bridge.active_session_id", "session-1",
                )
                copied = sqlite3.connect(crash_database)
                harness.store._connection.backup(copied)
                copied.close()
                if outcome == "directed_during_sol":
                    harness.sol.hold_answer_fable_question = child_released
                crash_captured.set()
                return result

            harness.store.reserve_fable_clarification_evidence_question = capture_directed_reservation  # type: ignore[method-assign]
            if outcome == "directed_before_spawn":
                original_answer_directed = harness.coordinator.answer_directed_question

                async def hold_before_child_spawn(
                    question: store_module.QuestionRecord,
                    **kwargs: object,
                ) -> None:
                    if question.text != second_directed.text:
                        await original_answer_directed(question, **kwargs)
                        return
                    await child_released.wait()

                harness.coordinator.answer_directed_question = hold_before_child_spawn  # type: ignore[method-assign]
        original_route = harness.coordinator._route_clarification

        async def route_after_accepted_result(*args: object, **kwargs: object) -> None:
            staged = harness.store.intervention("next-directed-outcome")
            assert staged is not None
            if (
                staged.directed_binding is None
                or staged.directed_binding.stage != "next_fable"
            ):
                await original_route(*args, **kwargs)
                return
            assert staged.status is store_module.InterventionStatus.RESUMING

            await original_route(*args, **kwargs)
            if directed_outcome:
                return
            terminal = harness.store.intervention("next-directed-outcome")
            assert terminal is not None
            assert terminal.status is store_module.InterventionStatus.RESUMED
            if outcome == "permission":
                task = harness.store.get_task("task-1", 1)
                assert task.state is TaskState.AWAITING_USER_INPUT
                assert task.continuation_state is TaskState.FABLE_CLARIFYING
                assert task.pending is not None
                assert task.pending["attempted_question"] == second_directed.to_dict()
                matching_events = tuple(
                    event for event in harness.store.events_after("session-1", 0)
                    if event.kind == "clarification"
                    and event.actor == "fable"
                    and event.payload.get("directed_question", {}).get("text")
                    == second_directed.text
                )
                permission_events = tuple(
                    event for event in harness.store.events_after("session-1", 0)
                    if event.kind == "conversation"
                    and event.payload.get("message_type") == "status"
                )
                assert len(permission_events) == 1
                assert matching_events[0].sequence < permission_events[0].sequence
            elif outcome == "escalation":
                task = harness.store.get_task("task-1", 1)
                assert task.state is TaskState.AWAITING_USER_INPUT
                assert task.continuation_state is TaskState.SOL_RUNNING
                assert task.pending is not None
                assert task.pending["question_for_user"] == "Please choose the exact downstream option."
                matching_events = tuple(
                    event for event in harness.store.events_after("session-1", 0)
                    if event.kind == "clarification"
                    and event.actor == "fable"
                    and event.payload.get("status") == "escalate_to_user"
                )
            else:
                task = harness.store.latest_task("task-1")
                assert task is not None
                assert task.revision == 2
                assert task.state is TaskState.AWAITING_SCOPE_APPROVAL
                matching_events = tuple(
                    event for event in harness.store.events_after("session-1", 0)
                    if event.kind == "clarification"
                    and event.actor == "fable"
                    and event.payload.get("scope_changed") is True
                )
            assert len(matching_events) == 1
            harness.store.set_setting("agent_bridge.active_session_id", "session-1")
            copied = sqlite3.connect(crash_database)
            harness.store._connection.backup(copied)
            copied.close()
            raise RuntimeError("controlled crash after next Fable directed outcome")

        harness.coordinator._route_clarification = route_after_accepted_result  # type: ignore[method-assign]
        harness.fable.next_clarifications.extend((
            _answer("I need one exact correction fact.", False, directed_question=first_directed),
            second_clarification,
        ))
        if directed_outcome:
            continuing = asyncio.create_task(workflows.continue_intervention(prepared))
            await crash_captured.wait()
            if outcome == "directed_during_sol":
                while len(harness.sol.answer_fable_question_run_ids) < 2:
                    await asyncio.sleep(0)
            child_run_id = crash_state["child_run_id"]
            assert isinstance(child_run_id, str)
            harness.runner.release_on_stop = child_released
            await harness.coordinator.stop_task("task-1")
            assert harness.runner.stops[-1] == child_run_id
            await continuing
        else:
            with pytest.raises(RuntimeError, match="controlled crash"):
                await workflows.continue_intervention(prepared)
        await approval
        harness.tracker.close()
        harness.store.close()

        if directed_outcome:
            child_run_id = crash_state["child_run_id"]
            exchange_consumed = crash_state["exchange_consumed"]
            assert isinstance(child_run_id, str)
            assert isinstance(exchange_consumed, int)
            await _recover_next_fable_reserved_child_once(
                harness=harness,
                crash_database=crash_database,
                intervention_id="next-directed-outcome",
                question_text=second_directed.text,
                child_run_id=child_run_id,
                exchange_consumed=exchange_consumed,
            )
        else:
            recovered = SQLiteStore(crash_database, clock=lambda: "2026-08-10T12:00:00Z")
            assert recovered.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)
            terminal = recovered.authenticated_intervention("next-directed-outcome")
            assert terminal is not None
            assert terminal.status is store_module.InterventionStatus.RESUMED
            if outcome == "scope":
                assert recovered.latest_task("task-1").state is TaskState.AWAITING_SCOPE_APPROVAL  # type: ignore[union-attr]
            else:
                assert recovered.get_task("task-1", 1).state is TaskState.AWAITING_USER_INPUT
            recovered.audit_legacy_project_ownership(str(harness.repo))
            recovered.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("route", "intervention_id"),
    (
        (ConversationTarget.SOL, "unknown-same-route"),
        (ConversationTarget.FABLE, "unknown-cross-route"),
    ),
)
def test_hub_unknown_ack_reopen_recovers_one_same_or_cross_route_attempt(
    harness: CoordinatorHarness,
    route: ConversationTarget,
    intervention_id: str,
) -> None:
    """Hub recovery must wait for one acknowledgment before replaying its exact route."""
    async def scenario() -> None:
        initial = harness.store.save_task(
            "session-1", harness.fable.brief, TaskState.AWAITING_USER_APPROVAL,
        )
        approved = harness.store.approve_task(
            initial.task_id, initial.revision, baseline_id="baseline-1",
        )
        active = harness.store.transition_task(
            approved.task_id, approved.revision,
            expected=TaskState.AWAITING_USER_APPROVAL, target=TaskState.SOL_RUNNING,
        )
        harness.store.set_sol_thread(active.task_id, active.revision, THREAD_ID)
        harness.store.set_fable_session(
            active.task_id, active.revision, "fable-session-1",
        )
        harness.store.start_agent_run("source-run", active.task_id, active.revision, "sol")
        harness.store.set_agent_run_session("source-run", THREAD_ID)
        harness.store.set_pending_context(
            active.task_id, active.revision, expected=TaskState.SOL_RUNNING,
            pending={"sol_run_id": "source-run", "prompt": "Build the bridge"},
        )
        pending = harness.store.create_intervention_and_request_stop(
            intervention_id=intervention_id, session_id="session-1",
            task_id=active.task_id, revision=active.revision,
            expected_source_generation=active.continuation_generation,
            message="Preserve this exact direction.", addressed_to=route, routed_to=route,
            run_id="source-run",
        )
        harness.store.finish_agent_run("source-run", status="interrupted", exit_code=-15)
        harness.store.mark_intervention_ready(intervention_id, run_id="source-run")
        await harness.coordinator.resume_intervention(
            intervention_id, resume_attempt_id="attempt-before-crash",
            resume_run_id="run-before-crash",
        )

        harness.tracker.close()
        harness.store.close()
        recovered = SQLiteStore(harness.database, clock=lambda: "2026-08-10T12:00:00Z")
        assert recovered.recover_active_tasks().tasks_interrupted == 1
        unknown = recovered.intervention(intervention_id)
        assert unknown is not None
        assert unknown.status.value == "resume_outcome_unknown"
        assert recovered.get_task(active.task_id, active.revision).continuation_generation == (
            unknown.resume_generation
        )

        stale_runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=recovered,
            coordinator=SimpleNamespace(),
        )
        stale_workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((stale_runtime,)), lease=ActiveAgentLease(),
            usage_credits_acknowledged=lambda: True,
        )
        with pytest.raises(RuntimeError, match="recovery is unavailable"):
            stale_workflows.prepare_recovery_resume(
                project_id=harness.coordinator.project_id, session_id="session-1",
                intervention_id=intervention_id,
                expected_resume_generation=unknown.resume_generation,
            )

        acknowledged = recovered.authorize_retry_after_unknown(
            intervention_id, expected_resume_generation=unknown.resume_generation,
            acknowledgment_id="acknowledgment-1",
        )
        assert recovered.authorize_retry_after_unknown(
            intervention_id, expected_resume_generation=unknown.resume_generation,
            acknowledgment_id="acknowledgment-1",
        ) == acknowledged
        with pytest.raises(RuntimeError, match="generation changed"):
            recovered.authorize_retry_after_unknown(
                intervention_id, expected_resume_generation=unknown.resume_generation,
                acknowledgment_id="acknowledgment-2",
            )
        recovered.close()

        reopened_store = SQLiteStore(harness.database, clock=lambda: "2026-08-10T12:00:00Z")
        reopened_tracker = RepositoryTracker(
            harness.repo, harness.artifacts, git_executable=GIT_EXECUTABLE,
        )
        resumed_fable = FakeFable(harness.fable.brief)
        resumed_sol = FakeSol()
        resumed_sol.queue(SolOutcome.from_dict({
            "status": "blocked",
            "summary": "The exact user decision is required.",
            "changed_files": [],
            "commands_run": [],
            "known_failures": [],
            "remaining_risks": [],
            "architecture_docs": "No durable architecture change.",
            "question": None,
        }))
        recreated = Coordinator(
            store=reopened_store, repository=reopened_tracker, runner=RecordingRunner(),
            fable=resumed_fable, sol=resumed_sol,
            ids=DeterministicIds(task_number=1, run_number=40), repo_root=harness.repo,
            repo_context="Binding AGENTS instructions.",
            trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
        )

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=recreated.project_id, store=reopened_store, coordinator=recreated,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_recovery_resume(
            project_id=recreated.project_id, session_id="session-1",
            intervention_id=intervention_id,
            expected_resume_generation=acknowledged.resume_generation,
        )
        await workflows.continue_intervention(prepared)

        resumed = reopened_store.intervention(intervention_id)
        assert resumed is not None
        assert resumed.status.value == "resumed"
        assert reopened_store.get_task(active.task_id, active.revision).continuation_generation == (
            resumed.resume_generation
        )
        if route is ConversationTarget.SOL:
            assert resumed_sol.resume_run_ids == [resumed.resume_run_id]
            assert resumed_fable.clarification_run_ids == []
        else:
            assert resumed_fable.clarification_run_ids == [resumed.resume_run_id]
            assert len(resumed_sol.resume_run_ids) == 1
        assert lease.snapshot() is None
        reopened_tracker.close()
        reopened_store.close()

    asyncio.run(scenario())


def test_hub_unknown_ack_reopen_recovers_one_directed_fable_attempt(
    harness: CoordinatorHarness,
) -> None:
    """An acknowledged paused Fable answer remains a directed continuation after reopen."""
    async def scenario() -> None:
        entered = asyncio.Event()
        release_source = asyncio.Event()
        harness.sol.queue(_directed_sol_question("Which exact rule applies?"))
        harness.fable.on_answer_sol_question = entered.set
        harness.fable.hold_answer_sol_question = release_source
        harness.runner.release_on_stop = release_source
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(harness.coordinator.approve_task("task-1", revision=1))
        await entered.wait()
        waiting = harness.store.get_task("task-1", 1)
        source = harness.store.active_run_for_task("task-1", 1)
        assert source is not None
        harness.store.set_agent_run_session(source.run_id, "fable-session-1")

        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=harness.store,
            coordinator=harness.coordinator,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        lease.acquire(project_id=runtime.project_id, session_id="session-1", task_id="task-1")
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = workflows.prepare_intervention(
            project_id=runtime.project_id, session_id="session-1", task_id="task-1",
            intent=InterventionIntent(
                intervention_id="unknown-directed", message="Use the exact approved rule.",
                addressed_to=ConversationTarget.FABLE, revision=1,
                continuation_generation=waiting.continuation_generation,
            ),
        )
        await harness.coordinator.continue_intervention(prepared.record.intervention_id)
        await approval
        lease.release(prepared.lease_token)
        await harness.coordinator.resume_intervention(
            prepared.record.intervention_id, resume_attempt_id="attempt-before-crash",
            resume_run_id="run-before-crash",
        )

        harness.tracker.close()
        harness.store.close()
        recovered = SQLiteStore(harness.database, clock=lambda: "2026-08-10T12:00:00Z")
        assert recovered.recover_active_tasks().tasks_interrupted == 1
        unknown = recovered.intervention("unknown-directed")
        assert unknown is not None
        assert unknown.status.value == "resume_outcome_unknown"

        stale_runtime = SimpleNamespace(
            project_id=harness.coordinator.project_id, store=recovered,
            coordinator=SimpleNamespace(),
        )
        stale_workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((stale_runtime,)), lease=ActiveAgentLease(),
            usage_credits_acknowledged=lambda: True,
        )
        with pytest.raises(RuntimeError, match="recovery is unavailable"):
            stale_workflows.prepare_recovery_resume(
                project_id=harness.coordinator.project_id, session_id="session-1",
                intervention_id="unknown-directed",
                expected_resume_generation=unknown.resume_generation,
            )
        acknowledged = recovered.authorize_retry_after_unknown(
            "unknown-directed", expected_resume_generation=unknown.resume_generation,
            acknowledgment_id="acknowledgment-1",
        )
        assert recovered.get_task("task-1", 1).continuation_generation == (
            acknowledged.resume_generation
        )
        assert recovered.authorize_retry_after_unknown(
            "unknown-directed", expected_resume_generation=unknown.resume_generation,
            acknowledgment_id="acknowledgment-1",
        ) == acknowledged
        with pytest.raises(RuntimeError, match="generation changed"):
            recovered.authorize_retry_after_unknown(
                "unknown-directed", expected_resume_generation=unknown.resume_generation,
                acknowledgment_id="acknowledgment-2",
            )
        recovered.close()

        reopened_store = SQLiteStore(harness.database, clock=lambda: "2026-08-10T12:00:00Z")
        reopened_tracker = RepositoryTracker(
            harness.repo, harness.artifacts, git_executable=GIT_EXECUTABLE,
        )
        resumed_fable = FakeFable(harness.fable.brief)
        resumed_sol = FakeSol()
        resumed_sol.queue(_completed())
        recreated = Coordinator(
            store=reopened_store, repository=reopened_tracker, runner=RecordingRunner(),
            fable=resumed_fable, sol=resumed_sol,
            ids=DeterministicIds(task_number=1, run_number=50), repo_root=harness.repo,
            repo_context="Binding AGENTS instructions.",
            trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
        )
        runtime = SimpleNamespace(
            project_id=recreated.project_id, store=reopened_store, coordinator=recreated,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        retry = workflows.prepare_recovery_resume(
            project_id=recreated.project_id, session_id="session-1",
            intervention_id="unknown-directed",
            expected_resume_generation=acknowledged.resume_generation,
        )
        await workflows.continue_intervention(retry)

        resumed = reopened_store.intervention("unknown-directed")
        assert resumed is not None
        assert resumed.status.value == "resumed"
        assert reopened_store.get_task("task-1", 1).continuation_generation == (
            resumed.resume_generation
        )
        assert resumed_fable.answer_sol_question_run_ids == [resumed.resume_run_id]
        assert [call[:2] for call in resumed_fable.answer_sol_question_prompts] == [(
            "task-1",
            "Which exact rule applies?\n\nIntervention guidance:\nUse the exact approved rule.",
        )]
        assert resumed_fable.answer_sol_question_prompts[0][2].endswith(
            "Binding AGENTS instructions."
        )
        assert len(resumed_sol.resume_run_ids) == 1
        assert lease.snapshot() is None
        reopened_tracker.close()
        reopened_store.close()

    asyncio.run(scenario())


def test_directed_user_question_pauses_and_agents_cannot_answer_it(
    harness: CoordinatorHarness,
) -> None:
    """Changing the recipient must not let an agent bypass the user answer CAS."""
    async def scenario() -> None:
        harness.sol.queue(_directed_sol_question(
            "Which deployment window does the user approve?", addressed_to="user",
        ))

        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        await harness.coordinator.approve_task("task-1", revision=1)

        task = harness.store.get_task("task-1", 1)
        assert task.state is TaskState.AWAITING_USER_INPUT
        question = harness.store._unanswered_question_for_task("task-1", 1)
        assert question is not None
        assert question.routed_to is ConversationTarget.USER

        with pytest.raises(RuntimeError, match="user-routed"):
            await harness.coordinator.answer_directed_question(question)

        persisted = harness.store.question(question.question_id)
        assert persisted is not None
        assert persisted.answer_text is None
        assert harness.fable.answer_sol_question_prompts == []
        assert harness.sol.resume_threads == []

    asyncio.run(scenario())


def test_directed_exchanges_pause_at_three_then_a_grant_allows_only_three_more(
    harness: CoordinatorHarness,
) -> None:
    """The fourth automatic hop must be a durable user permission boundary."""
    async def scenario() -> None:
        for ordinal in range(1, 8):
            harness.sol.queue(_directed_sol_question(
                f"Fable clarification {ordinal}?",
            ))

        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        await harness.coordinator.approve_task("task-1", revision=1)

        exhausted = harness.store.get_task("task-1", 1)
        assert exhausted.state is TaskState.AWAITING_USER_INPUT
        assert (exhausted.exchange_allowance, exhausted.exchange_consumed) == (0, 3)
        assert len(harness.fable.answer_sol_question_prompts) == 3
        permission = [
            event for event in harness.store.events_after("session-1", 0)
            if event.kind == "conversation"
            and event.payload["message_type"] == "status"
        ]
        assert len(permission) == 1
        assert permission[0].payload["sender"] == "system"
        assert permission[0].payload["addressed_to"] == "user"
        assert permission[0].payload["routed_to"] == "user"

        prepared_task = harness.coordinator.prepare_exchange_grant(
            session_id="session-1",
            task_id="task-1",
            revision=1,
            continuation_generation=1,
            request_id="grant-after-three",
        )
        assert prepared_task.state is TaskState.SOL_RUNNING
        prepared = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id,
            session_id="session-1",
            task_id="task-1",
            revision=1,
        )
        assert prepared is not None and prepared.action == "exchange_grant"

        await harness.coordinator.run_prepared_conversation_action(
            "task-1", "exchange_grant",
        )

        exhausted_again = harness.store.get_task("task-1", 1)
        assert exhausted_again.state is TaskState.AWAITING_USER_INPUT
        assert (exhausted_again.exchange_allowance, exhausted_again.exchange_consumed) == (0, 6)
        assert len(harness.fable.answer_sol_question_prompts) == 6
        permission_cards = [
            event for event in harness.store.events_after("session-1", 0)
            if event.kind == "conversation"
            and event.payload["message_type"] == "status"
        ]
        assert len(permission_cards) == 2

    asyncio.run(scenario())


def test_directed_exchange_restart_after_visible_reservation_reuses_one_charge_and_question(
    harness: CoordinatorHarness,
) -> None:
    """An atomic reservation/visible-question checkpoint survives full recreation."""
    async def scenario() -> None:
        harness.sol.queue(_directed_sol_question("Which invariant applies?"))
        original = harness.coordinator.answer_directed_question
        captured: list[object] = []

        async def crash_after_reservation(question: object) -> None:
            captured.append(question)
            raise RuntimeError("controlled crash after visible reservation")

        harness.coordinator.answer_directed_question = crash_after_reservation  # type: ignore[method-assign]
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        with pytest.raises(PreparedActionFailed):
            await harness.coordinator.approve_task("task-1", revision=1)

        paused = harness.store.get_task("task-1", 1)
        assert (paused.exchange_allowance, paused.exchange_consumed) == (2, 1)
        assert len(captured) == 1
        question = captured[0]
        assert isinstance(question, store_module.QuestionRecord)
        before_events = harness.store.events_after("session-1", 0)
        before_question_ids = harness.store._connection.execute(  # noqa: SLF001 - durable checkpoint
            "SELECT question_id, exchange_id FROM questions WHERE task_id = ? ORDER BY rowid",
            ("task-1",),
        ).fetchall()
        assert len(before_question_ids) == 1
        assert harness.fable.answer_sol_question_prompts == []
        fable_session_id = paused.fable_session_id
        sol_thread_id = paused.sol_thread_id

        # Reservation and visible-question publication are one SQLite transaction;
        # recreate exactly after that committed boundary, before any answer call.
        harness.tracker.close()
        harness.store.close()
        reopened_store = SQLiteStore(harness.database, clock=lambda: "2026-08-10T12:00:00Z")
        reopened_tracker = RepositoryTracker(
            harness.repo, harness.artifacts, git_executable=GIT_EXECUTABLE,
        )
        resumed_fable = FakeFable(harness.fable.brief)
        resumed_sol = FakeSol()
        resumed_sol.queue(_completed())
        recreated = Coordinator(
            store=reopened_store, repository=reopened_tracker, runner=RecordingRunner(),
            fable=resumed_fable, sol=resumed_sol,
            ids=DeterministicIds(task_number=1, run_number=40), repo_root=harness.repo,
            repo_context="Binding AGENTS instructions.",
            trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
        )
        reopened = reopened_store.get_task("task-1", 1)
        assert (
            reopened.continuation_generation, reopened.exchange_allowance,
            reopened.exchange_consumed, reopened.fable_session_id, reopened.sol_thread_id,
        ) == (1, 2, 1, fable_session_id, sol_thread_id)
        after_events = reopened_store.events_after("session-1", 0)
        assert after_events[:len(before_events)] == before_events
        assert [
            event for event in after_events
            if event.kind == "conversation" and event.payload["message_type"] == "answer"
        ] == [
            event for event in before_events
            if event.kind == "conversation" and event.payload["message_type"] == "answer"
        ]
        assert reopened_store._connection.execute(  # noqa: SLF001 - durable checkpoint
            "SELECT question_id, exchange_id FROM questions WHERE task_id = ? ORDER BY rowid",
            ("task-1",),
        ).fetchall() == before_question_ids

        persisted_question = reopened_store.question(question.question_id)
        assert persisted_question is not None
        await recreated.answer_directed_question(persisted_question)

        retried = reopened_store.get_task("task-1", 1)
        assert (retried.exchange_allowance, retried.exchange_consumed) == (2, 1)
        conversation_after = [
            event for event in reopened_store.events_after("session-1", 0)
            if event.kind == "conversation"
            and event.payload["message_type"] == "question"
        ]
        assert len(conversation_after) == 1
        assert conversation_after[0].payload["question_id"] == question.question_id
        assert len(resumed_fable.answer_sol_question_prompts) == 1
        assert resumed_sol.resume_threads == [sol_thread_id]
        assert harness.sol.resume_threads == []
        assert reopened_store._connection.execute(  # noqa: SLF001 - durable checkpoint
            "SELECT question_id, exchange_id FROM questions WHERE task_id = ? ORDER BY rowid",
            ("task-1",),
        ).fetchall() == before_question_ids
        reopened_tracker.close()
        reopened_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "tamper", [
        None, "malformed", "cross_project", "generation", "question", "preparation",
        "interrupt_rollback",
    ],
)
def test_directed_fable_answer_checkpoint_reopens_through_real_resume_once(
    harness: CoordinatorHarness, tamper: str | None,
) -> None:
    """A committed Fable answer resumes its exact Sol route without another Fable call."""
    async def scenario() -> None:
        harness.sol.queue(_directed_sol_question("Which approved invariant applies?"))
        original_handoff = harness.store.handoff_directed_fable_answer_same_scope

        def crash_before_handoff(record: object) -> TaskRecord:
            assert harness.coordinator._claimed_preparation_id is not None  # noqa: SLF001
            assert harness.store._connection.execute(  # noqa: SLF001 - checkpoint boundary
                "SELECT preparation_id FROM directed_fable_answer_checkpoints",
            ).fetchone()[0] == harness.coordinator._claimed_preparation_id
            raise RuntimeError("controlled checkpoint interruption")

        harness.store.handoff_directed_fable_answer_same_scope = crash_before_handoff  # type: ignore[method-assign]
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        if tamper == "interrupt_rollback":
            harness.store._connection.execute(  # noqa: SLF001 - atomic checkpoint boundary
                """
                CREATE TRIGGER fail_fable_checkpoint_interrupt
                BEFORE UPDATE OF status ON prepared_actions
                WHEN NEW.status = 'INTERRUPTED'
                BEGIN SELECT RAISE(ABORT, 'controlled checkpoint rollback'); END
                """
            )
        with pytest.raises(
            sqlite3.IntegrityError if tamper == "interrupt_rollback" else PreparedActionFailed,
        ):
            await harness.coordinator.approve_task("task-1", revision=1)
        interrupted = harness.store.get_task("task-1", 1)
        preparation = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id, session_id="session-1",
            task_id="task-1", revision=1,
        )
        assert preparation is not None
        checkpoint = harness.store.directed_fable_answer_checkpoint(preparation)
        raw_checkpoints = harness.store._connection.execute(  # noqa: SLF001 - checkpoint diagnosis
            "SELECT preparation_id, question_id, status FROM directed_fable_answer_checkpoints",
        ).fetchall()
        assert checkpoint is not None
        if tamper == "interrupt_rollback":
            assert preparation.status == "CLAIMED"
            assert interrupted.state is TaskState.SOL_RUNNING
            assert [tuple(row) for row in raw_checkpoints] == [
                (preparation.preparation_id, checkpoint.question_id, "PENDING"),
            ]
            return
        assert preparation.status == "INTERRUPTED", raw_checkpoints
        before_events = harness.store.events_after("session-1", 0)
        before_questions = harness.store._connection.execute(  # noqa: SLF001 - checkpoint identity
            "SELECT question_id, answer_text, exchange_id FROM questions WHERE task_id = ? ORDER BY rowid",
            ("task-1",),
        ).fetchall()
        assert len(harness.fable.answer_sol_question_prompts) == 1
        harness.store.set_setting("agent_bridge.active_session_id", "session-1")

        if tamper == "malformed":
            harness.store._connection.execute(  # noqa: SLF001 - startup audit boundary
                "UPDATE directed_fable_answer_checkpoints SET clarification_json = '{}'",
            )
        elif tamper == "cross_project":
            harness.store._connection.execute(  # noqa: SLF001 - startup audit boundary
                "UPDATE directed_fable_answer_checkpoints SET project_id = 'foreign-project'",
            )
        elif tamper == "generation":
            harness.store._connection.execute(  # noqa: SLF001 - startup audit boundary
                "UPDATE directed_fable_answer_checkpoints SET continuation_generation = 2",
            )
        elif tamper == "question":
            harness.store._connection.execute(  # noqa: SLF001 - startup audit boundary
                "UPDATE questions SET answered_by = 'sol' WHERE question_id = ?",
                (checkpoint.question_id,),
            )
        elif tamper == "preparation":
            harness.store._connection.execute(  # noqa: SLF001 - startup audit boundary
                "UPDATE directed_fable_answer_checkpoints SET preparation_id = 'missing-preparation'",
            )

        harness.tracker.close()
        harness.store.close()
        reopened_store = SQLiteStore(harness.database, clock=lambda: "2026-08-10T12:00:00Z")
        reopened_tracker = RepositoryTracker(
            harness.repo, harness.artifacts, git_executable=GIT_EXECUTABLE,
        )
        if tamper is not None:
            with pytest.raises(RuntimeError, match="legacy project ownership audit failed"):
                reopened_store.audit_legacy_project_ownership(str(harness.repo.resolve()))
            reopened_tracker.close()
            reopened_store.close()
            return
        reopened_store.audit_legacy_project_ownership(str(harness.repo.resolve()))
        resumed_fable = FakeFable(harness.fable.brief)
        resumed_sol = FakeSol()
        resumed_sol.queue(_completed())
        recreated = Coordinator(
            store=reopened_store, repository=reopened_tracker, runner=RecordingRunner(),
            fable=resumed_fable, sol=resumed_sol,
            ids=DeterministicIds(task_number=1, run_number=60), repo_root=harness.repo,
            repo_context="Binding AGENTS instructions.",
            trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
        )
        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=recreated.project_id,
            store=reopened_store,
            coordinator=recreated,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe,
                sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        stale = lease.acquire(
            project_id=recreated.project_id, session_id="session-1", task_id="task-1",
        )
        lease.release(stale)
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        resumed = await workflows.prepare_resume(
            project_id=recreated.project_id,
            session_id="session-1", task_id="task-1", revision=1,
        )
        assert resumed.token.generation == 2
        record = reopened_store.prepared_action(resumed.preparation_id)
        assert record is not None
        assert record.previous_preparation_id == preparation.preparation_id
        with pytest.raises(RuntimeError, match="no longer owns"):
            await workflows.run(PreparedWorkflow(resumed.preparation_id, stale, revision=1))
        await workflows.run(resumed)
        assert workflows.active_lease_snapshot() is None
        assert resumed_fable.answer_sol_question_prompts == []
        assert resumed_sol.resume_threads == [interrupted.sol_thread_id]
        after_events = reopened_store.events_after("session-1", 0)
        assert after_events[:len(before_events)] == before_events
        assert [
            event for event in after_events
            if event.kind == "conversation" and event.payload["message_type"] == "answer"
        ] == [
            event for event in before_events
            if event.kind == "conversation" and event.payload["message_type"] == "answer"
        ]
        assert reopened_store._connection.execute(  # noqa: SLF001 - checkpoint identity
            "SELECT question_id, answer_text, exchange_id FROM questions WHERE task_id = ? ORDER BY rowid",
            ("task-1",),
        ).fetchall() == before_questions
        assert reopened_store.directed_fable_answer_checkpoint(preparation) is None
        reopened_tracker.close()
        reopened_store.close()
        harness.store.handoff_directed_fable_answer_same_scope = original_handoff  # type: ignore[method-assign]

    asyncio.run(scenario())


def test_directed_fable_answer_handoff_consumes_before_sol_failure(
    harness: CoordinatorHarness,
) -> None:
    """A post-handoff Sol failure cannot resurrect the already-routed Fable answer."""
    async def scenario() -> None:
        harness.sol.queue(_directed_sol_question("Which approved invariant applies?"))
        original_resume = harness.coordinator._resume_sol

        async def crash_after_handoff(*_: object, **__: object) -> None:
            raise RuntimeError("controlled Sol handoff crash")

        harness.coordinator._resume_sol = crash_after_handoff  # type: ignore[method-assign]
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        with pytest.raises(PreparedActionFailed):
            await harness.coordinator.approve_task("task-1", revision=1)
        preparation = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id, session_id="session-1",
            task_id="task-1", revision=1,
        )
        assert preparation is not None
        assert harness.store.directed_fable_answer_checkpoint(preparation) is None
        assert harness.store.get_task("task-1", 1).pending is None
        assert len(harness.fable.answer_sol_question_prompts) == 1
        assert len([
            event for event in harness.store.events_after("session-1", 0)
            if event.kind == "clarification" and event.actor == "fable"
        ]) == 1
        harness.coordinator._resume_sol = original_resume  # type: ignore[method-assign]

    asyncio.run(scenario())


def test_directed_fable_answer_handoff_rolls_back_before_sol_runs(
    harness: CoordinatorHarness,
) -> None:
    """A same-scope handoff write failure leaves the pre-handoff checkpoint intact."""
    async def scenario() -> None:
        harness.sol.queue(_directed_sol_question("Which approved invariant applies?"))
        original_insert = harness.store._insert_event_in_transaction  # noqa: SLF001

        def fail_clarification(*args: object, **kwargs: object):
            if len(args) >= 4 and args[3] == "clarification":
                raise RuntimeError("controlled handoff transaction failure")
            return original_insert(*args, **kwargs)

        harness.store._insert_event_in_transaction = fail_clarification  # type: ignore[method-assign] # noqa: SLF001
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        with pytest.raises(PreparedActionFailed):
            await harness.coordinator.approve_task("task-1", revision=1)
        preparation = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id, session_id="session-1",
            task_id="task-1", revision=1,
        )
        assert preparation is not None
        checkpoint = harness.store.directed_fable_answer_checkpoint(preparation)
        task = harness.store.get_task("task-1", 1)
        assert checkpoint is not None
        assert task.state is TaskState.INTERRUPTED
        assert task.continuation_state is TaskState.SOL_RUNNING
        assert task.pending is not None
        assert harness.sol.resume_threads == []

    asyncio.run(scenario())


def test_directed_fable_scope_handoff_survives_post_handoff_crash(
    harness: CoordinatorHarness,
) -> None:
    """A crash after N+1 handoff leaves only the durable exact approval path."""
    async def scenario() -> None:
        revised = replace(
            harness.fable.brief, revision=2,
            allowed_paths=("bridge-output.txt", "bridge-extra.txt"),
        )
        harness.sol.queue(_directed_sol_question("May I add bridge-extra.txt?"))
        harness.fable.next_clarifications.append(
            _answer("Add the explicitly scoped file.", True, revised_brief=revised)
        )
        original_emit = harness.coordinator._emit_state

        def crash_after_scope_handoff(task: TaskRecord) -> None:
            if task.revision == 2:
                raise RuntimeError("controlled post-scope handoff crash")
            original_emit(task)

        harness.coordinator._emit_state = crash_after_scope_handoff  # type: ignore[method-assign]
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        with pytest.raises(PreparedActionFailed):
            await harness.coordinator.approve_task("task-1", revision=1)
        latest = harness.store.latest_task("task-1")
        assert latest is not None
        assert (latest.revision, latest.state) == (2, TaskState.AWAITING_SCOPE_APPROVAL)
        assert harness.store._connection.execute(  # noqa: SLF001 - durable handoff
            "SELECT COUNT(*) FROM directed_fable_answer_checkpoints WHERE status = 'PENDING'",
        ).fetchone()[0] == 0
        assert len(harness.fable.answer_sol_question_prompts) == 1
        harness.coordinator._emit_state = original_emit  # type: ignore[method-assign]
        await harness.coordinator.approve_task("task-1", revision=2)
        assert harness.sol.resume_threads == [THREAD_ID]

    asyncio.run(scenario())


def test_recovered_directed_fable_scope_checkpoint_hub_resume_consumes_once(
    harness: CoordinatorHarness,
) -> None:
    """A pre-handoff scope crash recreates exactly one N+1 through Hub Resume."""
    async def scenario() -> None:
        revised = replace(
            harness.fable.brief, revision=2,
            allowed_paths=("bridge-output.txt", "bridge-extra.txt"),
        )
        harness.sol.queue(_directed_sol_question("May I add bridge-extra.txt?"))
        harness.fable.next_clarifications.append(
            _answer("Add the explicitly scoped file.", True, revised_brief=revised)
        )
        original_route = harness.coordinator._route_directed_fable_answer

        async def crash_before_scope_handoff(*_: object, **__: object) -> None:
            raise RuntimeError("controlled pre-scope handoff crash")

        harness.coordinator._route_directed_fable_answer = crash_before_scope_handoff  # type: ignore[method-assign]
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        with pytest.raises(PreparedActionFailed):
            await harness.coordinator.approve_task("task-1", revision=1)
        prior = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id, session_id="session-1",
            task_id="task-1", revision=1,
        )
        assert prior is not None
        checkpoint = harness.store.directed_fable_answer_checkpoint(prior)
        assert checkpoint is not None
        parent = harness.store.question(checkpoint.question_id)
        assert parent is not None
        assert parent.answer_text is None and parent.answered_by is None
        paused = harness.store.get_task("task-1", 1)
        assert paused.state is TaskState.AWAITING_USER_INPUT
        assert paused.continuation_state is TaskState.SOL_RUNNING
        harness.tracker.close()
        harness.store.close()
        reopened_store = SQLiteStore(harness.database, clock=lambda: "2026-08-10T12:00:00Z")
        reopened_tracker = RepositoryTracker(
            harness.repo, harness.artifacts, git_executable=GIT_EXECUTABLE,
        )
        resumed_fable = FakeFable(harness.fable.brief)
        resumed_sol = FakeSol()
        recreated = Coordinator(
            store=reopened_store, repository=reopened_tracker, runner=RecordingRunner(),
            fable=resumed_fable, sol=resumed_sol,
            ids=DeterministicIds(task_number=1, run_number=80), repo_root=harness.repo,
            repo_context="Binding AGENTS instructions.",
            trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
        )
        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=recreated.project_id, store=reopened_store,
            coordinator=recreated,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe, sol_probe=sol_probe,
            ),
        )
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=ActiveAgentLease(),
            usage_credits_acknowledged=lambda: True,
        )
        resumed = await workflows.prepare_resume(
            project_id=recreated.project_id, session_id="session-1", task_id="task-1", revision=1,
        )
        await workflows.run(resumed)
        completed = reopened_store.prepared_action(resumed.preparation_id)
        assert completed is not None and completed.status == "COMPLETED"
        latest = reopened_store.latest_task("task-1")
        assert latest is not None and latest.revision == 2
        assert latest.state is TaskState.AWAITING_SCOPE_APPROVAL
        assert reopened_store.directed_fable_answer_checkpoint(prior) is None
        assert resumed_fable.answer_sol_question_prompts == []
        assert reopened_store.get_setting("agent_bridge.baseline.task-1.2") is not None
        events = reopened_store.events_after("session-1", 0)
        assert len([event for event in events if event.kind == "clarification" and event.actor == "fable"]) == 1
        assert len([event for event in events if event.kind == "task_brief" and event.actor == "fable"]) == 2
        reopened_tracker.close()
        reopened_store.close()
        harness.coordinator._route_directed_fable_answer = original_route  # type: ignore[method-assign]

    asyncio.run(scenario())


def test_directed_fable_scope_handoff_rolls_back_revision_setting_and_events(
    harness: CoordinatorHarness,
) -> None:
    """A scope handoff transaction failure leaves neither N+1 nor consumed state."""
    async def scenario() -> None:
        revised = replace(
            harness.fable.brief, revision=2,
            allowed_paths=("bridge-output.txt", "bridge-extra.txt"),
        )
        harness.sol.queue(_directed_sol_question("May I add bridge-extra.txt?"))
        harness.fable.next_clarifications.append(
            _answer("Add the explicitly scoped file.", True, revised_brief=revised)
        )
        original_insert = harness.store._insert_event_in_transaction  # noqa: SLF001

        def fail_scope_clarification(*args: object, **kwargs: object):
            if len(args) >= 4 and args[3] == "clarification":
                raise RuntimeError("controlled scope transaction failure")
            return original_insert(*args, **kwargs)

        harness.store._insert_event_in_transaction = fail_scope_clarification  # type: ignore[method-assign] # noqa: SLF001
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        with pytest.raises(PreparedActionFailed):
            await harness.coordinator.approve_task("task-1", revision=1)
        assert harness.store.latest_task("task-1").revision == 1  # type: ignore[union-attr]
        preparation = harness.store.latest_prepared_action_for_task(
            project_id=harness.coordinator.project_id, session_id="session-1",
            task_id="task-1", revision=1,
        )
        assert preparation is not None
        assert harness.store.directed_fable_answer_checkpoint(preparation) is not None
        assert harness.store.get_setting("agent_bridge.baseline.task-1.2") is None

    asyncio.run(scenario())


def test_directed_fable_scope_change_requires_exact_new_approval_before_sol_resumes(
    harness: CoordinatorHarness,
) -> None:
    """Fable keeps scope authority even while answering Sol's directed question."""
    async def scenario() -> None:
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", "bridge-extra.txt"),
        )
        harness.sol.queue(_directed_sol_question("May I add bridge-extra.txt?"))
        harness.fable.next_clarifications.append(
            _answer("Add the explicitly scoped file.", True, revised_brief=revised)
        )

        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        await harness.coordinator.approve_task("task-1", revision=1)

        revision_one = harness.store.get_task("task-1", 1)
        awaiting = harness.store.latest_task("task-1")
        assert awaiting is not None
        assert awaiting.revision == 2
        assert awaiting.state is TaskState.AWAITING_SCOPE_APPROVAL
        assert awaiting.approved_at is None
        assert awaiting.baseline_id == revision_one.baseline_id
        assert harness.sol.resume_threads == []
        assert harness.store._connection.execute(  # noqa: SLF001 - durable handoff
            "SELECT COUNT(*) FROM directed_fable_answer_checkpoints WHERE status = 'PENDING'",
        ).fetchone()[0] == 0
        assert len([
            event for event in harness.store.events_after("session-1", 0)
            if event.kind == "clarification" and event.actor == "fable"
        ]) == 1

        with pytest.raises(ValueError, match="revision"):
            await harness.coordinator.approve_task("task-1", revision=1)
        assert harness.sol.resume_threads == []

        await harness.coordinator.approve_task("task-1", revision=2)
        assert harness.sol.resume_threads == [THREAD_ID]

    asyncio.run(scenario())


def test_fable_directed_evidence_request_uses_exact_approved_sol_thread(
    harness: CoordinatorHarness,
) -> None:
    """Fable may ask Sol only inside the approved revision and exact thread."""
    async def scenario() -> None:
        harness.sol.queue(_completed())
        harness.sol.queue(_completed())
        harness.fable.next_verdicts.extend((
            _verdict(
                harness.fable.brief,
                directed_question=DirectedAgentQuestion(
                    addressed_to="sol",
                    text="Which approved seam is already exercised?",
                    reason="Fable needs evidence before final review.",
                ),
            ),
            _verdict(harness.fable.brief),
        ))

        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        await harness.coordinator.approve_task("task-1", revision=1)

        assert len(harness.sol.answer_fable_question_calls) == 1
        thread_id, brief, prompt = harness.sol.answer_fable_question_calls[0]
        assert thread_id == THREAD_ID
        assert brief.revision == 1
        assert prompt == "Which approved seam is already exercised?"
        assert harness.sol.resume_threads == []
        conversation = [
            event for event in harness.store.events_after("session-1", 0)
            if event.kind == "conversation"
        ]
        assert [event.payload["message_type"] for event in conversation] == [
            "question", "answer",
        ]
        assert conversation[0].payload["sender"] == "fable"
        assert conversation[0].payload["routed_to"] == "sol"
        assert conversation[1].payload["sender"] == "sol"
        assert conversation[1].payload["routed_to"] == "fable"
        assert harness.store.get_task("task-1", 1).state is TaskState.COMPLETED

    asyncio.run(scenario())


def test_fable_clarification_directed_evidence_request_resumes_its_exact_context(
    harness: CoordinatorHarness,
) -> None:
    """A contract-backed Fable evidence request must not be rejected as nested."""
    async def scenario() -> None:
        harness.sol.queue(_question("Which existing rule resolves this ambiguity?"))
        harness.sol.queue(_completed())
        harness.sol.queue(_completed())
        harness.fable.next_clarifications.extend((
            _answer(
                "I need one exact execution fact before answering.",
                False,
                directed_question=DirectedAgentQuestion(
                    addressed_to="sol",
                    text="Which approved rule is already exercised?",
                    reason="Fable needs exact execution evidence.",
                ),
            ),
            _answer("Use the confirmed approved rule.", False),
        ))
        harness.coordinator._bounded_compatibility_error = lambda error: error  # type: ignore[method-assign,return-value]

        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        await harness.coordinator.approve_task("task-1", revision=1)

        assert harness.sol.answer_fable_question_calls == [
            (
                THREAD_ID,
                harness.fable.brief,
                "Which approved rule is already exercised?",
            ),
        ]
        assert len(harness.fable.clarification_prompts) == 2
        assert "Sol evidence: Implemented and verified the approved change." in harness.fable.clarification_prompts[1]
        assert "User answer:" not in harness.fable.clarification_prompts[1]
        assert harness.sol.resume_threads == [THREAD_ID]
        conversation = [
            event for event in harness.store.events_after("session-1", 0)
            if event.kind == "conversation"
        ]
        assert [event.payload["message_type"] for event in conversation] == [
            "question", "answer",
        ]
        assert conversation[0].payload["sender"] == "fable"
        assert conversation[0].payload["routed_to"] == "sol"
        assert conversation[1].payload["sender"] == "sol"
        assert conversation[1].payload["routed_to"] == "fable"
        assert harness.store.get_task("task-1", 1).state is TaskState.COMPLETED

    asyncio.run(scenario())


def test_fable_answer_evidence_keeps_outer_sol_question_open_until_final_answer(
    harness: CoordinatorHarness,
) -> None:
    """A nested evidence reply must restore the outer Fable answer, not resume Sol."""
    async def scenario() -> None:
        harness.sol.queue(_directed_sol_question("Which approved constraint applies?"))
        harness.sol.queue(_completed(summary="Sol's persisted evidence only."))
        harness.sol.queue(_completed())
        harness.fable.next_clarifications.extend((
            _answer(
                "I need one execution fact before the final answer.",
                False,
                directed_question=DirectedAgentQuestion(
                    addressed_to="sol",
                    text="Which focused test proves the approved constraint?",
                    reason="Fable needs exact evidence before answering Sol.",
                ),
            ),
            _answer("Use the already-proven approved constraint.", False),
        ))

        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        await harness.coordinator.approve_task("task-1", revision=1)

        assert harness.sol.answer_fable_question_calls == [
            (
                THREAD_ID,
                harness.fable.brief,
                "Which focused test proves the approved constraint?",
            ),
        ]
        assert [entry[:2] for entry in harness.fable.answer_sol_question_prompts] == [
            ("task-1", "Which approved constraint applies?"),
            (
                "task-1",
                "Which approved constraint applies?\nSol evidence: Sol's persisted evidence only.",
            ),
        ]
        assert harness.sol.resume_threads == [THREAD_ID]
        conversation = [
            event.payload for event in harness.store.events_after("session-1", 0)
            if event.kind == "conversation"
        ]
        assert [event["message_type"] for event in conversation] == [
            "question", "question", "answer", "answer",
        ]
        assert conversation[1]["reply_to_question_id"] is None
        assert conversation[2]["reply_to_question_id"] == conversation[1]["question_id"]
        assert conversation[3]["reply_to_question_id"] == conversation[0]["question_id"]
        assert harness.store.get_task("task-1", 1).state is TaskState.COMPLETED

    asyncio.run(scenario())


@pytest.mark.parametrize("claim_before_recovery", (False, True))
def test_exhausted_fable_answer_evidence_pauses_then_retries_the_exact_outer_question(
    harness: CoordinatorHarness,
    claim_before_recovery: bool,
) -> None:
    """A +3 grant must retry the persisted Sol-question parent, not reroute it."""
    async def scenario() -> None:
        outer_text = "Which approved constraint applies?"
        harness.sol.queue(_directed_sol_question(outer_text))
        harness.sol.queue(_completed())
        harness.fable.next_clarifications.extend((
            _answer(
                "I need one execution fact before the final answer.",
                False,
                directed_question=DirectedAgentQuestion(
                    addressed_to="sol",
                    text="Which focused test proves the approved constraint?",
                    reason="Fable needs exact evidence before answering Sol.",
                ),
            ),
            _answer(
                "I need one execution fact before the final answer.",
                False,
                directed_question=DirectedAgentQuestion(
                    addressed_to="sol",
                    text="Which focused test proves the approved constraint?",
                    reason="Fable needs exact evidence before answering Sol.",
                ),
            ),
            _answer("Use the already-proven approved constraint.", False),
        ))

        def exhaust_before_nested_evidence() -> None:
            harness.store._connection.execute(
                "UPDATE tasks SET exchange_allowance = 0 WHERE task_id = ? AND revision = ?",
                ("task-1", 1),
            )

        harness.fable.on_answer_sol_question = exhaust_before_nested_evidence
        harness.coordinator._bounded_compatibility_error = lambda error: error  # type: ignore[method-assign,return-value]
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        await harness.coordinator.approve_task("task-1", revision=1)

        paused = harness.store.get_task("task-1", 1)
        assert (paused.state, paused.continuation_state, paused.exchange_allowance) == (
            TaskState.AWAITING_USER_INPUT, TaskState.FABLE_CLARIFYING, 0,
        )
        outer = harness.store.unanswered_question_for_task("task-1", 1)
        assert outer is not None and outer.text == outer_text
        harness.fable.on_answer_sol_question = None

        prepared = harness.coordinator._prepare_exchange_grant_action(
            session_id="session-1", task_id="task-1", revision=1,
            continuation_generation=paused.continuation_generation,
            request_id="retry-nested-answer-evidence",
            generation=17,
        )
        assert prepared.payload.outer_question_id == outer.question_id
        duplicate = harness.store.prepare_exchange_grant_action(
            project_id=harness.coordinator.project_id,
            session_id="session-1", task_id="task-1", revision=1,
            generation=17, payload=prepared.payload,
        )
        assert duplicate.preparation_id == prepared.preparation_id
        if claim_before_recovery:
            harness.store.claim_prepared_action(prepared.preparation_id, generation=17)
        assert harness.coordinator.recover_unfinished_prepared_actions().prepared_actions_recovered == 1
        resumed = harness.coordinator.prepare_resume(
            session_id="session-1", task_id="task-1", revision=1, generation=18,
        )
        assert resumed.preparation_id == prepared.preparation_id
        await harness.coordinator.run_prepared_action(resumed.preparation_id)

        assert [entry[:2] for entry in harness.fable.answer_sol_question_prompts] == [
            ("task-1", outer_text),
            ("task-1", outer_text),
            (
                "task-1",
                f"{outer_text}\nSol evidence: Implemented and verified the approved change.",
            ),
        ]
        assert len(harness.sol.answer_fable_question_calls) == 1
        assert harness.store._connection.execute(
            "SELECT COUNT(*) FROM exchange_grants WHERE request_id = ?",
            ("retry-nested-answer-evidence",),
        ).fetchone()[0] == 1
        assert harness.store.get_task("task-1", 1).state is TaskState.COMPLETED

    asyncio.run(scenario())


def test_legacy_user_answer_cannot_consume_a_durable_directed_question(
    harness: CoordinatorHarness,
) -> None:
    """Removing this guard would let an unbound legacy reply resume an exact question."""
    async def scenario() -> None:
        fable_started = asyncio.Event()
        release_fable = asyncio.Event()
        harness.sol.queue(_directed_sol_question("Which exact approved rule applies?"))
        harness.fable.hold_answer_sol_question = release_fable
        harness.fable.on_answer_sol_question = fable_started.set
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        approval = asyncio.create_task(
            harness.coordinator.approve_task("task-1", revision=1),
        )
        await fable_started.wait()
        task = harness.store.get_task("task-1", 1)
        before_events = harness.store.events_after("session-1", 0)

        with pytest.raises(ValueError, match="exact directed question"):
            await harness.coordinator.answer_user_question(
                "task-1", "An unbound legacy answer must not route.",
            )

        assert harness.store.get_task("task-1", 1) == task
        assert harness.store.events_after("session-1", 0) == before_events
        release_fable.set()
        await approval

    asyncio.run(scenario())


def test_clarification_grant_checkpoint_recovers_and_reuses_the_reserved_sol_question(
    harness: CoordinatorHarness,
) -> None:
    """A post-reservation crash must resume the exact child without another grant."""
    async def scenario() -> None:
        directed = DirectedAgentQuestion(
            addressed_to="sol",
            text="Which focused check proves the implementation?",
            reason="Fable needs one bounded execution fact.",
        )
        harness.sol.queue(_question("Which rule needs clarification?"))
        harness.fable.next_clarifications.append(
            _answer("I need one execution fact.", False, directed_question=directed)
        )
        original_clarify = harness.fable.clarify

        async def exhaust_before_fable_requests_evidence(**kwargs: object) -> AgentRunResult:
            harness.store._connection.execute(
                "UPDATE tasks SET exchange_allowance = 0 WHERE task_id = ? AND revision = ?",
                ("task-1", 1),
            )
            return await original_clarify(**kwargs)  # type: ignore[arg-type]

        harness.fable.clarify = exhaust_before_fable_requests_evidence  # type: ignore[method-assign]
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        await harness.coordinator.approve_task("task-1", revision=1)

        paused = harness.store.get_task("task-1", 1)
        assert (paused.state, paused.continuation_state, paused.exchange_allowance) == (
            TaskState.AWAITING_USER_INPUT, TaskState.FABLE_CLARIFYING, 0,
        )
        prepared = harness.coordinator._prepare_exchange_grant_action(
            session_id="session-1", task_id="task-1", revision=1,
            continuation_generation=paused.continuation_generation,
            request_id="clarification-checkpoint", generation=51,
        )
        claimed = harness.store.claim_prepared_action(
            prepared.preparation_id, generation=prepared.generation,
        )
        question_id, request_key = harness.coordinator._directed_question_identifiers(
            harness.store.get_task("task-1", 1), ConversationActor.FABLE, directed,
        )
        _, child = harness.store.reserve_fable_clarification_evidence_question(
            session_id="session-1", task_id="task-1", revision=1,
            expected_generation=claimed.payload.continuation_generation,
            question_id=question_id, request_key=request_key, text=directed.text,
            event=store_module.ConversationEnvelope(
                sender=ConversationActor.FABLE, addressed_to=ConversationTarget.SOL,
                routed_to=ConversationTarget.SOL,
                message_type=ConversationMessageType.QUESTION, text=directed.text,
                task_id="task-1", revision=1,
                continuation_generation=claimed.payload.continuation_generation,
                question_id=question_id,
            ),
        )
        checkpoint = harness.store.get_task("task-1", 1)
        assert (checkpoint.state, checkpoint.continuation_state) == (
            TaskState.AWAITING_USER_INPUT, TaskState.FABLE_CLARIFYING,
        )
        assert harness.store.question(child.question_id) == child
        harness.fable.clarify = original_clarify  # type: ignore[method-assign]

        harness.tracker.close()
        harness.store.close()
        reopened_store = SQLiteStore(harness.database, clock=lambda: "2026-08-10T12:00:00Z")
        reopened_tracker = RepositoryTracker(
            harness.repo, harness.artifacts, git_executable=GIT_EXECUTABLE,
        )
        resumed_sol = FakeSol()
        resumed_sol.queue(_completed(summary="The focused check passed."))
        resumed_sol.queue(_completed())
        recreated = Coordinator(
            store=reopened_store, repository=reopened_tracker, runner=RecordingRunner(),
            fable=harness.fable, sol=resumed_sol,
            ids=DeterministicIds(task_number=1, run_number=20), repo_root=harness.repo,
            repo_context="Binding AGENTS instructions.",
            trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
        )

        assert reopened_store.prepared_action(prepared.preparation_id).status == "RECOVERED"  # type: ignore[union-attr]
        assert recreated.recover_unfinished_prepared_actions().prepared_actions_recovered == 0
        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=recreated.project_id,
            store=reopened_store,
            coordinator=recreated,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe,
                sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        stale = lease.acquire(
            project_id=recreated.project_id, session_id="session-1", task_id="task-1",
        )
        lease.release(stale)
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        first_resume = await workflows.prepare_resume(
            project_id=recreated.project_id, session_id="session-1",
            task_id="task-1", revision=1,
        )
        assert (first_resume.preparation_id, first_resume.token.generation) == (
            prepared.preparation_id, 2,
        )
        with pytest.raises(RuntimeError, match="another workflow"):
            await workflows.prepare_resume(
                project_id=recreated.project_id, session_id="session-1",
                task_id="task-1", revision=1,
            )
        await workflows.run(first_resume)
        assert workflows.active_lease_snapshot() is None

        assert len(resumed_sol.answer_fable_question_calls) == 1
        assert resumed_sol.answer_fable_question_calls[0][2] == directed.text
        assert len(harness.fable.clarification_prompts) == 2
        assert "Sol evidence: The focused check passed." in harness.fable.clarification_prompts[1]
        assert "User answer:" not in harness.fable.clarification_prompts[1]
        assert reopened_store._connection.execute(
            "SELECT COUNT(*) FROM exchange_grants WHERE request_id = ?",
            ("clarification-checkpoint",),
        ).fetchone()[0] == 1
        conversation = [
            event.payload for event in reopened_store.events_after("session-1", 0)
            if event.kind == "conversation"
        ]
        assert [event["message_type"] for event in conversation].count("question") == 1
        assert [event["message_type"] for event in conversation].count("answer") == 1
        reopened_tracker.close()
        reopened_store.close()

    asyncio.run(scenario())


def test_question_grant_checkpoint_recovers_the_restored_outer_fable_pause(
    harness: CoordinatorHarness,
) -> None:
    """A crash after restoring Sol's parent question must not reroute or recharge it."""
    async def scenario() -> None:
        outer_text = "Which approved constraint applies?"
        directed = DirectedAgentQuestion(
            addressed_to="sol",
            text="Which focused test proves that constraint?",
            reason="Fable needs one exact execution fact.",
        )
        harness.sol.queue(_directed_sol_question(outer_text))
        harness.fable.next_clarifications.append(
            _answer("I need evidence first.", False, directed_question=directed)
        )
        harness.fable.on_answer_sol_question = lambda: harness.store._connection.execute(
            "UPDATE tasks SET exchange_allowance = 0 WHERE task_id = ? AND revision = ?",
            ("task-1", 1),
        )
        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        await harness.coordinator.approve_task("task-1", revision=1)

        paused = harness.store.get_task("task-1", 1)
        outer = harness.store.unanswered_question_for_task("task-1", 1)
        assert outer is not None
        assert (paused.state, paused.continuation_state, paused.exchange_allowance) == (
            TaskState.AWAITING_USER_INPUT, TaskState.FABLE_CLARIFYING, 0,
        )
        prepared = harness.coordinator._prepare_exchange_grant_action(
            session_id="session-1", task_id="task-1", revision=1,
            continuation_generation=paused.continuation_generation,
            request_id="question-checkpoint", generation=61,
        )
        claimed = harness.store.claim_prepared_action(
            prepared.preparation_id, generation=prepared.generation,
        )
        harness.store.restore_fable_answer_parent_for_retry(
            session_id="session-1", task_id="task-1", revision=1,
            expected_generation=claimed.payload.continuation_generation,
            outer_question_id=outer.question_id,
        )
        checkpoint = harness.store.get_task("task-1", 1)
        assert (checkpoint.state, checkpoint.continuation_state) == (
            TaskState.AWAITING_USER_INPUT, TaskState.SOL_RUNNING,
        )
        harness.fable.on_answer_sol_question = None

        harness.tracker.close()
        harness.store.close()
        reopened_store = SQLiteStore(harness.database, clock=lambda: "2026-08-10T12:00:00Z")
        reopened_tracker = RepositoryTracker(
            harness.repo, harness.artifacts, git_executable=GIT_EXECUTABLE,
        )
        resumed_sol = FakeSol()
        resumed_sol.queue(_completed(summary="The focused test passed."))
        resumed_sol.queue(_completed())
        harness.fable.next_clarifications.append(
            _answer("I need evidence first.", False, directed_question=directed)
        )
        harness.fable.next_clarifications.append(
            _answer("Use the verified constraint.", False)
        )
        recreated = Coordinator(
            store=reopened_store, repository=reopened_tracker, runner=RecordingRunner(),
            fable=harness.fable, sol=resumed_sol,
            ids=DeterministicIds(task_number=1, run_number=30), repo_root=harness.repo,
            repo_context="Binding AGENTS instructions.",
            trusted_shells={"bash": "/bin/bash", "sh": "/bin/sh"},
        )

        assert reopened_store.prepared_action(prepared.preparation_id).status == "RECOVERED"  # type: ignore[union-attr]
        assert recreated.recover_unfinished_prepared_actions().prepared_actions_recovered == 0
        async def fable_probe() -> tuple[bool, str]:
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        runtime = SimpleNamespace(
            project_id=recreated.project_id,
            store=reopened_store,
            coordinator=recreated,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe,
                sol_probe=sol_probe,
            ),
        )
        lease = ActiveAgentLease()
        stale = lease.acquire(
            project_id=recreated.project_id, session_id="session-1", task_id="task-1",
        )
        lease.release(stale)
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)), lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        resumed = await workflows.prepare_resume(
            project_id=recreated.project_id, session_id="session-1",
            task_id="task-1", revision=1,
        )
        assert (resumed.preparation_id, resumed.token.generation) == (
            prepared.preparation_id, 2,
        )
        assert reopened_store.prepared_action(prepared.preparation_id).generation == 2  # type: ignore[union-attr]
        with pytest.raises(RuntimeError, match="no longer owns"):
            await workflows.run(PreparedWorkflow(prepared.preparation_id, stale, revision=1))
        with pytest.raises(RuntimeError, match="another workflow"):
            await workflows.prepare_resume(
                project_id=recreated.project_id, session_id="session-1",
                task_id="task-1", revision=1,
            )
        await workflows.run(resumed)
        assert workflows.active_lease_snapshot() is None

        assert [entry[:2] for entry in harness.fable.answer_sol_question_prompts] == [
            ("task-1", outer_text),
            ("task-1", outer_text),
            ("task-1", f"{outer_text}\nSol evidence: The focused test passed."),
        ]
        assert len(resumed_sol.answer_fable_question_calls) == 1
        assert reopened_store._connection.execute(
            "SELECT COUNT(*) FROM exchange_grants WHERE request_id = ?",
            ("question-checkpoint",),
        ).fetchone()[0] == 1
        conversation = [
            event.payload for event in reopened_store.events_after("session-1", 0)
            if event.kind == "conversation"
        ]
        assert [event["message_type"] for event in conversation].count("question") == 2
        assert [event["message_type"] for event in conversation].count("answer") == 2
        reopened_tracker.close()
        reopened_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("parent_mode", ("clarification", "question"))
def test_fable_scope_change_wins_over_a_simultaneous_directed_question(
    harness: CoordinatorHarness,
    parent_mode: str,
) -> None:
    """A scope revision is N+1 authority; its ignored question cannot spend a hop."""
    async def scenario() -> None:
        directed = DirectedAgentQuestion(
            addressed_to="sol",
            text="Which exact fact should I use before widening scope?",
            reason="This must not run when Fable widens the revision.",
        )
        revised = replace(
            harness.fable.brief,
            revision=2,
            allowed_paths=("bridge-output.txt", "additional-output.txt"),
        )
        if parent_mode == "clarification":
            harness.sol.queue(_question("Which approved ambiguity needs a wider scope?"))
            harness.fable.next_clarifications.append(
                _answer(
                    "The revised scope is required.", True,
                    revised_brief=revised, directed_question=directed,
                )
            )
        else:
            harness.sol.queue(_directed_sol_question("Which approved constraint applies?"))
            harness.fable.next_clarifications.append(
                _answer(
                    "The revised scope is required.", True,
                    revised_brief=revised, directed_question=directed,
                )
            )

        await harness.coordinator.handle_user_request("session-1", "Build the bridge")
        await harness.coordinator.approve_task("task-1", revision=1)

        latest = harness.store.latest_task("task-1")
        assert latest is not None
        assert (latest.revision, latest.state) == (2, TaskState.AWAITING_SCOPE_APPROVAL)
        assert harness.sol.answer_fable_question_calls == []
        assert harness.store.get_task("task-1", 1).exchange_consumed == (
            0 if parent_mode == "clarification" else 1
        )
        questions = [
            event for event in harness.store.events_after("session-1", 0)
            if event.kind == "conversation" and event.payload["message_type"] == "question"
        ]
        assert len(questions) == (0 if parent_mode == "clarification" else 1)

    asyncio.run(scenario())


def _routing_task(
    *,
    state: TaskState,
    approved_at: str | None,
    task_id: str = "routing-task",
    revision: int = 2,
    continuation_generation: int = 7,
) -> TaskRecord:
    """Hand-built authenticated state for the side-effect-free selector."""
    return TaskRecord(
        task_id=task_id,
        revision=revision,
        session_id="session-1",
        state=state,
        brief=None,
        approved_at=approved_at,
        fable_session_id="fable-session-1",
        sol_thread_id=THREAD_ID,
        baseline_id="baseline-1" if approved_at is not None else None,
        correction_count=0,
        continuation_state=None,
        pending=None,
        continuation_generation=continuation_generation,
        exchange_allowance=3,
        exchange_consumed=0,
    )


def _bound_user_intent(
    *,
    addressed_to: ConversationTarget,
    task_id: str = "routing-task",
    revision: int = 2,
    continuation_generation: int = 7,
) -> UserConversationInput:
    return UserConversationInput(
        addressed_to=addressed_to,
        message_type=ConversationMessageType.STATEMENT,
        text="Continue with the recorded direction.",
        task_id=task_id,
        revision=revision,
        continuation_generation=continuation_generation,
    )


@pytest.mark.parametrize(
    "addressed_to",
    (ConversationTarget.FABLE, ConversationTarget.SOL, ConversationTarget.TEAM),
)
def test_routing_matrix_unbound_user_message_starts_a_new_fable_task(
    addressed_to: ConversationTarget,
) -> None:
    """A missing binding must not select an existing task or agent continuation."""
    intent = UserConversationInput(
        addressed_to=addressed_to,
        message_type=ConversationMessageType.STATEMENT,
        text="Start a separate task.",
    )

    decision = route_user_intent(None, intent)

    assert decision == RoutingDecision(
        addressed_to=addressed_to,
        routed_to=ConversationTarget.FABLE,
        mode=RoutingMode.NEW_FABLE_TASK,
        task_id=None,
        revision=None,
        continuation_generation=None,
    )


def test_routing_matrix_preapproval_sol_address_is_persisted_but_routes_to_fable() -> None:
    """Removing the approval guard would incorrectly send a pre-approval request to Sol."""
    task = _routing_task(
        state=TaskState.AWAITING_USER_APPROVAL,
        approved_at=None,
    )
    intent = _bound_user_intent(addressed_to=ConversationTarget.SOL)

    decision = route_user_intent(task, intent)

    assert decision == RoutingDecision(
        addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.FABLE,
        mode=RoutingMode.BOUND_CONTINUATION,
        task_id="routing-task",
        revision=2,
        continuation_generation=7,
    )


def test_routing_matrix_approved_nonterminal_sol_continuation_keeps_exact_binding() -> None:
    """A wrong route or stale binding would resume the wrong provider thread."""
    task = _routing_task(
        state=TaskState.SOL_RUNNING,
        approved_at="2026-08-11T12:00:00Z",
    )
    intent = _bound_user_intent(addressed_to=ConversationTarget.SOL)

    decision = route_user_intent(task, intent)

    assert decision == RoutingDecision(
        addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.SOL,
        mode=RoutingMode.BOUND_CONTINUATION,
        task_id="routing-task",
        revision=2,
        continuation_generation=7,
    )


@pytest.mark.parametrize("state", (TaskState.COMPLETED, TaskState.FAILED))
def test_routing_matrix_terminal_bound_message_creates_a_new_fable_task(
    state: TaskState,
) -> None:
    """A terminal task must never be resumed merely because its old ID was supplied."""
    task = _routing_task(state=state, approved_at="2026-08-11T12:00:00Z")
    intent = _bound_user_intent(addressed_to=ConversationTarget.SOL)

    decision = route_user_intent(task, intent)

    assert decision == RoutingDecision(
        addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.FABLE,
        mode=RoutingMode.NEW_FABLE_TASK,
        task_id=None,
        revision=None,
        continuation_generation=None,
    )


@pytest.mark.parametrize(
    "authenticated_task,intent",
    (
        (
            None,
            _bound_user_intent(addressed_to=ConversationTarget.SOL),
        ),
        (
            _routing_task(
                state=TaskState.SOL_RUNNING,
                approved_at="2026-08-11T12:00:00Z",
            ),
            _bound_user_intent(
                addressed_to=ConversationTarget.SOL,
                task_id="another-task",
            ),
        ),
        (
            _routing_task(
                state=TaskState.SOL_RUNNING,
                approved_at="2026-08-11T12:00:00Z",
            ),
            _bound_user_intent(
                addressed_to=ConversationTarget.SOL,
                revision=3,
            ),
        ),
        (
            _routing_task(
                state=TaskState.SOL_RUNNING,
                approved_at="2026-08-11T12:00:00Z",
            ),
            _bound_user_intent(
                addressed_to=ConversationTarget.SOL,
                continuation_generation=8,
            ),
        ),
    ),
)
def test_routing_matrix_unknown_or_stale_binding_never_falls_back(
    authenticated_task: TaskRecord | None,
    intent: UserConversationInput,
) -> None:
    """Changing any authenticated identity coordinate must reject rather than reroute."""
    with pytest.raises(RoutingError, match="conversation routing is unavailable"):
        route_user_intent(authenticated_task, intent)
