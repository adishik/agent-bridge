from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import hashlib
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from agent_bridge.adapters.base import AgentRunResult
from agent_bridge.adapters.claude_cli import ClaudeRunError
from agent_bridge.adapters.codex_cli import CodexRunError
from agent_bridge.contracts import (
    FableClarification,
    ReviewVerdict,
    SolOutcome,
    TaskBrief,
)
from agent_bridge.coordinator import (
    Coordinator,
    PreparedActionFailed,
    ResumeDriftBlocked,
)
from agent_bridge.process import ProcessRunner
from agent_bridge.repository import RepositoryTracker
from agent_bridge.state_machine import TaskState
from agent_bridge.store import (
    NewRequestPayload,
    PreparedActionOutcome,
    ResumePayload,
    ScopeApprovalContext,
    SQLiteStore,
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
) -> SolOutcome:
    return SolOutcome.from_dict({
        "status": "completed",
        "summary": "Implemented and verified the approved change.",
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


def _answer(
    text: str,
    scope_changed: bool,
    *,
    revised_brief: TaskBrief | None = None,
) -> FableClarification:
    return FableClarification.from_dict({
        "status": "answered",
        "answer": text,
        "reasoning": "The repository rules resolve the ambiguity.",
        "confidence": 0.95,
        "scope_changed": scope_changed,
        "revised_brief": None if revised_brief is None else revised_brief.to_dict(),
        "question_for_user": None,
    })


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
) -> ReviewVerdict:
    return ReviewVerdict.from_dict({
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
    })


@dataclass
class FakeFable:
    brief: TaskBrief
    clarification_prompts: list[str] = field(default_factory=list)
    review_prompts: list[str] = field(default_factory=list)
    plan_calls: list[tuple[str, str, str]] = field(default_factory=list)
    resume_plan_sessions: list[str] = field(default_factory=list)
    next_clarifications: deque[FableClarification] = field(default_factory=deque)
    next_verdicts: deque[ReviewVerdict] = field(default_factory=deque)
    hold_plan: asyncio.Event | None = None
    hold_clarification: asyncio.Event | None = None
    hold_review: asyncio.Event | None = None
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
        return _result(run_id, self.brief.to_dict(), session_id=session_id)

    async def clarify(
        self, *, run_id: str, session_id: str, prompt: str,
    ) -> AgentRunResult:
        self.clarification_prompts.append(prompt)
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

    async def review(
        self, *, run_id: str, session_id: str, prompt: str,
    ) -> AgentRunResult:
        self.review_prompts.append(prompt)
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
    next_outcomes: deque[tuple[SolOutcome, tuple[Mapping[str, object], ...]]] = field(
        default_factory=deque
    )
    hold_start: asyncio.Event | None = None
    on_start: Callable[[], None] | None = None
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
        self.resume_threads.append(thread_id)
        self.resume_prompts.append(prompt)
        outcome, events = self._next()
        return _result(run_id, outcome.to_dict(), session_id=thread_id, events=events)


class RecordingRunner(ProcessRunner):
    def __init__(self) -> None:
        super().__init__(stop_grace_seconds=0)
        self.stops: list[str] = []
        self.release_on_stop: asyncio.Event | None = None
        self.on_stop: Callable[[str], None] | None = None
        self.stop_error: BaseException | None = None

    async def stop(self, run_id: str) -> None:
        self.stops.append(run_id)
        if self.on_stop is not None:
            self.on_stop(run_id)
        if self.release_on_stop is not None:
            self.release_on_stop.set()
        if self.stop_error is not None:
            raise self.stop_error


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
            _answer("Add the explicitly scoped file.", True, revised_brief=revised)
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
