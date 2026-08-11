from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent_bridge.adapters.claude_cli import (
    ClaudeCLI,
    ClaudeRunError,
    SubscriptionAuthError,
)
from agent_bridge.adapters.codex_cli import CodexCLI, CodexRunError
from agent_bridge.app import BootstrapStatus, InMemoryEventBroadcaster, create_app
from agent_bridge.coordinator import Coordinator
from agent_bridge.process import LineCallback, ProcessResult, ProcessRunner
from agent_bridge.repository import RepositoryTracker
from agent_bridge.state_machine import TaskState
from agent_bridge.store import SQLiteStore


THREAD_ID = "0199a213-81c0-7800-8aa1-bbab2a035a53"
SESSION_ID = "session-e2e"
REQUIRED_TEST = "tests/fake_acceptance_test.py"
TRUSTED_PYTHON = str(Path(sys.executable).resolve(strict=True))
TEST_COMMAND = f"{TRUSTED_PYTHON} -m pytest -q {REQUIRED_TEST}"


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


class RecordingProcessRunner(ProcessRunner):
    """Observe the executable boundary before the real runner launches it."""

    def __init__(self, *, stop_grace_seconds: float) -> None:
        super().__init__(stop_grace_seconds=stop_grace_seconds)
        self.launches: list[dict[str, object]] = []

    async def run(
        self,
        *,
        run_id: str,
        argv: Sequence[str],
        cwd: str | Path,
        env: Mapping[str, str],
        stdin: bytes | None,
        on_line: LineCallback,
    ) -> ProcessResult:
        executable = Path(argv[0])
        if not executable.is_absolute():
            raise AssertionError("the E2E fixture forbids PATH-resolved processes")
        self.launches.append({
            "run_id": run_id,
            "executable": str(executable.resolve(strict=True)),
            "sentinel": env.get("AGENT_BRIDGE_TEST_FAKE") == "1",
        })
        return await super().run(
            run_id=run_id,
            argv=argv,
            cwd=cwd,
            env=env,
            stdin=stdin,
            on_line=on_line,
        )


def _brief(
    *,
    task_id: str = "task-1",
    title: str = "Add the bounded bridge fixture",
    allowed_path: str = "bridge_work",
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "revision": 1,
        "title": title,
        "objective": "Create one file inside the approved bridge work directory.",
        "context": ["This is a controlled fake-agent integration run."],
        "constraints": ["Use only the approved temporary repository."],
        "allowed_paths": [allowed_path],
        "out_of_scope": ["data", "outside-scope.txt"],
        "acceptance_criteria": ["The approved file is present and structurally tested."],
        "required_tests": [REQUIRED_TEST],
        "risks": ["Unexpected repository paths must fail closed."],
        "open_questions": [],
        "confidence": 0.95,
        "confidence_rationale": "The fixture owns the complete temporary repository.",
    }


def _completed(
    path: str = "bridge_work/output.txt",
    content: str = "implemented by fake Sol\n",
) -> dict[str, object]:
    return {
        "status": "completed",
        "summary": "The exact approved fake change is complete.",
        "changed_files": [path],
        "commands_run": [{
            "command": TEST_COMMAND,
            "exit_code": 0,
            "result": "passed",
        }],
        "known_failures": [],
        "remaining_risks": [],
        "architecture_docs": "No durable architecture change.",
        "question": None,
        "_mutations": [{"path": path, "content": content}],
    }


def _question() -> dict[str, object]:
    return {
        "status": "question",
        "summary": "The fake implementation needs one bounded answer.",
        "changed_files": [],
        "commands_run": [],
        "known_failures": [],
        "remaining_risks": ["The approved filename needs confirmation."],
        "architecture_docs": "No durable architecture change.",
        "question": {
            "ambiguity": "Should the approved file retain its planned name?",
            "why_it_matters": "The answer controls the exact approved path.",
            "options": ["Keep output.txt", "Request a new revision"],
            "recommendation": "Keep output.txt",
            "can_continue_safely": False,
        },
    }


def _clarification(*, escalate: bool = False) -> dict[str, object]:
    if escalate:
        return {
            "status": "escalate_to_user",
            "answer": None,
            "reasoning": "The approved brief does not resolve the user's preference.",
            "confidence": 0.25,
            "scope_changed": False,
            "revised_brief": None,
            "question_for_user": "Should Sol keep the approved output.txt filename?",
        }
    return {
        "status": "answered",
        "answer": "Keep the implementation within the exact approved scope.",
        "reasoning": "The approved brief already resolves the ambiguity.",
        "confidence": 0.95,
        "scope_changed": False,
        "revised_brief": None,
        "question_for_user": None,
    }


def _review(status: str = "approved") -> dict[str, object]:
    return {
        "status": status,
        "summary": "The fake evidence was reviewed against the exact brief.",
        "criteria": [{
            "criterion": "The approved file is present and structurally tested.",
            "evidence": ["Repository delta and exact command digest agree."],
            "satisfied": status == "approved",
        }],
        "test_assessment": "The required test has matching zero-exit structural evidence.",
        "scope_violations": [],
        "remaining_risks": [],
        "corrections": (
            ["Replace the first fake implementation with the corrected content."]
            if status == "corrections_required"
            else []
        ),
        "question_for_user": None,
    }


def _default_scenario() -> dict[str, object]:
    return {
        "plans": [_brief()],
        "outcomes": [_completed()],
        "clarifications": [_clarification()],
        "reviews": [_review()],
        "claude_modes": {},
        "codex_modes": [],
    }


def _write_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return path.resolve(strict=True)


def _fake_git_source(log_path: Path) -> str:
    return f'''#!/usr/bin/python3
import json
import os
from pathlib import Path
import sys

with Path({str(log_path)!r}).open("a", encoding="utf-8") as handle:
    for record in (
        {{
            "kind": "git_wrapper",
            "executable": str(Path(sys.argv[0]).resolve()),
            "argv": sys.argv[1:],
        }},
        {{
            "kind": "git_delegate",
            "executable": "/usr/bin/git",
            "argv": sys.argv[1:],
        }},
    ):
        handle.write(json.dumps(record, sort_keys=True) + "\\n")
    handle.flush()
    os.fsync(handle.fileno())
os.execve("/usr/bin/git", ["/usr/bin/git", *sys.argv[1:]], dict(os.environ))
'''


def _fake_shell_source(kind: str) -> str:
    return f'''#!/usr/bin/python3
import sys
print("refusing unexpected fake {kind} execution", file=sys.stderr)
raise SystemExit(97)
'''


def _run_git(executable: Path, repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        [str(executable), *arguments],
        cwd=repo,
        env={
            "AGENT_BRIDGE_TEST_FAKE": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "LANG": "C",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


@dataclass
class FakeBridge:
    root: Path
    repo: Path
    database: Path
    artifacts: Path
    schemas: Path
    captures: Path
    scenario_path: Path
    invocation_log: Path
    fake_claude: Path
    fake_codex: Path
    fake_git: Path
    fake_bash: Path
    fake_sh: Path
    store: SQLiteStore
    tracker: RepositoryTracker
    runner: RecordingProcessRunner
    fable: ClaudeCLI
    sol: CodexCLI
    ids: DeterministicIds
    coordinator: Coordinator
    broadcaster: InMemoryEventBroadcaster
    session_id: str = SESSION_ID

    @property
    def agent_executables(self) -> frozenset[str]:
        return frozenset((str(self.fake_claude), str(self.fake_codex)))

    @property
    def invocations(self) -> tuple[dict[str, Any], ...]:
        if not self.invocation_log.exists():
            return ()
        return tuple(
            json.loads(line)
            for line in self.invocation_log.read_text(encoding="utf-8").splitlines()
            if line
        )

    @property
    def live_call_count(self) -> int:
        return sum(
            launch.get("executable") not in self.agent_executables
            for launch in self.runner.launches
        )

    @property
    def infrastructure_invocations(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            invocation
            for invocation in self.invocations
            if invocation.get("kind") in {"git_wrapper", "git_delegate"}
        )

    def configure(self, **updates: object) -> None:
        scenario = _default_scenario()
        scenario.update(updates)
        self.scenario_path.write_text(
            json.dumps(scenario, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        for state_file in self.captures.glob("scenario-state-*.json"):
            state_file.unlink()

    def replace_scenario(self, scenario: Mapping[str, object]) -> None:
        self.scenario_path.write_text(
            json.dumps(scenario, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        for state_file in self.captures.glob("scenario-state-*.json"):
            state_file.unlink()

    def events(self):
        return self.store.events_after(self.session_id, 0)

    def latest(self, task_id: str):
        task = self.store.latest_task(task_id)
        assert task is not None
        return task

    def build_app(self):
        return create_app(
            coordinator=self.coordinator,
            store=self.store,
            static_dir=Path(__file__).parents[2] / "src" / "agent_bridge" / "static",
            session_key="e2e-browser-key",
            csrf_token="e2e-csrf-token",
            broadcaster=self.broadcaster,
            bootstrap_status=lambda: BootstrapStatus(
                session_id=self.session_id,
                fable_ready=True,
                fable_status="subscription_ready",
                sol_status="ready",
                repository=str(self.repo),
                branch="main",
            ),
        )

    def restart(self) -> tuple:
        self.tracker.close()
        self.store.close()
        self.store = SQLiteStore(
            self.database,
            clock=lambda: "2026-08-10T15:00:00Z",
            check_same_thread=False,
        )
        recovered = self.store.recover_active_tasks()
        self.tracker = RepositoryTracker(
            self.repo,
            self.artifacts,
            git_executable=self.fake_git,
        )
        self.ids = DeterministicIds(task_number=1, run_number=20)
        self.coordinator = Coordinator(
            store=self.store,
            repository=self.tracker,
            runner=self.runner,
            fable=self.fable,
            sol=self.sol,
            ids=self.ids,
            repo_root=self.repo,
            repo_context="Controlled temporary AGENTS context.",
            trusted_shells={"bash": self.fake_bash, "sh": self.fake_sh},
        )
        return recovered

    def close(self) -> None:
        self.tracker.close()
        self.store.close()


@pytest.fixture
def fake_bridge(
    tmp_path: Path,
    fake_claude: Path,
    fake_codex: Path,
) -> FakeBridge:
    state = tmp_path / "external-state"
    artifacts = tmp_path / "external-artifacts"
    schemas = tmp_path / "external-schemas"
    captures = tmp_path / "external-captures"
    binaries = tmp_path / "fake-bin"
    repo = tmp_path / "repo"
    for directory in (state, captures, binaries, repo):
        directory.mkdir()
    invocation_log = captures / "resolved-executables.jsonl"
    fake_git = _write_executable(
        binaries / "git", _fake_git_source(invocation_log)
    )
    fake_bash = _write_executable(binaries / "bash", _fake_shell_source("bash"))
    fake_sh = _write_executable(binaries / "sh", _fake_shell_source("sh"))

    _run_git(fake_git, repo, "init", "-b", "main")
    _run_git(fake_git, repo, "config", "user.email", "bridge@example.invalid")
    _run_git(fake_git, repo, "config", "user.name", "Bridge E2E")
    (repo / "fixture.txt").write_text("fixture\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / REQUIRED_TEST).write_text(
        "def test_fake_acceptance():\n    assert True\n", encoding="utf-8"
    )
    (repo / "data").mkdir()
    (repo / "data" / "frozen.txt").write_text("frozen\n", encoding="utf-8")
    _run_git(fake_git, repo, "add", "fixture.txt", REQUIRED_TEST, "data/frozen.txt")
    _run_git(fake_git, repo, "commit", "-m", "fixture")

    scenario_path = captures / "scenario.json"
    scenario_path.write_text(
        json.dumps(_default_scenario(), separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    database = state / "bridge.sqlite3"
    store = SQLiteStore(
        database,
        clock=lambda: "2026-08-10T14:00:00Z",
        check_same_thread=False,
    )
    store.create_session(SESSION_ID, str(repo))
    runner = RecordingProcessRunner(stop_grace_seconds=0.05)
    environment = {
        "AGENT_BRIDGE_TEST_FAKE": "1",
        "AGENT_BRIDGE_INVOCATION_LOG": str(invocation_log),
        "FAKE_AGENT_CAPTURE_DIR": str(captures),
        "FAKE_AGENT_SCENARIO": str(scenario_path),
        "FAKE_BRIDGE_REPO_ROOT": str(repo),
        "FAKE_BRIDGE_TEST_COMMAND": TEST_COMMAND,
        "FAKE_CLAUDE_TASK_ID": "task-1",
        "HOME": str(state),
        "PATH": "/path-resolution-is-forbidden",
        "LANG": "C",
        "LC_ALL": "C",
    }
    fable = ClaudeCLI(fake_claude, runner, env=environment, cwd=repo)
    sol = CodexCLI(
        fake_codex,
        runner,
        repo_root=repo,
        schema_dir=schemas,
        env=environment,
    )
    tracker = RepositoryTracker(repo, artifacts, git_executable=fake_git)
    ids = DeterministicIds()
    broadcaster = InMemoryEventBroadcaster()
    coordinator = Coordinator(
        store=store,
        repository=tracker,
        runner=runner,
        fable=fable,
        sol=sol,
        ids=ids,
        repo_root=repo,
        repo_context="Controlled temporary AGENTS context.",
        trusted_shells={"bash": fake_bash, "sh": fake_sh},
    )
    bridge = FakeBridge(
        root=tmp_path,
        repo=repo,
        database=database,
        artifacts=artifacts,
        schemas=schemas,
        captures=captures,
        scenario_path=scenario_path,
        invocation_log=invocation_log,
        fake_claude=fake_claude.resolve(strict=True),
        fake_codex=fake_codex.resolve(strict=True),
        fake_git=fake_git,
        fake_bash=fake_bash,
        fake_sh=fake_sh,
        store=store,
        tracker=tracker,
        runner=runner,
        fable=fable,
        sol=sol,
        ids=ids,
        coordinator=coordinator,
        broadcaster=broadcaster,
    )
    try:
        yield bridge
    finally:
        bridge.close()


def test_plan_approve_execute_review_complete(fake_bridge: FakeBridge) -> None:
    async def scenario() -> None:
        task_id = await fake_bridge.coordinator.handle_user_request(
            fake_bridge.session_id,
            "Add one small file under the exact approved path.",
        )
        awaiting = fake_bridge.latest(task_id)
        assert awaiting.state is TaskState.AWAITING_USER_APPROVAL
        assert awaiting.revision == 1

        with pytest.raises(ValueError, match="revision"):
            await fake_bridge.coordinator.approve_task(task_id, revision=2)
        assert not (fake_bridge.repo / "bridge_work" / "output.txt").exists()

        await fake_bridge.coordinator.approve_task(task_id, revision=1)
        completed = fake_bridge.latest(task_id)
        assert completed.state is TaskState.COMPLETED
        assert completed.approved_at == "2026-08-10T14:00:00Z"
        assert (fake_bridge.repo / "bridge_work" / "output.txt").read_text() == (
            "implemented by fake Sol\n"
        )

        events = fake_bridge.events()
        assert [event.kind for event in events].count("task_brief") == 1
        assert [event.kind for event in events].count("outcome") == 1
        assert [event.kind for event in events].count("review") == 1
        assert events[-1].kind == "review"
        assert any(
            event.kind == "task_state"
            and event.payload.get("state") == TaskState.COMPLETED.value
            for event in events
        )
        assert fake_bridge.live_call_count == 0
        invoked_agents = {
            str(launch["executable"]) for launch in fake_bridge.runner.launches
        }
        assert invoked_agents == {
            str(fake_bridge.fake_claude),
            str(fake_bridge.fake_codex),
        }
        assert all(
            launch["sentinel"] is True for launch in fake_bridge.runner.launches
        )
        infrastructure = fake_bridge.infrastructure_invocations
        assert {entry["kind"] for entry in infrastructure} == {
            "git_wrapper",
            "git_delegate",
        }
        assert {entry["executable"] for entry in infrastructure} == {
            str(fake_bridge.fake_git),
            "/usr/bin/git",
        }

    asyncio.run(scenario())


def test_question_user_answer_and_exact_session_thread_continuity(
    fake_bridge: FakeBridge,
) -> None:
    fake_bridge.configure(
        outcomes=[_question(), _completed()],
        clarifications=[_clarification(escalate=True)],
        reviews=[_review()],
    )

    async def scenario() -> None:
        task_id = await fake_bridge.coordinator.handle_user_request(
            fake_bridge.session_id, "Implement the approved file."
        )
        await fake_bridge.coordinator.approve_task(task_id, revision=1)
        paused = fake_bridge.latest(task_id)
        assert paused.state is TaskState.AWAITING_USER_INPUT
        assert paused.continuation_state is TaskState.SOL_RUNNING

        await fake_bridge.coordinator.answer_user_question(
            task_id, "Keep output.txt exactly as approved."
        )
        completed = fake_bridge.latest(task_id)
        assert completed.state is TaskState.COMPLETED
        assert completed.fable_session_id == "fable-session-1"
        assert completed.sol_thread_id == THREAD_ID

        claude_model_calls = [
            call for call in fake_bridge.invocations
            if call.get("kind") == "claude"
            and call.get("argv") != ["auth", "status", "--json"]
        ]
        assert len(claude_model_calls) == 3
        assert "--resume" not in claude_model_calls[0]["argv"]
        assert all(
            call["argv"][call["argv"].index("--resume") + 1] == "fable-session-1"
            for call in claude_model_calls[1:]
        )
        codex_calls = [
            call for call in fake_bridge.invocations if call.get("kind") == "codex"
        ]
        assert len(codex_calls) == 2
        assert codex_calls[1]["argv"][:2] == ["exec", "resume"]
        assert THREAD_ID in codex_calls[1]["argv"]
        assert fake_bridge.live_call_count == 0

    asyncio.run(scenario())


def test_correction_loop_reuses_exact_agent_identities(fake_bridge: FakeBridge) -> None:
    fake_bridge.configure(
        outcomes=[
            _completed(content="first implementation\n"),
            _completed(content="corrected implementation\n"),
        ],
        reviews=[_review("corrections_required"), _review("approved")],
    )

    async def scenario() -> None:
        task_id = await fake_bridge.coordinator.handle_user_request(
            fake_bridge.session_id, "Implement and review the approved file."
        )
        await fake_bridge.coordinator.approve_task(task_id, revision=1)

        task = fake_bridge.latest(task_id)
        assert task.state is TaskState.COMPLETED
        assert task.correction_count == 1
        assert task.fable_session_id == "fable-session-1"
        assert task.sol_thread_id == THREAD_ID
        assert (fake_bridge.repo / "bridge_work" / "output.txt").read_text() == (
            "corrected implementation\n"
        )
        reviews = [event for event in fake_bridge.events() if event.kind == "review"]
        assert [event.payload["status"] for event in reviews] == [
            "corrections_required",
            "approved",
        ]
        codex_calls = [
            call for call in fake_bridge.invocations if call.get("kind") == "codex"
        ]
        assert THREAD_ID in codex_calls[1]["argv"]
        assert fake_bridge.live_call_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("contract", "variant"),
    (
        ("plan", "missing"),
        ("plan", "empty"),
        ("plan", "non_mapping"),
        ("outcome", "missing"),
        ("outcome", "empty"),
        ("outcome", "non_mapping"),
        ("review", "missing"),
        ("review", "empty"),
        ("review", "non_mapping"),
    ),
)
def test_strict_scenario_never_manufactures_a_contract_or_completion(
    fake_bridge: FakeBridge,
    contract: str,
    variant: str,
) -> None:
    scenario = _default_scenario()
    scenario.pop("brief", None)
    scenario["plans"] = [_brief()]
    key = {"plan": "plans", "outcome": "outcomes", "review": "reviews"}[contract]
    if variant == "missing":
        scenario.pop(key, None)
    else:
        scenario[key] = [] if variant == "empty" else [[]]
    fake_bridge.replace_scenario(scenario)

    async def run() -> None:
        if contract == "plan":
            with pytest.raises(ClaudeRunError):
                await fake_bridge.coordinator.handle_user_request(
                    fake_bridge.session_id, "Reject a missing fake plan."
                )
            task_id = "task-1"
        else:
            task_id = await fake_bridge.coordinator.handle_user_request(
                fake_bridge.session_id, f"Reject a missing fake {contract}."
            )
            expected_error = CodexRunError if contract == "outcome" else ClaudeRunError
            with pytest.raises(expected_error):
                await fake_bridge.coordinator.approve_task(task_id, revision=1)
        task = fake_bridge.latest(task_id)
        assert task.state is TaskState.FAILED
        assert task.state is not TaskState.COMPLETED
        assert fake_bridge.live_call_count == 0

    asyncio.run(run())


def test_exhausted_review_sequence_fails_instead_of_approving(
    fake_bridge: FakeBridge,
) -> None:
    scenario = _default_scenario()
    scenario.pop("brief", None)
    scenario.update({
        "plans": [_brief()],
        "outcomes": [
            _completed(content="first strict implementation\n"),
            _completed(content="second strict implementation\n"),
        ],
        "reviews": [_review("corrections_required")],
    })
    fake_bridge.replace_scenario(scenario)

    async def run() -> None:
        task_id = await fake_bridge.coordinator.handle_user_request(
            fake_bridge.session_id, "Exhaust the strict review sequence."
        )
        with pytest.raises(ClaudeRunError):
            await fake_bridge.coordinator.approve_task(task_id, revision=1)
        task = fake_bridge.latest(task_id)
        assert task.state is TaskState.FAILED
        assert task.correction_count == 1
        assert task.state is not TaskState.COMPLETED
        assert fake_bridge.live_call_count == 0

    asyncio.run(run())


def test_concurrent_workflows_dequeue_unique_plan_slots_without_lost_history(
    fake_bridge: FakeBridge,
) -> None:
    scenario = _default_scenario()
    scenario.pop("brief", None)
    scenario["plans"] = [
        _brief(task_id="$TASK_ID", title="Concurrent slot alpha"),
        _brief(task_id="$TASK_ID", title="Concurrent slot beta"),
    ]
    fake_bridge.replace_scenario(scenario)

    async def run() -> None:
        task_ids = await asyncio.gather(
            fake_bridge.coordinator.handle_user_request(
                fake_bridge.session_id, "Concurrent workflow alpha."
            ),
            fake_bridge.coordinator.handle_user_request(
                fake_bridge.session_id, "Concurrent workflow beta."
            ),
        )
        assert task_ids == ["task-1", "task-2"]
        planned = [fake_bridge.latest(task_id) for task_id in task_ids]
        assert all(task.state is TaskState.AWAITING_USER_APPROVAL for task in planned)
        assert {task.brief.title for task in planned if task.brief is not None} == {
            "Concurrent slot alpha",
            "Concurrent slot beta",
        }
        scenario_state = json.loads(
            (fake_bridge.captures / "scenario-state-claude.json").read_text(
                encoding="utf-8"
            )
        )
        assert scenario_state["plan"] == 2
        history = json.loads(
            (fake_bridge.captures / "captured-env-history.json").read_text(
                encoding="utf-8"
            )
        )
        assert len(history) == 4
        child_invocations = [
            invocation
            for invocation in fake_bridge.invocations
            if invocation.get("kind") == "claude"
        ]
        assert len(child_invocations) == 4
        assert list(fake_bridge.captures.glob(".*.tmp")) == []
        assert len(fake_bridge.runner.launches) == 4
        assert all(
            launch["executable"] == str(fake_bridge.fake_claude)
            and launch["sentinel"] is True
            for launch in fake_bridge.runner.launches
        )
        assert fake_bridge.live_call_count == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    ("mutation_path", "expected_protected"),
    (("outside-scope.txt", False), ("data/frozen.txt", True)),
)
def test_unapproved_or_protected_repository_delta_cannot_complete(
    fake_bridge: FakeBridge,
    mutation_path: str,
    expected_protected: bool,
) -> None:
    fake_bridge.configure(outcomes=[_completed(path=mutation_path)])

    async def scenario() -> None:
        task_id = await fake_bridge.coordinator.handle_user_request(
            fake_bridge.session_id, "Attempt one controlled invalid mutation."
        )
        await fake_bridge.coordinator.approve_task(task_id, revision=1)

        task = fake_bridge.latest(task_id)
        assert task.state is TaskState.AWAITING_USER_INPUT
        assert task.state is not TaskState.COMPLETED
        review = [event for event in fake_bridge.events() if event.kind == "review"][-1]
        assert review.payload["status"] == "approved"
        prompt_capture = json.loads(
            (fake_bridge.captures / "captured-argv.json").read_text(encoding="utf-8")
        )[-1]
        assert mutation_path in prompt_capture
        if expected_protected:
            assert '"protected_changed_paths":["data/frozen.txt"]' in prompt_capture
        else:
            assert '"unexpected_paths":["outside-scope.txt"]' in prompt_capture
        assert fake_bridge.live_call_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("agent", "mode", "error_type"),
    (
        ("claude", "malformed_json", ClaudeRunError),
        ("claude", "nonzero", ClaudeRunError),
        ("codex", "malformed_json", CodexRunError),
        ("codex", "nonzero", CodexRunError),
    ),
)
def test_malformed_or_nonzero_agent_fails_closed(
    fake_bridge: FakeBridge,
    agent: str,
    mode: str,
    error_type: type[Exception],
) -> None:
    updates: dict[str, object]
    if agent == "claude":
        updates = {"claude_modes": {"plan": [mode]}}
    else:
        updates = {"codex_modes": [mode]}
    fake_bridge.configure(**updates)

    async def scenario() -> None:
        if agent == "claude":
            with pytest.raises(error_type):
                await fake_bridge.coordinator.handle_user_request(
                    fake_bridge.session_id, "Exercise a controlled agent failure."
                )
            task_id = "task-1"
        else:
            task_id = await fake_bridge.coordinator.handle_user_request(
                fake_bridge.session_id, "Exercise a controlled agent failure."
            )
            with pytest.raises(error_type):
                await fake_bridge.coordinator.approve_task(task_id, revision=1)
        assert fake_bridge.latest(task_id).state is TaskState.FAILED
        assert fake_bridge.latest(task_id).state is not TaskState.COMPLETED
        assert fake_bridge.live_call_count == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("agent", ("claude", "codex"))
def test_fake_agent_model_invocation_refuses_missing_sentinel(
    fake_bridge: FakeBridge, agent: str,
) -> None:
    if agent == "claude":
        fake_bridge.fable._env.pop("AGENT_BRIDGE_TEST_FAKE")
    else:
        fake_bridge.sol._env.pop("AGENT_BRIDGE_TEST_FAKE")

    async def scenario() -> None:
        if agent == "claude":
            with pytest.raises(SubscriptionAuthError, match="subscription"):
                await fake_bridge.coordinator.handle_user_request(
                    fake_bridge.session_id, "This must be refused by the fake."
                )
        else:
            await fake_bridge.coordinator.handle_user_request(
                fake_bridge.session_id, "This must be refused by fake Sol."
            )
            with pytest.raises(CodexRunError, match="non-zero"):
                await fake_bridge.coordinator.approve_task("task-1", revision=1)
        assert fake_bridge.latest("task-1").state is TaskState.FAILED
        refused = [
            launch
            for launch in fake_bridge.runner.launches
            if launch["executable"] == (
                str(fake_bridge.fake_claude)
                if agent == "claude"
                else str(fake_bridge.fake_codex)
            )
            and launch["sentinel"] is False
        ]
        assert len(refused) == 1
        assert fake_bridge.live_call_count == 0

    asyncio.run(scenario())


def test_stop_interrupts_only_the_active_fake_child(fake_bridge: FakeBridge) -> None:
    fake_bridge.configure(codex_modes=["slow_after_thread"])

    async def wait_for_marker(path: Path) -> None:
        for _ in range(250):
            if path.exists():
                return
            await asyncio.sleep(0.02)
        raise AssertionError("fake Codex did not reach its controlled stop marker")

    async def scenario() -> None:
        task_id = await fake_bridge.coordinator.handle_user_request(
            fake_bridge.session_id, "Start then stop the exact fake child."
        )
        approval = asyncio.create_task(
            fake_bridge.coordinator.approve_task(task_id, revision=1)
        )
        await wait_for_marker(fake_bridge.captures / "fake-codex-partials-ready.json")
        active = fake_bridge.store.active_run_for_task(task_id, 1)
        assert active is not None
        await fake_bridge.coordinator.stop_task(task_id)
        await approval

        task = fake_bridge.latest(task_id)
        assert task.state is TaskState.INTERRUPTED
        assert task.continuation_state is TaskState.SOL_RUNNING
        assert not fake_bridge.runner.is_running(active.run_id)
        assert fake_bridge.store.agent_run(active.run_id).status == "interrupted"
        assert not (fake_bridge.repo / "bridge_work" / "output.txt").exists()
        interrupted_baseline = fake_bridge.coordinator._load_baseline(task)
        delta = fake_bridge.tracker.compare(interrupted_baseline)
        assert delta.changed_paths == ()
        assert delta.unexpected_paths == ()
        assert delta.protected_changed_paths == ()
        assert fake_bridge.live_call_count == 0

    asyncio.run(scenario())


def test_restart_recovers_active_task_then_resumes_from_persisted_authority(
    fake_bridge: FakeBridge,
) -> None:
    async def scenario() -> None:
        task_id = await fake_bridge.coordinator.handle_user_request(
            fake_bridge.session_id, "Persist an approved task before restart."
        )
        task = fake_bridge.latest(task_id)
        assert task.brief is not None
        baseline = fake_bridge.tracker.capture(task.brief)
        approved = fake_bridge.store.approve_task_with_setting(
            task_id,
            1,
            brief=task.brief,
            baseline_id=baseline.baseline_id,
            expected=TaskState.AWAITING_USER_APPROVAL,
            setting=(
                fake_bridge.coordinator._baseline_key(task_id, 1),
                fake_bridge.coordinator._baseline_setting_value(task_id, 1, baseline),
            ),
        )
        active = fake_bridge.store.transition_task(
            task_id,
            1,
            expected=approved.state,
            target=TaskState.SOL_RUNNING,
        )
        fake_bridge.store.start_agent_run("stale-run", task_id, 1, "sol")
        fake_bridge.store.set_agent_run_process(
            "stale-run", pid=999_999, process_group_id=999_999
        )

        recovered = fake_bridge.restart()
        assert [task.task_id for task in recovered] == [task_id]
        interrupted = fake_bridge.latest(task_id)
        assert interrupted.state is TaskState.INTERRUPTED
        assert interrupted.continuation_state is active.state
        stale = fake_bridge.store.agent_run("stale-run")
        assert stale.status == "interrupted"
        assert stale.pid == 999_999

        await fake_bridge.coordinator.resume_task(task_id)
        completed = fake_bridge.latest(task_id)
        assert completed.state is TaskState.COMPLETED
        assert completed.sol_thread_id == THREAD_ID
        assert fake_bridge.live_call_count == 0

    asyncio.run(scenario())


def test_authenticated_websocket_replays_persisted_events_from_cursor(
    fake_bridge: FakeBridge,
) -> None:
    async def produce() -> None:
        task_id = await fake_bridge.coordinator.handle_user_request(
            fake_bridge.session_id, "Produce replayable fake workflow events."
        )
        await fake_bridge.coordinator.approve_task(task_id, revision=1)

    asyncio.run(produce())
    events = fake_bridge.events()
    cursor = events[2].sequence
    expected = [event.to_dict() for event in events if event.sequence > cursor]

    with TestClient(fake_bridge.build_app()) as client:
        assert client.get("/api/bootstrap").status_code == 403
        assert client.get("/?key=wrong", follow_redirects=False).status_code == 403
        authenticated = client.get(
            "/?key=e2e-browser-key", follow_redirects=False
        )
        assert authenticated.status_code == 303
        bootstrap = client.get("/api/bootstrap")
        assert bootstrap.status_code == 200
        assert bootstrap.json()["session_id"] == fake_bridge.session_id

        with client.websocket_connect(
            f"/ws?session_id={fake_bridge.session_id}&after={cursor}"
        ) as socket:
            replayed = [socket.receive_json() for _ in expected]

    assert replayed == expected
    assert all(event["sequence"] > cursor for event in replayed)
    assert fake_bridge.live_call_count == 0


def test_web_auth_failure_cannot_create_or_complete_a_task(
    fake_bridge: FakeBridge,
) -> None:
    with TestClient(fake_bridge.build_app()) as client:
        response = client.post(
            f"/api/sessions/{fake_bridge.session_id}/messages",
            headers={"X-CSRF-Token": "e2e-csrf-token"},
            json={"text": "This unauthenticated request must not run."},
        )
        assert response.status_code == 403

    assert fake_bridge.store.latest_task("task-1") is None
    assert fake_bridge.live_call_count == 0
