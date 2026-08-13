from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import httpx
import io
import json
import os
from pathlib import Path
import stat
import sqlite3
import subprocess
import sys
import time
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agent_bridge.adapters.claude_cli import (
    ClaudeCLI,
    ClaudeRunError,
    SubscriptionAuthError,
)
from agent_bridge.adapters.codex_cli import CodexCLI, CodexRunError
from agent_bridge.app import (
    BootstrapStatus,
    InMemoryEventBroadcaster,
    create_app,
    create_hub_app,
)
from agent_bridge.coordinator import Coordinator
from agent_bridge.contracts import TaskBrief
from agent_bridge.hub import (
    ActiveAgentLease,
    HubWorkflowOrchestrator,
    OwnedProjectRuntime,
    ProjectRegistry,
    RuntimeReadiness,
    RuntimeStatus,
)
from agent_bridge.hub_store import HubStore
from agent_bridge.process import LineCallback, ProcessResult, ProcessRunner
from agent_bridge.projects import ProjectSpec, build_project_specs
from agent_bridge.repository import RepositoryTracker
from agent_bridge.state_machine import TaskState
from agent_bridge.store import SQLiteStore
import agent_bridge.__main__ as launcher


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
    *,
    command: str = TEST_COMMAND,
    summary: str = "The exact approved fake change is complete.",
) -> dict[str, object]:
    return {
        "status": "completed",
        "summary": summary,
        "changed_files": [path],
        "commands_run": [{
            "command": command,
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


def _review(
    status: str = "approved",
    *,
    summary: str = "The fake evidence was reviewed against the exact brief.",
    evidence: str = "Repository delta and exact command digest agree.",
) -> dict[str, object]:
    return {
        "status": status,
        "summary": summary,
        "criteria": [{
            "criterion": "The approved file is present and structurally tested.",
            "evidence": [evidence],
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
    environment: dict[str, str]
    fable_session_id: str
    sol_thread_id: str
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


def _make_fake_bridge(
    root: Path,
    fake_claude: Path,
    fake_codex: Path,
    *,
    repo_name: str = "repo",
    fable_session_id: str = "fable-session-1",
    sol_thread_id: str = THREAD_ID,
) -> FakeBridge:
    root.mkdir(parents=True, exist_ok=True)
    state = root / "external-state"
    artifacts = root / "external-artifacts"
    schemas = root / "external-schemas"
    captures = root / "external-captures"
    binaries = root / "fake-bin"
    repo = root / repo_name
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
        "FAKE_CLAUDE_SESSION_ID": fable_session_id,
        "FAKE_CODEX_THREAD_ID": sol_thread_id,
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
    return FakeBridge(
        root=root,
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
        environment=environment,
        fable_session_id=fable_session_id,
        sol_thread_id=sol_thread_id,
        store=store,
        tracker=tracker,
        runner=runner,
        fable=fable,
        sol=sol,
        ids=ids,
        coordinator=coordinator,
        broadcaster=broadcaster,
    )


def _reopen_fake_bridge(previous: FakeBridge) -> tuple[FakeBridge, tuple]:
    """Rebuild every runtime-owned object from persisted fake project state."""
    environment = dict(previous.environment)
    store = SQLiteStore(
        previous.database,
        clock=lambda: "2026-08-10T15:00:00Z",
        check_same_thread=False,
    )
    recovered = store.recover_active_tasks()
    runner = RecordingProcessRunner(stop_grace_seconds=0.05)
    fable = ClaudeCLI(
        previous.fake_claude, runner, env=environment, cwd=previous.repo,
    )
    sol = CodexCLI(
        previous.fake_codex,
        runner,
        repo_root=previous.repo,
        schema_dir=previous.schemas,
        env=environment,
    )
    tracker = RepositoryTracker(
        previous.repo, previous.artifacts, git_executable=previous.fake_git,
    )
    ids = DeterministicIds(task_number=1, run_number=20)
    broadcaster = InMemoryEventBroadcaster()
    coordinator = Coordinator(
        store=store,
        repository=tracker,
        runner=runner,
        fable=fable,
        sol=sol,
        ids=ids,
        repo_root=previous.repo,
        repo_context="Controlled temporary AGENTS context.",
        trusted_shells={"bash": previous.fake_bash, "sh": previous.fake_sh},
    )
    return FakeBridge(
        root=previous.root,
        repo=previous.repo,
        database=previous.database,
        artifacts=previous.artifacts,
        schemas=previous.schemas,
        captures=previous.captures,
        scenario_path=previous.scenario_path,
        invocation_log=previous.invocation_log,
        fake_claude=previous.fake_claude,
        fake_codex=previous.fake_codex,
        fake_git=previous.fake_git,
        fake_bash=previous.fake_bash,
        fake_sh=previous.fake_sh,
        environment=environment,
        fable_session_id=previous.fable_session_id,
        sol_thread_id=previous.sol_thread_id,
        store=store,
        tracker=tracker,
        runner=runner,
        fable=fable,
        sol=sol,
        ids=ids,
        coordinator=coordinator,
        broadcaster=broadcaster,
        session_id=previous.session_id,
    ), recovered


@pytest.fixture
def fake_bridge(
    tmp_path: Path,
    fake_claude: Path,
    fake_codex: Path,
) -> FakeBridge:
    bridge = _make_fake_bridge(tmp_path, fake_claude, fake_codex)
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


@dataclass
class FakeHub:
    """Two complete fake-agent runtimes behind the production hub boundary."""

    bridges: dict[str, FakeBridge]
    specs: dict[str, ProjectSpec]
    hub_store: HubStore
    registry: ProjectRegistry
    workflows: HubWorkflowOrchestrator
    app: Any
    state_root: Path
    hub_path: Path
    immutable_roots: Mapping[str, Path]
    git_executable: Path

    @classmethod
    def create(
        cls,
        *,
        root: Path,
        fake_claude: Path,
        fake_codex: Path,
    ) -> "FakeHub":
        first = _make_fake_bridge(
            root / "first-parent",
            fake_claude,
            fake_codex,
            repo_name="same-name",
            fable_session_id="fable-alpha-session",
            sol_thread_id="0199a213-81c0-7800-8aa1-bbab2a035a51",
        )
        second = _make_fake_bridge(
            root / "second-parent",
            fake_claude,
            fake_codex,
            repo_name="same-name",
            fable_session_id="fable-beta-session",
            sol_thread_id="0199a213-81c0-7800-8aa1-bbab2a035a52",
        )
        bridges = {"alpha": first, "beta": second}
        state_root = root / "digest-state"
        immutable_roots = {label: bridge.repo for label, bridge in bridges.items()}
        spec_by_label = cls._build_specs(
            immutable_roots, state_root=state_root, git_executable=first.fake_git,
        )
        hub_path = state_root / "hub" / "hub.sqlite3"
        hub_store = cls._open_hub_store(hub_path)
        registry = cls._registry(bridges, spec_by_label)
        workflows = HubWorkflowOrchestrator(
            registry=registry,
            lease=ActiveAgentLease(),
            usage_credits_acknowledged=hub_store.usage_credits_acknowledged,
        )
        return cls(
            bridges=bridges,
            specs=spec_by_label,
            hub_store=hub_store,
            registry=registry,
            workflows=workflows,
            app=cls._app(registry, hub_store, workflows),
            state_root=state_root,
            hub_path=hub_path,
            immutable_roots=immutable_roots,
            git_executable=first.fake_git,
        )

    @staticmethod
    def _open_hub_store(path: Path) -> HubStore:
        path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_connect = sqlite3.connect
        with patch(
            "agent_bridge.hub_store.sqlite3.connect",
            side_effect=lambda *args, **kwargs: sqlite_connect(
                *args, check_same_thread=False, **kwargs
            ),
        ):
            return HubStore(
                path,
                clock=lambda: datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
            )

    @staticmethod
    def _build_specs(
        roots: Mapping[str, Path], *, state_root: Path, git_executable: Path,
    ) -> dict[str, ProjectSpec]:
        return {
            spec.label: spec
            for spec in build_project_specs(
                sorted(roots.items()),
                state_root=state_root,
                git_executable=git_executable,
            )
        }

    @staticmethod
    def _runtime(bridge: FakeBridge, spec: ProjectSpec) -> OwnedProjectRuntime:
        async def fable_probe() -> tuple[bool, str]:
            await bridge.fable.preflight()
            return True, "subscription_ready"

        async def sol_probe() -> str:
            return "ready"

        return OwnedProjectRuntime(
            spec=spec,
            store=bridge.store,
            tracker=bridge.tracker,
            runner=bridge.runner,
            fable=bridge.fable,
            sol=bridge.sol,
            coordinator=bridge.coordinator,
            broadcaster=bridge.broadcaster,
            readiness=RuntimeReadiness(
                initial=RuntimeStatus(True, "subscription_ready", "ready"),
                fable_probe=fable_probe,
                sol_probe=sol_probe,
                timeout_seconds=0.5,
            ),
            lock=launcher.acquire_instance_lock(spec.state_dir / "agent-bridge.lock"),
        )

    @classmethod
    def _registry(
        cls,
        bridges: Mapping[str, FakeBridge],
        specs: Mapping[str, ProjectSpec],
    ) -> ProjectRegistry:
        return ProjectRegistry(tuple(
            cls._runtime(bridges[label], specs[label])
            for label in sorted(bridges)
        ))

    @staticmethod
    def _app(
        registry: ProjectRegistry,
        hub_store: HubStore,
        workflows: HubWorkflowOrchestrator,
    ) -> Any:
        return create_hub_app(
            registry=registry,
            hub_store=hub_store,
            workflows=workflows,
            static_dir=Path(__file__).parents[2] / "src" / "agent_bridge" / "static",
            session_key="two-project-e2e-key",
            csrf_token="two-project-e2e-csrf",
        )

    def runtime(self, label: str) -> OwnedProjectRuntime:
        return self.registry.runtime(self.specs[label].project_id)  # type: ignore[return-value]

    def restart(self, *, labels: tuple[str, ...] | None = None) -> dict[str, tuple]:
        """Close every owned object, then reopen only the immutable allowlist."""
        selected = tuple(sorted(self.bridges if labels is None else labels))
        if not selected or any(label not in self.immutable_roots for label in selected):
            raise AssertionError("restart must use a non-empty immutable project allowlist")
        former_bridges = self.bridges
        self.registry.close()
        self.hub_store.close()
        reopened: dict[str, FakeBridge] = {}
        recovered: dict[str, tuple] = {}
        for label in selected:
            bridge, recovered_tasks = _reopen_fake_bridge(former_bridges[label])
            reopened[label] = bridge
            recovered[label] = recovered_tasks
        roots = {label: self.immutable_roots[label] for label in selected}
        self.bridges = reopened
        self.specs = self._build_specs(
            roots, state_root=self.state_root, git_executable=self.git_executable,
        )
        self.hub_store = self._open_hub_store(self.hub_path)
        self.registry = self._registry(self.bridges, self.specs)
        self.workflows = HubWorkflowOrchestrator(
            registry=self.registry,
            lease=ActiveAgentLease(),
            usage_credits_acknowledged=self.hub_store.usage_credits_acknowledged,
        )
        self.app = self._app(self.registry, self.hub_store, self.workflows)
        return recovered

    def close(self) -> None:
        self.registry.close()
        self.hub_store.close()


def _wait_for(predicate: Any, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before the E2E deadline")


def _authenticate_hub(client: TestClient) -> dict[str, str]:
    response = client.get("/?key=two-project-e2e-key", follow_redirects=False)
    assert response.status_code == 303
    headers = {"X-CSRF-Token": "two-project-e2e-csrf"}
    response = client.post(
        "/api/settings/usage-credits-acknowledgement",
        json={"acknowledged": True},
        headers=headers,
    )
    assert response.status_code == 202
    return headers


def _project_path(hub: FakeHub, label: str) -> str:
    return f"/api/projects/{hub.specs[label].project_id}"


def _event_documents(bridge: FakeBridge, session_id: str) -> list[dict[str, object]]:
    return [event.to_dict() for event in bridge.store.events_after(session_id, 0)]


@dataclass(frozen=True)
class ProjectProvenance:
    label: str
    output_path: str
    output_bytes: str
    command: str
    outcome_summary: str
    review_summary: str
    review_evidence: str
    fable_session_id: str
    sol_thread_id: str


ALPHA_PROVENANCE = ProjectProvenance(
    label="alpha",
    output_path="bridge_work/alpha-output.txt",
    output_bytes="alpha provenance bytes\n",
    command=f"{TRUSTED_PYTHON} -m pytest -q --color=never {REQUIRED_TEST}",
    outcome_summary="Alpha completed its distinct bounded change.",
    review_summary="Alpha review accepted its distinct evidence.",
    review_evidence="Alpha delta and Alpha command digest agree.",
    fable_session_id="fable-alpha-session",
    sol_thread_id="0199a213-81c0-7800-8aa1-bbab2a035a51",
)
BETA_PROVENANCE = ProjectProvenance(
    label="beta",
    output_path="bridge_work/beta-output.txt",
    output_bytes="beta provenance bytes\n",
    command=f"{TRUSTED_PYTHON} -m pytest -q --tb=short {REQUIRED_TEST}",
    outcome_summary="Beta completed its distinct bounded change.",
    review_summary="Beta review accepted its distinct evidence.",
    review_evidence="Beta delta and Beta command digest agree.",
    fable_session_id="fable-beta-session",
    sol_thread_id="0199a213-81c0-7800-8aa1-bbab2a035a52",
)


def _project_scenario(
    provenance: ProjectProvenance,
    *,
    plan_count: int,
    completed_workflow_count: int,
    slow_first_plan: bool = False,
    plan_paths: tuple[str, ...] | None = None,
    workflow_paths: tuple[str, ...] | None = None,
) -> dict[str, object]:
    resolved_plan_paths = plan_paths or (provenance.output_path,) * plan_count
    resolved_workflow_paths = workflow_paths or (
        provenance.output_path,
    ) * completed_workflow_count
    assert len(resolved_plan_paths) == plan_count
    assert len(resolved_workflow_paths) == completed_workflow_count
    return {
        "plans": [
            _brief(task_id="$TASK_ID", allowed_path=path)
            for path in resolved_plan_paths
        ],
        "outcomes": [
            result
            for index, path in enumerate(resolved_workflow_paths)
            for result in (
                _question(),
                _completed(
                    path=path,
                    content=(
                        provenance.output_bytes
                        if path == provenance.output_path
                        else f"{provenance.label} continuation {index}\n"
                    ),
                    command=provenance.command,
                    summary=provenance.outcome_summary,
                ),
            )
        ],
        "clarifications": [
            _clarification() for _ in range(completed_workflow_count)
        ],
        "reviews": [
            _review(
                summary=provenance.review_summary,
                evidence=provenance.review_evidence,
            )
            for _ in range(completed_workflow_count)
        ],
        "claude_modes": (
            {"plan": ["slow_after_init"]} if slow_first_plan else {}
        ),
        "codex_modes": [],
    }


def _response_signature(response: Any) -> tuple[int, bytes, tuple[tuple[bytes, bytes], ...]]:
    return (
        response.status_code,
        response.content,
        tuple(sorted(response.headers.raw)),
    )


def test_response_signature_detects_unlisted_and_duplicate_header_mutations() -> None:
    def response_with_raw_headers(*headers: tuple[bytes, bytes]) -> httpx.Response:
        return httpx.Response(
            status_code=404,
            content=b"project unavailable",
            headers=list(headers),
        )

    missing = response_with_raw_headers(
        (b"set-cookie", b"bridge=a; HttpOnly"),
        (b"set-cookie", b"project=missing; SameSite=strict"),
    )
    foreign_project = response_with_raw_headers(
        (b"set-cookie", b"bridge=a; HttpOnly"),
        (b"set-cookie", b"project=missing; SameSite=strict"),
        (b"x-project-exists", b"true"),
    )
    foreign_cookie = response_with_raw_headers(
        (b"set-cookie", b"bridge=a; HttpOnly"),
        (b"set-cookie", b"project=foreign; SameSite=strict"),
    )

    assert _response_signature(foreign_project) != _response_signature(missing)
    assert _response_signature(foreign_cookie) != _response_signature(missing)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _claude_contract_kind(call: Mapping[str, object]) -> str:
    argv = call["argv"]
    assert isinstance(argv, list)
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    properties = schema["properties"]
    if "task_id" in properties:
        return "plan"
    if "criteria" in properties:
        return "review"
    return "clarification"


def _assert_provider_continuity(
    bridge: FakeBridge,
    provenance: ProjectProvenance,
) -> None:
    claude_calls = [
        call
        for call in bridge.invocations
        if call.get("kind") == "claude"
        and call.get("argv") != ["auth", "status", "--json"]
    ]
    assert claude_calls
    plans = [call for call in claude_calls if _claude_contract_kind(call) == "plan"]
    reviews = [call for call in claude_calls if _claude_contract_kind(call) == "review"]
    clarifications = [
        call for call in claude_calls
        if _claude_contract_kind(call) == "clarification"
    ]
    assert plans and reviews and clarifications
    assert "--resume" not in plans[0]["argv"]
    for call in [*reviews, *clarifications, *plans[1:]]:
        argv = call["argv"]
        assert isinstance(argv, list)
        assert "--resume" in argv
        assert argv[argv.index("--resume") + 1] == provenance.fable_session_id

    codex_calls = [
        call for call in bridge.invocations if call.get("kind") == "codex"
    ]
    assert codex_calls
    initial = [call for call in codex_calls if call["argv"][:2] == ["exec", "--json"]]
    resumed = [call for call in codex_calls if call["argv"][:2] == ["exec", "resume"]]
    assert initial and resumed
    for call in resumed:
        argv = call["argv"]
        assert isinstance(argv, list)
        assert provenance.sol_thread_id in argv


def _assert_project_provenance(
    bridge: FakeBridge,
    provenance: ProjectProvenance,
    *,
    session_id: str,
    task_id: str,
) -> None:
    task = bridge.store.get_task(task_id, 1)
    assert task.state is TaskState.COMPLETED
    assert task.session_id == session_id
    assert task.brief is not None
    assert task.brief.allowed_paths == (provenance.output_path,)
    assert task.fable_session_id == provenance.fable_session_id
    assert task.sol_thread_id == provenance.sol_thread_id
    assert task.baseline_id is not None
    baseline = bridge.coordinator._load_baseline(task)
    delta = bridge.tracker.compare(baseline)
    assert delta.changed_paths == (provenance.output_path,)
    assert delta.unexpected_paths == ()
    assert delta.protected_changed_paths == ()
    assert (bridge.repo / provenance.output_path).read_text() == provenance.output_bytes

    documents = _event_documents(bridge, session_id)
    outcomes = [document["payload"] for document in documents if document["kind"] == "outcome"]
    digest = hashlib.sha256(provenance.command.encode()).hexdigest()
    assert outcomes[-1] == {
        "status": "completed",
        "summary": provenance.outcome_summary,
        "changed_files": [provenance.output_path],
        "known_failures": [],
        "remaining_risks": [],
        "architecture_docs": "No durable architecture change.",
        "question": None,
        "command_claims": [{"command_sha256": digest, "exit_code": 0}],
    }
    reviews = [document["payload"] for document in documents if document["kind"] == "review"]
    assert reviews[-1]["summary"] == provenance.review_summary
    assert reviews[-1]["criteria"] == [{
        "criterion": "The approved file is present and structurally tested.",
        "evidence": [provenance.review_evidence],
        "satisfied": True,
    }]
    observed_hashes = [
        event.payload.get("command_sha256")
        for event in bridge.store.events_after(session_id, 0)
        if event.kind == "agent_event"
        and event.payload.get("item_type") == "command_execution"
        and event.payload.get("type") == "item.completed"
    ]
    assert observed_hashes[-1:] == [digest]
    assert bridge.live_call_count == 0
    assert all(
        launch["executable"] in {str(bridge.fake_claude), str(bridge.fake_codex)}
        and launch["sentinel"] is True
        for launch in bridge.runner.launches
    )
    assert {entry["kind"] for entry in bridge.infrastructure_invocations} == {
        "git_wrapper", "git_delegate",
    }
    _assert_provider_continuity(bridge, provenance)


def _bridge_attack_snapshot(
    bridge: FakeBridge,
    *,
    project_id: str,
    session_ids: tuple[str, ...],
    task_ids: tuple[str, ...],
) -> dict[str, object]:
    tasks = {
        task_id: bridge.store.latest_task(task_id)
        for task_id in task_ids
    }
    prepared = {
        task_id: (
            None
            if task is None
            else bridge.store.latest_prepared_action_for_task(
                project_id=project_id,
                session_id=task.session_id,
                task_id=task_id,
                revision=task.revision,
            )
        )
        for task_id, task in tasks.items()
    }
    deltas = {}
    for task_id, task in tasks.items():
        if task is not None and task.baseline_id is not None:
            baseline = bridge.coordinator._load_baseline(task)
            deltas[task_id] = bridge.tracker.compare(baseline)
    return {
        "database": bridge.database.read_bytes(),
        "chats": bridge.store.list_chats(),
        "events": {
            session_id: _event_documents(bridge, session_id)
            for session_id in session_ids
        },
        "tasks": tasks,
        "prepared": prepared,
        "providers": tuple(
            invocation
            for invocation in bridge.invocations
            if invocation.get("kind") in {"claude", "codex"}
        ),
        "deltas": deltas,
    }


def _complete_http_workflow(
    client: TestClient,
    hub: FakeHub,
    *,
    label: str,
    session_id: str,
    headers: Mapping[str, str],
    text: str,
) -> str:
    bridge = hub.bridges[label]
    task_id = f"task-{session_id}"
    assert client.post(
        f"{_project_path(hub, label)}/chats/{session_id}/messages",
        json={"text": text},
        headers=headers,
    ).status_code == 202
    _wait_for(
        lambda: bridge.store.latest_task(task_id).state
        is TaskState.AWAITING_USER_APPROVAL  # type: ignore[union-attr]
    )
    assert client.post(
        f"{_project_path(hub, label)}/chats/{session_id}/tasks/{task_id}/approve",
        json={"revision": 1},
        headers=headers,
    ).status_code == 202
    _wait_for(
        lambda: bridge.store.latest_task(task_id).state
        is TaskState.COMPLETED  # type: ignore[union-attr]
    )
    return task_id


def test_two_project_http_websocket_workflow_isolated_by_hub_lease(
    tmp_path: Path,
    fake_claude: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A global lookup or lease release would cross the equal fake identities."""
    hub = FakeHub.create(
        root=tmp_path / "two-project", fake_claude=fake_claude, fake_codex=fake_codex,
    )
    first = hub.bridges["alpha"]
    second = hub.bridges["beta"]
    first.replace_scenario(_project_scenario(
        ALPHA_PROVENANCE,
        plan_count=4,
        completed_workflow_count=3,
        slow_first_plan=True,
        plan_paths=(
            ALPHA_PROVENANCE.output_path,
            ALPHA_PROVENANCE.output_path,
            "bridge_work/alpha-restarted.txt",
            "bridge_work/alpha-survivor.txt",
        ),
        workflow_paths=(
            ALPHA_PROVENANCE.output_path,
            "bridge_work/alpha-restarted.txt",
            "bridge_work/alpha-survivor.txt",
        ),
    ))
    second.replace_scenario(_project_scenario(
        BETA_PROVENANCE,
        plan_count=2,
        completed_workflow_count=2,
        plan_paths=(
            BETA_PROVENANCE.output_path,
            "bridge_work/beta-restarted.txt",
        ),
        workflow_paths=(
            BETA_PROVENANCE.output_path,
            "bridge_work/beta-restarted.txt",
        ),
    ))
    common_chat = "e2e-shared"
    common_task = f"task-{common_chat}"
    preparation_number = 0
    browser_id_values = iter((
        common_chat, common_chat, common_chat, common_chat,
        "alpha-restarted", "alpha-restarted",
        "beta-restarted", "beta-restarted",
        "alpha-survivor", "alpha-survivor",
    ))

    def deterministic_browser_ids(nbytes: int) -> str:
        nonlocal preparation_number
        if nbytes == 16:
            return next(browser_id_values)
        if nbytes == 24:
            preparation_number += 1
            return f"prepared-{preparation_number:040x}"
        raise AssertionError(f"unexpected test identifier width: {nbytes}")

    try:
        monkeypatch.setattr("agent_bridge.app.secrets.token_hex", deterministic_browser_ids)
        with TestClient(hub.app) as client:
            headers = _authenticate_hub(client)
            projects = client.get("/api/projects")
            assert projects.status_code == 200
            assert projects.json()["usage_credits_acknowledged"] is True
            assert projects.json()["active_lease"] is None
            assert {item["label"] for item in projects.json()["projects"]} == {
                "alpha", "beta",
            }

            first_chat = client.post(
                f"{_project_path(hub, 'alpha')}/chats", headers=headers,
            )
            second_chat = client.post(
                f"{_project_path(hub, 'beta')}/chats", headers=headers,
            )
            assert first_chat.status_code == second_chat.status_code == 201
            assert first_chat.json()["session_id"] == common_chat
            assert second_chat.json()["session_id"] == common_chat
            assert len(first.store.list_chats()) == len(second.store.list_chats()) == 2

            response = client.post(
                f"{_project_path(hub, 'alpha')}/chats/{common_chat}/messages",
                json={"text": "Plan the first equal-ID project."}, headers=headers,
            )
            assert response.status_code == 202
            _wait_for(lambda: (first.captures / "fake-claude-partials-ready.json").exists())
            active = hub.workflows.active_lease_snapshot()
            assert active is not None
            assert (active.project_id, active.session_id, active.task_id) == (
                hub.specs["alpha"].project_id, common_chat, common_task,
            )

            blocked = (
                client.post(
                    f"{_project_path(hub, 'beta')}/chats/{common_chat}/messages",
                    json={"text": "This must not reach the second provider."}, headers=headers,
                ),
                client.post(f"{_project_path(hub, 'beta')}/chats", headers=headers),
                client.get(
                    f"{_project_path(hub, 'beta')}/chats/{common_chat}/bootstrap"
                ),
            )
            assert [response.status_code for response in blocked] == [409, 409, 409]
            with pytest.raises(WebSocketDisconnect) as blocked_socket:
                with client.websocket_connect(
                    f"/ws?project_id={hub.specs['beta'].project_id}"
                    f"&session_id={common_chat}&after=0"
                ):
                    pass
            assert blocked_socket.value.code == 1008
            assert second.store.latest_task(common_task) is None
            assert second.store.events_after(common_chat, 0) == ()
            assert not [
                invocation for invocation in second.invocations
                if invocation.get("kind") in {"claude", "codex"}
            ]
            assert second.runner.launches == []
            assert not (second.repo / "bridge_work" / "output.txt").exists()

            assert client.post(
                f"{_project_path(hub, 'alpha')}/chats/{common_chat}"
                f"/tasks/{common_task}/stop",
                headers=headers,
            ).status_code == 202
            _wait_for(lambda: hub.workflows.active_lease_snapshot() is None)
            _wait_for(
                lambda: first.store.latest_task(common_task).state is TaskState.INTERRUPTED  # type: ignore[union-attr]
            )
            stopped_record = first.store.latest_prepared_action_for_task(
                project_id=hub.specs["alpha"].project_id,
                session_id=common_chat,
                task_id=common_task,
                revision=0,
            )
            assert stopped_record is not None
            assert stopped_record.status == "INTERRUPTED"
            resume = client.post(
                f"{_project_path(hub, 'alpha')}/chats/{common_chat}"
                f"/tasks/{common_task}/resume",
                headers=headers,
            )
            assert resume.status_code == 202, resume.text
            _wait_for(
                lambda: first.store.latest_task(common_task).state
                is TaskState.AWAITING_USER_APPROVAL  # type: ignore[union-attr]
            )
            assert client.post(
                f"{_project_path(hub, 'alpha')}/chats/{common_chat}"
                f"/tasks/{common_task}/approve",
                json={"revision": 1}, headers=headers,
            ).status_code == 202
            _wait_for(
                lambda: first.store.latest_task(common_task).state
                is TaskState.COMPLETED  # type: ignore[union-attr]
            )

            assert client.post(
                f"{_project_path(hub, 'beta')}/chats/{common_chat}/messages",
                json={"text": "Plan the second equal-ID project."}, headers=headers,
            ).status_code == 202
            _wait_for(
                lambda: second.store.latest_task(common_task).state
                is TaskState.AWAITING_USER_APPROVAL  # type: ignore[union-attr]
            )
            assert client.post(
                f"{_project_path(hub, 'beta')}/chats/{common_chat}"
                f"/tasks/{common_task}/approve",
                json={"revision": 1}, headers=headers,
            ).status_code == 202
            _wait_for(
                lambda: second.store.latest_task(common_task).state
                is TaskState.COMPLETED  # type: ignore[union-attr]
            )

            for label, bridge in hub.bridges.items():
                documents = _event_documents(bridge, common_chat)
                assert [document["sequence"] for document in documents] == list(
                    range(1, len(documents) + 1)
                )
                assert documents[0] == {
                    "sequence": 1,
                    "session_id": common_chat,
                    "task_id": common_task,
                    "actor": "user",
                    "kind": "message",
                    "payload": {
                        "text": (
                            "Plan the first equal-ID project."
                            if label == "alpha"
                            else "Plan the second equal-ID project."
                        )
                    },
                    "created_at": "2026-08-10T14:00:00Z",
                }
                with client.websocket_connect(
                    f"/ws?project_id={hub.specs[label].project_id}"
                    f"&session_id={common_chat}&after=0"
                ) as socket:
                    assert [socket.receive_json() for _ in documents] == documents
                with client.websocket_connect(
                    f"/ws?project_id={hub.specs[label].project_id}"
                    f"&session_id={common_chat}&after={documents[-2]['sequence']}"
                ) as socket:
                    assert socket.receive_json() == documents[-1]

        for bridge, provenance in (
            (first, ALPHA_PROVENANCE),
            (second, BETA_PROVENANCE),
        ):
            digest = hashlib.sha256(provenance.command.encode()).hexdigest()
            task = bridge.store.get_task(common_task, 1)
            assert task.state is TaskState.COMPLETED
            assert task.session_id == common_chat
            assert task.fable_session_id == provenance.fable_session_id
            assert task.sol_thread_id == provenance.sol_thread_id
            assert task.baseline_id is not None
            baseline = bridge.coordinator._load_baseline(task)
            delta = bridge.tracker.compare(baseline)
            assert delta.changed_paths == (provenance.output_path,)
            assert delta.unexpected_paths == ()
            assert delta.protected_changed_paths == ()
            assert (bridge.repo / provenance.output_path).read_text() == provenance.output_bytes
            outcomes = [
                document["payload"] for document in _event_documents(bridge, common_chat)
                if document["kind"] == "outcome"
            ]
            assert outcomes[-1] == {
                "status": "completed",
                "summary": provenance.outcome_summary,
                "changed_files": [provenance.output_path],
                "known_failures": [],
                "remaining_risks": [],
                "architecture_docs": "No durable architecture change.",
                "question": None,
                "command_claims": [{"command_sha256": digest, "exit_code": 0}],
            }
            observed_hashes = [
                event.payload.get("command_sha256")
                for event in bridge.store.events_after(common_chat, 0)
                if event.kind == "agent_event"
                and event.payload.get("item_type") == "command_execution"
                and event.payload.get("type") == "item.completed"
            ]
            assert observed_hashes == [digest]
            assert bridge.live_call_count == 0
            assert all(
                launch["executable"] in {
                    str(bridge.fake_claude), str(bridge.fake_codex),
                }
                and launch["sentinel"] is True
                for launch in bridge.runner.launches
            )
            assert {entry["kind"] for entry in bridge.infrastructure_invocations} == {
                "git_wrapper", "git_delegate",
            }
            provider_invocations = [
                entry for entry in bridge.invocations
                if entry.get("kind") in {"claude", "codex"}
            ]
            assert provider_invocations
            assert all(
                entry["executable"] in {
                    str(bridge.fake_claude), str(bridge.fake_codex),
                }
                for entry in provider_invocations
            )
            _assert_project_provenance(
                bridge, provenance, session_id=common_chat, task_id=common_task,
            )

        first_model_calls = [
            entry for entry in first.invocations
            if entry.get("kind") == "claude"
            and entry.get("argv") != ["auth", "status", "--json"]
        ]
        assert "--resume" not in first_model_calls[0]["argv"]
        assert any(
            "--resume" in entry["argv"]
            and entry["argv"][entry["argv"].index("--resume") + 1]
            == ALPHA_PROVENANCE.fable_session_id
            for entry in first_model_calls[1:]
        )

        first_task = first.store.get_task(common_task, 1)
        second_task = second.store.get_task(common_task, 1)
        assert first_task.baseline_id != second_task.baseline_id
        assert first_task.brief is not None and second_task.brief is not None
        assert first_task.brief.allowed_paths != second_task.brief.allowed_paths
        assert first_task.fable_session_id != second_task.fable_session_id
        assert first_task.sol_thread_id != second_task.sol_thread_id
        assert ALPHA_PROVENANCE.command != BETA_PROVENANCE.command
        assert hashlib.sha256(ALPHA_PROVENANCE.command.encode()).hexdigest() != (
            hashlib.sha256(BETA_PROVENANCE.command.encode()).hexdigest()
        )
        assert ALPHA_PROVENANCE.outcome_summary != BETA_PROVENANCE.outcome_summary
        assert ALPHA_PROVENANCE.review_evidence != BETA_PROVENANCE.review_evidence
        for counterfactual in (
            replace(BETA_PROVENANCE, command=ALPHA_PROVENANCE.command),
            replace(
                BETA_PROVENANCE,
                output_path=ALPHA_PROVENANCE.output_path,
                output_bytes=ALPHA_PROVENANCE.output_bytes,
                outcome_summary=ALPHA_PROVENANCE.outcome_summary,
            ),
            replace(
                BETA_PROVENANCE,
                review_summary=ALPHA_PROVENANCE.review_summary,
                review_evidence=ALPHA_PROVENANCE.review_evidence,
            ),
            replace(
                BETA_PROVENANCE,
                fable_session_id=ALPHA_PROVENANCE.fable_session_id,
                sol_thread_id=ALPHA_PROVENANCE.sol_thread_id,
            ),
        ):
            with pytest.raises(AssertionError):
                _assert_project_provenance(
                    second,
                    counterfactual,
                    session_id=common_chat,
                    task_id=common_task,
                )
        assert hub.hub_store.usage_credits_acknowledged() is True
        assert hub.workflows.active_lease_snapshot() is None

        common_documents = {
            label: _event_documents(bridge, common_chat)
            for label, bridge in hub.bridges.items()
        }
        first.store.create_planning_task(SESSION_ID, "recovery-only")
        former_hub_store = hub.hub_store
        former_registry = hub.registry
        former_app = hub.app
        former_runtimes = {
            label: hub.runtime(label) for label in ("alpha", "beta")
        }
        recovered = hub.restart()
        assert [task.task_id for task in recovered["alpha"]] == ["recovery-only"]
        assert recovered["beta"] == ()
        assert hub.hub_store is not former_hub_store
        assert hub.registry is not former_registry
        assert hub.app is not former_app
        assert former_hub_store._closed
        assert all(runtime._closed for runtime in former_runtimes.values())
        first = hub.bridges["alpha"]
        second = hub.bridges["beta"]
        assert hub.runtime("alpha") is not former_runtimes["alpha"]
        assert hub.runtime("beta") is not former_runtimes["beta"]
        assert first.store.get_task("recovery-only", 0).state is TaskState.INTERRUPTED
        assert second.store.get_task(common_task, 1).state is TaskState.COMPLETED
        assert hub.hub_store.usage_credits_acknowledged() is True
        assert hub.workflows.active_lease_snapshot() is None

        with TestClient(hub.app) as client:
            headers = _authenticate_hub(client)
            projects = client.get("/api/projects")
            assert projects.status_code == 200
            assert {project["label"] for project in projects.json()["projects"]} == {
                "alpha", "beta",
            }
            for label in ("alpha", "beta"):
                bootstrap = client.get(
                    f"{_project_path(hub, label)}/chats/{common_chat}/bootstrap"
                )
                assert bootstrap.status_code == 200
                assert bootstrap.json()["project_id"] == hub.specs[label].project_id
                documents = common_documents[label]
                with client.websocket_connect(
                    f"/ws?project_id={hub.specs[label].project_id}"
                    f"&session_id={common_chat}&after={documents[-2]['sequence']}"
                ) as socket:
                    assert socket.receive_json() == documents[-1]

            restarted_alpha = client.post(
                f"{_project_path(hub, 'alpha')}/chats", headers=headers,
            )
            assert restarted_alpha.status_code == 201
            assert restarted_alpha.json()["session_id"] == "alpha-restarted"
            _complete_http_workflow(
                client,
                hub,
                label="alpha",
                session_id="alpha-restarted",
                headers=headers,
                text="Safely continue Alpha after the full restart.",
            )
            restarted_beta = client.post(
                f"{_project_path(hub, 'beta')}/chats", headers=headers,
            )
            assert restarted_beta.status_code == 201
            assert restarted_beta.json()["session_id"] == "beta-restarted"
            _complete_http_workflow(
                client,
                hub,
                label="beta",
                session_id="beta-restarted",
                headers=headers,
                text="Safely continue Beta after the full restart.",
            )

        second.store.create_session("beta-only-chat", str(second.repo))
        second.store.save_task(
            "beta-only-chat",
            TaskBrief.from_dict(_brief(task_id="beta-only-task")),
            TaskState.AWAITING_USER_APPROVAL,
        )
        before_alpha = _bridge_attack_snapshot(
            first,
            project_id=hub.specs["alpha"].project_id,
            session_ids=(common_chat,),
            task_ids=(common_task,),
        )
        before_beta = _bridge_attack_snapshot(
            second,
            project_id=hub.specs["beta"].project_id,
            session_ids=(common_chat, "beta-only-chat"),
            task_ids=(common_task, "beta-only-task"),
        )
        with TestClient(hub.app) as client:
            headers = _authenticate_hub(client)
            request_cases = (
                ("get", "/chats/{chat}/bootstrap", None),
                ("post", "/chats/{chat}/messages", {"text": "hostile cross-project identifier"}),
                ("post", "/chats/{chat}/tasks/{task}/approve", {"revision": 1}),
                ("post", "/chats/{chat}/tasks/{task}/edit", _brief(task_id="beta-only-task")),
                ("post", "/chats/{chat}/tasks/{task}/reject", None),
                ("post", "/chats/{chat}/tasks/{task}/answer", {"answer": "hostile answer"}),
                ("post", "/chats/{chat}/tasks/{task}/stop", None),
                ("post", "/chats/{chat}/tasks/{task}/resume", None),
            )
            for method, suffix, body in request_cases:
                foreign_suffix = suffix.format(chat="beta-only-chat", task="beta-only-task")
                missing_suffix = suffix.format(chat="not-a-real-chat", task="not-a-real-task")
                request = getattr(client, method)
                foreign = request(
                    f"{_project_path(hub, 'alpha')}{foreign_suffix}",
                    headers=headers,
                    **({"json": body} if body is not None else {}),
                )
                missing = request(
                    f"{_project_path(hub, 'alpha')}{missing_suffix}",
                    headers=headers,
                    **({"json": body} if body is not None else {}),
                )
                assert _response_signature(foreign) == _response_signature(missing)
                assert foreign.status_code in {404, 409}
            for session_id in ("beta-only-chat", "not-a-real-chat"):
                with pytest.raises(WebSocketDisconnect) as rejected:
                    with client.websocket_connect(
                        f"/ws?project_id={hub.specs['alpha'].project_id}"
                        f"&session_id={session_id}&after=0"
                    ):
                        pass
                assert rejected.value.code == 1008
        assert _bridge_attack_snapshot(
            first,
            project_id=hub.specs["alpha"].project_id,
            session_ids=(common_chat,),
            task_ids=(common_task,),
        ) == before_alpha
        assert _bridge_attack_snapshot(
            second,
            project_id=hub.specs["beta"].project_id,
            session_ids=(common_chat, "beta-only-chat"),
            task_ids=(common_task, "beta-only-task"),
        ) == before_beta

        beta_project_id = hub.specs["beta"].project_id
        beta_state_dir = hub.specs["beta"].state_dir
        removed_state = _tree_bytes(second.root)
        removed_project_state = _tree_bytes(beta_state_dir)
        former_beta_runtime = hub.runtime("beta")
        hub.restart(labels=("alpha",))
        assert former_beta_runtime._closed
        assert "beta" not in hub.bridges
        assert "beta" not in hub.specs
        assert all(runtime.project_id != beta_project_id for runtime in hub.registry.projects())
        first = hub.bridges["alpha"]
        before_removed_attacks = _bridge_attack_snapshot(
            first,
            project_id=hub.specs["alpha"].project_id,
            session_ids=(common_chat, "alpha-restarted"),
            task_ids=(common_task, "task-alpha-restarted"),
        )
        missing_project_id = "f" * 32
        with TestClient(hub.app) as client:
            headers = _authenticate_hub(client)
            projects = client.get("/api/projects")
            assert projects.status_code == 200
            assert [project["label"] for project in projects.json()["projects"]] == ["alpha"]
            removed_base = f"/api/projects/{beta_project_id}"
            missing_base = f"/api/projects/{missing_project_id}"
            removed_cases = (
                ("get", "/chats", None),
                ("get", "/chats/e2e-shared/bootstrap", None),
                ("post", "/chats", None),
                ("post", "/chats/e2e-shared/messages", {"text": "removed project probe"}),
                ("post", "/chats/e2e-shared/tasks/task-e2e-shared/approve", {"revision": 1}),
                ("post", "/chats/e2e-shared/tasks/task-e2e-shared/edit", _brief(task_id="task-e2e-shared")),
                ("post", "/chats/e2e-shared/tasks/task-e2e-shared/reject", None),
                ("post", "/chats/e2e-shared/tasks/task-e2e-shared/answer", {"answer": "removed project probe"}),
                ("post", "/chats/e2e-shared/tasks/task-e2e-shared/stop", None),
                ("post", "/chats/e2e-shared/tasks/task-e2e-shared/resume", None),
            )
            for method, suffix, body in removed_cases:
                request = getattr(client, method)
                removed = request(
                    f"{removed_base}{suffix}",
                    headers=headers,
                    **({"json": body} if body is not None else {}),
                )
                missing = request(
                    f"{missing_base}{suffix}",
                    headers=headers,
                    **({"json": body} if body is not None else {}),
                )
                assert _response_signature(removed) == _response_signature(missing)
                assert removed.status_code == 404
            for project_id in (beta_project_id, missing_project_id):
                with pytest.raises(WebSocketDisconnect) as rejected:
                    with client.websocket_connect(
                        f"/ws?project_id={project_id}&session_id={common_chat}&after=0"
                    ):
                        pass
                assert rejected.value.code == 1008
        assert _bridge_attack_snapshot(
            first,
            project_id=hub.specs["alpha"].project_id,
            session_ids=(common_chat, "alpha-restarted"),
            task_ids=(common_task, "task-alpha-restarted"),
        ) == before_removed_attacks
        assert _tree_bytes(second.root) == removed_state
        assert _tree_bytes(beta_state_dir) == removed_project_state
        assert second.database.read_bytes() == removed_state[
            second.database.relative_to(second.root).as_posix()
        ]

        with TestClient(hub.app) as client:
            headers = _authenticate_hub(client)
            surviving_chat = client.post(
                f"{_project_path(hub, 'alpha')}/chats", headers=headers,
            )
            assert surviving_chat.status_code == 201
            assert surviving_chat.json()["session_id"] == "alpha-survivor"
            surviving_bootstrap = client.get(
                f"{_project_path(hub, 'alpha')}/chats/alpha-survivor/bootstrap"
            )
            assert surviving_bootstrap.status_code == 200
            assert surviving_bootstrap.json()["project_id"] == hub.specs["alpha"].project_id
        assert hub.hub_store.usage_credits_acknowledged() is True
        assert hub.workflows.active_lease_snapshot() is None
    finally:
        hub.close()


def test_partial_project_lock_failure_opens_no_runtime_or_database(
    tmp_path: Path,
    fake_claude: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving lock acquisition after state opening would leave startup residue."""
    root = tmp_path / "partial-lock"
    first = _make_fake_bridge(root / "first", fake_claude, fake_codex, repo_name="same-name")
    second = _make_fake_bridge(root / "second", fake_claude, fake_codex, repo_name="same-name")
    state_home = root / "state-home"
    calls: list[Path] = []
    original_acquire = launcher.acquire_instance_lock
    lifecycle: list[str] = []

    def fail_on_second_project(path: Path, **kwargs: object):
        calls.append(path)
        if len(calls) == 3:
            raise RuntimeError("controlled second-project lock failure")
        return original_acquire(path, **kwargs)

    def forbidden_open(*args: object, **kwargs: object) -> object:
        lifecycle.append("open")
        raise AssertionError("lock failure must precede database opening")

    def forbidden_runtime(*args: object, **kwargs: object) -> object:
        lifecycle.append("runtime")
        raise AssertionError("lock failure must precede runtime assembly")

    def forbidden_hub_store(*args: object, **kwargs: object) -> object:
        lifecycle.append("hub_store")
        raise AssertionError("lock failure must precede hub database opening")

    def forbidden_app(*args: object, **kwargs: object) -> object:
        lifecycle.append("app")
        raise AssertionError("lock failure must precede application creation")

    monkeypatch.setattr(launcher, "acquire_instance_lock", fail_on_second_project)
    monkeypatch.setattr(launcher, "_open_project_state", forbidden_open)
    monkeypatch.setattr(launcher, "assemble_project_runtime", forbidden_runtime)
    monkeypatch.setattr("agent_bridge.hub_store.HubStore", forbidden_hub_store)
    monkeypatch.setattr("agent_bridge.app.create_hub_app", forbidden_app)
    arguments = [
        "--project", f"alpha={first.repo}",
        "--project", f"beta={second.repo}",
        "--claude-executable", str(first.fake_claude),
        "--codex-executable", str(first.fake_codex),
        "--git-executable", str(first.fake_git),
        "--bash-executable", str(first.fake_bash),
        "--sh-executable", str(first.fake_sh),
    ]
    try:
        with pytest.raises(RuntimeError, match="second-project lock failure"):
            launcher.main(
                arguments,
                environ={
                    "AGENT_BRIDGE_TEST_FAKE": "1",
                    "XDG_STATE_HOME": str(state_home),
                    "PATH": "/path-resolution-is-forbidden",
                    "LANG": "C",
                    "LC_ALL": "C",
                },
                stdout=io.StringIO(),
                uvicorn_run=lambda *args, **kwargs: lifecycle.append("server"),
            )
        assert len(calls) == 3
        assert lifecycle == []
        assert not list((state_home / "agent-bridge").rglob("*.sqlite3"))
        assert not list((state_home / "agent-bridge").rglob("artifacts"))
        assert not list((state_home / "agent-bridge").rglob("schemas"))
    finally:
        first.close()
        second.close()


def test_audited_legacy_adoption_preserves_project_history_without_hub_ack(
    tmp_path: Path,
    fake_claude: Path,
    fake_codex: Path,
) -> None:
    """Copying the legacy acknowledgement into HubStore would enable model starts."""
    root = tmp_path / "legacy-adoption"
    bridge = _make_fake_bridge(root / "fixture", fake_claude, fake_codex, repo_name="legacy-repo")
    state_home = root / "state-home"
    legacy_state = state_home / "agent-bridge" / "legacy-repo"
    legacy_state.mkdir(parents=True)
    database = legacy_state / "bridge.sqlite3"
    store = SQLiteStore(
        database,
        clock=lambda: "2026-08-11T12:00:00Z",
        check_same_thread=False,
    )
    tracker = RepositoryTracker(
        bridge.repo, legacy_state / "artifacts", git_executable=bridge.fake_git,
    )
    brief = TaskBrief.from_dict(_brief(task_id="legacy-task"))
    store.create_session("legacy-chat", str(bridge.repo))
    store.create_session("legacy-other-chat", str(bridge.repo))
    store.set_setting("agent_bridge.active_session_id", "legacy-chat")
    store.set_setting("usage_credits_acknowledged", True)
    store.save_task("legacy-chat", brief, TaskState.AWAITING_USER_APPROVAL)
    baseline = tracker.capture(brief)
    store.approve_task_with_setting(
        "legacy-task",
        1,
        brief=brief,
        baseline_id=baseline.baseline_id,
        expected=TaskState.AWAITING_USER_APPROVAL,
        setting=(
            "agent_bridge.baseline.legacy-task.1",
            {
                "task_id": "legacy-task",
                "revision": 1,
                "baseline_id": baseline.baseline_id,
                "manifest": tracker.baseline_manifest(baseline),
            },
        ),
    )
    store.append_event(
        "legacy-chat", "legacy-task", "user", "message", {"text": "preserve me"},
    )
    store.append_event(
        "legacy-other-chat", None, "user", "message", {"text": "preserve other history"},
    )
    store.start_agent_run("legacy-run", "legacy-task", 1, "fable")
    store.finish_agent_run("legacy-run", status="completed", exit_code=0)
    expected_events = {
        session_id: [event.to_dict() for event in store.events_after(session_id, 0)]
        for session_id in ("legacy-chat", "legacy-other-chat")
    }
    expected_chats = store.list_chats()
    expected_chat_records = {
        session_id: store.chat(session_id)
        for session_id in ("legacy-chat", "legacy-other-chat")
    }
    expected_active_session = store.get_setting("agent_bridge.active_session_id")
    expected_task = store.get_task("legacy-task", 1)
    expected_run = store.agent_run("legacy-run")
    expected_baseline = store.get_setting("agent_bridge.baseline.legacy-task.1")
    expected_baseline_bytes = _tree_bytes(legacy_state / "artifacts")
    store.close()
    tracker.close()
    observed: list[dict[str, object]] = []

    def inspect_started_app(app: Any, *, host: str, port: int, reload: bool) -> None:
        runtime = app.state.project_registry.projects()[0]
        observed.append({
            "project_id": runtime.project_id,
            "state_dir": runtime.spec.state_dir,
            "events": {
                session_id: [
                    event.to_dict()
                    for event in runtime.store.events_after(session_id, 0)
                ]
                for session_id in ("legacy-chat", "legacy-other-chat")
            },
            "chats": runtime.store.list_chats(),
            "chat_records": {
                session_id: runtime.store.chat(session_id)
                for session_id in ("legacy-chat", "legacy-other-chat")
            },
            "active_session": runtime.store.get_setting("agent_bridge.active_session_id"),
            "task": runtime.store.get_task("legacy-task", 1),
            "run": runtime.store.agent_run("legacy-run"),
            "baseline": runtime.store.get_setting("agent_bridge.baseline.legacy-task.1"),
            "baseline_bytes": _tree_bytes(runtime.spec.state_dir / "artifacts"),
            "project_ack": runtime.store.get_setting("usage_credits_acknowledged"),
            "hub_ack": app.state.hub_store.usage_credits_acknowledged(),
        })

    arguments = [
        "--repo", str(bridge.repo),
        "--claude-executable", str(bridge.fake_claude),
        "--codex-executable", str(bridge.fake_codex),
        "--git-executable", str(bridge.fake_git),
        "--bash-executable", str(bridge.fake_bash),
        "--sh-executable", str(bridge.fake_sh),
    ]
    environment = {
        "AGENT_BRIDGE_TEST_FAKE": "1",
        "AGENT_BRIDGE_INVOCATION_LOG": str(bridge.invocation_log),
        "FAKE_AGENT_CAPTURE_DIR": str(bridge.captures),
        "FAKE_AGENT_SCENARIO": str(bridge.scenario_path),
        "FAKE_BRIDGE_REPO_ROOT": str(bridge.repo),
        "HOME": str(root / "home"),
        "XDG_STATE_HOME": str(state_home),
        "PATH": "/path-resolution-is-forbidden",
        "LANG": "C",
        "LC_ALL": "C",
    }
    try:
        for _ in range(2):
            assert launcher.main(
                arguments,
                environ=environment,
                stdout=io.StringIO(),
                uvicorn_run=inspect_started_app,
            ) == 0
        assert len(observed) == 2
        for checkpoint in observed:
            assert checkpoint["state_dir"] == legacy_state
            assert checkpoint["events"] == expected_events
            assert checkpoint["chats"] == expected_chats
            assert checkpoint["chat_records"] == expected_chat_records
            assert checkpoint["active_session"] == expected_active_session
            assert checkpoint["task"] == expected_task
            assert checkpoint["run"] == expected_run
            assert checkpoint["baseline"] == expected_baseline
            assert checkpoint["baseline_bytes"] == expected_baseline_bytes
            assert checkpoint["project_ack"] is True
            assert checkpoint["hub_ack"] is False
            assert not (
                state_home / "agent-bridge" / "projects" / checkpoint["project_id"]
            ).exists()
        assert bridge.live_call_count == 0
    finally:
        bridge.close()
