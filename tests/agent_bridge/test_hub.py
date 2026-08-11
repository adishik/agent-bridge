from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from agent_bridge.hub import (
    ActiveAgentLease,
    HubWorkflowOrchestrator,
    OwnedProjectRuntime,
    ProjectRegistry,
    RuntimeReadiness,
    RuntimeStatus,
)
from agent_bridge.projects import ProjectSpec


@dataclass
class _Ids:
    task_number: int = 0

    def new_task_id(self) -> str:
        self.task_number += 1
        return f"task-{self.task_number}"

    def new_run_id(self) -> str:
        return "run-unused"


@dataclass
class _RuntimeStore:
    sessions: set[str]
    task_rows: dict[tuple[str, int], object] | None = None
    listener_tokens: list[int] | None = None
    removed_listener_tokens: list[int] | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        if self.task_rows is None:
            self.task_rows = {}
        if self.listener_tokens is None:
            self.listener_tokens = []
        if self.removed_listener_tokens is None:
            self.removed_listener_tokens = []

    def session_exists(self, session_id: str) -> bool:
        return session_id in self.sessions

    def get_task(self, task_id: str, revision: int) -> object:
        return self.task_rows[(task_id, revision)]

    def add_event_listener(self, listener: object) -> int:
        token = len(self.listener_tokens) + 1
        self.listener_tokens.append(token)
        return token

    def remove_event_listener(self, token: int) -> None:
        self.removed_listener_tokens.append(token)

    def close(self) -> None:
        self.closed = True


@dataclass
class _RuntimeCoordinator:
    prepared: list[tuple[str, str, str]]
    run_task_ids: list[str]
    aborts: list[tuple[str, int, str, str]]
    stops: list[str]

    def prepare_user_request(self, session_id: str, text: str, task_id: str) -> object:
        self.prepared.append((session_id, text, task_id))
        return SimpleNamespace(task_id=task_id, session_id=session_id, revision=0)

    async def run_prepared_request(self, task_id: str) -> None:
        self.run_task_ids.append(task_id)

    def abort_prepared_action(
        self, task_id: str, revision: int, action: str, reason: str,
    ) -> object:
        self.aborts.append((task_id, revision, action, reason))
        return SimpleNamespace(task_id=task_id, revision=revision)

    async def stop_task(self, task_id: str) -> None:
        self.stops.append(task_id)


@dataclass
class _Runtime:
    project_id: str
    label: str
    repository: str
    branch: str
    store: _RuntimeStore
    coordinator: _RuntimeCoordinator
    readiness: RuntimeReadiness
    broadcaster: object = object()


def _readiness(
    *, fable: tuple[bool, str] = (True, "subscription_ready"), sol: str = "ready",
    fable_calls: list[None] | None = None, sol_calls: list[None] | None = None,
) -> RuntimeReadiness:
    async def fable_probe() -> tuple[bool, str]:
        if fable_calls is not None:
            fable_calls.append(None)
        return fable

    async def sol_probe() -> str:
        if sol_calls is not None:
            sol_calls.append(None)
        return sol

    return RuntimeReadiness(
        initial=RuntimeStatus(False, "checking", "checking"),
        fable_probe=fable_probe,
        sol_probe=sol_probe,
    )


def _runtime(project_id: str, *, sessions: set[str] | None = None) -> _Runtime:
    return _Runtime(
        project_id=project_id,
        label=project_id,
        repository=f"/repositories/{project_id}",
        branch="main",
        store=_RuntimeStore(sessions or {"chat-1"}),
        coordinator=_RuntimeCoordinator([], [], [], []),
        readiness=_readiness(),
    )


def test_runtime_readiness_refreshes_both_provider_states_before_start() -> None:
    async def exercise() -> None:
        fable_calls: list[None] = []
        sol_calls: list[None] = []
        readiness = _readiness(fable_calls=fable_calls, sol_calls=sol_calls)

        ready = await readiness.require_model_start_ready(
            usage_credits_acknowledged=True
        )

        assert ready == RuntimeStatus(True, "subscription_ready", "ready")
        assert readiness.snapshot() == ready
        assert fable_calls == [None]
        assert sol_calls == [None]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("fable", "sol", "expected"),
    (
        (
            (False, "subscription_unavailable"),
            "ready",
            RuntimeStatus(False, "subscription_unavailable", "ready"),
        ),
        (
            (True, "subscription_ready"),
            "unavailable",
            RuntimeStatus(True, "subscription_ready", "unavailable"),
        ),
        (
            (True, "not-a-status"),
            "ready",
            RuntimeStatus(False, "subscription_unavailable", "unavailable"),
        ),
    ),
)
def test_runtime_readiness_fails_closed_for_unavailable_or_malformed_probe_results(
    fable: tuple[bool, str], sol: str, expected: RuntimeStatus,
) -> None:
    async def exercise() -> None:
        readiness = _readiness(fable=fable, sol=sol)

        with pytest.raises(RuntimeError, match="not ready"):
            await readiness.require_model_start_ready(usage_credits_acknowledged=True)

        assert readiness.snapshot() == expected

    asyncio.run(exercise())


def test_runtime_readiness_rejects_unacknowledged_start_without_provider_probe() -> None:
    async def exercise() -> None:
        fable_calls: list[None] = []
        sol_calls: list[None] = []
        readiness = _readiness(fable_calls=fable_calls, sol_calls=sol_calls)

        with pytest.raises(PermissionError, match="acknowledged"):
            await readiness.require_model_start_ready(usage_credits_acknowledged=False)

        assert fable_calls == []
        assert sol_calls == []

    asyncio.run(exercise())


def test_registry_returns_only_the_selected_project_and_has_stable_order() -> None:
    first = _runtime("project-a")
    second = _runtime("project-b")
    registry = ProjectRegistry((second, first))

    selected = registry.runtime("project-a")

    assert selected is first
    assert selected.store is first.store
    assert tuple(runtime.project_id for runtime in registry.projects()) == (
        "project-a", "project-b",
    )


def test_registry_rejects_duplicate_and_unknown_project_ids_without_exposing_registry_contents() -> None:
    first = _runtime("project-a")
    duplicate = _runtime("project-a")

    with pytest.raises(ValueError, match="duplicate project id"):
        ProjectRegistry((first, duplicate))

    registry = ProjectRegistry((first,))
    with pytest.raises(LookupError) as error:
        registry.runtime("not-a-project")

    assert str(error.value) == "project not found"


@dataclass
class _CloseResource:
    name: str
    calls: list[str]
    raises: bool = False

    def close(self) -> None:
        self.calls.append(self.name)
        if self.raises:
            raise RuntimeError(self.name)


@dataclass
class _CloseStore(_RuntimeStore):
    calls: list[str] | None = None

    def close(self) -> None:
        assert self.calls is not None
        self.calls.append("store")
        super().close()


@dataclass
class _Lock:
    name: str
    calls: list[str]
    released: bool = False

    def release(self) -> None:
        self.calls.append(self.name)
        self.released = True


def _owned_runtime(project_id: str, calls: list[str], *, coordinator_raises: bool = False) -> OwnedProjectRuntime:
    store = _CloseStore({"chat-1"}, calls=calls)
    return OwnedProjectRuntime(
        spec=ProjectSpec(
            project_id=project_id,
            label=project_id,
            repo_root=Path(f"/repositories/{project_id}"),
            branch="main",
            state_dir=Path(f"/state/{project_id}"),
        ),
        store=store,  # type: ignore[arg-type]
        tracker=_CloseResource("tracker", calls),  # type: ignore[arg-type]
        runner=object(),  # type: ignore[arg-type]
        fable=object(),  # type: ignore[arg-type]
        sol=object(),  # type: ignore[arg-type]
        coordinator=_CloseResource("coordinator", calls, coordinator_raises),  # type: ignore[arg-type]
        broadcaster=SimpleNamespace(publish=lambda event: None),  # type: ignore[arg-type]
        readiness=_readiness(),
        lock=_Lock(f"lock-{project_id}", calls),  # type: ignore[arg-type]
    )


def test_registry_close_releases_owned_runtimes_in_reverse_order_after_close_failure() -> None:
    calls: list[str] = []
    first = _owned_runtime("project-a", calls)
    second = _owned_runtime("project-b", calls, coordinator_raises=True)
    registry = ProjectRegistry((first, second))

    with pytest.raises(RuntimeError, match="coordinator"):
        registry.close()
    registry.close()

    assert first.store.removed_listener_tokens == [1]
    assert second.store.removed_listener_tokens == [1]
    assert calls == [
        "coordinator", "tracker", "store", "lock-project-b",
        "coordinator", "tracker", "store", "lock-project-a",
    ]


def test_registry_close_does_not_close_a_non_owning_runtime() -> None:
    runtime = _runtime("project-a")
    registry = ProjectRegistry((runtime,))

    registry.close()

    assert runtime.store.closed is False


def test_active_agent_lease_allows_exactly_one_concurrent_acquisition() -> None:
    lease = ActiveAgentLease()
    barrier = threading.Barrier(3)
    results: list[object] = []

    def acquire(session_id: str) -> None:
        barrier.wait()
        try:
            results.append(lease.acquire(
                project_id="project-a", session_id=session_id, task_id="task-1",
            ))
        except RuntimeError as error:
            results.append(error)

    first = threading.Thread(target=acquire, args=("chat-a",))
    second = threading.Thread(target=acquire, args=("chat-b",))
    first.start()
    second.start()
    barrier.wait()
    first.join()
    second.join()

    tokens = [item for item in results if not isinstance(item, RuntimeError)]
    errors = [item for item in results if isinstance(item, RuntimeError)]
    assert len(tokens) == 1
    assert len(errors) == 1
    assert lease.snapshot() == tokens[0]


def test_acquire_new_keeps_task_id_creation_inside_the_lease_critical_section() -> None:
    class BlockingIds(_Ids):
        entered = threading.Event()
        release = threading.Event()

        def new_task_id(self) -> str:
            self.entered.set()
            assert self.release.wait(timeout=1)
            return super().new_task_id()

    lease = ActiveAgentLease()
    ids = BlockingIds()
    created: list[object] = []
    contender_finished = threading.Event()

    def create() -> None:
        created.append(lease.acquire_new(
            project_id="project-a", session_id="chat-a", ids=ids,
        ))

    def contend() -> None:
        try:
            created.append(lease.acquire(
                project_id="project-b", session_id="chat-b", task_id="task-b",
            ))
        except RuntimeError as error:
            created.append(error)
        finally:
            contender_finished.set()

    creator = threading.Thread(target=create)
    creator.start()
    assert ids.entered.wait(timeout=1)
    contender = threading.Thread(target=contend)
    contender.start()
    assert contender_finished.wait(timeout=0.05) is False
    ids.release.set()
    creator.join()
    contender.join()

    assert sum(not isinstance(item, RuntimeError) for item in created) == 1
    assert sum(isinstance(item, RuntimeError) for item in created) == 1
    assert lease.snapshot() is not None
    assert lease.snapshot().task_id == "task-1"  # type: ignore[union-attr]


def test_stale_or_repeated_release_cannot_clear_a_new_lease_generation() -> None:
    lease = ActiveAgentLease()
    first = lease.acquire(project_id="project-a", session_id="chat-a", task_id="task-a")
    lease.release(first)
    second = lease.acquire(project_id="project-b", session_id="chat-b", task_id="task-b")

    lease.release(first)
    assert lease.snapshot() == second
    lease.release(second)
    lease.release(second)
    assert lease.snapshot() is None


def test_workflow_preparation_holds_one_lease_for_fresh_readiness_and_releases_on_gate_failure() -> None:
    async def exercise() -> None:
        fable_calls: list[None] = []
        sol_calls: list[None] = []
        first = _runtime("project-a")
        first.readiness = _readiness(fable_calls=fable_calls, sol_calls=sol_calls)
        second = _runtime("project-b")
        lease = ActiveAgentLease()
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((first, second)),
            lease=lease,
            usage_credits_acknowledged=lambda: True,
        )

        prepared = await workflows.prepare_new_request(
            project_id="project-a", session_id="chat-1", text="Build it", ids=_Ids(),
        )
        with pytest.raises(RuntimeError, match="another workflow"):
            await workflows.prepare_new_request(
                project_id="project-b", session_id="chat-1", text="Also build", ids=_Ids(),
            )
        assert first.coordinator.prepared == [("chat-1", "Build it", "task-1")]
        assert fable_calls == [None]
        assert sol_calls == [None]

        workflows.abort_prepared(prepared, reason="untrusted scheduler detail")
        assert lease.snapshot() is None
        assert first.coordinator.aborts == [
            ("task-1", 0, "new_request", "scheduler_unavailable"),
        ]

        failed = _runtime("project-c")
        failed.readiness = _readiness(fable=(False, "subscription_unavailable"))
        failed_workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((failed,)),
            lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        with pytest.raises(RuntimeError, match="not ready"):
            await failed_workflows.prepare_new_request(
                project_id="project-c", session_id="chat-1", text="No child", ids=_Ids(),
            )
        assert failed.coordinator.prepared == []
        assert lease.snapshot() is None

    asyncio.run(exercise())


@pytest.mark.parametrize("action", ("approval", "resume", "answer"))
def test_held_lease_rejects_other_model_starting_routes_without_new_probes(
    action: str,
) -> None:
    async def exercise() -> None:
        fable_calls: list[None] = []
        sol_calls: list[None] = []
        first = _runtime("project-a", sessions={"chat-1", "chat-2"})
        first.readiness = _readiness(fable_calls=fable_calls, sol_calls=sol_calls)
        first.store.task_rows[("task-existing", 1)] = SimpleNamespace(
            session_id="chat-2"
        )
        second = _runtime("project-b")
        lease = ActiveAgentLease()
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((first, second)),
            lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = await workflows.prepare_new_request(
            project_id="project-a", session_id="chat-1", text="Build it", ids=_Ids(),
        )

        with pytest.raises(RuntimeError, match="another workflow"):
            await workflows.prepare_new_request(
                project_id="project-b", session_id="chat-1", text="Switch project", ids=_Ids(),
            )
        with pytest.raises(RuntimeError, match="another workflow"):
            await workflows.prepare_new_request(
                project_id="project-a", session_id="chat-2", text="New chat", ids=_Ids(),
            )
        with pytest.raises(RuntimeError, match="another workflow"):
            await workflows.prepare_existing_task(
                project_id="project-a",
                session_id="chat-2",
                task_id="task-existing",
                revision=1,
                action=action,
            )

        assert fable_calls == [None]
        assert sol_calls == [None]
        workflows.abort_prepared(prepared, reason="scheduler rejection")

    asyncio.run(exercise())


def test_running_a_prepared_request_releases_its_exact_lease_in_a_finally_block() -> None:
    async def exercise() -> None:
        runtime = _runtime("project-a")
        lease = ActiveAgentLease()
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)),
            lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = await workflows.prepare_new_request(
            project_id="project-a", session_id="chat-1", text="Build it", ids=_Ids(),
        )

        await workflows.run(prepared)

        assert runtime.coordinator.run_task_ids == ["task-1"]
        assert lease.snapshot() is None

    asyncio.run(exercise())


def test_abort_prepared_is_idempotent_and_stop_requires_the_exact_lease_owner() -> None:
    async def exercise() -> None:
        runtime = _runtime("project-a")
        lease = ActiveAgentLease()
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)),
            lease=lease,
            usage_credits_acknowledged=lambda: True,
        )
        prepared = await workflows.prepare_new_request(
            project_id="project-a", session_id="chat-1", text="Build it", ids=_Ids(),
        )

        with pytest.raises(RuntimeError, match="exact active workflow"):
            await workflows.stop(
                project_id="project-a", session_id="chat-1", task_id="other-task",
            )
        assert runtime.coordinator.stops == []

        await workflows.stop(
            project_id="project-a", session_id="chat-1", task_id="task-1",
        )
        assert runtime.coordinator.stops == ["task-1"]
        assert lease.snapshot() == prepared.token

        workflows.abort_prepared(prepared, reason="first")
        workflows.abort_prepared(prepared, reason="second")
        assert runtime.coordinator.aborts == [
            ("task-1", 0, "new_request", "scheduler_unavailable"),
        ]
        assert lease.snapshot() is None

    asyncio.run(exercise())
