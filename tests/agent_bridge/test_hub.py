from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
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
    PreparedWorkflow,
    StopReservation,
)
from agent_bridge.projects import ProjectSpec
from agent_bridge.state_machine import TaskState
from agent_bridge.store import (
    AnswerPayload,
    ApprovalPayload,
    NewRequestPayload,
    PreparedActionOutcome,
    PreparedActionRecord,
    ResumeDriftProjection,
    ResumePayload,
    SolResumeContext,
)


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
    prepared_rows: dict[str, PreparedActionRecord] | None = None

    def __post_init__(self) -> None:
        if self.task_rows is None:
            self.task_rows = {}
        if self.listener_tokens is None:
            self.listener_tokens = []
        if self.removed_listener_tokens is None:
            self.removed_listener_tokens = []
        if self.prepared_rows is None:
            self.prepared_rows = {}

    def session_exists(self, session_id: str) -> bool:
        return session_id in self.sessions

    def get_task(self, task_id: str, revision: int) -> object:
        return self.task_rows[(task_id, revision)]

    def prepared_action(self, preparation_id: str) -> PreparedActionRecord | None:
        return self.prepared_rows.get(preparation_id)

    def latest_prepared_action_for_task(
        self, *, project_id: str, session_id: str, task_id: str, revision: int,
    ) -> PreparedActionRecord | None:
        records = [
            record for record in self.prepared_rows.values()
            if (
                record.project_id == project_id
                and record.session_id == session_id
                and record.task_id == task_id
                and record.revision == revision
            )
        ]
        return None if not records else records[-1]

    def latest_task(self, task_id: str) -> object | None:
        rows = [
            value for (candidate, _), value in self.task_rows.items()
            if candidate == task_id
        ]
        return None if not rows else rows[-1]

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
    store: _RuntimeStore
    project_id: str
    prepared_actions: list[str] = field(default_factory=list)

    def _record(
        self, *, session_id: str, task_id: str, revision: int, action: str,
        generation: int,
    ) -> PreparedActionRecord:
        preparation_id = f"prepared-{task_id}-{generation}"
        continuation = SolResumeContext("thread-1", "run-1", "resume exactly")
        if action == "new_request":
            payload = NewRequestPayload(text="Build it")
            source = active = TaskState.FABLE_PLANNING
        elif action == "approval":
            payload = ApprovalPayload("baseline-1", None, None)
            source = TaskState.AWAITING_USER_APPROVAL
            active = TaskState.SOL_RUNNING
        elif action == "answer":
            payload = AnswerPayload("Use option A.", continuation)
            source = TaskState.AWAITING_USER_INPUT
            active = TaskState.SOL_RUNNING
        elif action == "resume":
            payload = ResumePayload(
                continuation,
                ResumeDriftProjection("unchanged", "Repository drift was checked.", ()),
            )
            source = TaskState.INTERRUPTED
            active = TaskState.SOL_RUNNING
        else:
            raise AssertionError("unexpected route")
        self.prepared_actions.append(action)
        pending_context = (
            None if action in {"new_request", "approval"} else continuation
        )
        record = PreparedActionRecord(
            preparation_id=preparation_id,
            project_id=self.project_id,
            session_id=session_id,
            task_id=task_id,
            revision=revision,
            action=action,  # type: ignore[arg-type]
            payload=payload,
            source_state=source,
            active_state=active,
            continuation_state=None,
            pending_context=pending_context,
            previous_preparation_id=None,
            status="PREPARED",
            reason=None,
            generation=generation,
        )
        self.store.prepared_rows[preparation_id] = record
        self.store.task_rows[(task_id, revision)] = SimpleNamespace(
            session_id=session_id, revision=revision, state=active,
        )
        return record

    def prepare_new_request(
        self, *, session_id: str, task_id: str, text: str, generation: int,
    ) -> PreparedActionRecord:
        self.prepared.append((session_id, text, task_id))
        return self._record(
            session_id=session_id, task_id=task_id, revision=0,
            action="new_request", generation=generation,
        )

    def prepare_user_request(self, session_id: str, text: str, task_id: str) -> object:
        self.prepared.append((session_id, text, task_id))
        return SimpleNamespace(task_id=task_id, session_id=session_id, revision=0)

    def prepare_approval(
        self, *, session_id: str, task_id: str, revision: int, generation: int,
    ) -> PreparedActionRecord:
        return self._record(
            session_id=session_id, task_id=task_id, revision=revision,
            action="approval", generation=generation,
        )

    def prepare_answer(
        self, *, session_id: str, task_id: str, revision: int, answer: str, generation: int,
    ) -> PreparedActionRecord:
        assert answer == "Use option A."
        return self._record(
            session_id=session_id, task_id=task_id, revision=revision,
            action="answer", generation=generation,
        )

    def prepare_resume(
        self, *, session_id: str, task_id: str, revision: int, generation: int,
    ) -> PreparedActionRecord:
        return self._record(
            session_id=session_id, task_id=task_id, revision=revision,
            action="resume", generation=generation,
        )

    async def run_prepared_request(self, task_id: str) -> None:
        self.run_task_ids.append(task_id)

    async def run_prepared_action(self, preparation_id: str) -> PreparedActionOutcome:
        record = self.store.prepared_action(preparation_id)
        assert record is not None
        self.run_task_ids.append(record.task_id)
        self.store.prepared_rows[preparation_id] = replace(record, status="COMPLETED")
        return PreparedActionOutcome("completed")

    def abort_prepared_action(
        self, preparation_id: str, *, generation: int, reason: str,
    ) -> object:
        record = self.store.prepared_action(preparation_id)
        assert record is not None
        self.aborts.append((record.task_id, record.revision, record.action, reason))
        self.store.prepared_rows[preparation_id] = replace(
            record, status="ABORTED", reason=reason,
        )
        return self.store.prepared_rows[preparation_id]

    def interrupt_claimed_prepared_action(
        self, preparation_id: str, *, generation: int, reason: str,
    ) -> object:
        record = self.store.prepared_action(preparation_id)
        assert record is not None
        self.store.prepared_rows[preparation_id] = replace(
            record, status="INTERRUPTED", reason=reason,
        )
        return self.store.prepared_rows[preparation_id]

    async def stop_task(self, task_id: str) -> None:
        self.stops.append(task_id)
        for key, task in self.store.task_rows.items():
            if key[0] == task_id:
                task.state = TaskState.INTERRUPTED


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
    store = _RuntimeStore(sessions or {"chat-1"})
    return _Runtime(
        project_id=project_id,
        label=project_id,
        repository=f"/repositories/{project_id}",
        branch="main",
        store=store,
        coordinator=_RuntimeCoordinator([], [], [], [], store, project_id),
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


def test_runtime_readiness_cancels_and_awaits_a_hanging_sibling_when_a_probe_fails() -> None:
    async def exercise() -> None:
        sibling_cancelled = asyncio.Event()

        async def fails() -> tuple[bool, str]:
            raise RuntimeError("controlled failure")

        async def hangs() -> str:
            try:
                await asyncio.Event().wait()
            finally:
                sibling_cancelled.set()

        readiness = RuntimeReadiness(
            initial=RuntimeStatus(False, "checking", "checking"),
            fable_probe=fails,
            sol_probe=hangs,
        )

        with pytest.raises(RuntimeError, match="not ready"):
            await readiness.require_model_start_ready(usage_credits_acknowledged=True)

        assert sibling_cancelled.is_set()

    asyncio.run(exercise())


@pytest.mark.parametrize("second_factory", ["raises", "non_awaitable"])
def test_runtime_readiness_cleans_up_the_first_probe_when_the_second_factory_fails(
    second_factory: str,
) -> None:
    async def exercise() -> None:
        started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        async def hangs() -> tuple[bool, str]:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                sibling_cancelled.set()

        active = asyncio.create_task(hangs())
        await started.wait()

        if second_factory == "raises":
            def invalid_second() -> str:
                raise RuntimeError("synchronous probe factory failure")
        else:
            def invalid_second() -> str:
                return "not-an-awaitable"

        readiness = RuntimeReadiness(
            initial=RuntimeStatus(False, "checking", "checking"),
            fable_probe=lambda: active,  # type: ignore[arg-type]
            sol_probe=invalid_second,  # type: ignore[arg-type]
        )

        with pytest.raises(RuntimeError, match="model runtime is not ready"):
            await readiness.require_model_start_ready(usage_credits_acknowledged=True)

        assert sibling_cancelled.is_set()

    asyncio.run(exercise())


def test_runtime_readiness_drains_active_probe_before_reraising_caller_cancellation() -> None:
    async def exercise() -> None:
        started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        async def hangs() -> tuple[bool, str]:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                sibling_cancelled.set()

        async def also_hangs() -> str:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        readiness = RuntimeReadiness(
            initial=RuntimeStatus(False, "checking", "checking"),
            fable_probe=hangs,
            sol_probe=also_hangs,
        )
        waiting = asyncio.create_task(
            readiness.require_model_start_ready(usage_credits_acknowledged=True)
        )
        await started.wait()
        waiting.cancel()

        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert sibling_cancelled.is_set()

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
            if action == "approval":
                await workflows.prepare_approval(
                    project_id="project-a", session_id="chat-2",
                    task_id="task-existing", revision=1,
                )
            elif action == "resume":
                await workflows.prepare_resume(
                    project_id="project-a", session_id="chat-2",
                    task_id="task-existing", revision=1,
                )
            else:
                await workflows.prepare_answer(
                    project_id="project-a", session_id="chat-2",
                    task_id="task-existing", revision=1, answer="Use option A.",
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


def test_hub_run_retains_the_exact_lease_when_terminal_persistence_fails() -> None:
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

        async def fail_before_terminal(preparation_id: str) -> PreparedActionOutcome:
            assert preparation_id == prepared.preparation_id
            raise RuntimeError("injected terminal persistence failure")

        runtime.coordinator.run_prepared_action = fail_before_terminal  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="terminal persistence"):
            await workflows.run(prepared)

        assert runtime.store.prepared_action(prepared.preparation_id).status == "PREPARED"  # type: ignore[union-attr]
        assert lease.snapshot() == prepared.token

    asyncio.run(exercise())


def test_hub_terminal_retry_keeps_the_lease_and_never_restarts_the_child() -> None:
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
        child_starts = 0
        terminal_attempts = 0

        async def retry_terminal_only(preparation_id: str) -> PreparedActionOutcome:
            nonlocal child_starts, terminal_attempts
            record = runtime.store.prepared_action(preparation_id)
            assert record is not None
            if record.status == "PREPARED":
                child_starts += 1
                runtime.store.prepared_rows[preparation_id] = replace(
                    record, status="CLAIMED",
                )
            terminal_attempts += 1
            if terminal_attempts == 1:
                raise RuntimeError("prepared action failed")
            claimed = runtime.store.prepared_action(preparation_id)
            assert claimed is not None and claimed.status == "CLAIMED"
            runtime.store.prepared_rows[preparation_id] = replace(
                claimed, status="COMPLETED",
            )
            return PreparedActionOutcome("completed")

        runtime.coordinator.run_prepared_action = retry_terminal_only  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="prepared action failed"):
            await workflows.run(prepared)

        claimed = runtime.store.prepared_action(prepared.preparation_id)
        assert claimed is not None and claimed.status == "CLAIMED"
        assert child_starts == 1
        assert lease.snapshot() == prepared.token

        await workflows.run(prepared)

        terminal = runtime.store.prepared_action(prepared.preparation_id)
        assert terminal is not None and terminal.status == "COMPLETED"
        assert child_starts == 1
        assert terminal_attempts == 2
        assert lease.snapshot() is None

    asyncio.run(exercise())


@pytest.mark.parametrize("route", ("approval", "answer", "resume"))
def test_hub_route_specific_preparations_run_the_matching_durable_action(
    route: str,
) -> None:
    async def exercise() -> None:
        runtime = _runtime("project-a")
        runtime.store.task_rows[("task-existing", 1)] = SimpleNamespace(
            session_id="chat-1", revision=1, state=TaskState.INTERRUPTED,
        )
        lease = ActiveAgentLease()
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)),
            lease=lease,
            usage_credits_acknowledged=lambda: True,
        )

        if route == "approval":
            prepared = await workflows.prepare_approval(
                project_id="project-a", session_id="chat-1",
                task_id="task-existing", revision=1,
            )
        elif route == "answer":
            prepared = await workflows.prepare_answer(
                project_id="project-a", session_id="chat-1",
                task_id="task-existing", revision=1, answer="Use option A.",
            )
        else:
            prepared = await workflows.prepare_resume(
                project_id="project-a", session_id="chat-1",
                task_id="task-existing", revision=1,
            )
        await workflows.run(prepared)

        assert runtime.coordinator.prepared_actions == [route]
        assert runtime.store.prepared_action(prepared.preparation_id).status == "COMPLETED"  # type: ignore[union-attr]
        assert lease.snapshot() is None

    asyncio.run(exercise())


@pytest.mark.parametrize("claimed", (False, True))
def test_hub_stop_interrupts_the_exact_prepared_action_before_or_after_claim(
    claimed: bool,
) -> None:
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
        if claimed:
            record = runtime.store.prepared_action(prepared.preparation_id)
            assert record is not None
            runtime.store.prepared_rows[prepared.preparation_id] = replace(
                record, status="CLAIMED",
            )

        reservation = workflows.reserve_stop(
            project_id="project-a", session_id="chat-1", task_id="task-1",
        )
        await workflows.stop(reservation=reservation)

        terminal = runtime.store.prepared_action(prepared.preparation_id)
        assert terminal is not None
        assert terminal.status == "INTERRUPTED"
        assert terminal.reason == "stop"
        assert runtime.coordinator.stops == ["task-1"]
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
        workflows.abort_prepared(prepared, reason="first")
        workflows.abort_prepared(prepared, reason="second")
        assert runtime.coordinator.aborts == [
            ("task-1", 0, "new_request", "scheduler_unavailable"),
        ]
        assert lease.snapshot() is None

        prepared = await workflows.prepare_new_request(
            project_id="project-a", session_id="chat-1", text="Build it", ids=_Ids(),
        )

        with pytest.raises(RuntimeError, match="exact active workflow"):
            workflows.reserve_stop(
                project_id="project-a", session_id="chat-1", task_id="other-task",
            )
        assert runtime.coordinator.stops == []

        reservation = workflows.reserve_stop(
            project_id="project-a", session_id="chat-1", task_id="task-1",
        )
        await workflows.stop(reservation=reservation)
        assert runtime.coordinator.stops == ["task-1"]
        assert lease.snapshot() is None

    asyncio.run(exercise())


def test_public_active_lease_guards_are_read_only_and_exact() -> None:
    """Removing route-side lease guards would allow foreign navigation work."""
    async def exercise() -> None:
        runtime = _runtime("project-a", sessions={"chat-1", "chat-2"})
        lease = ActiveAgentLease()
        workflows = HubWorkflowOrchestrator(
            registry=ProjectRegistry((runtime,)),
            lease=lease,
            usage_credits_acknowledged=lambda: True,
        )

        assert workflows.active_lease_snapshot() is None
        workflows.require_no_active_lease()
        prepared = await workflows.prepare_new_request(
            project_id="project-a", session_id="chat-1", text="Build it", ids=_Ids(),
        )

        assert workflows.active_lease_snapshot() == prepared.token
        with pytest.raises(RuntimeError, match="another workflow"):
            workflows.require_no_active_lease()
        assert workflows.require_navigation_allowed(
            project_id="project-a", session_id="chat-1",
        ) == prepared.token
        with pytest.raises(RuntimeError, match="active workflow"):
            workflows.require_navigation_allowed(
                project_id="project-a", session_id="chat-2",
            )
        with pytest.raises(RuntimeError, match="active workflow"):
            workflows.require_navigation_allowed(
                project_id="project-b", session_id="chat-1",
            )
        assert workflows.require_exact_stop_owner(
            project_id="project-a", session_id="chat-1", task_id="task-1",
        ) == prepared.token
        with pytest.raises(RuntimeError, match="exact active workflow"):
            workflows.require_exact_stop_owner(
                project_id="project-a", session_id="chat-1", task_id="other-task",
            )

        assert lease.snapshot() == prepared.token
        assert runtime.coordinator.stops == []

    asyncio.run(exercise())


def test_stop_releases_the_exact_lease_after_successful_finalization() -> None:
    """A successful Stop must not strand the global model-start lease."""
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

        reservation = workflows.reserve_stop(
            project_id="project-a", session_id="chat-1", task_id="task-1",
        )
        await workflows.stop(reservation=reservation)

        assert runtime.coordinator.stops == ["task-1"]
        assert workflows.active_lease_snapshot() is None

    asyncio.run(exercise())


def test_stop_reservation_prevents_same_identity_reacquire_until_exact_stop_finalizes() -> None:
    """Allowing release/acquire while reserved would replace generation one."""
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

        reservation = workflows.reserve_stop(
            project_id="project-a", session_id="chat-1", task_id="task-1",
        )
        lease.release(prepared.token)

        with pytest.raises(RuntimeError, match="exact active workflow"):
            workflows.reserve_stop(
                project_id="project-a",
                session_id="chat-1",
                task_id="task-1",
            )
        assert runtime.coordinator.stops == []
        assert lease.snapshot() == prepared.token

        with pytest.raises(RuntimeError, match="another workflow"):
            lease.acquire(
                project_id="project-a",
                session_id="chat-1",
                task_id="task-1",
            )
        assert runtime.coordinator.stops == []
        assert lease.snapshot() == prepared.token

        await workflows.stop(reservation=reservation)
        assert runtime.coordinator.stops == ["task-1"]
        assert lease.snapshot() is None

    asyncio.run(exercise())


def test_stop_rejects_a_reconstructed_reservation_claim() -> None:
    """Value-equal replacement claims must not gain authority to Stop."""
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
        reservation = workflows.reserve_stop(
            project_id="project-a", session_id="chat-1", task_id="task-1",
        )
        reconstructed = StopReservation(
            reservation._token, reservation._claim_id,
        )

        with pytest.raises(RuntimeError, match="exact active workflow"):
            await workflows.stop(reservation=reconstructed)
        assert runtime.coordinator.stops == []
        assert lease.snapshot() == prepared.token

        await workflows.stop(reservation=reservation)
        assert runtime.coordinator.stops == ["task-1"]
        assert lease.snapshot() is None

    asyncio.run(exercise())


def test_owner_release_during_stop_reservation_is_applied_after_stop_cancellation() -> None:
    """Dropping an ordinary release during a reservation would strand the lease."""
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
        entered = asyncio.Event()
        release = asyncio.Event()
        async def pause_stop(task_id: str) -> None:
            entered.set()
            await release.wait()
            raise RuntimeError("injected Stop failure")

        runtime.coordinator.stop_task = pause_stop  # type: ignore[method-assign]
        reservation = workflows.reserve_stop(
            project_id="project-a", session_id="chat-1", task_id="task-1",
        )
        stopping = asyncio.create_task(workflows.stop(reservation=reservation))
        await entered.wait()
        lease.release(prepared.token)
        assert lease.snapshot() == prepared.token
        release.set()
        with pytest.raises(RuntimeError, match="injected Stop failure"):
            await stopping
        assert runtime.coordinator.stops == []
        assert lease.snapshot() is None

    asyncio.run(exercise())


def test_cancelling_reserved_stop_keeps_its_still_running_owner_lease() -> None:
    """Cancelling Stop must not silently release a workflow that did not end."""
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
        entered = asyncio.Event()

        async def block_stop(task_id: str) -> None:
            entered.set()
            await asyncio.Future()

        runtime.coordinator.stop_task = block_stop  # type: ignore[method-assign]
        reservation = workflows.reserve_stop(
            project_id="project-a", session_id="chat-1", task_id="task-1",
        )
        stopping = asyncio.create_task(workflows.stop(reservation=reservation))
        await entered.wait()
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        assert lease.snapshot() == prepared.token
        lease.release(prepared.token)
        assert lease.snapshot() is None

    asyncio.run(exercise())


def test_abort_persistence_failure_keeps_the_exact_lease_retryable() -> None:
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
        original_abort = runtime.coordinator.abort_prepared_action

        def fail_abort(*args: object, **kwargs: object) -> object:
            raise RuntimeError("injected persistence failure")

        runtime.coordinator.abort_prepared_action = fail_abort  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="persistence failure"):
            workflows.abort_prepared(prepared, reason="scheduler rejected")
        assert lease.snapshot() == prepared.token

        runtime.coordinator.abort_prepared_action = original_abort  # type: ignore[method-assign]
        workflows.abort_prepared(prepared, reason="scheduler rejected")
        assert lease.snapshot() is None

    asyncio.run(exercise())


def test_forged_preparation_identifier_cannot_claim_abort_or_release_the_real_lease() -> None:
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
        forged = PreparedWorkflow(
            preparation_id="forged-preparation", token=prepared.token,
        )

        with pytest.raises(RuntimeError, match="not found"):
            await workflows.run(forged)
        assert lease.snapshot() == prepared.token
        with pytest.raises(RuntimeError, match="not found"):
            workflows.abort_prepared(forged, reason="scheduler rejected")
        assert lease.snapshot() == prepared.token

        workflows.abort_prepared(prepared, reason="scheduler rejected")
        assert lease.snapshot() is None

    asyncio.run(exercise())
