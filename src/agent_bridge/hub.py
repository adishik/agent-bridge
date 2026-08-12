"""Project-owned runtimes and the single in-process model-start lease."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
import fcntl
import os
from pathlib import Path
import threading
from typing import Protocol

from agent_bridge.adapters.base import FableAdapter, SolAdapter
from agent_bridge.app import EventBroadcaster
from agent_bridge.coordinator import Coordinator, IdFactory
from agent_bridge.process import ProcessRunner
from agent_bridge.projects import ProjectSpec
from agent_bridge.repository import RepositoryTracker
from agent_bridge.store import PreparedActionRecord, SQLiteStore


_FABLE_STATUSES = frozenset({
    "checking", "subscription_ready", "subscription_unavailable",
})
_SOL_STATUSES = frozenset({"checking", "ready", "running", "blocked", "unavailable"})
_SCHEDULER_UNAVAILABLE = "scheduler_unavailable"
_TERMINAL_PREPARED_STATUSES = frozenset({
    "COMPLETED", "FAILED", "ABORTED", "INTERRUPTED", "RECOVERED",
})


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """The current bounded, non-secret provider availability projection."""

    fable_ready: bool
    fable_status: str
    sol_status: str

    def __post_init__(self) -> None:
        if not isinstance(self.fable_ready, bool):
            raise ValueError("fable_ready must be a bool")
        if self.fable_status not in _FABLE_STATUSES:
            raise ValueError("fable_status is invalid")
        if self.fable_ready != (self.fable_status == "subscription_ready"):
            raise ValueError("fable_ready must match fable_status")
        if self.sol_status not in _SOL_STATUSES:
            raise ValueError("sol_status is invalid")


class RuntimeReadiness:
    """Refresh both provider states immediately before a model may start."""

    def __init__(
        self,
        *,
        initial: RuntimeStatus,
        fable_probe: Callable[[], Awaitable[tuple[bool, str]]],
        sol_probe: Callable[[], Awaitable[str]],
        timeout_seconds: float = 10.0,
    ) -> None:
        if not isinstance(initial, RuntimeStatus):
            raise ValueError("initial must be a RuntimeStatus")
        if not callable(fable_probe) or not callable(sol_probe):
            raise ValueError("provider probes must be callable")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive number")
        self._status = initial
        self._fable_probe = fable_probe
        self._sol_probe = sol_probe
        self._timeout_seconds = float(timeout_seconds)
        self._refresh_lock = asyncio.Lock()

    def snapshot(self) -> RuntimeStatus:
        return self._status

    async def require_model_start_ready(
        self, *, usage_credits_acknowledged: bool,
    ) -> RuntimeStatus:
        if usage_credits_acknowledged is not True:
            raise PermissionError("usage credits must be acknowledged")
        async with self._refresh_lock:
            probes: list[asyncio.Future[object]] = []
            try:
                probes.append(asyncio.ensure_future(self._fable_probe()))
                probes.append(asyncio.ensure_future(self._sol_probe()))
                fable_result, sol_status = await asyncio.wait_for(
                    asyncio.gather(*probes),
                    timeout=self._timeout_seconds,
                )
                if (
                    not isinstance(fable_result, tuple)
                    or len(fable_result) != 2
                    or not isinstance(fable_result[0], bool)
                    or not isinstance(fable_result[1], str)
                    or not isinstance(sol_status, str)
                ):
                    raise ValueError("provider probes returned an invalid status")
                status = RuntimeStatus(fable_result[0], fable_result[1], sol_status)
            except BaseException as error:
                for probe in probes:
                    if not probe.done():
                        probe.cancel()
                if probes:
                    await asyncio.gather(*probes, return_exceptions=True)
                self._status = RuntimeStatus(
                    False, "subscription_unavailable", "unavailable"
                )
                if isinstance(error, asyncio.CancelledError):
                    raise
                raise RuntimeError("model runtime is not ready") from error
            self._status = status
            if not status.fable_ready or status.sol_status != "ready":
                raise RuntimeError("model runtime is not ready")
            return status

    def invalidate_fable_subscription(self) -> None:
        self._status = RuntimeStatus(
            False, "subscription_unavailable", self._status.sol_status
        )


@dataclass(slots=True)
class InstanceLock:
    """One lock descriptor acquired by launcher startup and owned by a runtime."""

    path: Path
    descriptor: int
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)
            self.released = True


class AppProjectRuntime(Protocol):
    project_id: str
    label: str
    repository: str
    branch: str
    store: SQLiteStore
    coordinator: Coordinator
    broadcaster: EventBroadcaster
    readiness: RuntimeReadiness


@dataclass(slots=True)
class OwnedProjectRuntime:
    """A runtime whose listeners and local resources are owned by this hub."""

    spec: ProjectSpec
    store: SQLiteStore
    tracker: RepositoryTracker
    runner: ProcessRunner
    fable: FableAdapter
    sol: SolAdapter
    coordinator: Coordinator
    broadcaster: EventBroadcaster
    readiness: RuntimeReadiness
    lock: InstanceLock
    _event_listener_token: int | None = field(init=False, default=None, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._event_listener_token = self.store.add_event_listener(
            self.broadcaster.publish
        )

    @property
    def project_id(self) -> str:
        return self.spec.project_id

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def repository(self) -> str:
        return str(self.spec.repo_root)

    @property
    def branch(self) -> str:
        return self.spec.branch

    def close(self) -> None:
        """Release owned resources once, continuing cleanup after a close error."""
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []

        def release(call: Callable[[], None]) -> None:
            try:
                call()
            except BaseException as error:
                errors.append(error)

        if self._event_listener_token is not None:
            release(lambda: self.store.remove_event_listener(self._event_listener_token))
            self._event_listener_token = None
        release(self.coordinator.close)
        release(self.tracker.close)
        release(self.store.close)
        release(self.lock.release)
        if errors:
            raise errors[0]


@dataclass(frozen=True, slots=True)
class LeaseToken:
    generation: int
    project_id: str
    session_id: str
    task_id: str


class ActiveAgentLease:
    """One process-wide lease whose generation prevents stale release races."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._active: LeaseToken | None = None

    def acquire_new(
        self, *, project_id: str, session_id: str, ids: IdFactory,
    ) -> LeaseToken:
        self._require_id(project_id, "project_id")
        self._require_id(session_id, "session_id")
        if not callable(getattr(ids, "new_task_id", None)):
            raise ValueError("ids must create task identifiers")
        with self._lock:
            self._require_available()
            task_id = ids.new_task_id()
            self._require_id(task_id, "task_id")
            return self._activate(project_id, session_id, task_id)

    def acquire(
        self, *, project_id: str, session_id: str, task_id: str,
    ) -> LeaseToken:
        self._require_id(project_id, "project_id")
        self._require_id(session_id, "session_id")
        self._require_id(task_id, "task_id")
        with self._lock:
            self._require_available()
            return self._activate(project_id, session_id, task_id)

    def release(self, token: LeaseToken) -> None:
        if not isinstance(token, LeaseToken):
            raise ValueError("token must be a LeaseToken")
        with self._lock:
            if self._active == token:
                self._active = None

    def snapshot(self) -> LeaseToken | None:
        with self._lock:
            return self._active

    def _activate(self, project_id: str, session_id: str, task_id: str) -> LeaseToken:
        self._generation += 1
        token = LeaseToken(self._generation, project_id, session_id, task_id)
        self._active = token
        return token

    def _require_available(self) -> None:
        if self._active is not None:
            raise RuntimeError("another workflow already owns the active agent lease")

    @staticmethod
    def _require_id(value: object, name: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")


class ProjectRegistry:
    """Stable project lookup without cross-project persistence access."""

    def __init__(self, runtimes: Iterable[AppProjectRuntime]) -> None:
        provided = tuple(runtimes)
        by_id: dict[str, AppProjectRuntime] = {}
        for runtime in provided:
            project_id = getattr(runtime, "project_id", None)
            if not isinstance(project_id, str) or not project_id:
                raise ValueError("runtime project_id must be a non-empty string")
            if project_id in by_id:
                raise ValueError("duplicate project id")
            by_id[project_id] = runtime
        self._runtimes = tuple(by_id[key] for key in sorted(by_id))
        self._by_id = by_id
        self._closed = False

    def runtime(self, project_id: str) -> AppProjectRuntime:
        try:
            return self._by_id[project_id]
        except (KeyError, TypeError) as error:
            raise LookupError("project not found") from error

    def projects(self) -> tuple[AppProjectRuntime, ...]:
        return self._runtimes

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for runtime in reversed(self._runtimes):
            close = getattr(runtime, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]


@dataclass(frozen=True, slots=True)
class PreparedWorkflow:
    preparation_id: str
    token: LeaseToken
    revision: int = 0


class HubWorkflowOrchestrator:
    """The sole owner of leases across preparation and model execution."""

    def __init__(
        self,
        *,
        registry: ProjectRegistry,
        lease: ActiveAgentLease,
        usage_credits_acknowledged: Callable[[], bool],
    ) -> None:
        if not isinstance(registry, ProjectRegistry):
            raise ValueError("registry must be a ProjectRegistry")
        if not isinstance(lease, ActiveAgentLease):
            raise ValueError("lease must be an ActiveAgentLease")
        if not callable(usage_credits_acknowledged):
            raise ValueError("usage_credits_acknowledged must be callable")
        self._registry = registry
        self._lease = lease
        self._usage_credits_acknowledged = usage_credits_acknowledged
        self._aborted_tokens: set[LeaseToken] = set()

    def active_lease_snapshot(self) -> LeaseToken | None:
        """Return the current lease identity without exposing its owner."""
        return self._lease.snapshot()

    def require_no_active_lease(self) -> None:
        """Fail before a navigation or mutation may disturb an active workflow."""
        if self._lease.snapshot() is not None:
            raise RuntimeError("another workflow already owns the active agent lease")

    def require_navigation_allowed(
        self, *, project_id: str, session_id: str,
    ) -> LeaseToken | None:
        """Allow an idle browser or the exact browser owning the active task."""
        self._require_non_empty(project_id, "project_id")
        self._require_non_empty(session_id, "session_id")
        token = self._lease.snapshot()
        if token is not None and (
            token.project_id != project_id or token.session_id != session_id
        ):
            raise RuntimeError("active workflow belongs to another project or chat")
        return token

    def require_exact_stop_owner(
        self, *, project_id: str, session_id: str, task_id: str,
    ) -> LeaseToken:
        """Validate Stop at the HTTP boundary before it can schedule work."""
        self._require_non_empty(project_id, "project_id")
        self._require_non_empty(session_id, "session_id")
        self._require_non_empty(task_id, "task_id")
        token = self._lease.snapshot()
        if (
            token is None
            or token.project_id != project_id
            or token.session_id != session_id
            or token.task_id != task_id
        ):
            raise RuntimeError("stop requires the exact active workflow")
        return token

    async def prepare_new_request(
        self,
        *,
        project_id: str,
        session_id: str,
        text: str,
        ids: IdFactory,
    ) -> PreparedWorkflow:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        runtime = self._runtime_for_session(project_id, session_id)
        self._require_acknowledgement()
        token = self._lease.acquire_new(
            project_id=project_id, session_id=session_id, ids=ids,
        )
        try:
            await runtime.readiness.require_model_start_ready(
                usage_credits_acknowledged=True
            )
            record = runtime.coordinator.prepare_new_request(
                session_id=session_id,
                task_id=token.task_id,
                text=text,
                generation=token.generation,
            )
            record = self._bound_record(runtime, record.preparation_id, token)
            return PreparedWorkflow(
                preparation_id=record.preparation_id,
                token=token,
                revision=record.revision,
            )
        except BaseException:
            self._lease.release(token)
            raise

    async def prepare_approval(
        self, *, project_id: str, session_id: str, task_id: str, revision: int,
    ) -> PreparedWorkflow:
        return await self._prepare_existing(
            project_id=project_id, session_id=session_id, task_id=task_id,
            revision=revision,
            prepare=lambda runtime, token: runtime.coordinator.prepare_approval(
                session_id=session_id, task_id=task_id, revision=revision,
                generation=token.generation,
            ),
        )

    async def prepare_answer(
        self,
        *,
        project_id: str,
        session_id: str,
        task_id: str,
        revision: int,
        answer: str,
    ) -> PreparedWorkflow:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be non-empty")
        return await self._prepare_existing(
            project_id=project_id, session_id=session_id, task_id=task_id,
            revision=revision,
            prepare=lambda runtime, token: runtime.coordinator.prepare_answer(
                session_id=session_id, task_id=task_id, revision=revision,
                answer=answer, generation=token.generation,
            ),
        )

    async def prepare_resume(
        self, *, project_id: str, session_id: str, task_id: str, revision: int,
    ) -> PreparedWorkflow:
        return await self._prepare_existing(
            project_id=project_id, session_id=session_id, task_id=task_id,
            revision=revision,
            prepare=lambda runtime, token: runtime.coordinator.prepare_resume(
                session_id=session_id, task_id=task_id, revision=revision,
                generation=token.generation,
            ),
        )

    async def run(self, prepared: PreparedWorkflow) -> None:
        if not isinstance(prepared, PreparedWorkflow):
            raise ValueError("prepared must be a PreparedWorkflow")
        if self._lease.snapshot() != prepared.token:
            raise RuntimeError("prepared workflow no longer owns the active lease")
        runtime = self._registry.runtime(prepared.token.project_id)
        self._bound_record(runtime, prepared.preparation_id, prepared.token)
        try:
            await runtime.coordinator.run_prepared_action(prepared.preparation_id)
        except BaseException:
            record = self._bound_record(runtime, prepared.preparation_id, prepared.token)
            if record.status in _TERMINAL_PREPARED_STATUSES:
                self._lease.release(prepared.token)
            raise
        record = self._bound_record(runtime, prepared.preparation_id, prepared.token)
        if record.status not in _TERMINAL_PREPARED_STATUSES:
            raise RuntimeError("prepared action did not reach a terminal state")
        else:
            self._lease.release(prepared.token)

    def abort_prepared(self, prepared: PreparedWorkflow, *, reason: str) -> None:
        if not isinstance(prepared, PreparedWorkflow):
            raise ValueError("prepared must be a PreparedWorkflow")
        if prepared.token in self._aborted_tokens:
            return
        if self._lease.snapshot() != prepared.token:
            raise RuntimeError("prepared workflow no longer owns the active lease")
        runtime = self._registry.runtime(prepared.token.project_id)
        self._bound_record(runtime, prepared.preparation_id, prepared.token)
        runtime.coordinator.abort_prepared_action(
            prepared.preparation_id,
            generation=prepared.token.generation,
            reason=_SCHEDULER_UNAVAILABLE,
        )
        self._aborted_tokens.add(prepared.token)
        self._lease.release(prepared.token)

    async def stop(
        self, *, project_id: str, session_id: str, task_id: str,
    ) -> None:
        token = self.require_exact_stop_owner(
            project_id=project_id, session_id=session_id, task_id=task_id,
        )
        runtime = self._runtime_for_session(project_id, session_id)
        stopping = asyncio.create_task(runtime.coordinator.stop_task(task_id))
        try:
            # ``stop_task`` persists the interrupted task before it waits for
            # the child.  Let it reach that durable boundary, then record the
            # exact claim as stopped before the child completion can look like
            # a normal prepared-action completion.
            await asyncio.sleep(0)
            latest = runtime.store.latest_task(task_id)
            if latest is not None:
                record = runtime.store.latest_prepared_action_for_task(
                    project_id=project_id,
                    session_id=session_id,
                    task_id=task_id,
                    revision=latest.revision,
                )
                if (
                    record is not None
                    and record.status in {"PREPARED", "CLAIMED"}
                    and record.generation == token.generation
                ):
                    runtime.coordinator.interrupt_claimed_prepared_action(
                        record.preparation_id, generation=token.generation, reason="stop",
                    )
            await stopping
            self._lease.release(token)
        except BaseException:
            if not stopping.done():
                stopping.cancel()
                await asyncio.gather(stopping, return_exceptions=True)
            raise

    async def _prepare_existing(
        self,
        *,
        project_id: str,
        session_id: str,
        task_id: str,
        revision: int,
        prepare: Callable[[AppProjectRuntime, LeaseToken], PreparedActionRecord],
    ) -> PreparedWorkflow:
        self._require_non_empty(task_id, "task_id")
        self._require_non_negative_integer(revision, "revision")
        runtime = self._runtime_for_session(project_id, session_id)
        task = runtime.store.get_task(task_id, revision)
        if task.session_id != session_id:
            raise LookupError("task not found")
        self._require_acknowledgement()
        token = self._lease.acquire(
            project_id=project_id, session_id=session_id, task_id=task_id,
        )
        try:
            await runtime.readiness.require_model_start_ready(
                usage_credits_acknowledged=True
            )
            record = prepare(runtime, token)
            record = self._bound_record(runtime, record.preparation_id, token)
            return PreparedWorkflow(
                preparation_id=record.preparation_id,
                token=token,
                revision=record.revision,
            )
        except BaseException:
            self._lease.release(token)
            raise

    @staticmethod
    def _bound_record(
        runtime: AppProjectRuntime, preparation_id: str, token: LeaseToken,
    ) -> PreparedActionRecord:
        record = runtime.store.prepared_action(preparation_id)
        if record is None:
            raise RuntimeError("prepared action not found")
        if (
            record.project_id != token.project_id
            or record.session_id != token.session_id
            or record.task_id != token.task_id
            or record.generation != token.generation
        ):
            raise RuntimeError("prepared action does not match the active lease")
        return record

    def _runtime_for_session(
        self, project_id: str, session_id: str,
    ) -> AppProjectRuntime:
        self._require_non_empty(project_id, "project_id")
        self._require_non_empty(session_id, "session_id")
        runtime = self._registry.runtime(project_id)
        if not runtime.store.session_exists(session_id):
            raise LookupError("chat not found")
        return runtime

    def _require_acknowledgement(self) -> None:
        if self._usage_credits_acknowledged() is not True:
            raise PermissionError("usage credits must be acknowledged")

    @staticmethod
    def _require_non_empty(value: object, name: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")

    @staticmethod
    def _require_non_negative_integer(value: object, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
