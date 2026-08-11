"""Loopback browser API for the local Fable--Sol bridge.

The web boundary deliberately accepts task actions, not repository roots or
process identifiers.  Authentication is a keyed, HttpOnly browser session;
every HTTP mutation additionally requires a CSRF token obtained from the
authenticated bootstrap endpoint.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
import re
import secrets
import threading
from typing import Annotated, Any, Literal, Protocol

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Path as PathParameter,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_bridge.contracts import StreamEvent, TaskBrief
from agent_bridge.coordinator import Coordinator
from agent_bridge.store import (
    EVENT_REPLAY_PAGE_SIZE,
    SQLiteStore,
    TaskOverview,
    TaskRecord,
)


SESSION_COOKIE = "agent_bridge_session"
USAGE_CREDITS_SETTING = "usage_credits_acknowledged"
_SAFE_ID_BODY = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}"
_SAFE_ID_PATTERN = f"^{_SAFE_ID_BODY}$"
_SAFE_ID = re.compile(f"{_SAFE_ID_BODY}\\Z")


class EventSubscription(Protocol):
    async def __aenter__(self) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool | None:
        raise NotImplementedError


class EventBroadcaster(Protocol):
    def subscribe(self, session_id: str) -> EventSubscription:
        raise NotImplementedError

    def publish(self, event: StreamEvent) -> None:
        raise NotImplementedError


class _MemorySubscription:
    def __init__(
        self,
        broadcaster: "InMemoryEventBroadcaster",
        session_id: str,
    ) -> None:
        self._broadcaster = broadcaster
        self._session_id = session_id
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue(
            maxsize=broadcaster.max_queue_size
        )
        self._active = True
        self._wake_pending = False
        broadcaster._add(self)

    async def __aenter__(self) -> AsyncIterator[StreamEvent]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        self._broadcaster._remove(self)

    def __aiter__(self) -> "_MemorySubscription":
        return self

    async def __anext__(self) -> StreamEvent:
        item = await self._queue.get()
        self._broadcaster._consume_wake(self)
        if not isinstance(item, StreamEvent):
            raise RuntimeError("invalid broadcaster queue item")
        return item


class InMemoryEventBroadcaster:
    """Small synchronous publisher with per-session async subscriptions."""

    def __init__(self, *, max_queue_size: int = 256) -> None:
        if (
            not isinstance(max_queue_size, int)
            or isinstance(max_queue_size, bool)
            or max_queue_size < 1
        ):
            raise ValueError("max_queue_size must be a positive integer")
        self.max_queue_size = max_queue_size
        self._lock = threading.RLock()
        self._subscribers: dict[str, set[_MemorySubscription]] = {}

    def subscribe(self, session_id: str) -> _MemorySubscription:
        _require_safe_id(session_id, "session_id")
        return _MemorySubscription(self, session_id)

    def publish(self, event: StreamEvent) -> None:
        if not isinstance(event, StreamEvent):
            raise ValueError("event must be a StreamEvent")
        with self._lock:
            subscribers = tuple(self._subscribers.get(event.session_id, ()))
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        for subscriber in subscribers:
            with self._lock:
                if not subscriber._active or subscriber._wake_pending:
                    continue
                if subscriber._loop.is_closed():
                    self._remove(subscriber)
                    continue
                subscriber._wake_pending = True
            if current_loop is subscriber._loop:
                self._deliver_wake(subscriber, event)
                continue
            try:
                subscriber._loop.call_soon_threadsafe(
                    self._deliver_wake, subscriber, event
                )
            except RuntimeError:
                self._remove(subscriber)

    def _deliver_wake(
        self, subscriber: _MemorySubscription, event: StreamEvent
    ) -> None:
        with self._lock:
            if not subscriber._active:
                subscriber._wake_pending = False
                return
            if subscriber._queue.empty():
                subscriber._queue.put_nowait(event)

    def _consume_wake(self, subscriber: _MemorySubscription) -> None:
        with self._lock:
            subscriber._wake_pending = False

    def _add(self, subscriber: _MemorySubscription) -> None:
        with self._lock:
            self._subscribers.setdefault(subscriber._session_id, set()).add(
                subscriber
            )

    def _remove(self, subscriber: _MemorySubscription) -> None:
        with self._lock:
            subscriber._active = False
            subscriber._wake_pending = False
            subscribers = self._subscribers.get(subscriber._session_id)
            if subscribers is None:
                return
            subscribers.discard(subscriber)
            if not subscribers:
                self._subscribers.pop(subscriber._session_id, None)


class _StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class MessageRequest(_StrictRequest):
    text: str

    @field_validator("text")
    @classmethod
    def _text_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must be non-empty")
        return value


class AnswerRequest(_StrictRequest):
    answer: str

    @field_validator("answer")
    @classmethod
    def _answer_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must be non-empty")
        return value


class ApprovalRequest(_StrictRequest):
    revision: int = Field(ge=1)


class UsageCreditsAcknowledgementRequest(_StrictRequest):
    acknowledged: Literal[True]


class TaskBriefRequest(_StrictRequest):
    task_id: str
    revision: int
    title: str
    objective: str
    context: list[str]
    constraints: list[str]
    allowed_paths: list[str]
    out_of_scope: list[str]
    acceptance_criteria: list[str]
    required_tests: list[str]
    risks: list[str]
    open_questions: list[str]
    confidence: float
    confidence_rationale: str


TaskId = Annotated[
    str,
    PathParameter(min_length=1, max_length=128, pattern=_SAFE_ID_PATTERN),
]
SessionId = Annotated[
    str,
    PathParameter(min_length=1, max_length=128, pattern=_SAFE_ID_PATTERN),
]
SocketSessionId = Annotated[
    str,
    Query(min_length=1, max_length=128, pattern=_SAFE_ID_PATTERN),
]
ReplayCursor = Annotated[int, Query(ge=0, le=2**53 - 1)]


_FABLE_STATUSES = frozenset({
    "checking",
    "subscription_ready",
    "subscription_unavailable",
})
_SOL_STATUSES = frozenset({"checking", "ready", "running", "blocked", "unavailable"})


@dataclass(frozen=True)
class BootstrapStatus:
    """Injected, non-secret startup status exposed by the browser bootstrap."""

    session_id: str | None = None
    fable_ready: bool = False
    fable_status: str = "checking"
    sol_status: str = "checking"
    repository: str | None = None
    branch: str | None = None

    def __post_init__(self) -> None:
        if self.session_id is not None:
            _require_safe_id(self.session_id, "session_id")
        if not isinstance(self.fable_ready, bool):
            raise ValueError("fable_ready must be a bool")
        if self.fable_status not in _FABLE_STATUSES:
            raise ValueError("fable_status must be checking, subscription_ready, or subscription_unavailable")
        if self.fable_ready != (self.fable_status == "subscription_ready"):
            raise ValueError("fable_ready requires exact subscription_ready status")
        if self.sol_status not in _SOL_STATUSES:
            raise ValueError("sol_status is invalid")
        for name in ("repository", "branch"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be null or a non-empty string")


def _require_safe_id(value: str, name: str) -> None:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe identifier")


def _ascii_token(value: object, name: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty ASCII string")
    try:
        return value.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(f"{name} must be a non-empty ASCII string") from None


def _token_matches(candidate: str | None, expected: bytes) -> bool:
    if not isinstance(candidate, str):
        return False
    try:
        encoded = candidate.encode("ascii")
    except UnicodeEncodeError:
        return False
    return secrets.compare_digest(encoded, expected)


def create_app(
    *,
    coordinator: Coordinator,
    store: SQLiteStore,
    static_dir: str | Path,
    session_key: str,
    csrf_token: str,
    broadcaster: EventBroadcaster | None = None,
    bootstrap_status: Callable[[], BootstrapStatus] | None = None,
) -> FastAPI:
    """Create the authenticated loopback web application.

    Host binding is owned by the foreground launcher.  This factory has no
    browser-controlled repository, executable, process, or credential input.
    """

    session_key_bytes = _ascii_token(session_key, "session_key")
    csrf_token_bytes = _ascii_token(csrf_token, "csrf_token")
    static_root = Path(static_dir).resolve()
    index_path = static_root / "index.html"
    if not static_root.is_dir() or not index_path.is_file():
        raise ValueError("static_dir must contain index.html")
    event_broadcaster = broadcaster or InMemoryEventBroadcaster()
    status_provider = bootstrap_status or BootstrapStatus

    def thaw_projection(value: object) -> object:
        if isinstance(value, Mapping):
            return {key: thaw_projection(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [thaw_projection(item) for item in value]
        return value

    def task_snapshot(overview: TaskOverview) -> Mapping[str, object]:
        task = overview.task
        return {
            "task_id": task.task_id,
            "revision": task.revision,
            "state": task.state.value,
            "brief": None if task.brief is None else task.brief.to_dict(),
            "approved_at": task.approved_at,
            "correction_count": task.correction_count,
            "continuation_state": (
                None if task.continuation_state is None else task.continuation_state.value
            ),
            "updated_at": overview.updated_at,
            "active_agent": overview.active_agent,
            "active_started_at": overview.active_started_at,
            "revision_start_sequence": overview.revision_start_sequence,
            "outcome": thaw_projection(overview.outcome),
            "review": thaw_projection(overview.review),
            "clarification": thaw_projection(overview.clarification),
            "activity": thaw_projection(overview.activity),
        }

    def forward_committed_event(event: StreamEvent) -> None:
        if app.state.shutting_down:
            return
        try:
            event_broadcaster.publish(event)
        except BaseException as error:
            app.state.coroutine_observation_failures.append(
                {
                    "stage": "broadcast",
                    "error_type": type(error).__name__,
                }
            )

    @asynccontextmanager
    async def lifespan(lifespan_app: FastAPI) -> AsyncIterator[None]:
        lifespan_app.state.shutting_down = False
        listener_token = store.add_event_listener(forward_committed_event)
        try:
            yield
        finally:
            lifespan_app.state.shutting_down = True
            store.remove_event_listener(listener_token)
            active = tuple(lifespan_app.state.active_coroutines)
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
                await asyncio.sleep(0)
                lifespan_app.state.active_coroutines.difference_update(active)

    app = FastAPI(title="Agent Bridge", version="0.1.0", lifespan=lifespan)
    app.state.active_coroutines = set()
    app.state.coroutine_observation_failures = []
    app.state.event_broadcaster = event_broadcaster
    app.state.shutting_down = False

    def require_session(request: Request) -> None:
        if not _token_matches(
            request.cookies.get(SESSION_COOKIE), session_key_bytes
        ):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")

    def require_mutation(
        request: Request,
        x_csrf_token: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
    ) -> None:
        require_session(request)
        if request.query_params:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="mutation query parameters are not accepted",
            )
        if not _token_matches(x_csrf_token, csrf_token_bytes):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")

    def latest_task(task_id: str) -> TaskRecord:
        task = store.latest_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="task not found",
            )
        return task

    def require_known_session(session_id: str) -> None:
        if not store.session_exists(session_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="session not found",
            )

    def current_bootstrap_status() -> BootstrapStatus:
        runtime_status = status_provider()
        if not isinstance(runtime_status, BootstrapStatus):
            raise RuntimeError("bootstrap status provider returned an invalid value")
        return runtime_status

    def require_model_start_ready(session_id: str) -> None:
        runtime_status = current_bootstrap_status()
        if (
            store.get_setting(USAGE_CREDITS_SETTING) is not True
            or runtime_status.session_id != session_id
            or runtime_status.fable_ready is not True
            or runtime_status.fable_status != "subscription_ready"
            or runtime_status.sol_status != "ready"
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "model actions require Fable subscription readiness, "
                    "Sol readiness, and usage-credit acknowledgement "
                    "for this session"
                ),
            )

    def observe_coroutine(
        task: asyncio.Task[object],
        *,
        session_id: str,
        task_id: str | None,
        action: str,
    ) -> None:
        app.state.active_coroutines.discard(task)
        error: BaseException | None
        if task.cancelled() and app.state.shutting_down:
            return
        if task.cancelled():
            error = asyncio.CancelledError()
        else:
            try:
                error = task.exception()
            except BaseException as observation_error:
                error = observation_error
        if error is None:
            return
        try:
            store.append_event(
                session_id,
                task_id,
                "coordinator",
                "action_error",
                {
                    "action": action,
                    "error_type": type(error).__name__,
                },
            )
        except BaseException as persistence_error:
            app.state.coroutine_observation_failures.append(
                {
                    "stage": "persistence",
                    "action": action,
                    "error_type": type(persistence_error).__name__,
                }
            )
            return

    def schedule(
        coroutine: Coroutine[Any, Any, object],
        *,
        session_id: str,
        task_id: str | None,
        action: str,
    ) -> None:
        if app.state.shutting_down:
            coroutine.close()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="bridge is shutting down",
            )
        task = asyncio.create_task(coroutine, name=f"agent-bridge-{action}")
        app.state.active_coroutines.add(task)
        task.add_done_callback(
            lambda completed: observe_coroutine(
                completed,
                session_id=session_id,
                task_id=task_id,
                action=action,
            )
        )

    @app.get("/healthz")
    async def healthz() -> Mapping[str, str]:
        return {"status": "ok"}

    @app.get("/")
    async def index(
        request: Request,
        key: Annotated[str | None, Query()] = None,
    ) -> Response:
        if key is not None:
            if not _token_matches(key, session_key_bytes):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail="forbidden"
                )
            response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
            response.headers["Cache-Control"] = "no-store"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.set_cookie(
                SESSION_COOKIE,
                session_key,
                httponly=True,
                samesite="strict",
                path="/",
            )
            return response
        if not _token_matches(
            request.cookies.get(SESSION_COOKIE), session_key_bytes
        ):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")
        response = FileResponse(index_path)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/bootstrap", dependencies=[Depends(require_session)])
    async def bootstrap(request: Request, response: Response) -> Mapping[str, object]:
        if request.query_params:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="bootstrap query parameters are not accepted",
            )
        response.headers["Cache-Control"] = "no-store"
        runtime_status = current_bootstrap_status()
        session_id = runtime_status.session_id
        if session_id is not None and not store.session_exists(session_id):
            session_id = None
        tasks = (
            []
            if session_id is None
            else [
                task_snapshot(overview)
                for overview in store.latest_task_overviews(session_id)
            ]
        )
        return {
            "csrf_token": csrf_token,
            "usage_credits_acknowledged": (
                store.get_setting(USAGE_CREDITS_SETTING) is True
            ),
            "session_id": session_id,
            "fable_ready": runtime_status.fable_ready,
            "fable_status": runtime_status.fable_status,
            "sol_status": runtime_status.sol_status,
            "repository": runtime_status.repository,
            "branch": runtime_status.branch,
            "replay_after": (
                0 if session_id is None else store.browser_replay_floor(session_id)
            ),
            "tasks": tasks,
        }

    @app.post(
        "/api/settings/usage-credits-acknowledgement",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def acknowledge_usage_credits(
        body: UsageCreditsAcknowledgementRequest,
    ) -> Response:
        store.set_setting(USAGE_CREDITS_SETTING, body.acknowledged)
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/sessions/{session_id}/messages",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def send_message(session_id: SessionId, body: MessageRequest) -> Response:
        require_known_session(session_id)
        require_model_start_ready(session_id)
        schedule(
            coordinator.handle_user_request(session_id, body.text),
            session_id=session_id,
            task_id=None,
            action="message",
        )
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/tasks/{task_id}/approve",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def approve_task(task_id: TaskId, body: ApprovalRequest) -> Response:
        task = latest_task(task_id)
        if body.revision != task.revision:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="approval must name the latest exact revision",
            )
        if task.brief is None or task.brief.open_questions:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="resolve the TaskBrief open questions before approval",
            )
        require_model_start_ready(task.session_id)
        schedule(
            coordinator.approve_task(task_id, body.revision),
            session_id=task.session_id,
            task_id=task_id,
            action="approve",
        )
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/tasks/{task_id}/edit",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def edit_task(task_id: TaskId, body: TaskBriefRequest) -> Response:
        task = latest_task(task_id)
        try:
            brief = TaskBrief.from_dict(body.model_dump(mode="python"))
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from None
        if brief.task_id != task_id or brief.revision != task.revision + 1:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="edit must name the same task and next exact revision",
            )
        schedule(
            coordinator.edit_task(task_id, brief),
            session_id=task.session_id,
            task_id=task_id,
            action="edit",
        )
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/tasks/{task_id}/reject",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def reject_task(task_id: TaskId) -> Response:
        task = latest_task(task_id)
        schedule(
            coordinator.reject_task(task_id),
            session_id=task.session_id,
            task_id=task_id,
            action="reject",
        )
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/tasks/{task_id}/answer",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def answer_task(
        task_id: TaskId,
        body: AnswerRequest,
    ) -> Response:
        task = latest_task(task_id)
        require_model_start_ready(task.session_id)
        schedule(
            coordinator.answer_user_question(task_id, body.answer),
            session_id=task.session_id,
            task_id=task_id,
            action="answer",
        )
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/tasks/{task_id}/stop",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def stop_task(task_id: TaskId) -> Response:
        task = latest_task(task_id)
        schedule(
            coordinator.stop_task(task_id),
            session_id=task.session_id,
            task_id=task_id,
            action="stop",
        )
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/tasks/{task_id}/resume",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def resume_task(task_id: TaskId) -> Response:
        task = latest_task(task_id)
        require_model_start_ready(task.session_id)
        schedule(
            coordinator.resume_task(task_id),
            session_id=task.session_id,
            task_id=task_id,
            action="resume",
        )
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.get("/static/{asset_path:path}")
    async def static_asset(asset_path: str) -> Response:
        try:
            candidate = (static_root / asset_path).resolve(strict=True)
            candidate.relative_to(static_root)
            is_file = candidate.is_file()
        except (OSError, RuntimeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="asset not found"
            ) from None
        if not is_file or candidate.name == "index.html":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="asset not found"
            )
        return FileResponse(candidate)

    @app.websocket("/ws")
    async def websocket_events(
        websocket: WebSocket,
        session_id: SocketSessionId,
        after: ReplayCursor = 0,
    ) -> None:
        if (
            not _token_matches(
                websocket.cookies.get(SESSION_COOKIE), session_key_bytes
            )
            or _SAFE_ID.fullmatch(session_id) is None
            or not store.session_exists(session_id)
        ):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        await websocket.accept()
        last_sequence = after

        async def send_event_pages(cursor: int) -> int:
            while True:
                page = store.events_after(
                    session_id,
                    cursor,
                    limit=EVENT_REPLAY_PAGE_SIZE,
                )
                for event in page:
                    if event.sequence <= cursor:
                        continue
                    await websocket.send_json(event.to_dict())
                    cursor = event.sequence
                if len(page) < EVENT_REPLAY_PAGE_SIZE:
                    return cursor

        try:
            # Registration is synchronous, so an event created while replay is
            # being read is queued.  Replay is still sent before queued live data.
            async with event_broadcaster.subscribe(session_id) as subscription:
                last_sequence = await send_event_pages(last_sequence)

                while True:
                    live_task = asyncio.create_task(anext(subscription))
                    receive_task = asyncio.create_task(websocket.receive())
                    done, _ = await asyncio.wait(
                        {live_task, receive_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if receive_task in done:
                        incoming = receive_task.result()
                        live_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await live_task
                        if incoming.get("type") == "websocket.disconnect":
                            break
                        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
                        break

                    receive_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await receive_task
                    try:
                        live_task.result()
                    except StopAsyncIteration:
                        break
                    # Broadcasts are bounded level-triggered wakeups only.
                    # SQLite remains the ordered source of truth, so reordered
                    # or coalesced notifications cannot lose an event.
                    last_sequence = await send_event_pages(last_sequence)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


__all__ = [
    "BootstrapStatus",
    "EventBroadcaster",
    "InMemoryEventBroadcaster",
    "SESSION_COOKIE",
    "create_app",
]
