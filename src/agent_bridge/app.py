"""Loopback browser API for the local Fable--Sol bridge.

The web boundary deliberately accepts task actions, not repository roots or
process identifiers.  Authentication is a keyed, HttpOnly browser session;
every HTTP mutation additionally requires a CSRF token obtained from the
authenticated bootstrap endpoint.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
import re
import secrets
import threading
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol

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
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from agent_bridge.contracts import ConversationTarget, StreamEvent, TaskBrief
from agent_bridge.coordinator import Coordinator, InterventionIntent
from agent_bridge.store import (
    EVENT_REPLAY_PAGE_SIZE,
    MAX_CHAT_PAGE_SIZE,
    ChatCursor,
    ChatRecord,
    SQLiteStore,
    TaskOverview,
    TaskRecord,
    InterventionRecord,
    InterventionStatus,
)

if TYPE_CHECKING:
    from agent_bridge.hub import HubWorkflowOrchestrator, ProjectRegistry
    from agent_bridge.hub_store import HubStore


SESSION_COOKIE = "agent_bridge_session"
USAGE_CREDITS_SETTING = "usage_credits_acknowledged"
_SAFE_ID_BODY = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}"
_SAFE_ID_PATTERN = f"^{_SAFE_ID_BODY}$"
_SAFE_ID = re.compile(f"{_SAFE_ID_BODY}\\Z")
_MAX_BROWSER_TEXT_BYTES = 16 * 1024
_ACTIVITY_KINDS = frozenset({
    "action_error", "stop_error", "agent_event", "resume_drift",
})
_AGENT_ACTIVITY_STATUSES = frozenset({
    "completed", "declined", "failed", "in_progress", "interrupted",
})
_LOWER_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")


def _activity_projection(
    kind: object,
    activity: object,
) -> tuple[str | None, Mapping[str, object] | None]:
    """Project one browser-safe structural activity record without provider data."""
    if not isinstance(kind, str) or kind not in _ACTIVITY_KINDS:
        return None, None
    if kind != "agent_event" or not isinstance(activity, Mapping):
        return kind, {}
    status_value = activity.get("status")
    digest = activity.get("command_sha256")
    if not (
        isinstance(status_value, str)
        and status_value in _AGENT_ACTIVITY_STATUSES
        and isinstance(digest, str)
        and _LOWER_HEX_256.fullmatch(digest)
    ):
        return kind, {}
    return kind, {"status": status_value, "command_sha256": digest}


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
    text: StrictStr
    addressed_to: Literal["fable", "sol", "team"] = "fable"

    @field_validator("text")
    @classmethod
    def _text_must_be_non_empty(cls, value: str) -> str:
        return _non_empty_browser_text(value, "text")


class ContinuationMessageRequest(_StrictRequest):
    text: StrictStr
    addressed_to: Literal["fable", "sol"]
    revision: StrictInt = Field(ge=1)
    continuation_generation: StrictInt = Field(ge=1)

    @field_validator("text")
    @classmethod
    def _text_must_be_non_empty(cls, value: str) -> str:
        return _non_empty_browser_text(value, "text")


class QuestionAnswerRequest(_StrictRequest):
    text: StrictStr
    revision: StrictInt = Field(ge=1)
    question_id: StrictStr
    continuation_generation: StrictInt = Field(ge=1)

    @field_validator("text")
    @classmethod
    def _text_must_be_non_empty(cls, value: str) -> str:
        return _non_empty_browser_text(value, "text")

    @field_validator("question_id")
    @classmethod
    def _question_id_is_safe(cls, value: str) -> str:
        return _safe_browser_identifier(value, "question_id")


class ExchangeGrantRequest(_StrictRequest):
    revision: StrictInt = Field(ge=1)
    continuation_generation: StrictInt = Field(ge=1)
    request_id: StrictStr

    @field_validator("request_id")
    @classmethod
    def _request_id_is_safe(cls, value: str) -> str:
        return _safe_browser_identifier(value, "request_id")


class InterventionRequest(_StrictRequest):
    intervention_id: StrictStr
    message: StrictStr
    addressed_to: Literal["fable", "sol"]
    revision: StrictInt = Field(ge=0)
    continuation_generation: StrictInt = Field(ge=1)

    @field_validator("intervention_id")
    @classmethod
    def _intervention_id_is_safe(cls, value: str) -> str:
        return _safe_browser_identifier(value, "intervention_id")

    @field_validator("message")
    @classmethod
    def _message_is_safe(cls, value: str) -> str:
        return _non_empty_browser_text(value, "message")


class InterventionResumeRequest(_StrictRequest):
    expected_resume_generation: StrictInt = Field(ge=1)


class UnknownOutcomeRetryRequest(InterventionResumeRequest):
    acknowledgment_id: StrictStr
    acknowledge_possible_prior_execution: Literal[True]

    @field_validator("acknowledgment_id")
    @classmethod
    def _acknowledgment_id_is_safe(cls, value: str) -> str:
        return _safe_browser_identifier(value, "acknowledgment_id")


def _non_empty_browser_text(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    try:
        encoded_length = len(value.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    if encoded_length > _MAX_BROWSER_TEXT_BYTES:
        raise ValueError(f"{name} must be at most 16384 bytes")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _safe_browser_identifier(value: str, name: str) -> str:
    if _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe identifier")
    return value


async def _bounded_request_validation_error(
    _request: Request,
    _error: RequestValidationError,
) -> JSONResponse:
    """Never reflect untrusted request values in browser validation responses."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": "invalid request"},
    )


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
ProjectId = Annotated[
    str,
    PathParameter(min_length=1, max_length=128, pattern=_SAFE_ID_PATTERN),
]
ChatPageLimit = Annotated[int, Query(ge=1, le=MAX_CHAT_PAGE_SIZE)]
ChatCursorSequence = Annotated[int | None, Query(ge=0, le=2**53 - 1)]
ChatCursorSession = Annotated[
    str | None,
    Query(min_length=1, max_length=128, pattern=_SAFE_ID_PATTERN),
]
SocketProjectId = Annotated[
    str,
    Query(min_length=1, max_length=128, pattern=_SAFE_ID_PATTERN),
]


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


@dataclass(frozen=True, slots=True)
class CompatibilityProjectRuntime:
    """One non-owning app-facing projection for the legacy factory."""

    project_id: str
    label: str
    repository: str
    branch: str
    coordinator: Coordinator
    store: SQLiteStore
    broadcaster: EventBroadcaster
    readiness: object


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


def create_hub_app(
    *,
    registry: ProjectRegistry,
    hub_store: HubStore,
    workflows: HubWorkflowOrchestrator,
    static_dir: str | Path,
    session_key: str,
    csrf_token: str,
) -> FastAPI:
    """Create the project-scoped authenticated browser application.

    The registry lookup is deliberately the first application operation in
    every project route.  Session and task identifiers are only meaningful in
    the selected runtime, so no fallback or hub-wide persistence lookup exists.
    """

    session_key_bytes = _ascii_token(session_key, "session_key")
    csrf_token_bytes = _ascii_token(csrf_token, "csrf_token")
    static_root = Path(static_dir).resolve()
    index_path = static_root / "index.html"
    if not static_root.is_dir() or not index_path.is_file():
        raise ValueError("static_dir must contain index.html")

    def thaw_projection(value: object) -> object:
        if isinstance(value, Mapping):
            return {key: thaw_projection(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [thaw_projection(item) for item in value]
        return value

    def intervention_projection(record: InterventionRecord) -> Mapping[str, object]:
        unknown = record.status is InterventionStatus.RESUME_OUTCOME_UNKNOWN
        return {
            "intervention_id": record.intervention_id,
            "message": record.message,
            "addressed_to": record.addressed_to.value,
            "routed_to": record.routed_to.value,
            "status": record.status.value,
            "task_id": record.task_id,
            "revision": record.revision,
            "source_generation": record.source_generation,
            "resume_generation": record.resume_generation,
            "eligible": record.status in {
                InterventionStatus.PENDING_STOP,
                InterventionStatus.READY,
            },
            "visible_discontinuity": (
                record.continuation_state.value == "fable_planning"
                and record.fable_session_id is None
            ),
            "warning": (
                "prior resume outcome is unknown and may have executed"
                if unknown else None
            ),
        }

    def task_snapshot(runtime: object, overview: TaskOverview) -> Mapping[str, object]:
        task = overview.task
        activity_kind, activity = _activity_projection(
            overview.activity_kind, overview.activity,
        )
        store = getattr(runtime, "store")
        question = store.unanswered_question_for_task(task.task_id, task.revision)
        pending_question: Mapping[str, object] | None = None
        if question is not None and question.routed_to is ConversationTarget.USER:
            pending_question = {
                "question_id": question.question_id,
                "asked_by": question.asked_by.value,
                "addressed_to": question.addressed_to.value,
                "routed_to": question.routed_to.value,
                "text": question.text,
                "revision": question.revision,
                "continuation_generation": question.continuation_generation,
            }
        exchange_permission = store.current_exchange_permission(
            session_id=task.session_id,
            task_id=task.task_id,
            revision=task.revision,
        )
        intervention = store.current_visible_intervention_for_task(
            task.task_id, task.revision,
        )
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
            "continuation_generation": task.continuation_generation,
            "exchange_allowance": task.exchange_allowance,
            "exchange_consumed": task.exchange_consumed,
            "pending_question": pending_question,
            "exchange_permission": exchange_permission,
            "updated_at": overview.updated_at,
            "active_agent": overview.active_agent,
            "active_started_at": overview.active_started_at,
            "revision_start_sequence": overview.revision_start_sequence,
            "outcome": thaw_projection(overview.outcome),
            "review": thaw_projection(overview.review),
            "clarification": thaw_projection(overview.clarification),
            "activity_kind": activity_kind,
            "activity": activity,
            "intervention": (
                None if intervention is None else intervention_projection(intervention)
            ),
        }

    def chat_snapshot(chat: ChatRecord) -> Mapping[str, object]:
        return {
            "session_id": chat.session_id,
            "title": chat.title,
            "created_at": chat.created_at,
            "updated_at": chat.updated_at,
            "latest_sequence": chat.latest_sequence,
        }

    @asynccontextmanager
    async def lifespan(lifespan_app: FastAPI) -> AsyncIterator[None]:
        lifespan_app.state.shutting_down = False
        try:
            yield
        finally:
            lifespan_app.state.shutting_down = True
            active = tuple(lifespan_app.state.active_coroutines)
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
                await asyncio.sleep(0)
                lifespan_app.state.active_coroutines.difference_update(active)

    app = FastAPI(title="Agent Bridge", version="0.1.0", lifespan=lifespan)
    app.add_exception_handler(RequestValidationError, _bounded_request_validation_error)
    app.state.active_coroutines = set()
    app.state.coroutine_observation_failures = []
    app.state.shutting_down = False
    app.state.project_registry = registry
    app.state.hub_store = hub_store
    app.state.hub_orchestrator = workflows

    def require_session(request: Request) -> None:
        if not _token_matches(request.cookies.get(SESSION_COOKIE), session_key_bytes):
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

    def selected_runtime(project_id: str) -> object:
        """Resolve the opaque project before any caller-controlled identifier."""
        try:
            return registry.runtime(project_id)
        except LookupError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="project not found",
            ) from None

    def selected_chat(runtime: object, session_id: str) -> ChatRecord:
        chat = runtime.store.chat(session_id)
        if chat is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="chat not found"
            )
        return chat

    def selected_task(runtime: object, session_id: str, task_id: str) -> TaskRecord:
        task = runtime.store.latest_task(task_id)
        if task is None or task.session_id != session_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="task not found"
            )
        return task

    def selected_intervention(
        runtime: object, session_id: str, intervention_id: str,
    ) -> InterventionRecord:
        try:
            record = runtime.store.authenticated_intervention(intervention_id)
        except (RuntimeError, ValueError):
            record = None
        if record is None or record.session_id != session_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="intervention not found"
            )
        return record

    def workflow_http_error(error: BaseException) -> None:
        if isinstance(error, LookupError):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="chat or task not found"
            ) from None
        if isinstance(error, (PermissionError, RuntimeError, ValueError)):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="workflow is not currently available",
            ) from None
        raise error

    def observe_coroutine(
        task: asyncio.Task[object],
        *,
        runtime: object,
        session_id: str,
        task_id: str | None,
        action: str,
    ) -> None:
        app.state.active_coroutines.discard(task)
        if task.cancelled() and app.state.shutting_down:
            return
        try:
            error = asyncio.CancelledError() if task.cancelled() else task.exception()
        except BaseException as observation_error:
            error = observation_error
        if error is None:
            return
        try:
            runtime.store.append_event(
                session_id,
                task_id,
                "coordinator",
                "action_error",
                {"action": action, "error_type": type(error).__name__},
            )
        except BaseException as persistence_error:
            app.state.coroutine_observation_failures.append(
                {
                    "stage": "persistence",
                    "action": action,
                    "error_type": type(persistence_error).__name__,
                }
            )

    def install_action(
        *,
        runtime: object,
        session_id: str,
        task_id: str | None,
        action: str,
        coroutine_factory: Callable[[], Coroutine[object, object, None]],
    ) -> bool:
        if app.state.shutting_down:
            return False
        coroutine: Coroutine[object, object, None] | None = None
        try:
            coroutine = coroutine_factory()
            task = asyncio.create_task(coroutine, name=f"agent-bridge-{action}")
        except BaseException:
            if coroutine is not None:
                coroutine.close()
            return False
        app.state.active_coroutines.add(task)
        task.add_done_callback(
            lambda completed: observe_coroutine(
                completed,
                runtime=runtime,
                session_id=session_id,
                task_id=task_id,
                action=action,
            )
        )
        return True

    def install_stop_action(
        *,
        runtime: object,
        session_id: str,
        task_id: str,
        coroutine_factory: Callable[[], Coroutine[object, object, None]],
        cancel_reservation: Callable[[], None],
    ) -> bool:
        """Install a reserved Stop or synchronously return its exact claim."""
        if app.state.shutting_down:
            try:
                cancel_reservation()
            except BaseException as error:
                app.state.coroutine_observation_failures.append(
                    {"stage": "stop_reservation", "error_type": type(error).__name__}
                )
            return False
        coroutine: Coroutine[object, object, None] | None = None
        try:
            coroutine = coroutine_factory()
            task = asyncio.create_task(coroutine, name="agent-bridge-stop")
        except BaseException:
            if coroutine is not None:
                coroutine.close()
            try:
                cancel_reservation()
            except BaseException as error:
                app.state.coroutine_observation_failures.append(
                    {"stage": "stop_reservation", "error_type": type(error).__name__}
                )
            return False
        app.state.active_coroutines.add(task)
        task.add_done_callback(
            lambda completed: observe_coroutine(
                completed,
                runtime=runtime,
                session_id=session_id,
                task_id=task_id,
                action="stop",
            )
        )
        return True

    # This closure deliberately owns only the task it installs.  The workflow
    # object remains the sole owner of the lease and durable preparation.
    def install_prepared_action(
        *,
        prepared: object,
        coroutine_factory: Callable[[], Coroutine[object, object, None]],
        abort: Callable[[object, str], None],
    ) -> bool:
        if app.state.shutting_down:
            try:
                abort(prepared, "scheduler_unavailable")
            except BaseException as error:
                app.state.coroutine_observation_failures.append(
                    {"stage": "abort", "error_type": type(error).__name__}
                )
            return False
        coroutine: Coroutine[object, object, None] | None = None
        try:
            coroutine = coroutine_factory()
            task = asyncio.create_task(coroutine, name="agent-bridge-prepared")
        except BaseException:
            if coroutine is not None:
                coroutine.close()
            try:
                abort(prepared, "scheduler_unavailable")
            except BaseException as error:
                app.state.coroutine_observation_failures.append(
                    {"stage": "abort", "error_type": type(error).__name__}
                )
            return False
        app.state.active_coroutines.add(task)
        def observe_prepared(completed: asyncio.Task[object]) -> None:
            app.state.active_coroutines.discard(completed)
            try:
                error = completed.exception()
            except asyncio.CancelledError:
                if not app.state.shutting_down:
                    app.state.coroutine_observation_failures.append(
                        {"stage": "prepared", "outcome": "cancelled"}
                    )
                return
            except BaseException as observation_error:
                error = observation_error
            if error is not None:
                app.state.coroutine_observation_failures.append(
                    {"stage": "prepared", "error_type": type(error).__name__}
                )

        task.add_done_callback(observe_prepared)
        return True

    def recoverable_preparation(prepared: object) -> HTTPException:
        preparation_id = getattr(prepared, "preparation_id", None)
        token = getattr(prepared, "token", None)
        project_id = getattr(token, "project_id", getattr(prepared, "project_id", None))
        session_id = getattr(token, "session_id", getattr(prepared, "session_id", None))
        task_id = getattr(token, "task_id", getattr(prepared, "task_id", None))
        revision = getattr(prepared, "revision", None)
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "state": "recoverable",
                "preparation_id": preparation_id if isinstance(preparation_id, str) else None,
                "project_id": project_id if isinstance(project_id, str) else None,
                "session_id": session_id if isinstance(session_id, str) else None,
                "task_id": task_id if isinstance(task_id, str) else None,
                "revision": revision if isinstance(revision, int) and revision >= 0 else None,
            },
        )

    class BrowserIds:
        def new_task_id(self) -> str:
            return f"task-{secrets.token_hex(16)}"

    browser_ids = BrowserIds()

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
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")
            response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
            response.headers["Cache-Control"] = "no-store"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.set_cookie(
                SESSION_COOKIE, session_key, httponly=True, samesite="strict", path="/"
            )
            return response
        require_session(request)
        response = FileResponse(index_path)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/projects", dependencies=[Depends(require_session)])
    async def projects(request: Request, response: Response) -> Mapping[str, object]:
        if request.query_params:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="project query parameters are not accepted",
            )
        active = workflows.active_lease_snapshot()
        response.headers["Cache-Control"] = "no-store"
        return {
            "csrf_token": csrf_token,
            "usage_credits_acknowledged": hub_store.usage_credits_acknowledged(),
            "projects": [
                {
                    "project_id": runtime.project_id,
                    "label": runtime.label,
                    "branch": runtime.branch,
                    "readiness": {
                        "fable_ready": runtime.readiness.snapshot().fable_ready,
                        "fable_status": runtime.readiness.snapshot().fable_status,
                        "sol_status": runtime.readiness.snapshot().sol_status,
                    },
                }
                for runtime in registry.projects()
            ],
            "active_lease": (
                None
                if active is None
                else {
                    "project_id": active.project_id,
                    "session_id": active.session_id,
                    "task_id": active.task_id,
                }
            ),
        }

    @app.get("/api/projects/{project_id}/chats", dependencies=[Depends(require_session)])
    async def list_chats(
        project_id: ProjectId,
        before_sequence: ChatCursorSequence = None,
        before_session_id: ChatCursorSession = None,
        limit: ChatPageLimit = MAX_CHAT_PAGE_SIZE,
    ) -> Mapping[str, object]:
        runtime = selected_runtime(project_id)
        if (before_sequence is None) != (before_session_id is None):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="chat cursor requires both sequence and session id",
            )
        before = (
            None
            if before_sequence is None
            else ChatCursor(before_sequence, before_session_id)
        )
        return {"chats": [chat_snapshot(chat) for chat in runtime.store.list_chats(before=before, limit=limit)]}

    @app.post(
        "/api/projects/{project_id}/chats",
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(require_mutation)],
    )
    async def create_chat(project_id: ProjectId) -> Mapping[str, object]:
        runtime = selected_runtime(project_id)
        try:
            workflows.require_no_active_lease()
        except BaseException as error:
            workflow_http_error(error)
        return chat_snapshot(runtime.store.create_chat(runtime.repository))

    @app.get(
        "/api/projects/{project_id}/chats/{session_id}/bootstrap",
        dependencies=[Depends(require_session)],
    )
    async def bootstrap(
        project_id: ProjectId,
        session_id: SessionId,
        request: Request,
        response: Response,
    ) -> Mapping[str, object]:
        runtime = selected_runtime(project_id)
        if request.query_params:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="bootstrap query parameters are not accepted",
            )
        try:
            workflows.require_navigation_allowed(
                project_id=project_id, session_id=session_id,
            )
        except BaseException as error:
            workflow_http_error(error)
        selected_chat(runtime, session_id)
        response.headers["Cache-Control"] = "no-store"
        readiness = runtime.readiness.snapshot()
        return {
            "csrf_token": csrf_token,
            "usage_credits_acknowledged": hub_store.usage_credits_acknowledged(),
            "project_id": project_id,
            "session_id": session_id,
            "fable_ready": readiness.fable_ready,
            "fable_status": readiness.fable_status,
            "sol_status": readiness.sol_status,
            "branch": runtime.branch,
            "replay_after": runtime.store.browser_replay_floor(session_id),
            "tasks": [
                task_snapshot(runtime, overview)
                for overview in runtime.store.latest_task_overviews(session_id)
            ],
        }

    @app.post(
        "/api/settings/usage-credits-acknowledgement",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def acknowledge_usage_credits(
        body: UsageCreditsAcknowledgementRequest,
    ) -> Response:
        if body.acknowledged:
            hub_store.acknowledge_usage_credits()
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/projects/{project_id}/chats/{session_id}/messages",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def send_message(
        project_id: ProjectId, session_id: SessionId, body: MessageRequest,
    ) -> Response:
        runtime = selected_runtime(project_id)
        try:
            workflows.require_no_active_lease()
        except BaseException as error:
            workflow_http_error(error)
        selected_chat(runtime, session_id)
        try:
            prepared = await workflows.prepare_new_request(
                project_id=project_id,
                session_id=session_id,
                text=body.text,
                ids=browser_ids,
                addressed_to=ConversationTarget(body.addressed_to),
            )
        except BaseException as error:
            workflow_http_error(error)
        if not install_prepared_action(
            prepared=prepared,
            coroutine_factory=lambda: workflows.run(prepared),
            abort=lambda value, reason: workflows.abort_prepared(value, reason=reason),
        ):
            raise recoverable_preparation(prepared)
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/messages",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def send_continuation_message(
        project_id: ProjectId,
        session_id: SessionId,
        task_id: TaskId,
        body: ContinuationMessageRequest,
    ) -> Response:
        runtime = selected_runtime(project_id)
        selected_chat(runtime, session_id)
        try:
            prepared = await workflows.prepare_continuation_message(
                project_id=project_id,
                session_id=session_id,
                task_id=task_id,
                revision=body.revision,
                continuation_generation=body.continuation_generation,
                text=body.text,
                addressed_to=ConversationTarget(body.addressed_to),
            )
        except BaseException as error:
            workflow_http_error(error)
        if not install_prepared_action(
            prepared=prepared,
            coroutine_factory=lambda: workflows.run(prepared),
            abort=lambda value, reason: workflows.abort_prepared(value, reason=reason),
        ):
            raise recoverable_preparation(prepared)
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/approve",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def approve_task(
        project_id: ProjectId,
        session_id: SessionId,
        task_id: TaskId,
        body: ApprovalRequest,
    ) -> Response:
        runtime = selected_runtime(project_id)
        try:
            workflows.require_no_active_lease()
        except BaseException as error:
            workflow_http_error(error)
        selected_chat(runtime, session_id)
        task = selected_task(runtime, session_id, task_id)
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
        try:
            prepared = await workflows.prepare_approval(
                project_id=project_id,
                session_id=session_id,
                task_id=task_id,
                revision=body.revision,
            )
        except BaseException as error:
            workflow_http_error(error)
        if not install_prepared_action(
            prepared=prepared,
            coroutine_factory=lambda: workflows.run(prepared),
            abort=lambda value, reason: workflows.abort_prepared(value, reason=reason),
        ):
            raise recoverable_preparation(prepared)
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/edit",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def edit_task(
        project_id: ProjectId,
        session_id: SessionId,
        task_id: TaskId,
        body: TaskBriefRequest,
    ) -> Response:
        runtime = selected_runtime(project_id)
        selected_chat(runtime, session_id)
        task = selected_task(runtime, session_id, task_id)
        try:
            brief = TaskBrief.from_dict(body.model_dump(mode="python"))
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from None
        if brief.task_id != task_id or brief.revision != task.revision + 1:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="edit must name the same task and next exact revision",
            )
        if not install_action(
            runtime=runtime,
            session_id=session_id,
            task_id=task_id,
            action="edit",
            coroutine_factory=lambda: runtime.coordinator.edit_task(task_id, brief),
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="bridge is shutting down",
            )
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/reject",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def reject_task(
        project_id: ProjectId, session_id: SessionId, task_id: TaskId,
    ) -> Response:
        runtime = selected_runtime(project_id)
        selected_chat(runtime, session_id)
        selected_task(runtime, session_id, task_id)
        if not install_action(
            runtime=runtime,
            session_id=session_id,
            task_id=task_id,
            action="reject",
            coroutine_factory=lambda: runtime.coordinator.reject_task(task_id),
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="bridge is shutting down",
            )
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/answer",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def answer_task(
        project_id: ProjectId,
        session_id: SessionId,
        task_id: TaskId,
        body: QuestionAnswerRequest,
    ) -> Response:
        runtime = selected_runtime(project_id)
        selected_chat(runtime, session_id)
        try:
            prepared = await workflows.prepare_question_answer(
                project_id=project_id,
                session_id=session_id,
                task_id=task_id,
                revision=body.revision,
                continuation_generation=body.continuation_generation,
                question_id=body.question_id,
                answer=body.text,
            )
        except BaseException as error:
            workflow_http_error(error)
        if not install_prepared_action(
            prepared=prepared,
            coroutine_factory=lambda: workflows.run(prepared),
            abort=lambda value, reason: workflows.abort_prepared(value, reason=reason),
        ):
            raise recoverable_preparation(prepared)
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/exchanges/grant",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def grant_task_exchanges(
        project_id: ProjectId,
        session_id: SessionId,
        task_id: TaskId,
        body: ExchangeGrantRequest,
    ) -> Response:
        runtime = selected_runtime(project_id)
        selected_chat(runtime, session_id)
        try:
            prepared = await workflows.prepare_exchange_grant(
                project_id=project_id,
                session_id=session_id,
                task_id=task_id,
                revision=body.revision,
                continuation_generation=body.continuation_generation,
                request_id=body.request_id,
            )
        except BaseException as error:
            workflow_http_error(error)
        if not install_prepared_action(
            prepared=prepared,
            coroutine_factory=lambda: workflows.run(prepared),
            abort=lambda value, reason: workflows.abort_prepared(value, reason=reason),
        ):
            raise recoverable_preparation(prepared)
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/intervene",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def intervene_task(
        project_id: ProjectId, session_id: SessionId, task_id: TaskId,
        body: InterventionRequest,
    ) -> Mapping[str, object]:
        runtime = selected_runtime(project_id)
        selected_chat(runtime, session_id)
        task = selected_task(runtime, session_id, task_id)
        if hub_store.usage_credits_acknowledged() is not True:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="usage credits must be acknowledged")
        if body.revision != task.revision:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="intervention must name the latest exact revision")
        try:
            prepared = workflows.prepare_intervention(
                project_id=project_id, session_id=session_id, task_id=task_id,
                intent=InterventionIntent(
                    intervention_id=body.intervention_id, message=body.message,
                    addressed_to=ConversationTarget(body.addressed_to), revision=body.revision,
                    continuation_generation=body.continuation_generation,
                ),
            )
        except BaseException as error:
            workflow_http_error(error)
        installed = install_prepared_action(
            prepared=prepared,
            coroutine_factory=lambda: workflows.continue_intervention(prepared),
            abort=lambda value, reason: workflows.abort_prepared_intervention(value, reason=reason),
        )
        return {"intervention": intervention_projection(prepared.record), "scheduled": installed}

    @app.post(
        "/api/projects/{project_id}/chats/{session_id}/interventions/{intervention_id}/resume",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def resume_intervention(
        project_id: ProjectId, session_id: SessionId, intervention_id: TaskId,
        body: InterventionResumeRequest,
    ) -> Mapping[str, object]:
        runtime = selected_runtime(project_id)
        selected_chat(runtime, session_id)
        current = selected_intervention(runtime, session_id, intervention_id)
        if hub_store.usage_credits_acknowledged() is not True:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="usage credits must be acknowledged")
        try:
            if current.resume_generation != body.expected_resume_generation:
                raise RuntimeError("intervention resume generation changed")
            prepared = workflows.prepare_recovery_resume(
                project_id=project_id, session_id=session_id, intervention_id=intervention_id,
                expected_resume_generation=body.expected_resume_generation,
            )
        except BaseException as error:
            workflow_http_error(error)
        installed = install_prepared_action(
            prepared=prepared,
            coroutine_factory=lambda: workflows.continue_intervention(prepared),
            abort=lambda value, reason: workflows.abort_prepared_intervention(value, reason=reason),
        )
        return {"intervention": intervention_projection(prepared.record), "scheduled": installed}

    @app.post(
        "/api/projects/{project_id}/chats/{session_id}/interventions/{intervention_id}/authorize-retry",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def authorize_intervention_retry(
        project_id: ProjectId, session_id: SessionId, intervention_id: TaskId,
        body: UnknownOutcomeRetryRequest,
    ) -> Mapping[str, object]:
        runtime = selected_runtime(project_id)
        selected_chat(runtime, session_id)
        if hub_store.usage_credits_acknowledged() is not True:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="usage credits must be acknowledged")
        try:
            selected_intervention(runtime, session_id, intervention_id)
            authorized = runtime.store.authorize_retry_after_unknown(
                intervention_id, expected_resume_generation=body.expected_resume_generation,
                acknowledgment_id=body.acknowledgment_id,
            )
            prepared = workflows.prepare_recovery_resume(
                project_id=project_id, session_id=session_id, intervention_id=intervention_id,
                expected_resume_generation=authorized.resume_generation,
            )
        except BaseException as error:
            workflow_http_error(error)
        installed = install_prepared_action(
            prepared=prepared,
            coroutine_factory=lambda: workflows.continue_intervention(prepared),
            abort=lambda value, reason: workflows.abort_prepared_intervention(value, reason=reason),
        )
        return {"intervention": intervention_projection(prepared.record), "scheduled": installed}

    @app.post(
        "/api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/stop",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def stop_task(
        project_id: ProjectId, session_id: SessionId, task_id: TaskId,
    ) -> Response:
        runtime = selected_runtime(project_id)
        try:
            stop_reservation = workflows.reserve_stop(
                project_id=project_id, session_id=session_id, task_id=task_id,
            )
        except BaseException as error:
            workflow_http_error(error)
        try:
            selected_chat(runtime, session_id)
            selected_task(runtime, session_id, task_id)
        except BaseException:
            workflows.cancel_stop_reservation(stop_reservation)
            raise
        if not install_stop_action(
            runtime=runtime,
            session_id=session_id,
            task_id=task_id,
            coroutine_factory=lambda: workflows.stop(
                reservation=stop_reservation,
            ),
            cancel_reservation=lambda: workflows.cancel_stop_reservation(stop_reservation),
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="bridge is shutting down",
            )
        return Response(status_code=status.HTTP_202_ACCEPTED)

    @app.post(
        "/api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/resume",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_mutation)],
    )
    async def resume_task(
        project_id: ProjectId, session_id: SessionId, task_id: TaskId,
    ) -> Response:
        runtime = selected_runtime(project_id)
        try:
            workflows.require_no_active_lease()
        except BaseException as error:
            workflow_http_error(error)
        selected_chat(runtime, session_id)
        task = selected_task(runtime, session_id, task_id)
        try:
            prepared = await workflows.prepare_resume(
                project_id=project_id,
                session_id=session_id,
                task_id=task_id,
                revision=task.revision,
            )
        except BaseException as error:
            workflow_http_error(error)
        if not install_prepared_action(
            prepared=prepared,
            coroutine_factory=lambda: workflows.run(prepared),
            abort=lambda value, reason: workflows.abort_prepared(value, reason=reason),
        ):
            raise recoverable_preparation(prepared)
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
        project_id: SocketProjectId,
        session_id: SocketSessionId,
        after: ReplayCursor = 0,
    ) -> None:
        if (
            not _token_matches(websocket.cookies.get(SESSION_COOKIE), session_key_bytes)
            or _SAFE_ID.fullmatch(project_id) is None
            or _SAFE_ID.fullmatch(session_id) is None
        ):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        try:
            runtime = selected_runtime(project_id)
            workflows.require_navigation_allowed(
                project_id=project_id, session_id=session_id,
            )
            selected_chat(runtime, session_id)
        except (HTTPException, RuntimeError):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        await websocket.accept()
        last_sequence = after

        async def send_event_pages(cursor: int) -> int:
            while True:
                page = runtime.store.events_after(
                    session_id, cursor, limit=EVENT_REPLAY_PAGE_SIZE
                )
                for event in page:
                    if event.sequence <= cursor:
                        continue
                    await websocket.send_json(event.to_dict())
                    cursor = event.sequence
                if len(page) < EVENT_REPLAY_PAGE_SIZE:
                    return cursor

        try:
            async with runtime.broadcaster.subscribe(session_id) as subscription:
                last_sequence = await send_event_pages(last_sequence)
                while True:
                    live_task = asyncio.create_task(anext(subscription))
                    receive_task = asyncio.create_task(websocket.receive())
                    done, _ = await asyncio.wait(
                        {live_task, receive_task}, return_when=asyncio.FIRST_COMPLETED
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
                    last_sequence = await send_event_pages(last_sequence)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def create_app(
    *,
    coordinator: Coordinator,
    store: SQLiteStore,
    static_dir: str | Path,
    session_key: str,
    csrf_token: str,
    broadcaster: EventBroadcaster | None = None,
    bootstrap_status: Callable[[], BootstrapStatus] | None = None,
    readiness_check: Callable[[], Awaitable[BootstrapStatus]] | None = None,
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
        activity_kind, activity = _activity_projection(
            overview.activity_kind, overview.activity,
        )
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
            "activity_kind": activity_kind,
            "activity": activity,
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
    app.add_exception_handler(RequestValidationError, _bounded_request_validation_error)
    app.state.active_coroutines = set()
    app.state.coroutine_observation_failures = []
    app.state.event_broadcaster = event_broadcaster
    app.state.shutting_down = False

    # The legacy factory is intentionally non-owning.  It retains its original
    # routes while exposing a single opaque runtime projection for callers that
    # need the same app-facing shape as the hub factory.
    from agent_bridge.hub import ProjectRegistry, RuntimeReadiness, RuntimeStatus

    async def compatibility_status() -> BootstrapStatus:
        runtime_status = (
            await readiness_check() if readiness_check is not None else status_provider()
        )
        if not isinstance(runtime_status, BootstrapStatus):
            raise RuntimeError("bootstrap status provider returned an invalid value")
        return runtime_status

    async def compatibility_fable_probe() -> tuple[bool, str]:
        runtime_status = await compatibility_status()
        return runtime_status.fable_ready, runtime_status.fable_status

    async def compatibility_sol_probe() -> str:
        return (await compatibility_status()).sol_status

    compatibility_runtime = CompatibilityProjectRuntime(
        project_id="default-project",
        label="Default project",
        repository="default-project",
        branch="default",
        coordinator=coordinator,
        store=store,
        broadcaster=event_broadcaster,
        readiness=RuntimeReadiness(
            initial=RuntimeStatus(False, "checking", "checking"),
            fable_probe=compatibility_fable_probe,
            sol_probe=compatibility_sol_probe,
        ),
    )
    app.state.compatibility_runtime = compatibility_runtime
    app.state.project_registry = ProjectRegistry((compatibility_runtime,))

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

    async def require_model_start_ready(session_id: str) -> None:
        runtime_status = (
            await compatibility_status()
            if readiness_check is not None
            else current_bootstrap_status()
        )
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
        try:
            await compatibility_runtime.readiness.require_model_start_ready(
                usage_credits_acknowledged=True
            )
        except (PermissionError, RuntimeError):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "model actions require Fable subscription readiness, "
                    "Sol readiness, and usage-credit acknowledgement "
                    "for this session"
                ),
            ) from None

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
        await require_model_start_ready(session_id)
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
        await require_model_start_ready(task.session_id)
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
        await require_model_start_ready(task.session_id)
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
        await require_model_start_ready(task.session_id)
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
    "create_hub_app",
]
