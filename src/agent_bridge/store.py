"""SQLite persistence for the local agent bridge.

The store deliberately owns records and compare-and-swap updates, but not
workflow policy.  The coordinator supplies every state transition explicitly.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import sqlite3
import threading
from typing import TypeAlias

from agent_bridge.contracts import JsonValue, StreamEvent, TaskBrief, freeze_json
from agent_bridge.state_machine import TaskState, require_transition


Clock: TypeAlias = Callable[[], str]
EventListener: TypeAlias = Callable[[StreamEvent], None]
EventListenerIdentity: TypeAlias = tuple[str, int, int]
_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "interrupted"})
_ACTIVE_TASK_STATES = (
    TaskState.FABLE_PLANNING,
    TaskState.SOL_RUNNING,
    TaskState.FABLE_CLARIFYING,
    TaskState.FABLE_REVIEWING,
    TaskState.SOL_CORRECTING,
)
MAX_TASK_OVERVIEWS = 200
EVENT_REPLAY_PAGE_SIZE = 100
MAX_INITIAL_REPLAY_EVENTS = 300
MAX_CHAT_TITLE_LENGTH = 80
MAX_CHAT_PAGE_SIZE = 50
_NEW_CHAT_TITLE = "New chat"
_ACTIVE_SESSION_SETTING = "agent_bridge.active_session_id"
_BASELINE_SETTING_PREFIX = "agent_bridge.baseline."
_MAX_LEGACY_AUDIT_REASONS = 8


@dataclass(frozen=True)
class TaskRecord:
    """One immutable task revision and its persisted workflow metadata."""

    task_id: str
    revision: int
    session_id: str
    state: TaskState
    brief: TaskBrief | None
    approved_at: str | None
    fable_session_id: str | None
    sol_thread_id: str | None
    baseline_id: str | None
    correction_count: int
    continuation_state: TaskState | None
    pending: Mapping[str, JsonValue] | None


@dataclass(frozen=True)
class AgentRunRecord:
    """A single agent child process belonging to one exact task revision."""

    run_id: str
    task_id: str
    revision: int
    agent: str
    pid: int | None
    process_group_id: int | None
    cli_session_id: str | None
    started_at: str
    ended_at: str | None
    exit_code: int | None
    status: str


@dataclass(frozen=True)
class TaskOverview:
    """Safe task-list metadata without agent continuation identities or PIDs."""

    task: TaskRecord
    updated_at: str | None
    active_agent: str | None
    active_started_at: str | None
    revision_start_sequence: int | None
    outcome: Mapping[str, JsonValue] | None
    review: Mapping[str, JsonValue] | None
    clarification: Mapping[str, JsonValue] | None
    activity: Mapping[str, JsonValue] | None


@dataclass(frozen=True, slots=True)
class ChatRecord:
    """One persisted chat projected from its session and event history."""

    session_id: str
    repo_root: str
    title: str
    created_at: str
    updated_at: str
    latest_sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_string(self.session_id, "session_id"))
        object.__setattr__(self, "repo_root", _require_string(self.repo_root, "repo_root"))
        title = _require_string(self.title, "title")
        if len(title) > MAX_CHAT_TITLE_LENGTH:
            raise ValueError(f"title must be at most {MAX_CHAT_TITLE_LENGTH} characters")
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "created_at", _require_string(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _require_string(self.updated_at, "updated_at"))
        latest_sequence = _require_integer(self.latest_sequence, "latest_sequence")
        if latest_sequence < 0:
            raise ValueError("latest_sequence must be non-negative")
        object.__setattr__(self, "latest_sequence", latest_sequence)


@dataclass(frozen=True, slots=True)
class ChatCursor:
    """An exclusive cursor in the deterministic chat-recency ordering."""

    latest_sequence: int
    session_id: str

    def __post_init__(self) -> None:
        latest_sequence = _require_integer(self.latest_sequence, "latest_sequence")
        if latest_sequence < 0:
            raise ValueError("latest_sequence must be non-negative")
        object.__setattr__(self, "latest_sequence", latest_sequence)
        object.__setattr__(self, "session_id", _require_string(self.session_id, "session_id"))


def _utc_now() -> str:
    """Default process-edge clock for callers that do not inject one."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _event_listener_identity(listener: EventListener) -> EventListenerIdentity:
    bound_instance = getattr(listener, "__self__", None)
    bound_function = getattr(listener, "__func__", None)
    if bound_instance is not None and bound_function is not None:
        return ("bound", id(bound_instance), id(bound_function))
    return ("callable", id(listener), 0)


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _mutable_json(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _mutable_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_json(item) for item in value]
    return value


def _encode_json(value: object) -> str:
    """Validate and serialize JSON in one deterministic representation."""
    frozen = freeze_json(value)
    return json.dumps(_mutable_json(frozen), separators=(",", ":"), sort_keys=True)


def _decode_mapping(raw: str, name: str) -> Mapping[str, JsonValue]:
    try:
        frozen = freeze_json(json.loads(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"persisted {name} is invalid JSON") from error
    if not isinstance(frozen, Mapping):
        raise RuntimeError(f"persisted {name} must be an object")
    return frozen


_MIGRATION_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        repo_root TEXT NOT NULL,
        created_at TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT 'New chat',
        updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        session_id TEXT NOT NULL,
        state TEXT NOT NULL,
        brief_json TEXT,
        approved_at TEXT,
        fable_session_id TEXT,
        sol_thread_id TEXT,
        baseline_id TEXT,
        correction_count INTEGER NOT NULL DEFAULT 0,
        continuation_state TEXT,
        pending_json TEXT,
        PRIMARY KEY (task_id, revision),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        task_id TEXT,
        actor TEXT NOT NULL,
        kind TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_runs (
        run_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        agent TEXT NOT NULL,
        pid INTEGER,
        process_group_id INTEGER,
        cli_session_id TEXT,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        exit_code INTEGER,
        status TEXT NOT NULL,
        FOREIGN KEY (task_id, revision) REFERENCES tasks(task_id, revision)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS one_running_agent_run_per_task_revision
    ON agent_runs (task_id, revision)
    WHERE status = 'running'
    """,
    """
    CREATE INDEX IF NOT EXISTS events_session_sequence
    ON events (session_id, sequence)
    """,
    """
    CREATE INDEX IF NOT EXISTS events_session_task_sequence
    ON events (session_id, task_id, sequence DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS events_session_task_kind_sequence
    ON events (session_id, task_id, kind, sequence DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS events_session_sequence_desc
    ON events (session_id, sequence DESC)
    """,
)


class SQLiteStore:
    """A connection-owned SQLite persistence boundary for one bridge process."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Clock = _utc_now,
        check_same_thread: bool = True,
    ) -> None:
        if not isinstance(check_same_thread, bool):
            raise ValueError("check_same_thread must be a bool")
        self._clock = clock
        self._connection = sqlite3.connect(
            str(path),
            isolation_level=None,
            check_same_thread=check_same_thread,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._event_listener_lock = threading.RLock()
        self._event_listeners: dict[int, EventListener] = {}
        self._event_listener_tokens: dict[EventListenerIdentity, int] = {}
        self._next_event_listener_token = 0
        self._pending_listener_events: deque[StreamEvent] = deque()
        self._dispatching_listener_events = False
        self._migrate()

    def close(self) -> None:
        with self._event_listener_lock:
            self._event_listeners.clear()
            self._event_listener_tokens.clear()
            self._pending_listener_events.clear()
        self._connection.close()

    def add_event_listener(self, listener: EventListener) -> int:
        """Register a process-local observer for events committed by this store."""
        if not callable(listener):
            raise ValueError("listener must be callable")
        identity = _event_listener_identity(listener)
        with self._event_listener_lock:
            existing = self._event_listener_tokens.get(identity)
            if existing is not None:
                return existing
            self._next_event_listener_token += 1
            token = self._next_event_listener_token
            self._event_listeners[token] = listener
            self._event_listener_tokens[identity] = token
        return token

    def remove_event_listener(self, token: int) -> None:
        """Remove one observer; removing an absent token is intentionally safe."""
        if not isinstance(token, int) or isinstance(token, bool):
            raise ValueError("listener token must be an integer")
        with self._event_listener_lock:
            listener = self._event_listeners.pop(token, None)
            if listener is not None:
                identity = _event_listener_identity(listener)
                if self._event_listener_tokens.get(identity) == token:
                    self._event_listener_tokens.pop(identity, None)

    def _drain_event_listeners(self) -> None:
        while True:
            with self._event_listener_lock:
                if not self._pending_listener_events:
                    self._dispatching_listener_events = False
                    return
                event = self._pending_listener_events.popleft()
                listeners = tuple(self._event_listeners.values())
            for listener in listeners:
                try:
                    listener(event)
                except BaseException:
                    # The event is already committed.  An observer must never
                    # turn persistence into a retry that duplicates it.
                    continue

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _migrate(self) -> None:
        self._connection.execute("BEGIN")
        try:
            for statement in _MIGRATION_STATEMENTS:
                self._connection.execute(statement)
            self._migrate_session_chat_metadata()
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _migrate_session_chat_metadata(self) -> None:
        """Add the chat projection fields without rewriting legacy records."""
        columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(sessions)")
        }
        if "title" not in columns:
            self._connection.execute(
                "ALTER TABLE sessions ADD COLUMN title TEXT NOT NULL DEFAULT 'New chat'"
            )
        if "updated_at" not in columns:
            self._connection.execute("ALTER TABLE sessions ADD COLUMN updated_at TEXT")
        self._connection.execute(
            "UPDATE sessions SET updated_at = created_at WHERE updated_at IS NULL"
        )

    @contextmanager
    def _immediate_transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _timestamp(self) -> str:
        return _require_string(self._clock(), "clock result")

    def create_session(self, session_id: str, repo_root: str) -> None:
        """Create one session using the legacy call signature."""
        self.create_chat(repo_root, session_id=session_id)

    def create_chat(
        self, repo_root: str, *, session_id: str | None = None,
    ) -> ChatRecord:
        """Persist an empty chat bound to exactly one repository root."""
        repo_root = _require_string(repo_root, "repo_root")
        if session_id is not None:
            return self._insert_chat(_require_string(session_id, "session_id"), repo_root)
        while True:
            generated_session_id = secrets.token_hex(16)
            try:
                return self._insert_chat(generated_session_id, repo_root)
            except sqlite3.IntegrityError as error:
                if "sessions.session_id" not in str(error):
                    raise

    def _insert_chat(self, session_id: str, repo_root: str) -> ChatRecord:
        timestamp = self._timestamp()
        self._connection.execute(
            """
            INSERT INTO sessions (session_id, repo_root, created_at, title, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, repo_root, timestamp, _NEW_CHAT_TITLE, timestamp),
        )
        chat = self.chat(session_id)
        if chat is None:
            raise RuntimeError("inserted chat could not be read")
        return chat

    def chat(self, session_id: str) -> ChatRecord | None:
        """Return one chat projection, or ``None`` when its session is absent."""
        row = self._connection.execute(
            """
            SELECT
                sessions.session_id,
                sessions.repo_root,
                sessions.title,
                sessions.created_at,
                sessions.updated_at,
                COALESCE((
                    SELECT events.sequence
                    FROM events
                    WHERE events.session_id = sessions.session_id
                    ORDER BY events.sequence DESC
                    LIMIT 1
                ), 0) AS latest_sequence
            FROM sessions
            WHERE sessions.session_id = ?
            """,
            (_require_string(session_id, "session_id"),),
        ).fetchone()
        return None if row is None else self._chat_from_row(row)

    def list_chats(
        self,
        *,
        before: ChatCursor | None = None,
        limit: int = MAX_CHAT_PAGE_SIZE,
    ) -> tuple[ChatRecord, ...]:
        """List a bounded, cursor-paged projection ordered by event sequence."""
        limit = _require_integer(limit, "limit")
        if not 1 <= limit <= MAX_CHAT_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {MAX_CHAT_PAGE_SIZE}")
        if before is not None and not isinstance(before, ChatCursor):
            raise ValueError("before must be a ChatCursor or None")
        if before is not None:
            latest_sequence = _require_integer(before.latest_sequence, "latest_sequence")
            if latest_sequence < 0:
                raise ValueError("latest_sequence must be non-negative")
            session_id = _require_string(before.session_id, "session_id")
            where = "WHERE latest_sequence < ? OR (latest_sequence = ? AND session_id > ?)"
            parameters: tuple[object, ...] = (
                latest_sequence,
                latest_sequence,
                session_id,
                limit,
            )
        else:
            where = ""
            parameters = (limit,)
        rows = self._connection.execute(
            f"""
            WITH chat_rows AS (
                SELECT
                    sessions.session_id,
                    sessions.repo_root,
                    sessions.title,
                    sessions.created_at,
                    sessions.updated_at,
                    COALESCE((
                        SELECT events.sequence
                        FROM events
                        WHERE events.session_id = sessions.session_id
                        ORDER BY events.sequence DESC
                        LIMIT 1
                    ), 0) AS latest_sequence
                FROM sessions
            )
            SELECT * FROM chat_rows
            {where}
            ORDER BY latest_sequence DESC, session_id
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        return tuple(self._chat_from_row(row) for row in rows)

    def session_exists(self, session_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (_require_string(session_id, "session_id"),),
        ).fetchone()
        return row is not None

    def session_repo_root(self, session_id: str) -> str | None:
        """Return the exact repository bound to a persisted browser session."""
        row = self._connection.execute(
            "SELECT repo_root FROM sessions WHERE session_id = ?",
            (_require_string(session_id, "session_id"),),
        ).fetchone()
        return None if row is None else str(row["repo_root"])

    def create_planning_task(self, session_id: str, task_id: str) -> TaskRecord:
        self._connection.execute(
            """
            INSERT INTO tasks (task_id, revision, session_id, state, brief_json)
            VALUES (?, 0, ?, ?, NULL)
            """,
            (_require_string(task_id, "task_id"), _require_string(session_id, "session_id"), TaskState.FABLE_PLANNING.value),
        )
        return self.get_task(task_id, 0)

    def save_task(self, session_id: str, brief: TaskBrief, state: TaskState) -> TaskRecord:
        if not isinstance(brief, TaskBrief):
            raise ValueError("brief must be a TaskBrief")
        if brief.revision < 1:
            raise ValueError("task revision must be >= 1")
        if not isinstance(state, TaskState):
            raise ValueError("state must be a TaskState")
        _require_string(session_id, "session_id")
        with self._immediate_transaction():
            row = self._connection.execute(
                "SELECT MAX(revision) AS latest_revision FROM tasks WHERE task_id = ?",
                (brief.task_id,),
            ).fetchone()
            latest = None if row is None else row["latest_revision"]
            expected_revision = 1 if latest is None else int(latest) + 1
            if brief.revision != expected_revision:
                raise ValueError(f"task revision must be the next revision ({expected_revision})")
            self._connection.execute(
                """
                INSERT INTO tasks (task_id, revision, session_id, state, brief_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (brief.task_id, brief.revision, session_id, state.value, _encode_json(brief.to_dict())),
            )
        return self.get_task(brief.task_id, brief.revision)

    def get_task(self, task_id: str, revision: int) -> TaskRecord:
        row = self._connection.execute(
            "SELECT * FROM tasks WHERE task_id = ? AND revision = ?",
            (_require_string(task_id, "task_id"), _require_integer(revision, "revision")),
        ).fetchone()
        if row is None:
            raise RuntimeError("task record not found")
        return self._task_from_row(row)

    def task(self, task_id: str, revision: int) -> TaskRecord:
        """Compatibility-friendly spelling for retrieving one exact revision."""
        return self.get_task(task_id, revision)

    def latest_task(self, task_id: str) -> TaskRecord | None:
        row = self._connection.execute(
            "SELECT * FROM tasks WHERE task_id = ? ORDER BY revision DESC LIMIT 1",
            (_require_string(task_id, "task_id"),),
        ).fetchone()
        return None if row is None else self._task_from_row(row)

    def latest_task_overviews(self, session_id: str) -> tuple[TaskOverview, ...]:
        """Return one latest revision per task plus browser-safe activity metadata."""
        session_id = _require_string(session_id, "session_id")
        rows = self._connection.execute(
            """
            WITH latest AS (
                SELECT task_id, MAX(revision) AS revision
                FROM tasks
                WHERE session_id = ?
                GROUP BY task_id
            ),
            bounded_tasks AS (
              SELECT
                task.*,
                recent_event.created_at AS overview_updated_at,
                recent_event.sequence AS overview_sequence,
                (
                    SELECT run.agent
                    FROM agent_runs AS run
                    WHERE run.task_id = task.task_id
                      AND run.revision = task.revision
                      AND run.status = 'running'
                    LIMIT 1
                ) AS overview_active_agent,
                (
                    SELECT run.started_at
                    FROM agent_runs AS run
                    WHERE run.task_id = task.task_id
                      AND run.revision = task.revision
                      AND run.status = 'running'
                    LIMIT 1
                ) AS overview_active_started_at
              FROM tasks AS task
              JOIN latest
                ON latest.task_id = task.task_id
               AND latest.revision = task.revision
              LEFT JOIN events AS recent_event
                ON recent_event.sequence = (
                    SELECT event.sequence
                    FROM events AS event
                    WHERE event.session_id = task.session_id
                      AND event.task_id = task.task_id
                    ORDER BY event.sequence DESC
                    LIMIT 1
                )
              WHERE task.session_id = ?
              ORDER BY COALESCE(recent_event.sequence, 0) DESC, task.task_id
              LIMIT ?
            ),
            bounded_with_revision AS (
              SELECT
                bounded_tasks.*,
                (
                    SELECT event.sequence
                    FROM events AS event
                    WHERE event.session_id = bounded_tasks.session_id
                      AND event.task_id = bounded_tasks.task_id
                      AND event.kind = 'task_brief'
                      AND json_type(event.payload_json, '$.brief.revision') = 'integer'
                      AND json_extract(event.payload_json, '$.brief.revision') = bounded_tasks.revision
                    ORDER BY event.sequence DESC
                    LIMIT 1
                ) AS revision_start_sequence
              FROM bounded_tasks
            )
            SELECT
              bounded_with_revision.*,
              (
                SELECT event.payload_json
                FROM events AS event
                WHERE event.session_id = bounded_with_revision.session_id
                  AND event.task_id = bounded_with_revision.task_id
                  AND bounded_with_revision.revision_start_sequence IS NOT NULL
                  AND event.sequence > bounded_with_revision.revision_start_sequence
                  AND event.kind = 'outcome'
                ORDER BY event.sequence DESC
                LIMIT 1
              ) AS outcome_payload_json,
              (
                SELECT event.payload_json
                FROM events AS event
                WHERE event.session_id = bounded_with_revision.session_id
                  AND event.task_id = bounded_with_revision.task_id
                  AND bounded_with_revision.revision_start_sequence IS NOT NULL
                  AND event.sequence > bounded_with_revision.revision_start_sequence
                  AND event.kind = 'review'
                ORDER BY event.sequence DESC
                LIMIT 1
              ) AS review_payload_json,
              (
                SELECT event.payload_json
                FROM events AS event
                WHERE event.session_id = bounded_with_revision.session_id
                  AND event.task_id = bounded_with_revision.task_id
                  AND bounded_with_revision.revision_start_sequence IS NOT NULL
                  AND event.sequence > bounded_with_revision.revision_start_sequence
                  AND event.kind = 'clarification'
                ORDER BY event.sequence DESC
                LIMIT 1
              ) AS clarification_payload_json,
              (
                SELECT event.payload_json
                FROM events AS event
                WHERE event.session_id = bounded_with_revision.session_id
                  AND event.task_id = bounded_with_revision.task_id
                  AND bounded_with_revision.revision_start_sequence IS NOT NULL
                  AND event.sequence > bounded_with_revision.revision_start_sequence
                  AND event.kind IN (
                    'agent_event', 'resume_drift', 'stop_error', 'action_error'
                  )
                ORDER BY event.sequence DESC
                LIMIT 1
              ) AS activity_payload_json
            FROM bounded_with_revision
            ORDER BY COALESCE(overview_sequence, 0) DESC, task_id
            """,
            (session_id, session_id, MAX_TASK_OVERVIEWS),
        ).fetchall()
        return tuple(
            TaskOverview(
                task=self._task_from_row(row),
                updated_at=row["overview_updated_at"],
                active_agent=row["overview_active_agent"],
                active_started_at=row["overview_active_started_at"],
                revision_start_sequence=(
                    None
                    if row["revision_start_sequence"] is None
                    else int(row["revision_start_sequence"])
                ),
                outcome=(
                    None
                    if row["outcome_payload_json"] is None
                    else _decode_mapping(row["outcome_payload_json"], "outcome payload")
                ),
                review=(
                    None
                    if row["review_payload_json"] is None
                    else _decode_mapping(row["review_payload_json"], "review payload")
                ),
                clarification=(
                    None
                    if row["clarification_payload_json"] is None
                    else _decode_mapping(
                        row["clarification_payload_json"], "clarification payload"
                    )
                ),
                activity=(
                    None
                    if row["activity_payload_json"] is None
                    else _decode_mapping(row["activity_payload_json"], "activity payload")
                ),
            )
            for row in rows
        )

    def task_brief(self, task_id: str, revision: int) -> TaskBrief:
        task = self.get_task(task_id, revision)
        if task.brief is None:
            raise RuntimeError("revision-zero task has no brief")
        return task.brief

    def transition_task(
        self,
        task_id: str,
        revision: int,
        *,
        expected: TaskState,
        target: TaskState,
    ) -> TaskRecord:
        if not isinstance(expected, TaskState) or not isinstance(target, TaskState):
            raise ValueError("expected and target must be TaskState values")
        require_transition(expected, target)
        task_id = _require_string(task_id, "task_id")
        revision = _require_integer(revision, "revision")
        with self._immediate_transaction():
            row = self._connection.execute(
                """
                SELECT brief_json, continuation_state FROM tasks
                WHERE task_id = ? AND revision = ? AND state = ?
                """,
                (task_id, revision, expected.value),
            ).fetchone()
            if row is None:
                raise RuntimeError("task state changed concurrently")
            if revision == 0 and row["brief_json"] is None and target not in {
                TaskState.FAILED,
                TaskState.INTERRUPTED,
            }:
                raise RuntimeError("revision-zero task has no brief")
            clear_continuation = row["continuation_state"] is not None and target is TaskState.FAILED
            if row["continuation_state"] is not None and not clear_continuation:
                raise RuntimeError("task has continuation state; use resume_continuation")
            if clear_continuation:
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET state = ?, continuation_state = NULL, pending_json = NULL
                    WHERE task_id = ? AND revision = ? AND state = ?
                    """,
                    (target.value, task_id, revision, expected.value),
                )
            else:
                cursor = self._connection.execute(
                    """
                    UPDATE tasks SET state = ?
                    WHERE task_id = ? AND revision = ? AND state = ?
                    """,
                    (target.value, task_id, revision, expected.value),
                )
            if cursor.rowcount != 1:
                raise RuntimeError("task state changed concurrently")
        return self.get_task(task_id, revision)

    def transition_task_clearing_pending(
        self,
        task_id: str,
        revision: int,
        *,
        expected: TaskState,
        target: TaskState,
    ) -> TaskRecord:
        """Route a completed read-only agent phase and consume its context atomically."""
        require_transition(expected, target)
        task_id = _require_string(task_id, "task_id")
        revision = _require_integer(revision, "revision")
        with self._immediate_transaction():
            cursor = self._connection.execute(
                """
                UPDATE tasks SET state = ?, pending_json = NULL
                WHERE task_id = ? AND revision = ? AND state = ?
                  AND continuation_state IS NULL AND pending_json IS NOT NULL
                """,
                (target.value, task_id, revision, expected.value),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("task has no routable pending context")
        return self.get_task(task_id, revision)

    def pause_for_continuation(
        self,
        task_id: str,
        revision: int,
        *,
        expected: TaskState,
        target: TaskState,
        continuation_state: TaskState,
        pending: Mapping[str, object],
    ) -> TaskRecord:
        if not isinstance(continuation_state, TaskState):
            raise ValueError("continuation_state must be a TaskState")
        require_transition(expected, target)
        require_transition(target, continuation_state)
        frozen_pending = freeze_json(pending)
        if not isinstance(frozen_pending, Mapping):
            raise ValueError("pending must be an object")
        task_id = _require_string(task_id, "task_id")
        revision = _require_integer(revision, "revision")
        with self._immediate_transaction():
            row = self._connection.execute(
                """
                SELECT brief_json, continuation_state, pending_json FROM tasks
                WHERE task_id = ? AND revision = ? AND state = ?
                """,
                (task_id, revision, expected.value),
            ).fetchone()
            if row is None:
                raise RuntimeError("task state changed concurrently")
            if revision == 0 and row["brief_json"] is None and target not in {
                TaskState.FAILED,
                TaskState.INTERRUPTED,
            }:
                raise RuntimeError("revision-zero task has no brief")
            if row["continuation_state"] is not None or row["pending_json"] is not None:
                raise RuntimeError("task continuation context already exists")
            cursor = self._connection.execute(
                """
                UPDATE tasks
                SET state = ?, continuation_state = ?, pending_json = ?
                WHERE task_id = ? AND revision = ? AND state = ?
                  AND continuation_state IS NULL AND pending_json IS NULL
                """,
                (
                    target.value,
                    continuation_state.value,
                    _encode_json(frozen_pending),
                    task_id,
                    revision,
                    expected.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("task state changed concurrently")
        return self.get_task(task_id, revision)

    def replace_pending_for_continuation(
        self,
        task_id: str,
        revision: int,
        *,
        expected: TaskState,
        target: TaskState,
        continuation_state: TaskState,
        pending: Mapping[str, object],
    ) -> TaskRecord:
        """Pause a routed agent verdict while replacing its resumable context."""
        require_transition(expected, target)
        require_transition(target, continuation_state)
        frozen_pending = freeze_json(pending)
        if not isinstance(frozen_pending, Mapping):
            raise ValueError("pending must be an object")
        task_id = _require_string(task_id, "task_id")
        revision = _require_integer(revision, "revision")
        with self._immediate_transaction():
            cursor = self._connection.execute(
                """
                UPDATE tasks
                SET state = ?, continuation_state = ?, pending_json = ?
                WHERE task_id = ? AND revision = ? AND state = ?
                  AND continuation_state IS NULL AND pending_json IS NOT NULL
                """,
                (
                    target.value,
                    continuation_state.value,
                    _encode_json(frozen_pending),
                    task_id,
                    revision,
                    expected.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("task has no replaceable pending context")
        return self.get_task(task_id, revision)

    def retarget_continuation(
        self,
        task_id: str,
        revision: int,
        *,
        expected: TaskState,
        target: TaskState,
        pending: Mapping[str, object],
    ) -> TaskRecord:
        """Move a paused task while preserving its exact continuation state."""
        require_transition(expected, target)
        frozen_pending = freeze_json(pending)
        if not isinstance(frozen_pending, Mapping):
            raise ValueError("pending must be an object")
        task_id = _require_string(task_id, "task_id")
        revision = _require_integer(revision, "revision")
        with self._immediate_transaction():
            row = self._connection.execute(
                """
                SELECT continuation_state FROM tasks
                WHERE task_id = ? AND revision = ? AND state = ?
                """,
                (task_id, revision, expected.value),
            ).fetchone()
            if row is None:
                raise RuntimeError("task state changed concurrently")
            raw_continuation = row["continuation_state"]
            if raw_continuation is None:
                raise RuntimeError("task has no continuation state")
            continuation = TaskState(raw_continuation)
            require_transition(target, continuation)
            cursor = self._connection.execute(
                """
                UPDATE tasks SET state = ?, pending_json = ?
                WHERE task_id = ? AND revision = ? AND state = ?
                  AND continuation_state = ?
                """,
                (
                    target.value,
                    _encode_json(frozen_pending),
                    task_id,
                    revision,
                    expected.value,
                    continuation.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("task state changed concurrently")
        return self.get_task(task_id, revision)

    def set_pending_context(
        self,
        task_id: str,
        revision: int,
        *,
        expected: TaskState,
        pending: Mapping[str, object],
    ) -> TaskRecord:
        """Persist resumable context for one active read-only agent phase."""
        frozen_pending = freeze_json(pending)
        if not isinstance(frozen_pending, Mapping):
            raise ValueError("pending must be an object")
        cursor = self._connection.execute(
            """
            UPDATE tasks SET pending_json = ?
            WHERE task_id = ? AND revision = ? AND state = ?
              AND continuation_state IS NULL AND pending_json IS NULL
            """,
            (
                _encode_json(frozen_pending),
                _require_string(task_id, "task_id"),
                _require_integer(revision, "revision"),
                expected.value,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("task cannot accept pending context")
        return self.get_task(task_id, revision)

    def clear_pending_context(
        self, task_id: str, revision: int, *, expected: TaskState,
    ) -> TaskRecord:
        cursor = self._connection.execute(
            """
            UPDATE tasks SET pending_json = NULL
            WHERE task_id = ? AND revision = ? AND state = ?
              AND continuation_state IS NULL AND pending_json IS NOT NULL
            """,
            (
                _require_string(task_id, "task_id"),
                _require_integer(revision, "revision"),
                expected.value,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("task has no clearable pending context")
        return self.get_task(task_id, revision)

    def resume_nested_continuation(
        self,
        task_id: str,
        revision: int,
        *,
        active_state: TaskState,
        continuation_state: TaskState,
        pending: Mapping[str, object],
    ) -> TaskRecord:
        """Resume an interrupted Fable phase with its underlying continuation."""
        require_transition(TaskState.INTERRUPTED, active_state)
        require_transition(active_state, continuation_state)
        frozen_pending = freeze_json(pending)
        if not isinstance(frozen_pending, Mapping):
            raise ValueError("pending must be an object")
        with self._immediate_transaction():
            cursor = self._connection.execute(
                """
                UPDATE tasks
                SET state = ?, continuation_state = ?, pending_json = ?
                WHERE task_id = ? AND revision = ? AND state = ?
                  AND continuation_state = ?
                """,
                (
                    active_state.value,
                    continuation_state.value,
                    _encode_json(frozen_pending),
                    _require_string(task_id, "task_id"),
                    _require_integer(revision, "revision"),
                    TaskState.INTERRUPTED.value,
                    active_state.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("interrupted continuation changed concurrently")
        return self.get_task(task_id, revision)

    def resume_continuation(
        self,
        task_id: str,
        revision: int,
        *,
        expected: TaskState,
    ) -> TaskRecord:
        if not isinstance(expected, TaskState):
            raise ValueError("expected must be a TaskState")
        task_id = _require_string(task_id, "task_id")
        revision = _require_integer(revision, "revision")
        with self._immediate_transaction():
            row = self._connection.execute(
                "SELECT continuation_state FROM tasks WHERE task_id = ? AND revision = ? AND state = ?",
                (task_id, revision, expected.value),
            ).fetchone()
            if row is None:
                raise RuntimeError("task state changed concurrently")
            raw_target = row["continuation_state"]
            if raw_target is None:
                raise RuntimeError("task has no continuation state")
            target = TaskState(raw_target)
            require_transition(expected, target)
            cursor = self._connection.execute(
                """
                UPDATE tasks
                SET state = ?, continuation_state = NULL, pending_json = NULL
                WHERE task_id = ? AND revision = ? AND state = ? AND continuation_state = ?
                """,
                (target.value, task_id, revision, expected.value, target.value),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("task state changed concurrently")
        return self.get_task(task_id, revision)

    def mark_interrupted(
        self,
        task_id: str,
        revision: int,
        *,
        continuation: TaskState,
        cli_session_id: str | None = None,
    ) -> TaskRecord:
        if not isinstance(continuation, TaskState):
            raise ValueError("continuation must be a TaskState")
        task = self.get_task(task_id, revision)
        require_transition(task.state, TaskState.INTERRUPTED)
        require_transition(TaskState.INTERRUPTED, continuation)
        with self._immediate_transaction():
            cursor = self._connection.execute(
                """
                UPDATE tasks
                SET state = ?, continuation_state = ?,
                    fable_session_id = COALESCE(?, fable_session_id)
                WHERE task_id = ? AND revision = ? AND state = ?
                """,
                (
                    TaskState.INTERRUPTED.value,
                    continuation.value,
                    cli_session_id,
                    _require_string(task_id, "task_id"),
                    _require_integer(revision, "revision"),
                    task.state.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("task state changed concurrently")
        return self.get_task(task_id, revision)

    def approve_task(
        self,
        task_id: str,
        revision: int,
        *,
        baseline_id: str,
        expected: TaskState = TaskState.AWAITING_USER_APPROVAL,
    ) -> TaskRecord:
        self.task_brief(task_id, revision)
        if expected not in {
            TaskState.AWAITING_USER_APPROVAL,
            TaskState.AWAITING_SCOPE_APPROVAL,
        }:
            raise ValueError("approval state must await user or scope approval")
        with self._immediate_transaction():
            cursor = self._connection.execute(
                """
                UPDATE tasks SET approved_at = ?, baseline_id = ?
                WHERE task_id = ? AND revision = ? AND state = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM tasks AS newer
                      WHERE newer.task_id = ? AND newer.revision > ?
                  )
                """,
                (
                    self._timestamp(),
                    _require_string(baseline_id, "baseline_id"),
                    _require_string(task_id, "task_id"),
                    _require_integer(revision, "revision"),
                    expected.value,
                    task_id,
                    revision,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("task state changed concurrently")
        return self.get_task(task_id, revision)

    def approve_task_with_setting(
        self,
        task_id: str,
        revision: int,
        *,
        brief: TaskBrief,
        baseline_id: str,
        expected: TaskState,
        setting: tuple[str, object],
    ) -> TaskRecord:
        """Atomically attach one initial baseline and its durable manifest."""
        if not isinstance(brief, TaskBrief):
            raise ValueError("brief must be a TaskBrief")
        task_id = _require_string(task_id, "task_id")
        revision = _require_integer(revision, "revision")
        if brief.task_id != task_id or brief.revision != revision:
            raise ValueError("approval brief must match the exact task revision")
        if expected not in {
            TaskState.AWAITING_USER_APPROVAL,
            TaskState.AWAITING_SCOPE_APPROVAL,
        }:
            raise ValueError("approval state must await user or scope approval")
        if not isinstance(setting, tuple) or len(setting) != 2:
            raise ValueError("setting must be a key/value pair")
        setting_key = _require_string(setting[0], "setting key")
        encoded_setting = _encode_json(setting[1])
        encoded_brief = _encode_json(brief.to_dict())
        approved_at = self._timestamp()
        baseline_id = _require_string(baseline_id, "baseline_id")
        with self._immediate_transaction():
            self._connection.execute(
                """
                INSERT INTO settings (key, value_json) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                """,
                (setting_key, encoded_setting),
            )
            cursor = self._connection.execute(
                """
                UPDATE tasks SET approved_at = ?, baseline_id = ?
                WHERE task_id = ? AND revision = ? AND state = ?
                  AND brief_json = ? AND approved_at IS NULL AND baseline_id IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM tasks AS newer
                      WHERE newer.task_id = ? AND newer.revision > ?
                  )
                """,
                (
                    approved_at,
                    baseline_id,
                    task_id,
                    revision,
                    expected.value,
                    encoded_brief,
                    task_id,
                    revision,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "task approval identity changed concurrently"
                )
            approved = self.get_task(task_id, revision)
        return approved

    def save_scope_revision(
        self,
        session_id: str,
        brief: TaskBrief,
        *,
        fable_session_id: str,
        sol_thread_id: str,
        correction_count: int,
        continuation_state: TaskState,
        pending: Mapping[str, object],
        baseline_id: str,
        setting: tuple[str, object] | None = None,
    ) -> TaskRecord:
        """Persist a revised scope together with its resumable task identity."""
        if correction_count < 0:
            raise ValueError("correction_count must be >= 0")
        require_transition(TaskState.AWAITING_SCOPE_APPROVAL, continuation_state)
        frozen_pending = freeze_json(pending)
        if not isinstance(frozen_pending, Mapping):
            raise ValueError("pending must be an object")
        _require_string(session_id, "session_id")
        _require_string(fable_session_id, "fable_session_id")
        _require_string(sol_thread_id, "sol_thread_id")
        _require_string(baseline_id, "baseline_id")
        encoded_setting: tuple[str, str] | None = None
        if setting is not None:
            if not isinstance(setting, tuple) or len(setting) != 2:
                raise ValueError("setting must be a key/value pair")
            encoded_setting = (
                _require_string(setting[0], "setting key"),
                _encode_json(setting[1]),
            )
        with self._immediate_transaction():
            row = self._connection.execute(
                "SELECT MAX(revision) AS latest_revision FROM tasks WHERE task_id = ?",
                (brief.task_id,),
            ).fetchone()
            latest = None if row is None else row["latest_revision"]
            expected_revision = 1 if latest is None else int(latest) + 1
            if brief.revision != expected_revision:
                raise ValueError(f"task revision must be the next revision ({expected_revision})")
            self._connection.execute(
                """
                INSERT INTO tasks (
                    task_id, revision, session_id, state, brief_json,
                    fable_session_id, sol_thread_id, correction_count,
                    continuation_state, pending_json, baseline_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    brief.task_id,
                    brief.revision,
                    session_id,
                    TaskState.AWAITING_SCOPE_APPROVAL.value,
                    _encode_json(brief.to_dict()),
                    fable_session_id,
                    sol_thread_id,
                    correction_count,
                    continuation_state.value,
                    _encode_json(frozen_pending),
                    baseline_id,
                ),
            )
            if encoded_setting is not None:
                self._connection.execute(
                    """
                    INSERT INTO settings (key, value_json) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                    """,
                    encoded_setting,
                )
        return self.get_task(brief.task_id, brief.revision)

    def save_edited_revision(
        self,
        session_id: str,
        brief: TaskBrief,
        *,
        fable_session_id: str,
        sol_thread_id: str | None,
        baseline_id: str | None,
        correction_count: int,
        continuation_state: TaskState | None,
        pending: Mapping[str, object] | None,
        setting: tuple[str, object] | None = None,
    ) -> TaskRecord:
        """Create an unapproved edit without losing exact review/continuation IDs."""
        if correction_count < 0:
            raise ValueError("correction_count must be >= 0")
        _require_string(session_id, "session_id")
        _require_string(fable_session_id, "fable_session_id")
        if sol_thread_id is not None:
            _require_string(sol_thread_id, "sol_thread_id")
        if baseline_id is not None:
            _require_string(baseline_id, "baseline_id")
        if (continuation_state is None) != (pending is None):
            raise ValueError("continuation_state and pending must both be present or absent")
        if continuation_state is not None:
            require_transition(TaskState.AWAITING_USER_APPROVAL, continuation_state)
        frozen_pending = None if pending is None else freeze_json(pending)
        if frozen_pending is not None and not isinstance(frozen_pending, Mapping):
            raise ValueError("pending must be an object")
        encoded_setting: tuple[str, str] | None = None
        if setting is not None:
            if not isinstance(setting, tuple) or len(setting) != 2:
                raise ValueError("setting must be a key/value pair")
            encoded_setting = (
                _require_string(setting[0], "setting key"),
                _encode_json(setting[1]),
            )
        with self._immediate_transaction():
            row = self._connection.execute(
                "SELECT MAX(revision) AS latest_revision FROM tasks WHERE task_id = ?",
                (brief.task_id,),
            ).fetchone()
            latest = None if row is None else row["latest_revision"]
            expected_revision = 1 if latest is None else int(latest) + 1
            if brief.revision != expected_revision:
                raise ValueError(f"task revision must be the next revision ({expected_revision})")
            self._connection.execute(
                """
                INSERT INTO tasks (
                    task_id, revision, session_id, state, brief_json,
                    fable_session_id, sol_thread_id, baseline_id, correction_count,
                    continuation_state, pending_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    brief.task_id,
                    brief.revision,
                    session_id,
                    TaskState.AWAITING_USER_APPROVAL.value,
                    _encode_json(brief.to_dict()),
                    fable_session_id,
                    sol_thread_id,
                    baseline_id,
                    correction_count,
                    None if continuation_state is None else continuation_state.value,
                    None if frozen_pending is None else _encode_json(frozen_pending),
                ),
            )
            if encoded_setting is not None:
                self._connection.execute(
                    """
                    INSERT INTO settings (key, value_json) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
                    """,
                    encoded_setting,
                )
        return self.get_task(brief.task_id, brief.revision)

    def set_fable_session(self, task_id: str, revision: int, session_id: str) -> TaskRecord:
        return self._set_task_text(task_id, revision, "fable_session_id", session_id)

    def set_sol_thread(self, task_id: str, revision: int, thread_id: str) -> TaskRecord:
        return self._set_task_text(task_id, revision, "sol_thread_id", thread_id)

    def _set_task_text(self, task_id: str, revision: int, column: str, value: str) -> TaskRecord:
        cursor = self._connection.execute(
            f"UPDATE tasks SET {column} = ? WHERE task_id = ? AND revision = ?",
            (_require_string(value, column), _require_string(task_id, "task_id"), _require_integer(revision, "revision")),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("task record not found")
        return self.get_task(task_id, revision)

    def increment_correction_count(self, task_id: str, revision: int) -> TaskRecord:
        cursor = self._connection.execute(
            """
            UPDATE tasks SET correction_count = correction_count + 1
            WHERE task_id = ? AND revision = ?
            """,
            (_require_string(task_id, "task_id"), _require_integer(revision, "revision")),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("task record not found")
        return self.get_task(task_id, revision)

    def append_event(
        self,
        session_id: str,
        task_id: str | None,
        actor: str,
        kind: str,
        payload: Mapping[str, object],
    ) -> StreamEvent:
        frozen_payload = freeze_json(payload)
        if not isinstance(frozen_payload, Mapping):
            raise ValueError("payload must be an object")
        session_id = _require_string(session_id, "session_id")
        task_id = None if task_id is None else _require_string(task_id, "task_id")
        actor = _require_string(actor, "actor")
        kind = _require_string(kind, "kind")
        created_at = self._timestamp()
        values = (
            session_id,
            task_id,
            actor,
            kind,
            _encode_json(frozen_payload),
            created_at,
        )
        with self._event_listener_lock:
            with self._immediate_transaction():
                cursor = self._connection.execute(
                    """
                    INSERT INTO events (
                        session_id, task_id, actor, kind, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                row = self._connection.execute(
                    "SELECT * FROM events WHERE sequence = ?", (cursor.lastrowid,)
                ).fetchone()
                if row is None:
                    raise RuntimeError("inserted event could not be read")
                event = self._event_from_row(row)
                title = self._first_user_message_title(
                    session_id=session_id,
                    sequence=event.sequence,
                    actor=actor,
                    kind=kind,
                    payload=frozen_payload,
                )
                if title is None:
                    self._connection.execute(
                        "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                        (created_at, session_id),
                    )
                else:
                    self._connection.execute(
                        """
                        UPDATE sessions
                        SET title = ?, updated_at = ?
                        WHERE session_id = ?
                        """,
                        (title, created_at, session_id),
                    )
            self._pending_listener_events.append(event)
            if self._dispatching_listener_events:
                return event
            self._dispatching_listener_events = True
        self._drain_event_listeners()
        return event

    def _first_user_message_title(
        self,
        *,
        session_id: str,
        sequence: int,
        actor: str,
        kind: str,
        payload: Mapping[str, JsonValue],
    ) -> str | None:
        if actor != "user" or kind != "message":
            return None
        earlier = self._connection.execute(
            """
            SELECT 1 FROM events
            WHERE session_id = ?
              AND actor = 'user'
              AND kind = 'message'
              AND sequence < ?
            LIMIT 1
            """,
            (session_id, sequence),
        ).fetchone()
        if earlier is not None:
            return None
        raw_text = payload.get("text")
        if not isinstance(raw_text, str):
            return _NEW_CHAT_TITLE
        title = " ".join(raw_text.split())
        return _NEW_CHAT_TITLE if not title else title[:MAX_CHAT_TITLE_LENGTH]

    def events_after(
        self,
        session_id: str,
        sequence: int,
        *,
        limit: int | None = None,
    ) -> tuple[StreamEvent, ...]:
        session_id = _require_string(session_id, "session_id")
        sequence = _require_integer(sequence, "sequence")
        if limit is None:
            query = (
                "SELECT * FROM events "
                "WHERE session_id = ? AND sequence > ? ORDER BY sequence"
            )
            parameters = (session_id, sequence)
        else:
            limit = _require_integer(limit, "limit")
            if not 1 <= limit <= EVENT_REPLAY_PAGE_SIZE:
                raise ValueError(
                    f"limit must be between 1 and {EVENT_REPLAY_PAGE_SIZE}"
                )
            query = (
                "SELECT * FROM events "
                "WHERE session_id = ? AND sequence > ? ORDER BY sequence LIMIT ?"
            )
            parameters = (session_id, sequence, limit)
        rows = self._connection.execute(query, parameters).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def browser_replay_floor(self, session_id: str) -> int:
        """Return the cursor immediately before the recent browser replay window."""
        row = self._connection.execute(
            """
            SELECT sequence
            FROM events
            WHERE session_id = ?
            ORDER BY sequence DESC
            LIMIT 1 OFFSET ?
            """,
            (
                _require_string(session_id, "session_id"),
                MAX_INITIAL_REPLAY_EVENTS,
            ),
        ).fetchone()
        return 0 if row is None else int(row["sequence"])

    def start_agent_run(
        self,
        run_id: str,
        task_id: str,
        revision: int,
        agent: str,
    ) -> AgentRunRecord:
        try:
            self._connection.execute(
                """
                INSERT INTO agent_runs
                (run_id, task_id, revision, agent, started_at, status)
                VALUES (?, ?, ?, ?, ?, 'running')
                """,
                (
                    _require_string(run_id, "run_id"),
                    _require_string(task_id, "task_id"),
                    _require_integer(revision, "revision"),
                    _require_string(agent, "agent"),
                    self._timestamp(),
                ),
            )
        except sqlite3.IntegrityError as error:
            if "agent_runs.task_id, agent_runs.revision" in str(error):
                raise RuntimeError("task already has an active agent run") from error
            raise
        return self.agent_run(run_id)

    def set_agent_run_process(
        self,
        run_id: str,
        *,
        pid: int,
        process_group_id: int,
        cli_session_id: str | None = None,
    ) -> AgentRunRecord:
        cursor = self._connection.execute(
            """
            UPDATE agent_runs
            SET pid = ?, process_group_id = ?, cli_session_id = COALESCE(?, cli_session_id)
            WHERE run_id = ? AND status = 'running'
            """,
            (
                _require_integer(pid, "pid"),
                _require_integer(process_group_id, "process_group_id"),
                None if cli_session_id is None else _require_string(cli_session_id, "cli_session_id"),
                _require_string(run_id, "run_id"),
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("agent run is not active")
        return self.agent_run(run_id)

    def set_agent_run_session(self, run_id: str, cli_session_id: str) -> AgentRunRecord:
        cursor = self._connection.execute(
            "UPDATE agent_runs SET cli_session_id = ? WHERE run_id = ? AND status = 'running'",
            (_require_string(cli_session_id, "cli_session_id"), _require_string(run_id, "run_id")),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("agent run is not active")
        return self.agent_run(run_id)

    def finish_agent_run(self, run_id: str, *, status: str, exit_code: int) -> AgentRunRecord:
        if status not in _TERMINAL_RUN_STATUSES:
            raise ValueError("status must be a terminal agent-run status")
        cursor = self._connection.execute(
            """
            UPDATE agent_runs SET status = ?, exit_code = ?, ended_at = ?
            WHERE run_id = ? AND status = 'running'
            """,
            (status, _require_integer(exit_code, "exit_code"), self._timestamp(), _require_string(run_id, "run_id")),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("agent run already finished or not found")
        return self.agent_run(run_id)

    def agent_run(self, run_id: str) -> AgentRunRecord:
        row = self._connection.execute(
            "SELECT * FROM agent_runs WHERE run_id = ?", (_require_string(run_id, "run_id"),)
        ).fetchone()
        if row is None:
            raise RuntimeError("agent run not found")
        return self._agent_run_from_row(row)

    def active_run_for_task(self, task_id: str, revision: int) -> AgentRunRecord | None:
        row = self._connection.execute(
            """
            SELECT * FROM agent_runs
            WHERE task_id = ? AND revision = ? AND status = 'running'
            """,
            (_require_string(task_id, "task_id"), _require_integer(revision, "revision")),
        ).fetchone()
        return None if row is None else self._agent_run_from_row(row)

    def recover_active_tasks(self) -> tuple[TaskRecord, ...]:
        """Atomically retire process-local work left active by an old server.

        Persisted PIDs and process groups are inert audit data. Startup recovery
        never inspects or signals them because ownership cannot survive a process
        restart safely.
        """
        active_values = tuple(state.value for state in _ACTIVE_TASK_STATES)
        placeholders = ", ".join("?" for _ in active_values)
        identities: tuple[tuple[str, int], ...]
        with self._immediate_transaction():
            rows = self._connection.execute(
                f"""
                SELECT task.task_id, task.revision
                FROM tasks AS task
                WHERE task.state IN ({placeholders})
                  AND NOT EXISTS (
                      SELECT 1
                      FROM tasks AS newer
                      WHERE newer.task_id = task.task_id
                        AND newer.revision > task.revision
                  )
                ORDER BY task.task_id, task.revision
                """,
                active_values,
            ).fetchall()
            identities = tuple(
                (str(row["task_id"]), int(row["revision"])) for row in rows
            )
            self._connection.execute(
                f"""
                UPDATE tasks AS task
                SET continuation_state = state, state = ?
                WHERE task.state IN ({placeholders})
                  AND NOT EXISTS (
                      SELECT 1
                      FROM tasks AS newer
                      WHERE newer.task_id = task.task_id
                        AND newer.revision > task.revision
                  )
                """,
                (TaskState.INTERRUPTED.value, *active_values),
            )
            self._connection.execute(
                """
                UPDATE agent_runs
                SET status = 'interrupted', ended_at = ?
                WHERE status = 'running'
                """,
                (self._timestamp(),),
            )
        return tuple(self.get_task(task_id, revision) for task_id, revision in identities)

    def set_setting(self, key: str, value: object) -> None:
        self._connection.execute(
            """
            INSERT INTO settings (key, value_json) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json
            """,
            (_require_string(key, "key"), _encode_json(value)),
        )

    def get_setting(self, key: str) -> object | None:
        row = self._connection.execute(
            "SELECT value_json FROM settings WHERE key = ?", (_require_string(key, "key"),)
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError as error:
            raise RuntimeError("persisted setting is invalid JSON") from error

    def audit_legacy_project_ownership(self, canonical_repo_root: str) -> None:
        """Fail closed unless a legacy database belongs to one exact project root."""
        canonical_repo_root = _require_string(canonical_repo_root, "canonical_repo_root")
        self._connection.execute("BEGIN")
        try:
            reasons = self._legacy_project_ownership_reasons(canonical_repo_root)
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.rollback()
        if reasons:
            summary = ", ".join(sorted(reasons)[:_MAX_LEGACY_AUDIT_REASONS])
            raise RuntimeError(f"legacy project ownership audit failed: {summary}")

    def _legacy_project_ownership_reasons(
        self, canonical_repo_root: str,
    ) -> set[str]:
        """Collect generic integrity categories without exposing persisted values."""
        reasons: set[str] = set()
        foreign_key_rows = tuple(self._connection.execute("PRAGMA foreign_key_check"))
        if foreign_key_rows:
            reasons.add("foreign_key_integrity")

        session_rows = tuple(
            self._connection.execute("SELECT session_id, repo_root FROM sessions")
        )
        session_ids: set[str] = set()
        for row in session_rows:
            session_id = row["session_id"]
            repo_root = row["repo_root"]
            if (
                not isinstance(session_id, str)
                or not session_id.strip()
                or not isinstance(repo_root, str)
                or not repo_root.strip()
                or repo_root != canonical_repo_root
            ):
                reasons.add("session_ownership")
                continue
            session_ids.add(session_id)

        active_row = self._connection.execute(
            "SELECT value_json FROM settings WHERE key = ?",
            (_ACTIVE_SESSION_SETTING,),
        ).fetchone()
        if active_row is None:
            reasons.add("active_session")
        else:
            try:
                active_session = json.loads(active_row["value_json"])
            except (TypeError, json.JSONDecodeError):
                reasons.add("active_session")
            else:
                if (
                    not isinstance(active_session, str)
                    or not active_session.strip()
                    or active_session not in session_ids
                ):
                    reasons.add("active_session")

        task_rows = tuple(
            self._connection.execute(
                "SELECT task_id, revision, session_id, baseline_id FROM tasks"
            )
        )
        task_keys: set[tuple[str, int]] = set()
        task_baselines: dict[tuple[str, int], str | None] = {}
        task_sessions: dict[str, str] = {}
        task_revisions: dict[str, list[int]] = {}
        for row in task_rows:
            task_id = row["task_id"]
            revision = row["revision"]
            session_id = row["session_id"]
            baseline_id = row["baseline_id"]
            if (
                not isinstance(task_id, str)
                or not task_id.strip()
                or not isinstance(revision, int)
                or isinstance(revision, bool)
                or revision < 0
                or not isinstance(session_id, str)
                or session_id not in session_ids
                or (baseline_id is not None and (not isinstance(baseline_id, str) or not baseline_id))
            ):
                reasons.add("task_integrity")
                continue
            if task_id in task_sessions and task_sessions[task_id] != session_id:
                reasons.add("task_ownership")
            task_sessions[task_id] = session_id
            task_keys.add((task_id, revision))
            task_baselines[(task_id, revision)] = baseline_id
            task_revisions.setdefault(task_id, []).append(revision)
        for revisions in task_revisions.values():
            ordered = sorted(revisions)
            if ordered != list(range(ordered[0], ordered[-1] + 1)):
                reasons.add("task_revision_integrity")

        for row in self._connection.execute("SELECT session_id, task_id FROM events"):
            session_id = row["session_id"]
            task_id = row["task_id"]
            if not isinstance(session_id, str) or session_id not in session_ids:
                reasons.add("event_ownership")
                continue
            if task_id is not None:
                if not isinstance(task_id, str) or task_sessions.get(task_id) != session_id:
                    reasons.add("event_task_integrity")

        for row in self._connection.execute("SELECT task_id, revision FROM agent_runs"):
            task_id = row["task_id"]
            revision = row["revision"]
            if (
                not isinstance(task_id, str)
                or not isinstance(revision, int)
                or isinstance(revision, bool)
                or (task_id, revision) not in task_keys
            ):
                reasons.add("run_task_integrity")

        baseline_settings: dict[tuple[str, int], str] = {}
        for row in self._connection.execute("SELECT key, value_json FROM settings"):
            key = row["key"]
            if not isinstance(key, str):
                continue
            if key != _BASELINE_SETTING_PREFIX.removesuffix(".") and not key.startswith(
                _BASELINE_SETTING_PREFIX
            ):
                continue
            try:
                persisted = json.loads(row["value_json"])
            except (TypeError, json.JSONDecodeError):
                reasons.add("baseline_integrity")
                continue
            if not isinstance(persisted, dict):
                reasons.add("baseline_integrity")
                continue
            task_id = persisted.get("task_id")
            revision = persisted.get("revision")
            baseline_id = persisted.get("baseline_id")
            manifest = persisted.get("manifest")
            if (
                not isinstance(task_id, str)
                or not task_id.strip()
                or not isinstance(revision, int)
                or isinstance(revision, bool)
                or revision < 0
                or not isinstance(baseline_id, str)
                or not baseline_id
                or not isinstance(manifest, dict)
                or key != f"{_BASELINE_SETTING_PREFIX}{task_id}.{revision}"
                or manifest.get("baseline_id") != baseline_id
                or manifest.get("repo_root") != canonical_repo_root
                or task_baselines.get((task_id, revision)) != baseline_id
            ):
                reasons.add("baseline_integrity")
                continue
            baseline_settings[(task_id, revision)] = baseline_id
        for task_key, baseline_id in task_baselines.items():
            if baseline_id is not None and baseline_settings.get(task_key) != baseline_id:
                reasons.add("baseline_integrity")

        return reasons

    def _chat_from_row(self, row: sqlite3.Row) -> ChatRecord:
        updated_at = row["updated_at"]
        if updated_at is None:
            raise RuntimeError("persisted chat is missing its updated timestamp")
        try:
            return ChatRecord(
                session_id=row["session_id"],
                repo_root=row["repo_root"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=updated_at,
                latest_sequence=row["latest_sequence"],
            )
        except ValueError as error:
            raise RuntimeError("persisted chat is invalid") from error

    def _task_from_row(self, row: sqlite3.Row) -> TaskRecord:
        revision = int(row["revision"])
        raw_brief = row["brief_json"]
        if revision == 0:
            if raw_brief is not None:
                raise RuntimeError("revision-zero task must not have a brief")
            brief = None
        else:
            if raw_brief is None:
                raise RuntimeError("task revision is missing its brief")
            try:
                brief = TaskBrief.from_dict(_decode_mapping(raw_brief, "task brief"))
            except ValueError as error:
                raise RuntimeError("persisted task brief is invalid") from error
            if brief.revision != revision or brief.task_id != row["task_id"]:
                raise RuntimeError("persisted task brief identity does not match its record")
        raw_pending = row["pending_json"]
        pending = None if raw_pending is None else _decode_mapping(raw_pending, "pending context")
        raw_continuation = row["continuation_state"]
        return TaskRecord(
            task_id=row["task_id"],
            revision=revision,
            session_id=row["session_id"],
            state=TaskState(row["state"]),
            brief=brief,
            approved_at=row["approved_at"],
            fable_session_id=row["fable_session_id"],
            sol_thread_id=row["sol_thread_id"],
            baseline_id=row["baseline_id"],
            correction_count=int(row["correction_count"]),
            continuation_state=None if raw_continuation is None else TaskState(raw_continuation),
            pending=pending,
        )

    def _event_from_row(self, row: sqlite3.Row) -> StreamEvent:
        return StreamEvent(
            sequence=int(row["sequence"]),
            session_id=row["session_id"],
            task_id=row["task_id"],
            actor=row["actor"],
            kind=row["kind"],
            payload=_decode_mapping(row["payload_json"], "event payload"),
            created_at=row["created_at"],
        )

    def _agent_run_from_row(self, row: sqlite3.Row) -> AgentRunRecord:
        return AgentRunRecord(
            run_id=row["run_id"],
            task_id=row["task_id"],
            revision=int(row["revision"]),
            agent=row["agent"],
            pid=None if row["pid"] is None else int(row["pid"]),
            process_group_id=None if row["process_group_id"] is None else int(row["process_group_id"]),
            cli_session_id=row["cli_session_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            exit_code=None if row["exit_code"] is None else int(row["exit_code"]),
            status=row["status"],
        )
