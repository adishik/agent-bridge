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
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import threading
from typing import Literal, TypeAlias

from agent_bridge.contracts import (
    ConversationActor,
    ConversationEnvelope,
    ConversationMessageType,
    ConversationTarget,
    DirectedAgentQuestion,
    JsonValue,
    StreamEvent,
    TaskBrief,
    freeze_json,
)
from agent_bridge.projects import project_id_for_root
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
_SOL_TASK_STATES = frozenset({TaskState.SOL_RUNNING, TaskState.SOL_CORRECTING})
MAX_TASK_OVERVIEWS = 200
EVENT_REPLAY_PAGE_SIZE = 100
MAX_INITIAL_REPLAY_EVENTS = 300
MAX_CHAT_TITLE_LENGTH = 80
MAX_CHAT_PAGE_SIZE = 50
_NEW_CHAT_TITLE = "New chat"
_ACTIVE_SESSION_SETTING = "agent_bridge.active_session_id"
_BASELINE_SETTING_PREFIX = "agent_bridge.baseline."
_MAX_LEGACY_AUDIT_REASONS = 8
_MAX_PREPARED_TEXT_LENGTH = 16 * 1024
_MAX_RESUME_DRIFT_SUMMARY_LENGTH = 1024
_MAX_PREPARATION_ID_ATTEMPTS = 8
_STARTUP_RECOVERY_BATCH_SIZE = 128
INITIAL_INTERNAL_EXCHANGES = 3
EXCHANGE_GRANT_SIZE = 3
_PREPARED_ACTION_KINDS = frozenset({"new_request", "approval", "answer", "resume"})
_PREPARED_ACTION_STATUSES = frozenset({
    "PREPARED", "CLAIMED", "COMPLETED", "FAILED", "ABORTED", "INTERRUPTED", "RECOVERED",
})
_SAFE_PREPARED_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")

PreparedActionKind: TypeAlias = Literal["new_request", "approval", "answer", "resume"]
PreparedActionFailureReason: TypeAlias = Literal["nonresumable_failure"]
PreparedActionInterruptionReason: TypeAlias = Literal["stop", "adapter_interrupted"]
COMPATIBILITY_PREPARATION_GENERATION = 0


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
    continuation_generation: int
    exchange_allowance: int
    exchange_consumed: int


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    """One exact directed question and its optional single answer."""

    question_id: str
    session_id: str
    task_id: str
    revision: int
    continuation_generation: int
    asked_by: ConversationActor
    addressed_to: ConversationTarget
    routed_to: ConversationTarget
    text: str
    exchange_id: str | None
    answer_text: str | None
    answered_by: ConversationActor | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", _prepared_identifier(self.question_id, "question_id"))
        object.__setattr__(self, "session_id", _require_string(self.session_id, "session_id"))
        object.__setattr__(self, "task_id", _require_string(self.task_id, "task_id"))
        revision = _require_integer(self.revision, "revision")
        generation = _require_integer(self.continuation_generation, "continuation_generation")
        if revision < 1:
            raise ValueError("revision must be >= 1")
        if generation < 1:
            raise ValueError("continuation_generation must be >= 1")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "continuation_generation", generation)
        if not isinstance(self.asked_by, ConversationActor):
            raise ValueError("asked_by must be a ConversationActor")
        if not isinstance(self.addressed_to, ConversationTarget):
            raise ValueError("addressed_to must be a ConversationTarget")
        if not isinstance(self.routed_to, ConversationTarget):
            raise ValueError("routed_to must be a ConversationTarget")
        object.__setattr__(self, "text", _require_string(self.text, "text"))
        if self.exchange_id is not None:
            object.__setattr__(self, "exchange_id", _prepared_identifier(self.exchange_id, "exchange_id"))
        if (self.answer_text is None) != (self.answered_by is None):
            raise ValueError("answer_text and answered_by must both be present or absent")
        if self.answer_text is not None:
            object.__setattr__(self, "answer_text", _require_string(self.answer_text, "answer_text"))
        if self.answered_by is not None and not isinstance(self.answered_by, ConversationActor):
            raise ValueError("answered_by must be a ConversationActor")


@dataclass(frozen=True, slots=True)
class ExchangeReservation:
    """One durable, idempotent reservation of an internal dialogue slot."""

    exchange_id: str
    question_id: str
    ordinal: int
    continuation_generation: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "exchange_id", _prepared_identifier(self.exchange_id, "exchange_id"))
        object.__setattr__(self, "question_id", _prepared_identifier(self.question_id, "question_id"))
        ordinal = _require_integer(self.ordinal, "ordinal")
        generation = _require_integer(self.continuation_generation, "continuation_generation")
        if ordinal < 1:
            raise ValueError("ordinal must be >= 1")
        if generation < 1:
            raise ValueError("continuation_generation must be >= 1")
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "continuation_generation", generation)


@dataclass(frozen=True, slots=True)
class RecoverySummary:
    """Constant-size durable-transition counts from one startup recovery call."""

    prepared_actions_recovered: int
    tasks_interrupted: int
    agent_runs_interrupted: int

    def __post_init__(self) -> None:
        for name in (
            "prepared_actions_recovered",
            "tasks_interrupted",
            "agent_runs_interrupted",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def _prepared_text(value: object, name: str) -> str:
    text = _require_string(value, name)
    if len(text) > _MAX_PREPARED_TEXT_LENGTH:
        raise ValueError(f"{name} is too long")
    return text


def _prepared_identifier(value: object, name: str) -> str:
    identifier = _require_string(value, name)
    if _SAFE_PREPARED_IDENTIFIER.fullmatch(identifier) is None:
        raise ValueError(f"{name} must be a safe identifier")
    return identifier


@dataclass(frozen=True, slots=True)
class PreparedActionOutcome:
    category: Literal["completed", "stop", "adapter_interrupted", "nonresumable_failure"]

    def __post_init__(self) -> None:
        if self.category not in {
            "completed", "stop", "adapter_interrupted", "nonresumable_failure",
        }:
            raise ValueError("prepared action outcome category is invalid")


@dataclass(frozen=True, slots=True)
class NewRequestPayload:
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _prepared_text(self.text, "text"))


@dataclass(frozen=True, slots=True)
class SolResumeContext:
    sol_thread_id: str
    sol_run_id: str
    prompt: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sol_thread_id", _prepared_identifier(self.sol_thread_id, "sol_thread_id"))
        object.__setattr__(self, "sol_run_id", _prepared_identifier(self.sol_run_id, "sol_run_id"))
        object.__setattr__(self, "prompt", _prepared_text(self.prompt, "prompt"))


@dataclass(frozen=True, slots=True)
class ScopeApprovalContext:
    baseline_id: str
    approved_revision: int
    underlying_continuation: SolResumeContext | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_id", _prepared_identifier(self.baseline_id, "baseline_id"))
        revision = _require_integer(self.approved_revision, "approved_revision")
        if revision < 1:
            raise ValueError("approved_revision must be positive")
        object.__setattr__(self, "approved_revision", revision)
        if self.underlying_continuation is not None and not isinstance(
            self.underlying_continuation, SolResumeContext
        ):
            raise ValueError("scope underlying continuation is invalid")


@dataclass(frozen=True, slots=True)
class BaselineSetting:
    key: str
    value_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _prepared_identifier(self.key, "key"))
        value_json = _prepared_text(self.value_json, "value_json")
        try:
            decoded = json.loads(value_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("value_json must be canonical JSON") from error
        if not isinstance(decoded, Mapping):
            raise ValueError("value_json must encode an object")
        canonical = json.dumps(decoded, separators=(",", ":"), sort_keys=True)
        if canonical != value_json:
            raise ValueError("value_json must be canonical JSON")
        object.__setattr__(self, "value_json", value_json)


@dataclass(frozen=True, slots=True)
class ReviewContext:
    fable_session_id: str
    review_prompt: str
    completion_allowed: bool
    underlying_continuation: ScopeApprovalContext | SolResumeContext

    def __post_init__(self) -> None:
        object.__setattr__(self, "fable_session_id", _prepared_identifier(self.fable_session_id, "fable_session_id"))
        object.__setattr__(self, "review_prompt", _prepared_text(self.review_prompt, "review_prompt"))
        if not isinstance(self.completion_allowed, bool):
            raise ValueError("completion_allowed must be a bool")
        if not isinstance(self.underlying_continuation, (ScopeApprovalContext, SolResumeContext)):
            raise ValueError("review underlying continuation is invalid")


@dataclass(frozen=True, slots=True)
class ClarificationContext:
    fable_session_id: str
    clarification_prompt: str
    underlying_continuation: ScopeApprovalContext | SolResumeContext

    def __post_init__(self) -> None:
        object.__setattr__(self, "fable_session_id", _prepared_identifier(self.fable_session_id, "fable_session_id"))
        object.__setattr__(self, "clarification_prompt", _prepared_text(self.clarification_prompt, "clarification_prompt"))
        if not isinstance(self.underlying_continuation, (ScopeApprovalContext, SolResumeContext)):
            raise ValueError("clarification underlying continuation is invalid")


@dataclass(frozen=True, slots=True)
class AnswerContext:
    answer: str
    underlying_continuation: "PreparedContinuationContext"

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer", _prepared_text(self.answer, "answer"))
        if self.underlying_continuation is not None and not isinstance(
            self.underlying_continuation,
            (ScopeApprovalContext, ReviewContext, ClarificationContext, SolResumeContext),
        ):
            raise ValueError("answer underlying continuation is invalid")


@dataclass(frozen=True, slots=True)
class ResumeDriftProjection:
    status: Literal["unchanged", "drifted"]
    summary: str
    evidence_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {"unchanged", "drifted"}:
            raise ValueError("resume drift status is invalid")
        summary = _prepared_text(self.summary, "summary")
        if (
            len(summary) > _MAX_RESUME_DRIFT_SUMMARY_LENGTH
            or "/" in summary
            or "\\" in summary
            or any(ord(character) < 32 for character in summary)
        ):
            raise ValueError("resume drift summary is not safe")
        object.__setattr__(self, "summary", summary)
        hashes = tuple(self.evidence_hashes)
        if len(hashes) > 64 or not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in hashes
        ):
            raise ValueError("resume drift evidence hashes are invalid")
        object.__setattr__(self, "evidence_hashes", hashes)


PreparedContinuationContext: TypeAlias = (
    ScopeApprovalContext
    | ReviewContext
    | ClarificationContext
    | SolResumeContext
    | AnswerContext
    | None
)


@dataclass(frozen=True, slots=True)
class ApprovalPayload:
    baseline_id: str
    baseline_setting: BaselineSetting | None
    scope: ScopeApprovalContext | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_id", _prepared_identifier(self.baseline_id, "baseline_id"))
        if self.baseline_setting is not None and not isinstance(self.baseline_setting, BaselineSetting):
            raise ValueError("baseline_setting is invalid")
        if self.scope is not None and not isinstance(self.scope, ScopeApprovalContext):
            raise ValueError("scope is invalid")


@dataclass(frozen=True, slots=True)
class AnswerPayload:
    answer: str
    continuation: PreparedContinuationContext

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer", _prepared_text(self.answer, "answer"))
        if self.continuation is not None and not isinstance(
            self.continuation,
            (ScopeApprovalContext, ReviewContext, ClarificationContext, SolResumeContext, AnswerContext),
        ):
            raise ValueError("answer continuation is invalid")


@dataclass(frozen=True, slots=True)
class ResumePayload:
    continuation: PreparedContinuationContext
    drift_event: ResumeDriftProjection

    def __post_init__(self) -> None:
        if self.continuation is not None and not isinstance(
            self.continuation,
            (ScopeApprovalContext, ReviewContext, ClarificationContext, SolResumeContext, AnswerContext),
        ):
            raise ValueError("resume continuation is invalid")
        if not isinstance(self.drift_event, ResumeDriftProjection):
            raise ValueError("drift_event is invalid")


PreparedActionPayload: TypeAlias = NewRequestPayload | ApprovalPayload | AnswerPayload | ResumePayload


@dataclass(frozen=True, slots=True)
class PreparedActionRecord:
    preparation_id: str
    project_id: str
    session_id: str
    task_id: str
    revision: int
    action: PreparedActionKind
    payload: PreparedActionPayload
    source_state: TaskState
    active_state: TaskState
    continuation_state: TaskState | None
    pending_context: PreparedContinuationContext
    previous_preparation_id: str | None
    status: Literal["PREPARED", "CLAIMED", "COMPLETED", "FAILED", "ABORTED", "INTERRUPTED", "RECOVERED"]
    reason: str | None
    generation: int

    def __post_init__(self) -> None:
        for name in ("preparation_id", "project_id", "session_id", "task_id"):
            object.__setattr__(self, name, _prepared_identifier(getattr(self, name), name))
        revision = _require_integer(self.revision, "revision")
        generation = _require_integer(self.generation, "generation")
        if revision < 0 or generation < 0:
            raise ValueError("revision and generation must be non-negative")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "generation", generation)
        if self.action not in _PREPARED_ACTION_KINDS:
            raise ValueError("prepared action kind is invalid")
        payload_types = {
            "new_request": NewRequestPayload,
            "approval": ApprovalPayload,
            "answer": AnswerPayload,
            "resume": ResumePayload,
        }
        if not isinstance(self.payload, payload_types[self.action]):
            raise ValueError("prepared action payload does not match action")
        if not isinstance(self.source_state, TaskState) or not isinstance(self.active_state, TaskState):
            raise ValueError("prepared action states are invalid")
        if self.continuation_state is not None and not isinstance(self.continuation_state, TaskState):
            raise ValueError("prepared continuation state is invalid")
        if self.pending_context is not None and not isinstance(
            self.pending_context,
            (
                ScopeApprovalContext,
                ReviewContext,
                ClarificationContext,
                SolResumeContext,
                AnswerContext,
            ),
        ):
            raise ValueError("prepared pending context is invalid")
        if self.action == "new_request":
            expected_context = None
        elif self.action == "approval":
            expected_context = self.payload.scope
        else:
            expected_context = self.payload.continuation
        if self.pending_context != expected_context:
            raise ValueError("prepared action context does not match its payload")
        if self.previous_preparation_id is not None:
            object.__setattr__(self, "previous_preparation_id", _prepared_identifier(self.previous_preparation_id, "previous_preparation_id"))
        if self.status not in _PREPARED_ACTION_STATUSES:
            raise ValueError("prepared action status is invalid")
        if self.reason is not None:
            object.__setattr__(self, "reason", _prepared_text(self.reason, "reason"))
        if self.status in {"PREPARED", "CLAIMED", "COMPLETED", "RECOVERED"}:
            if self.reason is not None:
                raise ValueError("prepared action status must not have a reason")
        elif self.status == "FAILED" and self.reason != "nonresumable_failure":
            raise ValueError("failed prepared action reason is invalid")
        elif self.status == "INTERRUPTED" and self.reason not in {
            "stop", "adapter_interrupted",
        }:
            raise ValueError("interrupted prepared action reason is invalid")
        elif self.status == "ABORTED" and self.reason is None:
            raise ValueError("aborted prepared action reason is required")


def _context_to_data(
    context: PreparedContinuationContext, *, depth: int = 0,
) -> dict[str, object] | None:
    if depth > 8:
        raise ValueError("prepared continuation nesting is too deep")
    if context is None:
        return None
    if isinstance(context, SolResumeContext):
        return {
            "kind": "sol_resume",
            "sol_thread_id": context.sol_thread_id,
            "sol_run_id": context.sol_run_id,
            "prompt": context.prompt,
        }
    if isinstance(context, ScopeApprovalContext):
        return {
            "kind": "scope_approval",
            "baseline_id": context.baseline_id,
            "approved_revision": context.approved_revision,
            "underlying_continuation": _context_to_data(
                context.underlying_continuation, depth=depth + 1
            ),
        }
    if isinstance(context, ReviewContext):
        return {
            "kind": "review",
            "fable_session_id": context.fable_session_id,
            "review_prompt": context.review_prompt,
            "completion_allowed": context.completion_allowed,
            "underlying_continuation": _context_to_data(
                context.underlying_continuation, depth=depth + 1
            ),
        }
    if isinstance(context, ClarificationContext):
        return {
            "kind": "clarification",
            "fable_session_id": context.fable_session_id,
            "clarification_prompt": context.clarification_prompt,
            "underlying_continuation": _context_to_data(
                context.underlying_continuation, depth=depth + 1
            ),
        }
    if isinstance(context, AnswerContext):
        return {
            "kind": "answer",
            "answer": context.answer,
            "underlying_continuation": _context_to_data(
                context.underlying_continuation, depth=depth + 1
            ),
        }
    raise ValueError("prepared continuation is invalid")


def _context_from_data(
    value: object, *, depth: int = 0,
) -> PreparedContinuationContext:
    if depth > 8:
        raise RuntimeError("persisted prepared continuation is too deep")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError("persisted prepared continuation is invalid")
    kind = value.get("kind")
    try:
        if kind == "sol_resume" and set(value) == {
            "kind", "sol_thread_id", "sol_run_id", "prompt",
        }:
            return SolResumeContext(
                sol_thread_id=value["sol_thread_id"],
                sol_run_id=value["sol_run_id"],
                prompt=value["prompt"],
            )
        if kind == "scope_approval" and set(value) == {
            "kind", "baseline_id", "approved_revision", "underlying_continuation",
        }:
            underlying = _context_from_data(
                value["underlying_continuation"], depth=depth + 1
            )
            if underlying is not None and not isinstance(underlying, SolResumeContext):
                raise ValueError("scope continuation is invalid")
            return ScopeApprovalContext(
                baseline_id=value["baseline_id"],
                approved_revision=value["approved_revision"],
                underlying_continuation=underlying,
            )
        if kind == "review" and set(value) == {
            "kind", "fable_session_id", "review_prompt", "completion_allowed", "underlying_continuation",
        }:
            underlying = _context_from_data(
                value["underlying_continuation"], depth=depth + 1
            )
            if not isinstance(underlying, (ScopeApprovalContext, SolResumeContext)):
                raise ValueError("review continuation is invalid")
            return ReviewContext(
                fable_session_id=value["fable_session_id"],
                review_prompt=value["review_prompt"],
                completion_allowed=value["completion_allowed"],
                underlying_continuation=underlying,
            )
        if kind == "clarification" and set(value) == {
            "kind", "fable_session_id", "clarification_prompt", "underlying_continuation",
        }:
            underlying = _context_from_data(
                value["underlying_continuation"], depth=depth + 1
            )
            if not isinstance(underlying, (ScopeApprovalContext, SolResumeContext)):
                raise ValueError("clarification continuation is invalid")
            return ClarificationContext(
                fable_session_id=value["fable_session_id"],
                clarification_prompt=value["clarification_prompt"],
                underlying_continuation=underlying,
            )
        if kind == "answer" and set(value) == {
            "kind", "answer", "underlying_continuation",
        }:
            return AnswerContext(
                answer=value["answer"],
                underlying_continuation=_context_from_data(
                    value["underlying_continuation"], depth=depth + 1
                ),
            )
    except ValueError as error:
        raise RuntimeError("persisted prepared continuation is invalid") from error
    raise RuntimeError("persisted prepared continuation is invalid")


def _payload_to_data(payload: PreparedActionPayload) -> dict[str, object]:
    if isinstance(payload, NewRequestPayload):
        return {"kind": "new_request", "text": payload.text}
    if isinstance(payload, ApprovalPayload):
        return {
            "kind": "approval",
            "baseline_id": payload.baseline_id,
            "baseline_setting": None if payload.baseline_setting is None else {
                "key": payload.baseline_setting.key,
                "value_json": payload.baseline_setting.value_json,
            },
            "scope": _context_to_data(payload.scope),
        }
    if isinstance(payload, AnswerPayload):
        return {
            "kind": "answer",
            "answer": payload.answer,
            "continuation": _context_to_data(payload.continuation),
        }
    if isinstance(payload, ResumePayload):
        return {
            "kind": "resume",
            "continuation": _context_to_data(payload.continuation),
            "drift_event": {
                "status": payload.drift_event.status,
                "summary": payload.drift_event.summary,
                "evidence_hashes": list(payload.drift_event.evidence_hashes),
            },
        }
    raise ValueError("prepared action payload is invalid")


def _payload_from_data(value: object) -> PreparedActionPayload:
    if not isinstance(value, Mapping):
        raise RuntimeError("persisted prepared payload is invalid")
    kind = value.get("kind")
    try:
        if kind == "new_request" and set(value) == {"kind", "text"}:
            return NewRequestPayload(text=value["text"])
        if kind == "approval" and set(value) == {
            "kind", "baseline_id", "baseline_setting", "scope",
        }:
            raw_setting = value["baseline_setting"]
            if raw_setting is None:
                setting = None
            elif isinstance(raw_setting, Mapping) and set(raw_setting) == {"key", "value_json"}:
                setting = BaselineSetting(
                    key=raw_setting["key"], value_json=raw_setting["value_json"]
                )
            else:
                raise ValueError("baseline setting is invalid")
            scope = _context_from_data(value["scope"])
            if scope is not None and not isinstance(scope, ScopeApprovalContext):
                raise ValueError("scope context is invalid")
            return ApprovalPayload(
                baseline_id=value["baseline_id"], baseline_setting=setting, scope=scope,
            )
        if kind == "answer" and set(value) == {"kind", "answer", "continuation"}:
            return AnswerPayload(
                answer=value["answer"],
                continuation=_context_from_data(value["continuation"]),
            )
        if kind == "resume" and set(value) == {
            "kind", "continuation", "drift_event",
        }:
            drift = value["drift_event"]
            if not isinstance(drift, Mapping) or set(drift) != {
                "status", "summary", "evidence_hashes",
            }:
                raise ValueError("resume drift event is invalid")
            hashes = drift["evidence_hashes"]
            if not isinstance(hashes, (list, tuple)):
                raise ValueError("resume drift evidence hashes are invalid")
            return ResumePayload(
                continuation=_context_from_data(value["continuation"]),
                drift_event=ResumeDriftProjection(
                    status=drift["status"], summary=drift["summary"],
                    evidence_hashes=tuple(hashes),
                ),
            )
    except ValueError as error:
        raise RuntimeError("persisted prepared payload is invalid") from error
    raise RuntimeError("persisted prepared payload is invalid")


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
        continuation_pause_id TEXT,
        continuation_generation INTEGER NOT NULL DEFAULT 1
            CHECK (continuation_generation >= 1),
        exchange_allowance INTEGER NOT NULL DEFAULT 3
            CHECK (exchange_allowance >= 0),
        exchange_consumed INTEGER NOT NULL DEFAULT 0
            CHECK (exchange_consumed >= 0),
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
    CREATE TABLE IF NOT EXISTS prepared_actions (
        preparation_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        action TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        source_state TEXT NOT NULL,
        active_state TEXT NOT NULL,
        continuation_state TEXT,
        pending_context_json TEXT,
        previous_preparation_id TEXT,
        status TEXT NOT NULL,
        reason TEXT,
        generation INTEGER NOT NULL,
        FOREIGN KEY (task_id, revision) REFERENCES tasks(task_id, revision)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS questions (
        question_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        continuation_generation INTEGER NOT NULL CHECK (continuation_generation >= 1),
        asked_by TEXT NOT NULL,
        addressed_to TEXT NOT NULL,
        routed_to TEXT NOT NULL,
        text TEXT NOT NULL,
        exchange_id TEXT,
        continuation_state TEXT NOT NULL,
        pending_action_json TEXT NOT NULL,
        continuation_pause_id TEXT NOT NULL,
        answer_text TEXT,
        answered_by TEXT,
        CHECK (
            (answer_text IS NULL AND answered_by IS NULL)
            OR (answer_text IS NOT NULL AND answered_by IS NOT NULL)
        ),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id),
        FOREIGN KEY (task_id, revision) REFERENCES tasks(task_id, revision)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS exchange_reservations (
        exchange_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        question_id TEXT NOT NULL UNIQUE,
        request_key TEXT NOT NULL,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
        continuation_generation INTEGER NOT NULL CHECK (continuation_generation >= 1),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id),
        FOREIGN KEY (task_id, revision) REFERENCES tasks(task_id, revision),
        FOREIGN KEY (question_id) REFERENCES questions(question_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS exchange_grants (
        grant_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        request_id TEXT NOT NULL,
        permission_id TEXT NOT NULL,
        continuation_generation INTEGER NOT NULL CHECK (continuation_generation >= 1),
        grant_size INTEGER NOT NULL CHECK (grant_size = 3),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id),
        FOREIGN KEY (task_id, revision) REFERENCES tasks(task_id, revision)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS exchange_permissions (
        permission_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        continuation_generation INTEGER NOT NULL CHECK (continuation_generation >= 1),
        continuation_pause_id TEXT NOT NULL,
        grant_request_id TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id),
        FOREIGN KEY (task_id, revision) REFERENCES tasks(task_id, revision)
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
    """
    CREATE INDEX IF NOT EXISTS prepared_actions_identity
    ON prepared_actions (project_id, session_id, task_id, revision, status)
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS one_unanswered_question_per_task_revision
    ON questions (task_id, revision)
    WHERE answer_text IS NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS questions_session_task_revision
    ON questions (session_id, task_id, revision)
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS exchange_reservations_request_identity
    ON exchange_reservations (session_id, task_id, revision, request_key)
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS exchange_grants_request_identity
    ON exchange_grants (session_id, task_id, revision, request_id)
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS exchange_permissions_pause_identity
    ON exchange_permissions (
        session_id, task_id, revision, continuation_generation, continuation_pause_id
    )
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
            self._migrate_directed_conversation_schema()
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

    def _migrate_directed_conversation_schema(self) -> None:
        """Add directed-conversation task fields without rewriting legacy payloads."""
        columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(tasks)")
        }
        additions = (
            (
                "continuation_generation",
                """
                ALTER TABLE tasks
                ADD COLUMN continuation_generation INTEGER NOT NULL DEFAULT 1
                    CHECK (continuation_generation >= 1)
                """,
            ),
            (
                "exchange_allowance",
                """
                ALTER TABLE tasks
                ADD COLUMN exchange_allowance INTEGER NOT NULL DEFAULT 3
                    CHECK (exchange_allowance >= 0)
                """,
            ),
            (
                "exchange_consumed",
                """
                ALTER TABLE tasks
                ADD COLUMN exchange_consumed INTEGER NOT NULL DEFAULT 0
                    CHECK (exchange_consumed >= 0)
                """,
            ),
            (
                "continuation_pause_id",
                "ALTER TABLE tasks ADD COLUMN continuation_pause_id TEXT",
            ),
        )
        for column, statement in additions:
            if column not in columns:
                self._connection.execute(statement)
        grant_columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(exchange_grants)")
        }
        if "permission_id" not in grant_columns:
            self._connection.execute(
                "ALTER TABLE exchange_grants ADD COLUMN permission_id TEXT"
            )
        question_columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(questions)")
        }
        if "continuation_state" not in question_columns:
            self._connection.execute(
                "ALTER TABLE questions ADD COLUMN continuation_state TEXT"
            )
        if "pending_action_json" not in question_columns:
            self._connection.execute(
                "ALTER TABLE questions ADD COLUMN pending_action_json TEXT"
            )
        if "continuation_pause_id" not in question_columns:
            self._connection.execute(
                "ALTER TABLE questions ADD COLUMN continuation_pause_id TEXT"
            )
        self._connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS exchange_grants_permission_identity
            ON exchange_grants (session_id, task_id, revision, permission_id)
            WHERE permission_id IS NOT NULL
            """
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

    def question(self, question_id: str) -> QuestionRecord | None:
        """Return one durable directed question without crossing store boundaries."""
        row = self._connection.execute(
            "SELECT * FROM questions WHERE question_id = ?",
            (_prepared_identifier(question_id, "question_id"),),
        ).fetchone()
        return None if row is None else self._question_from_row(row)

    def pause_for_question(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
        question_id: str,
        asked_by: ConversationActor,
        addressed_to: ConversationTarget,
        routed_to: ConversationTarget,
        text: str,
        continuation_state: TaskState,
        pending_action: Mapping[str, object],
        event: ConversationEnvelope,
    ) -> QuestionRecord:
        """Pause one active task behind an exact question and visible event."""
        session_id, task_id, revision, expected_generation = self._directed_identity(
            session_id, task_id, revision, expected_generation,
        )
        question_id = _prepared_identifier(question_id, "question_id")
        self._validate_question_inputs(
            task_id=task_id,
            revision=revision,
            expected_generation=expected_generation,
            question_id=question_id,
            asked_by=asked_by,
            addressed_to=addressed_to,
            routed_to=routed_to,
            text=text,
            event=event,
        )
        frozen_pending = self._directed_pending_action(pending_action)
        self._validate_pause_transition(continuation_state)
        question = QuestionRecord(
            question_id=question_id,
            session_id=session_id,
            task_id=task_id,
            revision=revision,
            continuation_generation=expected_generation,
            asked_by=asked_by,
            addressed_to=addressed_to,
            routed_to=routed_to,
            text=text,
            exchange_id=None,
            answer_text=None,
            answered_by=None,
        )
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._directed_task_exact(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=expected_generation,
                )
                if self._unanswered_question_for_task(task_id, revision) is not None:
                    raise RuntimeError("task revision already has an unanswered question")
                self._require_task_can_pause(task, continuation_state)
                if self.question(question_id) is not None:
                    raise RuntimeError("question identifier already exists")
                pause_id = self._new_continuation_pause_id()
                self._insert_question(
                    question,
                    continuation_state=continuation_state,
                    pending_action=frozen_pending,
                    continuation_pause_id=pause_id,
                )
                self._pause_task_for_directed_action(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=expected_generation,
                    continuation_state=continuation_state,
                    pending_action=frozen_pending,
                    continuation_pause_id=pause_id,
                )
                emitted.append(self._insert_conversation_event_in_transaction(
                    session_id=session_id,
                    task_id=task_id,
                    event=event,
                ))
            self._publish_committed_events(emitted)
        return question

    def answer_question_and_prepare_resume(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        question_id: str,
        expected_generation: int,
        answer_text: str,
        answered_by: ConversationActor,
        pending_action: Mapping[str, object],
        event: ConversationEnvelope,
    ) -> QuestionRecord:
        """CAS one exact pending question, then durably prepare its continuation."""
        session_id, task_id, revision, expected_generation = self._directed_identity(
            session_id, task_id, revision, expected_generation,
        )
        question_id = _prepared_identifier(question_id, "question_id")
        answer_text = _require_string(answer_text, "answer_text")
        if not isinstance(answered_by, ConversationActor):
            raise ValueError("answered_by must be a ConversationActor")
        frozen_pending = self._directed_pending_action(pending_action)
        self._validate_answer_event_binding(
            event=event,
            task_id=task_id,
            revision=revision,
            expected_generation=expected_generation,
            question_id=question_id,
            answer_text=answer_text,
            answered_by=answered_by,
        )
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._directed_task_exact(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=expected_generation,
                )
                (
                    question,
                    question_continuation,
                    question_pending,
                    question_pause_id,
                ) = self._question_exact(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=expected_generation,
                    question_id=question_id,
                )
                if question.answer_text is not None:
                    raise RuntimeError("question was already answered")
                task_pause_id = self._directed_pause_id(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=expected_generation,
                )
                if (
                    task.state is not TaskState.AWAITING_USER_INPUT
                    or task.continuation_state is not question_continuation
                    or task.pending != question_pending
                    or task_pause_id != question_pause_id
                ):
                    raise RuntimeError("question continuation changed concurrently")
                continuation_state = question_continuation
                self._validate_pause_transition(continuation_state)
                expected_actor = self._answer_actor_for_routed_target(question.routed_to)
                if answered_by is not expected_actor:
                    raise RuntimeError("answer actor is not eligible for this routed question")
                reply_target = self._target_for_question_asker(question.asked_by)
                if event.addressed_to is not reply_target or event.routed_to is not reply_target:
                    raise RuntimeError("answer event does not route to the question asker")
                cursor = self._connection.execute(
                    """
                    UPDATE questions SET answer_text = ?, answered_by = ?
                    WHERE question_id = ? AND session_id = ? AND task_id = ?
                      AND revision = ? AND continuation_generation = ?
                      AND answer_text IS NULL AND answered_by IS NULL
                    """,
                    (
                        answer_text,
                        answered_by.value,
                        question_id,
                        session_id,
                        task_id,
                        revision,
                        expected_generation,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("question changed concurrently")
                if answered_by is ConversationActor.USER:
                    next_generation = self._reset_internal_exchanges_for_human_direction_in_transaction(
                        cursor,
                        session_id=session_id,
                        task_id=task_id,
                        revision=revision,
                        expected_generation=expected_generation,
                    )
                else:
                    next_generation = expected_generation
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET state = ?, continuation_state = NULL, pending_json = ?,
                        continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND session_id = ?
                      AND state = ? AND continuation_state = ?
                      AND continuation_generation = ? AND pending_json = ?
                      AND continuation_pause_id = ?
                    """,
                    (
                        continuation_state.value,
                        _encode_json(frozen_pending),
                        task_id,
                        revision,
                        session_id,
                        TaskState.AWAITING_USER_INPUT.value,
                        continuation_state.value,
                        next_generation,
                        _encode_json(question_pending),
                        question_pause_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("task question continuation changed concurrently")
                emitted.append(self._insert_conversation_event_in_transaction(
                    session_id=session_id,
                    task_id=task_id,
                    event=event,
                ))
            self._publish_committed_events(emitted)
        return QuestionRecord(
            question_id=question.question_id,
            session_id=question.session_id,
            task_id=question.task_id,
            revision=question.revision,
            continuation_generation=question.continuation_generation,
            asked_by=question.asked_by,
            addressed_to=question.addressed_to,
            routed_to=question.routed_to,
            text=question.text,
            exchange_id=question.exchange_id,
            answer_text=answer_text,
            answered_by=answered_by,
        )

    def reserve_internal_question(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
        question_id: str,
        request_key: str,
        asked_by: ConversationActor,
        addressed_to: ConversationTarget,
        routed_to: ConversationTarget,
        text: str,
        continuation_state: TaskState,
        pending_action: Mapping[str, object],
        event: ConversationEnvelope,
    ) -> tuple[ExchangeReservation, QuestionRecord]:
        """Reserve exactly one finite agent-to-agent exchange before it is visible."""
        session_id, task_id, revision, expected_generation = self._directed_identity(
            session_id, task_id, revision, expected_generation,
        )
        question_id = _prepared_identifier(question_id, "question_id")
        request_key = _prepared_identifier(request_key, "request_key")
        self._validate_question_inputs(
            task_id=task_id,
            revision=revision,
            expected_generation=expected_generation,
            question_id=question_id,
            asked_by=asked_by,
            addressed_to=addressed_to,
            routed_to=routed_to,
            text=text,
            event=event,
        )
        if asked_by not in {ConversationActor.FABLE, ConversationActor.SOL}:
            raise ValueError("internal questions must be asked by an agent")
        if routed_to not in {ConversationTarget.FABLE, ConversationTarget.SOL}:
            raise ValueError("internal questions must route to an agent")
        if routed_to.value == asked_by.value:
            raise ValueError("internal questions must route to the other agent")
        frozen_pending = self._directed_pending_action(pending_action)
        self._validate_pause_transition(continuation_state)
        emitted: list[StreamEvent] = []
        result: tuple[ExchangeReservation, QuestionRecord]
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._directed_task_exact(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=expected_generation,
                )
                existing = self._reservation_for_request_key(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    request_key=request_key,
                )
                if existing is not None:
                    reservation, question = existing
                    task_pause_id = self._directed_pause_id_exact(
                        session_id=session_id,
                        task_id=task_id,
                        revision=revision,
                        expected_generation=expected_generation,
                    )
                    question_pause_id = self._question_pause_id(question.question_id)
                    if (
                        reservation.question_id != question_id
                        or reservation.continuation_generation != expected_generation
                        or question.continuation_generation != expected_generation
                        or question.asked_by is not asked_by
                        or question.addressed_to is not addressed_to
                        or question.routed_to is not routed_to
                        or question.text != text
                        or question.answer_text is not None
                        or task.state is not TaskState.AWAITING_USER_INPUT
                        or task.continuation_state is not continuation_state
                        or task.pending != frozen_pending
                        or task_pause_id != question_pause_id
                    ):
                        raise RuntimeError("exchange request key does not match its reservation")
                    result = existing
                else:
                    self._require_task_can_pause(task, continuation_state)
                    if task.exchange_allowance <= 0:
                        raise RuntimeError("internal exchange allowance is exhausted")
                    if self._unanswered_question_for_task(task_id, revision) is not None:
                        raise RuntimeError("task revision already has an unanswered question")
                    if self.question(question_id) is not None:
                        raise RuntimeError("question identifier already exists")
                    exchange_id = self._new_exchange_id()
                    pause_id = self._new_continuation_pause_id()
                    reservation = ExchangeReservation(
                        exchange_id=exchange_id,
                        question_id=question_id,
                        ordinal=task.exchange_consumed + 1,
                        continuation_generation=expected_generation,
                    )
                    question = QuestionRecord(
                        question_id=question_id,
                        session_id=session_id,
                        task_id=task_id,
                        revision=revision,
                        continuation_generation=expected_generation,
                        asked_by=asked_by,
                        addressed_to=addressed_to,
                        routed_to=routed_to,
                        text=text,
                        exchange_id=exchange_id,
                        answer_text=None,
                        answered_by=None,
                    )
                    self._insert_question(
                        QuestionRecord(
                            question_id=question.question_id,
                            session_id=question.session_id,
                            task_id=question.task_id,
                            revision=question.revision,
                            continuation_generation=question.continuation_generation,
                            asked_by=question.asked_by,
                            addressed_to=question.addressed_to,
                            routed_to=question.routed_to,
                            text=question.text,
                            exchange_id=None,
                            answer_text=None,
                            answered_by=None,
                        ),
                        continuation_state=continuation_state,
                        pending_action=frozen_pending,
                        continuation_pause_id=pause_id,
                    )
                    self._connection.execute(
                        """
                        INSERT INTO exchange_reservations (
                            exchange_id, session_id, task_id, revision, question_id,
                            request_key, ordinal, continuation_generation
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            reservation.exchange_id,
                            session_id,
                            task_id,
                            revision,
                            question_id,
                            request_key,
                            reservation.ordinal,
                            expected_generation,
                        ),
                    )
                    cursor = self._connection.execute(
                        """
                        UPDATE questions SET exchange_id = ?
                        WHERE question_id = ? AND exchange_id IS NULL
                        """,
                        (exchange_id, question_id),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("exchange question changed concurrently")
                    cursor = self._connection.execute(
                        """
                        UPDATE tasks
                        SET exchange_allowance = exchange_allowance - 1,
                            exchange_consumed = exchange_consumed + 1,
                            state = ?, continuation_state = ?, pending_json = ?,
                            continuation_pause_id = ?
                        WHERE task_id = ? AND revision = ? AND session_id = ?
                          AND continuation_generation = ? AND state = ?
                          AND continuation_state IS NULL AND continuation_pause_id IS NULL
                          AND exchange_allowance > 0
                        """,
                        (
                            TaskState.AWAITING_USER_INPUT.value,
                            continuation_state.value,
                            _encode_json(frozen_pending),
                            pause_id,
                            task_id,
                            revision,
                            session_id,
                            expected_generation,
                            continuation_state.value,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("internal exchange changed concurrently")
                    emitted.append(self._insert_conversation_event_in_transaction(
                        session_id=session_id,
                        task_id=task_id,
                        event=event,
                    ))
                    result = (reservation, question)
            if emitted:
                self._publish_committed_events(emitted)
        return result

    def pause_for_exchange_permission(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
        attempted_question: DirectedAgentQuestion,
        continuation_state: TaskState,
        pending_action: Mapping[str, object],
        event: ConversationEnvelope,
    ) -> TaskRecord:
        """Persist an exhausted-hop permission pause before a fourth question starts."""
        session_id, task_id, revision, expected_generation = self._directed_identity(
            session_id, task_id, revision, expected_generation,
        )
        if not isinstance(attempted_question, DirectedAgentQuestion):
            raise ValueError("attempted_question must be a DirectedAgentQuestion")
        if attempted_question.addressed_to not in {"fable", "sol"}:
            raise ValueError("attempted internal question must target an agent")
        self._validate_pause_transition(continuation_state)
        self._validate_permission_event_binding(
            event=event,
            task_id=task_id,
            revision=revision,
            expected_generation=expected_generation,
        )
        frozen_pending = self._directed_pending_action(pending_action)
        if {"attempted_question", "exchange_permission_id"} & set(frozen_pending):
            raise ValueError("pending_action must not replace exchange permission state")
        permission_pending = {
            **frozen_pending,
            "attempted_question": {
                "addressed_to": attempted_question.addressed_to,
                "text": attempted_question.text,
                "reason": attempted_question.reason,
            },
        }
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._directed_task_exact(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=expected_generation,
                )
                self._require_task_can_pause(task, continuation_state)
                if task.exchange_allowance != 0:
                    raise RuntimeError("internal exchange allowance is not exhausted")
                pause_id = self._new_continuation_pause_id()
                permission_id = self._new_exchange_permission_id()
                self._pause_task_for_directed_action(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=expected_generation,
                    continuation_state=continuation_state,
                    pending_action=permission_pending,
                    continuation_pause_id=pause_id,
                )
                self._connection.execute(
                    """
                    INSERT INTO exchange_permissions (
                        permission_id, session_id, task_id, revision,
                        continuation_generation, continuation_pause_id, grant_request_id
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        permission_id,
                        session_id,
                        task_id,
                        revision,
                        expected_generation,
                        pause_id,
                    ),
                )
                emitted.append(self._insert_conversation_event_in_transaction(
                    session_id=session_id,
                    task_id=task_id,
                    event=event,
                ))
            self._publish_committed_events(emitted)
        return self.get_task(task_id, revision)

    def grant_internal_exchanges(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
        request_id: str,
    ) -> int:
        """Add one fixed, idempotent three-exchange grant to an exhausted task."""
        session_id, task_id, revision, expected_generation = self._directed_identity(
            session_id, task_id, revision, expected_generation,
        )
        request_id = _prepared_identifier(request_id, "request_id")
        with self._immediate_transaction():
            task = self._directed_task_exact(
                session_id=session_id,
                task_id=task_id,
                revision=revision,
                expected_generation=expected_generation,
            )
            pause_id = self._directed_pause_id(
                session_id=session_id,
                task_id=task_id,
                revision=revision,
                expected_generation=expected_generation,
            )
            permission = (
                None
                if pause_id is None
                else self._exchange_permission_for_pause(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=expected_generation,
                    continuation_pause_id=pause_id,
                )
            )
            existing = self._connection.execute(
                """
                SELECT permission_id, continuation_generation, grant_size FROM exchange_grants
                WHERE session_id = ? AND task_id = ? AND revision = ? AND request_id = ?
                """,
                (session_id, task_id, revision, request_id),
            ).fetchone()
            if existing is not None:
                if (
                    int(existing["continuation_generation"]) != expected_generation
                    or int(existing["grant_size"]) != EXCHANGE_GRANT_SIZE
                    or (
                        permission is not None
                        and existing["permission_id"] != permission["permission_id"]
                    )
                ):
                    raise RuntimeError("exchange grant request changed")
                return int(existing["grant_size"])
            if (
                task.state is not TaskState.AWAITING_USER_INPUT
                or task.continuation_state is None
                or task.exchange_allowance != 0
                or permission is None
            ):
                raise RuntimeError("task is not awaiting exchange permission")
            if permission["grant_request_id"] is not None:
                raise RuntimeError("exchange permission already received its grant")
            permission_id = str(permission["permission_id"])
            cursor = self._connection.execute(
                """
                UPDATE exchange_permissions SET grant_request_id = ?
                WHERE permission_id = ? AND grant_request_id IS NULL
                """,
                (request_id, permission_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("exchange permission already received its grant")
            self._connection.execute(
                """
                INSERT INTO exchange_grants (
                    session_id, task_id, revision, request_id, permission_id,
                    continuation_generation, grant_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    task_id,
                    revision,
                    request_id,
                    permission_id,
                    expected_generation,
                    EXCHANGE_GRANT_SIZE,
                ),
            )
            cursor = self._connection.execute(
                """
                UPDATE tasks SET exchange_allowance = exchange_allowance + ?
                WHERE task_id = ? AND revision = ? AND session_id = ?
                  AND continuation_generation = ? AND continuation_pause_id = ?
                  AND state = ? AND exchange_allowance = 0
                """,
                (
                    EXCHANGE_GRANT_SIZE,
                    task_id,
                    revision,
                    session_id,
                    expected_generation,
                    pause_id,
                    TaskState.AWAITING_USER_INPUT.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("exchange grant changed concurrently")
        return EXCHANGE_GRANT_SIZE

    @staticmethod
    def _directed_identity(
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
    ) -> tuple[str, str, int, int]:
        session_id = _require_string(session_id, "session_id")
        task_id = _require_string(task_id, "task_id")
        revision = _require_integer(revision, "revision")
        expected_generation = _require_integer(
            expected_generation, "expected_generation",
        )
        if revision < 1:
            raise ValueError("revision must be >= 1")
        if expected_generation < 1:
            raise ValueError("expected_generation must be >= 1")
        return session_id, task_id, revision, expected_generation

    @staticmethod
    def _directed_pending_action(pending_action: Mapping[str, object]) -> Mapping[str, JsonValue]:
        frozen = freeze_json(pending_action)
        if not isinstance(frozen, Mapping):
            raise ValueError("pending_action must be an object")
        return frozen

    @staticmethod
    def _validate_pause_transition(continuation_state: TaskState) -> None:
        if not isinstance(continuation_state, TaskState):
            raise ValueError("continuation_state must be a TaskState")
        require_transition(continuation_state, TaskState.AWAITING_USER_INPUT)
        require_transition(TaskState.AWAITING_USER_INPUT, continuation_state)

    @staticmethod
    def _validate_conversation_binding(
        *,
        event: ConversationEnvelope,
        task_id: str,
        revision: int,
        expected_generation: int,
    ) -> None:
        if not isinstance(event, ConversationEnvelope):
            raise ValueError("event must be a ConversationEnvelope")
        if (
            event.task_id != task_id
            or event.revision != revision
            or event.continuation_generation != expected_generation
        ):
            raise ValueError("event does not bind the exact task continuation")

    def _validate_question_inputs(
        self,
        *,
        task_id: str,
        revision: int,
        expected_generation: int,
        question_id: str,
        asked_by: ConversationActor,
        addressed_to: ConversationTarget,
        routed_to: ConversationTarget,
        text: str,
        event: ConversationEnvelope,
    ) -> None:
        if not isinstance(asked_by, ConversationActor):
            raise ValueError("asked_by must be a ConversationActor")
        if asked_by is ConversationActor.SYSTEM:
            raise ValueError("questions must be asked by a user or agent")
        if not isinstance(addressed_to, ConversationTarget):
            raise ValueError("addressed_to must be a ConversationTarget")
        if not isinstance(routed_to, ConversationTarget):
            raise ValueError("routed_to must be a ConversationTarget")
        if routed_to is ConversationTarget.TEAM:
            raise ValueError("questions must route to one exact recipient")
        _require_string(text, "text")
        self._validate_conversation_binding(
            event=event,
            task_id=task_id,
            revision=revision,
            expected_generation=expected_generation,
        )
        if (
            event.message_type is not ConversationMessageType.QUESTION
            or event.question_id != question_id
            or event.reply_to_question_id is not None
            or event.sender is not asked_by
            or event.addressed_to is not addressed_to
            or event.routed_to is not routed_to
            or event.text != text
        ):
            raise ValueError("question event does not match the exact question")

    def _validate_answer_event_binding(
        self,
        *,
        event: ConversationEnvelope,
        task_id: str,
        revision: int,
        expected_generation: int,
        question_id: str,
        answer_text: str,
        answered_by: ConversationActor,
    ) -> None:
        self._validate_conversation_binding(
            event=event,
            task_id=task_id,
            revision=revision,
            expected_generation=expected_generation,
        )
        if (
            event.message_type is not ConversationMessageType.ANSWER
            or event.question_id is not None
            or event.reply_to_question_id != question_id
            or event.sender is not answered_by
            or event.text != answer_text
        ):
            raise ValueError("answer event does not match the exact question answer")

    def _validate_permission_event_binding(
        self,
        *,
        event: ConversationEnvelope,
        task_id: str,
        revision: int,
        expected_generation: int,
    ) -> None:
        self._validate_conversation_binding(
            event=event,
            task_id=task_id,
            revision=revision,
            expected_generation=expected_generation,
        )
        if (
            event.message_type is not ConversationMessageType.STATUS
            or event.sender is not ConversationActor.SYSTEM
            or event.addressed_to is not ConversationTarget.USER
            or event.routed_to is not ConversationTarget.USER
            or event.question_id is not None
            or event.reply_to_question_id is not None
        ):
            raise ValueError("exchange permission event must be a system status card for the user")

    def _directed_task_exact(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
    ) -> TaskRecord:
        row = self._connection.execute(
            """
            SELECT * FROM tasks
            WHERE task_id = ? AND revision = ? AND session_id = ?
              AND continuation_generation = ?
            """,
            (task_id, revision, session_id, expected_generation),
        ).fetchone()
        if row is None:
            raise RuntimeError("task continuation identity changed")
        return self._task_from_row(row)

    def _require_task_can_pause(
        self, task: TaskRecord, continuation_state: TaskState,
    ) -> None:
        if task.state is not continuation_state or task.continuation_state is not None:
            raise RuntimeError("task continuation changed concurrently")

    def _directed_pause_id(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
    ) -> str | None:
        row = self._connection.execute(
            """
            SELECT continuation_pause_id FROM tasks
            WHERE task_id = ? AND revision = ? AND session_id = ?
              AND continuation_generation = ?
            """,
            (task_id, revision, session_id, expected_generation),
        ).fetchone()
        if row is None:
            raise RuntimeError("task continuation identity changed")
        raw_pause_id = row["continuation_pause_id"]
        if raw_pause_id is None:
            return None
        if not isinstance(raw_pause_id, str) or not _SAFE_PREPARED_IDENTIFIER.fullmatch(raw_pause_id):
            raise RuntimeError("task continuation pause identity is invalid")
        return raw_pause_id

    def _directed_pause_id_exact(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
    ) -> str:
        pause_id = self._directed_pause_id(
            session_id=session_id,
            task_id=task_id,
            revision=revision,
            expected_generation=expected_generation,
        )
        if pause_id is None:
            raise RuntimeError("task continuation pause identity is missing")
        return pause_id

    def _exchange_permission_for_pause(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
        continuation_pause_id: str,
    ) -> sqlite3.Row | None:
        return self._connection.execute(
            """
            SELECT * FROM exchange_permissions
            WHERE session_id = ? AND task_id = ? AND revision = ?
              AND continuation_generation = ? AND continuation_pause_id = ?
            """,
            (
                session_id,
                task_id,
                revision,
                expected_generation,
                continuation_pause_id,
            ),
        ).fetchone()

    def _unanswered_question_for_task(
        self, task_id: str, revision: int,
    ) -> QuestionRecord | None:
        row = self._connection.execute(
            """
            SELECT * FROM questions
            WHERE task_id = ? AND revision = ? AND answer_text IS NULL
            """,
            (task_id, revision),
        ).fetchone()
        return None if row is None else self._question_from_row(row)

    def _question_exact(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
        question_id: str,
    ) -> tuple[QuestionRecord, TaskState, Mapping[str, JsonValue], str]:
        row = self._connection.execute(
            """
            SELECT * FROM questions
            WHERE question_id = ? AND session_id = ? AND task_id = ?
              AND revision = ? AND continuation_generation = ?
            """,
            (
                question_id,
                session_id,
                task_id,
                revision,
                expected_generation,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("question continuation identity changed")
        question = self._question_from_row(row)
        raw_continuation = row["continuation_state"]
        raw_pending = row["pending_action_json"]
        raw_pause_id = row["continuation_pause_id"]
        if raw_continuation is None or raw_pending is None or raw_pause_id is None:
            raise RuntimeError("question continuation is missing")
        try:
            continuation = TaskState(raw_continuation)
            pending = _decode_mapping(raw_pending, "question pending action")
        except (TypeError, ValueError) as error:
            raise RuntimeError("question continuation is invalid") from error
        if not isinstance(raw_pause_id, str) or not _SAFE_PREPARED_IDENTIFIER.fullmatch(raw_pause_id):
            raise RuntimeError("question continuation pause identity is invalid")
        return question, continuation, pending, raw_pause_id

    def _question_pause_id(self, question_id: str) -> str:
        row = self._connection.execute(
            "SELECT continuation_pause_id FROM questions WHERE question_id = ?",
            (question_id,),
        ).fetchone()
        if row is None or row["continuation_pause_id"] is None:
            raise RuntimeError("question continuation pause identity is missing")
        pause_id = row["continuation_pause_id"]
        if not isinstance(pause_id, str) or not _SAFE_PREPARED_IDENTIFIER.fullmatch(pause_id):
            raise RuntimeError("question continuation pause identity is invalid")
        return pause_id

    def _reservation_for_request_key(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        request_key: str,
    ) -> tuple[ExchangeReservation, QuestionRecord] | None:
        row = self._connection.execute(
            """
            SELECT * FROM exchange_reservations
            WHERE session_id = ? AND task_id = ? AND revision = ? AND request_key = ?
            """,
            (session_id, task_id, revision, request_key),
        ).fetchone()
        if row is None:
            return None
        reservation = self._exchange_reservation_from_row(row)
        question = self.question(reservation.question_id)
        if question is None:
            raise RuntimeError("exchange reservation is missing its question")
        if (
            question.session_id != session_id
            or question.task_id != task_id
            or question.revision != revision
            or question.exchange_id != reservation.exchange_id
        ):
            raise RuntimeError("exchange reservation identity is invalid")
        return reservation, question

    def _insert_question(
        self,
        question: QuestionRecord,
        *,
        continuation_state: TaskState,
        pending_action: Mapping[str, JsonValue],
        continuation_pause_id: str,
    ) -> None:
        if not isinstance(continuation_state, TaskState):
            raise ValueError("question continuation_state must be a TaskState")
        frozen_pending = freeze_json(pending_action)
        if not isinstance(frozen_pending, Mapping):
            raise ValueError("question pending_action must be an object")
        continuation_pause_id = _prepared_identifier(
            continuation_pause_id, "question continuation_pause_id",
        )
        self._connection.execute(
            """
            INSERT INTO questions (
                question_id, session_id, task_id, revision, continuation_generation,
                asked_by, addressed_to, routed_to, text, exchange_id,
                continuation_state, pending_action_json, continuation_pause_id,
                answer_text, answered_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question.question_id,
                question.session_id,
                question.task_id,
                question.revision,
                question.continuation_generation,
                question.asked_by.value,
                question.addressed_to.value,
                question.routed_to.value,
                question.text,
                question.exchange_id,
                continuation_state.value,
                _encode_json(frozen_pending),
                continuation_pause_id,
                question.answer_text,
                None if question.answered_by is None else question.answered_by.value,
            ),
        )

    def _pause_task_for_directed_action(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
        continuation_state: TaskState,
        pending_action: Mapping[str, object],
        continuation_pause_id: str,
    ) -> None:
        continuation_pause_id = _prepared_identifier(
            continuation_pause_id, "continuation_pause_id",
        )
        cursor = self._connection.execute(
            """
            UPDATE tasks
            SET state = ?, continuation_state = ?, pending_json = ?,
                continuation_pause_id = ?
            WHERE task_id = ? AND revision = ? AND session_id = ?
              AND continuation_generation = ? AND state = ?
              AND continuation_state IS NULL AND continuation_pause_id IS NULL
            """,
            (
                TaskState.AWAITING_USER_INPUT.value,
                continuation_state.value,
                _encode_json(pending_action),
                continuation_pause_id,
                task_id,
                revision,
                session_id,
                expected_generation,
                continuation_state.value,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("task continuation changed concurrently")

    def _insert_conversation_event_in_transaction(
        self,
        *,
        session_id: str,
        task_id: str,
        event: ConversationEnvelope,
    ) -> StreamEvent:
        return self._insert_event_in_transaction(
            session_id,
            task_id,
            event.sender.value,
            "conversation",
            event.to_dict(),
        )

    @staticmethod
    def _answer_actor_for_routed_target(
        routed_to: ConversationTarget,
    ) -> ConversationActor:
        actors = {
            ConversationTarget.USER: ConversationActor.USER,
            ConversationTarget.FABLE: ConversationActor.FABLE,
            ConversationTarget.SOL: ConversationActor.SOL,
        }
        try:
            return actors[routed_to]
        except KeyError as error:
            raise RuntimeError("question has no exact answer recipient") from error

    @staticmethod
    def _target_for_question_asker(asked_by: ConversationActor) -> ConversationTarget:
        targets = {
            ConversationActor.USER: ConversationTarget.USER,
            ConversationActor.FABLE: ConversationTarget.FABLE,
            ConversationActor.SOL: ConversationTarget.SOL,
        }
        try:
            return targets[asked_by]
        except KeyError as error:
            raise RuntimeError("question has no reply target") from error

    def _new_exchange_id(self) -> str:
        for _ in range(_MAX_PREPARATION_ID_ATTEMPTS):
            exchange_id = secrets.token_hex(24)
            row = self._connection.execute(
                "SELECT 1 FROM exchange_reservations WHERE exchange_id = ?",
                (exchange_id,),
            ).fetchone()
            if row is None:
                return exchange_id
        raise RuntimeError("could not allocate exchange identifier")

    def _new_exchange_permission_id(self) -> str:
        for _ in range(_MAX_PREPARATION_ID_ATTEMPTS):
            permission_id = secrets.token_hex(24)
            row = self._connection.execute(
                "SELECT 1 FROM exchange_permissions WHERE permission_id = ?",
                (permission_id,),
            ).fetchone()
            if row is None:
                return permission_id
        raise RuntimeError("could not allocate exchange permission identifier")

    def _new_continuation_pause_id(self) -> str:
        for _ in range(_MAX_PREPARATION_ID_ATTEMPTS):
            pause_id = secrets.token_hex(24)
            row = self._connection.execute(
                """
                SELECT 1 FROM tasks WHERE continuation_pause_id = ?
                UNION ALL
                SELECT 1 FROM questions WHERE continuation_pause_id = ?
                UNION ALL
                SELECT 1 FROM exchange_permissions WHERE continuation_pause_id = ?
                LIMIT 1
                """,
                (pause_id, pause_id, pause_id),
            ).fetchone()
            if row is None:
                return pause_id
        raise RuntimeError("could not allocate continuation pause identifier")

    def _reset_internal_exchanges_for_human_direction_in_transaction(
        self,
        cursor: sqlite3.Cursor,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
    ) -> int:
        """Advance the exact generation and restore a finite budget in this transaction."""
        if (
            not isinstance(cursor, sqlite3.Cursor)
            or cursor.connection is not self._connection
            or not self._connection.in_transaction
        ):
            raise RuntimeError("human-direction reset requires the active store transaction")
        cursor.execute(
            """
            UPDATE tasks
            SET continuation_generation = continuation_generation + ?,
                exchange_allowance = ?, exchange_consumed = 0
            WHERE task_id = ? AND revision = ? AND session_id = ?
              AND continuation_generation = ?
            """,
            (
                1,
                INITIAL_INTERNAL_EXCHANGES,
                task_id,
                revision,
                session_id,
                expected_generation,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("task continuation generation changed concurrently")
        return expected_generation + 1

    def prepared_action(self, preparation_id: str) -> PreparedActionRecord | None:
        row = self._connection.execute(
            "SELECT * FROM prepared_actions WHERE preparation_id = ?",
            (_prepared_identifier(preparation_id, "preparation_id"),),
        ).fetchone()
        return None if row is None else self._prepared_action_from_row(row)

    def latest_prepared_action_for_task(
        self, *, project_id: str, session_id: str, task_id: str, revision: int,
    ) -> PreparedActionRecord | None:
        project_id, session_id, task_id, revision, _ = self._prepared_identity(
            project_id, session_id, task_id, revision, 0
        )
        row = self._connection.execute(
            """
            SELECT * FROM prepared_actions
            WHERE project_id = ? AND session_id = ? AND task_id = ? AND revision = ?
            ORDER BY rowid DESC LIMIT 1
            """,
            (project_id, session_id, task_id, revision),
        ).fetchone()
        return None if row is None else self._prepared_action_from_row(row)

    def prepare_new_request_action(
        self,
        *,
        project_id: str,
        session_id: str,
        task_id: str,
        generation: int,
        payload: NewRequestPayload,
    ) -> PreparedActionRecord:
        project_id, session_id, task_id, _, generation = self._prepared_identity(
            project_id, session_id, task_id, 0, generation
        )
        if generation < 0 or not isinstance(payload, NewRequestPayload):
            raise ValueError("new request preparation is invalid")
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                if not self.session_exists(session_id):
                    raise RuntimeError("prepared action session not found")
                self._verify_prepared_project_identity(project_id, session_id)
                self._connection.execute(
                    """
                    INSERT INTO tasks (task_id, revision, session_id, state, brief_json)
                    VALUES (?, 0, ?, ?, NULL)
                    """,
                    (task_id, session_id, TaskState.FABLE_PLANNING.value),
                )
                emitted.append(self._insert_event_in_transaction(
                    session_id, task_id, "user", "message", {"text": payload.text}
                ))
                record = self._insert_prepared_action(
                    project_id=project_id,
                    session_id=session_id,
                    task_id=task_id,
                    revision=0,
                    action="new_request",
                    payload=payload,
                    source_state=TaskState.FABLE_PLANNING,
                    active_state=TaskState.FABLE_PLANNING,
                    continuation_state=None,
                    pending_context=None,
                    previous_preparation_id=None,
                    generation=generation,
                )
            self._publish_committed_events(emitted)
        return record

    def prepare_approval_action(
        self,
        *,
        project_id: str,
        session_id: str,
        task_id: str,
        revision: int,
        generation: int,
        payload: ApprovalPayload,
    ) -> PreparedActionRecord:
        project_id, session_id, task_id, revision, generation = self._prepared_identity(
            project_id, session_id, task_id, revision, generation
        )
        if not isinstance(payload, ApprovalPayload):
            raise ValueError("approval preparation payload is invalid")
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._prepared_task_exact(session_id, task_id, revision)
                self._verify_prepared_project_identity(project_id, session_id)
                if task.state not in {
                    TaskState.AWAITING_USER_APPROVAL,
                    TaskState.AWAITING_SCOPE_APPROVAL,
                } or task.brief is None or task.brief.open_questions:
                    raise RuntimeError("task is not eligible for prepared approval")
                if task.state is TaskState.AWAITING_SCOPE_APPROVAL:
                    if payload.scope is None or task.continuation_state is None:
                        raise RuntimeError("prepared scope approval has no continuation")
                    active = task.continuation_state
                else:
                    if task.continuation_state is None:
                        if payload.scope is not None:
                            raise RuntimeError("prepared approval has unexpected continuation")
                        active = TaskState.SOL_RUNNING
                    else:
                        if payload.scope is None:
                            raise RuntimeError("prepared approval has no exact continuation")
                        active = task.continuation_state
                if payload.scope is not None:
                    self._validate_prepared_context(task, payload.scope)
                require_transition(task.state, active)
                if payload.baseline_setting is not None:
                    existing = self._connection.execute(
                        "SELECT value_json FROM settings WHERE key = ?",
                        (payload.baseline_setting.key,),
                    ).fetchone()
                    if existing is None:
                        self._connection.execute(
                            "INSERT INTO settings (key, value_json) VALUES (?, ?)",
                            (
                                payload.baseline_setting.key,
                                payload.baseline_setting.value_json,
                            ),
                        )
                    elif existing["value_json"] != payload.baseline_setting.value_json:
                        raise RuntimeError("prepared baseline setting changed")
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET approved_at = ?, baseline_id = ?, state = ?,
                        continuation_state = NULL, pending_json = NULL,
                        continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND session_id = ? AND state = ?
                      AND NOT EXISTS (
                        SELECT 1 FROM tasks AS newer
                        WHERE newer.task_id = ? AND newer.revision > ?
                      )
                    """,
                    (
                        self._timestamp(), payload.baseline_id, active.value,
                        task_id, revision, session_id, task.state.value, task_id, revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("prepared approval changed concurrently")
                emitted.append(self._insert_event_in_transaction(
                    session_id, task_id, "coordinator", "task_state",
                    {"state": active.value, "revision": revision},
                ))
                record = self._insert_prepared_action(
                    project_id=project_id, session_id=session_id, task_id=task_id,
                    revision=revision, action="approval", payload=payload,
                    source_state=task.state, active_state=active,
                    continuation_state=task.continuation_state, pending_context=payload.scope,
                    previous_preparation_id=None, generation=generation,
                )
            self._publish_committed_events(emitted)
        return record

    def prepare_answer_action(
        self,
        *,
        project_id: str,
        session_id: str,
        task_id: str,
        revision: int,
        generation: int,
        payload: AnswerPayload,
    ) -> PreparedActionRecord:
        project_id, session_id, task_id, revision, generation = self._prepared_identity(
            project_id, session_id, task_id, revision, generation
        )
        if not isinstance(payload, AnswerPayload):
            raise ValueError("answer preparation payload is invalid")
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._prepared_task_exact(session_id, task_id, revision)
                self._verify_prepared_project_identity(project_id, session_id)
                if task.state is not TaskState.AWAITING_USER_INPUT:
                    raise RuntimeError("task is not eligible for prepared answer")
                if task.continuation_state is None:
                    raise RuntimeError("prepared answer has no continuation")
                active = task.continuation_state
                self._validate_prepared_context(task, payload.continuation)
                require_transition(task.state, active)
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET state = ?, continuation_state = NULL, pending_json = NULL,
                        continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND session_id = ? AND state = ?
                    """,
                    (active.value, task_id, revision, session_id, task.state.value),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("prepared answer changed concurrently")
                emitted.append(self._insert_event_in_transaction(
                    session_id, task_id, "user", "message", {"text": payload.answer}
                ))
                record = self._insert_prepared_action(
                    project_id=project_id, session_id=session_id, task_id=task_id,
                    revision=revision, action="answer", payload=payload,
                    source_state=task.state, active_state=active,
                    continuation_state=task.continuation_state,
                    pending_context=payload.continuation,
                    previous_preparation_id=None, generation=generation,
                )
            self._publish_committed_events(emitted)
        return record

    def prepare_resume_action(
        self,
        *,
        project_id: str,
        session_id: str,
        task_id: str,
        revision: int,
        generation: int,
        payload: ResumePayload,
        previous_preparation_id: str | None,
    ) -> PreparedActionRecord:
        project_id, session_id, task_id, revision, generation = self._prepared_identity(
            project_id, session_id, task_id, revision, generation
        )
        if not isinstance(payload, ResumePayload):
            raise ValueError("resume preparation payload is invalid")
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._prepared_task_exact(session_id, task_id, revision)
                self._verify_prepared_project_identity(project_id, session_id)
                if task.state is not TaskState.INTERRUPTED:
                    raise RuntimeError("task is not eligible for prepared resume")
                if task.continuation_state is None:
                    raise RuntimeError("prepared resume has no continuation")
                active = task.continuation_state
                require_transition(task.state, active)
                predecessor = self._validate_predecessor(
                    project_id, session_id, task_id, revision, generation,
                    previous_preparation_id,
                )
                self._validate_prepared_context(
                    task,
                    payload.continuation,
                    predecessor=predecessor,
                )
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET state = ?, continuation_state = NULL, pending_json = NULL,
                        continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND session_id = ? AND state = ?
                    """,
                    (active.value, task_id, revision, session_id, TaskState.INTERRUPTED.value),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("prepared resume changed concurrently")
                emitted.append(self._insert_event_in_transaction(
                    session_id, task_id, "coordinator", "resume_drift",
                    {
                        "status": payload.drift_event.status,
                        "summary": payload.drift_event.summary,
                        "evidence_hashes": list(payload.drift_event.evidence_hashes),
                    },
                ))
                record = self._insert_prepared_action(
                    project_id=project_id, session_id=session_id, task_id=task_id,
                    revision=revision, action="resume", payload=payload,
                    source_state=TaskState.INTERRUPTED, active_state=active,
                    continuation_state=task.continuation_state,
                    pending_context=payload.continuation,
                    previous_preparation_id=previous_preparation_id, generation=generation,
                )
            self._publish_committed_events(emitted)
        return record

    def fail_resume_for_drift(
        self,
        *,
        project_id: str,
        session_id: str,
        task_id: str,
        revision: int,
        drift_event: ResumeDriftProjection,
    ) -> TaskRecord:
        """Persist an accepted drift block without creating runnable work."""
        project_id, session_id, task_id, revision, _ = self._prepared_identity(
            project_id, session_id, task_id, revision, 0,
        )
        if (
            not isinstance(drift_event, ResumeDriftProjection)
            or drift_event.status != "drifted"
        ):
            raise ValueError("resume drift failure must be a drifted projection")
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._prepared_task_exact(session_id, task_id, revision)
                self._verify_prepared_project_identity(project_id, session_id)
                if task.state is not TaskState.INTERRUPTED:
                    raise RuntimeError("task is not eligible for drift-blocked resume")
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET state = ?, continuation_state = NULL, pending_json = NULL,
                        continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND session_id = ? AND state = ?
                    """,
                    (
                        TaskState.FAILED.value,
                        task_id,
                        revision,
                        session_id,
                        TaskState.INTERRUPTED.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("drift-blocked resume changed concurrently")
                emitted.append(self._insert_event_in_transaction(
                    session_id,
                    task_id,
                    "coordinator",
                    "task_state",
                    {"state": TaskState.FAILED.value, "revision": revision},
                ))
                emitted.append(self._insert_event_in_transaction(
                    session_id,
                    task_id,
                    "coordinator",
                    "resume_drift",
                    {
                        "status": drift_event.status,
                        "summary": drift_event.summary,
                        "evidence_hashes": list(drift_event.evidence_hashes),
                    },
                ))
            self._publish_committed_events(emitted)
        return self.get_task(task_id, revision)

    def claim_prepared_action(
        self, preparation_id: str, *, generation: int,
    ) -> PreparedActionRecord:
        return self._transition_prepared_action(
            preparation_id, generation=generation, expected="PREPARED", target="CLAIMED", reason=None,
        )

    def complete_prepared_action(
        self, preparation_id: str, *, generation: int,
    ) -> PreparedActionRecord:
        return self._transition_prepared_action(
            preparation_id, generation=generation, expected="CLAIMED", target="COMPLETED", reason=None,
        )

    def fail_prepared_action(
        self, preparation_id: str, *, generation: int,
        reason: PreparedActionFailureReason,
    ) -> PreparedActionRecord:
        if reason != "nonresumable_failure":
            raise ValueError("prepared action failure reason is invalid")
        return self._transition_prepared_action(
            preparation_id, generation=generation, expected="CLAIMED", target="FAILED", reason=reason,
        )

    def interrupt_claimed_prepared_action(
        self, preparation_id: str, *, generation: int,
        reason: PreparedActionInterruptionReason,
    ) -> PreparedActionRecord:
        if reason not in {"stop", "adapter_interrupted"}:
            raise ValueError("prepared action interruption reason is invalid")
        with self._immediate_transaction():
            record = self._prepared_required(preparation_id)
            self._require_record_generation(record, generation)
            if record.status == "INTERRUPTED" and record.reason == reason:
                return record
            if record.status not in {"PREPARED", "CLAIMED"}:
                raise RuntimeError("prepared action is not interruptible")
            task = self._prepared_task_exact(record.session_id, record.task_id, record.revision)
            if task.state is not TaskState.INTERRUPTED:
                raise RuntimeError("prepared task is not interrupted")
            continuation = task.continuation_state or record.active_state
            if task.pending is not None and (
                task.continuation_state is not record.active_state
                or "sol_run_id" in task.pending
            ):
                pending = task.pending
            else:
                pending = self._prepared_pending_projection(record, reason=reason)
            task_cursor = self._connection.execute(
                """
                UPDATE tasks
                SET continuation_state = ?, pending_json = ?, continuation_pause_id = NULL
                WHERE task_id = ? AND revision = ? AND session_id = ? AND state = ?
                """,
                (
                    continuation.value,
                    _encode_json(pending),
                    record.task_id,
                    record.revision,
                    record.session_id,
                    TaskState.INTERRUPTED.value,
                ),
            )
            if task_cursor.rowcount != 1:
                raise RuntimeError("claimed prepared task changed concurrently")
            cursor = self._connection.execute(
                """
                UPDATE prepared_actions SET status = ?, reason = ?
                WHERE preparation_id = ? AND generation = ? AND status IN ('PREPARED', 'CLAIMED')
                """,
                ("INTERRUPTED", reason, record.preparation_id, generation),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("prepared action changed concurrently")
        return self._prepared_required(preparation_id)

    def abort_prepared_action(
        self, preparation_id: str, *, generation: int, reason: str,
    ) -> PreparedActionRecord:
        reason = _prepared_text(reason, "reason")
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                record = self._prepared_required(preparation_id)
                self._require_record_generation(record, generation)
                if record.status == "ABORTED":
                    if record.reason == reason:
                        return record
                    raise RuntimeError("prepared action was aborted differently")
                if record.status != "PREPARED":
                    raise RuntimeError("prepared action is not abortable")
                require_transition(record.active_state, TaskState.INTERRUPTED)
                task = self._prepared_task_exact(record.session_id, record.task_id, record.revision)
                if task.state is not record.active_state:
                    raise RuntimeError("prepared action task changed concurrently")
                pending = self._prepared_pending_projection(record, reason=reason)
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET state = ?, continuation_state = ?, pending_json = ?,
                        continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND session_id = ? AND state = ?
                    """,
                    (
                        TaskState.INTERRUPTED.value, record.active_state.value, _encode_json(pending),
                        record.task_id, record.revision, record.session_id, record.active_state.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("prepared abort changed concurrently")
                cursor = self._connection.execute(
                    """
                    UPDATE prepared_actions SET status = 'ABORTED', reason = ?
                    WHERE preparation_id = ? AND generation = ? AND status = 'PREPARED'
                    """,
                    (reason, record.preparation_id, generation),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("prepared abort changed concurrently")
                emitted.append(self._insert_event_in_transaction(
                    record.session_id, record.task_id, "coordinator", "task_state",
                    {"state": TaskState.INTERRUPTED.value, "revision": record.revision},
                ))
            self._publish_committed_events(emitted)
        return self._prepared_required(preparation_id)

    def recover_unfinished_prepared_actions(self) -> RecoverySummary:
        prepared_actions_recovered = 0
        tasks_interrupted = 0
        with self._immediate_transaction():
            last_preparation_id = ""
            while True:
                rows = self._connection.execute(
                    """
                    SELECT * FROM prepared_actions
                    WHERE status IN ('PREPARED', 'CLAIMED') AND preparation_id > ?
                    ORDER BY preparation_id LIMIT ?
                    """,
                    (last_preparation_id, _STARTUP_RECOVERY_BATCH_SIZE),
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    record = self._prepared_action_from_row(row)
                    task = self._prepared_task_exact(record.session_id, record.task_id, record.revision)
                    pending = self._prepared_pending_projection(record, reason="recovery")
                    if task.state is record.active_state:
                        require_transition(record.active_state, TaskState.INTERRUPTED)
                        cursor = self._connection.execute(
                            """
                            UPDATE tasks
                            SET state = ?, continuation_state = ?, pending_json = ?,
                                continuation_pause_id = NULL
                            WHERE task_id = ? AND revision = ? AND state = ?
                            """,
                            (
                                TaskState.INTERRUPTED.value,
                                record.active_state.value,
                                _encode_json(pending),
                                record.task_id,
                                record.revision,
                                record.active_state.value,
                            ),
                        )
                        if cursor.rowcount != 1:
                            raise RuntimeError("unfinished prepared action task changed")
                        tasks_interrupted += cursor.rowcount
                    elif task.state is not TaskState.INTERRUPTED:
                        raise RuntimeError("unfinished prepared action task changed")
                    else:
                        continuation = task.continuation_state or record.active_state
                        if task.pending is not None and (
                            task.continuation_state is not record.active_state
                            or "sol_run_id" in task.pending
                        ):
                            pending = task.pending
                        cursor = self._connection.execute(
                            """
                            UPDATE tasks
                            SET continuation_state = ?, pending_json = ?,
                                continuation_pause_id = NULL
                            WHERE task_id = ? AND revision = ? AND state = ?
                            """,
                            (
                                continuation.value,
                                _encode_json(pending),
                                record.task_id,
                                record.revision,
                                TaskState.INTERRUPTED.value,
                            ),
                        )
                        if cursor.rowcount != 1:
                            raise RuntimeError("unfinished prepared action task changed")
                    cursor = self._connection.execute(
                        """
                        UPDATE prepared_actions SET status = 'RECOVERED', reason = NULL
                        WHERE preparation_id = ? AND status IN ('PREPARED', 'CLAIMED')
                        """,
                        (record.preparation_id,),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("unfinished prepared action changed")
                    prepared_actions_recovered += cursor.rowcount
                last_preparation_id = str(rows[-1]["preparation_id"])
        return RecoverySummary(
            prepared_actions_recovered=prepared_actions_recovered,
            tasks_interrupted=tasks_interrupted,
            agent_runs_interrupted=0,
        )

    def _prepared_identity(
        self,
        project_id: object,
        session_id: object,
        task_id: object,
        revision: object,
        generation: object,
    ) -> tuple[str, str, str, int, int]:
        normalized_project = _prepared_identifier(project_id, "project_id")
        normalized_session = _prepared_identifier(session_id, "session_id")
        normalized_task = _prepared_identifier(task_id, "task_id")
        normalized_revision = _require_integer(revision, "revision")
        normalized_generation = _require_integer(generation, "generation")
        if normalized_revision < 0 or normalized_generation < 0:
            raise ValueError("revision and generation must be non-negative")
        return (
            normalized_project,
            normalized_session,
            normalized_task,
            normalized_revision,
            normalized_generation,
        )

    def _prepared_task_exact(
        self, session_id: str, task_id: str, revision: int,
    ) -> TaskRecord:
        row = self._connection.execute(
            """
            SELECT * FROM tasks
            WHERE task_id = ? AND revision = ? AND session_id = ?
            """,
            (task_id, revision, session_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("prepared action task not found")
        latest = self._connection.execute(
            "SELECT MAX(revision) AS revision FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if latest is None or latest["revision"] != revision:
            raise RuntimeError("prepared action task is not the latest revision")
        return self._task_from_row(row)

    def _verify_prepared_project_identity(self, project_id: str, session_id: str) -> None:
        """Bind a prepared row to the canonical session root when available.

        Some low-level store tests and legacy data use synthetic, non-existent
        repository strings.  They cannot be canonicalized, so the coordinator
        remains the trust boundary there.  Runtime-backed sessions always have
        an existing canonical root and therefore reject substituted browser
        project identifiers before any mutation.
        """
        root = self.session_repo_root(session_id)
        if root is None:
            raise RuntimeError("prepared action session not found")
        candidate = Path(root)
        if not candidate.exists():
            return
        try:
            expected = project_id_for_root(candidate.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise RuntimeError("prepared action project identity is invalid") from error
        if project_id != expected:
            raise RuntimeError("prepared action project identity does not match the session")

    def _new_preparation_id(self) -> str:
        for _ in range(_MAX_PREPARATION_ID_ATTEMPTS):
            candidate = secrets.token_hex(24)
            row = self._connection.execute(
                "SELECT 1 FROM prepared_actions WHERE preparation_id = ?", (candidate,)
            ).fetchone()
            if row is None:
                return candidate
        raise RuntimeError("could not allocate prepared action identifier")

    def _insert_prepared_action(
        self,
        *,
        project_id: str,
        session_id: str,
        task_id: str,
        revision: int,
        action: PreparedActionKind,
        payload: PreparedActionPayload,
        source_state: TaskState,
        active_state: TaskState,
        continuation_state: TaskState | None,
        pending_context: PreparedContinuationContext,
        previous_preparation_id: str | None,
        generation: int,
    ) -> PreparedActionRecord:
        preparation_id = self._new_preparation_id()
        record = PreparedActionRecord(
            preparation_id=preparation_id,
            project_id=project_id,
            session_id=session_id,
            task_id=task_id,
            revision=revision,
            action=action,
            payload=payload,
            source_state=source_state,
            active_state=active_state,
            continuation_state=continuation_state,
            pending_context=pending_context,
            previous_preparation_id=previous_preparation_id,
            status="PREPARED",
            reason=None,
            generation=generation,
        )
        self._connection.execute(
            """
            INSERT INTO prepared_actions (
                preparation_id, project_id, session_id, task_id, revision, action,
                payload_json, source_state, active_state, continuation_state,
                pending_context_json, previous_preparation_id, status, reason, generation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.preparation_id, record.project_id, record.session_id,
                record.task_id, record.revision, record.action,
                _encode_json(_payload_to_data(record.payload)), record.source_state.value,
                record.active_state.value,
                None if record.continuation_state is None else record.continuation_state.value,
                None if record.pending_context is None else _encode_json(
                    _context_to_data(record.pending_context)
                ),
                record.previous_preparation_id, record.status, record.reason,
                record.generation,
            ),
        )
        return record

    def _prepared_required(self, preparation_id: str) -> PreparedActionRecord:
        record = self.prepared_action(preparation_id)
        if record is None:
            raise RuntimeError("prepared action not found")
        return record

    @staticmethod
    def _require_record_generation(record: PreparedActionRecord, generation: int) -> None:
        generation = _require_integer(generation, "generation")
        if record.generation != generation:
            raise RuntimeError("prepared action generation changed")

    @staticmethod
    def _active_state_for_context(context: PreparedContinuationContext) -> TaskState:
        if isinstance(context, ReviewContext):
            return TaskState.FABLE_REVIEWING
        if isinstance(context, ClarificationContext):
            return TaskState.FABLE_CLARIFYING
        if isinstance(context, (ScopeApprovalContext, SolResumeContext, AnswerContext)):
            return TaskState.SOL_RUNNING
        return TaskState.FABLE_PLANNING

    @staticmethod
    def _prepared_pending_projection(
        record: PreparedActionRecord, *, reason: str,
    ) -> dict[str, object]:
        """Return the bounded resumable task projection for one durable row."""
        return {
            "prepared_action": {
                "preparation_id": record.preparation_id,
                "action": record.action,
                "reason": reason,
                "context": _context_to_data(record.pending_context),
            },
        }

    def _validate_predecessor(
        self,
        project_id: str,
        session_id: str,
        task_id: str,
        revision: int,
        generation: int,
        previous_preparation_id: str | None,
    ) -> PreparedActionRecord | None:
        existing = tuple(self._connection.execute(
            """
            SELECT preparation_id, generation FROM prepared_actions
            WHERE project_id = ? AND session_id = ? AND task_id = ? AND revision = ?
            ORDER BY rowid DESC
            """,
            (project_id, session_id, task_id, revision),
        ))
        if previous_preparation_id is None:
            if existing:
                raise RuntimeError("prepared resume requires a previous preparation")
            return None
        previous = self._prepared_required(previous_preparation_id)
        if (
            previous.project_id != project_id
            or previous.session_id != session_id
            or previous.task_id != task_id
            or previous.revision != revision
            or previous.status not in {"ABORTED", "RECOVERED", "INTERRUPTED"}
        ):
            raise RuntimeError("prepared resume predecessor is invalid")
        if not existing or existing[0]["preparation_id"] != previous_preparation_id:
            raise RuntimeError("prepared resume predecessor is not the latest preparation")
        if generation == COMPATIBILITY_PREPARATION_GENERATION:
            if previous.generation != COMPATIBILITY_PREPARATION_GENERATION:
                raise RuntimeError("prepared resume generation changed")
        elif generation <= previous.generation:
            raise RuntimeError("prepared resume generation did not advance")
        return previous

    def _validate_prepared_context(
        self,
        task: TaskRecord, context: PreparedContinuationContext,
        *,
        predecessor: PreparedActionRecord | None = None,
    ) -> None:
        """Reject a context substituted for the exact persisted continuation."""
        if task.continuation_state is None:
            raise RuntimeError("prepared continuation is missing")
        pending = task.pending or {}
        projection = pending.get("prepared_action")
        if self._is_initial_approval_resume_context(
            task, context, predecessor, projection,
        ):
            return
        if isinstance(projection, Mapping) and "context" in projection:
            try:
                frozen = _context_from_data(projection["context"])
            except RuntimeError as error:
                raise RuntimeError("prepared continuation does not match task") from error
            if frozen != context:
                raise RuntimeError("prepared continuation does not match task")
            return
        def sol_context_matches(value: object) -> bool:
            return (
                isinstance(value, SolResumeContext)
                and isinstance(task.sol_thread_id, str)
                and value.sol_thread_id == task.sol_thread_id
                and isinstance(pending.get("sol_run_id"), str)
                and value.sol_run_id == pending["sol_run_id"]
                and isinstance(pending.get("prompt"), str)
                and value.prompt == pending["prompt"]
            )

        def scope_context_matches(value: object) -> bool:
            return (
                isinstance(value, ScopeApprovalContext)
                and value.baseline_id == task.baseline_id
                and value.approved_revision == task.revision
                and (
                    value.underlying_continuation is None
                    or sol_context_matches(value.underlying_continuation)
                )
            )

        if task.continuation_state in _SOL_TASK_STATES:
            if isinstance(context, ScopeApprovalContext):
                if not scope_context_matches(context):
                    raise RuntimeError("prepared continuation does not match task")
                if context.underlying_continuation is None:
                    if task.sol_thread_id is None and not pending:
                        return
                    raise RuntimeError("prepared continuation does not match task")
                context = context.underlying_continuation
            if not sol_context_matches(context):
                raise RuntimeError("prepared continuation does not match task")
            return
        if task.continuation_state is TaskState.FABLE_REVIEWING:
            if not isinstance(context, ReviewContext):
                raise RuntimeError("prepared continuation does not match task")
            if (
                not isinstance(task.fable_session_id, str)
                or context.fable_session_id != task.fable_session_id
                or not isinstance(pending.get("review_prompt"), str)
                or context.review_prompt != pending["review_prompt"]
                or (
                    not isinstance(pending.get("completion_allowed"), bool)
                    or context.completion_allowed != pending["completion_allowed"]
                )
                or not (
                    scope_context_matches(context.underlying_continuation)
                    or sol_context_matches(context.underlying_continuation)
                )
            ):
                raise RuntimeError("prepared continuation does not match task")
            return
        if task.continuation_state is TaskState.FABLE_CLARIFYING:
            if not isinstance(context, ClarificationContext):
                raise RuntimeError("prepared continuation does not match task")
            if (
                not isinstance(task.fable_session_id, str)
                or context.fable_session_id != task.fable_session_id
                or not isinstance(pending.get("clarification_prompt"), str)
                or context.clarification_prompt != pending["clarification_prompt"]
                or not (
                    scope_context_matches(context.underlying_continuation)
                    or sol_context_matches(context.underlying_continuation)
                )
            ):
                raise RuntimeError("prepared continuation does not match task")
            return
        if task.continuation_state is TaskState.FABLE_PLANNING and context is None:
            return
        raise RuntimeError("prepared continuation does not match task")

    @staticmethod
    def _is_initial_approval_resume_context(
        task: TaskRecord,
        context: PreparedContinuationContext,
        predecessor: PreparedActionRecord | None,
        projection: object,
    ) -> bool:
        if (
            predecessor is None
            or predecessor.action != "approval"
            or not isinstance(predecessor.payload, ApprovalPayload)
            or predecessor.payload.scope is not None
            or task.continuation_state is not TaskState.SOL_RUNNING
            or task.sol_thread_id is not None
            or task.baseline_id != predecessor.payload.baseline_id
            or not isinstance(context, ScopeApprovalContext)
            or context != ScopeApprovalContext(
                baseline_id=predecessor.payload.baseline_id,
                approved_revision=task.revision,
                underlying_continuation=None,
            )
            or not isinstance(projection, Mapping)
        ):
            return False
        return (
            set(projection) == {"preparation_id", "action", "reason", "context"}
            and projection.get("preparation_id") == predecessor.preparation_id
            and projection.get("action") == "approval"
            and projection.get("context") is None
            and isinstance(projection.get("reason"), str)
        )

    def _transition_prepared_action(
        self,
        preparation_id: str,
        *,
        generation: int,
        expected: str,
        target: str,
        reason: str | None,
    ) -> PreparedActionRecord:
        with self._immediate_transaction():
            record = self._prepared_required(preparation_id)
            self._require_record_generation(record, generation)
            if record.status != expected:
                raise RuntimeError("prepared action state changed")
            if expected == "PREPARED":
                task = self._prepared_task_exact(
                    record.session_id, record.task_id, record.revision,
                )
                if task.state is not record.active_state:
                    raise RuntimeError("prepared action task changed concurrently")
            cursor = self._connection.execute(
                """
                UPDATE prepared_actions SET status = ?, reason = ?
                WHERE preparation_id = ? AND generation = ? AND status = ?
                """,
                (target, reason, record.preparation_id, generation, expected),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("prepared action state changed")
        return self._prepared_required(preparation_id)

    def _insert_event_in_transaction(
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
        created_at = self._timestamp()
        cursor = self._connection.execute(
            """
            INSERT INTO events (session_id, task_id, actor, kind, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, task_id, actor, kind, _encode_json(frozen_payload), created_at),
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
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title, created_at, session_id),
            )
        return event

    def _publish_committed_events(self, events: list[StreamEvent]) -> None:
        self._pending_listener_events.extend(events)
        if self._dispatching_listener_events:
            return
        self._dispatching_listener_events = True
        self._drain_event_listeners()

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
                    SET state = ?, continuation_state = NULL, pending_json = NULL,
                        continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND state = ?
                    """,
                    (target.value, task_id, revision, expected.value),
                )
            else:
                cursor = self._connection.execute(
                    """
                    UPDATE tasks SET state = ?, continuation_pause_id = NULL
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
                UPDATE tasks
                SET state = ?, pending_json = NULL, continuation_pause_id = NULL
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
                SET state = ?, continuation_state = ?, pending_json = ?,
                    continuation_pause_id = NULL
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
                SET state = ?, continuation_state = ?, pending_json = ?,
                    continuation_pause_id = NULL
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
                UPDATE tasks
                SET state = ?, pending_json = ?, continuation_pause_id = NULL
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
            UPDATE tasks SET pending_json = ?, continuation_pause_id = NULL
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
            UPDATE tasks SET pending_json = NULL, continuation_pause_id = NULL
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
                SET state = ?, continuation_state = ?, pending_json = ?,
                    continuation_pause_id = NULL
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
                SET state = ?, continuation_state = NULL, pending_json = NULL,
                    continuation_pause_id = NULL
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
                    fable_session_id = COALESCE(?, fable_session_id),
                    continuation_pause_id = NULL
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

    def recover_active_tasks(self) -> RecoverySummary:
        """Atomically retire process-local work left active by an old server.

        Persisted PIDs and process groups are inert audit data. Startup recovery
        never inspects or signals them because ownership cannot survive a process
        restart safely.
        """
        active_values = tuple(state.value for state in _ACTIVE_TASK_STATES)
        placeholders = ", ".join("?" for _ in active_values)
        tasks_interrupted = 0
        with self._immediate_transaction():
            while True:
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
                    LIMIT ?
                    """,
                    (*active_values, _STARTUP_RECOVERY_BATCH_SIZE),
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    task_id, revision = str(row["task_id"]), int(row["revision"])
                    cursor = self._connection.execute(
                        f"""
                        UPDATE tasks
                        SET continuation_state = state, state = ?,
                            continuation_pause_id = NULL
                        WHERE task_id = ? AND revision = ? AND state IN ({placeholders})
                        """,
                        (TaskState.INTERRUPTED.value, task_id, revision, *active_values),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("active task changed during startup recovery")
                    tasks_interrupted += cursor.rowcount
            cursor = self._connection.execute(
                """
                UPDATE agent_runs
                SET status = 'interrupted', ended_at = ?
                WHERE status = 'running'
                """,
                (self._timestamp(),),
            )
            agent_runs_interrupted = cursor.rowcount
        return RecoverySummary(
            prepared_actions_recovered=0,
            tasks_interrupted=tasks_interrupted,
            agent_runs_interrupted=agent_runs_interrupted,
        )

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

        def has_row(statement: str, parameters: tuple[object, ...] = ()) -> bool:
            return self._connection.execute(statement, parameters).fetchone() is not None

        if has_row("PRAGMA foreign_key_check"):
            reasons.add("foreign_key_integrity")
        if has_row(
            """
            SELECT 1 FROM sessions
            WHERE typeof(session_id) != 'text' OR trim(session_id) = ''
               OR typeof(repo_root) != 'text' OR trim(repo_root) = ''
               OR repo_root != ?
            LIMIT 1
            """,
            (canonical_repo_root,),
        ):
            reasons.add("session_ownership")

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
                    or not has_row(
                        "SELECT 1 FROM sessions WHERE session_id = ? AND repo_root = ? LIMIT 1",
                        (active_session, canonical_repo_root),
                    )
                ):
                    reasons.add("active_session")

        if has_row(
            """
            SELECT 1 FROM tasks AS task
            LEFT JOIN sessions AS session ON session.session_id = task.session_id
            WHERE typeof(task.task_id) != 'text' OR trim(task.task_id) = ''
               OR typeof(task.revision) != 'integer' OR task.revision < 0
               OR typeof(task.session_id) != 'text' OR session.session_id IS NULL
               OR (task.baseline_id IS NOT NULL AND (
                    typeof(task.baseline_id) != 'text' OR task.baseline_id = ''
               ))
            LIMIT 1
            """,
        ):
            reasons.add("task_integrity")
        if has_row(
            """
            SELECT 1 FROM tasks
            GROUP BY task_id
            HAVING COUNT(DISTINCT session_id) > 1
            LIMIT 1
            """,
        ):
            reasons.add("task_ownership")
        if has_row(
            """
            SELECT 1 FROM (
                SELECT task_id, MIN(revision) AS first_revision,
                       MAX(revision) AS last_revision, COUNT(*) AS revisions
                FROM tasks GROUP BY task_id
            )
            WHERE first_revision NOT IN (0, 1)
               OR revisions != last_revision - first_revision + 1
            LIMIT 1
            """,
        ):
            reasons.add("task_revision_integrity")

        if has_row(
            """
            SELECT 1 FROM events AS event
            LEFT JOIN sessions AS session ON session.session_id = event.session_id
            WHERE typeof(event.session_id) != 'text' OR session.session_id IS NULL
            LIMIT 1
            """,
        ):
            reasons.add("event_ownership")
        if has_row(
            """
            SELECT 1 FROM events AS event
            LEFT JOIN tasks AS task
              ON task.task_id = event.task_id AND task.session_id = event.session_id
            WHERE event.task_id IS NOT NULL AND (
                typeof(event.task_id) != 'text' OR trim(event.task_id) = ''
                OR task.task_id IS NULL
            )
            LIMIT 1
            """,
        ):
            reasons.add("event_task_integrity")
        if has_row(
            """
            SELECT 1 FROM agent_runs AS run
            LEFT JOIN tasks AS task
              ON task.task_id = run.task_id AND task.revision = run.revision
            WHERE typeof(run.task_id) != 'text' OR trim(run.task_id) = ''
               OR typeof(run.revision) != 'integer' OR task.task_id IS NULL
            LIMIT 1
            """,
        ):
            reasons.add("run_task_integrity")

        for row in self._connection.execute(
            """
            SELECT key, value_json FROM settings
            WHERE key = ? OR key LIKE ?
            """,
            (_BASELINE_SETTING_PREFIX.removesuffix("."), f"{_BASELINE_SETTING_PREFIX}%"),
        ):
            key = row["key"]
            try:
                persisted = json.loads(row["value_json"])
            except (TypeError, json.JSONDecodeError):
                reasons.add("baseline_integrity")
                continue
            if not isinstance(key, str) or not isinstance(persisted, dict):
                reasons.add("baseline_integrity")
                continue
            task_id = persisted.get("task_id")
            revision = persisted.get("revision")
            baseline_id = persisted.get("baseline_id")
            manifest = persisted.get("manifest")
            task_row = None
            if isinstance(task_id, str) and isinstance(revision, int) and not isinstance(revision, bool):
                task_row = self._connection.execute(
                    "SELECT baseline_id FROM tasks WHERE task_id = ? AND revision = ?",
                    (task_id, revision),
                ).fetchone()
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
                or task_row is None
                or task_row["baseline_id"] != baseline_id
            ):
                reasons.add("baseline_integrity")
        if has_row(
            """
            SELECT 1 FROM tasks AS task
            WHERE task.baseline_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM settings AS setting
                  WHERE setting.key = ? || task.task_id || '.' || task.revision
              )
            LIMIT 1
            """,
            (_BASELINE_SETTING_PREFIX,),
        ):
            reasons.add("baseline_integrity")

        expected_project_id = hashlib.sha256(
            os.fsencode(canonical_repo_root)
        ).hexdigest()[:32]
        for row in self._connection.execute(
            "SELECT rowid, * FROM prepared_actions ORDER BY rowid"
        ):
            try:
                record = self._prepared_action_from_row(row)
            except RuntimeError:
                reasons.add("prepared_action_integrity")
                continue
            if record.project_id != expected_project_id:
                reasons.add("prepared_action_ownership")
                continue
            if not self._legacy_prepared_action_is_authenticated(record, int(row["rowid"])):
                reasons.add("prepared_action_integrity")

        return reasons

    def _legacy_prepared_action_is_authenticated(
        self, record: PreparedActionRecord, rowid: int,
    ) -> bool:
        """Authenticate one recovery-capable row without retaining history."""
        task_row = self._connection.execute(
            "SELECT * FROM tasks WHERE task_id = ? AND revision = ?",
            (record.task_id, record.revision),
        ).fetchone()
        if task_row is None:
            return False
        try:
            task = self._task_from_row(task_row)
        except RuntimeError:
            return False
        if task.session_id != record.session_id:
            return False
        if record.action == "new_request":
            if (
                record.revision != 0
                or record.source_state is not TaskState.FABLE_PLANNING
                or record.active_state is not TaskState.FABLE_PLANNING
                or record.continuation_state is not None
                or record.pending_context is not None
                or record.previous_preparation_id is not None
            ):
                return False
        elif record.action == "approval":
            payload = record.payload
            if not isinstance(payload, ApprovalPayload) or task.baseline_id != payload.baseline_id:
                return False
            if payload.scope is not None and (
                payload.scope.baseline_id != payload.baseline_id
                or payload.scope.approved_revision != record.revision
            ):
                return False
            if record.source_state is TaskState.AWAITING_SCOPE_APPROVAL:
                if (
                    payload.scope is None
                    or record.continuation_state is None
                    or record.active_state is not record.continuation_state
                ):
                    return False
            elif record.source_state is TaskState.AWAITING_USER_APPROVAL:
                if payload.scope is None:
                    if (
                        record.continuation_state is not None
                        or record.active_state is not TaskState.SOL_RUNNING
                    ):
                        return False
                elif (
                    record.continuation_state is None
                    or record.active_state is not record.continuation_state
                ):
                    return False
            else:
                return False
            if record.previous_preparation_id is not None:
                return False
            if payload.baseline_setting is not None and not self._legacy_baseline_setting_matches(
                payload.baseline_setting, record.task_id, record.revision, payload.baseline_id,
            ):
                return False
        elif record.action == "answer":
            if (
                record.source_state is not TaskState.AWAITING_USER_INPUT
                or record.continuation_state is None
                or record.active_state is not record.continuation_state
                or record.previous_preparation_id is not None
            ):
                return False
            payload = record.payload
            if (
                not isinstance(payload, AnswerPayload)
                or payload.continuation != record.pending_context
                or not self._legacy_prepared_context_matches_task(
                    task=task,
                    active_state=record.active_state,
                    context=payload.continuation,
                )
                or not self._legacy_prepared_context_matches_task(
                    task=task,
                    active_state=record.active_state,
                    context=record.pending_context,
                )
            ):
                return False
        elif record.action == "resume":
            if (
                record.source_state is not TaskState.INTERRUPTED
                or record.continuation_state is None
                or record.active_state is not record.continuation_state
            ):
                return False
            if not self._legacy_prepared_lineage_matches(record, rowid):
                return False
        else:  # PreparedActionRecord normally prevents this; keep audit fail-closed.
            return False
        if record.action != "new_request":
            try:
                require_transition(record.source_state, record.active_state)
            except ValueError:
                return False
        return True

    def _legacy_prepared_context_matches_task(
        self,
        *,
        task: TaskRecord,
        active_state: TaskState,
        context: PreparedContinuationContext,
    ) -> bool:
        if active_state is TaskState.FABLE_REVIEWING:
            return (
                isinstance(context, ReviewContext)
                and context.fable_session_id == task.fable_session_id
                and self._legacy_continuation_identifiers_match_task(
                    task, context.underlying_continuation,
                )
            )
        return (
            active_state in _SOL_TASK_STATES
            and isinstance(context, (ScopeApprovalContext, SolResumeContext))
            and (
                not isinstance(context, ScopeApprovalContext)
                or context.underlying_continuation is not None
                or (task.sol_thread_id is None and not task.pending)
            )
            and self._legacy_continuation_identifiers_match_task(task, context)
        )

    def _legacy_continuation_identifiers_match_task(
        self,
        task: TaskRecord,
        context: PreparedContinuationContext,
    ) -> bool:
        if isinstance(context, SolResumeContext):
            run = self._connection.execute(
                """
                SELECT task_id, revision, cli_session_id FROM agent_runs
                WHERE run_id = ?
                """,
                (context.sol_run_id,),
            ).fetchone()
            return context.sol_thread_id == task.sol_thread_id and (
                run is None
                or (
                    run["task_id"] == task.task_id
                    and run["revision"] == task.revision
                    and run["cli_session_id"] == context.sol_thread_id
                )
            )
        if isinstance(context, ScopeApprovalContext):
            return (
                context.baseline_id == task.baseline_id
                and context.approved_revision == task.revision
                and (
                    context.underlying_continuation is None
                    or self._legacy_continuation_identifiers_match_task(
                        task, context.underlying_continuation,
                    )
                )
            )
        if isinstance(context, ReviewContext):
            return (
                context.fable_session_id == task.fable_session_id
                and self._legacy_continuation_identifiers_match_task(
                    task, context.underlying_continuation,
                )
            )
        return False

    def _legacy_baseline_setting_matches(
        self,
        setting: BaselineSetting,
        task_id: str,
        revision: int,
        baseline_id: str,
    ) -> bool:
        try:
            value = json.loads(setting.value_json)
        except (TypeError, json.JSONDecodeError):
            return False
        return (
            setting.key == f"{_BASELINE_SETTING_PREFIX}{task_id}.{revision}"
            and isinstance(value, Mapping)
            and value.get("task_id") == task_id
            and value.get("revision") == revision
            and value.get("baseline_id") == baseline_id
        )

    def _legacy_prepared_lineage_matches(
        self, record: PreparedActionRecord, rowid: int,
    ) -> bool:
        previous_id = record.previous_preparation_id
        previous_row = self._connection.execute(
            """
            SELECT rowid, * FROM prepared_actions
            WHERE preparation_id = ?
            """,
            (previous_id,),
        ).fetchone() if previous_id is not None else None
        latest = self._connection.execute(
            """
            SELECT preparation_id FROM prepared_actions
            WHERE project_id = ? AND session_id = ? AND task_id = ? AND revision = ?
              AND rowid < ?
            ORDER BY rowid DESC LIMIT 1
            """,
            (
                record.project_id, record.session_id, record.task_id,
                record.revision, rowid,
            ),
        ).fetchone()
        if previous_id is None:
            return latest is None
        if latest is None or latest["preparation_id"] != previous_id or previous_row is None:
            return False
        try:
            previous = self._prepared_action_from_row(previous_row)
        except RuntimeError:
            return False
        if (
            int(previous_row["rowid"]) >= rowid
            or previous.project_id != record.project_id
            or previous.session_id != record.session_id
            or previous.task_id != record.task_id
            or previous.revision != record.revision
            or previous.status not in {"ABORTED", "RECOVERED", "INTERRUPTED"}
        ):
            return False
        return (
            record.generation == COMPATIBILITY_PREPARATION_GENERATION
            and previous.generation == COMPATIBILITY_PREPARATION_GENERATION
        ) or (
            record.generation > previous.generation
        )

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
        try:
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
            continuation_generation = int(row["continuation_generation"])
            exchange_allowance = int(row["exchange_allowance"])
            exchange_consumed = int(row["exchange_consumed"])
            if continuation_generation < 1:
                raise RuntimeError("persisted task continuation generation is invalid")
            if exchange_allowance < 0 or exchange_consumed < 0:
                raise RuntimeError("persisted task exchange budget is invalid")
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
                continuation_state=(
                    None if raw_continuation is None else TaskState(raw_continuation)
                ),
                pending=pending,
                continuation_generation=continuation_generation,
                exchange_allowance=exchange_allowance,
                exchange_consumed=exchange_consumed,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError("persisted task is invalid") from error

    @staticmethod
    def _question_from_row(row: sqlite3.Row) -> QuestionRecord:
        try:
            raw_answered_by = row["answered_by"]
            return QuestionRecord(
                question_id=row["question_id"],
                session_id=row["session_id"],
                task_id=row["task_id"],
                revision=int(row["revision"]),
                continuation_generation=int(row["continuation_generation"]),
                asked_by=ConversationActor(row["asked_by"]),
                addressed_to=ConversationTarget(row["addressed_to"]),
                routed_to=ConversationTarget(row["routed_to"]),
                text=row["text"],
                exchange_id=row["exchange_id"],
                answer_text=row["answer_text"],
                answered_by=(
                    None
                    if raw_answered_by is None
                    else ConversationActor(raw_answered_by)
                ),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError("persisted question is invalid") from error

    @staticmethod
    def _exchange_reservation_from_row(row: sqlite3.Row) -> ExchangeReservation:
        try:
            return ExchangeReservation(
                exchange_id=row["exchange_id"],
                question_id=row["question_id"],
                ordinal=int(row["ordinal"]),
                continuation_generation=int(row["continuation_generation"]),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError("persisted exchange reservation is invalid") from error

    def _prepared_action_from_row(self, row: sqlite3.Row) -> PreparedActionRecord:
        try:
            payload = _payload_from_data(
                _decode_mapping(row["payload_json"], "prepared payload")
            )
            raw_context = row["pending_context_json"]
            context = None if raw_context is None else _context_from_data(
                _decode_mapping(raw_context, "prepared context")
            )
            raw_continuation = row["continuation_state"]
            return PreparedActionRecord(
                preparation_id=row["preparation_id"],
                project_id=row["project_id"],
                session_id=row["session_id"],
                task_id=row["task_id"],
                revision=int(row["revision"]),
                action=row["action"],
                payload=payload,
                source_state=TaskState(row["source_state"]),
                active_state=TaskState(row["active_state"]),
                continuation_state=(
                    None if raw_continuation is None else TaskState(raw_continuation)
                ),
                pending_context=context,
                previous_preparation_id=row["previous_preparation_id"],
                status=row["status"],
                reason=row["reason"],
                generation=int(row["generation"]),
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError("persisted prepared action is invalid") from error

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
