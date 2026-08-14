"""SQLite persistence for the local agent bridge.

The store deliberately owns records and compare-and-swap updates, but not
workflow policy.  The coordinator supplies every state transition explicitly.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
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
    FableClarification,
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
_INTERVENTION_SOURCE_AGENTS = {
    TaskState.FABLE_PLANNING: "fable",
    TaskState.FABLE_CLARIFYING: "fable",
    TaskState.FABLE_REVIEWING: "fable",
    TaskState.SOL_RUNNING: "sol",
    TaskState.SOL_CORRECTING: "sol",
}
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
_PREPARED_ACTION_KINDS = frozenset({
    "new_request",
    "approval",
    "answer",
    "resume",
    "continuation_message",
    "question_answer",
    "exchange_grant",
})
_PREPARED_ACTION_STATUSES = frozenset({
    "PREPARED", "CLAIMED", "COMPLETED", "FAILED", "ABORTED", "INTERRUPTED", "RECOVERED",
})
_SAFE_PREPARED_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")

PreparedActionKind: TypeAlias = Literal[
    "new_request",
    "approval",
    "answer",
    "resume",
    "continuation_message",
    "question_answer",
    "exchange_grant",
]
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
    nested_parent_kind: Literal["clarification", "question"] | None = None
    parent_question_id: str | None = None
    parent_continuation_pause_id: str | None = None

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
        if self.nested_parent_kind not in {None, "clarification", "question"}:
            raise ValueError("nested_parent_kind is invalid")
        if self.nested_parent_kind == "question":
            if self.parent_question_id is None or self.parent_continuation_pause_id is None:
                raise ValueError("nested question parent identity is required")
            object.__setattr__(self, "parent_question_id", _prepared_identifier(
                self.parent_question_id, "parent_question_id",
            ))
            object.__setattr__(self, "parent_continuation_pause_id", _prepared_identifier(
                self.parent_continuation_pause_id, "parent_continuation_pause_id",
            ))
        elif self.parent_question_id is not None or self.parent_continuation_pause_id is not None:
            raise ValueError("nested parent identity is invalid")


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


def _intervention_text(value: object) -> str:
    """Reuse the public conversation text grammar for intervention guidance."""
    return ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        message_type=ConversationMessageType.INTERVENTION,
        text=value,  # type: ignore[arg-type]
        task_id="intervention-task",
        revision=1,
        continuation_generation=1,
    ).text


def _prepared_identifier(value: object, name: str) -> str:
    identifier = _require_string(value, name)
    if _SAFE_PREPARED_IDENTIFIER.fullmatch(identifier) is None:
        raise ValueError(f"{name} must be a safe identifier")
    return identifier


def _prepared_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


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
    addressed_to: ConversationTarget | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _prepared_text(self.text, "text"))
        if self.addressed_to is not None and not isinstance(
            self.addressed_to, ConversationTarget,
        ):
            raise ValueError("addressed_to must be a ConversationTarget")


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
class DirectedFableAnswerCheckpoint:
    """One authenticated Fable answer committed before its Sol continuation."""

    preparation_id: str
    question_id: str
    continuation_generation: int
    clarification: FableClarification

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "preparation_id", _prepared_identifier(self.preparation_id, "preparation_id"),
        )
        object.__setattr__(
            self, "question_id", _prepared_identifier(self.question_id, "question_id"),
        )
        generation = _require_integer(
            self.continuation_generation, "continuation_generation",
        )
        if generation < 1:
            raise ValueError("continuation_generation must be positive")
        object.__setattr__(self, "continuation_generation", generation)
        if not isinstance(self.clarification, FableClarification):
            raise ValueError("checkpoint clarification is invalid")


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


@dataclass(frozen=True, slots=True)
class ContinuationMessagePayload:
    """One exact user-authored continuation routed to one persisted agent."""

    text: str
    addressed_to: ConversationTarget
    routed_to: ConversationTarget
    continuation_generation: int
    continuation: PreparedContinuationContext

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _prepared_text(self.text, "text"))
        if not isinstance(self.addressed_to, ConversationTarget):
            raise ValueError("addressed_to must be a ConversationTarget")
        if not isinstance(self.routed_to, ConversationTarget):
            raise ValueError("routed_to must be a ConversationTarget")
        if self.routed_to not in {ConversationTarget.FABLE, ConversationTarget.SOL}:
            raise ValueError("continuations must route to one agent")
        generation = _require_integer(
            self.continuation_generation, "continuation_generation",
        )
        if generation < 1:
            raise ValueError("continuation_generation must be positive")
        object.__setattr__(self, "continuation_generation", generation)
        if self.continuation is not None and not isinstance(
            self.continuation,
            (ScopeApprovalContext, ReviewContext, ClarificationContext, SolResumeContext, AnswerContext),
        ):
            raise ValueError("continuation is invalid")


@dataclass(frozen=True, slots=True)
class QuestionAnswerPayload:
    """A user answer bound to one persisted directed-question pause."""

    question_id: str
    answer: str
    continuation_generation: int
    continuation: PreparedContinuationContext

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", _prepared_identifier(self.question_id, "question_id"))
        object.__setattr__(self, "answer", _prepared_text(self.answer, "answer"))
        generation = _require_integer(
            self.continuation_generation, "continuation_generation",
        )
        if generation < 1:
            raise ValueError("continuation_generation must be positive")
        object.__setattr__(self, "continuation_generation", generation)
        if self.continuation is not None and not isinstance(
            self.continuation,
            (ScopeApprovalContext, ReviewContext, ClarificationContext, SolResumeContext, AnswerContext),
        ):
            raise ValueError("continuation is invalid")


@dataclass(frozen=True, slots=True)
class ExchangeGrantPayload:
    """One fixed +3 grant and its Store-authenticated attempted question."""

    request_id: str
    continuation_generation: int
    attempted_question: DirectedAgentQuestion
    continuation: PreparedContinuationContext
    parent_mode: Literal["top_level", "clarification", "question"]
    outer_question_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _prepared_identifier(self.request_id, "request_id"))
        generation = _require_integer(
            self.continuation_generation, "continuation_generation",
        )
        if generation < 1:
            raise ValueError("continuation_generation must be positive")
        object.__setattr__(self, "continuation_generation", generation)
        if not isinstance(self.attempted_question, DirectedAgentQuestion):
            raise ValueError("attempted_question must be a DirectedAgentQuestion")
        if self.continuation is not None and not isinstance(
            self.continuation,
            (ScopeApprovalContext, ReviewContext, ClarificationContext, SolResumeContext, AnswerContext),
        ):
            raise ValueError("continuation is invalid")
        if self.outer_question_id is not None:
            object.__setattr__(
                self, "outer_question_id",
                _prepared_identifier(self.outer_question_id, "outer_question_id"),
            )
        if self.parent_mode not in {"top_level", "clarification", "question"}:
            raise ValueError("exchange grant parent_mode is invalid")
        if (self.parent_mode == "question") != (self.outer_question_id is not None):
            raise ValueError("exchange grant parent identity is invalid")
        if self.parent_mode == "clarification" and not isinstance(self.continuation, ClarificationContext):
            raise ValueError("exchange grant clarification continuation is invalid")
        if self.parent_mode == "question" and not isinstance(self.continuation, SolResumeContext):
            raise ValueError("exchange grant question continuation is invalid")
        if self.parent_mode == "top_level" and isinstance(self.continuation, ClarificationContext):
            raise ValueError("exchange grant top-level continuation is invalid")


PreparedActionPayload: TypeAlias = (
    NewRequestPayload
    | ApprovalPayload
    | AnswerPayload
    | ResumePayload
    | ContinuationMessagePayload
    | QuestionAnswerPayload
    | ExchangeGrantPayload
)


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
            "continuation_message": ContinuationMessagePayload,
            "question_answer": QuestionAnswerPayload,
            "exchange_grant": ExchangeGrantPayload,
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
        elif self.action in {"answer", "resume"}:
            expected_context = self.payload.continuation
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
        data: dict[str, object] = {"kind": "new_request", "text": payload.text}
        if payload.addressed_to is not None:
            data["addressed_to"] = payload.addressed_to.value
        return data
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
    if isinstance(payload, ContinuationMessagePayload):
        return {
            "kind": "continuation_message",
            "text": payload.text,
            "addressed_to": payload.addressed_to.value,
            "routed_to": payload.routed_to.value,
            "continuation_generation": payload.continuation_generation,
            "continuation": _context_to_data(payload.continuation),
        }
    if isinstance(payload, QuestionAnswerPayload):
        return {
            "kind": "question_answer",
            "question_id": payload.question_id,
            "answer": payload.answer,
            "continuation_generation": payload.continuation_generation,
            "continuation": _context_to_data(payload.continuation),
        }
    if isinstance(payload, ExchangeGrantPayload):
        return {
            "kind": "exchange_grant",
            "request_id": payload.request_id,
            "continuation_generation": payload.continuation_generation,
            "attempted_question": payload.attempted_question.to_dict(),
            "continuation": _context_to_data(payload.continuation),
            "parent_mode": payload.parent_mode,
            "outer_question_id": payload.outer_question_id,
        }
    raise ValueError("prepared action payload is invalid")


def _payload_from_data(value: object) -> PreparedActionPayload:
    if not isinstance(value, Mapping):
        raise RuntimeError("persisted prepared payload is invalid")
    kind = value.get("kind")
    try:
        if kind == "new_request" and set(value) in (
            {"kind", "text"}, {"kind", "text", "addressed_to"},
        ):
            addressed_to = value.get("addressed_to")
            return NewRequestPayload(
                text=value["text"],
                addressed_to=(
                    None if addressed_to is None else ConversationTarget(addressed_to)
                ),
            )
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
        if kind == "continuation_message" and set(value) == {
            "kind",
            "text",
            "addressed_to",
            "routed_to",
            "continuation_generation",
            "continuation",
        }:
            return ContinuationMessagePayload(
                text=value["text"],
                addressed_to=ConversationTarget(value["addressed_to"]),
                routed_to=ConversationTarget(value["routed_to"]),
                continuation_generation=value["continuation_generation"],
                continuation=_context_from_data(value["continuation"]),
            )
        if kind == "question_answer" and set(value) == {
            "kind",
            "question_id",
            "answer",
            "continuation_generation",
            "continuation",
        }:
            return QuestionAnswerPayload(
                question_id=value["question_id"],
                answer=value["answer"],
                continuation_generation=value["continuation_generation"],
                continuation=_context_from_data(value["continuation"]),
            )
        if kind == "exchange_grant" and set(value) == {
            "kind",
            "request_id",
            "continuation_generation",
            "attempted_question",
            "continuation",
            "parent_mode",
            "outer_question_id",
        }:
            return ExchangeGrantPayload(
                request_id=value["request_id"],
                continuation_generation=value["continuation_generation"],
                attempted_question=DirectedAgentQuestion.from_dict(
                    _prepared_mapping(value["attempted_question"], "attempted_question")
                ),
                continuation=_context_from_data(value["continuation"]),
                parent_mode=value["parent_mode"],
                outer_question_id=value["outer_question_id"],
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


class InterventionStatus(str, Enum):
    PENDING_STOP = "pending_stop"
    READY = "ready"
    RESUMING = "resuming"
    RESUMED = "resumed"
    RESUME_OUTCOME_UNKNOWN = "resume_outcome_unknown"
    CANCELED_BY_STOP = "canceled_by_stop"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class _InterventionDirectedBinding:
    """Exact persisted agent-question boundary owned by one intervention."""

    kind: str
    stage: str
    question_id: str
    continuation_pause_id: str
    continuation_state: TaskState
    question_generation: int
    source_run_id: str
    source_agent: ConversationTarget
    source_provider_id: str
    asked_by: ConversationActor
    addressed_to: ConversationTarget
    routed_to: ConversationTarget
    nested_parent_kind: str | None
    parent_question_id: str | None
    parent_continuation_pause_id: str | None
    exchange_id: str | None
    exchange_request_key: str | None
    exchange_ordinal: int | None
    parent_exchange_id: str | None
    parent_exchange_request_key: str | None
    parent_exchange_ordinal: int | None
    next_attempt_id: str | None
    next_run_id: str | None
    next_provider_id: str | None
    next_task_state: TaskState | None
    next_continuation_state: TaskState | None

    def __post_init__(self) -> None:
        if self.kind not in {"initial", "nested_resume"}:
            raise ValueError("intervention directed binding kind is invalid")
        if self.stage not in {"active_question", "next_fable"}:
            raise ValueError("intervention directed binding stage is invalid")
        for name in (
            "question_id", "continuation_pause_id", "source_run_id", "source_provider_id",
        ):
            object.__setattr__(self, name, _prepared_identifier(getattr(self, name), name))
        generation = _require_integer(self.question_generation, "question_generation")
        if generation < 1:
            raise ValueError("intervention directed question generation is invalid")
        object.__setattr__(self, "question_generation", generation)
        if not isinstance(self.continuation_state, TaskState):
            raise ValueError("intervention directed continuation is invalid")
        if self.source_agent not in {ConversationTarget.FABLE, ConversationTarget.SOL}:
            raise ValueError("intervention directed source agent is invalid")
        if not isinstance(self.asked_by, ConversationActor) or self.asked_by not in {
            ConversationActor.FABLE, ConversationActor.SOL,
        }:
            raise ValueError("intervention directed question source is invalid")
        if (
            self.addressed_to not in {ConversationTarget.FABLE, ConversationTarget.SOL}
            or self.routed_to not in {ConversationTarget.FABLE, ConversationTarget.SOL}
            or self.routed_to is not self.source_agent
        ):
            raise ValueError("intervention directed route is invalid")
        if self.nested_parent_kind is None:
            if self.parent_question_id is not None or self.parent_continuation_pause_id is not None:
                raise ValueError("intervention directed parent is invalid")
        elif self.nested_parent_kind == "clarification":
            if self.parent_question_id is not None or self.parent_continuation_pause_id is not None:
                raise ValueError("intervention directed clarification parent is invalid")
        elif self.nested_parent_kind == "question":
            if self.parent_question_id is None or self.parent_continuation_pause_id is None:
                raise ValueError("intervention directed question parent is invalid")
            object.__setattr__(
                self, "parent_question_id",
                _prepared_identifier(self.parent_question_id, "parent_question_id"),
            )
            object.__setattr__(
                self, "parent_continuation_pause_id",
                _prepared_identifier(
                    self.parent_continuation_pause_id, "parent_continuation_pause_id",
                ),
            )
        else:
            raise ValueError("intervention directed parent kind is invalid")
        self._validate_reservation_identity(
            exchange_id=self.exchange_id,
            request_key=self.exchange_request_key,
            ordinal=self.exchange_ordinal,
            prefix="intervention directed",
        )
        self._validate_reservation_identity(
            exchange_id=self.parent_exchange_id,
            request_key=self.parent_exchange_request_key,
            ordinal=self.parent_exchange_ordinal,
            prefix="intervention directed parent",
        )
        if self.parent_question_id is None and self.parent_exchange_id is not None:
            raise ValueError("intervention directed parent reservation is invalid")
        if self.kind == "initial" and self.nested_parent_kind is not None:
            raise ValueError("initial intervention directed binding cannot be nested")
        if self.kind == "nested_resume" and self.nested_parent_kind is None:
            raise ValueError("nested intervention directed binding is not nested")
        next_identity = (
            self.next_attempt_id,
            self.next_run_id,
            self.next_provider_id,
            self.next_task_state,
            self.next_continuation_state,
        )
        if self.stage == "active_question":
            if any(value is not None for value in next_identity):
                raise ValueError("active intervention directed binding has a next stage")
            return
        if self.kind != "nested_resume" or self.source_agent is not ConversationTarget.SOL:
            raise ValueError("next Fable intervention stage is invalid")
        if (
            self.next_attempt_id is None
            or self.next_run_id is None
            or self.next_provider_id is None
        ):
            raise ValueError("next Fable intervention stage identity is incomplete")
        object.__setattr__(
            self, "next_attempt_id", _prepared_identifier(self.next_attempt_id, "next_attempt_id"),
        )
        object.__setattr__(
            self, "next_run_id", _prepared_identifier(self.next_run_id, "next_run_id"),
        )
        object.__setattr__(
            self,
            "next_provider_id",
            _prepared_identifier(self.next_provider_id, "next_provider_id"),
        )
        if not isinstance(self.next_task_state, TaskState):
            raise ValueError("next Fable intervention task state is invalid")
        if self.next_continuation_state is not None and not isinstance(
            self.next_continuation_state, TaskState,
        ):
            raise ValueError("next Fable intervention continuation is invalid")

    def _validate_reservation_identity(
        self,
        *,
        exchange_id: str | None,
        request_key: str | None,
        ordinal: int | None,
        prefix: str,
    ) -> None:
        present = (exchange_id is not None, request_key is not None, ordinal is not None)
        if any(present) and not all(present):
            raise ValueError(f"{prefix} reservation identity is incomplete")
        if exchange_id is None:
            return
        object.__setattr__(
            self,
            "exchange_id" if prefix == "intervention directed" else "parent_exchange_id",
            _prepared_identifier(exchange_id, "exchange_id"),
        )
        object.__setattr__(
            self,
            (
                "exchange_request_key"
                if prefix == "intervention directed"
                else "parent_exchange_request_key"
            ),
            _prepared_identifier(request_key, "request_key"),
        )
        parsed_ordinal = _require_integer(ordinal, "ordinal")
        if parsed_ordinal < 1:
            raise ValueError(f"{prefix} reservation ordinal is invalid")
        object.__setattr__(
            self,
            "exchange_ordinal" if prefix == "intervention directed" else "parent_exchange_ordinal",
            parsed_ordinal,
        )


@dataclass(frozen=True, slots=True)
class InterventionRecord:
    """One durable request to stop an exact run before continuing it."""

    intervention_id: str
    session_id: str
    task_id: str
    revision: int
    addressed_to: ConversationTarget
    routed_to: ConversationTarget
    message: str
    run_id: str
    continuation_state: TaskState
    source_generation: int
    resume_generation: int
    fable_session_id: str | None
    sol_thread_id: str | None
    resume_attempt_id: str | None
    resume_run_id: str | None
    status: InterventionStatus
    created_at: str
    directed_binding: _InterventionDirectedBinding | None = None

    def __post_init__(self) -> None:
        for name in ("intervention_id", "session_id", "task_id", "run_id"):
            object.__setattr__(self, name, _prepared_identifier(getattr(self, name), name))
        revision = _require_integer(self.revision, "revision")
        source_generation = _require_integer(self.source_generation, "source_generation")
        resume_generation = _require_integer(self.resume_generation, "resume_generation")
        if revision < 0 or source_generation < 1 or resume_generation < 1:
            raise ValueError("intervention revision or generation is invalid")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "source_generation", source_generation)
        object.__setattr__(self, "resume_generation", resume_generation)
        if (
            not isinstance(self.addressed_to, ConversationTarget)
            or self.addressed_to not in {ConversationTarget.FABLE, ConversationTarget.SOL}
            or not isinstance(self.routed_to, ConversationTarget)
            or self.routed_to not in {ConversationTarget.FABLE, ConversationTarget.SOL}
        ):
            raise ValueError("intervention recipient is invalid")
        object.__setattr__(self, "message", _intervention_text(self.message))
        if not isinstance(self.continuation_state, TaskState):
            raise ValueError("intervention continuation state is invalid")
        for name in ("fable_session_id", "sol_thread_id", "resume_attempt_id", "resume_run_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _prepared_identifier(value, name))
        if (self.resume_attempt_id is None) != (self.resume_run_id is None):
            raise ValueError("intervention resume ownership is incomplete")
        if not isinstance(self.status, InterventionStatus):
            raise ValueError("intervention status is invalid")
        object.__setattr__(self, "created_at", _require_string(self.created_at, "created_at"))
        if self.directed_binding is not None and not isinstance(
            self.directed_binding, _InterventionDirectedBinding,
        ):
            raise ValueError("intervention directed binding is invalid")


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


_PREVIOUS_INTERVENTION_DIRECTED_BINDING_KEYS = {
    "kind",
    "question_id",
    "continuation_pause_id",
    "continuation_state",
    "question_generation",
    "source_run_id",
    "source_agent",
    "source_provider_id",
    "asked_by",
    "addressed_to",
    "routed_to",
    "nested_parent_kind",
    "parent_question_id",
    "parent_continuation_pause_id",
}

_PRE_STAGE_INTERVENTION_DIRECTED_BINDING_KEYS = _PREVIOUS_INTERVENTION_DIRECTED_BINDING_KEYS | {
    "exchange_id",
    "exchange_request_key",
    "exchange_ordinal",
    "parent_exchange_id",
    "parent_exchange_request_key",
    "parent_exchange_ordinal",
}

_INTERVENTION_DIRECTED_BINDING_KEYS = _PRE_STAGE_INTERVENTION_DIRECTED_BINDING_KEYS | {
    "stage",
    "next_attempt_id",
    "next_run_id",
    "next_provider_id",
    "next_task_state",
    "next_continuation_state",
}


def _encode_intervention_directed_binding(
    binding: _InterventionDirectedBinding,
) -> str:
    return _encode_json({
        "kind": binding.kind,
        "stage": binding.stage,
        "question_id": binding.question_id,
        "continuation_pause_id": binding.continuation_pause_id,
        "continuation_state": binding.continuation_state.value,
        "question_generation": binding.question_generation,
        "source_run_id": binding.source_run_id,
        "source_agent": binding.source_agent.value,
        "source_provider_id": binding.source_provider_id,
        "asked_by": binding.asked_by.value,
        "addressed_to": binding.addressed_to.value,
        "routed_to": binding.routed_to.value,
        "nested_parent_kind": binding.nested_parent_kind,
        "parent_question_id": binding.parent_question_id,
        "parent_continuation_pause_id": binding.parent_continuation_pause_id,
        "exchange_id": binding.exchange_id,
        "exchange_request_key": binding.exchange_request_key,
        "exchange_ordinal": binding.exchange_ordinal,
        "parent_exchange_id": binding.parent_exchange_id,
        "parent_exchange_request_key": binding.parent_exchange_request_key,
        "parent_exchange_ordinal": binding.parent_exchange_ordinal,
        "next_attempt_id": binding.next_attempt_id,
        "next_run_id": binding.next_run_id,
        "next_provider_id": binding.next_provider_id,
        "next_task_state": (
            None if binding.next_task_state is None else binding.next_task_state.value
        ),
        "next_continuation_state": (
            None
            if binding.next_continuation_state is None
            else binding.next_continuation_state.value
        ),
    })


def _decode_intervention_directed_binding(
    raw: object,
) -> _InterventionDirectedBinding | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise RuntimeError("persisted intervention directed binding is invalid")
    value = _decode_mapping(raw, "intervention directed binding")
    if set(value) != _INTERVENTION_DIRECTED_BINDING_KEYS:
        raise RuntimeError("persisted intervention directed binding is invalid")
    try:
        return _InterventionDirectedBinding(
            kind=value["kind"],
            stage=value["stage"],
            question_id=value["question_id"],
            continuation_pause_id=value["continuation_pause_id"],
            continuation_state=TaskState(value["continuation_state"]),
            question_generation=value["question_generation"],
            source_run_id=value["source_run_id"],
            source_agent=ConversationTarget(value["source_agent"]),
            source_provider_id=value["source_provider_id"],
            asked_by=ConversationActor(value["asked_by"]),
            addressed_to=ConversationTarget(value["addressed_to"]),
            routed_to=ConversationTarget(value["routed_to"]),
            nested_parent_kind=value["nested_parent_kind"],
            parent_question_id=value["parent_question_id"],
            parent_continuation_pause_id=value["parent_continuation_pause_id"],
            exchange_id=value["exchange_id"],
            exchange_request_key=value["exchange_request_key"],
            exchange_ordinal=value["exchange_ordinal"],
            parent_exchange_id=value["parent_exchange_id"],
            parent_exchange_request_key=value["parent_exchange_request_key"],
            parent_exchange_ordinal=value["parent_exchange_ordinal"],
            next_attempt_id=value["next_attempt_id"],
            next_run_id=value["next_run_id"],
            next_provider_id=value["next_provider_id"],
            next_task_state=(
                None
                if value["next_task_state"] is None
                else TaskState(value["next_task_state"])
            ),
            next_continuation_state=(
                None
                if value["next_continuation_state"] is None
                else TaskState(value["next_continuation_state"])
            ),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("persisted intervention directed binding is invalid") from error


_MIGRATION_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        repo_root TEXT NOT NULL,
        created_at TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT 'New chat',
        title_initialized INTEGER NOT NULL DEFAULT 0
            CHECK (title_initialized IN (0, 1)),
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
        preparation_id TEXT NOT NULL,
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
        nested_parent_kind TEXT,
        parent_question_id TEXT,
        parent_continuation_pause_id TEXT,
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
    CREATE TABLE IF NOT EXISTS directed_fable_answer_checkpoints (
        preparation_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK (revision >= 1),
        question_id TEXT NOT NULL,
        continuation_generation INTEGER NOT NULL CHECK (continuation_generation >= 1),
        clarification_json TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('PENDING', 'CONSUMED')),
        PRIMARY KEY (preparation_id, question_id),
        FOREIGN KEY (question_id) REFERENCES questions(question_id),
        FOREIGN KEY (task_id, revision) REFERENCES tasks(task_id, revision)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS interventions (
        intervention_id TEXT PRIMARY KEY CHECK (length(intervention_id) BETWEEN 1 AND 128),
        session_id TEXT NOT NULL CHECK (length(session_id) BETWEEN 1 AND 128),
        task_id TEXT NOT NULL CHECK (length(task_id) BETWEEN 1 AND 128),
        revision INTEGER NOT NULL CHECK (revision >= 0),
        addressed_to TEXT NOT NULL CHECK (addressed_to IN ('fable', 'sol')),
        routed_to TEXT NOT NULL CHECK (routed_to IN ('fable', 'sol')),
        message TEXT NOT NULL CHECK (length(message) BETWEEN 1 AND 16384),
        run_id TEXT NOT NULL CHECK (length(run_id) BETWEEN 1 AND 128),
        continuation_state TEXT NOT NULL,
        source_generation INTEGER NOT NULL CHECK (source_generation >= 1),
        resume_generation INTEGER NOT NULL CHECK (resume_generation >= 1),
        fable_session_id TEXT,
        sol_thread_id TEXT,
        resume_attempt_id TEXT CHECK (
            resume_attempt_id IS NULL OR length(resume_attempt_id) BETWEEN 1 AND 128
        ),
        resume_run_id TEXT CHECK (
            resume_run_id IS NULL OR length(resume_run_id) BETWEEN 1 AND 128
        ),
        acknowledgment_id TEXT UNIQUE CHECK (
            acknowledgment_id IS NULL OR length(acknowledgment_id) BETWEEN 1 AND 128
        ),
        directed_binding_json TEXT,
        status TEXT NOT NULL CHECK (status IN (
            'pending_stop', 'ready', 'resuming', 'resumed',
            'resume_outcome_unknown', 'canceled_by_stop', 'failed'
        )),
        created_at TEXT NOT NULL,
        CHECK (
            (resume_attempt_id IS NULL AND resume_run_id IS NULL)
            OR (resume_attempt_id IS NOT NULL AND resume_run_id IS NOT NULL)
        ),
        FOREIGN KEY (session_id) REFERENCES sessions(session_id),
        FOREIGN KEY (task_id, revision) REFERENCES tasks(task_id, revision),
        FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
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
    CREATE UNIQUE INDEX IF NOT EXISTS prepared_actions_preparation_identifier
    ON prepared_actions (preparation_id)
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
    """
    CREATE INDEX IF NOT EXISTS interventions_task_status
    ON interventions (session_id, task_id, revision, status)
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
            self._migrate_intervention_schema()
            self._migrate_directed_conversation_schema()
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _migrate_session_chat_metadata(self) -> None:
        """Add chat projection fields and one-time title markers for legacy chats."""
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
        if "title_initialized" not in columns:
            self._connection.execute(
                """
                ALTER TABLE sessions
                ADD COLUMN title_initialized INTEGER NOT NULL DEFAULT 0
                    CHECK (title_initialized IN (0, 1))
                """
            )
            self._connection.execute(
                "UPDATE sessions SET title_initialized = 1 WHERE title <> ?",
                (_NEW_CHAT_TITLE,),
            )
            self._backfill_legacy_title_initialization_markers()

    def _backfill_legacy_title_initialization_markers(self) -> None:
        """Mark historical first-message chats in bounded pages during this migration."""
        last_sequence = 0
        while True:
            rows = self._connection.execute(
                """
                SELECT sequence, session_id, actor, kind, payload_json
                FROM events
                WHERE sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (last_sequence, _STARTUP_RECOVERY_BATCH_SIZE),
            ).fetchall()
            if not rows:
                return
            for row in rows:
                last_sequence = int(row["sequence"])
                try:
                    payload = _decode_mapping(str(row["payload_json"]), "event payload")
                except RuntimeError:
                    continue
                if not self._is_title_eligible_user_message(
                    str(row["actor"]), str(row["kind"]), payload,
                ):
                    continue
                self._connection.execute(
                    """
                    UPDATE sessions SET title_initialized = 1
                    WHERE session_id = ? AND title_initialized = 0
                    """,
                    (str(row["session_id"]),),
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
        if "nested_parent_kind" not in question_columns:
            self._connection.execute(
                "ALTER TABLE questions ADD COLUMN nested_parent_kind TEXT"
            )
        if "parent_question_id" not in question_columns:
            self._connection.execute(
                "ALTER TABLE questions ADD COLUMN parent_question_id TEXT"
            )
        if "parent_continuation_pause_id" not in question_columns:
            self._connection.execute(
                "ALTER TABLE questions ADD COLUMN parent_continuation_pause_id TEXT"
            )
        self._migrate_intervention_directed_bindings_in_transaction()
        self._validate_nested_question_rows_in_transaction()
        self._connection.execute("DROP INDEX IF EXISTS one_unanswered_question_per_task_revision")
        self._connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS one_unanswered_top_level_question_per_task_revision
            ON questions (task_id, revision)
            WHERE answer_text IS NULL AND nested_parent_kind IS NULL
            """
        )
        checkpoint_columns = {
            str(row["name"])
            for row in self._connection.execute(
                "PRAGMA table_info(directed_fable_answer_checkpoints)"
            )
        }
        if "project_id" not in checkpoint_columns:
            self._connection.execute(
                "ALTER TABLE directed_fable_answer_checkpoints ADD COLUMN project_id TEXT"
            )
            self._connection.execute(
                """
                UPDATE directed_fable_answer_checkpoints
                SET project_id = (
                    SELECT project_id FROM prepared_actions
                    WHERE prepared_actions.preparation_id
                      = directed_fable_answer_checkpoints.preparation_id
                )
                """
            )
            if self._connection.execute(
                "SELECT 1 FROM directed_fable_answer_checkpoints WHERE project_id IS NULL"
            ).fetchone() is not None:
                raise RuntimeError("Fable answer checkpoint migration is unauthenticated")
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS directed_fable_answer_checkpoints_identity
            ON directed_fable_answer_checkpoints (
                project_id, preparation_id, session_id, task_id, revision, status
            )
            """
        )
        self._connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS one_unanswered_nested_question_per_task_revision
            ON questions (task_id, revision)
            WHERE answer_text IS NULL AND nested_parent_kind IS NOT NULL
            """
        )
        self._connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS exchange_grants_permission_identity
            ON exchange_grants (session_id, task_id, revision, permission_id)
            WHERE permission_id IS NOT NULL
            """
        )

    def _migrate_intervention_schema(self) -> None:
        """Keep intervention migration inside the same rollback boundary as all DDL."""
        if self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'interventions'"
        ).fetchone() is None:
            raise RuntimeError("intervention migration did not create its table")
        columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(interventions)")
        }
        if "directed_binding_json" not in columns:
            self._connection.execute(
                "ALTER TABLE interventions ADD COLUMN directed_binding_json TEXT"
            )

    def _migrate_intervention_directed_bindings_in_transaction(self) -> None:
        """Backfill only exact preceding directed identities in bounded pages."""
        if not self._connection.in_transaction:
            raise RuntimeError("intervention binding migration requires a transaction")
        last_rowid = 0
        while True:
            rows = self._connection.execute(
                """
                SELECT rowid, * FROM interventions
                WHERE rowid > ? ORDER BY rowid LIMIT ?
                """,
                (last_rowid, _STARTUP_RECOVERY_BATCH_SIZE),
            ).fetchall()
            if not rows:
                return
            for row in rows:
                last_rowid = int(row["rowid"])
                try:
                    migrated = self._migrate_intervention_directed_binding_row(row)
                except (RuntimeError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        "intervention directed binding migration is unauthenticated"
                    ) from error
                if not migrated:
                    continue
                current = self._connection.execute(
                    "SELECT * FROM interventions WHERE rowid = ?",
                    (last_rowid,),
                ).fetchone()
                if current is None:
                    raise RuntimeError(
                        "intervention directed binding migration is unauthenticated"
                    )
                record = self._intervention_from_row(current)
                if not self._intervention_is_authenticated(
                    record, current["acknowledgment_id"],
                ):
                    raise RuntimeError(
                        "intervention directed binding migration is unauthenticated"
                    )

    def _migrate_intervention_directed_binding_row(self, row: sqlite3.Row) -> bool:
        raw = row["directed_binding_json"]
        legacy_record = dict(row)
        legacy_record["directed_binding_json"] = None
        record = self._intervention_from_row(legacy_record)
        if raw is not None:
            if not isinstance(raw, str):
                raise RuntimeError("persisted intervention directed binding is invalid")
            value = _decode_mapping(raw, "intervention directed binding")
            if set(value) == _INTERVENTION_DIRECTED_BINDING_KEYS:
                _decode_intervention_directed_binding(raw)
                return True
            if set(value) == _PRE_STAGE_INTERVENTION_DIRECTED_BINDING_KEYS:
                question_id = _prepared_identifier(value["question_id"], "question_id")
                question = self.question(question_id)
                if question is None:
                    raise RuntimeError("intervention directed question is missing")
                upgraded = {
                    **value,
                    "stage": "active_question",
                    "next_attempt_id": None,
                    "next_run_id": None,
                    "next_provider_id": None,
                    "next_task_state": None,
                    "next_continuation_state": None,
                }
                encoded = _encode_json(upgraded)
                binding = _decode_intervention_directed_binding(encoded)
                if binding is None:
                    raise RuntimeError("intervention directed binding is missing")
                binding = self._migrate_answered_nested_intervention_stage(
                    record, self.get_task(record.task_id, record.revision), binding,
                )
                encoded = _encode_intervention_directed_binding(binding)
                cursor = self._connection.execute(
                    """
                    UPDATE interventions SET directed_binding_json = ?
                    WHERE rowid = ? AND directed_binding_json = ?
                    """,
                    (encoded, int(row["rowid"]), raw),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("intervention directed binding migration changed")
                return True
            if set(value) != _PREVIOUS_INTERVENTION_DIRECTED_BINDING_KEYS:
                raise RuntimeError("persisted intervention directed binding is invalid")
            question_id = _prepared_identifier(value["question_id"], "question_id")
            question = self.question(question_id)
            if question is None:
                raise RuntimeError("intervention directed question is missing")
            exchange_id, request_key, ordinal = self._intervention_reservation_identity(
                question,
            )
            parent_exchange_id: str | None = None
            parent_request_key: str | None = None
            parent_ordinal: int | None = None
            if question.parent_question_id is not None:
                parent = self.question(question.parent_question_id)
                if parent is None:
                    raise RuntimeError("intervention directed parent is missing")
                parent_exchange_id, parent_request_key, parent_ordinal = (
                    self._intervention_reservation_identity(parent)
                )
            upgraded = {
                **value,
                "exchange_id": exchange_id,
                "exchange_request_key": request_key,
                "exchange_ordinal": ordinal,
                "parent_exchange_id": parent_exchange_id,
                "parent_exchange_request_key": parent_request_key,
                "parent_exchange_ordinal": parent_ordinal,
                "stage": "active_question",
                "next_attempt_id": None,
                "next_run_id": None,
                "next_provider_id": None,
                "next_task_state": None,
                "next_continuation_state": None,
            }
            encoded = _encode_json(upgraded)
            binding = _decode_intervention_directed_binding(encoded)
            if binding is None:
                raise RuntimeError("intervention directed binding is missing")
            binding = self._migrate_answered_nested_intervention_stage(
                record, self.get_task(record.task_id, record.revision), binding,
            )
            encoded = _encode_intervention_directed_binding(binding)
            cursor = self._connection.execute(
                """
                UPDATE interventions SET directed_binding_json = ?
                WHERE rowid = ? AND directed_binding_json = ?
                """,
                (encoded, int(row["rowid"]), raw),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("intervention directed binding migration changed")
            return True

        binding = self._infer_previous_intervention_directed_binding(record)
        if binding is None:
            return False
        cursor = self._connection.execute(
            """
            UPDATE interventions SET directed_binding_json = ?
            WHERE rowid = ? AND directed_binding_json IS NULL
            """,
            (_encode_intervention_directed_binding(binding), int(row["rowid"])),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("intervention directed binding migration changed")
        return True

    def _infer_previous_intervention_directed_binding(
        self, record: InterventionRecord,
    ) -> _InterventionDirectedBinding | None:
        task = self.get_task(record.task_id, record.revision)
        source = self.agent_run(record.run_id)
        terminal_answered = record.status in {
            InterventionStatus.RESUMED,
            InterventionStatus.CANCELED_BY_STOP,
        }
        nested_rows = self._connection.execute(
            """
            SELECT * FROM questions
            WHERE session_id = ? AND task_id = ? AND revision = ?
              AND continuation_generation = ? AND nested_parent_kind IS NOT NULL
            ORDER BY rowid LIMIT 2
            """,
            (record.session_id, record.task_id, record.revision, record.resume_generation),
        ).fetchall()
        if nested_rows:
            if (
                len(nested_rows) != 1
                or record.routed_to is not ConversationTarget.FABLE
                or record.continuation_state not in _SOL_TASK_STATES
                or record.status not in {
                    InterventionStatus.RESUMING,
                    InterventionStatus.RESUME_OUTCOME_UNKNOWN,
                    InterventionStatus.CANCELED_BY_STOP,
                }
            ):
                raise RuntimeError("nested intervention migration is ambiguous")
            question = self._question_from_row(nested_rows[0])
            provider_id = (
                task.fable_session_id
                if question.routed_to is ConversationTarget.FABLE
                else task.sol_thread_id
            )
            child_rows = self._connection.execute(
                """
                SELECT * FROM agent_runs
                WHERE task_id = ? AND revision = ? AND agent = ?
                  AND cli_session_id = ? AND run_id != ?
                  AND (? IS NULL OR run_id != ?)
                ORDER BY rowid LIMIT 2
                """,
                (
                    record.task_id,
                    record.revision,
                    question.routed_to.value,
                    provider_id,
                    record.run_id,
                    record.resume_run_id,
                    record.resume_run_id,
                ),
            ).fetchall()
            if len(child_rows) != 1:
                raise RuntimeError("nested intervention child migration is ambiguous")
            binding = self._make_intervention_directed_binding(
                kind="nested_resume",
                question=question,
                continuation_pause_id=self._question_pause_id(question.question_id),
                continuation_state=TaskState.FABLE_CLARIFYING,
                source_run=self._agent_run_from_row(child_rows[0]),
            )
            return self._migrate_answered_nested_intervention_stage(
                record, task, binding,
            )

        top_level_rows = self._connection.execute(
            """
            SELECT * FROM questions
            WHERE session_id = ? AND task_id = ? AND revision = ?
              AND continuation_generation = ? AND nested_parent_kind IS NULL
              AND routed_to = ? AND (? OR answer_text IS NULL)
            ORDER BY rowid LIMIT 2
            """,
            (
                record.session_id,
                record.task_id,
                record.revision,
                record.resume_generation,
                source.agent,
                int(terminal_answered),
            ),
        ).fetchall()
        if len(top_level_rows) == 1:
            question = self._question_from_row(top_level_rows[0])
            expected_source = _INTERVENTION_SOURCE_AGENTS.get(record.continuation_state)
            if terminal_answered:
                if source.agent == expected_source:
                    return None
                if (
                    record.resume_attempt_id is None
                    or record.resume_run_id is None
                    or source.agent not in {
                        ConversationTarget.FABLE.value,
                        ConversationTarget.SOL.value,
                    }
                    or question.answered_by is None
                    or question.answered_by.value != source.agent
                ):
                    raise RuntimeError("terminal intervention migration is ambiguous")
            return self._make_intervention_directed_binding(
                kind="initial",
                question=question,
                continuation_pause_id=self._question_pause_id(question.question_id),
                continuation_state=record.continuation_state,
                source_run=source,
            )
        expected_source = _INTERVENTION_SOURCE_AGENTS.get(record.continuation_state)
        if len(top_level_rows) > 1 or source.agent != expected_source:
            raise RuntimeError("intervention directed migration is ambiguous")
        return None

    def _migrate_answered_nested_intervention_stage(
        self,
        record: InterventionRecord,
        task: TaskRecord,
        binding: _InterventionDirectedBinding,
    ) -> _InterventionDirectedBinding:
        """Derive one durable successor only from an exact legacy answered child."""
        question = self.question(binding.question_id)
        if question is None:
            raise RuntimeError("intervention directed question is missing")
        if question.answer_text is None:
            return binding
        if (
            binding.kind != "nested_resume"
            or binding.stage != "active_question"
            or binding.source_agent is not ConversationTarget.SOL
            or record.status not in {
                InterventionStatus.RESUMING,
                InterventionStatus.RESUME_OUTCOME_UNKNOWN,
            }
            or record.routed_to is not ConversationTarget.FABLE
            or record.continuation_state not in _SOL_TASK_STATES
            or record.resume_attempt_id is None
            or record.resume_run_id is None
            or task.fable_session_id != record.fable_session_id
            or task.fable_session_id is None
            or question.answered_by is not ConversationActor.SOL
            or question.nested_parent_kind != binding.nested_parent_kind
            or question.continuation_generation != binding.question_generation
            or question.asked_by is not binding.asked_by
            or question.addressed_to is not binding.addressed_to
            or question.routed_to is not binding.routed_to
            or question.parent_question_id != binding.parent_question_id
            or question.parent_continuation_pause_id
            != binding.parent_continuation_pause_id
        ):
            raise RuntimeError("answered nested intervention migration is ambiguous")
        _, child_continuation, _, child_pause = self._question_exact(
            session_id=record.session_id,
            task_id=record.task_id,
            revision=record.revision,
            expected_generation=binding.question_generation,
            question_id=question.question_id,
        )
        source = self.agent_run(binding.source_run_id)
        if (
            child_continuation is not TaskState.FABLE_CLARIFYING
            or child_pause != binding.continuation_pause_id
            or source.task_id != record.task_id
            or source.revision != record.revision
            or source.agent != ConversationTarget.SOL.value
            or source.cli_session_id != record.sol_thread_id
            or source.cli_session_id != binding.source_provider_id
        ):
            raise RuntimeError("answered nested intervention migration is ambiguous")
        if question.nested_parent_kind == "clarification":
            next_task_state = TaskState.FABLE_CLARIFYING
            next_continuation_state = None
        elif question.nested_parent_kind == "question":
            if question.parent_question_id is None:
                raise RuntimeError("answered nested intervention parent is missing")
            parent, parent_continuation, _, parent_pause = self._question_exact(
                session_id=record.session_id,
                task_id=record.task_id,
                revision=record.revision,
                expected_generation=binding.question_generation,
                question_id=question.parent_question_id,
            )
            if (
                parent.answer_text is not None
                or parent.nested_parent_kind is not None
                or parent.asked_by is not ConversationActor.SOL
                or parent.routed_to is not ConversationTarget.FABLE
                or parent_continuation not in _SOL_TASK_STATES
                or parent_pause != binding.parent_continuation_pause_id
            ):
                raise RuntimeError("answered nested intervention parent changed")
            next_task_state = TaskState.AWAITING_USER_INPUT
            next_continuation_state = parent_continuation
        else:
            raise RuntimeError("answered nested intervention parent is invalid")
        resumed_continuation = next_continuation_state or next_task_state
        if record.status is InterventionStatus.RESUMING:
            task_matches = (
                task.state is next_task_state
                and task.continuation_state is next_continuation_state
            )
        else:
            task_matches = (
                task.state is TaskState.INTERRUPTED
                and task.continuation_state is resumed_continuation
            )
        if not task_matches:
            raise RuntimeError("answered nested intervention continuation changed")
        seed = _encode_json({
            "intervention_id": record.intervention_id,
            "resume_generation": record.resume_generation,
            "resume_attempt_id": record.resume_attempt_id,
            "question_id": binding.question_id,
            "source_run_id": binding.source_run_id,
        })
        next_run_id = f"migration-next-fable-{hashlib.sha256(seed.encode()).hexdigest()[:40]}"
        self._preallocate_next_fable_intervention_run(
            run_id=next_run_id,
            task_id=record.task_id,
            revision=record.revision,
            provider_id=task.fable_session_id,
            status=(
                "running"
                if record.status is InterventionStatus.RESUMING
                else "interrupted"
            ),
        )
        return replace(
            binding,
            stage="next_fable",
            next_attempt_id=record.resume_attempt_id,
            next_run_id=next_run_id,
            next_provider_id=task.fable_session_id,
            next_task_state=next_task_state,
            next_continuation_state=next_continuation_state,
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
            INSERT INTO sessions (
                session_id, repo_root, created_at, title, title_initialized, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, repo_root, timestamp, _NEW_CHAT_TITLE, 0, timestamp),
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

    def unanswered_question_for_task(self, task_id: str, revision: int) -> QuestionRecord | None:
        """Expose the exact durable-question guard for legacy compatibility routes."""
        return self._unanswered_question_for_task(
            _require_string(task_id, "task_id"), _require_integer(revision, "revision"),
        )

    def current_exchange_permission(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
    ) -> Mapping[str, str | int] | None:
        """Project only one current, ungranted exchange permission for the browser."""
        session_id = _require_string(session_id, "session_id")
        task_id = _require_string(task_id, "task_id")
        revision = _require_integer(revision, "revision")
        row = self._connection.execute(
            """
            SELECT permission.permission_id, permission.revision,
                   permission.continuation_generation
            FROM exchange_permissions AS permission
            JOIN tasks AS task
              ON task.task_id = permission.task_id
             AND task.revision = permission.revision
             AND task.session_id = permission.session_id
            WHERE permission.session_id = ? AND permission.task_id = ?
              AND permission.revision = ? AND permission.grant_request_id IS NULL
              AND task.state = ? AND task.exchange_allowance = 0
              AND task.continuation_generation = permission.continuation_generation
              AND task.continuation_pause_id = permission.continuation_pause_id
            """,
            (session_id, task_id, revision, TaskState.AWAITING_USER_INPUT.value),
        ).fetchone()
        if row is None:
            return None
        permission_id = row["permission_id"]
        if not isinstance(permission_id, str) or not _SAFE_PREPARED_IDENTIFIER.fullmatch(permission_id):
            raise RuntimeError("exchange permission identifier is invalid")
        return {
            "request_id": permission_id,
            "revision": int(row["revision"]),
            "continuation_generation": int(row["continuation_generation"]),
        }

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
        fable_checkpoint: FableClarification | None = None,
        checkpoint_preparation_id: str | None = None,
    ) -> QuestionRecord:
        """CAS one exact pending question, then durably prepare its continuation."""
        session_id, task_id, revision, expected_generation = self._directed_identity(
            session_id, task_id, revision, expected_generation,
        )
        question_id = _prepared_identifier(question_id, "question_id")
        answer_text = _require_string(answer_text, "answer_text")
        if not isinstance(answered_by, ConversationActor):
            raise ValueError("answered_by must be a ConversationActor")
        if fable_checkpoint is not None and (
            answered_by is not ConversationActor.FABLE
            or not isinstance(fable_checkpoint, FableClarification)
        ):
            raise ValueError("Fable checkpoint requires a Fable answer")
        if checkpoint_preparation_id is not None:
            checkpoint_preparation_id = _prepared_identifier(
                checkpoint_preparation_id, "checkpoint_preparation_id",
            )
        if (fable_checkpoint is None) != (checkpoint_preparation_id is None):
            raise ValueError("Fable checkpoint identity is incomplete")
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
                if question.nested_parent_kind is not None:
                    raise RuntimeError("nested questions require their exact nested answer path")
                if self._active_nested_child(question.question_id) is not None:
                    raise RuntimeError("nested question is still active")
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
                if fable_checkpoint is not None:
                    claimed = self._prepared_required(checkpoint_preparation_id)
                    if (
                        claimed.session_id != session_id
                        or claimed.task_id != task_id
                        or claimed.revision != revision
                        or claimed.status != "CLAIMED"
                    ):
                        raise RuntimeError("Fable answer has no exact claimed preparation")
                    self._connection.execute(
                        """
                        INSERT INTO directed_fable_answer_checkpoints (
                            preparation_id, project_id, session_id, task_id, revision, question_id,
                            continuation_generation, clarification_json, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                        """,
                        (
                            checkpoint_preparation_id, claimed.project_id, session_id, task_id,
                            revision, question_id, expected_generation,
                            _encode_json(fable_checkpoint.to_dict()),
                        ),
                    )
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

    def reserve_fable_clarification_evidence_question(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
        question_id: str,
        request_key: str,
        text: str,
        event: ConversationEnvelope,
        intervention_id: str | None = None,
        child_run_id: str | None = None,
    ) -> tuple[ExchangeReservation, QuestionRecord]:
        """Reserve Fable's one Sol evidence question from an active clarification."""
        return self._reserve_nested_fable_evidence_question(
            session_id=session_id, task_id=task_id, revision=revision,
            expected_generation=expected_generation, question_id=question_id,
            request_key=request_key, text=text, event=event,
            parent_kind="clarification", outer_question_id=None,
            intervention_id=intervention_id, child_run_id=child_run_id,
        )

    def reserve_fable_answer_evidence_question(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
        outer_question_id: str,
        question_id: str,
        request_key: str,
        text: str,
        event: ConversationEnvelope,
        intervention_id: str | None = None,
        child_run_id: str | None = None,
    ) -> tuple[ExchangeReservation, QuestionRecord]:
        """Reserve Fable's one Sol evidence question under one paused Sol question."""
        return self._reserve_nested_fable_evidence_question(
            session_id=session_id, task_id=task_id, revision=revision,
            expected_generation=expected_generation, question_id=question_id,
            request_key=request_key, text=text, event=event,
            parent_kind="question", outer_question_id=outer_question_id,
            intervention_id=intervention_id, child_run_id=child_run_id,
        )

    def _reserve_nested_fable_evidence_question(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
        question_id: str,
        request_key: str,
        text: str,
        event: ConversationEnvelope,
        parent_kind: Literal["clarification", "question"],
        outer_question_id: str | None,
        intervention_id: str | None,
        child_run_id: str | None,
    ) -> tuple[ExchangeReservation, QuestionRecord]:
        session_id, task_id, revision, expected_generation = self._directed_identity(
            session_id, task_id, revision, expected_generation,
        )
        question_id = _prepared_identifier(question_id, "question_id")
        request_key = _prepared_identifier(request_key, "request_key")
        if (intervention_id is None) != (child_run_id is None):
            raise ValueError("nested intervention preparation identity is incomplete")
        if intervention_id is not None:
            intervention_id = _prepared_identifier(intervention_id, "intervention_id")
            child_run_id = _prepared_identifier(child_run_id, "child_run_id")
        if outer_question_id is not None:
            outer_question_id = _prepared_identifier(outer_question_id, "outer_question_id")
        self._validate_question_inputs(
            task_id=task_id, revision=revision, expected_generation=expected_generation,
            question_id=question_id, asked_by=ConversationActor.FABLE,
            addressed_to=ConversationTarget.SOL, routed_to=ConversationTarget.SOL,
            text=text, event=event,
        )
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._directed_task_exact(
                    session_id=session_id, task_id=task_id, revision=revision,
                    expected_generation=expected_generation,
                )
                self._require_nested_agent_identity(task)
                intervention: InterventionRecord | None = None
                if intervention_id is not None:
                    intervention = self.authenticated_intervention(intervention_id)
                    binding = (
                        None if intervention is None else intervention.directed_binding
                    )
                    if (
                        intervention is None
                        or intervention.task_id != task_id
                        or intervention.revision != revision
                        or intervention.session_id != session_id
                        or intervention.status is not InterventionStatus.RESUMING
                        or intervention.routed_to is not ConversationTarget.FABLE
                        or intervention.continuation_state not in _SOL_TASK_STATES
                        or intervention.resume_generation != expected_generation
                        or intervention.resume_run_id is None
                        or (
                            binding is not None
                            and (
                                binding.kind != "nested_resume"
                                or binding.question_id != question_id
                                or binding.source_run_id != child_run_id
                            )
                        )
                    ):
                        raise RuntimeError("nested intervention preparation changed")
                    parent_run = self.agent_run(intervention.resume_run_id)
                    if (
                        parent_run.task_id != task_id
                        or parent_run.revision != revision
                        or parent_run.agent != ConversationTarget.FABLE.value
                        or parent_run.cli_session_id != task.fable_session_id
                        or parent_run.status != "completed"
                    ):
                        raise RuntimeError("nested intervention parent run changed")
                existing = self._reservation_for_request_key(
                    session_id=session_id, task_id=task_id, revision=revision,
                    request_key=request_key,
                )
                if existing is not None:
                    reservation, question = existing
                    if (
                        question.question_id != question_id
                        or question.nested_parent_kind != parent_kind
                        or question.parent_question_id != outer_question_id
                        or question.asked_by is not ConversationActor.FABLE
                        or question.routed_to is not ConversationTarget.SOL
                        or question.text != text
                        or question.answer_text is not None
                        or task.state is not TaskState.AWAITING_USER_INPUT
                        or task.continuation_state is not TaskState.FABLE_CLARIFYING
                        or self._directed_pause_id(
                            session_id=session_id, task_id=task_id, revision=revision,
                            expected_generation=expected_generation,
                        ) != self._question_pause_id(question.question_id)
                    ):
                        raise RuntimeError("nested evidence reservation changed")
                    if intervention is not None:
                        rebound = self._intervention_required(intervention.intervention_id)
                        if (
                            rebound.directed_binding is None
                            or rebound.directed_binding.question_id != question.question_id
                            or rebound.directed_binding.source_run_id != child_run_id
                        ):
                            raise RuntimeError("nested intervention reservation changed")
                    return existing
                if intervention is not None and intervention.directed_binding is not None:
                    raise RuntimeError("nested intervention reservation is missing")
                if task.exchange_allowance <= 0:
                    raise RuntimeError("internal exchange allowance is exhausted")
                active_nested = self._connection.execute(
                    """
                    SELECT 1 FROM questions WHERE task_id = ? AND revision = ?
                      AND answer_text IS NULL AND nested_parent_kind IS NOT NULL
                    """,
                    (task_id, revision),
                ).fetchone()
                if active_nested is not None:
                    raise RuntimeError("nested evidence question is already active")
                parent_pause_id: str | None = None
                if parent_kind == "clarification":
                    if (
                        task.state is not TaskState.FABLE_CLARIFYING
                        or task.continuation_state not in {None, TaskState.SOL_RUNNING}
                        or self._directed_pause_id(
                            session_id=session_id, task_id=task_id, revision=revision,
                            expected_generation=expected_generation,
                        ) is not None
                        or outer_question_id is not None
                    ):
                        raise RuntimeError("nested clarification parent changed")
                    parent_pending = task.pending
                else:
                    if outer_question_id is None:
                        raise RuntimeError("nested outer question identity is missing")
                    outer, outer_state, parent_pending, parent_pause_id = self._question_exact(
                        session_id=session_id, task_id=task_id, revision=revision,
                        expected_generation=expected_generation, question_id=outer_question_id,
                    )
                    if (
                        outer.nested_parent_kind is not None
                        or outer.asked_by is not ConversationActor.SOL
                        or outer.routed_to is not ConversationTarget.FABLE
                        or outer.answer_text is not None
                        or outer_state not in _SOL_TASK_STATES
                        or task.state is not TaskState.AWAITING_USER_INPUT
                        or task.continuation_state is not outer_state
                        or task.pending != parent_pending
                        or self._directed_pause_id(
                            session_id=session_id, task_id=task_id, revision=revision,
                            expected_generation=expected_generation,
                        ) != parent_pause_id
                    ):
                        raise RuntimeError("nested outer question parent changed")
                if parent_pending is None:
                    raise RuntimeError("nested Fable continuation is missing")
                pause_id = self._new_continuation_pause_id()
                exchange_id = self._new_exchange_id()
                reservation = ExchangeReservation(
                    exchange_id=exchange_id, question_id=question_id,
                    ordinal=task.exchange_consumed + 1,
                    continuation_generation=expected_generation,
                )
                question = QuestionRecord(
                    question_id=question_id, session_id=session_id, task_id=task_id,
                    revision=revision, continuation_generation=expected_generation,
                    asked_by=ConversationActor.FABLE, addressed_to=ConversationTarget.SOL,
                    routed_to=ConversationTarget.SOL, text=text, exchange_id=exchange_id,
                    answer_text=None, answered_by=None, nested_parent_kind=parent_kind,
                    parent_question_id=outer_question_id,
                    parent_continuation_pause_id=parent_pause_id,
                )
                self._insert_question(
                    question, continuation_state=TaskState.FABLE_CLARIFYING,
                    pending_action=parent_pending, continuation_pause_id=pause_id,
                )
                self._connection.execute(
                    """
                    INSERT INTO exchange_reservations (
                        exchange_id, session_id, task_id, revision, question_id,
                        request_key, ordinal, continuation_generation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (exchange_id, session_id, task_id, revision, question_id,
                     request_key, reservation.ordinal, expected_generation),
                )
                cursor = self._connection.execute(
                    """
                    UPDATE tasks SET exchange_allowance = exchange_allowance - 1,
                        exchange_consumed = exchange_consumed + 1, state = ?,
                        continuation_state = ?, pending_json = ?, continuation_pause_id = ?
                    WHERE task_id = ? AND revision = ? AND session_id = ?
                      AND continuation_generation = ? AND exchange_allowance > 0
                    """,
                    (TaskState.AWAITING_USER_INPUT.value, TaskState.FABLE_CLARIFYING.value,
                     _encode_json(parent_pending), pause_id, task_id, revision, session_id,
                     expected_generation),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("nested evidence question changed concurrently")
                if intervention is not None:
                    if task.sol_thread_id is None or child_run_id is None:
                        raise RuntimeError("nested intervention child identity is missing")
                    try:
                        self._connection.execute(
                            """
                            INSERT INTO agent_runs (
                                run_id, task_id, revision, agent, cli_session_id,
                                started_at, status
                            ) VALUES (?, ?, ?, 'sol', ?, ?, 'running')
                            """,
                            (
                                child_run_id,
                                task_id,
                                revision,
                                task.sol_thread_id,
                                self._timestamp(),
                            ),
                        )
                    except sqlite3.IntegrityError as error:
                        raise RuntimeError(
                            "nested intervention child run changed concurrently"
                        ) from error
                    binding = self._make_intervention_directed_binding(
                        kind="nested_resume",
                        question=question,
                        continuation_pause_id=pause_id,
                        continuation_state=TaskState.FABLE_CLARIFYING,
                        source_run=self.agent_run(child_run_id),
                    )
                    binding_cursor = self._connection.execute(
                        """
                        UPDATE interventions SET directed_binding_json = ?
                        WHERE intervention_id = ? AND status = ?
                          AND resume_generation = ? AND resume_attempt_id = ?
                          AND resume_run_id = ? AND directed_binding_json IS NULL
                        """,
                        (
                            _encode_intervention_directed_binding(binding),
                            intervention.intervention_id,
                            InterventionStatus.RESUMING.value,
                            expected_generation,
                            intervention.resume_attempt_id,
                            intervention.resume_run_id,
                        ),
                    )
                    if binding_cursor.rowcount != 1:
                        raise RuntimeError(
                            "nested intervention binding changed concurrently"
                        )
                emitted.append(self._insert_conversation_event_in_transaction(
                    session_id=session_id, task_id=task_id, event=event,
                ))
            self._publish_committed_events(emitted)
        return reservation, question

    def answer_fable_clarification_evidence_question_and_resume(self, **kwargs: object) -> QuestionRecord:
        """Answer nested evidence and restore the exact active Fable clarification."""
        return self._answer_nested_fable_evidence_question(
            **kwargs, parent_kind="clarification", outer_question_id=None,
        )

    def answer_fable_answer_evidence_question(self, **kwargs: object) -> QuestionRecord:
        """Answer nested evidence and restore the exact outer Sol-to-Fable pause."""
        return self._answer_nested_fable_evidence_question(
            **kwargs, parent_kind="question",
        )

    def _answer_nested_fable_evidence_question(
        self, *, session_id: object, task_id: object, revision: object,
        question_id: object, expected_generation: object, answer_text: object,
        event: object, parent_kind: Literal["clarification", "question"],
        outer_question_id: object | None = None,
        next_fable_run_id: object | None = None,
    ) -> QuestionRecord:
        session_id, task_id, revision, expected_generation = self._directed_identity(
            session_id, task_id, revision, expected_generation,
        )
        question_id = _prepared_identifier(question_id, "question_id")
        answer_text = _require_string(answer_text, "answer_text")
        outer_id = None if outer_question_id is None else _prepared_identifier(
            outer_question_id, "outer_question_id",
        )
        if next_fable_run_id is not None:
            next_fable_run_id = _prepared_identifier(
                next_fable_run_id, "next_fable_run_id",
            )
        if not isinstance(event, ConversationEnvelope):
            raise ValueError("event must be a ConversationEnvelope")
        self._validate_answer_event_binding(
            event=event, task_id=task_id, revision=revision,
            expected_generation=expected_generation, question_id=question_id,
            answer_text=answer_text, answered_by=ConversationActor.SOL,
        )
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._directed_task_exact(
                    session_id=session_id, task_id=task_id, revision=revision,
                    expected_generation=expected_generation,
                )
                self._require_nested_agent_identity(task)
                question, state, pending, pause_id = self._question_exact(
                    session_id=session_id, task_id=task_id, revision=revision,
                    expected_generation=expected_generation, question_id=question_id,
                )
                if (
                    question.nested_parent_kind != parent_kind
                    or question.parent_question_id != outer_id
                    or question.asked_by is not ConversationActor.FABLE
                    or question.routed_to is not ConversationTarget.SOL
                    or question.answer_text is not None
                    or state is not TaskState.FABLE_CLARIFYING
                    or task.state is not TaskState.AWAITING_USER_INPUT
                    or task.continuation_state is not TaskState.FABLE_CLARIFYING
                    or task.pending != pending
                    or self._directed_pause_id(
                        session_id=session_id, task_id=task_id, revision=revision,
                        expected_generation=expected_generation,
                    ) != pause_id
                ):
                    raise RuntimeError("nested evidence continuation changed")
                intervention = self.active_intervention_for_task(task_id, revision)
                binding = None if intervention is None else intervention.directed_binding
                owns_nested_child = (
                    intervention is not None
                    and intervention.status is InterventionStatus.RESUMING
                    and intervention.routed_to is ConversationTarget.FABLE
                    and binding is not None
                    and binding.kind == "nested_resume"
                    and binding.stage == "active_question"
                    and binding.question_id == question_id
                    and binding.source_agent is ConversationTarget.SOL
                    and binding.source_run_id != intervention.resume_run_id
                )
                if owns_nested_child != (next_fable_run_id is not None):
                    raise RuntimeError("nested intervention next Fable identity changed")
                restore_state = TaskState.FABLE_CLARIFYING
                restore_pending = pending
                restore_pause: str | None = None
                if parent_kind == "question":
                    if question.parent_continuation_pause_id is None or outer_id is None:
                        raise RuntimeError("nested outer question identity is invalid")
                    outer, restore_state, restore_pending, restore_pause = self._question_exact(
                        session_id=session_id, task_id=task_id, revision=revision,
                        expected_generation=expected_generation, question_id=outer_id,
                    )
                    if (
                        outer.nested_parent_kind is not None
                        or outer.asked_by is not ConversationActor.SOL
                        or outer.routed_to is not ConversationTarget.FABLE
                        or outer.answer_text is not None
                        or restore_state not in _SOL_TASK_STATES
                        or restore_pause != question.parent_continuation_pause_id
                    ):
                        raise RuntimeError("nested outer question parent changed")
                cursor = self._connection.execute(
                    """
                    UPDATE questions SET answer_text = ?, answered_by = ?
                    WHERE question_id = ? AND answer_text IS NULL AND answered_by IS NULL
                    """,
                    (answer_text, ConversationActor.SOL.value, question_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("nested evidence question changed concurrently")
                cursor = self._connection.execute(
                    """
                    UPDATE tasks SET state = ?, continuation_state = ?, pending_json = ?,
                        continuation_pause_id = ?
                    WHERE task_id = ? AND revision = ? AND session_id = ?
                      AND continuation_generation = ? AND state = ?
                      AND continuation_state = ? AND pending_json = ?
                      AND continuation_pause_id = ?
                    """,
                    (restore_state.value if parent_kind == "clarification" else TaskState.AWAITING_USER_INPUT.value,
                     None if parent_kind == "clarification" else restore_state.value,
                     _encode_json(restore_pending), restore_pause, task_id, revision,
                     session_id, expected_generation, TaskState.AWAITING_USER_INPUT.value,
                     TaskState.FABLE_CLARIFYING.value, _encode_json(pending), pause_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("nested evidence task changed concurrently")
                if intervention is not None and binding is not None and owns_nested_child:
                    if task.fable_session_id is None:
                        raise RuntimeError("nested intervention next Fable provider is missing")
                    self._preallocate_next_fable_intervention_run(
                        run_id=next_fable_run_id,
                        task_id=task_id,
                        revision=revision,
                        provider_id=task.fable_session_id,
                    )
                    next_binding = replace(
                        binding,
                        stage="next_fable",
                        question_generation=question.continuation_generation,
                        next_attempt_id=intervention.resume_attempt_id,
                        next_run_id=next_fable_run_id,
                        next_provider_id=task.fable_session_id,
                        next_task_state=(
                            TaskState.FABLE_CLARIFYING
                            if parent_kind == "clarification"
                            else TaskState.AWAITING_USER_INPUT
                        ),
                        next_continuation_state=(
                            None if parent_kind == "clarification" else restore_state
                        ),
                    )
                    binding_cursor = self._connection.execute(
                        """
                        UPDATE interventions SET directed_binding_json = ?
                        WHERE intervention_id = ? AND status = ?
                          AND resume_generation = ? AND resume_attempt_id = ?
                          AND resume_run_id = ? AND directed_binding_json = ?
                        """,
                        (
                            _encode_intervention_directed_binding(next_binding),
                            intervention.intervention_id,
                            InterventionStatus.RESUMING.value,
                            intervention.resume_generation,
                            intervention.resume_attempt_id,
                            intervention.resume_run_id,
                            _encode_intervention_directed_binding(binding),
                        ),
                    )
                    if binding_cursor.rowcount != 1:
                        raise RuntimeError(
                            "nested intervention next Fable binding changed concurrently"
                        )
                emitted.append(self._insert_conversation_event_in_transaction(
                    session_id=session_id, task_id=task_id, event=event,
                ))
            self._publish_committed_events(emitted)
        return replace(question, answer_text=answer_text, answered_by=ConversationActor.SOL)

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

    def pause_fable_clarification_evidence_permission(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_generation: int,
        attempted_question: DirectedAgentQuestion,
        event: ConversationEnvelope,
    ) -> TaskRecord:
        """Persist the one grantable nested-clarification pause without changing parent."""
        session_id, task_id, revision, expected_generation = self._directed_identity(
            session_id, task_id, revision, expected_generation,
        )
        if (
            not isinstance(attempted_question, DirectedAgentQuestion)
            or attempted_question.addressed_to != "sol"
        ):
            raise ValueError("nested attempted question is invalid")
        self._validate_permission_event_binding(
            event=event, task_id=task_id, revision=revision,
            expected_generation=expected_generation,
        )
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._directed_task_exact(
                    session_id=session_id, task_id=task_id, revision=revision,
                    expected_generation=expected_generation,
                )
                self._require_nested_agent_identity(task)
                if (
                    task.state is not TaskState.FABLE_CLARIFYING
                    or task.continuation_state not in {None, TaskState.SOL_RUNNING}
                    or task.exchange_allowance != 0
                    or task.pending is None
                ):
                    raise RuntimeError("nested clarification permission changed")
                pending = {
                    **task.pending,
                    "attempted_question": attempted_question.to_dict(),
                }
                pause_id = self._new_continuation_pause_id()
                permission_id = self._new_exchange_permission_id()
                cursor = self._connection.execute(
                    """
                    UPDATE tasks SET state = ?, continuation_state = ?, pending_json = ?,
                        continuation_pause_id = ?
                    WHERE task_id = ? AND revision = ? AND session_id = ?
                      AND continuation_generation = ? AND state = ?
                      AND continuation_pause_id IS NULL AND exchange_allowance = 0
                    """,
                    (TaskState.AWAITING_USER_INPUT.value, TaskState.FABLE_CLARIFYING.value,
                     _encode_json(pending), pause_id, task_id, revision, session_id,
                     expected_generation, TaskState.FABLE_CLARIFYING.value),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("nested clarification permission changed")
                self._connection.execute(
                    """
                    INSERT INTO exchange_permissions (
                        permission_id, session_id, task_id, revision,
                        continuation_generation, continuation_pause_id, grant_request_id
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (permission_id, session_id, task_id, revision,
                     expected_generation, pause_id),
                )
                emitted.append(self._insert_conversation_event_in_transaction(
                    session_id=session_id, task_id=task_id, event=event,
                ))
            self._publish_committed_events(emitted)
        return self.get_task(task_id, revision)

    def pause_fable_answer_evidence_permission(
        self, *, session_id: str, task_id: str, revision: int,
        expected_generation: int, outer_question_id: str,
        attempted_question: DirectedAgentQuestion, event: ConversationEnvelope,
    ) -> TaskRecord:
        """Pause one exact outer Sol question before Fable can seek Sol evidence."""
        session_id, task_id, revision, expected_generation = self._directed_identity(
            session_id, task_id, revision, expected_generation,
        )
        outer_question_id = _prepared_identifier(outer_question_id, "outer_question_id")
        if not isinstance(attempted_question, DirectedAgentQuestion) or attempted_question.addressed_to != "sol":
            raise ValueError("nested attempted question is invalid")
        self._validate_permission_event_binding(event=event, task_id=task_id, revision=revision, expected_generation=expected_generation)
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._directed_task_exact(session_id=session_id, task_id=task_id, revision=revision, expected_generation=expected_generation)
                outer, outer_state, pending, outer_pause_id = self._question_exact(
                    session_id=session_id, task_id=task_id, revision=revision,
                    expected_generation=expected_generation, question_id=outer_question_id,
                )
                if (
                    outer.nested_parent_kind is not None or outer.asked_by is not ConversationActor.SOL
                    or outer.routed_to is not ConversationTarget.FABLE or outer.answer_text is not None
                    or outer_state not in _SOL_TASK_STATES or task.state is not TaskState.AWAITING_USER_INPUT
                    or task.continuation_state is not outer_state or task.pending != pending
                    or self._directed_pause_id(session_id=session_id, task_id=task_id, revision=revision, expected_generation=expected_generation) != outer_pause_id
                    or task.exchange_allowance != 0
                ):
                    raise RuntimeError("nested outer evidence permission changed")
                pause_id = self._new_continuation_pause_id()
                permission_id = self._new_exchange_permission_id()
                paused_pending = {**pending, "attempted_question": attempted_question.to_dict()}
                cursor = self._connection.execute(
                    """UPDATE tasks SET continuation_state = ?, pending_json = ?, continuation_pause_id = ?
                    WHERE task_id = ? AND revision = ? AND session_id = ? AND state = ?
                      AND continuation_state = ? AND continuation_generation = ? AND continuation_pause_id = ?""",
                    (TaskState.FABLE_CLARIFYING.value, _encode_json(paused_pending), pause_id,
                     task_id, revision, session_id, TaskState.AWAITING_USER_INPUT.value,
                     outer_state.value, expected_generation, outer_pause_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("nested outer evidence permission changed")
                self._connection.execute(
                    """INSERT INTO exchange_permissions (permission_id, session_id, task_id, revision,
                    continuation_generation, continuation_pause_id, grant_request_id) VALUES (?, ?, ?, ?, ?, ?, NULL)""",
                    (permission_id, session_id, task_id, revision, expected_generation, pause_id),
                )
                emitted.append(self._insert_conversation_event_in_transaction(session_id=session_id, task_id=task_id, event=event))
            self._publish_committed_events(emitted)
        return self.get_task(task_id, revision)

    def restore_fable_answer_parent_for_retry(
        self, *, session_id: str, task_id: str, revision: int,
        expected_generation: int, outer_question_id: str,
    ) -> QuestionRecord:
        """Restore exactly the persisted outer pause after its authenticated +3 grant."""
        session_id, task_id, revision, expected_generation = self._directed_identity(session_id, task_id, revision, expected_generation)
        outer_question_id = _prepared_identifier(outer_question_id, "outer_question_id")
        with self._immediate_transaction():
            task = self._directed_task_exact(session_id=session_id, task_id=task_id, revision=revision, expected_generation=expected_generation)
            outer, outer_state, pending, outer_pause_id = self._question_exact(session_id=session_id, task_id=task_id, revision=revision, expected_generation=expected_generation, question_id=outer_question_id)
            if (outer.nested_parent_kind is not None or outer.answer_text is not None or outer_state not in _SOL_TASK_STATES
                or task.state is not TaskState.FABLE_CLARIFYING or task.continuation_state is not None):
                raise RuntimeError("nested outer evidence retry changed")
            cursor = self._connection.execute(
                """UPDATE tasks SET state = ?, continuation_state = ?, pending_json = ?, continuation_pause_id = ?
                WHERE task_id = ? AND revision = ? AND session_id = ? AND state = ? AND continuation_state IS NULL
                  AND continuation_generation = ?""",
                (TaskState.AWAITING_USER_INPUT.value, outer_state.value, _encode_json(pending), outer_pause_id,
                 task_id, revision, session_id, TaskState.FABLE_CLARIFYING.value, expected_generation),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("nested outer evidence retry changed")
        return outer

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

    @staticmethod
    def _require_nested_agent_identity(task: TaskRecord) -> None:
        if (
            task.approved_at is None
            or task.baseline_id is None
            or task.fable_session_id is None
            or task.sol_thread_id is None
        ):
            raise RuntimeError("nested evidence parent has no exact approved agent identity")

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
              AND nested_parent_kind IS NULL
            """,
            (task_id, revision),
        ).fetchone()
        return None if row is None else self._question_from_row(row)

    def _active_nested_child(self, parent_question_id: str) -> QuestionRecord | None:
        row = self._connection.execute(
            """
            SELECT * FROM questions
            WHERE parent_question_id = ? AND answer_text IS NULL
            """,
            (_prepared_identifier(parent_question_id, "parent_question_id"),),
        ).fetchone()
        return None if row is None else self._question_from_row(row)

    def nested_evidence_for_parent(self, parent_question_id: str) -> str | None:
        """Return one completed, authenticated direct child answer for outer Fable resumption."""
        row = self._connection.execute(
            """
            SELECT * FROM questions
            WHERE parent_question_id = ? AND nested_parent_kind = 'question'
              AND answer_text IS NOT NULL AND answered_by = ?
            ORDER BY rowid DESC LIMIT 1
            """,
            (_prepared_identifier(parent_question_id, "parent_question_id"), ConversationActor.SOL.value),
        ).fetchone()
        if row is None:
            return None
        question = self._question_from_row(row)
        if question.answer_text is None:
            raise RuntimeError("nested evidence answer is invalid")
        return question.answer_text

    def _validate_nested_question_rows_in_transaction(self) -> None:
        """Authenticate the two allowed non-recursive nested question shapes."""
        invalid = self._connection.execute(
            """
            SELECT 1 FROM questions AS child
            LEFT JOIN questions AS parent ON parent.question_id = child.parent_question_id
            WHERE
                (child.nested_parent_kind IS NULL AND (
                    child.parent_question_id IS NOT NULL
                    OR child.parent_continuation_pause_id IS NOT NULL
                ))
                OR (child.nested_parent_kind = 'clarification' AND (
                    child.parent_question_id IS NOT NULL
                    OR child.parent_continuation_pause_id IS NOT NULL
                ))
                OR (child.nested_parent_kind = 'question' AND (
                    child.parent_question_id IS NULL
                    OR child.parent_continuation_pause_id IS NULL
                    OR parent.question_id IS NULL
                    OR parent.nested_parent_kind IS NOT NULL
                    OR parent.session_id != child.session_id
                    OR parent.task_id != child.task_id
                    OR parent.revision != child.revision
                    OR parent.continuation_generation != child.continuation_generation
                    OR parent.continuation_pause_id != child.parent_continuation_pause_id
                ))
                OR child.nested_parent_kind NOT IN ('clarification', 'question')
            LIMIT 1
            """
        ).fetchone()
        if invalid is not None:
            raise RuntimeError("persisted nested question is invalid")
        sibling = self._connection.execute(
            """
            SELECT 1 FROM questions
            WHERE answer_text IS NULL AND nested_parent_kind IS NOT NULL
            GROUP BY task_id, revision HAVING COUNT(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if sibling is not None:
            raise RuntimeError("persisted nested question sibling is invalid")
        active_mismatch = self._connection.execute(
            """
            SELECT 1 FROM questions AS child
            JOIN tasks AS task
              ON task.task_id = child.task_id AND task.revision = child.revision
            LEFT JOIN questions AS parent ON parent.question_id = child.parent_question_id
            WHERE child.answer_text IS NULL AND child.nested_parent_kind IS NOT NULL
              AND (
                (task.state != ? AND NOT (
                    task.state = ? AND task.continuation_state = ?
                    AND EXISTS (
                        SELECT 1
                        FROM interventions AS intervention
                        JOIN agent_runs AS bound_run
                          ON bound_run.run_id = json_extract(
                              intervention.directed_binding_json, '$.source_run_id'
                          )
                        WHERE intervention.task_id = task.task_id
                          AND intervention.revision = task.revision
                          AND intervention.session_id = task.session_id
                          AND intervention.status IN (?, ?, ?, ?)
                          AND intervention.routed_to = ?
                          AND intervention.continuation_state IN (?, ?)
                          AND json_valid(intervention.directed_binding_json)
                          AND json_extract(
                              intervention.directed_binding_json, '$.kind'
                          ) = 'nested_resume'
                          AND json_extract(
                              intervention.directed_binding_json, '$.question_id'
                          ) = child.question_id
                          AND json_extract(
                              intervention.directed_binding_json,
                              '$.continuation_pause_id'
                          ) = child.continuation_pause_id
                          AND json_extract(
                              intervention.directed_binding_json, '$.continuation_state'
                          ) = task.continuation_state
                          AND json_extract(
                              intervention.directed_binding_json, '$.asked_by'
                          ) = child.asked_by
                          AND json_extract(
                              intervention.directed_binding_json, '$.addressed_to'
                          ) = child.addressed_to
                          AND json_extract(
                              intervention.directed_binding_json, '$.routed_to'
                          ) = child.routed_to
                          AND json_extract(
                              intervention.directed_binding_json, '$.nested_parent_kind'
                          ) IS child.nested_parent_kind
                          AND json_extract(
                              intervention.directed_binding_json, '$.parent_question_id'
                          ) IS child.parent_question_id
                          AND json_extract(
                              intervention.directed_binding_json,
                              '$.parent_continuation_pause_id'
                          ) IS child.parent_continuation_pause_id
                          AND bound_run.task_id = task.task_id
                          AND bound_run.revision = task.revision
                          AND bound_run.agent = child.routed_to
                          AND bound_run.agent = json_extract(
                              intervention.directed_binding_json, '$.source_agent'
                          )
                          AND bound_run.cli_session_id = json_extract(
                              intervention.directed_binding_json, '$.source_provider_id'
                          )
                          AND (
                              (intervention.acknowledgment_id IS NULL
                               AND child.continuation_generation = json_extract(
                                   intervention.directed_binding_json,
                                   '$.question_generation'
                               ))
                              OR (intervention.acknowledgment_id IS NOT NULL
                                  AND child.continuation_generation
                                      = intervention.resume_generation)
                          )
                          AND (
                              (intervention.status = ?
                               AND intervention.acknowledgment_id IS NOT NULL
                               AND intervention.resume_attempt_id IS NULL
                               AND intervention.resume_run_id IS NULL)
                              OR (intervention.status != ?
                                  AND intervention.resume_attempt_id IS NOT NULL
                                  AND intervention.resume_run_id IS NOT NULL)
                          )
                    )
                )) OR task.continuation_state != ?
                OR task.continuation_generation != child.continuation_generation
                OR task.continuation_pause_id != child.continuation_pause_id
                OR task.pending_json != child.pending_action_json
                OR (child.nested_parent_kind = 'question' AND (
                    parent.answer_text IS NOT NULL
                    OR parent.continuation_state NOT IN (?, ?)
                    OR parent.pending_action_json != child.pending_action_json
                ))
              )
            LIMIT 1
            """,
            (
                TaskState.AWAITING_USER_INPUT.value,
                TaskState.INTERRUPTED.value,
                TaskState.FABLE_CLARIFYING.value,
                InterventionStatus.RESUMING.value,
                InterventionStatus.RESUME_OUTCOME_UNKNOWN.value,
                InterventionStatus.READY.value,
                InterventionStatus.CANCELED_BY_STOP.value,
                ConversationTarget.FABLE.value,
                TaskState.SOL_RUNNING.value,
                TaskState.SOL_CORRECTING.value,
                InterventionStatus.READY.value,
                InterventionStatus.READY.value,
                TaskState.FABLE_CLARIFYING.value,
                TaskState.SOL_RUNNING.value,
                TaskState.SOL_CORRECTING.value,
            ),
        ).fetchone()
        if active_mismatch is not None:
            raise RuntimeError("persisted nested question state is invalid")

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
                nested_parent_kind, parent_question_id, parent_continuation_pause_id,
                answer_text, answered_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                question.nested_parent_kind,
                question.parent_question_id,
                question.parent_continuation_pause_id,
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
                if payload.addressed_to is None:
                    emitted.append(self._insert_event_in_transaction(
                        session_id, task_id, "user", "message", {"text": payload.text},
                    ))
                else:
                    emitted.append(self._insert_conversation_event_in_transaction(
                        session_id=session_id,
                        task_id=task_id,
                        event=ConversationEnvelope(
                            sender=ConversationActor.USER,
                            addressed_to=payload.addressed_to,
                            routed_to=ConversationTarget.FABLE,
                            message_type=ConversationMessageType.STATEMENT,
                            text=payload.text,
                        ),
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
                if self._unanswered_question_for_task(task_id, revision) is not None:
                    raise RuntimeError("exact directed question answer is required")
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

    def prepare_continuation_message_action(
        self,
        *,
        project_id: str,
        session_id: str,
        task_id: str,
        revision: int,
        generation: int,
        payload: ContinuationMessagePayload,
    ) -> PreparedActionRecord:
        """Atomically record and activate one exact non-question user continuation."""
        project_id, session_id, task_id, revision, generation = self._prepared_identity(
            project_id, session_id, task_id, revision, generation,
        )
        if not isinstance(payload, ContinuationMessagePayload):
            raise ValueError("continuation message preparation payload is invalid")
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._prepared_task_exact(session_id, task_id, revision)
                self._verify_prepared_project_identity(project_id, session_id)
                if (
                    task.continuation_generation != payload.continuation_generation
                    or task.state is not TaskState.AWAITING_USER_INPUT
                    or task.continuation_state is None
                ):
                    raise RuntimeError("task is not eligible for a bound continuation")
                if self._unanswered_question_for_task(task_id, revision) is not None:
                    raise RuntimeError("exact directed questions require their question answer action")
                pause_id = self._directed_pause_id(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=payload.continuation_generation,
                )
                if pause_id is not None and self._exchange_permission_for_pause(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=payload.continuation_generation,
                    continuation_pause_id=pause_id,
                ) is not None:
                    raise RuntimeError("exchange permission requires its grant action")
                self._validate_prepared_context(task, payload.continuation)
                active = task.continuation_state
                if self._target_for_prepared_continuation(payload.continuation) is not payload.routed_to:
                    raise RuntimeError("continuation route does not match its exact continuation")
                record = self._insert_prepared_action(
                    project_id=project_id,
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    action="continuation_message",
                    payload=payload,
                    source_state=TaskState.AWAITING_USER_INPUT,
                    active_state=active,
                    continuation_state=active,
                    pending_context=payload.continuation,
                    previous_preparation_id=None,
                    generation=generation,
                )
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET state = ?, continuation_state = NULL, pending_json = NULL,
                        continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND session_id = ?
                      AND continuation_generation = ? AND state = ?
                      AND continuation_state = ?
                    """,
                    (
                        active.value,
                        task_id,
                        revision,
                        session_id,
                        payload.continuation_generation,
                        TaskState.AWAITING_USER_INPUT.value,
                        active.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("bound continuation changed concurrently")
                emitted.append(self._insert_conversation_event_in_transaction(
                    session_id=session_id,
                    task_id=task_id,
                    event=ConversationEnvelope(
                        sender=ConversationActor.USER,
                        addressed_to=payload.addressed_to,
                        routed_to=payload.routed_to,
                        message_type=ConversationMessageType.STATEMENT,
                        text=payload.text,
                        task_id=task_id,
                        revision=revision,
                        continuation_generation=payload.continuation_generation,
                    ),
                ))
            self._publish_committed_events(emitted)
        return record

    def prepare_question_answer_action(
        self,
        *,
        project_id: str,
        session_id: str,
        task_id: str,
        revision: int,
        generation: int,
        payload: QuestionAnswerPayload,
    ) -> PreparedActionRecord:
        """Atomically answer one user-routed question and create its typed runner row."""
        project_id, session_id, task_id, revision, generation = self._prepared_identity(
            project_id, session_id, task_id, revision, generation,
        )
        if not isinstance(payload, QuestionAnswerPayload):
            raise ValueError("question answer preparation payload is invalid")
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._directed_task_exact(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=payload.continuation_generation,
                )
                self._prepared_task_exact(session_id, task_id, revision)
                self._verify_prepared_project_identity(project_id, session_id)
                (
                    question,
                    continuation_state,
                    question_pending,
                    question_pause_id,
                ) = self._question_exact(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=payload.continuation_generation,
                    question_id=payload.question_id,
                )
                if question.answer_text is not None:
                    raise RuntimeError("question was already answered")
                task_pause_id = self._directed_pause_id_exact(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=payload.continuation_generation,
                )
                if (
                    task.state is not TaskState.AWAITING_USER_INPUT
                    or task.continuation_state is not continuation_state
                    or task.pending != question_pending
                    or task_pause_id != question_pause_id
                    or question.routed_to is not ConversationTarget.USER
                ):
                    raise RuntimeError("question continuation changed concurrently")
                self._validate_prepared_context(task, payload.continuation)
                record = self._insert_prepared_action(
                    project_id=project_id,
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    action="question_answer",
                    payload=payload,
                    source_state=TaskState.AWAITING_USER_INPUT,
                    active_state=continuation_state,
                    continuation_state=continuation_state,
                    pending_context=payload.continuation,
                    previous_preparation_id=None,
                    generation=generation,
                )
                cursor = self._connection.execute(
                    """
                    UPDATE questions SET answer_text = ?, answered_by = ?
                    WHERE question_id = ? AND session_id = ? AND task_id = ?
                      AND revision = ? AND continuation_generation = ?
                      AND answer_text IS NULL AND answered_by IS NULL
                    """,
                    (
                        payload.answer,
                        ConversationActor.USER.value,
                        question.question_id,
                        session_id,
                        task_id,
                        revision,
                        payload.continuation_generation,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("question changed concurrently")
                next_generation = self._reset_internal_exchanges_for_human_direction_in_transaction(
                    cursor,
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=payload.continuation_generation,
                )
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET state = ?, continuation_state = NULL, pending_json = NULL,
                        continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND session_id = ?
                      AND state = ? AND continuation_state = ?
                      AND continuation_generation = ? AND pending_json = ?
                      AND continuation_pause_id = ?
                    """,
                    (
                        continuation_state.value,
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
                reply_target = self._target_for_question_asker(question.asked_by)
                emitted.append(self._insert_conversation_event_in_transaction(
                    session_id=session_id,
                    task_id=task_id,
                    event=ConversationEnvelope(
                        sender=ConversationActor.USER,
                        addressed_to=reply_target,
                        routed_to=reply_target,
                        message_type=ConversationMessageType.ANSWER,
                        text=payload.answer,
                        task_id=task_id,
                        revision=revision,
                        continuation_generation=payload.continuation_generation,
                        reply_to_question_id=question.question_id,
                    ),
                ))
            self._publish_committed_events(emitted)
        return record

    def prepare_exchange_grant_action(
        self,
        *,
        project_id: str,
        session_id: str,
        task_id: str,
        revision: int,
        generation: int,
        payload: ExchangeGrantPayload,
    ) -> PreparedActionRecord:
        """Atomically consume one permission, grant +3, and resume its exact action."""
        project_id, session_id, task_id, revision, generation = self._prepared_identity(
            project_id, session_id, task_id, revision, generation,
        )
        if not isinstance(payload, ExchangeGrantPayload):
            raise ValueError("exchange grant preparation payload is invalid")
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._directed_task_exact(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=payload.continuation_generation,
                )
                self._prepared_task_exact(session_id, task_id, revision)
                self._verify_prepared_project_identity(project_id, session_id)
                existing = self._prepared_exchange_grant_retry(
                    project_id=project_id,
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    payload=payload,
                )
                if existing is not None:
                    return existing
                if (
                    task.state is not TaskState.AWAITING_USER_INPUT
                    or task.continuation_state is None
                    or task.exchange_allowance != 0
                ):
                    raise RuntimeError("task is not awaiting exchange permission")
                self._validate_prepared_context(task, payload.continuation)
                pause_id = self._directed_pause_id_exact(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=payload.continuation_generation,
                )
                permission = self._exchange_permission_for_pause(
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    expected_generation=payload.continuation_generation,
                    continuation_pause_id=pause_id,
                )
                if permission is None or permission["grant_request_id"] is not None:
                    raise RuntimeError("exchange permission already received its grant")
                attempted = (task.pending or {}).get("attempted_question")
                try:
                    stored_attempted = DirectedAgentQuestion.from_dict(
                        _prepared_mapping(attempted, "attempted_question")
                    )
                except ValueError as error:
                    raise RuntimeError("exchange permission is missing its attempted question") from error
                if stored_attempted != payload.attempted_question:
                    raise RuntimeError("exchange permission attempted question changed")
                if payload.outer_question_id is not None:
                    outer, outer_state, outer_pending, outer_pause = self._question_exact(
                        session_id=session_id, task_id=task_id, revision=revision,
                        expected_generation=payload.continuation_generation,
                        question_id=payload.outer_question_id,
                    )
                    if (
                        outer.nested_parent_kind is not None
                        or outer.answer_text is not None
                        or outer.asked_by is not ConversationActor.SOL
                        or outer.routed_to is not ConversationTarget.FABLE
                        or outer_state not in _SOL_TASK_STATES
                        or outer_pending is None
                        or outer_pause is None
                    ):
                        raise RuntimeError("exchange grant outer question changed")
                active = task.continuation_state
                resumed_pending = (
                    task.pending
                    if payload.parent_mode == "clarification"
                    else None
                )
                cursor = self._connection.execute(
                    """
                    UPDATE exchange_permissions SET grant_request_id = ?
                    WHERE permission_id = ? AND grant_request_id IS NULL
                    """,
                    (payload.request_id, permission["permission_id"]),
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
                        payload.request_id,
                        permission["permission_id"],
                        payload.continuation_generation,
                        EXCHANGE_GRANT_SIZE,
                    ),
                )
                record = self._insert_prepared_action(
                    project_id=project_id,
                    session_id=session_id,
                    task_id=task_id,
                    revision=revision,
                    action="exchange_grant",
                    payload=payload,
                    source_state=TaskState.AWAITING_USER_INPUT,
                    active_state=active,
                    continuation_state=active,
                    pending_context=payload.continuation,
                    previous_preparation_id=None,
                    generation=generation,
                )
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET exchange_allowance = exchange_allowance + ?, state = ?,
                        continuation_state = NULL, pending_json = ?,
                        continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND session_id = ?
                      AND continuation_generation = ? AND continuation_pause_id = ?
                      AND state = ? AND continuation_state = ? AND exchange_allowance = 0
                    """,
                    (
                        EXCHANGE_GRANT_SIZE,
                        active.value,
                        None if resumed_pending is None else _encode_json(resumed_pending),
                        task_id,
                        revision,
                        session_id,
                        payload.continuation_generation,
                        pause_id,
                        TaskState.AWAITING_USER_INPUT.value,
                        active.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("exchange grant changed concurrently")
                emitted.append(self._insert_conversation_event_in_transaction(
                    session_id=session_id,
                    task_id=task_id,
                    event=ConversationEnvelope(
                        sender=ConversationActor.USER,
                        addressed_to=ConversationTarget.TEAM,
                        routed_to=ConversationTarget.FABLE,
                        message_type=ConversationMessageType.APPROVAL,
                        text="Allow three more internal exchanges.",
                        task_id=task_id,
                        revision=revision,
                    ),
                ))
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
                checkpoint_pending = (
                    predecessor is not None
                    and self.directed_fable_answer_checkpoint(predecessor) is not None
                )
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET state = ?, continuation_state = NULL, pending_json = ?,
                        continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND session_id = ? AND state = ?
                    """,
                    (
                        active.value,
                        None if not checkpoint_pending else _encode_json(task.pending),
                        task_id, revision, session_id, TaskState.INTERRUPTED.value,
                    ),
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

    def consume_directed_fable_answer_checkpoint(
        self, record: PreparedActionRecord, *, question_id: str,
    ) -> None:
        """Consume one checkpoint only after its reconstructed route succeeded."""
        record = self._checkpoint_record(record)
        with self._immediate_transaction():
            cursor = self._connection.execute(
                """
                UPDATE directed_fable_answer_checkpoints SET status = 'CONSUMED'
                WHERE preparation_id = ? AND project_id = ? AND session_id = ?
                  AND task_id = ? AND revision = ? AND question_id = ? AND status = 'PENDING'
                """,
                (
                    record.preparation_id, record.project_id, record.session_id,
                    record.task_id, record.revision,
                    _prepared_identifier(question_id, "question_id"),
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Fable answer checkpoint changed")

    def handoff_directed_fable_answer_same_scope(
        self, record: PreparedActionRecord,
    ) -> TaskRecord:
        """Durably hand an answered Fable continuation to Sol before Sol runs."""
        record = self._checkpoint_record(record)
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                checkpoint = self.directed_fable_answer_checkpoint(record)
                if checkpoint is None or record.status not in {"CLAIMED", "INTERRUPTED"}:
                    raise RuntimeError("Fable answer checkpoint is not handoff-ready")
                question = self.question(checkpoint.question_id)
                task = self._prepared_task_exact(record.session_id, record.task_id, record.revision)
                if (
                    question is None or question.answer_text != checkpoint.clarification.answer
                    or question.answered_by is not ConversationActor.FABLE
                    or question.continuation_generation != checkpoint.continuation_generation
                    or task.state not in _SOL_TASK_STATES or task.pending is None
                    or task.continuation_generation != checkpoint.continuation_generation
                ):
                    raise RuntimeError("Fable answer checkpoint changed")
                cursor = self._connection.execute(
                    """UPDATE tasks SET pending_json = NULL, continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND state = ? AND pending_json IS NOT NULL""",
                    (task.task_id, task.revision, task.state.value),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("Fable answer checkpoint task changed")
                cursor = self._connection.execute(
                    """UPDATE directed_fable_answer_checkpoints SET status = 'CONSUMED'
                    WHERE preparation_id = ? AND question_id = ? AND status = 'PENDING'""",
                    (record.preparation_id, checkpoint.question_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("Fable answer checkpoint changed")
                emitted.append(self._insert_event_in_transaction(
                    task.session_id, task.task_id, "fable", "clarification",
                    checkpoint.clarification.to_dict(),
                ))
            self._publish_committed_events(emitted)
        return self.get_task(record.task_id, record.revision)

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
                or isinstance(task.pending.get("intervention"), Mapping)
            ):
                pending = task.pending
            else:
                pending = self._prepared_pending_projection(record, reason=reason)
            pause_row = self._connection.execute(
                "SELECT continuation_pause_id FROM tasks WHERE task_id = ? AND revision = ?",
                (task.task_id, task.revision),
            ).fetchone()
            pause_id = None if pause_row is None else pause_row["continuation_pause_id"]
            preserve_pause = (
                pause_id is not None
                and self._unanswered_question_for_task(task.task_id, task.revision) is not None
            )
            task_cursor = self._connection.execute(
                """
                UPDATE tasks
                SET continuation_state = ?, pending_json = ?, continuation_pause_id = ?
                WHERE task_id = ? AND revision = ? AND session_id = ? AND state = ?
                """,
                (
                    continuation.value,
                    _encode_json(pending),
                    pause_id if preserve_pause else None,
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
            self._validate_nested_question_rows_in_transaction()
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
                    elif self._claimed_exchange_grant_checkpoint_child(record, task) is not None:
                        pass
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

    def directed_fable_answer_checkpoint(
        self, record: PreparedActionRecord,
    ) -> DirectedFableAnswerCheckpoint | None:
        """Load one still-pending, exact Fable-answer recovery checkpoint."""
        record = self._checkpoint_record(record)
        row = self._connection.execute(
            """
            SELECT preparation_id, question_id, continuation_generation, clarification_json
            FROM directed_fable_answer_checkpoints
            WHERE preparation_id = ? AND project_id = ? AND session_id = ?
              AND task_id = ? AND revision = ? AND status = 'PENDING'
            ORDER BY rowid DESC LIMIT 1
            """,
            (
                record.preparation_id, record.project_id, record.session_id,
                record.task_id, record.revision,
            ),
        ).fetchone()
        if row is None:
            return None
        try:
            clarification = FableClarification.from_dict(
                _decode_mapping(str(row["clarification_json"]), "Fable answer checkpoint")
            )
            return DirectedFableAnswerCheckpoint(
                preparation_id=str(row["preparation_id"]), question_id=str(row["question_id"]),
                continuation_generation=int(row["continuation_generation"]),
                clarification=clarification,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError("Fable answer checkpoint is invalid") from error

    def _checkpoint_record(self, record: PreparedActionRecord) -> PreparedActionRecord:
        if not isinstance(record, PreparedActionRecord):
            raise ValueError("Fable answer checkpoint record is invalid")
        persisted = self._prepared_required(record.preparation_id)
        if (
            persisted.project_id != record.project_id
            or persisted.session_id != record.session_id
            or persisted.task_id != record.task_id
            or persisted.revision != record.revision
        ):
            raise RuntimeError("Fable answer checkpoint preparation changed")
        return persisted

    def interrupt_claimed_fable_answer_checkpoint(
        self, preparation_id: str, *, generation: int,
    ) -> PreparedActionRecord:
        """Atomically preserve an answered Fable continuation for explicit resume."""
        with self._immediate_transaction():
            record = self._prepared_required(preparation_id)
            self._require_record_generation(record, generation)
            checkpoint = self.directed_fable_answer_checkpoint(record)
            if checkpoint is None or record.status != "CLAIMED":
                raise RuntimeError("Fable answer checkpoint is not interruptible")
            task = self._prepared_task_exact(record.session_id, record.task_id, record.revision)
            if (
                checkpoint.continuation_generation != task.continuation_generation
                or task.state is not record.active_state
                or task.continuation_state is not None
                or task.pending is None
            ):
                raise RuntimeError("Fable answer checkpoint changed")
            require_transition(task.state, TaskState.INTERRUPTED)
            cursor = self._connection.execute(
                """
                UPDATE tasks SET state = ?, continuation_state = ?
                WHERE task_id = ? AND revision = ? AND state = ?
                  AND continuation_generation = ?
                """,
                (
                    TaskState.INTERRUPTED.value, task.state.value,
                    task.task_id, task.revision, task.state.value,
                    checkpoint.continuation_generation,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Fable answer checkpoint task changed")
            cursor = self._connection.execute(
                """
                UPDATE prepared_actions SET status = 'INTERRUPTED', reason = 'adapter_interrupted'
                WHERE preparation_id = ? AND generation = ? AND status = 'CLAIMED'
                """,
                (preparation_id, generation),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Fable answer checkpoint preparation changed")
        return self._prepared_required(preparation_id)

    def resume_recovered_exchange_grant(
        self, preparation_id: str, *, generation: int,
    ) -> PreparedActionRecord:
        """Re-arm exactly one recovered typed grant without recreating its charge."""
        with self._immediate_transaction():
            record = self._prepared_required(preparation_id)
            self._require_record_generation(record, generation)
            if record.action != "exchange_grant" or record.status != "RECOVERED":
                raise RuntimeError("recovered exchange grant is not resumable")
            if not isinstance(record.payload, ExchangeGrantPayload):
                raise RuntimeError("recovered exchange grant payload is invalid")
            task = self._prepared_task_exact(record.session_id, record.task_id, record.revision)
            if (
                task.state is not TaskState.INTERRUPTED
                or task.continuation_state is not record.active_state
                or task.continuation_generation != record.payload.continuation_generation
            ):
                raise RuntimeError("recovered exchange grant task changed")
            cursor = self._connection.execute(
                """UPDATE tasks SET state = ?, continuation_state = NULL
                WHERE task_id = ? AND revision = ? AND session_id = ? AND state = ?
                  AND continuation_state = ? AND continuation_generation = ?""",
                (record.active_state.value, record.task_id, record.revision,
                 record.session_id, TaskState.INTERRUPTED.value,
                 record.active_state.value, record.payload.continuation_generation),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("recovered exchange grant task changed")
            cursor = self._connection.execute(
                """UPDATE prepared_actions SET status = 'PREPARED'
                WHERE preparation_id = ? AND generation = ? AND status = 'RECOVERED'""",
                (record.preparation_id, record.generation),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("recovered exchange grant changed")
        return self._prepared_required(preparation_id)

    def resume_claimed_exchange_grant_checkpoint(
        self, preparation_id: str, *, generation: int,
    ) -> QuestionRecord:
        """Authenticate a post-dispatch nested grant checkpoint without recharging it."""
        with self._immediate_transaction():
            record = self._prepared_required(preparation_id)
            self._require_record_generation(record, generation)
            if record.status not in {"CLAIMED", "RECOVERED"}:
                raise RuntimeError("exchange grant checkpoint is invalid")
            task = self._prepared_task_exact(record.session_id, record.task_id, record.revision)
            child = self._claimed_exchange_grant_checkpoint_child(record, task)
            if child is None:
                raise RuntimeError("exchange grant checkpoint changed")
            if record.status == "RECOVERED":
                cursor = self._connection.execute(
                    "UPDATE prepared_actions SET status = 'CLAIMED' WHERE preparation_id = ? AND generation = ? AND status = 'RECOVERED'",
                    (record.preparation_id, record.generation),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("exchange grant checkpoint changed")
        return child

    def rebind_recovered_exchange_grant_checkpoint(
        self,
        preparation_id: str,
        *,
        old_generation: int,
        generation: int,
        project_id: str,
        session_id: str,
        task_id: str,
        revision: int,
    ) -> PreparedActionRecord:
        """Atomically bind one recovered checkpoint to its newly acquired Hub lease."""
        _, session_id, task_id, revision, generation = self._prepared_identity(
            project_id, session_id, task_id, revision, generation,
        )
        old_generation = _require_integer(old_generation, "old_generation")
        if old_generation < 0:
            raise ValueError("old_generation must be non-negative")
        with self._immediate_transaction():
            record = self._prepared_required(preparation_id)
            if (
                record.project_id != project_id
                or record.session_id != session_id
                or record.task_id != task_id
                or record.revision != revision
                or record.generation != old_generation
                or record.status != "RECOVERED"
            ):
                raise RuntimeError("recovered exchange grant ownership changed")
            task = self._prepared_task_exact(session_id, task_id, revision)
            if self._claimed_exchange_grant_checkpoint_child(record, task) is None:
                raise RuntimeError("recovered exchange grant checkpoint changed")
            cursor = self._connection.execute(
                """UPDATE prepared_actions SET generation = ?, status = 'CLAIMED'
                WHERE preparation_id = ? AND project_id = ? AND session_id = ?
                  AND task_id = ? AND revision = ? AND generation = ?
                  AND status = 'RECOVERED'""",
                (
                    generation, preparation_id, project_id, session_id, task_id,
                    revision, old_generation,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("recovered exchange grant ownership changed")
        return self._prepared_required(preparation_id)

    def _claimed_exchange_grant_checkpoint_child(
        self, record: PreparedActionRecord, task: TaskRecord,
    ) -> QuestionRecord | None:
        """Return only an exact post-dispatch grant checkpoint, without repairing it."""
        payload = record.payload
        if (
            record.action != "exchange_grant"
            or record.status not in {"CLAIMED", "RECOVERED"}
            or not isinstance(payload, ExchangeGrantPayload)
            or payload.parent_mode not in {"clarification", "question"}
            or task.state is not TaskState.AWAITING_USER_INPUT
            or task.continuation_generation != payload.continuation_generation
        ):
            return None
        if payload.parent_mode == "question":
            if (
                payload.outer_question_id is None
                or task.continuation_state not in _SOL_TASK_STATES
            ):
                return None
            child, state, pending, pause_id = self._question_exact(
                session_id=record.session_id, task_id=record.task_id,
                revision=record.revision,
                expected_generation=payload.continuation_generation,
                question_id=payload.outer_question_id,
            )
            if (
                child.nested_parent_kind is not None
                or child.asked_by is not ConversationActor.SOL
                or child.routed_to is not ConversationTarget.FABLE
                or child.answer_text is not None
                or state is not task.continuation_state
                or pending != task.pending
                or pause_id != self._directed_pause_id(
                    session_id=record.session_id, task_id=record.task_id,
                    revision=record.revision,
                    expected_generation=payload.continuation_generation,
                )
            ):
                return None
            return child
        if (
            task.continuation_state is not TaskState.FABLE_CLARIFYING
            or not isinstance(payload.continuation, ClarificationContext)
            or task.pending is None
        ):
            return None
        try:
            self._validate_prepared_context(task, payload.continuation)
        except RuntimeError:
            return None
        rows = self._connection.execute(
            """SELECT * FROM questions WHERE task_id = ? AND revision = ?
            AND continuation_generation = ? AND answer_text IS NULL
            AND nested_parent_kind = 'clarification'""",
            (record.task_id, record.revision, payload.continuation_generation),
        ).fetchall()
        if len(rows) != 1:
            return None
        child = self._question_from_row(rows[0])
        if (
            child.parent_question_id is not None
            or child.asked_by is not ConversationActor.FABLE
            or child.routed_to is not ConversationTarget.SOL
            or child.text != payload.attempted_question.text
            or self._question_pause_id(child.question_id) != self._directed_pause_id(
                session_id=record.session_id, task_id=record.task_id,
                revision=record.revision,
                expected_generation=payload.continuation_generation,
            )
        ):
            return None
        return child

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
    def _target_for_prepared_continuation(
        context: PreparedContinuationContext,
    ) -> ConversationTarget:
        """Derive the only provider target permitted by a typed continuation."""
        while isinstance(context, AnswerContext):
            context = context.underlying_continuation
        if isinstance(context, (ScopeApprovalContext, SolResumeContext)):
            return ConversationTarget.SOL
        if isinstance(context, (ReviewContext, ClarificationContext)):
            return ConversationTarget.FABLE
        raise RuntimeError("prepared continuation has no agent route")

    def _prepared_exchange_grant_retry(
        self,
        *,
        project_id: str,
        session_id: str,
        task_id: str,
        revision: int,
        payload: ExchangeGrantPayload,
    ) -> PreparedActionRecord | None:
        """Return only the same still-runnable typed grant preparation."""
        rows = self._connection.execute(
            """
            SELECT * FROM prepared_actions
            WHERE project_id = ? AND session_id = ? AND task_id = ? AND revision = ?
              AND action = 'exchange_grant'
            ORDER BY rowid DESC
            """,
            (project_id, session_id, task_id, revision),
        ).fetchall()
        for row in rows:
            record = self._prepared_action_from_row(row)
            if not isinstance(record.payload, ExchangeGrantPayload):
                raise RuntimeError("prepared exchange grant payload is invalid")
            if record.payload.request_id != payload.request_id:
                continue
            if record.payload != payload:
                raise RuntimeError("exchange grant request changed")
            if record.status not in {"PREPARED", "CLAIMED"}:
                raise RuntimeError("exchange grant action is no longer runnable")
            return record
        return None

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
        if (
            task.continuation_state is TaskState.FABLE_CLARIFYING
            and isinstance(context, SolResumeContext)
            and not isinstance(pending.get("clarification_prompt"), str)
        ):
            parent = self._unanswered_question_for_task(task.task_id, task.revision)
            if parent is not None and parent.nested_parent_kind is None:
                parent_row = self._connection.execute(
                    "SELECT pending_action_json FROM questions WHERE question_id = ?",
                    (parent.question_id,),
                ).fetchone()
                if parent_row is not None:
                    original = _decode_mapping(parent_row["pending_action_json"], "parent pending")
                    if (
                        context.sol_thread_id == task.sol_thread_id
                        and context.sol_run_id == original.get("sol_run_id")
                        and context.prompt == original.get("prompt")
                    ):
                        return
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
        title = self._current_user_message_title(actor, kind, frozen_payload)
        if title is None:
            self._connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (created_at, session_id),
            )
        elif self._connection.execute(
            """
            UPDATE sessions SET title = ?, updated_at = ?, title_initialized = 1
            WHERE session_id = ? AND title_initialized = 0
            """,
            (title, created_at, session_id),
        ).rowcount != 1:
            self._connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (created_at, session_id),
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

    def replace_pending_context(
        self,
        task_id: str,
        revision: int,
        *,
        expected: TaskState,
        pending: Mapping[str, object],
    ) -> TaskRecord:
        """Replace one active phase's durable continuation before its next run."""
        frozen_pending = freeze_json(pending)
        if not isinstance(frozen_pending, Mapping):
            raise ValueError("pending must be an object")
        cursor = self._connection.execute(
            """
            UPDATE tasks SET pending_json = ?, continuation_pause_id = NULL
            WHERE task_id = ? AND revision = ? AND state = ?
              AND continuation_state IS NULL AND pending_json IS NOT NULL
            """,
            (
                _encode_json(frozen_pending),
                _require_string(task_id, "task_id"),
                _require_integer(revision, "revision"),
                expected.value,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("task has no replaceable pending context")
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

    def interrupt_directed_answer_for_stop(
        self, task_id: str, revision: int, *, run_id: str,
    ) -> TaskRecord:
        """Interrupt only an in-flight routed answer without consuming its pause."""
        task_id = _require_string(task_id, "task_id")
        revision = _require_integer(revision, "revision")
        run_id = _prepared_identifier(run_id, "run_id")
        with self._immediate_transaction():
            task = self.get_task(task_id, revision)
            question = self._unanswered_question_for_task(task_id, revision)
            if (
                task.state is not TaskState.AWAITING_USER_INPUT
                or task.continuation_state not in _ACTIVE_TASK_STATES
                or question is None
                or question.routed_to not in {
                    ConversationTarget.FABLE, ConversationTarget.SOL,
                }
                or question.continuation_generation != task.continuation_generation
            ):
                raise RuntimeError("directed answer identity changed")
            source = self.agent_run(run_id)
            if (
                source.task_id != task_id
                or source.revision != revision
                or source.status != "running"
                or source.agent != question.routed_to.value
            ):
                raise RuntimeError("directed answer identity changed")
            _, continuation, pending, question_pause = self._question_exact(
                session_id=task.session_id, task_id=task_id, revision=revision,
                expected_generation=task.continuation_generation,
                question_id=question.question_id,
            )
            pause = self._directed_pause_id(
                session_id=task.session_id, task_id=task_id, revision=revision,
                expected_generation=task.continuation_generation,
            )
            if (
                continuation is not task.continuation_state
                or pending != task.pending
                or pause != question_pause
            ):
                raise RuntimeError("directed answer identity changed")
            cursor = self._connection.execute(
                """
                UPDATE tasks SET state = ?
                WHERE task_id = ? AND revision = ? AND state = ?
                  AND continuation_state = ? AND continuation_generation = ?
                  AND pending_json = ? AND continuation_pause_id = ?
                """,
                (
                    TaskState.INTERRUPTED.value, task_id, revision,
                    TaskState.AWAITING_USER_INPUT.value, continuation.value,
                    task.continuation_generation, _encode_json(task.pending), pause,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("directed answer identity changed")
        return self.get_task(task_id, revision)

    def interrupt_fable_login_expired(
        self,
        *,
        session_id: str,
        task_id: str,
        revision: int,
        expected_state: TaskState,
        expected_fable_session_id: str | None,
        expected_pending: Mapping[str, object] | None,
        run_id: str,
        event: ConversationEnvelope,
        preparation_id: str,
        generation: int,
    ) -> TaskRecord:
        """Atomically retain one Fable continuation and its fixed login notice."""
        if expected_state not in {
            TaskState.FABLE_PLANNING,
            TaskState.FABLE_CLARIFYING,
            TaskState.FABLE_REVIEWING,
        }:
            raise ValueError("expired Fable login requires a Fable phase")
        session_id = _require_string(session_id, "session_id")
        task_id = _require_string(task_id, "task_id")
        revision = _require_integer(revision, "revision")
        if expected_fable_session_id is not None:
            expected_fable_session_id = _require_string(
                expected_fable_session_id, "expected_fable_session_id",
            )
        frozen_pending = None if expected_pending is None else freeze_json(expected_pending)
        if frozen_pending is not None and not isinstance(frozen_pending, Mapping):
            raise ValueError("expected_pending must be an object or null")
        run_id = _require_string(run_id, "run_id")
        if not isinstance(event, ConversationEnvelope) or (
            event.sender is not ConversationActor.SYSTEM
            or event.addressed_to is not ConversationTarget.USER
            or event.routed_to is not ConversationTarget.USER
            or event.message_type is not ConversationMessageType.STATUS
            or event.text
            != "Fable login expired. Run claude auth login on the host, then Resume."
            or event.question_id is not None
            or event.reply_to_question_id is not None
        ):
            raise ValueError("expired Fable login event is invalid")
        if revision == 0:
            if (
                event.task_id is not None
                or event.revision is not None
                or event.continuation_generation is not None
            ):
                raise ValueError("early planning login event must be unbound")
        elif (
            event.task_id != task_id
            or event.revision != revision
            or not isinstance(event.continuation_generation, int)
        ):
            raise ValueError("expired Fable login event does not bind the exact task")
        preparation_id = _prepared_identifier(preparation_id, "preparation_id")
        generation = _require_integer(generation, "generation")
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                task = self._prepared_task_exact(session_id, task_id, revision)
                if task.state is TaskState.INTERRUPTED:
                    if revision == 0:
                        notice_exists = any(
                            _decode_mapping(row["payload_json"], "conversation event")
                            == event.to_dict()
                            for row in self._connection.execute(
                                """
                                SELECT payload_json FROM events
                                WHERE session_id = ? AND task_id = ? AND kind = 'conversation'
                                """,
                                (session_id, task_id),
                            )
                        )
                    else:
                        notice_exists = self._conversation_event_exists(
                            session_id=session_id,
                            task_id=task_id,
                            sender=event.sender,
                            addressed_to=event.addressed_to,
                            routed_to=event.routed_to,
                            message_type=event.message_type,
                            text=event.text,
                            revision=revision,
                            continuation_generation=event.continuation_generation,
                        )
                    if (
                        task.continuation_state is not expected_state
                        or task.fable_session_id != expected_fable_session_id
                        or task.pending != frozen_pending
                        or not notice_exists
                    ):
                        raise RuntimeError("expired Fable login incident changed concurrently")
                    record = self._prepared_required(preparation_id)
                    self._require_record_generation(record, generation)
                    run = self.agent_run(run_id)
                    if (
                        not self._fable_login_preparation_matches(
                            record, task, expected_state, status="INTERRUPTED",
                        )
                        or run.task_id != task_id
                        or run.revision != revision
                        or run.agent != "fable"
                        or run.status != "interrupted"
                        or run.cli_session_id not in {None, expected_fable_session_id}
                    ):
                        raise RuntimeError("expired Fable login lifecycle changed concurrently")
                    return task
                if task.state is not expected_state:
                    raise RuntimeError("task is not in the expected Fable phase")
                if task.fable_session_id != expected_fable_session_id:
                    raise RuntimeError("Fable identity changed concurrently")
                if task.pending != frozen_pending:
                    raise RuntimeError("Fable pending context changed concurrently")
                if (
                    revision > 0
                    and event.continuation_generation != task.continuation_generation
                ):
                    raise RuntimeError("task continuation generation changed concurrently")
                record = self._prepared_required(preparation_id)
                self._require_record_generation(record, generation)
                run = self.agent_run(run_id)
                if (
                    not self._fable_login_preparation_matches(
                        record, task, expected_state, status="CLAIMED",
                    )
                    or run.task_id != task_id
                    or run.revision != revision
                    or run.agent != "fable"
                    or run.status != "running"
                    or run.cli_session_id not in {None, expected_fable_session_id}
                ):
                    raise RuntimeError("expired Fable login lifecycle changed concurrently")
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET state = ?, continuation_state = ?, continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND session_id = ? AND state = ?
                    """,
                    (
                        TaskState.INTERRUPTED.value,
                        expected_state.value,
                        task_id,
                        revision,
                        session_id,
                        expected_state.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("expired Fable login task changed concurrently")
                cursor = self._connection.execute(
                    """
                    UPDATE prepared_actions SET status = 'INTERRUPTED', reason = 'adapter_interrupted'
                    WHERE preparation_id = ? AND generation = ? AND status = 'CLAIMED'
                    """,
                    (preparation_id, generation),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("expired Fable login preparation changed concurrently")
                cursor = self._connection.execute(
                    """
                    UPDATE agent_runs SET status = 'interrupted', exit_code = ?, ended_at = ?
                    WHERE run_id = ? AND task_id = ? AND revision = ?
                      AND agent = 'fable' AND status = 'running'
                    """,
                    (-1, self._timestamp(), run_id, task_id, revision),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("expired Fable login run changed concurrently")
                emitted.append(self._insert_conversation_event_in_transaction(
                    session_id=session_id, task_id=task_id, event=event,
                ))
            self._publish_committed_events(emitted)
        return self.get_task(task_id, revision)

    def _fable_login_preparation_matches(
        self,
        record: PreparedActionRecord,
        task: TaskRecord,
        expected_state: TaskState,
        *,
        status: Literal["CLAIMED", "INTERRUPTED"],
    ) -> bool:
        """Authenticate the one claimed Hub action that owns a Fable run.

        A prepared Hub action can legitimately begin in Sol and subsequently
        route into Fable clarification or review.  Its typed lineage therefore
        authenticates the action and its context; the exact live Fable phase
        is authenticated separately from the current task row above.
        """
        if (
            record.session_id != task.session_id
            or record.task_id != task.task_id
            or record.revision != task.revision
            or record.status != status
            or (status == "INTERRUPTED" and record.reason != "adapter_interrupted")
            or (status == "CLAIMED" and record.reason is not None)
        ):
            return False
        row = self._connection.execute(
            "SELECT rowid FROM prepared_actions WHERE preparation_id = ?",
            (record.preparation_id,),
        ).fetchone()
        if row is None:
            return False
        try:
            self._verify_prepared_project_identity(record.project_id, record.session_id)
        except RuntimeError:
            return False
        if not self._legacy_prepared_action_is_authenticated(record, int(row["rowid"])):
            return False
        if expected_state is TaskState.FABLE_PLANNING:
            return (
                record.action in {"new_request", "resume"}
                and record.pending_context is None
            )
        return record.action != "new_request"

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
        directed_checkpoint: PreparedActionRecord | None = None,
        clarification: FableClarification | None = None,
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
        if (directed_checkpoint is None) != (clarification is None):
            raise ValueError("directed scope checkpoint is incomplete")
        if directed_checkpoint is not None:
            directed_checkpoint = self._checkpoint_record(directed_checkpoint)
            if clarification is None or not clarification.scope_changed:
                raise ValueError("directed scope checkpoint clarification is invalid")
        with self._immediate_transaction():
            checkpoint: DirectedFableAnswerCheckpoint | None = None
            if directed_checkpoint is not None:
                checkpoint = self.directed_fable_answer_checkpoint(directed_checkpoint)
                source = self._prepared_task_exact(
                    directed_checkpoint.session_id, directed_checkpoint.task_id,
                    directed_checkpoint.revision,
                )
                question = None if checkpoint is None else self.question(checkpoint.question_id)
                if (
                    checkpoint is None or question is None
                    or question.answer_text != clarification.answer
                    or question.answered_by is not ConversationActor.FABLE
                    or source.state not in _SOL_TASK_STATES
                    or source.continuation_generation != checkpoint.continuation_generation
                ):
                    raise RuntimeError("Fable answer checkpoint changed")
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
            if checkpoint is not None and directed_checkpoint is not None:
                cursor = self._connection.execute(
                    """UPDATE directed_fable_answer_checkpoints SET status = 'CONSUMED'
                    WHERE preparation_id = ? AND question_id = ? AND status = 'PENDING'""",
                    (directed_checkpoint.preparation_id, checkpoint.question_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("Fable answer checkpoint changed")
                self._insert_event_in_transaction(
                    session_id, brief.task_id, "fable", "task_brief",
                    {"brief": brief.to_dict()},
                )
                self._insert_event_in_transaction(
                    session_id, brief.task_id, "fable", "clarification",
                    clarification.to_dict(),
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
                title = self._current_user_message_title(actor, kind, frozen_payload)
                if title is None:
                    self._connection.execute(
                        "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                        (created_at, session_id),
                    )
                elif self._connection.execute(
                    """
                    UPDATE sessions SET title = ?, updated_at = ?, title_initialized = 1
                    WHERE session_id = ? AND title_initialized = 0
                    """,
                    (title, created_at, session_id),
                ).rowcount != 1:
                    self._connection.execute(
                        "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                        (created_at, session_id),
                    )
            self._pending_listener_events.append(event)
            if self._dispatching_listener_events:
                return event
            self._dispatching_listener_events = True
        self._drain_event_listeners()
        return event

    def _current_user_message_title(
        self,
        actor: str,
        kind: str,
        payload: Mapping[str, JsonValue],
    ) -> str | None:
        if not self._is_title_eligible_user_message(actor, kind, payload):
            return None
        raw_text = payload.get("text")
        if not isinstance(raw_text, str):
            return _NEW_CHAT_TITLE
        title = " ".join(raw_text.split())
        return _NEW_CHAT_TITLE if not title else title[:MAX_CHAT_TITLE_LENGTH]

    @staticmethod
    def _is_title_eligible_user_message(
        actor: str,
        kind: str,
        payload: Mapping[str, JsonValue],
    ) -> bool:
        if actor != ConversationActor.USER.value:
            return False
        if kind == "message":
            return isinstance(payload.get("text"), str)
        if kind != "conversation":
            return False
        try:
            envelope = ConversationEnvelope.from_dict(payload)
        except ValueError:
            return False
        return (
            envelope.sender is ConversationActor.USER
            and envelope.message_type is ConversationMessageType.STATEMENT
        )


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

    def intervention(self, intervention_id: str) -> InterventionRecord | None:
        row = self._connection.execute(
            "SELECT * FROM interventions WHERE intervention_id = ?",
            (_prepared_identifier(intervention_id, "intervention_id"),),
        ).fetchone()
        return None if row is None else self._intervention_from_row(row)

    def authenticated_intervention(self, intervention_id: str) -> InterventionRecord | None:
        """Return one intervention only when its current durable binding is intact."""
        row = self._connection.execute(
            "SELECT * FROM interventions WHERE intervention_id = ?",
            (_prepared_identifier(intervention_id, "intervention_id"),),
        ).fetchone()
        if row is None:
            return None
        record = self._intervention_from_row(row)
        if not self._intervention_is_authenticated(record, row["acknowledgment_id"]):
            raise RuntimeError("intervention binding is not authenticated")
        return record

    def active_intervention_for_task(
        self, task_id: str, revision: int,
    ) -> InterventionRecord | None:
        """Return the sole Stop-capable intervention for one exact task revision."""
        row = self._connection.execute(
            """
            SELECT * FROM interventions
            WHERE task_id = ? AND revision = ?
              AND status IN ('pending_stop', 'ready', 'resuming')
            ORDER BY created_at DESC, intervention_id DESC LIMIT 1
            """,
            (_prepared_identifier(task_id, "task_id"), _require_integer(revision, "revision")),
        ).fetchone()
        if row is None:
            return None
        record = self._intervention_from_row(row)
        if not self._intervention_is_authenticated(record, row["acknowledgment_id"]):
            raise RuntimeError("intervention binding is not authenticated")
        return record

    def prepared_nested_intervention_run(
        self, *, question_id: str, run_id: str | None = None,
    ) -> AgentRunRecord | None:
        """Return one atomically prepared nested child, authenticated to its question."""
        question_id = _prepared_identifier(question_id, "question_id")
        if run_id is not None:
            run_id = _prepared_identifier(run_id, "run_id")
        rows = self._connection.execute(
            """
            SELECT intervention.*
            FROM interventions AS intervention
            JOIN questions AS question
              ON question.session_id = intervention.session_id
             AND question.task_id = intervention.task_id
             AND question.revision = intervention.revision
            WHERE question.question_id = ? AND intervention.status = ?
            ORDER BY intervention.intervention_id LIMIT 2
            """,
            (question_id, InterventionStatus.RESUMING.value),
        ).fetchall()
        matches: list[tuple[InterventionRecord, object]] = []
        for row in rows:
            record = self._intervention_from_row(row)
            binding = record.directed_binding
            if (
                binding is not None
                and binding.kind == "nested_resume"
                and binding.question_id == question_id
                and (run_id is None or binding.source_run_id == run_id)
            ):
                matches.append((record, row["acknowledgment_id"]))
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError("prepared nested intervention child is ambiguous")
        record, acknowledgment_id = matches[0]
        if not self._intervention_is_authenticated(record, acknowledgment_id):
            raise RuntimeError("prepared nested intervention child is not authenticated")
        binding = record.directed_binding
        if binding is None:
            raise RuntimeError("prepared nested intervention child binding is missing")
        child = self.agent_run(binding.source_run_id)
        if child.status != "running":
            raise RuntimeError("prepared nested intervention child is not active")
        return child

    def start_next_fable_intervention_stage(
        self, intervention_id: str, *, run_id: str,
    ) -> AgentRunRecord:
        """Start only the exact Fable stage preallocated by a child-answer CAS."""
        intervention_id = _prepared_identifier(intervention_id, "intervention_id")
        run_id = _prepared_identifier(run_id, "run_id")
        with self._immediate_transaction():
            record = self.authenticated_intervention(intervention_id)
            if record is None:
                raise RuntimeError("intervention not found")
            binding = record.directed_binding
            task = self.get_task(record.task_id, record.revision)
            if (
                record.status is not InterventionStatus.RESUMING
                or binding is None
                or binding.stage != "next_fable"
                or binding.next_attempt_id != record.resume_attempt_id
                or binding.next_run_id != run_id
                or binding.next_provider_id != task.fable_session_id
                or task.state is not binding.next_task_state
                or task.continuation_state is not binding.next_continuation_state
            ):
                raise RuntimeError("nested intervention next Fable stage changed")
            row = self._connection.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("nested intervention next Fable run is missing")
            prepared = self._agent_run_from_row(row)
            if (
                prepared.task_id != record.task_id
                or prepared.revision != record.revision
                or prepared.agent != ConversationTarget.FABLE.value
                or prepared.cli_session_id != binding.next_provider_id
                or prepared.status != "running"
            ):
                raise RuntimeError("nested intervention next Fable run changed")
        return self.agent_run(run_id)

    def finish_next_fable_intervention_stage(
        self, intervention_id: str, *, run_id: str, exit_code: int | None,
    ) -> InterventionRecord:
        """Commit one completed next-Fable result with its intervention terminalization."""
        intervention_id = _prepared_identifier(intervention_id, "intervention_id")
        run_id = _prepared_identifier(run_id, "run_id")
        if exit_code is not None:
            exit_code = _require_integer(exit_code, "exit_code")
        with self._immediate_transaction():
            record = self.authenticated_intervention(intervention_id)
            if record is None:
                raise RuntimeError("intervention not found")
            binding = record.directed_binding
            task = self.get_task(record.task_id, record.revision)
            if (
                binding is None
                or binding.stage != "next_fable"
                or binding.next_run_id != run_id
                or binding.next_attempt_id != record.resume_attempt_id
                or task.state is not binding.next_task_state
                or task.continuation_state is not binding.next_continuation_state
            ):
                raise RuntimeError("nested intervention next Fable stage changed")
            row = self._connection.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("nested intervention next Fable run is missing")
            successor = self._agent_run_from_row(row)
            if (
                successor.task_id != record.task_id
                or successor.revision != record.revision
                or successor.agent != ConversationTarget.FABLE.value
                or successor.cli_session_id != binding.next_provider_id
            ):
                raise RuntimeError("nested intervention next Fable run changed")
            if record.status is InterventionStatus.RESUMED:
                if successor.status != "completed":
                    raise RuntimeError("nested intervention next Fable completion changed")
                return record
            if record.status is not InterventionStatus.RESUMING or successor.status != "running":
                raise RuntimeError("nested intervention next Fable stage changed")
            run_cursor = self._connection.execute(
                """
                UPDATE agent_runs SET status = 'completed', exit_code = ?, ended_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (exit_code, self._timestamp(), run_id),
            )
            if run_cursor.rowcount != 1:
                raise RuntimeError("nested intervention next Fable completion changed")
            intervention_cursor = self._connection.execute(
                """
                UPDATE interventions SET status = ?
                WHERE intervention_id = ? AND status = ? AND resume_generation = ?
                  AND resume_attempt_id = ? AND resume_run_id = ?
                """,
                (
                    InterventionStatus.RESUMED.value,
                    intervention_id,
                    InterventionStatus.RESUMING.value,
                    record.resume_generation,
                    record.resume_attempt_id,
                    record.resume_run_id,
                ),
            )
            if intervention_cursor.rowcount != 1:
                raise RuntimeError("nested intervention next Fable completion changed")
        return self._intervention_required(intervention_id)

    def handoff_next_fable_intervention_clarification_to_sol(
        self,
        intervention_id: str,
        *,
        run_id: str,
        exit_code: int | None,
        answer: str,
    ) -> TaskRecord:
        """Atomically consume one staged Fable answer before ordinary Sol resumes."""
        intervention_id = _prepared_identifier(intervention_id, "intervention_id")
        run_id = _prepared_identifier(run_id, "run_id")
        if exit_code is not None:
            exit_code = _require_integer(exit_code, "exit_code")
        answer = _intervention_text(answer)
        with self._immediate_transaction():
            record = self.authenticated_intervention(intervention_id)
            if record is None:
                raise RuntimeError("intervention not found")
            binding = record.directed_binding
            task = self.get_task(record.task_id, record.revision)
            if (
                binding is None
                or binding.stage != "next_fable"
                or binding.next_run_id != run_id
                or binding.next_attempt_id != record.resume_attempt_id
                or binding.next_task_state is not TaskState.FABLE_CLARIFYING
                or task.state is not TaskState.FABLE_CLARIFYING
                or task.continuation_state is not None
                or not isinstance(task.sol_thread_id, str)
                or not isinstance((task.pending or {}).get("sol_run_id"), str)
            ):
                raise RuntimeError("nested intervention next Fable stage changed")
            row = self._connection.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("nested intervention next Fable run is missing")
            successor = self._agent_run_from_row(row)
            if (
                successor.task_id != record.task_id
                or successor.revision != record.revision
                or successor.agent != ConversationTarget.FABLE.value
                or successor.cli_session_id != binding.next_provider_id
            ):
                raise RuntimeError("nested intervention next Fable run changed")
            pending = {
                "sol_run_id": (task.pending or {})["sol_run_id"],
                "prompt": answer,
            }
            if record.status is InterventionStatus.RESUMED:
                if successor.status != "completed" or task.state is not TaskState.SOL_RUNNING:
                    raise RuntimeError("nested intervention next Fable handoff changed")
                if task.pending != pending:
                    raise RuntimeError("nested intervention next Fable handoff changed")
                return task
            if record.status is not InterventionStatus.RESUMING or successor.status != "running":
                raise RuntimeError("nested intervention next Fable stage changed")
            run_cursor = self._connection.execute(
                """
                UPDATE agent_runs SET status = 'completed', exit_code = ?, ended_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (exit_code, self._timestamp(), run_id),
            )
            if run_cursor.rowcount != 1:
                raise RuntimeError("nested intervention next Fable handoff changed")
            task_cursor = self._connection.execute(
                """
                UPDATE tasks
                SET state = ?, continuation_state = NULL, pending_json = ?,
                    continuation_pause_id = NULL
                WHERE task_id = ? AND revision = ? AND state = ?
                  AND continuation_state IS NULL
                """,
                (
                    TaskState.SOL_RUNNING.value,
                    _encode_json(freeze_json(pending)),
                    record.task_id,
                    record.revision,
                    TaskState.FABLE_CLARIFYING.value,
                ),
            )
            if task_cursor.rowcount != 1:
                raise RuntimeError("nested intervention next Fable handoff changed")
            intervention_cursor = self._connection.execute(
                """
                UPDATE interventions SET status = ?
                WHERE intervention_id = ? AND status = ? AND resume_generation = ?
                  AND resume_attempt_id = ? AND resume_run_id = ?
                """,
                (
                    InterventionStatus.RESUMED.value,
                    intervention_id,
                    InterventionStatus.RESUMING.value,
                    record.resume_generation,
                    record.resume_attempt_id,
                    record.resume_run_id,
                ),
            )
            if intervention_cursor.rowcount != 1:
                raise RuntimeError("nested intervention next Fable handoff changed")
        return self.get_task(record.task_id, record.revision)

    def _preallocate_next_fable_intervention_run(
        self,
        *,
        run_id: str,
        task_id: str,
        revision: int,
        provider_id: str,
        status: Literal["running", "interrupted"] = "running",
    ) -> None:
        """Persist one exact next-Fable run before the adapter can be invoked."""
        row = self._connection.execute(
            "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,),
        ).fetchone()
        if row is not None:
            existing = self._agent_run_from_row(row)
            if (
                existing.task_id != task_id
                or existing.revision != revision
                or existing.agent != ConversationTarget.FABLE.value
                or existing.cli_session_id != provider_id
                or existing.status != status
            ):
                raise RuntimeError("nested intervention next Fable run changed")
            return
        self._connection.execute(
            """
            INSERT INTO agent_runs (
                run_id, task_id, revision, agent, cli_session_id, started_at, status
            ) VALUES (?, ?, ?, 'fable', ?, ?, ?)
            """,
            (run_id, task_id, revision, provider_id, self._timestamp(), status),
        )

    def create_intervention_and_request_stop(
        self,
        *,
        intervention_id: str,
        session_id: str,
        task_id: str,
        revision: int,
        expected_source_generation: int,
        message: str,
        addressed_to: ConversationTarget,
        routed_to: ConversationTarget,
        run_id: str,
    ) -> InterventionRecord:
        """Atomically persist one exact stop intent and its visible guidance."""
        intervention_id = _prepared_identifier(intervention_id, "intervention_id")
        session_id = _prepared_identifier(session_id, "session_id")
        task_id = _prepared_identifier(task_id, "task_id")
        revision = _require_integer(revision, "revision")
        expected_source_generation = _require_integer(
            expected_source_generation, "expected_source_generation",
        )
        run_id = _prepared_identifier(run_id, "run_id")
        message = _intervention_text(message)
        if revision < 0 or expected_source_generation < 1:
            raise ValueError("intervention revision or source generation is invalid")
        if (
            not isinstance(addressed_to, ConversationTarget)
            or addressed_to not in {ConversationTarget.FABLE, ConversationTarget.SOL}
            or not isinstance(routed_to, ConversationTarget)
            or routed_to not in {ConversationTarget.FABLE, ConversationTarget.SOL}
        ):
            raise ValueError("intervention recipient is invalid")
        emitted: list[StreamEvent] = []
        with self._event_listener_lock:
            with self._immediate_transaction():
                existing = self.intervention(intervention_id)
                if existing is not None:
                    if (
                        existing.session_id == session_id
                        and existing.task_id == task_id
                        and existing.revision == revision
                        and existing.source_generation == expected_source_generation
                        and existing.message == message
                        and existing.addressed_to is addressed_to
                        and existing.routed_to is routed_to
                        and existing.run_id == run_id
                    ):
                        return existing
                    raise RuntimeError("intervention identifier is already bound differently")
                task = self._prepared_task_exact(session_id, task_id, revision)
                directed_answer = task.state is TaskState.AWAITING_USER_INPUT
                if task.state not in _ACTIVE_TASK_STATES and not directed_answer:
                    raise RuntimeError("intervention task is not active")
                if task.continuation_generation != expected_source_generation:
                    raise RuntimeError("intervention source generation changed")
                source_run = self._connection.execute(
                    """
                    SELECT * FROM agent_runs
                    WHERE run_id = ? AND task_id = ? AND revision = ? AND status = 'running'
                    """,
                    (run_id, task_id, revision),
                ).fetchone()
                if source_run is None:
                    raise RuntimeError("intervention source run is not active")
                source = self._agent_run_from_row(source_run)
                directed_binding: _InterventionDirectedBinding | None = None
                if directed_answer:
                    if task.continuation_state not in _ACTIVE_TASK_STATES:
                        raise RuntimeError("intervention directed answer continuation changed")
                    question = self._unanswered_question_for_task(task_id, revision)
                    if (
                        question is None
                        or question.continuation_generation != expected_source_generation
                        or question.routed_to not in {
                            ConversationTarget.FABLE, ConversationTarget.SOL,
                        }
                    ):
                        raise RuntimeError("intervention directed answer identity changed")
                    _, question_continuation, question_pending, question_pause_id = (
                        self._question_exact(
                            session_id=session_id,
                            task_id=task_id,
                            revision=revision,
                            expected_generation=expected_source_generation,
                            question_id=question.question_id,
                        )
                    )
                    task_pause_id = self._directed_pause_id(
                        session_id=session_id,
                        task_id=task_id,
                        revision=revision,
                        expected_generation=expected_source_generation,
                    )
                    if (
                        question_continuation is not task.continuation_state
                        or question_pending != task.pending
                        or question_pause_id != task_pause_id
                        or source.agent != question.routed_to.value
                    ):
                        raise RuntimeError("intervention directed answer identity changed")
                    if source.agent == "fable":
                        self._require_exact_intervention_provider_identity(
                            task.fable_session_id, source.cli_session_id,
                        )
                    else:
                        self._require_exact_intervention_provider_identity(
                            task.sol_thread_id, source.cli_session_id,
                        )
                    continuation_state = task.continuation_state
                    resume_generation = expected_source_generation
                    directed_binding = self._make_intervention_directed_binding(
                        kind="initial",
                        question=question,
                        continuation_pause_id=question_pause_id,
                        continuation_state=continuation_state,
                        source_run=source,
                    )
                else:
                    self._require_intervention_source_identity(
                        task=task,
                        source_state=task.state,
                        source_run=source,
                    )
                    continuation_state = task.state
                    resume_generation = self._reset_internal_exchanges_for_human_direction_in_transaction(
                        self._connection.cursor(),
                        session_id=session_id,
                        task_id=task_id,
                        revision=revision,
                        expected_generation=expected_source_generation,
                    )
                if routed_to is ConversationTarget.SOL and (
                    continuation_state not in _SOL_TASK_STATES
                    or task.approved_at is None
                    or task.sol_thread_id is None
                ):
                    raise RuntimeError("intervention Sol recipient is not eligible")
                if directed_answer:
                    cursor = self._connection.execute(
                        """
                        UPDATE tasks SET state = ?
                        WHERE task_id = ? AND revision = ? AND session_id = ?
                          AND state = ? AND continuation_state = ?
                          AND continuation_generation = ? AND pending_json = ?
                          AND continuation_pause_id = ?
                        """,
                        (
                            TaskState.INTERRUPTED.value, task_id, revision, session_id,
                            TaskState.AWAITING_USER_INPUT.value, continuation_state.value,
                            expected_source_generation, _encode_json(task.pending), task_pause_id,
                        ),
                    )
                else:
                    stop_context = {
                        "intervention": {
                            "intervention_id": intervention_id,
                            "source_generation": expected_source_generation,
                            "source_run_id": run_id,
                            "continuation": (
                                None if task.pending is None else _mutable_json(task.pending)
                            ),
                        },
                    }
                    cursor = self._connection.execute(
                        """
                        UPDATE tasks
                        SET state = ?, continuation_state = ?, pending_json = ?,
                            continuation_pause_id = NULL
                        WHERE task_id = ? AND revision = ? AND session_id = ?
                          AND state = ? AND continuation_generation = ?
                        """,
                        (
                            TaskState.INTERRUPTED.value, continuation_state.value,
                            _encode_json(stop_context), task_id, revision, session_id,
                            continuation_state.value, resume_generation,
                        ),
                    )
                if cursor.rowcount != 1:
                    raise RuntimeError("intervention task changed concurrently")
                self._connection.execute(
                    """
                    INSERT INTO interventions (
                        intervention_id, session_id, task_id, revision,
                        addressed_to, routed_to, message, run_id, continuation_state,
                        source_generation, resume_generation, fable_session_id, sol_thread_id,
                        directed_binding_json, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        intervention_id, session_id, task_id, revision,
                        addressed_to.value, routed_to.value, message, run_id,
                        continuation_state.value, expected_source_generation, resume_generation,
                        task.fable_session_id, task.sol_thread_id,
                        (
                            None if directed_binding is None
                            else _encode_intervention_directed_binding(directed_binding)
                        ),
                        InterventionStatus.PENDING_STOP.value, self._timestamp(),
                    ),
                )
                emitted.append(self._insert_conversation_event_in_transaction(
                    session_id=session_id,
                    task_id=task_id,
                    event=ConversationEnvelope(
                        sender=ConversationActor.USER,
                        addressed_to=addressed_to,
                        routed_to=routed_to,
                        message_type=ConversationMessageType.INTERVENTION,
                        text=message,
                        task_id=None if revision == 0 else task_id,
                        revision=None if revision == 0 else revision,
                        continuation_generation=(
                            None if revision == 0 else expected_source_generation
                        ),
                    ),
                ))
            self._publish_committed_events(emitted)
        record = self.intervention(intervention_id)
        if record is None:
            raise RuntimeError("inserted intervention could not be read")
        return record

    def mark_intervention_ready(
        self, intervention_id: str, *, run_id: str,
    ) -> InterventionRecord:
        intervention_id = _prepared_identifier(intervention_id, "intervention_id")
        run_id = _prepared_identifier(run_id, "run_id")
        with self._immediate_transaction():
            record = self._intervention_required(intervention_id)
            if record.run_id != run_id:
                raise RuntimeError("intervention source run changed")
            if record.status is InterventionStatus.READY:
                return record
            if record.status is not InterventionStatus.PENDING_STOP:
                raise RuntimeError("intervention is not pending stop")
            source_run = self.agent_run(run_id)
            if source_run.status == "running":
                raise RuntimeError("intervention source run is still active")
            cursor = self._connection.execute(
                """
                UPDATE interventions SET status = ?
                WHERE intervention_id = ? AND run_id = ? AND status = ?
                """,
                (
                    InterventionStatus.READY.value, intervention_id, run_id,
                    InterventionStatus.PENDING_STOP.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("intervention status changed concurrently")
        return self._intervention_required(intervention_id)

    def claim_intervention_resume(
        self,
        intervention_id: str,
        *,
        expected_resume_generation: int,
        resume_attempt_id: str,
        resume_run_id: str,
    ) -> InterventionRecord:
        intervention_id = _prepared_identifier(intervention_id, "intervention_id")
        expected_resume_generation = _require_integer(
            expected_resume_generation, "expected_resume_generation",
        )
        resume_attempt_id = _prepared_identifier(resume_attempt_id, "resume_attempt_id")
        resume_run_id = _prepared_identifier(resume_run_id, "resume_run_id")
        with self._immediate_transaction():
            record = self._intervention_required(intervention_id)
            if record.resume_generation != expected_resume_generation:
                raise RuntimeError("intervention resume generation changed")
            if record.status is InterventionStatus.RESUMING:
                if (
                    record.resume_attempt_id == resume_attempt_id
                    and record.resume_run_id == resume_run_id
                ):
                    return record
                raise RuntimeError("intervention resume owner changed")
            if record.status is not InterventionStatus.READY:
                raise RuntimeError("intervention is not ready to resume")
            cursor = self._connection.execute(
                """
                UPDATE interventions
                SET status = ?, resume_attempt_id = ?, resume_run_id = ?
                WHERE intervention_id = ? AND resume_generation = ? AND status = ?
                  AND resume_attempt_id IS NULL AND resume_run_id IS NULL
                """,
                (
                    InterventionStatus.RESUMING.value, resume_attempt_id, resume_run_id,
                    intervention_id, expected_resume_generation, InterventionStatus.READY.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("intervention resume claim changed concurrently")
        return self._intervention_required(intervention_id)

    def begin_intervention_resume(
        self,
        intervention_id: str,
        *,
        expected_resume_generation: int,
        resume_attempt_id: str,
        resume_run_id: str,
    ) -> TaskRecord:
        """Claim one ready intervention and restore only its persisted continuation."""
        intervention_id = _prepared_identifier(intervention_id, "intervention_id")
        expected_resume_generation = _require_integer(
            expected_resume_generation, "expected_resume_generation",
        )
        resume_attempt_id = _prepared_identifier(resume_attempt_id, "resume_attempt_id")
        resume_run_id = _prepared_identifier(resume_run_id, "resume_run_id")
        with self._immediate_transaction():
            record = self._intervention_required(intervention_id)
            row = self._connection.execute(
                "SELECT acknowledgment_id, directed_binding_json FROM interventions WHERE intervention_id = ?",
                (intervention_id,),
            ).fetchone()
            if (
                record.resume_generation != expected_resume_generation
                or record.status is not InterventionStatus.READY
                or not self._intervention_is_authenticated(
                    record, None if row is None else row["acknowledgment_id"],
                )
            ):
                raise RuntimeError("intervention is not ready to resume")
            task = self.get_task(record.task_id, record.revision)
            acknowledgment_id = None if row is None else row["acknowledgment_id"]
            binding = record.directed_binding
            next_binding_json: str | None = None
            if binding is not None:
                if binding.stage == "next_fable":
                    resumed_continuation = (
                        binding.next_continuation_state or binding.next_task_state
                    )
                    if (
                        task.state is not TaskState.INTERRUPTED
                        or task.continuation_state is not resumed_continuation
                    ):
                        raise RuntimeError("intervention next Fable continuation changed")
                    cursor = self._connection.execute(
                        """
                        UPDATE tasks SET state = ?, continuation_state = ?
                        WHERE task_id = ? AND revision = ? AND state = ?
                          AND continuation_state = ? AND continuation_generation = ?
                        """,
                        (
                            binding.next_task_state.value,
                            None if binding.next_continuation_state is None
                            else binding.next_continuation_state.value,
                            record.task_id,
                            record.revision,
                            TaskState.INTERRUPTED.value,
                            resumed_continuation.value,
                            record.resume_generation,
                        ),
                    )
                    if binding.next_provider_id is None:
                        raise RuntimeError("intervention next Fable provider changed")
                    self._preallocate_next_fable_intervention_run(
                        run_id=resume_run_id,
                        task_id=record.task_id,
                        revision=record.revision,
                        provider_id=binding.next_provider_id,
                    )
                    next_binding_json = _encode_intervention_directed_binding(
                        replace(
                            binding,
                            next_attempt_id=resume_attempt_id,
                            next_run_id=resume_run_id,
                        )
                    )
                else:
                    question = self.question(binding.question_id)
                    if (
                        question is None
                        or task.state is not TaskState.INTERRUPTED
                        or task.continuation_state is not binding.continuation_state
                    ):
                        raise RuntimeError("intervention directed answer changed")
                    if binding.kind == "nested_resume" or (
                        binding.kind == "initial"
                        and record.routed_to is ConversationTarget.FABLE
                    ):
                        cursor = self._connection.execute(
                        """
                        UPDATE tasks SET state = ?
                        WHERE task_id = ? AND revision = ? AND state = ?
                          AND continuation_state = ? AND continuation_generation = ?
                          AND pending_json = ? AND continuation_pause_id = ?
                        """,
                        (
                            TaskState.AWAITING_USER_INPUT.value, record.task_id, record.revision,
                            TaskState.INTERRUPTED.value, binding.continuation_state.value,
                            record.resume_generation,
                            _encode_json(task.pending), binding.continuation_pause_id,
                        ),
                        )
                    elif (
                        binding.kind == "initial"
                        and
                        record.routed_to is ConversationTarget.SOL
                        and record.continuation_state in _SOL_TASK_STATES
                    ):
                        cursor = self._connection.execute(
                        """
                        UPDATE tasks
                        SET state = ?, continuation_state = NULL, pending_json = NULL,
                            continuation_pause_id = NULL
                        WHERE task_id = ? AND revision = ? AND state = ?
                          AND continuation_state = ? AND continuation_generation = ?
                        """,
                        (
                            record.continuation_state.value,
                            record.task_id,
                            record.revision,
                            TaskState.INTERRUPTED.value,
                            record.continuation_state.value,
                            record.resume_generation,
                        ),
                        )
                    else:
                        raise RuntimeError("intervention directed route changed")
            else:
                cross_route = (
                    record.routed_to is ConversationTarget.FABLE
                    and record.continuation_state in _SOL_TASK_STATES
                )
                resumed_continuation = (
                    TaskState.FABLE_CLARIFYING if cross_route
                    else record.continuation_state
                )
                expected_interrupted_continuation = record.continuation_state
                if acknowledgment_id is not None:
                    if (
                        task.state is not TaskState.INTERRUPTED
                        or task.continuation_state is not resumed_continuation
                    ):
                        raise RuntimeError("intervention continuation changed")
                    restored_state = resumed_continuation
                    restored_pending = task.pending
                    expected_interrupted_continuation = resumed_continuation
                else:
                    pending = task.pending or {}
                    stopped = pending.get("intervention")
                    if (
                        task.state is not TaskState.INTERRUPTED
                        or task.continuation_state is not record.continuation_state
                        or not isinstance(stopped, Mapping)
                        or stopped.get("intervention_id") != record.intervention_id
                        or stopped.get("source_generation") != record.source_generation
                        or stopped.get("source_run_id") != record.run_id
                    ):
                        raise RuntimeError("intervention continuation changed")
                    continuation = stopped.get("continuation")
                    if continuation is not None and not isinstance(continuation, Mapping):
                        raise RuntimeError("intervention continuation changed")
                    if cross_route:
                        if not isinstance(continuation, Mapping):
                            raise RuntimeError("intervention continuation changed")
                        restored_state = TaskState.FABLE_CLARIFYING
                        restored_pending = {
                            **continuation,
                            "clarification_prompt": record.message,
                        }
                    else:
                        restored_state = record.continuation_state
                        restored_pending = (
                            None if record.routed_to is ConversationTarget.SOL else continuation
                        )
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET state = ?, continuation_state = NULL, pending_json = ?,
                        continuation_pause_id = NULL
                    WHERE task_id = ? AND revision = ? AND state = ?
                      AND continuation_state = ? AND continuation_generation = ?
                    """,
                    (
                        restored_state.value,
                        None if restored_pending is None else _encode_json(restored_pending),
                        record.task_id, record.revision, TaskState.INTERRUPTED.value,
                        expected_interrupted_continuation.value, record.resume_generation,
                    ),
                )
            if cursor.rowcount != 1:
                raise RuntimeError("intervention continuation changed")
            if next_binding_json is None:
                cursor = self._connection.execute(
                    """
                    UPDATE interventions
                    SET status = ?, resume_attempt_id = ?, resume_run_id = ?
                    WHERE intervention_id = ? AND resume_generation = ? AND status = ?
                      AND resume_attempt_id IS NULL AND resume_run_id IS NULL
                    """,
                    (
                        InterventionStatus.RESUMING.value, resume_attempt_id, resume_run_id,
                        intervention_id, expected_resume_generation, InterventionStatus.READY.value,
                    ),
                )
            else:
                cursor = self._connection.execute(
                    """
                    UPDATE interventions
                    SET status = ?, resume_attempt_id = ?, resume_run_id = ?,
                        directed_binding_json = ?
                    WHERE intervention_id = ? AND resume_generation = ? AND status = ?
                      AND resume_attempt_id IS NULL AND resume_run_id IS NULL
                      AND directed_binding_json = ?
                    """,
                    (
                        InterventionStatus.RESUMING.value,
                        resume_attempt_id,
                        resume_run_id,
                        next_binding_json,
                        intervention_id,
                        expected_resume_generation,
                        InterventionStatus.READY.value,
                        None if row is None else row["directed_binding_json"],
                    ),
                )
            if cursor.rowcount != 1:
                raise RuntimeError("intervention resume claim changed concurrently")
        return self.get_task(record.task_id, record.revision)

    def complete_intervention(
        self,
        intervention_id: str,
        *,
        expected_resume_generation: int,
        resume_attempt_id: str,
        resume_run_id: str,
    ) -> InterventionRecord:
        intervention_id = _prepared_identifier(intervention_id, "intervention_id")
        expected_resume_generation = _require_integer(
            expected_resume_generation, "expected_resume_generation",
        )
        resume_attempt_id = _prepared_identifier(resume_attempt_id, "resume_attempt_id")
        resume_run_id = _prepared_identifier(resume_run_id, "resume_run_id")
        with self._immediate_transaction():
            record = self._intervention_required(intervention_id)
            if record.resume_generation != expected_resume_generation:
                raise RuntimeError("intervention resume generation changed")
            if (
                record.resume_attempt_id != resume_attempt_id
                or record.resume_run_id != resume_run_id
            ):
                raise RuntimeError("intervention resume owner changed")
            if record.status is InterventionStatus.RESUMED:
                return record
            if record.status is not InterventionStatus.RESUMING:
                raise RuntimeError("intervention is not resuming")
            cursor = self._connection.execute(
                """
                UPDATE interventions SET status = ?
                WHERE intervention_id = ? AND resume_generation = ?
                  AND resume_attempt_id = ? AND resume_run_id = ? AND status = ?
                """,
                (
                    InterventionStatus.RESUMED.value, intervention_id,
                    expected_resume_generation, resume_attempt_id, resume_run_id,
                    InterventionStatus.RESUMING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("intervention completion changed concurrently")
        return self._intervention_required(intervention_id)

    def cancel_intervention_by_stop(
        self, intervention_id: str, *, expected_resume_generation: int,
    ) -> InterventionRecord:
        intervention_id = _prepared_identifier(intervention_id, "intervention_id")
        expected_resume_generation = _require_integer(
            expected_resume_generation, "expected_resume_generation",
        )
        with self._immediate_transaction():
            record = self._intervention_required(intervention_id)
            if record.resume_generation != expected_resume_generation:
                raise RuntimeError("intervention resume generation changed")
            if record.status is InterventionStatus.CANCELED_BY_STOP:
                return record
            if record.status not in {
                InterventionStatus.PENDING_STOP,
                InterventionStatus.READY,
                InterventionStatus.RESUMING,
            }:
                raise RuntimeError("intervention cannot be canceled")
            task = self.get_task(record.task_id, record.revision)
            record = self._bind_nested_intervention_resume_in_transaction(record, task)
            if task.state in _ACTIVE_TASK_STATES:
                cursor = self._connection.execute(
                    """
                    UPDATE tasks SET state = ?, continuation_state = ?
                    WHERE task_id = ? AND revision = ? AND state = ?
                    """,
                    (
                        TaskState.INTERRUPTED.value, task.state.value,
                        record.task_id, record.revision, task.state.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("intervention task changed concurrently")
            elif task.state is TaskState.AWAITING_USER_INPUT:
                cursor = self._connection.execute(
                    """
                    UPDATE tasks SET state = ?
                    WHERE task_id = ? AND revision = ? AND state = ?
                    """,
                    (
                        TaskState.INTERRUPTED.value,
                        record.task_id, record.revision, TaskState.AWAITING_USER_INPUT.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("intervention task changed concurrently")
            elif task.state is not TaskState.INTERRUPTED:
                raise RuntimeError("intervention task changed concurrently")
            cursor = self._connection.execute(
                """
                UPDATE interventions SET status = ?
                WHERE intervention_id = ? AND resume_generation = ?
                  AND status IN ('pending_stop', 'ready', 'resuming')
                """,
                (
                    InterventionStatus.CANCELED_BY_STOP.value,
                    intervention_id, expected_resume_generation,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("intervention cancellation changed concurrently")
        return self._intervention_required(intervention_id)

    def mark_resume_outcome_unknown(
        self,
        intervention_id: str,
        *,
        resume_attempt_id: str,
        resume_run_id: str,
    ) -> InterventionRecord:
        intervention_id = _prepared_identifier(intervention_id, "intervention_id")
        resume_attempt_id = _prepared_identifier(resume_attempt_id, "resume_attempt_id")
        resume_run_id = _prepared_identifier(resume_run_id, "resume_run_id")
        with self._immediate_transaction():
            record = self._intervention_required(intervention_id)
            if (
                record.resume_attempt_id != resume_attempt_id
                or record.resume_run_id != resume_run_id
            ):
                raise RuntimeError("intervention resume owner changed")
            if record.status is InterventionStatus.RESUME_OUTCOME_UNKNOWN:
                return record
            if record.status is not InterventionStatus.RESUMING:
                raise RuntimeError("intervention outcome is not pending")
            cursor = self._connection.execute(
                """
                UPDATE interventions SET status = ?
                WHERE intervention_id = ? AND resume_attempt_id = ? AND resume_run_id = ?
                  AND status = ?
                """,
                (
                    InterventionStatus.RESUME_OUTCOME_UNKNOWN.value,
                    intervention_id, resume_attempt_id, resume_run_id,
                    InterventionStatus.RESUMING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("intervention outcome changed concurrently")
        return self._intervention_required(intervention_id)

    def authorize_retry_after_unknown(
        self,
        intervention_id: str,
        *,
        expected_resume_generation: int,
        acknowledgment_id: str,
    ) -> InterventionRecord:
        intervention_id = _prepared_identifier(intervention_id, "intervention_id")
        expected_resume_generation = _require_integer(
            expected_resume_generation, "expected_resume_generation",
        )
        acknowledgment_id = _prepared_identifier(acknowledgment_id, "acknowledgment_id")
        with self._immediate_transaction():
            record = self._intervention_required(intervention_id)
            row = self._connection.execute(
                "SELECT acknowledgment_id FROM interventions WHERE intervention_id = ?",
                (intervention_id,),
            ).fetchone()
            stored_acknowledgment = None if row is None else row["acknowledgment_id"]
            if (
                record.directed_binding is not None
                and not self._intervention_is_authenticated(record, stored_acknowledgment)
            ):
                raise RuntimeError("intervention binding is not authenticated")
            if (
                record.status is InterventionStatus.READY
                and record.resume_generation == expected_resume_generation + 1
                and stored_acknowledgment == acknowledgment_id
            ):
                return record
            if record.resume_generation != expected_resume_generation:
                raise RuntimeError("intervention resume generation changed")
            if record.status is not InterventionStatus.RESUME_OUTCOME_UNKNOWN:
                raise RuntimeError("intervention outcome is not unknown")
            task_cursor = self._connection.execute(
                """
                UPDATE tasks
                SET continuation_generation = continuation_generation + 1
                WHERE task_id = ? AND revision = ? AND session_id = ?
                  AND state = ? AND continuation_generation = ?
                """,
                (
                    record.task_id,
                    record.revision,
                    record.session_id,
                    TaskState.INTERRUPTED.value,
                    expected_resume_generation,
                ),
            )
            if task_cursor.rowcount != 1:
                raise RuntimeError("intervention task generation changed")
            binding = record.directed_binding
            if binding is not None and binding.stage == "active_question":
                question_cursor = self._connection.execute(
                    """
                    UPDATE questions
                    SET continuation_generation = continuation_generation + 1
                    WHERE question_id = ? AND session_id = ? AND task_id = ?
                      AND revision = ? AND answer_text IS NULL
                      AND continuation_generation = ?
                    """,
                    (
                        binding.question_id,
                        record.session_id,
                        record.task_id,
                        record.revision,
                        expected_resume_generation,
                    ),
                )
                if question_cursor.rowcount != 1:
                    raise RuntimeError("intervention directed question changed")
                self._advance_intervention_reservation_generation(
                    record=record,
                    question_id=binding.question_id,
                    exchange_id=binding.exchange_id,
                    request_key=binding.exchange_request_key,
                    ordinal=binding.exchange_ordinal,
                    expected_generation=expected_resume_generation,
                )
                if binding.parent_question_id is not None:
                    parent_cursor = self._connection.execute(
                        """
                        UPDATE questions
                        SET continuation_generation = continuation_generation + 1
                        WHERE question_id = ? AND session_id = ? AND task_id = ?
                          AND revision = ? AND answer_text IS NULL
                          AND continuation_generation = ?
                          AND continuation_pause_id = ?
                        """,
                        (
                            binding.parent_question_id,
                            record.session_id,
                            record.task_id,
                            record.revision,
                            expected_resume_generation,
                            binding.parent_continuation_pause_id,
                        ),
                    )
                    if parent_cursor.rowcount != 1:
                        raise RuntimeError("intervention directed parent question changed")
                    self._advance_intervention_reservation_generation(
                        record=record,
                        question_id=binding.parent_question_id,
                        exchange_id=binding.parent_exchange_id,
                        request_key=binding.parent_exchange_request_key,
                        ordinal=binding.parent_exchange_ordinal,
                        expected_generation=expected_resume_generation,
                    )
            cursor = self._connection.execute(
                """
                UPDATE interventions
                SET status = ?, resume_generation = resume_generation + 1,
                    resume_attempt_id = NULL, resume_run_id = NULL, acknowledgment_id = ?
                WHERE intervention_id = ? AND resume_generation = ?
                  AND status = ? AND acknowledgment_id IS NULL
                """,
                (
                    InterventionStatus.READY.value, acknowledgment_id, intervention_id,
                    expected_resume_generation, InterventionStatus.RESUME_OUTCOME_UNKNOWN.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("intervention retry authorization changed concurrently")
        return self._intervention_required(intervention_id)

    def _advance_intervention_reservation_generation(
        self,
        *,
        record: InterventionRecord,
        question_id: str,
        exchange_id: str | None,
        request_key: str | None,
        ordinal: int | None,
        expected_generation: int,
    ) -> None:
        present = (exchange_id is not None, request_key is not None, ordinal is not None)
        if not any(present):
            return
        if not all(present):
            raise RuntimeError("intervention directed reservation identity is incomplete")
        cursor = self._connection.execute(
            """
            UPDATE exchange_reservations
            SET continuation_generation = continuation_generation + 1
            WHERE exchange_id = ? AND session_id = ? AND task_id = ?
              AND revision = ? AND question_id = ? AND request_key = ?
              AND ordinal = ? AND continuation_generation = ?
            """,
            (
                exchange_id,
                record.session_id,
                record.task_id,
                record.revision,
                question_id,
                request_key,
                ordinal,
                expected_generation,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("intervention directed reservation changed")

    def _intervention_required(self, intervention_id: str) -> InterventionRecord:
        record = self.intervention(intervention_id)
        if record is None:
            raise RuntimeError("intervention not found")
        return record

    def _make_intervention_directed_binding(
        self,
        *,
        kind: str,
        stage: str = "active_question",
        question: QuestionRecord,
        continuation_pause_id: str,
        continuation_state: TaskState,
        source_run: AgentRunRecord,
        next_attempt_id: str | None = None,
        next_run_id: str | None = None,
        next_provider_id: str | None = None,
        next_task_state: TaskState | None = None,
        next_continuation_state: TaskState | None = None,
    ) -> _InterventionDirectedBinding:
        if source_run.cli_session_id is None:
            raise RuntimeError("intervention directed provider identity is missing")
        try:
            source_agent = ConversationTarget(source_run.agent)
        except ValueError as error:
            raise RuntimeError("intervention directed source agent is invalid") from error
        exchange_id, request_key, ordinal = self._intervention_reservation_identity(
            question,
        )
        parent_exchange_id: str | None = None
        parent_request_key: str | None = None
        parent_ordinal: int | None = None
        if question.parent_question_id is not None:
            parent = self.question(question.parent_question_id)
            if parent is None:
                raise RuntimeError("intervention directed parent question is missing")
            parent_exchange_id, parent_request_key, parent_ordinal = (
                self._intervention_reservation_identity(parent)
            )
        return _InterventionDirectedBinding(
            kind=kind,
            stage=stage,
            question_id=question.question_id,
            continuation_pause_id=continuation_pause_id,
            continuation_state=continuation_state,
            question_generation=question.continuation_generation,
            source_run_id=source_run.run_id,
            source_agent=source_agent,
            source_provider_id=source_run.cli_session_id,
            asked_by=question.asked_by,
            addressed_to=question.addressed_to,
            routed_to=question.routed_to,
            nested_parent_kind=question.nested_parent_kind,
            parent_question_id=question.parent_question_id,
            parent_continuation_pause_id=question.parent_continuation_pause_id,
            exchange_id=exchange_id,
            exchange_request_key=request_key,
            exchange_ordinal=ordinal,
            parent_exchange_id=parent_exchange_id,
            parent_exchange_request_key=parent_request_key,
            parent_exchange_ordinal=parent_ordinal,
            next_attempt_id=next_attempt_id,
            next_run_id=next_run_id,
            next_provider_id=next_provider_id,
            next_task_state=next_task_state,
            next_continuation_state=next_continuation_state,
        )

    def _intervention_reservation_identity(
        self, question: QuestionRecord,
    ) -> tuple[str | None, str | None, int | None]:
        if question.exchange_id is None:
            return None, None, None
        row = self._connection.execute(
            """
            SELECT * FROM exchange_reservations
            WHERE exchange_id = ? AND session_id = ? AND task_id = ?
              AND revision = ? AND question_id = ? AND continuation_generation = ?
            """,
            (
                question.exchange_id,
                question.session_id,
                question.task_id,
                question.revision,
                question.question_id,
                question.continuation_generation,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("intervention directed reservation is missing")
        try:
            request_key = _prepared_identifier(row["request_key"], "request_key")
            ordinal = _require_integer(row["ordinal"], "ordinal")
        except ValueError as error:
            raise RuntimeError("intervention directed reservation is invalid") from error
        if ordinal < 1:
            raise RuntimeError("intervention directed reservation is invalid")
        return question.exchange_id, request_key, ordinal

    def _bind_nested_intervention_resume_in_transaction(
        self, record: InterventionRecord, task: TaskRecord,
    ) -> InterventionRecord:
        """Snapshot one exact active nested provider boundary before interrupting it."""
        if record.directed_binding is not None:
            return record
        if (
            record.status is not InterventionStatus.RESUMING
            or record.routed_to is not ConversationTarget.FABLE
            or record.continuation_state not in _SOL_TASK_STATES
            or record.resume_attempt_id is None
            or record.resume_run_id is None
            or task.state is not TaskState.AWAITING_USER_INPUT
            or task.continuation_state is not TaskState.FABLE_CLARIFYING
            or task.continuation_generation != record.resume_generation
        ):
            return record
        row = self._connection.execute(
            """
            SELECT * FROM questions
            WHERE session_id = ? AND task_id = ? AND revision = ?
              AND answer_text IS NULL AND nested_parent_kind IS NOT NULL
            """,
            (record.session_id, record.task_id, record.revision),
        ).fetchone()
        active = self.active_run_for_task(record.task_id, record.revision)
        if row is None or active is None:
            return record
        question = self._question_from_row(row)
        _, continuation, pending, pause = self._question_exact(
            session_id=record.session_id,
            task_id=record.task_id,
            revision=record.revision,
            expected_generation=record.resume_generation,
            question_id=question.question_id,
        )
        task_pause = self._directed_pause_id(
            session_id=record.session_id,
            task_id=record.task_id,
            revision=record.revision,
            expected_generation=record.resume_generation,
        )
        if (
            continuation is not TaskState.FABLE_CLARIFYING
            or pending != task.pending
            or pause != task_pause
            or question.routed_to.value != active.agent
        ):
            raise RuntimeError("nested intervention directed binding changed")
        if active.agent == ConversationTarget.FABLE.value:
            self._require_exact_intervention_provider_identity(
                task.fable_session_id, active.cli_session_id,
            )
        elif active.agent == ConversationTarget.SOL.value:
            self._require_exact_intervention_provider_identity(
                task.sol_thread_id, active.cli_session_id,
            )
        else:
            raise RuntimeError("nested intervention directed source agent changed")
        binding = self._make_intervention_directed_binding(
            kind="nested_resume",
            question=question,
            continuation_pause_id=pause,
            continuation_state=continuation,
            source_run=active,
        )
        cursor = self._connection.execute(
            """
            UPDATE interventions SET directed_binding_json = ?
            WHERE intervention_id = ? AND status = ?
              AND resume_generation = ? AND resume_attempt_id = ? AND resume_run_id = ?
              AND directed_binding_json IS NULL
            """,
            (
                _encode_intervention_directed_binding(binding),
                record.intervention_id,
                InterventionStatus.RESUMING.value,
                record.resume_generation,
                record.resume_attempt_id,
                record.resume_run_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("nested intervention directed binding changed concurrently")
        return self._intervention_required(record.intervention_id)

    def intervention_resume_question(self, intervention_id: str) -> QuestionRecord | None:
        """Return only the exact persisted question boundary owned by this resume."""
        record = self.authenticated_intervention(intervention_id)
        if record is None or record.directed_binding is None:
            return None
        question = self.question(record.directed_binding.question_id)
        if question is None:
            raise RuntimeError("intervention directed question changed")
        return question

    def _intervention_is_directed_answer(self, record: InterventionRecord) -> bool:
        """Use the immutable persisted discriminator, never mutable phase mismatch."""
        return record.directed_binding is not None

    @staticmethod
    def _require_intervention_source_identity(
        *,
        task: TaskRecord,
        source_state: TaskState,
        source_run: AgentRunRecord,
    ) -> None:
        expected_agent = _INTERVENTION_SOURCE_AGENTS.get(source_state)
        if expected_agent is None or source_run.agent != expected_agent:
            raise RuntimeError("intervention source agent does not match its active phase")
        if expected_agent == "fable":
            fable_session_id = task.fable_session_id
            if source_state is TaskState.FABLE_PLANNING and fable_session_id is None:
                if source_run.cli_session_id is None:
                    return
                raise RuntimeError("intervention Fable provider identity changed")
            SQLiteStore._require_exact_intervention_provider_identity(
                fable_session_id, source_run.cli_session_id,
            )
            return
        SQLiteStore._require_exact_intervention_provider_identity(
            task.sol_thread_id, source_run.cli_session_id,
        )

    @staticmethod
    def _require_exact_intervention_provider_identity(
        stored_provider_id: object,
        run_provider_id: object,
    ) -> None:
        try:
            expected = _prepared_identifier(stored_provider_id, "intervention provider identity")
            actual = _prepared_identifier(run_provider_id, "intervention provider identity")
        except ValueError as error:
            raise RuntimeError("intervention provider identity changed") from error
        if actual != expected:
            raise RuntimeError("intervention provider identity changed")

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
            self._validate_nested_question_rows_in_transaction()
            for row in self._connection.execute(
                "SELECT * FROM interventions WHERE status = 'resuming' ORDER BY intervention_id"
            ):
                record = self._intervention_from_row(row)
                task = self.get_task(record.task_id, record.revision)
                if task.state is TaskState.INTERRUPTED:
                    continue
                record = self._bind_nested_intervention_resume_in_transaction(record, task)
                binding = record.directed_binding
                continuation = (
                    (
                        binding.next_continuation_state or binding.next_task_state
                        if binding is not None and binding.stage == "next_fable"
                        else binding.continuation_state if binding is not None else None
                    )
                    or TaskState.FABLE_CLARIFYING
                    if (
                        record.routed_to is ConversationTarget.FABLE
                        and record.continuation_state in _SOL_TASK_STATES
                    )
                    else record.continuation_state
                )
                cursor = self._connection.execute(
                    """
                    UPDATE tasks SET state = ?, continuation_state = ?
                    WHERE task_id = ? AND revision = ? AND state = ?
                    """,
                    (
                        TaskState.INTERRUPTED.value, continuation.value,
                        record.task_id, record.revision, task.state.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("resuming intervention task changed during recovery")
                tasks_interrupted += 1
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
            self._connection.execute(
                """
                UPDATE interventions
                SET status = ?
                WHERE status = ?
                  AND EXISTS (
                      SELECT 1 FROM tasks
                      WHERE tasks.task_id = interventions.task_id
                        AND tasks.revision = interventions.revision
                        AND tasks.session_id = interventions.session_id
                        AND tasks.state = ?
                        AND tasks.continuation_state = interventions.continuation_state
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM agent_runs
                      WHERE agent_runs.run_id = interventions.run_id
                        AND agent_runs.status = 'running'
                  )
                """,
                (
                    InterventionStatus.READY.value,
                    InterventionStatus.PENDING_STOP.value,
                    TaskState.INTERRUPTED.value,
                ),
            )
            self._validate_interventions_for_recovery_in_transaction()
            self._connection.execute(
                """
                UPDATE interventions SET status = ?
                WHERE status = ?
                """,
                (
                    InterventionStatus.RESUME_OUTCOME_UNKNOWN.value,
                    InterventionStatus.RESUMING.value,
                ),
            )
        return RecoverySummary(
            prepared_actions_recovered=0,
            tasks_interrupted=tasks_interrupted,
            agent_runs_interrupted=agent_runs_interrupted,
        )

    def _validate_interventions_for_recovery_in_transaction(self) -> None:
        if not self._connection.in_transaction:
            raise RuntimeError("intervention recovery validation requires a transaction")
        for row in self._connection.execute("SELECT * FROM interventions ORDER BY intervention_id"):
            record = self._intervention_from_row(row)
            if not self._intervention_is_authenticated(record, row["acknowledgment_id"]):
                raise RuntimeError("intervention recovery state is invalid")

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

    def audit_directed_fable_answer_checkpoints(self, canonical_repo_root: str) -> None:
        """Fail closed on unauthenticated directed-answer recovery state."""
        canonical_repo_root = _require_string(canonical_repo_root, "canonical_repo_root")
        expected_project_id = hashlib.sha256(
            os.fsencode(canonical_repo_root)
        ).hexdigest()[:32]
        self._connection.execute("BEGIN")
        try:
            reasons = self._directed_fable_answer_checkpoint_reasons(expected_project_id)
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.rollback()
        if reasons:
            summary = ", ".join(sorted(reasons)[:_MAX_LEGACY_AUDIT_REASONS])
            raise RuntimeError(f"directed Fable answer checkpoint audit failed: {summary}")

    def _directed_fable_answer_checkpoint_reasons(
        self, expected_project_id: str,
    ) -> set[str]:
        reasons: set[str] = set()
        for row in self._connection.execute(
            "SELECT rowid, * FROM directed_fable_answer_checkpoints ORDER BY rowid"
        ):
            if row["project_id"] != expected_project_id:
                reasons.add("fable_answer_checkpoint_ownership")
                continue
            prepared_row = self._connection.execute(
                """
                SELECT * FROM prepared_actions
                WHERE preparation_id = ? AND project_id = ? AND session_id = ?
                  AND task_id = ? AND revision = ?
                """,
                (row["preparation_id"], row["project_id"], row["session_id"],
                 row["task_id"], row["revision"]),
            ).fetchone()
            question_row = self._connection.execute(
                """
                SELECT * FROM questions
                WHERE question_id = ? AND session_id = ? AND task_id = ?
                  AND revision = ? AND continuation_generation = ?
                """,
                (row["question_id"], row["session_id"], row["task_id"],
                 row["revision"], row["continuation_generation"]),
            ).fetchone()
            try:
                clarification = FableClarification.from_dict(
                    _decode_mapping(str(row["clarification_json"]), "Fable answer checkpoint")
                )
            except (RuntimeError, TypeError, ValueError):
                reasons.add("fable_answer_checkpoint_integrity")
                continue
            if (
                prepared_row is None or question_row is None
                or row["status"] not in {"PENDING", "CONSUMED"}
                or question_row["answered_by"] != ConversationActor.FABLE.value
                or question_row["answer_text"] != clarification.answer
            ):
                reasons.add("fable_answer_checkpoint_integrity")
        return reasons

    def _legacy_project_ownership_reasons(
        self, canonical_repo_root: str,
    ) -> set[str]:
        """Collect generic integrity categories without exposing persisted values."""
        reasons: set[str] = set()

        def has_row(statement: str, parameters: tuple[object, ...] = ()) -> bool:
            return self._connection.execute(statement, parameters).fetchone() is not None

        if has_row("PRAGMA foreign_key_check"):
            reasons.add("foreign_key_integrity")
        try:
            self._validate_nested_question_rows_in_transaction()
        except RuntimeError:
            reasons.add("question_integrity")
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
        reasons.update(self._directed_fable_answer_checkpoint_reasons(expected_project_id))

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

        for row in self._connection.execute("SELECT * FROM interventions ORDER BY intervention_id"):
            try:
                record = self._intervention_from_row(row)
            except RuntimeError:
                reasons.add("intervention_integrity")
                continue
            if not self._intervention_is_authenticated(record, row["acknowledgment_id"]):
                reasons.add("intervention_integrity")

        return reasons

    def _intervention_directed_binding_is_authenticated(
        self,
        *,
        record: InterventionRecord,
        task: TaskRecord,
        binding: _InterventionDirectedBinding,
        acknowledgment_id: object,
    ) -> bool:
        if binding.stage == "next_fable":
            return self._next_fable_intervention_stage_is_authenticated(
                record=record,
                task=task,
                binding=binding,
                acknowledgment_id=acknowledgment_id,
            )
        question_row = self._connection.execute(
            "SELECT * FROM questions WHERE question_id = ?", (binding.question_id,)
        ).fetchone()
        source_row = self._connection.execute(
            "SELECT * FROM agent_runs WHERE run_id = ?", (binding.source_run_id,)
        ).fetchone()
        if question_row is None or source_row is None:
            return False
        try:
            question = self._question_from_row(question_row)
            source = self._agent_run_from_row(source_row)
            expected_generation = (
                binding.question_generation
                if acknowledgment_id is None
                else record.resume_generation
            )
            _, continuation, pending, pause = self._question_exact(
                session_id=record.session_id,
                task_id=record.task_id,
                revision=record.revision,
                expected_generation=expected_generation,
                question_id=binding.question_id,
            )
        except (RuntimeError, ValueError):
            return False
        if (
            question.continuation_generation != expected_generation
            or question.asked_by is not binding.asked_by
            or question.addressed_to is not binding.addressed_to
            or question.routed_to is not binding.routed_to
            or question.nested_parent_kind != binding.nested_parent_kind
            or question.parent_question_id != binding.parent_question_id
            or question.parent_continuation_pause_id != binding.parent_continuation_pause_id
            or continuation is not binding.continuation_state
            or pause != binding.continuation_pause_id
            or source.run_id != binding.source_run_id
            or source.task_id != record.task_id
            or source.revision != record.revision
            or source.agent != binding.source_agent.value
            or source.cli_session_id != binding.source_provider_id
        ):
            return False
        expected_provider = (
            record.fable_session_id
            if binding.source_agent is ConversationTarget.FABLE
            else record.sol_thread_id
        )
        if expected_provider != binding.source_provider_id:
            return False
        if not self._intervention_reservation_is_authenticated(
            record=record,
            question=question,
            exchange_id=binding.exchange_id,
            request_key=binding.exchange_request_key,
            ordinal=binding.exchange_ordinal,
            expected_generation=expected_generation,
        ):
            return False
        if binding.kind == "initial":
            if (
                binding.source_run_id != record.run_id
                or binding.continuation_state is not record.continuation_state
            ):
                return False
        elif (
            binding.kind != "nested_resume"
            or record.routed_to is not ConversationTarget.FABLE
            or record.continuation_state not in _SOL_TASK_STATES
            or binding.continuation_state is not TaskState.FABLE_CLARIFYING
            or binding.source_run_id == record.resume_run_id
        ):
            return False
        if binding.parent_question_id is not None:
            parent = self.question(binding.parent_question_id)
            if (
                parent is None
                or parent.continuation_generation != expected_generation
                or self._question_pause_id(parent.question_id)
                != binding.parent_continuation_pause_id
                or not self._intervention_reservation_is_authenticated(
                    record=record,
                    question=parent,
                    exchange_id=binding.parent_exchange_id,
                    request_key=binding.parent_exchange_request_key,
                    ordinal=binding.parent_exchange_ordinal,
                    expected_generation=expected_generation,
                )
            ):
                return False
        if (
            task.state in {TaskState.INTERRUPTED, TaskState.AWAITING_USER_INPUT}
            and question.answer_text is None
            and (
                task.continuation_state is not binding.continuation_state
                or task.pending != pending
                or self._directed_pause_id(
                    session_id=record.session_id,
                    task_id=record.task_id,
                    revision=record.revision,
                    expected_generation=expected_generation,
                ) != binding.continuation_pause_id
            )
        ):
            return False
        return True

    def _next_fable_intervention_stage_is_authenticated(
        self,
        *,
        record: InterventionRecord,
        task: TaskRecord,
        binding: _InterventionDirectedBinding,
        acknowledgment_id: object,
    ) -> bool:
        """Authenticate the consumed Sol child and its one exact Fable successor."""
        if (
            binding.kind != "nested_resume"
            or binding.source_agent is not ConversationTarget.SOL
            or record.routed_to is not ConversationTarget.FABLE
            or record.continuation_state not in _SOL_TASK_STATES
            or binding.continuation_state is not TaskState.FABLE_CLARIFYING
            or binding.source_run_id == record.resume_run_id
            or binding.next_provider_id != record.fable_session_id
            or binding.next_provider_id != task.fable_session_id
            or binding.next_run_id is None
            or binding.next_attempt_id is None
            or binding.next_task_state is None
        ):
            return False
        if (
            record.status in {
                InterventionStatus.RESUMING,
                InterventionStatus.RESUME_OUTCOME_UNKNOWN,
                InterventionStatus.RESUMED,
            }
            or (
                record.status is InterventionStatus.CANCELED_BY_STOP
                and record.resume_attempt_id is not None
            )
        ) and binding.next_attempt_id != record.resume_attempt_id:
            return False
        question_row = self._connection.execute(
            "SELECT * FROM questions WHERE question_id = ?", (binding.question_id,)
        ).fetchone()
        source_row = self._connection.execute(
            "SELECT * FROM agent_runs WHERE run_id = ?", (binding.source_run_id,)
        ).fetchone()
        if question_row is None or source_row is None:
            return False
        try:
            question = self._question_from_row(question_row)
            source = self._agent_run_from_row(source_row)
            _, continuation, _, pause = self._question_exact(
                session_id=record.session_id,
                task_id=record.task_id,
                revision=record.revision,
                expected_generation=binding.question_generation,
                question_id=binding.question_id,
            )
        except (RuntimeError, ValueError):
            return False
        if (
            question.answer_text is None
            or question.answered_by is not ConversationActor.SOL
            or question.continuation_generation != binding.question_generation
            or question.asked_by is not binding.asked_by
            or question.addressed_to is not binding.addressed_to
            or question.routed_to is not binding.routed_to
            or question.nested_parent_kind != binding.nested_parent_kind
            or question.parent_question_id != binding.parent_question_id
            or question.parent_continuation_pause_id != binding.parent_continuation_pause_id
            or continuation is not binding.continuation_state
            or pause != binding.continuation_pause_id
            or source.task_id != record.task_id
            or source.revision != record.revision
            or source.agent != ConversationTarget.SOL.value
            or source.cli_session_id != binding.source_provider_id
            or binding.source_provider_id != record.sol_thread_id
        ):
            return False
        if not self._intervention_reservation_is_authenticated(
            record=record,
            question=question,
            exchange_id=binding.exchange_id,
            request_key=binding.exchange_request_key,
            ordinal=binding.exchange_ordinal,
            expected_generation=binding.question_generation,
        ):
            return False
        if binding.parent_question_id is not None:
            try:
                parent, _, _, parent_pause = self._question_exact(
                    session_id=record.session_id,
                    task_id=record.task_id,
                    revision=record.revision,
                    expected_generation=binding.question_generation,
                    question_id=binding.parent_question_id,
                )
            except (RuntimeError, ValueError):
                return False
            if (
                parent.answer_text is not None
                or parent.continuation_generation != binding.question_generation
                or parent_pause != binding.parent_continuation_pause_id
                or not self._intervention_reservation_is_authenticated(
                    record=record,
                    question=parent,
                    exchange_id=binding.parent_exchange_id,
                    request_key=binding.parent_exchange_request_key,
                    ordinal=binding.parent_exchange_ordinal,
                    expected_generation=binding.question_generation,
                )
            ):
                return False
        next_row = self._connection.execute(
            "SELECT * FROM agent_runs WHERE run_id = ?", (binding.next_run_id,)
        ).fetchone()
        if next_row is None:
            return False
        try:
            successor = self._agent_run_from_row(next_row)
        except RuntimeError:
            return False
        if (
            successor.task_id != record.task_id
            or successor.revision != record.revision
            or successor.agent != ConversationTarget.FABLE.value
            or successor.cli_session_id != binding.next_provider_id
        ):
            return False
        resumed_continuation = (
            binding.next_continuation_state or binding.next_task_state
        )
        if record.status is InterventionStatus.RESUMING:
            if task.state is TaskState.INTERRUPTED:
                return task.continuation_state is resumed_continuation
            return (
                task.state is binding.next_task_state
                and task.continuation_state is binding.next_continuation_state
            )
        if record.status in {
            InterventionStatus.READY,
            InterventionStatus.RESUME_OUTCOME_UNKNOWN,
            InterventionStatus.CANCELED_BY_STOP,
        }:
            return (
                task.state is TaskState.INTERRUPTED
                and task.continuation_state is resumed_continuation
            )
        return True

    def _intervention_reservation_is_authenticated(
        self,
        *,
        record: InterventionRecord,
        question: QuestionRecord,
        exchange_id: str | None,
        request_key: str | None,
        ordinal: int | None,
        expected_generation: int,
    ) -> bool:
        present = (exchange_id is not None, request_key is not None, ordinal is not None)
        if not any(present):
            return question.exchange_id is None
        if not all(present) or question.exchange_id != exchange_id:
            return False
        row = self._connection.execute(
            """
            SELECT 1 FROM exchange_reservations
            WHERE exchange_id = ? AND session_id = ? AND task_id = ?
              AND revision = ? AND question_id = ? AND request_key = ?
              AND ordinal = ? AND continuation_generation = ?
            """,
            (
                exchange_id,
                record.session_id,
                record.task_id,
                record.revision,
                question.question_id,
                request_key,
                ordinal,
                expected_generation,
            ),
        ).fetchone()
        return row is not None

    def _intervention_is_authenticated(
        self, record: InterventionRecord, acknowledgment_id: object,
    ) -> bool:
        """Validate durable intervention authority without repairing persisted state."""
        task_row = self._connection.execute(
            """
            SELECT * FROM tasks
            WHERE task_id = ? AND revision = ? AND session_id = ?
            """,
            (record.task_id, record.revision, record.session_id),
        ).fetchone()
        run_row = self._connection.execute(
            "SELECT * FROM agent_runs WHERE run_id = ?", (record.run_id,)
        ).fetchone()
        if task_row is None or run_row is None:
            return False
        try:
            task = self._task_from_row(task_row)
            source_run = self._agent_run_from_row(run_row)
        except RuntimeError:
            return False
        if (
            source_run.task_id != record.task_id
            or source_run.revision != record.revision
            or task.fable_session_id != record.fable_session_id
            or task.sol_thread_id != record.sol_thread_id
            or record.status not in {
                InterventionStatus.PENDING_STOP,
                InterventionStatus.READY,
                InterventionStatus.RESUMING,
                InterventionStatus.RESUMED,
                InterventionStatus.RESUME_OUTCOME_UNKNOWN,
                InterventionStatus.CANCELED_BY_STOP,
                InterventionStatus.FAILED,
            }
        ):
            return False
        has_resume_owner = (
            record.resume_attempt_id is not None and record.resume_run_id is not None
        )
        binding = record.directed_binding
        if binding is None:
            if task.continuation_generation != record.resume_generation:
                return False
            try:
                self._require_intervention_source_identity(
                    task=task,
                    source_state=record.continuation_state,
                    source_run=source_run,
                )
            except RuntimeError:
                return False
        else:
            if not self._intervention_directed_binding_is_authenticated(
                record=record,
                task=task,
                binding=binding,
                acknowledgment_id=acknowledgment_id,
            ):
                return False
            if binding.kind == "nested_resume":
                try:
                    self._require_intervention_source_identity(
                        task=task,
                        source_state=record.continuation_state,
                        source_run=source_run,
                    )
                except RuntimeError:
                    return False
        resumed_continuation = (
            (
                binding.next_continuation_state or binding.next_task_state
                if binding is not None and binding.stage == "next_fable"
                else binding.continuation_state if binding is not None else None
            )
            or
            TaskState.FABLE_CLARIFYING
            if (
                record.routed_to is ConversationTarget.FABLE
                and record.continuation_state in _SOL_TASK_STATES
            )
            else record.continuation_state
        )
        if record.status in {
            InterventionStatus.PENDING_STOP,
            InterventionStatus.READY,
            InterventionStatus.CANCELED_BY_STOP,
        }:
            if task.state is not TaskState.INTERRUPTED:
                return False
            if (
                record.status is InterventionStatus.READY
                and acknowledgment_id is not None
            ):
                if task.continuation_state not in {
                    record.continuation_state,
                    resumed_continuation,
                }:
                    return False
            elif record.status is InterventionStatus.CANCELED_BY_STOP and (
                has_resume_owner
                or binding is not None and binding.stage == "next_fable"
            ):
                if task.continuation_state is not resumed_continuation:
                    return False
            elif task.continuation_state is not record.continuation_state:
                return False
        if record.status is InterventionStatus.RESUME_OUTCOME_UNKNOWN and (
            task.state is not TaskState.INTERRUPTED
            or task.continuation_state is not resumed_continuation
        ):
            return False
        if record.status is InterventionStatus.RESUMING and task.state not in {
            TaskState.INTERRUPTED,
            TaskState.AWAITING_USER_INPUT,
            *_ACTIVE_TASK_STATES,
        }:
            return False
        if not self._intervention_conversation_event_exists(record):
            return False
        if record.status in {
            InterventionStatus.RESUMING,
            InterventionStatus.RESUMED,
            InterventionStatus.RESUME_OUTCOME_UNKNOWN,
        } and not has_resume_owner:
            return False
        if record.status in {
            InterventionStatus.PENDING_STOP,
            InterventionStatus.READY,
        } and has_resume_owner:
            return False
        if acknowledgment_id is not None:
            try:
                _prepared_identifier(acknowledgment_id, "acknowledgment_id")
            except ValueError:
                return False
            if (
                record.status is InterventionStatus.PENDING_STOP
                or record.resume_generation != task.continuation_generation
            ):
                return False
        elif record.resume_generation != task.continuation_generation:
            return False
        return True

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
        elif record.action in {
            "continuation_message", "question_answer", "exchange_grant",
        }:
            if (
                record.source_state is not TaskState.AWAITING_USER_INPUT
                or record.continuation_state is None
                or record.active_state is not record.continuation_state
                or record.previous_preparation_id is not None
                or not self._conversation_prepared_context_matches_task(
                    task, record.pending_context,
                )
            ):
                return False
            if record.action == "continuation_message":
                payload = record.payload
                if (
                    not isinstance(payload, ContinuationMessagePayload)
                    or payload.continuation != record.pending_context
                    or payload.continuation_generation != task.continuation_generation
                    or self._target_for_prepared_continuation(payload.continuation)
                    is not payload.routed_to
                    or not self._conversation_event_exists(
                        session_id=record.session_id,
                        task_id=record.task_id,
                        sender=ConversationActor.USER,
                        addressed_to=payload.addressed_to,
                        routed_to=payload.routed_to,
                        message_type=ConversationMessageType.STATEMENT,
                        text=payload.text,
                        revision=record.revision,
                        continuation_generation=payload.continuation_generation,
                    )
                ):
                    return False
            elif record.action == "question_answer":
                payload = record.payload
                question = None
                if isinstance(payload, QuestionAnswerPayload):
                    question = self.question(payload.question_id)
                if (
                    not isinstance(payload, QuestionAnswerPayload)
                    or payload.continuation != record.pending_context
                    or task.continuation_generation != payload.continuation_generation + 1
                    or question is None
                    or question.session_id != record.session_id
                    or question.task_id != record.task_id
                    or question.revision != record.revision
                    or question.continuation_generation != payload.continuation_generation
                    or question.routed_to is not ConversationTarget.USER
                    or question.answer_text != payload.answer
                    or question.answered_by is not ConversationActor.USER
                    or not self._conversation_event_exists(
                        session_id=record.session_id,
                        task_id=record.task_id,
                        sender=ConversationActor.USER,
                        addressed_to=self._target_for_question_asker(question.asked_by),
                        routed_to=self._target_for_question_asker(question.asked_by),
                        message_type=ConversationMessageType.ANSWER,
                        text=payload.answer,
                        revision=record.revision,
                        continuation_generation=payload.continuation_generation,
                        reply_to_question_id=payload.question_id,
                    )
                ):
                    return False
            else:
                payload = record.payload
                grant = None
                if isinstance(payload, ExchangeGrantPayload):
                    grant = self._connection.execute(
                        """
                        SELECT grant_size FROM exchange_grants
                        WHERE session_id = ? AND task_id = ? AND revision = ?
                          AND request_id = ? AND continuation_generation = ?
                        """,
                        (
                            record.session_id,
                            record.task_id,
                            record.revision,
                            payload.request_id,
                            payload.continuation_generation,
                        ),
                    ).fetchone()
                if (
                    not isinstance(payload, ExchangeGrantPayload)
                    or payload.continuation != record.pending_context
                    or payload.continuation_generation != task.continuation_generation
                    or (
                        payload.parent_mode == "top_level"
                        and (payload.outer_question_id is not None or isinstance(payload.continuation, ClarificationContext))
                    )
                    or (
                        payload.parent_mode == "clarification"
                        and not isinstance(payload.continuation, ClarificationContext)
                    )
                    or (
                        payload.parent_mode == "question" and (
                            payload.outer_question_id is None
                            or (parent := self.question(payload.outer_question_id)) is None
                            or parent.task_id != record.task_id
                            or parent.revision != record.revision
                            or parent.nested_parent_kind is not None
                            or parent.asked_by is not ConversationActor.SOL
                            or parent.routed_to is not ConversationTarget.FABLE
                        )
                    )
                    or (payload.parent_mode == "clarification" and payload.outer_question_id is not None)
                    or grant is None
                    or int(grant["grant_size"]) != EXCHANGE_GRANT_SIZE
                    or not self._conversation_event_exists(
                        session_id=record.session_id,
                        task_id=record.task_id,
                        sender=ConversationActor.USER,
                        addressed_to=ConversationTarget.TEAM,
                        routed_to=ConversationTarget.FABLE,
                        message_type=ConversationMessageType.APPROVAL,
                        text="Allow three more internal exchanges.",
                        revision=record.revision,
                    )
                ):
                    return False
        else:  # PreparedActionRecord normally prevents this; keep audit fail-closed.
            return False
        if record.action != "new_request":
            try:
                require_transition(record.source_state, record.active_state)
            except ValueError:
                return False
        return True

    @staticmethod
    def _conversation_prepared_context_matches_task(
        task: TaskRecord,
        context: PreparedContinuationContext,
    ) -> bool:
        while isinstance(context, AnswerContext):
            context = context.underlying_continuation
        if isinstance(context, SolResumeContext):
            return task.sol_thread_id == context.sol_thread_id
        if isinstance(context, ScopeApprovalContext):
            return (
                task.baseline_id == context.baseline_id
                and task.revision == context.approved_revision
                and (
                    context.underlying_continuation is None
                    or task.sol_thread_id == context.underlying_continuation.sol_thread_id
                )
            )
        if isinstance(context, ReviewContext):
            return task.fable_session_id == context.fable_session_id
        if isinstance(context, ClarificationContext):
            return task.fable_session_id == context.fable_session_id
        return False

    def _conversation_event_exists(
        self,
        *,
        session_id: str,
        task_id: str,
        sender: ConversationActor,
        addressed_to: ConversationTarget,
        routed_to: ConversationTarget,
        message_type: ConversationMessageType,
        text: str,
        revision: int,
        continuation_generation: int | None = None,
        reply_to_question_id: str | None = None,
    ) -> bool:
        for row in self._connection.execute(
            """
            SELECT payload_json FROM events
            WHERE session_id = ? AND task_id = ? AND kind = 'conversation'
            """,
            (session_id, task_id),
        ):
            try:
                event = ConversationEnvelope.from_dict(
                    _decode_mapping(row["payload_json"], "conversation event")
                )
            except (RuntimeError, ValueError):
                continue
            if (
                event.sender is sender
                and event.addressed_to is addressed_to
                and event.routed_to is routed_to
                and event.message_type is message_type
                and event.text == text
                and event.task_id == task_id
                and event.revision == revision
                and event.continuation_generation == continuation_generation
                and event.reply_to_question_id == reply_to_question_id
            ):
                return True
        return False

    def _intervention_conversation_event_exists(self, record: InterventionRecord) -> bool:
        """Match the one visible intervention without inventing revision-zero binding."""
        for row in self._connection.execute(
            """
            SELECT payload_json FROM events
            WHERE session_id = ? AND task_id = ? AND actor = ? AND kind = 'conversation'
            """,
            (record.session_id, record.task_id, ConversationActor.USER.value),
        ):
            try:
                event = ConversationEnvelope.from_dict(
                    _decode_mapping(row["payload_json"], "conversation event")
                )
            except (RuntimeError, ValueError):
                continue
            if (
                event.sender is not ConversationActor.USER
                or event.addressed_to is not record.addressed_to
                or event.routed_to is not record.routed_to
                or event.message_type is not ConversationMessageType.INTERVENTION
                or event.text != record.message
            ):
                continue
            if record.revision == 0:
                if (
                    event.task_id is None
                    and event.revision is None
                    and event.continuation_generation is None
                ):
                    return True
            elif (
                event.task_id == record.task_id
                and event.revision == record.revision
                and event.continuation_generation == record.source_generation
            ):
                return True
        return False

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

    def _question_from_row(self, row: sqlite3.Row) -> QuestionRecord:
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
                nested_parent_kind=row["nested_parent_kind"],
                parent_question_id=row["parent_question_id"],
                parent_continuation_pause_id=row["parent_continuation_pause_id"],
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

    @staticmethod
    def _intervention_from_row(row: sqlite3.Row) -> InterventionRecord:
        try:
            return InterventionRecord(
                intervention_id=row["intervention_id"],
                session_id=row["session_id"],
                task_id=row["task_id"],
                revision=int(row["revision"]),
                addressed_to=ConversationTarget(row["addressed_to"]),
                routed_to=ConversationTarget(row["routed_to"]),
                message=row["message"],
                run_id=row["run_id"],
                continuation_state=TaskState(row["continuation_state"]),
                source_generation=int(row["source_generation"]),
                resume_generation=int(row["resume_generation"]),
                fable_session_id=row["fable_session_id"],
                sol_thread_id=row["sol_thread_id"],
                resume_attempt_id=row["resume_attempt_id"],
                resume_run_id=row["resume_run_id"],
                status=InterventionStatus(row["status"]),
                created_at=row["created_at"],
                directed_binding=_decode_intervention_directed_binding(
                    row["directed_binding_json"]
                ),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError("persisted intervention is invalid") from error

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
