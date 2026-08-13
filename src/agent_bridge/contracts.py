"""Immutable, JSON-compatible contracts for local bridge handoffs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
import re
from types import MappingProxyType
from typing import Literal, TypeAlias


JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


def freeze_json(value: object) -> JsonValue:
    """Recursively copy JSON-compatible data into immutable containers."""
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            frozen[key] = freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("value must be JSON-compatible")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError("value must be JSON-compatible")


def _thaw_json(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _mapping(data: object, name: str) -> Mapping[str, object]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{name} must be an object")
    for key in data:
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
    return data


def _require_fields(data: Mapping[str, object], name: str, fields: tuple[str, ...]) -> None:
    expected = set(fields)
    actual = set(data)
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        raise ValueError(f"{name} missing required fields: {', '.join(sorted(missing))}")
    if unexpected:
        raise ValueError(f"{name} has unexpected fields: {', '.join(sorted(unexpected))}")


def _string(value: object, name: str, *, non_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if non_empty and not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


_MAX_CONVERSATION_TEXT_LENGTH = 16 * 1024
_SAFE_CONVERSATION_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def _bounded_conversation_text(value: object, name: str) -> str:
    text = _string(value, name, non_empty=True)
    if len(text) > _MAX_CONVERSATION_TEXT_LENGTH:
        raise ValueError(f"{name} is too long")
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise ValueError(f"{name} must not contain control characters")
    return text


def _conversation_identifier(value: object, name: str) -> str:
    identifier = _string(value, name, non_empty=True)
    if _SAFE_CONVERSATION_IDENTIFIER.fullmatch(identifier) is None:
        raise ValueError(f"{name} must be a safe identifier")
    return identifier


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"{name} must be an array of strings")
    strings = tuple(value)
    if not all(isinstance(item, str) for item in strings):
        raise ValueError(f"{name} must be an array of strings")
    return strings


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _confidence(value: object, name: str = "confidence") -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return confidence


_TASK_BRIEF_FIELDS = (
    "task_id", "revision", "title", "objective", "context", "constraints",
    "allowed_paths", "out_of_scope", "acceptance_criteria", "required_tests",
    "risks", "open_questions", "confidence", "confidence_rationale",
)


@dataclass(frozen=True)
class TaskBrief:
    task_id: str
    revision: int
    title: str
    objective: str
    context: tuple[str, ...]
    constraints: tuple[str, ...]
    allowed_paths: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    required_tests: tuple[str, ...]
    risks: tuple[str, ...]
    open_questions: tuple[str, ...]
    confidence: float
    confidence_rationale: str

    def __post_init__(self) -> None:
        for name in (
            "context", "constraints", "allowed_paths", "out_of_scope",
            "acceptance_criteria", "required_tests", "risks", "open_questions",
        ):
            object.__setattr__(self, name, _string_tuple(getattr(self, name), name))
        object.__setattr__(self, "task_id", _string(self.task_id, "task_id", non_empty=True))
        object.__setattr__(self, "revision", _integer(self.revision, "revision"))
        object.__setattr__(self, "title", _string(self.title, "title", non_empty=True))
        object.__setattr__(self, "objective", _string(self.objective, "objective", non_empty=True))
        object.__setattr__(
            self,
            "confidence_rationale",
            _string(self.confidence_rationale, "confidence_rationale", non_empty=True),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if self.revision < 1:
            raise ValueError("revision must be >= 1")
        if not self.allowed_paths:
            raise ValueError("allowed_paths must be non-empty")
        if not self.acceptance_criteria:
            raise ValueError("acceptance_criteria must be non-empty")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TaskBrief":
        payload = _mapping(data, "TaskBrief")
        _require_fields(payload, "TaskBrief", _TASK_BRIEF_FIELDS)
        return cls(**{field: payload[field] for field in _TASK_BRIEF_FIELDS})  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "revision": self.revision,
            "title": self.title,
            "objective": self.objective,
            "context": list(self.context),
            "constraints": list(self.constraints),
            "allowed_paths": list(self.allowed_paths),
            "out_of_scope": list(self.out_of_scope),
            "acceptance_criteria": list(self.acceptance_criteria),
            "required_tests": list(self.required_tests),
            "risks": list(self.risks),
            "open_questions": list(self.open_questions),
            "confidence": self.confidence,
            "confidence_rationale": self.confidence_rationale,
        }


@dataclass(frozen=True)
class CommandReport:
    command: str
    exit_code: int
    result: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", _string(self.command, "command"))
        object.__setattr__(self, "exit_code", _integer(self.exit_code, "exit_code"))
        object.__setattr__(self, "result", _string(self.result, "result"))

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CommandReport":
        payload = _mapping(data, "CommandReport")
        fields = ("command", "exit_code", "result")
        _require_fields(payload, "CommandReport", fields)
        return cls(**{field: payload[field] for field in fields})  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"command": self.command, "exit_code": self.exit_code, "result": self.result}


@dataclass(frozen=True)
class SolQuestion:
    ambiguity: str
    why_it_matters: str
    options: tuple[str, ...]
    recommendation: str
    can_continue_safely: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "ambiguity", _string(self.ambiguity, "ambiguity"))
        object.__setattr__(self, "why_it_matters", _string(self.why_it_matters, "why_it_matters"))
        object.__setattr__(self, "options", _string_tuple(self.options, "options"))
        object.__setattr__(self, "recommendation", _string(self.recommendation, "recommendation"))
        object.__setattr__(
            self,
            "can_continue_safely",
            _boolean(self.can_continue_safely, "can_continue_safely"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SolQuestion":
        payload = _mapping(data, "SolQuestion")
        fields = ("ambiguity", "why_it_matters", "options", "recommendation", "can_continue_safely")
        _require_fields(payload, "SolQuestion", fields)
        return cls(**{field: payload[field] for field in fields})  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "ambiguity": self.ambiguity,
            "why_it_matters": self.why_it_matters,
            "options": list(self.options),
            "recommendation": self.recommendation,
            "can_continue_safely": self.can_continue_safely,
        }


_SOL_OUTCOME_FIELDS = (
    "status", "summary", "changed_files", "commands_run", "known_failures",
    "remaining_risks", "architecture_docs", "question",
)
_SOL_OUTCOME_STATUSES = frozenset({"completed", "question", "blocked", "failed"})


@dataclass(frozen=True)
class SolOutcome:
    status: str
    summary: str
    changed_files: tuple[str, ...]
    commands_run: tuple[CommandReport, ...]
    known_failures: tuple[str, ...]
    remaining_risks: tuple[str, ...]
    architecture_docs: str
    question: SolQuestion | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _string(self.status, "status"))
        object.__setattr__(self, "summary", _string(self.summary, "summary"))
        object.__setattr__(self, "changed_files", _string_tuple(self.changed_files, "changed_files"))
        if not isinstance(self.commands_run, Sequence) or isinstance(self.commands_run, str):
            raise ValueError("commands_run must be an array")
        commands = tuple(self.commands_run)
        if not all(isinstance(command, CommandReport) for command in commands):
            raise ValueError("commands_run must contain CommandReport values")
        object.__setattr__(self, "commands_run", commands)
        object.__setattr__(self, "known_failures", _string_tuple(self.known_failures, "known_failures"))
        object.__setattr__(self, "remaining_risks", _string_tuple(self.remaining_risks, "remaining_risks"))
        object.__setattr__(self, "architecture_docs", _string(self.architecture_docs, "architecture_docs"))
        if self.status not in _SOL_OUTCOME_STATUSES:
            raise ValueError("status must be completed, question, blocked, or failed")
        if self.status == "question":
            if not isinstance(self.question, SolQuestion) or len(self.question.options) < 2:
                raise ValueError("question outcomes require a question with at least two options")
        elif self.question is not None:
            raise ValueError("non-question outcomes must have a null question")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SolOutcome":
        payload = _mapping(data, "SolOutcome")
        _require_fields(payload, "SolOutcome", _SOL_OUTCOME_FIELDS)
        question_data = payload["question"]
        question = None if question_data is None else SolQuestion.from_dict(_mapping(question_data, "question"))
        commands_data = payload["commands_run"]
        if not isinstance(commands_data, Sequence) or isinstance(commands_data, str):
            raise ValueError("commands_run must be an array")
        return cls(
            status=payload["status"],
            summary=payload["summary"],
            changed_files=payload["changed_files"],
            commands_run=tuple(CommandReport.from_dict(_mapping(item, "commands_run")) for item in commands_data),
            known_failures=payload["known_failures"],
            remaining_risks=payload["remaining_risks"],
            architecture_docs=payload["architecture_docs"],
            question=question,
        )  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "summary": self.summary,
            "changed_files": list(self.changed_files),
            "commands_run": [command.to_dict() for command in self.commands_run],
            "known_failures": list(self.known_failures),
            "remaining_risks": list(self.remaining_risks),
            "architecture_docs": self.architecture_docs,
            "question": None if self.question is None else self.question.to_dict(),
        }


_FABLE_CLARIFICATION_FIELDS = (
    "status", "answer", "reasoning", "confidence", "scope_changed", "revised_brief", "question_for_user",
)
_FABLE_CLARIFICATION_STATUSES = frozenset({"answered", "escalate_to_user"})


@dataclass(frozen=True)
class FableClarification:
    status: str
    answer: str | None
    reasoning: str
    confidence: float
    scope_changed: bool
    revised_brief: TaskBrief | None
    question_for_user: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _string(self.status, "status"))
        if self.answer is not None:
            object.__setattr__(self, "answer", _string(self.answer, "answer"))
        object.__setattr__(self, "reasoning", _string(self.reasoning, "reasoning"))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "scope_changed", _boolean(self.scope_changed, "scope_changed"))
        if self.revised_brief is not None and not isinstance(self.revised_brief, TaskBrief):
            raise ValueError("revised_brief must be a TaskBrief or null")
        if self.question_for_user is not None:
            object.__setattr__(self, "question_for_user", _string(self.question_for_user, "question_for_user"))
        if self.status not in _FABLE_CLARIFICATION_STATUSES:
            raise ValueError("status must be answered or escalate_to_user")
        if self.status == "answered" and (self.answer is None or not self.answer.strip()):
            raise ValueError("answered clarifications require a non-empty answer")
        if self.scope_changed and self.revised_brief is None:
            raise ValueError("scope_changed clarifications require a revised_brief")
        if self.status == "escalate_to_user" and (
            self.question_for_user is None or not self.question_for_user.strip()
        ):
            raise ValueError("escalate_to_user clarifications require a non-empty question_for_user")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FableClarification":
        payload = _mapping(data, "FableClarification")
        _require_fields(payload, "FableClarification", _FABLE_CLARIFICATION_FIELDS)
        revised_data = payload["revised_brief"]
        return cls(
            status=payload["status"],
            answer=payload["answer"],
            reasoning=payload["reasoning"],
            confidence=payload["confidence"],
            scope_changed=payload["scope_changed"],
            revised_brief=(
                None if revised_data is None else TaskBrief.from_dict(_mapping(revised_data, "revised_brief"))
            ),
            question_for_user=payload["question_for_user"],
        )  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "answer": self.answer,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "scope_changed": self.scope_changed,
            "revised_brief": None if self.revised_brief is None else self.revised_brief.to_dict(),
            "question_for_user": self.question_for_user,
        }


@dataclass(frozen=True)
class CriterionEvidence:
    criterion: str
    evidence: tuple[str, ...]
    satisfied: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "criterion", _string(self.criterion, "criterion"))
        object.__setattr__(self, "evidence", _string_tuple(self.evidence, "evidence"))
        object.__setattr__(self, "satisfied", _boolean(self.satisfied, "satisfied"))

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CriterionEvidence":
        payload = _mapping(data, "CriterionEvidence")
        fields = ("criterion", "evidence", "satisfied")
        _require_fields(payload, "CriterionEvidence", fields)
        return cls(**{field: payload[field] for field in fields})  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"criterion": self.criterion, "evidence": list(self.evidence), "satisfied": self.satisfied}


_REVIEW_VERDICT_FIELDS = (
    "status", "summary", "criteria", "test_assessment", "scope_violations",
    "remaining_risks", "corrections", "question_for_user",
)
_REVIEW_VERDICT_STATUSES = frozenset({"approved", "corrections_required", "escalate_to_user"})


@dataclass(frozen=True)
class ReviewVerdict:
    status: str
    summary: str
    criteria: tuple[CriterionEvidence, ...]
    test_assessment: str
    scope_violations: tuple[str, ...]
    remaining_risks: tuple[str, ...]
    corrections: tuple[str, ...]
    question_for_user: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _string(self.status, "status"))
        object.__setattr__(self, "summary", _string(self.summary, "summary"))
        if not isinstance(self.criteria, Sequence) or isinstance(self.criteria, str):
            raise ValueError("criteria must be an array")
        criteria = tuple(self.criteria)
        if not all(isinstance(criterion, CriterionEvidence) for criterion in criteria):
            raise ValueError("criteria must contain CriterionEvidence values")
        object.__setattr__(self, "criteria", criteria)
        object.__setattr__(self, "test_assessment", _string(self.test_assessment, "test_assessment"))
        object.__setattr__(self, "scope_violations", _string_tuple(self.scope_violations, "scope_violations"))
        object.__setattr__(self, "remaining_risks", _string_tuple(self.remaining_risks, "remaining_risks"))
        object.__setattr__(self, "corrections", _string_tuple(self.corrections, "corrections"))
        if self.question_for_user is not None:
            object.__setattr__(self, "question_for_user", _string(self.question_for_user, "question_for_user"))
        if self.status not in _REVIEW_VERDICT_STATUSES:
            raise ValueError("status must be approved, corrections_required, or escalate_to_user")
        if self.status == "corrections_required" and not self.corrections:
            raise ValueError("corrections_required verdicts require corrections")
        if self.status == "escalate_to_user" and (
            self.question_for_user is None or not self.question_for_user.strip()
        ):
            raise ValueError("escalate_to_user verdicts require a non-empty question_for_user")
        if self.status == "approved" and (self.scope_violations or not all(item.satisfied for item in self.criteria)):
            raise ValueError("approved verdicts require no scope violations and all criteria satisfied")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ReviewVerdict":
        payload = _mapping(data, "ReviewVerdict")
        _require_fields(payload, "ReviewVerdict", _REVIEW_VERDICT_FIELDS)
        criteria_data = payload["criteria"]
        if not isinstance(criteria_data, Sequence) or isinstance(criteria_data, str):
            raise ValueError("criteria must be an array")
        return cls(
            status=payload["status"],
            summary=payload["summary"],
            criteria=tuple(CriterionEvidence.from_dict(_mapping(item, "criteria")) for item in criteria_data),
            test_assessment=payload["test_assessment"],
            scope_violations=payload["scope_violations"],
            remaining_risks=payload["remaining_risks"],
            corrections=payload["corrections"],
            question_for_user=payload["question_for_user"],
        )  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "summary": self.summary,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "test_assessment": self.test_assessment,
            "scope_violations": list(self.scope_violations),
            "remaining_risks": list(self.remaining_risks),
            "corrections": list(self.corrections),
            "question_for_user": self.question_for_user,
        }


class ConversationActor(str, Enum):
    USER = "user"
    FABLE = "fable"
    SOL = "sol"
    SYSTEM = "system"


class ConversationTarget(str, Enum):
    USER = "user"
    FABLE = "fable"
    SOL = "sol"
    TEAM = "team"


class ConversationMessageType(str, Enum):
    STATEMENT = "statement"
    QUESTION = "question"
    ANSWER = "answer"
    APPROVAL = "approval"
    INTERVENTION = "intervention"
    STATUS = "status"


_CONVERSATION_ENVELOPE_FIELDS = (
    "sender",
    "addressed_to",
    "routed_to",
    "message_type",
    "text",
    "task_id",
    "revision",
    "continuation_generation",
    "question_id",
    "reply_to_question_id",
)


def _conversation_binding(
    task_id: str | None,
    revision: int | None,
    continuation_generation: int | None,
) -> tuple[str | None, int | None, int | None]:
    if (task_id is None, revision is None, continuation_generation is None).count(True) not in (0, 3):
        raise ValueError("task_id, revision, and continuation_generation binding must be all-or-none")
    if task_id is not None:
        task_id = _conversation_identifier(task_id, "task_id")
        revision = _integer(revision, "revision")
        continuation_generation = _integer(continuation_generation, "continuation_generation")
        if revision < 1:
            raise ValueError("revision must be >= 1")
        if continuation_generation < 1:
            raise ValueError("continuation_generation must be >= 1")
    return task_id, revision, continuation_generation


def _conversation_question_pair(
    message_type: ConversationMessageType,
    task_id: str | None,
    revision: int | None,
    continuation_generation: int | None,
    question_id: str | None,
    reply_to_question_id: str | None,
) -> tuple[str | None, str | None]:
    if question_id is not None:
        question_id = _conversation_identifier(question_id, "question_id")
    if reply_to_question_id is not None:
        reply_to_question_id = _conversation_identifier(reply_to_question_id, "reply_to_question_id")
    if message_type is ConversationMessageType.QUESTION:
        if question_id is None or reply_to_question_id is not None:
            raise ValueError("questions require question_id and no reply_to_question_id")
        if task_id is None or revision is None or continuation_generation is None:
            raise ValueError("questions require task_id, revision, and continuation_generation")
    elif message_type is ConversationMessageType.ANSWER:
        if question_id is not None or reply_to_question_id is None:
            raise ValueError("answers require reply_to_question_id and no question_id")
        if task_id is None or revision is None or continuation_generation is None:
            raise ValueError("answers require task_id, revision, and continuation_generation")
    elif question_id is not None or reply_to_question_id is not None:
        raise ValueError("only questions and answers may bind question identifiers")
    return question_id, reply_to_question_id


@dataclass(frozen=True, slots=True)
class ConversationEnvelope:
    sender: ConversationActor
    addressed_to: ConversationTarget
    routed_to: ConversationTarget
    message_type: ConversationMessageType
    text: str
    task_id: str | None = None
    revision: int | None = None
    continuation_generation: int | None = None
    question_id: str | None = None
    reply_to_question_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sender, ConversationActor):
            raise ValueError("sender must be a ConversationActor")
        if not isinstance(self.addressed_to, ConversationTarget):
            raise ValueError("addressed_to must be a ConversationTarget")
        if not isinstance(self.routed_to, ConversationTarget):
            raise ValueError("routed_to must be a ConversationTarget")
        if not isinstance(self.message_type, ConversationMessageType):
            raise ValueError("message_type must be a ConversationMessageType")
        object.__setattr__(self, "text", _bounded_conversation_text(self.text, "text"))
        task_id, revision, continuation_generation = _conversation_binding(
            self.task_id, self.revision, self.continuation_generation
        )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "continuation_generation", continuation_generation)
        question_id, reply_to_question_id = _conversation_question_pair(
            self.message_type,
            task_id,
            revision,
            continuation_generation,
            self.question_id,
            self.reply_to_question_id,
        )
        object.__setattr__(self, "question_id", question_id)
        object.__setattr__(self, "reply_to_question_id", reply_to_question_id)
        if self.message_type is ConversationMessageType.APPROVAL and (
            task_id is None or revision is None
        ):
            raise ValueError("approvals require task_id and revision")
        if self.message_type is ConversationMessageType.STATUS and self.sender is not ConversationActor.SYSTEM:
            raise ValueError("status messages require sender=system")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ConversationEnvelope":
        payload = _mapping(data, "ConversationEnvelope")
        _require_fields(payload, "ConversationEnvelope", _CONVERSATION_ENVELOPE_FIELDS)
        try:
            sender = ConversationActor(payload["sender"])
        except (TypeError, ValueError) as error:
            raise ValueError("sender must be a known conversation actor") from error
        try:
            addressed_to = ConversationTarget(payload["addressed_to"])
        except (TypeError, ValueError) as error:
            raise ValueError("addressed_to must be a known conversation target") from error
        try:
            routed_to = ConversationTarget(payload["routed_to"])
        except (TypeError, ValueError) as error:
            raise ValueError("routed_to must be a known conversation target") from error
        try:
            message_type = ConversationMessageType(payload["message_type"])
        except (TypeError, ValueError) as error:
            raise ValueError("message_type must be a known conversation message type") from error
        return cls(
            sender=sender,
            addressed_to=addressed_to,
            routed_to=routed_to,
            message_type=message_type,
            text=payload["text"],
            task_id=payload["task_id"],
            revision=payload["revision"],
            continuation_generation=payload["continuation_generation"],
            question_id=payload["question_id"],
            reply_to_question_id=payload["reply_to_question_id"],
        )  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {
            "sender": self.sender.value,
            "addressed_to": self.addressed_to.value,
            "routed_to": self.routed_to.value,
            "message_type": self.message_type.value,
            "text": self.text,
            "task_id": self.task_id,
            "revision": self.revision,
            "continuation_generation": self.continuation_generation,
            "question_id": self.question_id,
            "reply_to_question_id": self.reply_to_question_id,
        }


@dataclass(frozen=True, slots=True)
class UserConversationInput:
    addressed_to: ConversationTarget
    message_type: ConversationMessageType
    text: str
    task_id: str | None = None
    revision: int | None = None
    question_id: str | None = None
    reply_to_question_id: str | None = None
    continuation_generation: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.addressed_to, ConversationTarget):
            raise ValueError("addressed_to must be a ConversationTarget")
        if not isinstance(self.message_type, ConversationMessageType):
            raise ValueError("message_type must be a ConversationMessageType")
        object.__setattr__(self, "text", _bounded_conversation_text(self.text, "text"))
        task_id, revision, continuation_generation = _conversation_binding(
            self.task_id, self.revision, self.continuation_generation
        )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "continuation_generation", continuation_generation)
        question_id, reply_to_question_id = _conversation_question_pair(
            self.message_type,
            task_id,
            revision,
            continuation_generation,
            self.question_id,
            self.reply_to_question_id,
        )
        object.__setattr__(self, "question_id", question_id)
        object.__setattr__(self, "reply_to_question_id", reply_to_question_id)
        if self.message_type is ConversationMessageType.APPROVAL and (
            task_id is None or revision is None
        ):
            raise ValueError("approvals require task_id and revision")


@dataclass(frozen=True, slots=True)
class DirectedAgentQuestion:
    addressed_to: Literal["user", "fable", "sol"]
    text: str
    reason: str

    def __post_init__(self) -> None:
        if self.addressed_to not in {"user", "fable", "sol"}:
            raise ValueError("addressed_to must be user, fable, or sol")
        object.__setattr__(self, "text", _bounded_conversation_text(self.text, "text"))
        object.__setattr__(self, "reason", _bounded_conversation_text(self.reason, "reason"))


@dataclass(frozen=True)
class StreamEvent:
    sequence: int
    session_id: str
    task_id: str | None
    actor: str
    kind: str
    payload: Mapping[str, object]
    created_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _integer(self.sequence, "sequence"))
        object.__setattr__(self, "session_id", _string(self.session_id, "session_id"))
        if self.task_id is not None:
            object.__setattr__(self, "task_id", _string(self.task_id, "task_id"))
        object.__setattr__(self, "actor", _string(self.actor, "actor"))
        object.__setattr__(self, "kind", _string(self.kind, "kind"))
        object.__setattr__(self, "created_at", _string(self.created_at, "created_at"))
        frozen_payload = freeze_json(self.payload)
        if not isinstance(frozen_payload, Mapping):
            raise ValueError("payload must be an object")
        object.__setattr__(self, "payload", frozen_payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "actor": self.actor,
            "kind": self.kind,
            "payload": _thaw_json(self.payload),
            "created_at": self.created_at,
        }


def _array_schema(items: dict[str, object]) -> dict[str, object]:
    return {"type": "array", "items": items}


def _object_schema(properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_STRING_SCHEMA: dict[str, object] = {"type": "string"}
_STRING_ARRAY_SCHEMA = _array_schema(_STRING_SCHEMA)
_COMMAND_REPORT_SCHEMA = _object_schema({
    "command": _STRING_SCHEMA,
    "exit_code": {"type": "integer"},
    "result": _STRING_SCHEMA,
})
_SOL_QUESTION_SCHEMA = _object_schema({
    "ambiguity": _STRING_SCHEMA,
    "why_it_matters": _STRING_SCHEMA,
    "options": _STRING_ARRAY_SCHEMA,
    "recommendation": _STRING_SCHEMA,
    "can_continue_safely": {"type": "boolean"},
})
_CRITERION_EVIDENCE_SCHEMA = _object_schema({
    "criterion": _STRING_SCHEMA,
    "evidence": _STRING_ARRAY_SCHEMA,
    "satisfied": {"type": "boolean"},
})

TASK_BRIEF_SCHEMA = _object_schema({
    "task_id": _STRING_SCHEMA,
    "revision": {"type": "integer", "minimum": 1},
    "title": _STRING_SCHEMA,
    "objective": _STRING_SCHEMA,
    "context": _STRING_ARRAY_SCHEMA,
    "constraints": _STRING_ARRAY_SCHEMA,
    "allowed_paths": {"type": "array", "items": _STRING_SCHEMA, "minItems": 1},
    "out_of_scope": _STRING_ARRAY_SCHEMA,
    "acceptance_criteria": {"type": "array", "items": _STRING_SCHEMA, "minItems": 1},
    "required_tests": _STRING_ARRAY_SCHEMA,
    "risks": _STRING_ARRAY_SCHEMA,
    "open_questions": _STRING_ARRAY_SCHEMA,
    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "confidence_rationale": _STRING_SCHEMA,
})

SOL_OUTCOME_SCHEMA = _object_schema({
    "status": {"type": "string", "enum": sorted(_SOL_OUTCOME_STATUSES)},
    "summary": _STRING_SCHEMA,
    "changed_files": _STRING_ARRAY_SCHEMA,
    "commands_run": _array_schema(_COMMAND_REPORT_SCHEMA),
    "known_failures": _STRING_ARRAY_SCHEMA,
    "remaining_risks": _STRING_ARRAY_SCHEMA,
    "architecture_docs": _STRING_SCHEMA,
    "question": {"anyOf": [_SOL_QUESTION_SCHEMA, {"type": "null"}]},
})

FABLE_CLARIFICATION_SCHEMA = _object_schema({
    "status": {"type": "string", "enum": sorted(_FABLE_CLARIFICATION_STATUSES)},
    "answer": {"type": ["string", "null"]},
    "reasoning": _STRING_SCHEMA,
    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "scope_changed": {"type": "boolean"},
    "revised_brief": {"anyOf": [TASK_BRIEF_SCHEMA, {"type": "null"}]},
    "question_for_user": {"type": ["string", "null"]},
})

REVIEW_VERDICT_SCHEMA = _object_schema({
    "status": {"type": "string", "enum": sorted(_REVIEW_VERDICT_STATUSES)},
    "summary": _STRING_SCHEMA,
    "criteria": _array_schema(_CRITERION_EVIDENCE_SCHEMA),
    "test_assessment": _STRING_SCHEMA,
    "scope_violations": _STRING_ARRAY_SCHEMA,
    "remaining_risks": _STRING_ARRAY_SCHEMA,
    "corrections": _STRING_ARRAY_SCHEMA,
    "question_for_user": {"type": ["string", "null"]},
})
