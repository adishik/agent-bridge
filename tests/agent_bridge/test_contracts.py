from __future__ import annotations

import json
from dataclasses import asdict, fields
from math import inf, nan

import pytest

from agent_bridge.contracts import (
    ConversationActor,
    ConversationEnvelope,
    ConversationMessageType,
    ConversationTarget,
    DirectedAgentQuestion,
    FableClarification,
    ReviewVerdict,
    SolOutcome,
    StreamEvent,
    TaskBrief,
    UserConversationInput,
)


VALID_BRIEF = {
    "task_id": "task-1",
    "revision": 1,
    "title": "Add bridge contracts",
    "objective": "Create immutable validated handoff contracts.",
    "context": ["The checkout may already be dirty."],
    "constraints": ["Fable is read-only."],
    "allowed_paths": ["src/agent_bridge", "tests/agent_bridge"],
    "out_of_scope": ["outside-project"],
    "acceptance_criteria": ["Invalid revisions raise ValueError."],
    "required_tests": ["tests/agent_bridge/test_contracts.py"],
    "risks": ["A mutable brief could detach approval from execution."],
    "open_questions": [],
    "confidence": 0.95,
    "confidence_rationale": "All fields are explicit.",
}


VALID_CONVERSATION_ENVELOPE = {
    "sender": "user",
    "addressed_to": "fable",
    "routed_to": "fable",
    "message_type": "question",
    "text": "Should the coordinator continue?",
    "task_id": "task-1",
    "revision": 1,
    "continuation_generation": 2,
    "question_id": "question-1",
    "reply_to_question_id": None,
}


def test_conversation_envelope_has_an_exact_canonical_json_round_trip() -> None:
    envelope = ConversationEnvelope.from_dict(VALID_CONVERSATION_ENVELOPE)

    assert envelope.to_dict() == VALID_CONVERSATION_ENVELOPE
    assert json.loads(json.dumps(envelope.to_dict(), sort_keys=True)) == VALID_CONVERSATION_ENVELOPE
    assert envelope.sender is ConversationActor.USER
    assert envelope.addressed_to is ConversationTarget.FABLE
    assert envelope.message_type is ConversationMessageType.QUESTION


def test_conversation_contracts_have_exact_fields_and_canonical_json_values() -> None:
    input_message = UserConversationInput(
        addressed_to=ConversationTarget.TEAM,
        message_type=ConversationMessageType.STATEMENT,
        text="The team can inspect this.",
    )
    directed_question = DirectedAgentQuestion(
        addressed_to="user",
        text="Continue?",
        reason="The coordinator needs a decision.",
    )

    assert tuple(field.name for field in fields(ConversationEnvelope)) == tuple(VALID_CONVERSATION_ENVELOPE)
    assert tuple(field.name for field in fields(UserConversationInput)) == (
        "addressed_to", "message_type", "text", "task_id", "revision", "question_id",
        "reply_to_question_id", "continuation_generation",
    )
    assert tuple(field.name for field in fields(DirectedAgentQuestion)) == (
        "addressed_to", "text", "reason",
    )
    assert json.loads(json.dumps(asdict(input_message), sort_keys=True)) == {
        "addressed_to": "team",
        "message_type": "statement",
        "text": "The team can inspect this.",
        "task_id": None,
        "revision": None,
        "question_id": None,
        "reply_to_question_id": None,
        "continuation_generation": None,
    }
    assert json.loads(json.dumps(asdict(directed_question), sort_keys=True)) == {
        "addressed_to": "user",
        "text": "Continue?",
        "reason": "The coordinator needs a decision.",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sender", "unknown"),
        ("addressed_to", "unknown"),
        ("routed_to", "unknown"),
        ("message_type", "unknown"),
        ("revision", True),
        ("continuation_generation", True),
        ("revision", 0),
        ("continuation_generation", 0),
        ("task_id", "task\n1"),
        ("task_id", "x" * 129),
        ("question_id", "question\x00"),
        ("text", ""),
        ("text", "contains\ncontrol"),
        ("text", "x" * (16 * 1024 + 1)),
    ],
)
def test_conversation_envelope_rejects_invalid_serialized_values(field: str, value: object) -> None:
    payload = dict(VALID_CONVERSATION_ENVELOPE)
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        ConversationEnvelope.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {key: value for key, value in VALID_CONVERSATION_ENVELOPE.items() if key != "routed_to"},
        {**VALID_CONVERSATION_ENVELOPE, "project_id": "outside-contract"},
        {**VALID_CONVERSATION_ENVELOPE, "message_type": "statement"},
        {**VALID_CONVERSATION_ENVELOPE, "message_type": "answer", "question_id": None},
        {**VALID_CONVERSATION_ENVELOPE, "question_id": "same", "reply_to_question_id": "same"},
        {
            **VALID_CONVERSATION_ENVELOPE,
            "message_type": "answer",
            "question_id": None,
            "reply_to_question_id": None,
        },
        {
            **VALID_CONVERSATION_ENVELOPE,
            "task_id": None,
            "revision": None,
            "continuation_generation": None,
        },
    ],
)
def test_conversation_envelope_rejects_invalid_field_sets_and_question_pairs(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ConversationEnvelope.from_dict(payload)


def test_conversation_envelope_requires_approval_binding_and_system_status_sender() -> None:
    with pytest.raises(ValueError, match="task_id"):
        ConversationEnvelope(
            sender=ConversationActor.USER,
            addressed_to=ConversationTarget.TEAM,
            routed_to=ConversationTarget.TEAM,
            message_type=ConversationMessageType.APPROVAL,
            text="Approved.",
        )
    with pytest.raises(ValueError, match="sender"):
        ConversationEnvelope(
            sender=ConversationActor.USER,
            addressed_to=ConversationTarget.TEAM,
            routed_to=ConversationTarget.TEAM,
            message_type=ConversationMessageType.STATUS,
            text="Waiting.",
        )


@pytest.mark.parametrize(
    ("character", "count", "is_valid"),
    [
        ("x", 16 * 1024, True),
        ("x", 16 * 1024 + 1, False),
        ("é", 8192, True),
        ("é", 8193, False),
        ("é", 9000, False),
    ],
)
def test_conversation_envelope_enforces_a_utf8_byte_limit(
    character: str, count: int, is_valid: bool,
) -> None:
    text = character * count
    kwargs = {
        "sender": ConversationActor.USER,
        "addressed_to": ConversationTarget.TEAM,
        "routed_to": ConversationTarget.TEAM,
        "message_type": ConversationMessageType.STATEMENT,
        "text": text,
    }

    if is_valid:
        assert ConversationEnvelope(**kwargs).text == text
    else:
        with pytest.raises(ValueError, match="text is too long"):
            ConversationEnvelope(**kwargs)


@pytest.mark.parametrize(
    ("character", "count", "is_valid"),
    [
        ("x", 16 * 1024, True),
        ("x", 16 * 1024 + 1, False),
        ("é", 8192, True),
        ("é", 8193, False),
        ("é", 9000, False),
    ],
)
def test_directed_agent_question_enforces_a_utf8_byte_limit(
    character: str, count: int, is_valid: bool,
) -> None:
    reason = character * count
    kwargs = {"addressed_to": "fable", "text": "Question?", "reason": reason}

    if is_valid:
        assert DirectedAgentQuestion(**kwargs).reason == reason
    else:
        with pytest.raises(ValueError, match="reason is too long"):
            DirectedAgentQuestion(**kwargs)


def test_conversation_envelope_approval_binds_exactly_task_and_revision() -> None:
    approval = ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.TEAM,
        routed_to=ConversationTarget.TEAM,
        message_type=ConversationMessageType.APPROVAL,
        text="Approved.",
        task_id="task-1",
        revision=1,
    )

    assert approval.continuation_generation is None
    for binding in (
        {"task_id": "task-1"},
        {"revision": 1},
        {"continuation_generation": 1},
        {"task_id": "task-1", "continuation_generation": 1},
        {"revision": 1, "continuation_generation": 1},
        {"task_id": "task-1", "revision": 1, "continuation_generation": 1},
    ):
        with pytest.raises(ValueError, match="approval"):
            ConversationEnvelope(
                sender=ConversationActor.USER,
                addressed_to=ConversationTarget.TEAM,
                routed_to=ConversationTarget.TEAM,
                message_type=ConversationMessageType.APPROVAL,
                text="Approved.",
                **binding,
            )


def test_user_conversation_input_validates_exact_optional_binding_before_routing() -> None:
    input_message = UserConversationInput(
        addressed_to=ConversationTarget.FABLE,
        message_type=ConversationMessageType.QUESTION,
        text="What remains?",
        task_id="task-1",
        revision=1,
        continuation_generation=2,
        question_id="question-1",
    )

    assert input_message.addressed_to is ConversationTarget.FABLE
    assert UserConversationInput(
        addressed_to=ConversationTarget.TEAM,
        message_type=ConversationMessageType.APPROVAL,
        text="Approved.",
        task_id="task-1",
        revision=1,
    ).continuation_generation is None
    with pytest.raises(ValueError, match="binding"):
        UserConversationInput(
            addressed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.STATEMENT,
            text="Unbound partial context.",
            task_id="task-1",
        )
    with pytest.raises(ValueError, match="question_id"):
        UserConversationInput(
            addressed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.QUESTION,
            text="Which revision?",
            task_id="task-1",
            revision=1,
            continuation_generation=2,
        )
    with pytest.raises(ValueError, match="reply_to_question_id"):
        UserConversationInput(
            addressed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.ANSWER,
            text="Revision one.",
            task_id="task-1",
            revision=1,
            continuation_generation=2,
        )
    with pytest.raises(TypeError, match="sender"):
        UserConversationInput(  # type: ignore[call-arg]
            addressed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.STATEMENT,
            text="Coordinator must set this.",
            sender=ConversationActor.USER,
        )
    with pytest.raises(TypeError, match="routed_to"):
        UserConversationInput(  # type: ignore[call-arg]
            addressed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.STATEMENT,
            text="Coordinator must route this.",
            routed_to=ConversationTarget.FABLE,
        )
    with pytest.raises(ValueError, match="status"):
        UserConversationInput(
            addressed_to=ConversationTarget.TEAM,
            message_type=ConversationMessageType.STATUS,
            text="The user cannot emit this.",
        )


def test_directed_agent_question_has_limited_targets_and_bounded_nonempty_text() -> None:
    question = DirectedAgentQuestion(
        addressed_to="sol",
        text="Which contract is ready?",
        reason="The review is blocked on the persisted envelope.",
    )

    assert question.addressed_to == "sol"
    with pytest.raises(ValueError, match="addressed_to"):
        DirectedAgentQuestion(
            addressed_to="team",  # type: ignore[arg-type]
            text="Invalid target.",
            reason="A direct question cannot target the whole team.",
        )
    with pytest.raises(ValueError, match="reason"):
        DirectedAgentQuestion(addressed_to="user", text="Question?", reason="x" * (16 * 1024 + 1))
    with pytest.raises(ValueError, match="reason"):
        DirectedAgentQuestion(addressed_to="user", text="Question?", reason="Contains\x7fcontrol")
    with pytest.raises(ValueError, match="text"):
        DirectedAgentQuestion(addressed_to="fable", text="", reason="Reason.")


def test_task_brief_is_immutable_and_round_trips() -> None:
    brief = TaskBrief.from_dict(VALID_BRIEF)
    assert brief.to_dict() == VALID_BRIEF
    with pytest.raises(AttributeError):
        brief.revision = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [("revision", 0), ("title", ""), ("allowed_paths", []),
     ("acceptance_criteria", []), ("confidence", 1.1)],
)
def test_task_brief_rejects_invalid_required_values(field: str, value: object) -> None:
    payload = dict(VALID_BRIEF)
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        TaskBrief.from_dict(payload)


def test_escalated_review_requires_a_question_for_the_user() -> None:
    with pytest.raises(ValueError, match="question_for_user"):
        ReviewVerdict(
            status="escalate_to_user",
            summary="A decision is needed.",
            criteria=(),
            test_assessment="Not run.",
            scope_violations=(),
            remaining_risks=(),
            corrections=(),
            question_for_user=None,
        )


@pytest.mark.parametrize("non_json_number", (nan, inf, -inf))
def test_stream_event_rejects_non_json_finite_payload_numbers(non_json_number: float) -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        StreamEvent(
            sequence=1,
            session_id="session-1",
            task_id="task-1",
            actor="sol",
            kind="progress",
            payload={"value": non_json_number},
            created_at="2026-08-10T00:00:00Z",
        )


def test_direct_malformed_nested_collections_raise_value_error() -> None:
    with pytest.raises(ValueError, match="commands_run"):
        SolOutcome(
            status="completed",
            summary="Complete.",
            changed_files=(),
            commands_run=None,  # type: ignore[arg-type]
            known_failures=(),
            remaining_risks=(),
            architecture_docs="None.",
            question=None,
        )

    with pytest.raises(ValueError, match="criteria"):
        ReviewVerdict(
            status="approved",
            summary="Approved.",
            criteria=None,  # type: ignore[arg-type]
            test_assessment="Passed.",
            scope_violations=(),
            remaining_risks=(),
            corrections=(),
            question_for_user=None,
        )


def test_stream_event_freezes_nested_payload_and_to_dict_returns_fresh_data() -> None:
    payload = {"nested": {"items": ["original", {"value": "stable"}]}}
    event = StreamEvent(
        sequence=1,
        session_id="session-1",
        task_id="task-1",
        actor="sol",
        kind="progress",
        payload=payload,
        created_at="2026-08-10T00:00:00Z",
    )

    with pytest.raises(TypeError):
        event.payload["nested"]["items"][1]["value"] = "blocked"  # type: ignore[index]
    with pytest.raises(AttributeError):
        event.payload["nested"]["items"].append("blocked")  # type: ignore[index]
    payload["nested"]["items"][1]["value"] = "mutated"
    payload["nested"]["items"].append("new")

    assert event.to_dict()["payload"] == {
        "nested": {"items": ["original", {"value": "stable"}]}
    }
    with pytest.raises(TypeError):
        event.payload["nested"] = {}  # type: ignore[index]

    mutable_copy = event.to_dict()
    mutable_copy["payload"]["nested"]["items"][1]["value"] = "copy-only"  # type: ignore[index]
    assert event.to_dict()["payload"] == {
        "nested": {"items": ["original", {"value": "stable"}]}
    }


def test_sol_outcome_enforces_question_discriminator_variants() -> None:
    completed = SolOutcome.from_dict({
        "status": "completed",
        "summary": "Complete.",
        "changed_files": [],
        "commands_run": [],
        "known_failures": [],
        "remaining_risks": [],
        "architecture_docs": "None.",
        "question": None,
    })
    assert completed.question is None

    question = SolOutcome.from_dict({
        "status": "question",
        "summary": "Clarification needed.",
        "changed_files": [],
        "commands_run": [],
        "known_failures": [],
        "remaining_risks": [],
        "architecture_docs": "None.",
        "question": {
            "ambiguity": "Approval intent is unclear.",
            "why_it_matters": "Execution must bind to approval.",
            "options": ["Resume", "Cancel"],
            "recommendation": "Ask the user.",
            "can_continue_safely": False,
        },
    })
    assert question.question is not None
    assert question.question.options == ("Resume", "Cancel")

    with pytest.raises(ValueError, match="at least two options"):
        SolOutcome.from_dict({
            **question.to_dict(),
            "question": {**question.question.to_dict(), "options": ["Resume"]},
        })

    with pytest.raises(ValueError, match="require a question"):
        SolOutcome.from_dict({**question.to_dict(), "question": None})

    with pytest.raises(ValueError, match="null question"):
        SolOutcome.from_dict({**completed.to_dict(), "question": question.question.to_dict()})


def test_fable_clarification_enforces_answer_scope_and_escalation_gates(
    valid_brief: TaskBrief,
) -> None:
    assert FableClarification(
        status="answered",
        answer="Stay within the approved paths.",
        reasoning="The scope is explicit.",
        confidence=0.9,
        scope_changed=False,
        revised_brief=None,
        question_for_user=None,
    ).answer == "Stay within the approved paths."
    assert FableClarification(
        status="answered",
        answer="The scope changed.",
        reasoning="The revised brief is explicit.",
        confidence=0.9,
        scope_changed=True,
        revised_brief=valid_brief,
        question_for_user=None,
    ).revised_brief is valid_brief
    assert FableClarification(
        status="escalate_to_user",
        answer=None,
        reasoning="The ambiguity changes scope.",
        confidence=0.9,
        scope_changed=False,
        revised_brief=None,
        question_for_user="Should the bridge continue?",
    ).question_for_user == "Should the bridge continue?"

    with pytest.raises(ValueError, match="non-empty answer"):
        FableClarification(
            status="answered",
            answer=None,
            reasoning="Missing answer.",
            confidence=0.9,
            scope_changed=False,
            revised_brief=None,
            question_for_user=None,
        )
    with pytest.raises(ValueError, match="non-empty answer"):
        FableClarification(
            status="answered",
            answer="   ",
            reasoning="Whitespace is not an answer.",
            confidence=0.9,
            scope_changed=False,
            revised_brief=None,
            question_for_user=None,
        )
    with pytest.raises(ValueError, match="revised_brief"):
        FableClarification(
            status="answered",
            answer="Scope changed.",
            reasoning="Missing revision.",
            confidence=0.9,
            scope_changed=True,
            revised_brief=None,
            question_for_user=None,
        )
    with pytest.raises(ValueError, match="question_for_user"):
        FableClarification(
            status="escalate_to_user",
            answer=None,
            reasoning="A user choice is needed.",
            confidence=0.9,
            scope_changed=False,
            revised_brief=None,
            question_for_user=None,
        )
    with pytest.raises(ValueError, match="question_for_user"):
        FableClarification(
            status="escalate_to_user",
            answer=None,
            reasoning="Whitespace does not ask the user anything.",
            confidence=0.9,
            scope_changed=False,
            revised_brief=None,
            question_for_user="   ",
        )
