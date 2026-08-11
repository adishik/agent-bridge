from __future__ import annotations

from math import inf, nan

import pytest

from agent_bridge.contracts import (
    FableClarification,
    ReviewVerdict,
    SolOutcome,
    StreamEvent,
    TaskBrief,
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
