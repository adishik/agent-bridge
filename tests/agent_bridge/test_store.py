from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import threading
import tracemalloc

import pytest

import agent_bridge.store as store_module
from agent_bridge.contracts import (
    ConversationActor,
    ConversationEnvelope,
    ConversationMessageType,
    ConversationTarget,
    DirectedAgentQuestion,
    FableClarification,
)
from agent_bridge.projects import project_id_for_root
from agent_bridge.state_machine import TaskState
from agent_bridge.store import (
    ApprovalPayload,
    AnswerContext,
    AnswerPayload,
    BaselineSetting,
    ClarificationContext,
    ContinuationMessagePayload,
    ExchangeReservation,
    ExchangeGrantPayload,
    EXCHANGE_GRANT_SIZE,
    INITIAL_INTERNAL_EXCHANGES,
    MAX_TASK_OVERVIEWS,
    NewRequestPayload,
    QuestionRecord,
    QuestionAnswerPayload,
    ResumeDriftProjection,
    ResumePayload,
    ReviewContext,
    ScopeApprovalContext,
    SQLiteStore,
    SolResumeContext,
)


def _store(tmp_path) -> SQLiteStore:
    return SQLiteStore(tmp_path / "bridge.sqlite3", clock=lambda: "2026-08-10T12:00:00Z")


def test_events_replay_in_monotonic_order(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    first = store.append_event("session-1", None, "user", "message", {"text": "hi"})
    second = store.append_event("session-1", None, "fable", "status", {"state": "planning"})

    assert first.created_at == "2026-08-10T12:00:00Z"
    assert second.sequence == first.sequence + 1
    assert store.events_after("session-1", first.sequence) == (second,)


def test_event_listeners_observe_each_committed_event_once_and_cannot_rollback(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    seen = []
    failed = []

    def committed_listener(event) -> None:
        assert store.events_after("session-1", event.sequence - 1) == (event,)
        seen.append(event)

    def failing_listener(event) -> None:
        failed.append(event)
        raise RuntimeError("listener failures are isolated after commit")

    committed_token = store.add_event_listener(committed_listener)
    failing_token = store.add_event_listener(failing_listener)
    event = store.append_event(
        "session-1", None, "coordinator", "status", {"state": "ready"}
    )

    assert seen == [event]
    assert failed == [event]
    assert store.events_after("session-1", 0) == (event,)

    store.remove_event_listener(committed_token)
    store.remove_event_listener(failing_token)
    second = store.append_event(
        "session-1", None, "coordinator", "status", {"state": "idle"}
    )
    assert seen == [event]
    assert failed == [event]
    assert store.events_after("session-1", event.sequence) == (second,)


def test_reentrant_event_append_is_dispatched_fifo_to_every_listener(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    first_listener_sequences: list[int] = []
    second_listener_sequences: list[int] = []
    inner_events = []

    def append_reentrantly(event) -> None:
        first_listener_sequences.append(event.sequence)
        if event.kind == "outer":
            inner_events.append(
                store.append_event(
                    "session-1",
                    None,
                    "coordinator",
                    "inner",
                    {"state": "inner"},
                )
            )

    def observe_all(event) -> None:
        second_listener_sequences.append(event.sequence)

    store.add_event_listener(append_reentrantly)
    store.add_event_listener(observe_all)
    outer = store.append_event(
        "session-1", None, "coordinator", "outer", {"state": "outer"}
    )

    assert len(inner_events) == 1
    inner = inner_events[0]
    assert inner.sequence == outer.sequence + 1
    assert first_listener_sequences == [outer.sequence, inner.sequence]
    assert second_listener_sequences == [outer.sequence, inner.sequence]


def test_duplicate_bound_event_listener_registration_is_idempotent(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")

    class Observer:
        def __init__(self) -> None:
            self.sequences: list[int] = []

        def receive(self, event) -> None:
            self.sequences.append(event.sequence)

    observer = Observer()
    first_token = store.add_event_listener(observer.receive)
    duplicate_token = store.add_event_listener(observer.receive)
    event = store.append_event(
        "session-1", None, "coordinator", "status", {"state": "ready"}
    )

    assert duplicate_token == first_token
    assert observer.sequences == [event.sequence]
    store.remove_event_listener(duplicate_token)
    store.remove_event_listener(first_token)
    store.append_event(
        "session-1", None, "coordinator", "status", {"state": "idle"}
    )
    assert observer.sequences == [event.sequence]


def test_event_listener_registration_is_thread_safe(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    failures: list[BaseException] = []

    def churn() -> None:
        try:
            for _ in range(100):
                token = store.add_event_listener(lambda event: None)
                store.remove_event_listener(token)
        except BaseException as error:
            failures.append(error)

    workers = [threading.Thread(target=churn) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)

    assert failures == []
    assert all(worker.is_alive() is False for worker in workers)
    event = store.append_event(
        "session-1", None, "coordinator", "status", {"state": "ready"}
    )
    assert store.events_after("session-1", 0) == (event,)


def test_scope_revision_event_batch_publishes_before_a_later_committed_event(
    tmp_path,
    valid_brief,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later writer cannot publish past a committed scope-answer batch."""
    store = SQLiteStore(
        tmp_path / "scope-event-order.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
        check_same_thread=False,
    )
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_fable_session(valid_brief.task_id, valid_brief.revision, "fable-session-1")
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "sol-thread-1")
    store._connection.execute(  # noqa: SLF001 - exact durable scope fixture
        "UPDATE tasks SET approved_at = ?, baseline_id = ? WHERE task_id = ? AND revision = ?",
        (
            "2026-08-10T12:00:00Z",
            "baseline-1",
            valid_brief.task_id,
            valid_brief.revision,
        ),
    )
    pending = {"sol_run_id": "sol-run-1", "prompt": "continue exactly"}
    _, question = store.reserve_internal_question(
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        expected_generation=1,
        question_id="scope-order-question",
        request_key="scope-order-request",
        asked_by=ConversationActor.SOL,
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        text="May the exact scope include one more file?",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action=pending,
        event=ConversationEnvelope(
            sender=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.QUESTION,
            text="May the exact scope include one more file?",
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            continuation_generation=1,
            question_id="scope-order-question",
        ),
    )
    revised = replace(
        valid_brief,
        revision=2,
        allowed_paths=(*valid_brief.allowed_paths, "scope-extra.txt"),
    )
    clarification = FableClarification.from_dict({
        "status": "answered",
        "answer": "Add the exact bounded path.",
        "reasoning": "The additional path is explicitly bounded.",
        "confidence": 0.9,
        "scope_changed": True,
        "revised_brief": revised.to_dict(),
        "question_for_user": None,
        "directed_question": None,
    })
    answer_event = ConversationEnvelope(
        sender=ConversationActor.FABLE,
        addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.SOL,
        message_type=ConversationMessageType.ANSWER,
        text=clarification.answer or "",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        continuation_generation=1,
        reply_to_question_id=question.question_id,
    )
    observed = []
    failing_observer = []

    def fail_after_commit(event) -> None:
        failing_observer.append(event.sequence)
        raise RuntimeError("listener failures remain isolated")

    store.add_event_listener(observed.append)
    store.add_event_listener(fail_after_commit)
    commit_entered = threading.Event()
    release_scope_publish = threading.Event()
    later_finished = threading.Event()
    failures: list[BaseException] = []
    original_transaction = store._immediate_transaction  # noqa: SLF001

    @contextmanager
    def block_scope_after_commit():
        with original_transaction():
            yield
        if threading.current_thread().name == "scope-revision-writer":
            commit_entered.set()
            if not release_scope_publish.wait(timeout=2):
                raise TimeoutError("scope publication was not released")

    monkeypatch.setattr(store, "_immediate_transaction", block_scope_after_commit)

    def save_scope() -> None:
        try:
            store.save_scope_revision(
                "session-1",
                revised,
                fable_session_id="fable-session-1",
                sol_thread_id="sol-thread-1",
                correction_count=0,
                continuation_state=TaskState.SOL_RUNNING,
                pending={"answer": clarification.answer, **pending},
                baseline_id="baseline-1",
                setting=("agent_bridge.baseline.task-1.2", {"baseline_id": "baseline-1"}),
                clarification=clarification,
                answered_question_id=question.question_id,
                answered_question_generation=1,
                answered_pending=pending,
                answer_event=answer_event,
            )
        except BaseException as error:
            failures.append(error)

    def append_later() -> None:
        try:
            store.append_event(
                "session-1",
                valid_brief.task_id,
                "coordinator",
                "later",
                {"state": "later"},
            )
        except BaseException as error:
            failures.append(error)
        finally:
            later_finished.set()

    scope_thread = threading.Thread(target=save_scope, name="scope-revision-writer")
    scope_thread.start()
    assert commit_entered.wait(timeout=2)
    later_thread = threading.Thread(target=append_later)
    later_thread.start()
    # The unfixed writer completes here and publishes out of order.  The fixed
    # writer remains behind the listener lock until the scope batch is released.
    later_finished.wait(timeout=0.5)
    release_scope_publish.set()
    scope_thread.join(timeout=2)
    later_thread.join(timeout=2)

    assert failures == []
    assert scope_thread.is_alive() is False
    assert later_thread.is_alive() is False
    durable = store.events_after("session-1", 0)
    relevant = tuple(
        event for event in durable
        if event.kind in {"task_brief", "clarification", "later"}
        or (
            event.kind == "conversation"
            and event.payload.get("reply_to_question_id") == question.question_id
        )
    )
    assert [event.kind for event in relevant] == [
        "conversation", "task_brief", "clarification", "later",
    ]
    assert [event.sequence for event in observed[-4:]] == [
        event.sequence for event in relevant
    ]
    assert failing_observer[-4:] == [event.sequence for event in relevant]


def test_scope_revision_failure_publishes_nothing_and_releases_listener_lock(
    tmp_path,
    valid_brief,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed scope transaction neither dispatches events nor strands writers."""
    store = SQLiteStore(
        tmp_path / "scope-event-rollback.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
        check_same_thread=False,
    )
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_fable_session(valid_brief.task_id, valid_brief.revision, "fable-session-1")
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "sol-thread-1")
    revised = replace(valid_brief, revision=2)
    clarification = FableClarification.from_dict({
        "status": "answered",
        "answer": "Keep the same bounded scope.",
        "reasoning": "The bounded fixture forces rollback.",
        "confidence": 0.9,
        "scope_changed": True,
        "revised_brief": revised.to_dict(),
        "question_for_user": None,
        "directed_question": None,
    })
    observed = []
    store.add_event_listener(observed.append)
    original_insert = store._insert_event_in_transaction  # noqa: SLF001

    def fail_task_brief(*args, **kwargs):
        kind = args[3] if len(args) >= 4 else kwargs.get("kind")
        if kind == "task_brief":
            raise RuntimeError("injected scope event failure")
        return original_insert(*args, **kwargs)

    monkeypatch.setattr(store, "_insert_event_in_transaction", fail_task_brief)
    with pytest.raises(RuntimeError, match="injected scope event failure"):
        store.save_scope_revision(
            "session-1",
            revised,
            fable_session_id="fable-session-1",
            sol_thread_id="sol-thread-1",
            correction_count=0,
            continuation_state=TaskState.SOL_RUNNING,
            pending={"answer": clarification.answer, "sol_run_id": "sol-run-1"},
            baseline_id="baseline-1",
            clarification=clarification,
        )
    assert observed == []
    assert store.latest_task(valid_brief.task_id).revision == 1  # type: ignore[union-attr]

    appended: list[object] = []
    worker = threading.Thread(target=lambda: appended.append(store.append_event(
        "session-1", valid_brief.task_id, "coordinator", "after_failure", {"ok": True},
    )))
    worker.start()
    worker.join(timeout=2)
    assert worker.is_alive() is False
    assert len(appended) == 1


def test_sqlite_thread_mode_defaults_safe_validates_type_and_can_be_opted_out(
    tmp_path,
) -> None:
    default = _store(tmp_path)
    default.create_session("session-1", "/repo")
    default_errors: list[BaseException] = []

    def use_default_from_thread() -> None:
        try:
            default.session_exists("session-1")
        except BaseException as error:
            default_errors.append(error)

    default_thread = threading.Thread(target=use_default_from_thread)
    default_thread.start()
    default_thread.join(timeout=2)
    assert len(default_errors) == 1
    assert isinstance(default_errors[0], sqlite3.ProgrammingError)

    with pytest.raises(ValueError, match="check_same_thread"):
        SQLiteStore(tmp_path / "invalid.sqlite3", check_same_thread=1)

    shared = SQLiteStore(
        tmp_path / "shared.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
        check_same_thread=False,
    )
    shared.create_session("session-1", "/repo")
    shared_results: list[bool] = []
    shared_thread = threading.Thread(
        target=lambda: shared_results.append(shared.session_exists("session-1"))
    )
    shared_thread.start()
    shared_thread.join(timeout=2)
    assert shared_thread.is_alive() is False
    assert shared_results == [True]


def test_transition_uses_revision_specific_compare_and_swap(tmp_path, valid_brief) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)

    transitioned = store.transition_task(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.AWAITING_USER_APPROVAL,
        target=TaskState.SOL_RUNNING,
    )

    assert transitioned.state is TaskState.SOL_RUNNING
    with pytest.raises(RuntimeError, match="task state changed concurrently"):
        store.transition_task(
            valid_brief.task_id,
            valid_brief.revision,
            expected=TaskState.AWAITING_USER_APPROVAL,
            target=TaskState.SOL_RUNNING,
        )


def test_planning_revision_has_no_brief_and_latest_revision_advances(tmp_path, valid_brief) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    planning = store.create_planning_task("session-1", valid_brief.task_id)

    assert planning.revision == 0
    assert planning.state is TaskState.FABLE_PLANNING
    assert planning.brief is None
    assert store.latest_task(valid_brief.task_id) == planning
    with pytest.raises(RuntimeError, match="revision-zero task has no brief"):
        store.task_brief(valid_brief.task_id, 0)

    first = store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    second_brief = replace(valid_brief, revision=2, title="Revised bridge contracts")
    second = store.save_task("session-1", second_brief, TaskState.AWAITING_USER_APPROVAL)

    assert first.brief == valid_brief
    assert second.brief == second_brief
    assert store.latest_task(valid_brief.task_id) == second
    with pytest.raises(ValueError, match="next revision"):
        store.save_task(
            "session-1",
            replace(valid_brief, revision=4, title="Skipped revision"),
            TaskState.AWAITING_USER_APPROVAL,
        )


def test_latest_task_overviews_are_session_scoped_latest_and_activity_safe(
    tmp_path, valid_brief,
) -> None:
    ticks = iter(
        (
            "2026-08-10T12:00:00Z",
            "2026-08-10T12:00:01Z",
            "2026-08-10T12:00:02Z",
            "2026-08-10T12:00:03Z",
            "2026-08-10T12:00:04Z",
            "2026-08-10T12:00:05Z",
        )
    )
    store = SQLiteStore(tmp_path / "overview.sqlite3", clock=lambda: next(ticks))
    store.create_session("session-1", "/repo")
    store.create_session("session-2", "/other")
    store.create_planning_task("session-1", valid_brief.task_id)
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    second = replace(valid_brief, revision=2, title="Latest exact revision")
    store.save_task("session-1", second, TaskState.AWAITING_USER_APPROVAL)
    store.create_planning_task("session-2", "other-task")
    event = store.append_event(
        "session-1",
        valid_brief.task_id,
        "coordinator",
        "task_state",
        {"state": "awaiting_user_approval", "revision": 2},
    )

    overviews = store.latest_task_overviews("session-1")

    assert len(overviews) == 1
    assert overviews[0].task.brief == second
    assert overviews[0].updated_at == event.created_at
    assert overviews[0].active_agent is None
    assert overviews[0].active_started_at is None
    assert store.latest_task_overviews("session-2")[0].task.task_id == "other-task"


def test_task_overview_projects_only_latest_evidence_after_exact_revision_boundary(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task(
        "session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL,
    )
    store.append_event(
        "session-1", valid_brief.task_id, "sol", "outcome",
        {"summary": "must not cross the brief boundary"},
    )
    boundary = store.append_event(
        "session-1", valid_brief.task_id, "fable", "task_brief",
        {"brief": valid_brief.to_dict()},
    )
    expected = {
        "outcome": {"summary": "current outcome"},
        "review": {"summary": "current review"},
        "clarification": {"reasoning": "current clarification"},
        "agent_event": {"status": "current activity", "command_sha256": "safe-hash"},
    }
    for kind, payload in expected.items():
        store.append_event(
            "session-1", valid_brief.task_id,
            "sol" if kind in {"outcome", "agent_event"} else "fable",
            kind,
            payload,
        )

    overview = store.latest_task_overviews("session-1")[0]

    assert overview.revision_start_sequence == boundary.sequence
    assert overview.outcome == expected["outcome"]
    assert overview.review == expected["review"]
    assert overview.clarification == expected["clarification"]
    assert overview.activity == expected["agent_event"]
    plan = store._connection.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT payload_json FROM events
        WHERE session_id = ? AND task_id = ? AND kind = ? AND sequence > ?
        ORDER BY sequence DESC LIMIT 1
        """,
        ("session-1", valid_brief.task_id, "outcome", boundary.sequence),
    ).fetchall()
    assert any("events_session_task_kind_sequence" in row["detail"] for row in plan)


def test_latest_task_overviews_have_a_deterministic_hard_bound(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    for index in range(MAX_TASK_OVERVIEWS + 5):
        store.save_task(
            "session-1",
            replace(valid_brief, task_id=f"task-{index:03d}"),
            TaskState.AWAITING_USER_APPROVAL,
        )

    overviews = store.latest_task_overviews("session-1")

    assert len(overviews) == MAX_TASK_OVERVIEWS
    assert overviews[0].task.task_id == "task-000"
    assert overviews[-1].task.task_id == f"task-{MAX_TASK_OVERVIEWS - 1:03d}"


def test_browser_replay_is_paged_recent_and_backed_by_event_indexes(
    tmp_path,
) -> None:
    assert getattr(store_module, "EVENT_REPLAY_PAGE_SIZE", None) == 100
    assert getattr(store_module, "MAX_INITIAL_REPLAY_EVENTS", None) == 300
    event_replay_page_size = store_module.EVENT_REPLAY_PAGE_SIZE
    max_initial_replay_events = store_module.MAX_INITIAL_REPLAY_EVENTS
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.create_session("session-2", "/other")
    total = max_initial_replay_events + event_replay_page_size + 5
    events = []
    for index in range(total):
        events.append(
            store.append_event(
                "session-1", None, "coordinator", "message", {"text": str(index)}
            )
        )
        if index % 7 == 0:
            store.append_event(
                "session-2", None, "coordinator", "message", {"text": str(index)}
            )

    first_page = store.events_after(
        "session-1", 0, limit=event_replay_page_size
    )
    assert len(first_page) == event_replay_page_size
    assert first_page[0] == events[0]
    assert first_page[-1] == events[event_replay_page_size - 1]
    with pytest.raises(ValueError, match="limit"):
        store.events_after("session-1", 0, limit=0)
    with pytest.raises(ValueError, match="limit"):
        store.events_after("session-1", 0, limit=event_replay_page_size + 1)

    floor = store.browser_replay_floor("session-1")
    replayed = []
    cursor = floor
    while True:
        page = store.events_after(
            "session-1", cursor, limit=event_replay_page_size
        )
        replayed.extend(page)
        if len(page) < event_replay_page_size:
            break
        cursor = page[-1].sequence
    assert replayed == events[-max_initial_replay_events:]
    assert floor == events[-max_initial_replay_events - 1].sequence

    indexes = {
        row["name"]
        for row in store._connection.execute("PRAGMA index_list(events)").fetchall()
    }
    assert {
        "events_session_sequence",
        "events_session_task_sequence",
        "events_session_task_kind_sequence",
    } <= indexes
    plan = store._connection.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT * FROM events
        WHERE session_id = ? AND sequence > ?
        ORDER BY sequence LIMIT ?
        """,
        ("session-1", 0, event_replay_page_size),
    ).fetchall()
    assert any("events_session_sequence" in row["detail"] for row in plan)


def test_task_recency_uses_latest_indexed_event_not_maximum_timestamp(
    tmp_path,
    valid_brief,
) -> None:
    ticks = iter((
        "2026-08-10T00:00:00Z",
        "2026-08-10T00:00:59Z",
        "2026-08-10T00:00:01Z",
    ))
    store = SQLiteStore(tmp_path / "recency.sqlite3", clock=lambda: next(ticks))
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    other_brief = replace(valid_brief, task_id="task-2")
    store.save_task("session-1", other_brief, TaskState.AWAITING_USER_APPROVAL)
    older_sequence = store.append_event(
        "session-1", other_brief.task_id, "coordinator", "message", {"text": "first"}
    )
    latest = store.append_event(
        "session-1", valid_brief.task_id, "coordinator", "message", {"text": "second"}
    )

    overviews = store.latest_task_overviews("session-1")
    assert older_sequence.created_at > latest.created_at
    assert [overview.task.task_id for overview in overviews] == [
        valid_brief.task_id,
        other_brief.task_id,
    ]
    assert overviews[0].updated_at == latest.created_at


def test_approval_and_agent_sessions_bind_to_one_task_revision(tmp_path, valid_brief) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)

    approved = store.approve_task(valid_brief.task_id, 1, baseline_id="baseline-1")
    with_sessions = store.set_fable_session(valid_brief.task_id, 1, "fable-session-1")
    with_sessions = store.set_sol_thread(valid_brief.task_id, 1, "sol-thread-1")
    corrected = store.increment_correction_count(valid_brief.task_id, 1)
    corrected = store.increment_correction_count(valid_brief.task_id, 1)
    revised = replace(valid_brief, revision=2, title="Second revision")
    store.save_task("session-1", revised, TaskState.AWAITING_USER_APPROVAL)

    assert approved.approved_at == "2026-08-10T12:00:00Z"
    assert approved.baseline_id == "baseline-1"
    assert with_sessions.fable_session_id == "fable-session-1"
    assert with_sessions.sol_thread_id == "sol-thread-1"
    assert corrected.correction_count == 2
    latest = store.latest_task(valid_brief.task_id)
    assert latest is not None
    assert latest.revision == 2
    assert latest.approved_at is None
    assert latest.fable_session_id is None
    assert latest.sol_thread_id is None
    assert latest.correction_count == 0


@pytest.mark.parametrize("atomic", (False, True))
def test_approval_rejects_a_revision_that_is_no_longer_latest(
    tmp_path, valid_brief, atomic: bool,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task(
        "session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL
    )
    store.save_task(
        "session-1",
        replace(valid_brief, revision=2, title="New latest revision"),
        TaskState.AWAITING_USER_APPROVAL,
    )
    key = "agent_bridge.baseline.task-1.1"

    with pytest.raises(RuntimeError, match="concurrently|latest"):
        if atomic:
            store.approve_task_with_setting(
                valid_brief.task_id,
                1,
                brief=valid_brief,
                baseline_id="stale-baseline",
                expected=TaskState.AWAITING_USER_APPROVAL,
                setting=(key, {"baseline_id": "stale-baseline"}),
            )
        else:
            store.approve_task(
                valid_brief.task_id, 1, baseline_id="stale-baseline"
            )

    stale = store.get_task(valid_brief.task_id, 1)
    assert stale.approved_at is None
    assert stale.baseline_id is None
    assert store.get_setting(key) is None


def test_atomic_initial_approval_rolls_back_setting_when_task_update_fails(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task(
        "session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL
    )
    key = f"agent_bridge.baseline.{valid_brief.task_id}.{valid_brief.revision}"
    store._connection.execute(
        """
        CREATE TRIGGER fail_initial_approval
        BEFORE UPDATE OF baseline_id ON tasks
        WHEN OLD.task_id = 'task-1' AND OLD.revision = 1
        BEGIN
            SELECT RAISE(FAIL, 'injected approval update failure');
        END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="injected approval update failure"):
        store.approve_task_with_setting(
            valid_brief.task_id,
            valid_brief.revision,
            brief=valid_brief,
            baseline_id="baseline-1",
            expected=TaskState.AWAITING_USER_APPROVAL,
            setting=(key, {"baseline_id": "baseline-1"}),
        )

    current = store.get_task(valid_brief.task_id, valid_brief.revision)
    assert current.state is TaskState.AWAITING_USER_APPROVAL
    assert current.approved_at is None
    assert current.baseline_id is None
    assert store.get_setting(key) is None

    store._connection.execute("DROP TRIGGER fail_initial_approval")
    mismatched = replace(valid_brief, title="Changed after capture")
    with pytest.raises(RuntimeError, match="identity changed concurrently"):
        store.approve_task_with_setting(
            valid_brief.task_id,
            valid_brief.revision,
            brief=mismatched,
            baseline_id="baseline-1",
            expected=TaskState.AWAITING_USER_APPROVAL,
            setting=(key, {"baseline_id": "baseline-1"}),
        )
    assert store.get_setting(key) is None


def test_continuation_context_is_cleared_atomically_on_resume(tmp_path, valid_brief) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    store.transition_task(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.AWAITING_USER_APPROVAL,
        target=TaskState.SOL_RUNNING,
    )
    awaiting_answer = store.pause_for_continuation(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=TaskState.SOL_RUNNING,
        pending={"question": "Which state directory should be used?"},
    )

    assert awaiting_answer.continuation_state is TaskState.SOL_RUNNING
    assert awaiting_answer.pending == {"question": "Which state directory should be used?"}
    resumed = store.resume_continuation(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.AWAITING_USER_INPUT,
    )
    assert resumed.state is TaskState.SOL_RUNNING
    assert resumed.continuation_state is None
    assert resumed.pending is None


def test_direct_transition_cannot_bypass_continuation_cleanup(tmp_path, valid_brief) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    store.transition_task(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.AWAITING_USER_APPROVAL,
        target=TaskState.SOL_RUNNING,
    )
    store.pause_for_continuation(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=TaskState.SOL_RUNNING,
        pending={"question": "Clarify the state directory."},
    )

    with pytest.raises(RuntimeError, match="use resume_continuation"):
        store.transition_task(
            valid_brief.task_id,
            valid_brief.revision,
            expected=TaskState.AWAITING_USER_INPUT,
            target=TaskState.SOL_RUNNING,
        )


def test_terminal_failure_clears_paused_continuation_context(tmp_path, valid_brief) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    store.transition_task(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.AWAITING_USER_APPROVAL,
        target=TaskState.SOL_RUNNING,
    )
    store.pause_for_continuation(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=TaskState.SOL_RUNNING,
        pending={"question": "Clarify the state directory."},
    )

    failed = store.transition_task(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.AWAITING_USER_INPUT,
        target=TaskState.FAILED,
    )

    assert failed.state is TaskState.FAILED
    assert failed.continuation_state is None
    assert failed.pending is None


def test_interruption_requires_a_resumable_continuation_target(tmp_path, valid_brief) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    store.transition_task(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.AWAITING_USER_APPROVAL,
        target=TaskState.SOL_RUNNING,
    )

    with pytest.raises(ValueError, match="illegal task transition"):
        store.mark_interrupted(
            valid_brief.task_id,
            valid_brief.revision,
            continuation=TaskState.COMPLETED,
        )


def test_expired_fable_login_requires_exact_claimed_fable_run_and_rolls_back(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    prepared = store.prepare_new_request_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id="task-login",
        generation=7,
        payload=NewRequestPayload(text="Plan the bounded task."),
    )
    claimed = store.claim_prepared_action(prepared.preparation_id, generation=7)
    started = store.start_agent_run("run-login", "task-login", 0, "fable")
    task = store.get_task("task-login", 0)
    notice = ConversationEnvelope(
        sender=ConversationActor.SYSTEM,
        addressed_to=ConversationTarget.USER,
        routed_to=ConversationTarget.USER,
        message_type=ConversationMessageType.STATUS,
        text="Fable login expired. Run claude auth login on the host, then Resume.",
    )

    with pytest.raises(RuntimeError, match="pending context"):
        store.interrupt_fable_login_expired(
            session_id="session-1",
            task_id="task-login",
            revision=0,
            expected_state=TaskState.FABLE_PLANNING,
            expected_fable_session_id=None,
            expected_pending={"tampered": "context"},
            run_id=started.run_id,
            preparation_id=claimed.preparation_id,
            generation=claimed.generation,
            event=notice,
        )
    assert store.get_task("task-login", 0) == task
    assert store.prepared_action(claimed.preparation_id) == claimed
    assert store.agent_run(started.run_id) == started

    other = store.prepare_new_request_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id="other-login-task",
        generation=1,
        payload=NewRequestPayload(text="Plan another bounded task."),
    )
    store.claim_prepared_action(other.preparation_id, generation=other.generation)
    wrong_run = store.start_agent_run("run-other", "other-login-task", 0, "fable")
    with pytest.raises(RuntimeError, match="lifecycle changed"):
        store.interrupt_fable_login_expired(
            session_id="session-1",
            task_id="task-login",
            revision=0,
            expected_state=TaskState.FABLE_PLANNING,
            expected_fable_session_id=None,
            expected_pending=None,
            run_id=wrong_run.run_id,
            preparation_id=claimed.preparation_id,
            generation=claimed.generation,
            event=notice,
        )
    assert store.get_task("task-login", 0) == task
    assert store.prepared_action(claimed.preparation_id) == claimed
    assert store.agent_run(started.run_id) == started

    store._connection.execute(
        """
        CREATE TRIGGER fail_login_guidance_event
        BEFORE INSERT ON events WHEN NEW.kind = 'conversation'
        BEGIN
            SELECT RAISE(ABORT, 'injected login guidance failure');
        END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="injected login guidance failure"):
        store.interrupt_fable_login_expired(
            session_id="session-1",
            task_id="task-login",
            revision=0,
            expected_state=TaskState.FABLE_PLANNING,
            expected_fable_session_id=None,
            expected_pending=None,
            run_id=started.run_id,
            preparation_id=claimed.preparation_id,
            generation=claimed.generation,
            event=notice,
        )

    assert store.get_task("task-login", 0) == task
    assert store.prepared_action(claimed.preparation_id) == claimed
    assert store.agent_run(started.run_id) == started
    initial_events = store.events_after("session-1", 0)
    assert len(initial_events) == 2

    store._connection.execute("DROP TRIGGER fail_login_guidance_event")
    interrupted = store.interrupt_fable_login_expired(
        session_id="session-1",
        task_id="task-login",
        revision=0,
        expected_state=TaskState.FABLE_PLANNING,
        expected_fable_session_id=None,
        expected_pending=None,
        run_id=started.run_id,
        preparation_id=claimed.preparation_id,
        generation=claimed.generation,
        event=notice,
    )
    assert interrupted.state is TaskState.INTERRUPTED
    assert interrupted.continuation_state is TaskState.FABLE_PLANNING
    assert store.prepared_action(claimed.preparation_id).status == "INTERRUPTED"
    assert store.prepared_action(claimed.preparation_id).reason == "adapter_interrupted"
    assert store.agent_run(started.run_id).status == "interrupted"
    assert store.agent_run(started.run_id).exit_code == -1
    assert len(store.events_after("session-1", 0)) == len(initial_events) + 1

    assert store.interrupt_fable_login_expired(
        session_id="session-1",
        task_id="task-login",
        revision=0,
        expected_state=TaskState.FABLE_PLANNING,
        expected_fable_session_id=None,
        expected_pending=None,
        run_id=started.run_id,
        preparation_id=claimed.preparation_id,
        generation=claimed.generation,
        event=notice,
    ) == interrupted
    assert len(store.events_after("session-1", 0)) == len(initial_events) + 1

    with pytest.raises(ValueError, match="preparation_id"):
        store.interrupt_fable_login_expired(
            session_id="session-1",
            task_id="task-login",
            revision=0,
            expected_state=TaskState.FABLE_PLANNING,
            expected_fable_session_id=None,
            expected_pending=None,
            run_id=started.run_id,
            preparation_id=None,  # type: ignore[arg-type]
            generation=claimed.generation,
            event=notice,
        )


def test_revision_zero_task_cannot_enter_approval_without_a_brief(tmp_path, valid_brief) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.create_planning_task("session-1", valid_brief.task_id)

    with pytest.raises(RuntimeError, match="revision-zero task has no brief"):
        store.transition_task(
            valid_brief.task_id,
            0,
            expected=TaskState.FABLE_PLANNING,
            target=TaskState.AWAITING_USER_APPROVAL,
        )


def test_revision_zero_task_cannot_pause_into_approval_without_a_brief(tmp_path, valid_brief) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.create_planning_task("session-1", valid_brief.task_id)

    with pytest.raises(RuntimeError, match="revision-zero task has no brief"):
        store.pause_for_continuation(
            valid_brief.task_id,
            0,
            expected=TaskState.FABLE_PLANNING,
            target=TaskState.AWAITING_USER_APPROVAL,
            continuation_state=TaskState.SOL_RUNNING,
            pending={"approval": "must not exist without a brief"},
        )


def test_second_pause_preserves_the_original_continuation_context(tmp_path, valid_brief) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    store.transition_task(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.AWAITING_USER_APPROVAL,
        target=TaskState.SOL_RUNNING,
    )
    original = store.pause_for_continuation(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=TaskState.SOL_RUNNING,
        pending={"question": "Use the state directory outside the repository?"},
    )

    with pytest.raises(RuntimeError, match="continuation context already exists"):
        store.pause_for_continuation(
            valid_brief.task_id,
            valid_brief.revision,
            expected=TaskState.AWAITING_USER_INPUT,
            target=TaskState.FABLE_REVIEWING,
            continuation_state=TaskState.SOL_CORRECTING,
            pending={"review": "replacement context"},
        )

    assert store.get_task(valid_brief.task_id, valid_brief.revision) == original


def test_agent_run_records_one_terminal_exit_status_and_exact_active_run(tmp_path, valid_brief) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    started = store.start_agent_run("run-1", valid_brief.task_id, 1, "sol")
    with_process = store.set_agent_run_process(
        "run-1", pid=101, process_group_id=101, cli_session_id="codex-thread-1"
    )

    assert started.status == "running"
    assert started.started_at == "2026-08-10T12:00:00Z"
    assert store.active_run_for_task(valid_brief.task_id, 1) == with_process
    with pytest.raises(RuntimeError, match="active agent run"):
        store.start_agent_run("run-2", valid_brief.task_id, 1, "sol")

    finished = store.finish_agent_run("run-1", status="completed", exit_code=0)
    assert finished.status == "completed"
    assert finished.exit_code == 0
    assert finished.ended_at == "2026-08-10T12:00:00Z"
    assert store.active_run_for_task(valid_brief.task_id, 1) is None
    with pytest.raises(RuntimeError, match="agent run already finished"):
        store.finish_agent_run("run-1", status="completed", exit_code=0)


def test_startup_recovery_interrupts_active_tasks_and_stale_runs_without_signals(
    tmp_path, valid_brief, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    active_states = (
        TaskState.FABLE_PLANNING,
        TaskState.SOL_RUNNING,
        TaskState.FABLE_CLARIFYING,
        TaskState.FABLE_REVIEWING,
        TaskState.SOL_CORRECTING,
    )
    identities: list[tuple[str, int, TaskState]] = []
    for index, state in enumerate(active_states):
        task_id = f"active-{index}"
        if state is TaskState.FABLE_PLANNING:
            store.create_planning_task("session-1", task_id)
            revision = 0
        else:
            brief = replace(valid_brief, task_id=task_id, title=f"Active {index}")
            store.save_task("session-1", brief, state)
            revision = brief.revision
        identities.append((task_id, revision, state))

    paused = replace(valid_brief, task_id="paused", title="Paused")
    store.save_task("session-1", paused, TaskState.AWAITING_USER_INPUT)
    store.start_agent_run("active-run", "active-1", 1, "sol")
    stored_process = store.set_agent_run_process(
        "active-run",
        pid=424242,
        process_group_id=424242,
        cli_session_id="codex-thread-stale",
    )
    store.start_agent_run("inconsistent-run", paused.task_id, paused.revision, "sol")
    signal_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, signal: signal_calls.append(("kill", pid)))
    monkeypatch.setattr(
        os, "killpg", lambda pgid, signal: signal_calls.append(("killpg", pgid)),
    )

    recovered = store.recover_active_tasks()

    assert recovered == store_module.RecoverySummary(
        prepared_actions_recovered=0,
        tasks_interrupted=5,
        agent_runs_interrupted=2,
    )
    for task_id, revision, prior_state in identities:
        task = store.get_task(task_id, revision)
        assert task.state is TaskState.INTERRUPTED
        assert task.continuation_state is prior_state
    assert store.get_task(paused.task_id, paused.revision).state is TaskState.AWAITING_USER_INPUT
    recovered_run = store.agent_run("active-run")
    assert recovered_run.status == "interrupted"
    assert recovered_run.ended_at == "2026-08-10T12:00:00Z"
    assert recovered_run.exit_code is None
    assert recovered_run.pid == stored_process.pid
    assert recovered_run.process_group_id == stored_process.process_group_id
    assert store.agent_run("inconsistent-run").status == "interrupted"
    assert store.active_run_for_task("active-1", 1) is None
    assert signal_calls == []
    assert store.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)


def test_startup_recovery_rolls_back_task_and_run_retirement_together(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.start_agent_run("run-1", valid_brief.task_id, valid_brief.revision, "sol")
    store._connection.execute(
        """
        CREATE TRIGGER fail_startup_run_recovery
        BEFORE UPDATE OF status ON agent_runs
        WHEN OLD.run_id = 'run-1'
        BEGIN
            SELECT RAISE(FAIL, 'injected startup recovery failure');
        END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="injected startup recovery failure"):
        store.recover_active_tasks()

    assert store.get_task(valid_brief.task_id, valid_brief.revision).state is TaskState.SOL_RUNNING
    assert store.agent_run("run-1").status == "running"


def test_startup_recovery_interrupts_only_latest_active_task_revision(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    historical = store.create_planning_task("session-1", valid_brief.task_id)
    store.start_agent_run("finished-plan", historical.task_id, 0, "fable")
    store.finish_agent_run("finished-plan", status="completed", exit_code=0)
    latest = store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.start_agent_run("active-sol", latest.task_id, latest.revision, "sol")

    recovered = store.recover_active_tasks()

    assert recovered == store_module.RecoverySummary(
        prepared_actions_recovered=0,
        tasks_interrupted=1,
        agent_runs_interrupted=1,
    )
    assert store.get_task(historical.task_id, historical.revision).state is (
        TaskState.FABLE_PLANNING
    )
    interrupted = store.get_task(latest.task_id, latest.revision)
    assert interrupted.state is TaskState.INTERRUPTED
    assert interrupted.continuation_state is TaskState.SOL_RUNNING
    assert store.agent_run("finished-plan").status == "completed"
    assert store.agent_run("active-sol").status == "interrupted"
    assert store.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)


def test_recovery_summary_counts_only_this_calls_transitions(tmp_path, valid_brief) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    prepared = tuple(
        store.prepare_new_request_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id=f"prepared-{index}",
            generation=index,
            payload=NewRequestPayload(text=f"prepared request {index}"),
        )
        for index in range(2)
    )

    assert store.recover_unfinished_prepared_actions() == store_module.RecoverySummary(
        prepared_actions_recovered=2,
        tasks_interrupted=2,
        agent_runs_interrupted=0,
    )
    for record in prepared:
        assert store.prepared_action(record.preparation_id).status == "RECOVERED"
        assert store.get_task(record.task_id, record.revision).state is TaskState.INTERRUPTED

    for index in range(3):
        store.create_planning_task("session-1", f"active-{index}")
        store.start_agent_run(f"active-run-{index}", f"active-{index}", 0, "fable")
    paused = replace(valid_brief, task_id="paused", title="Paused")
    store.save_task("session-1", paused, TaskState.AWAITING_USER_APPROVAL)
    store.start_agent_run("inactive-run", "paused", paused.revision, "fable")

    assert store.recover_active_tasks() == store_module.RecoverySummary(
        prepared_actions_recovered=0,
        tasks_interrupted=3,
        agent_runs_interrupted=4,
    )
    for index in range(3):
        assert store.get_task(f"active-{index}", 0).state is TaskState.INTERRUPTED
        assert store.agent_run(f"active-run-{index}").status == "interrupted"
    assert store.agent_run("inactive-run").status == "interrupted"
    assert store.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)


def test_recovery_summary_is_immutable_and_validated() -> None:
    summary_type = store_module.RecoverySummary
    summary = summary_type(1, 2, 3)

    assert summary.prepared_actions_recovered == 1
    assert summary.tasks_interrupted == 2
    assert summary.agent_runs_interrupted == 3
    with pytest.raises(ValueError, match="prepared_actions_recovered must be a non-negative integer"):
        summary_type(True, 0, 0)
    with pytest.raises(ValueError, match="tasks_interrupted must be a non-negative integer"):
        summary_type(0, 1.5, 0)
    with pytest.raises(ValueError, match="agent_runs_interrupted must be a non-negative integer"):
        summary_type(0, 0, -1)


def test_active_recoverable_rows_are_bounded(tmp_path) -> None:
    """Active recovery retains no Python-sized record collection after batching."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    recoverable_count = 4_000
    store._connection.executemany(
        """
        INSERT INTO tasks (task_id, revision, session_id, state, correction_count)
        VALUES (?, 0, 'session-1', ?, 0)
        """,
        (
            (f"active-bounded-{index:05d}", TaskState.FABLE_PLANNING.value)
            for index in range(recoverable_count)
        ),
    )

    tracemalloc.start()
    try:
        recovered = store.recover_active_tasks()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1_000_000
    assert recovered == store_module.RecoverySummary(0, recoverable_count, 0)
    assert store.get_task("active-bounded-00000", 0).state is TaskState.INTERRUPTED
    assert store.get_task("active-bounded-03999", 0).state is TaskState.INTERRUPTED


def test_session_repository_lookup_is_exact_and_absent_safe(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo/exact")

    assert store.session_repo_root("session-1") == "/repo/exact"
    assert store.session_repo_root("missing") is None


def test_usage_credit_acknowledgement_setting_persists_across_connections(tmp_path) -> None:
    path = tmp_path / "bridge.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    assert store.get_setting("usage_credits_acknowledged") is None
    store.set_setting("usage_credits_acknowledged", True)
    store.close()

    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    assert reopened.get_setting("usage_credits_acknowledged") is True


def _create_current_schema(path) -> sqlite3.Connection:
    """Create the schema released before chats became persistent records."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            repo_root TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE tasks (
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
        );
        CREATE TABLE events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            task_id TEXT,
            actor TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        CREATE TABLE agent_runs (
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
        );
        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        """
    )
    return connection


def _legacy_table_rows(connection: sqlite3.Connection) -> dict[str, tuple[tuple, ...]]:
    def rows(query: str) -> tuple[tuple, ...]:
        return tuple(tuple(row) for row in connection.execute(query))

    return {
        "sessions": rows(
            "SELECT session_id, repo_root, created_at FROM sessions ORDER BY session_id"
        ),
        "tasks": rows(
            """
            SELECT task_id, revision, session_id, state, brief_json, approved_at,
                   fable_session_id, sol_thread_id, baseline_id, correction_count,
                   continuation_state, pending_json
            FROM tasks ORDER BY task_id, revision
            """
        ),
        "events": rows("SELECT * FROM events ORDER BY sequence"),
        "agent_runs": rows("SELECT * FROM agent_runs ORDER BY run_id"),
        "settings": rows("SELECT * FROM settings ORDER BY key"),
    }


def _seed_legacy_database(path) -> dict[str, tuple[tuple, ...]]:
    connection = _create_current_schema(path)
    connection.executemany(
        "INSERT INTO sessions (session_id, repo_root, created_at) VALUES (?, ?, ?)",
        (
            ("active-session", "/repo", "2026-08-10T10:00:00Z"),
            ("other-session", "/repo", "2026-08-10T10:01:00Z"),
        ),
    )
    connection.execute(
        """
        INSERT INTO tasks (
            task_id, revision, session_id, state, brief_json, approved_at,
            fable_session_id, sol_thread_id, baseline_id, correction_count,
            continuation_state, pending_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task.legacy",
            1,
            "active-session",
            "awaiting_user_approval",
            '{"raw":"brief bytes"}',
            None,
            None,
            None,
            "baseline-legacy",
            0,
            None,
            '{"raw":"pending bytes"}',
        ),
    )
    connection.execute(
        """
        INSERT INTO events (session_id, task_id, actor, kind, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "other-session",
            None,
            "user",
            "message",
            '{"text":"preserved event bytes"}',
            "2026-08-10T10:02:00Z",
        ),
    )
    connection.execute(
        """
        INSERT INTO agent_runs (
            run_id, task_id, revision, agent, pid, process_group_id,
            cli_session_id, started_at, ended_at, exit_code, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run-legacy",
            "task.legacy",
            1,
            "sol",
            123,
            123,
            "thread-legacy",
            "2026-08-10T10:03:00Z",
            None,
            None,
            "running",
        ),
    )
    baseline = {
        "task_id": "task.legacy",
        "revision": 1,
        "baseline_id": "baseline-legacy",
        "manifest": {"baseline_id": "baseline-legacy", "repo_root": "/repo"},
    }
    connection.executemany(
        "INSERT INTO settings (key, value_json) VALUES (?, ?)",
        (
            ("agent_bridge.active_session_id", '"active-session"'),
            (
                "agent_bridge.baseline.task.legacy.1",
                json.dumps(baseline, separators=(",", ":"), sort_keys=True),
            ),
        ),
    )
    connection.commit()
    rows = _legacy_table_rows(connection)
    connection.close()
    return rows


def test_migration_preserves_legacy_bytes_and_backfills_bounded_chat_metadata(
    tmp_path,
) -> None:
    path = tmp_path / "legacy.sqlite3"
    before = _seed_legacy_database(path)

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    assert _legacy_table_rows(migrated._connection) == before
    active = migrated.chat("active-session")
    other = migrated.chat("other-session")
    assert active is not None
    assert active.title == "New chat"
    assert active.updated_at == "2026-08-10T10:00:00Z"
    assert active.latest_sequence == 0
    assert other is not None
    assert other.title == "New chat"
    assert other.updated_at == "2026-08-10T10:01:00Z"
    assert other.latest_sequence == 1
    selected = migrated.get_setting("agent_bridge.active_session_id")
    assert selected == "active-session"
    assert migrated.chat(selected) == active
    migrated.close()

    first_migration_bytes = path.read_bytes()
    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    assert _legacy_table_rows(reopened._connection) == before
    reopened.close()
    assert path.read_bytes() == first_migration_bytes


def test_chat_migration_rolls_back_every_ddl_change_after_backfill_failure(tmp_path) -> None:
    path = tmp_path / "rollback.sqlite3"
    connection = _create_current_schema(path)
    connection.execute(
        "INSERT INTO sessions (session_id, repo_root, created_at) VALUES (?, ?, ?)",
        ("session-1", "/repo", "2026-08-10T10:00:00Z"),
    )
    connection.execute(
        """
        CREATE TRIGGER fail_chat_backfill
        BEFORE UPDATE OF updated_at ON sessions
        BEGIN
            SELECT RAISE(ABORT, 'injected chat migration failure');
        END
        """
    )
    connection.commit()
    connection.close()

    with pytest.raises(sqlite3.IntegrityError, match="injected chat migration failure"):
        SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    inspected = sqlite3.connect(path)
    columns = tuple(row[1] for row in inspected.execute("PRAGMA table_info(sessions)"))
    assert columns == ("session_id", "repo_root", "created_at")
    assert inspected.execute("SELECT * FROM sessions").fetchall() == [
        ("session-1", "/repo", "2026-08-10T10:00:00Z")
    ]
    inspected.close()


def test_create_chat_uses_unique_cryptographic_ids_and_preserves_session_wrapper(
    tmp_path,
) -> None:
    store = _store(tmp_path)

    first = store.create_chat("/repo/one")
    second = store.create_chat("/repo/two")
    legacy_return = store.create_session("legacy-session", "/repo/legacy")

    assert first.session_id != second.session_id
    assert len(first.session_id) == 32
    assert set(first.session_id) <= set("0123456789abcdef")
    assert first.repo_root == "/repo/one"
    assert second.repo_root == "/repo/two"
    assert first.title == second.title == "New chat"
    assert first.latest_sequence == second.latest_sequence == 0
    assert legacy_return is None
    assert store.chat("legacy-session") is not None
    assert store.chat("legacy-session").repo_root == "/repo/legacy"  # type: ignore[union-attr]


def test_first_user_message_derives_the_only_bounded_unicode_chat_title(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_chat("/repo", session_id="chat-1")
    first_text = "  café\t" + ("文" * 100)

    store.append_event("chat-1", None, "fable", "message", {"text": "ignore me"})
    store.append_event("chat-1", None, "system", "message", {"text": "ignore me too"})
    store.append_event("chat-1", None, "user", "message", {"text": first_text})
    titled = store.chat("chat-1")
    store.append_event(
        "chat-1", None, "user", "message", {"text": "later user text cannot rename"}
    )

    assert titled is not None
    assert titled.title == "café " + ("文" * 75)
    assert len(titled.title) == 80
    assert store.chat("chat-1").title == titled.title  # type: ignore[union-attr]


def test_first_validated_user_conversation_statement_derives_the_only_chat_title(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    store.create_chat("/repo", session_id="chat-1")
    statement = ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.FABLE,
        message_type=ConversationMessageType.STATEMENT,
        text="  Plan this directed conversation.  ",
    )
    status = ConversationEnvelope(
        sender=ConversationActor.SYSTEM,
        addressed_to=ConversationTarget.USER,
        routed_to=ConversationTarget.USER,
        message_type=ConversationMessageType.STATUS,
        text="A bounded status.",
    )
    store.append_event("chat-1", None, "user", "conversation", status.to_dict())
    store.append_event("chat-1", None, "user", "conversation", statement.to_dict())
    titled = store.chat("chat-1")
    store.append_event(
        "chat-1", None, "user", "conversation", ConversationEnvelope(
            sender=ConversationActor.USER,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.STATEMENT,
            text="Later user text cannot rename this chat.",
        ).to_dict(),
    )

    assert titled is not None
    assert titled.title == "Plan this directed conversation."
    assert store.chat("chat-1").title == titled.title  # type: ignore[union-attr]


def test_first_legacy_user_text_equal_to_default_title_still_locks_chat_title(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_chat("/repo", session_id="chat-1")

    store.append_event("chat-1", None, "user", "message", {"text": "New chat"})
    store.append_event(
        "chat-1", None, "user", "message", {"text": "Later text cannot rename this."},
    )

    assert store.chat("chat-1").title == "New chat"  # type: ignore[union-attr]
    assert store._connection.execute(  # noqa: SLF001 - durable title marker contract
        "SELECT title_initialized FROM sessions WHERE session_id = ?", ("chat-1",)
    ).fetchone()[0] == 1


def test_first_conversation_text_equal_to_default_title_still_locks_chat_title(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_chat("/repo", session_id="chat-1")
    first = ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.FABLE,
        message_type=ConversationMessageType.STATEMENT,
        text="New chat",
    )
    later = ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        message_type=ConversationMessageType.STATEMENT,
        text="Later text cannot rename this.",
    )

    store.append_event("chat-1", None, "user", "conversation", first.to_dict())
    store.append_event("chat-1", None, "user", "conversation", later.to_dict())

    assert store.chat("chat-1").title == "New chat"  # type: ignore[union-attr]
    assert store._connection.execute(  # noqa: SLF001 - durable title marker contract
        "SELECT title_initialized FROM sessions WHERE session_id = ?", ("chat-1",)
    ).fetchone()[0] == 1


def test_concurrent_first_eligible_titles_commit_one_durable_title_marker(tmp_path) -> None:
    path = tmp_path / "concurrent-title.sqlite3"
    initial = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    initial.create_chat("/repo", session_id="chat-1")
    initial.close()

    first = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z", check_same_thread=False)
    second = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z", check_same_thread=False)
    barrier = threading.Barrier(2)
    failures: list[BaseException] = []

    def append_first_title(store: SQLiteStore, text: str) -> None:
        try:
            barrier.wait(timeout=2)
            store.append_event("chat-1", None, "user", "message", {"text": text})
        except BaseException as error:
            failures.append(error)

    workers = [
        threading.Thread(target=append_first_title, args=(first, "First one")),
        threading.Thread(target=append_first_title, args=(second, "First two")),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)

    assert all(worker.is_alive() is False for worker in workers)
    assert failures == []
    assert first.chat("chat-1").title in {"First one", "First two"}  # type: ignore[union-attr]
    assert first._connection.execute(  # noqa: SLF001 - durable title marker contract
        "SELECT title_initialized FROM sessions WHERE session_id = ?", ("chat-1",)
    ).fetchone()[0] == 1
    first.close()
    second.close()


def test_title_marker_migration_preserves_existing_first_titles_across_reopen(tmp_path) -> None:
    path = tmp_path / "pre-title-marker.sqlite3"
    connection = _create_current_schema(path)
    connection.executescript(
        """
        ALTER TABLE sessions ADD COLUMN title TEXT NOT NULL DEFAULT 'New chat';
        ALTER TABLE sessions ADD COLUMN updated_at TEXT;
        """
    )
    connection.executemany(
        """
        INSERT INTO sessions (session_id, repo_root, created_at, title, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            ("named", "/repo", "2026-08-10T10:00:00Z", "Existing title", "2026-08-10T10:00:00Z"),
            ("legacy", "/repo", "2026-08-10T10:00:00Z", "New chat", "2026-08-10T10:00:00Z"),
            ("conversation", "/repo", "2026-08-10T10:00:00Z", "New chat", "2026-08-10T10:00:00Z"),
            ("empty", "/repo", "2026-08-10T10:00:00Z", "New chat", "2026-08-10T10:00:00Z"),
        ),
    )
    statement = ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.TEAM,
        routed_to=ConversationTarget.FABLE,
        message_type=ConversationMessageType.STATEMENT,
        text="New chat",
    )
    connection.executemany(
        """
        INSERT INTO events (session_id, task_id, actor, kind, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            ("legacy", None, "user", "message", '{"text":"New chat"}', "2026-08-10T10:01:00Z"),
            ("conversation", None, "user", "conversation", json.dumps(statement.to_dict()), "2026-08-10T10:01:00Z"),
        ),
    )
    connection.commit()
    connection.close()

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    assert tuple(tuple(row) for row in migrated._connection.execute(  # noqa: SLF001 - schema migration contract
        "SELECT session_id, title_initialized FROM sessions ORDER BY session_id"
    )) == (("conversation", 1), ("empty", 0), ("legacy", 1), ("named", 1))
    migrated.close()

    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    reopened.append_event("legacy", None, "user", "message", {"text": "Do not rename"})
    reopened.append_event(
        "conversation", None, "user", "message", {"text": "Do not rename either"},
    )
    reopened.append_event("empty", None, "user", "message", {"text": "First later title"})

    assert reopened.chat("legacy").title == "New chat"  # type: ignore[union-attr]
    assert reopened.chat("conversation").title == "New chat"  # type: ignore[union-attr]
    assert reopened.chat("empty").title == "First later title"  # type: ignore[union-attr]
    assert tuple(tuple(row) for row in reopened._connection.execute(  # noqa: SLF001 - migration idempotency contract
        "SELECT session_id, title_initialized FROM sessions ORDER BY session_id"
    )) == (("conversation", 1), ("empty", 1), ("legacy", 1), ("named", 1))
    reopened.close()


def test_title_marker_migration_reads_legacy_history_in_bounded_pages(tmp_path, monkeypatch) -> None:
    path = tmp_path / "bounded-title-marker.sqlite3"
    connection = _create_current_schema(path)
    connection.executescript(
        """
        ALTER TABLE sessions ADD COLUMN title TEXT NOT NULL DEFAULT 'New chat';
        ALTER TABLE sessions ADD COLUMN updated_at TEXT;
        """
    )
    connection.execute(
        """
        INSERT INTO sessions (session_id, repo_root, created_at, title, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("chat-1", "/repo", "2026-08-10T10:00:00Z", "New chat", "2026-08-10T10:00:00Z"),
    )
    connection.executemany(
        """
        INSERT INTO events (session_id, task_id, actor, kind, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        tuple(
            ("chat-1", None, "fable", "status", '{"index":%d}' % index, "2026-08-10T10:01:00Z")
            for index in range(257)
        ),
    )
    connection.commit()
    statements: list[str] = []
    connection.close()

    original_connect = store_module.sqlite3.connect

    def observed_connect(*args, **kwargs):
        observed = original_connect(*args, **kwargs)
        observed.set_trace_callback(statements.append)
        return observed

    monkeypatch.setattr(store_module.sqlite3, "connect", observed_connect)

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    history_reads = [
        statement for statement in statements
        if "FROM events" in statement and "WHERE sequence >" in statement
    ]
    assert len(history_reads) == 4
    assert all("LIMIT 128" in statement for statement in history_reads)
    assert migrated._connection.execute(  # noqa: SLF001 - migration behavior contract
        "SELECT title_initialized FROM sessions WHERE session_id = ?", ("chat-1",)
    ).fetchone()[0] == 0
    migrated.close()


def test_title_marker_migration_rolls_back_its_additive_column_on_backfill_failure(tmp_path) -> None:
    path = tmp_path / "title-marker-rollback.sqlite3"
    connection = _create_current_schema(path)
    connection.executescript(
        """
        ALTER TABLE sessions ADD COLUMN title TEXT NOT NULL DEFAULT 'New chat';
        ALTER TABLE sessions ADD COLUMN updated_at TEXT;
        """
    )
    connection.execute(
        """
        INSERT INTO sessions (session_id, repo_root, created_at, title, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("chat-1", "/repo", "2026-08-10T10:00:00Z", "New chat", "2026-08-10T10:00:00Z"),
    )
    connection.execute(
        """
        INSERT INTO events (session_id, task_id, actor, kind, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("chat-1", None, "user", "message", '{"text":"first"}', "2026-08-10T10:01:00Z"),
    )
    connection.execute(
        """
        CREATE TRIGGER fail_title_marker_backfill
        BEFORE UPDATE OF title_initialized ON sessions
        BEGIN
            SELECT RAISE(ABORT, 'injected title marker migration failure');
        END
        """
    )
    connection.commit()
    connection.close()

    with pytest.raises(sqlite3.IntegrityError, match="injected title marker migration failure"):
        SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    inspected = sqlite3.connect(path)
    assert "title_initialized" not in tuple(
        row[1] for row in inspected.execute("PRAGMA table_info(sessions)")
    )
    inspected.close()


def test_first_conversation_title_never_scans_prior_event_history(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_chat("/repo", session_id="chat-1")
    for index in range(256):
        store.append_event(
            "chat-1", None, "fable", "message", {"text": f"agent {index}"},
        )
    statements: list[str] = []
    store._connection.set_trace_callback(statements.append)  # noqa: SLF001 - query budget evidence
    try:
        store.append_event(
            "chat-1", None, "user", "conversation", ConversationEnvelope(
                sender=ConversationActor.USER,
                addressed_to=ConversationTarget.SOL,
                routed_to=ConversationTarget.FABLE,
                message_type=ConversationMessageType.STATEMENT,
                text="Constant-time title.",
            ).to_dict(),
        )
    finally:
        store._connection.set_trace_callback(None)  # noqa: SLF001 - test cleanup

    assert store.chat("chat-1").title == "Constant-time title."  # type: ignore[union-attr]
    assert not any(
        "SELECT actor, kind, payload_json FROM events" in statement
        for statement in statements
    )


def test_chat_lists_use_event_sequence_recency_and_stable_cursor_pagination(tmp_path) -> None:
    store = _store(tmp_path)
    for session_id in ("active-a", "active-b", *(f"empty-{index:03d}" for index in range(103))):
        store.create_chat("/repo", session_id=session_id)
    older = store.append_event("active-a", None, "user", "message", {"text": "older"})
    newer = store.append_event("active-b", None, "user", "message", {"text": "newer"})

    all_records = store.list_chats(limit=50)
    assert [record.session_id for record in all_records[:2]] == ["active-b", "active-a"]
    assert [record.latest_sequence for record in all_records[:2]] == [
        newer.sequence,
        older.sequence,
    ]
    assert [record.session_id for record in all_records[2:]] == [
        f"empty-{index:03d}" for index in range(48)
    ]

    seen = []
    before = None
    while True:
        page = store.list_chats(before=before, limit=50)
        seen.extend(record.session_id for record in page)
        if len(page) < 50:
            break
        last = page[-1]
        before = store_module.ChatCursor(last.latest_sequence, last.session_id)

    assert seen == ["active-b", "active-a", *(f"empty-{index:03d}" for index in range(103))]
    assert len(seen) == len(set(seen))


def test_chat_page_inputs_reject_invalid_or_partial_cursors_before_sql(tmp_path) -> None:
    store = _store(tmp_path)
    statements: list[str] = []
    store._connection.set_trace_callback(statements.append)

    with pytest.raises(ValueError, match="limit"):
        store.list_chats(limit=0)
    with pytest.raises(ValueError, match="limit"):
        store.list_chats(limit=51)
    with pytest.raises(ValueError, match="latest_sequence"):
        store_module.ChatCursor(-1, "chat-1")
    with pytest.raises(ValueError, match="session_id"):
        store_module.ChatCursor(1, "")
    with pytest.raises(ValueError, match="before"):
        store.list_chats(before=object())

    assert statements == []


def _seed_auditable_legacy_database(path) -> None:
    connection = _create_current_schema(path)
    connection.execute(
        "INSERT INTO sessions (session_id, repo_root, created_at) VALUES (?, ?, ?)",
        ("session-1", "/repo", "2026-08-10T10:00:00Z"),
    )
    connection.execute(
        """
        INSERT INTO tasks (
            task_id, revision, session_id, state, brief_json, approved_at,
            fable_session_id, sol_thread_id, baseline_id, correction_count,
            continuation_state, pending_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task-1",
            1,
            "session-1",
            "awaiting_user_approval",
            "{}",
            None,
            None,
            None,
            "baseline-1",
            0,
            None,
            None,
        ),
    )
    connection.execute(
        """
        INSERT INTO events (session_id, task_id, actor, kind, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("session-1", "task-1", "user", "message", '{"text":"hello"}', "2026-08-10T10:01:00Z"),
    )
    connection.execute(
        """
        INSERT INTO agent_runs (
            run_id, task_id, revision, agent, pid, process_group_id,
            cli_session_id, started_at, ended_at, exit_code, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("run-1", "task-1", 1, "sol", None, None, None, "2026-08-10T10:02:00Z", None, None, "running"),
    )
    baseline = {
        "task_id": "task-1",
        "revision": 1,
        "baseline_id": "baseline-1",
        "manifest": {"baseline_id": "baseline-1", "repo_root": "/repo"},
    }
    connection.executemany(
        "INSERT INTO settings (key, value_json) VALUES (?, ?)",
        (
            ("agent_bridge.active_session_id", '"session-1"'),
            (
                "agent_bridge.baseline.task-1.1",
                json.dumps(baseline, separators=(",", ":"), sort_keys=True),
            ),
        ),
    )
    connection.commit()
    connection.close()


def test_legacy_project_ownership_audit_is_one_read_transaction(tmp_path) -> None:
    path = tmp_path / "audit.sqlite3"
    _seed_auditable_legacy_database(path)
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    statements: list[str] = []
    store._connection.set_trace_callback(statements.append)

    assert store.audit_legacy_project_ownership("/repo") is None

    assert statements[0] == "BEGIN"
    assert statements[-1] == "ROLLBACK"
    assert not any(statement.split(maxsplit=1)[0] in {"INSERT", "UPDATE", "DELETE"} for statement in statements)


@pytest.mark.parametrize(
    "corrupt",
    (
        pytest.param(
            lambda connection: connection.execute(
                "INSERT INTO sessions (session_id, repo_root, created_at) VALUES ('other-session', '/other', '2026-08-10T10:00:00Z')"
            ),
            id="mixed_session_roots",
        ),
        lambda connection: connection.execute(
            "UPDATE sessions SET repo_root = '/other' WHERE session_id = 'session-1'"
        ),
        lambda connection: connection.execute(
            "UPDATE sessions SET repo_root = '' WHERE session_id = 'session-1'"
        ),
        lambda connection: connection.execute(
            "UPDATE settings SET value_json = '\"missing-session\"' WHERE key = 'agent_bridge.active_session_id'"
        ),
        lambda connection: connection.execute(
            "INSERT INTO tasks (task_id, revision, session_id, state, correction_count) VALUES ('orphan-task', 1, 'missing-session', 'fable_planning', 0)"
        ),
        lambda connection: connection.execute(
            "INSERT INTO events (session_id, task_id, actor, kind, payload_json, created_at) VALUES ('missing-session', NULL, 'user', 'message', '{}', '2026-08-10T10:03:00Z')"
        ),
        pytest.param(
            lambda connection: connection.execute(
                "INSERT INTO events (session_id, task_id, actor, kind, payload_json, created_at) VALUES ('session-1', 'missing-task', 'user', 'message', '{}', '2026-08-10T10:03:00Z')"
            ),
            id="valid_session_event_with_missing_task",
        ),
        lambda connection: connection.execute(
            "INSERT INTO agent_runs (run_id, task_id, revision, agent, started_at, status) VALUES ('orphan-run', 'missing-task', 1, 'sol', '2026-08-10T10:03:00Z', 'running')"
        ),
        lambda connection: connection.execute(
            "INSERT INTO agent_runs (run_id, task_id, revision, agent, started_at, status) VALUES ('wrong-revision', 'task-1', 2, 'sol', '2026-08-10T10:03:00Z', 'running')"
        ),
        lambda connection: connection.execute(
            "INSERT INTO settings (key, value_json) VALUES ('agent_bridge.baseline.', '{}')"
        ),
        pytest.param(
            lambda connection: connection.execute(
                "UPDATE settings SET value_json = 'not-json' WHERE key = 'agent_bridge.baseline.task-1.1'"
            ),
            id="malformed_baseline_json",
        ),
        pytest.param(
            lambda connection: connection.execute(
                "UPDATE settings SET value_json = '[]' WHERE key = 'agent_bridge.baseline.task-1.1'"
            ),
            id="nonobject_baseline_payload",
        ),
        pytest.param(
            lambda connection: connection.execute(
                "UPDATE settings SET value_json = '{\"task_id\":\"other-task\",\"revision\":1,\"baseline_id\":\"baseline-1\",\"manifest\":{\"baseline_id\":\"baseline-1\",\"repo_root\":\"/repo\"}}' WHERE key = 'agent_bridge.baseline.task-1.1'"
            ),
            id="baseline_task_mismatch",
        ),
        lambda connection: connection.execute(
            "UPDATE settings SET value_json = '{\"task_id\":\"task-1\",\"revision\":2,\"baseline_id\":\"baseline-1\",\"manifest\":{\"baseline_id\":\"baseline-1\",\"repo_root\":\"/repo\"}}' WHERE key = 'agent_bridge.baseline.task-1.1'"
        ),
        lambda connection: connection.execute(
            "UPDATE settings SET value_json = '{\"task_id\":\"task-1\",\"revision\":1,\"baseline_id\":\"baseline-1\",\"manifest\":{\"baseline_id\":\"baseline-1\",\"repo_root\":\"/other\"}}' WHERE key = 'agent_bridge.baseline.task-1.1'"
        ),
    ),
)
def test_legacy_project_ownership_audit_fails_closed_for_corrupt_relationships(
    tmp_path, corrupt,
) -> None:
    path = tmp_path / "corrupt.sqlite3"
    _seed_auditable_legacy_database(path)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = OFF")
    corrupt(connection)
    connection.commit()
    connection.close()
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    with pytest.raises(RuntimeError, match="legacy project ownership audit failed") as error:
        store.audit_legacy_project_ownership("/repo")

    message = str(error.value)
    assert "/repo" not in message
    assert "session-1" not in message
    assert "task-1" not in message


def test_legacy_project_ownership_audit_rejects_a_task_whose_first_revision_is_two(
    tmp_path,
) -> None:
    path = tmp_path / "late-revision.sqlite3"
    _seed_auditable_legacy_database(path)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        INSERT INTO tasks (task_id, revision, session_id, state, correction_count)
        VALUES ('late-task', 2, 'session-1', 'awaiting_user_approval', 0)
        """
    )
    connection.commit()
    connection.close()
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    with pytest.raises(RuntimeError, match="legacy project ownership audit failed") as error:
        store.audit_legacy_project_ownership("/repo")

    assert "task_revision_integrity" in str(error.value)


def test_legacy_project_ownership_audit_aggregates_generic_reasons_without_mutation(
    tmp_path,
) -> None:
    path = tmp_path / "aggregate.sqlite3"
    _seed_auditable_legacy_database(path)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("UPDATE sessions SET repo_root = '/outside'")
    connection.execute(
        "UPDATE settings SET value_json = '\"missing-session\"' WHERE key = 'agent_bridge.active_session_id'"
    )
    connection.execute(
        "INSERT INTO agent_runs (run_id, task_id, revision, agent, started_at, status) VALUES ('orphan-run', 'missing-task', 1, 'sol', '2026-08-10T10:03:00Z', 'running')"
    )
    connection.commit()
    connection.close()
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    before = _legacy_table_rows(store._connection)

    with pytest.raises(RuntimeError, match="legacy project ownership audit failed") as error:
        store.audit_legacy_project_ownership("/repo")

    assert str(error.value).count(",") <= 7
    assert "/outside" not in str(error.value)
    assert _legacy_table_rows(store._connection) == before


def test_legacy_audit_rejects_a_foreign_prepared_action_before_recovery(
    tmp_path,
    valid_brief,
) -> None:
    """A prepared row can later drive recovery, so it must bind to this root."""
    repo = tmp_path / "repo"
    repo.mkdir()
    store = _store(tmp_path)
    store.create_session("session-1", str(repo))
    store.set_setting("agent_bridge.active_session_id", "session-1")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    baseline_setting = BaselineSetting(
        key="agent_bridge.baseline.task-1.1",
        value_json=json.dumps({
            "task_id": "task-1",
            "revision": 1,
            "baseline_id": "baseline-1",
            "manifest": {"baseline_id": "baseline-1", "repo_root": str(repo)},
        }, separators=(",", ":"), sort_keys=True),
    )
    prepared = store.prepare_approval_action(
        project_id=project_id_for_root(repo.resolve()),
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        generation=7,
        payload=ApprovalPayload("baseline-1", baseline_setting, None),
    )
    assert store.audit_legacy_project_ownership(str(repo.resolve())) is None
    store._connection.execute(
        "UPDATE prepared_actions SET project_id = ? WHERE preparation_id = ?",
        ("foreign-project", prepared.preparation_id),
    )

    with pytest.raises(RuntimeError, match="legacy project ownership audit failed"):
        store.audit_legacy_project_ownership(str(repo.resolve()))

    assert store.get_task(valid_brief.task_id, valid_brief.revision).state is TaskState.SOL_RUNNING


def test_legacy_audit_rejects_an_inconsistent_prepared_action_lineage(
    tmp_path,
    valid_brief,
) -> None:
    """A row that cannot be an exact resume lineage must not survive adoption."""
    repo = tmp_path / "repo"
    repo.mkdir()
    store = _store(tmp_path)
    store.create_session("session-1", str(repo))
    store.set_setting("agent_bridge.active_session_id", "session-1")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    baseline_setting = BaselineSetting(
        key="agent_bridge.baseline.task-1.1",
        value_json=json.dumps({
            "task_id": "task-1",
            "revision": 1,
            "baseline_id": "baseline-1",
            "manifest": {"baseline_id": "baseline-1", "repo_root": str(repo)},
        }, separators=(",", ":"), sort_keys=True),
    )
    prepared = store.prepare_approval_action(
        project_id=project_id_for_root(repo.resolve()),
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        generation=7,
        payload=ApprovalPayload("baseline-1", baseline_setting, None),
    )
    store._connection.execute(
        "UPDATE prepared_actions SET previous_preparation_id = ? WHERE preparation_id = ?",
        ("missing-predecessor", prepared.preparation_id),
    )

    with pytest.raises(RuntimeError, match="legacy project ownership audit failed"):
        store.audit_legacy_project_ownership(str(repo.resolve()))


def test_legacy_audit_rejects_a_prepared_scope_for_a_different_baseline(
    tmp_path,
    valid_brief,
) -> None:
    """A recovery-capable scope must remain bound to its approved baseline."""
    repo = tmp_path / "repo"
    repo.mkdir()
    store = _store(tmp_path)
    store.create_session("session-1", str(repo))
    store.set_setting("agent_bridge.active_session_id", "session-1")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    baseline_setting = BaselineSetting(
        key="agent_bridge.baseline.task-1.1",
        value_json=json.dumps({
            "task_id": "task-1",
            "revision": 1,
            "baseline_id": "baseline-1",
            "manifest": {"baseline_id": "baseline-1", "repo_root": str(repo)},
        }, separators=(",", ":"), sort_keys=True),
    )
    prepared = store.prepare_approval_action(
        project_id=project_id_for_root(repo.resolve()),
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        generation=7,
        payload=ApprovalPayload("baseline-1", baseline_setting, None),
    )
    payload = json.loads(store._connection.execute(
        "SELECT payload_json FROM prepared_actions WHERE preparation_id = ?",
        (prepared.preparation_id,),
    ).fetchone()["payload_json"])
    scope = {
        "kind": "scope_approval",
        "baseline_id": "foreign-baseline",
        "approved_revision": valid_brief.revision,
        "underlying_continuation": None,
    }
    payload["scope"] = scope
    store._connection.execute(
        """
        UPDATE prepared_actions
        SET payload_json = ?, pending_context_json = ?, continuation_state = ?
        WHERE preparation_id = ?
        """,
        (
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
            json.dumps(scope, separators=(",", ":"), sort_keys=True),
            TaskState.SOL_RUNNING.value,
            prepared.preparation_id,
        ),
    )

    with pytest.raises(RuntimeError, match="legacy project ownership audit failed"):
        store.audit_legacy_project_ownership(str(repo.resolve()))


def _prepared_answer_for_legacy_audit(
    tmp_path,
    valid_brief,
    *,
    active_state: TaskState,
    continuation,
) -> tuple[SQLiteStore, str, object]:
    """Persist one real Answer row for the supplied valid continuation family."""
    repo = tmp_path / "repo"
    repo.mkdir()
    canonical_root = str(repo.resolve())
    store = _store(tmp_path)
    store.create_session("session-1", canonical_root)
    store.set_setting("agent_bridge.active_session_id", "session-1")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    store.approve_task_with_setting(
        valid_brief.task_id,
        valid_brief.revision,
        brief=valid_brief,
        baseline_id="baseline-1",
        expected=TaskState.AWAITING_USER_APPROVAL,
        setting=(
            "agent_bridge.baseline.task-1.1",
            {
                "task_id": "task-1",
                "revision": 1,
                "baseline_id": "baseline-1",
                "manifest": {"baseline_id": "baseline-1", "repo_root": canonical_root},
            },
        ),
    )
    store.transition_task(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.AWAITING_USER_APPROVAL,
        target=TaskState.SOL_RUNNING,
    )
    store.set_sol_thread(
        valid_brief.task_id,
        valid_brief.revision,
        "11111111-1111-4111-8111-111111111111",
    )
    if active_state is TaskState.FABLE_REVIEWING:
        store.set_fable_session(valid_brief.task_id, valid_brief.revision, "fable-session")
        store.transition_task(
            valid_brief.task_id,
            valid_brief.revision,
            expected=TaskState.SOL_RUNNING,
            target=active_state,
        )
    elif active_state is TaskState.SOL_CORRECTING:
        store.set_fable_session(valid_brief.task_id, valid_brief.revision, "fable-session")
        store.transition_task(
            valid_brief.task_id,
            valid_brief.revision,
            expected=TaskState.SOL_RUNNING,
            target=TaskState.FABLE_REVIEWING,
        )
        store.transition_task(
            valid_brief.task_id,
            valid_brief.revision,
            expected=TaskState.FABLE_REVIEWING,
            target=TaskState.SOL_CORRECTING,
        )

    pending = {
        TaskState.SOL_RUNNING: {"prompt": "continue exact work", "sol_run_id": "run-sol"},
        TaskState.SOL_CORRECTING: {"prompt": "continue exact work", "sol_run_id": "run-sol"},
        TaskState.FABLE_REVIEWING: {
            "review_prompt": "review exact work", "completion_allowed": False,
        },
    }[active_state]
    waiting = store.pause_for_continuation(
        valid_brief.task_id,
        valid_brief.revision,
        expected=active_state,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=active_state,
        pending=pending,
    )
    prepared = store.prepare_answer_action(
        project_id=project_id_for_root(repo.resolve()),
        session_id=waiting.session_id,
        task_id=waiting.task_id,
        revision=waiting.revision,
        generation=1,
        payload=AnswerPayload(answer="continue", continuation=continuation),
    )
    return store, canonical_root, prepared


def _replace_prepared_answer_context(
    store: SQLiteStore,
    preparation_id: str,
    *,
    payload_context,
    pending_context=None,
) -> None:
    """Persist a deliberately substituted Answer payload/pending context pair."""
    if pending_context is None:
        pending_context = payload_context
    payload_data = store_module._context_to_data(payload_context)
    pending_data = store_module._context_to_data(pending_context)
    payload = json.loads(store._connection.execute(
        "SELECT payload_json FROM prepared_actions WHERE preparation_id = ?",
        (preparation_id,),
    ).fetchone()["payload_json"])
    payload["continuation"] = payload_data
    store._connection.execute(
        """
        UPDATE prepared_actions SET payload_json = ?, pending_context_json = ?
        WHERE preparation_id = ?
        """,
        (
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
            json.dumps(pending_data, separators=(",", ":"), sort_keys=True),
            preparation_id,
        ),
    )


def test_legacy_audit_rejects_top_level_scope_without_underlying_before_recovery(
    tmp_path,
    valid_brief,
) -> None:
    """An audited Answer cannot replace an exact Sol resume with a fresh start."""
    continuation = SolResumeContext(
        "11111111-1111-4111-8111-111111111111", "run-sol", "continue exact work",
    )
    store, canonical_root, prepared = _prepared_answer_for_legacy_audit(
        tmp_path,
        valid_brief,
        active_state=TaskState.SOL_RUNNING,
        continuation=continuation,
    )
    assert store.get_task(valid_brief.task_id, valid_brief.revision).sol_thread_id == (
        continuation.sol_thread_id
    )
    _replace_prepared_answer_context(
        store,
        prepared.preparation_id,
        payload_context=ScopeApprovalContext("baseline-1", 1, None),
    )
    before = _legacy_table_rows(store._connection)

    with pytest.raises(RuntimeError, match="legacy project ownership audit failed"):
        store.audit_legacy_project_ownership(canonical_root)

    assert _legacy_table_rows(store._connection) == before
    recovered = store.prepared_action(prepared.preparation_id)
    assert recovered is not None
    assert recovered.status == "PREPARED"
    assert store.get_task(valid_brief.task_id, valid_brief.revision).state is TaskState.SOL_RUNNING


@pytest.mark.parametrize("column", ("state", "continuation_state"))
def test_legacy_audit_redacts_corrupt_task_state_parse_failures(
    tmp_path,
    valid_brief,
    column: str,
) -> None:
    """Task-row enum corruption must fail through the fixed audit boundary."""
    store, canonical_root, prepared = _prepared_answer_for_legacy_audit(
        tmp_path,
        valid_brief,
        active_state=TaskState.SOL_RUNNING,
        continuation=SolResumeContext(
            "11111111-1111-4111-8111-111111111111", "run-sol", "continue exact work",
        ),
    )
    sentinel = "CORRUPT_TASK_STATE_SENTINEL_/outside/legacy"
    store._connection.execute(
        f"UPDATE tasks SET {column} = ? WHERE task_id = ? AND revision = ?",
        (sentinel, valid_brief.task_id, valid_brief.revision),
    )
    before = _legacy_table_rows(store._connection)

    with pytest.raises(RuntimeError, match="legacy project ownership audit failed") as error:
        store.audit_legacy_project_ownership(canonical_root)

    assert sentinel not in repr(error.value)
    assert "/outside/legacy" not in repr(error.value)
    assert _legacy_table_rows(store._connection) == before
    assert store.prepared_action(prepared.preparation_id) is not None


@pytest.mark.parametrize(
    ("prepared_active_state", "corrupt_active_state", "corrupt_context"),
    (
        pytest.param(
            TaskState.FABLE_REVIEWING,
            TaskState.FABLE_REVIEWING,
            SolResumeContext(
                sol_thread_id="11111111-1111-4111-8111-111111111111",
                sol_run_id="run-sol",
                prompt="continue exact work",
            ),
            id="reviewing_with_sol_context",
        ),
        pytest.param(
            TaskState.SOL_RUNNING,
            TaskState.FABLE_CLARIFYING,
            ReviewContext(
                fable_session_id="fable-session",
                review_prompt="review exact work",
                completion_allowed=False,
                underlying_continuation=ScopeApprovalContext(
                    baseline_id="baseline-1",
                    approved_revision=1,
                    underlying_continuation=SolResumeContext(
                        sol_thread_id="11111111-1111-4111-8111-111111111111",
                        sol_run_id="run-sol",
                        prompt="continue exact work",
                    ),
                ),
            ),
            id="clarifying_with_review_context",
        ),
        pytest.param(
            TaskState.SOL_RUNNING,
            TaskState.SOL_RUNNING,
            ClarificationContext(
                fable_session_id="fable-session",
                clarification_prompt="clarify exact work",
                underlying_continuation=ScopeApprovalContext(
                    baseline_id="baseline-1",
                    approved_revision=1,
                    underlying_continuation=SolResumeContext(
                        sol_thread_id="11111111-1111-4111-8111-111111111111",
                        sol_run_id="run-sol",
                        prompt="continue exact work",
                    ),
                ),
            ),
            id="sol_with_clarification_context",
        ),
        pytest.param(
            TaskState.SOL_RUNNING,
            TaskState.SOL_RUNNING,
            AnswerContext(
                answer="nested persisted answer",
                underlying_continuation=SolResumeContext(
                    sol_thread_id="11111111-1111-4111-8111-111111111111",
                    sol_run_id="run-sol",
                    prompt="continue exact work",
                ),
            ),
            id="sol_with_nested_answer_context",
        ),
    ),
)
def test_legacy_audit_rejects_answer_context_incompatible_with_active_state(
    tmp_path,
    valid_brief,
    prepared_active_state: TaskState,
    corrupt_active_state: TaskState,
    corrupt_context,
) -> None:
    """Recovery must not adopt an Answer row with a substituted typed context."""
    valid_scope = ScopeApprovalContext("baseline-1", 1, None)
    valid_continuation = {
        TaskState.SOL_RUNNING: SolResumeContext(
            "11111111-1111-4111-8111-111111111111", "run-sol", "continue exact work",
        ),
        TaskState.FABLE_REVIEWING: ReviewContext(
            "fable-session", "review exact work", False, valid_scope,
        ),
    }[prepared_active_state]
    store, canonical_root, prepared = _prepared_answer_for_legacy_audit(
        tmp_path,
        valid_brief,
        active_state=prepared_active_state,
        continuation=valid_continuation,
    )
    _replace_prepared_answer_context(
        store, prepared.preparation_id, payload_context=corrupt_context,
    )
    store._connection.execute(
        "UPDATE prepared_actions SET active_state = ?, continuation_state = ? WHERE preparation_id = ?",
        (corrupt_active_state.value, corrupt_active_state.value, prepared.preparation_id),
    )
    before = _legacy_table_rows(store._connection)

    with pytest.raises(RuntimeError, match="legacy project ownership audit failed"):
        store.audit_legacy_project_ownership(canonical_root)

    assert _legacy_table_rows(store._connection) == before
    assert (
        store.get_task(valid_brief.task_id, valid_brief.revision).state
        is prepared_active_state
    )


@pytest.mark.parametrize(
    ("active_state", "continuation"),
    (
        pytest.param(
            TaskState.SOL_RUNNING,
            SolResumeContext(
                "11111111-1111-4111-8111-111111111111", "run-sol", "continue exact work",
            ),
            id="sol_running",
        ),
        pytest.param(
            TaskState.SOL_RUNNING,
            ScopeApprovalContext(
                "baseline-1",
                1,
                SolResumeContext(
                    "11111111-1111-4111-8111-111111111111",
                    "run-sol",
                    "continue exact work",
                ),
            ),
            id="sol_running_top_level_scope",
        ),
        pytest.param(
            TaskState.SOL_CORRECTING,
            SolResumeContext(
                "11111111-1111-4111-8111-111111111111", "run-sol", "continue exact work",
            ),
            id="sol_correcting_without_agent_run",
        ),
        pytest.param(
            TaskState.FABLE_REVIEWING,
            ReviewContext(
                "fable-session", "review exact work", False,
                ScopeApprovalContext("baseline-1", 1, None),
            ),
            id="reviewing",
        ),
    ),
)
def test_legacy_audit_accepts_each_valid_prepared_answer_context_family(
    tmp_path,
    valid_brief,
    active_state: TaskState,
    continuation,
) -> None:
    """Every continuation family accepted by normal Answer preparation remains auditable."""
    store, canonical_root, prepared = _prepared_answer_for_legacy_audit(
        tmp_path, valid_brief, active_state=active_state, continuation=continuation,
    )

    assert store.audit_legacy_project_ownership(canonical_root) is None
    assert store.prepared_action(prepared.preparation_id) == prepared


def test_legacy_audit_rejects_answer_context_when_payload_and_pending_differ(
    tmp_path,
    valid_brief,
) -> None:
    """Payload and pending continuation must remain the same persisted context."""
    payload_context = SolResumeContext(
        "11111111-1111-4111-8111-111111111111", "run-sol", "continue exact work",
    )
    pending_context = SolResumeContext(
        "11111111-1111-4111-8111-111111111111", "run-sol-other", "continue other work",
    )
    store, canonical_root, prepared = _prepared_answer_for_legacy_audit(
        tmp_path,
        valid_brief,
        active_state=TaskState.SOL_RUNNING,
        continuation=payload_context,
    )
    _replace_prepared_answer_context(
        store,
        prepared.preparation_id,
        payload_context=payload_context,
        pending_context=pending_context,
    )
    before = _legacy_table_rows(store._connection)

    with pytest.raises(RuntimeError, match="legacy project ownership audit failed"):
        store.audit_legacy_project_ownership(canonical_root)

    assert _legacy_table_rows(store._connection) == before


@pytest.mark.parametrize(
    ("active_state", "valid_context", "corrupt_context"),
    (
        pytest.param(
            TaskState.FABLE_REVIEWING,
            ReviewContext(
                "fable-session", "review exact work", False,
                ScopeApprovalContext("baseline-1", 1, None),
            ),
            ReviewContext(
                "foreign-fable-session", "review exact work", False,
                ScopeApprovalContext("baseline-1", 1, None),
            ),
            id="wrong_fable_session",
        ),
        pytest.param(
            TaskState.SOL_RUNNING,
            SolResumeContext(
                "11111111-1111-4111-8111-111111111111", "run-sol", "continue exact work",
            ),
            SolResumeContext(
                "22222222-2222-4222-8222-222222222222", "run-sol", "continue exact work",
            ),
            id="wrong_sol_thread",
        ),
        pytest.param(
            TaskState.SOL_RUNNING,
            ScopeApprovalContext(
                "baseline-1", 1,
                SolResumeContext(
                    "11111111-1111-4111-8111-111111111111", "run-sol", "continue exact work",
                ),
            ),
            ScopeApprovalContext(
                "foreign-baseline", 1,
                SolResumeContext(
                    "11111111-1111-4111-8111-111111111111", "run-sol", "continue exact work",
                ),
            ),
            id="wrong_baseline",
        ),
        pytest.param(
            TaskState.SOL_RUNNING,
            ScopeApprovalContext(
                "baseline-1", 1,
                SolResumeContext(
                    "11111111-1111-4111-8111-111111111111", "run-sol", "continue exact work",
                ),
            ),
            ScopeApprovalContext(
                "baseline-1", 2,
                SolResumeContext(
                    "11111111-1111-4111-8111-111111111111", "run-sol", "continue exact work",
                ),
            ),
            id="wrong_approved_revision",
        ),
    ),
)
def test_legacy_audit_rejects_answer_context_with_wrong_task_identifier(
    tmp_path,
    valid_brief,
    active_state: TaskState,
    valid_context,
    corrupt_context,
) -> None:
    """Each independently persisted task identifier must authenticate exactly."""
    store, canonical_root, prepared = _prepared_answer_for_legacy_audit(
        tmp_path,
        valid_brief,
        active_state=active_state,
        continuation=valid_context,
    )
    _replace_prepared_answer_context(
        store, prepared.preparation_id, payload_context=corrupt_context,
    )
    before = _legacy_table_rows(store._connection)

    with pytest.raises(RuntimeError, match="legacy project ownership audit failed"):
        store.audit_legacy_project_ownership(canonical_root)

    assert _legacy_table_rows(store._connection) == before


def test_legacy_audit_rejects_answer_context_with_foreign_existing_sol_run(
    tmp_path,
    valid_brief,
) -> None:
    """A persisted Sol run id is authoritative only for its exact task and CLI thread."""
    continuation = SolResumeContext(
        "11111111-1111-4111-8111-111111111111", "run-sol", "continue exact work",
    )
    store, canonical_root, _ = _prepared_answer_for_legacy_audit(
        tmp_path,
        valid_brief,
        active_state=TaskState.SOL_RUNNING,
        continuation=continuation,
    )
    store.prepare_new_request_action(
        project_id=project_id_for_root(Path(canonical_root)),
        session_id="session-1",
        task_id="other-task",
        generation=1,
        payload=NewRequestPayload("other task"),
    )
    store.start_agent_run("run-sol", "other-task", 0, "sol")
    store.set_agent_run_session("run-sol", "foreign-thread")

    with pytest.raises(RuntimeError, match="legacy project ownership audit failed"):
        store.audit_legacy_project_ownership(canonical_root)


def test_legacy_audit_accepts_answer_context_with_matching_existing_sol_run(
    tmp_path,
    valid_brief,
) -> None:
    """An existing Sol run may authenticate the same task/revision/thread binding."""
    continuation = SolResumeContext(
        "11111111-1111-4111-8111-111111111111", "run-sol", "continue exact work",
    )
    store, canonical_root, _ = _prepared_answer_for_legacy_audit(
        tmp_path,
        valid_brief,
        active_state=TaskState.SOL_RUNNING,
        continuation=continuation,
    )
    store.start_agent_run("run-sol", valid_brief.task_id, valid_brief.revision, "sol")
    store.set_agent_run_session("run-sol", continuation.sol_thread_id)

    assert store.audit_legacy_project_ownership(canonical_root) is None


def test_prepared_recoverable_rows_are_bounded(
    tmp_path,
) -> None:
    """Prepared recovery retains no Python-sized record collection after batching."""
    repo = tmp_path / "repo"
    repo.mkdir()
    canonical_repo = str(repo.resolve())
    store = _store(tmp_path)
    store.create_session("session-1", canonical_repo)
    store.set_setting("agent_bridge.active_session_id", "session-1")
    recoverable_count = 4_000
    prepared = tuple(
        store.prepare_new_request_action(
            project_id=project_id_for_root(repo.resolve()),
            session_id="session-1",
            task_id=f"prepared-bounded-{index:05d}",
            generation=index,
            payload=NewRequestPayload(f"prepared request {index}"),
        )
        for index in range(recoverable_count)
    )

    tracemalloc.start()
    try:
        store.audit_legacy_project_ownership(canonical_repo)
        recovered = store.recover_unfinished_prepared_actions()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1_000_000
    assert recovered == store_module.RecoverySummary(recoverable_count, recoverable_count, 0)
    assert store.prepared_action(prepared[0].preparation_id).status == "RECOVERED"
    assert store.prepared_action(prepared[-1].preparation_id).status == "RECOVERED"
    assert store.get_task(prepared[0].task_id, prepared[0].revision).state is TaskState.INTERRUPTED
    assert store.get_task(prepared[-1].task_id, prepared[-1].revision).state is TaskState.INTERRUPTED


def test_prepared_action_new_request_is_store_owned_and_has_no_task_pending_context(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    observed = []
    store.add_event_listener(observed.append)

    prepared = store.prepare_new_request_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id="task-1",
        generation=1,
        payload=NewRequestPayload(text="Build the bridge"),
    )

    assert prepared.project_id == "a" * 32
    assert prepared.session_id == "session-1"
    assert prepared.task_id == "task-1"
    assert prepared.revision == 0
    assert prepared.action == "new_request"
    assert prepared.status == "PREPARED"
    assert store.prepared_action(prepared.preparation_id) == prepared
    task = store.get_task("task-1", 0)
    assert task.state is TaskState.FABLE_PLANNING
    assert task.pending is None
    assert [event.payload for event in observed] == [{"text": "Build the bridge"}]


def test_prepared_new_request_persists_requested_recipient_with_fable_owned_route(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")

    store.prepare_new_request_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id="task-directed",
        generation=1,
        payload=NewRequestPayload(
            text="Plan it with the team.", addressed_to=ConversationTarget.TEAM,
        ),
    )

    event = store.events_after("session-1", 0)[0]
    assert event.kind == "conversation"
    assert event.payload == ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.TEAM,
        routed_to=ConversationTarget.FABLE,
        message_type=ConversationMessageType.STATEMENT,
        text="Plan it with the team.",
    ).to_dict()


def test_prepared_action_claim_abort_and_recovery_are_exact_and_durable(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    prepared = store.prepare_new_request_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id="task-1",
        generation=7,
        payload=NewRequestPayload(text="Build the bridge"),
    )

    aborted = store.abort_prepared_action(
        prepared.preparation_id, generation=7, reason="scheduler_unavailable",
    )

    assert aborted.status == "ABORTED"
    assert store.get_task("task-1", 0).state is TaskState.INTERRUPTED
    assert store.prepared_action(prepared.preparation_id) == aborted
    repeated = store.abort_prepared_action(
        prepared.preparation_id, generation=7, reason="scheduler_unavailable",
    )
    assert repeated == aborted

    second = store.prepare_new_request_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id="task-2",
        generation=8,
        payload=NewRequestPayload(text="Another request"),
    )
    claimed = store.claim_prepared_action(second.preparation_id, generation=8)
    recovered = store.recover_unfinished_prepared_actions()

    assert claimed.status == "CLAIMED"
    assert recovered == store_module.RecoverySummary(1, 1, 0)
    assert store.prepared_action(claimed.preparation_id) == replace(
        claimed, status="RECOVERED", reason=None,
    )
    interrupted = store.get_task("task-2", 0)
    assert interrupted.state is TaskState.INTERRUPTED
    assert interrupted.continuation_state is TaskState.FABLE_PLANNING
    assert interrupted.pending == {
        "prepared_action": {
            "preparation_id": second.preparation_id,
            "action": "new_request",
            "reason": "recovery",
            "context": None,
        },
    }


def test_prepared_answer_and_resume_preserve_the_exact_continuation_and_lineage(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "thread-1")
    waiting = store.pause_for_continuation(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=TaskState.SOL_CORRECTING,
        pending={
            "prompt": "continue the exact correction",
            "sol_run_id": "run-1",
        },
    )
    continuation = SolResumeContext(
        sol_thread_id="thread-1",
        sol_run_id="run-1",
        prompt="continue the exact correction",
    )
    observed = []
    store.add_event_listener(observed.append)

    answered = store.prepare_answer_action(
        project_id="a" * 32,
        session_id=waiting.session_id,
        task_id=waiting.task_id,
        revision=waiting.revision,
        generation=11,
        payload=AnswerPayload(answer="Use the existing setting.", continuation=continuation),
    )

    assert answered.active_state is TaskState.SOL_CORRECTING
    assert answered.continuation_state is TaskState.SOL_CORRECTING
    assert answered.pending_context == continuation
    assert [event.payload for event in observed] == [{"text": "Use the existing setting."}]
    aborted = store.abort_prepared_action(
        answered.preparation_id, generation=11, reason="scheduler_unavailable",
    )
    after_abort = store.get_task(waiting.task_id, waiting.revision)
    assert after_abort.state is TaskState.INTERRUPTED
    assert after_abort.continuation_state is TaskState.SOL_CORRECTING
    assert after_abort.pending == {
        "prepared_action": {
            "preparation_id": answered.preparation_id,
            "action": "answer",
            "reason": "scheduler_unavailable",
            "context": {
                "kind": "sol_resume",
                "sol_thread_id": "thread-1",
                "sol_run_id": "run-1",
                "prompt": "continue the exact correction",
            },
        },
    }

    observed.clear()
    resumed = store.prepare_resume_action(
        project_id="a" * 32,
        session_id=waiting.session_id,
        task_id=waiting.task_id,
        revision=waiting.revision,
        generation=12,
        payload=ResumePayload(
            continuation=aborted.pending_context,
            drift_event=ResumeDriftProjection(
                status="unchanged", summary="Repository drift was checked.", evidence_hashes=(),
            ),
        ),
        previous_preparation_id=aborted.preparation_id,
    )

    assert resumed.previous_preparation_id == aborted.preparation_id
    assert resumed.active_state is TaskState.SOL_CORRECTING
    assert resumed.continuation_state is TaskState.SOL_CORRECTING
    assert [event.kind for event in observed] == ["resume_drift"]
    assert store.get_task(waiting.task_id, waiting.revision).pending is None


def test_prepared_approval_persists_the_canonical_baseline_setting_once(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    observed = []
    store.add_event_listener(observed.append)

    prepared = store.prepare_approval_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        generation=4,
        payload=store_module.ApprovalPayload(
            baseline_id="baseline-1",
            baseline_setting=BaselineSetting(
                key="baseline-setting-1",
                value_json='{"baseline_id":"baseline-1","manifest":{},"revision":1,"task_id":"task-1"}',
            ),
            scope=None,
        ),
    )

    assert prepared.source_state is TaskState.AWAITING_USER_APPROVAL
    assert prepared.active_state is TaskState.SOL_RUNNING
    assert store.get_setting("baseline-setting-1") == {
        "baseline_id": "baseline-1",
        "manifest": {},
        "revision": 1,
        "task_id": "task-1",
    }
    assert [event.kind for event in observed] == ["task_state"]


def test_prepared_resume_drift_projection_rejects_raw_path_like_content() -> None:
    with pytest.raises(ValueError, match="not safe"):
        ResumeDriftProjection(
            status="drifted",
            summary="Changed /private/repository/path after the stop.",
            evidence_hashes=(),
        )


def test_prepared_action_rejects_a_project_id_substituted_for_an_existing_session_root(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    store.create_session("session-1", str(repo))

    with pytest.raises(RuntimeError, match="project identity"):
        store.prepare_new_request_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id="task-1",
            generation=1,
            payload=NewRequestPayload(text="Build the bridge"),
        )

    prepared = store.prepare_new_request_action(
        project_id=project_id_for_root(repo.resolve()),
        session_id="session-1",
        task_id="task-1",
        generation=1,
        payload=NewRequestPayload(text="Build the bridge"),
    )
    assert prepared.project_id == project_id_for_root(repo.resolve())


def test_resume_drift_failure_is_an_atomic_no_child_terminal_transition(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.mark_interrupted(
        valid_brief.task_id,
        valid_brief.revision,
        continuation=TaskState.SOL_RUNNING,
    )
    observed = []
    store.add_event_listener(observed.append)

    failed = store.fail_resume_for_drift(
        project_id="a" * 32,
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        drift_event=ResumeDriftProjection(
            status="drifted",
            summary="Repository drift prevented automatic resume.",
            evidence_hashes=(),
        ),
    )

    assert failed.state is TaskState.FAILED
    assert failed.continuation_state is None
    assert failed.pending is None
    assert store.latest_prepared_action_for_task(
        project_id="a" * 32,
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
    ) is None
    assert [(event.kind, event.payload) for event in observed] == [
        ("task_state", {"state": TaskState.FAILED.value, "revision": 1}),
        ("resume_drift", {
            "status": "drifted",
            "summary": "Repository drift prevented automatic resume.",
            "evidence_hashes": (),
        }),
    ]


def test_prepared_resume_rejects_a_cross_context_before_clearing_interruption(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    interrupted = store.mark_interrupted(
        valid_brief.task_id,
        valid_brief.revision,
        continuation=TaskState.SOL_RUNNING,
    )

    with pytest.raises(RuntimeError, match="continuation does not match"):
        store.prepare_resume_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id=interrupted.task_id,
            revision=interrupted.revision,
            generation=0,
            payload=ResumePayload(
                continuation=None,
                drift_event=ResumeDriftProjection(
                    status="unchanged",
                    summary="Repository drift was checked.",
                    evidence_hashes=(),
                ),
            ),
            previous_preparation_id=None,
        )

    unchanged = store.get_task(interrupted.task_id, interrupted.revision)
    assert unchanged.state is TaskState.INTERRUPTED
    assert unchanged.continuation_state is TaskState.SOL_RUNNING


def test_prepared_stop_before_claim_is_durable_and_resumable(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    prepared = store.prepare_new_request_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id="task-1",
        generation=9,
        payload=NewRequestPayload(text="Build the bridge"),
    )
    store.mark_interrupted(
        "task-1", 0, continuation=TaskState.FABLE_PLANNING,
    )

    interrupted = store.interrupt_claimed_prepared_action(
        prepared.preparation_id, generation=9, reason="stop",
    )

    assert interrupted.status == "INTERRUPTED"
    assert interrupted.reason == "stop"
    task = store.get_task("task-1", 0)
    assert task.state is TaskState.INTERRUPTED
    assert task.continuation_state is TaskState.FABLE_PLANNING
    assert task.pending == {
        "prepared_action": {
            "preparation_id": prepared.preparation_id,
            "action": "new_request",
            "reason": "stop",
            "context": None,
        },
    }


def test_prepared_approval_setting_conflict_rolls_back_all_route_mutation(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.AWAITING_USER_APPROVAL)
    key = "baseline-setting-1"
    existing = '{"baseline_id":"old","manifest":{},"revision":1,"task_id":"task-1"}'
    expected = '{"baseline_id":"baseline-1","manifest":{},"revision":1,"task_id":"task-1"}'
    store._connection.execute(
        "INSERT INTO settings (key, value_json) VALUES (?, ?)", (key, existing),
    )
    observed = []
    store.add_event_listener(observed.append)

    with pytest.raises(RuntimeError, match="baseline setting changed"):
        store.prepare_approval_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            generation=5,
            payload=ApprovalPayload(
                baseline_id="baseline-1",
                baseline_setting=BaselineSetting(key=key, value_json=expected),
                scope=None,
            ),
        )

    task = store.get_task(valid_brief.task_id, valid_brief.revision)
    assert task.state is TaskState.AWAITING_USER_APPROVAL
    assert task.baseline_id is None
    assert store.get_setting(key) == json.loads(existing)
    assert observed == []
    assert store.latest_prepared_action_for_task(
        project_id="a" * 32,
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
    ) is None


def test_prepared_resume_requires_the_latest_lineage_and_no_generation_zero_collision(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "thread-1")
    store.set_pending_context(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.SOL_RUNNING,
        pending={"sol_run_id": "run-1", "prompt": "exact prompt"},
    )
    store.mark_interrupted(
        valid_brief.task_id, valid_brief.revision, continuation=TaskState.SOL_RUNNING,
    )
    continuation = SolResumeContext("thread-1", "run-1", "exact prompt")
    drift = ResumeDriftProjection(
        status="unchanged", summary="Repository drift was checked.", evidence_hashes=(),
    )
    first = store.prepare_resume_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        generation=0,
        payload=ResumePayload(continuation=continuation, drift_event=drift),
        previous_preparation_id=None,
    )
    store.abort_prepared_action(
        first.preparation_id, generation=0, reason="scheduler_unavailable",
    )
    second = store.prepare_resume_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        generation=4,
        payload=ResumePayload(continuation=continuation, drift_event=drift),
        previous_preparation_id=first.preparation_id,
    )
    store.abort_prepared_action(
        second.preparation_id, generation=4, reason="scheduler_unavailable",
    )

    with pytest.raises(RuntimeError, match="latest"):
        store.prepare_resume_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            generation=5,
            payload=ResumePayload(continuation=continuation, drift_event=drift),
            previous_preparation_id=first.preparation_id,
        )
    with pytest.raises(RuntimeError, match="previous preparation"):
        store.prepare_resume_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            generation=0,
            payload=ResumePayload(continuation=continuation, drift_event=drift),
            previous_preparation_id=None,
        )


def test_prepared_answer_rejects_a_cross_context_before_clearing_pending(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    waiting = store.pause_for_continuation(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=TaskState.SOL_RUNNING,
        pending={"prompt": "exact prompt"},
    )

    with pytest.raises(RuntimeError, match="continuation does not match"):
        store.prepare_answer_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id=waiting.task_id,
            revision=waiting.revision,
            generation=3,
            payload=AnswerPayload(
                answer="Use option A.",
                continuation=SolResumeContext("thread-1", "run-1", "different prompt"),
            ),
        )

    unchanged = store.get_task(waiting.task_id, waiting.revision)
    assert unchanged.state is TaskState.AWAITING_USER_INPUT
    assert unchanged.pending == {"prompt": "exact prompt"}


def test_generation_zero_resume_must_link_its_latest_terminal_generation_zero_predecessor(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    initial = store.prepare_new_request_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id="task-1",
        generation=0,
        payload=NewRequestPayload(text="Build the bridge"),
    )
    terminal = store.abort_prepared_action(
        initial.preparation_id, generation=0, reason="scheduler_unavailable",
    )

    resumed = store.prepare_resume_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id="task-1",
        revision=0,
        generation=0,
        payload=ResumePayload(
            continuation=None,
            drift_event=ResumeDriftProjection(
                status="unchanged", summary="Repository drift was checked.", evidence_hashes=(),
            ),
        ),
        previous_preparation_id=terminal.preparation_id,
    )

    assert resumed.previous_preparation_id == terminal.preparation_id
    assert resumed.generation == 0
    latest = store.abort_prepared_action(
        resumed.preparation_id, generation=0, reason="scheduler_unavailable",
    )
    payload = ResumePayload(
        continuation=None,
        drift_event=ResumeDriftProjection(
            status="unchanged", summary="Repository drift was checked.", evidence_hashes=(),
        ),
    )

    with pytest.raises(RuntimeError, match="latest"):
        store.prepare_resume_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id="task-1",
            revision=0,
            generation=0,
            payload=payload,
            previous_preparation_id=terminal.preparation_id,
        )
    with pytest.raises(RuntimeError, match="previous preparation"):
        store.prepare_resume_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id="task-1",
            revision=0,
            generation=0,
            payload=payload,
            previous_preparation_id=None,
        )

    other = store.prepare_new_request_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id="task-2",
        generation=0,
        payload=NewRequestPayload(text="Build the other bridge"),
    )
    other = store.abort_prepared_action(
        other.preparation_id, generation=0, reason="scheduler_unavailable",
    )
    with pytest.raises(RuntimeError, match="invalid"):
        store.prepare_resume_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id="task-1",
            revision=0,
            generation=0,
            payload=payload,
            previous_preparation_id=other.preparation_id,
        )

    claimed = store.prepare_resume_action(
        project_id="a" * 32,
        session_id="session-1",
        task_id="task-1",
        revision=0,
        generation=0,
        payload=payload,
        previous_preparation_id=latest.preparation_id,
    )
    store.claim_prepared_action(claimed.preparation_id, generation=0)
    store.mark_interrupted("task-1", 0, continuation=TaskState.FABLE_PLANNING)
    with pytest.raises(RuntimeError, match="invalid"):
        store.prepare_resume_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id="task-1",
            revision=0,
            generation=0,
            payload=payload,
            previous_preparation_id=claimed.preparation_id,
        )


def test_legacy_answer_rejects_missing_sol_run_id_without_clearing_task(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "thread-1")
    waiting = store.pause_for_continuation(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=TaskState.SOL_RUNNING,
        pending={"prompt": "continue exactly"},
    )

    with pytest.raises(RuntimeError, match="continuation does not match"):
        store.prepare_answer_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id=waiting.task_id,
            revision=waiting.revision,
            generation=0,
            payload=AnswerPayload(
                answer="Use option A.",
                continuation=SolResumeContext(
                    sol_thread_id="thread-1",
                    sol_run_id="run-1",
                    prompt="continue exactly",
                ),
            ),
        )

    unchanged = store.get_task(waiting.task_id, waiting.revision)
    assert unchanged.state is TaskState.AWAITING_USER_INPUT
    assert unchanged.pending == {"prompt": "continue exactly"}


def test_legacy_scope_approval_rejects_missing_sol_run_id_without_clearing_task(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "thread-1")
    store._connection.execute(
        "UPDATE tasks SET baseline_id = ? WHERE task_id = ? AND revision = ?",
        ("baseline-1", valid_brief.task_id, valid_brief.revision),
    )
    clarifying = store.pause_for_continuation(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.FABLE_CLARIFYING,
        continuation_state=TaskState.SOL_RUNNING,
        pending={"clarification_prompt": "Clarify the exact scope."},
    )
    scope = store.retarget_continuation(
        clarifying.task_id,
        clarifying.revision,
        expected=TaskState.FABLE_CLARIFYING,
        target=TaskState.AWAITING_SCOPE_APPROVAL,
        pending={"answer": "Add the approved path.", "prompt": "resume exactly"},
    )

    with pytest.raises(RuntimeError, match="continuation does not match"):
        store.prepare_approval_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id=scope.task_id,
            revision=scope.revision,
            generation=0,
            payload=ApprovalPayload(
                baseline_id="baseline-1",
                baseline_setting=None,
                scope=ScopeApprovalContext(
                    baseline_id="baseline-1",
                    approved_revision=scope.revision,
                    underlying_continuation=SolResumeContext(
                        sol_thread_id="thread-1",
                        sol_run_id="run-1",
                        prompt="resume exactly",
                    ),
                ),
            ),
        )

    unchanged = store.get_task(scope.task_id, scope.revision)
    assert unchanged.state is TaskState.AWAITING_SCOPE_APPROVAL
    assert unchanged.pending == {
        "answer": "Add the approved path.", "prompt": "resume exactly",
    }


def test_legacy_review_requires_an_actual_boolean_completion_guard(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.FABLE_REVIEWING)
    store.set_fable_session(valid_brief.task_id, valid_brief.revision, "fable-session-1")
    waiting = store.pause_for_continuation(
        valid_brief.task_id,
        valid_brief.revision,
        expected=TaskState.FABLE_REVIEWING,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=TaskState.FABLE_REVIEWING,
        pending={"review_prompt": "Review the exact output.", "completion_allowed": 1},
    )

    with pytest.raises(RuntimeError, match="continuation does not match"):
        store.prepare_answer_action(
            project_id="a" * 32,
            session_id="session-1",
            task_id=waiting.task_id,
            revision=waiting.revision,
            generation=0,
            payload=AnswerPayload(
                answer="Use option A.",
                continuation=ReviewContext(
                    fable_session_id="fable-session-1",
                    review_prompt="Review the exact output.",
                    completion_allowed=True,
                    underlying_continuation=SolResumeContext(
                        sol_thread_id="thread-1",
                        sol_run_id="run-1",
                        prompt="resume exactly",
                    ),
                ),
            ),
        )

    unchanged = store.get_task(waiting.task_id, waiting.revision)
    assert unchanged.state is TaskState.AWAITING_USER_INPUT
    assert unchanged.pending == {
        "review_prompt": "Review the exact output.", "completion_allowed": 1,
    }


def _create_pre_directed_conversation_schema(path) -> sqlite3.Connection:
    """Create the exact Phase 1 schema before directed conversations."""
    connection = _create_current_schema(path)
    connection.executescript(
        """
        ALTER TABLE sessions ADD COLUMN title TEXT NOT NULL DEFAULT 'New chat';
        ALTER TABLE sessions ADD COLUMN updated_at TEXT;
        CREATE TABLE prepared_actions (
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
        );
        CREATE UNIQUE INDEX one_running_agent_run_per_task_revision
        ON agent_runs (task_id, revision) WHERE status = 'running';
        CREATE INDEX events_session_sequence ON events (session_id, sequence);
        CREATE INDEX events_session_task_sequence
        ON events (session_id, task_id, sequence DESC);
        CREATE INDEX events_session_task_kind_sequence
        ON events (session_id, task_id, kind, sequence DESC);
        CREATE INDEX events_session_sequence_desc ON events (session_id, sequence DESC);
        CREATE INDEX prepared_actions_identity
        ON prepared_actions (project_id, session_id, task_id, revision, status);
        """
    )
    return connection


def _pre_directed_rows(connection: sqlite3.Connection) -> dict[str, tuple[tuple, ...]]:
    def rows(query: str) -> tuple[tuple, ...]:
        return tuple(tuple(row) for row in connection.execute(query))

    return {
        "sessions": rows(
            """
            SELECT session_id, repo_root, created_at, title, updated_at
            FROM sessions ORDER BY session_id
            """
        ),
        "tasks": rows(
            """
            SELECT task_id, revision, session_id, state, brief_json, approved_at,
                   fable_session_id, sol_thread_id, baseline_id, correction_count,
                   continuation_state, pending_json
            FROM tasks ORDER BY task_id, revision
            """
        ),
        "events": rows("SELECT * FROM events ORDER BY sequence"),
        "agent_runs": rows("SELECT * FROM agent_runs ORDER BY run_id"),
        "settings": rows("SELECT * FROM settings ORDER BY key"),
        "prepared_actions": rows("SELECT * FROM prepared_actions ORDER BY preparation_id"),
    }


def _seed_pre_directed_conversation_database(path) -> dict[str, tuple[tuple, ...]]:
    connection = _create_pre_directed_conversation_schema(path)
    connection.execute(
        """
        INSERT INTO sessions (session_id, repo_root, created_at, title, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "session-legacy",
            "/repo/legacy",
            "2026-08-10T10:00:00Z",
            "Existing chat",
            "2026-08-10T10:01:00Z",
        ),
    )
    connection.execute(
        """
        INSERT INTO tasks (
            task_id, revision, session_id, state, brief_json, approved_at,
            fable_session_id, sol_thread_id, baseline_id, correction_count,
            continuation_state, pending_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task.legacy",
            1,
            "session-legacy",
            TaskState.SOL_RUNNING.value,
            '{"raw":"brief bytes"}',
            "2026-08-10T10:02:00Z",
            "fable-session",
            "sol-thread",
            "baseline-legacy",
            2,
            None,
            '{"raw":"pending bytes"}',
        ),
    )
    connection.execute(
        """
        INSERT INTO events (session_id, task_id, actor, kind, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "session-legacy",
            "task.legacy",
            "sol",
            "outcome",
            '{"raw":"event bytes"}',
            "2026-08-10T10:03:00Z",
        ),
    )
    connection.execute(
        """
        INSERT INTO prepared_actions (
            preparation_id, project_id, session_id, task_id, revision, action,
            payload_json, source_state, active_state, continuation_state,
            pending_context_json, previous_preparation_id, status, reason, generation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "prepared-legacy",
            "a" * 32,
            "session-legacy",
            "task.legacy",
            1,
            "resume",
            '{"raw":"prepared bytes"}',
            TaskState.SOL_RUNNING.value,
            TaskState.SOL_RUNNING.value,
            None,
            None,
            None,
            "COMPLETED",
            None,
            0,
        ),
    )
    connection.commit()
    seeded = _pre_directed_rows(connection)
    connection.close()
    return seeded


def test_directed_question_exchange_migration_is_additive_idempotent_and_byte_safe(
    tmp_path,
) -> None:
    path = tmp_path / "pre-directed.sqlite3"
    before = _seed_pre_directed_conversation_database(path)

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    assert _pre_directed_rows(migrated._connection) == before
    task_columns = tuple(
        row["name"] for row in migrated._connection.execute("PRAGMA table_info(tasks)")
    )
    assert task_columns[-4:] == (
        "continuation_generation",
        "exchange_allowance",
        "exchange_consumed",
        "continuation_pause_id",
    )
    assert tuple(migrated._connection.execute(
        """
        SELECT continuation_generation, exchange_allowance, exchange_consumed,
               continuation_pause_id
        FROM tasks WHERE task_id = ? AND revision = ?
        """,
        ("task.legacy", 1),
    ).fetchone()) == (1, INITIAL_INTERNAL_EXCHANGES, 0, None)
    tables = {
        row["name"]
        for row in migrated._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {
        "questions", "exchange_reservations", "exchange_grants", "exchange_permissions",
    } <= tables
    question_columns = tuple(
        row["name"]
        for row in migrated._connection.execute("PRAGMA table_info(questions)")
    )
    assert "continuation_state" in question_columns
    assert "pending_action_json" in question_columns
    assert "continuation_pause_id" in question_columns
    assert {
        "nested_parent_kind", "parent_question_id", "parent_continuation_pause_id",
    } <= set(question_columns)
    grant_columns = tuple(
        row["name"]
        for row in migrated._connection.execute("PRAGMA table_info(exchange_grants)")
    )
    assert "permission_id" in grant_columns
    indexes = {
        row["name"]
        for row in migrated._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
    }
    assert {
        "one_unanswered_top_level_question_per_task_revision",
        "one_unanswered_nested_question_per_task_revision",
        "exchange_reservations_request_identity",
        "exchange_grants_request_identity",
        "exchange_grants_permission_identity",
        "exchange_permissions_pause_identity",
        "prepared_actions_preparation_identifier",
        "directed_fable_answer_checkpoints_identity",
    } <= indexes
    checkpoint_columns = tuple(
        row["name"]
        for row in migrated._connection.execute(
            "PRAGMA table_info(directed_fable_answer_checkpoints)"
        )
    )
    assert "project_id" in checkpoint_columns
    migrated.close()

    first_migration_bytes = path.read_bytes()
    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    assert _pre_directed_rows(reopened._connection) == before
    reopened.close()
    assert path.read_bytes() == first_migration_bytes


def test_pre_directed_preparation_identifiers_are_globally_unique_before_checkpoint_migration(
    tmp_path,
) -> None:
    """The released legacy schema rejects an ambiguous cross-project preparation ID."""
    connection = _create_pre_directed_conversation_schema(tmp_path / "legacy.sqlite3")
    connection.execute(
        """
        INSERT INTO sessions (session_id, repo_root, created_at, title, updated_at)
        VALUES ('session-1', '/repo/one', '2026-08-10T10:00:00Z', 'One', '2026-08-10T10:00:00Z')
        """
    )
    connection.execute(
        """
        INSERT INTO tasks (
            task_id, revision, session_id, state, correction_count
        ) VALUES ('task-1', 1, 'session-1', 'SOL_RUNNING', 0)
        """
    )
    values = (
        'prepared-global', 'a' * 32, 'session-1', 'task-1', 1, 'approval', '{}',
        'AWAITING_USER_APPROVAL', 'SOL_RUNNING', None, None, None, 'COMPLETED', None, 0,
    )
    connection.execute(
        """
        INSERT INTO prepared_actions (
            preparation_id, project_id, session_id, task_id, revision, action,
            payload_json, source_state, active_state, continuation_state,
            pending_context_json, previous_preparation_id, status, reason, generation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO prepared_actions (
                preparation_id, project_id, session_id, task_id, revision, action,
                payload_json, source_state, active_state, continuation_state,
                pending_context_json, previous_preparation_id, status, reason, generation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ('prepared-global', 'b' * 32, *values[2:]),
        )
    connection.close()


def test_directed_question_exchange_migration_rolls_back_all_ddl_on_injected_failure(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "directed-rollback.sqlite3"
    connection = _create_pre_directed_conversation_schema(path)
    connection.execute(
        """
        INSERT INTO sessions (session_id, repo_root, created_at, title, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("session-1", "/repo", "2026-08-10T10:00:00Z", "New chat", "2026-08-10T10:00:00Z"),
    )
    connection.commit()
    connection.close()

    def fail_after_directed_ddl(self) -> None:
        raise sqlite3.IntegrityError("injected directed migration failure")

    monkeypatch.setattr(
        SQLiteStore,
        "_migrate_directed_conversation_schema",
        fail_after_directed_ddl,
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected directed migration failure"):
        SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    inspected = sqlite3.connect(path)
    task_columns = tuple(row[1] for row in inspected.execute("PRAGMA table_info(tasks)"))
    assert task_columns == (
        "task_id", "revision", "session_id", "state", "brief_json", "approved_at",
        "fable_session_id", "sol_thread_id", "baseline_id", "correction_count",
        "continuation_state", "pending_json",
    )
    tables = {
        row[0] for row in inspected.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert "questions" not in tables
    assert "exchange_reservations" not in tables
    assert "exchange_grants" not in tables
    assert "exchange_permissions" not in tables
    inspected.close()


def _conversation_question(
    *,
    sender: ConversationActor,
    addressed_to: ConversationTarget,
    routed_to: ConversationTarget,
    task_id: str,
    revision: int,
    generation: int,
    question_id: str,
    text: str,
) -> ConversationEnvelope:
    return ConversationEnvelope(
        sender=sender,
        addressed_to=addressed_to,
        routed_to=routed_to,
        message_type=ConversationMessageType.QUESTION,
        text=text,
        task_id=task_id,
        revision=revision,
        continuation_generation=generation,
        question_id=question_id,
    )


def _conversation_answer(
    *,
    sender: ConversationActor,
    addressed_to: ConversationTarget,
    routed_to: ConversationTarget,
    task_id: str,
    revision: int,
    generation: int,
    question_id: str,
    text: str,
) -> ConversationEnvelope:
    return ConversationEnvelope(
        sender=sender,
        addressed_to=addressed_to,
        routed_to=routed_to,
        message_type=ConversationMessageType.ANSWER,
        text=text,
        task_id=task_id,
        revision=revision,
        continuation_generation=generation,
        reply_to_question_id=question_id,
    )


def _conversation_permission(
    *, task_id: str, revision: int, generation: int,
) -> ConversationEnvelope:
    return ConversationEnvelope(
        sender=ConversationActor.SYSTEM,
        addressed_to=ConversationTarget.USER,
        routed_to=ConversationTarget.USER,
        message_type=ConversationMessageType.STATUS,
        text="The internal exchange limit was reached. Allow three more exchanges?",
        task_id=task_id,
        revision=revision,
        continuation_generation=generation,
    )


def _save_active_directed_task(store, session_id: str, brief) -> None:
    store.save_task(session_id, brief, TaskState.SOL_RUNNING)


def test_intervention_migration_is_additive_idempotent_and_preserves_current_rows(
    tmp_path, valid_brief,
) -> None:
    """Adding interventions must not rewrite current durable rows on reopen."""
    path = tmp_path / "intervention-migration.sqlite3"
    initial = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    initial.create_session("session-1", "/repo")
    initial.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    initial.start_agent_run("source-run-1", valid_brief.task_id, valid_brief.revision, "sol")
    before = {
        table: tuple(initial._connection.execute(f"SELECT * FROM {table} ORDER BY rowid"))
        for table in ("sessions", "tasks", "events", "agent_runs", "settings")
    }
    initial.close()

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    assert {
        table: tuple(migrated._connection.execute(f"SELECT * FROM {table} ORDER BY rowid"))
        for table in before
    } == before
    intervention_columns = {
        row["name"] for row in migrated._connection.execute("PRAGMA table_info(interventions)")
    }
    assert {
        "intervention_id", "session_id", "task_id", "revision", "run_id",
        "source_generation", "resume_generation", "fable_session_id", "sol_thread_id",
        "resume_attempt_id", "resume_run_id", "acknowledgment_id", "status", "created_at",
    } <= intervention_columns
    migrated.close()

    first_migration_bytes = path.read_bytes()
    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    reopened.close()
    assert path.read_bytes() == first_migration_bytes


def test_directed_intervention_binding_migration_is_additive_idempotent_and_preserves_rows(
    tmp_path, valid_brief,
) -> None:
    """An existing intervention table gains only the durable directed discriminator."""
    path = tmp_path / "pre-directed-binding.sqlite3"
    initial = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    initial.create_session("session-1", "/repo")
    initial.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    initial.set_sol_thread(valid_brief.task_id, valid_brief.revision, "provider-1")
    initial.start_agent_run(
        "source-run-1", valid_brief.task_id, valid_brief.revision, "sol",
    )
    initial.set_agent_run_session("source-run-1", "provider-1")
    created = initial.create_intervention_and_request_stop(
        intervention_id="intervention-1", session_id="session-1",
        task_id=valid_brief.task_id, revision=valid_brief.revision,
        expected_source_generation=1, message="Keep the exact direction.",
        addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
        run_id="source-run-1",
    )
    initial.close()

    legacy = sqlite3.connect(path)
    columns = {
        row[1] for row in legacy.execute("PRAGMA table_info(interventions)")
    }
    if "directed_binding_json" in columns:
        legacy.execute("ALTER TABLE interventions DROP COLUMN directed_binding_json")
        legacy.commit()
    legacy.close()

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    assert {
        row["name"] for row in migrated._connection.execute(
            "PRAGMA table_info(interventions)"
        )
    } >= {"directed_binding_json"}
    assert migrated.intervention(created.intervention_id) == created
    migrated.close()

    first_migration_bytes = path.read_bytes()
    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    reopened.close()
    assert path.read_bytes() == first_migration_bytes


def _seed_previous_directed_intervention_families(
    path: Path, valid_brief, *, binding_column: str,
) -> tuple[tuple[object, ...], ...]:
    """Create reachable directed rows, then restore the exact preceding schema shape."""
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    store.create_session("session-1", "/repo")
    for status in (
        store_module.InterventionStatus.PENDING_STOP,
        store_module.InterventionStatus.READY,
        store_module.InterventionStatus.RESUMING,
        store_module.InterventionStatus.RESUME_OUTCOME_UNKNOWN,
        store_module.InterventionStatus.RESUMED,
        store_module.InterventionStatus.CANCELED_BY_STOP,
    ):
        suffix = status.value
        brief = replace(valid_brief, task_id=f"directed-{suffix}")
        store.save_task("session-1", brief, TaskState.SOL_RUNNING)
        store.set_sol_thread(brief.task_id, brief.revision, f"sol-{suffix}")
        store.set_fable_session(brief.task_id, brief.revision, f"fable-{suffix}")
        _, question = store.reserve_internal_question(
            session_id="session-1", task_id=brief.task_id, revision=brief.revision,
            expected_generation=1, question_id=f"question-{suffix}",
            request_key=f"request-{suffix}", asked_by=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
            text=f"Which exact {suffix} fact applies?",
            continuation_state=TaskState.SOL_RUNNING,
            pending_action={"sol_run_id": f"sol-run-{suffix}", "prompt": suffix},
            event=ConversationEnvelope(
                sender=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
                routed_to=ConversationTarget.FABLE,
                message_type=ConversationMessageType.QUESTION,
                text=f"Which exact {suffix} fact applies?", task_id=brief.task_id,
                revision=brief.revision, continuation_generation=1,
                question_id=f"question-{suffix}",
            ),
        )
        assert question.exchange_id is not None
        source_run_id = f"source-{suffix}"
        store.start_agent_run(source_run_id, brief.task_id, brief.revision, "fable")
        store.set_agent_run_session(source_run_id, f"fable-{suffix}")
        created = store.create_intervention_and_request_stop(
            intervention_id=f"intervention-{suffix}", session_id="session-1",
            task_id=brief.task_id, revision=brief.revision,
            expected_source_generation=1, message=f"Keep {suffix} exact.",
            addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
            run_id=source_run_id,
        )
        if status is store_module.InterventionStatus.PENDING_STOP:
            continue
        store.finish_agent_run(source_run_id, status="interrupted", exit_code=-15)
        store.mark_intervention_ready(created.intervention_id, run_id=source_run_id)
        if status is store_module.InterventionStatus.READY:
            continue
        store.begin_intervention_resume(
            created.intervention_id, expected_resume_generation=created.resume_generation,
            resume_attempt_id=f"attempt-{suffix}", resume_run_id=f"resume-{suffix}",
        )
        if status is store_module.InterventionStatus.RESUME_OUTCOME_UNKNOWN:
            store.mark_resume_outcome_unknown(
                created.intervention_id, resume_attempt_id=f"attempt-{suffix}",
                resume_run_id=f"resume-{suffix}",
            )
            store._connection.execute(
                """
                UPDATE tasks SET state = ?
                WHERE task_id = ? AND revision = ? AND state = ?
                """,
                (
                    TaskState.INTERRUPTED.value,
                    brief.task_id,
                    brief.revision,
                    TaskState.AWAITING_USER_INPUT.value,
                ),
            )
        elif status in {
            store_module.InterventionStatus.RESUMED,
            store_module.InterventionStatus.CANCELED_BY_STOP,
        }:
            # The immediately preceding bytes can contain a completed directed
            # exchange whose durable binding column has not yet been added.
            store._connection.execute(
                """
                UPDATE questions
                SET answer_text = ?, answered_by = ?
                WHERE question_id = ? AND answer_text IS NULL
                """,
                (
                    f"Exact {suffix} answer.", ConversationActor.FABLE.value,
                    f"question-{suffix}",
                ),
            )
            if status is store_module.InterventionStatus.RESUMED:
                store.complete_intervention(
                    created.intervention_id,
                    expected_resume_generation=created.resume_generation,
                    resume_attempt_id=f"attempt-{suffix}",
                    resume_run_id=f"resume-{suffix}",
                )
            else:
                store.cancel_intervention_by_stop(
                    created.intervention_id,
                    expected_resume_generation=created.resume_generation,
                )
    store.close()

    preceding = sqlite3.connect(path)
    if binding_column == "absent":
        preceding.execute("ALTER TABLE interventions DROP COLUMN directed_binding_json")
    else:
        preceding.execute("UPDATE interventions SET directed_binding_json = NULL")
    preceding.commit()
    unchanged = tuple(
        tuple(row) for row in preceding.execute(
            """
            SELECT intervention_id, session_id, task_id, revision, addressed_to,
                   routed_to, message, run_id, continuation_state, source_generation,
                   resume_generation, fable_session_id, sol_thread_id,
                   resume_attempt_id, resume_run_id, acknowledgment_id, status, created_at
            FROM interventions ORDER BY intervention_id
            """
        )
    )
    preceding.close()
    return unchanged


@pytest.mark.parametrize("binding_column", ("absent", "null"))
def test_directed_intervention_binding_migration_backfills_each_live_family_exactly(
    tmp_path, valid_brief, monkeypatch: pytest.MonkeyPatch, binding_column: str,
) -> None:
    """The immediately preceding schema backfills only complete directed identities."""
    path = tmp_path / f"previous-directed-{binding_column}.sqlite3"
    unchanged = _seed_previous_directed_intervention_families(
        path, valid_brief, binding_column=binding_column,
    )
    before = sqlite3.connect(path)
    unrelated_before = {
        table: tuple(tuple(row) for row in before.execute(f"SELECT * FROM {table} ORDER BY rowid"))
        for table in (
            "sessions", "tasks", "events", "agent_runs", "questions",
            "exchange_reservations",
        )
    }
    before.close()
    monkeypatch.setattr(store_module, "_STARTUP_RECOVERY_BATCH_SIZE", 2)

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    for status in (
        store_module.InterventionStatus.PENDING_STOP,
        store_module.InterventionStatus.READY,
        store_module.InterventionStatus.RESUMING,
        store_module.InterventionStatus.RESUME_OUTCOME_UNKNOWN,
        store_module.InterventionStatus.RESUMED,
        store_module.InterventionStatus.CANCELED_BY_STOP,
    ):
        record = migrated.authenticated_intervention(f"intervention-{status.value}")
        assert record is not None
        assert record.status is status
        assert record.directed_binding is not None
        assert record.directed_binding.exchange_id is not None
    assert tuple(
        tuple(row) for row in migrated._connection.execute(
            """
            SELECT intervention_id, session_id, task_id, revision, addressed_to,
                   routed_to, message, run_id, continuation_state, source_generation,
                   resume_generation, fable_session_id, sol_thread_id,
                   resume_attempt_id, resume_run_id, acknowledgment_id, status, created_at
            FROM interventions ORDER BY intervention_id
            """
        )
    ) == unchanged
    assert {
        table: tuple(
            tuple(row) for row in migrated._connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            )
        )
        for table in unrelated_before
    } == unrelated_before
    migrated.close()

    migrated_bytes = path.read_bytes()
    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    for status in (
        store_module.InterventionStatus.PENDING_STOP,
        store_module.InterventionStatus.READY,
        store_module.InterventionStatus.RESUMING,
        store_module.InterventionStatus.RESUME_OUTCOME_UNKNOWN,
        store_module.InterventionStatus.RESUMED,
        store_module.InterventionStatus.CANCELED_BY_STOP,
    ):
        assert reopened.authenticated_intervention(
            f"intervention-{status.value}"
        ) is not None
    reopened.close()
    assert path.read_bytes() == migrated_bytes


@pytest.mark.parametrize("binding_column", ("absent", "null"))
def test_terminal_ordinary_migration_ignores_compatible_answered_questions(
    tmp_path, valid_brief, binding_column: str,
) -> None:
    """A matching answered row is ordinary when its source matches the active phase."""
    path = tmp_path / f"ordinary-terminal-{binding_column}.sqlite3"
    _seed_previous_directed_intervention_families(
        path, valid_brief, binding_column=binding_column,
    )
    preceding = sqlite3.connect(path)
    for status in ("resumed", "canceled_by_stop"):
        preceding.execute(
            """
            UPDATE questions
            SET asked_by = 'fable', addressed_to = 'sol', routed_to = 'sol',
                answered_by = 'sol'
            WHERE question_id = ?
            """,
            (f"question-{status}",),
        )
        preceding.execute(
            """
            UPDATE agent_runs SET agent = 'sol', cli_session_id = ?
            WHERE run_id = ?
            """,
            (f"sol-{status}", f"source-{status}"),
        )
    preceding.commit()
    ordinary_before = tuple(tuple(row) for row in preceding.execute(
        """
        SELECT intervention_id, run_id, continuation_state, resume_attempt_id,
               resume_run_id, acknowledgment_id, status
        FROM interventions
        WHERE intervention_id IN ('intervention-resumed', 'intervention-canceled_by_stop')
        ORDER BY intervention_id
        """
    ))
    preceding.close()

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    for status in ("resumed", "canceled_by_stop"):
        record = migrated.intervention(f"intervention-{status}")
        assert record is not None
        assert record.directed_binding is None
    assert tuple(tuple(row) for row in migrated._connection.execute(
        """
        SELECT intervention_id, run_id, continuation_state, resume_attempt_id,
               resume_run_id, acknowledgment_id, status
        FROM interventions
        WHERE intervention_id IN ('intervention-resumed', 'intervention-canceled_by_stop')
        ORDER BY intervention_id
        """
    )) == ordinary_before
    migrated.close()


@pytest.mark.parametrize("binding_column", ("absent", "null"))
@pytest.mark.parametrize(
    "fault", (
        "incomplete", "ambiguous", "foreign_provider", "foreign_question", "owner",
    ),
)
def test_directed_intervention_binding_migration_rejects_unsafe_rows_and_rolls_back(
    tmp_path, valid_brief, binding_column: str, fault: str,
) -> None:
    """An incomplete or ambiguous binding rejects the whole migration transaction."""
    path = tmp_path / f"{fault}-directed-{binding_column}.sqlite3"
    _seed_previous_directed_intervention_families(
        path, valid_brief, binding_column=binding_column,
    )
    preceding = sqlite3.connect(path)
    if fault == "incomplete":
        preceding.execute(
            "DELETE FROM exchange_reservations WHERE question_id = ?",
            ("question-ready",),
        )
    elif fault == "foreign_provider":
        preceding.execute(
            """
            UPDATE agent_runs SET cli_session_id = 'foreign-provider'
            WHERE run_id = 'source-ready'
            """
        )
    elif fault == "foreign_question":
        preceding.execute(
            """
            UPDATE questions SET routed_to = 'sol'
            WHERE question_id = 'question-ready'
            """
        )
    elif fault == "owner":
        preceding.execute(
            """
            UPDATE interventions SET resume_attempt_id = NULL, resume_run_id = NULL
            WHERE intervention_id = 'intervention-resumed'
            """
        )
    else:
        preceding.execute("DROP INDEX one_unanswered_top_level_question_per_task_revision")
        preceding.execute(
            """
            INSERT INTO questions (
                question_id, session_id, task_id, revision, continuation_generation,
                asked_by, addressed_to, routed_to, text, exchange_id,
                continuation_state, pending_action_json, continuation_pause_id,
                nested_parent_kind, parent_question_id, parent_continuation_pause_id,
                answer_text, answered_by
            )
            SELECT
                'ambiguous-question-ready', session_id, task_id, revision,
                continuation_generation, asked_by, addressed_to, routed_to,
                'Which other exact ready fact applies?', 'ambiguous-exchange-ready',
                continuation_state, pending_action_json, continuation_pause_id,
                nested_parent_kind, parent_question_id, parent_continuation_pause_id,
                answer_text, answered_by
            FROM questions WHERE question_id = 'question-ready'
            """
        )
        preceding.execute(
            """
            INSERT INTO exchange_reservations (
                exchange_id, session_id, task_id, revision, question_id,
                request_key, ordinal, continuation_generation
            )
            SELECT
                'ambiguous-exchange-ready', session_id, task_id, revision,
                'ambiguous-question-ready', 'ambiguous-request-ready',
                ordinal, continuation_generation
            FROM exchange_reservations WHERE question_id = 'question-ready'
            """
        )
    preceding.commit()
    columns_before = tuple(
        row[1] for row in preceding.execute("PRAGMA table_info(interventions)")
    )
    rows_before = tuple(
        tuple(row) for row in preceding.execute(
            "SELECT * FROM interventions ORDER BY intervention_id"
        )
    )
    questions_before = tuple(
        tuple(row) for row in preceding.execute("SELECT * FROM questions ORDER BY question_id")
    )
    reservations_before = tuple(
        tuple(row) for row in preceding.execute(
            "SELECT * FROM exchange_reservations ORDER BY question_id"
        )
    )
    preceding.close()

    with pytest.raises(RuntimeError, match="migration.*unauthenticated"):
        SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    inspected = sqlite3.connect(path)
    assert tuple(
        row[1] for row in inspected.execute("PRAGMA table_info(interventions)")
    ) == columns_before
    assert tuple(
        tuple(row) for row in inspected.execute(
            "SELECT * FROM interventions ORDER BY intervention_id"
        )
    ) == rows_before
    assert tuple(
        tuple(row) for row in inspected.execute("SELECT * FROM questions ORDER BY question_id")
    ) == questions_before
    assert tuple(
        tuple(row) for row in inspected.execute(
            "SELECT * FROM exchange_reservations ORDER BY question_id"
        )
    ) == reservations_before
    inspected.close()


def test_directed_intervention_binding_migration_rolls_back_with_outer_transaction(
    tmp_path, monkeypatch,
) -> None:
    """A failure after the additive column leaves the pre-migration schema intact."""
    path = tmp_path / "directed-binding-rollback.sqlite3"
    initial = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    initial.close()
    legacy = sqlite3.connect(path)
    columns = {
        row[1] for row in legacy.execute("PRAGMA table_info(interventions)")
    }
    if "directed_binding_json" in columns:
        legacy.execute("ALTER TABLE interventions DROP COLUMN directed_binding_json")
        legacy.commit()
    legacy.close()

    migrate = SQLiteStore._migrate_intervention_schema

    def migrate_then_fail(self) -> None:
        migrate(self)
        raise sqlite3.IntegrityError("injected directed binding migration failure")

    monkeypatch.setattr(SQLiteStore, "_migrate_intervention_schema", migrate_then_fail)
    with pytest.raises(
        sqlite3.IntegrityError, match="injected directed binding migration failure",
    ):
        SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    inspected = sqlite3.connect(path)
    assert "directed_binding_json" not in {
        row[1] for row in inspected.execute("PRAGMA table_info(interventions)")
    }
    inspected.close()


def test_intervention_migration_rolls_back_ddl_after_an_injected_failure(
    tmp_path, monkeypatch,
) -> None:
    path = tmp_path / "intervention-rollback.sqlite3"
    legacy = _create_current_schema(path)
    legacy.execute(
        "INSERT INTO sessions (session_id, repo_root, created_at) VALUES (?, ?, ?)",
        ("session-1", "/repo", "2026-08-10T12:00:00Z"),
    )
    legacy.commit()
    legacy.close()

    def fail_intervention_migration(self) -> None:
        raise sqlite3.IntegrityError("injected intervention migration failure")

    monkeypatch.setattr(
        SQLiteStore, "_migrate_intervention_schema", fail_intervention_migration,
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected intervention migration failure"):
        SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    inspected = sqlite3.connect(path)
    assert inspected.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'interventions'"
    ).fetchone() is None
    assert inspected.execute("SELECT session_id FROM sessions").fetchone() == ("session-1",)
    inspected.close()


def test_nested_intervention_reservation_exact_retry_reuses_atomic_child_binding(
    tmp_path, valid_brief,
) -> None:
    """The pre-invocation transaction is exactly idempotent without duplicate events."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "sol-thread-1")
    store.set_fable_session(valid_brief.task_id, valid_brief.revision, "fable-session-1")
    store._connection.execute(
        """
        UPDATE tasks SET approved_at = ?, baseline_id = ?, pending_json = ?
        WHERE task_id = ? AND revision = ?
        """,
        (
            "2026-08-10T12:00:00Z", "baseline-1",
            store_module._encode_json({
                "sol_run_id": "source-run-1", "prompt": "continue exactly",
            }),
            valid_brief.task_id, valid_brief.revision,
        ),
    )
    store.start_agent_run(
        "source-run-1", valid_brief.task_id, valid_brief.revision, "sol",
    )
    store.set_agent_run_session("source-run-1", "sol-thread-1")
    created = store.create_intervention_and_request_stop(
        intervention_id="nested-reservation-intervention",
        session_id="session-1", task_id=valid_brief.task_id,
        revision=valid_brief.revision, expected_source_generation=1,
        message="Keep the nested evidence exact.",
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        run_id="source-run-1",
    )
    store.finish_agent_run("source-run-1", status="interrupted", exit_code=-15)
    store.mark_intervention_ready(created.intervention_id, run_id="source-run-1")
    store.begin_intervention_resume(
        created.intervention_id, expected_resume_generation=created.resume_generation,
        resume_attempt_id="resume-attempt-1", resume_run_id="resume-run-1",
    )
    store.start_agent_run(
        "resume-run-1", valid_brief.task_id, valid_brief.revision, "fable",
    )
    store.set_agent_run_session("resume-run-1", "fable-session-1")
    store.finish_agent_run("resume-run-1", status="completed", exit_code=0)
    event = ConversationEnvelope(
        sender=ConversationActor.FABLE, addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.SOL,
        message_type=ConversationMessageType.QUESTION,
        text="Which exact evidence is verified?", task_id=valid_brief.task_id,
        revision=valid_brief.revision, continuation_generation=created.resume_generation,
        question_id="nested-question-1",
    )
    arguments = {
        "session_id": "session-1", "task_id": valid_brief.task_id,
        "revision": valid_brief.revision,
        "expected_generation": created.resume_generation,
        "question_id": "nested-question-1", "request_key": "nested-request-1",
        "text": "Which exact evidence is verified?", "event": event,
        "intervention_id": created.intervention_id, "child_run_id": "child-run-1",
    }

    first = store.reserve_fable_clarification_evidence_question(**arguments)
    second = store.reserve_fable_clarification_evidence_question(**arguments)

    assert second == first
    bound = store.authenticated_intervention(created.intervention_id)
    assert bound is not None and bound.directed_binding is not None
    assert bound.directed_binding.source_run_id == "child-run-1"
    assert store.agent_run("child-run-1").status == "running"
    assert len(tuple(
        persisted for persisted in store.events_after("session-1", 0)
        if persisted.kind == "conversation"
        and persisted.payload.get("question_id") == "nested-question-1"
    )) == 1


def _seed_predecessorless_active_nested_intervention(
    path: Path,
    valid_brief,
    *,
    binding_shape: str,
    acknowledged: bool,
) -> None:
    """Create the exact pre-predecessor active-child crash image."""
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00.500000Z")
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "sol-thread-1")
    store.set_fable_session(valid_brief.task_id, valid_brief.revision, "fable-session-1")
    store._connection.execute(
        """
        UPDATE tasks SET approved_at = ?, baseline_id = ?, pending_json = ?
        WHERE task_id = ? AND revision = ?
        """,
        (
            "2026-08-10T12:00:00.500000Z",
            "baseline-1",
            store_module._encode_json({
                "sol_run_id": "source-run-1", "prompt": "continue exactly",
            }),
            valid_brief.task_id,
            valid_brief.revision,
        ),
    )
    store.start_agent_run(
        "source-run-1", valid_brief.task_id, valid_brief.revision, "sol",
    )
    store.set_agent_run_session("source-run-1", "sol-thread-1")
    created = store.create_intervention_and_request_stop(
        intervention_id="legacy-active-child",
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        expected_source_generation=1,
        message="Keep the nested evidence exact.",
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        run_id="source-run-1",
    )
    store.finish_agent_run("source-run-1", status="interrupted", exit_code=-15)
    store.mark_intervention_ready(created.intervention_id, run_id="source-run-1")
    store.begin_intervention_resume(
        created.intervention_id,
        expected_resume_generation=created.resume_generation,
        resume_attempt_id="fable-attempt-1",
        resume_run_id="fable-predecessor-1",
    )
    store.start_agent_run(
        "fable-predecessor-1", valid_brief.task_id, valid_brief.revision, "fable",
    )
    store.set_agent_run_session("fable-predecessor-1", "fable-session-1")
    store.finish_agent_run("fable-predecessor-1", status="completed", exit_code=0)
    store.reserve_fable_clarification_evidence_question(
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        expected_generation=created.resume_generation,
        question_id="legacy-active-question",
        request_key="legacy-active-request",
        text="Which exact fact is verified?",
        event=_conversation_question(
            sender=ConversationActor.FABLE,
            addressed_to=ConversationTarget.SOL,
            routed_to=ConversationTarget.SOL,
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            generation=created.resume_generation,
            question_id="legacy-active-question",
            text="Which exact fact is verified?",
        ),
        intervention_id=created.intervention_id,
        child_run_id="legacy-sol-child-1",
    )
    store._connection.execute(
        "UPDATE agent_runs SET started_at = ? WHERE run_id = ?",
        ("2026-08-10T12:00:00.600000Z", "fable-predecessor-1"),
    )
    store._connection.execute(
        "UPDATE agent_runs SET started_at = ? WHERE run_id = ?",
        ("2026-08-10T12:00:00.800000Z", "legacy-sol-child-1"),
    )
    assert store.recover_active_tasks() == store_module.RecoverySummary(0, 1, 1)
    unknown = store.authenticated_intervention(created.intervention_id)
    assert unknown is not None
    if acknowledged:
        ready = store.authorize_retry_after_unknown(
            created.intervention_id,
            expected_resume_generation=unknown.resume_generation,
            acknowledgment_id="legacy-active-ack",
        )
        assert ready.status is store_module.InterventionStatus.READY
    store.close()

    preceding = sqlite3.connect(path)
    if binding_shape == "absent":
        preceding.execute("ALTER TABLE interventions DROP COLUMN directed_binding_json")
    elif binding_shape == "null":
        preceding.execute(
            "UPDATE interventions SET directed_binding_json = NULL "
            "WHERE intervention_id = 'legacy-active-child'"
        )
    elif binding_shape == "stage_less":
        row = preceding.execute(
            "SELECT directed_binding_json FROM interventions "
            "WHERE intervention_id = 'legacy-active-child'"
        ).fetchone()
        assert row is not None
        binding = json.loads(row[0])
        for key in (
            "stage", "next_attempt_id", "next_predecessor_run_id", "next_run_id",
            "next_provider_id", "next_task_state", "next_continuation_state",
        ):
            binding.pop(key)
        preceding.execute(
            "UPDATE interventions SET directed_binding_json = ? "
            "WHERE intervention_id = 'legacy-active-child'",
            (json.dumps(binding, sort_keys=True, separators=(",", ":")),),
        )
    else:
        raise AssertionError(f"unknown binding shape {binding_shape}")
    preceding.commit()
    preceding.close()


@pytest.mark.parametrize("binding_shape", ("absent", "null", "stage_less"))
@pytest.mark.parametrize("acknowledged", (False, True))
def test_active_question_migration_retains_exact_fable_predecessor_through_sol_retry(
    tmp_path,
    valid_brief,
    binding_shape: str,
    acknowledged: bool,
) -> None:
    """A retried Sol child can never become its own next-Fable predecessor."""
    path = tmp_path / f"active-predecessor-{binding_shape}-{acknowledged}.sqlite3"
    _seed_predecessorless_active_nested_intervention(
        path, valid_brief, binding_shape=binding_shape, acknowledged=acknowledged,
    )

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:01Z")
    record = migrated.authenticated_intervention("legacy-active-child")
    assert record is not None and record.directed_binding is not None
    assert record.directed_binding.stage == "active_question"
    assert record.directed_binding.next_predecessor_run_id == "fable-predecessor-1"
    if not acknowledged:
        record = migrated.authorize_retry_after_unknown(
            record.intervention_id,
            expected_resume_generation=record.resume_generation,
            acknowledgment_id="legacy-active-ack",
        )
    migrated.begin_intervention_resume(
        record.intervention_id,
        expected_resume_generation=record.resume_generation,
        resume_attempt_id="sol-retry-attempt",
        resume_run_id="sol-retry-run",
    )
    migrated.start_agent_run(
        "sol-retry-run", valid_brief.task_id, valid_brief.revision, "sol",
    )
    migrated.set_agent_run_session("sol-retry-run", "sol-thread-1")
    migrated.finish_agent_run("sol-retry-run", status="completed", exit_code=0)
    migrated.answer_fable_clarification_evidence_question_and_resume(
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        question_id="legacy-active-question",
        expected_generation=record.resume_generation,
        answer_text="The exact fact is verified.",
        next_fable_run_id="next-fable-after-sol-retry",
        event=_conversation_answer(
            sender=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            generation=record.resume_generation,
            question_id="legacy-active-question",
            text="The exact fact is verified.",
        ),
    )

    promoted = migrated.authenticated_intervention(record.intervention_id)
    assert promoted is not None and promoted.directed_binding is not None
    assert promoted.directed_binding.stage == "next_fable"
    assert promoted.directed_binding.next_predecessor_run_id == "fable-predecessor-1"
    assert promoted.directed_binding.next_predecessor_run_id != "sol-retry-run"


@pytest.mark.parametrize("fault", ("missing", "two", "provider", "status"))
def test_active_question_predecessor_migration_fails_closed_and_rolls_back(
    tmp_path,
    valid_brief,
    fault: str,
) -> None:
    """Only one exact terminal Fable owner may authenticate an old active child."""
    path = tmp_path / f"active-predecessor-{fault}.sqlite3"
    _seed_predecessorless_active_nested_intervention(
        path, valid_brief, binding_shape="null", acknowledged=True,
    )
    preceding = sqlite3.connect(path)
    if fault == "missing":
        preceding.execute("DELETE FROM agent_runs WHERE run_id = 'fable-predecessor-1'")
    elif fault == "two":
        child_rowid = preceding.execute(
            "SELECT rowid FROM agent_runs WHERE run_id = 'legacy-sol-child-1'"
        ).fetchone()[0]
        preceding.execute(
            "UPDATE agent_runs SET rowid = ? WHERE run_id = 'legacy-sol-child-1'",
            (child_rowid + 2,),
        )
        preceding.execute(
            """
            INSERT INTO agent_runs (
                rowid,
                run_id, task_id, revision, agent, cli_session_id,
                started_at, ended_at, exit_code, status
            ) VALUES (?, 'second-fable-predecessor', ?, ?, 'fable', 'fable-session-1',
                      '2026-08-10T12:00:00.700000Z',
                      '2026-08-10T12:00:00.750000Z', 0, 'completed')
            """,
            (child_rowid + 1, valid_brief.task_id, valid_brief.revision),
        )
    elif fault == "provider":
        preceding.execute(
            "UPDATE agent_runs SET cli_session_id = 'wrong-provider' "
            "WHERE run_id = 'fable-predecessor-1'"
        )
    elif fault == "status":
        preceding.execute(
            "UPDATE agent_runs SET status = 'running', ended_at = NULL, exit_code = NULL "
            "WHERE run_id = 'fable-predecessor-1'"
        )
    else:
        raise AssertionError(f"unknown predecessor fault {fault}")
    preceding.commit()
    tables = ("tasks", "interventions", "agent_runs", "questions", "exchange_reservations")
    before = {
        table: tuple(tuple(row) for row in preceding.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in tables
    }
    preceding.close()

    with pytest.raises(RuntimeError, match="migration"):
        SQLiteStore(path, clock=lambda: "2026-08-10T12:00:01Z")

    inspected = sqlite3.connect(path)
    assert {
        table: tuple(tuple(row) for row in inspected.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in tables
    } == before
    inspected.close()


def test_create_intervention_stops_exact_active_run_and_emits_one_user_message(
    tmp_path, valid_brief,
) -> None:
    """Removing any one write from creation must leave no durable intervention."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "sol-thread-1")
    store.start_agent_run("source-run-1", valid_brief.task_id, valid_brief.revision, "sol")
    store.set_agent_run_process(
        "source-run-1", pid=101, process_group_id=101, cli_session_id="sol-thread-1",
    )

    created = store.create_intervention_and_request_stop(
        intervention_id="intervention-1",
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        expected_source_generation=1,
        message="Pause and inspect the current output.",
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        run_id="source-run-1",
    )

    assert created == store.intervention("intervention-1")
    assert created.status is store_module.InterventionStatus.PENDING_STOP
    assert created.continuation_state is TaskState.SOL_RUNNING
    assert created.source_generation == 1
    assert created.resume_generation == 2
    assert created.fable_session_id is None
    assert created.sol_thread_id == "sol-thread-1"
    interrupted = store.get_task(valid_brief.task_id, valid_brief.revision)
    assert interrupted.state is TaskState.INTERRUPTED
    assert interrupted.continuation_state is TaskState.SOL_RUNNING
    assert interrupted.continuation_generation == 2
    assert interrupted.pending == {
        "intervention": {
            "intervention_id": "intervention-1",
            "source_generation": 1,
            "source_run_id": "source-run-1",
            "continuation": None,
        },
    }
    events = store.events_after("session-1", 0)
    assert len(events) == 1
    event = ConversationEnvelope.from_dict(events[0].payload)
    assert event == ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        message_type=ConversationMessageType.INTERVENTION,
        text="Pause and inspect the current output.",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        continuation_generation=1,
    )

    assert store.create_intervention_and_request_stop(
        intervention_id="intervention-1",
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        expected_source_generation=1,
        message="Pause and inspect the current output.",
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        run_id="source-run-1",
    ) == created
    assert store.events_after("session-1", 0) == events


def test_intervention_stops_only_the_exact_active_directed_answer_without_losing_pause(
    tmp_path, valid_brief,
) -> None:
    """A Stop during an answer must retain the unanswered question's exact pause."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "sol-thread-1")
    store.set_fable_session(valid_brief.task_id, valid_brief.revision, "fable-session-1")
    store.pause_for_question(
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        expected_generation=1,
        question_id="question-1",
        asked_by=ConversationActor.SOL,
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        text="Which exact constraint applies?",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"sol_run_id": "sol-run-1", "prompt": "continue exactly"},
        event=ConversationEnvelope(
            sender=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.QUESTION,
            text="Which exact constraint applies?",
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            continuation_generation=1,
            question_id="question-1",
        ),
    )
    before = store.get_task(valid_brief.task_id, valid_brief.revision)
    before_pause = store._connection.execute(
        "SELECT continuation_pause_id FROM tasks WHERE task_id = ? AND revision = ?",
        (valid_brief.task_id, valid_brief.revision),
    ).fetchone()["continuation_pause_id"]
    store.start_agent_run("fable-answer-run-1", valid_brief.task_id, valid_brief.revision, "fable")
    store.set_agent_run_session("fable-answer-run-1", "fable-session-1")

    created = store.create_intervention_and_request_stop(
        intervention_id="directed-answer-intervention",
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        expected_source_generation=1,
        message="Pause the answer and consider this guidance.",
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        run_id="fable-answer-run-1",
    )

    interrupted = store.get_task(valid_brief.task_id, valid_brief.revision)
    assert created.continuation_state is TaskState.SOL_RUNNING
    assert created.source_generation == created.resume_generation == 1
    assert interrupted.state is TaskState.INTERRUPTED
    assert interrupted.continuation_state is TaskState.SOL_RUNNING
    assert interrupted.pending == before.pending
    after_pause = store._connection.execute(
        "SELECT continuation_pause_id FROM tasks WHERE task_id = ? AND revision = ?",
        (valid_brief.task_id, valid_brief.revision),
    ).fetchone()["continuation_pause_id"]
    assert after_pause == before_pause
    question = store.unanswered_question_for_task(valid_brief.task_id, valid_brief.revision)
    assert question is not None
    assert question.question_id == "question-1"


def test_directed_answer_intervention_recovery_authenticates_the_preserved_pause(
    tmp_path, valid_brief,
) -> None:
    """Removing directed-answer recovery authentication would reject this exact paused form on reopen."""
    path = tmp_path / "directed-intervention-recovery.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "sol-thread-1")
    store.set_fable_session(valid_brief.task_id, valid_brief.revision, "fable-session-1")
    store.pause_for_question(
        session_id="session-1", task_id=valid_brief.task_id,
        revision=valid_brief.revision, expected_generation=1, question_id="question-1",
        asked_by=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE, text="Which exact constraint applies?",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"sol_run_id": "sol-run-1", "prompt": "continue exactly"},
        event=ConversationEnvelope(
            sender=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE, message_type=ConversationMessageType.QUESTION,
            text="Which exact constraint applies?", task_id=valid_brief.task_id,
            revision=valid_brief.revision, continuation_generation=1, question_id="question-1",
        ),
    )
    store.start_agent_run("fable-answer-run-1", valid_brief.task_id, valid_brief.revision, "fable")
    store.set_agent_run_session("fable-answer-run-1", "fable-session-1")
    created = store.create_intervention_and_request_stop(
        intervention_id="directed-answer-intervention", session_id="session-1",
        task_id=valid_brief.task_id, revision=valid_brief.revision,
        expected_source_generation=1, message="Pause the answer.",
        addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
        run_id="fable-answer-run-1",
    )
    store.close()

    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    assert reopened.recover_active_tasks().agent_runs_interrupted == 1
    recovered = reopened.authenticated_intervention(created.intervention_id)
    assert recovered is not None
    assert recovered.status is store_module.InterventionStatus.READY
    task = reopened.get_task(valid_brief.task_id, valid_brief.revision)
    question = reopened.unanswered_question_for_task(valid_brief.task_id, valid_brief.revision)
    assert task.state is TaskState.INTERRUPTED
    assert task.continuation_state is TaskState.SOL_RUNNING
    assert question is not None and question.question_id == "question-1"


@pytest.mark.parametrize(
    ("run_id", "agent", "provider_id"),
    (
        ("wrong-answer-run", "sol", "sol-thread-1"),
        ("wrong-provider-run", "fable", "other-fable-session"),
    ),
)
def test_intervention_rejects_mismatched_directed_answer_identity_without_mutation(
    tmp_path, valid_brief, run_id: str, agent: str, provider_id: str,
) -> None:
    """A different paused-answer worker must not consume this question's identity."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "sol-thread-1")
    store.set_fable_session(valid_brief.task_id, valid_brief.revision, "fable-session-1")
    store.pause_for_question(
        session_id="session-1", task_id=valid_brief.task_id,
        revision=valid_brief.revision, expected_generation=1, question_id="question-1",
        asked_by=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE, text="Which exact constraint applies?",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"sol_run_id": "sol-run-1", "prompt": "continue exactly"},
        event=ConversationEnvelope(
            sender=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE, message_type=ConversationMessageType.QUESTION,
            text="Which exact constraint applies?", task_id=valid_brief.task_id,
            revision=valid_brief.revision, continuation_generation=1, question_id="question-1",
        ),
    )
    store.start_agent_run(run_id, valid_brief.task_id, valid_brief.revision, agent)
    store.set_agent_run_session(run_id, provider_id)
    before = store.get_task(valid_brief.task_id, valid_brief.revision)

    with pytest.raises(RuntimeError, match="directed answer|provider identity"):
        store.create_intervention_and_request_stop(
            intervention_id="directed-answer-intervention", session_id="session-1",
            task_id=valid_brief.task_id, revision=valid_brief.revision,
            expected_source_generation=1, message="Pause the answer.",
            addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
            run_id=run_id,
        )

    assert store.intervention("directed-answer-intervention") is None
    assert store.get_task(valid_brief.task_id, valid_brief.revision) == before


def _directed_intervention_claim_for_tamper_matrix(store, valid_brief):
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "provider-1")
    store.set_fable_session(valid_brief.task_id, valid_brief.revision, "provider-1")
    store.reserve_internal_question(
        session_id="session-1", task_id=valid_brief.task_id,
        revision=valid_brief.revision, expected_generation=1,
        question_id="original-question", request_key="original-request",
        asked_by=ConversationActor.SOL,
        addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
        text="Which exact constraint applies?", continuation_state=TaskState.SOL_RUNNING,
        pending_action={"sol_run_id": "sol-run-1", "prompt": "continue exactly"},
        event=ConversationEnvelope(
            sender=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.QUESTION,
            text="Which exact constraint applies?", task_id=valid_brief.task_id,
            revision=valid_brief.revision, continuation_generation=1,
            question_id="original-question",
        ),
    )
    store.start_agent_run(
        "original-source-run", valid_brief.task_id, valid_brief.revision, "fable",
    )
    store.set_agent_run_session("original-source-run", "provider-1")
    created = store.create_intervention_and_request_stop(
        intervention_id="directed-terminal", session_id="session-1",
        task_id=valid_brief.task_id, revision=valid_brief.revision,
        expected_source_generation=1, message="Keep the exact directed answer.",
        addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
        run_id="original-source-run",
    )
    store.finish_agent_run("original-source-run", status="interrupted", exit_code=-15)
    store.mark_intervention_ready(created.intervention_id, run_id=created.run_id)
    store.begin_intervention_resume(
        created.intervention_id, expected_resume_generation=created.resume_generation,
        resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
    )
    return created


def _tamper_directed_intervention_binding(store, mutation: str) -> None:
    if mutation == "source_agent":
        store._connection.execute(
            "UPDATE agent_runs SET agent = 'sol' WHERE run_id = 'original-source-run'"
        )
    elif mutation == "provider":
        store._connection.execute(
            """
            UPDATE agent_runs SET cli_session_id = 'substituted-provider'
            WHERE run_id = 'original-source-run'
            """
        )
    elif mutation == "question":
        store._connection.execute(
            """
            UPDATE interventions
            SET directed_binding_json = json_set(
                directed_binding_json, '$.question_id', 'substituted-question'
            )
            WHERE intervention_id = 'directed-terminal'
            """
        )
    elif mutation == "pause":
        store._connection.execute(
            """
            UPDATE questions SET continuation_pause_id = 'substituted-pause'
            WHERE question_id = 'original-question'
            """
        )
    elif mutation == "route":
        store._connection.execute(
            """
            UPDATE questions SET addressed_to = 'sol', routed_to = 'sol'
            WHERE question_id = 'original-question'
            """
        )
    elif mutation == "reservation_missing":
        store._connection.execute(
            "DELETE FROM exchange_reservations WHERE question_id = 'original-question'"
        )
    elif mutation == "reservation_substituted":
        store._connection.execute(
            """
            UPDATE questions SET exchange_id = 'substituted-exchange'
            WHERE question_id = 'original-question'
            """
        )
        store._connection.execute(
            """
            UPDATE exchange_reservations SET exchange_id = 'substituted-exchange'
            WHERE question_id = 'original-question'
            """
        )
    else:
        raise AssertionError(f"unknown mutation {mutation}")


def _nested_parent_intervention_claim_for_tamper_matrix(store, valid_brief):
    """Build one live nested Sol child under an exact reserved outer question."""
    created = _directed_intervention_claim_for_tamper_matrix(store, valid_brief)
    store._connection.execute(
        """
        UPDATE tasks SET approved_at = ?, baseline_id = ?
        WHERE task_id = ? AND revision = ?
        """,
        ("2026-08-10T12:00:00Z", "baseline-1", valid_brief.task_id, valid_brief.revision),
    )
    store.start_agent_run(
        "resume-run-1", valid_brief.task_id, valid_brief.revision, "fable",
    )
    store.set_agent_run_session("resume-run-1", "provider-1")
    store.finish_agent_run("resume-run-1", status="completed", exit_code=0)
    _, child = store.reserve_fable_answer_evidence_question(
        session_id="session-1", task_id=valid_brief.task_id,
        revision=valid_brief.revision, expected_generation=created.resume_generation,
        outer_question_id="original-question", question_id="nested-question",
        request_key="nested-request", text="Which exact nested fact applies?",
        event=ConversationEnvelope(
            sender=ConversationActor.FABLE, addressed_to=ConversationTarget.SOL,
            routed_to=ConversationTarget.SOL,
            message_type=ConversationMessageType.QUESTION,
            text="Which exact nested fact applies?", task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            continuation_generation=created.resume_generation,
            question_id="nested-question",
        ),
    )
    store.start_agent_run(
        "nested-source-run", valid_brief.task_id, valid_brief.revision, "sol",
    )
    store.set_agent_run_session("nested-source-run", "provider-1")
    store._connection.execute(
        "UPDATE interventions SET directed_binding_json = NULL WHERE intervention_id = ?",
        (created.intervention_id,),
    )
    with store._immediate_transaction():
        rebound = store._bind_nested_intervention_resume_in_transaction(
            store.intervention(created.intervention_id),
            store.get_task(valid_brief.task_id, valid_brief.revision),
        )
    store._connection.execute(
        """
        UPDATE agent_runs SET agent = 'sol', cli_session_id = 'provider-1'
        WHERE run_id = 'original-source-run'
        """
    )
    assert rebound.directed_binding is not None
    assert rebound.directed_binding.parent_question_id == "original-question"
    assert child.parent_question_id == "original-question"
    return rebound


def _next_fable_intervention_stage_for_tamper_matrix(store, valid_brief):
    """Advance one exact nested Sol child through the child-answer CAS only."""
    created = _nested_parent_intervention_claim_for_tamper_matrix(store, valid_brief)
    store.finish_agent_run("nested-source-run", status="completed", exit_code=0)
    store.answer_fable_answer_evidence_question(
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        outer_question_id="original-question",
        question_id="nested-question",
        expected_generation=created.resume_generation,
        answer_text="The exact nested fact is verified.",
        next_fable_run_id="next-fable-run",
        event=ConversationEnvelope(
            sender=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            message_type=ConversationMessageType.ANSWER,
            text="The exact nested fact is verified.",
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            continuation_generation=created.resume_generation,
            reply_to_question_id="nested-question",
        ),
    )
    stage = store.intervention(created.intervention_id)
    assert stage is not None and stage.directed_binding is not None
    assert stage.directed_binding.stage == "next_fable"
    return stage


def _prepare_staged_scope_checkpoint_owner(
    store,
    valid_brief,
    *,
    action: str,
    session_id: str,
    provider_id: str,
    owned,
):
    """Create one real prepared action that can reach a Sol-to-Fable scope answer."""
    parent_pending = {
        "sol_run_id": owned("sol-run-1"),
        "prompt": "continue exactly",
    }
    context = SolResumeContext(
        sol_thread_id=provider_id,
        sol_run_id=parent_pending["sol_run_id"],
        prompt=parent_pending["prompt"],
    )
    approval = store.prepare_approval_action(
        project_id="a" * 32,
        session_id=session_id,
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        generation=41 if action == "approval" else 40,
        payload=ApprovalPayload(
            baseline_id="baseline-1",
            baseline_setting=None,
            scope=None,
        ),
    )
    approval = store.claim_prepared_action(
        approval.preparation_id,
        generation=approval.generation,
    )
    if action == "approval":
        return approval, parent_pending
    if action == "resume":
        store.pause_for_continuation(
            valid_brief.task_id,
            valid_brief.revision,
            expected=TaskState.SOL_RUNNING,
            target=TaskState.INTERRUPTED,
            continuation_state=TaskState.SOL_RUNNING,
            pending=parent_pending,
        )
        approval = store.interrupt_claimed_prepared_action(
            approval.preparation_id,
            generation=approval.generation,
            reason="adapter_interrupted",
        )
        prepared = store.prepare_resume_action(
            project_id="a" * 32,
            session_id=session_id,
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            generation=41,
            payload=ResumePayload(
                continuation=context,
                drift_event=ResumeDriftProjection(
                    status="unchanged",
                    summary="Repository drift was checked.",
                    evidence_hashes=(),
                ),
            ),
            previous_preparation_id=approval.preparation_id,
        )
    else:
        store.complete_prepared_action(
            approval.preparation_id,
            generation=approval.generation,
        )
    if action == "resume":
        pass
    elif action in {"answer", "continuation_message"}:
        store.pause_for_continuation(
            valid_brief.task_id,
            valid_brief.revision,
            expected=TaskState.SOL_RUNNING,
            target=TaskState.AWAITING_USER_INPUT,
            continuation_state=TaskState.SOL_RUNNING,
            pending=parent_pending,
        )
        if action == "answer":
            prepared = store.prepare_answer_action(
                project_id="a" * 32,
                session_id=session_id,
                task_id=valid_brief.task_id,
                revision=valid_brief.revision,
                generation=41,
                payload=AnswerPayload(
                    answer="Continue with the exact approved path.",
                    continuation=context,
                ),
            )
        else:
            prepared = store.prepare_continuation_message_action(
                project_id="a" * 32,
                session_id=session_id,
                task_id=valid_brief.task_id,
                revision=valid_brief.revision,
                generation=41,
                payload=ContinuationMessagePayload(
                    text="Continue with the exact approved path.",
                    addressed_to=ConversationTarget.SOL,
                    routed_to=ConversationTarget.SOL,
                    continuation_generation=1,
                    continuation=context,
                ),
            )
    elif action == "question_answer":
        store.pause_for_question(
            session_id=session_id,
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            expected_generation=1,
            question_id=owned("scope-owner-user-question"),
            asked_by=ConversationActor.SOL,
            addressed_to=ConversationTarget.USER,
            routed_to=ConversationTarget.USER,
            text="Which exact approved option should Sol use?",
            continuation_state=TaskState.SOL_RUNNING,
            pending_action=parent_pending,
            event=_conversation_question(
                sender=ConversationActor.SOL,
                addressed_to=ConversationTarget.USER,
                routed_to=ConversationTarget.USER,
                task_id=valid_brief.task_id,
                revision=valid_brief.revision,
                generation=1,
                question_id=owned("scope-owner-user-question"),
                text="Which exact approved option should Sol use?",
            ),
        )
        prepared = store.prepare_question_answer_action(
            project_id="a" * 32,
            session_id=session_id,
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            generation=41,
            payload=QuestionAnswerPayload(
                question_id=owned("scope-owner-user-question"),
                answer="Use the exact approved option.",
                continuation_generation=1,
                continuation=context,
            ),
        )
    elif action == "exchange_grant":
        attempted = DirectedAgentQuestion(
            addressed_to="fable",
            text="Resolve the exact approved ambiguity.",
            reason="The finite internal exchange allowance is exhausted.",
        )
        store._connection.execute(  # noqa: SLF001 - exact permission fixture
            "UPDATE tasks SET exchange_allowance = 0 WHERE task_id = ? AND revision = ?",
            (valid_brief.task_id, valid_brief.revision),
        )
        store.pause_for_exchange_permission(
            session_id=session_id,
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            expected_generation=1,
            attempted_question=attempted,
            continuation_state=TaskState.SOL_RUNNING,
            pending_action=parent_pending,
            event=_conversation_permission(
                task_id=valid_brief.task_id,
                revision=valid_brief.revision,
                generation=1,
            ),
        )
        prepared = store.prepare_exchange_grant_action(
            project_id="a" * 32,
            session_id=session_id,
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            generation=41,
            payload=ExchangeGrantPayload(
                request_id=owned("scope-owner-grant-request"),
                continuation_generation=1,
                attempted_question=attempted,
                continuation=context,
                parent_mode="top_level",
            ),
        )
    else:
        raise AssertionError(f"unsupported staged checkpoint owner action {action}")
    prepared = store.claim_prepared_action(
        prepared.preparation_id,
        generation=prepared.generation,
    )
    return prepared, parent_pending


def _staged_scope_checkpoint_for_owner_matrix(
    store,
    valid_brief,
    *,
    prefix: str = "",
    preparation_action: str = "approval",
):
    """Build one accepted scope checkpoint with a still-running exact Fable owner."""
    def owned(value: str) -> str:
        return value if not prefix else f"{prefix}-{value}"

    session_id = owned("session-1")
    provider_id = owned("provider-1")
    store.create_session(session_id, "/repo")
    store.save_task(session_id, valid_brief, TaskState.AWAITING_USER_APPROVAL)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, provider_id)
    store.set_fable_session(valid_brief.task_id, valid_brief.revision, provider_id)
    prepared, parent_pending = _prepare_staged_scope_checkpoint_owner(
        store,
        valid_brief,
        action=preparation_action,
        session_id=session_id,
        provider_id=provider_id,
        owned=owned,
    )
    question_generation = store.get_task(
        valid_brief.task_id, valid_brief.revision,
    ).continuation_generation
    _, parent = store.reserve_internal_question(
        session_id=session_id,
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        expected_generation=question_generation,
        question_id=owned("scope-owner-parent"),
        request_key=owned("scope-owner-parent-request"),
        asked_by=ConversationActor.SOL,
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        text="May the bounded scope include one exact path?",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action=parent_pending,
        event=_conversation_question(
            sender=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            generation=question_generation,
            question_id=owned("scope-owner-parent"),
            text="May the bounded scope include one exact path?",
        ),
    )
    store.start_agent_run(
        owned("scope-owner-source"), valid_brief.task_id, valid_brief.revision, "fable",
    )
    store.set_agent_run_session(owned("scope-owner-source"), provider_id)
    created = store.create_intervention_and_request_stop(
        intervention_id=owned("scope-owner-intervention"),
        session_id=session_id,
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        expected_source_generation=question_generation,
        message="Keep the scope answer exact.",
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        run_id=owned("scope-owner-source"),
    )
    store.finish_agent_run(
        owned("scope-owner-source"), status="interrupted", exit_code=-15,
    )
    store.mark_intervention_ready(created.intervention_id, run_id=created.run_id)
    store.begin_intervention_resume(
        created.intervention_id,
        expected_resume_generation=created.resume_generation,
        resume_attempt_id=owned("scope-owner-attempt"),
        resume_run_id=owned("scope-owner-resume"),
    )
    store.start_agent_run(
        owned("scope-owner-resume"), valid_brief.task_id, valid_brief.revision, "fable",
    )
    store.set_agent_run_session(owned("scope-owner-resume"), provider_id)
    store.finish_agent_run(
        owned("scope-owner-resume"), status="completed", exit_code=0,
    )
    _, child = store.reserve_fable_answer_evidence_question(
        session_id=session_id,
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        expected_generation=created.resume_generation,
        outer_question_id=parent.question_id,
        question_id=owned("scope-owner-child"),
        request_key=owned("scope-owner-child-request"),
        text="Which exact fact permits that bounded path?",
        event=_conversation_question(
            sender=ConversationActor.FABLE,
            addressed_to=ConversationTarget.SOL,
            routed_to=ConversationTarget.SOL,
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            generation=created.resume_generation,
            question_id=owned("scope-owner-child"),
            text="Which exact fact permits that bounded path?",
        ),
        intervention_id=created.intervention_id,
        child_run_id=owned("scope-owner-child-run"),
    )
    store.finish_agent_run(
        owned("scope-owner-child-run"), status="completed", exit_code=0,
    )
    store.answer_fable_answer_evidence_question(
        session_id=session_id,
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        outer_question_id=parent.question_id,
        question_id=child.question_id,
        expected_generation=created.resume_generation,
        answer_text="The exact fact permits only the bounded path.",
        next_fable_run_id=owned("scope-owner-next-fable"),
        event=_conversation_answer(
            sender=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            generation=created.resume_generation,
            question_id=child.question_id,
            text="The exact fact permits only the bounded path.",
        ),
    )
    stage = store.authenticated_intervention(created.intervention_id)
    assert stage is not None and stage.directed_binding is not None
    assert stage.directed_binding.stage == "next_fable"
    clarification = FableClarification.from_dict({
        "status": "answered",
        "answer": "Add only the explicitly bounded path.",
        "reasoning": "The exact evidence supports one bounded scope revision.",
        "confidence": 0.9,
        "scope_changed": True,
        "revised_brief": replace(
            valid_brief,
            revision=2,
            allowed_paths=(*valid_brief.allowed_paths, "scope-extra.txt"),
        ).to_dict(),
        "question_for_user": None,
        "directed_question": None,
    })
    checkpoint = store.checkpoint_directed_fable_scope_answer(
        prepared,
        question_id=parent.question_id,
        continuation_generation=created.resume_generation,
        clarification=clarification,
        completed_next_fable_intervention_id=stage.intervention_id,
        completed_next_fable_run_id=stage.directed_binding.next_run_id,
    )
    return prepared, stage, checkpoint, clarification, parent_pending


def _scope_owner_fixture_snapshot(store) -> dict[str, tuple[tuple[object, ...], ...]]:
    tables = (
        "tasks",
        "settings",
        "events",
        "agent_runs",
        "interventions",
        "prepared_actions",
        "questions",
        "exchange_reservations",
        "directed_fable_answer_checkpoints",
    )
    return {
        table: tuple(tuple(row) for row in store._connection.execute(  # noqa: SLF001
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in tables
    }


def _consume_staged_scope_checkpoint(
    store,
    valid_brief,
    *,
    prepared,
    stage,
    clarification,
    parent_pending,
):
    binding = stage.directed_binding
    assert binding is not None and binding.next_run_id is not None
    revised = replace(
        valid_brief,
        revision=2,
        allowed_paths=(*valid_brief.allowed_paths, "scope-extra.txt"),
    )
    return store.save_scope_revision(
        "session-1",
        revised,
        fable_session_id="provider-1",
        sol_thread_id="provider-1",
        correction_count=0,
        continuation_state=TaskState.SOL_RUNNING,
        pending={"answer": clarification.answer, **parent_pending},
        baseline_id="baseline-1",
        setting=("agent_bridge.baseline.task-1.2", {"baseline_id": "baseline-1"}),
        directed_checkpoint=prepared,
        clarification=clarification,
        completed_next_fable_intervention_id=stage.intervention_id,
        completed_next_fable_run_id=binding.next_run_id,
        completed_next_fable_exit_code=0,
        answered_question_id="scope-owner-parent",
        answered_question_generation=stage.resume_generation,
        answered_pending=parent_pending,
        answer_event=_conversation_answer(
            sender=ConversationActor.FABLE,
            addressed_to=ConversationTarget.SOL,
            routed_to=ConversationTarget.SOL,
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            generation=stage.resume_generation,
            question_id="scope-owner-parent",
            text=clarification.answer,
        ),
    )


_SCOPE_CHECKPOINT_OWNER_KEYS = {
    "intervention_id",
    "run_id",
    "resume_attempt_id",
    "provider_id",
    "session_id",
    "task_id",
    "revision",
    "generation",
    "question_id",
    "parent_question_id",
    "preparation_id",
}


def test_staged_scope_checkpoint_persists_its_complete_exact_owner(
    tmp_path,
    valid_brief,
) -> None:
    """The accepted checkpoint itself names the only stage it may consume."""
    store = SQLiteStore(
        tmp_path / "scope-checkpoint-owner.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    prepared, stage, checkpoint, _, _ = _staged_scope_checkpoint_for_owner_matrix(
        store, valid_brief,
    )
    binding = stage.directed_binding
    assert binding is not None
    row = store._connection.execute(  # noqa: SLF001 - persisted owner contract
        "SELECT stage_kind, stage_owner_json FROM directed_fable_answer_checkpoints "
        "WHERE preparation_id = ? AND question_id = ?",
        (prepared.preparation_id, checkpoint.question_id),
    ).fetchone()
    assert row is not None
    assert row["stage_kind"] == "next_fable"
    owner = json.loads(row["stage_owner_json"])
    assert set(owner) == _SCOPE_CHECKPOINT_OWNER_KEYS
    assert owner == {
        "intervention_id": stage.intervention_id,
        "run_id": binding.next_run_id,
        "resume_attempt_id": stage.resume_attempt_id,
        "provider_id": binding.next_provider_id,
        "session_id": stage.session_id,
        "task_id": stage.task_id,
        "revision": stage.revision,
        "generation": stage.resume_generation,
        "question_id": binding.question_id,
        "parent_question_id": binding.parent_question_id,
        "preparation_id": prepared.preparation_id,
    }


@pytest.mark.parametrize("operation", ("recover", "save"))
@pytest.mark.parametrize(
    "mutation",
    (
        "missing_owner",
        "missing_owner_field",
        "stage_kind",
        "intervention",
        "missing_intervention",
        "run",
        "missing_run",
        "attempt",
        "provider",
        "session",
        "task",
        "revision",
        "generation",
        "question",
        "parent",
        "preparation",
        "missing_preparation",
    ),
)
def test_staged_scope_checkpoint_owner_tampering_fails_closed_and_rolls_back(
    tmp_path,
    valid_brief,
    operation: str,
    mutation: str,
) -> None:
    """Recovery and consumption accept only the checkpoint's exact stage owner."""
    store = SQLiteStore(
        tmp_path / f"scope-owner-{operation}-{mutation}.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    prepared, stage, _, clarification, parent_pending = (
        _staged_scope_checkpoint_for_owner_matrix(store, valid_brief)
    )
    owner_mutations = {
        "intervention": ("intervention_id", "substituted-intervention"),
        "run": ("run_id", "substituted-run"),
        "attempt": ("resume_attempt_id", "substituted-attempt"),
        "provider": ("provider_id", "substituted-provider"),
        "session": ("session_id", "substituted-session"),
        "task": ("task_id", "substituted-task"),
        "revision": ("revision", 999),
        "generation": ("generation", 999),
        "question": ("question_id", "substituted-question"),
        "parent": ("parent_question_id", "substituted-parent"),
        "preparation": ("preparation_id", "substituted-preparation"),
    }
    if mutation == "missing_owner":
        store._connection.execute(  # noqa: SLF001
            "UPDATE directed_fable_answer_checkpoints SET stage_owner_json = NULL"
        )
    elif mutation == "missing_owner_field":
        store._connection.execute(  # noqa: SLF001
            "UPDATE directed_fable_answer_checkpoints "
            "SET stage_owner_json = json_remove(stage_owner_json, '$.provider_id')"
        )
    elif mutation == "stage_kind":
        store._connection.execute(  # noqa: SLF001
            "UPDATE directed_fable_answer_checkpoints SET stage_kind = 'prepared_answer'"
        )
    elif mutation == "missing_intervention":
        store._connection.execute(  # noqa: SLF001
            "DELETE FROM interventions WHERE intervention_id = ?",
            (stage.intervention_id,),
        )
    elif mutation == "missing_run":
        assert stage.directed_binding is not None
        store._connection.execute(  # noqa: SLF001
            "DELETE FROM agent_runs WHERE run_id = ?",
            (stage.directed_binding.next_run_id,),
        )
    elif mutation == "missing_preparation":
        store._connection.execute(  # noqa: SLF001
            "DELETE FROM prepared_actions WHERE preparation_id = ?",
            (prepared.preparation_id,),
        )
    else:
        key, value = owner_mutations[mutation]
        store._connection.execute(  # noqa: SLF001
            "UPDATE directed_fable_answer_checkpoints "
            "SET stage_owner_json = json_set(stage_owner_json, ?, json(?))",
            (f"$.{key}", json.dumps(value)),
        )
    before = _scope_owner_fixture_snapshot(store)

    with pytest.raises(RuntimeError, match="checkpoint|stage|owner|intervention|preparation"):
        if operation == "recover":
            store.recoverable_next_fable_scope_stage(prepared)
        else:
            _consume_staged_scope_checkpoint(
                store,
                valid_brief,
                prepared=prepared,
                stage=stage,
                clarification=clarification,
                parent_pending=parent_pending,
            )

    assert _scope_owner_fixture_snapshot(store) == before


def test_exact_staged_scope_checkpoint_reopens_and_consumes_once(
    tmp_path,
    valid_brief,
) -> None:
    """An untampered checkpoint survives reopen and consumes its one owner once."""
    path = tmp_path / "scope-checkpoint-replay.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    prepared, stage, _, clarification, parent_pending = (
        _staged_scope_checkpoint_for_owner_matrix(store, valid_brief)
    )
    expected_stage = (stage.intervention_id, stage.directed_binding.next_run_id)  # type: ignore[union-attr]
    store.close()

    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:01Z")
    assert reopened.recoverable_next_fable_scope_stage(prepared) == expected_stage
    saved = _consume_staged_scope_checkpoint(
        reopened,
        valid_brief,
        prepared=prepared,
        stage=stage,
        clarification=clarification,
        parent_pending=parent_pending,
    )
    assert saved.revision == 2
    assert reopened.directed_fable_answer_checkpoint(prepared) is None
    assert reopened.intervention(stage.intervention_id).status is (  # type: ignore[union-attr]
        store_module.InterventionStatus.RESUMED
    )
    assert reopened.agent_run(expected_stage[1]).status == "completed"
    after_first = _scope_owner_fixture_snapshot(reopened)

    with pytest.raises(RuntimeError, match="checkpoint|stage|completion"):
        _consume_staged_scope_checkpoint(
            reopened,
            valid_brief,
            prepared=prepared,
            stage=stage,
            clarification=clarification,
            parent_pending=parent_pending,
        )
    assert _scope_owner_fixture_snapshot(reopened) == after_first
    reopened.close()

    audited = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:02Z")
    assert audited.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)
    assert _scope_owner_fixture_snapshot(audited) == after_first
    audited.close()


def test_staged_scope_checkpoint_owner_migration_is_exact_and_byte_idempotent(
    tmp_path,
    valid_brief,
) -> None:
    """The preceding checkpoint schema backfills only its one authenticated stage."""
    path = tmp_path / "scope-checkpoint-owner-migration.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    prepared, stage, checkpoint, _, _ = _staged_scope_checkpoint_for_owner_matrix(
        store, valid_brief,
    )
    expected_stage = (stage.intervention_id, stage.directed_binding.next_run_id)  # type: ignore[union-attr]
    store.close()

    legacy = sqlite3.connect(path)
    legacy.execute(
        """
        CREATE TABLE preceding_directed_fable_answer_checkpoints AS
        SELECT preparation_id, project_id, session_id, task_id, revision,
               question_id, continuation_generation, clarification_json, status
        FROM directed_fable_answer_checkpoints
        """
    )
    legacy.execute("DROP TABLE directed_fable_answer_checkpoints")
    legacy.execute(
        "ALTER TABLE preceding_directed_fable_answer_checkpoints "
        "RENAME TO directed_fable_answer_checkpoints"
    )
    legacy.commit()
    legacy.close()

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:01Z")
    row = migrated._connection.execute(  # noqa: SLF001 - migration contract
        "SELECT stage_kind, stage_owner_json "
        "FROM directed_fable_answer_checkpoints WHERE preparation_id = ?",
        (prepared.preparation_id,),
    ).fetchone()
    assert row is not None and row["stage_kind"] == "next_fable"
    assert set(json.loads(row["stage_owner_json"])) == _SCOPE_CHECKPOINT_OWNER_KEYS
    assert migrated.recoverable_next_fable_scope_stage(prepared) == expected_stage
    assert migrated.directed_fable_answer_checkpoint(prepared) == checkpoint
    migrated.close()

    migrated_bytes = path.read_bytes()
    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:02Z")
    assert reopened.recoverable_next_fable_scope_stage(prepared) == expected_stage
    reopened.close()
    assert path.read_bytes() == migrated_bytes


def _restore_pre_stage_owner_checkpoint_schema(path: Path) -> None:
    """Restore only the immediately preceding checkpoint table shape."""
    preceding = sqlite3.connect(path)
    preceding.execute(
        """
        CREATE TABLE preceding_directed_fable_answer_checkpoints AS
        SELECT preparation_id, project_id, session_id, task_id, revision,
               question_id, continuation_generation, clarification_json, status
        FROM directed_fable_answer_checkpoints
        """
    )
    preceding.execute("DROP TABLE directed_fable_answer_checkpoints")
    preceding.execute(
        "ALTER TABLE preceding_directed_fable_answer_checkpoints "
        "RENAME TO directed_fable_answer_checkpoints"
    )
    preceding.commit()
    preceding.close()


@pytest.mark.parametrize("terminal", ("resumed", "canceled_by_stop"))
def test_terminal_staged_scope_checkpoint_migration_preserves_exact_owner(
    tmp_path,
    valid_brief,
    terminal: str,
) -> None:
    """Consumed staged checkpoints retain their one authenticated terminal owner."""
    path = tmp_path / f"terminal-scope-owner-{terminal}.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    prepared, stage, _, clarification, parent_pending = (
        _staged_scope_checkpoint_for_owner_matrix(store, valid_brief)
    )
    binding = stage.directed_binding
    assert binding is not None and binding.next_run_id is not None
    if terminal == "resumed":
        _consume_staged_scope_checkpoint(
            store,
            valid_brief,
            prepared=prepared,
            stage=stage,
            clarification=clarification,
            parent_pending=parent_pending,
        )
        expected_status = store_module.InterventionStatus.RESUMED
        expected_run_status = "completed"
    else:
        store.cancel_intervention_by_stop(
            stage.intervention_id,
            expected_resume_generation=stage.resume_generation,
        )
        expected_status = store_module.InterventionStatus.CANCELED_BY_STOP
        expected_run_status = "interrupted"
    store.close()
    _restore_pre_stage_owner_checkpoint_schema(path)

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:01Z")
    row = migrated._connection.execute(  # noqa: SLF001 - migration contract
        "SELECT stage_kind, stage_owner_json, status "
        "FROM directed_fable_answer_checkpoints WHERE preparation_id = ?",
        (prepared.preparation_id,),
    ).fetchone()
    assert row is not None
    assert (row["stage_kind"], row["status"]) == ("next_fable", "CONSUMED")
    owner = json.loads(row["stage_owner_json"])
    assert owner["intervention_id"] == stage.intervention_id
    assert owner["run_id"] == binding.next_run_id
    assert migrated.authenticated_intervention(stage.intervention_id).status is expected_status  # type: ignore[union-attr]
    assert migrated.agent_run(binding.next_run_id).status == expected_run_status
    assert migrated._directed_fable_answer_checkpoint_reasons("a" * 32) == set()  # noqa: SLF001
    migrated.close()

    migrated_bytes = path.read_bytes()
    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:02Z")
    assert reopened._directed_fable_answer_checkpoint_reasons("a" * 32) == set()  # noqa: SLF001
    reopened.close()
    assert path.read_bytes() == migrated_bytes


@pytest.mark.parametrize("terminal", ("pending", "resumed", "canceled_by_stop"))
def test_resume_owned_staged_scope_checkpoint_migration_preserves_exact_owner(
    tmp_path,
    valid_brief,
    terminal: str,
) -> None:
    """A real resumed workflow may own every valid staged checkpoint form."""
    path = tmp_path / f"resume-scope-owner-{terminal}.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    prepared, stage, _, clarification, parent_pending = (
        _staged_scope_checkpoint_for_owner_matrix(
            store,
            valid_brief,
            preparation_action="resume",
        )
    )
    binding = stage.directed_binding
    assert prepared.action == "resume"
    assert binding is not None and binding.next_run_id is not None
    if terminal == "pending":
        expected_checkpoint_status = "PENDING"
        expected_intervention_status = store_module.InterventionStatus.RESUMING
        expected_run_status = "running"
    elif terminal == "resumed":
        _consume_staged_scope_checkpoint(
            store,
            valid_brief,
            prepared=prepared,
            stage=stage,
            clarification=clarification,
            parent_pending=parent_pending,
        )
        expected_checkpoint_status = "CONSUMED"
        expected_intervention_status = store_module.InterventionStatus.RESUMED
        expected_run_status = "completed"
    else:
        store.cancel_intervention_by_stop(
            stage.intervention_id,
            expected_resume_generation=stage.resume_generation,
        )
        expected_checkpoint_status = "CONSUMED"
        expected_intervention_status = store_module.InterventionStatus.CANCELED_BY_STOP
        expected_run_status = "interrupted"
    store.close()
    _restore_pre_stage_owner_checkpoint_schema(path)

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:01Z")
    row = migrated._connection.execute(  # noqa: SLF001 - migration contract
        "SELECT stage_kind, stage_owner_json, status "
        "FROM directed_fable_answer_checkpoints WHERE preparation_id = ?",
        (prepared.preparation_id,),
    ).fetchone()
    assert row is not None
    assert (row["stage_kind"], row["status"]) == (
        "next_fable",
        expected_checkpoint_status,
    )
    assert json.loads(row["stage_owner_json"])["preparation_id"] == (
        prepared.preparation_id
    )
    assert migrated.prepared_action(prepared.preparation_id).action == "resume"
    assert migrated.authenticated_intervention(stage.intervention_id).status is (  # type: ignore[union-attr]
        expected_intervention_status
    )
    assert migrated.agent_run(binding.next_run_id).status == expected_run_status
    assert migrated._directed_fable_answer_checkpoint_reasons("a" * 32) == set()  # noqa: SLF001


@pytest.mark.parametrize(
    "action",
    ("answer", "continuation_message", "question_answer", "exchange_grant"),
)
def test_pending_staged_scope_checkpoint_migration_accepts_other_reachable_owners(
    tmp_path,
    valid_brief,
    action: str,
) -> None:
    """Every other real prepared workflow that can ask Fable remains migratable."""
    path = tmp_path / f"{action}-scope-owner.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    prepared, stage, _, _, _ = _staged_scope_checkpoint_for_owner_matrix(
        store,
        valid_brief,
        preparation_action=action,
    )
    binding = stage.directed_binding
    assert prepared.action == action
    assert binding is not None and binding.next_run_id is not None
    store.close()
    _restore_pre_stage_owner_checkpoint_schema(path)

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:01Z")
    row = migrated._connection.execute(  # noqa: SLF001 - migration contract
        "SELECT stage_kind, stage_owner_json, status "
        "FROM directed_fable_answer_checkpoints WHERE preparation_id = ?",
        (prepared.preparation_id,),
    ).fetchone()
    assert row is not None
    assert (row["stage_kind"], row["status"]) == ("next_fable", "PENDING")
    assert json.loads(row["stage_owner_json"])["preparation_id"] == (
        prepared.preparation_id
    )
    assert migrated.recoverable_next_fable_scope_stage(prepared) == (
        stage.intervention_id,
        binding.next_run_id,
    )
    assert migrated._directed_fable_answer_checkpoint_reasons("a" * 32) == set()  # noqa: SLF001


@pytest.mark.parametrize(
    "fault", ("missing_history", "missing_run", "ambiguous_history"),
)
def test_staged_scope_checkpoint_migration_requires_unique_positive_history(
    tmp_path,
    valid_brief,
    fault: str,
) -> None:
    """No live match is not evidence of an ordinary prepared-answer checkpoint."""
    path = tmp_path / f"scope-owner-migration-{fault}.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    _, stage, _, _, _ = _staged_scope_checkpoint_for_owner_matrix(store, valid_brief)
    store.close()
    _restore_pre_stage_owner_checkpoint_schema(path)

    preceding = sqlite3.connect(path)
    if fault == "missing_history":
        preceding.execute(
            "DELETE FROM interventions WHERE intervention_id = ?",
            (stage.intervention_id,),
        )
    elif fault == "missing_run":
        assert stage.directed_binding is not None
        preceding.execute(
            "DELETE FROM agent_runs WHERE run_id = ?",
            (stage.directed_binding.next_run_id,),
        )
    else:
        preceding.execute(
            """
            INSERT INTO interventions (
                intervention_id, session_id, task_id, revision, addressed_to,
                routed_to, message, run_id, continuation_state, source_generation,
                resume_generation, fable_session_id, sol_thread_id,
                resume_attempt_id, resume_run_id, acknowledgment_id, status,
                directed_binding_json, created_at
            )
            SELECT 'ambiguous-scope-owner', session_id, task_id, revision,
                   addressed_to, routed_to, message, run_id, continuation_state,
                   source_generation, resume_generation, fable_session_id,
                   sol_thread_id, resume_attempt_id, resume_run_id,
                   acknowledgment_id, status, directed_binding_json, created_at
            FROM interventions WHERE intervention_id = ?
            """,
            (stage.intervention_id,),
        )
    preceding.commit()
    schema_before = tuple(
        row[1] for row in preceding.execute(
            "PRAGMA table_info(directed_fable_answer_checkpoints)"
        )
    )
    unrelated_before = tuple(
        tuple(row) for row in preceding.execute("SELECT * FROM events ORDER BY sequence")
    )
    preceding.close()
    bytes_before = path.read_bytes()

    with pytest.raises(
        RuntimeError,
        match="checkpoint stage migration|migration is unauthenticated",
    ):
        SQLiteStore(path, clock=lambda: "2026-08-10T12:00:01Z")

    inspected = sqlite3.connect(path)
    assert tuple(
        row[1] for row in inspected.execute(
            "PRAGMA table_info(directed_fable_answer_checkpoints)"
        )
    ) == schema_before
    assert tuple(
        tuple(row) for row in inspected.execute("SELECT * FROM events ORDER BY sequence")
    ) == unrelated_before
    inspected.close()
    assert path.read_bytes() == bytes_before


@pytest.mark.parametrize(
    "fault",
    (
        "missing_preparation",
        "substituted_preparation",
        "wrong_project",
        "wrong_task",
        "wrong_revision",
        "wrong_generation",
        "wrong_question",
        "wrong_clarification",
        "duplicate_checkpoint",
        "duplicate_preparation",
    ),
)
def test_staged_scope_checkpoint_migration_authenticates_inverse_exact_identity(
    tmp_path,
    valid_brief,
    fault: str,
) -> None:
    """One legacy checkpoint and one exact preparation may own one exact stage."""
    path = tmp_path / f"scope-owner-inverse-{fault}.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    prepared, _, _, _, _ = _staged_scope_checkpoint_for_owner_matrix(
        store, valid_brief,
    )
    store.close()
    _restore_pre_stage_owner_checkpoint_schema(path)

    preceding = sqlite3.connect(path)
    prepared_row = preceding.execute(
        "SELECT * FROM prepared_actions WHERE preparation_id = ?",
        (prepared.preparation_id,),
    ).fetchone()
    assert prepared_row is not None
    prepared_columns = tuple(
        row[1] for row in preceding.execute("PRAGMA table_info(prepared_actions)")
    )

    def clone_preparation(
        preparation_id: str, *, project_id: str | None = None,
    ) -> None:
        values = list(prepared_row)
        values[prepared_columns.index("preparation_id")] = preparation_id
        values[prepared_columns.index("status")] = "INTERRUPTED"
        values[prepared_columns.index("reason")] = "adapter_interrupted"
        if project_id is not None:
            values[prepared_columns.index("project_id")] = project_id
        placeholders = ", ".join("?" for _ in prepared_columns)
        preceding.execute(
            f"INSERT INTO prepared_actions ({', '.join(prepared_columns)}) "
            f"VALUES ({placeholders})",
            tuple(values),
        )

    if fault == "missing_preparation":
        preceding.execute(
            "DELETE FROM prepared_actions WHERE preparation_id = ?",
            (prepared.preparation_id,),
        )
    elif fault == "substituted_preparation":
        clone_preparation("substituted-preparation")
        preceding.execute(
            "UPDATE directed_fable_answer_checkpoints SET preparation_id = ?",
            ("substituted-preparation",),
        )
    elif fault == "wrong_project":
        preceding.execute(
            "UPDATE directed_fable_answer_checkpoints SET project_id = ?",
            ("b" * 32,),
        )
    elif fault == "wrong_task":
        preceding.execute(
            "UPDATE directed_fable_answer_checkpoints SET task_id = 'wrong-task'"
        )
    elif fault == "wrong_revision":
        preceding.execute(
            "UPDATE directed_fable_answer_checkpoints SET revision = 999"
        )
    elif fault == "wrong_generation":
        preceding.execute(
            "UPDATE directed_fable_answer_checkpoints "
            "SET continuation_generation = continuation_generation + 10"
        )
    elif fault == "wrong_question":
        preceding.execute(
            "UPDATE directed_fable_answer_checkpoints "
            "SET question_id = 'scope-owner-child'"
        )
    elif fault == "wrong_clarification":
        preceding.execute(
            "UPDATE directed_fable_answer_checkpoints SET clarification_json = json_set("
            "clarification_json, '$.revised_brief.task_id', 'wrong-task')"
        )
    elif fault == "duplicate_checkpoint":
        preceding.execute(
            "INSERT INTO directed_fable_answer_checkpoints "
            "SELECT * FROM directed_fable_answer_checkpoints"
        )
    elif fault == "duplicate_preparation":
        clone_preparation("duplicate-preparation")
    else:
        raise AssertionError(f"unknown inverse migration fault {fault}")
    preceding.commit()
    schema_before = tuple(
        row[1] for row in preceding.execute(
            "PRAGMA table_info(directed_fable_answer_checkpoints)"
        )
    )
    events_before = tuple(
        tuple(row) for row in preceding.execute("SELECT * FROM events ORDER BY sequence")
    )
    preceding.close()
    bytes_before = path.read_bytes()

    with pytest.raises(RuntimeError, match="migration|checkpoint|preparation"):
        SQLiteStore(path, clock=lambda: "2026-08-10T12:00:01Z")

    inspected = sqlite3.connect(path)
    assert tuple(
        row[1] for row in inspected.execute(
            "PRAGMA table_info(directed_fable_answer_checkpoints)"
        )
    ) == schema_before
    assert tuple(
        tuple(row) for row in inspected.execute("SELECT * FROM events ORDER BY sequence")
    ) == events_before
    inspected.close()
    assert path.read_bytes() == bytes_before


def test_staged_scope_checkpoint_migration_pages_rows_and_limits_owner_candidates(
    tmp_path,
    valid_brief,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Migration retains no unbounded row or ambiguity collection."""
    path = tmp_path / "paged-scope-owner-migration.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    for index in range(6):
        _staged_scope_checkpoint_for_owner_matrix(
            store,
            replace(valid_brief, task_id=f"paged-task-{index}"),
            prefix=f"paged-{index}",
        )
    store.close()
    _restore_pre_stage_owner_checkpoint_schema(path)

    statements: list[str] = []
    original_connect = store_module.sqlite3.connect

    def observed_connect(*args, **kwargs):
        observed = original_connect(*args, **kwargs)
        observed.set_trace_callback(statements.append)
        return observed

    monkeypatch.setattr(store_module, "_STARTUP_RECOVERY_BATCH_SIZE", 2)
    monkeypatch.setattr(store_module.sqlite3, "connect", observed_connect)
    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:01Z")
    assert migrated._directed_fable_answer_checkpoint_reasons("a" * 32) == set()  # noqa: SLF001
    migrated.close()

    page_reads = [
        statement for statement in statements
        if "FROM directed_fable_answer_checkpoints" in statement
        and "WHERE rowid >" in statement
    ]
    candidate_reads = [
        statement for statement in statements
        if statement.lstrip().startswith("SELECT * FROM interventions")
        and "status IN" in statement
        and "json_extract" in statement
        and "canceled_by_stop" in statement
    ]
    inverse_reads = [
        statement for statement in statements
        if statement.lstrip().startswith(
            "SELECT rowid FROM directed_fable_answer_checkpoints"
        )
        and "continuation_generation =" in statement
    ]
    preparation_reads = [
        statement for statement in statements
        if statement.lstrip().startswith("SELECT rowid, * FROM prepared_actions")
        and "payload_json =" in statement
        and "previous_preparation_id IS" in statement
    ]
    assert len(page_reads) == 4
    assert all("LIMIT 2" in statement for statement in page_reads)
    assert candidate_reads
    assert all("LIMIT 2" in statement for statement in candidate_reads)
    assert inverse_reads
    assert all("LIMIT 2" in statement for statement in inverse_reads)
    assert preparation_reads
    assert all("ORDER BY rowid LIMIT 2" in statement for statement in preparation_reads)


def test_child_answer_cas_preallocates_the_exact_next_fable_run(
    tmp_path, valid_brief,
) -> None:
    """The successor run row exists before any next-Fable provider spawn."""
    store = SQLiteStore(
        tmp_path / "next-fable-preallocated.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    stage = _next_fable_intervention_stage_for_tamper_matrix(store, valid_brief)
    binding = stage.directed_binding
    assert binding is not None
    successor = store.agent_run(binding.next_run_id)
    assert successor.task_id == valid_brief.task_id
    assert successor.revision == valid_brief.revision
    assert successor.agent == ConversationTarget.FABLE.value
    assert successor.cli_session_id == binding.next_provider_id
    assert successor.status == "running"


def test_completed_next_fable_stage_atomically_terminalizes_its_intervention(
    tmp_path, valid_brief,
) -> None:
    """A completed next Fable result cannot leave the intervention resumable."""
    store = SQLiteStore(
        tmp_path / "next-fable-terminal.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    stage = _next_fable_intervention_stage_for_tamper_matrix(store, valid_brief)
    binding = stage.directed_binding
    assert binding is not None and binding.next_run_id is not None
    store.start_next_fable_intervention_stage(
        stage.intervention_id, run_id=binding.next_run_id,
    )

    terminal = store.finish_next_fable_intervention_stage(
        stage.intervention_id, run_id=binding.next_run_id, exit_code=0,
    )

    assert terminal.status is store_module.InterventionStatus.RESUMED
    assert store.agent_run(binding.next_run_id).status == "completed"
    assert store.authenticated_intervention(stage.intervention_id) == terminal
    assert store.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)


def test_next_fable_directed_result_atomically_reserves_its_exact_sol_child(
    tmp_path,
    valid_brief,
) -> None:
    """The accepted Fable stage cannot terminalize before a durable child owner exists."""
    store = SQLiteStore(
        tmp_path / "next-fable-directed-child.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    stage = _next_fable_intervention_stage_for_tamper_matrix(store, valid_brief)
    binding = stage.directed_binding
    assert binding is not None and binding.next_run_id is not None
    store.start_next_fable_intervention_stage(
        stage.intervention_id, run_id=binding.next_run_id,
    )
    directed = DirectedAgentQuestion(
        addressed_to="sol",
        text="Which exact downstream fact is verified?",
        reason="Fable needs one bounded fact.",
    )
    event = _conversation_question(
        sender=ConversationActor.FABLE,
        addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.SOL,
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        generation=stage.resume_generation,
        question_id="second-sol-child-question",
        text=directed.text,
    )

    first = store.reserve_fable_answer_evidence_question(
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        expected_generation=stage.resume_generation,
        outer_question_id="original-question",
        question_id="second-sol-child-question",
        request_key="second-sol-child-request",
        text=directed.text,
        event=event,
        intervention_id=stage.intervention_id,
        child_run_id="second-sol-child-run",
        completed_next_fable_intervention_id=stage.intervention_id,
        completed_next_fable_run_id=binding.next_run_id,
        completed_next_fable_exit_code=0,
        clarification=FableClarification.from_dict({
            "status": "answered",
            "answer": "One more exact fact is required.",
            "reasoning": "The bounded evidence needs one successor.",
            "confidence": 0.9,
            "scope_changed": False,
            "revised_brief": None,
            "question_for_user": None,
            "directed_question": directed.to_dict(),
        }),
    )
    second = store.reserve_fable_answer_evidence_question(
        session_id="session-1",
        task_id=valid_brief.task_id,
        revision=valid_brief.revision,
        expected_generation=stage.resume_generation,
        outer_question_id="original-question",
        question_id="second-sol-child-question",
        request_key="second-sol-child-request",
        text=directed.text,
        event=event,
        intervention_id=stage.intervention_id,
        child_run_id="second-sol-child-run",
        completed_next_fable_intervention_id=stage.intervention_id,
        completed_next_fable_run_id=binding.next_run_id,
        completed_next_fable_exit_code=0,
        clarification=FableClarification.from_dict({
            "status": "answered",
            "answer": "One more exact fact is required.",
            "reasoning": "The bounded evidence needs one successor.",
            "confidence": 0.9,
            "scope_changed": False,
            "revised_brief": None,
            "question_for_user": None,
            "directed_question": directed.to_dict(),
        }),
    )

    assert second == first
    rebound = store.authenticated_intervention(stage.intervention_id)
    assert rebound is not None and rebound.directed_binding is not None
    assert rebound.status is store_module.InterventionStatus.RESUMING
    assert rebound.directed_binding.stage == "active_question"
    assert rebound.directed_binding.question_id == "second-sol-child-question"
    assert rebound.directed_binding.source_run_id == "second-sol-child-run"
    assert rebound.directed_binding.next_predecessor_run_id == binding.next_run_id
    assert store.agent_run(binding.next_run_id).status == "completed"
    child = store.agent_run("second-sol-child-run")
    assert child.agent == "sol"
    assert child.cli_session_id == "provider-1"
    assert child.status == "running"
    assert len(tuple(
        persisted for persisted in store.events_after("session-1", 0)
        if persisted.kind == "conversation"
        and persisted.payload.get("question_id") == "second-sol-child-question"
    )) == 1


def test_next_fable_answer_permission_pause_orders_cause_before_status_and_retries_after_rollback(
    tmp_path,
    valid_brief,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Fable result precedes its derived warning, with rollback leaving neither."""
    store = SQLiteStore(
        tmp_path / "permission-order-answer.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    stage = _next_fable_intervention_stage_for_tamper_matrix(store, valid_brief)
    binding = stage.directed_binding
    assert binding is not None and binding.next_run_id is not None
    store.start_next_fable_intervention_stage(
        stage.intervention_id, run_id=binding.next_run_id,
    )
    store._connection.execute(
        "UPDATE tasks SET exchange_allowance = 0 WHERE task_id = ? AND revision = ?",
        (valid_brief.task_id, valid_brief.revision),
    )
    directed = DirectedAgentQuestion(
        addressed_to="sol",
        text="Which exact downstream fact is verified?",
        reason="Fable needs one bounded fact.",
    )
    clarification = FableClarification.from_dict({
        "status": "answered",
        "answer": "One more exact fact is required.",
        "reasoning": "The bounded evidence needs one successor.",
        "confidence": 0.9,
        "scope_changed": False,
        "revised_brief": None,
        "question_for_user": None,
        "directed_question": directed.to_dict(),
    })
    before = {
        table: tuple(tuple(row) for row in store._connection.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in (
            "tasks", "interventions", "agent_runs", "events", "exchange_permissions",
        )
    }
    insert_status = store._insert_conversation_event_in_transaction

    def fail_status_insert(**_: object) -> object:
        raise RuntimeError("injected permission status failure")

    monkeypatch.setattr(store, "_insert_conversation_event_in_transaction", fail_status_insert)
    common = {
        "session_id": "session-1",
        "task_id": valid_brief.task_id,
        "revision": valid_brief.revision,
        "expected_generation": stage.resume_generation,
        "attempted_question": directed,
        "event": _conversation_permission(
            task_id=valid_brief.task_id,
            revision=valid_brief.revision,
            generation=stage.resume_generation,
        ),
        "completed_next_fable_intervention_id": stage.intervention_id,
        "completed_next_fable_run_id": binding.next_run_id,
        "completed_next_fable_exit_code": 0,
        "clarification": clarification,
    }
    pause = store.pause_fable_answer_evidence_permission
    common["outer_question_id"] = "original-question"
    with pytest.raises(RuntimeError, match="injected permission status failure"):
        pause(**common)
    assert {
        table: tuple(tuple(row) for row in store._connection.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in before
    } == before

    monkeypatch.setattr(store, "_insert_conversation_event_in_transaction", insert_status)
    pause(**common)

    causal = tuple(
        event for event in store.events_after("session-1", 0)
        if (
            event.kind == "clarification"
            and event.payload.get("answer") == clarification.answer
        )
        or (
            event.kind == "conversation"
            and event.payload.get("message_type") == "status"
        )
    )
    assert [event.kind for event in causal] == ["clarification", "conversation"]
    assert len(causal) == 2


def _seed_pre_stage_answered_nested_intervention(
    path: Path, valid_brief, *, outcome: str, binding_shape: str,
    preserve_successor: bool = False,
) -> dict[str, tuple[tuple[object, ...], ...]]:
    """Restore a post-answer crash image from before the stage discriminator."""
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    stage = _next_fable_intervention_stage_for_tamper_matrix(store, valid_brief)
    assert stage.directed_binding is not None
    if outcome == "unknown":
        assert store.recover_active_tasks() == store_module.RecoverySummary(0, 1, 1)
    elif outcome != "resuming":
        raise AssertionError(f"unknown post-answer outcome {outcome}")
    store.close()

    preceding = sqlite3.connect(path)
    if binding_shape == "stage_less":
        row = preceding.execute(
            """
            SELECT directed_binding_json FROM interventions
            WHERE intervention_id = 'directed-terminal'
            """
        ).fetchone()
        assert row is not None
        binding = json.loads(row[0])
        for key in (
            "stage", "next_attempt_id", "next_run_id", "next_provider_id",
            "next_task_state", "next_continuation_state",
        ):
            binding.pop(key)
        preceding.execute(
            """
            UPDATE interventions SET directed_binding_json = ?
            WHERE intervention_id = 'directed-terminal'
            """,
            (json.dumps(binding, sort_keys=True, separators=(",", ":")),),
        )
    elif binding_shape == "null":
        preceding.execute(
            """
            UPDATE interventions SET directed_binding_json = NULL
            WHERE intervention_id = 'directed-terminal'
            """
        )
    elif binding_shape == "absent":
        preceding.execute("ALTER TABLE interventions DROP COLUMN directed_binding_json")
    else:
        raise AssertionError(f"unknown binding shape {binding_shape}")
    if not preserve_successor:
        preceding.execute("DELETE FROM agent_runs WHERE run_id = 'next-fable-run'")
    preceding.commit()
    tables = (
        "tasks", "interventions", "agent_runs", "questions", "exchange_reservations",
    )
    before = {
        table: tuple(tuple(row) for row in preceding.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in tables
    }
    preceding.close()
    return before


@pytest.mark.parametrize("outcome", ("resuming", "unknown"))
@pytest.mark.parametrize("binding_shape", ("absent", "null", "stage_less"))
def test_pre_stage_answered_nested_migration_binds_one_recoverable_successor(
    tmp_path, valid_brief, outcome: str, binding_shape: str,
) -> None:
    """One legacy post-answer identity migrates to one durable Fable retry stage."""
    path = tmp_path / f"pre-stage-{outcome}-{binding_shape}.sqlite3"
    before = _seed_pre_stage_answered_nested_intervention(
        path, valid_brief, outcome=outcome, binding_shape=binding_shape,
    )

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    record = migrated.authenticated_intervention("directed-terminal")
    assert record is not None and record.directed_binding is not None
    binding = record.directed_binding
    assert binding.stage == "next_fable"
    assert binding.next_attempt_id == record.resume_attempt_id
    assert binding.next_run_id is not None
    successor = migrated.agent_run(binding.next_run_id)
    assert successor.agent == "fable"
    assert successor.cli_session_id == record.fable_session_id
    assert successor.status == ("running" if outcome == "resuming" else "interrupted")
    child = migrated.question(binding.question_id)
    assert child is not None and child.answer_text == "The exact nested fact is verified."
    events_before = tuple(migrated.events_after("session-1", 0))
    migrated.close()

    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    same = reopened.authenticated_intervention("directed-terminal")
    assert same == record
    assert tuple(reopened.events_after("session-1", 0)) == events_before
    if outcome == "resuming":
        assert reopened.recover_active_tasks() == store_module.RecoverySummary(0, 1, 1)
    else:
        assert reopened.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)
    unknown = reopened.authenticated_intervention("directed-terminal")
    assert unknown is not None
    assert unknown.status is store_module.InterventionStatus.RESUME_OUTCOME_UNKNOWN
    assert unknown.directed_binding is not None
    assert reopened.question(unknown.directed_binding.question_id) == child

    acknowledged = reopened.authorize_retry_after_unknown(
        unknown.intervention_id,
        expected_resume_generation=unknown.resume_generation,
        acknowledgment_id="legacy-next-fable-ack",
    )
    reopened.begin_intervention_resume(
        acknowledged.intervention_id,
        expected_resume_generation=acknowledged.resume_generation,
        resume_attempt_id="legacy-next-fable-retry-attempt",
        resume_run_id="legacy-next-fable-retry-run",
    )
    retried = reopened.authenticated_intervention("directed-terminal")
    assert retried is not None and retried.directed_binding is not None
    retry_binding = retried.directed_binding
    assert retry_binding.next_attempt_id == "legacy-next-fable-retry-attempt"
    assert retry_binding.next_run_id == "legacy-next-fable-retry-run"
    started = reopened.start_next_fable_intervention_stage(
        retried.intervention_id, run_id="legacy-next-fable-retry-run",
    )
    assert reopened.start_next_fable_intervention_stage(
        retried.intervention_id, run_id="legacy-next-fable-retry-run",
    ) == started
    assert started.status == "running"
    assert reopened.question(retry_binding.question_id) == child
    assert tuple(reopened.events_after("session-1", 0)) == events_before
    assert tuple(
        tuple(row) for row in reopened._connection.execute(
            "SELECT * FROM questions ORDER BY rowid"
        )
    ) == before["questions"]
    reopened.close()


@pytest.mark.parametrize("outcome", ("resuming", "unknown"))
@pytest.mark.parametrize("binding_shape", ("absent", "null", "stage_less"))
def test_pre_stage_answered_nested_migration_reuses_one_exact_existing_successor(
    tmp_path, valid_brief, outcome: str, binding_shape: str,
) -> None:
    """A historical started successor remains the one authenticated retry stage."""
    path = tmp_path / f"pre-stage-existing-{outcome}-{binding_shape}.sqlite3"
    _seed_pre_stage_answered_nested_intervention(
        path, valid_brief, outcome=outcome, binding_shape=binding_shape,
        preserve_successor=True,
    )

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    record = migrated.authenticated_intervention("directed-terminal")
    assert record is not None and record.directed_binding is not None
    binding = record.directed_binding
    assert binding.stage == "next_fable"
    assert binding.next_attempt_id == "attempt-1"
    assert binding.next_run_id == "next-fable-run"
    assert migrated.agent_run("next-fable-run").status == (
        "running" if outcome == "resuming" else "interrupted"
    )
    child = migrated.question("nested-question")
    assert child is not None and child.answer_text == "The exact nested fact is verified."
    events = tuple(migrated.events_after("session-1", 0))
    migrated.close()

    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    same = reopened.authenticated_intervention("directed-terminal")
    assert same == record
    assert tuple(reopened.events_after("session-1", 0)) == events
    if outcome == "resuming":
        assert reopened.recover_active_tasks() == store_module.RecoverySummary(0, 1, 1)
    else:
        assert reopened.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)
    unknown = reopened.authenticated_intervention("directed-terminal")
    assert unknown is not None
    acknowledged = reopened.authorize_retry_after_unknown(
        unknown.intervention_id,
        expected_resume_generation=unknown.resume_generation,
        acknowledgment_id="existing-next-fable-ack",
    )
    reopened.begin_intervention_resume(
        acknowledged.intervention_id,
        expected_resume_generation=acknowledged.resume_generation,
        resume_attempt_id="existing-next-fable-retry-attempt",
        resume_run_id="existing-next-fable-retry-run",
    )
    retried = reopened.authenticated_intervention("directed-terminal")
    assert retried is not None and retried.directed_binding is not None
    assert retried.directed_binding.next_run_id == "existing-next-fable-retry-run"
    assert reopened.start_next_fable_intervention_stage(
        retried.intervention_id, run_id="existing-next-fable-retry-run",
    ).status == "running"
    assert reopened.question("nested-question") == child
    assert tuple(reopened.events_after("session-1", 0)) == events
    reopened.close()


@pytest.mark.parametrize(
    ("outcome", "preserve_successor", "second_eligible", "expects_migration"),
    (
        ("resuming", True, False, True),
        ("unknown", True, False, True),
        ("resuming", False, False, True),
        ("unknown", True, True, False),
    ),
)
def test_pre_stage_answered_migration_filters_historical_fable_runs_before_uniqueness(
    tmp_path,
    valid_brief,
    outcome: str,
    preserve_successor: bool,
    second_eligible: bool,
    expects_migration: bool,
) -> None:
    """Only exact eligible successor runs participate in predecessor migration."""
    path = tmp_path / (
        "pre-stage-filtered-"
        f"{outcome}-{preserve_successor}-{second_eligible}.sqlite3"
    )
    _seed_pre_stage_answered_nested_intervention(
        path,
        valid_brief,
        outcome=outcome,
        binding_shape="absent",
        preserve_successor=preserve_successor,
    )
    preceding = sqlite3.connect(path)
    for run_id in ("historical-plan-run", "historical-clarification-run"):
        preceding.execute(
            """
            INSERT INTO agent_runs (
                run_id, task_id, revision, agent, cli_session_id, started_at,
                ended_at, exit_code, status
            ) VALUES (?, ?, ?, 'fable', ?, ?, ?, 0, 'completed')
            """,
            (
                run_id,
                valid_brief.task_id,
                valid_brief.revision,
                "provider-1",
                "2026-08-09T12:00:00Z",
                "2026-08-09T12:05:00Z",
            ),
        )
    if second_eligible:
        preceding.execute(
            """
            INSERT INTO agent_runs (
                run_id, task_id, revision, agent, cli_session_id, started_at,
                ended_at, exit_code, status
            ) VALUES ('second-eligible-successor', ?, ?, 'fable', ?, ?, ?, -15, 'interrupted')
            """,
            (
                valid_brief.task_id,
                valid_brief.revision,
                "provider-1",
                "2026-08-10T12:00:00Z",
                "2026-08-10T12:01:00Z",
            ),
        )
    preceding.commit()
    before = tuple(
        tuple(row)
        for row in preceding.execute("SELECT * FROM agent_runs ORDER BY rowid")
    )
    preceding.close()

    if not expects_migration:
        with pytest.raises(RuntimeError, match="migration"):
            SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
        after_connection = sqlite3.connect(path)
        after = tuple(
            tuple(row)
            for row in after_connection.execute("SELECT * FROM agent_runs ORDER BY rowid")
        )
        after_connection.close()
        assert after == before
        return

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    record = migrated.authenticated_intervention("directed-terminal")
    assert record is not None and record.directed_binding is not None
    binding = record.directed_binding
    assert binding.stage == "next_fable"
    if preserve_successor:
        assert binding.next_run_id == "next-fable-run"
    else:
        assert binding.next_run_id is not None
        assert binding.next_run_id.startswith("migration-next-fable-")
    assert migrated.agent_run("historical-plan-run").status == "completed"
    assert migrated.agent_run("historical-clarification-run").status == "completed"
    migrated.close()


def test_pre_stage_successor_migration_preserves_fractional_order_within_one_second(
    tmp_path,
    valid_brief,
) -> None:
    """A completed earlier fraction cannot hide the later structural successor."""
    path = tmp_path / "pre-stage-fractional-successor.sqlite3"
    _seed_pre_stage_answered_nested_intervention(
        path,
        valid_brief,
        outcome="unknown",
        binding_shape="absent",
        preserve_successor=True,
    )
    preceding = sqlite3.connect(path)
    preceding.execute(
        "UPDATE interventions SET created_at = '2026-08-10T12:00:00.500000Z' "
        "WHERE intervention_id = 'directed-terminal'"
    )
    preceding.execute(
        "UPDATE agent_runs SET started_at = '2026-08-10T12:00:00.600000Z' "
        "WHERE run_id = 'nested-source-run'"
    )
    preceding.execute(
        "UPDATE agent_runs SET started_at = '2026-08-10T12:00:00.550000Z' "
        "WHERE run_id = 'resume-run-1'"
    )
    preceding.execute(
        "UPDATE agent_runs SET started_at = '2026-08-10T12:00:00.700000Z' "
        "WHERE run_id = 'next-fable-run'"
    )
    preceding.execute(
        """
        INSERT INTO agent_runs (
            run_id, task_id, revision, agent, cli_session_id,
            started_at, ended_at, exit_code, status
        ) VALUES ('same-second-historical-fable', ?, ?, 'fable', 'provider-1',
                  '2026-08-10T12:00:00.100000Z',
                  '2026-08-10T12:00:00.200000Z', 0, 'completed')
        """,
        (valid_brief.task_id, valid_brief.revision),
    )
    preceding.commit()
    preceding.close()

    migrated = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:01Z")

    record = migrated.authenticated_intervention("directed-terminal")
    assert record is not None and record.directed_binding is not None
    assert record.directed_binding.next_run_id == "next-fable-run"
    assert migrated.agent_run("same-second-historical-fable").status == "completed"


@pytest.mark.parametrize("mutation", ("two_candidates", "provider", "session", "status"))
def test_pre_stage_answered_nested_migration_rejects_an_ambiguous_or_mismatched_successor(
    tmp_path, valid_brief, mutation: str,
) -> None:
    """Legacy candidate reuse fails closed unless one successor is exact."""
    path = tmp_path / f"pre-stage-mismatch-{mutation}.sqlite3"
    _seed_pre_stage_answered_nested_intervention(
        path, valid_brief, outcome="resuming", binding_shape="absent",
        preserve_successor=True,
    )
    preceding = sqlite3.connect(path)
    if mutation == "two_candidates":
        preceding.execute(
            """
            INSERT INTO agent_runs (
                run_id, task_id, revision, agent, cli_session_id, started_at, status
            ) VALUES ('second-next-fable-run', ?, ?, 'fable', ?, ?, 'interrupted')
            """,
            (
                valid_brief.task_id, valid_brief.revision, "provider-1",
                "2026-08-10T12:00:00Z",
            ),
        )
    elif mutation == "provider":
        preceding.execute(
            "UPDATE agent_runs SET cli_session_id = 'other-fable-session' "
            "WHERE run_id = 'next-fable-run'"
        )
    elif mutation == "session":
        preceding.execute(
            "UPDATE tasks SET fable_session_id = 'other-fable-session' "
            "WHERE task_id = ? AND revision = ?",
            (valid_brief.task_id, valid_brief.revision),
        )
    elif mutation == "status":
        preceding.execute(
            "UPDATE agent_runs SET status = 'completed' WHERE run_id = 'next-fable-run'"
        )
    else:
        raise AssertionError(f"unknown successor mutation {mutation}")
    preceding.commit()
    tables = (
        "tasks", "interventions", "agent_runs", "questions", "exchange_reservations",
    )
    before = {
        table: tuple(tuple(row) for row in preceding.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in tables
    }
    preceding.close()

    with pytest.raises(RuntimeError, match="migration"):
        SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    after_connection = sqlite3.connect(path)
    after = {
        table: tuple(tuple(row) for row in after_connection.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in tables
    }
    after_connection.close()
    assert after == before


def test_pre_stage_answered_migration_rejects_a_substituted_successor_collision(
    tmp_path, valid_brief,
) -> None:
    """A deterministic legacy successor ID cannot bind a substituted run row."""
    path = tmp_path / "pre-stage-successor-collision.sqlite3"
    _seed_pre_stage_answered_nested_intervention(
        path, valid_brief, outcome="resuming", binding_shape="absent",
    )
    preceding = sqlite3.connect(path)
    row = preceding.execute(
        """
        SELECT intervention_id, resume_generation, resume_attempt_id
        FROM interventions WHERE intervention_id = 'directed-terminal'
        """
    ).fetchone()
    assert row is not None
    seed = store_module._encode_json({
        "intervention_id": row[0],
        "resume_generation": row[1],
        "resume_attempt_id": row[2],
        "question_id": "nested-question",
        "source_run_id": "nested-source-run",
    })
    successor_id = (
        "migration-next-fable-"
        f"{store_module.hashlib.sha256(seed.encode()).hexdigest()[:40]}"
    )
    preceding.execute(
        """
        INSERT INTO agent_runs (
            run_id, task_id, revision, agent, cli_session_id, started_at, status
        ) VALUES (?, ?, ?, 'sol', ?, ?, 'running')
        """,
        (
            successor_id, valid_brief.task_id, valid_brief.revision,
            "fable-session-1", "2026-08-10T12:00:00Z",
        ),
    )
    preceding.commit()
    tables = (
        "tasks", "interventions", "agent_runs", "questions", "exchange_reservations",
    )
    before = {
        table: tuple(tuple(entry) for entry in preceding.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in tables
    }
    preceding.close()

    with pytest.raises(RuntimeError):
        SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")

    after_connection = sqlite3.connect(path)
    after = {
        table: tuple(tuple(entry) for entry in after_connection.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in tables
    }
    after_connection.close()
    assert after == before


def _tamper_next_fable_intervention_stage(store, mutation: str) -> None:
    if mutation == "run":
        store._connection.execute(
            """
            UPDATE interventions SET directed_binding_json = json_set(
                directed_binding_json, '$.next_run_id', 'nested-source-run'
            ) WHERE intervention_id = 'directed-terminal'
            """
        )
    elif mutation == "missing_run":
        store._connection.execute(
            "DELETE FROM agent_runs WHERE run_id = 'next-fable-run'"
        )
    elif mutation == "provider":
        store._connection.execute(
            """
            UPDATE interventions SET directed_binding_json = json_set(
                directed_binding_json, '$.next_provider_id', 'substituted-provider'
            ) WHERE intervention_id = 'directed-terminal'
            """
        )
    elif mutation == "session":
        store._connection.execute(
            """
            UPDATE tasks SET fable_session_id = 'substituted-session'
            WHERE task_id = ? AND revision = ?
            """,
            ("task-1", 1),
        )
    elif mutation == "question":
        store._connection.execute(
            """
            UPDATE interventions SET directed_binding_json = json_set(
                directed_binding_json, '$.question_id', 'original-question'
            ) WHERE intervention_id = 'directed-terminal'
            """
        )
    elif mutation == "parent":
        store._connection.execute(
            """
            UPDATE interventions SET directed_binding_json = json_set(
                directed_binding_json, '$.parent_question_id', 'nested-question'
            ) WHERE intervention_id = 'directed-terminal'
            """
        )
    elif mutation == "route":
        store._connection.execute(
            """
            UPDATE questions SET addressed_to = 'fable', routed_to = 'fable'
            WHERE question_id = 'nested-question'
            """
        )
    elif mutation == "generation":
        store._connection.execute(
            """
            UPDATE interventions SET directed_binding_json = json_set(
                directed_binding_json, '$.question_generation', 999
            ) WHERE intervention_id = 'directed-terminal'
            """
        )
    elif mutation == "attempt":
        store._connection.execute(
            """
            UPDATE interventions SET directed_binding_json = json_set(
                directed_binding_json, '$.next_attempt_id', 'substituted-attempt'
            ) WHERE intervention_id = 'directed-terminal'
            """
        )
    else:
        raise AssertionError(f"unknown next-stage mutation {mutation}")


@pytest.mark.parametrize(
    "mutation",
    (
        "run", "missing_run", "provider", "session", "question", "parent", "route",
        "generation", "attempt",
    ),
)
@pytest.mark.parametrize("operation", ("invoke", "retry"))
def test_next_fable_intervention_stage_tampering_fails_before_invoke_or_retry_and_rolls_back(
    tmp_path, valid_brief, mutation: str, operation: str,
) -> None:
    """The consumed child permits only its exact next Fable stage."""
    store = SQLiteStore(
        tmp_path / f"next-{operation}-{mutation}.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    stage = _next_fable_intervention_stage_for_tamper_matrix(store, valid_brief)
    if operation == "retry":
        store.recover_active_tasks()
    _tamper_next_fable_intervention_stage(store, mutation)
    before = {
        table: tuple(
            tuple(row) for row in store._connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            )
        )
        for table in ("tasks", "interventions", "questions", "exchange_reservations", "agent_runs")
    }

    if operation == "invoke":
        with pytest.raises(RuntimeError, match="binding is not authenticated"):
            store.start_next_fable_intervention_stage(
                stage.intervention_id, run_id="next-fable-run",
            )
    else:
        with pytest.raises(RuntimeError, match="binding is not authenticated"):
            store.authorize_retry_after_unknown(
                stage.intervention_id,
                expected_resume_generation=stage.resume_generation,
                acknowledgment_id=f"ack-{mutation}",
            )

    assert {
        table: tuple(
            tuple(row) for row in store._connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            )
        )
        for table in before
    } == before


@pytest.mark.parametrize("after_acknowledgment", (False, True))
def test_stop_authenticates_the_staged_next_fable_continuation(
    tmp_path, valid_brief, after_acknowledgment: bool,
) -> None:
    """Stop remains exact before invocation and after UNKNOWN acknowledgment."""
    store = SQLiteStore(
        tmp_path / f"next-stop-{after_acknowledgment}.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    stage = _next_fable_intervention_stage_for_tamper_matrix(store, valid_brief)
    if after_acknowledgment:
        unknown = store.recover_active_tasks()
        assert unknown.tasks_interrupted == 1
        stage = store.authorize_retry_after_unknown(
            stage.intervention_id,
            expected_resume_generation=stage.resume_generation,
            acknowledgment_id="next-stop-ack",
        )
    canceled = store.cancel_intervention_by_stop(
        stage.intervention_id,
        expected_resume_generation=stage.resume_generation,
    )
    assert canceled.status is store_module.InterventionStatus.CANCELED_BY_STOP
    assert canceled.directed_binding is not None
    assert canceled.directed_binding.stage == "next_fable"
    assert store.authenticated_intervention(stage.intervention_id) == canceled
    task = store.get_task(valid_brief.task_id, valid_brief.revision)
    assert task.state is TaskState.INTERRUPTED
    assert task.continuation_state is TaskState.SOL_RUNNING


@pytest.mark.parametrize("source_status", ("completed", "failed", "interrupted"))
def test_repeated_and_reopened_stop_accepts_authenticated_terminal_directed_source(
    tmp_path,
    valid_brief,
    source_status: str,
) -> None:
    """The first Stop deliberately preserves a terminal active-question source."""
    path = tmp_path / f"terminal-source-stop-{source_status}.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    created = _directed_intervention_claim_for_tamper_matrix(store, valid_brief)
    store._connection.execute(  # noqa: SLF001 - exact terminal source matrix
        "UPDATE agent_runs SET status = ?, ended_at = ?, exit_code = ? WHERE run_id = ?",
        (
            source_status,
            "2026-08-10T12:00:00Z",
            0 if source_status == "completed" else -1,
            "original-source-run",
        ),
    )
    canceled = store.cancel_intervention_by_stop(
        created.intervention_id,
        expected_resume_generation=created.resume_generation,
    )
    assert canceled.status is store_module.InterventionStatus.CANCELED_BY_STOP
    assert store.agent_run("original-source-run").status == source_status
    assert store.cancel_intervention_by_stop(
        canceled.intervention_id,
        expected_resume_generation=canceled.resume_generation,
    ) == canceled
    store.close()

    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:01Z")
    assert reopened.cancel_intervention_by_stop(
        canceled.intervention_id,
        expected_resume_generation=canceled.resume_generation,
    ) == canceled
    assert reopened.agent_run("original-source-run").status == source_status
    assert reopened.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)
    assert reopened.authenticated_intervention(canceled.intervention_id) == canceled
    reopened.close()


@pytest.mark.parametrize("mutation", ("source_run", "source_owner"))
def test_repeated_terminal_directed_source_stop_rejects_wrong_owner_without_mutation(
    tmp_path,
    valid_brief,
    mutation: str,
) -> None:
    """Canonical replay cannot substitute a compatible terminal source identity."""
    store = SQLiteStore(
        tmp_path / f"terminal-source-stop-{mutation}.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    created = _directed_intervention_claim_for_tamper_matrix(store, valid_brief)
    canceled = store.cancel_intervention_by_stop(
        created.intervention_id,
        expected_resume_generation=created.resume_generation,
    )
    if mutation == "source_run":
        store._connection.execute(  # noqa: SLF001 - exact owner tamper
            "UPDATE interventions SET directed_binding_json = json_set("
            "directed_binding_json, '$.source_run_id', 'substituted-source') "
            "WHERE intervention_id = ?",
            (canceled.intervention_id,),
        )
    else:
        store._connection.execute(  # noqa: SLF001 - exact owner tamper
            "UPDATE agent_runs SET cli_session_id = 'substituted-provider' "
            "WHERE run_id = 'original-source-run'"
        )
    before = _scope_owner_fixture_snapshot(store)

    with pytest.raises(RuntimeError, match="binding|run|owner|authenticated"):
        store.cancel_intervention_by_stop(
            canceled.intervention_id,
            expected_resume_generation=canceled.resume_generation,
        )

    assert _scope_owner_fixture_snapshot(store) == before


def test_staged_scope_stop_uniqueness_queries_are_bounded(
    tmp_path,
    valid_brief,
) -> None:
    """Stop ambiguity checks bound compatible persisted history at two rows."""
    store = SQLiteStore(
        tmp_path / "bounded-staged-scope-stop.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    _, stage, _, _, _ = _staged_scope_checkpoint_for_owner_matrix(store, valid_brief)
    row = store._connection.execute(  # noqa: SLF001 - compatible history fixture
        "SELECT * FROM interventions WHERE intervention_id = ?",
        (stage.intervention_id,),
    ).fetchone()
    assert row is not None
    columns = tuple(row.keys())
    placeholders = ", ".join("?" for _ in columns)
    for index in range(64):
        values = list(row)
        values[columns.index("intervention_id")] = f"unrelated-stage-{index}"
        binding = json.loads(values[columns.index("directed_binding_json")])
        binding["parent_question_id"] = f"unrelated-parent-{index}"
        values[columns.index("directed_binding_json")] = json.dumps(
            binding, sort_keys=True, separators=(",", ":"),
        )
        store._connection.execute(  # noqa: SLF001 - compatible history fixture
            f"INSERT INTO interventions ({', '.join(columns)}) VALUES ({placeholders})",
            tuple(values),
        )
    statements: list[str] = []
    store._connection.set_trace_callback(statements.append)  # noqa: SLF001
    try:
        store.cancel_intervention_by_stop(
            stage.intervention_id,
            expected_resume_generation=stage.resume_generation,
        )
    finally:
        store._connection.set_trace_callback(None)  # noqa: SLF001

    checkpoint_reads = [
        statement for statement in statements
        if "FROM directed_fable_answer_checkpoints" in statement
        and "WHERE session_id =" in statement
        and "ORDER BY rowid" in statement
    ]
    preparation_reads = [
        statement for statement in statements
        if "FROM prepared_actions" in statement
        and "status IN" in statement
        and "ORDER BY rowid" in statement
    ]
    history_reads = [
        statement for statement in statements
        if "FROM interventions" in statement and "json_extract" in statement
    ]
    assert checkpoint_reads and all("LIMIT 2" in row for row in checkpoint_reads)
    assert preparation_reads and all("LIMIT 2" in row for row in preparation_reads)
    assert history_reads and all("LIMIT 2" in row for row in history_reads)


@pytest.mark.parametrize("operation", ("retry", "stop"))
@pytest.mark.parametrize("mutation", ("attempt", "run", "compatible_run"))
def test_acknowledged_next_fable_stage_retains_and_authenticates_its_predecessor_owner(
    tmp_path, valid_brief, operation: str, mutation: str,
) -> None:
    """READY must retain the acknowledged stage owner until retry or Stop consumes it."""
    store = SQLiteStore(
        tmp_path / f"next-ready-{operation}-{mutation}.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    stage = _next_fable_intervention_stage_for_tamper_matrix(store, valid_brief)
    assert store.recover_active_tasks() == store_module.RecoverySummary(0, 1, 1)
    unknown = store.authenticated_intervention(stage.intervention_id)
    assert unknown is not None
    acknowledged = store.authorize_retry_after_unknown(
        unknown.intervention_id,
        expected_resume_generation=unknown.resume_generation,
        acknowledgment_id="next-ready-ack",
    )
    assert acknowledged.status is store_module.InterventionStatus.READY
    assert acknowledged.resume_attempt_id == "attempt-1"
    assert acknowledged.resume_run_id == "resume-run-1"
    if mutation == "attempt":
        store._connection.execute(
            "UPDATE interventions SET resume_attempt_id = 'substituted-attempt' "
            "WHERE intervention_id = ?",
            (acknowledged.intervention_id,),
        )
    else:
        if mutation == "compatible_run":
            store._connection.execute(
                """
                INSERT INTO agent_runs (
                    run_id, task_id, revision, agent, cli_session_id, started_at,
                    ended_at, exit_code, status
                ) VALUES ('compatible-historical-fable', ?, ?, 'fable', ?, ?, ?, 0, 'completed')
                """,
                (
                    valid_brief.task_id,
                    valid_brief.revision,
                    "provider-1",
                    "2026-08-09T12:00:00Z",
                    "2026-08-09T12:05:00Z",
                ),
            )
        store._connection.execute(
            "UPDATE interventions SET resume_run_id = ? "
            "WHERE intervention_id = ?",
            (
                (
                    "compatible-historical-fable"
                    if mutation == "compatible_run" else "substituted-run"
                ),
                acknowledged.intervention_id,
            ),
        )
    before = {
        table: tuple(tuple(row) for row in store._connection.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in ("tasks", "interventions", "questions", "exchange_reservations", "agent_runs")
    }

    if operation == "retry":
        with pytest.raises(RuntimeError, match="not ready to resume"):
            store.begin_intervention_resume(
                acknowledged.intervention_id,
                expected_resume_generation=acknowledged.resume_generation,
                resume_attempt_id="next-ready-retry-attempt",
                resume_run_id="next-ready-retry-run",
            )
    else:
        with pytest.raises(RuntimeError, match="binding is not authenticated"):
            store.cancel_intervention_by_stop(
                acknowledged.intervention_id,
                expected_resume_generation=acknowledged.resume_generation,
            )

    assert {
        table: tuple(tuple(row) for row in store._connection.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in before
    } == before


def test_unknown_next_fable_stage_rejects_a_compatible_substituted_predecessor_run(
    tmp_path,
    valid_brief,
) -> None:
    """UNKNOWN cannot replace its exact owner with compatible terminal Fable history."""
    store = SQLiteStore(
        tmp_path / "next-unknown-compatible-predecessor.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    stage = _next_fable_intervention_stage_for_tamper_matrix(store, valid_brief)
    assert store.recover_active_tasks() == store_module.RecoverySummary(0, 1, 1)
    store._connection.execute(
        """
        INSERT INTO agent_runs (
            run_id, task_id, revision, agent, cli_session_id, started_at,
            ended_at, exit_code, status
        ) VALUES ('compatible-historical-fable', ?, ?, 'fable', ?, ?, ?, 0, 'completed')
        """,
        (
            valid_brief.task_id,
            valid_brief.revision,
            "provider-1",
            "2026-08-09T12:00:00Z",
            "2026-08-09T12:05:00Z",
        ),
    )
    store._connection.execute(
        "UPDATE interventions SET resume_run_id = 'compatible-historical-fable' "
        "WHERE intervention_id = ?",
        (stage.intervention_id,),
    )
    before = {
        table: tuple(tuple(row) for row in store._connection.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in ("tasks", "interventions", "questions", "exchange_reservations", "agent_runs")
    }

    with pytest.raises(RuntimeError, match="binding is not authenticated"):
        store.authorize_retry_after_unknown(
            stage.intervention_id,
            expected_resume_generation=stage.resume_generation,
            acknowledgment_id="compatible-predecessor-ack",
        )

    assert {
        table: tuple(tuple(row) for row in store._connection.execute(
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in before
    } == before


@pytest.mark.parametrize("mutation", ("parent_reservation_missing", "parent_reservation_substituted"))
@pytest.mark.parametrize("operation", ("acknowledge", "retry"))
def test_nested_directed_unknown_flow_requires_exact_parent_reservation_and_rolls_back(
    tmp_path, valid_brief, mutation: str, operation: str,
) -> None:
    """Nested acknowledgment and retry reject a wrong parent without mutation."""
    store = SQLiteStore(
        tmp_path / f"parent-{operation}-{mutation}.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    created = _nested_parent_intervention_claim_for_tamper_matrix(store, valid_brief)
    store.recover_active_tasks()
    expected_generation = created.resume_generation
    if operation == "retry":
        acknowledged = store.authorize_retry_after_unknown(
            created.intervention_id,
            expected_resume_generation=created.resume_generation,
            acknowledgment_id=f"ack-{mutation}",
        )
        expected_generation = acknowledged.resume_generation
    if mutation == "parent_reservation_missing":
        store._connection.execute(
            "DELETE FROM exchange_reservations WHERE question_id = 'original-question'"
        )
    else:
        store._connection.execute(
            """
            UPDATE questions SET exchange_id = 'substituted-parent-exchange'
            WHERE question_id = 'original-question'
            """
        )
        store._connection.execute(
            """
            UPDATE exchange_reservations SET exchange_id = 'substituted-parent-exchange'
            WHERE question_id = 'original-question'
            """
        )
    before = {
        table: tuple(
            tuple(row) for row in store._connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            )
        )
        for table in ("tasks", "interventions", "questions", "exchange_reservations")
    }

    if operation == "acknowledge":
        with pytest.raises(RuntimeError, match="binding is not authenticated"):
            store.authorize_retry_after_unknown(
                created.intervention_id,
                expected_resume_generation=expected_generation,
                acknowledgment_id=f"ack-{mutation}",
            )
    else:
        with pytest.raises(RuntimeError, match="not ready to resume"):
            store.begin_intervention_resume(
                created.intervention_id,
                expected_resume_generation=expected_generation,
                resume_attempt_id="retry-attempt", resume_run_id="retry-run",
            )

    assert {
        table: tuple(
            tuple(row) for row in store._connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            )
        )
        for table in before
    } == before


@pytest.mark.parametrize(
    "status",
    (
        store_module.InterventionStatus.RESUMING,
        store_module.InterventionStatus.RESUME_OUTCOME_UNKNOWN,
        store_module.InterventionStatus.RESUMED,
        store_module.InterventionStatus.CANCELED_BY_STOP,
    ),
)
@pytest.mark.parametrize(
    "mutation", (
        "source_agent", "provider", "question", "pause", "route",
        "reservation_missing", "reservation_substituted",
    ),
)
def test_directed_intervention_terminal_forms_authenticate_the_original_binding(
    tmp_path, valid_brief, status, mutation: str,
) -> None:
    """Every post-claim form rejects substitution of its original directed boundary."""
    store = SQLiteStore(
        tmp_path / f"{status.value}-{mutation}.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    created = _directed_intervention_claim_for_tamper_matrix(store, valid_brief)
    if status is store_module.InterventionStatus.RESUME_OUTCOME_UNKNOWN:
        store.recover_active_tasks()
    elif status is store_module.InterventionStatus.RESUMED:
        store.complete_intervention(
            created.intervention_id, expected_resume_generation=created.resume_generation,
            resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
        )
    elif status is store_module.InterventionStatus.CANCELED_BY_STOP:
        store.cancel_intervention_by_stop(
            created.intervention_id, expected_resume_generation=created.resume_generation,
        )
    _tamper_directed_intervention_binding(store, mutation)

    with pytest.raises(RuntimeError, match="binding is not authenticated"):
        store.authenticated_intervention(created.intervention_id)


@pytest.mark.parametrize("mutation", ("reservation_missing", "reservation_substituted"))
def test_directed_unknown_ack_reservation_mismatch_rolls_back_every_generation(
    tmp_path, valid_brief, mutation: str,
) -> None:
    """Unknown acknowledgment requires its exact reservation before changing generations."""
    store = SQLiteStore(
        tmp_path / f"ack-{mutation}.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    created = _directed_intervention_claim_for_tamper_matrix(store, valid_brief)
    store.recover_active_tasks()
    _tamper_directed_intervention_binding(store, mutation)
    before = {
        table: tuple(
            tuple(row) for row in store._connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            )
        )
        for table in ("tasks", "interventions", "questions", "exchange_reservations")
    }

    with pytest.raises(RuntimeError, match="binding is not authenticated"):
        store.authorize_retry_after_unknown(
            created.intervention_id, expected_resume_generation=created.resume_generation,
            acknowledgment_id=f"ack-{mutation}",
        )

    assert {
        table: tuple(
            tuple(row) for row in store._connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            )
        )
        for table in before
    } == before


@pytest.mark.parametrize("mutation", ("reservation_missing", "reservation_substituted"))
def test_directed_acknowledged_retry_rejects_reservation_mismatch_without_mutation(
    tmp_path, valid_brief, mutation: str,
) -> None:
    """A prepared retry reauthenticates the reservation before claiming its new owner."""
    store = SQLiteStore(
        tmp_path / f"retry-{mutation}.sqlite3",
        clock=lambda: "2026-08-10T12:00:00Z",
    )
    created = _directed_intervention_claim_for_tamper_matrix(store, valid_brief)
    store.recover_active_tasks()
    acknowledged = store.authorize_retry_after_unknown(
        created.intervention_id, expected_resume_generation=created.resume_generation,
        acknowledgment_id=f"ack-{mutation}",
    )
    _tamper_directed_intervention_binding(store, mutation)
    before = {
        table: tuple(
            tuple(row) for row in store._connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            )
        )
        for table in ("tasks", "interventions", "questions", "exchange_reservations")
    }

    with pytest.raises(RuntimeError, match="not ready to resume"):
        store.begin_intervention_resume(
            created.intervention_id,
            expected_resume_generation=acknowledged.resume_generation,
            resume_attempt_id="retry-attempt", resume_run_id="retry-run",
        )

    assert {
        table: tuple(
            tuple(row) for row in store._connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid"
            )
        )
        for table in before
    } == before


def test_corrupt_ordinary_terminal_intervention_cannot_mimic_a_directed_binding(
    tmp_path, valid_brief,
) -> None:
    """A terminal ordinary row stays ordinary after a source phase-mismatch substitution."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "provider-sol")
    store.set_fable_session(valid_brief.task_id, valid_brief.revision, "provider-fable")
    store.start_agent_run(
        "ordinary-source", valid_brief.task_id, valid_brief.revision, "sol",
    )
    store.set_agent_run_session("ordinary-source", "provider-sol")
    store.set_pending_context(
        valid_brief.task_id, valid_brief.revision, expected=TaskState.SOL_RUNNING,
        pending={"sol_run_id": "ordinary-source", "prompt": "continue exactly"},
    )
    created = store.create_intervention_and_request_stop(
        intervention_id="ordinary-terminal", session_id="session-1",
        task_id=valid_brief.task_id, revision=valid_brief.revision,
        expected_source_generation=1, message="Keep the ordinary continuation.",
        addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
        run_id="ordinary-source",
    )
    store.finish_agent_run("ordinary-source", status="interrupted", exit_code=-15)
    store.mark_intervention_ready(created.intervention_id, run_id=created.run_id)
    store.begin_intervention_resume(
        created.intervention_id, expected_resume_generation=created.resume_generation,
        resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
    )
    store.complete_intervention(
        created.intervention_id, expected_resume_generation=created.resume_generation,
        resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
    )
    store._connection.execute(
        """
        UPDATE agent_runs SET agent = 'fable', cli_session_id = 'provider-fable'
        WHERE run_id = 'ordinary-source'
        """
    )

    with pytest.raises(RuntimeError, match="binding is not authenticated"):
        store.authenticated_intervention(created.intervention_id)


@pytest.mark.parametrize(
    ("state", "expected_agent"),
    (
        (TaskState.FABLE_PLANNING, "fable"),
        (TaskState.FABLE_CLARIFYING, "fable"),
        (TaskState.FABLE_REVIEWING, "fable"),
        (TaskState.SOL_RUNNING, "sol"),
        (TaskState.SOL_CORRECTING, "sol"),
    ),
)
def test_intervention_creation_requires_the_exact_agent_for_its_active_phase(
    tmp_path, valid_brief, state: TaskState, expected_agent: str,
) -> None:
    """A source run for another agent must not interrupt an active task."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    task_id = f"agent-phase-{state.value}"
    if state is TaskState.FABLE_PLANNING:
        task = store.create_planning_task("session-1", task_id)
    else:
        task = store.save_task(
            "session-1", replace(valid_brief, task_id=task_id), state,
        )
    source_run = f"source-{state.value}"
    store.start_agent_run(source_run, task.task_id, task.revision, "other")
    before = store.get_task(task.task_id, task.revision)

    with pytest.raises(RuntimeError, match="source agent"):
        store.create_intervention_and_request_stop(
            intervention_id=f"intervention-{state.value}", session_id="session-1",
            task_id=task.task_id, revision=task.revision, expected_source_generation=1,
            message="Pause exact work.", addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE, run_id=source_run,
        )

    assert expected_agent in {"fable", "sol"}
    assert store.intervention(f"intervention-{state.value}") is None
    assert store.get_task(task.task_id, task.revision) == before
    assert store.events_after("session-1", 0) == ()


@pytest.mark.parametrize(
    ("state", "task_provider", "run_provider"),
    (
        (TaskState.FABLE_PLANNING, None, "fable-session-1"),
        (TaskState.FABLE_PLANNING, "fable-session-1", None),
        (TaskState.FABLE_PLANNING, "fable-session-1", "other-fable-session"),
        (TaskState.FABLE_CLARIFYING, "fable-session-1", None),
        (TaskState.FABLE_REVIEWING, "fable-session-1", "other-fable-session"),
        (TaskState.SOL_RUNNING, "sol-thread-1", None),
        (TaskState.SOL_CORRECTING, "sol-thread-1", "other-sol-thread"),
        (TaskState.SOL_RUNNING, "-malformed-thread", "-malformed-thread"),
    ),
)
def test_intervention_creation_requires_exact_validated_provider_identity(
    tmp_path, valid_brief, state: TaskState, task_provider: str | None, run_provider: str | None,
) -> None:
    """Missing, changed, and malformed provider identities must not create stop intent."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    task_id = f"provider-phase-{state.value}"
    if state is TaskState.FABLE_PLANNING:
        task = store.create_planning_task("session-1", task_id)
        expected_agent = "fable"
        if task_provider is not None:
            store.set_fable_session(task.task_id, task.revision, task_provider)
    else:
        task = store.save_task(
            "session-1", replace(valid_brief, task_id=task_id), state,
        )
        expected_agent = "fable" if state in {
            TaskState.FABLE_CLARIFYING, TaskState.FABLE_REVIEWING,
        } else "sol"
        if expected_agent == "fable":
            store.set_fable_session(task.task_id, task.revision, task_provider)
        else:
            store.set_sol_thread(task.task_id, task.revision, task_provider)
    source_run = f"source-{state.value}"
    store.start_agent_run(source_run, task.task_id, task.revision, expected_agent)
    if run_provider is not None:
        store.set_agent_run_session(source_run, run_provider)
    before = store.get_task(task.task_id, task.revision)

    with pytest.raises(RuntimeError, match="provider identity"):
        store.create_intervention_and_request_stop(
            intervention_id=f"provider-intervention-{state.value}", session_id="session-1",
            task_id=task.task_id, revision=task.revision, expected_source_generation=1,
            message="Pause exact work.", addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE, run_id=source_run,
        )

    assert store.intervention(f"provider-intervention-{state.value}") is None
    assert store.get_task(task.task_id, task.revision) == before
    assert store.events_after("session-1", 0) == ()


def test_intervention_allows_early_fable_planning_only_before_a_session_exists(tmp_path) -> None:
    """The first Fable plan may stop before a provider session ID has been issued."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    task = store.create_planning_task("session-1", "early-fable-task")
    store.start_agent_run("early-fable-run", task.task_id, task.revision, "fable")

    created = store.create_intervention_and_request_stop(
        intervention_id="early-fable-intervention", session_id="session-1",
        task_id=task.task_id, revision=task.revision, expected_source_generation=1,
        message="Pause before the initial Fable session.", addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE, run_id="early-fable-run",
    )

    assert created.fable_session_id is None
    assert created.sol_thread_id is None
    assert created.continuation_state is TaskState.FABLE_PLANNING
    events = store.events_after("session-1", 0)
    assert len(events) == 1
    event = ConversationEnvelope.from_dict(events[0].payload)
    assert event.sender is ConversationActor.USER
    assert event.message_type is ConversationMessageType.INTERVENTION
    assert event.text == "Pause before the initial Fable session."
    assert event.task_id is None


def test_early_fable_session_binding_is_atomic_and_authenticates_after_reopen(
    tmp_path,
) -> None:
    """A stopped first Fable session must bind its run, task, and intervention together."""
    path = tmp_path / "early-fable-session-binding.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    repo = tmp_path / "repo"
    repo.mkdir()
    store.create_session("session-1", str(repo))
    task = store.create_planning_task("session-1", "early-fable-task")
    store.start_agent_run("early-fable-run", task.task_id, task.revision, "fable")
    created = store.create_intervention_and_request_stop(
        intervention_id="early-fable-intervention", session_id="session-1",
        task_id=task.task_id, revision=task.revision, expected_source_generation=1,
        message="Pause before the initial Fable session.", addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE, run_id="early-fable-run",
    )

    bound = store.bind_fresh_fable_session_to_pending_intervention(
        project_id=project_id_for_root(repo),
        intervention_id=created.intervention_id,
        session_id="session-1",
        task_id=task.task_id,
        revision=task.revision,
        expected_source_generation=created.source_generation,
        expected_resume_generation=created.resume_generation,
        source_run_id=created.run_id,
        expected_resume_attempt_id=None,
        expected_resume_run_id=None,
        fable_session_id="fable-session-1",
        continuation=TaskState.FABLE_PLANNING,
    )

    assert bound.fable_session_id == "fable-session-1"
    assert store.get_task(task.task_id, task.revision).fable_session_id == "fable-session-1"
    assert store.agent_run(created.run_id).cli_session_id == "fable-session-1"
    assert store.bind_fresh_fable_session_to_pending_intervention(
        project_id=project_id_for_root(repo),
        intervention_id=created.intervention_id,
        session_id="session-1",
        task_id=task.task_id,
        revision=task.revision,
        expected_source_generation=created.source_generation,
        expected_resume_generation=created.resume_generation,
        source_run_id=created.run_id,
        expected_resume_attempt_id=None,
        expected_resume_run_id=None,
        fable_session_id="fable-session-1",
        continuation=TaskState.FABLE_PLANNING,
    ) == bound

    store.finish_agent_run(created.run_id, status="interrupted", exit_code=-15)
    ready = store.mark_intervention_ready(created.intervention_id, run_id=created.run_id)
    store.set_setting("agent_bridge.active_session_id", "session-1")
    store.close()

    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    assert reopened.authenticated_intervention(ready.intervention_id) == ready
    reopened.audit_legacy_project_ownership(str(repo))
    reopened.close()


def test_claimed_fresh_fable_plan_session_binds_without_rewriting_no_id_source(
    tmp_path,
) -> None:
    """A no-ID source Stop authenticates the exact fresh claimed plan session."""
    path = tmp_path / "claimed-fresh-fable-session.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    repo = tmp_path / "repo"
    repo.mkdir()
    store.create_session("session-1", str(repo))
    task = store.create_planning_task("session-1", "claimed-fresh-fable-task")
    store.start_agent_run("claimed-fresh-fable-source", task.task_id, task.revision, "fable")
    created = store.create_intervention_and_request_stop(
        intervention_id="claimed-fresh-fable-intervention", session_id="session-1",
        task_id=task.task_id, revision=task.revision, expected_source_generation=1,
        message="Continue from a fresh Fable session.", addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE, run_id="claimed-fresh-fable-source",
    )
    store.finish_agent_run(created.run_id, status="interrupted", exit_code=-15)
    store.mark_intervention_ready(created.intervention_id, run_id=created.run_id)
    resumed = store.begin_intervention_resume(
        created.intervention_id,
        expected_resume_generation=created.resume_generation,
        resume_attempt_id="claimed-fresh-fable-attempt",
        resume_run_id="claimed-fresh-fable-run",
    )
    assert resumed.state is TaskState.FABLE_PLANNING
    assert resumed.continuation_state is None
    store.start_agent_run(
        "claimed-fresh-fable-run", task.task_id, task.revision, "fable",
    )

    bound = store.bind_fresh_fable_session_to_pending_intervention(
        project_id=project_id_for_root(repo),
        intervention_id=created.intervention_id,
        session_id="session-1",
        task_id=task.task_id,
        revision=task.revision,
        expected_source_generation=created.source_generation,
        expected_resume_generation=created.resume_generation,
        source_run_id=created.run_id,
        expected_resume_attempt_id="claimed-fresh-fable-attempt",
        expected_resume_run_id="claimed-fresh-fable-run",
        fable_session_id="claimed-fresh-fable-session",
        continuation=TaskState.FABLE_PLANNING,
    )

    assert bound.fable_session_id == "claimed-fresh-fable-session"
    assert store.get_task(task.task_id, task.revision).fable_session_id == (
        "claimed-fresh-fable-session"
    )
    assert store.agent_run(created.run_id).cli_session_id is None
    assert store.agent_run("claimed-fresh-fable-run").cli_session_id == (
        "claimed-fresh-fable-session"
    )
    assert store.bind_fresh_fable_session_to_pending_intervention(
        project_id=project_id_for_root(repo),
        intervention_id=created.intervention_id,
        session_id="session-1",
        task_id=task.task_id,
        revision=task.revision,
        expected_source_generation=created.source_generation,
        expected_resume_generation=created.resume_generation,
        source_run_id=created.run_id,
        expected_resume_attempt_id="claimed-fresh-fable-attempt",
        expected_resume_run_id="claimed-fresh-fable-run",
        fable_session_id="claimed-fresh-fable-session",
        continuation=TaskState.FABLE_PLANNING,
    ) == bound
    before_tamper = (
        store.get_task(task.task_id, task.revision),
        store.intervention(created.intervention_id),
        store.agent_run(created.run_id),
        store.agent_run("claimed-fresh-fable-run"),
    )
    with pytest.raises(RuntimeError, match="session changed"):
        store.bind_fresh_fable_session_to_pending_intervention(
            project_id=project_id_for_root(repo),
            intervention_id=created.intervention_id,
            session_id="session-1",
            task_id=task.task_id,
            revision=task.revision,
            expected_source_generation=created.source_generation,
            expected_resume_generation=created.resume_generation,
            source_run_id=created.run_id,
            expected_resume_attempt_id="claimed-fresh-fable-attempt",
            expected_resume_run_id="claimed-fresh-fable-run",
            fable_session_id="forged-fable-session",
            continuation=TaskState.FABLE_PLANNING,
        )
    assert (
        store.get_task(task.task_id, task.revision),
        store.intervention(created.intervention_id),
        store.agent_run(created.run_id),
        store.agent_run("claimed-fresh-fable-run"),
    ) == before_tamper

    store.set_setting("agent_bridge.active_session_id", "session-1")
    store.close()
    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    assert reopened.authenticated_intervention(created.intervention_id) == bound
    reopened.audit_legacy_project_ownership(str(repo))
    reopened.close()


@pytest.mark.parametrize(
    "mutation",
    ("project", "generation", "source", "resume_owner", "provider", "continuation"),
)
def test_early_fable_session_binding_rejects_mismatch_without_partial_write(
    tmp_path,
    mutation: str,
) -> None:
    """Every identity coordinate is a CAS guard for the fresh Fable binding."""
    store = _store(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    store.create_session("session-1", str(repo))
    task = store.create_planning_task("session-1", "early-fable-task")
    store.start_agent_run("early-fable-run", task.task_id, task.revision, "fable")
    created = store.create_intervention_and_request_stop(
        intervention_id="early-fable-intervention", session_id="session-1",
        task_id=task.task_id, revision=task.revision, expected_source_generation=1,
        message="Pause before the initial Fable session.", addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE, run_id="early-fable-run",
    )
    arguments: dict[str, object] = {
        "project_id": project_id_for_root(repo),
        "intervention_id": created.intervention_id,
        "session_id": "session-1",
        "task_id": task.task_id,
        "revision": task.revision,
        "expected_source_generation": created.source_generation,
        "expected_resume_generation": created.resume_generation,
        "source_run_id": created.run_id,
        "expected_resume_attempt_id": None,
        "expected_resume_run_id": None,
        "fable_session_id": "fable-session-1",
        "continuation": TaskState.FABLE_PLANNING,
    }
    if mutation == "project":
        arguments["project_id"] = "other-project"
    elif mutation == "generation":
        arguments["expected_source_generation"] = created.source_generation + 1
    elif mutation == "source":
        arguments["source_run_id"] = "other-source-run"
    elif mutation == "resume_owner":
        arguments["expected_resume_attempt_id"] = "claimed-attempt"
    elif mutation == "provider":
        arguments["fable_session_id"] = "--forged-session"
    elif mutation == "continuation":
        arguments["continuation"] = TaskState.SOL_RUNNING
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    before = {
        "task": store.get_task(task.task_id, task.revision),
        "record": store.intervention(created.intervention_id),
        "run": store.agent_run(created.run_id),
        "events": store.events_after("session-1", 0),
    }

    with pytest.raises((RuntimeError, ValueError)):
        store.bind_fresh_fable_session_to_pending_intervention(**arguments)  # type: ignore[arg-type]

    assert {
        "task": store.get_task(task.task_id, task.revision),
        "record": store.intervention(created.intervention_id),
        "run": store.agent_run(created.run_id),
        "events": store.events_after("session-1", 0),
    } == before


def test_intervention_status_transitions_are_exact_owner_bound_and_idempotent(
    tmp_path, valid_brief,
) -> None:
    """Changing a status, owner, or generation must reject the affected transition."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "sol-thread-1")
    store.start_agent_run("source-run-1", valid_brief.task_id, valid_brief.revision, "sol")
    store.set_agent_run_session("source-run-1", "sol-thread-1")
    created = store.create_intervention_and_request_stop(
        intervention_id="intervention-1", session_id="session-1", task_id=valid_brief.task_id,
        revision=valid_brief.revision, expected_source_generation=1, message="Pause here.",
        addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
        run_id="source-run-1",
    )
    store.finish_agent_run("source-run-1", status="interrupted", exit_code=0)

    ready = store.mark_intervention_ready("intervention-1", run_id="source-run-1")
    assert ready.status is store_module.InterventionStatus.READY
    assert store.mark_intervention_ready("intervention-1", run_id="source-run-1") == ready
    with pytest.raises(RuntimeError, match="source run"):
        store.mark_intervention_ready("intervention-1", run_id="other-run")
    with pytest.raises(RuntimeError, match="generation"):
        store.claim_intervention_resume(
            "intervention-1", expected_resume_generation=1,
            resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
        )

    claimed = store.claim_intervention_resume(
        "intervention-1", expected_resume_generation=created.resume_generation,
        resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
    )
    assert claimed.status is store_module.InterventionStatus.RESUMING
    assert store.claim_intervention_resume(
        "intervention-1", expected_resume_generation=created.resume_generation,
        resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
    ) == claimed
    with pytest.raises(RuntimeError, match="owner"):
        store.complete_intervention(
            "intervention-1", expected_resume_generation=created.resume_generation,
            resume_attempt_id="attempt-other", resume_run_id="resume-run-1",
        )
    completed = store.complete_intervention(
        "intervention-1", expected_resume_generation=created.resume_generation,
        resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
    )
    assert completed.status is store_module.InterventionStatus.RESUMED

    store._connection.execute(
        "UPDATE interventions SET status = 'resuming' WHERE intervention_id = 'intervention-1'"
    )
    unknown = store.mark_resume_outcome_unknown(
        "intervention-1", resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
    )
    assert unknown.status is store_module.InterventionStatus.RESUME_OUTCOME_UNKNOWN
    retried = store.authorize_retry_after_unknown(
        "intervention-1", expected_resume_generation=created.resume_generation,
        acknowledgment_id="ack-1",
    )
    assert retried.status is store_module.InterventionStatus.READY
    assert retried.resume_generation == created.resume_generation + 1
    assert store.authorize_retry_after_unknown(
        "intervention-1", expected_resume_generation=created.resume_generation,
        acknowledgment_id="ack-1",
    ) == retried
    canceled = store.cancel_intervention_by_stop(
        "intervention-1", expected_resume_generation=retried.resume_generation,
    )
    assert canceled.status is store_module.InterventionStatus.CANCELED_BY_STOP
    with pytest.raises(RuntimeError, match="ready"):
        store.claim_intervention_resume(
            "intervention-1", expected_resume_generation=retried.resume_generation,
            resume_attempt_id="attempt-2", resume_run_id="resume-run-2",
        )


def test_intervention_retry_audit_preserves_the_source_bound_user_event(
    tmp_path, valid_brief,
) -> None:
    """Retry generation must not rewrite or invalidate the original intervention."""
    path = tmp_path / "intervention-ack.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "sol-thread-1")
    store.start_agent_run("source-run-1", valid_brief.task_id, valid_brief.revision, "sol")
    store.set_agent_run_session("source-run-1", "sol-thread-1")
    created = store.create_intervention_and_request_stop(
        intervention_id="acknowledged-intervention", session_id="session-1",
        task_id=valid_brief.task_id, revision=valid_brief.revision,
        expected_source_generation=1, message="Pause before retry.",
        addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
        run_id="source-run-1",
    )
    store.finish_agent_run("source-run-1", status="interrupted", exit_code=0)
    store.mark_intervention_ready(created.intervention_id, run_id="source-run-1")
    store.claim_intervention_resume(
        created.intervention_id, expected_resume_generation=created.resume_generation,
        resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
    )
    store.mark_resume_outcome_unknown(
        created.intervention_id, resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
    )

    retried = store.authorize_retry_after_unknown(
        created.intervention_id, expected_resume_generation=created.resume_generation,
        acknowledgment_id="ack-1",
    )

    assert retried.status is store_module.InterventionStatus.READY
    assert retried.resume_generation == created.resume_generation + 1
    assert store.get_task(
        valid_brief.task_id, valid_brief.revision,
    ).continuation_generation == retried.resume_generation
    events = store.events_after("session-1", 0)
    assert len(events) == 1
    assert ConversationEnvelope.from_dict(events[0].payload).continuation_generation == 1
    store.set_setting("agent_bridge.active_session_id", "session-1")
    assert store.audit_legacy_project_ownership("/repo") is None
    store.close()

    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    assert reopened.recover_active_tasks() == store_module.RecoverySummary(0, 0, 0)
    claimed = reopened.claim_intervention_resume(
        created.intervention_id, expected_resume_generation=retried.resume_generation,
        resume_attempt_id="attempt-2", resume_run_id="resume-run-2",
    )
    assert claimed.status is store_module.InterventionStatus.RESUMING
    assert len(reopened.events_after("session-1", 0)) == 1
    reopened._connection.execute(
        "UPDATE interventions SET acknowledgment_id = ? WHERE intervention_id = ?",
        ("-tampered-ack", created.intervention_id),
    )
    with pytest.raises(RuntimeError, match="legacy project ownership audit failed"):
        reopened.audit_legacy_project_ownership("/repo")


def test_intervention_startup_recovery_never_replays_a_committed_resume_claim(
    tmp_path, valid_brief,
) -> None:
    """A crash after claim must require an acknowledgment, never a provider retry."""
    path = tmp_path / "intervention-recovery.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "sol-thread-1")
    store.start_agent_run("source-run-1", valid_brief.task_id, valid_brief.revision, "sol")
    store.set_agent_run_session("source-run-1", "sol-thread-1")
    store.set_pending_context(
        valid_brief.task_id, valid_brief.revision, expected=TaskState.SOL_RUNNING,
        pending={"sol_run_id": "source-run-1", "prompt": "continue exactly"},
    )
    pending = store.create_intervention_and_request_stop(
        intervention_id="pending-stop", session_id="session-1", task_id=valid_brief.task_id,
        revision=valid_brief.revision, expected_source_generation=1, message="Pause first.",
        addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
        run_id="source-run-1",
    )
    store.close()

    recovered = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    summary = recovered.recover_active_tasks()
    assert summary.agent_runs_interrupted == 1
    ready = recovered.intervention("pending-stop")
    assert ready is not None
    assert ready.status is store_module.InterventionStatus.READY
    assert recovered.agent_run("source-run-1").status == "interrupted"
    assert recovered.mark_intervention_ready("pending-stop", run_id="source-run-1") == ready
    claimed_task = recovered.begin_intervention_resume(
        "pending-stop", expected_resume_generation=pending.resume_generation,
        resume_attempt_id="attempt-1", resume_run_id="resume-run-1",
    )
    assert claimed_task.state is TaskState.FABLE_CLARIFYING
    claimed = recovered.intervention("pending-stop")
    assert claimed is not None
    assert claimed.status is store_module.InterventionStatus.RESUMING
    recovered.close()

    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    assert reopened.recover_active_tasks().agent_runs_interrupted == 0
    unknown = reopened.intervention("pending-stop")
    assert unknown is not None
    assert unknown.status is store_module.InterventionStatus.RESUME_OUTCOME_UNKNOWN
    recovered_task = reopened.get_task(valid_brief.task_id, valid_brief.revision)
    assert recovered_task.state is TaskState.INTERRUPTED
    assert recovered_task.continuation_state is TaskState.FABLE_CLARIFYING
    with pytest.raises(RuntimeError, match="ready"):
        reopened.claim_intervention_resume(
            "pending-stop", expected_resume_generation=pending.resume_generation,
            resume_attempt_id="attempt-2", resume_run_id="resume-run-2",
        )
    assert reopened.authorize_retry_after_unknown(
        "pending-stop", expected_resume_generation=pending.resume_generation,
        acknowledgment_id="ack-1",
    ).status is store_module.InterventionStatus.READY


def test_intervention_rejects_hostile_reuse_and_audits_durable_bindings(
    tmp_path, valid_brief,
) -> None:
    """A substituted binding must not mutate this Store or cross into another one."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    store.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    store.set_sol_thread(valid_brief.task_id, valid_brief.revision, "sol-thread-1")
    store.start_agent_run("source-run-1", valid_brief.task_id, valid_brief.revision, "sol")
    store.set_agent_run_session("source-run-1", "sol-thread-1")
    created = store.create_intervention_and_request_stop(
        intervention_id="shared-id", session_id="session-1", task_id=valid_brief.task_id,
        revision=valid_brief.revision, expected_source_generation=1, message="Keep scope.",
        addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
        run_id="source-run-1",
    )
    before_task = store.get_task(valid_brief.task_id, valid_brief.revision)
    before_events = store.events_after("session-1", 0)
    with pytest.raises(RuntimeError, match="bound differently"):
        store.create_intervention_and_request_stop(
            intervention_id="shared-id", session_id="session-1", task_id=valid_brief.task_id,
            revision=valid_brief.revision, expected_source_generation=1, message="Change scope.",
            addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
            run_id="source-run-1",
        )
    assert store.intervention("shared-id") == created
    assert store.get_task(valid_brief.task_id, valid_brief.revision) == before_task
    assert store.events_after("session-1", 0) == before_events

    foreign = SQLiteStore(tmp_path / "foreign.sqlite3", clock=lambda: "2026-08-10T12:00:00Z")
    foreign.create_session("session-1", "/other-repo")
    foreign.save_task("session-1", valid_brief, TaskState.SOL_RUNNING)
    foreign.set_sol_thread(valid_brief.task_id, valid_brief.revision, "sol-thread-1")
    foreign.start_agent_run("source-run-1", valid_brief.task_id, valid_brief.revision, "sol")
    foreign.set_agent_run_session("source-run-1", "sol-thread-1")
    assert foreign.intervention("shared-id") is None
    assert foreign.create_intervention_and_request_stop(
        intervention_id="shared-id", session_id="session-1", task_id=valid_brief.task_id,
        revision=valid_brief.revision, expected_source_generation=1, message="Foreign scope.",
        addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
        run_id="source-run-1",
    ).intervention_id == "shared-id"
    assert store.intervention("shared-id") == created

    store.finish_agent_run("source-run-1", status="interrupted", exit_code=0)
    store.mark_intervention_ready("shared-id", run_id="source-run-1")
    store.set_setting("agent_bridge.active_session_id", "session-1")
    assert store.audit_legacy_project_ownership("/repo") is None

    store._connection.execute(
        "UPDATE interventions SET continuation_state = ? WHERE intervention_id = ?",
        (TaskState.COMPLETED.value, "shared-id"),
    )
    with pytest.raises(RuntimeError, match="legacy project ownership audit failed"):
        store.audit_legacy_project_ownership("/repo")


def test_question_answers_compare_and_swap_exact_identity_and_legal_recipient(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo-one")
    store.create_session("session-2", "/repo-two")
    first_brief = replace(valid_brief, task_id="task-one", revision=1)
    other_brief = replace(valid_brief, task_id="task-two", revision=1)
    second_revision = replace(first_brief, revision=2, title="Second exact revision")
    _save_active_directed_task(store, "session-1", first_brief)
    _save_active_directed_task(store, "session-1", other_brief)
    _save_active_directed_task(store, "session-1", second_revision)
    question = store.pause_for_question(
        session_id="session-1",
        task_id="task-one",
        revision=1,
        expected_generation=1,
        question_id="question-one",
        asked_by=ConversationActor.FABLE,
        addressed_to=ConversationTarget.USER,
        routed_to=ConversationTarget.USER,
        text="Which approved option should be used?",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"next": "resume-sol"},
        event=_conversation_question(
            sender=ConversationActor.FABLE,
            addressed_to=ConversationTarget.USER,
            routed_to=ConversationTarget.USER,
            task_id="task-one",
            revision=1,
            generation=1,
            question_id="question-one",
            text="Which approved option should be used?",
        ),
    )
    assert question == QuestionRecord(
        question_id="question-one",
        session_id="session-1",
        task_id="task-one",
        revision=1,
        continuation_generation=1,
        asked_by=ConversationActor.FABLE,
        addressed_to=ConversationTarget.USER,
        routed_to=ConversationTarget.USER,
        text="Which approved option should be used?",
        exchange_id=None,
        answer_text=None,
        answered_by=None,
    )
    assert store.question("question-one") == question
    waiting = store.get_task("task-one", 1)
    assert waiting.state is TaskState.AWAITING_USER_INPUT
    assert waiting.continuation_state is TaskState.SOL_RUNNING
    assert waiting.pending == {"next": "resume-sol"}
    assert waiting.continuation_generation == 1
    assert waiting.exchange_allowance == INITIAL_INTERNAL_EXCHANGES
    assert waiting.exchange_consumed == 0

    other_question = store.pause_for_question(
        session_id="session-1",
        task_id="task-two",
        revision=1,
        expected_generation=1,
        question_id="question-other-task",
        asked_by=ConversationActor.FABLE,
        addressed_to=ConversationTarget.USER,
        routed_to=ConversationTarget.USER,
        text="Choose the separate task option.",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"next": "resume-other"},
        event=_conversation_question(
            sender=ConversationActor.FABLE,
            addressed_to=ConversationTarget.USER,
            routed_to=ConversationTarget.USER,
            task_id="task-two",
            revision=1,
            generation=1,
            question_id="question-other-task",
            text="Choose the separate task option.",
        ),
    )
    assert other_question.question_id == "question-other-task"
    with pytest.raises(RuntimeError, match="unanswered question"):
        store.pause_for_question(
            session_id="session-1",
            task_id="task-one",
            revision=1,
            expected_generation=1,
            question_id="second-question-one",
            asked_by=ConversationActor.FABLE,
            addressed_to=ConversationTarget.USER,
            routed_to=ConversationTarget.USER,
            text="A second question must wait.",
            continuation_state=TaskState.SOL_RUNNING,
            pending_action={"next": "must-not-persist"},
            event=_conversation_question(
                sender=ConversationActor.FABLE,
                addressed_to=ConversationTarget.USER,
                routed_to=ConversationTarget.USER,
                task_id="task-one",
                revision=1,
                generation=1,
                question_id="second-question-one",
                text="A second question must wait.",
            ),
        )

    def assert_rejected(
        *, session_id: str, revision: int, question_id: str, generation: int,
    ) -> None:
        before_task = store.get_task("task-one", 1)
        before_events = store.events_after("session-1", 0)
        before_runs = tuple(store._connection.execute("SELECT * FROM agent_runs"))
        with pytest.raises(RuntimeError):
            store.answer_question_and_prepare_resume(
                session_id=session_id,
                task_id="task-one",
                revision=revision,
                question_id=question_id,
                expected_generation=generation,
                answer_text="Use option A.",
                answered_by=ConversationActor.USER,
                pending_action={"next": "must-not-prepare"},
                event=_conversation_answer(
                    sender=ConversationActor.USER,
                    addressed_to=ConversationTarget.FABLE,
                    routed_to=ConversationTarget.FABLE,
                    task_id="task-one",
                    revision=revision,
                    generation=generation,
                    question_id=question_id,
                    text="Use option A.",
                ),
            )
        assert store.get_task("task-one", 1) == before_task
        assert store.events_after("session-1", 0) == before_events
        assert tuple(store._connection.execute("SELECT * FROM agent_runs")) == before_runs
        assert store.question("question-one") == question

    assert_rejected(
        session_id="session-1", revision=1, question_id="question-one", generation=2,
    )
    assert_rejected(
        session_id="session-1", revision=1, question_id="wrong-question", generation=1,
    )
    assert_rejected(
        session_id="session-2", revision=1, question_id="question-one", generation=1,
    )
    assert_rejected(
        session_id="session-1", revision=2, question_id="question-one", generation=1,
    )
    answered = store.answer_question_and_prepare_resume(
        session_id="session-1",
        task_id="task-one",
        revision=1,
        question_id="question-one",
        expected_generation=1,
        answer_text="Use option A.",
        answered_by=ConversationActor.USER,
        pending_action={"next": "resume-with-user-answer"},
        event=_conversation_answer(
            sender=ConversationActor.USER,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            task_id="task-one",
            revision=1,
            generation=1,
            question_id="question-one",
            text="Use option A.",
        ),
    )
    assert answered.answer_text == "Use option A."
    assert answered.answered_by is ConversationActor.USER
    resumed = store.get_task("task-one", 1)
    assert resumed.state is TaskState.SOL_RUNNING
    assert resumed.continuation_state is None
    assert resumed.pending == {"next": "resume-with-user-answer"}
    assert resumed.continuation_generation == 2
    assert resumed.exchange_allowance == INITIAL_INTERNAL_EXCHANGES
    assert resumed.exchange_consumed == 0

    with pytest.raises(RuntimeError):
        store.answer_question_and_prepare_resume(
            session_id="session-1",
            task_id="task-one",
            revision=1,
            question_id="question-one",
            expected_generation=1,
            answer_text="Duplicate answer.",
            answered_by=ConversationActor.USER,
            pending_action={"next": "must-not-prepare"},
            event=_conversation_answer(
                sender=ConversationActor.USER,
                addressed_to=ConversationTarget.FABLE,
                routed_to=ConversationTarget.FABLE,
                task_id="task-one",
                revision=1,
                generation=1,
                question_id="question-one",
                text="Duplicate answer.",
            ),
        )

    agent_question = store.pause_for_question(
        session_id="session-1",
        task_id="task-one",
        revision=1,
        expected_generation=2,
        question_id="question-for-fable",
        asked_by=ConversationActor.SOL,
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        text="What does the approved brief mean here?",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"next": "resume-sol-after-fable"},
        event=_conversation_question(
            sender=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            task_id="task-one",
            revision=1,
            generation=2,
            question_id="question-for-fable",
            text="What does the approved brief mean here?",
        ),
    )
    with pytest.raises(RuntimeError, match="answer actor"):
        store.answer_question_and_prepare_resume(
            session_id="session-1",
            task_id="task-one",
            revision=1,
            question_id=agent_question.question_id,
            expected_generation=2,
            answer_text="Sol cannot answer its own routed question.",
            answered_by=ConversationActor.SOL,
            pending_action={"next": "must-not-prepare"},
            event=_conversation_answer(
                sender=ConversationActor.SOL,
                addressed_to=ConversationTarget.SOL,
                routed_to=ConversationTarget.SOL,
                task_id="task-one",
                revision=1,
                generation=2,
                question_id=agent_question.question_id,
                text="Sol cannot answer its own routed question.",
            ),
        )
    agent_answered = store.answer_question_and_prepare_resume(
        session_id="session-1",
        task_id="task-one",
        revision=1,
        question_id=agent_question.question_id,
        expected_generation=2,
        answer_text="Follow the approved brief exactly.",
        answered_by=ConversationActor.FABLE,
        pending_action={"next": "resume-sol-after-fable-answer"},
        event=_conversation_answer(
            sender=ConversationActor.FABLE,
            addressed_to=ConversationTarget.SOL,
            routed_to=ConversationTarget.SOL,
            task_id="task-one",
            revision=1,
            generation=2,
            question_id=agent_question.question_id,
            text="Follow the approved brief exactly.",
        ),
    )
    assert agent_answered.answered_by is ConversationActor.FABLE
    assert store.get_task("task-one", 1).continuation_generation == 2

    foreign = SQLiteStore(tmp_path / "foreign-project.sqlite3", clock=lambda: "2026-08-10T12:00:00Z")
    foreign.create_session("session-1", "/other-project")
    _save_active_directed_task(foreign, "session-1", first_brief)
    assert foreign.question("question-one") is None
    foreign.close()


def test_internal_exchange_reservations_are_bounded_and_grants_are_idempotent(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="exchange-task", revision=1)
    _save_active_directed_task(store, "session-1", brief)

    def reserve(question_id: str, request_key: str):
        return store.reserve_internal_question(
            session_id="session-1",
            task_id="exchange-task",
            revision=1,
            expected_generation=1,
            question_id=question_id,
            request_key=request_key,
            asked_by=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            text=f"Question {question_id} for Fable.",
            continuation_state=TaskState.SOL_RUNNING,
            pending_action={"next": f"resume-{question_id}"},
            event=_conversation_question(
                sender=ConversationActor.SOL,
                addressed_to=ConversationTarget.FABLE,
                routed_to=ConversationTarget.FABLE,
                task_id="exchange-task",
                revision=1,
                generation=1,
                question_id=question_id,
                text=f"Question {question_id} for Fable.",
            ),
        )

    def answer(question_id: str):
        return store.answer_question_and_prepare_resume(
            session_id="session-1",
            task_id="exchange-task",
            revision=1,
            question_id=question_id,
            expected_generation=1,
            answer_text=f"Answer {question_id} from Fable.",
            answered_by=ConversationActor.FABLE,
            pending_action={"next": f"resume-after-{question_id}"},
            event=_conversation_answer(
                sender=ConversationActor.FABLE,
                addressed_to=ConversationTarget.SOL,
                routed_to=ConversationTarget.SOL,
                task_id="exchange-task",
                revision=1,
                generation=1,
                question_id=question_id,
                text=f"Answer {question_id} from Fable.",
            ),
        )

    first_reservation, first_question = reserve("exchange-question-1", "request-1")
    assert first_reservation == ExchangeReservation(
        exchange_id=first_question.exchange_id,
        question_id="exchange-question-1",
        ordinal=1,
        continuation_generation=1,
    )
    assert first_question.exchange_id == first_reservation.exchange_id
    assert store.get_task("exchange-task", 1).exchange_allowance == 2
    assert store.get_task("exchange-task", 1).exchange_consumed == 1
    answer("exchange-question-1")

    second_reservation, _ = reserve("exchange-question-2", "request-2")
    answer("exchange-question-2")
    third_reservation, third_question = reserve("exchange-question-3", "request-3")
    assert (first_reservation.ordinal, second_reservation.ordinal, third_reservation.ordinal) == (1, 2, 3)
    exhausted = store.get_task("exchange-task", 1)
    assert exhausted.exchange_allowance == 0
    assert exhausted.exchange_consumed == 3
    assert third_question.answer_text is None
    answered_third = answer("exchange-question-3")
    assert answered_third.answered_by is ConversationActor.FABLE
    assert store.get_task("exchange-task", 1).exchange_allowance == 0

    paused = store.pause_for_exchange_permission(
        session_id="session-1",
        task_id="exchange-task",
        revision=1,
        expected_generation=1,
        attempted_question=DirectedAgentQuestion(
            addressed_to="fable",
            text="A fourth question needs permission.",
            reason="The initial exchange allowance is exhausted.",
        ),
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"next": "retry-fourth-question"},
        event=_conversation_permission(task_id="exchange-task", revision=1, generation=1),
    )
    assert paused.state is TaskState.AWAITING_USER_INPUT
    assert paused.continuation_state is TaskState.SOL_RUNNING
    assert dict(paused.pending or {}) == {
        "next": "retry-fourth-question",
        "attempted_question": {
            "addressed_to": "fable",
            "text": "A fourth question needs permission.",
            "reason": "The initial exchange allowance is exhausted.",
        },
    }
    assert store._connection.execute(
        "SELECT COUNT(*) FROM exchange_permissions WHERE grant_request_id IS NULL"
    ).fetchone()[0] == 1
    assert store._connection.execute("SELECT COUNT(*) FROM exchange_reservations").fetchone()[0] == 3
    assert store._connection.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 3

    assert store.grant_internal_exchanges(
        session_id="session-1",
        task_id="exchange-task",
        revision=1,
        expected_generation=1,
        request_id="grant-1",
    ) == EXCHANGE_GRANT_SIZE
    assert store.grant_internal_exchanges(
        session_id="session-1",
        task_id="exchange-task",
        revision=1,
        expected_generation=1,
        request_id="grant-1",
    ) == EXCHANGE_GRANT_SIZE
    before_distinct_grant = store.get_task("exchange-task", 1)
    with pytest.raises(RuntimeError, match="permission"):
        store.grant_internal_exchanges(
            session_id="session-1",
            task_id="exchange-task",
            revision=1,
            expected_generation=1,
            request_id="grant-2",
        )
    assert store.get_task("exchange-task", 1) == before_distinct_grant
    granted = store.get_task("exchange-task", 1)
    assert granted.exchange_allowance == EXCHANGE_GRANT_SIZE
    assert granted.exchange_consumed == 3
    assert store._connection.execute("SELECT COUNT(*) FROM exchange_grants").fetchone()[0] == 1
    assert store._connection.execute(
        """
        SELECT COUNT(*) FROM tasks
        WHERE exchange_allowance IS NULL OR exchange_consumed IS NULL
        """
    ).fetchone()[0] == 0
    assert store._connection.execute(
        "SELECT grant_request_id FROM exchange_permissions"
    ).fetchone()[0] == "grant-1"

    store.resume_continuation(
        "exchange-task", 1, expected=TaskState.AWAITING_USER_INPUT,
    )
    fourth_reservation, _ = reserve("exchange-question-4", "request-4")
    assert fourth_reservation.ordinal == 4
    assert store.get_task("exchange-task", 1).exchange_allowance == 2


def test_current_exchange_permission_projects_only_the_exact_ungranted_pause(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="permission-projection-task", revision=1)
    _save_active_directed_task(store, "session-1", brief)
    store._connection.execute(
        "UPDATE tasks SET exchange_allowance = 0 WHERE task_id = ? AND revision = ?",
        (brief.task_id, brief.revision),
    )
    paused = store.pause_for_exchange_permission(
        session_id="session-1",
        task_id=brief.task_id,
        revision=brief.revision,
        expected_generation=1,
        attempted_question=DirectedAgentQuestion(
            addressed_to="fable",
            text="A fourth question needs permission.",
            reason="The exchange allowance is exhausted.",
        ),
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"next": "retry-fourth-question", "provider_id": "must-not-project"},
        event=_conversation_permission(task_id=brief.task_id, revision=1, generation=1),
    )
    permission_id = store._connection.execute(  # noqa: SLF001 - exact persisted fixture
        "SELECT permission_id FROM exchange_permissions"
    ).fetchone()[0]

    assert store.current_exchange_permission(
        session_id="session-1",
        task_id=brief.task_id,
        revision=brief.revision,
    ) == {
        "request_id": permission_id,
        "revision": brief.revision,
        "continuation_generation": paused.continuation_generation,
    }

    store._connection.execute(  # noqa: SLF001 - stale generation must not project
        "UPDATE tasks SET continuation_generation = continuation_generation + 1 "
        "WHERE task_id = ? AND revision = ?",
        (brief.task_id, brief.revision),
    )
    assert store.current_exchange_permission(
        session_id="session-1", task_id=brief.task_id, revision=brief.revision,
    ) is None

    store._connection.execute(  # noqa: SLF001 - restore exact fixture before grant
        "UPDATE tasks SET continuation_generation = ? WHERE task_id = ? AND revision = ?",
        (paused.continuation_generation, brief.task_id, brief.revision),
    )
    assert store.grant_internal_exchanges(
        session_id="session-1",
        task_id=brief.task_id,
        revision=brief.revision,
        expected_generation=paused.continuation_generation,
        request_id=permission_id,
    ) == EXCHANGE_GRANT_SIZE
    assert store.current_exchange_permission(
        session_id="session-1", task_id=brief.task_id, revision=brief.revision,
    ) is None


def test_internal_exchange_request_key_is_concurrent_idempotent_and_transactional(
    tmp_path, valid_brief,
) -> None:
    path = tmp_path / "concurrent-exchanges.sqlite3"
    initial = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    initial.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="concurrent-task", revision=1)
    _save_active_directed_task(initial, "session-1", brief)
    initial.close()

    first = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z", check_same_thread=False)
    second = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z", check_same_thread=False)
    barrier = threading.Barrier(2)
    results: list[tuple[ExchangeReservation, QuestionRecord]] = []
    failures: list[BaseException] = []

    def reserve_once(store) -> None:
        try:
            barrier.wait(timeout=2)
            results.append(store.reserve_internal_question(
                session_id="session-1",
                task_id="concurrent-task",
                revision=1,
                expected_generation=1,
                question_id="concurrent-question",
                request_key="concurrent-request",
                asked_by=ConversationActor.SOL,
                addressed_to=ConversationTarget.FABLE,
                routed_to=ConversationTarget.FABLE,
                text="One concurrent question.",
                continuation_state=TaskState.SOL_RUNNING,
                pending_action={"next": "resume-concurrent"},
                event=_conversation_question(
                    sender=ConversationActor.SOL,
                    addressed_to=ConversationTarget.FABLE,
                    routed_to=ConversationTarget.FABLE,
                    task_id="concurrent-task",
                    revision=1,
                    generation=1,
                    question_id="concurrent-question",
                    text="One concurrent question.",
                ),
            ))
        except BaseException as error:
            failures.append(error)

    workers = [threading.Thread(target=reserve_once, args=(store,)) for store in (first, second)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
    assert all(worker.is_alive() is False for worker in workers)
    assert failures == []
    assert len(results) == 2
    assert results[0] == results[1]
    assert first._connection.execute("SELECT COUNT(*) FROM exchange_reservations").fetchone()[0] == 1
    assert first._connection.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 1
    assert len(first.events_after("session-1", 0)) == 1
    persisted = first.get_task("concurrent-task", 1)
    assert persisted.exchange_allowance == 2
    assert persisted.exchange_consumed == 1
    assert persisted.pending == {"next": "resume-concurrent"}
    first.close()
    second.close()

    rollback = _store(tmp_path)
    rollback.create_session("rollback-session", "/repo")
    rollback_brief = replace(valid_brief, task_id="rollback-task", revision=1)
    _save_active_directed_task(rollback, "rollback-session", rollback_brief)
    seen = []
    rollback.add_event_listener(seen.append)
    rollback._connection.execute(
        """
        CREATE TRIGGER fail_conversation_event
        BEFORE INSERT ON events WHEN NEW.kind = 'conversation'
        BEGIN
            SELECT RAISE(ABORT, 'injected conversation event failure');
        END
        """
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected conversation event failure"):
        rollback.reserve_internal_question(
            session_id="rollback-session",
            task_id="rollback-task",
            revision=1,
            expected_generation=1,
            question_id="rollback-question",
            request_key="rollback-request",
            asked_by=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            text="This transaction must roll back.",
            continuation_state=TaskState.SOL_RUNNING,
            pending_action={"next": "must-not-persist"},
            event=_conversation_question(
                sender=ConversationActor.SOL,
                addressed_to=ConversationTarget.FABLE,
                routed_to=ConversationTarget.FABLE,
                task_id="rollback-task",
                revision=1,
                generation=1,
                question_id="rollback-question",
                text="This transaction must roll back.",
            ),
        )
    assert rollback.get_task("rollback-task", 1).state is TaskState.SOL_RUNNING
    assert rollback._connection.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 0
    assert rollback._connection.execute("SELECT COUNT(*) FROM exchange_reservations").fetchone()[0] == 0
    assert rollback.events_after("rollback-session", 0) == ()
    assert seen == []


def test_internal_exchange_request_key_rejects_substituted_pending_action(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="request-key-task", revision=1)
    _save_active_directed_task(store, "session-1", brief)
    event = _conversation_question(
        sender=ConversationActor.SOL,
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        task_id="request-key-task",
        revision=1,
        generation=1,
        question_id="request-key-question",
        text="One exact durable question.",
    )
    first = store.reserve_internal_question(
        session_id="session-1",
        task_id="request-key-task",
        revision=1,
        expected_generation=1,
        question_id="request-key-question",
        request_key="request-key",
        asked_by=ConversationActor.SOL,
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        text="One exact durable question.",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"next": "resume-original"},
        event=event,
    )
    before_task = store.get_task("request-key-task", 1)
    before_events = store.events_after("session-1", 0)

    with pytest.raises(RuntimeError, match="request key"):
        store.reserve_internal_question(
            session_id="session-1",
            task_id="request-key-task",
            revision=1,
            expected_generation=1,
            question_id="request-key-question",
            request_key="request-key",
            asked_by=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            text="One exact durable question.",
            continuation_state=TaskState.SOL_RUNNING,
            pending_action={"next": "substituted-resume"},
            event=event,
        )

    assert store.get_task("request-key-task", 1) == before_task
    assert store.events_after("session-1", 0) == before_events
    assert store.reserve_internal_question(
        session_id="session-1",
        task_id="request-key-task",
        revision=1,
        expected_generation=1,
        question_id="request-key-question",
        request_key="request-key",
        asked_by=ConversationActor.SOL,
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        text="One exact durable question.",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"next": "resume-original"},
        event=event,
    ) == first


def test_question_generation_reset_and_non_question_work_do_not_consume_exchanges(
    tmp_path, valid_brief,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = _store(tmp_path)
    store.create_session("session-1", str(repo))
    first = replace(valid_brief, task_id="generation-task", revision=1)
    _save_active_directed_task(store, "session-1", first)
    store.set_fable_session("generation-task", 1, "fable-session")
    store.set_sol_thread("generation-task", 1, "sol-thread")
    store.increment_correction_count("generation-task", 1)
    unchanged = store.get_task("generation-task", 1)
    assert unchanged.exchange_allowance == INITIAL_INTERNAL_EXCHANGES
    assert unchanged.exchange_consumed == 0
    assert unchanged.continuation_generation == 1

    reservation, _ = store.reserve_internal_question(
        session_id="session-1",
        task_id="generation-task",
        revision=1,
        expected_generation=1,
        question_id="generation-question",
        request_key="generation-request",
        asked_by=ConversationActor.SOL,
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        text="Please resolve this internal ambiguity.",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"next": "resume-after-internal-answer"},
        event=_conversation_question(
            sender=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            task_id="generation-task",
            revision=1,
            generation=1,
            question_id="generation-question",
            text="Please resolve this internal ambiguity.",
        ),
    )
    assert reservation.ordinal == 1
    assert store.get_task("generation-task", 1).exchange_allowance == 2
    store.answer_question_and_prepare_resume(
        session_id="session-1",
        task_id="generation-task",
        revision=1,
        question_id="generation-question",
        expected_generation=1,
        answer_text="The approved brief controls the ambiguity.",
        answered_by=ConversationActor.FABLE,
        pending_action={"next": "resume-after-internal-answer"},
        event=_conversation_answer(
            sender=ConversationActor.FABLE,
            addressed_to=ConversationTarget.SOL,
            routed_to=ConversationTarget.SOL,
            task_id="generation-task",
            revision=1,
            generation=1,
            question_id="generation-question",
            text="The approved brief controls the ambiguity.",
        ),
    )
    store.pause_for_question(
        session_id="session-1",
        task_id="generation-task",
        revision=1,
        expected_generation=1,
        question_id="generation-user-question",
        asked_by=ConversationActor.SOL,
        addressed_to=ConversationTarget.USER,
        routed_to=ConversationTarget.USER,
        text="Please provide new human direction.",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"next": "resume-after-human-direction"},
        event=_conversation_question(
            sender=ConversationActor.SOL,
            addressed_to=ConversationTarget.USER,
            routed_to=ConversationTarget.USER,
            task_id="generation-task",
            revision=1,
            generation=1,
            question_id="generation-user-question",
            text="Please provide new human direction.",
        ),
    )
    user_answer = store.answer_question_and_prepare_resume(
        session_id="session-1",
        task_id="generation-task",
        revision=1,
        question_id="generation-user-question",
        expected_generation=1,
        answer_text="Use the approved option.",
        answered_by=ConversationActor.USER,
        pending_action={"next": "resume-with-new-direction"},
        event=_conversation_answer(
            sender=ConversationActor.USER,
            addressed_to=ConversationTarget.SOL,
            routed_to=ConversationTarget.SOL,
            task_id="generation-task",
            revision=1,
            generation=1,
            question_id="generation-user-question",
            text="Use the approved option.",
        ),
    )
    assert user_answer.answered_by is ConversationActor.USER
    reset = store.get_task("generation-task", 1)
    assert reset.continuation_generation == 2
    assert reset.exchange_allowance == INITIAL_INTERNAL_EXCHANGES
    assert reset.exchange_consumed == 0
    with pytest.raises(RuntimeError):
        store.grant_internal_exchanges(
            session_id="session-1",
            task_id="generation-task",
            revision=1,
            expected_generation=1,
            request_id="stale-generation-grant",
        )

    store.transition_task(
        "generation-task", 1,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.FABLE_REVIEWING,
    )
    after_review_transition = store.get_task("generation-task", 1)
    assert after_review_transition.exchange_allowance == INITIAL_INTERNAL_EXCHANGES
    assert after_review_transition.exchange_consumed == 0

    approved = replace(valid_brief, task_id="approval-budget-task", revision=1)
    store.save_task("session-1", approved, TaskState.AWAITING_USER_APPROVAL)
    store.prepare_approval_action(
        project_id=project_id_for_root(repo),
        session_id="session-1",
        task_id="approval-budget-task",
        revision=1,
        generation=1,
        payload=ApprovalPayload(baseline_id="baseline-one", baseline_setting=None, scope=None),
    )
    next_revision = replace(approved, revision=2, title="Approved second revision")
    store.save_task("session-1", next_revision, TaskState.AWAITING_USER_APPROVAL)
    store.prepare_approval_action(
        project_id=project_id_for_root(repo),
        session_id="session-1",
        task_id="approval-budget-task",
        revision=2,
        generation=1,
        payload=ApprovalPayload(baseline_id="baseline-two", baseline_setting=None, scope=None),
    )
    separate = store.get_task("approval-budget-task", 2)
    assert separate.continuation_generation == 1
    assert separate.exchange_allowance == INITIAL_INTERNAL_EXCHANGES
    assert separate.exchange_consumed == 0


def test_human_direction_reset_helper_rejects_autocommit_invocation(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="reset-helper-task", revision=1)
    _save_active_directed_task(store, "session-1", brief)
    before = store.get_task("reset-helper-task", 1)

    with pytest.raises(RuntimeError, match="transaction"):
        store._reset_internal_exchanges_for_human_direction_in_transaction(
            store._connection.cursor(),
            session_id="session-1",
            task_id="reset-helper-task",
            revision=1,
            expected_generation=1,
        )

    assert store.get_task("reset-helper-task", 1) == before


def test_answer_rejects_a_question_detached_from_its_exact_pause(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="detached-question-task", revision=1)
    _save_active_directed_task(store, "session-1", brief)
    store.pause_for_question(
        session_id="session-1",
        task_id="detached-question-task",
        revision=1,
        expected_generation=1,
        question_id="detached-question",
        asked_by=ConversationActor.SOL,
        addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE,
        text="Resolve the original exact ambiguity.",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"next": "resume-original-question"},
        event=_conversation_question(
            sender=ConversationActor.SOL,
            addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE,
            task_id="detached-question-task",
            revision=1,
            generation=1,
            question_id="detached-question",
            text="Resolve the original exact ambiguity.",
        ),
    )
    store.resume_continuation(
        "detached-question-task",
        1,
        expected=TaskState.AWAITING_USER_INPUT,
    )
    store.pause_for_continuation(
        "detached-question-task",
        1,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=TaskState.SOL_RUNNING,
        pending={"next": "resume-original-question"},
    )
    before_task = store.get_task("detached-question-task", 1)
    before_question = store.question("detached-question")
    before_events = store.events_after("session-1", 0)

    with pytest.raises(RuntimeError, match="question continuation"):
        store.answer_question_and_prepare_resume(
            session_id="session-1",
            task_id="detached-question-task",
            revision=1,
            question_id="detached-question",
            expected_generation=1,
            answer_text="This answer belongs to the original pause only.",
            answered_by=ConversationActor.FABLE,
            pending_action={"next": "must-not-replace-unrelated-pause"},
            event=_conversation_answer(
                sender=ConversationActor.FABLE,
                addressed_to=ConversationTarget.SOL,
                routed_to=ConversationTarget.SOL,
                task_id="detached-question-task",
                revision=1,
                generation=1,
                question_id="detached-question",
                text="This answer belongs to the original pause only.",
            ),
        )

    assert store.get_task("detached-question-task", 1) == before_task
    assert store.question("detached-question") == before_question
    assert store.events_after("session-1", 0) == before_events


def test_exchange_permission_requires_store_provenance_and_a_system_status_card(
    tmp_path, valid_brief,
) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="permission-provenance-task", revision=1)
    _save_active_directed_task(store, "session-1", brief)
    store._connection.execute(
        """
        UPDATE tasks SET exchange_allowance = 0
        WHERE task_id = ? AND revision = ?
        """,
        ("permission-provenance-task", 1),
    )
    store.pause_for_continuation(
        "permission-provenance-task",
        1,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=TaskState.SOL_RUNNING,
        pending={
            "next": "forged-permission",
            "attempted_question": {
                "addressed_to": "fable",
                "text": "A fabricated fourth question.",
                "reason": "No durable permission record exists.",
            },
            "exchange_permission_id": "f" * 48,
        },
    )
    before_forged_grant = store.get_task("permission-provenance-task", 1)

    with pytest.raises(RuntimeError, match="permission"):
        store.grant_internal_exchanges(
            session_id="session-1",
            task_id="permission-provenance-task",
            revision=1,
            expected_generation=1,
            request_id="forged-grant",
        )

    assert store.get_task("permission-provenance-task", 1) == before_forged_grant
    assert store._connection.execute(
        "SELECT COUNT(*) FROM exchange_grants"
    ).fetchone()[0] == 0


def test_exchange_permission_requires_a_system_status_card(tmp_path, valid_brief) -> None:
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="permission-card-task", revision=1)
    _save_active_directed_task(store, "session-1", brief)
    store._connection.execute(
        """
        UPDATE tasks SET exchange_allowance = 0
        WHERE task_id = ? AND revision = ?
        """,
        ("permission-card-task", 1),
    )
    before_invalid_card = store.get_task("permission-card-task", 1)
    before_events = store.events_after("session-1", 0)

    with pytest.raises(ValueError, match="exchange permission event"):
        store.pause_for_exchange_permission(
            session_id="session-1",
            task_id="permission-card-task",
            revision=1,
            expected_generation=1,
            attempted_question=DirectedAgentQuestion(
                addressed_to="fable",
                text="A fourth question needs permission.",
                reason="The allowance is exhausted.",
            ),
            continuation_state=TaskState.SOL_RUNNING,
            pending_action={"next": "retry-fourth-question"},
            event=ConversationEnvelope(
                sender=ConversationActor.FABLE,
                addressed_to=ConversationTarget.USER,
                routed_to=ConversationTarget.USER,
                message_type=ConversationMessageType.STATEMENT,
                text="This is not an exchange permission card.",
                task_id="permission-card-task",
                revision=1,
                continuation_generation=1,
            ),
        )

    assert store.get_task("permission-card-task", 1) == before_invalid_card
    assert store.events_after("session-1", 0) == before_events


def test_nested_fable_clarification_evidence_pauses_and_restores_exact_context(
    tmp_path, valid_brief,
) -> None:
    """A nested evidence answer must restore Fable, never resume Sol directly."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="nested-clarification", revision=1)
    _save_active_directed_task(store, "session-1", brief)
    pending = {
        "clarification_prompt": "Resolve the exact approved ambiguity.",
        "sol_run_id": "sol-run-1",
        "prompt": "Resume only the approved work.",
    }
    store._connection.execute(
        """
        UPDATE tasks SET state = ?, approved_at = ?, fable_session_id = ?, sol_thread_id = ?,
            baseline_id = ?, pending_json = ? WHERE task_id = ? AND revision = ?
        """,
        (
            TaskState.FABLE_CLARIFYING.value, "2026-08-10T12:00:00Z", "fable-session-1", "sol-thread-1",
            "baseline-1", json.dumps(pending, separators=(",", ":"), sort_keys=True),
            brief.task_id, brief.revision,
        ),
    )

    reservation, nested = store.reserve_fable_clarification_evidence_question(
        session_id="session-1", task_id=brief.task_id, revision=brief.revision,
        expected_generation=1, question_id="nested-clarification-question",
        request_key="nested-clarification-request",
        text="Which approved rule is already exercised?",
        event=_conversation_question(
            sender=ConversationActor.FABLE, addressed_to=ConversationTarget.SOL,
            routed_to=ConversationTarget.SOL, task_id=brief.task_id,
            revision=brief.revision, generation=1,
            question_id="nested-clarification-question",
            text="Which approved rule is already exercised?",
        ),
    )

    assert reservation.ordinal == 1
    assert nested.nested_parent_kind == "clarification"
    assert nested.parent_question_id is None
    paused = store.get_task(brief.task_id, brief.revision)
    assert (paused.state, paused.continuation_state, paused.pending) == (
        TaskState.AWAITING_USER_INPUT, TaskState.FABLE_CLARIFYING, pending,
    )
    assert (paused.exchange_allowance, paused.exchange_consumed) == (2, 1)

    with pytest.raises(RuntimeError, match="nested"):
        store.answer_question_and_prepare_resume(
            session_id="session-1", task_id=brief.task_id, revision=brief.revision,
            question_id=nested.question_id, expected_generation=1,
            answer_text="The existing focused test proves it.",
            answered_by=ConversationActor.SOL, pending_action={"must": "not resume"},
            event=_conversation_answer(
                sender=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
                routed_to=ConversationTarget.FABLE, task_id=brief.task_id,
                revision=brief.revision, generation=1, question_id=nested.question_id,
                text="The existing focused test proves it.",
            ),
        )

    answered = store.answer_fable_clarification_evidence_question_and_resume(
        session_id="session-1", task_id=brief.task_id, revision=brief.revision,
        question_id=nested.question_id, expected_generation=1,
        answer_text="The existing focused test proves it.",
        event=_conversation_answer(
            sender=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE, task_id=brief.task_id,
            revision=brief.revision, generation=1, question_id=nested.question_id,
            text="The existing focused test proves it.",
        ),
    )

    assert answered.answered_by is ConversationActor.SOL
    resumed = store.get_task(brief.task_id, brief.revision)
    assert (resumed.state, resumed.continuation_state, resumed.pending) == (
        TaskState.FABLE_CLARIFYING, None, pending,
    )
    assert (resumed.exchange_allowance, resumed.exchange_consumed) == (2, 1)
    assert [event.payload["message_type"] for event in store.events_after("session-1", 0)] == [
        "question", "answer",
    ]


def test_interrupt_directed_answer_for_stop_preserves_an_exact_nested_child(
    tmp_path,
    valid_brief,
) -> None:
    """A nested Sol child must keep its pause when its exact run is terminalized."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="nested-stop", revision=1)
    _save_active_directed_task(store, "session-1", brief)
    pending = {
        "clarification_prompt": "Resolve the exact approved ambiguity.",
        "sol_run_id": "sol-run-1",
        "prompt": "Resume only the approved work.",
    }
    store._connection.execute(
        """
        UPDATE tasks SET state = ?, approved_at = ?, fable_session_id = ?, sol_thread_id = ?,
            baseline_id = ?, pending_json = ? WHERE task_id = ? AND revision = ?
        """,
        (
            TaskState.FABLE_CLARIFYING.value, "2026-08-10T12:00:00Z",
            "fable-session-1", "sol-thread-1", "baseline-1",
            json.dumps(pending, separators=(",", ":"), sort_keys=True),
            brief.task_id, brief.revision,
        ),
    )
    _, nested = store.reserve_fable_clarification_evidence_question(
        session_id="session-1", task_id=brief.task_id, revision=brief.revision,
        expected_generation=1, question_id="nested-stop-question",
        request_key="nested-stop-request", text="Which approved rule is verified?",
        event=_conversation_question(
            sender=ConversationActor.FABLE, addressed_to=ConversationTarget.SOL,
            routed_to=ConversationTarget.SOL, task_id=brief.task_id,
            revision=brief.revision, generation=1,
            question_id="nested-stop-question", text="Which approved rule is verified?",
        ),
    )
    store.start_agent_run("nested-stop-run", brief.task_id, brief.revision, "sol")
    store.set_agent_run_session("nested-stop-run", "sol-thread-1")
    before = (
        store.get_task(brief.task_id, brief.revision),
        store.question(nested.question_id),
        store.agent_run("nested-stop-run"),
        store.events_after("session-1", 0),
    )

    with pytest.raises(RuntimeError):
        store.interrupt_directed_answer_for_stop(
            brief.task_id, brief.revision,
            run_id="nested-stop-run", question_id="other-question",
        )
    assert (
        store.get_task(brief.task_id, brief.revision),
        store.question(nested.question_id),
        store.agent_run("nested-stop-run"),
        store.events_after("session-1", 0),
    ) == before

    interrupted = store.interrupt_directed_answer_for_stop(
        brief.task_id, brief.revision,
        run_id="nested-stop-run", question_id=nested.question_id,
    )

    assert interrupted.state is TaskState.INTERRUPTED
    assert interrupted.continuation_state is TaskState.FABLE_CLARIFYING
    assert store.question(nested.question_id) == nested
    assert store.agent_run("nested-stop-run").status == "running"
    assert store.events_after("session-1", 0) == before[3]


@pytest.mark.parametrize(
    "mutation",
    (
        "exact",
        "missing_parent",
        "changed_parent",
        "answered_parent",
        "changed_parent_generation",
        "changed_parent_route",
        "downgraded_child_to_clarification",
        "substituted_binding_child",
        "substituted_binding_parent",
    ),
)
def test_interrupt_directed_answer_for_stop_authenticates_question_nested_parent_and_binding(
    tmp_path,
    valid_brief,
    mutation: str,
) -> None:
    """A nested-question child can stop only under its exact parent intervention image."""
    store = _store(tmp_path)
    record = _nested_parent_intervention_claim_for_tamper_matrix(store, valid_brief)
    binding = record.directed_binding
    assert binding is not None
    child = store.question(binding.question_id)
    assert child is not None
    assert child.nested_parent_kind == "question"
    assert child.parent_question_id == binding.parent_question_id
    assert binding.parent_question_id == "original-question"

    if mutation == "missing_parent":
        store._connection.execute(  # noqa: SLF001 - parent identity tamper boundary
            "UPDATE questions SET parent_question_id = NULL WHERE question_id = ?",
            (child.question_id,),
        )
    elif mutation == "changed_parent":
        store._connection.execute(  # noqa: SLF001 - parent substitution boundary
            "UPDATE questions SET parent_question_id = 'substituted-parent' WHERE question_id = ?",
            (child.question_id,),
        )
    elif mutation == "answered_parent":
        store._connection.execute(  # noqa: SLF001 - parent lifecycle tamper boundary
            "UPDATE questions SET answer_text = 'answered early', answered_by = 'fable' "
            "WHERE question_id = 'original-question'",
        )
    elif mutation == "changed_parent_generation":
        store._connection.execute(  # noqa: SLF001 - parent generation tamper boundary
            "UPDATE questions SET continuation_generation = continuation_generation + 1 "
            "WHERE question_id = 'original-question'",
        )
    elif mutation == "changed_parent_route":
        store._connection.execute(  # noqa: SLF001 - parent route tamper boundary
            "UPDATE questions SET addressed_to = 'sol', routed_to = 'sol' "
            "WHERE question_id = 'original-question'",
        )
    elif mutation == "downgraded_child_to_clarification":
        store._connection.execute(  # noqa: SLF001 - valid child-shape downgrade boundary
            "UPDATE questions SET nested_parent_kind = 'clarification', "
            "parent_question_id = NULL, parent_continuation_pause_id = NULL "
            "WHERE question_id = ?",
            (child.question_id,),
        )
    elif mutation == "substituted_binding_child":
        store._connection.execute(  # noqa: SLF001 - owning binding tamper boundary
            "UPDATE interventions SET directed_binding_json = json_set("
            "directed_binding_json, '$.question_id', 'substituted-child') "
            "WHERE intervention_id = ?",
            (record.intervention_id,),
        )
    elif mutation == "substituted_binding_parent":
        store._connection.execute(  # noqa: SLF001 - owning binding tamper boundary
            "UPDATE interventions SET directed_binding_json = json_set("
            "directed_binding_json, '$.parent_question_id', 'substituted-parent') "
            "WHERE intervention_id = ?",
            (record.intervention_id,),
        )
    elif mutation != "exact":
        raise AssertionError(f"unknown mutation: {mutation}")

    tables = ("tasks", "questions", "agent_runs", "interventions", "events")
    before = {
        table: tuple(tuple(row) for row in store._connection.execute(  # noqa: SLF001
            f"SELECT * FROM {table} ORDER BY rowid"
        ))
        for table in tables
    }
    if mutation == "exact":
        interrupted = store.interrupt_directed_answer_for_stop(
            valid_brief.task_id,
            valid_brief.revision,
            run_id=binding.source_run_id,
            question_id=child.question_id,
        )
        assert interrupted.state is TaskState.INTERRUPTED
        assert interrupted.continuation_state is TaskState.FABLE_CLARIFYING
    else:
        with pytest.raises(RuntimeError):
            store.interrupt_directed_answer_for_stop(
                valid_brief.task_id,
                valid_brief.revision,
                run_id=binding.source_run_id,
                question_id=child.question_id,
            )
        assert {
            table: tuple(tuple(row) for row in store._connection.execute(  # noqa: SLF001
                f"SELECT * FROM {table} ORDER BY rowid"
            ))
            for table in tables
        } == before


def test_nested_fable_answer_evidence_binds_outer_question_and_reopens_idempotently(
    tmp_path, valid_brief,
) -> None:
    """A nested child must not close or double-charge the durable outer question."""
    path = tmp_path / "nested-outer.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="nested-outer", revision=1)
    _save_active_directed_task(store, "session-1", brief)
    store._connection.execute(
        """
        UPDATE tasks SET approved_at = ?, fable_session_id = ?, sol_thread_id = ?, baseline_id = ?
        WHERE task_id = ? AND revision = ?
        """,
        ("2026-08-10T12:00:00Z", "fable-session-1", "sol-thread-1", "baseline-1", brief.task_id, brief.revision),
    )
    _, outer = store.reserve_internal_question(
        session_id="session-1", task_id=brief.task_id, revision=brief.revision,
        expected_generation=1, question_id="outer-sol-question",
        request_key="outer-sol-request", asked_by=ConversationActor.SOL,
        addressed_to=ConversationTarget.FABLE, routed_to=ConversationTarget.FABLE,
        text="Which exact approved constraint resolves this?",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"sol_run_id": "sol-run-1", "prompt": "Resume exactly."},
        event=_conversation_question(
            sender=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE, task_id=brief.task_id,
            revision=brief.revision, generation=1, question_id="outer-sol-question",
            text="Which exact approved constraint resolves this?",
        ),
    )
    _, nested = store.reserve_fable_answer_evidence_question(
        session_id="session-1", task_id=brief.task_id, revision=brief.revision,
        expected_generation=1, outer_question_id=outer.question_id,
        question_id="nested-sol-evidence", request_key="nested-sol-evidence-request",
        text="Which focused test already proves that constraint?",
        event=_conversation_question(
            sender=ConversationActor.FABLE, addressed_to=ConversationTarget.SOL,
            routed_to=ConversationTarget.SOL, task_id=brief.task_id,
            revision=brief.revision, generation=1, question_id="nested-sol-evidence",
            text="Which focused test already proves that constraint?",
        ),
    )

    assert nested.nested_parent_kind == "question"
    assert nested.parent_question_id == outer.question_id
    assert store.question(outer.question_id) == outer
    assert store.get_task(brief.task_id, brief.revision).exchange_consumed == 2
    with pytest.raises(RuntimeError, match="nested"):
        store.answer_question_and_prepare_resume(
            session_id="session-1", task_id=brief.task_id, revision=brief.revision,
            question_id=outer.question_id, expected_generation=1,
            answer_text="This must wait for nested evidence.",
            answered_by=ConversationActor.FABLE, pending_action={"must": "not resume"},
            event=_conversation_answer(
                sender=ConversationActor.FABLE, addressed_to=ConversationTarget.SOL,
                routed_to=ConversationTarget.SOL, task_id=brief.task_id,
                revision=brief.revision, generation=1, question_id=outer.question_id,
                text="This must wait for nested evidence.",
            ),
        )
    store.close()

    reopened = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    replayed, replayed_nested = reopened.reserve_fable_answer_evidence_question(
        session_id="session-1", task_id=brief.task_id, revision=brief.revision,
        expected_generation=1, outer_question_id=outer.question_id,
        question_id=nested.question_id, request_key="nested-sol-evidence-request",
        text="Which focused test already proves that constraint?",
        event=_conversation_question(
            sender=ConversationActor.FABLE, addressed_to=ConversationTarget.SOL,
            routed_to=ConversationTarget.SOL, task_id=brief.task_id,
            revision=brief.revision, generation=1, question_id=nested.question_id,
            text="Which focused test already proves that constraint?",
        ),
    )
    assert (replayed.question_id, replayed_nested) == (nested.question_id, nested)
    assert reopened.get_task(brief.task_id, brief.revision).exchange_consumed == 2
    assert len(reopened.events_after("session-1", 0)) == 2

    answered = reopened.answer_fable_answer_evidence_question(
        session_id="session-1", task_id=brief.task_id, revision=brief.revision,
        outer_question_id=outer.question_id, question_id=nested.question_id,
        expected_generation=1, answer_text="test_store.py proves the exact pause contract.",
        event=_conversation_answer(
            sender=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE, task_id=brief.task_id,
            revision=brief.revision, generation=1, question_id=nested.question_id,
            text="test_store.py proves the exact pause contract.",
        ),
    )
    assert answered.answered_by is ConversationActor.SOL
    assert reopened.question(outer.question_id) == outer
    assert reopened.get_task(brief.task_id, brief.revision).state is TaskState.AWAITING_USER_INPUT
    reopened.close()


def test_nested_question_migration_audit_and_recovery_reject_mixed_parent_identity(
    tmp_path, valid_brief,
) -> None:
    """A malformed child marker must fail closed before audit or recovery acts."""
    path = tmp_path / "nested-invalid.sqlite3"
    store = SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="nested-invalid", revision=1)
    _save_active_directed_task(store, "session-1", brief)
    store.pause_for_question(
        session_id="session-1", task_id=brief.task_id, revision=brief.revision,
        expected_generation=1, question_id="top-level-question",
        asked_by=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
        routed_to=ConversationTarget.FABLE, text="Which exact fact applies?",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"sol_run_id": "sol-run-1", "prompt": "Resume exactly."},
        event=_conversation_question(
            sender=ConversationActor.SOL, addressed_to=ConversationTarget.FABLE,
            routed_to=ConversationTarget.FABLE, task_id=brief.task_id,
            revision=brief.revision, generation=1, question_id="top-level-question",
            text="Which exact fact applies?",
        ),
    )
    store._connection.execute(
        "UPDATE questions SET parent_question_id = ? WHERE question_id = ?",
        ("forged-parent", "top-level-question"),
    )

    with pytest.raises(RuntimeError, match="question_integrity"):
        store.audit_legacy_project_ownership("/repo")
    with pytest.raises(RuntimeError, match="nested question"):
        store.recover_active_tasks()
    store.close()

    with pytest.raises(RuntimeError, match="nested question"):
        SQLiteStore(path, clock=lambda: "2026-08-10T12:00:00Z")


def test_zero_allowance_nested_clarification_persists_permission_without_losing_parent(
    tmp_path, valid_brief,
) -> None:
    """Removing the nested permission branch would drop a grantable Fable parent."""
    store = _store(tmp_path)
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="nested-limit", revision=1)
    _save_active_directed_task(store, "session-1", brief)
    pending = {
        "clarification_prompt": "Keep this exact Fable clarification.",
        "sol_run_id": "sol-run-1",
        "prompt": "Resume exactly.",
    }
    store._connection.execute(
        """
        UPDATE tasks SET state = ?, approved_at = ?, fable_session_id = ?, sol_thread_id = ?,
            baseline_id = ?, exchange_allowance = 0, pending_json = ?
        WHERE task_id = ? AND revision = ?
        """,
        (
            TaskState.FABLE_CLARIFYING.value, "2026-08-10T12:00:00Z",
            "fable-session-1", "sol-thread-1", "baseline-1",
            json.dumps(pending, separators=(",", ":"), sort_keys=True),
            brief.task_id, brief.revision,
        ),
    )

    paused = store.pause_fable_clarification_evidence_permission(
        session_id="session-1", task_id=brief.task_id, revision=brief.revision,
        expected_generation=1,
        attempted_question=DirectedAgentQuestion(
            addressed_to="sol", text="Which exact evidence is missing?",
            reason="Fable needs one bounded execution fact.",
        ),
        event=_conversation_permission(task_id=brief.task_id, revision=brief.revision, generation=1),
    )

    assert (paused.state, paused.continuation_state, paused.pending) == (
        TaskState.AWAITING_USER_INPUT, TaskState.FABLE_CLARIFYING,
        {**pending, "attempted_question": {
            "addressed_to": "sol", "text": "Which exact evidence is missing?",
            "reason": "Fable needs one bounded execution fact.",
        }},
    )
    assert store.grant_internal_exchanges(
        session_id="session-1", task_id=brief.task_id, revision=brief.revision,
        expected_generation=1, request_id="nested-limit-grant",
    ) == EXCHANGE_GRANT_SIZE


def test_typed_conversation_preparations_are_atomic_and_survive_reload(
    tmp_path, valid_brief,
) -> None:
    """Removing any typed row, CAS, or event must make this workflow fail closed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    store = _store(tmp_path)
    store.create_session("session-1", str(repo))
    project_id = project_id_for_root(repo)
    context = SolResumeContext(
        sol_thread_id="sol-thread-1",
        sol_run_id="sol-run-1",
        prompt="Continue the exact approved work.",
    )

    continuation_brief = replace(
        valid_brief, task_id="typed-continuation-task", revision=1,
    )
    _save_active_directed_task(store, "session-1", continuation_brief)
    store.pause_for_continuation(
        "typed-continuation-task",
        1,
        expected=TaskState.SOL_RUNNING,
        target=TaskState.AWAITING_USER_INPUT,
        continuation_state=TaskState.SOL_RUNNING,
        pending={"sol_run_id": context.sol_run_id, "prompt": context.prompt},
    )
    store.set_sol_thread("typed-continuation-task", 1, context.sol_thread_id)
    continuation_payload = ContinuationMessagePayload(
        text="Continue with the documented approach.",
        addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.SOL,
        continuation_generation=1,
        continuation=context,
    )

    continuation = store.prepare_continuation_message_action(
        project_id=project_id,
        session_id="session-1",
        task_id="typed-continuation-task",
        revision=1,
        generation=41,
        payload=continuation_payload,
    )

    assert continuation.action == "continuation_message"
    assert continuation.payload == continuation_payload
    assert continuation.source_state is TaskState.AWAITING_USER_INPUT
    assert continuation.active_state is TaskState.SOL_RUNNING
    assert store.get_task("typed-continuation-task", 1).state is TaskState.SOL_RUNNING
    assert store.events_after("session-1", 0)[-1].payload == ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.SOL,
        message_type=ConversationMessageType.STATEMENT,
        text="Continue with the documented approach.",
        task_id="typed-continuation-task",
        revision=1,
        continuation_generation=1,
    ).to_dict()

    answer_brief = replace(valid_brief, task_id="typed-answer-task", revision=1)
    _save_active_directed_task(store, "session-1", answer_brief)
    store.pause_for_question(
        session_id="session-1",
        task_id="typed-answer-task",
        revision=1,
        expected_generation=1,
        question_id="typed-user-question",
        asked_by=ConversationActor.SOL,
        addressed_to=ConversationTarget.USER,
        routed_to=ConversationTarget.USER,
        text="Which approved option should Sol use?",
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"sol_run_id": context.sol_run_id, "prompt": context.prompt},
        event=_conversation_question(
            sender=ConversationActor.SOL,
            addressed_to=ConversationTarget.USER,
            routed_to=ConversationTarget.USER,
            task_id="typed-answer-task",
            revision=1,
            generation=1,
            question_id="typed-user-question",
            text="Which approved option should Sol use?",
        ),
    )
    store.set_sol_thread("typed-answer-task", 1, context.sol_thread_id)
    answer_payload = QuestionAnswerPayload(
        question_id="typed-user-question",
        answer="Use the documented option.",
        continuation_generation=1,
        continuation=context,
    )

    answer = store.prepare_question_answer_action(
        project_id=project_id,
        session_id="session-1",
        task_id="typed-answer-task",
        revision=1,
        generation=42,
        payload=answer_payload,
    )

    assert answer.action == "question_answer"
    assert answer.payload == answer_payload
    assert store.question("typed-user-question").answer_text == "Use the documented option."
    answered_task = store.get_task("typed-answer-task", 1)
    assert answered_task.state is TaskState.SOL_RUNNING
    assert answered_task.continuation_generation == 2
    assert store.events_after("session-1", 0)[-1].payload == _conversation_answer(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.SOL,
        routed_to=ConversationTarget.SOL,
        task_id="typed-answer-task",
        revision=1,
        generation=1,
        question_id="typed-user-question",
        text="Use the documented option.",
    ).to_dict()

    grant_brief = replace(valid_brief, task_id="typed-grant-task", revision=1)
    _save_active_directed_task(store, "session-1", grant_brief)
    store._connection.execute(
        "UPDATE tasks SET exchange_allowance = 0 WHERE task_id = ? AND revision = ?",
        ("typed-grant-task", 1),
    )
    attempted = DirectedAgentQuestion(
        addressed_to="fable",
        text="Please resolve the fourth exact ambiguity.",
        reason="The initial finite allowance is exhausted.",
    )
    store.pause_for_exchange_permission(
        session_id="session-1",
        task_id="typed-grant-task",
        revision=1,
        expected_generation=1,
        attempted_question=attempted,
        continuation_state=TaskState.SOL_RUNNING,
        pending_action={"sol_run_id": context.sol_run_id, "prompt": context.prompt},
        event=_conversation_permission(
            task_id="typed-grant-task", revision=1, generation=1,
        ),
    )
    store.set_sol_thread("typed-grant-task", 1, context.sol_thread_id)
    grant_payload = ExchangeGrantPayload(
        request_id="typed-grant-request",
        continuation_generation=1,
        attempted_question=attempted,
        continuation=context,
        parent_mode="top_level",
    )

    grant = store.prepare_exchange_grant_action(
        project_id=project_id,
        session_id="session-1",
        task_id="typed-grant-task",
        revision=1,
        generation=43,
        payload=grant_payload,
    )

    assert grant.action == "exchange_grant"
    assert grant.payload == grant_payload
    granted_task = store.get_task("typed-grant-task", 1)
    assert granted_task.state is TaskState.SOL_RUNNING
    assert granted_task.exchange_allowance == EXCHANGE_GRANT_SIZE
    assert store.events_after("session-1", 0)[-1].payload == ConversationEnvelope(
        sender=ConversationActor.USER,
        addressed_to=ConversationTarget.TEAM,
        routed_to=ConversationTarget.FABLE,
        message_type=ConversationMessageType.APPROVAL,
        text="Allow three more internal exchanges.",
        task_id="typed-grant-task",
        revision=1,
    ).to_dict()

    store.set_setting("agent_bridge.active_session_id", "session-1")
    store.close()
    reloaded = SQLiteStore(
        tmp_path / "bridge.sqlite3", clock=lambda: "2026-08-10T12:00:00Z",
    )
    assert reloaded.prepared_action(continuation.preparation_id) == continuation
    assert reloaded.prepared_action(answer.preparation_id) == answer
    assert reloaded.prepared_action(grant.preparation_id) == grant
    reloaded.audit_legacy_project_ownership(str(repo.resolve()))


@pytest.mark.parametrize(
    "tamper",
    ("mode", "outer", "task", "revision", "generation", "answer", "identity"),
)
def test_claimed_question_grant_checkpoint_rejects_each_tampered_authority_link(
    tmp_path, valid_brief, tamper: str,
) -> None:
    """Changing any persisted parent checkpoint link must fail before another agent starts."""
    store = _store(tmp_path)
    (tmp_path / "repo").mkdir()
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="checkpoint-tamper", revision=1)
    _save_active_directed_task(store, "session-1", brief)
    context = SolResumeContext("sol-thread-1", "sol-run-1", "Resume exactly.")
    pending = {"sol_run_id": context.sol_run_id, "prompt": context.prompt}
    store._connection.execute(
        """UPDATE tasks SET state = ?, continuation_state = ?, pending_json = ?,
        continuation_pause_id = ?, fable_session_id = ?, sol_thread_id = ?,
        approved_at = ?, baseline_id = ? WHERE task_id = ? AND revision = ?""",
        (TaskState.AWAITING_USER_INPUT.value, TaskState.SOL_RUNNING.value,
         json.dumps(pending, separators=(",", ":"), sort_keys=True), "outer-pause",
         "fable-session-1", context.sol_thread_id, "2026-08-10T12:00:00Z", "baseline-1",
         brief.task_id, brief.revision),
    )
    outer = QuestionRecord(
        "outer-question", "session-1", brief.task_id, brief.revision, 1,
        ConversationActor.SOL, ConversationTarget.FABLE, ConversationTarget.FABLE,
        "Which exact rule applies?", "outer-exchange", None, None,
    )
    store._insert_question(
        outer, continuation_state=TaskState.SOL_RUNNING, pending_action=pending,
        continuation_pause_id="outer-pause",
    )
    payload = ExchangeGrantPayload(
        request_id="checkpoint-grant", continuation_generation=1,
        attempted_question=DirectedAgentQuestion(
            addressed_to="sol", text="Which exact test proves it?", reason="Need one fact.",
        ),
        continuation=context, parent_mode="question", outer_question_id=outer.question_id,
    )
    record = store._insert_prepared_action(
        project_id=project_id_for_root(tmp_path / "repo"), session_id="session-1",
        task_id=brief.task_id, revision=brief.revision, action="exchange_grant",
        payload=payload, source_state=TaskState.AWAITING_USER_INPUT,
        active_state=TaskState.SOL_RUNNING, continuation_state=TaskState.SOL_RUNNING,
        pending_context=context, previous_preparation_id=None, generation=9,
    )
    store._connection.execute(
        "UPDATE prepared_actions SET status = 'CLAIMED' WHERE preparation_id = ?",
        (record.preparation_id,),
    )
    if tamper == "mode":
        store._connection.execute("UPDATE prepared_actions SET payload_json = json_set(payload_json, '$.parent_mode', 'clarification') WHERE preparation_id = ?", (record.preparation_id,))
    elif tamper == "outer":
        store._connection.execute("UPDATE prepared_actions SET payload_json = json_set(payload_json, '$.outer_question_id', 'forged-outer') WHERE preparation_id = ?", (record.preparation_id,))
    elif tamper == "task":
        with pytest.raises(sqlite3.IntegrityError):
            store._connection.execute("UPDATE prepared_actions SET task_id = 'forged-task' WHERE preparation_id = ?", (record.preparation_id,))
        return
    elif tamper == "revision":
        with pytest.raises(sqlite3.IntegrityError):
            store._connection.execute("UPDATE prepared_actions SET revision = 2 WHERE preparation_id = ?", (record.preparation_id,))
        return
    elif tamper == "generation":
        store._connection.execute("UPDATE tasks SET continuation_generation = 2 WHERE task_id = ?", (brief.task_id,))
    elif tamper == "answer":
        store._connection.execute("UPDATE questions SET answer_text = 'forged', answered_by = 'fable' WHERE question_id = ?", (outer.question_id,))
    else:
        store._connection.execute("UPDATE questions SET asked_by = 'fable' WHERE question_id = ?", (outer.question_id,))

    with pytest.raises(RuntimeError):
        store.resume_claimed_exchange_grant_checkpoint(record.preparation_id, generation=9)


def test_recovered_question_grant_checkpoint_rebinds_only_one_fresh_lease_generation(
    tmp_path, valid_brief,
) -> None:
    """Removing the CAS would let a stale Hub lease run a recovered checkpoint."""
    store = _store(tmp_path)
    (tmp_path / "repo").mkdir()
    store.create_session("session-1", "/repo")
    brief = replace(valid_brief, task_id="checkpoint-rebind", revision=1)
    _save_active_directed_task(store, "session-1", brief)
    context = SolResumeContext("sol-thread-1", "sol-run-1", "Resume exactly.")
    pending = {"sol_run_id": context.sol_run_id, "prompt": context.prompt}
    store._connection.execute(
        """UPDATE tasks SET state = ?, continuation_state = ?, pending_json = ?,
        continuation_pause_id = ?, fable_session_id = ?, sol_thread_id = ?,
        approved_at = ?, baseline_id = ? WHERE task_id = ? AND revision = ?""",
        (TaskState.AWAITING_USER_INPUT.value, TaskState.SOL_RUNNING.value,
         json.dumps(pending, separators=(",", ":"), sort_keys=True), "rebind-pause",
         "fable-session-1", context.sol_thread_id, "2026-08-10T12:00:00Z", "baseline-1",
         brief.task_id, brief.revision),
    )
    outer = QuestionRecord(
        "rebind-outer", "session-1", brief.task_id, brief.revision, 1,
        ConversationActor.SOL, ConversationTarget.FABLE, ConversationTarget.FABLE,
        "Which exact rule applies?", "rebind-exchange", None, None,
    )
    store._insert_question(
        outer, continuation_state=TaskState.SOL_RUNNING, pending_action=pending,
        continuation_pause_id="rebind-pause",
    )
    record = store._insert_prepared_action(
        project_id=project_id_for_root(tmp_path / "repo"), session_id="session-1",
        task_id=brief.task_id, revision=brief.revision, action="exchange_grant",
        payload=ExchangeGrantPayload(
            request_id="rebind-grant", continuation_generation=1,
            attempted_question=DirectedAgentQuestion(
                addressed_to="sol", text="Which exact test proves it?", reason="Need one fact.",
            ),
            continuation=context, parent_mode="question", outer_question_id=outer.question_id,
        ),
        source_state=TaskState.AWAITING_USER_INPUT, active_state=TaskState.SOL_RUNNING,
        continuation_state=TaskState.SOL_RUNNING, pending_context=context,
        previous_preparation_id=None, generation=9,
    )
    store._connection.execute(
        "UPDATE prepared_actions SET status = 'RECOVERED' WHERE preparation_id = ?",
        (record.preparation_id,),
    )

    rebound = store.rebind_recovered_exchange_grant_checkpoint(
        record.preparation_id, old_generation=9, generation=14,
        project_id=record.project_id, session_id=record.session_id,
        task_id=record.task_id, revision=record.revision,
    )

    assert (rebound.generation, rebound.status) == (14, "CLAIMED")
    assert store.get_task(brief.task_id, brief.revision).continuation_generation == 1
    assert store.question(outer.question_id) == outer
    with pytest.raises(RuntimeError):
        store.rebind_recovered_exchange_grant_checkpoint(
            record.preparation_id, old_generation=9, generation=15,
            project_id=record.project_id, session_id=record.session_id,
            task_id=record.task_id, revision=record.revision,
        )
