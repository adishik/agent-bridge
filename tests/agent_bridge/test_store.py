from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import threading
import tracemalloc

import pytest

import agent_bridge.store as store_module
from agent_bridge.projects import project_id_for_root
from agent_bridge.state_machine import TaskState
from agent_bridge.store import (
    ApprovalPayload,
    AnswerContext,
    AnswerPayload,
    BaselineSetting,
    ClarificationContext,
    MAX_TASK_OVERVIEWS,
    NewRequestPayload,
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
        "tasks": rows("SELECT * FROM tasks ORDER BY task_id, revision"),
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
