from __future__ import annotations

from dataclasses import replace
import os
import sqlite3
import threading

import pytest

import agent_bridge.store as store_module
from agent_bridge.state_machine import TaskState
from agent_bridge.store import MAX_TASK_OVERVIEWS, SQLiteStore


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

    assert tuple((task.task_id, task.revision) for task in recovered) == tuple(
        (task_id, revision) for task_id, revision, _ in identities
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
    assert store.recover_active_tasks() == ()


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

    assert tuple((task.task_id, task.revision) for task in recovered) == (
        (latest.task_id, latest.revision),
    )
    assert store.get_task(historical.task_id, historical.revision).state is (
        TaskState.FABLE_PLANNING
    )
    interrupted = store.get_task(latest.task_id, latest.revision)
    assert interrupted.state is TaskState.INTERRUPTED
    assert interrupted.continuation_state is TaskState.SOL_RUNNING
    assert store.agent_run("finished-plan").status == "completed"
    assert store.agent_run("active-sol").status == "interrupted"
    assert store.recover_active_tasks() == ()


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
