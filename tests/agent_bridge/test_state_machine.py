import pytest

from agent_bridge.state_machine import ALLOWED_TRANSITIONS, TaskState, require_transition


EXPECTED_TRANSITIONS = {
    TaskState.IDLE: frozenset({TaskState.FABLE_PLANNING}),
    TaskState.FABLE_PLANNING: frozenset({
        TaskState.AWAITING_USER_APPROVAL,
        TaskState.FAILED,
        TaskState.INTERRUPTED,
    }),
    TaskState.AWAITING_USER_APPROVAL: frozenset({
        TaskState.SOL_RUNNING,
        TaskState.FAILED,
    }),
    TaskState.SOL_RUNNING: frozenset({
        TaskState.FABLE_CLARIFYING,
        TaskState.FABLE_REVIEWING,
        TaskState.AWAITING_USER_INPUT,
        TaskState.FAILED,
        TaskState.INTERRUPTED,
    }),
    TaskState.FABLE_CLARIFYING: frozenset({
        TaskState.SOL_RUNNING,
        TaskState.SOL_CORRECTING,
        TaskState.AWAITING_USER_INPUT,
        TaskState.AWAITING_SCOPE_APPROVAL,
        TaskState.FAILED,
        TaskState.INTERRUPTED,
    }),
    TaskState.AWAITING_USER_INPUT: frozenset({
        TaskState.SOL_RUNNING,
        TaskState.SOL_CORRECTING,
        TaskState.FABLE_REVIEWING,
        TaskState.FAILED,
    }),
    TaskState.AWAITING_SCOPE_APPROVAL: frozenset({
        TaskState.SOL_RUNNING,
        TaskState.SOL_CORRECTING,
        TaskState.FAILED,
    }),
    TaskState.FABLE_REVIEWING: frozenset({
        TaskState.COMPLETED,
        TaskState.SOL_CORRECTING,
        TaskState.AWAITING_USER_INPUT,
        TaskState.FAILED,
        TaskState.INTERRUPTED,
    }),
    TaskState.SOL_CORRECTING: frozenset({
        TaskState.FABLE_CLARIFYING,
        TaskState.FABLE_REVIEWING,
        TaskState.AWAITING_USER_INPUT,
        TaskState.FAILED,
        TaskState.INTERRUPTED,
    }),
    TaskState.INTERRUPTED: frozenset({
        TaskState.FABLE_PLANNING,
        TaskState.SOL_RUNNING,
        TaskState.FABLE_CLARIFYING,
        TaskState.FABLE_REVIEWING,
        TaskState.SOL_CORRECTING,
        TaskState.FAILED,
    }),
    TaskState.COMPLETED: frozenset(),
    TaskState.FAILED: frozenset(),
}


@pytest.mark.parametrize(
    ("current", "target", "legal"),
    [
        (current, target, target in targets)
        for current, targets in EXPECTED_TRANSITIONS.items()
        for target in TaskState
    ],
)
def test_require_transition_matches_every_approved_state_edge(
    current: TaskState,
    target: TaskState,
    legal: bool,
) -> None:
    if legal:
        require_transition(current, target)
    else:
        with pytest.raises(ValueError, match="illegal task transition"):
            require_transition(current, target)


def test_allowed_transition_table_has_every_state_and_no_extra_edge() -> None:
    assert ALLOWED_TRANSITIONS == EXPECTED_TRANSITIONS


def test_planning_cannot_bypass_approval_into_sol_running() -> None:
    require_transition(TaskState.AWAITING_USER_APPROVAL, TaskState.SOL_RUNNING)
    with pytest.raises(ValueError, match="illegal task transition"):
        require_transition(TaskState.FABLE_PLANNING, TaskState.SOL_RUNNING)


def test_terminal_and_non_resume_transitions_are_rejected() -> None:
    with pytest.raises(ValueError, match="illegal task transition"):
        require_transition(TaskState.FAILED, TaskState.SOL_RUNNING)

    require_transition(TaskState.INTERRUPTED, TaskState.SOL_RUNNING)

    with pytest.raises(ValueError, match="illegal task transition"):
        require_transition(TaskState.INTERRUPTED, TaskState.COMPLETED)
