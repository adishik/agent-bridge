"""Explicit, finite task lifecycle transitions for the local bridge."""

from __future__ import annotations

from enum import StrEnum


class TaskState(StrEnum):
    IDLE = "idle"
    FABLE_PLANNING = "fable_planning"
    AWAITING_USER_APPROVAL = "awaiting_user_approval"
    SOL_RUNNING = "sol_running"
    FABLE_CLARIFYING = "fable_clarifying"
    AWAITING_USER_INPUT = "awaiting_user_input"
    AWAITING_SCOPE_APPROVAL = "awaiting_scope_approval"
    FABLE_REVIEWING = "fable_reviewing"
    SOL_CORRECTING = "sol_correcting"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


ALLOWED_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
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


def require_transition(current: TaskState, target: TaskState) -> None:
    """Raise when a requested lifecycle edge is outside the finite state map."""
    if target not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"illegal task transition: {current.value} -> {target.value}")
