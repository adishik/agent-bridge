"""Normalized adapter result and protocol boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from agent_bridge.contracts import TaskBrief


@dataclass(frozen=True)
class AgentRunResult:
    run_id: str
    cli_session_id: str | None
    payload: Mapping[str, object] | None
    events: tuple[Mapping[str, object], ...]
    stderr: tuple[str, ...]
    exit_code: int
    interrupted: bool


class FableAdapter(Protocol):
    async def plan(
        self, *, run_id: str, task_id: str, prompt: str, context: str,
    ) -> AgentRunResult:
        raise NotImplementedError

    async def resume_plan(
        self,
        *,
        run_id: str,
        session_id: str,
        task_id: str,
        prompt: str,
        context: str,
    ) -> AgentRunResult:
        raise NotImplementedError

    async def clarify(
        self, *, run_id: str, session_id: str, prompt: str,
    ) -> AgentRunResult:
        raise NotImplementedError

    async def review(
        self, *, run_id: str, session_id: str, prompt: str,
    ) -> AgentRunResult:
        raise NotImplementedError


class SolAdapter(Protocol):
    async def start(
        self, *, run_id: str, brief: TaskBrief, context: str,
    ) -> AgentRunResult:
        raise NotImplementedError

    async def resume(
        self, *, run_id: str, thread_id: str, prompt: str,
    ) -> AgentRunResult:
        raise NotImplementedError
