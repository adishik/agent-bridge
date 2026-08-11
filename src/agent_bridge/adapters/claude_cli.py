"""Subscription-only, read-only Claude CLI adapter for Fable."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path

from agent_bridge.adapters.base import AgentRunResult
from agent_bridge.contracts import (
    FABLE_CLARIFICATION_SCHEMA,
    REVIEW_VERDICT_SCHEMA,
    TASK_BRIEF_SCHEMA,
    FableClarification,
    ReviewVerdict,
    TaskBrief,
    freeze_json,
)
from agent_bridge.process import ProcessResult, ProcessRunner


METERED_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_")
METERED_ENV_KEYS = frozenset({
    "CLAUDE_CODE_OAUTH_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "GOOGLE_APPLICATION_CREDENTIALS",
})


class SubscriptionAuthError(RuntimeError):
    """Claude CLI is not unambiguously using a saved paid subscription."""


class ClaudeRunError(RuntimeError):
    """A completed Claude CLI run did not satisfy the adapter contract."""

    def __init__(self, message: str, *, result: AgentRunResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class _EventParseError(ValueError):
    def __init__(self, message: str, events: tuple[Mapping[str, object], ...]) -> None:
        super().__init__(message)
        self.events = events


@dataclass(frozen=True)
class _ParsedEvents:
    audit_events: tuple[Mapping[str, object], ...]
    structured_output: Mapping[str, object] | None
    result_seen: bool


@dataclass(frozen=True)
class ClaudeAuthStatus:
    logged_in: bool
    auth_method: str
    api_provider: str
    subscription_type: str


def sanitized_claude_env(source: Mapping[str, str]) -> dict[str, str]:
    """Return a copy without API, gateway, or alternate-provider selectors."""
    return {
        key: value
        for key, value in source.items()
        if key not in METERED_ENV_KEYS
        and not key.startswith(METERED_ENV_PREFIXES)
    }


class ClaudeCLI:
    """Run Fable through one explicit Claude executable with read-only authority."""

    def __init__(
        self,
        executable: str | Path,
        runner: ProcessRunner,
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | Path,
    ) -> None:
        executable_path = Path(executable)
        if not executable_path.is_absolute():
            raise ValueError("executable must be an absolute path")
        if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
            raise ValueError("executable must be an executable file")
        self.executable = executable_path
        self._runner = runner
        self._env = sanitized_claude_env(os.environ if env is None else env)
        self._cwd = Path(cwd)
        self._preflight_ids = itertools.count(1)

    async def preflight(self) -> ClaudeAuthStatus:
        """Require saved first-party Claude subscription authentication."""
        run_id = f"claude-subscription-preflight-{next(self._preflight_ids)}"
        result = await self._run_preflight(run_id)
        if result.interrupted:
            raise SubscriptionAuthError("Claude subscription authentication was interrupted")
        return self._parse_auth_status(result)

    async def plan(
        self, *, run_id: str, task_id: str, prompt: str, context: str,
    ) -> AgentRunResult:
        contract_prompt = self._prompt(
            contract_name="TaskBrief",
            prompt=prompt,
            task_id=task_id,
            context=context,
        )
        return await self._run_contract(
            run_id=run_id,
            schema=TASK_BRIEF_SCHEMA,
            contract_name="TaskBrief",
            prompt=contract_prompt,
            session_id=None,
            expected_task_id=task_id,
        )

    async def resume_plan(
        self,
        *,
        run_id: str,
        session_id: str,
        task_id: str,
        prompt: str,
        context: str,
    ) -> AgentRunResult:
        contract_prompt = self._prompt(
            contract_name="TaskBrief",
            prompt=prompt,
            task_id=task_id,
            context=context,
        )
        return await self._run_contract(
            run_id=run_id,
            schema=TASK_BRIEF_SCHEMA,
            contract_name="TaskBrief",
            prompt=contract_prompt,
            session_id=session_id,
            expected_task_id=task_id,
        )

    async def clarify(
        self, *, run_id: str, session_id: str, prompt: str,
    ) -> AgentRunResult:
        return await self._run_contract(
            run_id=run_id,
            schema=FABLE_CLARIFICATION_SCHEMA,
            contract_name="FableClarification",
            prompt=self._prompt(contract_name="FableClarification", prompt=prompt),
            session_id=session_id,
            expected_task_id=None,
        )

    async def review(
        self, *, run_id: str, session_id: str, prompt: str,
    ) -> AgentRunResult:
        return await self._run_contract(
            run_id=run_id,
            schema=REVIEW_VERDICT_SCHEMA,
            contract_name="ReviewVerdict",
            prompt=self._prompt(contract_name="ReviewVerdict", prompt=prompt),
            session_id=session_id,
            expected_task_id=None,
        )

    async def _run_preflight(self, run_id: str) -> ProcessResult:
        return await self._runner.run(
            run_id=run_id,
            argv=(str(self.executable), "auth", "status", "--json"),
            cwd=self._cwd,
            env=self._env,
            stdin=None,
            on_line=lambda stream, line: None,
        )

    def _parse_auth_status(self, result: ProcessResult) -> ClaudeAuthStatus:
        if result.exit_code != 0 or len(result.stdout) != 1:
            raise SubscriptionAuthError("Claude subscription authentication could not be verified")
        try:
            status = json.loads(result.stdout[0])
        except (json.JSONDecodeError, TypeError):
            raise SubscriptionAuthError(
                "Claude subscription authentication could not be verified"
            ) from None
        if not isinstance(status, Mapping):
            raise SubscriptionAuthError("Claude subscription authentication could not be verified")
        subscription_type = status.get("subscriptionType")
        if (
            status.get("loggedIn") is not True
            or status.get("authMethod") != "claude.ai"
            or status.get("apiProvider") != "firstParty"
            or not isinstance(subscription_type, str)
            or not subscription_type.strip()
        ):
            raise SubscriptionAuthError("Claude subscription authentication is required")
        return ClaudeAuthStatus(
            logged_in=True,
            auth_method="claude.ai",
            api_provider="firstParty",
            subscription_type=subscription_type,
        )

    async def _run_contract(
        self,
        *,
        run_id: str,
        schema: Mapping[str, object],
        contract_name: str,
        prompt: str,
        session_id: str | None,
        expected_task_id: str | None,
    ) -> AgentRunResult:
        auth_result = await self._run_preflight(run_id)
        if auth_result.interrupted:
            return AgentRunResult(
                run_id=run_id,
                cli_session_id=None,
                payload=None,
                events=(),
                stderr=(),
                exit_code=auth_result.exit_code,
                interrupted=True,
            )
        self._parse_auth_status(auth_result)

        argv = (
            str(self.executable),
            "--safe-mode",
            "-p",
            "--model", "fable",
            "--permission-mode", "plan",
            "--tools", "Read,Glob,Grep",
            "--no-chrome",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--json-schema", json.dumps(schema, separators=(",", ":"), sort_keys=True),
        )
        if session_id is None:
            argv = (*argv, prompt)
        else:
            argv = (*argv, "--resume", session_id, prompt)

        process = await self._runner.run(
            run_id=run_id,
            argv=argv,
            cwd=self._cwd,
            env=self._env,
            stdin=None,
            on_line=lambda stream, line: None,
        )
        try:
            parsed = self._parse_events(process.stdout, interrupted=process.interrupted)
        except _EventParseError as error:
            raise ClaudeRunError(
                str(error),
                result=self._failed_result(run_id, process, error.events),
            ) from None
        events = parsed.audit_events
        cli_session_id = self._session_id(events)
        if (
            session_id is not None
            and cli_session_id is not None
            and cli_session_id != session_id
        ):
            raise ClaudeRunError(
                "Claude resumed a different session than requested",
                result=self._failed_result(
                    run_id,
                    process,
                    events,
                    cli_session_id,
                    interrupted=process.interrupted,
                ),
            )
        if process.interrupted:
            return AgentRunResult(
                run_id=run_id,
                cli_session_id=cli_session_id,
                payload=None,
                events=events,
                stderr=self._stderr_summary(process.stderr),
                exit_code=process.exit_code,
                interrupted=True,
            )
        if process.exit_code != 0:
            raise ClaudeRunError(
                "Claude exited with a non-zero exit status",
                result=self._failed_result(run_id, process, events, cli_session_id),
            )
        if cli_session_id is None:
            raise ClaudeRunError(
                "Claude output is missing the required system/init event",
                result=self._failed_result(run_id, process, events),
            )
        if not parsed.result_seen or parsed.structured_output is None:
            raise ClaudeRunError(
                "Claude output is missing the required result event",
                result=self._failed_result(run_id, process, events, cli_session_id),
            )
        try:
            payload = self._validate_payload(contract_name, parsed.structured_output)
        except ClaudeRunError as error:
            raise ClaudeRunError(
                str(error),
                result=self._failed_result(run_id, process, events, cli_session_id),
            ) from None
        if expected_task_id is not None and payload.get("task_id") != expected_task_id:
            raise ClaudeRunError(
                "Claude TaskBrief returned a different coordinator task ID",
                result=self._failed_result(run_id, process, events, cli_session_id),
            )
        return AgentRunResult(
            run_id=run_id,
            cli_session_id=cli_session_id,
            payload=payload,
            events=events,
            stderr=self._stderr_summary(process.stderr),
            exit_code=process.exit_code,
            interrupted=False,
        )

    @staticmethod
    def _parse_events(
        lines: tuple[str, ...], *, interrupted: bool,
    ) -> _ParsedEvents:
        events: list[Mapping[str, object]] = []
        structured_output: Mapping[str, object] | None = None
        result_seen = False
        for line in lines:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                if interrupted:
                    continue
                raise _EventParseError(
                    "Claude emitted an invalid JSON event", tuple(events),
                ) from None
            if not isinstance(event, Mapping):
                if interrupted:
                    continue
                raise _EventParseError(
                    "Claude emitted a JSON event that is not an object", tuple(events),
                )
            event_type = event.get("type")
            audit_event: dict[str, object] | None = None
            if event_type == "system" and event.get("subtype") == "init":
                audit_event = {"type": "system", "subtype": "init"}
                session_id = event.get("session_id")
                if isinstance(session_id, str) and session_id:
                    audit_event["session_id"] = session_id
            elif isinstance(event_type, str) and event_type in {
                "assistant", "user", "stream_event",
            }:
                audit_event = {"type": event_type}
            elif event_type == "result":
                result_seen = True
                candidate = event.get("structured_output")
                structured_output = candidate if isinstance(candidate, Mapping) else None
                audit_event = {
                    "type": "result",
                    "has_structured_output": structured_output is not None,
                }
            if audit_event is not None:
                frozen = freeze_json(audit_event)
                if not isinstance(frozen, Mapping):
                    raise RuntimeError("audit event normalization did not produce an object")
                events.append(frozen)
        return _ParsedEvents(
            audit_events=tuple(events),
            structured_output=structured_output,
            result_seen=result_seen,
        )

    @staticmethod
    def _failed_result(
        run_id: str,
        process: ProcessResult,
        events: tuple[Mapping[str, object], ...],
        cli_session_id: str | None = None,
        *,
        interrupted: bool = False,
    ) -> AgentRunResult:
        if cli_session_id is None:
            cli_session_id = ClaudeCLI._session_id(events)
        return AgentRunResult(
            run_id=run_id,
            cli_session_id=cli_session_id,
            payload=None,
            events=events,
            stderr=ClaudeCLI._stderr_summary(process.stderr),
            exit_code=process.exit_code,
            interrupted=interrupted,
        )

    @staticmethod
    def _stderr_summary(stderr: tuple[str, ...]) -> tuple[str, ...]:
        if not stderr:
            return ()
        if len(stderr) > 9_999:
            return ("stderr_lines=10000+",)
        return (f"stderr_lines={len(stderr)}",)

    @staticmethod
    def _session_id(events: tuple[Mapping[str, object], ...]) -> str | None:
        for event in events:
            if event.get("type") == "system" and event.get("subtype") == "init":
                session_id = event.get("session_id")
                if isinstance(session_id, str) and session_id:
                    return session_id
        return None

    @staticmethod
    def _validate_payload(
        contract_name: str, payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        try:
            if contract_name == "TaskBrief":
                return TaskBrief.from_dict(payload).to_dict()
            if contract_name == "FableClarification":
                return FableClarification.from_dict(payload).to_dict()
            if contract_name == "ReviewVerdict":
                return ReviewVerdict.from_dict(payload).to_dict()
        except ValueError:
            raise ClaudeRunError(
                f"Claude structured output failed {contract_name} schema validation"
            ) from None
        raise ClaudeRunError("unsupported Claude contract")

    @staticmethod
    def _prompt(
        *,
        contract_name: str,
        prompt: str,
        task_id: str | None = None,
        context: str | None = None,
    ) -> str:
        sections = [f"Requested final contract: {contract_name}."]
        if task_id is not None:
            sections.append(f"Coordinator-generated task ID: {task_id}.")
        if context is not None:
            sections.append(f"Applicable AGENTS.md and repository context:\n{context}")
        sections.extend((
            f"Coordinator request:\n{prompt}",
            "Only JSON matching the supplied schema may be final output.",
        ))
        return "\n\n".join(sections)
