"""Workspace-scoped Codex CLI adapter for the Sol implementation agent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
from uuid import UUID

from agent_bridge.adapters.base import AgentRunResult
from agent_bridge.contracts import (
    SOL_OUTCOME_SCHEMA,
    SolOutcome,
    TaskBrief,
    freeze_json,
)
from agent_bridge.process import ProcessResult, ProcessRunner


_AUDIT_ITEM_TYPES = frozenset({
    "agent_message",
    "command_execution",
    "file_change",
    "plan",
    "todo_list",
})
_AUDIT_ITEM_STATUSES = frozenset({
    "completed",
    "declined",
    "failed",
    "in_progress",
    "interrupted",
})
MAX_CODEX_AUDIT_EVENTS = 1_024
_SOL_SCHEMA_FILENAME = "sol-outcome.json"


def _serialized_sol_outcome_schema() -> bytes:
    return json.dumps(
        SOL_OUTCOME_SCHEMA,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validated_sol_schema_file_descriptor(descriptor: int) -> None:
    """Fail closed unless *descriptor* is the exact read-only schema file."""
    if (
        not isinstance(descriptor, int)
        or isinstance(descriptor, bool)
        or descriptor < 0
    ):
        raise ValueError("schema_file_fd must be an open read-only schema file descriptor")
    try:
        if fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
            raise ValueError("schema_file_fd must be an open read-only schema file descriptor")
        entry = os.fstat(descriptor)
        expected = _serialized_sol_outcome_schema()
        if not stat.S_ISREG(entry.st_mode) or entry.st_size != len(expected):
            raise ValueError("schema_file_fd must be an open read-only schema file descriptor")
        duplicate = os.dup(descriptor)
        try:
            contents = os.pread(duplicate, len(expected) + 1, 0)
        finally:
            os.close(duplicate)
        if contents != expected:
            raise ValueError("schema_file_fd must be an open read-only schema file descriptor")
    except OSError as error:
        raise ValueError("schema_file_fd must be an open read-only schema file descriptor") from error


def _validated_sol_schema_directory_descriptor(descriptor: int) -> None:
    if (
        not isinstance(descriptor, int)
        or isinstance(descriptor, bool)
        or descriptor < 0
    ):
        raise ValueError("schema directory descriptor must be an open directory descriptor")
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("schema directory descriptor must be an open directory descriptor")
    except OSError as error:
        raise ValueError("schema directory descriptor must be an open directory descriptor") from error


def materialize_sol_schema_file(directory_fd: int) -> int:
    """Atomically return one caller-owned, authenticated schema-file descriptor."""
    _validated_sol_schema_directory_descriptor(directory_fd)
    expected = _serialized_sol_outcome_schema()
    write_descriptor = -1
    schema_file_descriptor = -1
    temporary_name: str | None = None
    try:
        temporary_name = f".{_SOL_SCHEMA_FILENAME}-{os.urandom(16).hex()}.tmp"
        write_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(write_descriptor, 0o600)
        written = 0
        while written < len(expected):
            count = os.write(write_descriptor, expected[written:])
            if count <= 0:
                raise OSError("schema write did not make progress")
            written += count
        os.fsync(write_descriptor)
        os.close(write_descriptor)
        write_descriptor = -1
        os.replace(
            temporary_name,
            _SOL_SCHEMA_FILENAME,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = None
        os.fsync(directory_fd)
        schema_file_descriptor = os.open(
            _SOL_SCHEMA_FILENAME,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        _validated_sol_schema_file_descriptor(schema_file_descriptor)
        result = schema_file_descriptor
        schema_file_descriptor = -1
        return result
    except OSError as error:
        raise ValueError("Sol schema file could not be materialized") from error
    finally:
        if write_descriptor >= 0:
            os.close(write_descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        if schema_file_descriptor >= 0:
            os.close(schema_file_descriptor)


class CodexRunError(RuntimeError):
    """A completed Codex CLI run did not satisfy the adapter contract."""

    def __init__(self, message: str, *, result: AgentRunResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class _EventParseError(ValueError):
    def __init__(
        self,
        message: str,
        events: tuple[Mapping[str, object], ...],
        thread_id: str | None,
    ) -> None:
        super().__init__(message)
        self.events = events
        self.thread_id = thread_id


@dataclass(frozen=True)
class _ParsedEvents:
    audit_events: tuple[Mapping[str, object], ...]
    thread_id: str | None
    final_message: str | None


class CodexCLI:
    """Run Sol through one injected Codex executable with bounded authority."""

    def __init__(
        self,
        executable: str | Path,
        runner: ProcessRunner,
        *,
        repo_root: str | Path,
        schema_dir: str | Path,
        env: Mapping[str, str] | None = None,
        schema_file_fd: int | None = None,
    ) -> None:
        executable_path = Path(executable)
        if not executable_path.is_absolute():
            raise ValueError("executable must be an absolute path")
        if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
            raise ValueError("executable must be an executable file")
        repo_root_path = Path(repo_root)
        if not repo_root_path.is_absolute():
            raise ValueError("repo_root must be an absolute path")
        schema_dir_path = Path(schema_dir)
        if not schema_dir_path.is_absolute():
            raise ValueError("schema_dir must be an absolute path")
        self.executable = executable_path
        self._runner = runner
        self.repo_root = repo_root_path
        self._env = dict(os.environ if env is None else env)
        if schema_file_fd is None:
            self._schema_pass_fds: tuple[int, ...] = ()
            self.schema_path = schema_dir_path / _SOL_SCHEMA_FILENAME
            self._materialize_schema()
        else:
            _validated_sol_schema_file_descriptor(schema_file_fd)
            self._schema_pass_fds = (schema_file_fd,)
            self.schema_path = Path(f"/proc/self/fd/{schema_file_fd}")

    async def start(
        self, *, run_id: str, brief: TaskBrief, context: str,
    ) -> AgentRunResult:
        if not isinstance(brief, TaskBrief):
            raise ValueError("brief must be a TaskBrief")
        prompt = self._start_prompt(brief, context)
        argv = (
            str(self.executable),
            "exec", "--json",
            "--model", "gpt-5.6-sol",
            "--sandbox", "workspace-write",
            "--approve-for-me",
            "--cd", str(self.repo_root),
            "--output-schema", str(self.schema_path),
            prompt,
        )
        return await self._run_contract(
            run_id=run_id,
            argv=argv,
            expected_thread_id=None,
            pass_fds=self._schema_pass_fds,
        )

    async def resume(
        self, *, run_id: str, thread_id: str, prompt: str,
    ) -> AgentRunResult:
        thread_id = self._canonical_thread_id(thread_id)
        argv = (
            str(self.executable),
            "exec", "resume", "--json",
            "--model", "gpt-5.6-sol",
            "--output-schema", str(self.schema_path),
            thread_id,
            self._resume_prompt(prompt),
        )
        return await self._run_contract(
            run_id=run_id,
            argv=argv,
            expected_thread_id=thread_id,
            pass_fds=self._schema_pass_fds,
        )

    def _materialize_schema(self) -> None:
        schema_dir = self.schema_path.parent
        schema_dir.mkdir(parents=True, exist_ok=True)
        directory_descriptor = -1
        schema_file_descriptor = -1
        try:
            directory_descriptor = os.open(
                schema_dir,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            )
            schema_file_descriptor = materialize_sol_schema_file(directory_descriptor)
        finally:
            if schema_file_descriptor >= 0:
                os.close(schema_file_descriptor)
            if directory_descriptor >= 0:
                os.close(directory_descriptor)

    async def _run_contract(
        self,
        *,
        run_id: str,
        argv: tuple[str, ...],
        expected_thread_id: str | None,
        pass_fds: tuple[int, ...] = (),
    ) -> AgentRunResult:
        process_arguments: dict[str, object] = {
            "run_id": run_id,
            "argv": argv,
            "cwd": self.repo_root,
            "env": self._env,
            "stdin": None,
            "on_line": lambda stream, line: None,
        }
        if pass_fds:
            process_arguments["pass_fds"] = pass_fds
        process = await self._runner.run(**process_arguments)
        try:
            parsed = self._parse_events(process.stdout, interrupted=process.interrupted)
        except _EventParseError as error:
            error_thread_id = error.thread_id
            if (
                expected_thread_id is not None
                and error_thread_id != expected_thread_id
            ):
                error_thread_id = None
            raise CodexRunError(
                str(error),
                result=self._failed_result(
                    run_id,
                    process,
                    error.events,
                    error_thread_id,
                    interrupted=process.interrupted,
                ),
            ) from None

        if (
            expected_thread_id is not None
            and parsed.thread_id is not None
            and parsed.thread_id != expected_thread_id
        ):
            raise CodexRunError(
                "Codex resumed a different thread than requested",
                result=self._failed_result(
                    run_id,
                    process,
                    parsed.audit_events,
                    None,
                    interrupted=process.interrupted,
                ),
            )
        if process.interrupted:
            return AgentRunResult(
                run_id=run_id,
                cli_session_id=parsed.thread_id,
                payload=None,
                events=parsed.audit_events,
                stderr=self._stderr_summary(process.stderr),
                exit_code=process.exit_code,
                interrupted=True,
            )
        if process.exit_code != 0:
            raise CodexRunError(
                "Codex exited with a non-zero exit status",
                result=self._failed_result(
                    run_id,
                    process,
                    parsed.audit_events,
                    parsed.thread_id,
                ),
            )
        if parsed.thread_id is None:
            raise CodexRunError(
                "Codex output is missing the required thread.started event",
                result=self._failed_result(run_id, process, parsed.audit_events),
            )
        if parsed.final_message is None:
            raise CodexRunError(
                "Codex output is missing the required final agent message",
                result=self._failed_result(
                    run_id,
                    process,
                    parsed.audit_events,
                    parsed.thread_id,
                ),
            )
        try:
            candidate = json.loads(parsed.final_message)
            if not isinstance(candidate, Mapping):
                raise ValueError("SolOutcome must be an object")
            payload = SolOutcome.from_dict(candidate).to_dict()
        except (json.JSONDecodeError, TypeError, ValueError):
            raise CodexRunError(
                "Codex final agent message failed SolOutcome schema validation",
                result=self._failed_result(
                    run_id,
                    process,
                    parsed.audit_events,
                    parsed.thread_id,
                ),
            ) from None
        return AgentRunResult(
            run_id=run_id,
            cli_session_id=parsed.thread_id,
            payload=payload,
            events=parsed.audit_events,
            stderr=self._stderr_summary(process.stderr),
            exit_code=process.exit_code,
            interrupted=False,
        )

    @staticmethod
    def _parse_events(
        lines: tuple[str, ...], *, interrupted: bool,
    ) -> _ParsedEvents:
        events: list[Mapping[str, object]] = []
        dropped_audit_events = 0
        thread_id: str | None = None
        final_message: str | None = None
        for line in lines:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                if interrupted:
                    continue
                raise _EventParseError(
                    "Codex emitted an invalid JSON event", tuple(events), thread_id,
                ) from None
            if not isinstance(event, Mapping):
                if interrupted:
                    continue
                raise _EventParseError(
                    "Codex emitted a JSON event that is not an object",
                    tuple(events),
                    thread_id,
                )

            event_type = event.get("type")
            audit_event: dict[str, object] | None = None
            if event_type == "thread.started":
                candidate_thread = event.get("thread_id")
                if isinstance(candidate_thread, str) and candidate_thread:
                    try:
                        candidate_thread = CodexCLI._canonical_thread_id(candidate_thread)
                    except ValueError:
                        raise _EventParseError(
                            "Codex emitted a non-canonical thread ID",
                            tuple(events),
                            thread_id,
                        ) from None
                    if thread_id is not None and candidate_thread != thread_id:
                        raise _EventParseError(
                            "Codex emitted conflicting thread.started events",
                            tuple(events),
                            thread_id,
                        )
                    thread_id = candidate_thread
                    audit_event = {
                        "type": "thread.started",
                        "thread_id": candidate_thread,
                    }
            elif event_type in {"item.started", "item.updated", "item.completed"}:
                item = event.get("item")
                if isinstance(item, Mapping):
                    item_type = item.get("type")
                    if isinstance(item_type, str) and item_type in _AUDIT_ITEM_TYPES:
                        audit_event = {"type": event_type, "item_type": item_type}
                        status = item.get("status")
                        if (
                            event_type == "item.completed"
                            and isinstance(status, str)
                            and status in _AUDIT_ITEM_STATUSES
                        ):
                            audit_event["status"] = status
                        if item_type == "command_execution":
                            command = item.get("command")
                            if isinstance(command, str):
                                audit_event["command_sha256"] = CodexCLI._sha256(command)
                            if event_type == "item.completed":
                                exit_code = item.get("exit_code")
                                if isinstance(exit_code, int) and not isinstance(exit_code, bool):
                                    audit_event["exit_code"] = exit_code
                                output = item.get("aggregated_output")
                                if isinstance(output, str):
                                    audit_event.update({
                                        "output_sha256": CodexCLI._sha256(output),
                                        "output_bytes": len(output.encode("utf-8")),
                                        "output_lines": len(output.splitlines()),
                                    })
                        if event_type == "item.completed" and item_type == "agent_message":
                            text = item.get("text")
                            if isinstance(text, str):
                                final_message = text

            if audit_event is not None:
                frozen = freeze_json(audit_event)
                if not isinstance(frozen, Mapping):
                    raise RuntimeError("audit event normalization did not produce an object")
                if len(events) < MAX_CODEX_AUDIT_EVENTS - 1:
                    events.append(frozen)
                else:
                    dropped_audit_events += 1
        if dropped_audit_events:
            summary = freeze_json({
                "type": "audit_events_truncated",
                "dropped_count": dropped_audit_events,
            })
            if not isinstance(summary, Mapping):
                raise RuntimeError("audit event normalization did not produce an object")
            events.append(summary)
        return _ParsedEvents(
            audit_events=tuple(events),
            thread_id=thread_id,
            final_message=final_message,
        )

    @staticmethod
    def _failed_result(
        run_id: str,
        process: ProcessResult,
        events: tuple[Mapping[str, object], ...],
        thread_id: str | None = None,
        *,
        interrupted: bool = False,
    ) -> AgentRunResult:
        return AgentRunResult(
            run_id=run_id,
            cli_session_id=thread_id,
            payload=None,
            events=events,
            stderr=CodexCLI._stderr_summary(process.stderr),
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
    def _canonical_thread_id(thread_id: object) -> str:
        if not isinstance(thread_id, str):
            raise ValueError("thread_id must be a canonical UUID")
        try:
            parsed = UUID(thread_id)
        except (ValueError, AttributeError):
            raise ValueError("thread_id must be a canonical UUID") from None
        if str(parsed) != thread_id:
            raise ValueError("thread_id must be a canonical UUID")
        return thread_id

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _start_prompt(brief: TaskBrief, context: str) -> str:
        brief_json = json.dumps(
            brief.to_dict(),
            separators=(",", ":"),
            sort_keys=True,
        )
        return "\n\n".join((
            "Execute only this explicitly approved TaskBrief revision.",
            f"Approved TaskBrief JSON:\n{brief_json}",
            f"Applicable AGENTS rules and repository baseline context:\n{context}",
            f"Approved allowed paths: {', '.join(brief.allowed_paths)}.",
            (
                "Do not commit, stage files, push, open or merge a pull request, "
                "reset, revert, delete worktrees, or perform unrelated cleanup."
            ),
            (
                "Do not use any paid service, external provider, network call, "
                "credential, or secret."
            ),
            (
                "Required final contract: SolOutcome. The final response must be "
                "only JSON matching the supplied output schema."
            ),
        ))

    @staticmethod
    def _resume_prompt(prompt: str) -> str:
        return "\n\n".join((
            f"Approved continuation context:\n{prompt}",
            (
                "Continue only within the latest exact user-approved TaskBrief "
                "revision supplied in the conversation and workspace-write "
                "authority. Do not commit, stage, push, use a paid service, "
                "access credentials, or expand beyond that revision."
            ),
            (
                "Required final contract: SolOutcome. The final response must be "
                "only JSON matching the supplied output schema."
            ),
        ))
