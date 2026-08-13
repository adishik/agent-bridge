#!/usr/bin/python3
"""Controlled Codex CLI double. Never imports or invokes the real CLI."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time


THREAD_ID = "0199a213-81c0-7800-8aa1-bbab2a035a53"
_MAX_JSON_BYTES = 1024 * 1024
_MAX_HISTORY_ITEMS = 4096
_CONFIGURATION_EXIT = 88
_MISSING = object()


class _FakeConfigurationError(RuntimeError):
    pass


def _capture_path(name: str) -> Path:
    directory = os.environ.get("FAKE_AGENT_CAPTURE_DIR")
    return Path(name) if directory is None else Path(directory) / name


@contextmanager
def _exclusive_lock():
    path = _capture_path("fake-agent-state.lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise _FakeConfigurationError("lock unavailable") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise _FakeConfigurationError("lock is not regular")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_json(
    path: Path,
    *,
    default: object = _MISSING,
    expected: type,
) -> object:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        schema_target = os.readlink(path)
    except OSError:
        schema_target = ""
    if schema_target.startswith("/memfd:agent-bridge-sol-schema"):
        # The adapter deliberately supplies its anonymous schema capability
        # through this procfs link; ordinary JSON inputs remain no-follow.
        pass
    else:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        if default is _MISSING:
            raise _FakeConfigurationError("JSON input is missing") from None
        return default
    except OSError as error:
        raise _FakeConfigurationError("JSON input unavailable") from error
    try:
        entry = os.fstat(descriptor)
        if not stat.S_ISREG(entry.st_mode) or entry.st_size > _MAX_JSON_BYTES:
            raise _FakeConfigurationError("JSON input is not bounded regular data")
        chunks: list[bytes] = []
        size = 0
        while block := os.read(descriptor, min(65536, _MAX_JSON_BYTES - size + 1)):
            size += len(block)
            if size > _MAX_JSON_BYTES:
                raise _FakeConfigurationError("JSON input exceeds bound")
            chunks.append(block)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _FakeConfigurationError("JSON input is invalid") from error
    if not isinstance(value, expected):
        raise _FakeConfigurationError("JSON input has wrong type")
    return value


def _atomic_write_json(path: Path, value: object) -> None:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(encoded) > _MAX_JSON_BYTES:
        raise _FakeConfigurationError("JSON output exceeds bound")
    descriptor = -1
    temporary = ""
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}-",
            suffix=".tmp",
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        raise _FakeConfigurationError("atomic JSON write failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _append_invocation_locked(argv: list[str]) -> None:
    log_path = os.environ.get("AGENT_BRIDGE_INVOCATION_LOG")
    if log_path is None:
        return
    encoded = (json.dumps({
        "kind": "codex",
        "executable": str(Path(sys.argv[0]).resolve()),
        "argv": argv,
    }, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > _MAX_JSON_BYTES:
        raise _FakeConfigurationError("invocation record exceeds bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(log_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise _FakeConfigurationError("invocation log is not regular")
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_capture(name: str, value: object) -> None:
    with _exclusive_lock():
        _atomic_write_json(_capture_path(name), value)


def _record_invocation(argv: list[str]) -> None:
    history_path = _capture_path("captured-codex-argv-history.json")
    with _exclusive_lock():
        _atomic_write_json(_capture_path("captured-codex-argv.json"), argv)
        history = _read_json(history_path, default=[], expected=list)
        if not all(
            isinstance(item, list)
            and all(isinstance(argument, str) for argument in item)
            for item in history
        ):
            raise _FakeConfigurationError("invocation history is invalid")
        if len(history) >= _MAX_HISTORY_ITEMS:
            raise _FakeConfigurationError("invocation history is full")
        history.append(argv)
        _atomic_write_json(history_path, history)
        _append_invocation_locked(argv)


def _completed_outcome() -> dict[str, object]:
    return {
        "status": "completed",
        "summary": "The approved bridge change is complete.",
        "changed_files": ["src/agent_bridge/adapters/codex_cli.py"],
        "commands_run": [{
            "command": "pytest tests/agent_bridge/test_codex_cli.py",
            "exit_code": 0,
            "result": "passed",
        }],
        "known_failures": [],
        "remaining_risks": [],
        "architecture_docs": "No architecture update is required.",
        "question": None,
    }


def _question_outcome() -> dict[str, object]:
    return {
        "status": "question",
        "summary": "One bounded ambiguity remains.",
        "changed_files": [],
        "commands_run": [],
        "known_failures": [],
        "remaining_risks": ["The allowed location is ambiguous."],
        "architecture_docs": "No architecture update is required.",
        "question": {
            "ambiguity": "Which approved directory should contain the adapter?",
            "why_it_matters": "The answer determines the changed path.",
            "options": ["src/agent_bridge", "tests/agent_bridge"],
            "recommendation": "Use src/agent_bridge.",
            "can_continue_safely": False,
            "directed_question": None,
        },
    }


def _emit(value: object) -> None:
    print(json.dumps(value, separators=(",", ":")), flush=True)


def _scenario_outcome(
    fallback: dict[str, object],
) -> tuple[dict[str, object], str | None]:
    scenario_value = os.environ.get("FAKE_AGENT_SCENARIO")
    if scenario_value is None:
        return fallback, None
    with _exclusive_lock():
        scenario = _read_json(Path(scenario_value), expected=dict)
        outcomes = scenario.get("outcomes")
        if not isinstance(outcomes, list):
            raise _FakeConfigurationError("outcome sequence is missing")
        state_path = _capture_path("scenario-state-codex.json")
        state = _read_json(state_path, default={}, expected=dict)
        if not all(
            isinstance(key, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for key, value in state.items()
        ):
            raise _FakeConfigurationError("scenario state is invalid")
        index = state.get("outcome", 0)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise _FakeConfigurationError("outcome counter is invalid")
        if index >= len(outcomes):
            raise _FakeConfigurationError("outcome sequence is exhausted")
        candidate = outcomes[index]
        if not isinstance(candidate, dict):
            raise _FakeConfigurationError("outcome contract is not an object")
        payload = dict(candidate)
        modes = scenario.get("codex_modes", [])
        if not isinstance(modes, list):
            raise _FakeConfigurationError("Codex mode sequence is invalid")
        mode = modes[index] if index < len(modes) else None
        if mode is not None and not isinstance(mode, str):
            raise _FakeConfigurationError("Codex mode is invalid")
        state["outcome"] = index + 1
        _atomic_write_json(state_path, state)
    return payload, mode


def _apply_mutations(payload: dict[str, object]) -> None:
    mutations = payload.pop("_mutations", [])
    repo_root_value = os.environ.get("FAKE_BRIDGE_REPO_ROOT")
    if not mutations:
        return
    if repo_root_value is None or not isinstance(mutations, list):
        raise _FakeConfigurationError("fake mutations are invalid")
    repo_root = Path(repo_root_value).resolve(strict=True)
    for mutation in mutations:
        if not isinstance(mutation, dict):
            raise _FakeConfigurationError("fake mutation is not an object")
        path_value = mutation.get("path")
        content = mutation.get("content")
        if not isinstance(path_value, str) or not isinstance(content, str):
            raise _FakeConfigurationError("fake mutation fields are invalid")
        relative = Path(path_value)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise _FakeConfigurationError("fake mutation is not repository-relative")
        destination = repo_root.joinpath(relative)
        if not destination.resolve().is_relative_to(repo_root):
            raise _FakeConfigurationError("fake mutation escaped the repository")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


def _main() -> int:
    argv = sys.argv[1:]
    _record_invocation(argv)
    if os.environ.get("AGENT_BRIDGE_TEST_FAKE") != "1":
        print("refusing fake Codex invocation without sentinel", file=sys.stderr)
        return 91

    schema_path = Path(argv[argv.index("--output-schema") + 1])
    _read_json(schema_path, expected=dict)
    is_resume = len(argv) > 1 and argv[1] == "resume"
    fallback = _completed_outcome() if is_resume else _question_outcome()
    payload, scenario_mode = _scenario_outcome(fallback)
    directed_question_target = os.environ.get("FAKE_CODEX_DIRECTED_QUESTION_TARGET")
    if directed_question_target is not None:
        if directed_question_target not in {"user", "fable", "sol"}:
            raise _FakeConfigurationError("directed question target is invalid")
        payload = _question_outcome()
        question = payload["question"]
        if not isinstance(question, dict):
            raise _FakeConfigurationError("directed question fixture is invalid")
        question["directed_question"] = {
            "addressed_to": directed_question_target,
            "text": "Which focused test is approved?",
            "reason": "Sol cannot widen the approved execution scope.",
        }
    mode = scenario_mode or os.environ.get("FAKE_CODEX_MODE", "success")
    if mode == "slow_before_thread":
        time.sleep(60)
        return 0

    thread_id = os.environ.get("FAKE_CODEX_THREAD_ID", THREAD_ID)
    _emit({
        "type": "thread.started",
        "thread_id": thread_id,
        "account": "SECRET_ACCOUNT_SENTINEL",
    })
    if mode == "missing_thread":
        pass
    scenario_enabled = os.environ.get("FAKE_AGENT_SCENARIO") is not None
    command = "SECRET_COMMAND_SENTINEL --token SECRET_TOKEN_SENTINEL"
    command_output = "SECRET_COMMAND_OUTPUT_SENTINEL"
    if scenario_enabled and payload.get("status") == "completed":
        command_output = "bounded structural output"
        reports = payload.get("commands_run", [])
        if isinstance(reports, list) and reports and isinstance(reports[0], dict):
            candidate_command = reports[0].get("command")
            if isinstance(candidate_command, str):
                command = candidate_command
    if not scenario_enabled or payload.get("status") == "completed":
        _emit({
            "type": "item.started",
            "item": {
                "id": "command-1",
                "type": "command_execution",
                "command": command,
                "aggregated_output": "SECRET_COMMAND_OUTPUT_SENTINEL",
            },
        })
        _emit({
            "type": "item.completed",
            "item": {
                "id": "command-1",
                "type": "command_execution",
                "command": command,
                "aggregated_output": command_output,
                "exit_code": 0,
                "status": "completed",
            },
        })
    for event_type in ("item.started", "item.updated", "item.completed"):
        _emit({
            "type": event_type,
            "item": {
                "id": "todo-1",
                "type": "todo_list",
                "items": [{
                    "text": "SECRET_TODO_SENTINEL",
                    "completed": event_type == "item.completed",
                }],
            },
        })
    _emit({
        "type": "item.completed",
        "item": {
            "id": "file-1",
            "type": "file_change",
            "changes": [{
                "path": "SECRET_PATH_SENTINEL",
                "diff": "SECRET_DIFF_SENTINEL",
            }],
            "status": "completed",
        },
    })
    _emit({
        "type": "item.completed",
        "item": {"id": "plan-1", "type": "plan", "text": "SECRET_PLAN_SENTINEL"},
    })
    _emit({
        "type": "item.completed",
        "item": {
            "id": "message-progress",
            "type": "agent_message",
            "text": "SECRET_AGENT_MESSAGE_SENTINEL",
        },
    })
    _emit({"type": "unknown.secret", "token": "SECRET_UNKNOWN_SENTINEL"})
    if mode in {"conflicting_thread", "slow_after_conflicting_thread"}:
        _emit({
            "type": "thread.started",
            "thread_id": "0199a213-81c0-7800-8aa1-bbab2a035a55",
            "account": "SECRET_ACCOUNT_SENTINEL",
        })
    _write_capture("fake-codex-partials-ready.json", True)
    print("SECRET_STDERR_SENTINEL", file=sys.stderr, flush=True)

    if mode in {"slow_after_thread", "slow_after_conflicting_thread"}:
        time.sleep(60)
        return 0
    _apply_mutations(payload)
    if mode == "malformed_json":
        print("not-json", flush=True)
    if mode != "missing_agent_message":
        if mode == "invalid_payload":
            payload = {"status": "not-a-contract"}
        _emit({
            "type": "item.completed",
            "item": {
                "id": "message-final",
                "type": "agent_message",
                "text": json.dumps(payload, separators=(",", ":")),
                "extra": "SECRET_FINAL_EVENT_SENTINEL",
            },
        })
    if mode == "nonzero":
        print("SECRET_FAILURE_DETAIL_SENTINEL", file=sys.stderr, flush=True)
        return 7
    return 0


def main() -> int:
    try:
        return _main()
    except _FakeConfigurationError:
        print("fake Codex configuration error", file=sys.stderr)
        return _CONFIGURATION_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
