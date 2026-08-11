#!/usr/bin/python3
"""Controlled Claude CLI double. Never imports or invokes the real CLI."""

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
        "kind": "claude",
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
    environment = dict(os.environ)
    history_path = _capture_path("captured-env-history.json")
    with _exclusive_lock():
        _atomic_write_json(_capture_path("captured-env.json"), environment)
        _atomic_write_json(_capture_path("captured-argv.json"), argv)
        history = _read_json(history_path, default=[], expected=list)
        if not all(
            isinstance(item, dict)
            and all(isinstance(key, str) and isinstance(value, str) for key, value in item.items())
            for item in history
        ):
            raise _FakeConfigurationError("invocation history is invalid")
        if len(history) >= _MAX_HISTORY_ITEMS:
            raise _FakeConfigurationError("invocation history is full")
        history.append(environment)
        _atomic_write_json(history_path, history)
        _append_invocation_locked(argv)


def _task_brief() -> dict[str, object]:
    return {
        "task_id": os.environ.get("FAKE_CLAUDE_TASK_ID", "task-1"),
        "revision": 1,
        "title": "Plan the task",
        "objective": "Produce a read-only plan.",
        "context": ["Repository context was supplied."],
        "constraints": ["Fable is read-only."],
        "allowed_paths": ["src/agent_bridge"],
        "out_of_scope": ["outside-project"],
        "acceptance_criteria": ["The plan is structured."],
        "required_tests": ["tests/agent_bridge/test_claude_cli.py"],
        "risks": ["Authentication must fail closed."],
        "open_questions": [],
        "confidence": 0.9,
        "confidence_rationale": "The requested scope is explicit.",
    }


def _clarification() -> dict[str, object]:
    return {
        "status": "answered",
        "answer": "Keep the implementation within the approved scope.",
        "reasoning": "The existing task brief already answers the question.",
        "confidence": 0.9,
        "scope_changed": False,
        "revised_brief": None,
        "question_for_user": None,
    }


def _review() -> dict[str, object]:
    return {
        "status": "approved",
        "summary": "The evidence satisfies the acceptance criterion.",
        "criteria": [{
            "criterion": "The plan is structured.",
            "evidence": ["Focused fake-only tests passed."],
            "satisfied": True,
        }],
        "test_assessment": "The focused tests are adequate.",
        "scope_violations": [],
        "remaining_risks": [],
        "corrections": [],
        "question_for_user": None,
    }


def _model_payload(argv: list[str]) -> dict[str, object]:
    kind = _contract_kind(argv)
    return {"plan": _task_brief, "review": _review}.get(
        kind, _clarification
    )()


def _contract_kind(argv: list[str]) -> str:
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    properties = schema["properties"]
    if "task_id" in properties:
        return "plan"
    if "criteria" in properties:
        return "review"
    return "clarification"


def _expected_task_id(argv: list[str]) -> str:
    marker = "Coordinator-generated task ID: "
    prompt = argv[-1]
    if marker not in prompt:
        raise _FakeConfigurationError("plan prompt has no task identity")
    candidate = prompt.split(marker, 1)[1].split(".", 1)[0]
    if (
        not candidate
        or len(candidate) > 128
        or not all(character.isalnum() or character in "._:-" for character in candidate)
    ):
        raise _FakeConfigurationError("plan prompt task identity is unsafe")
    return candidate


def _scenario_result(
    kind: str,
    fallback: dict[str, object],
    argv: list[str],
) -> tuple[dict[str, object], str | None]:
    scenario_value = os.environ.get("FAKE_AGENT_SCENARIO")
    if scenario_value is None:
        return fallback, None
    sequence_key = {
        "plan": "plans",
        "clarification": "clarifications",
        "review": "reviews",
    }[kind]
    with _exclusive_lock():
        scenario = _read_json(Path(scenario_value), expected=dict)
        sequence = scenario.get(sequence_key)
        if not isinstance(sequence, list):
            raise _FakeConfigurationError("scenario sequence is missing")
        state_path = _capture_path("scenario-state-claude.json")
        state = _read_json(state_path, default={}, expected=dict)
        if not all(
            isinstance(key, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for key, value in state.items()
        ):
            raise _FakeConfigurationError("scenario state is invalid")
        raw_index = state.get(kind, 0)
        if not isinstance(raw_index, int) or isinstance(raw_index, bool) or raw_index < 0:
            raise _FakeConfigurationError("scenario counter is invalid")
        if raw_index >= len(sequence):
            raise _FakeConfigurationError("scenario sequence is exhausted")
        candidate = sequence[raw_index]
        if not isinstance(candidate, dict):
            raise _FakeConfigurationError("scenario contract is not an object")
        payload = dict(candidate)
        if kind == "plan" and payload.get("task_id") == "$TASK_ID":
            payload["task_id"] = _expected_task_id(argv)

        configured_modes = scenario.get("claude_modes", {})
        if not isinstance(configured_modes, dict):
            raise _FakeConfigurationError("Claude modes are not an object")
        modes = configured_modes.get(kind, [])
        if not isinstance(modes, list):
            raise _FakeConfigurationError("Claude mode sequence is invalid")
        mode = modes[raw_index] if raw_index < len(modes) else None
        if mode is not None and not isinstance(mode, str):
            raise _FakeConfigurationError("Claude mode is invalid")
        state[kind] = raw_index + 1
        _atomic_write_json(state_path, state)
    return payload, mode


def _main() -> int:
    argv = sys.argv[1:]
    _record_invocation(argv)
    if os.environ.get("AGENT_BRIDGE_TEST_FAKE") != "1":
        print("refusing fake Claude invocation without sentinel", file=sys.stderr)
        return 91
    if argv == ["auth", "status", "--json"]:
        status_value = os.environ.get("FAKE_CLAUDE_AUTH_STATUS")
        if status_value == "malformed":
            print("not-json", flush=True)
        else:
            print(status_value or json.dumps({
                "loggedIn": True,
                "authMethod": "claude.ai",
                "apiProvider": "firstParty",
                "subscriptionType": "max",
            }, separators=(",", ":")), flush=True)
        return int(os.environ.get("FAKE_CLAUDE_AUTH_EXIT", "0"))

    kind = _contract_kind(argv)
    payload, scenario_mode = _scenario_result(
        kind, _model_payload(argv), argv
    )
    mode = scenario_mode or os.environ.get("FAKE_CLAUDE_MODE", "success")
    default_session_id = (
        argv[argv.index("--resume") + 1] if "--resume" in argv else "fable-session-1"
    )
    init = {
        "type": "system",
        "subtype": "init",
        "session_id": os.environ.get("FAKE_CLAUDE_SESSION_ID", default_session_id),
    }
    assistant = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "Working read-only."}]},
    }
    user = {"type": "user", "message": "SECRET_USER_TEXT_SENTINEL"}
    stream_event = {
        "type": "stream_event",
        "event": {"delta": {"text": "SECRET_STREAM_TEXT_SENTINEL"}},
    }
    unknown = {"type": "credential_dump", "token": "SECRET_UNKNOWN_EVENT_SENTINEL"}
    nonstring_unknown = {"type": [], "token": "SECRET_NONSTRING_TYPE_SENTINEL"}
    if mode == "invalid_payload":
        payload = {"status": "not-a-contract"}
    result = {"type": "result", "subtype": "success", "structured_output": payload}
    if os.environ.get("FAKE_CLAUDE_SECRET_OUTPUT") == "1":
        init.update({
            "email": "SECRET_EMAIL_SENTINEL@example.invalid",
            "organization": "SECRET_ORGANIZATION_SENTINEL",
            "env": {"ANTHROPIC_API_KEY": "SECRET_CREDENTIAL_SENTINEL"},
        })
        assistant["message"] = {
            "content": [{"type": "text", "text": "SECRET_ASSISTANT_TEXT_SENTINEL"}],
            "credential": "SECRET_CREDENTIAL_SENTINEL",
        }
        result["unknown"] = "SECRET_RESULT_FIELD_SENTINEL"

    if mode == "slow_before_init":
        time.sleep(60)
        return 0
    if mode != "missing_init":
        print(json.dumps(init, separators=(",", ":")), flush=True)
    print(json.dumps(assistant, separators=(",", ":")), flush=True)
    if os.environ.get("FAKE_CLAUDE_SECRET_OUTPUT") == "1":
        print(json.dumps(user, separators=(",", ":")), flush=True)
        print(json.dumps(stream_event, separators=(",", ":")), flush=True)
        print(json.dumps(unknown, separators=(",", ":")), flush=True)
        print(json.dumps(nonstring_unknown, separators=(",", ":")), flush=True)
        print("SECRET_STDERR_SENTINEL", file=sys.stderr, flush=True)
    _write_capture("fake-claude-partials-ready.json", True)
    if mode == "malformed_json":
        print("not-json", flush=True)
    if mode == "slow_after_init":
        time.sleep(60)
        return 0
    if mode != "missing_result":
        print(json.dumps(result, separators=(",", ":")), flush=True)
    if mode == "nonzero":
        print("controlled model failure", file=sys.stderr, flush=True)
        return 7
    return 0


def main() -> int:
    try:
        return _main()
    except _FakeConfigurationError:
        print("fake Claude configuration error", file=sys.stderr)
        return _CONFIGURATION_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
