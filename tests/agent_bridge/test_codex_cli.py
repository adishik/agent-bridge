from __future__ import annotations

import asyncio
import fcntl
import json
import os
from pathlib import Path
import stat
import sys

import pytest

from agent_bridge.adapters.codex_cli import (
    CodexCLI,
    CodexRunError,
    materialize_sol_schema_file,
)
from agent_bridge.contracts import SOL_OUTCOME_SCHEMA, TaskBrief
from agent_bridge.process import ProcessRunner


SAFE_ENV = {"AGENT_BRIDGE_TEST_FAKE": "1", "LANG": "C.UTF-8", "PATH": "/not-used"}
THREAD_ID = "0199a213-81c0-7800-8aa1-bbab2a035a53"
SECRET_SENTINELS = (
    "SECRET_ACCOUNT_SENTINEL",
    "SECRET_COMMAND_SENTINEL",
    "SECRET_TOKEN_SENTINEL",
    "SECRET_COMMAND_OUTPUT_SENTINEL",
    "SECRET_TODO_SENTINEL",
    "SECRET_PATH_SENTINEL",
    "SECRET_DIFF_SENTINEL",
    "SECRET_PLAN_SENTINEL",
    "SECRET_AGENT_MESSAGE_SENTINEL",
    "SECRET_UNKNOWN_SENTINEL",
    "SECRET_STDERR_SENTINEL",
    "SECRET_FINAL_EVENT_SENTINEL",
    "SECRET_FAILURE_DETAIL_SENTINEL",
)


def _adapter(fake_codex: Path, tmp_path: Path, **extra_env: str) -> CodexCLI:
    return CodexCLI(
        fake_codex,
        ProcessRunner(stop_grace_seconds=0.02),
        repo_root=tmp_path,
        schema_dir=tmp_path / "schemas",
        env={**SAFE_ENV, **extra_env},
    )


def _assert_no_secret_sentinel(value: object) -> None:
    representation = repr(value)
    for sentinel in SECRET_SENTINELS:
        assert sentinel not in representation


async def _wait_for_capture(tmp_path: Path) -> None:
    path = tmp_path / "captured-codex-argv.json"
    deadline = asyncio.get_running_loop().time() + 3
    while not path.exists():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("fake Codex invocation did not start")
        await asyncio.sleep(0.005)


async def _wait_for_partials(tmp_path: Path) -> None:
    path = tmp_path / "fake-codex-partials-ready.json"
    deadline = asyncio.get_running_loop().time() + 3
    while not path.exists():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("fake Codex partial events were not emitted")
        await asyncio.sleep(0.005)


@pytest.mark.parametrize("executable", ("codex", "bin/codex"))
def test_sol_rejects_bare_and_relative_executable_names(
    executable: str, tmp_path: Path,
) -> None:
    with pytest.raises(ValueError) as raised:
        CodexCLI(
            executable,
            ProcessRunner(),
            repo_root=tmp_path,
            schema_dir=tmp_path / "schemas",
            env=SAFE_ENV,
        )
    assert str(raised.value) == "executable must be an absolute path"


@pytest.mark.parametrize("kind", ("directory", "non_executable"))
def test_sol_requires_an_absolute_executable_file(kind: str, tmp_path: Path) -> None:
    executable = tmp_path / "candidate"
    if kind == "directory":
        executable.mkdir()
    else:
        executable.write_text("not executable", encoding="utf-8")
        executable.chmod(0o600)

    with pytest.raises(ValueError) as raised:
        CodexCLI(
            executable,
            ProcessRunner(),
            repo_root=tmp_path,
            schema_dir=tmp_path / "schemas",
            env=SAFE_ENV,
        )
    assert str(raised.value) == "executable must be an executable file"


@pytest.mark.parametrize(
    ("repo_root", "schema_dir", "match"),
    [
        (Path("relative-repo"), None, "repo_root must be an absolute path"),
        (None, Path("relative-schema"), "schema_dir must be an absolute path"),
    ],
)
def test_sol_requires_absolute_workspace_and_schema_paths(
    fake_codex: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path | None,
    schema_dir: Path | None,
    match: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=match):
        CodexCLI(
            fake_codex,
            ProcessRunner(),
            repo_root=tmp_path if repo_root is None else repo_root,
            schema_dir=tmp_path / "schemas" if schema_dir is None else schema_dir,
            env=SAFE_ENV,
        )


def test_sol_start_uses_workspace_write_safe_flags_and_revision_bound_prompt(
    fake_codex: Path, brief: TaskBrief, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        result = await _adapter(fake_codex, tmp_path).start(
            run_id="run-1",
            brief=brief,
            context="AGENTS rules and immutable baseline baseline-1",
        )

        assert result.cli_session_id == THREAD_ID
        assert result.payload is not None
        assert result.payload["status"] == "question"
        argv = json.loads((tmp_path / "captured-codex-argv.json").read_text())
        assert argv[:2] == ["exec", "--json"]
        assert argv[argv.index("--model"):argv.index("--model") + 2] == [
            "--model", "gpt-5.6-sol",
        ]
        assert argv[argv.index("--sandbox"):argv.index("--sandbox") + 2] == [
            "--sandbox", "workspace-write",
        ]
        assert argv[argv.index("--cd"):argv.index("--cd") + 2] == [
            "--cd", str(tmp_path),
        ]
        assert "--approve-for-me" in argv
        forbidden = {
            "--dangerously-bypass-approvals-and-sandbox",
            "--dangerously-bypass-hook-trust",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
        }
        assert not (forbidden & set(argv))
        prompt = argv[-1]
        assert "AGENTS rules" in prompt
        assert "baseline-1" in prompt
        assert '"revision":1' in prompt
        assert json.dumps(brief.to_dict(), separators=(",", ":"), sort_keys=True) in prompt
        assert "src/agent_bridge" in prompt
        assert "Do not commit" in prompt
        assert "paid service" in prompt
        assert "SolOutcome" in prompt

    asyncio.run(scenario())


def test_sol_materializes_schema_with_owner_only_mode(
    fake_codex: Path, tmp_path: Path,
) -> None:
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    schema_path = schema_dir / "sol-outcome.json"
    schema_path.write_text("old", encoding="utf-8")
    schema_path.chmod(0o644)

    adapter = _adapter(fake_codex, tmp_path)

    assert adapter.schema_path == schema_path
    assert json.loads(schema_path.read_text(encoding="utf-8")) == SOL_OUTCOME_SCHEMA
    assert stat.S_IMODE(schema_path.stat().st_mode) == 0o600


def test_materialize_sol_schema_file_returns_exact_read_only_regular_file(
    tmp_path: Path,
) -> None:
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    directory_fd = os.open(schemas, os.O_RDONLY | os.O_DIRECTORY)
    schema_file_fd = -1
    try:
        schema_file_fd = materialize_sol_schema_file(directory_fd)
        assert stat.S_ISREG(os.fstat(schema_file_fd).st_mode)
        assert fcntl.fcntl(schema_file_fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        duplicate = os.dup(schema_file_fd)
        try:
            assert os.read(duplicate, 1_000_000) == json.dumps(
                SOL_OUTCOME_SCHEMA, separators=(",", ":"), sort_keys=True,
            ).encode("utf-8")
        finally:
            os.close(duplicate)
    finally:
        if schema_file_fd >= 0:
            os.close(schema_file_fd)
        os.close(directory_fd)


@pytest.mark.parametrize("kind", ("directory", "writable", "closed", "wrong_content"))
def test_sol_rejects_noncanonical_schema_file_descriptors(
    fake_codex: Path, tmp_path: Path, kind: str,
) -> None:
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    if kind == "directory":
        descriptor = os.open(schemas, os.O_RDONLY | os.O_DIRECTORY)
    else:
        path = schemas / f"{kind}.json"
        path.write_text(
            "{}" if kind == "wrong_content" else json.dumps(SOL_OUTCOME_SCHEMA),
            encoding="utf-8",
        )
        descriptor = os.open(
            path,
            os.O_WRONLY if kind == "writable" else os.O_RDONLY,
        )
        if kind == "closed":
            os.close(descriptor)
    try:
        with pytest.raises(ValueError, match="schema_file_fd"):
            CodexCLI(
                fake_codex,
                ProcessRunner(),
                repo_root=tmp_path,
                schema_dir=schemas,
                schema_file_fd=descriptor,
                env=SAFE_ENV,
            )
    finally:
        if kind != "closed":
            os.close(descriptor)


def test_sol_keeps_an_injected_schema_file_available_to_its_child(
    brief: TaskBrief, tmp_path: Path,
) -> None:
    """A child gets the one schema file through its caller-owned descriptor."""
    async def scenario() -> None:
        capability_codex = tmp_path / "schema-file-codex"
        capability_codex.write_text(
            f"#!{sys.executable}\n"
            "import json\n"
            "from pathlib import Path\n"
            "import sys\n"
            "if sys.argv[1:] == ['--version']:\n"
            "    print('codex-cli capability')\n"
            "    raise SystemExit(0)\n"
            "schema_path = Path(sys.argv[sys.argv.index('--output-schema') + 1])\n"
            "json.loads(schema_path.read_text(encoding='utf-8'))\n"
            "print(json.dumps({'type': 'thread.started', 'thread_id': '0199a213-81c0-7800-8aa1-bbab2a035a53'}))\n"
            "print(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': json.dumps({'status': 'completed', 'summary': 'schema read', 'changed_files': [], 'commands_run': [], 'known_failures': [], 'remaining_risks': [], 'architecture_docs': 'No change.', 'question': None})}}))\n",
            encoding="utf-8",
        )
        capability_codex.chmod(0o700)
        state = tmp_path / "state"
        state.mkdir()
        schemas = state / "schemas"
        schemas.mkdir()
        directory_fd = os.open(schemas, os.O_RDONLY | os.O_DIRECTORY)
        descriptor = materialize_sol_schema_file(directory_fd)
        try:
            adapter = CodexCLI(
                capability_codex,
                ProcessRunner(stop_grace_seconds=0.02),
                repo_root=tmp_path,
                schema_dir=schemas,
                schema_file_fd=descriptor,
                env=SAFE_ENV,
            )
            assert adapter.schema_path == Path(f"/proc/self/fd/{descriptor}")
            result = await adapter.start(
                run_id="descriptor-anchored-schema", brief=brief, context="context",
            )
        finally:
            os.close(descriptor)
            os.close(directory_fd)
        assert result.payload is not None

    asyncio.run(scenario())


def test_sol_resume_uses_exact_thread_and_validates_outcome(
    fake_codex: Path, brief: TaskBrief, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        adapter = _adapter(fake_codex, tmp_path)
        await adapter.start(run_id="run-1", brief=brief, context="context")
        result = await adapter.resume(
            run_id="run-2",
            thread_id=THREAD_ID,
            prompt="Fable answered: keep the change inside src/agent_bridge.",
        )

        assert result.cli_session_id == THREAD_ID
        assert result.payload is not None
        assert result.payload["status"] == "completed"
        argv = json.loads((tmp_path / "captured-codex-argv.json").read_text())
        assert argv[:4] == ["exec", "resume", "--json", "--model"]
        assert argv[4] == "gpt-5.6-sol"
        assert argv[argv.index("--output-schema") + 1] == str(adapter.schema_path)
        assert argv[-2] == THREAD_ID
        assert "Fable answered" in argv[-1]
        assert "latest exact user-approved TaskBrief revision" in argv[-1]
        assert "original approved TaskBrief revision" not in argv[-1]
        assert "--sandbox" not in argv
        assert "--approve-for-me" not in argv
        assert "--cd" not in argv
        assert "--ephemeral" not in argv

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "thread_id",
    (
        "--last",
        "--dangerously-bypass-approvals-and-sandbox",
        "thread id with spaces",
        "thread-id\n--ephemeral",
        "not-a-uuid",
    ),
)
def test_sol_resume_rejects_noncanonical_thread_ids_before_invocation(
    fake_codex: Path, tmp_path: Path, thread_id: str,
) -> None:
    async def scenario() -> None:
        with pytest.raises(ValueError, match="canonical UUID"):
            await _adapter(fake_codex, tmp_path).resume(
                run_id="run-invalid-thread",
                thread_id=thread_id,
                prompt="Continue.",
            )
        assert not (tmp_path / "captured-codex-argv.json").exists()

    asyncio.run(scenario())


def test_sol_rejects_noncanonical_thread_id_emitted_by_codex(
    fake_codex: Path, brief: TaskBrief, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        with pytest.raises(CodexRunError, match="canonical thread ID") as raised:
            await _adapter(
                fake_codex,
                tmp_path,
                FAKE_CODEX_THREAD_ID="--last",
            ).start(run_id="run-bad-emitted-thread", brief=brief, context="context")
        assert raised.value.result is not None
        assert raised.value.result.cli_session_id is None

    asyncio.run(scenario())


def test_sol_returns_only_structural_audit_events_and_summarized_stderr(
    fake_codex: Path, brief: TaskBrief, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        result = await _adapter(fake_codex, tmp_path).start(
            run_id="run-safe-audit", brief=brief, context="context",
        )

        assert tuple(dict(event) for event in result.events) == (
            {"type": "thread.started", "thread_id": THREAD_ID},
            {
                "type": "item.started",
                "item_type": "command_execution",
                "command_sha256": "9f5fc393a0a72fb9b8b443518c2e307e40dc0907d94dbf64828c6d48402f9137",
            },
            {
                "type": "item.completed",
                "item_type": "command_execution",
                "status": "completed",
                "command_sha256": "9f5fc393a0a72fb9b8b443518c2e307e40dc0907d94dbf64828c6d48402f9137",
                "exit_code": 0,
                "output_sha256": "e0a53721297ab30510ae594e41dbc6cfce7965b95bf1dc82ba1d447ae499f9dc",
                "output_bytes": 30,
                "output_lines": 1,
            },
            {"type": "item.started", "item_type": "todo_list"},
            {"type": "item.updated", "item_type": "todo_list"},
            {"type": "item.completed", "item_type": "todo_list"},
            {"type": "item.completed", "item_type": "file_change", "status": "completed"},
            {"type": "item.completed", "item_type": "plan"},
            {"type": "item.completed", "item_type": "agent_message"},
            {"type": "item.completed", "item_type": "agent_message"},
        )
        assert result.stderr == ("stderr_lines=1",)
        _assert_no_secret_sentinel(result)

    asyncio.run(scenario())


def test_sol_parser_coalesces_a_structural_event_flood_but_keeps_completion_data() -> None:
    """Audit retention must not scale linearly with an untrusted event stream."""
    flood = json.dumps({
        "type": "item.updated",
        "item": {"type": "todo_list"},
    })
    lines = (
        json.dumps({"type": "thread.started", "thread_id": THREAD_ID}),
        *(flood for _ in range(1_300)),
        json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "{}"},
        }),
    )

    parsed = CodexCLI._parse_events(lines, interrupted=False)

    assert parsed.thread_id == THREAD_ID
    assert parsed.final_message == "{}"
    assert len(parsed.audit_events) <= 1_024


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("missing_thread", "thread"),
        ("missing_agent_message", "agent message"),
        ("invalid_payload", "schema"),
        ("malformed_json", "JSON"),
    ],
)
def test_completed_sol_run_rejects_incomplete_or_invalid_output(
    fake_codex: Path, brief: TaskBrief, tmp_path: Path, mode: str, match: str,
) -> None:
    async def scenario() -> None:
        extra_env = {"FAKE_CODEX_MODE": mode}
        if mode == "missing_thread":
            extra_env["FAKE_CODEX_THREAD_ID"] = ""
        with pytest.raises(CodexRunError, match=match) as raised:
            await _adapter(fake_codex, tmp_path, **extra_env).start(
                run_id=f"run-{mode}", brief=brief, context="context",
            )
        result = raised.value.result
        assert result is not None
        assert result.payload is None
        assert result.interrupted is False
        _assert_no_secret_sentinel(raised.value)
        _assert_no_secret_sentinel(result)

    asyncio.run(scenario())


def test_nonzero_sol_run_error_retains_safe_audit_only(
    fake_codex: Path, brief: TaskBrief, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        with pytest.raises(CodexRunError, match="exit") as raised:
            await _adapter(fake_codex, tmp_path, FAKE_CODEX_MODE="nonzero").start(
                run_id="run-nonzero", brief=brief, context="context",
            )
        result = raised.value.result
        assert result is not None
        assert result.exit_code == 7
        assert result.payload is None
        assert result.stderr == ("stderr_lines=2",)
        _assert_no_secret_sentinel(raised.value)
        _assert_no_secret_sentinel(result)

    asyncio.run(scenario())


def test_resume_rejects_a_different_thread_reported_by_codex(
    fake_codex: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        with pytest.raises(CodexRunError, match="different thread") as raised:
            await _adapter(
                fake_codex,
                tmp_path,
                FAKE_CODEX_THREAD_ID="0199a213-81c0-7800-8aa1-bbab2a035a54",
            ).resume(
                run_id="run-wrong-thread",
                thread_id=THREAD_ID,
                prompt="Continue.",
            )
        assert raised.value.result is not None
        assert raised.value.result.cli_session_id is None

    asyncio.run(scenario())


def test_interrupted_resume_mismatch_never_exposes_wrong_continuation_thread(
    fake_codex: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.02)
        adapter = CodexCLI(
            fake_codex,
            runner,
            repo_root=tmp_path,
            schema_dir=tmp_path / "schemas",
            env={
                **SAFE_ENV,
                "FAKE_CODEX_MODE": "slow_after_thread",
                "FAKE_CODEX_THREAD_ID": "0199a213-81c0-7800-8aa1-bbab2a035a54",
            },
        )
        run = asyncio.create_task(adapter.resume(
            run_id="run-interrupted-wrong-thread",
            thread_id=THREAD_ID,
            prompt="Continue.",
        ))
        await _wait_for_capture(tmp_path)
        await _wait_for_partials(tmp_path)
        await runner.stop("run-interrupted-wrong-thread")

        with pytest.raises(CodexRunError, match="different thread") as raised:
            await run
        result = raised.value.result
        assert result is not None
        assert result.interrupted is True
        assert result.cli_session_id is None
        assert result.payload is None

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ("malformed_json", "conflicting_thread"))
def test_resume_parse_error_never_exposes_wrong_continuation_thread(
    fake_codex: Path, tmp_path: Path, mode: str,
) -> None:
    async def scenario() -> None:
        with pytest.raises(CodexRunError) as raised:
            await _adapter(
                fake_codex,
                tmp_path,
                FAKE_CODEX_MODE=mode,
                FAKE_CODEX_THREAD_ID="0199a213-81c0-7800-8aa1-bbab2a035a54",
            ).resume(
                run_id=f"run-wrong-thread-{mode}",
                thread_id=THREAD_ID,
                prompt="Continue.",
            )
        result = raised.value.result
        assert result is not None
        assert result.interrupted is False
        assert result.cli_session_id is None
        assert result.payload is None

    asyncio.run(scenario())


def test_interrupted_resume_parse_error_never_exposes_wrong_continuation_thread(
    fake_codex: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.02)
        adapter = CodexCLI(
            fake_codex,
            runner,
            repo_root=tmp_path,
            schema_dir=tmp_path / "schemas",
            env={
                **SAFE_ENV,
                "FAKE_CODEX_MODE": "slow_after_conflicting_thread",
                "FAKE_CODEX_THREAD_ID": "0199a213-81c0-7800-8aa1-bbab2a035a54",
            },
        )
        run = asyncio.create_task(adapter.resume(
            run_id="run-interrupted-conflicting-thread",
            thread_id=THREAD_ID,
            prompt="Continue.",
        ))
        await _wait_for_capture(tmp_path)
        await _wait_for_partials(tmp_path)
        await runner.stop("run-interrupted-conflicting-thread")

        with pytest.raises(CodexRunError, match="conflicting") as raised:
            await run
        result = raised.value.result
        assert result is not None
        assert result.interrupted is True
        assert result.cli_session_id is None
        assert result.payload is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mode", "expected_thread"),
    [("slow_before_thread", None), ("slow_after_thread", THREAD_ID)],
)
def test_interrupted_sol_run_returns_partial_result_only_after_observed_thread(
    fake_codex: Path,
    brief: TaskBrief,
    tmp_path: Path,
    mode: str,
    expected_thread: str | None,
) -> None:
    async def scenario() -> None:
        runner = ProcessRunner(stop_grace_seconds=0.02)
        adapter = CodexCLI(
            fake_codex,
            runner,
            repo_root=tmp_path,
            schema_dir=tmp_path / "schemas",
            env={**SAFE_ENV, "FAKE_CODEX_MODE": mode},
        )
        run = asyncio.create_task(adapter.start(
            run_id=f"run-{mode}", brief=brief, context="context",
        ))
        await _wait_for_capture(tmp_path)
        if mode == "slow_after_thread":
            await _wait_for_partials(tmp_path)
        await runner.stop(f"run-{mode}")
        result = await run

        assert result.interrupted is True
        assert result.cli_session_id == expected_thread
        assert result.payload is None
        _assert_no_secret_sentinel(result)

    asyncio.run(scenario())
