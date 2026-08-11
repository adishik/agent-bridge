from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

import agent_bridge.__main__ as launcher
from agent_bridge.adapters.codex_cli import CodexCLI
from agent_bridge.process import ProcessRunner
from agent_bridge.state_machine import TaskState
from agent_bridge.store import SQLiteStore
from agent_bridge.__main__ import (
    main,
    parse_settings,
    prepare_state_dir,
    select_port,
    ssh_forward_command,
)


def _write_executable(path: Path, source: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(0o700)
    return path.resolve()


def _fake_tools(tmp_path: Path) -> dict[str, Path]:
    tools = tmp_path / "tools"
    tools.mkdir()
    log_path = tmp_path / "commands.jsonl"
    common_log = """
import json
import os
from pathlib import Path
import sys
import time

control_dir = Path(sys.argv[0]).parent
log_path = Path(__LOG_PATH__)
with log_path.open("a", encoding="utf-8") as log:
    log.write(json.dumps({
            "tool": Path(sys.argv[0]).name,
            "argv": sys.argv[1:],
            "environment": {
                key: os.environ[key]
                for key in (
                    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                    "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN",
                    "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                    "CLAUDE_CODE_USE_FOUNDRY",
                    "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY",
                    "AZURE_CLIENT_SECRET", "GOOGLE_APPLICATION_CREDENTIALS",
                    "DATABASE_URL", "SSH_AUTH_SOCK", "SSH_CONNECTION", "PASSWORD",
                    "HOME", "CODEX_HOME", "XDG_CONFIG_HOME", "PATH",
                    "XDG_CACHE_HOME", "XDG_DATA_HOME", "TERM", "TMPDIR",
                    "USER", "LOGNAME", "VIRTUAL_ENV", "LC_CTYPE",
                    "GIT_DIR", "GIT_WORK_TREE", "GIT_CONFIG_GLOBAL",
                    "GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM",
                    "GIT_TERMINAL_PROMPT", "GIT_OPTIONAL_LOCKS",
                    "GIT_LFS_SKIP_SMUDGE",
                    "GCM_INTERACTIVE", "LC_ALL", "LANG",
                )
                if key in os.environ
            },
        }, sort_keys=True) + "\\n")

def maybe_hang(probe):
    if (control_dir / ("hang-" + probe)).exists():
        (control_dir / ("hung-" + probe + ".pid")).write_text(str(os.getpid()))
        time.sleep(2)
""".replace("__LOG_PATH__", repr(str(log_path)))
    claude = _write_executable(
        tools / "claude",
        common_log
        + """
if sys.argv[1:] == ["--version"]:
    maybe_hang("claude-version")
    print("Claude Code 9.9.9")
elif sys.argv[1:] == ["auth", "status", "--json"]:
    maybe_hang("claude-auth")
    if os.environ.get("AGENT_BRIDGE_TEST_AUTH") == "invalid":
        print(json.dumps({
            "loggedIn": False, "authMethod": "none",
            "apiProvider": "none", "subscriptionType": "",
        }))
    else:
        print(json.dumps({
            "loggedIn": True, "authMethod": "claude.ai",
            "apiProvider": "firstParty", "subscriptionType": "max",
        }))
else:
    raise SystemExit(91)
""",
    )
    codex = _write_executable(
        tools / "codex",
        common_log
        + """
if sys.argv[1:] == ["--version"]:
    maybe_hang("codex-version")
    print("codex-cli 9.9.9")
elif sys.argv[1:3] == ["exec", "--json"]:
    print(json.dumps({
        "type": "thread.started",
        "thread_id": "0199a213-81c0-7800-8aa1-bbab2a035a53",
    }))
    print(json.dumps({
        "type": "item.completed",
        "item": {
            "id": "message-final", "type": "agent_message",
            "text": json.dumps({
                "status": "completed", "summary": "Fake Sol completed.",
                "changed_files": [], "commands_run": [], "known_failures": [],
                "remaining_risks": [], "architecture_docs": "No change.",
                "question": None,
            }),
        },
    }))
else:
    raise SystemExit(92)
""",
    )
    git = _write_executable(
        tools / "git",
        common_log
        + """
if (control_dir / "git-fail").exists():
    print("not a repository", file=sys.stderr)
    raise SystemExit(1)
if sys.argv[-2:] == ["rev-parse", "--show-toplevel"]:
    override = control_dir / "git-root"
    print(override.read_text() if override.exists() else os.getcwd())
elif sys.argv[-2:] == ["branch", "--show-current"]:
    print("feat/test-launcher")
else:
    raise SystemExit(93)
""",
    )
    bash = _write_executable(tools / "bash", "raise SystemExit(94)\n")
    sh = _write_executable(tools / "sh", "raise SystemExit(95)\n")
    return {"claude": claude, "codex": codex, "git": git, "bash": bash, "sh": sh}


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("# Test rules\nUse fake tools only.\n", encoding="utf-8")
    return repo.resolve()


def _args(repo: Path, tools: dict[str, Path], *, host: str = "127.0.0.1") -> list[str]:
    return [
        "--repo", str(repo), "--host", host, "--port", "56590",
        "--claude-executable", str(tools["claude"]),
        "--codex-executable", str(tools["codex"]),
        "--git-executable", str(tools["git"]),
        "--bash-executable", str(tools["bash"]),
        "--sh-executable", str(tools["sh"]),
    ]


def _environment(tmp_path: Path, repo: Path) -> dict[str, str]:
    return {
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "HOME": str(tmp_path / "home"),
        "CODEX_HOME": str(tmp_path / "codex-home"),
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "USER": "bridgeuser",
        "SSH_CONNECTION": "192.0.2.10 55481 198.51.100.20 22",
        "AGENT_BRIDGE_TEST_LOG": str(tmp_path / "commands.jsonl"),
        "ANTHROPIC_API_KEY": "provider-secret-must-not-leak",
        "ANTHROPIC_AUTH_TOKEN": "provider-token-must-not-leak",
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token-must-not-leak",
        "OPENAI_API_KEY": "openai-secret-must-not-leak",
        "AWS_SECRET_ACCESS_KEY": "aws-secret-must-not-leak",
        "AZURE_CLIENT_SECRET": "azure-secret-must-not-leak",
        "GOOGLE_APPLICATION_CREDENTIALS": "/secret/google.json",
        "DATABASE_URL": "postgres://secret",
        "SSH_AUTH_SOCK": "/secret/ssh-agent.sock",
        "PASSWORD": "password-secret-must-not-leak",
    }


def test_importing_main_keeps_optional_web_stack_unloaded() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    package_root = (source_root / "agent_bridge").resolve()
    assert package_root.is_dir()
    result = subprocess.run(
        [sys.executable, "-c", (
            "import os; from pathlib import Path; import sys; import agent_bridge; "
            "assert Path(agent_bridge.__file__).resolve().parent == "
            "Path(os.environ['AGENT_BRIDGE_IMPORT_ROOT']).resolve(); "
            "import agent_bridge.__main__; "
            "assert 'fastapi' not in sys.modules; assert 'uvicorn' not in sys.modules"
        )],
        cwd=source_root.parent,
        env={
            "PYTHONPATH": str(source_root),
            "AGENT_BRIDGE_IMPORT_ROOT": str(package_root),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_parse_settings_rejects_public_bind_missing_repo_and_non_git(tmp_path: Path) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    environ = _environment(tmp_path, repo)

    with pytest.raises(ValueError, match="loopback"):
        parse_settings(_args(repo, tools, host="0.0.0.0"), environ=environ)
    with pytest.raises(ValueError, match="repository"):
        parse_settings(_args(repo / "missing", tools), environ=environ)
    with pytest.raises(ValueError, match="Git repository"):
        (tools["git"].parent / "git-fail").touch()
        try:
            parse_settings(_args(repo, tools), environ=environ)
        finally:
            (tools["git"].parent / "git-fail").unlink()
    other = tmp_path / "other"
    other.mkdir()
    (tools["git"].parent / "git-root").write_text(str(other), encoding="utf-8")
    with pytest.raises(ValueError, match="top level"):
        parse_settings(_args(repo, tools), environ=environ)


def test_parse_settings_requires_absolute_regular_executables(tmp_path: Path) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    environ = _environment(tmp_path, repo)
    args = _args(repo, tools)
    args[args.index("--claude-executable") + 1] = str(tmp_path / "missing-claude")

    with pytest.raises(ValueError, match="Claude executable"):
        parse_settings(args, environ=environ)


def test_state_directory_is_external_owner_only_and_repository_specific(tmp_path: Path) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    settings = parse_settings(
        _args(repo, tools), environ=_environment(tmp_path, repo),
    )

    state_dir = prepare_state_dir(settings)

    assert state_dir == tmp_path / "state" / "agent-bridge" / "repo"
    assert not state_dir.is_relative_to(repo)
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
    state_dir.chmod(0o777)
    assert prepare_state_dir(settings) == state_dir
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700


def test_state_directory_inside_repository_is_rejected(tmp_path: Path) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    environ = {
        **_environment(tmp_path, repo),
        "XDG_STATE_HOME": str(repo / "state"),
    }
    settings = parse_settings(_args(repo, tools), environ=environ)

    with pytest.raises(ValueError, match="outside repository"):
        prepare_state_dir(settings)


def test_repository_context_reads_only_bounded_regular_repo_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path)
    external = tmp_path / "external-secret"
    external.write_text("EXTERNAL-CREDENTIAL-MATERIAL", encoding="utf-8")
    agents = repo / "AGENTS.md"
    agents.unlink()
    agents.symlink_to(external)

    symlink_context = launcher.read_repository_context(repo)

    assert "EXTERNAL-CREDENTIAL-MATERIAL" not in symlink_context
    agents.unlink()
    os.mkfifo(agents)
    started = time.monotonic()
    fifo_context = launcher.read_repository_context(repo)
    assert time.monotonic() - started < 0.2
    assert "Repository root:" in fifo_context
    agents.unlink()
    agents.write_text("stable context", encoding="utf-8")
    original_open = os.open
    raced = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal raced
        if path == "AGENTS.md" and kwargs.get("dir_fd") is not None and not raced:
            raced = True
            agents.unlink()
            agents.write_text("RACE-CREDENTIAL-MATERIAL", encoding="utf-8")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", racing_open)
    race_context = launcher.read_repository_context(repo)
    monkeypatch.setattr(os, "open", original_open)
    assert raced is True
    assert "RACE-CREDENTIAL-MATERIAL" not in race_context
    agents.unlink()
    agents.write_bytes(b"A" * (launcher.MAX_REPO_CONTEXT_BYTES + 100))
    bounded = launcher.read_repository_context(repo)
    assert len(bounded.encode("utf-8")) <= launcher.MAX_REPO_CONTEXT_BYTES


def test_codex_environment_is_minimal_for_version_and_actual_model_execution(
    tmp_path: Path, valid_brief,
) -> None:
    tools = _fake_tools(tmp_path)
    source = {
        "HOME": str(tmp_path / "home"),
        "CODEX_HOME": str(tmp_path / "codex-home"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "TERM": "xterm-256color",
        "USER": "adi",
        "VIRTUAL_ENV": "/venv",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "OPENAI_API_KEY": "openai-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "AZURE_CLIENT_SECRET": "azure-secret",
        "GOOGLE_APPLICATION_CREDENTIALS": "/secret/google.json",
        "DATABASE_URL": "postgres://secret",
        "SSH_AUTH_SOCK": "/secret/agent.sock",
        "PASSWORD": "password-secret",
    }

    environment = launcher.codex_environment(source)
    allowed_keys = (
        "HOME", "CODEX_HOME", "XDG_CONFIG_HOME", "PATH", "LANG",
        "TERM", "USER", "VIRTUAL_ENV",
    )
    assert environment == {key: source[key] for key in allowed_keys}
    hostile_path_environment = launcher.codex_environment({
        **source,
        "PATH": "relative::/usr/bin:/usr/bin:/bin",
    })
    assert hostile_path_environment["PATH"] == "/usr/bin:/bin"

    async def scenario() -> None:
        adapter = CodexCLI(
            tools["codex"],
            ProcessRunner(stop_grace_seconds=0.02),
            repo_root=tmp_path,
            schema_dir=tmp_path / "schemas",
            env=environment,
        )
        await adapter.start(
            run_id="run-environment",
            brief=valid_brief,
            context="Fake model environment boundary.",
        )

    asyncio.run(scenario())
    call = json.loads((tmp_path / "commands.jsonl").read_text().splitlines()[-1])
    assert call["argv"][:2] == ["exec", "--json"]
    assert call["environment"] == {key: source[key] for key in allowed_keys}


def test_git_validation_uses_fixed_minimal_environment(tmp_path: Path) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    hostile = {
        **_environment(tmp_path, repo),
        "GIT_DIR": "/outside/.git",
        "GIT_WORK_TREE": "/outside",
        "GIT_CONFIG_GLOBAL": "/outside/config",
        "GIT_CONFIG_SYSTEM": "/outside/config",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "credential.helper",
        "GIT_CONFIG_VALUE_0": "malicious",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "DATABASE_URL": "postgres://secret",
        "SSH_AUTH_SOCK": "/secret/agent.sock",
    }

    parse_settings(_args(repo, tools), environ=hostile)

    calls = [
        json.loads(line)
        for line in (tmp_path / "commands.jsonl").read_text().splitlines()
    ]
    git_calls = [call for call in calls if call["tool"] == "git"]
    assert len(git_calls) == 2
    expected = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_LFS_SKIP_SMUDGE": "1",
        "GCM_INTERACTIVE": "Never",
        "LC_ALL": "C",
        "LANG": "C",
    }
    assert all(call["environment"] == expected for call in git_calls)


def test_ssh_forward_uses_current_server_and_rejects_unsafe_user() -> None:
    assert ssh_forward_command(
        port=56590, user="bridgeuser",
        ssh_connection="192.0.2.10 55481 198.51.100.20 22",
    ) == "ssh -N -L 56590:127.0.0.1:56590 bridgeuser@198.51.100.20"
    with pytest.raises(ValueError, match="user"):
        ssh_forward_command(
            port=56590, user="bridgeuser; touch /tmp/unsafe",
            ssh_connection="192.0.2.10 55481 198.51.100.20 22",
        )


def test_zero_port_uses_one_temporary_loopback_socket() -> None:
    calls: list[object] = []

    class FakeSocket:
        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *args):
            calls.append("close")

        def bind(self, address):
            calls.append(("bind", address))

        def getsockname(self):
            return ("127.0.0.1", 43123)

    selected = select_port(
        "localhost", 0, socket_factory=lambda *args: FakeSocket(),
    )

    assert selected == 43123
    assert calls == ["enter", ("bind", ("127.0.0.1", 0)), "close"]


def test_localhost_is_normalized_before_selection_output_and_uvicorn(tmp_path: Path) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    environ = _environment(tmp_path, repo)
    args = _args(repo, tools, host="localhost")
    assert parse_settings(args, environ=environ).host == "127.0.0.1"
    output = io.StringIO()

    def run_uvicorn(app, *, host: str, port: int, reload: bool) -> None:
        assert (host, port, reload) == ("127.0.0.1", 56590, False)
        assert json.loads(output.getvalue())["url"].startswith("http://127.0.0.1:")

    assert main(
        args, environ=environ, stdout=output, uvicorn_run=run_uvicorn,
    ) == 0


def test_foreground_launch_injects_complete_status_reuses_session_and_leaks_no_credentials(
    tmp_path: Path,
) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    environ = _environment(tmp_path, repo)
    observed_sessions: list[str] = []
    startup_records: list[dict[str, object]] = []

    for _ in range(2):
        output = io.StringIO()

        def run_uvicorn(app, *, host: str, port: int, reload: bool) -> None:
            assert (host, port, reload) == ("127.0.0.1", 56590, False)
            lines = output.getvalue().splitlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            startup_records.append(record)
            key = parse_qs(urlparse(str(record["url"])).query)["key"][0]
            with TestClient(app) as client:
                assert client.get(f"/?key={key}", follow_redirects=False).status_code == 303
                bootstrap = client.get("/api/bootstrap").json()
            assert bootstrap["fable_ready"] is True
            assert bootstrap["fable_status"] == "subscription_ready"
            assert bootstrap["sol_status"] == "ready"
            assert bootstrap["repository"] == str(repo)
            assert bootstrap["branch"] == "feat/test-launcher"
            observed_sessions.append(bootstrap["session_id"])

        assert main(
            _args(repo, tools), environ=environ, stdout=output,
            uvicorn_run=run_uvicorn,
        ) == 0

    assert observed_sessions[0] == observed_sessions[1]
    for record in startup_records:
        assert record["port"] == 56590
        assert record["fable_status"] == "subscription_ready"
        assert record["sol_version"] == "codex-cli 9.9.9"
        assert record["ssh_command"] == (
            "ssh -N -L 56590:127.0.0.1:56590 bridgeuser@198.51.100.20"
        )
        serialized = json.dumps(record, sort_keys=True)
        for secret in (
            environ["ANTHROPIC_API_KEY"],
            environ["ANTHROPIC_AUTH_TOKEN"],
            environ["CLAUDE_CODE_OAUTH_TOKEN"],
            environ["OPENAI_API_KEY"],
            environ["AWS_SECRET_ACCESS_KEY"],
            environ["AZURE_CLIENT_SECRET"],
            environ["GOOGLE_APPLICATION_CREDENTIALS"],
            environ["DATABASE_URL"],
            environ["SSH_AUTH_SOCK"],
            environ["PASSWORD"],
        ):
            assert secret not in serialized
    state_dir = tmp_path / "state" / "agent-bridge" / "repo"
    assert stat.S_IMODE((state_dir / "bridge.sqlite3").stat().st_mode) == 0o600
    calls = [
        json.loads(line)
        for line in (tmp_path / "commands.jsonl").read_text().splitlines()
    ]
    model_calls = [call for call in calls if call["tool"] in {"claude", "codex"}]
    assert [call["argv"] for call in model_calls] == [
        ["--version"], ["auth", "status", "--json"], ["--version"],
        ["--version"], ["auth", "status", "--json"], ["--version"],
    ]
    claude_forbidden = {
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
    }
    assert all(
        claude_forbidden.isdisjoint(call["environment"])
        for call in model_calls
        if call["tool"] == "claude"
    )
    expected_codex_environment = {
        key: environ[key]
        for key in (
            "HOME", "CODEX_HOME", "XDG_CONFIG_HOME", "PATH", "LANG", "USER",
        )
    }
    assert all(
        call["environment"] == expected_codex_environment
        for call in model_calls
        if call["tool"] == "codex"
    )


def test_single_instance_lock_prevents_second_recovery_and_releases_after_exit(
    tmp_path: Path, valid_brief,
) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    environ = _environment(tmp_path, repo)
    args = _args(repo, tools)
    state_dir = tmp_path / "state" / "agent-bridge" / "repo"
    inner_attempted = False

    def first_uvicorn(app, *, host: str, port: int, reload: bool) -> None:
        nonlocal inner_attempted
        database = state_dir / "bridge.sqlite3"
        side_store = SQLiteStore(database)
        side_store.create_session("recovery-session", str(repo))
        side_store.save_task("recovery-session", valid_brief, TaskState.SOL_RUNNING)
        side_store.close()
        before = database.read_bytes()

        with pytest.raises(ValueError, match="already running"):
            main(
                args,
                environ=environ,
                stdout=io.StringIO(),
                uvicorn_run=lambda *args, **kwargs: None,
            )
        inner_attempted = True
        assert database.read_bytes() == before
        check = SQLiteStore(database)
        assert check.get_task(valid_brief.task_id, 1).state is TaskState.SOL_RUNNING
        check.close()

    assert main(
        args, environ=environ, stdout=io.StringIO(), uvicorn_run=first_uvicorn,
    ) == 0
    assert inner_attempted is True
    lock_path = state_dir / "agent-bridge.lock"
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600

    assert main(
        args,
        environ=environ,
        stdout=io.StringIO(),
        uvicorn_run=lambda *args, **kwargs: None,
    ) == 0
    check = SQLiteStore(state_dir / "bridge.sqlite3")
    assert check.get_task(valid_brief.task_id, 1).state is TaskState.INTERRUPTED
    check.close()

    def fail_uvicorn(*args, **kwargs) -> None:
        raise RuntimeError("injected foreground failure")

    with pytest.raises(RuntimeError, match="foreground failure"):
        main(
            args,
            environ=environ,
            stdout=io.StringIO(),
            uvicorn_run=fail_uvicorn,
        )
    assert main(
        args,
        environ=environ,
        stdout=io.StringIO(),
        uvicorn_run=lambda *args, **kwargs: None,
    ) == 0


@pytest.mark.parametrize(
    ("probe", "expected_fable", "expected_sol"),
    (
        ("claude-version", "subscription_unavailable", "ready"),
        ("claude-auth", "subscription_unavailable", "ready"),
        ("codex-version", "subscription_ready", "unavailable"),
    ),
)
def test_preflight_deadline_retires_hung_fake_child_and_starts_disabled_ui(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe: str,
    expected_fable: str,
    expected_sol: str,
) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    environ = _environment(tmp_path, repo)
    (tools["claude"].parent / f"hang-{probe}").touch()
    monkeypatch.setattr(
        launcher, "PREFLIGHT_TIMEOUT_SECONDS", 0.05, raising=False,
    )
    output = io.StringIO()
    observed: list[dict[str, object]] = []
    started = time.monotonic()

    def run_uvicorn(app, *, host: str, port: int, reload: bool) -> None:
        record = json.loads(output.getvalue())
        observed.append(record)

    assert main(
        _args(repo, tools),
        environ=environ,
        stdout=output,
        uvicorn_run=run_uvicorn,
    ) == 0
    assert time.monotonic() - started < 1.0
    assert observed[0]["fable_status"] == expected_fable
    assert observed[0]["sol_status"] == expected_sol
    pid = int((tools["claude"].parent / f"hung-{probe}.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_invalid_subscription_still_starts_with_server_gate_disabled(tmp_path: Path) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    environ = {**_environment(tmp_path, repo), "AGENT_BRIDGE_TEST_AUTH": "invalid"}
    output = io.StringIO()
    observed: list[dict[str, object]] = []

    def run_uvicorn(app, *, host: str, port: int, reload: bool) -> None:
        record = json.loads(output.getvalue())
        key = parse_qs(urlparse(str(record["url"])).query)["key"][0]
        with TestClient(app) as client:
            client.get(f"/?key={key}", follow_redirects=False)
            observed.append(client.get("/api/bootstrap").json())

    assert main(
        _args(repo, tools), environ=environ, stdout=output,
        uvicorn_run=run_uvicorn,
    ) == 0

    record = json.loads(output.getvalue())
    assert record["fable_status"] == "subscription_unavailable"
    assert observed[0]["fable_ready"] is False
    assert observed[0]["fable_status"] == "subscription_unavailable"
    assert observed[0]["sol_status"] == "ready"


def test_path_resolution_is_owned_only_by_main_module() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "agent_bridge"
    python_files = tuple(root.rglob("*.py"))
    assert python_files
    for path in python_files:
        if path.name != "__main__.py":
            assert "shutil.which(" not in path.read_text(encoding="utf-8")
