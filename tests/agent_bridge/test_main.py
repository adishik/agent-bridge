from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
import sqlite3
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
from agent_bridge.projects import ProjectSpec, project_id_for_root
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
elif sys.argv[-2:] == ["branch", "--show-current"] or sys.argv[-4:] == ["symbolic-ref", "--quiet", "--short", "HEAD"]:
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

    help_result = subprocess.run(
        [sys.executable, "-m", "agent_bridge", "--help"],
        cwd=source_root.parent,
        env={"PYTHONPATH": str(source_root)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    help_text = help_result.stdout.lower()
    assert "repository authority" in help_text
    assert "loopback-only" in help_text
    assert "127.0.0.1" in help_result.stdout
    assert "56590" in help_result.stdout
    for option in (
        "--claude-executable",
        "--codex-executable",
        "--git-executable",
        "--bash-executable",
        "--sh-executable",
    ):
        assert option in help_result.stdout
    normalized_help = " ".join(help_text.split())
    assert normalized_help.count("must be an absolute executable path") == 5


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

    assert state_dir == tmp_path / "state" / "agent-bridge" / "hub"
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
    state_dir = tmp_path / "state" / "agent-bridge" / "projects" / project_id_for_root(repo)
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
    state_dir = tmp_path / "state" / "agent-bridge" / "projects" / project_id_for_root(repo)
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


def _project_spec(repo: Path, label: str, state_dir: Path) -> ProjectSpec:
    return ProjectSpec(
        project_id=project_id_for_root(repo),
        label=label,
        repo_root=repo,
        branch="feat/test-launcher",
        state_dir=state_dir,
    )


def test_project_cli_requires_one_immutable_allowlist_and_uses_digest_state(
    tmp_path: Path,
) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    environment = _environment(tmp_path, repo)

    with pytest.raises(SystemExit) as missing:
        parse_settings(_args(repo, tools)[2:], environ=environment)
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as mixed:
        parse_settings(
            [
                *_args(repo, tools),
                "--project", f"second={repo}",
            ],
            environ=environment,
        )
    assert mixed.value.code == 2

    repo_settings = parse_settings(_args(repo, tools), environ=environment)
    named_settings = parse_settings(
        [
            "--project", f"renamed={repo}",
            "--claude-executable", str(tools["claude"]),
            "--codex-executable", str(tools["codex"]),
            "--git-executable", str(tools["git"]),
            "--bash-executable", str(tools["bash"]),
            "--sh-executable", str(tools["sh"]),
        ],
        environ=environment,
    )

    expected_id = project_id_for_root(repo)
    assert repo_settings.projects == (
        ProjectSpec(
            project_id=expected_id,
            label="repo",
            repo_root=repo,
            branch="feat/test-launcher",
            state_dir=tmp_path / "state" / "agent-bridge" / "projects" / expected_id,
        ),
    )
    assert named_settings.projects[0].project_id == expected_id
    assert named_settings.projects[0].state_dir == repo_settings.projects[0].state_dir
    assert named_settings.projects[0].label == "renamed"
    assert repo_settings.hub_state_dir == tmp_path / "state" / "agent-bridge" / "hub"
    help_text = launcher._parser().format_help().lower()
    assert "restart-required" in help_text
    assert "immutable allowlist" in help_text


def test_main_acquires_every_lock_before_opening_any_database_and_releases_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = _fake_tools(tmp_path)
    first_repo = _repo(tmp_path)
    second_repo = tmp_path / "second-repo"
    second_repo.mkdir()
    (second_repo / "AGENTS.md").write_text("# second\n", encoding="utf-8")
    third_repo = tmp_path / "third-repo"
    third_repo.mkdir()
    (third_repo / "AGENTS.md").write_text("# third\n", encoding="utf-8")
    state_root = tmp_path / "state" / "agent-bridge"
    specs = tuple(sorted((
        _project_spec(first_repo, "first", state_root / "projects" / project_id_for_root(first_repo)),
        ProjectSpec("2" * 32, "second", second_repo, "main", state_root / "projects" / ("2" * 32)),
        ProjectSpec("1" * 32, "third", third_repo, "main", state_root / "projects" / ("1" * 32)),
    ), key=lambda spec: spec.project_id))
    settings = launcher.Settings(
        projects=specs,
        hub_state_dir=state_root / "hub",
        host="127.0.0.1",
        port=56590,
        claude_executable=tools["claude"],
        codex_executable=tools["codex"],
        git_executable=tools["git"],
        bash_executable=tools["bash"],
        sh_executable=tools["sh"],
    )
    events: list[str] = []

    class FakeLock:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.released = False

        def release(self) -> None:
            self.released = True
            events.append(f"release:{self.path.parent.name}")

    acquired: list[FakeLock] = []

    def prepare(candidate_settings: launcher.Settings, *, candidate: Path) -> Path:
        assert candidate_settings is settings
        events.append(f"validate:{candidate.name}")
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def acquire(path: Path) -> FakeLock:
        events.append(f"lock:{path.parent.name}")
        if path.parent.name == "2" * 32:
            raise ValueError("injected second lock failure")
        lock = FakeLock(path)
        acquired.append(lock)
        return lock

    monkeypatch.setattr(launcher, "parse_settings", lambda *args, **kwargs: settings)
    monkeypatch.setattr(launcher, "prepare_state_dir", prepare)
    monkeypatch.setattr(launcher, "acquire_instance_lock", acquire)

    with pytest.raises(ValueError, match="second lock failure"):
        main([], environ=_environment(tmp_path, first_repo), uvicorn_run=lambda *args, **kwargs: None)

    assert events[:4] == [
        "validate:hub",
        *(f"validate:{spec.project_id}" for spec in specs),
    ]
    failed_at = next(index for index, spec in enumerate(specs) if spec.project_id == "2" * 32)
    assert events[4:] == [
        "lock:hub",
        *(f"lock:{spec.project_id}" for spec in specs[:failed_at + 1]),
        *(f"release:{spec.project_id}" for spec in reversed(specs[:failed_at])),
        "release:hub",
    ]
    assert all(lock.released for lock in acquired)
    assert not list(state_root.rglob("*.sqlite3"))


def test_main_closes_constructed_runtime_and_every_lock_when_later_assembly_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = _fake_tools(tmp_path)
    first_repo = _repo(tmp_path)
    second_repo = tmp_path / "second-repo"
    second_repo.mkdir()
    (second_repo / "AGENTS.md").write_text("# second\n", encoding="utf-8")
    state_root = tmp_path / "state" / "agent-bridge"
    specs = tuple(sorted((
        _project_spec(first_repo, "first", state_root / "projects" / project_id_for_root(first_repo)),
        ProjectSpec("1" * 32, "second", second_repo, "main", state_root / "projects" / ("1" * 32)),
    ), key=lambda spec: spec.project_id))
    settings = launcher.Settings(
        projects=specs,
        hub_state_dir=state_root / "hub",
        host="127.0.0.1",
        port=56590,
        claude_executable=tools["claude"],
        codex_executable=tools["codex"],
        git_executable=tools["git"],
        bash_executable=tools["bash"],
        sh_executable=tools["sh"],
    )
    events: list[str] = []

    class FakeLock:
        def __init__(self, path: Path) -> None:
            self.path = path

        def release(self) -> None:
            events.append(f"release:{self.path.parent.name}")

    class FakeHubStore:
        def __init__(self, path: Path, *, clock) -> None:
            events.append("hub-store")

        def close(self) -> None:
            events.append("hub-close")

        def usage_credits_acknowledged(self) -> bool:
            return False

    class FakeRuntime:
        def __init__(self, spec: ProjectSpec, lock: FakeLock) -> None:
            self.spec = spec
            self.lock = lock

        def close(self) -> None:
            events.append(f"runtime-close:{self.spec.project_id}")
            self.lock.release()

    def prepare(candidate_settings: launcher.Settings, *, candidate: Path) -> Path:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def assemble(spec: ProjectSpec, **kwargs: object) -> FakeRuntime:
        events.append(f"assemble:{spec.project_id}")
        if spec.project_id == specs[1].project_id:
            raise RuntimeError("injected runtime two failure")
        return FakeRuntime(spec, kwargs["lock"])

    monkeypatch.setattr(launcher, "parse_settings", lambda *args, **kwargs: settings)
    monkeypatch.setattr(launcher, "prepare_state_dir", prepare)
    monkeypatch.setattr(launcher, "acquire_instance_lock", lambda path: FakeLock(path))
    monkeypatch.setattr("agent_bridge.hub_store.HubStore", FakeHubStore)
    monkeypatch.setattr(launcher, "assemble_project_runtime", assemble)

    with pytest.raises(RuntimeError, match="runtime two failure"):
        main([], environ=_environment(tmp_path, first_repo), uvicorn_run=lambda *args, **kwargs: None)

    assert events == [
        "hub-store",
        f"assemble:{specs[0].project_id}",
        f"assemble:{specs[1].project_id}",
        f"runtime-close:{specs[0].project_id}",
        f"release:{specs[0].project_id}",
        "hub-close",
        f"release:{specs[1].project_id}",
        "release:hub",
    ]


@pytest.mark.parametrize("argument_kind", ("repo", "project"))
def test_legacy_state_is_audited_and_adopted_in_place(
    tmp_path: Path,
    argument_kind: str,
) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    environment = _environment(tmp_path, repo)
    legacy_state = tmp_path / "state" / "agent-bridge" / "repo"
    legacy_state.mkdir(parents=True)
    database = legacy_state / "bridge.sqlite3"
    store = SQLiteStore(database)
    store.create_session("session-1", str(repo))
    store.set_setting("agent_bridge.active_session_id", "session-1")
    store.close()
    args = _args(repo, tools) if argument_kind == "repo" else [
        "--project", f"chosen={repo}",
        "--claude-executable", str(tools["claude"]),
        "--codex-executable", str(tools["codex"]),
        "--git-executable", str(tools["git"]),
        "--bash-executable", str(tools["bash"]),
        "--sh-executable", str(tools["sh"]),
    ]

    assert main(args, environ=environment, stdout=io.StringIO(), uvicorn_run=lambda *args, **kwargs: None) == 0

    assert legacy_state.is_dir()
    assert database.is_file()
    assert not (tmp_path / "state" / "agent-bridge" / "projects" / project_id_for_root(repo)).exists()


@pytest.mark.parametrize(
    "corrupt",
    (
        pytest.param(
            lambda connection: connection.execute(
                "UPDATE sessions SET repo_root = '/other' WHERE session_id = 'session-1'"
            ),
            id="mixed-root",
        ),
        pytest.param(
            lambda connection: connection.execute(
                "INSERT INTO events (session_id, task_id, actor, kind, payload_json, created_at) VALUES ('missing', NULL, 'user', 'message', '{}', '2026-08-11T00:00:00Z')"
            ),
            id="orphan",
        ),
        pytest.param(
            lambda connection: connection.execute(
                "UPDATE settings SET value_json = '\"missing\"' WHERE key = 'agent_bridge.active_session_id'"
            ),
            id="invalid-active-session",
        ),
        pytest.param(
            lambda connection: connection.execute(
                "INSERT INTO settings (key, value_json) VALUES ('agent_bridge.baseline.', 'not-json')"
            ),
            id="corrupt-baseline",
        ),
        pytest.param(
            lambda connection: connection.execute(
                "INSERT INTO tasks (task_id, revision, session_id, state, correction_count) VALUES ('orphan', 1, 'missing', 'fable_planning', 0)"
            ),
            id="failed-foreign-key",
        ),
    ),
)
def test_invalid_legacy_state_aborts_before_recovery(
    tmp_path: Path,
    corrupt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    environment = _environment(tmp_path, repo)
    legacy_state = tmp_path / "state" / "agent-bridge" / "repo"
    legacy_state.mkdir(parents=True)
    database = legacy_state / "bridge.sqlite3"
    store = SQLiteStore(database)
    store.create_session("session-1", str(repo))
    store.set_setting("agent_bridge.active_session_id", "session-1")
    store.close()
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys = OFF")
    corrupt(connection)
    connection.commit()
    connection.close()
    recoveries: list[Path] = []
    original_recover = SQLiteStore.recover_active_tasks

    def recover(self: SQLiteStore):
        recoveries.append(database)
        return original_recover(self)

    monkeypatch.setattr(SQLiteStore, "recover_active_tasks", recover)

    with pytest.raises(RuntimeError, match="legacy project ownership audit failed"):
        main(_args(repo, tools), environ=environment, stdout=io.StringIO(), uvicorn_run=lambda *args, **kwargs: None)

    assert recoveries == []


def test_legacy_and_digest_state_for_one_root_is_rejected_as_ambiguous(tmp_path: Path) -> None:
    tools = _fake_tools(tmp_path)
    repo = _repo(tmp_path)
    state_root = tmp_path / "state" / "agent-bridge"
    (state_root / "repo").mkdir(parents=True)
    (state_root / "projects" / project_id_for_root(repo)).mkdir(parents=True)

    with pytest.raises(ValueError, match="ambiguous"):
        parse_settings(_args(repo, tools), environ=_environment(tmp_path, repo))


def test_two_roots_cannot_claim_one_existing_legacy_state_directory(tmp_path: Path) -> None:
    tools = _fake_tools(tmp_path)
    first_parent = tmp_path / "first"
    second_parent = tmp_path / "second"
    first_parent.mkdir()
    second_parent.mkdir()
    first = first_parent / "repo"
    second = second_parent / "repo"
    first.mkdir()
    second.mkdir()
    (first / "AGENTS.md").write_text("# first\n", encoding="utf-8")
    (second / "AGENTS.md").write_text("# second\n", encoding="utf-8")
    legacy = tmp_path / "state" / "agent-bridge" / "repo"
    legacy.mkdir(parents=True)
    arguments = [
        "--project", f"first={first}",
        "--project", f"second={second}",
        "--claude-executable", str(tools["claude"]),
        "--codex-executable", str(tools["codex"]),
        "--git-executable", str(tools["git"]),
        "--bash-executable", str(tools["bash"]),
        "--sh-executable", str(tools["sh"]),
    ]

    with pytest.raises(ValueError, match="legacy"):
        parse_settings(arguments, environ=_environment(tmp_path, first))
