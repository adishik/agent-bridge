"""Foreground launcher for the loopback-only local agent bridge."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
from typing import Protocol, TextIO


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
_SAFE_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}\Z")
_SAFE_SLUG_CHAR = re.compile(r"[^a-z0-9._-]+")
_ACTIVE_SESSION_SETTING = "agent_bridge.active_session_id"
MAX_REPO_CONTEXT_BYTES = 256 * 1024
PREFLIGHT_TIMEOUT_SECONDS = 10.0
_CODEX_ENVIRONMENT_KEYS = (
    "HOME",
    "CODEX_HOME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "USER",
    "LOGNAME",
    "VIRTUAL_ENV",
)
_GIT_ENVIRONMENT = {
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


class UvicornRun(Protocol):
    def __call__(self, app: object, *, host: str, port: int, reload: bool) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    branch: str
    host: str
    port: int
    state_dir: Path
    claude_executable: Path
    codex_executable: Path
    git_executable: Path
    bash_executable: Path
    sh_executable: Path


@dataclass(frozen=True)
class _PreflightStatus:
    fable_version: str | None
    fable_ready: bool
    fable_status: str
    sol_version: str | None
    sol_status: str


class _Ids:
    def new_task_id(self) -> str:
        return f"task-{secrets.token_hex(16)}"

    def new_run_id(self) -> str:
        return f"run-{secrets.token_hex(16)}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Agent Bridge")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=56590)
    parser.add_argument("--claude-executable")
    parser.add_argument("--codex-executable")
    parser.add_argument("--git-executable")
    parser.add_argument("--bash-executable")
    parser.add_argument("--sh-executable")
    return parser


def _resolve_executable(
    configured: str | None,
    *,
    command: str,
    label: str,
) -> Path:
    candidate = configured if configured is not None else shutil.which(command)
    if candidate is None:
        raise ValueError(f"{label} executable was not found")
    raw_path = Path(candidate)
    if not raw_path.is_absolute():
        raise ValueError(f"{label} executable must be an absolute path")
    try:
        path = raw_path.resolve(strict=True)
        entry = path.stat()
    except OSError as error:
        raise ValueError(f"{label} executable must be an executable file") from error
    if not stat.S_ISREG(entry.st_mode) or not os.access(path, os.X_OK):
        raise ValueError(f"{label} executable must be an executable file")
    return path


def _run_git(
    git_executable: Path,
    repo_root: Path,
    arguments: Sequence[str],
) -> str:
    command = (
        str(git_executable),
        "--no-pager",
        "-c", f"core.excludesFile={os.devnull}",
        "-c", "core.fsmonitor=false",
        "-c", "credential.helper=",
        "-c", "core.hooksPath=/dev/null",
        *arguments,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=dict(_GIT_ENVIRONMENT),
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("repository must be a readable Git repository") from error
    if completed.returncode != 0:
        raise ValueError("repository must be a readable Git repository")
    return completed.stdout.strip()


def _repository_slug(repo_root: Path) -> str:
    slug = _SAFE_SLUG_CHAR.sub("-", repo_root.name.lower()).strip("-.")
    return slug or "repository"


def codex_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Return only cached-auth and local-tool essentials for Codex children."""
    if not isinstance(source, Mapping):
        raise ValueError("source environment must be a mapping")
    environment = {
        key: value
        for key in _CODEX_ENVIRONMENT_KEYS
        if isinstance((value := source.get(key)), str)
    }
    raw_path = environment.get("PATH", os.defpath)
    safe_path = tuple(
        component
        for component in raw_path.split(os.pathsep)
        if component
        and Path(component).is_absolute()
        and "\n" not in component
        and "\r" not in component
    )
    environment["PATH"] = os.pathsep.join(dict.fromkeys(safe_path)) or os.defpath
    return environment


def _context_fallback(repo_root: Path) -> str:
    return f"Repository root: {repo_root}"


def read_repository_context(repo_root: Path) -> str:
    """Read AGENTS.md through authenticated descriptors without following links."""
    fallback = _context_fallback(repo_root)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        return fallback
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | close_on_exec
    file_flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_NONBLOCK", 0)
        | close_on_exec
    )
    root_descriptor: int | None = None
    descriptor: int | None = None
    try:
        root_descriptor = os.open(repo_root, root_flags)
        before = os.stat("AGENTS.md", dir_fd=root_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_REPO_CONTEXT_BYTES:
            return fallback
        descriptor = os.open("AGENTS.md", file_flags, dir_fd=root_descriptor)
        opened = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if not stat.S_ISREG(opened.st_mode) or opened_identity != before_identity:
            return fallback
        chunks: list[bytes] = []
        size = 0
        while block := os.read(
            descriptor,
            min(64 * 1024, MAX_REPO_CONTEXT_BYTES - size + 1),
        ):
            size += len(block)
            if size > MAX_REPO_CONTEXT_BYTES:
                return fallback
            chunks.append(block)
        final = os.fstat(descriptor)
        current = os.stat("AGENTS.md", dir_fd=root_descriptor, follow_symlinks=False)
        final_identity = (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        )
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if final_identity != opened_identity or current_identity != opened_identity:
            return fallback
        text = b"".join(chunks).decode("utf-8", errors="replace").strip()
        return text or fallback
    except OSError:
        return fallback
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)


def parse_settings(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Parse and validate every filesystem/process boundary before web imports."""
    environment = dict(os.environ if environ is None else environ)
    arguments = _parser().parse_args(argv)
    if arguments.host not in _LOOPBACK_HOSTS:
        raise ValueError("host must be a loopback address")
    if arguments.port < 0 or arguments.port > 65535:
        raise ValueError("port must be between 0 and 65535")
    try:
        repo_root = Path(arguments.repo).resolve(strict=True)
    except OSError as error:
        raise ValueError("repository must be an existing directory") from error
    if not repo_root.is_dir():
        raise ValueError("repository must be an existing directory")

    claude = _resolve_executable(
        arguments.claude_executable, command="claude", label="Claude",
    )
    codex = _resolve_executable(
        arguments.codex_executable, command="codex", label="Codex",
    )
    git = _resolve_executable(
        arguments.git_executable, command="git", label="Git",
    )
    bash = _resolve_executable(
        arguments.bash_executable, command="bash", label="Bash",
    )
    sh = _resolve_executable(
        arguments.sh_executable, command="sh", label="sh",
    )
    top_level_text = _run_git(
        git, repo_root, ("rev-parse", "--show-toplevel"),
    )
    try:
        top_level = Path(top_level_text).resolve(strict=True)
    except OSError as error:
        raise ValueError("Git top level is not an existing directory") from error
    if top_level != repo_root:
        raise ValueError("repository must equal its Git top level")
    branch = _run_git(
        git, repo_root, ("branch", "--show-current"),
    )
    if not branch:
        branch = "detached"

    xdg_text = environment.get("XDG_STATE_HOME")
    if xdg_text:
        state_home = Path(xdg_text)
        if not state_home.is_absolute():
            raise ValueError("XDG_STATE_HOME must be an absolute path")
    else:
        state_home = Path.home() / ".local" / "state"
    state_dir = Path(os.path.abspath(state_home / "agent-bridge" / _repository_slug(repo_root)))
    return Settings(
        repo_root=repo_root,
        branch=branch,
        host="127.0.0.1",
        port=arguments.port,
        state_dir=state_dir,
        claude_executable=claude,
        codex_executable=codex,
        git_executable=git,
        bash_executable=bash,
        sh_executable=sh,
    )


def prepare_state_dir(settings: Settings) -> Path:
    """Create one external, non-symlinked, owner-only runtime directory."""
    if not isinstance(settings, Settings):
        raise ValueError("settings must be Settings")
    candidate = settings.state_dir
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_candidate.is_relative_to(settings.repo_root):
        raise ValueError("state directory must remain outside repository")
    try:
        candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError("state directory must be a safe directory")
        if candidate.resolve(strict=True).is_relative_to(settings.repo_root):
            raise ValueError("state directory must remain outside repository")
        candidate.chmod(0o700)
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("state directory must be a safe writable directory") from error
    return candidate


def select_port(
    host: str,
    port: int,
    *,
    socket_factory: Callable[..., object] = socket.socket,
) -> int:
    """Resolve port zero with one temporary IPv4 loopback socket."""
    if host not in _LOOPBACK_HOSTS:
        raise ValueError("host must be a loopback address")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    if port:
        return port
    with socket_factory(socket.AF_INET, socket.SOCK_STREAM) as temporary:
        temporary.bind(("127.0.0.1", 0))
        selected = temporary.getsockname()[1]
    if not isinstance(selected, int) or not 1 <= selected <= 65535:
        raise RuntimeError("operating system returned an invalid loopback port")
    return selected


def ssh_forward_command(*, port: int, user: str, ssh_connection: str) -> str:
    """Build a display-only forwarding command from the current SSH session."""
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if not isinstance(user, str) or _SAFE_USER.fullmatch(user) is None:
        raise ValueError("user must be a safe local account name")
    if not isinstance(ssh_connection, str):
        raise ValueError("SSH_CONNECTION must contain four fields")
    fields = ssh_connection.split()
    if len(fields) != 4:
        raise ValueError("SSH_CONNECTION must contain four fields")
    try:
        ipaddress.ip_address(fields[0])
        server = ipaddress.ip_address(fields[2])
        client_port = int(fields[1])
        server_port = int(fields[3])
    except ValueError as error:
        raise ValueError("SSH_CONNECTION is invalid") from error
    if not 1 <= client_port <= 65535 or not 1 <= server_port <= 65535:
        raise ValueError("SSH_CONNECTION is invalid")
    server_text = f"[{server}]" if server.version == 6 else str(server)
    return f"ssh -N -L {port}:127.0.0.1:{port} {user}@{server_text}"


def _secure_regular_file(path: Path) -> None:
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            entry = os.fstat(descriptor)
            if not stat.S_ISREG(entry.st_mode):
                raise ValueError("database must be a regular file")
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("database must be a safe owner-only file") from error


def acquire_instance_lock(state_dir: Path) -> int:
    """Acquire the repository state's nonblocking process-lifetime lock."""
    path = state_dir / "agent-bridge.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        entry = os.fstat(descriptor)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_uid != os.geteuid()
            or entry.st_nlink != 1
        ):
            raise ValueError("instance lock must be an owner-controlled regular file")
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("agent bridge is already running for this repository") from error
        return descriptor
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _prepare_private_directory(path: Path) -> Path:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("runtime subdirectory must be a safe directory")
        path.chmod(0o700)
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("runtime subdirectory must be writable") from error
    return path


async def _version(
    *,
    runner: object,
    run_id: str,
    executable: Path,
    cwd: Path,
    environment: Mapping[str, str],
) -> str | None:
    result = await runner.run(
        run_id=run_id,
        argv=(str(executable), "--version"),
        cwd=cwd,
        env=environment,
        stdin=None,
        on_line=lambda stream, line: None,
    )
    if result.interrupted or result.exit_code != 0 or not result.stdout:
        return None
    version = result.stdout[0].strip()
    return version or None


async def _run_preflights(
    *,
    runner: object,
    fable: object,
    settings: Settings,
    claude_source_environment: Mapping[str, str],
    codex_child_environment: Mapping[str, str],
) -> _PreflightStatus:
    from agent_bridge.adapters.claude_cli import (
        SubscriptionAuthError,
        sanitized_claude_env,
    )

    claude_environment = sanitized_claude_env(claude_source_environment)
    try:
        fable_version = await asyncio.wait_for(
            _version(
                runner=runner,
                run_id="startup-claude-version",
                executable=settings.claude_executable,
                cwd=settings.repo_root,
                environment=claude_environment,
            ),
            timeout=PREFLIGHT_TIMEOUT_SECONDS,
        )
    except (OSError, RuntimeError, TimeoutError):
        fable_version = None
    try:
        await asyncio.wait_for(
            fable.preflight(),
            timeout=PREFLIGHT_TIMEOUT_SECONDS,
        )
        subscription_ready = True
    except (OSError, RuntimeError, SubscriptionAuthError, TimeoutError):
        subscription_ready = False
    try:
        sol_version = await asyncio.wait_for(
            _version(
                runner=runner,
                run_id="startup-codex-version",
                executable=settings.codex_executable,
                cwd=settings.repo_root,
                environment=codex_child_environment,
            ),
            timeout=PREFLIGHT_TIMEOUT_SECONDS,
        )
    except (OSError, RuntimeError, TimeoutError):
        sol_version = None
    fable_ready = fable_version is not None and subscription_ready
    return _PreflightStatus(
        fable_version=fable_version,
        fable_ready=fable_ready,
        fable_status=(
            "subscription_ready" if fable_ready else "subscription_unavailable"
        ),
        sol_version=sol_version,
        sol_status="ready" if sol_version is not None else "unavailable",
    )


def _active_session(store: object, repo_root: Path) -> str:
    configured = store.get_setting(_ACTIVE_SESSION_SETTING)
    if isinstance(configured, str):
        bound_root = store.session_repo_root(configured)
        if bound_root == str(repo_root):
            return configured
    while True:
        session_id = f"session-{secrets.token_hex(16)}"
        if not store.session_exists(session_id):
            break
    store.create_session(session_id, str(repo_root))
    store.set_setting(_ACTIVE_SESSION_SETTING, session_id)
    return session_id


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    uvicorn_run: UvicornRun | None = None,
) -> int:
    """Assemble the local bridge and run Uvicorn in the foreground."""
    environment = dict(os.environ if environ is None else environ)
    output = sys.stdout if stdout is None else stdout
    settings = parse_settings(argv, environ=environment)
    state_dir = prepare_state_dir(settings)
    instance_lock = acquire_instance_lock(state_dir)
    try:
        database_path = state_dir / "bridge.sqlite3"
        _secure_regular_file(database_path)
        artifacts = _prepare_private_directory(state_dir / "artifacts")
        schemas = _prepare_private_directory(state_dir / "schemas")

        # Optional web and agent dependencies stay outside module import time.
        from agent_bridge.adapters.claude_cli import ClaudeCLI
        from agent_bridge.adapters.codex_cli import CodexCLI
        from agent_bridge.app import BootstrapStatus, create_app
        from agent_bridge.coordinator import Coordinator
        from agent_bridge.process import ProcessRunner
        from agent_bridge.repository import RepositoryTracker
        from agent_bridge.store import SQLiteStore

        store = SQLiteStore(database_path, check_same_thread=False)
    except BaseException:
        os.close(instance_lock)
        raise
    repository = None
    try:
        store.recover_active_tasks()
        session_id = _active_session(store, settings.repo_root)
        runner = ProcessRunner()
        codex_child_environment = codex_environment(environment)
        fable = ClaudeCLI(
            settings.claude_executable,
            runner,
            env=environment,
            cwd=settings.repo_root,
        )
        sol = CodexCLI(
            settings.codex_executable,
            runner,
            repo_root=settings.repo_root,
            schema_dir=schemas,
            env=codex_child_environment,
        )
        preflight = asyncio.run(_run_preflights(
            runner=runner,
            fable=fable,
            settings=settings,
            claude_source_environment=environment,
            codex_child_environment=codex_child_environment,
        ))
        repository = RepositoryTracker(
            settings.repo_root,
            artifacts,
            git_executable=settings.git_executable,
        )
        coordinator = Coordinator(
            store=store,
            repository=repository,
            runner=runner,
            fable=fable,
            sol=sol,
            ids=_Ids(),
            repo_root=settings.repo_root,
            repo_context=read_repository_context(settings.repo_root),
            trusted_shells={
                "bash": settings.bash_executable,
                "sh": settings.sh_executable,
            },
        )
        status = BootstrapStatus(
            session_id=session_id,
            fable_ready=preflight.fable_ready,
            fable_status=preflight.fable_status,
            sol_status=preflight.sol_status,
            repository=str(settings.repo_root),
            branch=settings.branch,
        )
        session_key = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        app = create_app(
            coordinator=coordinator,
            store=store,
            static_dir=Path(__file__).resolve().parent / "static",
            session_key=session_key,
            csrf_token=csrf_token,
            bootstrap_status=lambda: status,
        )
        port = select_port(settings.host, settings.port)
        public_host = "127.0.0.1"
        url = f"http://{public_host}:{port}/?key={session_key}"
        ssh_command = None
        user = environment.get("USER")
        ssh_connection = environment.get("SSH_CONNECTION")
        if user is not None and ssh_connection is not None:
            ssh_command = ssh_forward_command(
                port=port,
                user=user,
                ssh_connection=ssh_connection,
            )
        startup = {
            "port": port,
            "url": url,
            "fable_status": preflight.fable_status,
            "fable_version": preflight.fable_version,
            "sol_status": preflight.sol_status,
            "sol_version": preflight.sol_version,
            "ssh_command": ssh_command,
            "repository": str(settings.repo_root),
            "branch": settings.branch,
        }
        output.write(json.dumps(startup, separators=(",", ":"), sort_keys=True) + "\n")
        output.flush()
        if uvicorn_run is None:
            import uvicorn

            run_server = uvicorn.run
        else:
            run_server = uvicorn_run
        run_server(app, host=settings.host, port=port, reload=False)
        return 0
    finally:
        try:
            if repository is not None:
                repository.close()
        finally:
            try:
                store.close()
            finally:
                os.close(instance_lock)


if __name__ == "__main__":
    raise SystemExit(main())
