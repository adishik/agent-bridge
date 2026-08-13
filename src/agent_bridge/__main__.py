"""Foreground launcher for the loopback-only local agent bridge."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
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
from typing import TYPE_CHECKING, Protocol, TextIO

from agent_bridge.projects import (
    ProjectSpec,
    build_project_specs,
    parse_project_argument,
    project_id_for_root,
)

if TYPE_CHECKING:
    from agent_bridge.coordinator import IdFactory
    from agent_bridge.hub import InstanceLock, OwnedProjectRuntime


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


@dataclass(frozen=True, slots=True)
class Settings:
    projects: tuple[ProjectSpec, ...]
    hub_state_dir: Path
    host: str
    port: int
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


@dataclass(slots=True)
class _StateDirectory:
    """A retained no-follow directory authority for startup-owned state."""

    path: Path
    descriptor: int | None

    def child_path(self, name: str) -> Path:
        if self.descriptor is None:
            raise RuntimeError("state directory authority is closed")
        if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("state child name is invalid")
        return Path(f"/proc/self/fd/{self.descriptor}/{name}")

    def close(self) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


@dataclass(slots=True)
class _OpenedProjectState:
    """One migrated project database held until every legacy audit completes."""

    spec: ProjectSpec
    lock: InstanceLock
    store: object
    artifacts: Path
    schemas: Path
    state_directory: _StateDirectory
    artifact_descriptor: int | None
    schema_descriptor: int | None

    def release_state_authority(self) -> None:
        if self.artifact_descriptor is not None:
            os.close(self.artifact_descriptor)
            self.artifact_descriptor = None
        if self.schema_descriptor is not None:
            os.close(self.schema_descriptor)
            self.schema_descriptor = None
        self.state_directory.close()


class _Ids:
    def new_task_id(self) -> str:
        return f"task-{secrets.token_hex(16)}"

    def new_run_id(self) -> str:
        return f"run-{secrets.token_hex(16)}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local Agent Bridge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    projects = parser.add_mutually_exclusive_group(required=True)
    projects.add_argument(
        "--repo",
        help=(
            "Single repository authority for this run; it becomes a "
            "restart-required immutable allowlist entry."
        ),
    )
    projects.add_argument(
        "--project",
        action="append",
        metavar="LABEL=/ABSOLUTE/REPOSITORY",
        help=(
            "Repeatable restart-required immutable allowlist entry. "
            "Labels are display-only."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Loopback-only host, normalized to 127.0.0.1.",
    )
    parser.add_argument(
        "--port", type=int, default=56590, help="Loopback listener port.",
    )
    parser.add_argument(
        "--claude-executable",
        help="Claude executable; must be an absolute executable path.",
    )
    parser.add_argument(
        "--codex-executable",
        help="Codex executable; must be an absolute executable path.",
    )
    parser.add_argument(
        "--git-executable",
        help="Git executable; must be an absolute executable path.",
    )
    parser.add_argument(
        "--bash-executable",
        help="Bash executable; must be an absolute executable path.",
    )
    parser.add_argument(
        "--sh-executable",
        help="sh executable; must be an absolute executable path.",
    )
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
    xdg_text = environment.get("XDG_STATE_HOME")
    if xdg_text:
        state_home = Path(xdg_text)
        if not state_home.is_absolute():
            raise ValueError("XDG_STATE_HOME must be an absolute path")
    else:
        state_home = Path.home() / ".local" / "state"
    state_root = Path(os.path.abspath(state_home / "agent-bridge"))
    entries = _project_entries(arguments)
    candidate_specs = _candidate_project_specs(entries, state_root)
    legacy_paths = _resolve_legacy_paths(candidate_specs, state_root)
    preliminary_settings = Settings(
        projects=candidate_specs,
        hub_state_dir=state_root / "hub",
        host="127.0.0.1",
        port=arguments.port,
        claude_executable=claude,
        codex_executable=codex,
        git_executable=git,
        bash_executable=bash,
        sh_executable=sh,
    )
    # Validate every selected state target before startup opens descriptor-
    # anchored directories for the first write.
    preliminary_candidates = (
        preliminary_settings.hub_state_dir,
        state_root / "projects",
        *(
            legacy_paths.get(spec.project_id, spec.state_dir)
            for spec in candidate_specs
        ),
    )
    for candidate in preliminary_candidates:
        _validate_state_dir(preliminary_settings, candidate=candidate)
    try:
        specs = build_project_specs(
            entries,
            state_root=state_root,
            git_executable=git,
            create_state_dirs=False,
        )
    except ValueError as error:
        if str(error) == "Git top-level probe failed":
            raise ValueError("repository must be a readable Git repository") from error
        if str(error) == "Git top-level does not match the canonical project root":
            raise ValueError("repository must equal its Git top level") from error
        raise
    selected: list[ProjectSpec] = []
    for spec in specs:
        legacy_path = legacy_paths.get(spec.project_id)
        if legacy_path is None:
            selected.append(spec)
            continue
        selected.append(replace(spec, state_dir=legacy_path))
    return Settings(
        projects=tuple(selected),
        hub_state_dir=state_root / "hub",
        host="127.0.0.1",
        port=arguments.port,
        claude_executable=claude,
        codex_executable=codex,
        git_executable=git,
        bash_executable=bash,
        sh_executable=sh,
    )


def _project_entries(arguments: argparse.Namespace) -> tuple[tuple[str, Path], ...]:
    if arguments.repo is not None:
        raw_root = Path(arguments.repo)
        if not raw_root.is_absolute():
            raw_root = Path(os.path.abspath(raw_root))
        return ((_repository_slug(raw_root), raw_root),)
    values = tuple(arguments.project or ())
    try:
        return tuple(parse_project_argument(value) for value in values)
    except ValueError:
        raise


def _candidate_project_specs(
    entries: Sequence[tuple[str, Path]], state_root: Path,
) -> tuple[ProjectSpec, ...]:
    """Resolve the immutable authorities needed for no-write state validation."""
    candidates: list[ProjectSpec] = []
    roots_by_project_id: dict[str, Path] = {}
    for label, root in entries:
        try:
            canonical_root = root.resolve(strict=True)
        except OSError as error:
            raise ValueError("repository must be an existing directory") from error
        if not canonical_root.is_dir():
            raise ValueError("repository must be an existing directory")
        project_id = project_id_for_root(canonical_root)
        claimed_root = roots_by_project_id.get(project_id)
        if claimed_root is not None and claimed_root != canonical_root:
            raise ValueError("project identity collision")
        roots_by_project_id[project_id] = canonical_root
        candidates.append(ProjectSpec(
            project_id=project_id,
            label=label,
            repo_root=canonical_root,
            branch="pending",
            state_dir=state_root / "projects" / project_id,
        ))
    return tuple(sorted(candidates, key=lambda spec: spec.project_id))


def _resolve_legacy_paths(
    specs: Sequence[ProjectSpec], state_root: Path,
) -> dict[str, Path]:
    """Select auditable basename state without opening a database."""
    candidates: dict[Path, list[Path]] = {}
    selected: dict[str, Path] = {}
    for spec in specs:
        legacy_path = state_root / _repository_slug(spec.repo_root)
        digest_path = state_root / "projects" / spec.project_id
        legacy_exists = legacy_path.exists()
        digest_exists = digest_path.exists()
        if legacy_exists and digest_exists:
            raise ValueError("legacy and digest project state are ambiguous")
        if legacy_exists:
            candidates.setdefault(legacy_path, []).append(spec.repo_root)
            selected[spec.project_id] = legacy_path
    for roots in candidates.values():
        if len(roots) > 1:
            raise ValueError("multiple configured roots claim one legacy state directory")
    return selected


def _validate_state_dir(settings: Settings, *, candidate: Path) -> Path:
    """Check a state target without creating it or following symlinked parents."""
    if not isinstance(settings, Settings):
        raise ValueError("settings must be Settings")
    state_dir = candidate
    if not isinstance(state_dir, Path) or not state_dir.is_absolute():
        raise ValueError("state directory must be an absolute path")
    roots = tuple(spec.repo_root for spec in settings.projects)
    if not roots:
        raise ValueError("settings must contain at least one project")
    if any(state_dir.is_relative_to(root) for root in roots):
        raise ValueError("state directory must remain outside repository")
    ancestor = state_dir
    while True:
        if ancestor.is_symlink():
            raise ValueError("state directory must not have a symlinked ancestor")
        if ancestor == ancestor.parent:
            break
        ancestor = ancestor.parent
    resolved_candidate = state_dir.resolve(strict=False)
    if any(resolved_candidate.is_relative_to(root) for root in roots):
        raise ValueError("state directory must remain outside repository")
    return state_dir


def _open_nofollow_directory(path: Path, *, create: bool) -> int:
    """Open an absolute directory by descriptor without following ancestors."""
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
    ):
        raise ValueError("state directory must support absolute no-follow access")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_state_directory(settings: Settings, *, candidate: Path) -> _StateDirectory:
    """Validate, create, and retain one state target through one descriptor."""
    state_dir = _validate_state_dir(settings, candidate=candidate)
    descriptor: int | None = None
    try:
        descriptor = _open_nofollow_directory(state_dir, create=True)
        entry = os.fstat(descriptor)
        if not stat.S_ISDIR(entry.st_mode) or entry.st_uid != os.geteuid():
            raise ValueError("state directory must be an owner-controlled directory")
        os.fchmod(descriptor, 0o700)
        return _StateDirectory(path=state_dir, descriptor=descriptor)
    except ValueError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ValueError("state directory must be a safe writable directory") from error


def prepare_state_dir(settings: Settings, *, candidate: Path | None = None) -> Path:
    """Create one external, non-symlinked, owner-only runtime directory."""
    state_dir = settings.hub_state_dir if candidate is None else candidate
    authority = _open_state_directory(settings, candidate=state_dir)
    authority.close()
    return state_dir


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


def _secure_regular_file(path: Path, *, directory_fd: int | None = None) -> None:
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        if directory_fd is None:
            descriptor = os.open(path, flags, 0o600)
        else:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
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


def acquire_instance_lock(path: Path, *, directory_fd: int | None = None) -> InstanceLock:
    """Acquire one nonblocking process-lifetime lock at its explicit path."""
    from agent_bridge.hub import InstanceLock

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("instance lock path must be absolute")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        if directory_fd is None:
            descriptor = os.open(path, flags, 0o600)
        else:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=directory_fd)
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
        return InstanceLock(path=path, descriptor=descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _open_private_directory(path: Path, *, directory_fd: int | None = None) -> int:
    """Create/open one owner-only state child under retained authority."""
    descriptor: int | None = None
    try:
        if directory_fd is None:
            descriptor = _open_nofollow_directory(path, create=True)
        else:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            try:
                os.mkdir(path.name, mode=0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            descriptor = os.open(path.name, flags, dir_fd=directory_fd)
        entry = os.fstat(descriptor)
        if not stat.S_ISDIR(entry.st_mode) or entry.st_uid != os.geteuid():
            raise ValueError("runtime subdirectory must be an owner-controlled directory")
        os.fchmod(descriptor, 0o700)
        return descriptor
    except ValueError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ValueError("runtime subdirectory must be writable") from error


def _prepare_private_directory(path: Path, *, directory_fd: int | None = None) -> Path:
    descriptor: int | None = None
    try:
        descriptor = _open_private_directory(path, directory_fd=directory_fd)
        return path
    finally:
        if descriptor is not None:
            os.close(descriptor)


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
    spec: ProjectSpec,
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
                cwd=spec.repo_root,
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
                cwd=spec.repo_root,
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


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _store_clock(clock: Callable[[], datetime]) -> Callable[[], str]:
    def timestamp() -> str:
        value = clock()
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.isoformat()
    return timestamp


async def _fresh_fable_probe(fable: object) -> tuple[bool, str]:
    from agent_bridge.adapters.claude_cli import SubscriptionAuthError

    try:
        await asyncio.wait_for(fable.preflight(), timeout=PREFLIGHT_TIMEOUT_SECONDS)
    except (OSError, RuntimeError, SubscriptionAuthError, TimeoutError):
        return False, "subscription_unavailable"
    return True, "subscription_ready"


async def _fresh_sol_probe(
    *, runner: object, settings: Settings, spec: ProjectSpec, environment: Mapping[str, str],
) -> str:
    try:
        version = await asyncio.wait_for(
            _version(
                runner=runner,
                run_id=f"readiness-codex-version-{spec.project_id}",
                executable=settings.codex_executable,
                cwd=spec.repo_root,
                environment=environment,
            ),
            timeout=PREFLIGHT_TIMEOUT_SECONDS,
        )
    except (OSError, RuntimeError, TimeoutError):
        return "unavailable"
    return "ready" if version is not None else "unavailable"


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


def _is_legacy_state(spec: ProjectSpec, settings: Settings) -> bool:
    return spec.state_dir == (
        settings.hub_state_dir.parent / _repository_slug(spec.repo_root)
    )


def _open_project_state(
    spec: ProjectSpec,
    *,
    lock: InstanceLock,
    clock: Callable[[], datetime],
    state_directory: _StateDirectory | None = None,
) -> _OpenedProjectState:
    """Open and migrate one project database without recovering work."""
    from agent_bridge.store import SQLiteStore

    owns_state_directory = state_directory is None
    database_path = spec.state_dir / "bridge.sqlite3"
    store = None
    artifact_descriptor: int | None = None
    schema_descriptor: int | None = None
    try:
        if state_directory is None:
            descriptor = _open_nofollow_directory(spec.state_dir, create=True)
            state_directory = _StateDirectory(spec.state_dir, descriptor)
            os.fchmod(descriptor, 0o700)
        if state_directory.descriptor is None:
            raise RuntimeError("state directory authority is closed")
        _secure_regular_file(database_path, directory_fd=state_directory.descriptor)
        artifact_descriptor = _open_private_directory(
            spec.state_dir / "artifacts", directory_fd=state_directory.descriptor,
        )
        schema_descriptor = _open_private_directory(
            spec.state_dir / "schemas", directory_fd=state_directory.descriptor,
        )
        store = SQLiteStore(
            state_directory.child_path("bridge.sqlite3"),
            clock=_store_clock(clock),
            check_same_thread=False,
        )
        return _OpenedProjectState(
            spec=spec,
            lock=lock,
            store=store,
            artifacts=spec.state_dir / "artifacts",
            schemas=Path(f"/proc/self/fd/{schema_descriptor}"),
            state_directory=state_directory,
            artifact_descriptor=artifact_descriptor,
            schema_descriptor=schema_descriptor,
        )
    except BaseException:
        if store is not None:
            try:
                store.close()
            except BaseException:
                pass
        if artifact_descriptor is not None:
            os.close(artifact_descriptor)
        if schema_descriptor is not None:
            os.close(schema_descriptor)
        if owns_state_directory and state_directory is not None:
            state_directory.close()
        raise


def _audit_project_state(opened: _OpenedProjectState, settings: Settings) -> None:
    """Perform the exact legacy-root audit before any project may recover."""
    if _is_legacy_state(opened.spec, settings):
        opened.store.audit_legacy_project_ownership(str(opened.spec.repo_root))


def assemble_project_runtime(
    spec: ProjectSpec,
    *,
    lock: InstanceLock,
    settings: Settings,
    environment: Mapping[str, str],
    ids: IdFactory,
    clock: Callable[[], datetime],
    _opened_state: _OpenedProjectState | None = None,
) -> OwnedProjectRuntime:
    """Build one fully isolated project runtime after the legacy audit phase."""
    from agent_bridge.adapters.claude_cli import ClaudeCLI, SubscriptionAuthError
    from agent_bridge.adapters.codex_cli import CodexCLI
    from agent_bridge.app import InMemoryEventBroadcaster
    from agent_bridge.coordinator import Coordinator
    from agent_bridge.hub import OwnedProjectRuntime, RuntimeReadiness, RuntimeStatus
    from agent_bridge.process import ProcessRunner
    from agent_bridge.repository import RepositoryTracker

    if not isinstance(spec, ProjectSpec):
        raise ValueError("spec must be a ProjectSpec")
    if not isinstance(settings, Settings):
        raise ValueError("settings must be Settings")
    opened = _opened_state
    opened_here = opened is None
    tracker = None
    try:
        if opened is None:
            opened = _open_project_state(spec, lock=lock, clock=clock)
            _audit_project_state(opened, settings)
        elif opened.spec != spec or opened.lock is not lock:
            raise ValueError("opened project state does not match runtime inputs")
        opened.store.recover_active_tasks()
        runner = ProcessRunner()
        codex_child_environment = codex_environment(environment)
        readiness_holder: list[RuntimeReadiness] = []

        class _ReadinessBoundClaudeCLI(ClaudeCLI):
            async def _run_contract(self, **kwargs: object) -> object:
                try:
                    return await super()._run_contract(**kwargs)
                except SubscriptionAuthError:
                    if readiness_holder:
                        readiness_holder[0].invalidate_fable_subscription()
                    raise

        fable = _ReadinessBoundClaudeCLI(
            settings.claude_executable,
            runner,
            env=environment,
            cwd=spec.repo_root,
        )
        sol = CodexCLI(
            settings.codex_executable,
            runner,
            repo_root=spec.repo_root,
            schema_dir=opened.schemas,
            env=codex_child_environment,
            schema_directory_fd=opened.schema_descriptor,
        )
        preflight = asyncio.run(_run_preflights(
            runner=runner,
            fable=fable,
            settings=settings,
            spec=spec,
            claude_source_environment=environment,
            codex_child_environment=codex_child_environment,
        ))
        readiness = RuntimeReadiness(
            initial=RuntimeStatus(
                preflight.fable_ready,
                preflight.fable_status,
                preflight.sol_status,
            ),
            fable_probe=lambda: _fresh_fable_probe(fable),
            sol_probe=lambda: _fresh_sol_probe(
                runner=runner,
                settings=settings,
                spec=spec,
                environment=codex_child_environment,
            ),
            timeout_seconds=PREFLIGHT_TIMEOUT_SECONDS,
        )
        readiness_holder.append(readiness)
        # The bootstrap screen displays this one bounded startup snapshot;
        # model starts always use the fresh readiness probes above.
        readiness._startup_preflight = preflight  # type: ignore[attr-defined]
        tracker = RepositoryTracker(
            spec.repo_root,
            opened.artifacts,
            git_executable=settings.git_executable,
            artifact_directory_fd=opened.artifact_descriptor,
        )
        coordinator = Coordinator(
            store=opened.store,
            repository=tracker,
            runner=runner,
            fable=fable,
            sol=sol,
            ids=ids,
            repo_root=spec.repo_root,
            repo_context=read_repository_context(spec.repo_root),
            trusted_shells={
                "bash": settings.bash_executable,
                "sh": settings.sh_executable,
            },
        )
        return OwnedProjectRuntime(
            spec=spec,
            store=opened.store,
            tracker=tracker,
            runner=runner,
            fable=fable,
            sol=sol,
            coordinator=coordinator,
            broadcaster=InMemoryEventBroadcaster(),
            readiness=readiness,
            lock=lock,
            state_authority_close=opened.release_state_authority,
        )
    except BaseException:
        if tracker is not None:
            try:
                tracker.close()
            except BaseException:
                pass
        if opened_here and opened is not None:
            try:
                opened.store.close()
            except BaseException:
                pass
            opened.release_state_authority()
        raise


def _release_locks(locks: Sequence[object]) -> None:
    for lock in reversed(tuple(locks)):
        try:
            lock.release()
        except BaseException:
            continue


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
    ordered_specs = tuple(sorted(settings.projects, key=lambda spec: spec.project_id))
    state_authorities: dict[Path, _StateDirectory] = {}
    locks: list[object] = []
    try:
        for state_dir in (settings.hub_state_dir, *(spec.state_dir for spec in ordered_specs)):
            prepare_state_dir(settings, candidate=state_dir)
            state_authorities[state_dir] = _open_state_directory(
                settings, candidate=state_dir,
            )
        hub_lock = acquire_instance_lock(
            settings.hub_state_dir / "agent-bridge.lock",
            directory_fd=state_authorities[settings.hub_state_dir].descriptor,
        )
        locks.append(hub_lock)
        project_locks: dict[str, object] = {}
        for spec in ordered_specs:
            lock = acquire_instance_lock(
                spec.state_dir / "agent-bridge.lock",
                directory_fd=state_authorities[spec.state_dir].descriptor,
            )
            locks.append(lock)
            project_locks[spec.project_id] = lock
    except BaseException:
        _release_locks(locks)
        for authority in state_authorities.values():
            authority.close()
        raise

    hub_store = None
    runtimes: list[object] = []
    opened_states: list[_OpenedProjectState] = []
    registry = None
    owned_project_ids: set[str] = set()
    try:
        from agent_bridge.app import create_hub_app
        from agent_bridge.hub import ActiveAgentLease, HubWorkflowOrchestrator, ProjectRegistry
        from agent_bridge.hub_store import HubStore

        hub_database = settings.hub_state_dir / "hub.sqlite3"
        hub_authority = state_authorities[settings.hub_state_dir]
        _secure_regular_file(hub_database, directory_fd=hub_authority.descriptor)
        hub_store = HubStore(hub_authority.child_path("hub.sqlite3"), clock=_now)
        for spec in ordered_specs:
            opened_states.append(_open_project_state(
                spec,
                lock=project_locks[spec.project_id],
                clock=_now,
                state_directory=state_authorities[spec.state_dir],
            ))
        for opened in opened_states:
            _audit_project_state(opened, settings)
        for opened in opened_states:
            spec = opened.spec
            runtime = assemble_project_runtime(
                spec,
                lock=project_locks[spec.project_id],
                settings=settings,
                environment=environment,
                ids=_Ids(),
                clock=_now,
                _opened_state=opened,
            )
            runtimes.append(runtime)
            owned_project_ids.add(spec.project_id)
        registry = ProjectRegistry(runtimes)
        # This binds the durable account acknowledgement to the hub rather
        # than copying the former project-local setting into any project DB.
        orchestrator = HubWorkflowOrchestrator(
            registry=registry,
            lease=ActiveAgentLease(),
            usage_credits_acknowledged=hub_store.usage_credits_acknowledged,
        )
        primary = runtimes[0]
        _active_session(primary.store, primary.spec.repo_root)
        preflight = primary.readiness._startup_preflight
        session_key = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        app = create_hub_app(
            registry=registry,
            hub_store=hub_store,
            workflows=orchestrator,
            static_dir=Path(__file__).resolve().parent / "static",
            session_key=session_key,
            csrf_token=csrf_token,
        )
        app.state.launcher_settings = settings
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
            "repository": str(primary.spec.repo_root),
            "branch": primary.spec.branch,
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
        if registry is not None:
            try:
                registry.close()
            except BaseException:
                pass
        else:
            for runtime in reversed(runtimes):
                try:
                    runtime.close()
                except BaseException:
                    pass
        for opened in reversed(opened_states):
            if opened.spec.project_id not in owned_project_ids:
                try:
                    opened.store.close()
                except BaseException:
                    pass
            opened.release_state_authority()
        if hub_store is not None:
            try:
                hub_store.close()
            except BaseException:
                pass
        _release_locks((
            hub_lock,
            *(
                project_locks[spec.project_id]
                for spec in ordered_specs
                if spec.project_id not in owned_project_ids
            ),
        ))
        for authority in state_authorities.values():
            authority.close()


if __name__ == "__main__":
    raise SystemExit(main())
