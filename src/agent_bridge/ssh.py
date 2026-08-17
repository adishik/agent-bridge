"""Bounded local inputs for a future SSH-backed Agent Bridge runner."""

from __future__ import annotations

import argparse
import base64
import codecs
import hashlib
import http.cookiejar
import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shlex
import shutil
import stat
import subprocess
import time
import socket
import sys
from typing import TextIO
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

from agent_bridge.projects import MAX_PROJECTS, parse_project_argument


_MAX_DESTINATION_LENGTH = 255
_RELEASE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+!-]{0,63}", re.ASCII)
_MAX_BOOTSTRAP_SOURCE_BYTES = 128 * 1024
_MAX_BOOTSTRAP_OUTPUT_BYTES = 64 * 1024
_MAX_BOOTSTRAP_PYTHON_BYTES = 4096
_MAX_DIAGNOSTIC_BYTES = 4096
_BOOTSTRAP_TEARDOWN_HEADROOM_SECONDS = 60.0
_BOOTSTRAP_TIMEOUT_SECONDS = 690.0
_STARTUP_TIMEOUT_SECONDS = 30.0
_STARTUP_GIT_PROBES_PER_PROJECT = 2
_STARTUP_PROVIDER_PREFLIGHTS_PER_PROJECT = 3
_STARTUP_COMMAND_TIMEOUT_SECONDS = 10.0
_STARTUP_PROVIDER_CANCELLATION_SECONDS = 5.0
_STARTUP_TIMEOUT_PER_PROJECT_SECONDS = (
    _STARTUP_GIT_PROBES_PER_PROJECT * _STARTUP_COMMAND_TIMEOUT_SECONDS
    + _STARTUP_PROVIDER_PREFLIGHTS_PER_PROJECT
    * (_STARTUP_COMMAND_TIMEOUT_SECONDS + _STARTUP_PROVIDER_CANCELLATION_SECONDS)
)
_STARTUP_NON_SUBPROCESS_OVERHEAD_PER_PROJECT_SECONDS = 10.0
_STARTUP_GLOBAL_OVERHEAD_SECONDS = 10.0
_MAX_PRESTARTUP_LOG_BYTES = 64 * 1024
_MAX_JSON_FIELD_BYTES = 256
_MAX_JSON_DELIMITER_BYTES = 64
_BOOTSTRAP_FIELDS = frozenset({"protocol", "python", "remote_port", "version"})

MAX_STARTUP_BYTES = 64 * 1024
STARTUP_FIELDS = frozenset({
    "port",
    "url",
    "fable_status",
    "fable_version",
    "sol_status",
    "sol_version",
    "ssh_command",
    "repository",
    "branch",
})
_MAX_STARTUP_STRING_BYTES = 4096
_MAX_FAILURE_DRAIN_PASSES = 32
_MAX_TUNNEL_DIAGNOSTIC_CAPTURE_BYTES = _MAX_DIAGNOSTIC_BYTES + _MAX_STARTUP_STRING_BYTES


class SSHLaunchError(RuntimeError):
    """A bounded actionable failure from the SSH connection workflow."""


class _RetryablePortCollision(SSHLaunchError):
    """An automatically selected listener or forward collided before readiness."""

    def __init__(self, message: str, *, endpoint: str) -> None:
        if endpoint not in {"local", "remote"}:
            raise ValueError("collision endpoint must be local or remote")
        super().__init__(message)
        self.endpoint = endpoint


@dataclass(frozen=True, slots=True)
class SSHSettings:
    ssh_executable: Path
    destination: str
    remote_arguments: tuple[str, ...]
    local_port: int
    remote_port: int
    python_command: str
    open_browser: bool
    version: str


@dataclass(frozen=True, slots=True)
class BootstrapRecord:
    protocol: int
    python: str
    remote_port: int
    version: str


def parse_ssh_settings(
    argv: Sequence[str] | None = None,
    *,
    executable_finder: Callable[[str], str | None] = shutil.which,
    distribution_reader: Callable[[str], importlib.metadata.Distribution] = (
        importlib.metadata.distribution
    ),
) -> SSHSettings:
    arguments = _ssh_parser().parse_args(argv)
    destination = _validate_destination(arguments.destination)
    ssh_executable = _resolve_local_ssh(executable_finder("ssh"))
    try:
        distribution = distribution_reader("agent-bridge")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(
            "SSH requires an installed published package-index release; "
            "source-checkout-only installs are not accepted"
        ) from error
    version = _installed_release_version(distribution)
    remote_arguments = _remote_bridge_arguments(arguments)
    return SSHSettings(
        ssh_executable=ssh_executable,
        destination=destination,
        remote_arguments=remote_arguments,
        local_port=_validate_port(arguments.local_port, label="local port"),
        remote_port=_validate_port(arguments.remote_port, label="remote port"),
        python_command=_validate_remote_command(arguments.python),
        open_browser=not arguments.no_open,
        version=version,
    )


def _ssh_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("destination")
    projects = parser.add_mutually_exclusive_group(required=True)
    projects.add_argument("--repo")
    projects.add_argument("--project", action="append")
    parser.add_argument("--local-port", default=0)
    parser.add_argument("--remote-port", default=0)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--claude-executable")
    parser.add_argument("--codex-executable")
    parser.add_argument("--git-executable")
    parser.add_argument("--bash-executable")
    parser.add_argument("--sh-executable")
    return parser


def _validate_destination(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_DESTINATION_LENGTH
        or value.startswith("-")
        or any(character.isspace() or _is_control_character(character) for character in value)
    ):
        raise ValueError("destination must be a safe SSH alias or user@host")
    return value


def _resolve_local_ssh(candidate: str | None) -> Path:
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("SSH executable was not found")
    try:
        executable = Path(candidate).resolve(strict=True)
        entry = executable.stat()
    except (OSError, ValueError) as error:
        raise ValueError("SSH executable must be an absolute regular executable") from error
    if not executable.is_absolute() or not stat.S_ISREG(entry.st_mode) or not os.access(
        executable, os.X_OK
    ):
        raise ValueError("SSH executable must be an absolute regular executable")
    return executable


def _installed_release_version(distribution: importlib.metadata.Distribution) -> str:
    try:
        name = distribution.metadata["Name"]
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("installed distribution metadata must name agent-bridge") from error
    if name != "agent-bridge":
        raise ValueError("installed distribution metadata must name agent-bridge")
    version = distribution.version
    if not isinstance(version, str) or _RELEASE_VERSION.fullmatch(version) is None:
        raise ValueError("installed distribution version is not a bounded release token")
    try:
        direct_url = distribution.read_text("direct_url.json")
    except (AttributeError, OSError) as error:
        raise ValueError("installed distribution provenance could not be read") from error
    if direct_url is not None:
        if not isinstance(direct_url, str):
            raise ValueError("installed distribution provenance is malformed")
        try:
            json.loads(direct_url)
        except json.JSONDecodeError as error:
            raise ValueError("installed distribution provenance is malformed") from error
        raise ValueError("installed distribution provenance must be a package-index release")
    _validate_distribution_binding(distribution)
    return version


def _validate_distribution_binding(distribution: importlib.metadata.Distribution) -> None:
    package_root = Path(__file__).resolve().parent
    try:
        records = distribution.files
    except (AttributeError, OSError) as error:
        raise ValueError("installed distribution RECORD could not be read") from error
    if not records:
        raise ValueError("installed distribution RECORD is missing")
    recorded_package_files: set[str] = set()
    for record in records:
        relative = str(record)
        package_relative = _package_record_relative(relative)
        if package_relative is None:
            continue
        package_parts = PurePosixPath(package_relative).parts[1:]
        generated_bytecode = (
            "__pycache__" in package_parts
            and PurePosixPath(package_relative).suffix == ".pyc"
        )
        try:
            installed_path = Path(distribution.locate_file(record)).resolve(strict=True)
        except (AttributeError, OSError, RuntimeError, ValueError) as error:
            raise ValueError("installed distribution does not match the imported package") from error
        if package_relative in recorded_package_files:
            raise ValueError("installed distribution RECORD contains a duplicate package file")
        expected_path = package_root.joinpath(*PurePosixPath(package_relative).parts[1:])
        try:
            actual_path = expected_path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError("installed distribution does not match the imported package") from error
        if (
            installed_path != actual_path
            or not _is_within(package_root, actual_path)
            or expected_path.is_symlink()
        ):
            raise ValueError("installed distribution does not match the imported package")
        if not generated_bytecode:
            _validate_record_integrity(actual_path, record)
        recorded_package_files.add(package_relative)

    _validate_recorded_package_tree(package_root, recorded_package_files)


def _package_record_relative(relative: str) -> str | None:
    """Return a canonical agent_bridge RECORD entry, or None for dist metadata."""
    if not relative.startswith("agent_bridge/"):
        return None
    if "\\" in relative:
        raise ValueError("installed distribution RECORD package path is invalid")
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or str(path) != relative
        or len(path.parts) < 2
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("installed distribution RECORD package path is invalid")
    return relative


def _is_within(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_recorded_package_tree(package_root: Path, recorded: set[str]) -> None:
    """Require RECORD integrity for every runtime-importable package file."""
    try:
        entries = sorted(package_root.rglob("*"))
    except OSError as error:
        raise ValueError("installed distribution package tree could not be read") from error
    for path in entries:
        relative_path = path.relative_to(package_root)
        relative = relative_path.as_posix()
        try:
            entry = path.lstat()
        except OSError as error:
            raise ValueError("installed distribution package tree could not be read") from error
        if "__pycache__" in relative_path.parts:
            if stat.S_ISDIR(entry.st_mode):
                continue
            if stat.S_ISREG(entry.st_mode) and path.suffix == ".pyc":
                continue
            raise ValueError("installed distribution package cache entry is unsafe")
        if stat.S_ISDIR(entry.st_mode):
            continue
        if not stat.S_ISREG(entry.st_mode):
            raise ValueError("installed distribution package file is unsafe")
        if f"agent_bridge/{relative}" not in recorded:
            raise ValueError("installed distribution RECORD is missing a package file")


def _validate_record_integrity(path: Path, record: object) -> None:
    size = getattr(record, "size", None)
    try:
        entry = path.lstat()
    except OSError as error:
        raise ValueError("installed distribution RECORD package file could not be read") from error
    if (
        not stat.S_ISREG(entry.st_mode)
        or type(size) is not int
        or entry.st_size != size
    ):
        raise ValueError("installed distribution RECORD size does not match the imported package")
    hash_info = getattr(record, "hash", None)
    if hash_info is None:
        raise ValueError("installed distribution RECORD hash is missing")
    mode = getattr(hash_info, "mode", None)
    expected = getattr(hash_info, "value", None)
    if (
        not isinstance(mode, str)
        or mode not in {"sha256", "sha384", "sha512"}
        or not isinstance(expected, str)
    ):
        raise ValueError("installed distribution RECORD hash is invalid")
    try:
        digest = hashlib.new(mode, path.read_bytes()).digest()
    except (OSError, ValueError) as error:
        raise ValueError("installed distribution RECORD hash is invalid") from error
    actual = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    if actual != expected:
        raise ValueError("installed distribution RECORD hash does not match the imported package")


def _remote_bridge_arguments(arguments: argparse.Namespace) -> tuple[str, ...]:
    remote_arguments: list[str]
    if arguments.repo is not None:
        remote_arguments = ["--repo", _validate_remote_path(arguments.repo, label="repository")]
    else:
        labels: set[str] = set()
        remote_arguments = []
        for value in arguments.project or ():
            label, _ = parse_project_argument(value)
            if label.casefold() in labels:
                raise ValueError("duplicate project label")
            labels.add(label.casefold())
            _, root = value.split("=", 1)
            _validate_remote_path(root, label="project repository")
            remote_arguments.extend(("--project", value))
    if arguments.claude_executable is not None:
        remote_arguments.extend((
            "--claude-executable",
            _validate_remote_path(arguments.claude_executable, label="Claude executable"),
        ))
    if arguments.codex_executable is not None:
        remote_arguments.extend((
            "--codex-executable",
            _validate_remote_path(arguments.codex_executable, label="Codex executable"),
        ))
    for option, value, label in (
        ("--git-executable", arguments.git_executable, "Git executable"),
        ("--bash-executable", arguments.bash_executable, "Bash executable"),
        ("--sh-executable", arguments.sh_executable, "sh executable"),
    ):
        if value is not None:
            remote_arguments.extend((option, _validate_remote_path(value, label=label)))
    return tuple(remote_arguments)


def _startup_timeout_seconds(settings: SSHSettings) -> float:
    """Budget all bounded sequential remote startup work before its record."""
    if settings.remote_arguments[:1] == ("--repo",):
        projects = 1
    else:
        projects = sum(
            option == "--project" for option in settings.remote_arguments
        )
    if not 1 <= projects <= MAX_PROJECTS:
        raise SSHLaunchError(f"too many projects; maximum is {MAX_PROJECTS}")
    return (
        projects
        * (
            _STARTUP_TIMEOUT_PER_PROJECT_SECONDS
            + _STARTUP_NON_SUBPROCESS_OVERHEAD_PER_PROJECT_SECONDS
        )
        + _STARTUP_GLOBAL_OVERHEAD_SECONDS
    )


def _validate_port(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 65535")
    if isinstance(value, int):
        port = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value, re.ASCII):
        port = int(value)
    else:
        raise ValueError(f"{label} must be between 0 and 65535")
    if not 0 <= port <= 65535:
        raise ValueError(f"{label} must be between 0 and 65535")
    return port


def _validate_remote_command(value: object) -> str:
    if value == "python3":
        return value
    return _validate_remote_path(value, label="Python command")


def _validate_remote_path(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(_is_control_character(character) for character in value)
        or not PurePosixPath(value).is_absolute()
    ):
        raise ValueError(f"remote {label} must be a safe absolute POSIX path")
    return value


def _is_control_character(value: str) -> bool:
    return ord(value) < 32 or ord(value) == 127


def build_bootstrap_argv(
    settings: SSHSettings,
    *,
    bootstrap_source: str,
) -> tuple[str, ...]:
    """Build the one fixed OpenSSH bootstrap invocation."""
    remote = shlex.join((
        settings.python_command,
        "-I",
        "-c",
        bootstrap_source,
        settings.version,
        str(settings.remote_port),
    ))
    return (str(settings.ssh_executable), "-T", "--", settings.destination, remote)


def run_remote_bootstrap(
    settings: SSHSettings,
    *,
    source_reader: Callable[[], str] | None = None,
    process_runner: Callable[[tuple[str, ...]], subprocess.CompletedProcess[bytes]] | None = None,
) -> BootstrapRecord:
    """Bootstrap the exact remote release through one bounded SSH command."""
    try:
        source = (source_reader or _read_bootstrap_source)()
    except (OSError, ValueError, UnicodeError) as error:
        raise SSHLaunchError("could not read the local SSH bootstrap source") from error
    _validate_bootstrap_source(source)
    argv = build_bootstrap_argv(settings, bootstrap_source=source)
    try:
        result = (process_runner or _run_bounded_ssh)(argv)
    except subprocess.TimeoutExpired as error:
        raise SSHLaunchError("SSH bootstrap timed out") from error
    except OSError as error:
        raise SSHLaunchError("could not start SSH bootstrap") from error
    if result.returncode != 0:
        raise SSHLaunchError(
            "SSH bootstrap failed" + _diagnostic_suffix(result.stdout, result.stderr)
        )
    return _parse_bootstrap_record(result.stdout, settings)


def _read_bootstrap_source() -> str:
    path = Path(__file__).with_name("_remote_bootstrap.py")
    entry = path.lstat()
    if not stat.S_ISREG(entry.st_mode) or stat.S_ISLNK(entry.st_mode):
        raise ValueError("bootstrap source is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("bootstrap source cannot be opened safely")
    descriptor = os.open(path, flags | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != entry.st_dev
            or opened.st_ino != entry.st_ino
            or opened.st_size > _MAX_BOOTSTRAP_SOURCE_BYTES
        ):
            raise ValueError("bootstrap source is not a bounded regular file")
        content = bytearray()
        while len(content) <= _MAX_BOOTSTRAP_SOURCE_BYTES:
            chunk = os.read(descriptor, _MAX_BOOTSTRAP_SOURCE_BYTES + 1 - len(content))
            if not chunk:
                break
            content.extend(chunk)
        if len(content) > _MAX_BOOTSTRAP_SOURCE_BYTES:
            raise ValueError("bootstrap source exceeds its size limit")
    finally:
        os.close(descriptor)
    return bytes(content).decode("utf-8")


def _validate_bootstrap_source(source: object) -> None:
    if not isinstance(source, str) or len(source.encode("utf-8")) > _MAX_BOOTSTRAP_SOURCE_BYTES:
        raise SSHLaunchError("local SSH bootstrap source is invalid")


def _run_bounded_ssh(argv: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    selector: selectors.BaseSelector | None = None
    output: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    try:
        if process.stdout is None or process.stderr is None:
            raise SSHLaunchError("SSH bootstrap could not capture its output")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        deadline = time.monotonic() + _BOOTSTRAP_TIMEOUT_SECONDS
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            events = selector.select(remaining)
            if not events:
                raise TimeoutError
            for key, _ in events:
                stream = output[key.data]
                chunk = os.read(key.fd, _MAX_BOOTSTRAP_OUTPUT_BYTES - len(stream) + 1)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                stream.extend(chunk)
                if len(stream) > _MAX_BOOTSTRAP_OUTPUT_BYTES:
                    raise OverflowError
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        returncode = process.wait(timeout=remaining)
    except TimeoutError as error:
        _stop_process(process)
        raise SSHLaunchError(
            "SSH bootstrap timed out" + _diagnostic_suffix(output["stdout"], output["stderr"])
        ) from error
    except OverflowError as error:
        _stop_process(process)
        raise SSHLaunchError(
            "SSH bootstrap output exceeded its limit"
            + _diagnostic_suffix(output["stdout"], output["stderr"])
        ) from error
    except BaseException:
        _stop_process(process)
        raise
    finally:
        if selector is not None:
            selector.close()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
    return subprocess.CompletedProcess(
        argv,
        returncode,
        bytes(output["stdout"]),
        bytes(output["stderr"]),
    )


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        process.wait()
        return
    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _diagnostic_suffix(stdout: bytes | bytearray, stderr: bytes | bytearray) -> str:
    if not stdout and not stderr:
        return ""
    redacted = _redact_diagnostic(bytes(stdout) + bytes(stderr))
    return f": {redacted[-_MAX_DIAGNOSTIC_BYTES:]}"


def _redact_diagnostic(diagnostic: bytes) -> str:
    text = diagnostic.decode("utf-8", errors="replace")
    redacted = re.sub(
        r"(?i)([?&]key=)[^&#\s'\"<>]+",
        r"\1<redacted>",
        text,
    )
    redacted = re.sub(
        r'(?i)("(?:key|access_key)"\s*:\s*")[^"]*',
        r"\1<redacted>",
        redacted,
    )
    return redacted


def _parse_bootstrap_record(stdout: bytes, settings: SSHSettings) -> BootstrapRecord:
    if (
        len(stdout) > _MAX_BOOTSTRAP_OUTPUT_BYTES
        or not stdout.endswith(b"\n")
        or stdout.count(b"\n") != 1
    ):
        raise SSHLaunchError("remote bootstrap record is invalid")
    try:
        record = json.loads(stdout[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SSHLaunchError("remote bootstrap record is invalid") from error
    if not isinstance(record, dict) or set(record) != _BOOTSTRAP_FIELDS:
        raise SSHLaunchError("remote bootstrap record has an invalid shape")
    protocol = record["protocol"]
    python = record["python"]
    remote_port = record["remote_port"]
    version = record["version"]
    if not isinstance(protocol, int) or isinstance(protocol, bool) or protocol != 1:
        raise SSHLaunchError("remote bootstrap protocol is invalid")
    if not isinstance(version, str) or version != settings.version:
        raise SSHLaunchError("remote bootstrap version is invalid")
    if not isinstance(python, str):
        raise SSHLaunchError("remote bootstrap python is invalid")
    try:
        encoded_python = python.encode("utf-8")
    except UnicodeError as error:
        raise SSHLaunchError("remote bootstrap python is invalid") from error
    if (
        not python
        or len(encoded_python) > _MAX_BOOTSTRAP_PYTHON_BYTES
        or any(_is_control_character(character) for character in python)
        or not PurePosixPath(python).is_absolute()
    ):
        raise SSHLaunchError("remote bootstrap python is invalid")
    if (
        not isinstance(remote_port, int)
        or isinstance(remote_port, bool)
        or not 1 <= remote_port <= 65535
    ):
        raise SSHLaunchError("remote bootstrap port is invalid")
    return BootstrapRecord(
        protocol=protocol,
        python=python,
        remote_port=remote_port,
        version=version,
    )


def build_tunnel_argv(
    settings: SSHSettings,
    bootstrap: BootstrapRecord,
    *,
    local_port: int,
) -> tuple[str, ...]:
    """Build the foreground SSH tunnel with loopback-only endpoints."""
    _require_selected_port(local_port, label="local port")
    _require_selected_port(bootstrap.remote_port, label="remote port")
    runtime_argv = (
        bootstrap.python,
        "-I",
        "-m",
        "agent_bridge",
        *settings.remote_arguments,
        "--host",
        "127.0.0.1",
        "--port",
        str(bootstrap.remote_port),
    )
    # The remote shell receives a single, quoted command argument.  Merge the
    # application streams there so they retain their source order; OpenSSH's
    # own diagnostics remain on this process's stderr.
    remote_command = "exec " + shlex.join(runtime_argv) + " 2>&1"
    forward = f"127.0.0.1:{local_port}:127.0.0.1:{bootstrap.remote_port}"
    return (
        str(settings.ssh_executable),
        "-T",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        forward,
        "--",
        settings.destination,
        remote_command,
    )


def _require_selected_port(port: object, *, label: str) -> None:
    if (
        not isinstance(port, int)
        or isinstance(port, bool)
        or not 1 <= port <= 65535
    ):
        raise SSHLaunchError(f"{label} is invalid")


def parse_remote_startup(
    line: bytes,
    *,
    expected_remote_port: int,
) -> Mapping[str, object]:
    if len(line) > MAX_STARTUP_BYTES or not line.endswith(b"\n"):
        raise SSHLaunchError("remote startup record is invalid")
    try:
        record = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SSHLaunchError("remote startup record is invalid") from error
    if not isinstance(record, dict) or set(record) != STARTUP_FIELDS:
        raise SSHLaunchError("remote startup record has an invalid shape")
    _validate_remote_startup_fields(record, expected_remote_port)
    return record


def _validate_remote_startup_fields(
    record: Mapping[str, object], expected_remote_port: int
) -> None:
    _require_selected_port(expected_remote_port, label="expected remote port")
    if (
        not isinstance(record["port"], int)
        or isinstance(record["port"], bool)
        or record["port"] != expected_remote_port
    ):
        raise SSHLaunchError("remote startup port is invalid")
    _startup_key(record["url"], expected_remote_port=expected_remote_port)
    for name in ("fable_status", "sol_status", "repository", "branch"):
        _require_bounded_startup_string(record[name], allow_none=False)
    for name in ("fable_version", "sol_version", "ssh_command"):
        _require_bounded_startup_string(record[name], allow_none=True)


def _require_bounded_startup_string(value: object, *, allow_none: bool) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, str):
        raise SSHLaunchError("remote startup record has invalid fields")
    try:
        encoded_value = value.encode("utf-8")
    except UnicodeError as error:
        raise SSHLaunchError("remote startup record has invalid fields") from error
    if (
        not value
        or len(encoded_value) > _MAX_STARTUP_STRING_BYTES
        or any(_is_control_character(character) for character in value)
    ):
        raise SSHLaunchError("remote startup record has invalid fields")


def _startup_key(raw_url: object, *, expected_remote_port: int | None = None) -> str:
    if not isinstance(raw_url, str):
        raise SSHLaunchError("remote startup URL is invalid")
    if any(_is_control_character(character) for character in raw_url):
        raise SSHLaunchError("remote startup URL is invalid")
    try:
        encoded_url = raw_url.encode("utf-8")
    except UnicodeError as error:
        raise SSHLaunchError("remote startup URL is invalid") from error
    if len(encoded_url) > _MAX_STARTUP_STRING_BYTES:
        raise SSHLaunchError("remote startup URL is invalid")
    try:
        parsed = urllib.parse.urlsplit(raw_url)
        query = urllib.parse.parse_qs(parsed.query, strict_parsing=True, keep_blank_values=True)
        port = parsed.port
    except ValueError as error:
        raise SSHLaunchError("remote startup URL is invalid") from error
    values = query.get("key", ())
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/"
        or parsed.fragment
        or set(query) != {"key"}
        or len(values) != 1
        or re.fullmatch(r"[A-Za-z0-9_-]{16,128}", values[0]) is None
        or (expected_remote_port is not None and port != expected_remote_port)
        or port is None
        or not 1 <= port <= 65535
    ):
        raise SSHLaunchError("remote startup URL is invalid")
    if raw_url != f"http://127.0.0.1:{port}/?key={values[0]}":
        raise SSHLaunchError("remote startup URL is invalid")
    return values[0]


def localized_startup(
    remote: Mapping[str, object],
    *,
    local_port: int,
) -> dict[str, object]:
    _require_selected_port(local_port, label="local port")
    try:
        key = _startup_key(remote["url"])
    except KeyError as error:
        raise SSHLaunchError("remote startup URL is invalid") from error
    localized = dict(remote)
    localized["port"] = local_port
    localized["url"] = f"http://127.0.0.1:{local_port}/?key={key}"
    return localized


def wait_for_readiness(
    url: str,
    *,
    process: subprocess.Popen[bytes],
    opener: Callable[..., object] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    timeout_seconds: float = 30.0,
    progress: Callable[[], None] | None = None,
) -> None:
    request_opener = _loopback_readiness_opener().open if opener is None else opener
    deadline = monotonic() + timeout_seconds
    while True:
        if progress is not None:
            progress()
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        if process.poll() is not None:
            raise SSHLaunchError("SSH exited before Agent Bridge was ready")
        try:
            with request_opener(url, timeout=min(1.0, remaining)) as response:
                final_url = response.geturl() if hasattr(response, "geturl") else None
                if response.status == 200 and final_url in (None, url):
                    if progress is not None:
                        progress()
                    if deadline - monotonic() <= 0:
                        break
                    if process.poll() is not None:
                        raise SSHLaunchError("SSH exited before Agent Bridge was ready")
                    return
                raise SSHLaunchError("forwarded Agent Bridge returned an invalid status")
        except urllib.error.HTTPError as error:
            if error.code == 403:
                raise SSHLaunchError("forwarded Agent Bridge rejected its access key") from error
            if _is_keyed_root_admission(error):
                if progress is not None:
                    progress()
                if deadline - monotonic() <= 0:
                    break
                if process.poll() is not None:
                    raise SSHLaunchError("SSH exited before Agent Bridge was ready")
                return
            raise SSHLaunchError("forwarded Agent Bridge returned an invalid status") from error
        except (TimeoutError, urllib.error.URLError, ConnectionError):
            if progress is not None:
                progress()
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            if process.poll() is not None:
                raise SSHLaunchError("SSH exited before Agent Bridge was ready")
            sleeper(min(0.1, remaining))
    raise SSHLaunchError("Agent Bridge did not become ready before the deadline")


def _is_keyed_root_admission(error: urllib.error.HTTPError) -> bool:
    headers = error.headers
    return (
        error.code == 303
        and headers is not None
        and headers.get("Location") == "/"
        and (headers.get("Set-Cookie") or "").startswith("agent_bridge_session=")
    )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


def _loopback_readiness_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
        _NoRedirectHandler(),
    )


def _select_local_port(
    requested: object,
    *,
    socket_factory: Callable[..., socket.socket] = socket.socket,
) -> int:
    """Reserve an IPv4 loopback port long enough to choose a tunnel endpoint."""
    requested_port = _validate_port(requested, label="local port")
    if requested_port:
        return requested_port
    try:
        with socket_factory(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.bind(("127.0.0.1", 0))
            selected = candidate.getsockname()[1]
    except OSError as error:
        raise SSHLaunchError("could not select a local loopback port") from error
    _require_selected_port(selected, label="local port")
    return selected


def _bounded_tail(value: bytearray, chunk: bytes) -> None:
    value.extend(chunk)
    if len(value) > _MAX_DIAGNOSTIC_BYTES:
        del value[:-_MAX_DIAGNOSTIC_BYTES]


class _BoundedDiagnosticOutput:
    def __init__(self) -> None:
        self._value = bytearray()

    def write(self, value: str) -> int:
        self._value.extend(value.encode("utf-8", errors="replace"))
        if len(self._value) > _MAX_TUNNEL_DIAGNOSTIC_CAPTURE_BYTES:
            del self._value[:-_MAX_TUNNEL_DIAGNOSTIC_CAPTURE_BYTES]
        return len(value)

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        return bytes(self._value).decode("utf-8", errors="ignore")


def _diagnostic_tail(value: str, maximum: int) -> str:
    if maximum <= 0:
        return ""
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= maximum:
        return value
    return encoded[-maximum:].decode("utf-8", errors="ignore")


def _render_sanitized_diagnostics(sources: Mapping[str, str]) -> str:
    entries = [
        (label, sources.get(stream, ""))
        for stream, label in (("stdout", "remote output"), ("stderr", "SSH diagnostics"))
        if sources.get(stream, "")
    ]
    if not entries:
        return ""
    labels_size = sum(len(f"{label}: ".encode("utf-8")) for label, unused in entries)
    separator_size = len(" | ".encode("utf-8")) * (len(entries) - 1)
    remaining = max(0, _MAX_DIAGNOSTIC_BYTES - labels_size - separator_size)
    lengths = [len(value.encode("utf-8", errors="replace")) for unused, value in entries]
    allocations = [0] * len(entries)
    pending = set(range(len(entries)))
    while pending and remaining:
        share = remaining // len(pending)
        if not share:
            for index in sorted(pending):
                if not remaining:
                    break
                allocations[index] += 1
                remaining -= 1
            break
        used = 0
        for index in pending:
            added = min(lengths[index] - allocations[index], share)
            allocations[index] += added
            used += added
        if not used:
            break
        remaining -= used
        pending = {index for index in pending if allocations[index] < lengths[index]}
    rendered = " | ".join(
        f"{label}: {_diagnostic_tail(value, allocations[index])}"
        for index, (label, value) in enumerate(entries)
    )
    return _diagnostic_tail(rendered, _MAX_DIAGNOSTIC_BYTES)


class _TunnelDiagnostics(dict[str, bytearray]):
    """Keep raw collision tails and independently sanitized display tails."""

    def __init__(self) -> None:
        super().__init__(stdout=bytearray(), stderr=bytearray())
        self._key: str | None = None
        self._outputs: dict[str, _BoundedDiagnosticOutput] = {}
        self._writers: dict[str, _RedactedLogWriter] = {}
        self._finalized: dict[str, str] | None = None
        self.activate(None)

    def activate(
        self,
        key: str | None,
        *,
        histories: Mapping[str, bytes] | None = None,
    ) -> None:
        if self._finalized is not None:
            raise RuntimeError("tunnel diagnostics are finalized")
        self._key = key
        self._outputs = {
            stream: _BoundedDiagnosticOutput() for stream in ("stdout", "stderr")
        }
        self._writers = {
            stream: _RedactedLogWriter(self._outputs[stream], key=key)
            for stream in ("stdout", "stderr")
        }
        if histories is not None:
            for stream in ("stdout", "stderr"):
                self._writers[stream].feed(histories.get(stream, b""))

    def append(self, stream: str, chunk: bytes) -> None:
        if self._finalized is not None:
            raise RuntimeError("tunnel diagnostics are finalized")
        value = self[stream]
        value.extend(chunk)
        if len(value) > _MAX_TUNNEL_DIAGNOSTIC_CAPTURE_BYTES:
            del value[:-_MAX_TUNNEL_DIAGNOSTIC_CAPTURE_BYTES]
        self._writers[stream].feed(chunk)

    def finalize(self, *, key: str | None) -> Mapping[str, str]:
        if self._finalized is None:
            if key != self._key:
                self.activate(key, histories={stream: bytes(self[stream]) for stream in self})
            for writer in self._writers.values():
                writer.flush()
            self._finalized = {
                stream: output.getvalue() for stream, output in self._outputs.items()
            }
        elif key != self._key:
            raise RuntimeError("tunnel diagnostics are finalized for a different key")
        return self._finalized

    def render(self, *, key: str | None) -> str:
        return _render_sanitized_diagnostics(self.finalize(key=key))


def _new_tunnel_diagnostics() -> _TunnelDiagnostics:
    return _TunnelDiagnostics()


def _activate_tunnel_diagnostics(
    diagnostics: _TunnelDiagnostics,
    *,
    key: str,
    histories: Mapping[str, bytes] | None = None,
) -> None:
    diagnostics.activate(key, histories=histories)


def _append_tunnel_diagnostic(
    diagnostics: _TunnelDiagnostics,
    stream: str,
    chunk: bytes,
) -> None:
    diagnostics.append(stream, chunk)


class _RedactedLogWriter:
    """Bounded incremental sanitizer for visible remote output."""

    def __init__(self, output: TextIO, *, key: str | None) -> None:
        self._output = output
        self._key = b"" if key is None else key.encode("ascii")
        self._plain = bytearray()
        self._state = "plain"
        self._field = bytearray()
        self._escaped = False
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._decoder_finalized = False

    def feed(self, chunk: bytes) -> None:
        if self._decoder_finalized:
            raise RuntimeError("cannot write logs after final flush")
        for value in chunk:
            self._feed_byte(value)

    def flush(self) -> None:
        if self._decoder_finalized:
            return
        if self._state in {"json_name", "json_colon", "json_value_start"}:
            field = bytes(self._field)
            self._field.clear()
            self._state = "plain"
            self._replay_plain(field)
        self._state = "plain"
        self._field.clear()
        self._flush_plain(final=True)
        tail = self._decoder.decode(b"", final=True)
        self._decoder_finalized = True
        if tail:
            self._output.write(tail)
            self._output.flush()

    def _feed_byte(self, value: int) -> None:
        if self._state in {"query", "quoted_query"}:
            self._feed_query_byte(value)
            return
        if self._state == "quoted_plain":
            self._feed_quoted_plain_byte(value)
            return
        if self._state == "json_value":
            if self._escaped:
                self._escaped = False
            elif value == ord("\\"):
                self._escaped = True
            elif value == ord('"'):
                self._write_raw(b'"')
                self._state = "plain"
            return
        if self._state == "json_name":
            self._field.append(value)
            if len(self._field) > _MAX_JSON_FIELD_BYTES:
                field = bytes(self._field)
                self._field.clear()
                self._state = "plain"
                self._replay_plain(field, query_state="quoted_query")
                if self._state == "plain" and not (
                    value == ord('"') and not self._quoted_escape_state(field[:-1])
                ):
                    self._state = "quoted_plain"
                    self._escaped = self._quoted_escape_state(field)
            elif (
                value == ord('"')
                and len(self._field) > 1
                and not self._quoted_escape_state(bytes(self._field[:-1]))
            ):
                if bytes(self._field).lower() in {b'"key"', b'"access_key"'}:
                    self._write_raw(bytes(self._field))
                    self._field.clear()
                    self._state = "json_colon"
                else:
                    field = bytes(self._field)
                    self._field.clear()
                    self._state = "plain"
                    self._replay_plain(field)
            return
        if self._state == "json_colon":
            if value in b" \t\r\n":
                self._write_raw(bytes((value,)))
            elif value == ord(":"):
                self._write_raw(b":")
                self._state = "json_value_start"
            else:
                self._state = "plain"
                self._feed_plain_byte(value)
            return
        if self._state == "json_value_start":
            if value in b" \t\r\n":
                self._write_raw(bytes((value,)))
            elif value == ord('"'):
                self._write_raw(b'"<redacted>')
                self._state = "json_value"
                self._escaped = False
            else:
                self._state = "plain"
                self._feed_plain_byte(value)
            return
        if value == ord('"'):
            self._flush_plain(final=True)
            self._field = bytearray(b'"')
            self._state = "json_name"
            return
        self._feed_plain_byte(value)

    def _feed_plain_byte(self, value: int, *, query_state: str = "query") -> None:
        self._plain.append(value)
        lowered = bytes(self._plain).lower()
        for marker in (b"?key=", b"&key="):
            position = lowered.find(marker)
            if position >= 0:
                before = bytes(self._plain[:position])
                self._emit_plain(before)
                self._write_raw(bytes(self._plain[position:position + len(marker)]))
                self._write_raw(b"<redacted>")
                del self._plain[:position + len(marker)]
                self._state = query_state
                return
        self._flush_plain(final=False)

    def _feed_query_byte(self, value: int) -> None:
        quoted = self._state == "quoted_query"
        if quoted and self._escaped:
            self._escaped = False
            return
        if quoted and value == ord("\\"):
            self._escaped = True
            return
        if value in b" \t\v\f\r\n'\"<>&#":
            self._state = "quoted_plain" if quoted else "plain"
            if quoted:
                self._feed_quoted_plain_byte(value)
            else:
                self._feed_plain_byte(value)

    def _feed_quoted_plain_byte(self, value: int) -> None:
        if self._escaped:
            self._escaped = False
        elif value == ord("\\"):
            self._escaped = True
        elif value == ord('"'):
            self._feed_plain_byte(value, query_state="quoted_query")
            if self._state == "quoted_plain":
                self._state = "plain"
            return
        self._feed_plain_byte(value, query_state="quoted_query")

    @staticmethod
    def _quoted_escape_state(value: bytes) -> bool:
        escaped = False
        for item in value[1:]:
            if escaped:
                escaped = False
            elif item == ord("\\"):
                escaped = True
        return escaped

    def _replay_plain(self, value: bytes, *, query_state: str = "query") -> None:
        for item in value:
            if self._state in {"query", "quoted_query"}:
                self._feed_query_byte(item)
            elif self._state == "quoted_plain":
                self._feed_quoted_plain_byte(item)
            else:
                self._feed_plain_byte(item, query_state=query_state)

    def _flush_plain(self, *, final: bool) -> None:
        while self._plain:
            position, secret_length = self._secret_position(bytes(self._plain))
            if position >= 0:
                self._emit_plain(bytes(self._plain[:position]))
                self._write_raw(b"<redacted>")
                del self._plain[:position + secret_length]
                continue
            keep = 0 if final else max(len(self._key) - 1, len(b"?key=") - 1)
            if len(self._plain) <= keep:
                return
            release = len(self._plain) - keep
            self._emit_plain(bytes(self._plain[:release]))
            del self._plain[:release]

    def _emit_plain(self, value: bytes) -> None:
        if value:
            self._write_raw(
                value.replace(self._key, b"<redacted>") if self._key else value,
            )

    def _secret_position(self, value: bytes) -> tuple[int, int]:
        if not self._key:
            return -1, 0
        position = value.find(self._key)
        return (position, len(self._key)) if position >= 0 else (-1, 0)

    def _write_raw(self, value: bytes) -> None:
        if value:
            decoded = self._decoder.decode(value, final=False)
            if decoded:
                self._output.write(decoded)
                self._output.flush()


def _redact_tunnel_diagnostics(
    diagnostics: Mapping[str, bytes | bytearray],
    *,
    key: str | None = None,
) -> str:
    if isinstance(diagnostics, _TunnelDiagnostics):
        return diagnostics.render(key=key)
    sources: dict[str, str] = {}
    for stream in ("stdout", "stderr"):
        raw = bytes(diagnostics.get(stream, b""))
        if not raw:
            continue
        if key is None:
            redacted = _redact_diagnostic(raw)
        else:
            output = io.StringIO()
            writer = _RedactedLogWriter(output, key=key)
            writer.feed(raw)
            writer.flush()
            redacted = output.getvalue()
        if redacted:
            sources[stream] = redacted
    return _render_sanitized_diagnostics(sources)


def _collision_from_diagnostic(
    diagnostics: Mapping[str, bytes | bytearray],
    *,
    after_startup: bool,
) -> str | None:
    if isinstance(diagnostics, _TunnelDiagnostics):
        sources = diagnostics.finalize(key=diagnostics._key)
    else:
        sources = {
            stream: _redact_diagnostic(bytes(diagnostics.get(stream, b"")))
            for stream in ("stdout", "stderr")
        }

    def has_collision(stream: str) -> bool:
        text = sources.get(stream, "").lower()
        return "address already in use" in text or "cannot listen to port" in text

    if after_startup:
        return "remote" if has_collision("stdout") else None
    if has_collision("stderr"):
        return "local"
    return "remote" if has_collision("stdout") else None


def _startup_failure(
    process: subprocess.Popen[bytes],
    diagnostics: Mapping[str, bytes | bytearray],
) -> SSHLaunchError:
    suffix = _redact_tunnel_diagnostics(diagnostics)
    collision = _collision_from_diagnostic(diagnostics, after_startup=False)
    if collision is not None:
        return _RetryablePortCollision(
            "SSH tunnel listener collision" + (f": {suffix}" if suffix else ""),
            endpoint=collision,
        )
    if process.poll() is None:
        return SSHLaunchError("SSH did not provide a remote startup record")
    message = "SSH exited before Agent Bridge startup"
    if suffix:
        message += f": {suffix}"
    return SSHLaunchError(message)


def _read_startup_record(
    process: subprocess.Popen[bytes],
    selector: selectors.BaseSelector,
    diagnostics: _TunnelDiagnostics,
    *,
    expected_remote_port: int,
    monotonic: Callable[[], float],
    timeout_seconds: float = _STARTUP_TIMEOUT_SECONDS,
) -> tuple[bytes, bytes, bytes, bytes]:
    if process.stdout is None or process.stderr is None:
        raise SSHLaunchError("SSH tunnel could not capture its output")
    startup = bytearray()
    prestartup_stdout = bytearray()
    prestartup_stderr = bytearray()
    startup_remainder = bytearray()
    startup_line: bytes | None = None
    deadline = monotonic() + timeout_seconds
    while selector.get_map():
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise SSHLaunchError("SSH startup deadline exceeded")
        events = selector.select(min(0.1, remaining))
        for event, _ in events:
            while True:
                try:
                    chunk = os.read(event.fd, _MAX_DIAGNOSTIC_BYTES + 1)
                except BlockingIOError:
                    break
                if not chunk:
                    selector.unregister(event.fileobj)
                    break
                if event.data == "stderr":
                    _append_tunnel_diagnostic(diagnostics, "stderr", chunk)
                    prestartup_stderr.extend(chunk)
                    if len(prestartup_stderr) > _MAX_PRESTARTUP_LOG_BYTES:
                        raise SSHLaunchError("SSH pre-startup stderr exceeded its limit")
                    continue
                if startup_line is not None:
                    startup_remainder.extend(chunk)
                    if len(startup_remainder) > _MAX_PRESTARTUP_LOG_BYTES:
                        raise SSHLaunchError("SSH startup stdout remainder exceeded its limit")
                    _append_tunnel_diagnostic(diagnostics, "stdout", chunk)
                    continue
                startup.extend(chunk)
                while b"\n" in startup:
                    line, _, remainder = startup.partition(b"\n")
                    if len(line) + 1 > MAX_STARTUP_BYTES:
                        raise SSHLaunchError("remote startup output exceeded its limit")
                    candidate = line + b"\n"
                    del startup[:len(candidate)]
                    try:
                        parse_remote_startup(
                            candidate, expected_remote_port=expected_remote_port,
                        )
                    except SSHLaunchError:
                        prestartup_stdout.extend(candidate)
                        if len(prestartup_stdout) > _MAX_PRESTARTUP_LOG_BYTES:
                            raise SSHLaunchError("SSH pre-startup stdout exceeded its limit")
                        _append_tunnel_diagnostic(diagnostics, "stdout", candidate)
                        continue
                    startup_line = candidate
                    startup_remainder.extend(startup)
                    if len(startup_remainder) > _MAX_PRESTARTUP_LOG_BYTES:
                        raise SSHLaunchError("SSH startup stdout remainder exceeded its limit")
                    if startup:
                        _append_tunnel_diagnostic(diagnostics, "stdout", bytes(startup))
                    startup.clear()
                    break
                if startup_line is None and len(startup) > MAX_STARTUP_BYTES:
                    raise SSHLaunchError("remote startup output exceeded its limit")
        if startup_line is not None:
            while _drain_tunnel_streams(
                selector, diagnostics, log_writers=None, timeout=0.0,
                prestartup_stderr=prestartup_stderr,
                startup_remainder=startup_remainder,
            ):
                pass
            return (
                startup_line,
                bytes(prestartup_stdout),
                bytes(startup_remainder),
                bytes(prestartup_stderr),
            )
        if process.poll() is not None and not events:
            break
    if startup:
        prestartup_stdout.extend(startup)
        _append_tunnel_diagnostic(diagnostics, "stdout", bytes(startup))
    raise _startup_failure(process, diagnostics)


def _drain_tunnel_streams(
    selector: selectors.BaseSelector,
    diagnostics: _TunnelDiagnostics,
    *,
    log_writers: Mapping[str, _RedactedLogWriter] | None,
    timeout: float,
    prestartup_stderr: bytearray | None = None,
    startup_remainder: bytearray | None = None,
) -> list[tuple[selectors.SelectorKey, int]]:
    events = selector.select(timeout)
    for event, _ in events:
        try:
            chunk = os.read(event.fd, _MAX_DIAGNOSTIC_BYTES + 1)
        except BlockingIOError:
            continue
        if not chunk:
            selector.unregister(event.fileobj)
            continue
        _append_tunnel_diagnostic(diagnostics, event.data, chunk)
        if log_writers is not None:
            log_writers[event.data].feed(chunk)
        elif event.data == "stderr" and prestartup_stderr is not None:
            prestartup_stderr.extend(chunk)
            if len(prestartup_stderr) > _MAX_PRESTARTUP_LOG_BYTES:
                raise SSHLaunchError("SSH pre-startup stderr exceeded its limit")
        elif event.data == "stdout" and startup_remainder is not None:
            startup_remainder.extend(chunk)
            if len(startup_remainder) > _MAX_PRESTARTUP_LOG_BYTES:
                raise SSHLaunchError("SSH startup stdout remainder exceeded its limit")
    return events


def _wait_for_foreground_process(
    process: subprocess.Popen[bytes],
    selector: selectors.BaseSelector,
    diagnostics: _TunnelDiagnostics,
    *,
    key: str,
    log_writers: Mapping[str, _RedactedLogWriter],
) -> int:
    while selector.get_map():
        events = _drain_tunnel_streams(
            selector, diagnostics, log_writers=log_writers, timeout=0.1,
        )
        if process.poll() is not None and not events:
            break
    for log_writer in log_writers.values():
        log_writer.flush()
    returncode = process.wait()
    if returncode != 0:
        diagnostic_text = _redact_tunnel_diagnostics(diagnostics, key=key)
        message = f"SSH tunnel exited with status {returncode}"
        if diagnostic_text:
            message += f": {diagnostic_text}"
        raise SSHLaunchError(message)
    return 0


def _stop_ssh_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        process.wait()
        return
    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _close_tunnel_streams(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _run_tunnel_attempt(
    settings: SSHSettings,
    bootstrap: BootstrapRecord,
    *,
    local_port: int,
    stdout: TextIO,
    stderr: TextIO,
    browser_open: Callable[[str], bool],
    popen_factory: Callable[..., subprocess.Popen[bytes]],
    readiness_opener: Callable[..., object] | None,
) -> int:
    tunnel_argv = build_tunnel_argv(settings, bootstrap, local_port=local_port)
    try:
        process = popen_factory(
            tunnel_argv,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
    except OSError as error:
        raise SSHLaunchError("could not start the SSH tunnel") from error
    selector: selectors.BaseSelector | None = None
    waited = False
    try:
        if process.stdout is None or process.stderr is None:
            raise SSHLaunchError("SSH tunnel could not capture its output")
        try:
            os.set_blocking(process.stdout.fileno(), False)
            os.set_blocking(process.stderr.fileno(), False)
        except OSError as error:
            raise SSHLaunchError("SSH tunnel pipes could not become nonblocking") from error
        selector = selectors.DefaultSelector()
        diagnostics = _new_tunnel_diagnostics()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        startup, prestartup_stdout, startup_remainder, prestartup_stderr = _read_startup_record(
            process,
            selector,
            diagnostics,
            expected_remote_port=bootstrap.remote_port,
            monotonic=time.monotonic,
            timeout_seconds=_startup_timeout_seconds(settings),
        )
        remote = parse_remote_startup(
            startup,
            expected_remote_port=bootstrap.remote_port,
        )
        localized = localized_startup(remote, local_port=local_port)
        url = localized["url"]
        if not isinstance(url, str):
            raise SSHLaunchError("localized startup URL is invalid")
        key = _startup_key(url)
        _activate_tunnel_diagnostics(
            diagnostics,
            key=key,
            histories={
                "stdout": prestartup_stdout + startup_remainder,
                "stderr": prestartup_stderr,
            },
        )
        log_writers = {
            "stdout": _RedactedLogWriter(stderr, key=key),
            "stderr": _RedactedLogWriter(stderr, key=key),
        }
        log_writers["stdout"].feed(prestartup_stdout)
        log_writers["stderr"].feed(prestartup_stderr)
        log_writers["stdout"].feed(startup_remainder)
        try:
            wait_for_readiness(
                url,
                process=process,
                opener=readiness_opener,
                progress=lambda: _drain_tunnel_streams(
                    selector, diagnostics, log_writers=log_writers, timeout=0.0,
                ),
            )
        except SSHLaunchError as error:
            for index in range(_MAX_FAILURE_DRAIN_PASSES):
                events = _drain_tunnel_streams(
                    selector,
                    diagnostics,
                    log_writers=log_writers,
                    timeout=0.1 if index == 0 else 0.0,
                )
                if not events:
                    break
            collision = _collision_from_diagnostic(diagnostics, after_startup=True)
            if collision is not None:
                raise _RetryablePortCollision(str(error), endpoint=collision) from error
            if process.poll() is not None:
                diagnostic_text = _redact_tunnel_diagnostics(diagnostics, key=key)
                if diagnostic_text:
                    raise SSHLaunchError(f"{error}: {diagnostic_text}") from error
            raise
        stdout.write(url + "\n")
        stdout.flush()
        if settings.open_browser and not browser_open(url):
            stderr.write("agent-bridge ssh: could not open browser; use the URL above\n")
            stderr.flush()
        result = _wait_for_foreground_process(
            process, selector, diagnostics, key=key, log_writers=log_writers,
        )
        waited = True
        return result
    finally:
        try:
            if "log_writers" in locals():
                for log_writer in log_writers.values():
                    log_writer.flush()
        finally:
            try:
                if selector is not None:
                    selector.close()
            finally:
                try:
                    if not waited:
                        _stop_ssh_process(process)
                finally:
                    _close_tunnel_streams(process)


def run_ssh(
    settings: SSHSettings,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    browser_open: Callable[[str], bool] = webbrowser.open,
    bootstrap_runner: Callable[[SSHSettings], BootstrapRecord] = run_remote_bootstrap,
    popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    readiness_opener: Callable[..., object] | None = None,
) -> int:
    """Run one attached SSH tunnel, retrying only automatic listener collisions."""
    for attempt in range(5):
        bootstrap = bootstrap_runner(settings)
        local_port = _select_local_port(settings.local_port)
        try:
            return _run_tunnel_attempt(
                settings,
                bootstrap,
                local_port=local_port,
                stdout=stdout,
                stderr=stderr,
                browser_open=browser_open,
                popen_factory=popen_factory,
                readiness_opener=readiness_opener,
            )
        except _RetryablePortCollision as error:
            fixed_collision = (
                (error.endpoint == "local" and settings.local_port != 0)
                or (error.endpoint == "remote" and settings.remote_port != 0)
            )
            if fixed_collision or attempt == 4:
                raise SSHLaunchError(str(error)) from error
    raise SSHLaunchError("SSH launch exhausted its retry limit")


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Parse and run the foreground SSH command without masking programmer errors."""
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    try:
        settings = parse_ssh_settings(argv)
        return run_ssh(settings, stdout=output, stderr=errors)
    except (ValueError, SSHLaunchError) as error:
        errors.write(f"agent-bridge ssh: {error}\n")
        errors.flush()
        return 2
    except KeyboardInterrupt:
        return 130
