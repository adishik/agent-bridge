from __future__ import annotations

import io
import importlib.metadata
import json
import os
from email.message import Message
from dataclasses import replace
from pathlib import Path
import py_compile
import shlex
import shutil
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.error

import pytest

from agent_bridge import _remote_bootstrap as bootstrap
import agent_bridge.ssh as ssh
from agent_bridge.ssh import (
    BootstrapRecord,
    SSHLaunchError,
    build_bootstrap_argv,
    build_tunnel_argv,
    localized_startup,
    parse_remote_startup,
    parse_ssh_settings,
    run_ssh,
    run_remote_bootstrap,
    wait_for_readiness,
)


_DEFAULT_RECORDS = object()


@pytest.fixture(scope="module")
def _clean_release_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy a bytecode-free installed-package fixture once for release checks."""
    source_package = Path(ssh.__file__).resolve().parent
    package_root = tmp_path_factory.mktemp("released-agent-bridge") / "agent_bridge"
    shutil.copytree(
        source_package,
        package_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return package_root


@pytest.fixture(autouse=True)
def _use_clean_release_package(
    monkeypatch: pytest.MonkeyPatch, _clean_release_package: Path
) -> None:
    # The repository checkout may retain ignored bytecode from other test lanes;
    # release-metadata fixtures model a clean published installation instead.
    monkeypatch.setattr(ssh, "__file__", str(_clean_release_package / "ssh.py"))


class _ReleasedDistribution:
    def __init__(
        self,
        version: str,
        *,
        name: str | None = "agent-bridge",
        direct_url: str | None = None,
        install_root: Path | None = None,
        files: object = _DEFAULT_RECORDS,
    ) -> None:
        self.metadata = {} if name is None else {"Name": name}
        self.version = version
        self._direct_url = direct_url
        self._install_root = install_root or Path(ssh.__file__).resolve().parents[1]
        self.files = _release_records() if files is _DEFAULT_RECORDS else files

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return self._direct_url

    def locate_file(self, path: object) -> Path:
        return self._install_root / str(path)


class _ReleaseRecord:
    def __init__(self, relative: str, *, size: int | None, hash_info: object) -> None:
        self._relative = relative
        self.size = size
        self.hash = hash_info

    def __str__(self) -> str:
        return self._relative


def _release_records() -> tuple[_ReleaseRecord, ...]:
    records: list[_ReleaseRecord] = []
    package_root = Path(ssh.__file__).resolve().parent
    for path in sorted(package_root.rglob("*")):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        relative = path.relative_to(package_root.parent).as_posix()
        digest = ssh.base64.urlsafe_b64encode(
            ssh.hashlib.sha256(path.read_bytes()).digest()
        ).decode("ascii").rstrip("=")
        records.append(
            _ReleaseRecord(
                relative,
                size=path.stat().st_size,
                hash_info=SimpleNamespace(mode="sha256", value=digest),
            )
        )
    return tuple(records)


def _released_distribution(
    version: str,
    *,
    name: str | None = "agent-bridge",
    direct_url: str | None = None,
    install_root: Path | None = None,
    files: object = _DEFAULT_RECORDS,
) -> _ReleasedDistribution:
    return _ReleasedDistribution(
        version,
        name=name,
        direct_url=direct_url,
        install_root=install_root,
        files=files,
    )


def _write_executable(path: Path, source: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return path


def _parse(
    argv: list[str],
    *,
    ssh: Path | None,
    distribution: _ReleasedDistribution | None = None,
):
    return parse_ssh_settings(
        argv,
        executable_finder=lambda name: str(ssh) if name == "ssh" and ssh else None,
        distribution_reader=lambda name: distribution or _released_distribution("0.1.0"),
    )


def test_parse_ssh_settings_preserves_remote_authority_and_defaults(tmp_path: Path) -> None:
    ssh = _write_executable(tmp_path / "ssh", "raise SystemExit(0)\n")

    settings = parse_ssh_settings(
        ["workbox", "--repo", "/srv/demo"],
        executable_finder=lambda name: str(ssh) if name == "ssh" else None,
        distribution_reader=lambda name: _released_distribution("0.1.0"),
    )

    assert settings.destination == "workbox"
    assert settings.remote_arguments == ("--repo", "/srv/demo")
    assert settings.local_port == 0
    assert settings.remote_port == 0
    assert settings.python_command == "python3"
    assert settings.open_browser is True
    assert settings.version == "0.1.0"


def test_parse_ssh_settings_preserves_multi_project_and_remote_overrides(
    tmp_path: Path,
) -> None:
    ssh = _write_executable(tmp_path / "ssh", "raise SystemExit(0)\n")

    settings = parse_ssh_settings(
        [
            "adi@workbox",
            "--project", "app=/srv/app",
            "--project", "docs=/srv/docs with space",
            "--local-port", "43123",
            "--remote-port", "53123",
            "--python", "/opt/Python 3/bin/python3",
            "--no-open",
            "--claude-executable", "/home/adi/.local/bin/claude",
            "--codex-executable", "/home/adi/.local/bin/codex",
        ],
        executable_finder=lambda name: str(ssh),
        distribution_reader=lambda name: _released_distribution("0.1.0"),
    )

    assert settings.remote_arguments == (
        "--project", "app=/srv/app",
        "--project", "docs=/srv/docs with space",
        "--claude-executable", "/home/adi/.local/bin/claude",
        "--codex-executable", "/home/adi/.local/bin/codex",
    )
    assert settings.open_browser is False


@pytest.mark.parametrize(
    ("argv", "message"),
    (
        (["", "--repo", "/srv/demo"], "destination"),
        (["--repo", "/srv/demo", "--", "-workbox"], "destination"),
        (["work box", "--repo", "/srv/demo"], "destination"),
        (["workbox\x00", "--repo", "/srv/demo"], "destination"),
        (["w" * 256, "--repo", "/srv/demo"], "destination"),
        (["workbox", "--repo", "relative/repo"], "remote repository"),
        (["workbox", "--repo", "/srv/\x00demo"], "remote repository"),
        (["workbox", "--project", "unsafe.label=/srv/demo"], "project"),
        (
            ["workbox", "--project", "app=/srv/app", "--project", "APP=/srv/other"],
            "duplicate project label",
        ),
        (["workbox", "--repo", "/srv/demo", "--python", "python"], "remote Python"),
        (
            ["workbox", "--repo", "/srv/demo", "--claude-executable", "claude"],
            "remote Claude",
        ),
        (
            ["workbox", "--repo", "/srv/demo", "--codex-executable", "codex"],
            "remote Codex",
        ),
        (["workbox", "--repo", "/srv/demo", "--local-port", "nope"], "local port"),
        (["workbox", "--repo", "/srv/demo", "--remote-port", "65536"], "remote port"),
    ),
)
def test_parse_ssh_settings_rejects_invalid_authorities(
    tmp_path: Path,
    argv: list[str],
    message: str,
) -> None:
    # This catches accepting malformed local or remote process authorities.
    ssh = _write_executable(tmp_path / "ssh", "raise SystemExit(0)\n")

    with pytest.raises(ValueError, match=message):
        _parse(argv, ssh=ssh)


def test_parse_ssh_settings_rejects_missing_local_ssh() -> None:
    # This catches constructing a runner with no local SSH program.
    with pytest.raises(ValueError, match="SSH executable"):
        _parse(["workbox", "--repo", "/srv/demo"], ssh=None)


def test_parse_ssh_settings_translates_missing_release_metadata_to_an_actionable_error(
    tmp_path: Path,
) -> None:
    # This catches source-checkout-only metadata failures escaping the SSH launcher.
    executable = _write_executable(tmp_path / "ssh", "raise SystemExit(0)\n")

    with pytest.raises(ValueError, match="published package-index release"):
        parse_ssh_settings(
            ["workbox", "--repo", "/srv/demo"],
            executable_finder=lambda name: str(executable),
            distribution_reader=lambda name: (_ for _ in ()).throw(
                importlib.metadata.PackageNotFoundError(name)
            ),
        )


def test_release_metadata_must_bind_to_the_imported_agent_bridge_package() -> None:
    # This catches clean unrelated release metadata authorizing shadow checkout bytes.
    with pytest.raises(ValueError, match="does not match the imported package"):
        ssh._installed_release_version(
            _released_distribution("0.1.0", install_root=Path("/opt/clean-release"))
        )


def test_release_metadata_rejects_an_available_record_hash_mismatch() -> None:
    # This catches accepting tampered transmitted/imported bytes when RECORD provides a hash.
    class Record:
        hash = SimpleNamespace(mode="sha256", value="not-the-real-digest")
        size = Path(ssh.__file__).stat().st_size

        def __str__(self) -> str:
            return "agent_bridge/ssh.py"

    record = Record()

    records = list(_release_records())
    records[next(index for index, item in enumerate(records) if str(item) == "agent_bridge/ssh.py")] = record
    with pytest.raises(ValueError, match="RECORD hash"):
        ssh._installed_release_version(_released_distribution("0.1.0", files=records))


@pytest.mark.parametrize("files", (None, ()))
def test_release_metadata_requires_an_available_complete_record(files: object) -> None:
    # This catches a release claim with no authenticated package-file manifest.
    with pytest.raises(ValueError, match="RECORD"):
        ssh._installed_release_version(_released_distribution("0.1.0", files=files))


def test_release_metadata_requires_a_size_and_hash_for_every_essential_file() -> None:
    # This catches accepting a RECORD entry that names bytes without authenticating them.
    records = list(_release_records())
    records[next(index for index, item in enumerate(records) if str(item) == "agent_bridge/ssh.py")] = _ReleaseRecord(
        "agent_bridge/ssh.py", size=Path(ssh.__file__).stat().st_size, hash_info=None
    )

    with pytest.raises(ValueError, match="RECORD hash"):
        ssh._installed_release_version(_released_distribution("0.1.0", files=records))


def test_release_metadata_rejects_tampering_in_any_recorded_runtime_module() -> None:
    # This catches authenticating only a hand-picked subset while a later import uses
    # altered bytes from another shipped agent_bridge module.
    records = list(_release_records())
    index = next(index for index, item in enumerate(records) if str(item) == "agent_bridge/projects.py")
    records[index] = _ReleaseRecord(
        "agent_bridge/projects.py",
        size=Path(ssh.__file__).with_name("projects.py").stat().st_size,
        hash_info=SimpleNamespace(mode="sha256", value="not-the-real-digest"),
    )

    with pytest.raises(ValueError, match="RECORD hash"):
        ssh._installed_release_version(_released_distribution("0.1.0", files=records))


def test_release_metadata_accepts_regular_pip_package_bytecode_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This catches rejecting pip's ordinary regular __pycache__ RECORD entries,
    # whose generated bytecode intentionally has no RECORD hash or size.
    package_root = tmp_path / "site-packages" / "agent_bridge"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    source = package_root / "ssh.py"
    source.write_text('VALUE = "RELEASE"\n', encoding="utf-8")
    bytecode = package_root / "__pycache__" / (
        f"ssh.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
    )
    bytecode.parent.mkdir()
    py_compile.compile(str(source), cfile=str(bytecode), doraise=True)
    monkeypatch.setattr(ssh, "__file__", str(source))
    records = [
        *_release_records(),
        _ReleaseRecord(
            "agent_bridge/" + bytecode.relative_to(package_root).as_posix(),
            size=None,
            hash_info=None,
        ),
    ]

    assert ssh._installed_release_version(
        _released_distribution("0.1.0", files=records)
    ) == "0.1.0"


@pytest.mark.parametrize(
    "distribution",
    (
        _released_distribution("0.1.0", name=None),
        _released_distribution("0.1.0", name="another-package"),
        _released_distribution("0.1.0 dev"),
        _released_distribution("v" * 65),
        _released_distribution("0.1.0", direct_url="{\"url\": \"https://example.invalid/pkg.whl\"}"),
        _released_distribution(
            "0.1.0",
            direct_url="{\"url\": \"file:///tmp/agent-bridge.whl\"}",
        ),
        _released_distribution(
            "0.1.0",
            direct_url=(
                "{\"url\": \"https://example.invalid/agent-bridge\", "
                "\"dir_info\": {\"editable\": true}}"
            ),
        ),
        _released_distribution("0.1.0", direct_url="not json"),
    ),
)
def test_parse_ssh_settings_rejects_untrusted_distribution_provenance(
    tmp_path: Path,
    distribution: _ReleasedDistribution,
) -> None:
    # This catches using an editable, direct, or incorrectly identified package install.
    ssh = _write_executable(tmp_path / "ssh", "raise SystemExit(0)\n")

    with pytest.raises(ValueError, match="distribution|release|provenance"):
        _parse(["workbox", "--repo", "/srv/demo"], ssh=ssh, distribution=distribution)


def _settings(tmp_path: Path, *, remote_arguments: tuple[str, ...] = ("--repo", "/srv/demo")):
    executable = _write_executable(tmp_path / "ssh", "raise SystemExit(0)\n")
    return ssh.SSHSettings(
        ssh_executable=executable,
        destination="workbox",
        remote_arguments=remote_arguments,
        local_port=0,
        remote_port=0,
        python_command="python3",
        open_browser=True,
        version="0.1.0",
    )


def _bootstrap_record() -> BootstrapRecord:
    return BootstrapRecord(
        protocol=1,
        python="/home/test/.cache/agent-bridge/runtime/0.1.0/venv/bin/python",
        remote_port=53123,
        version="0.1.0",
    )


def _startup_record(*, key: str = "A" * 32) -> dict[str, object]:
    return {
        "port": 53123,
        "url": f"http://127.0.0.1:53123/?key={key}",
        "fable_status": "subscription_ready",
        "fable_version": "1.2.3",
        "sol_status": "ready",
        "sol_version": "4.5.6",
        "ssh_command": None,
        "repository": "/srv/demo",
        "branch": "main",
    }


def test_bootstrap_argv_quotes_the_program_as_literal_remote_argv(tmp_path: Path) -> None:
    # This catches interpolating bootstrap program inputs into executable remote shell text.
    settings = _settings(tmp_path)

    assert build_bootstrap_argv(settings, bootstrap_source="print('bootstrap')") == (
        str(settings.ssh_executable),
        "-T",
        "--",
        "workbox",
        shlex.join(("python3", "-I", "-c", "print('bootstrap')", "0.1.0", "0")),
    )


def test_local_bootstrap_budget_has_headroom_beyond_the_remote_runtime_sequence() -> None:
    # This catches a local SSH deadline equaling the remote probe/install/verify maximum.
    remote_sequence = (
        2 * bootstrap._VERIFY_TIMEOUT_SECONDS + 2 * bootstrap._INSTALL_TIMEOUT_SECONDS
    )
    assert ssh._BOOTSTRAP_TIMEOUT_SECONDS >= (
        remote_sequence + ssh._BOOTSTRAP_TEARDOWN_HEADROOM_SECONDS
    )
    assert ssh._BOOTSTRAP_TEARDOWN_HEADROOM_SECONDS >= 60.0


def test_tunnel_argv_quotes_unusual_repository_values_as_literal_remote_argv(
    tmp_path: Path,
) -> None:
    # This catches a repository value becoming a second shell command or option.
    repository = "/srv/a space/'quote'/\"double\"/$(not-run);--leading-dash"
    settings = _settings(tmp_path, remote_arguments=("--repo", repository))
    bootstrap = _bootstrap_record()

    argv = build_tunnel_argv(settings, bootstrap, local_port=43123)

    assert argv[:8] == (
        str(settings.ssh_executable),
        "-T",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        "127.0.0.1:43123:127.0.0.1:53123",
        "--",
        "workbox",
    )
    assert argv[8].startswith("exec ")
    assert argv[8].endswith(" 2>&1")
    assert shlex.split(argv[8][len("exec ") : -len(" 2>&1")]) == [
        bootstrap.python,
        "-I",
        "-m",
        "agent_bridge",
        "--repo",
        repository,
        "--host",
        "127.0.0.1",
        "--port",
        "53123",
    ]


def test_run_remote_bootstrap_decodes_one_exact_validated_record(
    tmp_path: Path,
) -> None:
    # This catches accepting a bootstrap record other than the requested runtime authority.
    settings = _settings(tmp_path)
    record = {
        "protocol": 1,
        "python": "/home/test/.cache/agent-bridge/runtime/0.1.0/venv/bin/python",
        "remote_port": 53123,
        "version": "0.1.0",
    }
    calls: list[tuple[str, ...]] = []

    actual = run_remote_bootstrap(
        settings,
        source_reader=lambda: "print('bootstrap')",
        process_runner=lambda argv: (
            calls.append(argv)
            or subprocess.CompletedProcess(argv, 0, json.dumps(record).encode() + b"\n", b"")
        ),
    )

    assert actual == _bootstrap_record()
    assert calls == [build_bootstrap_argv(settings, bootstrap_source="print('bootstrap')")]


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "message"),
    (
        (0, b'{"protocol":1}\n', b"", "shape"),
        (0, b'{"protocol":1,"python":"/x","remote_port":1,"version":"0.1.0","extra":true}\n', b"", "shape"),
        (0, b'{"protocol":1,"python":"/x","remote_port":1,"version":"0.1.0}\n{}\n', b"", "invalid"),
        (0, b"\xff\n", b"", "invalid"),
        (0, b"x" * (64 * 1024 + 1), b"", "invalid"),
        (1, b"", b"remote failed", "failed"),
        (0, b'{"protocol":2,"python":"/x","remote_port":1,"version":"0.1.0"}\n', b"", "protocol"),
        (0, b'{"protocol":1,"python":"/x","remote_port":1,"version":"0.1.1"}\n', b"", "version"),
        (0, b'{"protocol":1,"python":"relative/python","remote_port":1,"version":"0.1.0"}\n', b"", "python"),
        (0, b'{"protocol":1,"python":"/x","remote_port":65536,"version":"0.1.0"}\n', b"", "port"),
    ),
)
def test_run_remote_bootstrap_rejects_invalid_or_untrusted_protocol_output(
    tmp_path: Path,
    returncode: int,
    stdout: bytes,
    stderr: bytes,
    message: str,
) -> None:
    # This catches accepting malformed, mismatched, or unbounded remote bootstrap output.
    settings = _settings(tmp_path)

    with pytest.raises(SSHLaunchError, match=message):
        run_remote_bootstrap(
            settings,
            source_reader=lambda: "print('bootstrap')",
            process_runner=lambda argv: subprocess.CompletedProcess(
                argv, returncode, stdout, stderr
            ),
        )


def test_run_remote_bootstrap_redacts_access_keys_from_failure_diagnostics(
    tmp_path: Path,
) -> None:
    # This catches a remote diagnostic exposing a browser access key.
    settings = _settings(tmp_path)
    access_key = "A" * 32

    with pytest.raises(SSHLaunchError) as error:
        run_remote_bootstrap(
            settings,
            source_reader=lambda: "print('bootstrap')",
            process_runner=lambda argv: subprocess.CompletedProcess(
                argv,
                1,
                b"",
                f"http://127.0.0.1:53123/?key={access_key}".encode(),
            ),
        )

    assert access_key not in str(error.value)
    assert "<redacted>" in str(error.value)


def test_run_remote_bootstrap_rejects_a_timed_out_runner(tmp_path: Path) -> None:
    # This catches treating a hung remote bootstrap as a usable SSH session.
    settings = _settings(tmp_path)

    def timeout(argv: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(argv, 180)

    with pytest.raises(SSHLaunchError, match="timed out"):
        run_remote_bootstrap(
            settings,
            source_reader=lambda: "print('bootstrap')",
            process_runner=timeout,
        )


@pytest.mark.parametrize("kind", ("symlink", "oversized"))
def test_default_bootstrap_source_must_be_a_bounded_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    # This catches sending an attacker-controlled or unexpectedly large local bootstrap program.
    package = tmp_path / "agent_bridge"
    package.mkdir()
    (package / "ssh.py").write_text("# test module\n", encoding="utf-8")
    source = package / "_remote_bootstrap.py"
    if kind == "symlink":
        target = tmp_path / "bootstrap-source.py"
        target.write_text("print('bootstrap')\n", encoding="utf-8")
        source.symlink_to(target)
    else:
        source.write_bytes(b"x" * (128 * 1024 + 1))
    monkeypatch.setattr(ssh, "__file__", str(package / "ssh.py"))

    with pytest.raises(SSHLaunchError, match="bootstrap source"):
        run_remote_bootstrap(
            _settings(tmp_path),
            process_runner=lambda argv: pytest.fail("unsafe source launched SSH"),
        )


def test_parse_remote_startup_validates_and_localizes_the_keyed_loopback_url() -> None:
    # This catches forwarding a non-loopback record or losing its opaque access key.
    remote = _startup_record()

    parsed = parse_remote_startup(
        json.dumps(remote, separators=(",", ":")).encode() + b"\n",
        expected_remote_port=53123,
    )
    local = localized_startup(parsed, local_port=43123)

    assert local == {
        **remote,
        "port": 43123,
        "url": "http://127.0.0.1:43123/?key=" + "A" * 32,
    }
    assert parsed is not local


@pytest.mark.parametrize(
    "mutator",
    (
        lambda record: record.pop("branch"),
        lambda record: record.__setitem__("extra", "value"),
        lambda record: record.__setitem__("port", 53124),
        lambda record: record.__setitem__("url", "https://127.0.0.1:53123/?key=" + "A" * 32),
        lambda record: record.__setitem__("url", "http://localhost:53123/?key=" + "A" * 32),
        lambda record: record.__setitem__("url", "http://127.0.0.1:53123/?key=" + "A" * 32 + "&key=" + "B" * 32),
        lambda record: record.__setitem__("url", "http://user@127.0.0.1:53123/?key=" + "A" * 32),
        lambda record: record.__setitem__("url", "\ud800"),
        lambda record: record.__setitem__("fable_status", None),
        lambda record: record.__setitem__("ssh_command", 1),
    ),
)
def test_parse_remote_startup_rejects_untrusted_shapes_types_and_urls(mutator) -> None:
    # This catches accepting a remote startup record that can redirect or confuse the local client.
    record = _startup_record()
    mutator(record)

    with pytest.raises(SSHLaunchError):
        parse_remote_startup(
            json.dumps(record, separators=(",", ":")).encode() + b"\n",
            expected_remote_port=53123,
        )


class _Response:
    def __init__(self, status: int, *, final_url: str | None = None) -> None:
        self.status = status
        self._final_url = final_url

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *unused: object) -> None:
        return None

    def geturl(self) -> str | None:
        return self._final_url


class _Process:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


class _TunnelProcess:
    """A foreground SSH child with real readable pipes and recorded cleanup."""

    def __init__(
        self,
        stdout: bytes,
        stderr: bytes = b"",
        *,
        returncode: int | None = None,
        events: list[object] | None = None,
    ) -> None:
        self.stdout = self._pipe(stdout)
        self.stderr = self._pipe(stderr)
        self.returncode = returncode
        self.events = events if events is not None else []
        self.terminated = False
        self.killed = False
        self.wait_calls: list[float | None] = []

    @staticmethod
    def _pipe(payload: bytes):
        reader, writer = os.pipe()
        os.write(writer, payload)
        os.close(writer)
        return os.fdopen(reader, "rb", buffering=0)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        self.events.append("wait")
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.events.append("terminate")
        self.returncode = 143

    def kill(self) -> None:
        self.killed = True
        self.events.append("kill")
        self.returncode = 137


class _LiveTunnelProcess(_TunnelProcess):
    """Pipe-backed child that can emit after the startup record was consumed."""

    def __init__(self, stdout: bytes) -> None:
        self.stdout, self._stdout_writer = self._live_pipe(stdout)
        self.stderr, self._stderr_writer = self._live_pipe(b"")
        self.returncode = None
        self.events = []
        self.terminated = False
        self.killed = False
        self.wait_calls = []

    @staticmethod
    def _live_pipe(payload: bytes):
        reader, writer = os.pipe()
        if payload:
            os.write(writer, payload)
        return os.fdopen(reader, "rb", buffering=0), writer

    def write_stderr(self, payload: bytes, *, returncode: int | None = None) -> None:
        os.write(self._stderr_writer, payload)
        if returncode is not None:
            self.returncode = returncode
            self._close_writers()

    def write_stdout(self, payload: bytes, *, returncode: int | None = None) -> None:
        os.write(self._stdout_writer, payload)
        if returncode is not None:
            self.returncode = returncode
            self._close_writers()

    def _close_writers(self) -> None:
        for name in ("_stdout_writer", "_stderr_writer"):
            descriptor = getattr(self, name, None)
            if descriptor is not None:
                os.close(descriptor)
                setattr(self, name, None)

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        self._close_writers()
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 143
        self._close_writers()

    def kill(self) -> None:
        self.killed = True
        self.returncode = 137
        self._close_writers()


class _SilentSelector:
    def __init__(self) -> None:
        self.registered: dict[int, object] = {}
        self.select_calls = 0

    def register(self, pipe: object, unused_events: int, data: str) -> None:
        self.registered[pipe.fileno()] = pipe  # type: ignore[attr-defined]

    def unregister(self, pipe: object) -> None:
        self.registered.pop(pipe.fileno(), None)  # type: ignore[attr-defined]

    def get_map(self) -> dict[int, object]:
        return self.registered

    def select(self, unused_timeout: float) -> list[object]:
        self.select_calls += 1
        if self.select_calls > 2:
            raise RuntimeError("silent startup exceeded bounded test cycles")
        return []

    def close(self) -> None:
        return None


class _RecordingOutput(io.StringIO):
    def __init__(self, events: list[object]) -> None:
        super().__init__()
        self.events = events

    def write(self, value: str) -> int:
        if value.startswith("http://127.0.0.1:"):
            self.events.append(("output", value.rstrip("\n")))
        return super().write(value)


class _BrokenOutput(io.StringIO):
    def write(self, value: str) -> int:
        raise BrokenPipeError("closed test sink")


_SESSION_KEY = "S" * 32


def _startup_line(*, key: str = _SESSION_KEY, port: int = 53123) -> bytes:
    record = _startup_record(key=key)
    record["port"] = port
    record["url"] = f"http://127.0.0.1:{port}/?key={key}"
    return json.dumps(record, separators=(",", ":")).encode() + b"\n"


def test_foreground_ssh_waits_for_readiness_before_output_and_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches printing or browsing a keyed URL before its owning tunnel is ready.
    settings = _settings(tmp_path)
    events: list[object] = []
    process = _TunnelProcess(_startup_line(), events=events)
    output = _RecordingOutput(events)
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    assert run_ssh(
        settings,
        stdout=output,
        bootstrap_runner=lambda unused: events.append("bootstrap") or _bootstrap_record(),
        popen_factory=lambda argv, **kwargs: (
            events.append(("tunnel", argv[5])) or process
        ),
        readiness_opener=lambda url, **kwargs: (
            events.append(("ready", url)) or _Response(200)
        ),
        browser_open=lambda url: events.append(("browser", url)) or True,
    ) == 0

    assert events == [
        "bootstrap",
        ("tunnel", "127.0.0.1:43123:127.0.0.1:53123"),
        ("ready", f"http://127.0.0.1:43123/?key={_SESSION_KEY}"),
        ("output", f"http://127.0.0.1:43123/?key={_SESSION_KEY}"),
        ("browser", f"http://127.0.0.1:43123/?key={_SESSION_KEY}"),
        "wait",
    ]


def test_foreground_ssh_uses_exact_attached_popen_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches changing the one tunnel owner into a shell or detached subprocess.
    process = _TunnelProcess(_startup_line())
    captured: dict[str, object] = {}
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    assert run_ssh(
        _settings(tmp_path),
        stdout=io.StringIO(),
        bootstrap_runner=lambda unused: _bootstrap_record(),
        popen_factory=lambda argv, **kwargs: captured.update(kwargs) or process,
        readiness_opener=lambda *args, **kwargs: _Response(200),
        browser_open=lambda url: True,
    ) == 0

    assert captured == {
        "stdin": None,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "shell": False,
    }


def test_foreground_ssh_no_open_still_prints_once_and_stays_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches --no-open skipping readiness, URL output, or foreground ownership.
    settings = replace(_settings(tmp_path), open_browser=False)
    events: list[object] = []
    process = _TunnelProcess(_startup_line(), events=events)
    output = _RecordingOutput(events)
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    assert run_ssh(
        settings,
        stdout=output,
        bootstrap_runner=lambda unused: _bootstrap_record(),
        popen_factory=lambda *args, **kwargs: process,
        readiness_opener=lambda *args, **kwargs: _Response(200),
        browser_open=lambda url: pytest.fail("--no-open must not call the browser"),
    ) == 0

    assert output.getvalue().count(f"?key={_SESSION_KEY}") == 1
    assert process.wait_calls == [None]


def test_foreground_ssh_warns_if_the_browser_cannot_open_but_stays_attached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches treating browser integration as permission to abandon the session.
    process = _TunnelProcess(_startup_line())
    errors = io.StringIO()
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    assert run_ssh(
        _settings(tmp_path),
        stdout=io.StringIO(),
        stderr=errors,
        bootstrap_runner=lambda unused: _bootstrap_record(),
        popen_factory=lambda *args, **kwargs: process,
        readiness_opener=lambda *args, **kwargs: _Response(200),
        browser_open=lambda url: False,
    ) == 0

    assert "could not open browser" in errors.getvalue()
    assert "?key=" not in errors.getvalue()
    assert process.wait_calls == [None]


def test_foreground_ssh_retries_an_automatic_local_forward_collision_and_reaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # This catches retrying a failed automatic forward without reaping its SSH owner.
    settings = _settings(tmp_path)
    first = _TunnelProcess(b"", b"bind [127.0.0.1]:43123: Address already in use\n", returncode=255)
    second = _TunnelProcess(_startup_line())
    processes = iter((first, second))
    ports = iter((43123, 43124))
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: next(ports))

    assert run_ssh(
        settings,
        stdout=io.StringIO(),
        bootstrap_runner=lambda unused: _bootstrap_record(),
        popen_factory=lambda *args, **kwargs: next(processes),
        readiness_opener=lambda *args, **kwargs: _Response(200),
        browser_open=lambda url: True,
    ) == 0

    assert first.wait_calls == [None]


def test_foreground_ssh_does_not_retry_a_fixed_local_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # This catches silently changing a port the user explicitly selected.
    base = _settings(tmp_path)
    settings = replace(base, local_port=43123)
    process = _TunnelProcess(b"", b"bind [127.0.0.1]:43123: Address already in use\n", returncode=255)
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: requested)
    calls: list[object] = []

    with pytest.raises(SSHLaunchError, match="Address already in use"):
        run_ssh(
            settings,
            stdout=io.StringIO(),
            bootstrap_runner=lambda unused: calls.append("bootstrap") or _bootstrap_record(),
            popen_factory=lambda *args, **kwargs: process,
            readiness_opener=lambda *args, **kwargs: _Response(200),
        )

    assert calls == ["bootstrap"]
    assert process.wait_calls == [None]


def test_foreground_ssh_retries_an_automatic_remote_listener_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # This catches retrying a remote automatic listener without obtaining a new bootstrap port.
    settings = _settings(tmp_path)
    first = _TunnelProcess(b"remote listener: address already in use\n", returncode=1)
    second = _TunnelProcess(_startup_line(port=53124))
    records = iter((
        _bootstrap_record(),
        replace(_bootstrap_record(), remote_port=53124),
    ))
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)
    forwards: list[str] = []

    assert run_ssh(
        settings,
        stdout=io.StringIO(),
        bootstrap_runner=lambda unused: next(records),
        popen_factory=lambda argv, **kwargs: forwards.append(argv[5]) or (
            first if len(forwards) == 1 else second
        ),
        readiness_opener=lambda *args, **kwargs: _Response(200),
        browser_open=lambda url: True,
    ) == 0

    assert forwards == [
        "127.0.0.1:43123:127.0.0.1:53123",
        "127.0.0.1:43123:127.0.0.1:53124",
    ]
    assert first.wait_calls == [None]


def test_foreground_ssh_retries_remote_collision_reported_after_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches ignoring a real Uvicorn bind failure emitted after its startup JSON.
    first = _LiveTunnelProcess(_startup_line())
    second = _TunnelProcess(_startup_line(port=53124))
    records = iter((_bootstrap_record(), replace(_bootstrap_record(), remote_port=53124)))
    processes = iter((first, second))
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)
    readiness_calls = 0

    def readiness(url: str, **unused: object) -> _Response:
        nonlocal readiness_calls
        readiness_calls += 1
        if readiness_calls == 1:
            first.write_stdout(b"ERROR: [Errno 98] address already in use\n", returncode=1)
            raise urllib.error.URLError("refused")
        return _Response(200)

    assert run_ssh(
        _settings(tmp_path),
        stdout=io.StringIO(),
        bootstrap_runner=lambda unused: next(records),
        popen_factory=lambda *args, **kwargs: next(processes),
        readiness_opener=readiness,
        browser_open=lambda url: True,
    ) == 0

    assert first.wait_calls == [None]


def test_foreground_ssh_drains_queued_post_startup_collision_to_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches stopping after two ready reads before the queued remote bind failure.
    first = _LiveTunnelProcess(_startup_line())
    second = _TunnelProcess(_startup_line(port=53124))
    records = iter((_bootstrap_record(), replace(_bootstrap_record(), remote_port=53124)))
    processes = iter((first, second))
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)
    readiness_calls = 0

    def readiness(url: str, **unused: object) -> _Response:
        nonlocal readiness_calls
        readiness_calls += 1
        if readiness_calls == 1:
            first.write_stdout(b"x" * 4097 + b"y" * 4097 + b"bind: address already in use\n", returncode=1)
            raise urllib.error.URLError("refused")
        return _Response(200)

    assert run_ssh(
        _settings(tmp_path),
        stdout=io.StringIO(),
        bootstrap_runner=lambda unused: next(records),
        popen_factory=lambda *args, **kwargs: next(processes),
        readiness_opener=readiness,
        browser_open=lambda url: True,
    ) == 0

    assert first.wait_calls == [None]


def test_foreground_ssh_classifies_post_startup_bind_as_remote_with_fixed_local_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches a post-startup application bind being mistaken for a local SSH forward.
    first = _LiveTunnelProcess(_startup_line())
    second = _TunnelProcess(_startup_line(port=53124))
    records = iter((_bootstrap_record(), replace(_bootstrap_record(), remote_port=53124)))
    processes = iter((first, second))
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)
    readiness_calls = 0

    def readiness(url: str, **unused: object) -> _Response:
        nonlocal readiness_calls
        readiness_calls += 1
        if readiness_calls == 1:
            first.write_stdout(b"bind: address already in use\n", returncode=1)
            raise urllib.error.URLError("refused")
        return _Response(200)

    assert run_ssh(
        replace(_settings(tmp_path), local_port=43123),
        stdout=io.StringIO(),
        bootstrap_runner=lambda unused: next(records),
        popen_factory=lambda *args, **kwargs: next(processes),
        readiness_opener=readiness,
        browser_open=lambda url: True,
    ) == 0


def test_foreground_ssh_refuses_fixed_remote_collision_reported_after_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches changing a user-fixed remote port after a post-startup bind failure.
    process = _LiveTunnelProcess(_startup_line())
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)
    bootstrap_calls: list[object] = []

    def readiness(url: str, **unused: object) -> _Response:
        process.write_stdout(b"ERROR: [Errno 98] address already in use\n", returncode=1)
        raise urllib.error.URLError("refused")

    with pytest.raises(SSHLaunchError, match="SSH exited"):
        run_ssh(
            replace(_settings(tmp_path), remote_port=53123),
            stdout=io.StringIO(),
            bootstrap_runner=lambda unused: bootstrap_calls.append(None) or _bootstrap_record(),
            popen_factory=lambda *args, **kwargs: process,
            readiness_opener=readiness,
        )

    assert bootstrap_calls == [None]
    assert process.wait_calls == [None]


def test_foreground_ssh_times_out_a_live_silent_startup_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches a live SSH child holding both pipes open without a startup record forever.
    process = _LiveTunnelProcess(b"")
    selector = _SilentSelector()
    clock = iter((0.0, ssh._startup_timeout_seconds(_settings(tmp_path)))).__next__
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)
    monkeypatch.setattr(ssh.selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(ssh.time, "monotonic", clock)

    with pytest.raises(SSHLaunchError, match="startup deadline"):
        run_ssh(
            _settings(tmp_path),
            stdout=io.StringIO(),
            bootstrap_runner=lambda unused: _bootstrap_record(),
            popen_factory=lambda *args, **kwargs: process,
        )

    assert process.terminated is True
    assert process.wait_calls == [1.0]


def test_tunnel_selector_construction_failure_stops_reaps_and_closes_its_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches selector setup escaping after foreground-tunnel Popen ownership begins.
    process = _TunnelProcess(b"")
    monkeypatch.setattr(ssh.os, "set_blocking", lambda *args: None)
    monkeypatch.setattr(
        ssh.selectors, "DefaultSelector", lambda: (_ for _ in ()).throw(OSError("fd limit"))
    )

    with pytest.raises(OSError, match="fd limit"):
        ssh._run_tunnel_attempt(
            _settings(tmp_path),
            _bootstrap_record(),
            local_port=43123,
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            browser_open=lambda unused: True,
            popen_factory=lambda *args, **kwargs: process,
            readiness_opener=None,
        )

    assert process.terminated is True
    assert process.wait_calls == [1.0]
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_tunnel_selector_registration_failure_stops_reaps_closes_pipes_and_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches a partial selector setup leaking the exact tunnel child.
    process = _TunnelProcess(b"")

    class FailingSelector:
        closed = False

        def register(self, *unused: object) -> None:
            raise OSError("fd limit")

        def close(self) -> None:
            self.closed = True

    selector = FailingSelector()
    monkeypatch.setattr(ssh.os, "set_blocking", lambda *args: None)
    monkeypatch.setattr(ssh.selectors, "DefaultSelector", lambda: selector)

    with pytest.raises(OSError, match="fd limit"):
        ssh._run_tunnel_attempt(
            _settings(tmp_path),
            _bootstrap_record(),
            local_port=43123,
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            browser_open=lambda unused: True,
            popen_factory=lambda *args, **kwargs: process,
            readiness_opener=None,
        )

    assert process.terminated is True
    assert process.wait_calls == [1.0]
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert selector.closed is True


@pytest.mark.parametrize("projects", (1, 32))
def test_startup_deadline_covers_subprocess_work_and_explicit_overhead_for_every_project_count(
    tmp_path: Path, projects: int,
) -> None:
    # This catches a raw subprocess sum leaving no valid startup assembly headroom.
    settings = replace(
        _settings(tmp_path),
        remote_arguments=tuple(
            argument
            for index in range(projects)
            for argument in ("--project", f"project{index}=/srv/project{index}")
        ),
    )

    assert ssh._startup_timeout_seconds(settings) >= (
        projects * (
            ssh._STARTUP_TIMEOUT_PER_PROJECT_SECONDS
            + ssh._STARTUP_NON_SUBPROCESS_OVERHEAD_PER_PROJECT_SECONDS
        )
        + ssh._STARTUP_GLOBAL_OVERHEAD_SECONDS
    )


@pytest.mark.parametrize("projects", (1, 32))
def test_startup_deadline_includes_each_provider_timeout_cancellation_cleanup(
    tmp_path: Path, projects: int,
) -> None:
    # This catches wait_for cancellation consuming its ProcessRunner 5-second TERM grace.
    settings = replace(
        _settings(tmp_path),
        remote_arguments=tuple(
            argument
            for index in range(projects)
            for argument in ("--project", f"project{index}=/srv/project{index}")
        ),
    )

    assert ssh._startup_timeout_seconds(settings) >= (
        projects * (2 * 10.0 + 3 * (10.0 + 5.0) + 10.0) + 10.0
    )


def test_foreground_ssh_scans_remote_warnings_before_the_valid_startup_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches remote 2>&1 warnings being mistaken for the startup protocol record.
    process = _TunnelProcess(b"Python warning: optional tool unavailable\n" + _startup_line())
    errors = io.StringIO()
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    assert run_ssh(
        _settings(tmp_path),
        stdout=io.StringIO(),
        stderr=errors,
        bootstrap_runner=lambda unused: _bootstrap_record(),
        popen_factory=lambda *args, **kwargs: process,
        readiness_opener=lambda *args, **kwargs: _Response(200),
        browser_open=lambda url: True,
    ) == 0

    assert "optional tool unavailable" in errors.getvalue()
    assert _SESSION_KEY not in errors.getvalue()


def test_foreground_ssh_includes_source_merged_remote_startup_failure_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches turning a useful remote application failure into a protocol-only error.
    process = _TunnelProcess(
        b"agent-bridge: repository does not exist: /missing/repository\n",
        returncode=2,
    )
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    with pytest.raises(SSHLaunchError, match="repository does not exist"):
        run_ssh(
            _settings(tmp_path),
            stdout=io.StringIO(),
            bootstrap_runner=lambda unused: _bootstrap_record(),
            popen_factory=lambda *args, **kwargs: process,
            readiness_opener=lambda *args, **kwargs: _Response(200),
        )

    assert process.wait_calls == [None]


def test_foreground_ssh_bounds_multiple_malformed_remote_startup_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches accepting an unbounded series of non-protocol remote stdout lines.
    monkeypatch.setattr(ssh, "_MAX_PRESTARTUP_LOG_BYTES", 12)
    process = _TunnelProcess(b"bad-1\nbad-2\nbad-3\n", returncode=2)
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    with pytest.raises(SSHLaunchError, match="pre-startup stdout exceeded"):
        run_ssh(
            _settings(tmp_path),
            stdout=io.StringIO(),
            bootstrap_runner=lambda unused: _bootstrap_record(),
            popen_factory=lambda *args, **kwargs: process,
            readiness_opener=lambda *args, **kwargs: _Response(200),
        )

    assert process.wait_calls == [None]


@pytest.mark.parametrize(
    ("prefix", "suffix"),
    (
        (_SESSION_KEY[:10].encode(), _SESSION_KEY[10:].encode()),
        (b"?KeY=" + _SESSION_KEY[:10].encode(), _SESSION_KEY[10:].encode() + b"&tail=1"),
        (b'{"Access_Key":"' + _SESSION_KEY[:10].encode(), _SESSION_KEY[10:].encode() + b'"}'),
    ),
)
def test_tunnel_diagnostics_redact_remote_secrets_despite_interposed_ssh_stderr(
    prefix: bytes, suffix: bytes,
) -> None:
    # This catches rebuilding a remote token by locally joining independent source frames.
    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
    ssh._append_tunnel_diagnostic(diagnostics, "stdout", prefix + suffix)
    ssh._append_tunnel_diagnostic(diagnostics, "stderr", b"OpenSSH warning\n")

    rendered = ssh._redact_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)

    assert _SESSION_KEY not in rendered
    assert "OpenSSH warning" in rendered
    assert "<redacted>" in rendered


def test_tunnel_diagnostics_detect_remote_collision_despite_ssh_interposition() -> None:
    # This catches selector interleaving hiding a real remote Uvicorn bind failure.
    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._append_tunnel_diagnostic(diagnostics, "stdout", b"ERROR: address already in use\n")
    ssh._append_tunnel_diagnostic(diagnostics, "stderr", b"OpenSSH warning\n")

    assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) == "remote"


def test_tunnel_diagnostics_cannot_forge_a_collision_from_two_sources() -> None:
    # This catches independent source fragments manufacturing a retry condition.
    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._append_tunnel_diagnostic(diagnostics, "stdout", b"address already ")
    ssh._append_tunnel_diagnostic(diagnostics, "stderr", b"in use\n")

    assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) is None


@pytest.mark.parametrize("marker", (b"", b"?key=", b'"access_key":"'))
def test_tunnel_diagnostic_tail_redacts_each_source_before_final_truncation(marker: bytes) -> None:
    # This catches dropping a marker/key prefix before independently sanitizing its source tail.
    key = "K" * 128
    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._activate_tunnel_diagnostics(diagnostics, key=key)
    ssh._append_tunnel_diagnostic(
        diagnostics,
        "stdout",
        b"x" * ssh._MAX_TUNNEL_DIAGNOSTIC_CAPTURE_BYTES
        + marker
        + key.encode()
        + b"z" * (ssh._MAX_DIAGNOSTIC_BYTES - 64),
    )

    rendered = ssh._redact_tunnel_diagnostics(diagnostics, key=key)

    assert key not in rendered
    assert "<redacted>" in rendered


@pytest.mark.parametrize("marker", (b"?key=", b'"access_key":"'))
def test_tunnel_diagnostic_tail_suppresses_an_unbounded_generic_value(marker: bytes) -> None:
    # This catches raw-tail truncation forgetting a query/JSON secret state before display truncation.
    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
    secret = b"V" * 20_000
    ssh._append_tunnel_diagnostic(diagnostics, "stdout", marker + secret)

    rendered = ssh._redact_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)

    assert "V" * 96 not in rendered
    assert "<redacted>" in rendered


@pytest.mark.parametrize("generic_mapping", (False, True))
def test_tunnel_diagnostics_cap_the_labelled_two_source_render_globally(
    generic_mapping: bool,
) -> None:
    # This catches joining independently bounded source tails into an over-4-KiB error.
    sources = {
        "stdout": b"r" * 5_000 + b"?key=" + _SESSION_KEY.encode(),
        "stderr": b"s" * 5_000 + b"?key=" + _SESSION_KEY.encode(),
    }
    if generic_mapping:
        diagnostics: dict[str, bytes | bytearray] = sources
    else:
        diagnostics = ssh._new_tunnel_diagnostics()
        ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
        for stream, value in sources.items():
            ssh._append_tunnel_diagnostic(diagnostics, stream, value)

    rendered = ssh._redact_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)

    assert len(rendered.encode("utf-8")) <= ssh._MAX_DIAGNOSTIC_BYTES
    assert _SESSION_KEY not in rendered
    assert "remote output:" in rendered
    assert "SSH diagnostics:" in rendered


@pytest.mark.parametrize(
    ("marker", "expected"),
    (
        (b"?key=", "remote"),
        (b"&key=", "remote"),
        (b'{"key":"', None),
        (b'{"access_key":"', None),
    ),
)
def test_tunnel_diagnostics_classify_long_generic_values_only_after_their_real_boundary(
    marker: bytes, expected: str | None,
) -> None:
    # This catches treating an unescaped query-space boundary like a protected value byte.
    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
    ssh._append_tunnel_diagnostic(
        diagnostics,
        "stdout",
        marker
        + b"V" * (ssh._MAX_TUNNEL_DIAGNOSTIC_CAPTURE_BYTES + 1)
        + b" address already in use",
    )

    assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) == expected


def test_tunnel_diagnostics_classify_a_genuine_collision_after_long_ordinary_output() -> None:
    # This catches losing a real remote bind failure while retaining sanitized source tails.
    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
    ssh._append_tunnel_diagnostic(
        diagnostics,
        "stdout",
        b"ordinary output " * 1_000 + b"address already in use",
    )

    assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) == "remote"


def test_tunnel_diagnostic_classification_finalizes_a_stable_snapshot() -> None:
    # This catches collision inspection drifting from later render output or accepting later bytes.
    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
    ssh._append_tunnel_diagnostic(diagnostics, "stdout", b"address already in use")

    first_collision = ssh._collision_from_diagnostic(diagnostics, after_startup=True)
    first_render = ssh._redact_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)

    assert first_collision == "remote"
    assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) == first_collision
    assert ssh._redact_tunnel_diagnostics(diagnostics, key=_SESSION_KEY) == first_render
    with pytest.raises(RuntimeError, match="finalized"):
        ssh._append_tunnel_diagnostic(diagnostics, "stdout", b"later output")


@pytest.mark.parametrize("delimiter", (b" ", b"\t", b"'", b'"', b"<", b">", b"&", b"#", b"\r", b"\n"))
def test_tunnel_diagnostics_restore_standard_query_boundaries_for_visible_collisions(
    delimiter: bytes,
) -> None:
    # This catches treating ordinary request/log boundaries as part of a query value.
    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
    ssh._append_tunnel_diagnostic(
        diagnostics,
        "stdout",
        b"GET /?KeY=generic-value" + delimiter + b"address already in use\n",
    )

    rendered = ssh._redact_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)

    assert "generic-value" not in rendered
    assert "address already in use" in rendered
    assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) == "remote"


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        (
            b'INFO "GET /?KeY=generic-value HTTP/1.1" 500 ordinary suffix\n',
            'INFO "GET /?KeY=<redacted> HTTP/1.1" 500 ordinary suffix\n',
        ),
        (
            b'{"url":"http://x/?KeY=generic-value","status":500}\n',
            '{"url":"http://x/?KeY=<redacted>","status":500}\n',
        ),
    ),
)
def test_redacted_log_writer_scans_generic_queries_inside_ordinary_quoted_text_at_every_split(
    payload: bytes,
    expected: str,
) -> None:
    # This catches bypassing the query scanner when a non-secret quoted sequence closes.
    for split in range(len(payload) + 1):
        output = io.StringIO()
        writer = ssh._RedactedLogWriter(output, key=_SESSION_KEY)
        writer.feed(payload[:split])
        writer.feed(payload[split:])
        writer.flush()
        assert output.getvalue() == expected


def test_redacted_log_writer_scans_a_marker_before_a_large_ordinary_quoted_field_overflows() -> None:
    # This catches an overlong ordinary quoted field releasing an earlier generic query value.
    output = io.StringIO()
    writer = ssh._RedactedLogWriter(output, key=_SESSION_KEY)
    payload = (
        b'"ordinary ?key='
        + b"V" * (ssh._MAX_TUNNEL_DIAGNOSTIC_CAPTURE_BYTES + 1)
        + b"&tail=ordinary\n"
    )

    for value in payload:
        writer.feed(bytes((value,)))
    writer.flush()

    assert "V" * 96 not in output.getvalue()
    assert "&tail=ordinary\n" in output.getvalue()


def test_tunnel_diagnostics_do_not_classify_a_long_json_url_query_value() -> None:
    # This catches a quoted JSON URL replaying protected collision words as ordinary diagnostics.
    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
    payload = (
        b'{"url":"http://x/?key='
        + b"V" * (ssh._MAX_TUNNEL_DIAGNOSTIC_CAPTURE_BYTES + 1)
        + b'address already in use","status":500}'
    )

    for value in payload:
        ssh._append_tunnel_diagnostic(diagnostics, "stdout", bytes((value,)))

    assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) is None


def test_tunnel_diagnostics_retain_a_genuine_collision_followed_by_four_kib_of_output() -> None:
    # This catches using the shorter display tail rather than the accepted collision horizon.
    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
    ssh._append_tunnel_diagnostic(
        diagnostics,
        "stdout",
        b"address already in use\n" + b"x" * ssh._MAX_DIAGNOSTIC_BYTES,
    )

    assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) == "remote"


@pytest.mark.parametrize("small_stream", ("stdout", "stderr"))
def test_tunnel_diagnostic_renderer_water_fills_asymmetric_source_tails(
    small_stream: str,
) -> None:
    # This catches overwriting a prior allocation and leaving usable diagnostic bytes empty.
    diagnostics = {
        "stdout": b"r" if small_stream == "stdout" else b"r" * 5_000,
        "stderr": b"s" if small_stream == "stderr" else b"s" * 5_000,
    }

    rendered = ssh._redact_tunnel_diagnostics(diagnostics)

    assert len(rendered.encode("utf-8")) == ssh._MAX_DIAGNOSTIC_BYTES


def test_tunnel_diagnostic_renderer_water_fills_multibyte_source_tails_as_valid_utf8() -> None:
    # This catches byte-budget allocation splitting a code point or discarding most of both tails.
    diagnostics = {"stdout": "é".encode() * 5_000, "stderr": "漢".encode() * 5_000}

    rendered = ssh._redact_tunnel_diagnostics(diagnostics)

    assert rendered.encode("utf-8").decode("utf-8") == rendered
    assert ssh._MAX_DIAGNOSTIC_BYTES - len(rendered.encode("utf-8")) < 4


@pytest.mark.parametrize("field", ("key", "access_key"))
@pytest.mark.parametrize(
    "quoted_length",
    (256, 257, 300, ssh._MAX_TUNNEL_DIAGNOSTIC_CAPTURE_BYTES + 1),
)
def test_tunnel_diagnostics_keep_quote_parity_after_a_long_ordinary_json_token(
    field: str,
    quoted_length: int,
) -> None:
    # This catches an overflowing non-secret JSON token disabling the following protected field.
    phrase = b"address already in use"
    payload = (
        b'{"long":"'
        + b"x" * quoted_length
        + b'","'
        + field.encode()
        + b'":"'
        + phrase
        + b'"}'
    )

    if quoted_length > ssh._MAX_JSON_FIELD_BYTES:
        transition = len(b'{"long":"') + ssh._MAX_JSON_FIELD_BYTES
        splits = (0, 1, transition - 1, transition, transition + 1, len(payload) - 1, len(payload))
    else:
        splits = range(len(payload) + 1)
    for split in splits:
        diagnostics = ssh._new_tunnel_diagnostics()
        ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
        ssh._append_tunnel_diagnostic(diagnostics, "stdout", payload[:split])
        ssh._append_tunnel_diagnostic(diagnostics, "stdout", payload[split:])
        rendered = ssh._redact_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
        assert phrase.decode() not in rendered
        assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) is None

    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
    for value in payload:
        ssh._append_tunnel_diagnostic(diagnostics, "stdout", bytes((value,)))
    rendered = ssh._redact_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
    assert phrase.decode() not in rendered
    assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) is None


def test_tunnel_diagnostics_track_an_escaped_quote_across_the_overflow_boundary() -> None:
    # This catches an overflow-boundary backslash making an escaped quote close the token.
    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
    phrase = b"address already in use"
    payload = (
        b'{"long":"'
        + b"x" * 255
        + b'\\"ordinary","key":"'
        + phrase
        + b'"}'
    )

    for value in payload:
        ssh._append_tunnel_diagnostic(diagnostics, "stdout", bytes((value,)))
    rendered = ssh._redact_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)

    assert '\\"ordinary' in rendered
    assert phrase.decode() not in rendered
    assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) is None


@pytest.mark.parametrize("field", ("key", "access_key"))
@pytest.mark.parametrize("backslashes", (1, 3, 5))
def test_tunnel_diagnostics_keep_escape_parity_at_the_json_field_boundary(
    field: str,
    backslashes: int,
) -> None:
    # This catches an escaped quote at byte 256 closing a non-secret JSON token.
    phrase = b"address already in use"
    payload = (
        b'{"long":"'
        + b"x" * (254 - backslashes)
        + b"\\" * backslashes
        + b'"ordinary-tail","'
        + field.encode()
        + b'":"'
        + phrase
        + b'"}'
    )

    for split in range(len(payload) + 1):
        diagnostics = ssh._new_tunnel_diagnostics()
        ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
        ssh._append_tunnel_diagnostic(diagnostics, "stdout", payload[:split])
        ssh._append_tunnel_diagnostic(diagnostics, "stdout", payload[split:])
        rendered = ssh._redact_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
        assert "ordinary-tail" in rendered
        assert phrase.decode() not in rendered
        assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) is None

    diagnostics = ssh._new_tunnel_diagnostics()
    ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
    for value in payload:
        ssh._append_tunnel_diagnostic(diagnostics, "stdout", bytes((value,)))
    rendered = ssh._redact_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
    assert "ordinary-tail" in rendered
    assert phrase.decode() not in rendered
    assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) is None


@pytest.mark.parametrize("field", ("key", "access_key"))
@pytest.mark.parametrize("backslashes", (2, 4))
def test_tunnel_diagnostics_treat_even_escape_parity_as_a_normal_json_close(
    field: str,
    backslashes: int,
) -> None:
    # This catches treating an even backslash run as an escaped JSON quote.
    phrase = b"address already in use"
    payload = (
        b'{"long":"'
        + b"x" * (254 - backslashes)
        + b"\\" * backslashes
        + b'","'
        + field.encode()
        + b'":"'
        + phrase
        + b'"}'
    )

    for split in range(len(payload) + 1):
        diagnostics = ssh._new_tunnel_diagnostics()
        ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
        ssh._append_tunnel_diagnostic(diagnostics, "stdout", payload[:split])
        ssh._append_tunnel_diagnostic(diagnostics, "stdout", payload[split:])
        rendered = ssh._redact_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
        assert phrase.decode() not in rendered
        assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) is None


@pytest.mark.parametrize("field", ("key", "access_key"))
def test_tunnel_diagnostics_keep_odd_escape_parity_when_overflow_precedes_the_quote(
    field: str,
) -> None:
    # This catches changing the neighboring overflow transition while fixing byte 256.
    phrase = b"address already in use"
    payload = (
        b'{"long":"'
        + b"x" * 254
        + b"\\"
        + b'"ordinary-tail","'
        + field.encode()
        + b'":"'
        + phrase
        + b'"}'
    )

    for split in range(len(payload) + 1):
        diagnostics = ssh._new_tunnel_diagnostics()
        ssh._activate_tunnel_diagnostics(diagnostics, key=_SESSION_KEY)
        ssh._append_tunnel_diagnostic(diagnostics, "stdout", payload[:split])
        ssh._append_tunnel_diagnostic(diagnostics, "stdout", payload[split:])
        assert ssh._collision_from_diagnostic(diagnostics, after_startup=True) is None


def test_redacted_log_writer_preserves_short_escaped_nonsecret_json_text() -> None:
    # This catches the protected-field fix rewriting an ordinary short escaped JSON string.
    payload = b'{"long":"short\\"ordinary","status":"visible"}\n'
    for split in range(len(payload) + 1):
        output = io.StringIO()
        writer = ssh._RedactedLogWriter(output, key=_SESSION_KEY)
        writer.feed(payload[:split])
        writer.feed(payload[split:])
        writer.flush()
        assert output.getvalue() == payload.decode()


def test_foreground_ssh_streams_redacted_remote_logs_after_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches dropping source-merged attached remote logs or leaking a split key.
    process = _LiveTunnelProcess(_startup_line())
    errors = io.StringIO()
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    def readiness(url: str, **unused: object) -> _Response:
        process.write_stdout(b"remote log ?key=" + _SESSION_KEY[:10].encode())
        process.write_stdout(_SESSION_KEY[10:].encode() + b"\nready log\n")
        process._close_writers()
        return _Response(200)

    assert run_ssh(
        _settings(tmp_path),
        stdout=io.StringIO(),
        stderr=errors,
        bootstrap_runner=lambda unused: _bootstrap_record(),
        popen_factory=lambda *args, **kwargs: process,
        readiness_opener=readiness,
        browser_open=lambda url: True,
    ) == 0

    assert "ready log" in errors.getvalue()
    assert _SESSION_KEY not in errors.getvalue()
    assert "<redacted>" in errors.getvalue()


@pytest.mark.parametrize(
    ("prefix", "suffix"),
    (
        (b"prefix ?key=", b"&next=visible\n"),
        (b"prefix &key=", b"#fragment\n"),
        (b'{"key":"', b'"}\n'),
        (b'{"access_key":"', b'"}\n'),
    ),
)
def test_redacted_log_writer_suppresses_long_split_secret_values(prefix: bytes, suffix: bytes) -> None:
    # This catches releasing a later slice of a generic query or JSON secret value.
    output = io.StringIO()
    writer = ssh._RedactedLogWriter(output, key=_SESSION_KEY)
    value = b"V" * 96

    writer.feed(prefix + value[:40])
    writer.feed(value[40:] + suffix)
    writer.flush()

    assert value.decode() not in output.getvalue()
    assert "<redacted>" in output.getvalue()


def test_redacted_log_writer_preserves_a_long_ordinary_line_losslessly() -> None:
    # This catches truncating visible attached output to the 4 KiB diagnostic tail.
    output = io.StringIO()
    writer = ssh._RedactedLogWriter(output, key=_SESSION_KEY)
    line = "log:" + "x" * 5001 + "\n"

    writer.feed(line.encode())
    writer.flush()

    assert output.getvalue() == line


def test_redacted_log_writer_suppresses_the_known_key_split_across_chunks() -> None:
    # This catches emitting either half of the exact known key before the next chunk arrives.
    output = io.StringIO()
    writer = ssh._RedactedLogWriter(output, key=_SESSION_KEY)

    writer.feed(b"before " + _SESSION_KEY[:10].encode())
    writer.feed(_SESSION_KEY[10:].encode() + b" after\n")
    writer.flush()

    assert _SESSION_KEY not in output.getvalue()
    assert output.getvalue() == "before <redacted> after\n"


@pytest.mark.parametrize("bytewise", (False, True))
def test_redacted_log_writer_preserves_valid_utf8_across_arbitrary_byte_splits(bytewise: bool) -> None:
    # This catches independent per-slice UTF-8 decoding corrupting ordinary remote logs.
    output = io.StringIO()
    writer = ssh._RedactedLogWriter(output, key=_SESSION_KEY)
    line = ("é漢🚀 ordinary log " * 64) + "\n"
    payload = line.encode("utf-8")

    if bytewise:
        for value in payload:
            writer.feed(bytes((value,)))
    else:
        writer.feed(payload)
    writer.flush()

    assert output.getvalue() == line


@pytest.mark.parametrize(
    "payload",
    (
        b"prefix ?KEY=" + b"V" * 96 + b"&tail=visible\n",
        b"prefix &KeY=" + b"V" * 96 + b"#tail\n",
        b'{"KEY":"' + b"V" * 96 + b'"}\n',
        b'{"Access_Key":"' + b"V" * 96 + b'"}\n',
    ),
)
def test_redacted_log_writer_redacts_case_insensitive_markers_at_every_split(payload: bytes) -> None:
    # This catches upper/mixed-case secret markers leaking at an arbitrary chunk boundary.
    secret = b"V" * 96
    for split in range(len(payload) + 1):
        output = io.StringIO()
        writer = ssh._RedactedLogWriter(output, key=_SESSION_KEY)
        writer.feed(payload[:split])
        writer.feed(payload[split:])
        writer.flush()
        assert secret.decode() not in output.getvalue()
        assert "<redacted>" in output.getvalue()


def test_redacted_log_writer_accepts_unbounded_json_whitespace_before_a_secret_value() -> None:
    # This catches abandoning a valid JSON key delimiter merely because whitespace is long.
    output = io.StringIO()
    writer = ssh._RedactedLogWriter(output, key=_SESSION_KEY)
    secret = "V" * 96
    payload = ('{"KEY"' + " " * 80 + ':"' + secret + '"}\n').encode()

    for value in payload:
        writer.feed(bytes((value,)))
    writer.flush()

    assert secret not in output.getvalue()
    assert "<redacted>" in output.getvalue()
    assert " " * 80 in output.getvalue()


def test_redacted_log_writers_keep_reverse_source_frames_separate() -> None:
    # This catches concatenating OpenSSH stderr with remote-app stdout into a false token.
    remote_output = io.StringIO()
    ssh_output = io.StringIO()
    remote_writer = ssh._RedactedLogWriter(remote_output, key=_SESSION_KEY)
    ssh_writer = ssh._RedactedLogWriter(ssh_output, key=_SESSION_KEY)
    suffix = _SESSION_KEY[10:]

    ssh_writer.feed(suffix.encode() + b"\n")
    remote_writer.feed(("?KEY=" + _SESSION_KEY + "\n").encode())
    remote_writer.flush()
    ssh_writer.flush()

    assert _SESSION_KEY not in remote_output.getvalue()
    assert "<redacted>" in remote_output.getvalue()
    assert ssh_output.getvalue() == suffix + "\n"


def test_redacted_log_writer_keeps_all_state_bounded_for_large_ordinary_and_secret_inputs() -> None:
    # This catches a line or unterminated value making the incremental sanitizer unbounded.
    writer = ssh._RedactedLogWriter(io.StringIO(), key=_SESSION_KEY)

    for payload in (b"x" * 20_000, b"?KEY=" + b"V" * 20_000, b'{"KEY":"' + b"V" * 20_000):
        writer.feed(payload)
        assert len(writer._plain) <= len(_SESSION_KEY)
        assert len(writer._field) <= ssh._MAX_JSON_FIELD_BYTES


@pytest.mark.parametrize(
    ("payload", "secret"),
    (
        (b"plain \xc3\xa9\xe6\xbc\xa2\xf0\x9f\x9a\x80\n", None),
        (b"?key=" + b"V" * 96 + b"&tail=1\n", b"V" * 96),
        (b"?KEY=" + b"V" * 96 + b"&tail=1\n", b"V" * 96),
        (b'{"key":"' + b"V" * 96 + b'"}\n', b"V" * 96),
        (b'{"Access_Key":"' + b"V" * 96 + b'"}\n', b"V" * 96),
        (b"before " + _SESSION_KEY.encode() + b" after\n", _SESSION_KEY.encode()),
    ),
)
def test_redacted_log_writer_fuzzes_every_two_chunk_split_and_bytewise_input(
    payload: bytes, secret: bytes | None,
) -> None:
    # This catches a state transition that is only unsafe at one byte boundary.
    for split in range(len(payload) + 1):
        output = io.StringIO()
        writer = ssh._RedactedLogWriter(output, key=_SESSION_KEY)
        writer.feed(payload[:split])
        writer.feed(payload[split:])
        writer.flush()
        if secret is None:
            assert output.getvalue() == payload.decode("utf-8")
        else:
            assert secret.decode() not in output.getvalue()

    output = io.StringIO()
    writer = ssh._RedactedLogWriter(output, key=_SESSION_KEY)
    for value in payload:
        writer.feed(bytes((value,)))
    writer.flush()
    if secret is None:
        assert output.getvalue() == payload.decode("utf-8")
    else:
        assert secret.decode() not in output.getvalue()


def test_redacted_log_writer_fuzzes_both_two_stream_delivery_orders() -> None:
    # Remote application output is merged at the remote shell; OpenSSH stderr
    # stays a distinct source frame and must not complete its token locally.
    prefix = b"?key=" + _SESSION_KEY[:10].encode()
    suffix = _SESSION_KEY[10:].encode() + b"\n"
    for ssh_chunk, remote_chunks in ((suffix, (prefix, suffix)), (prefix, (prefix, suffix))):
        remote_output = io.StringIO()
        ssh_output = io.StringIO()
        remote_writer = ssh._RedactedLogWriter(remote_output, key=_SESSION_KEY)
        ssh_writer = ssh._RedactedLogWriter(ssh_output, key=_SESSION_KEY)
        ssh_writer.feed(ssh_chunk)
        for chunk in remote_chunks:
            remote_writer.feed(chunk)
        remote_writer.flush()
        ssh_writer.flush()
        assert _SESSION_KEY not in remote_output.getvalue()
        assert remote_output.getvalue() == "?key=<redacted>\n"
        if ssh_chunk == prefix:
            assert ssh_output.getvalue() == "?key=<redacted>"
        else:
            assert ssh_output.getvalue() == ssh_chunk.decode()


def test_foreground_ssh_preserves_reverse_prestartup_ssh_stderr_as_a_separate_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches merging OpenSSH stderr with a remote-app stdout query fragment locally.
    suffix = _SESSION_KEY[10:]
    process = _LiveTunnelProcess(_startup_line() + ("?KEY=" + _SESSION_KEY[:10] + "\n").encode())
    process.write_stderr(suffix.encode())
    errors = io.StringIO()
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    def readiness(url: str, **unused: object) -> _Response:
        process._close_writers()
        return _Response(200)

    assert run_ssh(
        _settings(tmp_path),
        stdout=io.StringIO(),
        stderr=errors,
        bootstrap_runner=lambda unused: _bootstrap_record(),
        popen_factory=lambda *args, **kwargs: process,
        readiness_opener=readiness,
        browser_open=lambda url: True,
    ) == 0

    assert suffix in errors.getvalue()
    assert _SESSION_KEY not in errors.getvalue()


@pytest.mark.parametrize("fragment_length", range(1, 32))
def test_redacted_log_writer_preserves_every_ordinary_known_key_suffix_before_an_unrelated_query(
    fragment_length: int,
) -> None:
    # This catches guessing that ordinary text before a query marker is a cross-stream secret fragment.
    key = "AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    suffix = key[-fragment_length:]
    output = io.StringIO()
    writer = ssh._RedactedLogWriter(output, key=key)

    writer.feed(("ordinary:" + suffix + "?key=unrelated\n").encode())
    writer.flush()

    assert output.getvalue() == "ordinary:" + suffix + "?key=<redacted>\n"


@pytest.mark.parametrize("offset", range(1, 32))
def test_redacted_log_writer_preserves_every_ordinary_known_key_rotation(offset: int) -> None:
    # This catches treating a cyclic rotation as if it were the exact access key.
    key = "AbCdEfGhIjKlMnOpQrStUvWxYz012345"
    rotation = key[offset:] + key[:offset]
    output = io.StringIO()
    writer = ssh._RedactedLogWriter(output, key=key)

    writer.feed(("ordinary rotation:" + rotation + "\n").encode())
    writer.flush()

    assert output.getvalue() == "ordinary rotation:" + rotation + "\n"


def test_redacted_log_writer_handles_a_large_maximum_length_key_without_quadratic_rotation_scans() -> None:
    # This catches reintroducing per-byte scans over every rotation of a 128-character key.
    key = "A" * 128
    output = io.StringIO()
    writer = ssh._RedactedLogWriter(output, key=key)
    payload = b"x" * 20_000 + b"\n"
    started = time.perf_counter()

    writer.feed(payload)
    writer.flush()

    assert output.getvalue() == payload.decode()
    assert time.perf_counter() - started < 0.75


def test_foreground_ssh_displays_pre_startup_stderr_after_key_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches discarding useful remote stderr that arrived before startup JSON.
    process = _LiveTunnelProcess(_startup_line())
    process.write_stderr(b"remote pre-startup log\n")
    errors = io.StringIO()
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    def readiness(url: str, **unused: object) -> _Response:
        process._close_writers()
        return _Response(200)

    assert run_ssh(
        _settings(tmp_path),
        stdout=io.StringIO(),
        stderr=errors,
        bootstrap_runner=lambda unused: _bootstrap_record(),
        popen_factory=lambda *args, **kwargs: process,
        readiness_opener=readiness,
        browser_open=lambda url: True,
    ) == 0

    assert "remote pre-startup log" in errors.getvalue()


def test_foreground_ssh_cleans_up_if_visible_log_sink_breaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches a logging exception skipping selector, process, and pipe cleanup.
    process = _TunnelProcess(_startup_line() + b"remote log\n")
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    with pytest.raises(BrokenPipeError, match="closed test sink"):
        run_ssh(
            _settings(tmp_path),
            stdout=io.StringIO(),
            stderr=_BrokenOutput(),
            bootstrap_runner=lambda unused: _bootstrap_record(),
            popen_factory=lambda *args, **kwargs: process,
            readiness_opener=lambda *args, **kwargs: _Response(200),
        )

    assert process.terminated is True
    assert process.wait_calls == [1.0]
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_foreground_ssh_stops_after_five_automatic_collisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # This catches an unbounded retry loop when automatic SSH forwarding cannot bind.
    settings = _settings(tmp_path)
    processes = [
        _TunnelProcess(b"", b"bind [127.0.0.1]:43123: Address already in use\n", returncode=255)
        for unused in range(5)
    ]
    attempted = list(processes)
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    with pytest.raises(SSHLaunchError, match="listener collision"):
        run_ssh(
            settings,
            stdout=io.StringIO(),
            bootstrap_runner=lambda unused: _bootstrap_record(),
            popen_factory=lambda *args, **kwargs: processes.pop(0),
            readiness_opener=lambda *args, **kwargs: _Response(200),
        )

    assert processes == []
    assert all(process.wait_calls == [None] for process in attempted)


def test_foreground_ssh_does_not_retry_a_fixed_remote_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches changing the remote listener port after the user explicitly fixed it.
    settings = replace(_settings(tmp_path), remote_port=53123)
    process = _TunnelProcess(b"remote listener: address already in use\n", returncode=1)
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)
    bootstrap_calls: list[object] = []

    with pytest.raises(SSHLaunchError, match="listener collision"):
        run_ssh(
            settings,
            stdout=io.StringIO(),
            bootstrap_runner=lambda unused: bootstrap_calls.append(None) or _bootstrap_record(),
            popen_factory=lambda *args, **kwargs: process,
            readiness_opener=lambda *args, **kwargs: _Response(200),
        )

    assert bootstrap_calls == [None]
    assert process.wait_calls == [None]


def test_foreground_ssh_reaps_an_early_authentication_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # This catches SSH authentication failure leaving a child behind before startup output.
    process = _TunnelProcess(b"", b"Permission denied (publickey).\n", returncode=255)
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    with pytest.raises(SSHLaunchError, match="exited before"):
        run_ssh(
            _settings(tmp_path),
            stdout=io.StringIO(),
            bootstrap_runner=lambda unused: _bootstrap_record(),
            popen_factory=lambda *args, **kwargs: process,
            readiness_opener=lambda *args, **kwargs: _Response(200),
        )

    assert process.wait_calls == [None]
    assert process.terminated is False


def test_foreground_ssh_reaps_a_host_key_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # This catches host-key rejection taking a path that does not reap the SSH child.
    process = _TunnelProcess(b"", b"Host key verification failed.\n", returncode=255)
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    with pytest.raises(SSHLaunchError, match="Host key verification failed"):
        run_ssh(
            _settings(tmp_path),
            stdout=io.StringIO(),
            bootstrap_runner=lambda unused: _bootstrap_record(),
            popen_factory=lambda *args, **kwargs: process,
        )

    assert process.wait_calls == [None]


def test_foreground_ssh_redacts_post_startup_nonzero_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches a remote post-startup failure exposing the access key in its diagnostic.
    process = _LiveTunnelProcess(_startup_line())
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    def readiness(url: str, **unused: object) -> _Response:
        process.write_stdout(f"remote failure ?key={_SESSION_KEY}\n".encode(), returncode=1)
        raise urllib.error.URLError("refused")

    with pytest.raises(SSHLaunchError) as error:
        run_ssh(
            _settings(tmp_path),
            stdout=io.StringIO(),
            bootstrap_runner=lambda unused: _bootstrap_record(),
            popen_factory=lambda *args, **kwargs: process,
            readiness_opener=readiness,
        )

    assert _SESSION_KEY not in str(error.value)
    assert "<redacted>" in str(error.value)


def test_foreground_ssh_reaps_malformed_bounded_startup_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # This catches accepting or abandoning an oversized remote startup protocol record.
    monkeypatch.setattr(ssh, "MAX_STARTUP_BYTES", 4)
    process = _TunnelProcess(b"x" * 5)
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    with pytest.raises(SSHLaunchError, match="exceeded"):
        run_ssh(
            _settings(tmp_path),
            stdout=io.StringIO(),
            bootstrap_runner=lambda unused: _bootstrap_record(),
            popen_factory=lambda *args, **kwargs: process,
            readiness_opener=lambda *args, **kwargs: _Response(200),
        )

    assert process.terminated is True


def test_foreground_ssh_reaps_after_readiness_when_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches Ctrl+C leaving the foreground SSH child alive after a ready URL.
    process = _TunnelProcess(_startup_line())
    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)

    def interrupted_wait(timeout: float | None = None) -> int:
        if timeout is None and not process.terminated:
            raise KeyboardInterrupt
        if timeout == 1.0 and not process.killed:
            raise subprocess.TimeoutExpired(("ssh",), timeout)
        return 0

    process.wait = interrupted_wait  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        run_ssh(
            _settings(tmp_path),
            stdout=io.StringIO(),
            bootstrap_runner=lambda unused: _bootstrap_record(),
            popen_factory=lambda *args, **kwargs: process,
            readiness_opener=lambda *args, **kwargs: _Response(200),
            browser_open=lambda url: True,
        )

    assert process.terminated is True
    assert process.killed is True


def test_ssh_main_converts_expected_launch_errors_to_one_actionable_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches expected SSH setup errors escaping as tracebacks or multiple messages.
    errors = io.StringIO()
    monkeypatch.setattr(ssh, "parse_ssh_settings", lambda argv: (_ for _ in ()).throw(ValueError("bad destination")))

    assert ssh.main(["bad"], stdout=io.StringIO(), stderr=errors) == 2
    assert errors.getvalue() == "agent-bridge ssh: bad destination\n"


def test_ssh_main_returns_130_for_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches Ctrl+C escaping the public SSH command after its child cleanup path.
    monkeypatch.setattr(ssh, "parse_ssh_settings", lambda argv: _settings(tmp_path))
    monkeypatch.setattr(ssh, "run_ssh", lambda settings, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt))

    assert ssh.main(["workbox", "--repo", "/srv/demo"]) == 130


def test_ssh_help_lists_the_remote_connection_options(capsys: pytest.CaptureFixture[str]) -> None:
    # This catches the SSH entry point exposing a partial or local-only help surface.
    with pytest.raises(SystemExit, match="0"):
        ssh.main(["--help"])

    help_text = capsys.readouterr().out
    for option in ("--local-port", "--remote-port", "--python", "--no-open"):
        assert option in help_text


def test_wait_for_readiness_retries_connection_failures_until_http_200() -> None:
    # This catches opening the browser before the keyed forwarded endpoint is actually ready.
    results = iter((urllib.error.URLError("refused"), TimeoutError(), _Response(200)))
    sleeps: list[float] = []

    def opener(*unused: object, **unused_kwargs: object) -> _Response:
        result = next(results)
        if isinstance(result, BaseException):
            raise result
        return result

    wait_for_readiness(
        "http://127.0.0.1:43123/?key=" + "A" * 32,
        process=_Process(),  # type: ignore[arg-type]
        opener=opener,
        monotonic=lambda: 0.0,
        sleeper=sleeps.append,
    )

    assert sleeps == [0.1, 0.1]


def test_wait_for_readiness_accepts_keyed_root_cookie_admission_without_redirect_following() -> None:
    # This catches rejecting the real keyed-root 303 before the browser can use that exact URL.
    url = "http://127.0.0.1:43123/?key=" + "A" * 32
    headers = Message()
    headers["Location"] = "/"
    headers["Set-Cookie"] = "agent_bridge_session=" + "A" * 32 + "; HttpOnly; SameSite=strict"
    calls: list[str] = []

    def opener(request_url: str, **unused: object) -> object:
        calls.append(request_url)
        raise urllib.error.HTTPError(request_url, 303, "See Other", headers, io.BytesIO())

    wait_for_readiness(url, process=_Process(), opener=opener, monotonic=lambda: 0.0)

    assert calls == [url]


def test_loopback_readiness_opener_disables_environment_proxies_and_redirects() -> None:
    # This catches the keyed loopback URL being sent to an HTTP proxy or followed to another URL.
    opener = ssh._loopback_readiness_opener()
    proxy_handlers = [
        handler for handler in opener.handlers if isinstance(handler, urllib.request.ProxyHandler)
    ]

    assert proxy_handlers == []
    assert any(type(handler).__name__ == "_NoRedirectHandler" for handler in opener.handlers)


def test_run_ssh_defaults_to_the_loopback_keyed_root_opener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches production bypassing the proxy-disabled, no-redirect keyed-root opener.
    process = _TunnelProcess(_startup_line())
    captured: list[object] = []

    def readiness(value: str, *, opener: object, **unused: object) -> None:
        captured.append(opener)

    monkeypatch.setattr(ssh, "_select_local_port", lambda requested: 43123)
    monkeypatch.setattr(ssh, "wait_for_readiness", readiness)

    assert run_ssh(
        _settings(tmp_path),
        stdout=io.StringIO(),
        bootstrap_runner=lambda unused: _bootstrap_record(),
        popen_factory=lambda *args, **kwargs: process,
        browser_open=lambda unused: True,
    ) == 0
    assert captured == [None]


@pytest.mark.parametrize(
    ("opener", "process", "monotonic", "message"),
    (
        (
            lambda *args, **kwargs: (_ for _ in ()).throw(
                urllib.error.HTTPError("http://x", 403, "forbidden", {}, io.BytesIO())
            ),
            _Process(),
            lambda: 0.0,
            "access key",
        ),
        (lambda *args, **kwargs: _Response(500), _Process(), lambda: 0.0, "invalid status"),
        (lambda *args, **kwargs: _Response(200), _Process(1), lambda: 0.0, "SSH exited"),
        (
            lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("refused")),
            _Process(),
            iter((0.0, 1.0)).__next__,
            "deadline",
        ),
    ),
)
def test_wait_for_readiness_fails_closed_for_bad_status_exit_or_deadline(
    opener,
    process: _Process,
    monotonic,
    message: str,
) -> None:
    # This catches silently accepting unauthorized, failed, or abandoned SSH readiness checks.
    with pytest.raises(SSHLaunchError, match=message):
        wait_for_readiness(
            "http://127.0.0.1:43123/?key=" + "A" * 32,
            process=process,  # type: ignore[arg-type]
            opener=opener,
            monotonic=monotonic,
            sleeper=lambda seconds: None,
            timeout_seconds=1.0,
        )


@pytest.mark.parametrize(
    "marker",
    (b"?key=", b"&key=", b'"key":"', b'"access_key":"'),
)
def test_bootstrap_diagnostic_redacts_each_secret_marker_across_the_tail_boundary(
    tmp_path: Path, marker: bytes
) -> None:
    # This catches selecting a diagnostic tail after its access-key marker was discarded.
    settings = _settings(tmp_path)
    access_key = b"A" * 32
    diagnostic = marker + access_key + b"x" * (4096 - len(access_key))

    with pytest.raises(SSHLaunchError) as error:
        run_remote_bootstrap(
            settings,
            source_reader=lambda: "print('bootstrap')",
            process_runner=lambda argv: subprocess.CompletedProcess(argv, 1, b"", diagnostic),
        )

    assert access_key.decode() not in str(error.value)
    assert "<redacted>" in str(error.value)


@pytest.mark.parametrize(
    "url",
    (
        "http://127.0.0.1:53123/?key=%41" + "A" * 31,
        "http://127.0.0.1:53123/?key=" + "A" * 32 + "#",
        "http://127.0.0.1:53123/\n?key=" + "A" * 32,
        "http://127.0.0.1:053123/?key=" + "A" * 32,
        "http://127.0.0.1:53123/?key=" + "A" * 32 + "?",
    ),
)
def test_parse_remote_startup_rejects_noncanonical_keyed_loopback_urls(url: str) -> None:
    # This catches parser-normalized aliases reaching the localized browser endpoint.
    record = _startup_record()
    record["url"] = url

    with pytest.raises(SSHLaunchError, match="URL"):
        parse_remote_startup(
            json.dumps(record, separators=(",", ":")).encode() + b"\n",
            expected_remote_port=53123,
        )


@pytest.mark.parametrize(
    "field",
    (
        "fable_status",
        "fable_version",
        "sol_status",
        "sol_version",
        "ssh_command",
        "repository",
        "branch",
    ),
)
def test_parse_remote_startup_rejects_lone_surrogates_in_every_string_field(field: str) -> None:
    # This catches JSON escapes crossing the UTF-8 protocol boundary as a raw encoding error.
    record = _startup_record()
    record[field] = "\ud800"

    with pytest.raises(SSHLaunchError):
        parse_remote_startup(
            json.dumps(record, separators=(",", ":")).encode() + b"\n",
            expected_remote_port=53123,
        )


@pytest.mark.parametrize(
    "python",
    ("/runtime/\ud800/python", "/" + "x" * 4097),
)
def test_run_remote_bootstrap_rejects_non_utf8_or_unbounded_runtime_python(
    tmp_path: Path, python: str
) -> None:
    # This catches carrying a malformed remote executable into tunnel construction.
    settings = _settings(tmp_path)
    record = {
        "protocol": 1,
        "python": python,
        "remote_port": 53123,
        "version": "0.1.0",
    }

    with pytest.raises(SSHLaunchError, match="python"):
        run_remote_bootstrap(
            settings,
            source_reader=lambda: "print('bootstrap')",
            process_runner=lambda argv: subprocess.CompletedProcess(
                argv, 0, json.dumps(record).encode() + b"\n", b""
            ),
        )


class _FakeClock:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def test_wait_for_readiness_caps_request_timeout_to_its_remaining_deadline() -> None:
    # This catches a final request being allowed to run beyond the configured readiness deadline.
    clock = _FakeClock((0.0, 0.95, 1.0))
    timeouts: list[float] = []

    def opener(*unused: object, timeout: float) -> _Response:
        timeouts.append(timeout)
        raise urllib.error.URLError("refused")

    with pytest.raises(SSHLaunchError, match="deadline"):
        wait_for_readiness(
            "http://127.0.0.1:43123/?key=" + "A" * 32,
            process=_Process(),  # type: ignore[arg-type]
            opener=opener,
            monotonic=clock,
            sleeper=lambda seconds: pytest.fail("deadline failure must not sleep"),
            timeout_seconds=1.0,
        )

    assert timeouts == [pytest.approx(0.05)]


def test_wait_for_readiness_caps_retry_sleep_to_its_remaining_deadline() -> None:
    # This catches unconditional 100 ms sleeps after the readiness budget is almost exhausted.
    clock = _FakeClock((0.0, 0.95, 0.96, 1.0))
    sleeps: list[float] = []

    def opener(*unused: object, timeout: float) -> _Response:
        raise urllib.error.URLError("refused")

    with pytest.raises(SSHLaunchError, match="deadline"):
        wait_for_readiness(
            "http://127.0.0.1:43123/?key=" + "A" * 32,
            process=_Process(),  # type: ignore[arg-type]
            opener=opener,
            monotonic=clock,
            sleeper=sleeps.append,
            timeout_seconds=1.0,
        )

    assert sleeps == [pytest.approx(0.04)]


def test_readiness_progress_bounds_continuously_readable_tunnel_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches one noisy remote stream starving the monotonic readiness deadline.
    pipe = _CollectorPipe(10)
    key = _collector_key(pipe, "stdout")

    class AlwaysReadySelector:
        def select(self, timeout: float) -> list[tuple[SimpleNamespace, int]]:
            assert timeout == 0.0
            return [(key, 0)]

        def unregister(self, unused_pipe: _CollectorPipe) -> None:
            pytest.fail("a continuously readable stream must not be treated as EOF")

    selector = AlwaysReadySelector()
    diagnostics = ssh._new_tunnel_diagnostics()
    reads = 0

    def continuous_read(unused_descriptor: int, unused_size: int) -> bytes:
        nonlocal reads
        reads += 1
        if reads > 1:
            pytest.fail("one readiness progress call drained an unbounded stream")
        return b"remote log\n"

    monkeypatch.setattr(ssh.os, "read", continuous_read)

    with pytest.raises(SSHLaunchError, match="deadline"):
        wait_for_readiness(
            "http://127.0.0.1:43123/?key=" + "A" * 32,
            process=_Process(),  # type: ignore[arg-type]
            opener=lambda *unused, **unused_kwargs: pytest.fail("deadline must win before HTTP"),
            monotonic=iter((0.0, 1.0)).__next__,
            timeout_seconds=1.0,
            progress=lambda: ssh._drain_tunnel_streams(
                selector, diagnostics, log_writers=None, timeout=0.0,
            ),
        )

    assert reads == 1


@pytest.mark.parametrize(
    "response",
    (
        _Response(302),
        _Response(200, final_url="http://127.0.0.1:43124/?key=" + "A" * 32),
    ),
)
def test_wait_for_readiness_rejects_redirects_and_changed_final_urls(response: _Response) -> None:
    # This catches accepting HTTP success reached by following a redirect away from the keyed URL.
    url = "http://127.0.0.1:43123/?key=" + "A" * 32

    with pytest.raises(SSHLaunchError, match="invalid status"):
        wait_for_readiness(
            url,
            process=_Process(),  # type: ignore[arg-type]
            opener=lambda *unused, **unused_kwargs: response,
            monotonic=lambda: 0.0,
        )


class _CollectorPipe:
    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor
        self.closed = False

    def fileno(self) -> int:
        return self.descriptor

    def close(self) -> None:
        self.closed = True


class _CollectorProcess:
    def __init__(self, *, kill_after_terminate: bool = False) -> None:
        self.stdout = _CollectorPipe(10)
        self.stderr = _CollectorPipe(11)
        self.kill_after_terminate = kill_after_terminate
        self.terminated = False
        self.killed = False
        self.wait_calls: list[float | None] = []

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.kill_after_terminate and timeout == 1.0 and not self.killed:
            raise subprocess.TimeoutExpired(("ssh",), timeout)
        return 0


class _CollectorSelector:
    def __init__(self, batches: list[list[SimpleNamespace]]) -> None:
        self.batches = batches
        self.registered: dict[int, _CollectorPipe] = {}
        self.closed = False

    def register(self, pipe: _CollectorPipe, unused_events: int, data: str) -> None:
        self.registered[pipe.descriptor] = pipe

    def unregister(self, pipe: _CollectorPipe) -> None:
        self.registered.pop(pipe.descriptor)

    def get_map(self) -> dict[int, _CollectorPipe]:
        return self.registered

    def select(self, unused_timeout: float) -> list[tuple[SimpleNamespace, int]]:
        return [(key, 0) for key in self.batches.pop(0)]

    def close(self) -> None:
        self.closed = True


def _collector_key(pipe: _CollectorPipe, stream: str) -> SimpleNamespace:
    return SimpleNamespace(fd=pipe.descriptor, fileobj=pipe, data=stream)


def _run_collector(
    monkeypatch: pytest.MonkeyPatch,
    process: _CollectorProcess,
    selector: _CollectorSelector,
    chunks: dict[int, bytes],
    *,
    monotonic,
) -> subprocess.CompletedProcess[bytes]:
    monkeypatch.setattr(ssh.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(ssh.selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(ssh.os, "read", lambda descriptor, size: chunks[descriptor])
    monkeypatch.setattr(ssh.time, "monotonic", monotonic)
    return ssh._run_bounded_ssh(("ssh",))


@pytest.mark.parametrize("stream", ("stdout", "stderr"))
def test_bounded_collector_caps_each_stream_and_terminates_reaps_and_closes(
    monkeypatch: pytest.MonkeyPatch, stream: str
) -> None:
    # This catches dropping either independent stream cap or cleanup after an overflow.
    process = _CollectorProcess()
    pipe = getattr(process, stream)
    selector = _CollectorSelector([[_collector_key(pipe, stream)]])

    with pytest.raises(SSHLaunchError, match="exceeded"):
        _run_collector(
            monkeypatch,
            process,
            selector,
            {pipe.descriptor: b"x" * (64 * 1024 + 1)},
            monotonic=lambda: 0.0,
        )

    assert process.terminated is True
    assert process.killed is False
    assert process.wait_calls == [1.0]
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert selector.closed is True


def test_bounded_collector_deadline_terminates_reaps_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches a bootstrap deadline leaving a child or either pipe behind.
    process = _CollectorProcess()
    selector = _CollectorSelector([])

    with pytest.raises(SSHLaunchError, match="timed out"):
        _run_collector(
            monkeypatch,
            process,
            selector,
            {},
            monotonic=_FakeClock((0.0, ssh._BOOTSTRAP_TIMEOUT_SECONDS)),
        )

    assert process.terminated is True
    assert process.killed is False
    assert process.wait_calls == [1.0]
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert selector.closed is True


def test_bounded_collector_kills_after_a_failed_terminate_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches returning from overflow while an unresponsive SSH child remains alive.
    process = _CollectorProcess(kill_after_terminate=True)
    selector = _CollectorSelector([[_collector_key(process.stdout, "stdout")]])

    with pytest.raises(SSHLaunchError, match="exceeded"):
        _run_collector(
            monkeypatch,
            process,
            selector,
            {process.stdout.descriptor: b"x" * (64 * 1024 + 1)},
            monotonic=lambda: 0.0,
        )

    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls == [1.0, None]
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_bounded_collector_reaps_a_successful_process_and_closes_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches returning a completed bootstrap result without final wait or pipe cleanup.
    process = _CollectorProcess()
    selector = _CollectorSelector([
        [_collector_key(process.stdout, "stdout")],
        [_collector_key(process.stderr, "stderr")],
    ])

    result = _run_collector(
        monkeypatch,
        process,
        selector,
        {process.stdout.descriptor: b"", process.stderr.descriptor: b""},
        monotonic=lambda: 0.0,
    )

    assert result.returncode == 0
    assert process.terminated is False
    assert process.killed is False
    assert process.wait_calls == [ssh._BOOTSTRAP_TIMEOUT_SECONDS]
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert selector.closed is True


def test_bootstrap_selector_construction_failure_stops_reaps_and_closes_its_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches selector setup escaping after Popen and leaking the owned SSH child.
    process = _CollectorProcess()
    monkeypatch.setattr(ssh.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        ssh.selectors, "DefaultSelector", lambda: (_ for _ in ()).throw(OSError("fd limit"))
    )

    with pytest.raises(OSError, match="fd limit"):
        ssh._run_bounded_ssh(("ssh",))

    assert process.terminated is True
    assert process.wait_calls == [1.0]
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_bootstrap_selector_registration_failure_stops_reaps_closes_pipes_and_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches selector registration leaking a locally owned bootstrap child.
    process = _CollectorProcess()

    class FailingSelector:
        closed = False

        def register(self, *unused: object) -> None:
            raise OSError("fd limit")

        def close(self) -> None:
            self.closed = True

    selector = FailingSelector()
    monkeypatch.setattr(ssh.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(ssh.selectors, "DefaultSelector", lambda: selector)

    with pytest.raises(OSError, match="fd limit"):
        ssh._run_bounded_ssh(("ssh",))

    assert process.terminated is True
    assert process.wait_calls == [1.0]
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert selector.closed is True


@pytest.mark.parametrize(
    "marker",
    (b"?key=", b"&key=", b'"key":"', b'"access_key":"'),
)
def test_bootstrap_diagnostic_redacts_secret_tokens_split_across_streams(
    tmp_path: Path, marker: bytes
) -> None:
    # This catches combining independently redacted streams into a new raw key token.
    access_key = b"A" * 32

    with pytest.raises(SSHLaunchError) as error:
        run_remote_bootstrap(
            _settings(tmp_path),
            source_reader=lambda: "print('bootstrap')",
            process_runner=lambda argv: subprocess.CompletedProcess(
                argv, 1, marker, access_key
            ),
        )

    assert access_key.decode() not in str(error.value)
    assert "<redacted>" in str(error.value)


class _MutableClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _PollingProcess:
    def __init__(self, polls: tuple[int | None, ...]) -> None:
        self._polls = iter(polls)

    def poll(self) -> int | None:
        return next(self._polls)


def test_wait_for_readiness_rejects_http_200_that_returns_after_the_deadline() -> None:
    # This catches a successful request extending the readiness deadline before acceptance.
    clock = _MutableClock(0.0)

    def opener(*unused: object, **unused_kwargs: object) -> _Response:
        clock.now = 1.0
        return _Response(200)

    with pytest.raises(SSHLaunchError, match="deadline"):
        wait_for_readiness(
            "http://127.0.0.1:43123/?key=" + "A" * 32,
            process=_Process(),  # type: ignore[arg-type]
            opener=opener,
            monotonic=clock,
            timeout_seconds=1.0,
        )


def test_wait_for_readiness_rejects_http_200_after_ssh_exits() -> None:
    # This catches accepting an HTTP response after the owning SSH process has already died.
    with pytest.raises(SSHLaunchError, match="SSH exited"):
        wait_for_readiness(
            "http://127.0.0.1:43123/?key=" + "A" * 32,
            process=_PollingProcess((None, 1)),  # type: ignore[arg-type]
            opener=lambda *unused, **unused_kwargs: _Response(200),
            monotonic=lambda: 0.0,
        )
