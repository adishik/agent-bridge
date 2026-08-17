from __future__ import annotations

import io
import json
import os
import base64
import hashlib
from pathlib import Path
import py_compile
import socket
import stat
import subprocess
import shutil
import sys

import pytest

from agent_bridge import _remote_bootstrap as bootstrap


def _fake_venv_creator(path: Path) -> None:
    python = path / "bin" / "python"
    python.parent.mkdir(parents=True)
    shutil.copy2(Path(sys.executable).resolve(), python)
    python.chmod(0o700)


def _fake_runner(
    calls: list[tuple[str, ...]],
    *,
    reported_version: str,
    install_returncode: int = 0,
) -> object:
    def run(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(argv))
        if argv[2:5] == ("-m", "pip", "install"):
            return subprocess.CompletedProcess(argv, install_returncode, "", "")
        return subprocess.CompletedProcess(argv, 0, reported_version, "")

    return run


def _offline_venv(path: Path) -> Path:
    assert bootstrap.venv is not None
    bootstrap.venv.EnvBuilder(with_pip=False).create(path)
    return path / "bin" / "python"


def _write_offline_agent_bridge_fixture(
    site_packages: Path, *, include_main: bool = True,
) -> None:
    package = site_packages / "agent_bridge"
    package.mkdir(parents=True)
    contents = {
        "agent_bridge/__init__.py": b"\n",
        "agent_bridge/ssh.py": b"\n",
        "agent_bridge/_remote_bootstrap.py": b"\n",
        "agent_bridge/projects.py": b"VALUE = 'RELEASE'\n",
        "agent_bridge-0.1.0.dist-info/METADATA": (
            b"Metadata-Version: 2.1\nName: agent-bridge\nVersion: 0.1.0\n"
        ),
    }
    if include_main:
        contents["agent_bridge/__main__.py"] = (
            b"from . import projects\n"
            b"if __name__ == '__main__':\n"
            b"    print(projects.VALUE)\n"
        )
    records: list[str] = []
    for relative, content in contents.items():
        destination = site_packages / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
        records.append(f"{relative},sha256={digest.decode('ascii')},{len(content)}")
    (site_packages / "agent_bridge-0.1.0.dist-info" / "RECORD").write_text(
        "\n".join(records + ["agent_bridge-0.1.0.dist-info/RECORD,,"]) + "\n",
        encoding="utf-8",
    )


def test_remote_verifier_requires_real_recorded_agent_bridge_package(
    tmp_path: Path,
) -> None:
    # This catches accepting a metadata-only dist-info cache hit without installed package bytes.
    good_python = _offline_venv(tmp_path / "good")
    good_site = good_python.parents[1] / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    _write_offline_agent_bridge_fixture(good_site)
    metadata_only_python = _offline_venv(tmp_path / "metadata-only")
    metadata_only_site = (
        metadata_only_python.parents[1]
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    metadata_only = metadata_only_site / "agent_bridge-0.1.0.dist-info"
    metadata_only.mkdir(parents=True)
    (metadata_only / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: agent-bridge\nVersion: 0.1.0\n",
        encoding="utf-8",
    )

    verification = bootstrap._run((
        str(good_python),
        "-I",
        "-B",
        "-S",
        "-c",
        bootstrap._VERIFY_PROGRAM,
        str(good_python),
        str(good_python.parents[1]),
    ))
    assert verification.returncode == 0, verification.stderr
    assert bootstrap._reports_exact_version(good_python, "0.1.0", bootstrap._run) is True
    assert bootstrap._reports_exact_version(
        metadata_only_python, "0.1.0", bootstrap._run
    ) is False
    (good_site / "agent_bridge" / "ssh.py").write_text("tampered\n", encoding="utf-8")
    assert bootstrap._reports_exact_version(good_python, "0.1.0", bootstrap._run) is False


def test_remote_verifier_rejects_any_tampered_recorded_runtime_module(
    tmp_path: Path,
) -> None:
    """Every shipped package module must bind RECORD bytes before launch."""
    python = _offline_venv(tmp_path / "tampered-projects")
    site_packages = (
        python.parents[1]
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    _write_offline_agent_bridge_fixture(site_packages)
    projects = site_packages / "agent_bridge" / "projects.py"
    projects.write_text("VALUE = 'TAMPERED'\n", encoding="utf-8")

    launch = subprocess.run(
        (str(python), "-I", "-m", "agent_bridge"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert launch.stdout == "TAMPERED\n"
    assert bootstrap._reports_exact_version(python, "0.1.0", bootstrap._run) is False


def test_remote_verifier_rejects_an_unrecorded_package_file(tmp_path: Path) -> None:
    """No package file may exist outside the authenticated package RECORD."""
    python = _offline_venv(tmp_path / "unrecorded-module")
    site_packages = (
        python.parents[1]
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    _write_offline_agent_bridge_fixture(site_packages)
    (site_packages / "agent_bridge" / "unrecorded.py").write_text(
        "VALUE = 'untrusted'\n", encoding="utf-8"
    )

    assert bootstrap._reports_exact_version(python, "0.1.0", bootstrap._run) is False


def test_remote_verifier_rejects_timestamp_valid_unrecorded_bytecode(
    tmp_path: Path,
) -> None:
    """Authenticated source must not launch different cached bytecode."""
    python = _offline_venv(tmp_path / "poisoned-bytecode")
    site_packages = (
        python.parents[1]
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    _write_offline_agent_bridge_fixture(site_packages)
    projects = site_packages / "agent_bridge" / "projects.py"
    altered_source = tmp_path / "altered-projects.py"
    altered_source.write_text("VALUE = 'POISON!'\n", encoding="utf-8")
    assert altered_source.stat().st_size == projects.stat().st_size
    bytecode = (
        projects.parent
        / "__pycache__"
        / f"projects.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
    )
    bytecode.parent.mkdir()
    py_compile.compile(str(altered_source), cfile=str(bytecode), doraise=True)
    altered_stat = altered_source.stat()
    os.utime(projects, ns=(altered_stat.st_atime_ns, altered_stat.st_mtime_ns))

    launch = subprocess.run(
        (str(python), "-I", "-m", "agent_bridge"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert launch.stdout == "POISON!\n"
    assert bootstrap._reports_exact_version(python, "0.1.0", bootstrap._run) is False


def test_private_cache_validation_rejects_agent_bridge_bytecode(tmp_path: Path) -> None:
    """The local cache gate must also reject unrecorded package bytecode."""
    tree = tmp_path / "runtime"
    bytecode = (
        tree
        / "venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "agent_bridge"
        / "__pycache__"
        / "projects.cpython-312.pyc"
    )
    bytecode.parent.mkdir(parents=True)
    bytecode.write_bytes(b"unrecorded bytecode")

    with pytest.raises(RuntimeError, match="bytecode"):
        bootstrap._validate_private_tree(tree)


def test_remote_verifier_rejects_pth_startup_code_without_executing_it(
    tmp_path: Path,
) -> None:
    """Candidate verification must not run arbitrary site startup code."""
    python = _offline_venv(tmp_path / "pth-startup")
    site_packages = (
        python.parents[1]
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    _write_offline_agent_bridge_fixture(site_packages)
    sentinel = tmp_path / "pth-executed"
    (site_packages / "startup-side-effect.pth").write_text(
        f"import pathlib; pathlib.Path({str(sentinel)!r}).write_text('executed')\n",
        encoding="utf-8",
    )

    assert bootstrap._reports_exact_version(python, "0.1.0", bootstrap._run) is False
    assert sentinel.exists() is False


def test_remote_verifier_hashes_invalid_package_bytes_before_importing_them(
    tmp_path: Path,
) -> None:
    """A RECORD-mismatched package initializer must not run during verification."""
    python = _offline_venv(tmp_path / "import-after-record")
    site_packages = (
        python.parents[1]
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    _write_offline_agent_bridge_fixture(site_packages)
    sentinel = tmp_path / "invalid-init-executed"
    (site_packages / "agent_bridge" / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n",
        encoding="utf-8",
    )

    assert bootstrap._reports_exact_version(python, "0.1.0", bootstrap._run) is False
    assert sentinel.exists() is False


def test_remote_verifier_rejects_an_installed_release_without_its_executed_main_module(
    tmp_path: Path,
) -> None:
    # This catches accepting a cache that cannot run ``python -m agent_bridge``.
    python = _offline_venv(tmp_path / "missing-main")
    site_packages = (
        python.parents[1]
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    _write_offline_agent_bridge_fixture(site_packages, include_main=False)

    assert bootstrap._reports_exact_version(python, "0.1.0", bootstrap._run) is False
    launch = subprocess.run(
        (str(python), "-I", "-m", "agent_bridge"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert launch.returncode != 0


def test_remote_verifier_rejects_external_pth_imports_and_direct_url_provenance(
    tmp_path: Path,
) -> None:
    # This catches reusing a venv that imports a valid-looking release from outside it.
    external_site = tmp_path / "external-site"
    _write_offline_agent_bridge_fixture(external_site)
    external_python = _offline_venv(tmp_path / "pth")
    external_venv_site = (
        external_python.parents[1]
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    (external_venv_site / "external.pth").write_text(
        f"import sys; sys.path.insert(0, {str(external_site)!r})\n",
        encoding="utf-8",
    )
    direct_python = _offline_venv(tmp_path / "direct-url")
    direct_site = (
        direct_python.parents[1]
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    _write_offline_agent_bridge_fixture(direct_site)
    (direct_site / "agent_bridge-0.1.0.dist-info" / "direct_url.json").write_text(
        '{"url":"file:///tmp/agent-bridge"}\n', encoding="utf-8"
    )

    assert bootstrap._reports_exact_version(
        external_python, "0.1.0", bootstrap._run
    ) is False
    assert bootstrap._reports_exact_version(direct_python, "0.1.0", bootstrap._run) is False


def test_remote_verifier_rejects_a_cache_local_wrapper_to_an_external_venv(
    tmp_path: Path,
) -> None:
    """The selected cache executable, rather than its self-report, is authority."""
    external_python = _offline_venv(tmp_path / "external")
    external_site = (
        external_python.parents[1]
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    _write_offline_agent_bridge_fixture(external_site)

    selected_python = tmp_path / "cache" / "0.1.0" / "venv" / "bin" / "python"
    selected_python.parent.mkdir(parents=True)
    selected_python.write_text(
        f"#!/bin/sh\nexec {str(external_python)!r} \"$@\"\n", encoding="utf-8"
    )
    selected_python.chmod(0o700)

    external_launch = subprocess.run(
        (str(selected_python), "-I", "-c", "import agent_bridge; print(agent_bridge.__file__)"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert external_launch.returncode == 0
    assert str(external_site) in external_launch.stdout

    assert bootstrap._reports_exact_version(
        selected_python, "0.1.0", bootstrap._run
    ) is False


def test_remote_verifier_accepts_a_normal_symlinked_venv_interpreter(
    tmp_path: Path,
) -> None:
    """A standard venv symlink is authenticated from the bootstrap interpreter."""
    assert bootstrap.venv is not None
    venv_path = tmp_path / "symlinked"
    bootstrap.venv.EnvBuilder(with_pip=False, symlinks=True).create(venv_path)
    python = venv_path / "bin" / "python"
    site_packages = (
        venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )
    _write_offline_agent_bridge_fixture(site_packages)

    assert python.is_symlink()
    assert bootstrap._reports_exact_version(python, "0.1.0", bootstrap._run) is True


def test_cache_hit_rejects_a_verifier_only_wrapper_before_it_can_launch_elsewhere(
    tmp_path: Path,
) -> None:
    """A candidate must be trusted before its verifier command is ever executed."""
    external_python = _offline_venv(tmp_path / "external")
    external_site = (
        external_python.parents[1]
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    _write_offline_agent_bridge_fixture(external_site)
    (external_site / "agent_bridge" / "__main__.py").write_text(
        "print('EXTERNAL-LAUNCH')\n", encoding="utf-8"
    )
    cache_root = tmp_path / "runtime"
    candidate = cache_root / "0.1.0" / "venv" / "bin" / "python"
    candidate.parent.mkdir(parents=True)
    cache_root.chmod(0o700)
    candidate.write_text(
        "#!/bin/sh\n"
        "if [ \"$2\" = \"-c\" ]; then printf '0.1.0'; exit 0; fi\n"
        f"exec {str(external_python)!r} \"$@\"\n",
        encoding="utf-8",
    )
    candidate.chmod(0o700)
    launch = subprocess.run(
        (str(candidate), "-I", "-m", "agent_bridge"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert launch.stdout == "EXTERNAL-LAUNCH\n"
    calls: list[tuple[str, ...]] = []

    def trusted_copy_creator(path: Path) -> None:
        python = path / "bin" / "python"
        python.parent.mkdir(parents=True)
        shutil.copy2(Path(sys.executable).resolve(), python)
        python.chmod(0o700)

    bootstrap.ensure_runtime(
        "0.1.0",
        cache_root=cache_root,
        venv_creator=trusted_copy_creator,
        command_runner=_fake_runner(calls, reported_version="0.1.0"),
    )

    assert len([argv for argv in calls if argv[2:5] == ("-m", "pip", "install")]) == 1


def test_first_bootstrap_installs_exact_release_and_cache_hit_does_not_reinstall(
    tmp_path: Path,
) -> None:
    # This catches omitting the exact pinned installation or reinstalling a valid cache.
    calls: list[tuple[str, ...]] = []
    first = bootstrap.ensure_runtime(
        "0.1.0",
        cache_root=tmp_path / "runtime",
        venv_creator=_fake_venv_creator,
        command_runner=_fake_runner(calls, reported_version="0.1.0"),
    )
    second = bootstrap.ensure_runtime(
        "0.1.0",
        cache_root=tmp_path / "runtime",
        venv_creator=lambda path: pytest.fail("cache hit recreated venv"),
        command_runner=_fake_runner(calls, reported_version="0.1.0"),
    )

    assert first == second
    installs = [argv for argv in calls if argv[2:5] == ("-m", "pip", "install")]
    assert len(installs) == 1
    assert installs[0][1:] == (
        "-I",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--no-compile",
        "agent-bridge==0.1.0",
    )
    assert Path(installs[0][0]).parents[2].name.startswith(".0.1.0-")
    verifications = [
        argv for argv in calls if argv[1:5] == ("-I", "-B", "-S", "-c")
    ]
    assert len(verifications) == 2


def test_first_bootstrap_runs_ensurepip_in_the_owned_bounded_runner_before_pip(
    tmp_path: Path,
) -> None:
    # This catches venv creation spawning an unowned, unbounded ensurepip child.
    calls: list[tuple[str, ...]] = []

    bootstrap.ensure_runtime(
        "0.1.0",
        cache_root=tmp_path / "runtime",
        venv_creator=_fake_venv_creator,
        command_runner=_fake_runner(calls, reported_version="0.1.0"),
    )

    ensurepip_index = next(
        index for index, argv in enumerate(calls) if argv[2:4] == ("-m", "ensurepip")
    )
    pip_index = next(
        index for index, argv in enumerate(calls) if argv[2:5] == ("-m", "pip", "install")
    )
    assert calls[ensurepip_index][1:] == (
        "-I", "-m", "ensurepip", "--upgrade", "--default-pip"
    )
    assert ensurepip_index < pip_index


def test_first_bootstrap_removes_ensurepip_pth_before_verification_and_publication(
    tmp_path: Path,
) -> None:
    """Python 3.11's standard ensurepip hook must not reach the verifier."""
    hook: Path | None = None
    hook_missing_when_verified: list[bool] = []

    def creator(path: Path) -> None:
        nonlocal hook
        _fake_venv_creator(path)
        site_packages = (
            path
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        site_packages.mkdir(parents=True)
        hook = site_packages / "distutils-precedence.pth"
        hook.write_text("import _distutils_hack\n", encoding="utf-8")

    def runner(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        if argv[1:5] == ("-I", "-B", "-S", "-c"):
            assert hook is not None
            hook_missing_when_verified.append(not hook.exists())
        return subprocess.CompletedProcess(argv, 0, "0.1.0", "")

    runtime = bootstrap.ensure_runtime(
        "0.1.0",
        cache_root=tmp_path / "runtime",
        venv_creator=creator,
        command_runner=runner,
    )

    assert hook is not None
    assert hook_missing_when_verified == [True]
    assert not hook.exists()
    assert runtime == tmp_path / "runtime" / "0.1.0" / "venv" / "bin" / "python"


def test_bootstrap_rejects_an_unsafe_site_packages_pth_and_cleans_the_temporary(
    tmp_path: Path,
) -> None:
    """A non-regular `.pth` must fail before a partial runtime is published."""
    def creator(path: Path) -> None:
        _fake_venv_creator(path)
        unsafe = (
            path
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "unexpected.pth"
        )
        unsafe.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="site-packages .pth"):
        bootstrap.ensure_runtime(
            "0.1.0",
            cache_root=tmp_path / "runtime",
            venv_creator=creator,
            command_runner=_fake_runner([], reported_version="0.1.0"),
        )

    assert not (tmp_path / "runtime" / "0.1.0").exists()
    assert not list((tmp_path / "runtime").glob(".0.1.0-*"))


def test_partial_cache_is_rebuilt_before_installing(tmp_path: Path) -> None:
    # This catches treating a partial runtime as a cache hit.
    stale = tmp_path / "runtime" / "0.1.0" / "venv" / "bin" / "python"
    stale.parent.mkdir(parents=True)
    (tmp_path / "runtime").chmod(0o700)
    stale.write_text("partial", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    runtime = bootstrap.ensure_runtime(
        "0.1.0",
        cache_root=tmp_path / "runtime",
        venv_creator=_fake_venv_creator,
        command_runner=_fake_runner(calls, reported_version="0.1.0"),
    )

    assert runtime.stat().st_size == Path(sys.executable).resolve().stat().st_size
    assert len([argv for argv in calls if argv[2:5] == ("-m", "pip", "install")]) == 1


def test_malformed_published_final_is_removed_and_rebuilt_under_its_version_lock(
    tmp_path: Path,
) -> None:
    # This catches a rejected final symlink permanently blocking its exact release cache.
    root = tmp_path / "runtime"
    final = root / "0.1.0"
    final.mkdir(parents=True)
    root.chmod(0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    (final / "poison").symlink_to(outside, target_is_directory=True)
    calls: list[tuple[str, ...]] = []

    runtime = bootstrap.ensure_runtime(
        "0.1.0",
        cache_root=root,
        venv_creator=_fake_venv_creator,
        command_runner=_fake_runner(calls, reported_version="0.1.0"),
    )

    assert runtime.stat().st_size == Path(sys.executable).resolve().stat().st_size
    assert not (final / "poison").exists()
    assert len([argv for argv in calls if argv[2:5] == ("-m", "pip", "install")]) == 1


def test_retry_sweeps_only_its_abandoned_private_temporary_runtime_under_lock(
    tmp_path: Path,
) -> None:
    # This catches interrupted .VERSION-* runtime trees accumulating across rebuild attempts.
    root = tmp_path / "runtime"
    abandoned = root / ".0.1.0-abandoned"
    abandoned.mkdir(parents=True)
    root.chmod(0o700)
    (abandoned / "partial").write_text("partial", encoding="utf-8")

    bootstrap.ensure_runtime(
        "0.1.0",
        cache_root=root,
        venv_creator=_fake_venv_creator,
        command_runner=_fake_runner([], reported_version="0.1.0"),
    )

    assert not abandoned.exists()


def test_runtime_lifecycle_handles_an_interrupt_during_venv_creation_and_removes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches an unhandled signal during venv setup leaving a partial temporary runtime.
    holder: dict[str, object] = {}

    def install(interrupted: object) -> dict[int, object]:
        holder["interrupted"] = interrupted
        return {}

    def creator(path: Path) -> None:
        callback = holder.get("interrupted")
        if callback is not None:
            callback(bootstrap.signal.SIGTERM, None)  # type: ignore[operator]
        _fake_venv_creator(path)

    monkeypatch.setattr(bootstrap, "_install_cleanup_signal_handlers", install)

    with pytest.raises(RuntimeError, match="interrupted"):
        bootstrap.ensure_runtime(
            "0.1.0",
            cache_root=tmp_path / "runtime",
            venv_creator=creator,
            command_runner=_fake_runner([], reported_version="0.1.0"),
        )

    assert not list((tmp_path / "runtime").glob(".0.1.0-*"))


def test_runtime_lifecycle_replays_a_second_signal_after_private_temp_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches cleanup swallowing a later terminal signal or letting it interrupt removal.
    holder: dict[str, object] = {}
    original_remove = bootstrap._remove_private_tree
    injected = False

    def install(interrupted: object) -> dict[int, object]:
        holder["interrupted"] = interrupted
        return {}

    def creator(path: Path) -> None:
        holder["interrupted"](bootstrap.signal.SIGTERM, None)  # type: ignore[operator]

    def remove(path: Path, *, root: Path) -> None:
        nonlocal injected
        if path.name.startswith(".0.1.0-") and not injected:
            injected = True
            holder["interrupted"](bootstrap.signal.SIGINT, None)  # type: ignore[operator]
        original_remove(path, root=root)

    monkeypatch.setattr(bootstrap, "_install_cleanup_signal_handlers", install)
    monkeypatch.setattr(bootstrap, "_remove_private_tree", remove)

    with pytest.raises(RuntimeError, match=f"signal {bootstrap.signal.SIGINT}"):
        bootstrap.ensure_runtime(
            "0.1.0",
            cache_root=tmp_path / "runtime",
            venv_creator=creator,
            command_runner=_fake_runner([], reported_version="0.1.0"),
        )

    assert injected is True
    assert not list((tmp_path / "runtime").glob(".0.1.0-*"))


def test_install_failure_leaves_no_final_runtime(tmp_path: Path) -> None:
    # This catches publishing a failed or partial installation as a reusable runtime.
    calls: list[tuple[str, ...]] = []

    with pytest.raises(RuntimeError, match="installation failed"):
        bootstrap.ensure_runtime(
            "0.1.0",
            cache_root=tmp_path / "runtime",
            venv_creator=_fake_venv_creator,
            command_runner=_fake_runner(
                calls, reported_version="0.1.0", install_returncode=1
            ),
        )

    assert not (tmp_path / "runtime" / "0.1.0").exists()


def test_installed_version_mismatch_is_rejected_and_not_cached(tmp_path: Path) -> None:
    # This catches accepting a different release than the one requested remotely.
    with pytest.raises(RuntimeError, match="does not match"):
        bootstrap.ensure_runtime(
            "0.1.0",
            cache_root=tmp_path / "runtime",
            venv_creator=_fake_venv_creator,
            command_runner=_fake_runner([], reported_version="0.1.1"),
        )

    assert not (tmp_path / "runtime" / "0.1.0").exists()


def test_default_venv_creator_publishes_runtime_with_offline_fake_install(
    tmp_path: Path,
) -> None:
    # This catches rejecting the stdlib venv's known internal lib64 -> lib link.
    calls: list[tuple[str, ...]] = []

    runtime = bootstrap.ensure_runtime(
        "0.1.0",
        cache_root=tmp_path / "runtime",
        command_runner=_fake_runner(calls, reported_version="0.1.0"),
    )

    assert runtime.is_file()
    assert not (runtime.parents[1] / "lib64").exists()
    assert len([argv for argv in calls if argv[2:5] == ("-m", "pip", "install")]) == 1


def test_only_stdlib_lib64_to_lib_link_is_removed_before_validation(tmp_path: Path) -> None:
    # This catches allowing an altered internal symlink while accommodating stdlib venv.
    def creator(path: Path) -> None:
        _fake_venv_creator(path)
        (path / "elsewhere").mkdir()
        (path / "lib64").symlink_to("elsewhere", target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        bootstrap.ensure_runtime(
            "0.1.0",
            cache_root=tmp_path / "runtime",
            venv_creator=creator,
            command_runner=_fake_runner([], reported_version="0.1.0"),
        )

    assert not (tmp_path / "runtime" / "0.1.0").exists()
    assert not list((tmp_path / "runtime").glob(".0.1.0-*"))


def test_rejected_tree_removal_does_not_follow_a_directory_swapped_to_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches recursive pathname cleanup chmodding/scanning an outside symlink target.
    root = tmp_path / "runtime"
    rejected = root / "rejected"
    nested = rejected / "nested"
    nested.mkdir(parents=True)
    root.chmod(0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.chmod(0o755)
    sentinel = outside / "sentinel"
    sentinel.write_text("outside bytes survive", encoding="utf-8")
    original_rename = bootstrap.os.rename
    swapped = False

    def rename_at(source: object, destination: object, *args: object, **kwargs: object) -> None:
        nonlocal swapped
        if source == "nested" and kwargs.get("src_dir_fd") is not None and not swapped:
            original_rename("nested", "nested-before-swap", src_dir_fd=kwargs["src_dir_fd"],
                            dst_dir_fd=kwargs["src_dir_fd"])
            os.symlink(outside, "nested", dir_fd=kwargs["src_dir_fd"])
            swapped = True
        original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(bootstrap.os, "rename", rename_at)

    with pytest.raises((OSError, RuntimeError)):
        bootstrap._remove_private_tree(rejected, root=root)

    assert swapped is True
    assert sentinel.read_text(encoding="utf-8") == "outside bytes survive"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o755


def test_rejected_tree_removal_rejects_a_root_ancestor_swapped_to_outside_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches reopening a previously accepted cache root through a swapped ancestor.
    original_parent = tmp_path / "original-parent"
    root = original_parent / "runtime"
    original_rejected = root / "rejected"
    original_rejected.mkdir(parents=True)
    root.chmod(0o700)
    outside_parent = tmp_path / "outside-parent"
    outside_root = outside_parent / "runtime"
    outside_rejected = outside_root / "rejected"
    outside_rejected.mkdir(parents=True)
    outside_root.chmod(0o700)
    sentinel = outside_rejected / "sentinel"
    sentinel.write_text("outside bytes survive", encoding="utf-8")
    original_open = bootstrap.os.open
    swapped = False

    def open_root(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == root and not swapped:
            original_parent.rename(tmp_path / "original-parent-before-swap")
            original_parent.symlink_to(outside_parent, target_is_directory=True)
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(bootstrap.os, "open", open_root)

    with pytest.raises(RuntimeError, match="changed"):
        bootstrap._remove_private_tree(original_rejected, root=root)

    assert swapped is True
    assert sentinel.read_text(encoding="utf-8") == "outside bytes survive"
    assert stat.S_IMODE(outside_root.stat().st_mode) == 0o700


def test_rejected_tree_removal_rejects_a_child_directory_replaced_by_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches a nofollow open deleting a different directory than lstat inspected.
    root = tmp_path / "runtime"
    nested = root / "rejected" / "nested"
    nested.mkdir(parents=True)
    root.chmod(0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.chmod(0o755)
    sentinel = outside / "sentinel"
    sentinel.write_text("outside bytes survive", encoding="utf-8")
    original_rename = bootstrap.os.rename
    swapped = False

    def rename_child(source: object, destination: object, *args: object, **kwargs: object) -> None:
        nonlocal swapped
        if source == "nested" and kwargs.get("src_dir_fd") is not None and not swapped:
            original_rename("nested", "nested-before-swap", src_dir_fd=kwargs["src_dir_fd"],
                            dst_dir_fd=kwargs["src_dir_fd"])
            original_rename(outside, "nested", dst_dir_fd=kwargs["src_dir_fd"])
            swapped = True
        original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(bootstrap.os, "rename", rename_child)

    with pytest.raises(RuntimeError, match="changed"):
        bootstrap._remove_private_tree(root / "rejected", root=root)

    assert swapped is True
    matches = list(root.rglob(sentinel.name))
    assert any(path.read_text(encoding="utf-8") == "outside bytes survive" for path in matches)
    assert any(stat.S_IMODE(path.parent.stat().st_mode) == 0o755 for path in matches)


def test_rejected_tree_removal_preserves_a_leaf_swapped_during_quarantine_detach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checked leaf must be identity-bound before any destructive unlink."""
    root = tmp_path / "runtime"
    rejected = root / "rejected"
    rejected.mkdir(parents=True)
    root.chmod(0o700)
    leaf = rejected / "leaf"
    leaf.write_text("rejected bytes", encoding="utf-8")
    parked = rejected / "leaf-parked"
    unrelated = rejected / "unrelated"
    unrelated.write_text("outside bytes survive", encoding="utf-8")
    original_rename = bootstrap.os.rename
    swapped = False

    def swap_before_detach(
        source: object, destination: object, *args: object, **kwargs: object,
    ) -> None:
        nonlocal swapped
        if source == "leaf" and kwargs.get("src_dir_fd") is not None and not swapped:
            original_rename("leaf", "leaf-parked", src_dir_fd=kwargs["src_dir_fd"],
                            dst_dir_fd=kwargs["src_dir_fd"])
            original_rename("unrelated", "leaf", src_dir_fd=kwargs["src_dir_fd"],
                            dst_dir_fd=kwargs["src_dir_fd"])
            swapped = True
        original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(bootstrap.os, "rename", swap_before_detach)

    with pytest.raises(RuntimeError, match="changed"):
        bootstrap._remove_private_tree(rejected, root=root)

    assert swapped is True
    assert any(
        candidate.read_text(encoding="utf-8") == "rejected bytes"
        for candidate in root.rglob(parked.name)
    )
    assert any(
        candidate.is_file()
        and candidate.read_text(encoding="utf-8") == "outside bytes survive"
        for candidate in root.rglob("*")
    )


def test_rejected_tree_removal_preserves_final_directory_swapped_during_quarantine_detach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final rmdir target must be bound before the directory can be removed."""
    root = tmp_path / "runtime"
    rejected = root / "rejected"
    rejected.mkdir(parents=True)
    root.chmod(0o700)
    parked = root / "rejected-parked"
    unrelated = root / "unrelated"
    unrelated.mkdir()
    unrelated_identity = unrelated.stat()
    original_rename = bootstrap.os.rename
    swapped = False

    def swap_before_detach(
        source: object, destination: object, *args: object, **kwargs: object,
    ) -> None:
        nonlocal swapped
        if source == "rejected" and kwargs.get("src_dir_fd") is not None and not swapped:
            original_rename("rejected", "rejected-parked", src_dir_fd=kwargs["src_dir_fd"],
                            dst_dir_fd=kwargs["src_dir_fd"])
            original_rename("unrelated", "rejected", src_dir_fd=kwargs["src_dir_fd"],
                            dst_dir_fd=kwargs["src_dir_fd"])
            swapped = True
        original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(bootstrap.os, "rename", swap_before_detach)

    with pytest.raises(RuntimeError, match="changed"):
        bootstrap._remove_private_tree(rejected, root=root)

    assert swapped is True
    assert parked.is_dir()
    assert any(
        candidate.is_dir()
        and candidate.stat().st_dev == unrelated_identity.st_dev
        and candidate.stat().st_ino == unrelated_identity.st_ino
        for candidate in root.iterdir()
    )


def test_immutable_version_mismatch_is_removed_and_rebuilt(tmp_path: Path) -> None:
    # This catches inability to invalidate a published immutable runtime.
    calls: list[tuple[str, ...]] = []
    first = bootstrap.ensure_runtime(
        "0.1.0",
        cache_root=tmp_path / "runtime",
        venv_creator=_fake_venv_creator,
        command_runner=_fake_runner(calls, reported_version="0.1.0"),
    )
    versions = iter(("0.1.1", "0.1.0"))

    def mismatch_runner(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[2:5] == ("-m", "pip", "install"):
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[2:4] == ("-m", "ensurepip"):
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, next(versions), "")

    second = bootstrap.ensure_runtime(
        "0.1.0",
        cache_root=tmp_path / "runtime",
        venv_creator=_fake_venv_creator,
        command_runner=mismatch_runner,
    )

    assert second == first
    assert len([argv for argv in calls if argv[2:5] == ("-m", "pip", "install")]) == 2


def test_immutable_preparation_failure_does_not_publish_a_final_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This catches a chmod failure exposing a writable or partial final runtime.
    original_chmod = bootstrap.os.chmod
    failed = False

    def fail_for_venv(path: str | Path, mode: int, **kwargs: object) -> None:
        nonlocal failed
        if Path(path).name == "venv" and not failed:
            failed = True
            raise PermissionError("chmod denied")
        original_chmod(path, mode, **kwargs)

    monkeypatch.setattr(bootstrap.os, "chmod", fail_for_venv)

    with pytest.raises(PermissionError, match="chmod denied"):
        bootstrap.ensure_runtime(
            "0.1.0",
            cache_root=tmp_path / "runtime",
            venv_creator=_fake_venv_creator,
            command_runner=_fake_runner([], reported_version="0.1.0"),
        )

    assert not (tmp_path / "runtime" / "0.1.0").exists()


def test_unlaunchable_cached_python_is_removed_and_rebuilt(tmp_path: Path) -> None:
    # This catches propagating a cache-probe execution error instead of rebuilding.
    stale = tmp_path / "runtime" / "0.1.0" / "venv" / "bin" / "python"
    stale.parent.mkdir(parents=True)
    (tmp_path / "runtime").chmod(0o700)
    stale.write_text("not a Python executable", encoding="utf-8")
    stale.chmod(0o700)
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[1:3] == ("-I", "-c") and argv[0] == str(stale):
            raise OSError("Exec format error")
        if argv[2:5] == ("-m", "pip", "install"):
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "0.1.0", "")

    runtime = bootstrap.ensure_runtime(
        "0.1.0",
        cache_root=tmp_path / "runtime",
        venv_creator=_fake_venv_creator,
        command_runner=runner,
    )

    assert runtime.stat().st_size == Path(sys.executable).resolve().stat().st_size
    assert len([argv for argv in calls if argv[2:5] == ("-m", "pip", "install")]) == 1


def test_successful_runtime_is_immutable_after_atomic_publish(tmp_path: Path) -> None:
    # This catches leaving a published version writable after the atomic final rename.
    runtime = bootstrap.ensure_runtime(
        "0.1.0",
        cache_root=tmp_path / "runtime",
        venv_creator=_fake_venv_creator,
        command_runner=_fake_runner([], reported_version="0.1.0"),
    )

    assert runtime.stat().st_mode & 0o222 == 0
    assert runtime.parents[2].stat().st_mode & 0o222 == 0


def test_cache_root_symlink_is_rejected_before_use(tmp_path: Path) -> None:
    # This catches following an attacker-controlled cache-root symlink.
    target = tmp_path / "target"
    target.mkdir()
    cache_root = tmp_path / "runtime"
    cache_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        bootstrap.ensure_runtime(
            "0.1.0",
            cache_root=cache_root,
            venv_creator=_fake_venv_creator,
            command_runner=_fake_runner([], reported_version="0.1.0"),
        )

    assert target.exists()


def test_non_owner_cache_root_is_rejected_before_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This catches use of a cache location not controlled by the effective user.
    cache_root = tmp_path / "runtime"
    cache_root.mkdir()
    cache_root.chmod(0o700)
    monkeypatch.setattr(bootstrap.os, "geteuid", lambda: cache_root.stat().st_uid + 1)

    with pytest.raises(RuntimeError, match="owner"):
        bootstrap.ensure_runtime(
            "0.1.0",
            cache_root=cache_root,
            venv_creator=_fake_venv_creator,
            command_runner=_fake_runner([], reported_version="0.1.0"),
        )


def test_group_or_other_writable_cache_root_is_rejected_before_locking(tmp_path: Path) -> None:
    # This catches an attacker-swappable cache authority being accepted before chmod narrows it.
    cache_root = tmp_path / "runtime"
    cache_root.mkdir(mode=0o700)
    cache_root.chmod(0o777)

    with pytest.raises(RuntimeError, match="externally writable"):
        bootstrap.ensure_runtime(
            "0.1.0",
            cache_root=cache_root,
            venv_creator=_fake_venv_creator,
            command_runner=_fake_runner([], reported_version="0.1.0"),
        )


def test_cache_ancestor_accepts_only_root_or_effective_user_owned_sticky_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches treating an arbitrary attacker's sticky directory like trusted /tmp.
    effective_uid = 1000
    monkeypatch.setattr(bootstrap.os, "geteuid", lambda: effective_uid)
    base = (stat.S_IFDIR | 0o1777, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    root_owned = os.stat_result(base[:4] + (0,) + base[5:])
    self_owned = os.stat_result(base[:4] + (effective_uid,) + base[5:])
    foreign_owned = os.stat_result(base[:4] + (effective_uid + 1,) + base[5:])

    assert bootstrap._require_safe_cache_ancestor(root_owned) is None
    assert bootstrap._require_safe_cache_ancestor(self_owned) is None
    with pytest.raises(RuntimeError, match="externally writable"):
        bootstrap._require_safe_cache_ancestor(foreign_owned)


def test_cache_ancestor_rejects_a_foreign_owned_nonwritable_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches treating a presently-0755 foreign parent as durable cache authority.
    monkeypatch.setattr(bootstrap.os, "geteuid", lambda: 1000)
    foreign_owned = os.stat_result((stat.S_IFDIR | 0o755, 0, 0, 0, 1001, 0, 0, 0, 0, 0))

    with pytest.raises(RuntimeError, match="owner"):
        bootstrap._require_safe_cache_ancestor(foreign_owned)


@pytest.mark.parametrize("version", ("", "../0.1.0", "release version", "v" * 65))
def test_ensure_runtime_rejects_unsafe_versions(tmp_path: Path, version: str) -> None:
    # This catches version text escaping the one-directory-per-release cache layout.
    with pytest.raises(ValueError, match="version"):
        bootstrap.ensure_runtime(
            version,
            cache_root=tmp_path / "runtime",
            venv_creator=_fake_venv_creator,
            command_runner=_fake_runner([], reported_version="0.1.0"),
        )


def test_bootstrap_main_rejects_python_before_311(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches trying to bootstrap on a remote interpreter below the support floor.
    monkeypatch.setattr(bootstrap.sys, "version_info", (3, 10, 14))

    with pytest.raises(RuntimeError, match="Python 3.11"):
        bootstrap.bootstrap_main(["0.1.0", "53123"])


def test_ensure_runtime_rejects_venv_without_a_python_executable(tmp_path: Path) -> None:
    # This catches proceeding to pip when venv setup did not provide an interpreter.
    with pytest.raises(RuntimeError, match="venv.*python"):
        bootstrap.ensure_runtime(
            "0.1.0",
            cache_root=tmp_path / "runtime",
            venv_creator=lambda path: path.mkdir(parents=True),
            command_runner=_fake_runner([], reported_version="0.1.0"),
        )


def test_create_venv_reports_an_unavailable_venv_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This catches an opaque attribute failure on Python installations without venv.
    monkeypatch.setattr(bootstrap, "venv", None)

    with pytest.raises(RuntimeError, match="does not provide venv"):
        bootstrap._create_venv(tmp_path / "venv")


def test_ensure_runtime_rejects_missing_pip(tmp_path: Path) -> None:
    # This catches accepting a venv whose pip installation command failed.
    with pytest.raises(RuntimeError, match="installation failed"):
        bootstrap.ensure_runtime(
            "0.1.0",
            cache_root=tmp_path / "runtime",
            venv_creator=_fake_venv_creator,
            command_runner=_fake_runner([], reported_version="0.1.0", install_returncode=1),
        )


def test_select_remote_port_keeps_a_requested_port_without_binding() -> None:
    # This catches changing a caller's already validated fixed remote port.
    assert bootstrap.select_remote_port(
        53123,
        socket_factory=lambda *args: pytest.fail("fixed port should not probe"),
    ) == 53123


def test_select_remote_port_returns_available_high_loopback_candidate() -> None:
    # This catches selecting a low or publicly-bound automatic remote port.
    binds: list[tuple[str, int]] = []

    class Probe:
        def __enter__(self) -> Probe:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def bind(self, address: tuple[str, int]) -> None:
            binds.append(address)

    assert bootstrap.select_remote_port(
        0, socket_factory=lambda *args: Probe(), candidate_factory=lambda: 53123
    ) == 53123
    assert binds == [("127.0.0.1", 53123)]


def test_select_remote_port_retries_collision_then_fails_after_32_attempts() -> None:
    # This catches unbounded collision retries or returning an occupied candidate.
    candidates = iter(range(49152, 49184))

    class BusyProbe:
        def __enter__(self) -> BusyProbe:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def bind(self, address: tuple[str, int]) -> None:
            raise OSError("busy")

    with pytest.raises(RuntimeError, match="could not select"):
        bootstrap.select_remote_port(
            0,
            socket_factory=lambda *args: BusyProbe(),
            candidate_factory=lambda: next(candidates),
        )


def test_select_remote_port_retries_a_collision_before_returning_a_candidate() -> None:
    # This catches stopping after the first occupied high loopback candidate.
    candidates = iter((53123, 53124))
    probes = 0

    class Probe:
        def __enter__(self) -> Probe:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def bind(self, address: tuple[str, int]) -> None:
            nonlocal probes
            probes += 1
            if probes == 1:
                raise OSError("busy")

    assert bootstrap.select_remote_port(
        0,
        socket_factory=lambda *args: Probe(),
        candidate_factory=lambda: next(candidates),
    ) == 53124
    assert probes == 2


@pytest.mark.parametrize("port", (True, "53123", -1, 65536))
def test_select_remote_port_rejects_invalid_requested_ports(port: object) -> None:
    # This catches non-integer and out-of-range port authorities.
    with pytest.raises(ValueError, match="remote port"):
        bootstrap.select_remote_port(port)  # type: ignore[arg-type]


def test_bootstrap_main_emits_one_compact_protocol_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # This catches leaking bootstrap details instead of emitting the fixed protocol record.
    runtime_python = tmp_path / "runtime" / "0.1.0" / "venv" / "bin" / "python"
    output = io.StringIO()
    monkeypatch.setattr(bootstrap, "ensure_runtime", lambda version: runtime_python)

    assert bootstrap.bootstrap_main(["0.1.0", "53123"], stdout=output) == 0
    assert output.getvalue() == (
        '{"protocol":1,"python":"'
        + str(runtime_python)
        + '","remote_port":53123,"version":"0.1.0"}\n'
    )
    assert json.loads(output.getvalue()) == {
        "protocol": 1,
        "python": str(runtime_python),
        "remote_port": 53123,
        "version": "0.1.0",
    }


class _RuntimePipe:
    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor
        self.closed = False

    def fileno(self) -> int:
        return self.descriptor

    def close(self) -> None:
        self.closed = True


class _RuntimeProcess:
    def __init__(self) -> None:
        self.pid = 741
        self.stdout = _RuntimePipe(10)
        self.stderr = _RuntimePipe(11)
        self.returncode: int | None = None
        self.wait_calls: list[float | None] = []
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = 137


class _RuntimeSelector:
    def __init__(self, batches: list[list[tuple[_RuntimePipe, str]]]) -> None:
        self.batches = batches
        self.registered: dict[int, _RuntimePipe] = {}
        self.timeouts: list[float] = []
        self.closed = False

    def register(self, pipe: _RuntimePipe, unused_events: int, data: str) -> None:
        self.registered[pipe.descriptor] = pipe

    def unregister(self, pipe: _RuntimePipe) -> None:
        self.registered.pop(pipe.descriptor)

    def get_map(self) -> dict[int, _RuntimePipe]:
        return self.registered

    def select(self, timeout: float) -> list[tuple[object, int]]:
        self.timeouts.append(timeout)
        return [
            (type("Key", (), {"fd": pipe.descriptor, "fileobj": pipe, "data": stream})(), 0)
            for pipe, stream in self.batches.pop(0)
        ]

    def close(self) -> None:
        self.closed = True


def test_runtime_install_uses_a_separate_budget_and_terminates_its_owned_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches first install sharing verification limits or killing only its direct pip parent.
    process = _RuntimeProcess()
    selector = _RuntimeSelector([[]])
    popen_kwargs: dict[str, object] = {}
    group_signals: list[tuple[int, int]] = []
    clock = iter((0.0, 0.0)).__next__

    monkeypatch.setattr(
        bootstrap.subprocess,
        "Popen",
        lambda *args, **kwargs: popen_kwargs.update(kwargs) or process,
    )
    monkeypatch.setattr(bootstrap.selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(bootstrap.time, "monotonic", clock)
    monkeypatch.setattr(bootstrap.os, "killpg", lambda pid, signal: group_signals.append((pid, signal)))

    with pytest.raises(RuntimeError, match="installation timed out"):
        bootstrap._run(("/runtime/python", "-I", "-m", "pip", "install", "agent-bridge==0.1.0"))

    assert popen_kwargs["start_new_session"] is True
    assert selector.timeouts == [300.0]
    assert group_signals == [
        (741, bootstrap.signal.SIGTERM),
        (741, bootstrap.signal.SIGKILL),
    ]
    assert process.wait_calls == [1.0]
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert selector.closed is True


def test_runtime_interrupt_during_spawn_cleans_the_exact_owned_group_and_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches a signal between child creation and the old cleanup-protected region.
    process = _RuntimeProcess()
    selector = _RuntimeSelector([])
    handlers: dict[str, object] = {}
    group_signals: list[tuple[int, int]] = []

    def install(*args: object) -> dict[int, object]:
        if not args:
            raise RuntimeError("remote bootstrap interrupted by signal")
        handlers["interrupted"] = args[0]
        return {}

    def spawn(*args: object, **kwargs: object) -> _RuntimeProcess:
        handler = handlers.get("interrupted")
        if handler is not None:
            handler(bootstrap.signal.SIGTERM, None)  # type: ignore[operator]
        return process

    monkeypatch.setattr(bootstrap, "_install_cleanup_signal_handlers", install)
    monkeypatch.setattr(bootstrap.subprocess, "Popen", spawn)
    monkeypatch.setattr(bootstrap.selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(
        bootstrap.os, "killpg", lambda pid, value: group_signals.append((pid, value))
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        bootstrap._run(("/runtime/python", "-I", "-m", "pip", "install", "agent-bridge==0.1.0"))

    assert group_signals == [
        (process.pid, bootstrap.signal.SIGTERM),
        (process.pid, bootstrap.signal.SIGKILL),
    ]
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_runtime_repeated_interrupt_does_not_preempt_exact_group_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches a second signal escaping the cleanup handler before SIGKILL/reap.
    process = _RuntimeProcess()
    selector = _RuntimeSelector([[]])
    holder: dict[str, object] = {}
    signals: list[int] = []

    def install(interrupted: object) -> dict[int, object]:
        holder["interrupted"] = interrupted
        return {}

    def killpg(unused_pid: int, signum: int) -> None:
        signals.append(signum)
        if signum == bootstrap.signal.SIGTERM:
            holder["interrupted"](bootstrap.signal.SIGTERM, None)  # type: ignore[operator]

    monkeypatch.setattr(bootstrap, "_install_cleanup_signal_handlers", install)
    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(bootstrap.selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(bootstrap.time, "monotonic", iter((0.0, 0.0)).__next__)
    monkeypatch.setattr(bootstrap.os, "killpg", killpg)

    with pytest.raises(RuntimeError, match="interrupted"):
        bootstrap._run(("/runtime/python", "-I", "-m", "pip", "install", "agent-bridge==0.1.0"))

    assert signals == [bootstrap.signal.SIGTERM, bootstrap.signal.SIGKILL]
    assert process.wait_calls == [1.0]


def test_runtime_install_allows_output_larger_than_version_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catches a normal pip progress stream failing under the 4 KiB verification cap.
    process = _RuntimeProcess()
    selector = _RuntimeSelector([
        [(process.stdout, "stdout")],
        [(process.stdout, "stdout")],
        [(process.stderr, "stderr")],
    ])
    chunks = {
        process.stdout.descriptor: [b"x" * 8192, b""],
        process.stderr.descriptor: [b""],
    }

    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(bootstrap.selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(
        bootstrap.os,
        "read",
        lambda descriptor, size: chunks[descriptor].pop(0),
    )
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: 0.0)

    result = bootstrap._run(
        ("/runtime/python", "-I", "-m", "pip", "install", "agent-bridge==0.1.0")
    )

    assert result.stdout == "x" * 8192
    assert process.wait_calls == [300.0]
