"""Small stdlib-only runtime bootstrap sent to a POSIX SSH host."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import selectors
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import TextIO

try:
    import venv
except ImportError:  # pragma: no cover - exercised on remote Python installations only
    venv = None  # type: ignore[assignment]


PROTOCOL_VERSION = 1
PACKAGE_NAME = "agent-bridge"

_RELEASE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+!-]{0,63}", re.ASCII)
_VERIFY_PROGRAM = """\
import base64
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import re
import stat
import sys


def fail():
    raise SystemExit(1)


if len(sys.argv) != 3:
    fail()
expected_executable = Path(sys.argv[1])
expected_prefix = Path(sys.argv[2])
if not expected_executable.is_absolute() or not expected_prefix.is_absolute():
    fail()
# Keep the requested venv launcher as a lexical authority.  Resolving it here
# would turn an ordinary venv's python symlink into the base interpreter and
# permit a cache-local wrapper to substitute another venv.
if os.path.abspath(sys.executable) != os.fspath(expected_executable):
    fail()
# Before Python 3.14, ``-S`` intentionally leaves sys.prefix at the base
# interpreter. Bind the selected venv from the caller-supplied lexical
# executable/prefix pair instead of letting site initialization self-report it.
if (
    expected_prefix != expected_executable.parent.parent
    or not (expected_prefix / "pyvenv.cfg").is_file()
):
    fail()
venv_root = expected_prefix.resolve()


def within_venv(path):
    try:
        path.relative_to(venv_root)
    except ValueError:
        return False
    return True


def within(root, path):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


site_packages_entry = (
    expected_prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
)
try:
    site_packages = site_packages_entry.resolve(strict=True)
    site_packages_mode = site_packages_entry.lstat().st_mode
except Exception:
    fail()
if (
    not within_venv(site_packages)
    or site_packages_entry.is_symlink()
    or not stat.S_ISDIR(site_packages_mode)
):
    fail()
try:
    if any(entry.suffix == ".pth" for entry in site_packages.iterdir()):
        fail()
except Exception:
    fail()
# ``-S`` leaves the selected venv's site-packages off sys.path.  Add only its
# fixed directory after rejecting executable .pth startup hooks, without ever
# calling site.main().
sys.path.insert(0, os.fspath(site_packages))

distribution = importlib.metadata.distribution("agent-bridge")
name = distribution.metadata.get("Name")
if not isinstance(name, str) or re.sub(r"[-_.]+", "-", name).lower() != "agent-bridge":
    fail()
files = distribution.files
if files is None or distribution.read_text("direct_url.json") is not None:
    fail()
package_entry = site_packages / "agent_bridge"
try:
    package_root = package_entry.resolve(strict=True)
    package_mode = package_entry.lstat().st_mode
except Exception:
    fail()
if (
    package_entry.is_symlink()
    or not stat.S_ISDIR(package_mode)
    or not within_venv(package_root)
    or not within(site_packages, package_root)
):
    fail()
records = {}
for record in files:
    name = str(record)
    try:
        recorded_path = Path(distribution.locate_file(record)).resolve(strict=True)
    except Exception:
        fail()
    if not within_venv(recorded_path):
        fail()
    if not name.startswith("agent_bridge/"):
        continue
    if chr(92) in name:
        fail()
    parts = Path(name).parts
    if (
        len(parts) < 2
        or parts[0] != "agent_bridge"
        or any(part in {"", ".", ".."} for part in parts)
        or name in records
    ):
        fail()
    expected_entry = package_root.joinpath(*parts[1:])
    try:
        expected = expected_entry.resolve(strict=True)
        entry = expected_entry.lstat()
    except Exception:
        fail()
    if (
        expected != recorded_path
        or not within_venv(expected)
        or not within(package_root, expected)
        or expected_entry.is_symlink()
        or not stat.S_ISREG(entry.st_mode)
        or record.hash is None
        or type(record.size) is not int
    ):
        fail()
    algorithm = record.hash.mode
    encoded = record.hash.value
    if (
        not isinstance(algorithm, str)
        or algorithm not in {"sha256", "sha384", "sha512"}
        or not isinstance(encoded, str)
    ):
        fail()
    try:
        digest = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        content = expected.read_bytes()
    except Exception:
        fail()
    if len(content) != record.size or hashlib.new(algorithm, content).digest() != digest:
        fail()
    records[name] = expected

for current, directories, names in os.walk(package_root):
    current_path = Path(current)
    for directory in tuple(directories):
        entry = current_path / directory
        try:
            mode = entry.lstat().st_mode
        except Exception:
            fail()
        if directory == "__pycache__":
            if not stat.S_ISDIR(mode):
                fail()
        elif not stat.S_ISDIR(mode):
            fail()
    for name in names:
        entry = current_path / name
        try:
            mode = entry.lstat().st_mode
        except Exception:
            fail()
        if name.endswith(".pyc") and entry.parent.name == "__pycache__":
            fail()
        if not stat.S_ISREG(mode):
            fail()
        relative = entry.relative_to(package_root).as_posix()
        if f"agent_bridge/{relative}" not in records:
            fail()

package = importlib.import_module("agent_bridge")
for relative, module in (
    ("agent_bridge/__init__.py", package),
    ("agent_bridge/__main__.py", importlib.import_module("agent_bridge.__main__")),
):
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or Path(module_file).resolve() != records.get(relative):
        fail()
print(distribution.version, end="")
"""
_VERIFY_TIMEOUT_SECONDS = 15
_VERIFY_OUTPUT_LIMIT = 4096
_INSTALL_TIMEOUT_SECONDS = 300
_INSTALL_OUTPUT_LIMIT = 256 * 1024
_MAX_SITE_PACKAGES_PTH_HOOKS = 32
_FD_SAFE_REMOVAL_AVAILABLE = (
    hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.rename in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
    and os.rmdir in os.supports_dir_fd
)


def _create_venv(path: Path) -> None:
    if venv is None:
        raise RuntimeError("remote Python does not provide venv")
    venv.EnvBuilder(with_pip=False).create(path)


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run one fixed runtime command with a bounded command-specific policy."""
    timeout_seconds, output_limit, label = _command_policy(argv)
    process: subprocess.Popen[str] | None = None
    selector: selectors.BaseSelector | None = None
    previous_handlers: Mapping[int, object] = {}
    pending_signals: list[int] = []
    teardown_signals: list[int] = []
    teardown_in_progress = False

    def interrupted(signum: int, frame: object) -> None:
        if teardown_in_progress:
            # The original failure is already propagating.  Defer repeated
            # terminal signals until this exact child group is stopped/reaped.
            teardown_signals.append(signum)
            return
        if process is None:
            pending_signals.append(signum)
            return
        raise RuntimeError(f"remote bootstrap interrupted by signal {signum}")

    try:
        previous_handlers = _install_cleanup_signal_handlers(interrupted)
        if pending_signals:
            raise RuntimeError(f"remote bootstrap interrupted by signal {pending_signals[0]}")
        process = subprocess.Popen(
            tuple(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        if pending_signals:
            raise RuntimeError(f"remote bootstrap interrupted by signal {pending_signals[0]}")
        if process.stdout is None or process.stderr is None:
            raise RuntimeError(f"{label} could not capture output")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        output = {"stdout": bytearray(), "stderr": bytearray()}
        deadline = time.monotonic() + timeout_seconds
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"{label} timed out")
            events = selector.select(remaining)
            if not events:
                raise RuntimeError(f"{label} timed out")
            for key, _ in events:
                chunk = os.read(key.fd, output_limit - len(output[key.data]) + 1)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                output[key.data].extend(chunk)
                if len(output[key.data]) > output_limit:
                    raise RuntimeError(f"{label} output exceeded its limit")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"{label} timed out")
        returncode = process.wait(timeout=remaining)
    except BaseException as error:
        if process is not None:
            teardown_in_progress = True
            try:
                _stop_process_group(process)
            finally:
                teardown_in_progress = False
        if teardown_signals:
            raise RuntimeError(
                f"remote bootstrap interrupted by signal {teardown_signals[0]}"
            ) from error
        raise
    finally:
        _restore_signal_handlers(previous_handlers)
        if selector is not None:
            selector.close()
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
    return subprocess.CompletedProcess(
        tuple(argv),
        returncode,
        bytes(output["stdout"]).decode("utf-8", errors="replace"),
        bytes(output["stderr"]).decode("utf-8", errors="replace"),
    )


def _install_cleanup_signal_handlers(
    interrupted: Callable[[int, object], None],
) -> dict[int, object]:
    previous: dict[int, object] = {}

    for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        try:
            previous[signum] = signal.signal(signum, interrupted)
        except ValueError:
            # Signal handlers are unavailable only away from the remote main thread.
            return previous
    return previous


def _restore_signal_handlers(previous: Mapping[int, object]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _command_policy(argv: Sequence[str]) -> tuple[int, int, str]:
    if tuple(argv[2:5]) == ("-m", "pip", "install"):
        return _INSTALL_TIMEOUT_SECONDS, _INSTALL_OUTPUT_LIMIT, "remote Agent Bridge installation"
    if tuple(argv[2:4]) == ("-m", "ensurepip"):
        return _INSTALL_TIMEOUT_SECONDS, _INSTALL_OUTPUT_LIMIT, "remote Python ensurepip"
    return _VERIFY_TIMEOUT_SECONDS, _VERIFY_OUTPUT_LIMIT, "remote version verification"


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass
    # The session leader may exit after SIGTERM while an inherited child remains
    # in the group. Escalate the exact, separately owned group before returning.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait()


def ensure_runtime(
    version: str,
    *,
    cache_root: Path | None = None,
    venv_creator: Callable[[Path], None] = _create_venv,
    command_runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run,
) -> Path:
    """Return an immutable, exact-version private runtime interpreter."""
    checked_version = _validate_version(version)
    root = _prepare_private_cache_root(
        cache_root or Path.home() / ".cache" / "agent-bridge" / "runtime"
    )
    final = root / checked_version
    with _version_lock(root, checked_version):
        temporary: Path | None = None
        previous_handlers: Mapping[int, object] = {}
        cleanup_in_progress = False
        deferred_signals: list[int] = []

        def interrupted(signum: int, frame: object) -> None:
            if cleanup_in_progress:
                deferred_signals.append(signum)
                return
            raise RuntimeError(f"remote bootstrap interrupted by signal {signum}")

        try:
            # This covers stale-tree sweep, temporary creation, venv setup, and
            # the exception cleanup path.  _run() temporarily installs its own
            # child-aware handler while an exact process group exists.
            previous_handlers = _install_cleanup_signal_handlers(interrupted)
            trusted_bootstrap = _trusted_bootstrap_interpreter()
            _sweep_abandoned_temporaries(root, checked_version)
            if _path_lexists(final):
                try:
                    _validate_private_tree(final, trusted_bootstrap=trusted_bootstrap)
                except RuntimeError:
                    _remove_private_tree(final, root=root)
            cached_python = final / "venv" / "bin" / "python"
            if _reports_exact_version(
                cached_python,
                checked_version,
                command_runner,
                cache_probe=True,
                trusted_bootstrap=trusted_bootstrap,
            ):
                return cached_python
            _remove_private_tree(final, root=root)
            temporary = Path(tempfile.mkdtemp(prefix=f".{checked_version}-", dir=root))
            venv_path = temporary / "venv"
            venv_creator(venv_path)
            _remove_stdlib_lib64_link(venv_path)
            python = venv_path / "bin" / "python"
            if not _is_private_executable(python, trusted_bootstrap=trusted_bootstrap):
                raise RuntimeError("remote venv did not create a private python executable")
            _require_success(
                command_runner(
                    (
                        str(python),
                        "-I",
                        "-m",
                        "ensurepip",
                        "--upgrade",
                        "--default-pip",
                    )
                ),
                "remote Python ensurepip failed",
            )
            _require_success(
                command_runner(
                    (
                        str(python),
                        "-I",
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "--no-input",
                        "--no-compile",
                        f"{PACKAGE_NAME}=={checked_version}",
                    )
                ),
                "remote Agent Bridge installation failed",
            )
            _remove_site_packages_pth_hooks(venv_path)
            _validate_private_tree(temporary, trusted_bootstrap=trusted_bootstrap)
            if not _reports_exact_version(
                python, checked_version, command_runner, trusted_bootstrap=trusted_bootstrap,
            ):
                raise RuntimeError("remote Agent Bridge version does not match")
            _make_tree_immutable(temporary, trusted_bootstrap=trusted_bootstrap)
            temporary.rename(final)
            return final / "venv" / "bin" / "python"
        except BaseException as error:
            if temporary is not None:
                cleanup_in_progress = True
                try:
                    _remove_private_tree(temporary, root=root)
                finally:
                    cleanup_in_progress = False
            if deferred_signals:
                raise RuntimeError(
                    f"remote bootstrap interrupted by signal {deferred_signals[0]}"
                ) from error
            raise
        finally:
            _restore_signal_handlers(previous_handlers)


def _sweep_abandoned_temporaries(root: Path, version: str) -> None:
    """Delete only interrupted temporaries for this locked release."""
    prefix = f".{version}-"
    try:
        candidates = tuple(path for path in root.iterdir() if path.name.startswith(prefix))
    except OSError as error:
        raise RuntimeError("cache runtime directory could not be scanned") from error
    for candidate in candidates:
        _remove_private_tree(candidate, root=root)


def _random_high_port() -> int:
    return 49152 + secrets.randbelow(16384)


def select_remote_port(
    requested: int,
    *,
    socket_factory: Callable[..., socket.socket] = socket.socket,
    candidate_factory: Callable[[], int] = _random_high_port,
) -> int:
    """Return a requested port or a currently free high IPv4 loopback port."""
    if not isinstance(requested, int) or isinstance(requested, bool):
        raise ValueError("remote port must be an integer")
    if not 0 <= requested <= 65535:
        raise ValueError("remote port must be between 0 and 65535")
    if requested:
        return requested
    for _ in range(32):
        candidate = candidate_factory()
        if not 49152 <= candidate <= 65535:
            raise RuntimeError("remote port candidate is outside the high range")
        try:
            with socket_factory(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", candidate))
        except OSError:
            continue
        return candidate
    raise RuntimeError("could not select an available remote high port")


def bootstrap_main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
) -> int:
    """Bootstrap one runtime and write its single protocol-v1 JSON record."""
    version, raw_port = tuple(sys.argv[1:] if argv is None else argv)
    _require_python_311()
    python = ensure_runtime(version)
    record = {
        "protocol": PROTOCOL_VERSION,
        "python": str(python),
        "remote_port": select_remote_port(int(raw_port)),
        "version": version,
    }
    stdout.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    stdout.flush()
    return 0


def _reports_exact_version(
    python: Path,
    version: str,
    command_runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]],
    *,
    cache_probe: bool = False,
    trusted_bootstrap: tuple[Path, os.stat_result, bytes] | None = None,
) -> bool:
    if trusted_bootstrap is None:
        try:
            trusted_bootstrap = _trusted_bootstrap_interpreter()
        except RuntimeError:
            return False
    if not _is_private_executable(python, trusted_bootstrap=trusted_bootstrap):
        return False
    expected_executable = Path(os.path.abspath(os.fspath(python)))
    expected_prefix = expected_executable.parent.parent
    try:
        result = command_runner((
            str(expected_executable),
            "-I",
            "-B",
            "-S",
            "-c",
            _VERIFY_PROGRAM,
            str(expected_executable),
            str(expected_prefix),
        ))
    except (OSError, RuntimeError, subprocess.SubprocessError):
        if cache_probe:
            return False
        raise
    return result.returncode == 0 and result.stdout == version


def _require_success(result: subprocess.CompletedProcess[str], message: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(message)


def _validate_version(version: object) -> str:
    if not isinstance(version, str) or _RELEASE_VERSION.fullmatch(version) is None:
        raise ValueError("remote runtime version must be a bounded release token")
    return version


def _require_python_311() -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError("remote Python 3.11 or newer is required")


def _prepare_private_cache_root(cache_root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(cache_root)))
    parts = root.parts
    current = Path(root.anchor)
    controlled_parent: Path | None = None
    missing_parts: tuple[str, ...] = ()
    for index, part in enumerate(parts[1:]):
        candidate = current / part
        if not _path_lexists(candidate):
            missing_parts = parts[index + 1 :]
            break
        entry = _lstat(candidate)
        _require_directory_shape(entry)
        _require_safe_cache_ancestor(entry)
        if entry.st_uid == os.geteuid():
            controlled_parent = candidate
        elif controlled_parent is not None:
            raise RuntimeError("cache component is not owned by the effective user")
        current = candidate
    if controlled_parent is None:
        raise RuntimeError("cache root has no owner-controlled parent")
    for part in missing_parts:
        current /= part
        os.mkdir(current, 0o700)
        _require_private_directory(current, _lstat(current))
    if current != root:
        raise RuntimeError("cache root could not be created")
    root_entry = _lstat(root)
    _require_private_directory(root, root_entry)
    if root_entry.st_uid != os.geteuid():
        raise RuntimeError("cache root is not owned by the effective user")
    os.chmod(root, 0o700)
    return root


@contextmanager
def _version_lock(root: Path, version: str) -> Iterator[None]:
    lock = root / f".{version}.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock, flags, 0o600)
    except OSError as error:
        raise RuntimeError("cache lock file is unsafe") from error
    try:
        lock_entry = os.fstat(descriptor)
        if not stat.S_ISREG(lock_entry.st_mode) or lock_entry.st_uid != os.geteuid():
            raise RuntimeError("cache lock file is unsafe")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _remove_private_tree(path: Path, *, root: Path) -> None:
    if path.parent != root:
        raise RuntimeError("cache removal escaped its runtime root")
    # Bind the descriptor to the authority checked before opening.  O_NOFOLLOW
    # protects only the final component, so an ancestor can otherwise be
    # exchanged for a symlink between pathname validation and open().
    expected_root = _lstat(root)
    _require_private_directory(root, expected_root)
    if expected_root.st_mode & 0o022:
        raise RuntimeError("cache root is externally writable")
    root_descriptor = _open_trusted_cache_root(root, expected_root=expected_root)
    try:
        _remove_rejected_private_tree(path.name, parent_fd=root_descriptor)
    finally:
        os.close(root_descriptor)


def _open_trusted_cache_root(root: Path, *, expected_root: os.stat_result) -> int:
    if not _FD_SAFE_REMOVAL_AVAILABLE:
        raise RuntimeError("safe cache removal is unavailable on this platform")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise RuntimeError("cache root could not be opened safely") from error
    try:
        entry = os.fstat(descriptor)
        if not _same_file(entry, expected_root):
            raise RuntimeError("cache root changed while opening")
        _require_private_directory(root, entry)
        if entry.st_mode & 0o022:
            raise RuntimeError("cache root is externally writable")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _remove_rejected_private_tree(name: str, *, parent_fd: int) -> None:
    """Remove one rejected cache entry without resolving child pathnames."""
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise RuntimeError("cache component could not be inspected safely") from error
    quarantined_name = _quarantine_rejected_entry(
        name, parent_fd=parent_fd, expected=entry,
    )
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        try:
            os.unlink(quarantined_name, dir_fd=parent_fd)
        except OSError as error:
            raise RuntimeError("cache component could not be removed safely") from error
        return
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(quarantined_name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise RuntimeError("cache component could not be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        if not _same_file(entry, opened):
            raise RuntimeError("cache component changed while opening")
        _require_private_directory(Path(quarantined_name), opened)
        os.fchmod(descriptor, 0o700)
        try:
            with os.scandir(descriptor) as children:
                child_names = tuple(child.name for child in children)
        except (OSError, TypeError) as error:
            raise RuntimeError("cache component could not be scanned safely") from error
        for child_name in child_names:
            _remove_rejected_private_tree(child_name, parent_fd=descriptor)
        try:
            current = os.stat(quarantined_name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(current.st_mode) or not _same_file(entry, current):
                raise RuntimeError("cache component changed while removing")
            os.rmdir(quarantined_name, dir_fd=parent_fd)
        except OSError as error:
            raise RuntimeError("cache component could not be removed safely") from error
    finally:
        os.close(descriptor)


def _quarantine_rejected_entry(
    name: str,
    *,
    parent_fd: int,
    expected: os.stat_result,
) -> str:
    """Detach an entry to an unpredictable private sibling before deletion.

    POSIX has no fd-based unlink/rmdir operation.  The private cache root keeps
    other users out, and this unguessable sibling prevents an external actor
    from naming a replacement between the identity check and destructive call.
    A process already running as the effective user can enumerate and rename
    entries in its own private cache directory, which POSIX cannot fully
    distinguish from this process; on any observed identity mismatch we leave
    both entries in place and fail closed rather than deleting either one.
    """
    for _ in range(8):
        quarantined_name = f".agent-bridge-removing-{secrets.token_hex(32)}"
        try:
            os.stat(quarantined_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise RuntimeError("cache component could not be inspected safely") from error
        else:
            continue
        try:
            os.rename(
                name,
                quarantined_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except FileNotFoundError as error:
            raise RuntimeError("cache component changed while quarantining") from error
        except OSError as error:
            raise RuntimeError("cache component could not be quarantined safely") from error
        try:
            quarantined = os.stat(
                quarantined_name, dir_fd=parent_fd, follow_symlinks=False,
            )
        except OSError as error:
            raise RuntimeError("cache component changed while quarantining") from error
        if not _same_file(expected, quarantined):
            # Do not rename back: without Linux renameat2(RENAME_NOREPLACE), a
            # second rename can overwrite a new same-user entry.  The moved
            # substitute remains under the private parent and no destructive
            # operation has occurred.
            raise RuntimeError("cache component changed while quarantining")
        return quarantined_name
    raise RuntimeError("cache component could not be quarantined safely")


def _validate_private_tree(
    path: Path,
    *,
    trusted_bootstrap: tuple[Path, os.stat_result, bytes] | None = None,
    tree_root: Path | None = None,
) -> None:
    if tree_root is None:
        tree_root = path
    if _is_unrecorded_agent_bridge_bytecode(path, tree_root=tree_root):
        raise RuntimeError("cache package bytecode must not be present")
    entry = _lstat(path)
    _require_private_directory(path, entry)
    for child in os.scandir(path):
        child_path = Path(child.path)
        child_entry = _lstat(child_path)
        if stat.S_ISLNK(child_entry.st_mode):
            if (
                trusted_bootstrap is not None
                and _is_standard_venv_interpreter_link(
                    child_path, tree_root=tree_root, trusted_bootstrap=trusted_bootstrap,
                )
            ):
                continue
            raise RuntimeError("cache component must not be a symlink")
        if child_entry.st_uid != os.geteuid():
            raise RuntimeError("cache component is not owned by the effective user")
        if stat.S_ISDIR(child_entry.st_mode):
            _validate_private_tree(
                child_path,
                trusted_bootstrap=trusted_bootstrap,
                tree_root=tree_root,
            )
        elif not stat.S_ISREG(child_entry.st_mode):
            raise RuntimeError("cache component is not a regular file")


def _is_unrecorded_agent_bridge_bytecode(path: Path, *, tree_root: Path) -> bool:
    """Reject bytecode that could run instead of RECORD-authenticated source."""
    try:
        parts = path.relative_to(tree_root).parts
    except ValueError:
        return True
    try:
        package_index = parts.index("agent_bridge")
    except ValueError:
        return False
    package_parts = parts[package_index + 1 :]
    return path.suffix == ".pyc" or "__pycache__" in package_parts


def _make_tree_immutable(
    path: Path, *, trusted_bootstrap: tuple[Path, os.stat_result, bytes],
) -> None:
    _validate_private_tree(path, trusted_bootstrap=trusted_bootstrap)
    for directory, _, files in os.walk(path, topdown=False, followlinks=False):
        directory_path = Path(directory)
        for name in files:
            file_path = directory_path / name
            mode = _lstat(file_path).st_mode
            if stat.S_ISLNK(mode):
                continue
            os.chmod(file_path, 0o500 if mode & 0o111 else 0o400)
        os.chmod(directory_path, 0o500)


def _restore_private_tree_for_removal(path: Path) -> None:
    """Restore traversal/write modes only after the complete tree is validated."""
    _validate_private_tree(path)
    for directory, _, _ in os.walk(path, topdown=True, followlinks=False):
        os.chmod(directory, 0o700)


def _remove_stdlib_lib64_link(venv_path: Path) -> None:
    """Remove only venv's known internal ``lib64 -> lib`` compatibility link."""
    _require_private_directory(venv_path, _lstat(venv_path))
    lib64 = venv_path / "lib64"
    if not _path_lexists(lib64):
        return
    entry = _lstat(lib64)
    if stat.S_ISLNK(entry.st_mode) and os.readlink(lib64) == "lib":
        lib64.unlink()


def _remove_site_packages_pth_hooks(venv_path: Path) -> None:
    """Remove bounded regular site hooks created by ensurepip before verification."""
    site_packages = (
        venv_path
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if not _path_lexists(site_packages):
        # Lightweight fake runners can model pip success without creating its
        # package directory. A real installation is rejected by the verifier.
        return
    try:
        directory = _lstat(site_packages)
    except RuntimeError as error:
        raise RuntimeError("site-packages directory is unsafe") from error
    if stat.S_ISLNK(directory.st_mode) or not stat.S_ISDIR(directory.st_mode):
        raise RuntimeError("site-packages directory is unsafe")
    if directory.st_uid != os.geteuid():
        raise RuntimeError("site-packages directory is unsafe")
    try:
        entries = tuple(os.scandir(site_packages))
    except OSError as error:
        raise RuntimeError("site-packages directory could not be scanned") from error
    hooks = tuple(entry for entry in entries if entry.name.endswith(".pth"))
    if len(hooks) > _MAX_SITE_PACKAGES_PTH_HOOKS:
        raise RuntimeError("site-packages has too many .pth hooks")
    for hook in hooks:
        path = Path(hook.path)
        try:
            entry = path.lstat()
        except OSError as error:
            raise RuntimeError("site-packages .pth could not be inspected safely") from error
        if not stat.S_ISREG(entry.st_mode) or entry.st_uid != os.geteuid():
            raise RuntimeError("site-packages .pth is unsafe")
        try:
            path.unlink()
        except OSError as error:
            raise RuntimeError("site-packages .pth could not be removed safely") from error


def _trusted_bootstrap_interpreter() -> tuple[Path, os.stat_result, bytes]:
    """Capture the running bootstrap interpreter before examining a cache entry."""
    if not isinstance(sys.executable, str) or not sys.executable:
        raise RuntimeError("remote bootstrap interpreter is unavailable")
    configured = Path(os.path.abspath(sys.executable))
    try:
        resolved = configured.resolve(strict=True)
        entry = resolved.stat()
    except OSError as error:
        raise RuntimeError("remote bootstrap interpreter could not be inspected") from error
    if not stat.S_ISREG(entry.st_mode) or not entry.st_mode & 0o111:
        raise RuntimeError("remote bootstrap interpreter is unsafe")
    return resolved, entry, _sha256_file(resolved)


def _is_private_executable(
    path: Path,
    *,
    trusted_bootstrap: tuple[Path, os.stat_result, bytes] | None = None,
) -> bool:
    if trusted_bootstrap is None:
        try:
            trusted_bootstrap = _trusted_bootstrap_interpreter()
        except RuntimeError:
            return False
    if not _path_lexists(path):
        return False
    try:
        link_entry = _lstat(path)
        resolved = path.resolve(strict=True)
        target_entry = resolved.stat()
    except OSError:
        return False
    if (
        link_entry.st_uid != os.geteuid()
        or not stat.S_ISREG(target_entry.st_mode)
        or not os.access(path, os.X_OK)
    ):
        return False
    trusted_path, trusted_entry, trusted_digest = trusted_bootstrap
    if _same_file(target_entry, trusted_entry):
        return True
    if target_entry.st_size != trusted_entry.st_size:
        return False
    try:
        return secrets.compare_digest(_sha256_file(resolved), trusted_digest)
    except OSError:
        return False


def _is_standard_venv_interpreter_link(
    path: Path,
    *,
    tree_root: Path,
    trusted_bootstrap: tuple[Path, os.stat_result, bytes],
) -> bool:
    bin_directory = tree_root / "venv" / "bin"
    if path.parent != bin_directory or path.name not in {
        "python",
        "python3",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
    }:
        return False
    return _is_private_executable(path, trusted_bootstrap=trusted_bootstrap)


def _sha256_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def _require_private_directory(path: Path, entry: os.stat_result) -> None:
    _require_directory_shape(entry)
    if entry.st_uid != os.geteuid():
        raise RuntimeError("cache component is not owned by the effective user")


def _require_safe_cache_ancestor(entry: os.stat_result) -> None:
    if entry.st_uid not in {0, os.geteuid()}:
        raise RuntimeError("cache ancestor is externally writable or has an untrusted owner")
    if not entry.st_mode & 0o022:
        return
    if (
        entry.st_mode & stat.S_ISVTX
        and entry.st_uid in {0, os.geteuid()}
    ):
        return
    raise RuntimeError("cache ancestor is externally writable")


def _require_directory_shape(entry: os.stat_result) -> None:
    if stat.S_ISLNK(entry.st_mode):
        raise RuntimeError("cache component must not be a symlink")
    if not stat.S_ISDIR(entry.st_mode):
        raise RuntimeError("cache component is not a directory")


def _lstat(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        raise RuntimeError("cache component could not be inspected") from error


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


if __name__ == "__main__":
    try:
        raise SystemExit(bootstrap_main())
    except Exception as error:
        print(f"remote bootstrap failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
