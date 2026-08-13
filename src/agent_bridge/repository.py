"""Read-only repository baselines for a single approved bridge task.

The tracker deliberately keeps the snapshot outside the target repository.  It
does not attempt to repair a checkout: its only job is to report the delta from
the moment Sol was allowed to start.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import difflib
import errno
import hashlib
import os
from pathlib import Path, PurePosixPath
import selectors
import stat
import subprocess
import time
from types import MappingProxyType
from typing import Callable, Mapping
import uuid

from agent_bridge.contracts import TaskBrief


_DEFAULT_PROTECTED_PREFIXES = (
    PurePosixPath("data"),
    PurePosixPath("benchmarks"),
    PurePosixPath(".git"),
    PurePosixPath(".artifacts"),
    PurePosixPath(".superpowers/brainstorm"),
)
_MAX_GIT_CONFIGURATION_BYTES = 1024 * 1024
_MAX_GIT_INDEX_BYTES = 64 * 1024 * 1024


class ProtectedPathApprovalRequired(RuntimeError):
    """Raised when capture would snapshot a path requiring separate approval."""


class _UnsafeRepositoryPath(RuntimeError):
    """A repository entry changed type while being read safely."""


class _SnapshotTooLarge(RuntimeError):
    """A copied baseline image exceeded its separately-approved size limit."""


@dataclass(frozen=True)
class PathBaseline:
    path: str
    existed: bool
    kind: str
    size: int
    sha256: str | None
    before_image: Path | None
    protected_reason: str | None


@dataclass(frozen=True)
class WorkspaceBaseline:
    baseline_id: str
    repo_root: Path
    repo_root_sha256: str
    head: str
    branch: str
    git_marker_sha256: str
    git_dir: Path
    git_dir_sha256: str
    common_dir: Path
    common_dir_sha256: str
    git_index_sha256: str
    allowed_paths: tuple[str, ...]
    paths: tuple[PathBaseline, ...]
    git_control: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class WorkspaceDelta:
    """A deterministic, non-mutating comparison against one baseline."""

    changed_paths: tuple[str, ...]
    unexpected_paths: tuple[str, ...]
    protected_changed_paths: tuple[str, ...]
    preexisting_unchanged_paths: tuple[str, ...]
    text_diffs: Mapping[str, str] = field(default_factory=dict)
    binary_changed_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "text_diffs", MappingProxyType(dict(self.text_diffs)))


def validate_allowed_path(value: str) -> PurePosixPath:
    """Return a normalized repository-relative allowed path, or raise."""
    if not isinstance(value, str):
        raise ValueError(f"allowed path must be repository-relative: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"allowed path must be repository-relative: {value}")
    return path


def _open_parent_directory(repo_root: Path, path: Path) -> tuple[int, str]:
    """Open a repository-relative path's parent without following any component."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise _UnsafeRepositoryPath("safe repository reads require O_NOFOLLOW")
    try:
        relative = path.relative_to(repo_root)
    except ValueError as error:
        raise _UnsafeRepositoryPath(f"path is outside repository: {path}") from error
    if not relative.parts:
        raise _UnsafeRepositoryPath("repository root is not a file entry")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(repo_root, flags)
        for component in relative.parts[:-1]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except FileNotFoundError:
        try:
            os.close(descriptor)
        except UnboundLocalError:
            pass
        raise
    except OSError as error:
        try:
            os.close(descriptor)
        except UnboundLocalError:
            pass
        raise _UnsafeRepositoryPath(f"unsafe repository path: {path}") from error
    return descriptor, relative.parts[-1]


def _entry_stat(repo_root: Path, path: Path) -> os.stat_result | None:
    try:
        parent, name = _open_parent_directory(repo_root, path)
    except FileNotFoundError:
        return None
    try:
        return os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise _UnsafeRepositoryPath(f"unsafe repository path: {path}") from error
    finally:
        os.close(parent)


def _read_regular_bytes(repo_root: Path, path: Path, max_bytes: int) -> bytes:
    """Read one regular repository entry without exceeding a total-call budget."""
    chunks: list[bytes] = []
    size = 0
    with _open_regular(repo_root, path) as source:
        if os.fstat(source.fileno()).st_size > max_bytes:
            raise _SnapshotTooLarge(path.as_posix())
        while block := source.read(min(1024 * 1024, max_bytes - size + 1)):
            if size + len(block) > max_bytes:
                raise _SnapshotTooLarge(path.as_posix())
            chunks.append(block)
            size += len(block)
    return b"".join(chunks)


def _metadata_sha256(entry: os.stat_result) -> str:
    """Build a no-content identity that same-user writes cannot restore exactly."""
    digest = hashlib.sha256()
    for value in (
        stat.S_IFMT(entry.st_mode),
        entry.st_dev,
        entry.st_ino,
        entry.st_size,
        entry.st_mtime_ns,
        entry.st_ctime_ns,
    ):
        digest.update(str(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _directory_sha256(entry: os.stat_result) -> str:
    """Identify a directory itself without treating child churn as a change."""
    digest = hashlib.sha256()
    for value in (stat.S_IFMT(entry.st_mode), entry.st_dev, entry.st_ino):
        digest.update(str(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _entry_identity_sha256(entry: os.stat_result) -> str:
    if stat.S_ISDIR(entry.st_mode):
        return _directory_sha256(entry)
    return _metadata_sha256(entry)


def _path_kind(repo_root: Path, path: Path) -> str:
    entry = _entry_stat(repo_root, path)
    if entry is None:
        return "missing"
    mode = entry.st_mode
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISDIR(mode):
        return "directory"
    return "other"


def _open_regular(repo_root: Path, path: Path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise _UnsafeRepositoryPath("safe repository reads require O_NOFOLLOW")
    parent, name = _open_parent_directory(repo_root, path)
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except OSError as error:
        raise _UnsafeRepositoryPath(f"unsafe repository path: {path}") from error
    finally:
        os.close(parent)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise _UnsafeRepositoryPath(f"repository path is not a regular file: {path}")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _copy_regular_file(
    repo_root: Path,
    source_path: Path,
    destination_path: Path | int,
    max_snapshot_bytes: int,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    destination = (
        os.fdopen(destination_path, "wb")
        if isinstance(destination_path, int)
        else destination_path.open("wb")
    )
    with _open_regular(repo_root, source_path) as source, destination:
        if os.fstat(source.fileno()).st_size > max_snapshot_bytes:
            raise _SnapshotTooLarge(source_path.as_posix())
        while block := source.read(1024 * 1024):
            if size + len(block) > max_snapshot_bytes:
                raise _SnapshotTooLarge(source_path.as_posix())
            destination.write(block)
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _open_absolute_directory(path: Path, *, create: bool) -> int:
    """Open an absolute directory by components without following links."""
    if not path.is_absolute() or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("artifact_directory must be an absolute no-follow path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, dir_fd=descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _absolute_entry_sha256(
    path: Path, *, hash_regular_content: bool, max_bytes: int
) -> tuple[str, int]:
    """Fingerprint one absolute entry without following any path component."""
    parent = _open_absolute_directory(path.parent, create=False)
    try:
        try:
            entry = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return hashlib.sha256(b"missing").hexdigest(), 0
        if not hash_regular_content or not stat.S_ISREG(entry.st_mode):
            return _entry_identity_sha256(entry), 0
        contents = _read_absolute_regular_bytes(path, max_bytes)
        digest = hashlib.sha256(b"regular\0" + contents).hexdigest()
        return digest, len(contents)
    finally:
        os.close(parent)


def _read_absolute_regular_bytes(path: Path, max_bytes: int) -> bytes:
    parent = _open_absolute_directory(path.parent, create=False)
    try:
        entry = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(entry.st_mode) or entry.st_size > max_bytes:
            if entry.st_size > max_bytes:
                raise _SnapshotTooLarge(path.as_posix())
            raise _UnsafeRepositoryPath(f"absolute path is not regular: {path}")
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
            dir_fd=parent,
        )
    finally:
        os.close(parent)
    try:
        current = os.fstat(descriptor)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != entry.st_dev
            or current.st_ino != entry.st_ino
            or current.st_size != entry.st_size
        ):
            raise _UnsafeRepositoryPath(
                f"absolute path changed while opening: {path}"
            )
        chunks: list[bytes] = []
        size = 0
        while block := os.read(descriptor, min(1024 * 1024, max_bytes - size + 1)):
            if size + len(block) > max_bytes:
                raise _SnapshotTooLarge(path.as_posix())
            chunks.append(block)
            size += len(block)
        final = os.fstat(descriptor)
        if (
            final.st_dev != entry.st_dev
            or final.st_ino != entry.st_ino
            or final.st_size != entry.st_size
        ):
            raise _UnsafeRepositoryPath(f"absolute path changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_open_regular_bytes(descriptor: int, max_bytes: int) -> bytes:
    """Read one already-open regular file without changing its shared offset."""
    entry = os.fstat(descriptor)
    if not stat.S_ISREG(entry.st_mode):
        raise _UnsafeRepositoryPath("open snapshot is not a regular file")
    if entry.st_size > max_bytes:
        raise _SnapshotTooLarge("open snapshot exceeds its content bound")
    chunks: list[bytes] = []
    size = 0
    while block := os.pread(
        descriptor, min(1024 * 1024, max_bytes - size + 1), size
    ):
        if size + len(block) > max_bytes:
            raise _SnapshotTooLarge("open snapshot exceeds its content bound")
        chunks.append(block)
        size += len(block)
    final = os.fstat(descriptor)
    if (
        final.st_dev != entry.st_dev
        or final.st_ino != entry.st_ino
        or final.st_size != entry.st_size
        or final.st_mtime_ns != entry.st_mtime_ns
        or final.st_ctime_ns != entry.st_ctime_ns
    ):
        raise _UnsafeRepositoryPath("open snapshot changed while reading")
    return b"".join(chunks)


def _is_within(path: PurePosixPath, prefix: PurePosixPath) -> bool:
    return path == prefix or prefix in path.parents


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class RepositoryTracker:
    """Capture and compare a git working tree without changing it."""

    def __init__(
        self,
        repo_root: str | Path,
        artifact_directory: str | Path,
        *,
        git_executable: str | Path,
        git_timeout_seconds: float = 10.0,
        max_snapshot_bytes: int = 5 * 1024 * 1024,
        max_ignored_entries: int = 250_000,
        max_ignored_bytes: int | None = None,
        max_inventory_entries: int = 250_000,
        max_inventory_listing_bytes: int = 16 * 1024 * 1024,
        max_inventory_content_bytes: int = 64 * 1024 * 1024,
        protected_prefixes: tuple[str, ...] | None = None,
        artifact_directory_fd: int | None = None,
    ) -> None:
        self._artifact_fd: int | None = None
        self._repo_root = Path(repo_root).resolve()
        self._artifact_directory = Path(os.path.abspath(artifact_directory))
        if not self._repo_root.is_dir():
            raise ValueError("repo_root must be an existing directory")
        if self._artifact_directory.is_relative_to(self._repo_root):
            raise ValueError("artifact_directory must be outside repo_root")
        candidate_git = Path(git_executable)
        if not candidate_git.is_absolute() or Path(os.path.abspath(candidate_git)) != candidate_git:
            raise ValueError("Git executable must be an absolute canonical path")
        if (
            not isinstance(git_timeout_seconds, (int, float))
            or isinstance(git_timeout_seconds, bool)
            or git_timeout_seconds <= 0
        ):
            raise ValueError("git_timeout_seconds must be positive")
        if max_snapshot_bytes < 0:
            raise ValueError("max_snapshot_bytes must be >= 0")
        if (
            not isinstance(max_ignored_entries, int)
            or isinstance(max_ignored_entries, bool)
            or max_ignored_entries < 0
        ):
            raise ValueError("max_ignored_entries must be a non-negative integer")
        if max_ignored_bytes is not None and (
            not isinstance(max_ignored_bytes, int)
            or isinstance(max_ignored_bytes, bool)
            or max_ignored_bytes < 0
        ):
            raise ValueError("max_ignored_bytes must be None or a non-negative integer")
        for value, name in (
            (max_inventory_entries, "max_inventory_entries"),
            (max_inventory_listing_bytes, "max_inventory_listing_bytes"),
            (max_inventory_content_bytes, "max_inventory_content_bytes"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        prefixes = _DEFAULT_PROTECTED_PREFIXES if protected_prefixes is None else tuple(
            validate_allowed_path(value) for value in protected_prefixes
        )
        try:
            parent = _open_absolute_directory(candidate_git.parent, create=False)
            try:
                executable_entry = os.stat(
                    candidate_git.name, dir_fd=parent, follow_symlinks=False
                )
                executable_accessible = os.access(
                    candidate_git.name,
                    os.X_OK,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
            finally:
                os.close(parent)
        except OSError as error:
            raise ValueError("Git executable must be a safe regular executable") from error
        if (
            not stat.S_ISREG(executable_entry.st_mode)
            or not executable_accessible
        ):
            raise ValueError("Git executable must be a safe regular executable")
        try:
            if artifact_directory_fd is None:
                self._artifact_fd = _open_absolute_directory(self._artifact_directory, create=True)
            else:
                if (
                    not isinstance(artifact_directory_fd, int)
                    or isinstance(artifact_directory_fd, bool)
                    or artifact_directory_fd < 0
                ):
                    raise ValueError("artifact_directory_fd must be an open directory descriptor")
                descriptor = os.dup(artifact_directory_fd)
                try:
                    is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
                except BaseException:
                    os.close(descriptor)
                    raise
                if not is_directory:
                    os.close(descriptor)
                    raise ValueError("artifact_directory_fd must be an open directory descriptor")
                self._artifact_fd = descriptor
        except OSError as error:
            raise ValueError("artifact_directory must be a safe writable directory") from error
        self._max_snapshot_bytes = max_snapshot_bytes
        self._git_executable = candidate_git
        self._git_timeout_seconds = float(git_timeout_seconds)
        self._max_ignored_entries = max_ignored_entries
        self._max_ignored_bytes = max_ignored_bytes
        self._max_inventory_entries = max_inventory_entries
        self._max_inventory_listing_bytes = max_inventory_listing_bytes
        self._max_inventory_content_bytes = max_inventory_content_bytes
        self._protected_prefixes = prefixes

    def close(self) -> None:
        """Release the retained artifact-directory descriptor exactly once."""
        if self._artifact_fd is not None:
            os.close(self._artifact_fd)
            self._artifact_fd = None

    def __enter__(self) -> "RepositoryTracker":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_open(self) -> int:
        if self._artifact_fd is None:
            raise RuntimeError("RepositoryTracker is closed")
        return self._artifact_fd

    def capture(self, brief: TaskBrief) -> WorkspaceBaseline:
        """Record before-images for the approved scope in an external artifact dir."""
        self._require_open()
        if not isinstance(brief, TaskBrief):
            raise ValueError("brief must be a TaskBrief")
        allowed = tuple(validate_allowed_path(value) for value in brief.allowed_paths)
        for path in allowed:
            protected_reason = self._protected_reason(path)
            if protected_reason is not None:
                raise ProtectedPathApprovalRequired(
                    f"protected path requires separate approval: {path} ({protected_reason})"
                )

        git_dir, common_dir = self._structural_git_roots()
        repo_root_sha256 = self._absolute_identity(self._repo_root)
        git_marker_sha256 = self._absolute_identity(
            self._repo_root / ".git", hash_regular_content=True
        )
        git_dir_sha256 = self._absolute_identity(git_dir)
        common_dir_sha256 = self._absolute_identity(common_dir)
        self._validate_git_relationship(git_dir, common_dir)
        self._validate_local_git_configuration(git_dir, common_dir)
        self._validate_git_index_path(git_dir)
        pre_git_control = self._git_control_manifest(git_dir, common_dir)

        tracked_paths, git_index_sha256 = self._index_inventory_and_identity(
            git_dir, common_dir
        )
        filesystem_paths = self._filesystem_inventory()
        inventory = self._inventory(
            tracked_paths=tracked_paths,
            filesystem_paths=filesystem_paths,
        )
        if self._git_control_manifest(git_dir, common_dir) != pre_git_control:
            raise ProtectedPathApprovalRequired(
                "Git control snapshot changed during filesystem inventory"
            )
        for path in inventory:
            relative = PurePosixPath(path)
            source = self._repo_root / relative
            if self._is_allowed(relative, allowed):
                try:
                    source_kind = _path_kind(self._repo_root, source)
                except _UnsafeRepositoryPath as error:
                    raise ProtectedPathApprovalRequired(
                        f"unsafe path requires separate approval: {path}"
                    ) from error
                if source_kind == "symlink":
                    raise ProtectedPathApprovalRequired(
                        f"unsafe symlink path requires separate approval: {path}"
                    )
                if source_kind not in {"regular", "missing", "directory"}:
                    raise ProtectedPathApprovalRequired(
                        f"non-regular path requires separate approval: {path}"
                    )
                protected_reason = self._protected_reason(relative)
                if protected_reason is not None:
                    raise ProtectedPathApprovalRequired(
                        f"protected path requires separate approval: {path} ({protected_reason})"
                    )

        self._validate_git_root_values(
            repo_root_sha256=repo_root_sha256,
            git_marker_sha256=git_marker_sha256,
            git_dir=git_dir,
            git_dir_sha256=git_dir_sha256,
            common_dir=common_dir,
            common_dir_sha256=common_dir_sha256,
        )
        head = self._anchored_git_text(
            git_dir, common_dir, "rev-parse", "HEAD"
        ).strip()
        branch = self._anchored_git_text(
            git_dir, common_dir, "branch", "--show-current"
        ).strip()
        git_control = self._git_control_manifest(git_dir, common_dir)
        if git_control != pre_git_control:
            raise ProtectedPathApprovalRequired(
                "Git control snapshot changed during identity capture"
            )
        baseline_id = uuid.uuid4().hex
        records = self._capture_path_records(baseline_id, inventory, allowed)
        return WorkspaceBaseline(
            baseline_id=baseline_id,
            repo_root=self._repo_root,
            repo_root_sha256=repo_root_sha256,
            head=head,
            branch=branch,
            git_marker_sha256=git_marker_sha256,
            git_dir=git_dir,
            git_dir_sha256=git_dir_sha256,
            common_dir=common_dir,
            common_dir_sha256=common_dir_sha256,
            git_index_sha256=git_index_sha256,
            allowed_paths=tuple(path.as_posix() for path in allowed),
            paths=records,
            git_control=git_control,
        )

    def _capture_path_records(
        self,
        baseline_id: str,
        inventory: tuple[str, ...],
        allowed: tuple[PurePosixPath, ...],
    ) -> tuple[PathBaseline, ...]:
        records: list[PathBaseline] = []
        created_artifacts: list[PurePosixPath] = []
        content_bytes_read = 0
        try:
            for path in inventory:
                relative = PurePosixPath(path)
                source = self._repo_root / relative
                is_allowed = self._is_allowed(relative, allowed)
                before_image: Path | None = None
                try:
                    source_kind = _path_kind(self._repo_root, source)
                except _UnsafeRepositoryPath as error:
                    raise ProtectedPathApprovalRequired(
                        f"unsafe path requires separate approval: {path}"
                    ) from error
                if source_kind == "missing":
                    records.append(PathBaseline(
                        path=path,
                        existed=False,
                        kind="missing",
                        size=0,
                        sha256=None,
                        before_image=None,
                        protected_reason=self._protected_reason(relative),
                    ))
                    continue
                if source_kind not in {"regular", "symlink"}:
                    entry = _entry_stat(self._repo_root, source)
                    records.append(PathBaseline(
                        path=path,
                        existed=True,
                        kind=source_kind,
                        size=0 if entry is None else entry.st_size,
                        sha256=None if entry is None else _entry_identity_sha256(entry),
                        before_image=None,
                        protected_reason=self._protected_reason(relative),
                    ))
                    continue
                size: int
                digest: str | None
                if is_allowed:
                    artifact_path = PurePosixPath(baseline_id) / "before" / relative
                    before_image = self._artifact_directory / artifact_path
                    try:
                        snapshot_limit = min(
                            self._max_snapshot_bytes,
                            self._max_inventory_content_bytes - content_bytes_read,
                        )
                        destination = self._open_artifact_output(artifact_path)
                        created_artifacts.append(artifact_path)
                        size, digest = _copy_regular_file(
                            self._repo_root, source, destination, snapshot_limit
                        )
                        content_bytes_read += size
                    except _SnapshotTooLarge as error:
                        raise ProtectedPathApprovalRequired(
                            f"inventory aggregate content budget requires separate approval: "
                            f"{path} exceeds remaining bytes={snapshot_limit}"
                        ) from error
                    except _UnsafeRepositoryPath as error:
                        raise ProtectedPathApprovalRequired(
                            f"unsafe path requires separate approval: {path}"
                        ) from error
                else:
                    try:
                        entry = _entry_stat(self._repo_root, source)
                        digest = None if entry is None else _entry_identity_sha256(entry)
                    except _UnsafeRepositoryPath as error:
                        raise ProtectedPathApprovalRequired(
                            f"unsafe path requires separate approval: {path}"
                        ) from error
                    if entry is None:
                        records.append(PathBaseline(
                            path=path,
                            existed=False,
                            kind="missing",
                            size=0,
                            sha256=None,
                            before_image=None,
                            protected_reason=self._protected_reason(relative),
                        ))
                        continue
                    size = entry.st_size
                records.append(PathBaseline(
                    path=path,
                    existed=True,
                    kind=source_kind,
                    size=size,
                    sha256=digest,
                    before_image=before_image,
                    protected_reason=self._protected_reason(relative),
                ))

            existing = {record.path for record in records}
            for path in allowed:
                normalized = path.as_posix()
                try:
                    is_directory = self._allowed_path_is_directory(path)
                except _UnsafeRepositoryPath as error:
                    raise ProtectedPathApprovalRequired(
                        f"unsafe path requires separate approval: {path}"
                    ) from error
                if normalized not in existing and not is_directory:
                    records.append(PathBaseline(
                        path=normalized,
                        existed=False,
                        kind="missing",
                        size=0,
                        sha256=None,
                        before_image=None,
                        protected_reason=None,
                    ))
            return tuple(sorted(records, key=lambda record: record.path))
        except BaseException:
            for artifact_path in reversed(created_artifacts):
                try:
                    self._remove_artifact_output(artifact_path)
                except FileNotFoundError:
                    pass
                self._prune_artifact_parents(artifact_path)
            raise

    def baseline_manifest(self, baseline: WorkspaceBaseline) -> dict[str, object]:
        """Return the complete JSON-compatible manifest needed after restart."""
        self._validate_baseline_identity(baseline)
        return {
            "baseline_id": baseline.baseline_id,
            "repo_root": str(baseline.repo_root),
            "repo_root_sha256": baseline.repo_root_sha256,
            "head": baseline.head,
            "branch": baseline.branch,
            "git_marker_sha256": baseline.git_marker_sha256,
            "git_dir": str(baseline.git_dir),
            "git_dir_sha256": baseline.git_dir_sha256,
            "common_dir": str(baseline.common_dir),
            "common_dir_sha256": baseline.common_dir_sha256,
            "git_index_sha256": baseline.git_index_sha256,
            "allowed_paths": list(baseline.allowed_paths),
            "git_control": [
                {"path": path, "sha256": digest}
                for path, digest in baseline.git_control
            ],
            "paths": [
                {
                    "path": record.path,
                    "existed": record.existed,
                    "kind": record.kind,
                    "size": record.size,
                    "sha256": record.sha256,
                    "before_image": (
                        None if record.before_image is None else str(record.before_image)
                    ),
                    "protected_reason": record.protected_reason,
                }
                for record in baseline.paths
            ],
        }

    def discard_baseline(self, baseline: WorkspaceBaseline) -> None:
        """Remove artifacts from one unpersisted initial capture."""
        if not isinstance(baseline, WorkspaceBaseline):
            raise ValueError("baseline must be a WorkspaceBaseline")
        if baseline.repo_root != self._repo_root:
            raise ValueError("baseline belongs to a different repository")
        if (
            len(baseline.baseline_id) != 32
            or any(character not in "0123456789abcdef" for character in baseline.baseline_id)
        ):
            raise ValueError("baseline identity is invalid")
        for record in reversed(baseline.paths):
            if record.before_image is None:
                continue
            path = validate_allowed_path(record.path)
            relative = PurePosixPath(baseline.baseline_id) / "before" / path
            if record.before_image != self._artifact_directory / relative:
                raise ValueError("baseline artifact path mismatch")
            try:
                self._remove_artifact_output(relative)
            except FileNotFoundError:
                pass
            self._prune_artifact_parents(relative)

    def widen_baseline(
        self, baseline: WorkspaceBaseline, brief: TaskBrief,
    ) -> WorkspaceBaseline:
        """Carry an original baseline into a newly approved path scope."""
        self._validate_baseline_identity(baseline)
        if not isinstance(brief, TaskBrief):
            raise ValueError("brief must be a TaskBrief")
        allowed = tuple(validate_allowed_path(value) for value in brief.allowed_paths)
        for path in allowed:
            protected_reason = self._protected_reason(path)
            if protected_reason is not None:
                raise ProtectedPathApprovalRequired(
                    f"protected path requires separate approval: {path} ({protected_reason})"
                )
        prior_allowed = tuple(
            validate_allowed_path(value) for value in baseline.allowed_paths
        )
        records = {record.path: record for record in baseline.paths}
        control_changes = self._git_control_snapshot_changes(baseline)
        if control_changes:
            raise ProtectedPathApprovalRequired(
                "Git control snapshot changed before scope widening: "
                + ", ".join(control_changes)
            )
        tracked_paths, current_index_sha256 = self._index_inventory_and_identity(
            baseline.git_dir, baseline.common_dir
        )
        if current_index_sha256 != baseline.git_index_sha256:
            raise ProtectedPathApprovalRequired(
                "Git semantic index changed before scope widening"
            )
        filesystem_paths = self._filesystem_inventory()
        created_artifacts: list[PurePosixPath] = []
        content_bytes_read = sum(
            record.size for record in baseline.paths if record.before_image is not None
        )
        try:
            inventory = self._inventory(
                tracked_paths=tracked_paths,
                filesystem_paths=filesystem_paths,
            )
            if self._git_control_snapshot_changes(baseline):
                raise ProtectedPathApprovalRequired(
                    "Git control snapshot changed during filesystem inventory"
                )
            for raw_path in inventory:
                path = PurePosixPath(raw_path)
                newly_allowed = self._is_allowed(path, allowed) and not self._is_allowed(
                    path, prior_allowed
                )
                if not newly_allowed:
                    continue
                source = self._repo_root / path
                try:
                    source_kind = _path_kind(self._repo_root, source)
                    entry = _entry_stat(self._repo_root, source)
                except _UnsafeRepositoryPath as error:
                    raise ProtectedPathApprovalRequired(
                        f"unsafe path requires separate approval: {path}"
                    ) from error
                if source_kind == "symlink":
                    raise ProtectedPathApprovalRequired(
                        f"unsafe symlink path requires separate approval: {path}"
                    )
                if source_kind != "regular" or entry is None:
                    raise ProtectedPathApprovalRequired(
                        f"non-regular path requires separate approval: {path}"
                    )
                original = records.get(raw_path)
                if original is None or not original.existed or original.sha256 is None:
                    raise ProtectedPathApprovalRequired(
                        f"path changed since the original baseline: {path}"
                    )
                if _metadata_sha256(entry) != original.sha256:
                    raise ProtectedPathApprovalRequired(
                        f"path changed since the original baseline: {path}"
                    )
                if original.before_image is not None:
                    continue
                artifact_path = PurePosixPath(baseline.baseline_id) / "before" / path
                before_image = self._artifact_directory / artifact_path
                snapshot_limit = min(
                    self._max_snapshot_bytes,
                    self._max_inventory_content_bytes - content_bytes_read,
                )
                try:
                    if entry.st_size > snapshot_limit:
                        raise _SnapshotTooLarge(source.as_posix())
                    destination = self._open_artifact_output(artifact_path)
                    # From this point onward every failure path must remove the
                    # newly created artifact, including final authentication and
                    # caller-side manifest serialization/persistence failures.
                    created_artifacts.append(artifact_path)
                    size, digest = _copy_regular_file(
                        self._repo_root, source, destination, snapshot_limit
                    )
                except _SnapshotTooLarge as error:
                    raise ProtectedPathApprovalRequired(
                        f"inventory content requires separate approval: {path} exceeds "
                        f"snapshot/content budget={snapshot_limit}"
                    ) from error
                except _UnsafeRepositoryPath as error:
                    raise ProtectedPathApprovalRequired(
                        f"unsafe path requires separate approval: {path}"
                    ) from error
                final_entry = _entry_stat(self._repo_root, source)
                if (
                    size != original.size
                    or final_entry is None
                    or _metadata_sha256(final_entry) != original.sha256
                ):
                    raise ProtectedPathApprovalRequired(
                        f"path changed while widening the original baseline: {path}"
                    )
                content_bytes_read += size
                records[raw_path] = replace(
                    original,
                    sha256=digest,
                    before_image=before_image,
                )
            return WorkspaceBaseline(
                baseline_id=baseline.baseline_id,
                repo_root=baseline.repo_root,
                repo_root_sha256=baseline.repo_root_sha256,
                head=baseline.head,
                branch=baseline.branch,
                git_marker_sha256=baseline.git_marker_sha256,
                git_dir=baseline.git_dir,
                git_dir_sha256=baseline.git_dir_sha256,
                common_dir=baseline.common_dir,
                common_dir_sha256=baseline.common_dir_sha256,
                git_index_sha256=baseline.git_index_sha256,
                allowed_paths=tuple(path.as_posix() for path in allowed),
                paths=tuple(sorted(records.values(), key=lambda record: record.path)),
                git_control=baseline.git_control,
            )
        except BaseException:
            for artifact_path in reversed(created_artifacts):
                self._remove_artifact_output(artifact_path)
                self._prune_artifact_parents(artifact_path)
            raise

    def discard_widening(
        self, original: WorkspaceBaseline, widened: WorkspaceBaseline,
    ) -> None:
        """Remove only before-images introduced by one unpersisted widening."""
        self._validate_baseline_identity(original)
        if not isinstance(widened, WorkspaceBaseline):
            raise ValueError("widened baseline must be a WorkspaceBaseline")
        if original.baseline_id != widened.baseline_id:
            raise ValueError("widened baseline identity mismatch")
        if widened.repo_root != original.repo_root:
            raise ValueError("widened baseline repository mismatch")
        prior = {record.path: record for record in original.paths}
        for record in widened.paths:
            previous = prior.get(record.path)
            if record.before_image is None or (
                previous is not None and previous.before_image is not None
            ):
                continue
            try:
                path = validate_allowed_path(record.path)
            except ValueError as error:
                raise ValueError("widened baseline contains an unsafe path") from error
            relative = PurePosixPath(original.baseline_id) / "before" / path
            expected = self._artifact_directory / relative
            if record.before_image != expected:
                raise ValueError("widened baseline artifact path mismatch")
            try:
                self._remove_artifact_output(relative)
            except FileNotFoundError:
                pass
            self._prune_artifact_parents(relative)

    def restore_baseline(
        self, manifest: object, *, expected_baseline_id: str,
    ) -> WorkspaceBaseline:
        """Reconstruct and validate one persisted baseline without repository writes."""
        if not isinstance(manifest, Mapping):
            raise RuntimeError("persisted baseline manifest must be an object")
        fields = {
            "baseline_id", "repo_root", "repo_root_sha256", "head", "branch",
            "git_marker_sha256", "git_dir", "git_dir_sha256", "common_dir",
            "common_dir_sha256", "git_index_sha256", "allowed_paths", "paths",
            "git_control",
        }
        if set(manifest) != fields:
            raise RuntimeError("persisted baseline manifest has invalid fields")
        baseline_id = manifest["baseline_id"]
        if baseline_id != expected_baseline_id:
            raise RuntimeError("persisted baseline identity mismatch")
        if not isinstance(baseline_id, str) or not baseline_id:
            raise RuntimeError("persisted baseline identity is invalid")
        if manifest["repo_root"] != str(self._repo_root):
            raise RuntimeError("persisted baseline repository root mismatch")
        head = manifest["head"]
        branch = manifest["branch"]
        repo_root_sha256 = manifest["repo_root_sha256"]
        git_marker_sha256 = manifest["git_marker_sha256"]
        git_dir_raw = manifest["git_dir"]
        git_dir_sha256 = manifest["git_dir_sha256"]
        common_dir_raw = manifest["common_dir"]
        common_dir_sha256 = manifest["common_dir_sha256"]
        git_index_sha256 = manifest["git_index_sha256"]
        allowed_raw = manifest["allowed_paths"]
        paths_raw = manifest["paths"]
        git_control_raw = manifest["git_control"]
        if not isinstance(head, str) or not isinstance(branch, str):
            raise RuntimeError("persisted baseline git identity is invalid")
        if not all(
            _is_sha256(value)
            for value in (
                repo_root_sha256,
                git_marker_sha256,
                git_dir_sha256,
                common_dir_sha256,
                git_index_sha256,
            )
        ):
            raise RuntimeError("persisted baseline root identity is invalid")
        if not isinstance(git_dir_raw, str) or not isinstance(common_dir_raw, str):
            raise RuntimeError("persisted baseline Git roots are invalid")
        git_dir = Path(git_dir_raw)
        common_dir = Path(common_dir_raw)
        if (
            not git_dir.is_absolute()
            or not common_dir.is_absolute()
            or Path(os.path.abspath(git_dir)) != git_dir
            or Path(os.path.abspath(common_dir)) != common_dir
        ):
            raise RuntimeError("persisted baseline Git roots are invalid")
        if not isinstance(allowed_raw, list) or not all(
            isinstance(path, str) for path in allowed_raw
        ):
            raise RuntimeError("persisted baseline allowed paths are invalid")
        allowed = tuple(validate_allowed_path(path).as_posix() for path in allowed_raw)
        if not isinstance(paths_raw, list):
            raise RuntimeError("persisted baseline paths are invalid")
        if not isinstance(git_control_raw, list):
            raise RuntimeError("persisted Git control manifest is invalid")
        git_control: list[tuple[str, str]] = []
        for item in git_control_raw:
            if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
                raise RuntimeError("persisted Git control entry is invalid")
            control_path = item["path"]
            control_digest = item["sha256"]
            if not isinstance(control_path, str) or not isinstance(control_digest, str):
                raise RuntimeError("persisted Git control entry identity is invalid")
            try:
                normalized_control = validate_allowed_path(control_path)
            except ValueError as error:
                raise RuntimeError(
                    "persisted Git control entry identity is invalid"
                ) from error
            if not _is_within(normalized_control, PurePosixPath(".git")):
                raise RuntimeError("persisted Git control entry is outside .git")
            if not _is_sha256(control_digest):
                raise RuntimeError("persisted Git control entry digest is invalid")
            git_control.append((normalized_control.as_posix(), control_digest))
        if (
            not git_control
            or git_control != sorted(git_control)
            or len({path for path, _ in git_control}) != len(git_control)
            or not {".git", ".git/HEAD", ".git/config"}.issubset(
                path for path, _ in git_control
            )
        ):
            raise RuntimeError("persisted Git control manifest is incomplete")
        records: list[PathBaseline] = []
        for raw_record in paths_raw:
            records.append(self._restore_path_baseline(baseline_id, raw_record))
        if (
            records != sorted(records, key=lambda record: record.path)
            or len({record.path for record in records}) != len(records)
        ):
            raise RuntimeError("persisted path baselines are not canonical")
        baseline = WorkspaceBaseline(
            baseline_id=baseline_id,
            repo_root=self._repo_root,
            repo_root_sha256=str(repo_root_sha256),
            head=head,
            branch=branch,
            git_marker_sha256=str(git_marker_sha256),
            git_dir=git_dir,
            git_dir_sha256=str(git_dir_sha256),
            common_dir=common_dir,
            common_dir_sha256=str(common_dir_sha256),
            git_index_sha256=str(git_index_sha256),
            allowed_paths=allowed,
            paths=tuple(records),
            git_control=tuple(git_control),
        )
        self._validate_baseline_identity(baseline)
        return baseline

    def _restore_path_baseline(
        self, baseline_id: str, raw_record: object,
    ) -> PathBaseline:
        if not isinstance(raw_record, Mapping):
            raise RuntimeError("persisted path baseline must be an object")
        fields = {
            "path", "existed", "kind", "size", "sha256", "before_image",
            "protected_reason",
        }
        if set(raw_record) != fields:
            raise RuntimeError("persisted path baseline has invalid fields")
        raw_path = raw_record["path"]
        existed = raw_record["existed"]
        kind = raw_record["kind"]
        size = raw_record["size"]
        digest = raw_record["sha256"]
        before_raw = raw_record["before_image"]
        protected_reason = raw_record["protected_reason"]
        if not isinstance(raw_path, str):
            raise RuntimeError("persisted path baseline path is invalid")
        path = validate_allowed_path(raw_path).as_posix()
        if not isinstance(existed, bool):
            raise RuntimeError("persisted path baseline existed flag is invalid")
        if kind not in {"missing", "regular", "directory", "symlink", "other"}:
            raise RuntimeError("persisted path baseline kind is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError("persisted path baseline size is invalid")
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError("persisted path baseline digest is invalid")
        if protected_reason is not None and not isinstance(protected_reason, str):
            raise RuntimeError("persisted path baseline protection is invalid")
        before_image: Path | None
        if before_raw is None:
            before_image = None
        elif isinstance(before_raw, str):
            before_image = Path(before_raw)
            expected = self._artifact_directory / baseline_id / "before" / path
            if before_image != expected:
                raise RuntimeError("persisted before-image path is outside its baseline")
        else:
            raise RuntimeError("persisted before-image path is invalid")
        if not existed and (
            kind != "missing"
            or size != 0
            or digest is not None
            or before_image is not None
        ):
            raise RuntimeError("persisted path baseline combination is invalid")
        if existed and (kind == "missing" or digest is None):
            raise RuntimeError("persisted path baseline combination is invalid")
        if before_image is not None and kind != "regular":
            raise RuntimeError("persisted path baseline combination is invalid")
        return PathBaseline(
            path=path,
            existed=existed,
            kind=str(kind),
            size=size,
            sha256=digest,
            before_image=before_image,
            protected_reason=protected_reason,
        )

    def _validate_baseline_identity(self, baseline: WorkspaceBaseline) -> None:
        if not isinstance(baseline, WorkspaceBaseline):
            raise ValueError("baseline must be a WorkspaceBaseline")
        if baseline.repo_root != self._repo_root:
            raise ValueError("baseline belongs to a different repository")
        if tuple(sorted(record.path for record in baseline.paths)) != tuple(
            record.path for record in baseline.paths
        ) or len({record.path for record in baseline.paths}) != len(baseline.paths):
            raise RuntimeError("baseline path records are not canonical")
        allowed = tuple(validate_allowed_path(path) for path in baseline.allowed_paths)
        self._validate_git_root_values(
            repo_root_sha256=baseline.repo_root_sha256,
            git_marker_sha256=baseline.git_marker_sha256,
            git_dir=baseline.git_dir,
            git_dir_sha256=baseline.git_dir_sha256,
            common_dir=baseline.common_dir,
            common_dir_sha256=baseline.common_dir_sha256,
        )
        total_before_bytes = 0
        for record in baseline.paths:
            path = validate_allowed_path(record.path)
            if not record.existed and (
                record.kind != "missing"
                or record.size != 0
                or record.sha256 is not None
                or record.before_image is not None
            ):
                raise RuntimeError("baseline path record combination is invalid")
            if record.existed and (
                record.kind not in {"regular", "directory", "symlink", "other"}
                or not _is_sha256(record.sha256)
            ):
                raise RuntimeError("baseline path record combination is invalid")
            requires_before_image = (
                record.existed
                and record.kind == "regular"
                and self._is_allowed(path, allowed)
            )
            if requires_before_image != (record.before_image is not None):
                raise RuntimeError(
                    "baseline before-image is required exactly for allowed regular files"
                )
            if record.before_image is None:
                continue
            if record.kind != "regular":
                raise RuntimeError("baseline path record combination is invalid")
            if not self._is_allowed(path, allowed):
                raise RuntimeError("baseline before-image is outside the allowed scope")
            expected = self._artifact_directory / baseline.baseline_id / "before" / record.path
            if record.before_image != expected:
                raise RuntimeError("baseline before-image path is outside its artifact directory")
            total_before_bytes += record.size
            if total_before_bytes > self._max_inventory_content_bytes:
                raise RuntimeError("aggregate before-image integrity budget exceeded")
            self._validate_before_image(record)

    def compare(self, baseline: WorkspaceBaseline) -> WorkspaceDelta:
        """Report repository changes since ``baseline`` without writing to it."""
        self._require_open()
        if not isinstance(baseline, WorkspaceBaseline):
            raise ValueError("baseline must be a WorkspaceBaseline")
        self._validate_baseline_identity(baseline)
        allowed = tuple(validate_allowed_path(path) for path in baseline.allowed_paths)
        prior = {record.path: record for record in baseline.paths}
        control_changes = self._git_control_snapshot_changes(baseline)
        if control_changes:
            return self._protected_control_delta(control_changes)
        tracked_paths, current_index_sha256 = self._index_inventory_and_identity(
            baseline.git_dir, baseline.common_dir
        )
        filesystem_paths = self._filesystem_inventory()
        current = set(self._inventory(
            tracked_paths=tracked_paths,
            filesystem_paths=filesystem_paths,
        ))
        control_changes = self._git_control_snapshot_changes(baseline)
        if control_changes:
            return self._protected_control_delta(control_changes)
        paths = sorted(current | set(prior))
        changed: list[str] = []
        unexpected: list[str] = []
        protected_changed: list[str] = list(
            self._git_semantic_changes(baseline, current_index_sha256)
        )
        control_changes = self._git_control_snapshot_changes(baseline)
        if control_changes:
            return self._protected_control_delta(control_changes)
        unchanged: list[str] = []
        text_diffs: dict[str, str] = {}
        binary_changed: list[str] = []
        current_kinds: dict[str, str] = {}
        content_bytes_read = 0

        for path in paths:
            relative = PurePosixPath(path)
            record = prior.get(path)
            current_path = self._repo_root / relative
            current_bytes: bytes | None = None
            try:
                current_kind = _path_kind(self._repo_root, current_path) if path in current else "missing"
                current_kinds[path] = current_kind
                if current_kind == "regular" and self._is_allowed(relative, allowed):
                    current_bytes = _read_regular_bytes(
                        self._repo_root,
                        current_path,
                        self._max_inventory_content_bytes - content_bytes_read,
                    )
                    content_bytes_read += len(current_bytes)
                    current_hash = hashlib.sha256(current_bytes).hexdigest()
                elif current_kind == "missing":
                    current_hash = None
                else:
                    entry = _entry_stat(self._repo_root, current_path)
                    current_hash = None if entry is None else _entry_identity_sha256(entry)
            except _SnapshotTooLarge as error:
                raise ProtectedPathApprovalRequired(
                    f"inventory content exceeds max_inventory_content_bytes: {path}"
                ) from error
            except _UnsafeRepositoryPath:
                current_hash = None
                current_kind = "unsafe"
            prior_hash = None if record is None else record.sha256
            if current_hash == prior_hash:
                if record is not None and record.existed and self._is_allowed(relative, allowed):
                    unchanged.append(path)
                continue

            structural_directory = current_kind == "directory" or (
                record is not None and record.kind == "directory"
            )
            in_allowed_scope = self._is_allowed(relative, allowed) or (
                structural_directory
                and any(relative in candidate.parents for candidate in allowed)
            )
            protected_reason = self._protected_reason(relative)
            if protected_reason is not None or current_kind in {"symlink", "unsafe"}:
                protected_changed.append(path)
            elif in_allowed_scope:
                changed.append(path)
                if current_kind == "regular" or (record is not None and record.before_image is not None):
                    diff = self._text_diff(record, current_bytes, path)
                    if diff is None:
                        binary_changed.append(path)
                    else:
                        text_diffs[path] = diff
            else:
                unexpected.append(path)

        return WorkspaceDelta(
            changed_paths=self._leaf_changes(changed, prior, current_kinds),
            unexpected_paths=self._leaf_changes(unexpected, prior, current_kinds),
            protected_changed_paths=self._leaf_changes(
                sorted(set(protected_changed)), prior, current_kinds
            ),
            preexisting_unchanged_paths=tuple(unchanged),
            text_diffs=text_diffs,
            binary_changed_paths=tuple(binary_changed),
        )

    @staticmethod
    def _protected_control_delta(
        control_changes: tuple[str, ...]
    ) -> WorkspaceDelta:
        protected = set(control_changes)
        if any(
            _is_within(PurePosixPath(path), PurePosixPath(".git/objects"))
            for path in control_changes
        ):
            protected.add(".git/index")
        return WorkspaceDelta(
            changed_paths=(),
            unexpected_paths=(),
            protected_changed_paths=tuple(sorted(protected)),
            preexisting_unchanged_paths=(),
        )

    @staticmethod
    def _leaf_changes(
        paths: list[str],
        prior: Mapping[str, PathBaseline],
        current_kinds: Mapping[str, str],
    ) -> tuple[str, ...]:
        """Suppress container-only ancestors while retaining exact empty dirs."""
        candidates = tuple(sorted(set(paths)))
        result: list[str] = []
        for path in candidates:
            record = prior.get(path)
            was_directory = record is not None and record.kind == "directory"
            is_directory = current_kinds.get(path) == "directory"
            relative = PurePosixPath(path)
            if (was_directory or is_directory) and any(
                relative in PurePosixPath(other).parents for other in candidates
            ):
                continue
            result.append(path)
        return tuple(result)

    def _inventory(
        self,
        *,
        tracked_paths: frozenset[str],
        filesystem_paths: frozenset[str],
    ) -> tuple[str, ...]:
        merged = tracked_paths | filesystem_paths
        if len(merged) > self._max_inventory_entries:
            raise ProtectedPathApprovalRequired(
                "inventory exceeds max_inventory_entries"
            )
        listing_bytes = sum(
            len(path.encode("utf-8")) + 1 for path in merged
        )
        if listing_bytes > self._max_inventory_listing_bytes:
            raise ProtectedPathApprovalRequired(
                "inventory exceeds max_inventory_listing_bytes"
            )
        return tuple(sorted(merged))

    def _parse_tracked_paths(self, raw_listing: bytes) -> frozenset[str]:
        paths: set[str] = set()
        for raw in raw_listing.split(b"\0"):
            if not raw:
                continue
            try:
                paths.add(validate_allowed_path(raw.decode("utf-8")).as_posix())
            except UnicodeDecodeError as error:
                raise ValueError("repository paths must be valid UTF-8") from error
            if len(paths) > self._max_inventory_entries:
                raise ProtectedPathApprovalRequired(
                    "tracked inventory exceeds max_inventory_entries"
                )
        return frozenset(paths)

    def _git_text(self, *args: str) -> str:
        return self._git_bytes("Git identity", *args).decode("utf-8")

    def _git_bytes(self, label: str, *args: str) -> bytes:
        """Read bounded Git output without allocating beyond the configured cap."""
        return self._run_git_bytes(label, args, git_dir=None, common_dir=None)

    def _anchored_git_text(
        self, git_dir: Path, common_dir: Path, *args: str
    ) -> str:
        return self._anchored_git_bytes(
            git_dir, common_dir, "Git identity", *args
        ).decode("utf-8")

    def _anchored_git_bytes(
        self, git_dir: Path, common_dir: Path, label: str, *args: str
    ) -> bytes:
        controls_before = self._git_execution_control_snapshot(
            git_dir, common_dir
        )
        output = self._run_git_bytes(
            label,
            args,
            git_dir=git_dir,
            common_dir=common_dir,
        )
        if self._git_execution_control_snapshot(
            git_dir, common_dir
        ) != controls_before:
            raise ProtectedPathApprovalRequired(
                f"{label} raced with a Git control change"
            )
        return output

    def _run_git_bytes(
        self,
        label: str,
        args: tuple[str, ...],
        *,
        git_dir: Path | None,
        common_dir: Path | None,
        index_file: Path | None = None,
        inherited_fds: tuple[int, ...] = (),
    ) -> bytes:
        if (git_dir is None) != (common_dir is None):
            raise ValueError("Git anchors must be supplied together")
        command = [
            str(self._git_executable),
            "--no-pager",
            "-c",
            f"core.excludesFile={os.devnull}",
            "-c",
            "core.ignoreCase=false",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "credential.helper=",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            f"core.worktree={self._repo_root}",
        ]
        if git_dir is not None:
            command.extend((
                f"--git-dir={git_dir}",
                f"--work-tree={self._repo_root}",
            ))
        command.extend(args)
        environment = {
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
        if git_dir is not None and common_dir is not None:
            environment.update({
                "GIT_DIR": str(git_dir),
                "GIT_WORK_TREE": str(self._repo_root),
                "GIT_INDEX_FILE": str(index_file or (git_dir / "index")),
                "GIT_COMMON_DIR": str(common_dir),
                "GIT_OBJECT_DIRECTORY": str(common_dir / "objects"),
                "GIT_ALTERNATE_OBJECT_DIRECTORIES": "",
            })
        process = subprocess.Popen(
            command,
            cwd=self._repo_root,
            env=environment,
            pass_fds=inherited_fds,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if process.stdout is None:
            process.kill()
            process.wait()
            raise RuntimeError("Git stdout pipe was not created")
        chunks: list[bytes] = []
        size = 0
        deadline = time.monotonic() + self._git_timeout_seconds
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    self._terminate_git_process(process)
                    raise ProtectedPathApprovalRequired(
                        f"{label} exceeded Git subprocess deadline"
                    )
                block = os.read(
                    process.stdout.fileno(),
                    min(64 * 1024, self._max_inventory_listing_bytes - size + 1),
                )
                if not block:
                    break
                if size + len(block) > self._max_inventory_listing_bytes:
                    self._terminate_git_process(process)
                    raise ProtectedPathApprovalRequired(
                        f"{label} exceeds max_inventory_listing_bytes"
                    )
                chunks.append(block)
                size += len(block)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate_git_process(process)
                raise ProtectedPathApprovalRequired(
                    f"{label} exceeded Git subprocess deadline"
                )
            try:
                exit_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                self._terminate_git_process(process)
                raise ProtectedPathApprovalRequired(
                    f"{label} exceeded Git subprocess deadline"
                ) from error
        finally:
            selector.close()
            process.stdout.close()
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, command)
        return b"".join(chunks)

    @staticmethod
    def _terminate_git_process(process: subprocess.Popen[bytes]) -> None:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _filesystem_inventory(self) -> frozenset[str]:
        """List every present worktree entry with one bounded no-follow walk."""
        stack: list[PurePosixPath] = [PurePosixPath()]
        paths: set[str] = set()
        visited = 0
        listing_bytes = 0
        legacy_bytes = 0
        while stack:
            prefix = stack.pop()
            descriptor = _open_absolute_directory(
                self._repo_root / prefix, create=False
            )
            try:
                with os.scandir(os.dup(descriptor)) as entries:
                    for entry in entries:
                        try:
                            encoded_name = entry.name.encode("utf-8")
                        except UnicodeEncodeError as error:
                            raise ValueError(
                                "repository paths must be valid UTF-8"
                            ) from error
                        relative = prefix / entry.name
                        visited += 1
                        listing_bytes += len(encoded_name) + len(prefix.as_posix()) + 2
                        if visited > self._max_inventory_entries:
                            raise ProtectedPathApprovalRequired(
                                "filesystem inventory exceeds max_inventory_entries"
                            )
                        if listing_bytes > self._max_inventory_listing_bytes:
                            raise ProtectedPathApprovalRequired(
                                "filesystem inventory exceeds max_inventory_listing_bytes"
                            )
                        if visited > self._max_ignored_entries:
                            raise ProtectedPathApprovalRequired(
                                "ignored inventory exceeds max_ignored_entries"
                            )
                        if self._is_artifact_path(relative):
                            continue
                        entry_stat = entry.stat(follow_symlinks=False)
                        if self._max_ignored_bytes is not None:
                            legacy_bytes += entry_stat.st_size
                            if legacy_bytes > self._max_ignored_bytes:
                                raise ProtectedPathApprovalRequired(
                                    "ignored inventory exceeds max_ignored_bytes"
                                )
                        paths.add(relative.as_posix())
                        if stat.S_ISDIR(entry_stat.st_mode):
                            if relative == PurePosixPath(".git"):
                                continue
                            stack.append(relative)
            finally:
                os.close(descriptor)
        return frozenset(paths)

    def _absolute_identity(
        self, path: Path, *, hash_regular_content: bool = False
    ) -> str:
        try:
            digest, _ = _absolute_entry_sha256(
                path,
                hash_regular_content=hash_regular_content,
                max_bytes=self._max_inventory_content_bytes,
            )
            return digest
        except (OSError, _SnapshotTooLarge, _UnsafeRepositoryPath) as error:
            raise ProtectedPathApprovalRequired(
                f"Git root/control identity is unsafe: {path}"
            ) from error

    def _structural_git_roots(self) -> tuple[Path, Path]:
        """Derive exact normal/linked Git roots without invoking Git."""
        marker = self._repo_root / ".git"
        try:
            marker_entry = _entry_stat(self._repo_root, marker)
            if marker_entry is None:
                raise _UnsafeRepositoryPath("repository .git marker is missing")
            if stat.S_ISDIR(marker_entry.st_mode):
                git_dir = marker
                common_dir = marker
            elif stat.S_ISREG(marker_entry.st_mode):
                marker_text = _read_regular_bytes(
                    self._repo_root,
                    marker,
                    min(4096, self._max_inventory_content_bytes),
                ).decode("utf-8").strip()
                if not marker_text.startswith("gitdir: "):
                    raise _UnsafeRepositoryPath(
                        "linked-worktree .git marker is invalid"
                    )
                declared = Path(marker_text.removeprefix("gitdir: "))
                if not declared.is_absolute():
                    declared = self._repo_root / declared
                git_dir = Path(os.path.abspath(declared))
                commondir_path = git_dir / "commondir"
                try:
                    common_text = _read_absolute_regular_bytes(
                        commondir_path, 4096
                    ).decode("utf-8").strip()
                except FileNotFoundError:
                    common_dir = git_dir
                else:
                    if not common_text:
                        raise _UnsafeRepositoryPath(
                            "linked-worktree commondir is invalid"
                        )
                    common_candidate = Path(common_text)
                    if not common_candidate.is_absolute():
                        common_candidate = git_dir / common_candidate
                    common_dir = Path(os.path.abspath(common_candidate))
            else:
                raise _UnsafeRepositoryPath(
                    "repository .git marker has unsafe type"
                )
            git_descriptor = _open_absolute_directory(git_dir, create=False)
            os.close(git_descriptor)
            common_descriptor = _open_absolute_directory(common_dir, create=False)
            os.close(common_descriptor)
            return git_dir, common_dir
        except (
            OSError,
            UnicodeDecodeError,
            _SnapshotTooLarge,
            _UnsafeRepositoryPath,
        ) as error:
            raise ProtectedPathApprovalRequired(
                "Git root relationship is unsafe"
            ) from error

    def _validate_git_relationship(self, git_dir: Path, common_dir: Path) -> None:
        derived_git, derived_common = self._structural_git_roots()
        if derived_git != git_dir or derived_common != common_dir:
            raise ProtectedPathApprovalRequired("Git root relationship changed")

    def _validate_local_git_configuration(
        self, git_dir: Path, common_dir: Path
    ) -> None:
        """Reject local Git configuration that can redirect trusted reads."""
        paths = {
            common_dir / "config",
            common_dir / "config.worktree",
            git_dir / "config",
            git_dir / "config.worktree",
        }
        consumed = 0
        for path in sorted(paths):
            try:
                entry_digest, _ = _absolute_entry_sha256(
                    path,
                    hash_regular_content=False,
                    max_bytes=0,
                )
                del entry_digest
                contents = _read_absolute_regular_bytes(
                    path, _MAX_GIT_CONFIGURATION_BYTES - consumed
                )
            except FileNotFoundError:
                continue
            except (OSError, _SnapshotTooLarge, _UnsafeRepositoryPath) as error:
                raise ProtectedPathApprovalRequired(
                    "Git local configuration has an unsafe type or size"
                ) from error
            consumed += len(contents)
            self._validate_git_config_text(path, contents)

        for alternates in (
            common_dir / "objects" / "info" / "alternates",
            common_dir / "objects" / "info" / "http-alternates",
        ):
            try:
                contents = _read_absolute_regular_bytes(
                    alternates, _MAX_GIT_CONFIGURATION_BYTES - consumed
                )
            except FileNotFoundError:
                continue
            except (OSError, _SnapshotTooLarge, _UnsafeRepositoryPath) as error:
                raise ProtectedPathApprovalRequired(
                    "Git alternate object configuration is unsafe"
                ) from error
            consumed += len(contents)
            if contents.strip():
                raise ProtectedPathApprovalRequired(
                    "Git alternate object configuration is not allowed"
                )

    def _git_execution_control_snapshot(
        self, git_dir: Path, common_dir: Path
    ) -> tuple[tuple[str, str], ...]:
        """Authenticate only controls Git may consult around one subprocess."""
        self._validate_git_relationship(git_dir, common_dir)
        self._validate_local_git_configuration(git_dir, common_dir)
        self._validate_git_index_path(git_dir)
        paths = {
            self._repo_root / ".git",
            git_dir,
            common_dir,
            common_dir / "config",
            common_dir / "config.worktree",
            git_dir / "config",
            git_dir / "config.worktree",
            common_dir / "objects" / "info" / "alternates",
            common_dir / "objects" / "info" / "http-alternates",
        }
        consumed = 0
        snapshot: list[tuple[str, str]] = []
        try:
            for path in sorted(paths):
                digest, size = _absolute_entry_sha256(
                    path,
                    hash_regular_content=True,
                    max_bytes=_MAX_GIT_CONFIGURATION_BYTES - consumed,
                )
                consumed += size
                snapshot.append((str(path), digest))
        except (OSError, _SnapshotTooLarge, _UnsafeRepositoryPath) as error:
            raise ProtectedPathApprovalRequired(
                "Git execution control snapshot is unsafe"
            ) from error
        return tuple(snapshot)

    def _validate_git_config_text(self, path: Path, contents: bytes) -> None:
        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProtectedPathApprovalRequired(
                f"Git local configuration is not UTF-8: {path}"
            ) from error
        section = ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", ";")):
                continue
            if line.endswith("\\"):
                raise ProtectedPathApprovalRequired(
                    "Git local configuration continuations are not allowed"
                )
            if line.startswith("["):
                closing = line.find("]")
                if closing <= 1:
                    raise ProtectedPathApprovalRequired(
                        "Git local configuration section is invalid"
                    )
                header = line[1:closing].strip()
                section = header.split(maxsplit=1)[0].strip('"').casefold()
                if section.startswith("include"):
                    raise ProtectedPathApprovalRequired(
                        "Git local configuration includes are not allowed"
                    )
                continue
            if "=" in line:
                raw_key, raw_value = line.split("=", 1)
            else:
                fields = line.split(maxsplit=1)
                raw_key = fields[0]
                raw_value = "true" if len(fields) == 1 else fields[1]
            key = raw_key.strip().casefold().replace("-", "")
            value = raw_value.strip().strip('"').casefold()
            if key.startswith("include"):
                raise ProtectedPathApprovalRequired(
                    "Git local configuration includes are not allowed"
                )
            if section != "core":
                continue
            if key == "fsmonitor" and value in {"", "0", "false", "no", "off"}:
                continue
            safe_path_values = {
                "excludesfile": os.devnull.casefold(),
                "hookspath": os.devnull.casefold(),
                "worktree": str(self._repo_root).casefold(),
            }
            if key in {"excludesfile", "fsmonitor", "hookspath", "worktree"}:
                if safe_path_values.get(key) != value:
                    raise ProtectedPathApprovalRequired(
                        f"Git local configuration may redirect trusted reads: core.{key}"
                    )

    def _validate_git_root_values(
        self,
        *,
        repo_root_sha256: str,
        git_marker_sha256: str,
        git_dir: Path,
        git_dir_sha256: str,
        common_dir: Path,
        common_dir_sha256: str,
    ) -> None:
        if self._absolute_identity(self._repo_root) != repo_root_sha256:
            raise ProtectedPathApprovalRequired("Git root identity changed")
        if self._absolute_identity(
            self._repo_root / ".git", hash_regular_content=True
        ) != git_marker_sha256:
            raise ProtectedPathApprovalRequired("Git root marker identity changed")
        if self._absolute_identity(git_dir) != git_dir_sha256:
            raise ProtectedPathApprovalRequired("Git root identity changed")
        if self._absolute_identity(common_dir) != common_dir_sha256:
            raise ProtectedPathApprovalRequired("Git common root identity changed")
        self._validate_git_relationship(git_dir, common_dir)
        self._validate_local_git_configuration(git_dir, common_dir)
        self._validate_git_index_path(git_dir)

    def _validate_git_index_path(self, git_dir: Path) -> None:
        descriptor = _open_absolute_directory(git_dir, create=False)
        try:
            try:
                entry = os.stat("index", dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError as error:
                raise ProtectedPathApprovalRequired(
                    "Git index is missing from its captured root"
                ) from error
            if not stat.S_ISREG(entry.st_mode):
                raise ProtectedPathApprovalRequired(
                    "Git index must remain a no-follow regular file"
                )
        except OSError as error:
            raise ProtectedPathApprovalRequired(
                "Git index identity is unsafe"
            ) from error
        finally:
            os.close(descriptor)

    def _reject_split_index(self, git_dir: Path, common_dir: Path) -> None:
        visited = 0
        listing_bytes = 0
        for root in {git_dir, common_dir}:
            descriptor = _open_absolute_directory(root, create=False)
            try:
                with os.scandir(os.dup(descriptor)) as entries:
                    for entry in entries:
                        visited += 1
                        listing_bytes += len(entry.name.encode("utf-8")) + 1
                        if (
                            visited > self._max_inventory_entries
                            or listing_bytes > self._max_inventory_listing_bytes
                        ):
                            raise ProtectedPathApprovalRequired(
                                "Git split-index inspection exceeds inventory bounds"
                            )
                        if entry.name.startswith("sharedindex."):
                            raise ProtectedPathApprovalRequired(
                                "Git split/shared index is not supported"
                            )
            except UnicodeEncodeError as error:
                raise ProtectedPathApprovalRequired(
                    "Git split-index paths must be valid UTF-8"
                ) from error
            finally:
                os.close(descriptor)

    def _index_inventory_and_identity(
        self, git_dir: Path, common_dir: Path
    ) -> tuple[frozenset[str], str]:
        """Read Git's index only through one private authenticated snapshot."""
        self._validate_git_index_path(git_dir)
        self._reject_split_index(git_dir, common_dir)
        index_path = git_dir / "index"
        try:
            source_bytes = _read_absolute_regular_bytes(
                index_path, _MAX_GIT_INDEX_BYTES
            )
        except (OSError, _SnapshotTooLarge, _UnsafeRepositoryPath) as error:
            raise ProtectedPathApprovalRequired(
                "Git index cannot be snapshotted safely"
            ) from error
        artifact_path = (
            PurePosixPath(".git-index-snapshots") / uuid.uuid4().hex / "index"
        )
        destination = self._open_artifact_output(artifact_path)
        snapshot_descriptor = -1
        snapshot_linked = True
        try:
            with os.fdopen(destination, "wb") as output:
                output.write(source_bytes)
            snapshot_descriptor = self._open_artifact_input(artifact_path)
            os.fchmod(snapshot_descriptor, 0o400)
            self._remove_artifact_output(artifact_path)
            snapshot_linked = False
            self._prune_artifact_parents(artifact_path)
            snapshot_path = Path("/proc/self/fd") / str(snapshot_descriptor)
            controls = self._git_execution_control_snapshot(git_dir, common_dir)
            tracked_listing = self._run_git_bytes(
                "authenticated tracked index",
                ("ls-files", "-z", "--cached"),
                git_dir=git_dir,
                common_dir=common_dir,
                index_file=snapshot_path,
                inherited_fds=(snapshot_descriptor,),
            )
            self._authenticate_index_snapshot(
                index_path, snapshot_descriptor, source_bytes
            )
            if self._git_execution_control_snapshot(
                git_dir, common_dir
            ) != controls:
                raise ProtectedPathApprovalRequired(
                    "tracked index raced with a Git control change"
                )
            semantic_listing = self._run_git_bytes(
                "Git semantic index",
                ("ls-files", "--stage", "-v", "-z"),
                git_dir=git_dir,
                common_dir=common_dir,
                index_file=snapshot_path,
                inherited_fds=(snapshot_descriptor,),
            )
            self._authenticate_index_snapshot(
                index_path, snapshot_descriptor, source_bytes
            )
            if self._git_execution_control_snapshot(
                git_dir, common_dir
            ) != controls:
                raise ProtectedPathApprovalRequired(
                    "semantic index raced with a Git control change"
                )
            return (
                self._parse_tracked_paths(tracked_listing),
                hashlib.sha256(semantic_listing).hexdigest(),
            )
        finally:
            if snapshot_descriptor != -1:
                os.close(snapshot_descriptor)
            if snapshot_linked:
                try:
                    self._remove_artifact_output(artifact_path)
                except FileNotFoundError:
                    pass
                self._prune_artifact_parents(artifact_path)

    @staticmethod
    def _authenticate_index_snapshot(
        source_path: Path, snapshot_descriptor: int, expected: bytes
    ) -> None:
        try:
            source = _read_absolute_regular_bytes(
                source_path, _MAX_GIT_INDEX_BYTES
            )
            snapshot = _read_open_regular_bytes(
                snapshot_descriptor, _MAX_GIT_INDEX_BYTES
            )
        except (OSError, _SnapshotTooLarge, _UnsafeRepositoryPath) as error:
            raise ProtectedPathApprovalRequired(
                "Git index changed type while its snapshot was in use"
            ) from error
        if source != expected or snapshot != expected:
            raise ProtectedPathApprovalRequired(
                "Git index bytes changed while its snapshot was in use"
            )

    def _git_control_manifest(
        self, git_dir: Path, common_dir: Path
    ) -> tuple[tuple[str, str], ...]:
        budget = [0, 0, 0]
        manifest = {
            ".git": self._absolute_identity(
                self._repo_root / ".git", hash_regular_content=True
            )
        }
        budget[0] = 1
        budget[1] = len(b".git")
        try:
            self._extend_git_control_tree(
                manifest,
                git_dir,
                lambda relative: PurePosixPath(".git") / relative,
                budget,
            )
            if common_dir != git_dir:
                self._extend_git_control_tree(
                    manifest,
                    common_dir,
                    self._common_git_control_name,
                    budget,
                )
        except (OSError, _SnapshotTooLarge, _UnsafeRepositoryPath) as error:
            raise ProtectedPathApprovalRequired(
                "Git control inventory exceeds a safe identity/content bound"
            ) from error
        return tuple(sorted(manifest.items()))

    def _extend_git_control_tree(
        self,
        manifest: dict[str, str],
        root: Path,
        conceptual_name: Callable[[PurePosixPath], PurePosixPath],
        budget: list[int],
    ) -> None:
        """Add bounded no-follow semantic Git controls and object metadata."""
        stack: list[PurePosixPath] = [PurePosixPath()]
        while stack:
            prefix = stack.pop()
            descriptor = _open_absolute_directory(root / prefix, create=False)
            try:
                with os.scandir(os.dup(descriptor)) as entries:
                    for entry in entries:
                        try:
                            entry.name.encode("utf-8")
                        except UnicodeEncodeError as error:
                            raise ValueError(
                                "Git control paths must be valid UTF-8"
                            ) from error
                        relative = prefix / entry.name
                        conceptual = conceptual_name(relative)
                        normalized = validate_allowed_path(
                            conceptual.as_posix()
                        ).as_posix()
                        budget[0] += 1
                        budget[1] += len(normalized.encode("utf-8")) + 1
                        if budget[0] > self._max_inventory_entries:
                            raise ProtectedPathApprovalRequired(
                                "Git control inventory exceeds max_inventory_entries"
                            )
                        if budget[1] > self._max_inventory_listing_bytes:
                            raise ProtectedPathApprovalRequired(
                                "Git control inventory exceeds max_inventory_listing_bytes"
                            )
                        entry_stat = entry.stat(follow_symlinks=False)
                        if normalized == ".git/index":
                            continue
                        if stat.S_ISREG(entry_stat.st_mode) and not _is_within(
                            PurePosixPath(normalized), PurePosixPath(".git/objects")
                        ):
                            remaining = self._max_inventory_content_bytes - budget[2]
                            digest, consumed = _absolute_entry_sha256(
                                root / relative,
                                hash_regular_content=True,
                                max_bytes=remaining,
                            )
                            budget[2] += consumed
                        else:
                            digest = _entry_identity_sha256(entry_stat)
                        previous = manifest.get(normalized)
                        if previous is not None and previous != digest:
                            raise ProtectedPathApprovalRequired(
                                "Git control inventory has an ambiguous identity"
                            )
                        manifest[normalized] = digest
                        if stat.S_ISDIR(entry_stat.st_mode):
                            stack.append(relative)
            finally:
                os.close(descriptor)

    @staticmethod
    def _common_git_control_name(relative: PurePosixPath) -> PurePosixPath:
        first = relative.parts[0]
        if first in {"hooks", "info", "objects", "refs"} or (
            len(relative.parts) == 1
            and first in {"config", "packed-refs"}
        ):
            return PurePosixPath(".git") / relative
        return PurePosixPath(".git/common") / relative

    def _git_control_snapshot_changes(
        self, baseline: WorkspaceBaseline
    ) -> tuple[str, ...]:
        self._validate_git_relationship(baseline.git_dir, baseline.common_dir)
        self._validate_local_git_configuration(
            baseline.git_dir, baseline.common_dir
        )
        self._validate_git_index_path(baseline.git_dir)
        current = dict(
            self._git_control_manifest(baseline.git_dir, baseline.common_dir)
        )
        prior = dict(baseline.git_control)
        changed = {
            path
            for path in current.keys() | prior.keys()
            if current.get(path) != prior.get(path)
        }
        return tuple(sorted(changed))

    def _git_semantic_changes(
        self, baseline: WorkspaceBaseline, current_index_sha256: str
    ) -> tuple[str, ...]:
        changed: set[str] = set()
        if current_index_sha256 != baseline.git_index_sha256:
            changed.add(".git/index")
        head = self._anchored_git_text(
            baseline.git_dir, baseline.common_dir, "rev-parse", "HEAD"
        ).strip()
        self._require_unchanged_git_controls(baseline, "Git HEAD identity")
        branch = self._anchored_git_text(
            baseline.git_dir, baseline.common_dir, "branch", "--show-current"
        ).strip()
        self._require_unchanged_git_controls(baseline, "Git branch identity")
        if head != baseline.head or branch != baseline.branch:
            changed.add(".git/HEAD")
        return tuple(sorted(changed))

    def _require_unchanged_git_controls(
        self, baseline: WorkspaceBaseline, label: str
    ) -> None:
        changes = self._git_control_snapshot_changes(baseline)
        if changes:
            raise ProtectedPathApprovalRequired(
                f"{label} raced with a Git control change"
            )

    def _protected_reason(self, path: PurePosixPath) -> str | None:
        for prefix in self._protected_prefixes:
            if _is_within(path, prefix):
                return prefix.as_posix()
        return None

    def _is_allowed(self, path: PurePosixPath, allowed: tuple[PurePosixPath, ...]) -> bool:
        return any(_is_within(path, candidate) for candidate in allowed)

    def _open_artifact_parent(self, path: PurePosixPath, *, create: bool) -> tuple[int, str]:
        descriptor = os.dup(self._require_open())
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            for component in path.parts[:-1]:
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except OSError as error:
            os.close(descriptor)
            raise RuntimeError("baseline artifact path is unsafe") from error
        return descriptor, path.parts[-1]

    def _open_artifact_output(self, path: PurePosixPath) -> int:
        parent, name = self._open_artifact_parent(path, create=True)
        try:
            return os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
        except OSError as error:
            raise RuntimeError("baseline artifact path is unsafe") from error
        finally:
            os.close(parent)

    def _open_artifact_input(self, path: PurePosixPath) -> int:
        parent, name = self._open_artifact_parent(path, create=False)
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                dir_fd=parent,
            )
        except OSError as error:
            raise RuntimeError("baseline artifact path is unsafe") from error
        finally:
            os.close(parent)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise RuntimeError("baseline artifact path is unsafe")
        return descriptor

    def _remove_artifact_output(self, path: PurePosixPath) -> None:
        parent, name = self._open_artifact_parent(path, create=False)
        try:
            os.unlink(name, dir_fd=parent)
        finally:
            os.close(parent)

    def _prune_artifact_parents(self, path: PurePosixPath) -> None:
        current = path.parent
        while current.parts:
            parent, name = self._open_artifact_parent(current, create=False)
            try:
                os.rmdir(name, dir_fd=parent)
            except FileNotFoundError:
                pass
            except OSError as error:
                if error.errno in {errno.ENOTEMPTY, errno.EEXIST}:
                    return
                raise
            finally:
                os.close(parent)
            current = current.parent

    def _validate_before_image(self, record: PathBaseline) -> bytes:
        if record.before_image is None or record.sha256 is None:
            raise RuntimeError("baseline before-image integrity check failed")
        try:
            relative = PurePosixPath(record.before_image.relative_to(self._artifact_directory).as_posix())
            validate_allowed_path(relative.as_posix())
            parent, name = self._open_artifact_parent(relative, create=False)
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                    dir_fd=parent,
                )
            finally:
                os.close(parent)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise RuntimeError("baseline artifact is not a regular file")
                with os.fdopen(descriptor, "rb") as source:
                    descriptor = -1
                    contents = source.read(record.size + 1)
            finally:
                if descriptor != -1:
                    os.close(descriptor)
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError("baseline before-image integrity check failed") from error
        if len(contents) != record.size or hashlib.sha256(contents).hexdigest() != record.sha256:
            raise RuntimeError("baseline before-image integrity check failed")
        return contents

    def _read_before_image(self, record: PathBaseline) -> str:
        contents = self._validate_before_image(record)
        try:
            return contents.decode("utf-8")
        except UnicodeDecodeError:
            raise

    def _allowed_path_is_directory(self, path: PurePosixPath) -> bool:
        candidate = self._repo_root / path
        return _path_kind(self._repo_root, candidate) == "directory"

    def _is_artifact_path(self, path: PurePosixPath) -> bool:
        try:
            artifact_relative = self._artifact_directory.relative_to(self._repo_root)
        except ValueError:
            return False
        return _is_within(path, PurePosixPath(artifact_relative.as_posix()))

    def _text_diff(
        self, record: PathBaseline | None, current_bytes: bytes | None, path: str
    ) -> str | None:
        try:
            before = (
                self._read_before_image(record).splitlines(keepends=True)
                if record is not None and record.before_image is not None
                else []
            )
            current = (
                current_bytes.decode("utf-8").splitlines(keepends=True)
                if current_bytes is not None
                else []
            )
        except UnicodeDecodeError:
            return None
        return "".join(difflib.unified_diff(
            before,
            current,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        ))
