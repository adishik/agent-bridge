"""Immutable startup project configuration with opaque state identities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import time


MAX_PROJECTS = 32
_LABEL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}", re.ASCII)
_MAX_GIT_OUTPUT_BYTES = 64 * 1024
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


@dataclass(frozen=True, slots=True)
class ProjectSpec:
    project_id: str
    label: str
    repo_root: Path
    branch: str
    state_dir: Path


def parse_project_argument(value: str) -> tuple[str, Path]:
    """Parse one ``LABEL=/absolute/repository`` startup argument."""
    if not isinstance(value, str) or _contains_control_character(value):
        raise ValueError("project argument must contain a safe label and absolute root")
    try:
        label, root_value = value.split("=", 1)
    except ValueError as error:
        raise ValueError("project argument must be LABEL=/absolute/root") from error
    _validate_label(label)
    root = Path(root_value)
    if not root.is_absolute():
        raise ValueError("project root must be absolute")
    return label, root


def project_id_for_root(repo_root: Path) -> str:
    """Return the opaque, label-independent identity of one canonical root."""
    canonical_root = _require_exact_canonical_root(repo_root)
    return _project_id_from_canonical_root(canonical_root)


def build_project_specs(
    entries: Sequence[tuple[str, Path]],
    *,
    state_root: Path,
    git_executable: Path,
    probe_timeout_seconds: float = 10.0,
) -> tuple[ProjectSpec, ...]:
    """Validate explicit projects, then create their isolated state directories."""
    normalized_entries = tuple(entries)
    if not normalized_entries:
        return ()
    if len(normalized_entries) > MAX_PROJECTS:
        raise ValueError(f"too many projects; maximum is {MAX_PROJECTS}")
    if not isinstance(probe_timeout_seconds, (int, float)) or isinstance(
        probe_timeout_seconds, bool
    ):
        raise ValueError("probe_timeout_seconds must be a finite positive number")
    normalized_probe_timeout_seconds = float(probe_timeout_seconds)
    if (
        not math.isfinite(normalized_probe_timeout_seconds)
        or normalized_probe_timeout_seconds <= 0
    ):
        raise ValueError("probe_timeout_seconds must be a finite positive number")

    validated_git = _validate_git_executable(git_executable)
    validated_state_root = _validate_state_root(state_root)
    labels: set[str] = set()
    roots: set[Path] = set()
    prepared: list[tuple[str, Path, str, str]] = []
    for entry in normalized_entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError("project entries must be (label, root) pairs")
        label, root = entry
        _validate_label(label)
        folded_label = label.casefold()
        if folded_label in labels:
            raise ValueError("duplicate project label")
        labels.add(folded_label)
        canonical_root = _resolve_project_root(root)
        if canonical_root in roots:
            raise ValueError("duplicate canonical project root")
        roots.add(canonical_root)
        project_id = _project_id_from_canonical_root(canonical_root)
        top_level = _run_git_probe(
            validated_git,
            canonical_root,
            ("rev-parse", "--show-toplevel"),
            timeout_seconds=normalized_probe_timeout_seconds,
            label="Git top-level probe",
        )
        if top_level != str(canonical_root):
            raise ValueError("Git top-level does not match the canonical project root")
        branch = _run_git_probe(
            validated_git,
            canonical_root,
            ("symbolic-ref", "--quiet", "--short", "HEAD"),
            timeout_seconds=normalized_probe_timeout_seconds,
            label="Git branch probe",
        )
        if not branch:
            raise ValueError("Git branch probe returned an empty branch")
        prepared.append((project_id, canonical_root, label, branch))

    state_parent = validated_state_root / "projects"
    specs = tuple(
        ProjectSpec(
            project_id=project_id,
            label=label,
            repo_root=canonical_root,
            branch=branch,
            state_dir=state_parent / project_id,
        )
        for project_id, canonical_root, label, branch in sorted(prepared)
    )
    try:
        for spec in specs:
            spec.state_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError("could not create project state directory") from error
    return specs


def _contains_control_character(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _validate_label(label: object) -> None:
    if (
        not isinstance(label, str)
        or _contains_control_character(label)
        or _LABEL_PATTERN.fullmatch(label) is None
    ):
        raise ValueError("project label must match [A-Za-z][A-Za-z0-9_-]{0,31}")


def _resolve_project_root(root: object) -> Path:
    if not isinstance(root, Path):
        raise ValueError("project root must be a Path")
    if not root.is_absolute() or _contains_control_character(str(root)):
        raise ValueError("project root must be a safe absolute path")
    try:
        canonical_root = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("project root must exist") from error
    if not canonical_root.is_dir():
        raise ValueError("project root must be a directory")
    if not os.access(canonical_root, os.R_OK | os.X_OK):
        raise ValueError("project root must be readable")
    return canonical_root


def _require_exact_canonical_root(root: Path) -> Path:
    canonical_root = _resolve_project_root(root)
    if root != canonical_root:
        raise ValueError("project root must be an exact canonical path")
    return canonical_root


def _project_id_from_canonical_root(canonical_root: Path) -> str:
    return hashlib.sha256(os.fsencode(str(canonical_root))).hexdigest()[:32]


def _validate_state_root(state_root: object) -> Path:
    if not isinstance(state_root, Path):
        raise ValueError("state_root must be a Path")
    if not state_root.is_absolute() or _contains_control_character(str(state_root)):
        raise ValueError("state_root must be a safe absolute path")
    return state_root


def _validate_git_executable(git_executable: object) -> Path:
    if not isinstance(git_executable, Path):
        raise ValueError("Git executable must be an absolute regular executable")
    if not git_executable.is_absolute() or git_executable != Path(os.path.abspath(git_executable)):
        raise ValueError("Git executable must be an absolute regular executable")
    try:
        entry = git_executable.lstat()
    except OSError as error:
        raise ValueError("Git executable must be an absolute regular executable") from error
    if not stat.S_ISREG(entry.st_mode) or not os.access(git_executable, os.X_OK):
        raise ValueError("Git executable must be an absolute regular executable")
    return git_executable


def _run_git_probe(
    git_executable: Path,
    repo_root: Path,
    args: tuple[str, ...],
    *,
    timeout_seconds: float,
    label: str,
) -> str:
    command = (str(git_executable), "--no-pager", *args)
    try:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=_GIT_ENVIRONMENT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        raise ValueError(f"{label} could not start") from error
    if process.stdout is None:
        process.kill()
        process.wait()
        raise ValueError(f"{label} did not provide stdout")

    chunks: list[bytes] = []
    size = 0
    deadline = time.monotonic() + timeout_seconds
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                _terminate_process(process)
                raise ValueError(f"{label} exceeded subprocess deadline")
            block = os.read(process.stdout.fileno(), _MAX_GIT_OUTPUT_BYTES - size + 1)
            if not block:
                break
            if size + len(block) > _MAX_GIT_OUTPUT_BYTES:
                _terminate_process(process)
                raise ValueError(f"{label} exceeded output limit")
            chunks.append(block)
            size += len(block)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process(process)
            raise ValueError(f"{label} exceeded subprocess deadline")
        try:
            exit_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            _terminate_process(process)
            raise ValueError(f"{label} exceeded subprocess deadline") from error
    finally:
        selector.close()
        process.stdout.close()
    if exit_code != 0:
        raise ValueError(f"{label} failed")
    try:
        records = b"".join(chunks).decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} returned invalid output") from error
    if len(records) != 1 or not records[0]:
        raise ValueError(f"{label} must return exactly one record")
    return records[0]


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
