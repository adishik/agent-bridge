from __future__ import annotations

import subprocess
from contextlib import contextmanager
import os
from pathlib import Path
import stat

import pytest

from agent_bridge.contracts import TaskBrief
from agent_bridge import repository
from agent_bridge.repository import (
    ProtectedPathApprovalRequired,
    RepositoryTracker as _RepositoryTracker,
    validate_allowed_path,
)


GIT_EXECUTABLE = Path("/usr/bin/git")


def RepositoryTracker(repo_root: Path, artifact_directory: Path, **kwargs):
    return _RepositoryTracker(
        repo_root,
        artifact_directory,
        git_executable=GIT_EXECUTABLE,
        **kwargs,
    )


def _brief(*, allowed_paths: tuple[str, ...]) -> TaskBrief:
    return TaskBrief.from_dict({
        "task_id": "task-1",
        "revision": 1,
        "title": "Repository delta",
        "objective": "Measure only task-relative changes.",
        "context": [],
        "constraints": [],
        "allowed_paths": list(allowed_paths),
        "out_of_scope": [],
        "acceptance_criteria": ["The delta preserves the pre-task before-image."],
        "required_tests": [],
        "risks": [],
        "open_questions": [],
        "confidence": 1.0,
        "confidence_rationale": "Test fixture.",
    })


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    )
    return result.stdout


def _repository(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Bridge Test")
    for relative_path, contents in files.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    _git(repo, "add", *files)
    _git(repo, "commit", "-m", "fixture")
    return repo


def test_delta_separates_preexisting_changes_from_task_changes(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"clean.py": "CLEAN = 1\n", "dirty.py": "DIRTY = 1\n"})
    (repo / "dirty.py").write_text("DIRTY = 2\n")

    tracker = RepositoryTracker(repo, tmp_path / "artifacts", max_snapshot_bytes=1024)
    baseline = tracker.capture(_brief(allowed_paths=("dirty.py", "new.py")))
    (repo / "dirty.py").write_text("DIRTY = 3\n")
    (repo / "new.py").write_text("NEW = 1\n")
    delta = tracker.compare(baseline)

    assert delta.changed_paths == ("dirty.py", "new.py")
    assert delta.preexisting_unchanged_paths == ()
    assert "-DIRTY = 2" in delta.text_diffs["dirty.py"]
    assert "+DIRTY = 3" in delta.text_diffs["dirty.py"]


def test_outside_scope_edit_is_unexpected(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "ALLOWED = 1\n", "outside.py": "OUTSIDE = 1\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))

    (repo / "outside.py").write_text("OUTSIDE = 2\n")
    delta = tracker.compare(baseline)

    assert delta.changed_paths == ()
    assert delta.unexpected_paths == ("outside.py",)


@pytest.mark.parametrize(
    ("relative", "allowed_paths", "bucket"),
    (
        ("newdir", ("newdir",), "changed_paths"),
        ("outside-empty", ("allowed.py",), "unexpected_paths"),
        ("data/empty", ("allowed.py",), "protected_changed_paths"),
    ),
)
def test_empty_directory_additions_are_classified_by_scope(
    tmp_path: Path,
    relative: str,
    allowed_paths: tuple[str, ...],
    bucket: str,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=allowed_paths))
    target.mkdir()

    delta = tracker.compare(baseline)

    assert getattr(delta, bucket) == (relative,)


@pytest.mark.parametrize(
    ("relative", "allowed_paths", "bucket"),
    (
        ("allowed-empty", ("allowed-empty",), "changed_paths"),
        ("outside-empty", ("allowed.py",), "unexpected_paths"),
        ("data/empty", ("allowed.py",), "protected_changed_paths"),
    ),
)
def test_empty_directory_removals_are_classified_by_scope(
    tmp_path: Path,
    relative: str,
    allowed_paths: tuple[str, ...],
    bucket: str,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    (repo / relative).mkdir(parents=True)
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=allowed_paths))
    (repo / relative).rmdir()

    delta = tracker.compare(baseline)

    assert getattr(delta, bucket) == (relative,)


@pytest.mark.parametrize(
    ("relative", "allowed_paths", "bucket", "from_directory"),
    (
        ("allowed-entry", ("allowed-entry",), "changed_paths", False),
        ("outside-entry", ("allowed.py",), "unexpected_paths", True),
        ("data/entry", ("allowed.py",), "protected_changed_paths", False),
    ),
)
def test_file_directory_transitions_are_classified_by_scope(
    tmp_path: Path,
    relative: str,
    allowed_paths: tuple[str, ...],
    bucket: str,
    from_directory: bool,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if from_directory:
        target.mkdir()
    else:
        target.write_text("before\n", encoding="utf-8")
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=allowed_paths))
    if from_directory:
        target.rmdir()
        target.write_text("after\n", encoding="utf-8")
    else:
        target.unlink()
        target.mkdir()

    delta = tracker.compare(baseline)

    assert getattr(delta, bucket) == (relative,)


def test_protected_path_requires_approval_before_capture(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"data/frozen.txt": "frozen\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")

    with pytest.raises(ProtectedPathApprovalRequired, match="data/frozen.txt"):
        tracker.capture(_brief(allowed_paths=("data/frozen.txt",)))


@pytest.mark.parametrize("value", ("/absolute.py", "../escape.py", "nested/../../escape.py", ""))
def test_allowed_paths_must_be_repository_relative(value: str) -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        validate_allowed_path(value)


def test_large_preexisting_untracked_file_requires_separate_approval(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"tracked.py": "TRACKED = 1\n"})
    (repo / "large.py").write_text("x" * 17)
    tracker = RepositoryTracker(repo, tmp_path / "artifacts", max_snapshot_bytes=16)

    with pytest.raises(ProtectedPathApprovalRequired, match="large.py"):
        tracker.capture(_brief(allowed_paths=("large.py",)))


def test_compare_never_changes_file_contents(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    (repo / "allowed.py").write_text("after\n")
    before_compare = (repo / "allowed.py").read_bytes()

    tracker.compare(baseline)

    assert (repo / "allowed.py").read_bytes() == before_compare


def test_before_images_cannot_be_configured_inside_the_target_repository(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})

    with pytest.raises(ValueError, match="outside repo_root"):
        RepositoryTracker(repo, repo / ".artifacts")


@pytest.mark.parametrize("kind", ("relative", "symlink", "nonexec"))
def test_git_executable_must_be_explicit_absolute_regular_executable(
    tmp_path: Path, kind: str,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    candidate: Path | str
    if kind == "relative":
        candidate = "git"
    else:
        target = tmp_path / "git-target"
        target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        if kind == "nonexec":
            target.chmod(0o600)
            candidate = target
        else:
            target.chmod(0o700)
            link = tmp_path / "git-link"
            link.symlink_to(target)
            candidate = link

    with pytest.raises(ValueError, match="Git executable"):
        _RepositoryTracker(
            repo,
            tmp_path / "artifacts",
            git_executable=candidate,
        )


def test_in_scope_symlink_requires_separate_approval(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    external_secret = tmp_path / "external-secret.txt"
    external_secret.write_text("do not snapshot\n")
    (repo / "allowed.py").unlink()
    (repo / "allowed.py").symlink_to(external_secret)
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")

    with pytest.raises(ProtectedPathApprovalRequired, match="symlink"):
        tracker.capture(_brief(allowed_paths=("allowed.py",)))


def test_deletions_have_text_and_binary_delta_records(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"text.py": "before\n", "binary.bin": "placeholder\n"})
    (repo / "binary.bin").write_bytes(b"\x00\xff")
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("text.py", "binary.bin")))
    (repo / "text.py").unlink()
    (repo / "binary.bin").unlink()

    delta = tracker.compare(baseline)

    assert delta.changed_paths == ("binary.bin", "text.py")
    assert "-before" in delta.text_diffs["text.py"]
    assert delta.binary_changed_paths == ("binary.bin",)


def test_file_replaced_by_directory_is_a_safe_removal_delta(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    (repo / "allowed.py").unlink()
    (repo / "allowed.py").mkdir()

    delta = tracker.compare(baseline)

    assert delta.changed_paths == ("allowed.py",)
    assert "-before" in delta.text_diffs["allowed.py"]


def test_in_scope_parent_symlink_requires_separate_approval(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"safe/note.py": "before\n"})
    (repo / ".gitignore").write_text("safe\n")
    external_directory = tmp_path / "external"
    external_directory.mkdir()
    (external_directory / "note.py").write_text("do not snapshot\n")
    (repo / "safe" / "note.py").unlink()
    (repo / "safe").rmdir()
    (repo / "safe").symlink_to(external_directory, target_is_directory=True)
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")

    with pytest.raises(ProtectedPathApprovalRequired, match="unsafe"):
        tracker.capture(_brief(allowed_paths=("safe",)))


def test_new_text_and_binary_files_have_content_delta_records(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"existing.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("new.txt", "new.bin")))
    (repo / "new.txt").write_text("added\n")
    (repo / "new.bin").write_bytes(b"\x00\xff")

    delta = tracker.compare(baseline)

    assert delta.changed_paths == ("new.bin", "new.txt")
    assert "+added" in delta.text_diffs["new.txt"]
    assert delta.binary_changed_paths == ("new.bin",)


def test_tracked_path_missing_at_capture_is_preserved_in_baseline(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"tracked.py": "before\n"})
    (repo / "tracked.py").unlink()
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")

    baseline = tracker.capture(_brief(allowed_paths=("other.py",)))

    record = next(record for record in baseline.paths if record.path == "tracked.py")
    assert record.existed is False
    assert record.sha256 is None


def test_tracked_gitlink_is_preserved_in_baseline(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"tracked.py": "before\n"})
    nested_parent = tmp_path / "nested"
    nested_parent.mkdir()
    nested = _repository(nested_parent, {"child.py": "child\n"})
    gitlink_head = _git(nested, "rev-parse", "HEAD").strip()
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{gitlink_head},submodule")
    _git(repo, "commit", "-m", "gitlink")
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")

    baseline = tracker.capture(_brief(allowed_paths=("other.py",)))

    record = next(record for record in baseline.paths if record.path == "submodule")
    assert record.existed is False
    assert record.sha256 is None


def test_copy_size_limit_is_checked_on_the_copied_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "x\n"})
    original_copy = repository._copy_regular_file

    def grow_before_copy(
        repo_root: Path, source: Path, destination: Path, max_snapshot_bytes: int,
    ) -> tuple[int, str]:
        source.write_text("this is now too large\n")
        return original_copy(repo_root, source, destination, max_snapshot_bytes)

    monkeypatch.setattr(repository, "_copy_regular_file", grow_before_copy)
    tracker = RepositoryTracker(repo, tmp_path / "artifacts", max_snapshot_bytes=2)

    with pytest.raises(ProtectedPathApprovalRequired, match="allowed.py"):
        tracker.capture(_brief(allowed_paths=("allowed.py",)))


def test_tampered_before_image_is_rejected_before_diffing(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    record = next(record for record in baseline.paths if record.path == "allowed.py")
    assert record.before_image is not None
    record.before_image.write_text("tampered\n")
    (repo / "allowed.py").write_text("after\n")

    with pytest.raises(RuntimeError, match="baseline before-image integrity"):
        tracker.compare(baseline)


def test_persisted_baseline_round_trip_preserves_artifacts_and_rejects_wrong_identity(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    artifact_directory = tmp_path / "artifacts"
    tracker = RepositoryTracker(repo, artifact_directory)
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    manifest = tracker.baseline_manifest(baseline)

    restored = tracker.restore_baseline(
        manifest, expected_baseline_id=baseline.baseline_id
    )

    assert restored == baseline
    allowed = next(record for record in restored.paths if record.path == "allowed.py")
    assert allowed.before_image is not None
    assert allowed.before_image.is_relative_to(artifact_directory)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        tracker.restore_baseline(manifest, expected_baseline_id="wrong-baseline")

    wrong_root = dict(manifest)
    wrong_root["repo_root"] = str(tmp_path / "other-repo")
    with pytest.raises(RuntimeError, match="repository root mismatch"):
        tracker.restore_baseline(
            wrong_root, expected_baseline_id=baseline.baseline_id
        )


@pytest.mark.parametrize("tamper", ("missing", "symlink", "oversize", "same_size"))
def test_restore_eagerly_validates_every_before_image(
    tmp_path: Path, tamper: str,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    artifacts = tmp_path / "artifacts"
    tracker = RepositoryTracker(repo, artifacts)
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    manifest = tracker.baseline_manifest(baseline)
    record = next(record for record in baseline.paths if record.path == "allowed.py")
    assert record.before_image is not None
    if tamper == "missing":
        record.before_image.unlink()
    elif tamper == "symlink":
        record.before_image.unlink()
        record.before_image.symlink_to(tmp_path / "external.txt")
    elif tamper == "oversize":
        record.before_image.write_bytes(b"before\nextra")
    else:
        record.before_image.write_bytes(b"tamper\n")

    with pytest.raises(RuntimeError, match="before-image integrity"):
        tracker.restore_baseline(
            manifest, expected_baseline_id=baseline.baseline_id
        )


def test_compare_validates_before_images_even_when_working_file_is_unchanged(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    record = next(record for record in baseline.paths if record.path == "allowed.py")
    assert record.before_image is not None
    record.before_image.write_bytes(b"tamper\n")

    with pytest.raises(RuntimeError, match="before-image integrity"):
        tracker.compare(baseline)


def test_restore_enforces_aggregate_before_image_budget(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    artifacts = tmp_path / "artifacts"
    capturing = RepositoryTracker(repo, artifacts)
    baseline = capturing.capture(_brief(allowed_paths=("allowed.py",)))
    manifest = capturing.baseline_manifest(baseline)
    restoring = RepositoryTracker(
        repo, artifacts, max_inventory_content_bytes=4
    )

    with pytest.raises(RuntimeError, match="aggregate before-image"):
        restoring.restore_baseline(
            manifest, expected_baseline_id=baseline.baseline_id
        )


def test_restore_rejects_noncanonical_path_records(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    manifest = tracker.baseline_manifest(baseline)
    duplicate = dict(manifest)
    duplicate["paths"] = [*manifest["paths"], manifest["paths"][0]]  # type: ignore[index]

    with pytest.raises(RuntimeError, match="canonical"):
        tracker.restore_baseline(
            duplicate, expected_baseline_id=baseline.baseline_id
        )


@pytest.mark.parametrize(
    "mutation",
    (
        {"existed": False},
        {"sha256": None},
        {"kind": "directory"},
    ),
)
def test_restore_rejects_inconsistent_before_image_record_combinations(
    tmp_path: Path, mutation: dict[str, object],
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    manifest = tracker.baseline_manifest(baseline)
    invalid = dict(manifest)
    raw_records = list(manifest["paths"])  # type: ignore[arg-type]
    allowed_index = next(
        index
        for index, raw in enumerate(raw_records)
        if isinstance(raw, dict) and raw.get("path") == "allowed.py"
    )
    raw_records[allowed_index] = {**raw_records[allowed_index], **mutation}  # type: ignore[arg-type]
    invalid["paths"] = raw_records

    with pytest.raises(RuntimeError, match="combination"):
        tracker.restore_baseline(
            invalid, expected_baseline_id=baseline.baseline_id
        )


def test_restore_rejects_missing_before_image_reference_for_allowed_regular_file(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    manifest = tracker.baseline_manifest(baseline)
    invalid = dict(manifest)
    invalid["paths"] = [
        {**raw, "before_image": None} if raw["path"] == "allowed.py" else raw
        for raw in manifest["paths"]  # type: ignore[union-attr]
    ]

    with pytest.raises(RuntimeError, match="before-image.*required"):
        tracker.restore_baseline(
            invalid, expected_baseline_id=baseline.baseline_id
        )


def test_widen_baseline_rejects_newly_allowed_symlink(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    external = tmp_path / "external.txt"
    external.write_text("external\n", encoding="utf-8")
    (repo / "linked.py").symlink_to(external)
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))

    with pytest.raises(ProtectedPathApprovalRequired, match="symlink"):
        tracker.widen_baseline(
            baseline, _brief(allowed_paths=("allowed.py", "linked.py"))
        )


def test_widen_baseline_rejects_newly_allowed_oversized_file(tmp_path: Path) -> None:
    repo = _repository(
        tmp_path, {"allowed.py": "before\n", "large.py": "x" * 17}
    )
    tracker = RepositoryTracker(
        repo, tmp_path / "artifacts", max_snapshot_bytes=16
    )
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))

    with pytest.raises(ProtectedPathApprovalRequired, match="large.py"):
        tracker.widen_baseline(
            baseline, _brief(allowed_paths=("allowed.py", "large.py"))
        )


def test_failed_multi_path_widening_cleans_artifacts_and_can_retry(
    tmp_path: Path,
) -> None:
    repo = _repository(
        tmp_path,
        {
            "allowed.py": "before\n",
            "nested/first.py": "first before\n",
        },
    )
    external = tmp_path / "external.txt"
    external.write_text("external\n", encoding="utf-8")
    (repo / "second.py").symlink_to(external)
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    original = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    widened_brief = _brief(
        allowed_paths=("allowed.py", "nested/first.py", "second.py")
    )

    with pytest.raises(ProtectedPathApprovalRequired, match="second.py"):
        tracker.widen_baseline(original, widened_brief)

    before_root = (
        tmp_path / "artifacts" / original.baseline_id / "before"
    )
    assert not (before_root / "nested").exists()
    assert (before_root / "allowed.py").exists()

    retried = tracker.widen_baseline(
        original, _brief(allowed_paths=("allowed.py", "nested/first.py"))
    )

    first = next(
        record for record in retried.paths if record.path == "nested/first.py"
    )
    assert first.before_image is not None
    assert first.before_image.read_text(encoding="utf-8") == "first before\n"


@pytest.mark.parametrize("failure", ("unsafe", "io"))
def test_widening_registers_artifact_cleanup_before_final_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    repo = _repository(
        tmp_path, {"allowed.py": "before\n", "candidate.py": "candidate\n"}
    )
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    original = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    original_copy = repository._copy_regular_file
    original_stat = repository._entry_stat
    copied = False

    def recording_copy(*args: object, **kwargs: object):
        nonlocal copied
        result = original_copy(*args, **kwargs)  # type: ignore[arg-type]
        copied = True
        return result

    def failing_final_stat(repo_root: Path, path: Path):
        if copied and path.name == "candidate.py":
            if failure == "unsafe":
                raise repository._UnsafeRepositoryPath("injected final stat failure")
            raise OSError("injected final stat I/O failure")
        return original_stat(repo_root, path)

    monkeypatch.setattr(repository, "_copy_regular_file", recording_copy)
    monkeypatch.setattr(repository, "_entry_stat", failing_final_stat)
    with pytest.raises((repository._UnsafeRepositoryPath, OSError)):
        tracker.widen_baseline(
            original,
            _brief(allowed_paths=("allowed.py", "candidate.py")),
        )

    expected = (
        tmp_path
        / "artifacts"
        / original.baseline_id
        / "before"
        / "candidate.py"
    )
    assert not expected.exists()
    monkeypatch.setattr(repository, "_copy_regular_file", original_copy)
    monkeypatch.setattr(repository, "_entry_stat", original_stat)
    retried = tracker.widen_baseline(
        original, _brief(allowed_paths=("allowed.py", "candidate.py"))
    )
    assert next(
        record for record in retried.paths if record.path == "candidate.py"
    ).before_image == expected


def test_discard_widening_removes_tampered_new_artifact(tmp_path: Path) -> None:
    repo = _repository(
        tmp_path, {"allowed.py": "before\n", "candidate.py": "candidate\n"}
    )
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    original = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    widened = tracker.widen_baseline(
        original, _brief(allowed_paths=("allowed.py", "candidate.py"))
    )
    candidate = next(
        record for record in widened.paths if record.path == "candidate.py"
    )
    assert candidate.before_image is not None
    candidate.before_image.write_text("tampered\n", encoding="utf-8")

    tracker.discard_widening(original, widened)

    assert not candidate.before_image.exists()
    allowed = next(record for record in original.paths if record.path == "allowed.py")
    assert allowed.before_image is not None and allowed.before_image.exists()


def test_discard_widening_prunes_an_artifact_free_baseline_tree(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path, {"candidate.py": "candidate\n"})
    artifacts = tmp_path / "artifacts"
    tracker = RepositoryTracker(repo, artifacts)
    original = tracker.capture(_brief(allowed_paths=("future.py",)))
    widened = tracker.widen_baseline(
        original, _brief(allowed_paths=("future.py", "candidate.py"))
    )

    tracker.discard_widening(original, widened)

    assert list(artifacts.iterdir()) == []


def test_widening_aggregate_budget_includes_existing_before_images_and_can_retry(
    tmp_path: Path,
) -> None:
    repo = _repository(
        tmp_path,
        {
            "allowed.py": "a" * 40_000,
            "too-large-together.py": "b" * 30_000,
            "small.py": "c" * 10_000,
        },
    )
    tracker = RepositoryTracker(
        repo,
        tmp_path / "artifacts",
        max_snapshot_bytes=50_000,
        max_inventory_content_bytes=64 * 1024,
    )
    original = tracker.capture(_brief(allowed_paths=("allowed.py",)))

    with pytest.raises(ProtectedPathApprovalRequired, match="content budget"):
        tracker.widen_baseline(
            original,
            _brief(allowed_paths=("allowed.py", "too-large-together.py")),
        )

    retried = tracker.widen_baseline(
        original, _brief(allowed_paths=("allowed.py", "small.py"))
    )
    small = next(record for record in retried.paths if record.path == "small.py")
    assert small.before_image is not None
    assert small.before_image.read_bytes() == b"c" * 10_000


def test_capture_rejects_an_explicitly_allowed_ignored_symlink(tmp_path: Path) -> None:
    repo = _repository(
        tmp_path, {".gitignore": "ignored-link\n", "allowed.py": "before\n"}
    )
    external = tmp_path / "external.txt"
    external.write_text("external\n", encoding="utf-8")
    (repo / "ignored-link").symlink_to(external)
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")

    with pytest.raises(ProtectedPathApprovalRequired, match="symlink"):
        tracker.capture(_brief(allowed_paths=("ignored-link",)))


def test_widen_baseline_rejects_a_newly_allowed_ignored_symlink(
    tmp_path: Path,
) -> None:
    repo = _repository(
        tmp_path, {".gitignore": "ignored-link\n", "allowed.py": "before\n"}
    )
    external = tmp_path / "external.txt"
    external.write_text("external\n", encoding="utf-8")
    (repo / "ignored-link").symlink_to(external)
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))

    with pytest.raises(ProtectedPathApprovalRequired, match="symlink"):
        tracker.widen_baseline(
            baseline, _brief(allowed_paths=("allowed.py", "ignored-link"))
        )


def test_capture_rejects_ignored_symlink_descendant_of_allowed_directory(
    tmp_path: Path,
) -> None:
    repo = _repository(
        tmp_path,
        {".gitignore": "allowed/ignored-link\n", "allowed/tracked.py": "before\n"},
    )
    external = tmp_path / "external.txt"
    external.write_text("external\n", encoding="utf-8")
    (repo / "allowed" / "ignored-link").symlink_to(external)
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")

    with pytest.raises(ProtectedPathApprovalRequired, match="symlink"):
        tracker.capture(_brief(allowed_paths=("allowed",)))


def test_compare_tracks_ignored_regular_file_changes_inside_and_outside_scope(
    tmp_path: Path,
) -> None:
    repo = _repository(
        tmp_path,
        {
            ".gitignore": "allowed/ignored.txt\noutside-ignored.txt\n",
            "allowed/tracked.py": "before\n",
        },
    )
    (repo / "allowed" / "ignored.txt").write_text("before\n", encoding="utf-8")
    (repo / "outside-ignored.txt").write_text("outside before\n", encoding="utf-8")
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed",)))

    (repo / "allowed" / "ignored.txt").write_text("after\n", encoding="utf-8")
    (repo / "outside-ignored.txt").write_text("outside after\n", encoding="utf-8")
    delta = tracker.compare(baseline)

    assert delta.changed_paths == ("allowed/ignored.txt",)
    assert delta.unexpected_paths == ("outside-ignored.txt",)


@pytest.mark.parametrize("relative", ("outside-ignored.txt", "data/ignored.txt"))
@pytest.mark.parametrize("mutation", ("content", "deletion", "symlink"))
def test_compare_detects_preexisting_ignored_outside_and_protected_mutations(
    tmp_path: Path, relative: str, mutation: str,
) -> None:
    repo = _repository(
        tmp_path,
        {
            ".gitignore": "outside-ignored.txt\ndata/\n",
            "allowed.py": "before\n",
        },
    )
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("before\n", encoding="utf-8")
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))

    if mutation == "content":
        target.write_text("after\n", encoding="utf-8")
    elif mutation == "deletion":
        target.unlink()
    else:
        external = tmp_path / "external-secret.txt"
        external.write_text("must not be read\n", encoding="utf-8")
        target.unlink()
        target.symlink_to(external)

    delta = tracker.compare(baseline)

    if relative.startswith("data/") or mutation == "symlink":
        assert delta.protected_changed_paths == (relative,)
        assert delta.unexpected_paths == ()
    else:
        assert delta.unexpected_paths == (relative,)
        assert delta.protected_changed_paths == ()


@pytest.mark.parametrize("relative", ("outside-new.txt", "data/new.txt"))
def test_compare_detects_ignored_outside_and_protected_additions(
    tmp_path: Path, relative: str,
) -> None:
    repo = _repository(
        tmp_path,
        {
            ".gitignore": "outside-new.txt\ndata/\n",
            "allowed.py": "before\n",
        },
    )
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("new\n", encoding="utf-8")

    delta = tracker.compare(baseline)

    if relative.startswith("data/"):
        assert delta.protected_changed_paths == (relative,)
    else:
        assert delta.unexpected_paths == (relative,)


@pytest.mark.parametrize("relative", ("outside-fifo", "data/newfifo"))
def test_compare_detects_ignored_special_additions_without_opening_them(
    tmp_path: Path, relative: str,
) -> None:
    repo = _repository(
        tmp_path,
        {
            ".gitignore": "outside-fifo\ndata/\n",
            "allowed.py": "before\n",
        },
    )
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(target)

    delta = tracker.compare(baseline)

    if relative.startswith("data/"):
        assert delta.protected_changed_paths == (relative,)
    else:
        assert delta.unexpected_paths == (relative,)


@pytest.mark.parametrize("relative", ("ordinary-fifo", "data/ordinary-fifo"))
def test_compare_detects_non_ignored_special_additions_without_opening_them(
    tmp_path: Path, relative: str,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(target)

    delta = tracker.compare(baseline)

    if relative.startswith("data/"):
        assert delta.protected_changed_paths == (relative,)
    else:
        assert delta.unexpected_paths == (relative,)


def test_compare_detects_git_config_mutation_as_protected(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))

    _git(repo, "config", "bridge.mutated", "true")
    delta = tracker.compare(baseline)

    assert ".git/config" in delta.protected_changed_paths


def test_compare_detects_branch_identity_change_as_protected(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))

    _git(repo, "checkout", "-b", "changed-branch")
    delta = tracker.compare(baseline)

    assert ".git/HEAD" in delta.protected_changed_paths


def test_git_status_stat_cache_refresh_is_not_a_protected_change(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    (repo / "allowed.py").write_text("after\n", encoding="utf-8")

    _git(repo, "status", "--short")
    delta = tracker.compare(baseline)

    assert delta.changed_paths == ("allowed.py",)
    assert delta.protected_changed_paths == ()


@pytest.mark.parametrize("flag", ("--assume-unchanged", "--skip-worktree"))
def test_semantic_index_flags_are_protected_changes(
    tmp_path: Path, flag: str,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))

    _git(repo, "update-index", flag, "allowed.py")
    delta = tracker.compare(baseline)

    assert ".git/index" in delta.protected_changed_paths


@pytest.mark.parametrize(
    "mutation", ("assume-unchanged", "skip-worktree", "staged")
)
def test_transient_baseline_index_bytes_cannot_hide_final_semantic_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    repo = _repository(
        tmp_path, {"allowed.py": "before\n", "second.py": "before\n"}
    )
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    index = baseline.git_dir / "index"
    baseline_bytes = index.read_bytes()
    if mutation == "assume-unchanged":
        _git(repo, "update-index", "--assume-unchanged", "allowed.py")
    elif mutation == "skip-worktree":
        _git(repo, "update-index", "--skip-worktree", "allowed.py")
    else:
        (repo / "second.py").write_text("staged\n", encoding="utf-8")
        _git(repo, "add", "second.py")
    malicious_bytes = index.read_bytes()
    assert malicious_bytes != baseline_bytes
    original_run = tracker._run_git_bytes

    def supply_baseline_only_during_semantic_command(
        label: str, *args: object, **kwargs: object,
    ):
        if label != "Git semantic index":
            return original_run(label, *args, **kwargs)  # type: ignore[arg-type]
        index.write_bytes(baseline_bytes)
        try:
            return original_run(label, *args, **kwargs)  # type: ignore[arg-type]
        finally:
            index.write_bytes(malicious_bytes)

    monkeypatch.setattr(
        tracker, "_run_git_bytes", supply_baseline_only_during_semantic_command
    )
    delta = tracker.compare(baseline)

    assert ".git/index" in delta.protected_changed_paths
    assert index.read_bytes() == malicious_bytes


def test_split_index_is_rejected_before_semantic_git(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    _git(repo, "update-index", "--split-index")
    assert tuple((repo / ".git").glob("sharedindex.*"))
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")

    with pytest.raises(ProtectedPathApprovalRequired, match="split|shared"):
        tracker.capture(_brief(allowed_paths=("allowed.py",)))


def test_compare_rejects_symlinked_external_index_before_spawning_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    index = baseline.git_dir / "index"
    external = tmp_path / "external-index"
    external.write_bytes(index.read_bytes())
    index.unlink()
    index.symlink_to(external)

    def forbidden_git(*args: object, **kwargs: object) -> object:
        raise AssertionError("Git must not run after its index changes type")

    monkeypatch.setattr(repository.subprocess, "Popen", forbidden_git)
    with pytest.raises(ProtectedPathApprovalRequired, match="index"):
        tracker.compare(baseline)


def test_staging_is_detected_by_semantic_index_identity(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    (repo / "allowed.py").write_text("after\n", encoding="utf-8")

    _git(repo, "add", "allowed.py")
    delta = tracker.compare(baseline)

    assert ".git/index" in delta.protected_changed_paths


def test_compare_rejects_changed_git_marker_before_spawning_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _repository(tmp_path, {"allowed.py": "before\n"})
    linked = tmp_path / "linked"
    _git(primary, "worktree", "add", "-b", "linked-guard", str(linked))
    tracker = RepositoryTracker(linked, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    marker = linked / ".git"
    marker.write_text(marker.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    def forbidden_git(*args: object, **kwargs: object) -> object:
        raise AssertionError("Git must not run after its marker changes")

    monkeypatch.setattr(repository.subprocess, "Popen", forbidden_git)
    with pytest.raises(ProtectedPathApprovalRequired, match="Git root"):
        tracker.compare(baseline)


def test_compare_short_circuits_changed_control_before_any_git_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    (repo / ".git" / "config").write_text(
        (repo / ".git" / "config").read_text(encoding="utf-8")
        + "\n[bridge]\n\tchanged = true\n",
        encoding="utf-8",
    )

    def forbidden_git(*args: object, **kwargs: object) -> object:
        raise AssertionError("Git must not run after a control snapshot changes")

    monkeypatch.setattr(repository.subprocess, "Popen", forbidden_git)
    delta = tracker.compare(baseline)

    assert ".git/config" in delta.protected_changed_paths


@pytest.mark.parametrize("linked", (False, True))
def test_git_commands_ignore_hostile_environment_and_external_fsmonitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked: bool,
) -> None:
    primary = _repository(tmp_path, {"allowed.py": "before\n"})
    repo = primary
    if linked:
        repo = tmp_path / "linked-sanitized"
        _git(primary, "worktree", "add", "-b", "linked-sanitized", str(repo))
    external_parent = tmp_path / "external"
    external_parent.mkdir()
    external_repo = _repository(external_parent, {"other.py": "outside\n"})
    sentinel = tmp_path / "fsmonitor-ran"
    fsmonitor = tmp_path / "fsmonitor.sh"
    fsmonitor.write_text(f"#!/bin/sh\ntouch '{sentinel}'\nexit 1\n", encoding="utf-8")
    fsmonitor.chmod(0o700)
    hostile_global = tmp_path / "hostile.gitconfig"
    hostile_global.write_text(
        f"[core]\n\tfsmonitor = {fsmonitor}\n", encoding="utf-8"
    )
    hostile_values = {
        "GIT_DIR": str(external_repo / ".git"),
        "GIT_WORK_TREE": str(external_repo),
        "GIT_INDEX_FILE": str(external_repo / ".git" / "index"),
        "GIT_OBJECT_DIRECTORY": str(external_repo / ".git" / "objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(external_repo / ".git" / "objects"),
        "GIT_COMMON_DIR": str(external_repo / ".git"),
        "GIT_CEILING_DIRECTORIES": str(tmp_path),
        "GIT_CONFIG_GLOBAL": str(hostile_global),
        "GIT_CONFIG_SYSTEM": str(hostile_global),
        "GIT_CONFIG_NOSYSTEM": "0",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.fsmonitor",
        "GIT_CONFIG_VALUE_0": str(fsmonitor),
    }
    for key, value in hostile_values.items():
        monkeypatch.setenv(key, value)
    original_popen = repository.subprocess.Popen
    observed_environments: list[dict[str, str]] = []
    snapshot_modes: list[int] = []

    def recording_popen(*args: object, **kwargs: object):
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        observed_environments.append(dict(environment))
        index_file = Path(environment["GIT_INDEX_FILE"])
        if index_file.as_posix().startswith("/proc/self/fd/"):
            descriptor = int(index_file.name)
            assert descriptor in kwargs.get("pass_fds", ())
            snapshot_modes.append(stat.S_IMODE(index_file.stat().st_mode))
        return original_popen(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(repository.subprocess, "Popen", recording_popen)

    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    delta = tracker.compare(baseline)

    assert delta.changed_paths == ()
    assert delta.unexpected_paths == ()
    assert delta.protected_changed_paths == ()
    assert not sentinel.exists()
    assert observed_environments
    assert all(str(fsmonitor) not in environment.values() for environment in observed_environments)
    anchored = [environment for environment in observed_environments if "GIT_DIR" in environment]
    assert anchored
    assert all(environment["GIT_WORK_TREE"] == str(repo) for environment in anchored)
    observed_indexes = {
        Path(environment["GIT_INDEX_FILE"]) for environment in anchored
    }
    assert baseline.git_dir / "index" in observed_indexes
    snapshot_indexes = {
        path for path in observed_indexes if path != baseline.git_dir / "index"
    }
    assert snapshot_indexes
    assert all(
        path.as_posix().startswith("/proc/self/fd/") for path in snapshot_indexes
    )
    assert snapshot_modes and set(snapshot_modes) == {0o400}
    assert all(environment["GIT_COMMON_DIR"] == str(baseline.common_dir) for environment in anchored)
    assert all(environment["GIT_OBJECT_DIRECTORY"] == str(baseline.common_dir / "objects") for environment in anchored)
    assert all(environment["GIT_CONFIG_GLOBAL"] == os.devnull for environment in observed_environments)
    assert all(environment["GIT_CONFIG_SYSTEM"] == os.devnull for environment in observed_environments)
    expected_base = {
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_TERMINAL_PROMPT",
        "GIT_OPTIONAL_LOCKS",
        "GIT_LFS_SKIP_SMUDGE",
        "GCM_INTERACTIVE",
        "LC_ALL",
        "LANG",
    }
    expected_anchored = expected_base | {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    }
    assert all(set(environment) == expected_anchored for environment in anchored)


def test_exact_git_executable_ignores_fake_path_and_still_blocks_protected_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    sentinel = tmp_path / "fake-git-ran"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"#!/bin/sh\ntouch '{sentinel}'\nexec /usr/bin/git \"$@\"\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o700)
    monkeypatch.setenv("PATH", str(fake_bin))
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    protected = repo / "data" / "secret"
    protected.parent.mkdir()
    protected.write_text("secret\n", encoding="utf-8")

    delta = tracker.compare(baseline)

    assert "data/secret" in delta.protected_changed_paths
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "section",
    (
        '[include]\n\tpath = /tmp/external-task7-config\n',
        '[includeIf "gitdir:/tmp/"]\n\tpath = /tmp/external-task7-config\n',
    ),
)
def test_external_git_config_includes_are_rejected_before_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, section: str,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    with (repo / ".git" / "config").open("a", encoding="utf-8") as config:
        config.write(section)

    def forbidden_git(*args: object, **kwargs: object) -> object:
        raise AssertionError("Git must not run with an external config include")

    monkeypatch.setattr(repository.subprocess, "Popen", forbidden_git)
    with pytest.raises(ProtectedPathApprovalRequired, match="configuration"):
        RepositoryTracker(repo, tmp_path / "artifacts").capture(
            _brief(allowed_paths=("allowed.py",))
        )


def test_linked_worktree_config_is_inspected_before_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _repository(tmp_path, {"allowed.py": "before\n"})
    linked = tmp_path / "linked-config"
    _git(primary, "worktree", "add", "-b", "linked-config", str(linked))
    _git(primary, "config", "extensions.worktreeConfig", "true")
    marker = (linked / ".git").read_text(encoding="utf-8").strip()
    git_dir = Path(marker.removeprefix("gitdir: "))
    (git_dir / "config.worktree").write_text(
        '[includeIf "gitdir:/tmp/"]\n\tpath = /tmp/external-task7-config\n',
        encoding="utf-8",
    )

    def forbidden_git(*args: object, **kwargs: object) -> object:
        raise AssertionError("Git must not run with an unsafe worktree config")

    monkeypatch.setattr(repository.subprocess, "Popen", forbidden_git)
    with pytest.raises(ProtectedPathApprovalRequired, match="configuration"):
        RepositoryTracker(linked, tmp_path / "artifacts").capture(
            _brief(allowed_paths=("allowed.py",))
        )


@pytest.mark.parametrize("key", ("fsmonitor", "hooksPath", "worktree"))
def test_redirecting_local_core_paths_are_rejected_before_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    with (repo / ".git" / "config").open("a", encoding="utf-8") as config:
        config.write(f"\n[core]\n\t{key} = /tmp/external-task7-path\n")

    def forbidden_git(*args: object, **kwargs: object) -> object:
        raise AssertionError("Git must not run with a redirecting local config")

    monkeypatch.setattr(repository.subprocess, "Popen", forbidden_git)
    with pytest.raises(ProtectedPathApprovalRequired, match="redirect"):
        RepositoryTracker(repo, tmp_path / "artifacts").capture(
            _brief(allowed_paths=("allowed.py",))
        )


@pytest.mark.parametrize("external_kind", ("fifo", "symlink"))
def test_external_core_excludes_file_is_rejected_without_access(
    tmp_path: Path,
    external_kind: str,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    external = tmp_path / "external-excludes"
    if external_kind == "fifo":
        os.mkfifo(external)
    else:
        target = tmp_path / "external-target"
        target.write_text("data/secret\n", encoding="utf-8")
        external.symlink_to(target)
    with (repo / ".git" / "config").open("a", encoding="utf-8") as config:
        config.write(f"\n[core]\n\texcludesFile = {external}\n")

    with pytest.raises(ProtectedPathApprovalRequired, match="configuration"):
        RepositoryTracker(repo, tmp_path / "artifacts").capture(
            _brief(allowed_paths=("allowed.py",))
        )


def test_case_colliding_protected_paths_are_both_visible_with_ignorecase_config(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n", "data/SAFE": "upper\n"})
    _git(repo, "config", "core.ignoreCase", "true")
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    (repo / "data" / "safe").write_text("lower\n", encoding="utf-8")

    delta = tracker.compare(baseline)

    assert "data/safe" in delta.protected_changed_paths


def test_config_race_to_fifo_is_detected_before_next_git_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    original = tracker._anchored_git_bytes
    raced = False

    def race_after_first_command(*args: object, **kwargs: object):
        nonlocal raced
        output = original(*args, **kwargs)  # type: ignore[arg-type]
        if not raced:
            raced = True
            config = repo / ".git" / "config"
            config.unlink()
            os.mkfifo(config)
        return output

    monkeypatch.setattr(tracker, "_anchored_git_bytes", race_after_first_command)
    with pytest.raises(ProtectedPathApprovalRequired, match="control|configuration"):
        tracker.compare(baseline)


def test_git_subprocess_deadline_is_bounded(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    slow_git = tmp_path / "slow-git"
    slow_git.write_text("#!/bin/sh\nexec /bin/sleep 5\n", encoding="utf-8")
    slow_git.chmod(0o700)
    artifacts = tmp_path / "artifacts"
    tracker = _RepositoryTracker(
        repo,
        artifacts,
        git_executable=slow_git,
        git_timeout_seconds=0.05,
    )

    with pytest.raises(ProtectedPathApprovalRequired, match="deadline"):
        tracker.capture(_brief(allowed_paths=("allowed.py",)))
    assert list(artifacts.iterdir()) == []


def test_config_race_to_external_fifo_is_stopped_by_git_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    external_fifo = tmp_path / "external-config"
    os.mkfifo(external_fifo)
    original_popen = repository.subprocess.Popen
    raced = False

    def race_before_launch(*args: object, **kwargs: object):
        nonlocal raced
        if not raced:
            raced = True
            (repo / ".git" / "config").write_text(
                f"[include]\n\tpath = {external_fifo}\n", encoding="utf-8"
            )
        return original_popen(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(repository.subprocess, "Popen", race_before_launch)
    tracker = _RepositoryTracker(
        repo,
        tmp_path / "artifacts",
        git_executable=GIT_EXECUTABLE,
        git_timeout_seconds=0.1,
    )

    with pytest.raises(ProtectedPathApprovalRequired, match="deadline"):
        tracker.capture(_brief(allowed_paths=("allowed.py",)))


def test_capture_and_compare_allow_explicit_nested_missing_path(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    relative = "nested/not-yet/deep.py"
    baseline = tracker.capture(_brief(allowed_paths=(relative,)))
    target = repo / relative
    target.parent.mkdir(parents=True)
    target.write_text("created\n", encoding="utf-8")

    delta = tracker.compare(baseline)

    assert delta.changed_paths == (relative,)
    assert delta.unexpected_paths == ()


@pytest.mark.parametrize("linked", (False, True))
def test_compare_rejects_replaced_exact_git_dir_before_spawning_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked: bool,
) -> None:
    primary = _repository(tmp_path, {"allowed.py": "before\n"})
    repo = primary
    if linked:
        repo = tmp_path / "linked"
        _git(primary, "worktree", "add", "-b", "linked-root-guard", str(repo))
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    displaced = baseline.git_dir.with_name(baseline.git_dir.name + "-displaced")
    baseline.git_dir.rename(displaced)
    baseline.git_dir.mkdir()

    def forbidden_git(*args: object, **kwargs: object) -> object:
        raise AssertionError("Git must not run after its exact root changes")

    monkeypatch.setattr(repository.subprocess, "Popen", forbidden_git)
    with pytest.raises(ProtectedPathApprovalRequired, match="Git root"):
        tracker.compare(baseline)


def test_compare_detects_common_git_config_mutation_from_linked_worktree(
    tmp_path: Path,
) -> None:
    primary = _repository(tmp_path, {"allowed.py": "before\n"})
    linked = tmp_path / "linked"
    _git(primary, "worktree", "add", "-b", "linked-branch", str(linked))
    tracker = RepositoryTracker(linked, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))

    _git(linked, "config", "bridge.mutated", "true")
    delta = tracker.compare(baseline)

    assert ".git/config" in delta.protected_changed_paths


@pytest.mark.parametrize(
    ("mutation", "expected_path"),
    (
        ("hook", ".git/hooks/pre-commit"),
        ("non_current_ref", ".git/refs/heads/other"),
        ("exclude", ".git/info/exclude"),
    ),
)
def test_compare_detects_bounded_git_control_tree_mutations(
    tmp_path: Path, mutation: str, expected_path: str,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    if mutation == "non_current_ref":
        (repo / "allowed.py").write_text("second\n", encoding="utf-8")
        _git(repo, "add", "allowed.py")
        _git(repo, "commit", "-m", "second")
        _git(repo, "branch", "other", "HEAD~1")
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))

    if mutation == "hook":
        (repo / ".git" / "hooks" / "pre-commit").write_text(
            "#!/bin/sh\nexit 0\n", encoding="utf-8"
        )
    elif mutation == "non_current_ref":
        _git(repo, "update-ref", "refs/heads/other", "HEAD")
    else:
        with (repo / ".git" / "info" / "exclude").open(
            "a", encoding="utf-8"
        ) as excluded:
            excluded.write("hidden-by-sol\n")

    delta = tracker.compare(baseline)

    assert expected_path in delta.protected_changed_paths


@pytest.mark.parametrize("mutation", ("content", "deletion", "symlink"))
def test_ignored_brainstorm_entries_are_protected_not_invisible(
    tmp_path: Path, mutation: str,
) -> None:
    relative = ".superpowers/brainstorm/note.txt"
    repo = _repository(
        tmp_path,
        {
            ".gitignore": ".superpowers/brainstorm/\n",
            "allowed.py": "before\n",
        },
    )
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("before\n", encoding="utf-8")
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))

    if mutation == "content":
        target.write_text("after\n", encoding="utf-8")
    elif mutation == "deletion":
        target.unlink()
    else:
        external = tmp_path / "external-secret.txt"
        external.write_text("must not be read\n", encoding="utf-8")
        target.unlink()
        target.symlink_to(external)

    delta = tracker.compare(baseline)

    assert delta.protected_changed_paths == (relative,)


def test_ignored_brainstorm_addition_is_protected(tmp_path: Path) -> None:
    relative = ".superpowers/brainstorm/new.txt"
    repo = _repository(
        tmp_path,
        {
            ".gitignore": ".superpowers/brainstorm/\n",
            "allowed.py": "before\n",
        },
    )
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("new\n", encoding="utf-8")

    delta = tracker.compare(baseline)

    assert delta.protected_changed_paths == (relative,)


@pytest.mark.parametrize("ignore_source", ("root", "nested", "info"))
def test_transient_ignore_rules_cannot_hide_protected_addition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ignore_source: str,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    secret = repo / "data" / "secret"
    secret.parent.mkdir()
    secret.write_text("secret\n", encoding="utf-8")
    if ignore_source == "root":
        ignore_path = repo / ".gitignore"
        ignored_text = "data/secret\n"
    elif ignore_source == "nested":
        ignore_path = repo / "data" / ".gitignore"
        ignored_text = "secret\n"
    else:
        ignore_path = repo / ".git" / "info" / "exclude"
        ignored_text = "data/secret\n"
    original_contents = (
        ignore_path.read_bytes() if ignore_path.exists() else None
    )
    original_inventory = tracker._filesystem_inventory
    raced = False

    def inventory_with_transient_ignore() -> frozenset[str]:
        nonlocal raced
        raced = True
        ignore_path.parent.mkdir(parents=True, exist_ok=True)
        ignore_path.write_text(ignored_text, encoding="utf-8")
        try:
            return original_inventory()
        finally:
            if original_contents is None:
                ignore_path.unlink()
            else:
                ignore_path.write_bytes(original_contents)

    monkeypatch.setattr(
        tracker, "_filesystem_inventory", inventory_with_transient_ignore
    )
    delta = tracker.compare(baseline)

    assert raced
    assert "data/secret" in delta.protected_changed_paths


@pytest.mark.parametrize("later_failure", ("oversized", "tampered"))
def test_initial_capture_failure_removes_every_created_baseline_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    later_failure: str,
) -> None:
    repo = _repository(
        tmp_path,
        {"a-small.py": "ok\n", "z-later.py": "too large\n"},
    )
    artifacts = tmp_path / "artifacts"
    tracker = RepositoryTracker(
        repo,
        artifacts,
        max_snapshot_bytes=4 if later_failure == "oversized" else 1024,
    )
    if later_failure == "tampered":
        original_copy = repository._copy_regular_file

        def tamper_later(*args: object, **kwargs: object):
            result = original_copy(*args, **kwargs)  # type: ignore[arg-type]
            source_path = args[1]
            if isinstance(source_path, Path) and source_path.name == "z-later.py":
                raise RuntimeError("injected post-copy validation failure")
            return result

        monkeypatch.setattr(repository, "_copy_regular_file", tamper_later)

    with pytest.raises(
        (ProtectedPathApprovalRequired, RuntimeError), match="budget|validation"
    ):
        tracker.capture(
            _brief(allowed_paths=("a-small.py", "z-later.py"))
        )

    assert list(artifacts.iterdir()) == []


@pytest.mark.parametrize(
    ("tracker_kwargs", "files"),
    (
        ({"max_ignored_entries": 1}, {"first.bin": b"1", "second.bin": b"2"}),
        ({"max_ignored_bytes": 4}, {"large.bin": b"12345"}),
    ),
)
def test_ignored_inventory_fails_closed_at_configured_traversal_and_read_bounds(
    tmp_path: Path, tracker_kwargs: dict[str, int], files: dict[str, bytes],
) -> None:
    repo = _repository(
        tmp_path, {".gitignore": "*.bin\n", "allowed.py": "before\n"}
    )
    for name, contents in files.items():
        (repo / name).write_bytes(contents)
    tracker = RepositoryTracker(
        repo, tmp_path / "artifacts", **tracker_kwargs
    )

    with pytest.raises(ProtectedPathApprovalRequired, match="ignored inventory"):
        tracker.capture(_brief(allowed_paths=("allowed.py",)))


@pytest.mark.parametrize(
    ("tracker_kwargs", "files", "allowed_paths"),
    (
        (
            {"max_inventory_entries": 1},
            {"outside.py": b"outside\n"},
            ("allowed.py",),
        ),
        (
            {"max_inventory_listing_bytes": 8},
            {"long-outside-name.py": b"outside\n"},
            ("allowed.py",),
        ),
        (
            {"max_inventory_content_bytes": 4},
            {},
            ("allowed.py",),
        ),
    ),
)
def test_all_inventory_classes_fail_closed_before_entry_listing_or_content_bounds(
    tmp_path: Path,
    tracker_kwargs: dict[str, int],
    files: dict[str, bytes],
    allowed_paths: tuple[str, ...],
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    for name, contents in files.items():
        (repo / name).write_bytes(contents)
    tracker = RepositoryTracker(
        repo, tmp_path / "artifacts", **tracker_kwargs
    )

    with pytest.raises(ProtectedPathApprovalRequired, match="inventory"):
        tracker.capture(_brief(allowed_paths=allowed_paths))


def test_default_metadata_inventory_handles_more_than_ten_thousand_ignored_entries(
    tmp_path: Path,
) -> None:
    repo = _repository(
        tmp_path, {".gitignore": "ignored/\n", "allowed.py": "before\n"}
    )
    ignored = repo / "ignored"
    ignored.mkdir()
    for index in range(10_001):
        (ignored / f"entry-{index:05d}").touch()
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")

    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))

    ignored_records = [
        record for record in baseline.paths if record.path.startswith("ignored/")
    ]
    assert len(ignored_records) == 10_001


def test_copy_never_writes_a_mid_copy_growth_past_the_snapshot_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "x"})
    source_path = repo / "allowed.py"
    artifact_path = tmp_path / "copy.bin"
    original_open = repository._open_regular

    class GrowingSource:
        def __init__(self, source: object) -> None:
            self._source = source
            self._grew = False

        def __enter__(self) -> "GrowingSource":
            self._source.__enter__()  # type: ignore[union-attr]
            return self

        def __exit__(self, *args: object) -> None:
            self._source.__exit__(*args)  # type: ignore[union-attr]

        def fileno(self) -> int:
            return self._source.fileno()  # type: ignore[union-attr]

        def read(self, size: int) -> bytes:
            if not self._grew:
                self._grew = True
                with source_path.open("ab") as growth:
                    growth.write(b"growth")
            return self._source.read(size)  # type: ignore[union-attr]

    @contextmanager
    def growing_open(repo_root: Path, path: Path):
        with original_open(repo_root, path) as source:
            yield GrowingSource(source)

    monkeypatch.setattr(repository, "_open_regular", growing_open)

    with pytest.raises(repository._SnapshotTooLarge):
        repository._copy_regular_file(repo, source_path, artifact_path, max_snapshot_bytes=1)

    assert artifact_path.read_bytes() == b""


def test_fifo_before_image_is_rejected_without_blocking(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    record = next(record for record in baseline.paths if record.path == "allowed.py")
    assert record.before_image is not None
    record.before_image.unlink()
    os.mkfifo(record.before_image)
    (repo / "allowed.py").write_text("after\n")

    with pytest.raises(RuntimeError, match="baseline before-image integrity"):
        tracker.compare(baseline)


def test_artifact_anchor_survives_path_replacement_by_ancestor_symlink(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    artifact_directory = tmp_path / "artifacts"
    tracker = RepositoryTracker(repo, artifact_directory)
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    original_directory = tmp_path / "artifact-original"
    artifact_directory.rename(original_directory)
    replacement = tmp_path / "artifact-replacement"
    replacement.mkdir()
    artifact_directory.symlink_to(replacement, target_is_directory=True)
    (repo / "allowed.py").write_text("after\n")

    delta = tracker.compare(baseline)

    assert "-before" in delta.text_diffs["allowed.py"]


def test_tracker_close_is_idempotent_and_context_managed(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    tracker.close()
    tracker.close()
    with pytest.raises(RuntimeError, match="closed"):
        tracker.capture(_brief(allowed_paths=("allowed.py",)))

    with RepositoryTracker(repo, tmp_path / "artifacts-2") as managed:
        managed.capture(_brief(allowed_paths=("allowed.py",)))
    with pytest.raises(RuntimeError, match="closed"):
        managed.capture(_brief(allowed_paths=("allowed.py",)))


def test_oversized_tampered_before_image_is_bounded_and_rejected(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")
    baseline = tracker.capture(_brief(allowed_paths=("allowed.py",)))
    record = next(record for record in baseline.paths if record.path == "allowed.py")
    assert record.before_image is not None
    record.before_image.write_bytes(b"x" * (record.size + 1))
    (repo / "allowed.py").write_text("after\n")

    with pytest.raises(RuntimeError, match="baseline before-image integrity"):
        tracker.compare(baseline)


def test_non_utf8_git_path_is_rejected_explicitly(tmp_path: Path) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})
    raw_path = os.fsencode(repo) + b"/invalid-\xff"
    descriptor = os.open(raw_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    _git(repo, "add", os.fsdecode(raw_path))
    _git(repo, "commit", "-m", "non-utf8")
    tracker = RepositoryTracker(repo, tmp_path / "artifacts")

    with pytest.raises(ValueError, match="valid UTF-8"):
        tracker.capture(_brief(allowed_paths=("allowed.py",)))


def test_constructor_validation_happens_before_artifact_descriptor_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repository(tmp_path, {"allowed.py": "before\n"})

    def unexpected_open(path: Path, *, create: bool) -> int:
        raise AssertionError("artifact descriptor should not open")

    monkeypatch.setattr(repository, "_open_absolute_directory", unexpected_open)

    with pytest.raises(ValueError, match="max_snapshot_bytes"):
        RepositoryTracker(repo, tmp_path / "artifacts", max_snapshot_bytes=-1)
