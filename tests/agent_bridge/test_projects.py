from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

from agent_bridge.projects import (
    MAX_PROJECTS,
    build_project_specs,
    parse_project_argument,
    project_id_for_root,
)


GIT_EXECUTABLE = Path(shutil.which("git") or "/usr/bin/git").resolve()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        [str(GIT_EXECUTABLE), *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _repository(parent: Path, name: str) -> Path:
    repo = parent / name
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Agent Bridge Test")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "fixture")
    return repo


def _projects(
    entries: list[tuple[str, Path]],
    tmp_path: Path,
    **kwargs: object,
):
    return build_project_specs(
        entries,
        state_root=tmp_path / "state",
        git_executable=GIT_EXECUTABLE,
        **kwargs,
    )


def _fake_git(path: Path, source: str) -> Path:
    path.write_text(
        "#!" + sys.executable + "\n" + source,
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def test_parse_project_argument_accepts_ascii_label_and_absolute_root() -> None:
    assert parse_project_argument("alpha_2-frontend=/absolute/root") == (
        "alpha_2-frontend",
        Path("/absolute/root"),
    )


@pytest.mark.parametrize("label", ("a", "A", "alpha", "Z9", "a_b-c"))
def test_parse_project_argument_accepts_the_label_grammar(label: str) -> None:
    assert parse_project_argument(f"{label}=/absolute/root")[0] == label


@pytest.mark.parametrize(
    "value",
    (
        "alpha",
        "= /absolute/root",
        "=/absolute/root",
        "1alpha=/absolute/root",
        "alpha.name=/absolute/root",
        "a" * 33 + "=/absolute/root",
        "alpha=relative/root",
        "alpha=/absolute/root\n",
        "alpha\x00=/absolute/root",
        "../../escape=/absolute/root",
    ),
)
def test_parse_project_argument_rejects_invalid_grammar_or_root(value: str) -> None:
    # This catches accepting labels or roots that could later become authority paths.
    with pytest.raises(ValueError):
        parse_project_argument(value)


def test_build_rejects_case_insensitive_duplicate_labels_before_state_creation(
    tmp_path: Path,
) -> None:
    first = _repository(tmp_path, "first")
    second = _repository(tmp_path, "second")
    state_root = tmp_path / "state"

    with pytest.raises(ValueError, match="label"):
        build_project_specs(
            [("Alpha", first), ("alpha", second)],
            state_root=state_root,
            git_executable=GIT_EXECUTABLE,
        )

    assert not state_root.exists()


def test_build_rejects_duplicate_canonical_roots_through_symlink_alias(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path, "repo")
    alias = tmp_path / "repo-alias"
    alias.symlink_to(repo, target_is_directory=True)

    with pytest.raises(ValueError, match="duplicate.*root"):
        _projects([("primary", repo), ("alias", alias)], tmp_path)


def test_build_limits_the_registry_to_max_projects(tmp_path: Path) -> None:
    repo = _repository(tmp_path, "repo")
    entries = [(f"p{index}", repo) for index in range(MAX_PROJECTS + 1)]

    with pytest.raises(ValueError, match="too many"):
        _projects(entries, tmp_path)


@pytest.mark.parametrize("timeout", (float("inf"), float("nan")))
def test_build_rejects_non_finite_git_probe_timeouts(
    tmp_path: Path,
    timeout: float,
) -> None:
    # This catches removing a finite deadline before subprocess setup can begin.
    repo = _repository(tmp_path, "repo")
    state_root = tmp_path / "state"

    with pytest.raises(ValueError, match="finite"):
        build_project_specs(
            [("repo", repo)],
            state_root=state_root,
            git_executable=GIT_EXECUTABLE,
            probe_timeout_seconds=timeout,
        )

    assert not state_root.exists()


def test_build_rejects_non_git_root_without_creating_state(tmp_path: Path) -> None:
    root = tmp_path / "not-a-repository"
    root.mkdir()
    state_root = tmp_path / "state"

    with pytest.raises(ValueError, match="Git"):
        build_project_specs(
            [("plain", root)],
            state_root=state_root,
            git_executable=GIT_EXECUTABLE,
        )

    assert not state_root.exists()


def test_build_rejects_unreadable_root_without_starting_a_probe(tmp_path: Path) -> None:
    repo = _repository(tmp_path, "repo")
    original_mode = stat.S_IMODE(repo.stat().st_mode)
    repo.chmod(0)
    try:
        with pytest.raises(ValueError, match="readable"):
            _projects([("blocked", repo)], tmp_path)
    finally:
        repo.chmod(original_mode)


def test_build_rejects_git_probe_timeout_without_creating_state(tmp_path: Path) -> None:
    repo = _repository(tmp_path, "repo")
    slow_git = _fake_git(
        tmp_path / "slow-git",
        "import time\ntime.sleep(5)\n",
    )
    state_root = tmp_path / "state"

    with pytest.raises(ValueError, match="deadline"):
        build_project_specs(
            [("slow", repo)],
            state_root=state_root,
            git_executable=slow_git,
            probe_timeout_seconds=0.05,
        )

    assert not state_root.exists()


def test_same_basename_repositories_have_distinct_identity_and_state(tmp_path: Path) -> None:
    first_parent = tmp_path / "first-parent"
    second_parent = tmp_path / "second-parent"
    first_parent.mkdir()
    second_parent.mkdir()
    first = _repository(first_parent, "app")
    second = _repository(second_parent, "app")

    specs = _projects([("first", first), ("second", second)], tmp_path)

    assert len(specs) == 2
    assert specs[0].project_id != specs[1].project_id
    assert specs[0].state_dir != specs[1].state_dir
    assert all(spec.state_dir.is_dir() for spec in specs)
    assert tuple(spec.project_id for spec in specs) == tuple(sorted(spec.project_id for spec in specs))


def test_renaming_a_label_does_not_change_state_identity(tmp_path: Path) -> None:
    repo = _repository(tmp_path, "repo")

    first = _projects([("before", repo)], tmp_path)[0]
    renamed = _projects([("after", repo)], tmp_path)[0]

    assert renamed.project_id == first.project_id
    assert renamed.state_dir == first.state_dir
    assert renamed.label == "after"


def test_project_id_requires_an_exact_canonical_root(tmp_path: Path) -> None:
    repo = _repository(tmp_path, "repo")
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)

    assert project_id_for_root(repo) == project_id_for_root(repo.resolve())
    with pytest.raises(ValueError, match="canonical"):
        project_id_for_root(alias)


def test_invalid_label_and_control_character_root_never_start_a_git_probe(
    tmp_path: Path,
) -> None:
    log = tmp_path / "captured.jsonl"
    fake_git = _fake_git(
        tmp_path / "capture-git",
        "import json, os, pathlib\n"
        f"pathlib.Path({str(log)!r}).open('a', encoding='utf-8').write(json.dumps({{'argv': __import__('sys').argv, 'env': dict(os.environ)}}) + '\\n')\n"
        "print(os.getcwd())\n",
    )
    repo = _repository(tmp_path, "repo")

    with pytest.raises(ValueError):
        build_project_specs(
            [("../../escape", repo)],
            state_root=tmp_path / "state-label",
            git_executable=fake_git,
        )
    with pytest.raises(ValueError):
        build_project_specs(
            [("safe", tmp_path / "root\nwith-control")],
            state_root=tmp_path / "state-root",
            git_executable=fake_git,
        )

    assert not log.exists()


def test_git_probe_rejects_multiple_records(tmp_path: Path) -> None:
    repo = _repository(tmp_path, "repo")
    fake_git = _fake_git(
        tmp_path / "multi-record-git",
        "import os\nprint(os.getcwd())\nprint('/another/root')\n",
    )

    with pytest.raises(ValueError, match="record"):
        build_project_specs(
            [("repo", repo)],
            state_root=tmp_path / "state",
            git_executable=fake_git,
        )


def test_git_probe_uses_only_deterministic_non_secret_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repository(tmp_path, "repo")
    log = tmp_path / "captured.jsonl"
    fake_git = _fake_git(
        tmp_path / "capture-git",
        "import json, os, pathlib, sys\n"
        f"pathlib.Path({str(log)!r}).open('a', encoding='utf-8').write(json.dumps({{'argv': sys.argv, 'cwd': os.getcwd(), 'env': dict(os.environ)}}) + '\\n')\n"
        "if '--show-toplevel' in sys.argv:\n"
        "    print(os.getcwd())\n"
        "else:\n"
        "    print('main')\n",
    )
    hostile = {
        "GIT_DIR": "/hostile/git-dir",
        "GIT_WORK_TREE": "/hostile/work-tree",
        "GIT_CONFIG_GLOBAL": "/hostile/global-config",
        "GIT_CONFIG_SYSTEM": "/hostile/system-config",
        "OPENAI_API_KEY": "provider-secret",
        "ANTHROPIC_API_KEY": "another-provider-secret",
    }
    for name, value in hostile.items():
        monkeypatch.setenv(name, value)

    specs = build_project_specs(
        [("safe", repo)],
        state_root=tmp_path / "state",
        git_executable=fake_git,
    )

    assert specs[0].repo_root == repo.resolve()
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    for record in records:
        assert record["cwd"] == str(repo.resolve())
        assert all(value not in record["env"].values() for value in hostile.values())
        assert set(record["env"]) == {
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
        assert "--git-dir=/hostile/git-dir" not in record["argv"]
