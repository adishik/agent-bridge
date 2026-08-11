from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from agent_bridge.contracts import TaskBrief


@pytest.fixture
def valid_brief() -> TaskBrief:
    return TaskBrief.from_dict({
        "task_id": "task-1",
        "revision": 1,
        "title": "Add bridge contracts",
        "objective": "Create immutable validated handoff contracts.",
        "context": ["The checkout may already be dirty."],
        "constraints": ["Fable is read-only."],
        "allowed_paths": ["src/agent_bridge", "tests/agent_bridge"],
        "out_of_scope": ["outside-project"],
        "acceptance_criteria": ["Invalid revisions raise ValueError."],
        "required_tests": ["tests/agent_bridge/test_contracts.py"],
        "risks": ["Approval must bind to one immutable revision."],
        "open_questions": [],
        "confidence": 0.95,
        "confidence_rationale": "All fields are explicit.",
    })


@pytest.fixture
def brief(valid_brief: TaskBrief) -> TaskBrief:
    return valid_brief


@pytest.fixture
def fake_claude(tmp_path: Path) -> Path:
    fixture = Path(__file__).parent / "fixtures" / "fake_claude.py"
    executable = tmp_path / "claude"
    shutil.copyfile(fixture, executable)
    executable.chmod(0o700)
    return executable


@pytest.fixture
def fake_codex(tmp_path: Path) -> Path:
    fixture = Path(__file__).parent / "fixtures" / "fake_codex.py"
    executable = tmp_path / "codex"
    shutil.copyfile(fixture, executable)
    executable.chmod(0o700)
    return executable
