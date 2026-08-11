from __future__ import annotations

import hashlib
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]
APACHE_2_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def _metadata() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_project_metadata_and_console_entry_point() -> None:
    project = _metadata()["project"]
    assert project["name"] == "agent-bridge"
    assert project["version"] == "0.1.0"
    assert project["requires-python"] == ">=3.11"
    assert project["license"] == "Apache-2.0"
    assert project["authors"] == [{"name": "Adi Shik"}]
    assert project["scripts"] == {
        "agent-bridge": "agent_bridge.__main__:main",
    }


def test_apache_license_and_notice_are_exact() -> None:
    license_bytes = (ROOT / "LICENSE").read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == APACHE_2_SHA256
    assert (ROOT / "NOTICE").read_text(encoding="utf-8") == (
        "Agent Bridge\nCopyright 2026 Adi Shik\n"
    )


def test_readme_is_generic_and_copy_pasteable() -> None:
    metadata = _metadata()["project"]
    assert metadata["readme"] == "README.md"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "pipx install ." in readme
    assert "agent-bridge --repo /path/to/project" in readme
    assert "ssh -N -L 56590:127.0.0.1:56590 YOUR_SSH_ALIAS" in readme
    assert "Claude Code subscription" in readme
    assert "usage credits" in readme.lower()
    assert "Codex CLI" in readme


def test_static_browser_assets_are_declared_as_package_data() -> None:
    package_data = _metadata()["tool"]["setuptools"]["package-data"]
    assert package_data["agent_bridge"] == [
        "static/*.html",
        "static/*.css",
        "static/*.js",
    ]
