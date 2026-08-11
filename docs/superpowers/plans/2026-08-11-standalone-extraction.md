# Standalone Agent Bridge Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mechanically extract the existing Agent Bridge into a fresh, installable, independently verifiable Apache-2.0 repository without carrying private project identifiers, filesystem links, runtime state, or Git history.

**Architecture:** Copy the existing package, static browser assets, fake-agent fixtures, and behavioral tests into a conventional `src/agent_bridge` layout, then change only namespace, package-resource, packaging, and generic operator-facing text. Preserve the current Claude/Fable and Codex/Sol adapters, coordinator, repository tracker, state machine, persistence, launcher, web API, UI, and security behavior.

**Tech Stack:** Python 3.11+, setuptools, FastAPI, Uvicorn, SQLite, vanilla HTML/CSS/JavaScript, pytest, fake Claude/Codex executables.

## Global Constraints

- This is a mechanical extraction, not a rewrite.
- The distribution name is exactly `agent-bridge`; the import package is exactly `agent_bridge`; the console command is exactly `agent-bridge`.
- Python support is `>=3.11`.
- Version 1 retains only the current Fable/Claude and Sol/Codex stack.
- Fable continues to require the installed Claude Code subscription login with no API-key or usage-based fallback.
- Preserve every existing approval, exact-revision, server gate, repository-boundary, recovery, audit-redaction, process, and fake-agent behavior.
- Runtime state remains outside target repositories under the existing XDG state mechanism.
- Do not migrate databases, conversations, baselines, schemas, credentials, or runtime artifacts.
- The repository must have fresh Git history, no remote, no symlinks, no submodules, no worktree link, and no dependency on another checkout.
- No private project identifier, path, branding, example, or metadata may enter a commit, commit message, package archive, or generated report.
- The full Apache-2.0 license applies only to the standalone repository; `NOTICE` names `Adi Shik` and the year `2026`.
- All validation is fake-only. Do not run live models, provider login flows, browser servers, network services, API keys, or paid operations.
- Do not modify, delete, stage, commit, or otherwise mutate the private source checkout.
- Stage explicit standalone paths only; never use `git add -A`.
- Do not configure a remote, push, publish a package, or delete the private copy.

## External execution inputs

The following values are required only in the implementation shell. They must
be absolute, must remain outside this repository, and must never be written to
tracked files, reports, command transcripts committed to Git, or commit
messages:

- `AGENT_BRIDGE_SOURCE_PACKAGE`: directory containing the approved source package.
- `AGENT_BRIDGE_SOURCE_TESTS`: directory containing the approved package tests and fake fixtures.
- `AGENT_BRIDGE_LEGACY_IMPORT`: exact previous Python import prefix.
- `AGENT_BRIDGE_LEGACY_PACKAGE_PATH`: exact previous slash-separated package path used in fixtures.
- `AGENT_BRIDGE_LEGACY_TEST_PATH`: exact previous slash-separated test path used in fixtures.
- `AGENT_BRIDGE_FORBIDDEN_PATTERNS`: external newline-delimited ripgrep pattern file containing all private names, variants, absolute paths, and legacy package references that must not survive extraction.
- `AGENT_BRIDGE_SOURCE_PACKAGE_DIGEST_FILE`: external file that stores only the approved source-package digest between tasks.
- `AGENT_BRIDGE_SOURCE_TESTS_DIGEST_FILE`: external file that stores only the approved source-test digest between tasks.

Before Task 1, validate the boundary without printing any input value:

```bash
test -n "${AGENT_BRIDGE_SOURCE_PACKAGE:?}"
test -n "${AGENT_BRIDGE_SOURCE_TESTS:?}"
test -n "${AGENT_BRIDGE_LEGACY_IMPORT:?}"
test -n "${AGENT_BRIDGE_LEGACY_PACKAGE_PATH:?}"
test -n "${AGENT_BRIDGE_LEGACY_TEST_PATH:?}"
test -f "${AGENT_BRIDGE_FORBIDDEN_PATTERNS:?}"
test -n "${AGENT_BRIDGE_SOURCE_PACKAGE_DIGEST_FILE:?}"
test -n "${AGENT_BRIDGE_SOURCE_TESTS_DIGEST_FILE:?}"
test "$(realpath "$AGENT_BRIDGE_SOURCE_PACKAGE")" != "$(git rev-parse --show-toplevel)"
test "$(realpath "$AGENT_BRIDGE_SOURCE_TESTS")" != "$(git rev-parse --show-toplevel)"
test -z "$(find "$AGENT_BRIDGE_SOURCE_PACKAGE" "$AGENT_BRIDGE_SOURCE_TESTS" -type l -print -quit)"
test -z "$(git remote)"
test "$(git rev-list --max-parents=0 --count HEAD)" -eq 1
```

Expected: every command exits 0 and prints no private value.

## File map

### Standalone metadata

- `.gitignore`: ignores local environments, caches, build outputs, and runtime residue.
- `pyproject.toml`: project metadata, dependencies, setuptools configuration, package data, console entry point, and pytest defaults.
- `LICENSE`: exact Apache License 2.0 text.
- `NOTICE`: standalone copyright notice.
- `MANIFEST.in`: explicit source-distribution inclusions.
- `README.md`: generic installation, security, workflow, SSH, state, and troubleshooting guide.

### Runtime package

- `src/agent_bridge/__init__.py`: package identity only.
- `src/agent_bridge/contracts.py`: immutable validated task and agent contracts.
- `src/agent_bridge/state_machine.py`: task states and allowed transitions.
- `src/agent_bridge/process.py`: bounded child-process execution and stopping.
- `src/agent_bridge/store.py`: SQLite persistence, events, recovery, and browser projections.
- `src/agent_bridge/repository.py`: approval baselines, Git authentication, and delta enforcement.
- `src/agent_bridge/coordinator.py`: user/Fable/Sol orchestration and approval lifecycle.
- `src/agent_bridge/adapters/base.py`: adapter protocols and result contract.
- `src/agent_bridge/adapters/claude_cli.py`: subscription-only read-only Fable adapter.
- `src/agent_bridge/adapters/codex_cli.py`: sandboxed Sol adapter.
- `src/agent_bridge/app.py`: authenticated HTTP/WebSocket application.
- `src/agent_bridge/__main__.py`: foreground CLI assembly and preflights.
- `src/agent_bridge/static/index.html`: generic browser workspace markup.
- `src/agent_bridge/static/styles.css`: browser workspace styles.
- `src/agent_bridge/static/app.js`: browser state, controls, replay, and rendering.

### Tests

- `tests/agent_bridge/conftest.py`: generic briefs and fake executable fixtures.
- `tests/agent_bridge/fixtures/fake_claude.py`: strict fake Fable executable.
- `tests/agent_bridge/fixtures/fake_codex.py`: strict fake Sol executable.
- `tests/agent_bridge/test_contracts.py`: contract validation.
- `tests/agent_bridge/test_state_machine.py`: transition validation.
- `tests/agent_bridge/test_process.py`: process lifecycle and stop races.
- `tests/agent_bridge/test_store.py`: persistence, event ordering, and recovery.
- `tests/agent_bridge/test_repository.py`: filesystem and Git boundary authentication.
- `tests/agent_bridge/test_claude_cli.py`: subscription and read-only Fable behavior.
- `tests/agent_bridge/test_codex_cli.py`: Sol schema, resume, environment, and audit behavior.
- `tests/agent_bridge/test_coordinator.py`: full orchestration state routing.
- `tests/agent_bridge/test_web.py`: HTTP, CSRF, bootstrap, WebSocket, and server gates.
- `tests/agent_bridge/test_static_ui.py`: safe browser DOM and controller behavior.
- `tests/agent_bridge/test_main.py`: launcher, lock, preflight, and executable wiring.
- `tests/agent_bridge/test_e2e_fake_agents.py`: real-stack fake-agent workflows.
- `tests/agent_bridge/test_packaging_metadata.py`: standalone metadata, license, README, and package-data assertions.

---

### Task 1: Establish the licensed standalone package skeleton

**Files:**
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `NOTICE`
- Create: `pyproject.toml`
- Create: `src/agent_bridge/__init__.py`
- Create: `tests/agent_bridge/__init__.py`
- Create: `tests/agent_bridge/test_packaging_metadata.py`

**Interfaces:**
- Consumes: the approved repository name, package name, CLI name, Python floor, dependency set, license, and copyright holder.
- Produces: installable project metadata; `agent_bridge` package namespace; `agent-bridge = "agent_bridge.__main__:main"`; development environment contract for all later tasks.

- [ ] **Step 1: Add the metadata tests before creating metadata**

Create `tests/agent_bridge/__init__.py`:

```python
"""Tests for the local Agent Bridge."""
```

Create `tests/agent_bridge/test_packaging_metadata.py`:

```python
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
```

- [ ] **Step 2: Run the metadata test and confirm honest RED**

Create the repository-local test environment, install pytest, and run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'pytest>=8.0'
.venv/bin/python -m pytest -q tests/agent_bridge/test_packaging_metadata.py
```

Expected: FAIL because `pyproject.toml`, `LICENSE`, and `NOTICE` do not exist.

- [ ] **Step 3: Add the exact standalone metadata and package shell**

Create `pyproject.toml` with these values:

```toml
[project]
name = "agent-bridge"
version = "0.1.0"
description = "Local browser coordination between a planning agent and an execution agent"
requires-python = ">=3.11"
license = "Apache-2.0"
license-files = ["LICENSE", "NOTICE"]
authors = [{ name = "Adi Shik" }]
dependencies = [
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
]

[project.optional-dependencies]
dev = [
  "build>=1.2",
  "httpx>=0.27",
  "pytest>=8.0",
  "pytest-timeout>=2.3",
  "pytest-xdist>=3.6",
]

[project.scripts]
agent-bridge = "agent_bridge.__main__:main"

[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = { "" = "src" }
include-package-data = true

[tool.setuptools.packages.find]
where = ["src"]
namespaces = false

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

Create `src/agent_bridge/__init__.py`:

```python
"""Contracts and state transitions for the local agent bridge."""
```

Create `.gitignore`:

```gitignore
.venv/
.pytest_cache/
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.coverage
htmlcov/
*.sqlite3
*.sqlite3-*
*.lock
```

Copy the exact system Apache-2.0 text and verify its known digest:

```bash
test "$(sha256sum /usr/share/common-licenses/Apache-2.0 | cut -d' ' -f1)" = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
cp /usr/share/common-licenses/Apache-2.0 LICENSE
sha256sum LICENSE
```

Create `NOTICE` exactly:

```text
Agent Bridge
Copyright 2026 Adi Shik
```

- [ ] **Step 4: Run the metadata tests**

Run:

```bash
python3 -m pytest -q tests/agent_bridge/test_packaging_metadata.py
```

Expected: 2 passed.

- [ ] **Step 5: Create and populate the repository-local development environment**

Populate the existing repository-local environment:

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q tests/agent_bridge/test_packaging_metadata.py
```

Expected: editable installation succeeds entirely from this repository and 2 tests pass.

- [ ] **Step 6: Run the privacy and Git-boundary checks before the first implementation commit**

Run:

```bash
! rg --hidden --quiet -i -f "$AGENT_BRIDGE_FORBIDDEN_PATTERNS" . --glob '!.git/**' --glob '!.venv/**'
test -z "$(find . -path ./.git -prune -o -path ./.venv -prune -o -type l -print -quit)"
test -z "$(git remote)"
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 7: Commit the package skeleton**

```bash
git add .gitignore LICENSE NOTICE pyproject.toml src/agent_bridge/__init__.py tests/agent_bridge/__init__.py tests/agent_bridge/test_packaging_metadata.py
git diff --cached --check
git commit -m "chore: establish standalone package metadata"
```

Expected: the commit contains only the seven listed paths and its message contains no private identifier.

---

### Task 2: Mechanically extract and sanitize the implementation and behavioral tests

**Files:**
- Modify: `src/agent_bridge/__init__.py`
- Create: `src/agent_bridge/__main__.py`
- Create: `src/agent_bridge/app.py`
- Create: `src/agent_bridge/contracts.py`
- Create: `src/agent_bridge/coordinator.py`
- Create: `src/agent_bridge/process.py`
- Create: `src/agent_bridge/repository.py`
- Create: `src/agent_bridge/state_machine.py`
- Create: `src/agent_bridge/store.py`
- Create: `src/agent_bridge/adapters/__init__.py`
- Create: `src/agent_bridge/adapters/base.py`
- Create: `src/agent_bridge/adapters/claude_cli.py`
- Create: `src/agent_bridge/adapters/codex_cli.py`
- Create: `src/agent_bridge/static/index.html`
- Create: `src/agent_bridge/static/styles.css`
- Create: `src/agent_bridge/static/app.js`
- Modify: `tests/agent_bridge/__init__.py`
- Create: `tests/agent_bridge/conftest.py`
- Create: `tests/agent_bridge/fixtures/fake_claude.py`
- Create: `tests/agent_bridge/fixtures/fake_codex.py`
- Create: `tests/agent_bridge/test_claude_cli.py`
- Create: `tests/agent_bridge/test_codex_cli.py`
- Create: `tests/agent_bridge/test_contracts.py`
- Create: `tests/agent_bridge/test_coordinator.py`
- Create: `tests/agent_bridge/test_e2e_fake_agents.py`
- Create: `tests/agent_bridge/test_main.py`
- Create: `tests/agent_bridge/test_process.py`
- Create: `tests/agent_bridge/test_repository.py`
- Create: `tests/agent_bridge/test_state_machine.py`
- Create: `tests/agent_bridge/test_static_ui.py`
- Create: `tests/agent_bridge/test_store.py`
- Create: `tests/agent_bridge/test_web.py`

**Interfaces:**
- Consumes: the external source-package and source-test directories; the three external legacy namespace/path values; Task 1's `agent_bridge` package shell and dev environment.
- Produces: the unchanged runtime stack under `agent_bridge`; the complete fake-only behavioral suite under `tests/agent_bridge`; generic repository-relative fixture paths rooted at `src/agent_bridge` and `tests/agent_bridge`.

- [ ] **Step 1: Snapshot and validate the private source without modifying it**

Run the following from the standalone repository:

```bash
test -d "$AGENT_BRIDGE_SOURCE_PACKAGE"
test -d "$AGENT_BRIDGE_SOURCE_TESTS"
test -f "$AGENT_BRIDGE_SOURCE_PACKAGE/__main__.py"
test -f "$AGENT_BRIDGE_SOURCE_TESTS/test_e2e_fake_agents.py"
test -z "$(find "$AGENT_BRIDGE_SOURCE_PACKAGE" "$AGENT_BRIDGE_SOURCE_TESTS" -type l -print -quit)"
find "$AGENT_BRIDGE_SOURCE_PACKAGE" -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum | sha256sum > "$AGENT_BRIDGE_SOURCE_PACKAGE_DIGEST_FILE"
find "$AGENT_BRIDGE_SOURCE_TESTS" -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum | sha256sum > "$AGENT_BRIDGE_SOURCE_TESTS_DIGEST_FILE"
test -s "$AGENT_BRIDGE_SOURCE_PACKAGE_DIGEST_FILE"
test -s "$AGENT_BRIDGE_SOURCE_TESTS_DIGEST_FILE"
```

Expected: inputs are regular directory trees with no symlink, and both external digest files are nonempty and contain only one SHA-256 line.

- [ ] **Step 2: Copy and mechanically rename tests first**

Run:

```bash
rsync -a --exclude '__pycache__/' --exclude '*.pyc' "$AGENT_BRIDGE_SOURCE_TESTS/" tests/agent_bridge/
```

Then run this bounded mechanical rewrite over test text only:

```python
from __future__ import annotations

import os
from pathlib import Path

replacements = (
    (os.environ["AGENT_BRIDGE_LEGACY_TEST_PATH"], "tests/agent_bridge"),
    (os.environ["AGENT_BRIDGE_LEGACY_PACKAGE_PATH"], "src/agent_bridge"),
    (os.environ["AGENT_BRIDGE_LEGACY_IMPORT"], "agent_bridge"),
)
for path in sorted(Path("tests/agent_bridge").rglob("*")):
    if not path.is_file() or path.suffix not in {".py", ".js", ".html", ".css"}:
        continue
    original = path.read_text(encoding="utf-8")
    rewritten = original
    for old, new in replacements:
        rewritten = rewritten.replace(old, new)
    if rewritten != original:
        path.write_text(rewritten, encoding="utf-8")
```

Run the Python block through `.venv/bin/python` from the repository root. This is an approved bulk mechanical rewrite; do not store it as a repository script.

- [ ] **Step 3: Run a focused test and confirm honest RED before copying source**

Run:

```bash
.venv/bin/python -m pytest -q tests/agent_bridge/test_contracts.py
```

Expected: collection fails with `ModuleNotFoundError` for `agent_bridge.contracts` because only the package shell exists.

- [ ] **Step 4: Copy the runtime package without its private operator guide**

Run:

```bash
rsync -a --exclude 'README.md' --exclude '__pycache__/' --exclude '*.pyc' "$AGENT_BRIDGE_SOURCE_PACKAGE/" src/agent_bridge/
```

Run the same mechanical rewrite algorithm from Step 2 over
`src/agent_bridge`, using the same three replacements and the same text suffix
allowlist.

Expected: implementation bytes differ from the source only where package
imports, package fixture paths, or test fixture paths use the new standalone
namespace.

- [ ] **Step 5: Make the two known operator-facing fixtures generic before any commit**

In `src/agent_bridge/static/index.html`, require these exact public strings:

```html
<title>Agent Bridge</title>
```

and:

```html
<strong>Agent Bridge</strong>
```

In `tests/agent_bridge/conftest.py`, keep the copied `TaskBrief` structure but require these standalone fixture values:

```python
"allowed_paths": ["src/agent_bridge", "tests/agent_bridge"],
"out_of_scope": ["outside-project"],
"required_tests": ["tests/agent_bridge/test_contracts.py"],
```

Do not alter any validation rule or assertion strength.

- [ ] **Step 6: Prove the privacy scan is clean before running or staging copied code**

Run:

```bash
! rg --hidden --quiet -i -f "$AGENT_BRIDGE_FORBIDDEN_PATTERNS" src tests --glob '!__pycache__/**' --glob '!*.pyc'
! rg --quiet -F "$AGENT_BRIDGE_LEGACY_IMPORT" src tests
! rg --quiet -F "$AGENT_BRIDGE_LEGACY_PACKAGE_PATH" src tests
! rg --quiet -F "$AGENT_BRIDGE_LEGACY_TEST_PATH" src tests
test -z "$(find src tests -type l -print -quit)"
```

Expected: all commands exit 0. Any hit is a hard stop: do not stage the copy and do not substitute a weaker pattern.

- [ ] **Step 7: Reinstall the editable package and run narrow suites**

Run:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_contracts.py \
  tests/agent_bridge/test_state_machine.py \
  tests/agent_bridge/test_process.py \
  tests/agent_bridge/test_store.py \
  tests/agent_bridge/test_repository.py
```

Expected: all selected tests pass with no import from another checkout.

- [ ] **Step 8: Run adapter, coordinator, web, launcher, and fake E2E suites**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_claude_cli.py \
  tests/agent_bridge/test_codex_cli.py \
  tests/agent_bridge/test_coordinator.py \
  tests/agent_bridge/test_web.py \
  tests/agent_bridge/test_static_ui.py \
  tests/agent_bridge/test_main.py \
  tests/agent_bridge/test_e2e_fake_agents.py
```

Expected: all tests pass using only fake Claude, fake Codex, temporary Git repositories, and loopback-local test clients. No live model or browser server starts.

- [ ] **Step 9: Run the complete extracted suite**

Run:

```bash
.venv/bin/python -m pytest -q tests/agent_bridge
```

Expected: the complete copied suite plus the metadata tests passes. Investigate any count difference from the source suite; do not silently omit, skip, or weaken a copied test.

- [ ] **Step 10: Prove the source tree was not changed**

Run:

```bash
test "$(cat "$AGENT_BRIDGE_SOURCE_PACKAGE_DIGEST_FILE")" = "$(find "$AGENT_BRIDGE_SOURCE_PACKAGE" -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum | sha256sum)"
test "$(cat "$AGENT_BRIDGE_SOURCE_TESTS_DIGEST_FILE")" = "$(find "$AGENT_BRIDGE_SOURCE_TESTS" -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum | sha256sum)"
```

Expected: both exact digest comparisons pass.

- [ ] **Step 11: Stage, inspect, rescan, and commit the extraction**

```bash
git add src/agent_bridge tests/agent_bridge
git diff --cached --check
! git diff --cached --text | rg --quiet -i -f "$AGENT_BRIDGE_FORBIDDEN_PATTERNS"
! git diff --cached --text | rg --quiet -F "$AGENT_BRIDGE_LEGACY_IMPORT"
git diff --cached --stat
git commit -m "feat: extract standalone agent bridge"
```

Expected: the commit contains only runtime source, static assets, fake fixtures,
and tests; no README from the source checkout and no private identifier appears
in content or the commit message.

---

### Task 3: Add the generic operator guide and source-distribution manifest

**Files:**
- Create: `README.md`
- Create: `MANIFEST.in`
- Modify: `pyproject.toml`
- Modify: `tests/agent_bridge/test_packaging_metadata.py`

**Interfaces:**
- Consumes: the installed CLI and preserved runtime behavior from Task 2.
- Produces: generic public installation and operation documentation; `project.readme = "README.md"`; explicit source-distribution inclusions.

- [ ] **Step 1: Add a failing operator-guide contract**

Append to `tests/agent_bridge/test_packaging_metadata.py`:

```python
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
```

- [ ] **Step 2: Run the test and confirm honest RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/agent_bridge/test_packaging_metadata.py::test_readme_is_generic_and_copy_pasteable
```

Expected: FAIL because `project.readme` and `README.md` do not exist.

- [ ] **Step 3: Write the standalone README without copying private examples**

Create `README.md` with these exact sections and operational requirements:

1. `# Agent Bridge` — describe a single-user, loopback-only browser chat where Fable/Claude plans and reviews and Sol/Codex executes.
2. `## Install` — show both:

   ```bash
   pipx install .
   ```

   and:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -e '.[dev]'
   ```

3. `## Prerequisites` — Python 3.11+, Git, Claude Code with subscription login, Codex CLI, Bash, and `sh`; explicitly state that Agent Bridge supplies no API-key fallback.
4. `## Subscription and usage safety` — retain the bounded provider-environment cleanup, safe Claude auth projection, separate account-level usage-credit acknowledgement, and the warning that Sol's version preflight does not prove billing method.
5. `## Run` — show exactly:

   ```bash
   agent-bridge --repo /path/to/project
   ```

   Explain the default `127.0.0.1:56590`, foreground ownership, keyed URL, and no public bind/background/tmux operation.
6. `## SSH access` — show exactly:

   ```bash
   ssh -N -L 56590:127.0.0.1:56590 YOUR_SSH_ALIAS
   ```

   Explain that the command runs on the user's local computer and the browser opens the keyed loopback URL.
7. `## Workflow` — document request, exact revision review, approve/edit/reject, Sol execution, Fable review, bounded correction, user escalation, Stop, and explicit Resume.
8. `## Repository safety` — document dirty-baseline preservation, allowed/unexpected/protected deltas, no automatic Git mutation, and user-owned integration.
9. `## State and recovery` — document XDG state, 0700 directories, 0600 files, kernel lock, latest active revision interruption, inert historical PID/PGID, and the current same-basename namespace tradeoff.
10. `## Troubleshooting` — cover missing executables, Fable subscription unavailable, Sol executable unavailable, occupied port, active lock, and reconnect behavior.
11. `## Development` — show `.venv/bin/python -m pytest -q tests/agent_bridge` and state that tests use fake agents and temporary repositories only.
12. `## License` — state Apache-2.0 and `Copyright 2026 Adi Shik`.

Every repository example must use `/path/to/project` or a shell variable. Every SSH example must use `YOUR_SSH_ALIAS`. Do not include a personal hostname, username, IP address, home directory, private project name, or source-checkout path.

- [ ] **Step 4: Wire README metadata and the source manifest**

Add to `[project]` in `pyproject.toml`:

```toml
readme = "README.md"
```

Create `MANIFEST.in`:

```text
include LICENSE
include NOTICE
include README.md
recursive-include docs *.md
recursive-include src/agent_bridge/static *.html *.css *.js
recursive-include tests *.py
```

- [ ] **Step 5: Verify docs and metadata**

Run:

```bash
.venv/bin/python -m pytest -q tests/agent_bridge/test_packaging_metadata.py
.venv/bin/python -m agent_bridge --help
git diff --check
! rg --hidden --quiet -i -f "$AGENT_BRIDGE_FORBIDDEN_PATTERNS" README.md MANIFEST.in pyproject.toml docs src tests --glob '!__pycache__/**' --glob '!*.pyc'
```

Expected: metadata tests pass; help lists required `--repo`, loopback host/default port, and optional absolute executable overrides; all scans pass.

- [ ] **Step 6: Commit the generic operator documentation**

```bash
git add README.md MANIFEST.in pyproject.toml tests/agent_bridge/test_packaging_metadata.py
git diff --cached --check
! git diff --cached --text | rg --quiet -i -f "$AGENT_BRIDGE_FORBIDDEN_PATTERNS"
git commit -m "docs: add standalone operator guide"
```

Expected: one documentation/metadata commit with no source-specific identifier.

---

### Task 4: Prove wheel, source distribution, and isolated installation behavior

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/agent_bridge/test_packaging_metadata.py`

**Interfaces:**
- Consumes: Task 3 metadata and the complete `agent_bridge` package.
- Produces: explicit static package-data declaration; inspected wheel and source distribution; clean-environment CLI and resource proof.

- [ ] **Step 1: Add a failing static package-data contract**

Append to `tests/agent_bridge/test_packaging_metadata.py`:

```python
def test_static_browser_assets_are_declared_as_package_data() -> None:
    package_data = _metadata()["tool"]["setuptools"]["package-data"]
    assert package_data["agent_bridge"] == [
        "static/*.html",
        "static/*.css",
        "static/*.js",
    ]
```

- [ ] **Step 2: Run the test and confirm honest RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/agent_bridge/test_packaging_metadata.py::test_static_browser_assets_are_declared_as_package_data
```

Expected: FAIL with missing `package-data` metadata.

- [ ] **Step 3: Declare exact package data**

Add to `pyproject.toml`:

```toml
[tool.setuptools.package-data]
agent_bridge = [
  "static/*.html",
  "static/*.css",
  "static/*.js",
]
```

- [ ] **Step 4: Run metadata and full source-tree tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/agent_bridge/test_packaging_metadata.py
.venv/bin/python -m pytest -q tests/agent_bridge
```

Expected: all metadata and copied behavioral tests pass.

- [ ] **Step 5: Build wheel and source distribution from a clean output directory**

Run:

```bash
test ! -e dist
.venv/bin/python -m build
find dist -maxdepth 1 -type f -name 'agent_bridge-0.1.0-*.whl' -print -quit | grep -q .
test -f dist/agent_bridge-0.1.0.tar.gz
```

Expected: exactly one wheel and one source archive are produced without reading another checkout.

- [ ] **Step 6: Inspect archive membership and privacy**

Run this read-only verifier with `.venv/bin/python`:

```python
from __future__ import annotations

from pathlib import Path
import tarfile
import zipfile

dist = Path("dist")
wheel = next(dist.glob("agent_bridge-0.1.0-*.whl"))
sdist = dist / "agent_bridge-0.1.0.tar.gz"
with zipfile.ZipFile(wheel) as archive:
    wheel_names = set(archive.namelist())
assert "agent_bridge/static/index.html" in wheel_names
assert "agent_bridge/static/styles.css" in wheel_names
assert "agent_bridge/static/app.js" in wheel_names
assert not any(name.startswith("tests/") for name in wheel_names)
assert all(
    name.startswith("agent_bridge/") or ".dist-info/" in name
    for name in wheel_names
)
with tarfile.open(sdist, "r:gz") as archive:
    source_names = set(archive.getnames())
prefix = "agent_bridge-0.1.0/"
for required in (
    "LICENSE",
    "NOTICE",
    "README.md",
    "pyproject.toml",
    "src/agent_bridge/static/index.html",
):
    assert prefix + required in source_names
```

Extract both archives into separate `mktemp -d` directories and run:

```bash
! rg --hidden --quiet -i -f "$AGENT_BRIDGE_FORBIDDEN_PATTERNS" "$WHEEL_EXTRACT_DIR" "$SDIST_EXTRACT_DIR"
test -z "$(find "$WHEEL_EXTRACT_DIR" "$SDIST_EXTRACT_DIR" -type l -print -quit)"
```

Expected: required assets exist; wheel membership is package-only; archives contain no private identifier or symlink.

- [ ] **Step 7: Install the wheel into a fresh environment outside the checkout**

Create a temporary directory with `mktemp -d`, set its absolute path as
`SMOKE_ROOT`, and run:

```bash
WHEEL=$(realpath dist/agent_bridge-0.1.0-*.whl)
python3 -m venv "$SMOKE_ROOT/venv"
"$SMOKE_ROOT/venv/bin/python" -m pip install "$WHEEL"
mkdir "$SMOKE_ROOT/work"
cd "$SMOKE_ROOT/work"
"$SMOKE_ROOT/venv/bin/agent-bridge" --help
"$SMOKE_ROOT/venv/bin/python" - <<'PY'
from importlib.metadata import version
from importlib.resources import files

assert version("agent-bridge") == "0.1.0"
root = files("agent_bridge")
for name in ("index.html", "styles.css", "app.js"):
    asset = root.joinpath("static", name)
    assert asset.is_file()
    assert asset.read_text(encoding="utf-8").strip()
PY
```

Expected: the command and resources work while the current directory is outside the source checkout and `PYTHONPATH` is unset.

- [ ] **Step 8: Commit package-data metadata**

Return to the repository and run:

```bash
git add pyproject.toml tests/agent_bridge/test_packaging_metadata.py
git diff --cached --check
git commit -m "build: package the browser workspace"
```

Expected: the commit contains exactly the package-data declaration and its test. Build outputs remain ignored and untracked.

---

### Task 5: Certify the standalone boundary and record verification

**Files:**
- Create: `docs/superpowers/results/2026-08-11-standalone-extraction-verification.md`

**Interfaces:**
- Consumes: complete standalone source, tests, docs, wheel, source distribution, and external forbidden-pattern file.
- Produces: final fake-only verification evidence and a clean local repository ready for a separately authorized publication decision.

- [ ] **Step 1: Run the highest-value fake-agent gates**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_claude_cli.py \
  tests/agent_bridge/test_codex_cli.py \
  tests/agent_bridge/test_coordinator.py \
  tests/agent_bridge/test_e2e_fake_agents.py
```

Expected: all adapter, coordinator, question/answer, correction, Stop, recovery, protected-delta, and fake provenance cases pass without live services.

- [ ] **Step 2: Run the complete standalone suite**

Run:

```bash
.venv/bin/python -m pytest -q tests/agent_bridge
```

Expected: every copied and standalone packaging test passes; no test is skipped or weakened because of extraction.

- [ ] **Step 3: Run lightweight import and compile checks**

Run:

```bash
.venv/bin/python -m compileall -q src/agent_bridge tests/agent_bridge
env -u PYTHONPATH .venv/bin/python - <<'PY'
import sys
import agent_bridge
import agent_bridge.__main__

assert agent_bridge.__file__ is not None
for forbidden in ("torch", "numpy"):
    assert forbidden not in sys.modules
PY
node --input-type=module -e "import('./src/agent_bridge/static/app.js')"
```

Expected: compile and imports succeed; unrelated ML packages are not imported; JavaScript parses as an ES module.

- [ ] **Step 4: Run the final tracked-content, history, and filesystem audit**

Run:

```bash
git diff --check
test -z "$(git remote)"
test ! -e .gitmodules
test -z "$(git ls-files -s | awk '$1 == "120000" { print $4; exit }')"
test -z "$(find . -path ./.git -prune -o -path ./.venv -prune -o -path ./dist -prune -o -type l -print -quit)"
! git grep -I -n -i -f "$AGENT_BRIDGE_FORBIDDEN_PATTERNS"
! git log --format='%H%n%B' | rg --quiet -i -f "$AGENT_BRIDGE_FORBIDDEN_PATTERNS"
! rg --hidden --quiet -i -f "$AGENT_BRIDGE_FORBIDDEN_PATTERNS" dist
git fsck --full
```

Expected: no remote, submodule, symlink, external reference, private identifier, history leak, or Git integrity error.

- [ ] **Step 5: Re-run installed-wheel smoke from outside both checkouts**

Use a new `mktemp -d` directory and repeat Task 4 Step 7 with `PYTHONPATH`
unset. Additionally run from outside the checkout:

```bash
"$SMOKE_ROOT/venv/bin/python" - <<'PY'
from pathlib import Path
import sys
import agent_bridge

module_path = Path(agent_bridge.__file__).resolve(strict=True)
assert module_path.is_relative_to(Path(sys.prefix).resolve(strict=True))
PY
```

Expected: `agent_bridge` and its static assets load from the fresh environment,
not from any source checkout.

- [ ] **Step 6: Confirm the source copy remains unchanged**

Recompute the Task 2 source-package and source-test digests from the external
directories and compare them with
`AGENT_BRIDGE_SOURCE_PACKAGE_DIGEST_FILE` and
`AGENT_BRIDGE_SOURCE_TESTS_DIGEST_FILE` using the exact Task 2 Step 10 commands.

Expected: exact equality; do not store source paths or digests in this repository.

- [ ] **Step 7: Write the generic verification report**

Create `docs/superpowers/results/2026-08-11-standalone-extraction-verification.md` with:

```markdown
# Standalone Agent Bridge Extraction Verification

**Date:** 2026-08-11
**Version:** 0.1.0
**Status:** Ready for a separate publication decision

## Scope

The existing Agent Bridge implementation and fake-only behavioral suite were
mechanically extracted into the standalone `agent_bridge` package. No runtime
state, credentials, external repository link, or Git history was imported.

## Verification

- Focused adapter/coordinator/fake-agent suite: PASS (181 passed)
- Complete standalone suite: PASS (582 passed)
- Wheel and source distribution build: PASS
- Fresh-environment wheel installation and external-directory CLI smoke: PASS
- Static browser resource loading from installed wheel: PASS
- Lightweight Python and JavaScript import checks: PASS
- Tracked content, archive, and Git-history privacy scan: PASS
- Remote/submodule/symlink/worktree-link audit: PASS
- Source-copy immutability digest comparison: PASS

## Safety

All agent workflows used explicit fake executables and temporary repositories.
No live model, provider login, API key, paid service, browser server, remote Git
operation, or package publication was used.

## Deferred decisions

- Public hosting and remote configuration
- Package-index publication
- Collision-proof state namespaces for equal repository basenames
```

The recorded counts must match the observed final commands. Do not include
source paths, source project names, usernames, hostnames, IP
addresses, credentials, or raw environment values.

- [ ] **Step 8: Rescan the report and commit the certification**

Run:

```bash
! rg --hidden --quiet -i -f "$AGENT_BRIDGE_FORBIDDEN_PATTERNS" docs/superpowers/results
git add docs/superpowers/results/2026-08-11-standalone-extraction-verification.md
git diff --cached --check
git commit -m "docs: certify standalone extraction"
```

Expected: one generic evidence commit.

- [ ] **Step 9: Perform final post-commit verification**

Run:

```bash
git status --short --branch
git remote -v
git log --oneline --decorate
git diff HEAD^ --check
! git grep -I -n -i -f "$AGENT_BRIDGE_FORBIDDEN_PATTERNS"
! git log --format='%H%n%B' | rg --quiet -i -f "$AGENT_BRIDGE_FORBIDDEN_PATTERNS"
```

Expected: clean `main`, no remote output, standalone-only commits, clean diff,
and no private-pattern match. Stop here. Do not publish or modify the source
checkout.
