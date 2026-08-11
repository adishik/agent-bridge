# Standalone Agent Bridge Extraction Design

**Status:** Approved design, implementation pending

**Date:** 2026-08-11

**Project:** Agent Bridge

**Copyright holder:** Adi Shik

## Purpose

Agent Bridge is a local browser application in which a planning/review agent and
an execution agent collaborate on changes to a user-selected Git repository.
The standalone project packages the existing, proven implementation for reuse
across unrelated repositories.

This work is an extraction, not a rewrite. The implementation, state machine,
security boundaries, browser workflow, adapters, and tests are to be copied
substantially unchanged. Only changes required for an independent installable
package are in scope.

## Goals

- Create a completely independent repository named `agent-bridge`.
- Publish an installable Python package named `agent_bridge`.
- Provide the console command `agent-bridge`.
- Support arbitrary local Git repositories through `--repo`.
- Retain the existing Fable/Claude and Sol/Codex operating model.
- Preserve all approval, review, repository-boundary, recovery, and audit
  behavior.
- Make the project suitable for later public sharing under Apache-2.0.
- Prove that built packages and installed commands have no dependency on the
  private source checkout.

## Non-goals

- Rewriting or redesigning the application.
- Introducing a general provider plug-in system.
- Adding new model providers, remote execution, or hosted operation.
- Migrating existing tasks, conversations, baselines, databases, or state.
- Changing the state-directory naming scheme during extraction.
- Publishing, pushing, or creating a public remote as part of extraction.
- Deleting or modifying the private source implementation.

## Independence boundary

The standalone repository is created with a fresh `git init`. It must not use
or retain:

- copied Git history;
- a worktree, submodule, subtree, or shared Git directory;
- symlinks or filesystem references to the private source checkout;
- editable-install or `PYTHONPATH` dependencies on another checkout;
- private project branding, paths, examples, identifiers, data, or metadata;
- runtime state or artifacts copied from an earlier installation.

Only the Agent Bridge implementation and its directly relevant tests are
copied. Privacy checks use patterns supplied from outside the standalone
repository so private identifiers do not become repository content merely to
implement the check.

The repository remains local with no configured remote until a later,
separately authorized publication step.

## Repository layout

```text
agent-bridge/
├── LICENSE
├── NOTICE
├── README.md
├── pyproject.toml
├── docs/
│   └── superpowers/
│       ├── plans/
│       └── specs/
├── src/
│   └── agent_bridge/
│       ├── adapters/
│       ├── static/
│       ├── __init__.py
│       ├── __main__.py
│       ├── app.py
│       ├── contracts.py
│       ├── coordinator.py
│       ├── process.py
│       ├── repository.py
│       ├── state_machine.py
│       └── store.py
└── tests/
    └── agent_bridge/
```

The exact test filenames remain aligned with the copied implementation. The
layout changes from an application-internal namespace to the conventional
`src/agent_bridge` package without changing module responsibilities.

## Mechanical extraction changes

The implementation change set is intentionally narrow:

1. Copy the existing Agent Bridge source modules and static assets.
2. Move the package into `src/agent_bridge`.
3. Replace internal package imports with the `agent_bridge` namespace.
4. Move the relevant tests and update import and fixture paths.
5. Replace product-specific browser titles and operator examples with generic
   Agent Bridge wording.
6. Add standalone packaging metadata, licensing, and operator documentation.
7. Make package-resource lookup work from an installed wheel.

The extraction must not opportunistically refactor coordinator logic, adapter
behavior, persistence, repository tracking, process management, or UI state
handling. Behavioral improvements discovered during extraction are documented
for later work unless they are strictly required for the standalone package to
function.

## Packaging

- Python requirement: 3.11 or newer.
- Distribution name: `agent-bridge`.
- Import package: `agent_bridge`.
- Build backend: setuptools through `pyproject.toml`.
- Console entry point:

  ```toml
  agent-bridge = "agent_bridge.__main__:main"
  ```

- Static browser assets are included as package data.
- Current web runtime dependencies, including FastAPI and Uvicorn, are declared
  explicitly.
- Test and build dependencies are exposed through a development extra.
- Claude Code, Codex CLI, Git, and trusted shells remain external executable
  prerequisites rather than Python dependencies.

The package must build both a wheel and source distribution. The wheel must run
from outside the source tree in a clean virtual environment.

## Licensing

The standalone project is licensed under Apache License 2.0.

- `LICENSE` contains the full unmodified Apache-2.0 license text.
- `pyproject.toml` declares SPDX identifier `Apache-2.0`.
- `NOTICE` contains:

  ```text
  Agent Bridge
  Copyright 2026 Adi Shik
  ```

The license applies only to content committed to this standalone repository.

## Runtime model

The command:

```bash
agent-bridge --repo /absolute/path/to/project
```

selects the sole repository authority for that server instance. The repository
path is explicit and must resolve to a valid local Git checkout.

Version 1 keeps the two-agent roles fixed:

- **Fable / Claude:** plans, reviews, and requests clarification using the
  installed Claude Code subscription login.
- **Sol / Codex:** executes approved work using the installed Codex CLI.

The implementation must not introduce an Anthropic API-key or usage-based
fallback. Readiness, authentication, and usage-credit acknowledgement remain
distinct gates.

## Browser and safety behavior

Existing behavior is preserved:

- loopback-only binding;
- keyed browser URL and authentication cookie;
- CSRF validation on mutations;
- strict WebSocket replay and session filtering;
- exact task-revision approval;
- explicit usage-credit acknowledgement;
- Fable subscription and Sol readiness checks;
- Stop available independently of model readiness;
- bounded browser histories and task lists;
- safe text rendering for untrusted agent output.

No public bind, hosted mode, detached server, or automatic browser exposure is
added.

## Repository protection

The repository tracker remains authoritative for:

- approval-time baselines;
- allowed, unexpected, and protected deltas;
- symlink and special-file handling;
- Git metadata and index authentication;
- required-test evidence;
- before-image integrity;
- interruption and resume drift checks.

Bridge state remains outside the target repository under the user's XDG state
directory. The application performs no automatic Git commit, push, reset,
checkout, merge, or cleanup operation.

The current state namespace behavior, including the documented collision risk
for repositories with the same basename, is retained for this mechanical
extraction. A collision-proof namespace is a separate behavioral change.

## State and recovery

Existing SQLite persistence, event sequencing, task revision semantics,
single-instance locking, stale-run retirement, and interrupted-task recovery
are preserved. The extraction does not import state from another installation.

Each standalone installation starts with its own XDG-managed state. Process IDs
stored from prior runs remain audit information and are never treated as proof
that an old process is live.

## Testing strategy

Extraction testing uses fake agents only. It must not invoke live models,
provider authentication changes, network services, API keys, or paid usage.

Verification proceeds from narrow to broad:

1. Import and path tests for the renamed package.
2. Store, repository tracker, coordinator, adapter, web, and static UI suites.
3. Full fake-agent end-to-end workflows.
4. Complete standalone test suite.
5. Wheel and source-distribution builds.
6. Archive-content inspection.
7. Clean-environment installation and CLI smoke tests from outside the checkout.
8. Lightweight import checks.
9. External privacy and independence audit.

The copied behavioral assertions are retained. Tests may be updated for module
and fixture paths but must not be weakened merely to accommodate extraction.

## Independence acceptance criteria

The extraction is acceptable only when all of the following hold:

- The standalone repository has its own Git directory and begins with fresh
  standalone commits.
- `git remote -v` is empty.
- No tracked symlink, submodule, worktree pointer, or external path reference
  exists.
- No private project name, path, branding, example, or identifier occurs in
  tracked files, build metadata, wheel contents, source distribution, or Git
  commit messages.
- Source and test imports resolve exclusively through `agent_bridge`.
- A fresh environment can install the built wheel and invoke `agent-bridge`
  while its working directory is outside the source checkout.
- The installed package includes all required static browser assets.
- Full fake-only tests pass without credentials or network access.
- The original private checkout remains unmodified by extraction.

## Implementation sequence

1. Establish standalone licensing and packaging skeleton.
2. Copy source, tests, and static assets without history.
3. Perform namespace and package-resource path changes.
4. Replace product-specific wording and examples.
5. Run focused tests and resolve extraction-only failures.
6. Run the complete fake-only standalone suite.
7. Build and inspect distribution artifacts.
8. Install and smoke-test in an isolated environment.
9. Run privacy, independence, and Git-boundary audits.
10. Commit the reviewed extraction locally.

No publication or remote operation is included in this sequence.

## Risks and controls

### Accidental private coupling

**Risk:** An absolute path, example, package reference, or static asset retains a
private identifier.

**Control:** Scan tracked files, Git history, package metadata, and built
archives using externally supplied patterns; inspect links and package paths.

### Unintentional behavioral rewrite

**Risk:** Packaging changes alter security or orchestration behavior.

**Control:** Keep changes mechanical, retain existing assertions, compare test
coverage, and defer unrelated refactors.

### Installed-package resource failure

**Risk:** Static UI or schema resources work only from a source checkout.

**Control:** Use package-relative resources, build a wheel, install it in a
fresh environment, and exercise the CLI outside the checkout.

### Live-provider invocation during validation

**Risk:** A test accidentally calls a real CLI or consumes paid service usage.

**Control:** Use absolute fake executables, hostile-path tests, sentinels, and
runner-side executable provenance checks. Do not run a live smoke test without
new explicit approval.

## Completion boundary

Extraction is complete when the standalone local repository passes all
acceptance gates and contains a reviewed local commit. Creating a hosting
account, configuring a remote, pushing, publishing a package, or deleting the
private copy requires a separate user decision.
