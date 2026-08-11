# Agent Bridge

Agent Bridge is a single-user, loopback-only browser chat for coordinating two
local command-line agents. Fable (Claude) plans and reviews work; Sol (Codex)
executes the approved work in the selected Git repository. The browser is a
control surface for that handoff, not a public service.

## Install

With `pipx`:

```bash
pipx install .
```

For a development checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Prerequisites

You need Python 3.11 or newer, Git, Bash, `sh`, the Claude Code CLI with a
Claude Code subscription login, and the Codex CLI. Agent Bridge supplies no
API-key fallback: Fable must use the installed Claude Code subscription
authentication, and Sol must use the local Codex CLI. Make sure both CLIs are
available on `PATH`, or provide absolute executable overrides when starting
the bridge.

## Subscription and usage safety

Agent Bridge is designed for subscription-backed local use. Before starting a
model action, it performs bounded startup checks and cleans provider-specific
environment variables from child processes. Claude receives only the safe
authentication projection needed for its saved subscription login; unrelated
provider credentials, tokens, and connection settings are not forwarded.
Codex receives a separately filtered local-tool environment.

The browser has a separate account-level acknowledgement: confirm that Claude
account usage credits are disabled before sending or approving work. This
acknowledgement is not inferred from the CLI. Sol's version preflight only
proves that the Codex executable responds; it does not prove the billing
method, account, or usage mode behind that executable.

## Run

From this checkout, install, or another directory containing the package, run:

```bash
agent-bridge --repo /path/to/project
```

The default listener is `127.0.0.1:56590`. The process stays in the
foreground and owns the server lifetime; stop it with the normal terminal
interrupt. Startup prints a keyed loopback URL such as
`http://127.0.0.1:56590/?key=<key>`. Open that exact URL in one browser session.
There is no public bind, background daemon, or tmux operation. If the port is
busy, choose another local port with `--port` and use the port printed at
startup.

The launcher accepts `--repo` and optional absolute executable overrides:
`--claude-executable`, `--codex-executable`, `--git-executable`,
`--bash-executable`, and `--sh-executable`. It also accepts a loopback-only
`--host` and a numeric `--port`; non-loopback hosts are rejected.

## SSH access

When the bridge runs on a remote machine, run this command on your local
computer:

```bash
ssh -N -L 56590:127.0.0.1:56590 YOUR_SSH_ALIAS
```

Keep that SSH forwarding command open. Then open the keyed loopback URL in a
browser on your local computer, using the same port and key printed by the
foreground bridge. The forwarding binds the local loopback interface to the
remote loopback listener; it does not make Agent Bridge public.

## Workflow

1. Submit a request in the browser. Fable receives the repository context and
   returns a structured plan, allowed paths, and required checks.
2. Review the exact task revision shown by the bridge, including its scope and
   requested test command.
3. Approve that revision, edit it, or reject it. An edit creates the next
   revision; approve only the exact revision currently displayed.
4. After approval, Sol executes within the approved repository scope. The
   bridge records the commands and compares the checkout with the baseline
   captured immediately before execution.
5. Fable reviews Sol's result and the observed repository delta. It can accept
   the result, ask a bounded correction, or ask the user for clarification.
6. A bounded correction stays tied to the same task and revision lineage. If
   the requested change widens scope or cannot be resolved safely, the bridge
   escalates to the user instead of guessing.
7. Use **Stop** to interrupt an active run. An interruption is recorded and
   does not silently continue.
8. Use **Resume** only after explicitly choosing to resume the interrupted
   task. Reconnect first if the browser was closed; a historical process ID is
   not evidence that work is still running.

## Repository safety

Agent Bridge captures a dirty baseline and preserves it. Pre-existing changes
remain user-owned and are not reset, stashed, cleaned, committed, or pushed by
the bridge. Sol may change only the allowed repository-relative paths in the
approved revision.

After execution, the bridge distinguishes:

- allowed deltas, which are reviewed against the approved scope;
- unexpected deltas, which block completion and require user attention; and
- protected deltas, including repository control data and protected project
  areas, which require separate user approval.

The bridge does not automatically repair or integrate any delta. Inspect the
working tree and decide how to commit, revert, merge, or otherwise integrate
the result yourself.

## State and recovery

Runtime state lives outside the target repository under the platform's XDG
state location (`XDG_STATE_HOME` when set, otherwise the user's normal local
state directory). Repository state directories and runtime subdirectories are
owner-only (`0700`); the SQLite database, lock, and other state files are
owner-only (`0600`). A kernel-held instance lock prevents two bridge processes
from coordinating the same repository at once.

On startup, the latest active revision is recovered as interrupted rather than
being assumed safe to continue. Historical process IDs and process-group IDs
are inert records for audit and recovery; they are never treated as permission
to signal or resume a process. Resume is always an explicit user action.

The current state namespace is derived from the repository basename. This
keeps state paths predictable and outside repositories, but two different
repositories with the same basename share a namespace. Use one bridge instance
per basename at a time and treat the lock error as a signal to inspect the
existing state before proceeding.

## Troubleshooting

- **Missing executable:** install the named tool, put it on `PATH`, or pass an
  absolute path with the matching `--*-executable` option. The launcher checks
  every configured executable before starting the server.
- **Fable subscription unavailable:** sign in through Claude Code's supported
  subscription flow, then restart the foreground bridge. API keys and
  usage-based fallbacks are not accepted.
- **Sol executable unavailable:** install or authenticate the Codex CLI, or
  pass its absolute executable path with `--codex-executable`; the startup
  version check must succeed.
- **Occupied port:** stop the other local listener, or pass a free loopback
  port with `--port`; use the resulting keyed URL and matching SSH forward.
- **Active lock:** another instance owns the repository's kernel lock. Close
  that instance cleanly or inspect its foreground terminal and state before
  starting another one; do not delete the lock file while it is held.
- **Reconnect behavior:** reopen the exact keyed URL while the foreground
  process is still running. If the process stopped, restart it and follow its
  new startup URL; review any recovered interrupted revision before choosing
  Resume.

## Development

Run the Agent Bridge tests with:

```bash
.venv/bin/python -m pytest -q tests/agent_bridge
```

Tests use fake agents and temporary repositories only. They do not require
live model logins, provider credentials, browser servers, or paid services.

## License

Agent Bridge is licensed under Apache-2.0.

Copyright 2026 Adi Shik
