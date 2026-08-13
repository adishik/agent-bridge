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
model action, it performs bounded startup checks. Claude's child environment
uses a bounded denylist of known provider selectors and overrides; it is not a
general secret scrubber. Start Agent Bridge from a clean launch shell and do
not place unrelated credentials in that shell's environment. Codex receives a
separately filtered minimal local-tool environment.

The browser has a separate account-level acknowledgement: confirm that Claude
account usage credits are disabled before sending or approving work. This
acknowledgement is not inferred from the CLI. Sol's version preflight only
proves that the Codex executable responds; it does not prove the billing
method, account, or usage mode behind that executable.

## Run

From this checkout, install, or another directory containing the package, run
one immutable project allowlist. For a single-project compatible launch:

```bash
agent-bridge --repo /path/to/project
```

For more than one project, repeat `--project` with a display label and an
absolute Git-root path:

```bash
agent-bridge \
  --project app=/absolute/path/to/app \
  --project docs=/absolute/path/to/docs
```

`--repo` remains the compatible one-project spelling; it cannot be combined
with `--project`. The selected roots are the complete authority for a running
bridge: labels are display-only, roots are canonicalized and validated as Git
top levels, and the allowlist cannot be changed through the browser. Stop the
foreground bridge and restart with a new command to add, remove, or rename a
project. The launcher rejects duplicate roots, duplicate labels, non-absolute
roots, and unsafe executable paths before serving the browser.

The default listener is `127.0.0.1:56590`. The process stays in the
foreground and owns the server lifetime; stop it with the normal terminal
interrupt. Startup prints a keyed loopback URL such as
`http://127.0.0.1:56590/?key=<key>`. Open that exact URL in one browser session.
There is no public bind, background daemon, or tmux operation. If the port is
busy, choose another local port with `--port` and use the port printed at
startup.

The launcher accepts `--repo`, repeatable `--project`, and optional absolute
executable overrides:
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

1. Select a project, then select or create a chat within that project. Chats,
   task IDs, and provider-session IDs are project-local: identical identifiers
   in two projects are not shared state.
2. Submit a request in the selected chat. Fable receives that repository's
   context and returns a structured plan, allowed paths, and required checks.
3. Review the exact task revision shown by the bridge, including its scope and
   requested test command.
4. Approve that revision, edit it, or reject it. An edit creates the next
   revision; approve only the exact revision currently displayed.
5. After approval, Sol executes within the approved repository scope. The
   bridge records the commands and compares the checkout with the baseline
   captured immediately before execution.
6. Fable reviews Sol's result and the observed repository delta. It can accept
   the result, ask a bounded correction, or ask the user for clarification.
7. A bounded correction stays tied to the same task and revision lineage. If
   the requested change widens scope or cannot be resolved safely, the bridge
   escalates to the user instead of guessing.
8. Use **Stop** to interrupt an active run. An interruption is recorded and
   does not silently continue.
9. Use **Resume** only after explicitly choosing to resume the interrupted
   task. Reconnect first if the browser was closed; a historical process ID is
   not evidence that work is still running.

There is one hub-wide active workflow, not one per project. While a model
workflow owns that lease, model-starting actions and opening another
project/chat's live workflow are rejected until it reaches a terminal state or
is stopped. This is deliberate: finish or stop the active workflow before
changing the selected project.

The composer always records the selected recipient. An ordinary request to
Fable begins planning. An ordinary request addressed to Sol is still visibly
routed through Fable until the exact displayed task revision has been approved;
it does not start Sol early. Fable is the planner and reviewer with authority
over intent and scope. Sol is the executor only for an exact approved,
non-terminal revision.

During an approved task, either agent may ask the other a structured question.
The question and its exact reply are shown in the main conversation before the
next agent runs, and each reply is bound to its project, chat, task, revision,
continuation generation, and question identifier. A reply that does not match
that exact question is rejected; it cannot be applied to another task. Agent
messages are limited to the task discussion: directed messaging never supplies
file paths, shell commands, executable names, environment values, or other
execution instructions.

Automatic agent-to-agent exchanges are deliberately bounded. Each approved
revision starts with three exchanges. At the limit the bridge pauses visibly and
asks the user to allow exactly three more; it never grants an open-ended agent
conversation. A scope-changing answer creates the next revision and requires
approval of that exact revision before Sol can continue. The user remains the
control point for approvals, question answers, Stop, and Resume. Directed
intervention and cross-project handoffs remain out of scope pending Phase 3.

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

Runtime state lives outside the target repositories under the platform's XDG
state location (`XDG_STATE_HOME` when set, otherwise the user's normal local
state directory). A multi-project launch has one hub database at
`<state>/agent-bridge/hub/hub.sqlite3` and one digest-identified project state
directory at `<state>/agent-bridge/projects/<project-id>/`, where
`<project-id>` is derived from the canonical repository root rather than its
label or basename. Project directories contain their own bridge database,
artifacts, schemas, and lock. The hub owns only account-level settings. State
directories and runtime subdirectories are owner-only (`0700`); SQLite
databases, locks, and other state files are owner-only (`0600`).

The usage-credit acknowledgement is one hub-wide account acknowledgement, not
one acknowledgement per project. Confirm it after startup before starting any
model action. A historical project-local acknowledgement is retained for audit
but never copied into the hub, so it cannot silently enable a multi-project
launch.

For an existing single-project installation, Agent Bridge recognizes the
historical basename state directory only after an exact ownership audit. The
audit requires every historical chat, task, event, baseline setting, and agent
run to belong to that one canonical repository; malformed, mixed-root, orphan,
or ambiguous legacy/digest state aborts startup. A successful audit adopts the
legacy directory in place: it preserves historical chats, events, baselines,
and runs instead of copying them into a digest directory. It likewise preserves
the old project-local acknowledgement without treating it as the new hub
acknowledgement.

A kernel-held hub lock plus per-project locks prevents overlapping bridge
instances. Startup acquires every lock before opening a hub or project
database; a partial lock failure starts no server and performs no recovery.

On startup, the latest active revision is recovered as interrupted rather than
being assumed safe to continue. Historical process IDs and process-group IDs
are inert records for audit and recovery; they are never treated as permission
to signal or resume a process. Resume is always an explicit user action.

## Troubleshooting

- **Missing executable:** install the named tool, put it on `PATH`, or pass an
  absolute path with the matching `--*-executable` option. The launcher checks
  every configured executable before starting the server.
- **Fable subscription unavailable:** sign in through Claude Code's supported
  subscription flow (for example, run its normal `claude auth login` flow in
  the launch shell), then restart the foreground bridge. API keys and
  usage-based fallbacks are not accepted.
- **Sol executable unavailable:** install or authenticate the Codex CLI, or
  pass its absolute executable path with `--codex-executable`; the startup
  version check must succeed.
- **Occupied port:** stop the other local listener, or pass a free loopback
  port with `--port`; use the resulting keyed URL and matching SSH forward.
- **Active lock or workflow:** another bridge instance may own a hub/project
  kernel lock, or the current bridge may have one active workflow. Close the
  other foreground instance, or finish/stop the active workflow before
  continuing; do not delete a lock file while it is held.
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
