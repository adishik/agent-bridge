# SSH Remote Bootstrap Design

## Goal

Provide a first-class open-source remote workflow that requires no manual Agent Bridge installation on the SSH host:

```bash
agent-bridge ssh my-server --repo /absolute/remote/repository
```

The command uses the user's existing OpenSSH configuration and credentials, prepares a version-matched unprivileged Agent Bridge runtime remotely, starts the bridge against the remote repository and remote Claude/Codex CLIs, creates a loopback-only local tunnel, and opens or prints the correct keyed local URL.

## Scope and prerequisites

The local machine requires:

- the Agent Bridge CLI;
- an OpenSSH-compatible `ssh` executable; and
- a configured SSH destination, normally an alias in `~/.ssh/config`.

The remote machine requires:

- a POSIX-compatible SSH session;
- Python 3.11 or newer with `venv` support;
- network access to the configured Python package index during the first bootstrap of an Agent Bridge version;
- Git; and
- authenticated Claude and Codex CLIs available on the remote `PATH`.

Agent Bridge itself does not need to be installed manually on the remote machine. The first version does not promise offline bootstrap, Windows SSH hosts, bundled Claude/Codex installation, or remote credential setup. A future prebuilt-runtime feature may remove the remote Python/package-index prerequisites without changing the public connection command.

## Public CLI

Add an `ssh` subcommand while preserving every existing direct-launch spelling:

```bash
agent-bridge ssh SSH_DESTINATION --repo /absolute/remote/repository
agent-bridge ssh SSH_DESTINATION \
  --project app=/absolute/remote/app \
  --project docs=/absolute/remote/docs
```

Supported connection options are deliberately small:

- `--repo` or repeatable `--project`, with the same label/path rules as direct launch;
- `--local-port`, default `0` for an automatically selected loopback port;
- `--remote-port`, default `0` for a bounded automatically selected high port;
- `--python`, default `python3`, naming the remote Python command;
- `--no-open`, which prints but does not open the browser URL; and
- the existing remote executable overrides for Claude, Codex, Git, Bash, and sh.

The SSH destination may be a normal alias or `user@host`. Values that begin with `-`, contain control characters, or otherwise risk option injection are rejected. Arbitrary `ssh` flags are not accepted; users configure ProxyJump, identity files, ports, and host keys through standard OpenSSH configuration.

The compatible `--repo` spelling must work for any otherwise-valid absolute Git root, including directory names containing dots, beginning with digits, or exceeding the display-label limit. Its implicit display label is normalized into the existing `[A-Za-z][A-Za-z0-9_-]{0,31}` grammar without changing the canonical repository authority. This correction applies equally to direct and SSH launches so users do not need to invent a `--project` label as a workaround.

## Remote bootstrap

The local launcher resolves its exact installed Agent Bridge version and requests the same public version remotely. It invokes the remote Python through SSH to:

1. validate Python 3.11+ and `venv` support;
2. create `~/.cache/agent-bridge/runtime/<version>/venv` without root privileges;
3. install `agent-bridge==<exact-version>` through that environment's pip when the cache is absent or invalid;
4. verify the installed distribution reports the exact requested version; and
5. return a small versioned JSON bootstrap result.

The cache is immutable per version. A failed or partial installation is removed before retry; a valid cached version is reused without reinstalling. Persistent chats and project state continue to use the normal remote state directory and are not stored in the runtime cache.

Version bootstrap is release-only in the first implementation. The local distribution name and version must identify an Agent Bridge release resolvable from the configured package index. An editable install or any local/direct-URL installation provenance is rejected even when its version string equals a published release, because equal metadata would not prove equal code. A source-checkout-only or unpublished version fails before the tunnel starts with an actionable message; it never silently substitutes another version or package. Supporting unpublished development checkouts or offline transfer requires the deferred prebuilt-runtime/bundle mechanism.

No shell profile, system package, service, daemon, SSH key, or global Python environment is modified. Bootstrap commands and results are bounded. Repository paths and executable overrides are passed as quoted arguments, never interpolated into executable shell source.

## Tunnel and process lifecycle

After bootstrap, the launcher:

1. selects a free local IPv4 loopback port;
2. chooses a bounded remote high port, retrying a small fixed number of times if the remote listener reports a collision;
3. starts one foreground OpenSSH process with `ExitOnForwardFailure=yes`, a local `-L` forward, and a remote `exec` of the cached Agent Bridge runtime;
4. reads the remote launcher's single JSON startup record;
5. rewrites only its loopback port to the selected local port while preserving the access key;
6. polls the forwarded keyed root until it is reachable or the SSH process exits;
7. opens the URL with the platform browser unless `--no-open` was supplied; and
8. remains attached to the SSH process until normal interruption or remote exit.

The remote Agent Bridge remains bound to `127.0.0.1`; the local forward also binds only `127.0.0.1`. No public listener is introduced. `Ctrl+C` terminates the foreground SSH session; the remote command is executed directly so session closure terminates the remote bridge. The versioned runtime cache and normal Agent Bridge state remain for later connections.

Only the local URL is presented as actionable. Remote logs remain visible with access keys redacted outside the one required keyed URL. The launcher never stores passwords, private keys, access keys, or SSH-agent material.

## Failure behavior

Every preflight failure occurs before a browser is opened and returns a concise actionable error:

- SSH executable missing;
- destination rejected as unsafe;
- SSH authentication or host-key failure;
- remote Python missing or older than 3.11;
- remote `venv` unavailable;
- package-index/runtime installation failure;
- remote Git, Claude, or Codex unavailable;
- repository path invalid or not a Git top level;
- local forward unavailable;
- remote port collision after bounded retries;
- malformed or mismatched bootstrap/startup JSON; or
- SSH session exiting before HTTP readiness.

Failures do not fall back to a public bind, weaken host-key checking, prompt Agent Bridge to collect SSH credentials, or reuse a mismatched cached runtime.

## Components

Keep responsibilities separate:

- the existing direct launcher remains responsible for parsing and running a bridge on its own host;
- a new SSH connection module validates destination/options, runs bounded SSH subprocesses, manages bootstrap/cache protocol, chooses ports, rewrites the keyed URL, checks readiness, and owns cleanup;
- the CLI entry point performs compatibility-preserving subcommand dispatch and user-facing output only; and
- tests use a fake OpenSSH executable and fake HTTP readiness seam. They do not contact a real SSH host, package index, Claude, or Codex.

No coordinator, Store, Hub, provider adapter, browser API, database schema, or task protocol change is required.

## Testing and documentation

Focused tests must cover:

- existing direct-launch argument compatibility;
- compatible `--repo` label normalization for dotted, numeric-leading, and overlong directory names;
- safe SSH destination and remote argument quoting;
- exact version cache hit, first install, partial-install rollback, and version mismatch rejection;
- missing/old Python, venv/install failure, missing remote tools, and invalid repository;
- local/remote port selection, collision retry, `ExitOnForwardFailure`, and loopback-only forwarding;
- startup JSON validation, access-key-preserving local URL rewrite, delayed HTTP readiness, and no browser open before readiness;
- `--no-open` and browser-open success;
- `Ctrl+C`, early SSH exit, child cleanup, and no detached remote process;
- secret redaction and bounded diagnostics; and
- multi-project argument preservation.

README documentation will give a one-command example, prerequisites, first-connect cache behavior, SSH-config guidance, cleanup/cache location, security model, and troubleshooting. It will clearly state that Claude and Codex still need to be installed and authenticated remotely and that first bootstrap requires remote package-index access.

## Deferred work

- signed/prebuilt offline runtime bundles by OS and architecture;
- automatic Claude or Codex installation/authentication;
- browser-only SSH initiation;
- Windows SSH hosts;
- background sessions or reconnectable daemons;
- arbitrary SSH option passthrough; and
- remote repositories reached through a second SSH hop outside normal `~/.ssh/config` handling.
