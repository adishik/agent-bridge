# SSH Remote Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `agent-bridge ssh SSH_DESTINATION --repo /remote/repository` as a safe one-command workflow that prepares an exact released Agent Bridge runtime remotely, opens a loopback-only SSH tunnel, and launches the keyed local browser URL without a manual remote Agent Bridge installation.

**Architecture:** Keep direct launch unchanged and dispatch only the leading `ssh` spelling to a new local orchestration module. Send a small stdlib-only bootstrap program through OpenSSH to create or validate an immutable per-version remote venv, then run the existing launcher through one foreground SSH process; all remote arguments are shell-quoted, both listeners stay on IPv4 loopback, and browser opening waits for the exact keyed URL to answer.

**Tech Stack:** Python 3.11+, stdlib `argparse`, `importlib.metadata`, `venv`, `fcntl`, `subprocess`, `selectors`, `socket`, `urllib.request`, `webbrowser`, OpenSSH, pytest, and fake executables only.

## Global Constraints

- Source design: `docs/superpowers/specs/2026-08-16-ssh-remote-bootstrap-design.md`.
- Create a fresh linked worktree from committed HEAD with the `using-git-worktrees` skill before implementation. Do not alter, move, stage, or copy over the current uncommitted conversation UI source/test changes.
- Bring only this approved spec and plan into the SSH worktree before code work; the SSH feature is independent of the UI delta.
- Preserve every existing direct spelling such as `agent-bridge --repo /path`; only a leading literal `ssh` selects the new workflow.
- Normalize the implicit display label for `--repo` into `[A-Za-z][A-Za-z0-9_-]{0,31}` so dotted, numeric-leading, and overlong Git-root basenames work without requiring `--project`.
- The local and remote listener addresses are always `127.0.0.1`. Never add a public-bind fallback.
- The remote host is POSIX and must already have Python 3.11+ with `venv`, Git, and authenticated Claude/Codex CLIs. Agent Bridge never installs or authenticates provider CLIs.
- Runtime bootstrap is release-only: install the exact local distribution version from the remote package index, reject editable/local/direct-URL distribution provenance, and fail closed for an unavailable/unpublished or mismatched version.
- Do not accept arbitrary SSH flags or collect/store SSH passwords, private keys, agent material, browser keys, or provider credentials. OpenSSH config owns identity files, ports, ProxyJump, and host-key policy.
- No coordinator, Store, Hub, adapter, browser API, database, task protocol, or static UI change belongs in this plan.
- Tests use fake SSH/Python/provider tools and injected HTTP/browser seams only. Do not use a live SSH host, network package index, browser server, Claude, Codex, credentials, or paid service.
- Use `/home/adi/agent-bridge/.venv/bin/python` with `PYTHONPATH="$PWD/src"` in the isolated worktree.
- Do not commit, push, change remotes, or remove worktrees unless the user explicitly asks. Replace commit steps with explicit-path diff/review checkpoints.
- Every implementation task follows honest RED/GREEN TDD and ends at an independently reviewable boundary.

---

### Task 1: Bounded SSH command model and parser

**Files:**
- Create: `src/agent_bridge/ssh.py`
- Create: `tests/agent_bridge/test_ssh.py`

**Interfaces:**
- Consumes: existing `agent_bridge.projects.parse_project_argument()` and the `agent-bridge` distribution metadata.
- Produces: `SSHSettings` and `parse_ssh_settings()`; no public launcher dispatch is added until Task 4 has a complete foreground runner.

- [ ] **Step 1: Write failing SSH argument and compatibility tests**

Create `tests/agent_bridge/test_ssh.py` with a helper that supplies an absolute fake `ssh` executable and a fake installed distribution exposing `metadata["Name"]`, `version`, and `read_text("direct_url.json")`. Cover the single-repository and multi-project forms:

```python
def test_parse_ssh_settings_preserves_remote_authority_and_defaults(tmp_path: Path) -> None:
    ssh = _write_executable(tmp_path / "ssh", "raise SystemExit(0)\n")
    settings = parse_ssh_settings(
        ["workbox", "--repo", "/srv/demo"],
        executable_finder=lambda name: str(ssh) if name == "ssh" else None,
        distribution_reader=lambda name: _released_distribution("0.1.0"),
    )
    assert settings.destination == "workbox"
    assert settings.remote_arguments == ("--repo", "/srv/demo")
    assert settings.local_port == 0
    assert settings.remote_port == 0
    assert settings.python_command == "python3"
    assert settings.open_browser is True
    assert settings.version == "0.1.0"


def test_parse_ssh_settings_preserves_multi_project_and_remote_overrides(
    tmp_path: Path,
) -> None:
    ssh = _write_executable(tmp_path / "ssh", "raise SystemExit(0)\n")
    settings = parse_ssh_settings(
        [
            "adi@workbox",
            "--project", "app=/srv/app",
            "--project", "docs=/srv/docs with space",
            "--local-port", "43123",
            "--remote-port", "53123",
            "--python", "/opt/Python 3/bin/python3",
            "--no-open",
            "--claude-executable", "/home/adi/.local/bin/claude",
            "--codex-executable", "/home/adi/.local/bin/codex",
        ],
        executable_finder=lambda name: str(ssh),
        distribution_reader=lambda name: _released_distribution("0.1.0"),
    )
    assert settings.remote_arguments == (
        "--project", "app=/srv/app",
        "--project", "docs=/srv/docs with space",
        "--claude-executable", "/home/adi/.local/bin/claude",
        "--codex-executable", "/home/adi/.local/bin/codex",
    )
    assert settings.open_browser is False
```

Parameterize rejections for an empty destination, a destination beginning with `-`, whitespace/control characters, overlong destination, relative repository, unsafe/duplicate project label, relative remote executable override, nonnumeric/out-of-range ports, missing local `ssh`, missing/wrong-name distribution metadata, a version outside the bounded release-token grammar, editable `direct_url.json`, local-file/direct-URL provenance, and malformed provenance JSON.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_ssh.py -k ssh_settings
```

Expected: collection fails because `agent_bridge.ssh` and its public interfaces do not exist.

- [ ] **Step 3: Implement the bounded SSH settings model and parser**

Create `src/agent_bridge/ssh.py` with stdlib-only imports at module import time and these exact public interfaces:

```python
@dataclass(frozen=True, slots=True)
class SSHSettings:
    ssh_executable: Path
    destination: str
    remote_arguments: tuple[str, ...]
    local_port: int
    remote_port: int
    python_command: str
    open_browser: bool
    version: str


def parse_ssh_settings(
    argv: Sequence[str] | None = None,
    *,
    executable_finder: Callable[[str], str | None] = shutil.which,
    distribution_reader: Callable[[str], importlib.metadata.Distribution] = (
        importlib.metadata.distribution
    ),
) -> SSHSettings:
    arguments = _ssh_parser().parse_args(argv)
    destination = _validate_destination(arguments.destination)
    ssh_executable = _resolve_local_ssh(executable_finder("ssh"))
    version = _installed_release_version(distribution_reader("agent-bridge"))
    remote_arguments = _remote_bridge_arguments(arguments)
    return SSHSettings(
        ssh_executable=ssh_executable,
        destination=destination,
        remote_arguments=remote_arguments,
        local_port=_validate_port(arguments.local_port, label="local port"),
        remote_port=_validate_port(arguments.remote_port, label="remote port"),
        python_command=_validate_remote_command(arguments.python),
        open_browser=not arguments.no_open,
        version=version,
    )
```

Use one mutually exclusive required `--repo`/repeatable `--project` group. Require remote repository and executable paths to be safe absolute POSIX paths without control characters. Call `parse_project_argument()` for every project value, reject case-insensitive duplicate labels, and retain the original ordered strings in `remote_arguments`. Destination validation permits an OpenSSH alias or `user@host` as one non-whitespace argument up to 255 characters, but rejects leading `-` and control characters. Resolve the local SSH executable to an absolute regular executable. Bound the release token to 64 ASCII version characters (`[A-Za-z0-9][A-Za-z0-9._+!-]{0,63}`). `_installed_release_version()` must require canonical distribution name `agent-bridge`, parse `direct_url.json` if present, and reject every present direct URL (including editable and local wheel/source installs); normal package-index installations have no direct-URL record.

Do not define a temporary `main()` or `run_ssh()` stub in this task. Task 4 adds the public command only with its complete implementation.

- [ ] **Step 4: Run the complete parser test module**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_ssh.py
```

Expected: all Task 1 parser tests pass.

- [ ] **Step 5: Review the explicit Task 1 diff without committing**

```bash
git diff --check -- src/agent_bridge/ssh.py tests/agent_bridge/test_ssh.py
git diff -- src/agent_bridge/ssh.py tests/agent_bridge/test_ssh.py
```

Expected: only the bounded SSH input model and focused tests exist; direct startup is untouched.

---

### Task 2: Stdlib-only immutable remote runtime bootstrap

**Files:**
- Create: `src/agent_bridge/_remote_bootstrap.py`
- Create: `tests/agent_bridge/test_remote_bootstrap.py`

**Interfaces:**
- Consumes: exact release version and requested remote port supplied by Task 1.
- Produces: `ensure_runtime()`, `select_remote_port()`, `bootstrap_main()`, and one protocol-v1 JSON record containing `protocol`, `python`, `remote_port`, and `version`.

- [ ] **Step 1: Write failing bootstrap cache, security, and port tests**

Create a fake venv creator that makes only `venv/bin/python`, and a fake command runner that records exact argv and returns the requested version for verification. Cover:

```python
def test_first_bootstrap_installs_exact_release_and_cache_hit_does_not_reinstall(
    tmp_path: Path,
) -> None:
    calls = []
    first = ensure_runtime(
        "0.1.0",
        cache_root=tmp_path / "runtime",
        venv_creator=_fake_venv_creator,
        command_runner=_fake_runner(calls, reported_version="0.1.0"),
    )
    second = ensure_runtime(
        "0.1.0",
        cache_root=tmp_path / "runtime",
        venv_creator=lambda path: pytest.fail("cache hit recreated venv"),
        command_runner=_fake_runner(calls, reported_version="0.1.0"),
    )
    assert first == second
    installs = [argv for argv in calls if argv[1:4] == ("-m", "pip", "install")]
    assert installs == [(
        str(first), "-m", "pip", "install", "--disable-pip-version-check",
        "--no-input", "agent-bridge==0.1.0",
    )]
```

Add honest cases for an invalid partial cache being deleted/rebuilt, install failure leaving no final runtime, installed version mismatch, exact version verification on every cache hit, a symlinked/non-owner-controlled cache component, unsafe version, Python below 3.11, missing `venv`/pip, fixed remote port, automatic high-loopback port, candidate collision retry/exhaustion, and invalid port.

Test `bootstrap_main()` with in-memory stdout so it emits exactly one compact JSON line and no secret/environment data:

```python
assert json.loads(output.getvalue()) == {
    "protocol": 1,
    "python": str(runtime_python),
    "remote_port": 53123,
    "version": "0.1.0",
}
```

- [ ] **Step 2: Run bootstrap tests and verify RED**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_remote_bootstrap.py
```

Expected: collection fails because `_remote_bootstrap` does not exist.

- [ ] **Step 3: Implement exact cache validation and atomic creation**

Create a package module that imports only stdlib modules and can run when its source is sent to a remote `python -c`. Define:

```python
PROTOCOL_VERSION = 1
PACKAGE_NAME = "agent-bridge"


def ensure_runtime(
    version: str,
    *,
    cache_root: Path | None = None,
    venv_creator: Callable[[Path], None] = _create_venv,
    command_runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] = _run,
) -> Path:
    checked_version = _validate_version(version)
    root = _prepare_private_cache_root(
        cache_root or Path.home() / ".cache" / "agent-bridge" / "runtime"
    )
    final = root / checked_version
    with _version_lock(root, checked_version):
        cached_python = final / "venv" / "bin" / "python"
        if _reports_exact_version(cached_python, checked_version, command_runner):
            return cached_python
        _remove_private_tree(final, root=root)
        temporary = Path(tempfile.mkdtemp(prefix=f".{checked_version}-", dir=root))
        try:
            venv_path = temporary / "venv"
            venv_creator(venv_path)
            python = venv_path / "bin" / "python"
            _require_success(command_runner((
                str(python), "-m", "pip", "install",
                "--disable-pip-version-check", "--no-input",
                f"{PACKAGE_NAME}=={checked_version}",
            )), "remote Agent Bridge installation failed")
            if not _reports_exact_version(python, checked_version, command_runner):
                raise RuntimeError("remote Agent Bridge version does not match")
            temporary.rename(final)
            return final / "venv" / "bin" / "python"
        except BaseException:
            _remove_private_tree(temporary, root=root)
            raise
```

`_create_venv()` must call `venv.EnvBuilder(with_pip=True).create(path)`. `_reports_exact_version()` must run the candidate interpreter with a fixed `importlib.metadata.version("agent-bridge")` program, a timeout, bounded output, no shell, and exact string comparison. Create the runtime root and lock files owner-only; reject symlinks or components not owned by the current effective user before removal or use. Serialize same-version creation with `fcntl.flock` and keep the final version directory immutable after a successful rename.

- [ ] **Step 4: Add bounded remote port selection and JSON protocol**

Implement:

```python
def select_remote_port(
    requested: int,
    *,
    socket_factory: Callable[..., socket.socket] = socket.socket,
    candidate_factory: Callable[[], int] = _random_high_port,
) -> int:
    if not isinstance(requested, int) or isinstance(requested, bool):
        raise ValueError("remote port must be an integer")
    if not 0 <= requested <= 65535:
        raise ValueError("remote port must be between 0 and 65535")
    if requested:
        return requested
    for _ in range(32):
        candidate = candidate_factory()
        if not 49152 <= candidate <= 65535:
            raise RuntimeError("remote port candidate is outside the high range")
        try:
            with socket_factory(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", candidate))
        except OSError:
            continue
        return candidate
    raise RuntimeError("could not select an available remote high port")


def bootstrap_main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
) -> int:
    version, raw_port = tuple(sys.argv[1:] if argv is None else argv)
    _require_python_311()
    python = ensure_runtime(version)
    record = {
        "protocol": PROTOCOL_VERSION,
        "python": str(python),
        "remote_port": select_remote_port(int(raw_port)),
        "version": version,
    }
    stdout.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    stdout.flush()
    return 0
```

Define `_random_high_port()` as `49152 + secrets.randbelow(16384)`. Use a normal `if __name__ == "__main__":` exit. Emit failures on stderr with a nonzero exit and no JSON success record.

- [ ] **Step 5: Run focused bootstrap tests and compile the standalone source**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_remote_bootstrap.py
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m py_compile \
  src/agent_bridge/_remote_bootstrap.py
```

Expected: all cache/protocol/security tests pass, and the module compiles without importing Agent Bridge internals.

- [ ] **Step 6: Review the explicit Task 2 diff without committing**

```bash
git diff --check -- src/agent_bridge/_remote_bootstrap.py \
  tests/agent_bridge/test_remote_bootstrap.py
git diff -- src/agent_bridge/_remote_bootstrap.py \
  tests/agent_bridge/test_remote_bootstrap.py
```

Expected: the bootstrap module is stdlib-only, package/version strings are fixed, subprocesses use argv without `shell=True`, and cache deletion cannot escape the validated runtime root.

---

### Task 3: OpenSSH protocol, exact startup validation, and readiness gate

**Files:**
- Modify: `src/agent_bridge/ssh.py`
- Modify: `tests/agent_bridge/test_ssh.py`

**Interfaces:**
- Consumes: `SSHSettings` from Task 1 and `_remote_bootstrap.py` source/protocol from Task 2.
- Produces: `BootstrapRecord`, `run_remote_bootstrap()`, `build_tunnel_argv()`, `parse_remote_startup()`, `localized_startup()`, and `wait_for_readiness()`.

- [ ] **Step 1: Write failing command-construction and bootstrap-protocol tests**

Add tests with paths containing spaces, quotes, `$()`, semicolons, and leading dashes inside repository values. Require all values to remain literal remote argv after `shlex.split(remote_command)`, never executable shell fragments. Pin the local OpenSSH argv:

```python
assert build_bootstrap_argv(settings, bootstrap_source="print('bootstrap')") == (
    str(settings.ssh_executable),
    "-T",
    "--",
    "workbox",
    shlex.join((
        "python3", "-c", "print('bootstrap')", "0.1.0", "0",
    )),
)
```

Use a fake bounded subprocess result for `run_remote_bootstrap()` and assert exact decoding into:

```python
BootstrapRecord(
    protocol=1,
    python="/home/test/.cache/agent-bridge/runtime/0.1.0/venv/bin/python",
    remote_port=53123,
    version="0.1.0",
)
```

Reject missing/extra JSON fields, multiple records, invalid UTF-8, over-limit output, timeout, nonzero SSH exit, wrong protocol/version, nonabsolute runtime Python, out-of-range port, and diagnostics containing a URL access key.

- [ ] **Step 2: Write failing tunnel/startup/readiness tests**

Pin the tunnel arguments and remote command:

```python
argv = build_tunnel_argv(settings, bootstrap, local_port=43123)
assert argv[:8] == (
    str(settings.ssh_executable), "-T",
    "-o", "ExitOnForwardFailure=yes",
    "-L", "127.0.0.1:43123:127.0.0.1:53123",
    "--", "workbox",
)
remote = shlex.split(argv[8])
assert remote == [
    "exec",
    bootstrap.python,
    "-m", "agent_bridge",
    "--repo", "/srv/demo",
    "--host", "127.0.0.1",
    "--port", "53123",
]
```

Because `exec` is a shell builtin, construct the final string as `"exec " + shlex.join(runtime_argv)` and assert the test parses the tail after removing the literal prefix.

For `parse_remote_startup()`, use the existing nine-field direct startup record and require the remote URL scheme/host/port to be exactly `http`, `127.0.0.1`, and the selected remote port with one nonempty bounded `key` query value. `localized_startup()` must return a new mapping with `port=43123` and `url=http://127.0.0.1:43123/?key=<same-key>` while preserving status, versions, repository, and branch.

Test `wait_for_readiness()` with an injected opener sequence of connection refusal, timeout, then HTTP 200. Assert HTTP 403 fails immediately as keyed-auth rejection, other non-200 responses fail, a process exit aborts polling, and deadline exhaustion reports one concise error.

- [ ] **Step 3: Run the focused protocol tests and verify RED**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_ssh.py \
  -k 'bootstrap or tunnel or startup or readiness or quoting'
```

Expected: FAIL because Task 1 contains only parsing/dispatch.

- [ ] **Step 4: Implement bounded bootstrap invocation and validation**

In `ssh.py`, add:

```python
@dataclass(frozen=True, slots=True)
class BootstrapRecord:
    protocol: int
    python: str
    remote_port: int
    version: str


def build_bootstrap_argv(
    settings: SSHSettings,
    *,
    bootstrap_source: str,
) -> tuple[str, ...]:
    remote = shlex.join((
        settings.python_command,
        "-c",
        bootstrap_source,
        settings.version,
        str(settings.remote_port),
    ))
    return (str(settings.ssh_executable), "-T", "--", settings.destination, remote)
```

Read `_remote_bootstrap.py` from the installed package path, bound it to 128 KiB, and reject a nonregular/symlinked source. `run_remote_bootstrap()` must use a no-shell SSH child, a 180-second deadline, and a selector-based stdout/stderr collector capped at 64 KiB per stream. Terminate, then kill after one second, on timeout/overflow. Require exit zero and exactly one JSON line; retain only a bounded redacted diagnostic tail on failure.

Validate the four exact JSON keys and construct `BootstrapRecord`. Require protocol `1`, exact requested version, an absolute POSIX `python`, and a port from 1 through 65535.

- [ ] **Step 5: Implement tunnel argv, startup localization, and keyed HTTP readiness**

Add these signatures:

```python
MAX_STARTUP_BYTES = 64 * 1024
STARTUP_FIELDS = frozenset({
    "port", "url",
    "fable_status", "fable_version",
    "sol_status", "sol_version",
    "ssh_command", "repository", "branch",
})


def build_tunnel_argv(
    settings: SSHSettings,
    bootstrap: BootstrapRecord,
    *,
    local_port: int,
) -> tuple[str, ...]:
    runtime_argv = (
        bootstrap.python,
        "-m", "agent_bridge",
        *settings.remote_arguments,
        "--host", "127.0.0.1",
        "--port", str(bootstrap.remote_port),
    )
    remote_command = "exec " + shlex.join(runtime_argv)
    forward = (
        f"127.0.0.1:{local_port}:"
        f"127.0.0.1:{bootstrap.remote_port}"
    )
    return (
        str(settings.ssh_executable),
        "-T",
        "-o", "ExitOnForwardFailure=yes",
        "-L", forward,
        "--", settings.destination,
        remote_command,
    )


def parse_remote_startup(
    line: bytes,
    *,
    expected_remote_port: int,
) -> Mapping[str, object]:
    if len(line) > MAX_STARTUP_BYTES or not line.endswith(b"\n"):
        raise SSHLaunchError("remote startup record is invalid")
    try:
        record = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SSHLaunchError("remote startup record is invalid") from error
    if not isinstance(record, dict) or set(record) != STARTUP_FIELDS:
        raise SSHLaunchError("remote startup record has an invalid shape")
    _validate_remote_startup_fields(record, expected_remote_port)
    return record


def _startup_key(raw_url: object) -> str:
    if not isinstance(raw_url, str) or len(raw_url) > 4096:
        raise SSHLaunchError("remote startup URL is invalid")
    parsed = urllib.parse.urlsplit(raw_url)
    query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
    values = query.get("key", ())
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.path != "/"
        or parsed.fragment
        or set(query) != {"key"}
        or len(values) != 1
        or re.fullmatch(r"[A-Za-z0-9_-]{16,128}", values[0]) is None
    ):
        raise SSHLaunchError("remote startup URL is invalid")
    return values[0]


def localized_startup(
    remote: Mapping[str, object],
    *,
    local_port: int,
) -> dict[str, object]:
    key = _startup_key(remote["url"])
    localized = dict(remote)
    localized["port"] = local_port
    localized["url"] = f"http://127.0.0.1:{local_port}/?key={key}"
    return localized


def wait_for_readiness(
    url: str,
    *,
    process: subprocess.Popen[bytes],
    opener: Callable[..., object] = urllib.request.urlopen,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    timeout_seconds: float = 30.0,
) -> None:
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        if process.poll() is not None:
            raise SSHLaunchError("SSH exited before Agent Bridge was ready")
        try:
            with opener(url, timeout=1.0) as response:
                if response.status == 200:
                    return
                raise SSHLaunchError("forwarded Agent Bridge returned an invalid status")
        except urllib.error.HTTPError as error:
            if error.code == 403:
                raise SSHLaunchError("forwarded Agent Bridge rejected its access key") from error
            raise SSHLaunchError("forwarded Agent Bridge returned an invalid status") from error
        except (TimeoutError, urllib.error.URLError, ConnectionError):
            sleeper(0.1)
    raise SSHLaunchError("Agent Bridge did not become ready before the deadline")
```

Build OpenSSH with `-T`, `-o ExitOnForwardFailure=yes`, and exactly one IPv4-loopback `-L`. Append the destination after `--`. Build the remote runtime argv from the already-validated fixed executable, `-m agent_bridge`, Task 1's ordered remote arguments, `--host 127.0.0.1`, and the selected remote port; prepend only the literal shell builtin `exec ` and quote every argv item with `shlex.join()`.

Bound the startup line to 64 KiB and exact UTF-8. Validate its known field types plus the exact keyed loopback URL. Do not print or log the raw remote record. Readiness sends a GET to the localized keyed URL with a one-second per-attempt timeout, accepts only HTTP 200, sleeps at most 100 ms between connection failures, and checks `process.poll()` on every iteration.

`_validate_remote_startup_fields()` must require `record["port"] == expected_remote_port`; bounded strings for status/repository/branch; string-or-null versions and SSH hint; and `_startup_key(record["url"])`. It must also require the parsed URL's explicit port to equal `expected_remote_port` and reject userinfo, a second query value, or any extra field.

- [ ] **Step 6: Run Task 3 and cumulative SSH tests**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_remote_bootstrap.py tests/agent_bridge/test_ssh.py
```

Expected: all bootstrap, quoting, protocol, startup, and readiness tests pass.

- [ ] **Step 7: Review the explicit Task 3 diff without committing**

```bash
git diff --check -- src/agent_bridge/ssh.py tests/agent_bridge/test_ssh.py
git diff -- src/agent_bridge/ssh.py tests/agent_bridge/test_ssh.py
```

Expected: no `shell=True`, no user input concatenated outside `shlex.join`, no public bind, no raw access key in diagnostics, and every wait/output path is bounded.

---

### Task 4: Foreground session ownership, collision retry, browser opening, and cleanup

**Files:**
- Modify: `src/agent_bridge/ssh.py`
- Modify: `src/agent_bridge/__main__.py:148-198`
- Modify: `src/agent_bridge/__main__.py:1113-1124`
- Modify: `tests/agent_bridge/test_ssh.py`
- Modify: `tests/agent_bridge/test_main.py:200-246`

**Interfaces:**
- Consumes: all Task 1-3 interfaces.
- Produces: `run_ssh()` and the completed `ssh.main()` command path with one attached OpenSSH child and deterministic cleanup.

- [ ] **Step 1: Write failing happy-path and browser-gate tests**

Use a fake OpenSSH executable for bootstrap command logging plus an injected tunnel `Popen` factory whose stdout contains the exact remote startup line. Inject readiness and browser seams; assert the observable order:

```python
assert events == [
    "bootstrap",
    ("tunnel", "127.0.0.1:43123:127.0.0.1:53123"),
    ("ready", "http://127.0.0.1:43123/?key=session-key"),
    ("output", "http://127.0.0.1:43123/?key=session-key"),
    ("browser", "http://127.0.0.1:43123/?key=session-key"),
    "wait",
]
```

Assert `--no-open` omits only the browser call, still prints the URL once, and remains attached. If `webbrowser.open()` returns false, assert a bounded warning is printed while the SSH session remains active.

Extend `tests/agent_bridge/test_main.py` at this point so the old direct help and launch tests remain unchanged, root help contains the exact remote example, `agent-bridge ssh --help` contains `--local-port`, `--remote-port`, `--python`, and `--no-open`, and only a leading literal `ssh` dispatches:

```python
def test_main_dispatches_only_leading_ssh_without_parsing_local_repo(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "agent_bridge.ssh.main",
        lambda argv, **kwargs: calls.append((tuple(argv), kwargs)) or 17,
    )
    assert launcher.main(["ssh", "workbox", "--repo", "/srv/demo"]) == 17
    assert calls[0][0] == ("workbox", "--repo", "/srv/demo")
```

Add the regression matrix for the user's real `--repo` failure:

```python
@pytest.mark.parametrize(
    ("basename", "expected_label"),
    (
        ("agent-bridge-demo.mgF8bo", "agent-bridge-demo-mgf8bo"),
        ("123-demo", "repo-123-demo"),
        ("a" * 80, "a" * 32),
    ),
)
def test_repo_shorthand_always_derives_a_valid_project_label(
    tmp_path: Path,
    basename: str,
    expected_label: str,
) -> None:
    repo = _named_repo(tmp_path, basename)
    tools = _fake_tools(tmp_path)
    settings = parse_settings(_args(repo, tools), environ=_environment(tmp_path, repo))
    assert settings.projects[0].label == expected_label
    assert settings.projects[0].repo_root == repo
```

- [ ] **Step 2: Write failing retry, early-exit, interrupt, and cleanup tests**

Cover:

- an auto-selected local forward collision followed by a new local port;
- an auto-selected remote listener collision followed by a new bootstrap-selected remote port;
- no retry for a user-fixed colliding local or remote port;
- at most five automatic attempts;
- SSH exit before startup JSON and after startup but before readiness;
- SSH authentication and host-key rejection before startup;
- remote Python/venv bootstrap failure, missing remote Git/Claude/Codex, and invalid remote repository diagnostics;
- malformed/overlong startup output;
- stderr diagnostics bounded and access-key-redacted;
- `KeyboardInterrupt` after readiness calls `terminate()`, waits one second, then `kill()` only if still alive;
- successful remote exit returns zero without kill;
- nonzero remote exit raises a concise launch error; and
- every failed attempt is reaped before retry.

Use fake process objects that record `poll`, `terminate`, `kill`, and `wait`; no background process or socket may survive a test.

- [ ] **Step 3: Run lifecycle tests and verify RED**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_ssh.py \
  -k 'foreground or browser or collision or interrupt or cleanup or early_exit'
```

Expected: FAIL because `run_ssh()` and completed `ssh.main()` do not exist.

- [ ] **Step 4: Implement one-attempt ownership and bounded stream handling**

Define the public launch error and one private retry signal:

```python
class SSHLaunchError(RuntimeError):
    """A bounded actionable failure from the SSH connection workflow."""


class _RetryablePortCollision(SSHLaunchError):
    """An automatically selected listener or forward collided before readiness."""

    def __init__(self, message: str, *, endpoint: str) -> None:
        if endpoint not in {"local", "remote"}:
            raise ValueError("collision endpoint must be local or remote")
        super().__init__(message)
        self.endpoint = endpoint
```

Implement `_run_tunnel_attempt()` so it:

1. computes `tunnel_argv = build_tunnel_argv(settings, bootstrap, local_port=local_port)` and starts `subprocess.Popen(tunnel_argv, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)`;
2. reads one bounded stdout startup record while concurrently draining a bounded stderr diagnostic tail;
3. validates/localizes the record and waits for HTTP readiness;
4. prints only concise status lines plus exactly one actionable localized keyed URL;
5. calls the browser seam only after readiness;
6. continues draining/redacting remote stdout/stderr while synchronously waiting for the foreground SSH child; and
7. always reaps the child in `finally`.

Use selectors rather than detached reader jobs. Redaction replaces the exact access key and any `?key=` query value in diagnostics with `<redacted>`. A valid URL is printed only through the dedicated ready-output function.

Implement `_stop_ssh_process()` as:

```python
def _stop_ssh_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        process.wait()
        return
    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
```

Do not start a remote daemon, tmux session, local detached child, or shell-owned background process.

- [ ] **Step 5: Implement top-level retry and user-facing errors**

Add:

```python
def run_ssh(
    settings: SSHSettings,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    browser_open: Callable[[str], bool] = webbrowser.open,
    bootstrap_runner: Callable[[SSHSettings], BootstrapRecord] = run_remote_bootstrap,
    popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    readiness_opener: Callable[..., object] = urllib.request.urlopen,
) -> int:
    for attempt in range(5):
        bootstrap = bootstrap_runner(settings)
        local_port = _select_local_port(settings.local_port)
        try:
            return _run_tunnel_attempt(
                settings, bootstrap, local_port=local_port,
                stdout=stdout, stderr=stderr,
                browser_open=browser_open,
                popen_factory=popen_factory,
                readiness_opener=readiness_opener,
            )
        except _RetryablePortCollision as error:
            fixed_collision = (
                (error.endpoint == "local" and settings.local_port != 0)
                or (error.endpoint == "remote" and settings.remote_port != 0)
            )
            if fixed_collision or attempt == 4:
                raise SSHLaunchError(str(error)) from error
    raise SSHLaunchError("SSH launch exhausted its retry limit")
```

Implement `_select_local_port(requested, socket_factory=socket.socket) -> int` with the same validation and IPv4-loopback temporary-bind pattern as Task 2's remote selector. Retry only the endpoint whose bounded SSH/remote diagnostic identifies a listener/forward collision, and only when that endpoint was automatic; preserve a user-fixed port on the other side. Re-running bootstrap is a cache hit and obtains a fresh remote port only when `--remote-port 0` was selected.

Complete `ssh.main()` by parsing settings, calling `run_ssh()`, converting expected validation/bootstrap/launch exceptions into one `agent-bridge ssh: <actionable message>` stderr line and exit code `2`, and allowing `KeyboardInterrupt` to perform cleanup then return `130`. Do not catch `BaseException` around programming errors.

At the start of `agent_bridge.__main__.main()`, normalize the argument source once and dispatch before `parse_settings()`:

```python
raw_arguments = tuple(sys.argv[1:] if argv is None else argv)
if raw_arguments[:1] == ("ssh",):
    from agent_bridge.ssh import main as ssh_main
    if uvicorn_run is not None:
        raise ValueError("uvicorn_run is available only for direct launch")
    return ssh_main(raw_arguments[1:], stdout=stdout)
settings = parse_settings(raw_arguments, environ=environment)
```

Add this exact epilog to the existing direct parser without converting it to subparsers or moving local-launch code:

```text
Remote: agent-bridge ssh SSH_DESTINATION --repo /absolute/remote/repository
```

Narrow `_repository_slug()` to the actual label alphabet and bound it without changing canonical root identity:

```python
def _repository_slug(repo_root: Path) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", repo_root.name.lower()).strip("-_")
    if not slug:
        return "repository"
    if not slug[0].isalpha():
        slug = f"repo-{slug}"
    return slug[:32].rstrip("-_") or "repository"
```

Existing basenames that already satisfy the project-label grammar retain the same label and legacy state-directory spelling. Previously failing dotted/numeric/overlong names gain the first valid spelling they can persist.

- [ ] **Step 6: Run cumulative SSH and direct-launch tests**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_remote_bootstrap.py tests/agent_bridge/test_ssh.py \
  tests/agent_bridge/test_main.py
```

Expected: all new session/lifecycle cases and all existing direct-launch behavior pass.

- [ ] **Step 7: Review the explicit Task 4 diff without committing**

```bash
git diff --check -- src/agent_bridge/ssh.py src/agent_bridge/__main__.py \
  tests/agent_bridge/test_ssh.py tests/agent_bridge/test_main.py
git diff -- src/agent_bridge/ssh.py src/agent_bridge/__main__.py \
  tests/agent_bridge/test_ssh.py tests/agent_bridge/test_main.py
```

Expected: one foreground SSH owner, no leaked process on any branch, no key outside the ready URL, no public listener, and no unbounded retry/read/output path.

---

### Task 5: Copy-pasteable documentation, full fake verification, and independent review

**Files:**
- Modify: `README.md:23-30`
- Modify: `README.md:87-100`
- Modify: `README.md:262-284`
- Modify: `tests/agent_bridge/test_packaging_metadata.py:36-45`
- Verify: `src/agent_bridge/_remote_bootstrap.py`
- Verify: `src/agent_bridge/ssh.py`
- Verify: `src/agent_bridge/__main__.py`
- Verify: `tests/agent_bridge/test_remote_bootstrap.py`
- Verify: `tests/agent_bridge/test_ssh.py`
- Verify: `tests/agent_bridge/test_main.py`
- Verify: `docs/superpowers/specs/2026-08-16-ssh-remote-bootstrap-design.md`
- Verify: `docs/superpowers/plans/2026-08-16-ssh-remote-bootstrap.md`

**Interfaces:**
- Consumes: completed Tasks 1-4.
- Produces: a documented, fake-verified, independently reviewed SSH workflow ready for a release and later live user test.

- [ ] **Step 1: Write failing README contract assertions**

Update `test_readme_is_generic_and_copy_pasteable()` to require:

```python
for text in (
    "agent-bridge ssh YOUR_SSH_ALIAS --repo /absolute/remote/repository",
    "~/.cache/agent-bridge/runtime/<version>",
    "Python 3.11",
    "Claude and Codex",
    "--no-open",
    "127.0.0.1",
    "source-checkout-only",
):
    assert text in readme
```

Retain the manual `ssh -N -L` assertion as an explicitly documented advanced fallback, not the primary workflow.

- [ ] **Step 2: Run the README test and verify RED**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_packaging_metadata.py -k readme
```

Expected: FAIL because README still presents only the manual two-terminal tunnel.

- [ ] **Step 3: Document the exact one-command workflow and its limits**

Replace the SSH section's primary instructions with:

```bash
agent-bridge ssh YOUR_SSH_ALIAS --repo /absolute/remote/repository
```

Explain that this command runs on the local machine, uses `~/.ssh/config`, auto-opens a local keyed URL, stays attached, and stops with Ctrl+C. State that Agent Bridge is cached remotely under `~/.cache/agent-bridge/runtime/<version>` without root/global installation, while the remote host must already have Python 3.11+ with `venv`, Git, and authenticated Claude and Codex CLIs. State that first bootstrap needs remote package-index access and only a published exact Agent Bridge version is accepted; source-checkout-only/unpublished builds fail clearly. Include `--no-open`, multi-project syntax, executable overrides as remote paths, cache removal, and the existing manual launch+tunnel as an advanced fallback.

Add troubleshooting entries for SSH authentication/host keys, remote Python/venv, unpublished version, missing remote Claude/Codex, remote repository validation, local/remote port exhaustion, and keyed readiness failure. Never recommend disabling host-key checking or public binding.

- [ ] **Step 4: Run the complete focused lane**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_remote_bootstrap.py \
  tests/agent_bridge/test_ssh.py \
  tests/agent_bridge/test_main.py \
  tests/agent_bridge/test_packaging_metadata.py
```

Expected: all tests pass with fake-only execution.

- [ ] **Step 5: Run the complete Agent Bridge suite**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge
```

Expected: 100% pass. Diagnose any failure before changing code; rerun timeout-prone failures serially before treating them as regressions.

- [ ] **Step 6: Run static, packaging, and scope verification**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m py_compile \
  src/agent_bridge/_remote_bootstrap.py src/agent_bridge/ssh.py \
  src/agent_bridge/__main__.py \
  tests/agent_bridge/test_remote_bootstrap.py tests/agent_bridge/test_ssh.py \
  tests/agent_bridge/test_main.py tests/agent_bridge/test_packaging_metadata.py
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -c \
  'import sys; import agent_bridge.__main__; assert "fastapi" not in sys.modules; assert "uvicorn" not in sys.modules'
git diff --check
git status --short
```

Expected: compilation/import/diff checks pass. Only the approved SSH spec, plan, README, launcher/bootstrap modules, and focused tests are modified in the isolated worktree.

- [ ] **Step 7: Request independent Sol review**

Request a read-only `gpt-5.6-sol` review of the complete SSH diff. Require explicit Critical/Important findings for destination/argument injection, cache ownership/symlink/deletion safety, exact-version authority, subprocess deadlines/output bounds, loopback forwarding, startup/key validation, browser timing, port retry, Ctrl+C/early-exit cleanup, direct-launch compatibility, fake-only test adequacy, and README honesty.

- [ ] **Step 8: Address confirmed findings test-first and rerun every gate**

For every confirmed finding, add an honest focused RED, make the smallest in-scope fix, rerun the focused lane, full suite, static checks, and fresh independent review until the reviewer reports READY with zero Critical/Important findings.

- [ ] **Step 9: Hand off without committing or live SSH execution**

Report changed files, focused/full test counts, review verdict, known limitations, and the release-only bootstrap requirement. State that no live SSH/provider/network test, commit, push, remote mutation, or existing UI-worktree mutation occurred. Ask the user separately before committing/publishing or performing a live SSH acceptance test.
