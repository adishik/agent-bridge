# Multi-Project Chat Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an immutable multi-project startup registry, isolated per-project runtimes, persistent chats, a hub-wide active-agent lease, and project-aware HTTP/WebSocket boundaries without changing the existing planning/execution semantics.

**Architecture:** Introduce pure project-configuration types, one small hub settings store, and a `ProjectRegistry` that owns independent existing runtimes. Extend the current SQLite store additively so sessions are chats with bounded titles and recency. Route every browser operation by opaque project ID before querying a project store. Keep `create_app` as a one-project compatibility wrapper while the launcher assembles `create_hub_app` from all validated projects.

**Tech Stack:** Python 3.11+, argparse, pathlib, hashlib, SQLite, FastAPI/Starlette, browser-native ES modules, pytest, Node test harnesses.

## Global constraints

- Work only in the standalone `agent-bridge` repository and an isolated feature worktree.
- Do not access or mention any private source checkout, identifier, path, state, or history.
- Tests use temporary Git repositories, fake executables, and local clients only. Do not invoke live agents, provider auth, network services, or a browser server.
- Preserve `--repo` as the compatible single-project shortcut. Browser requests never introduce filesystem paths.
- Every project owns a separate store, tracker, runner, adapters, coordinator, broadcaster, artifacts directory, and lock.
- Resolve and validate every chosen state directory, then acquire the hub lock and all project locks in project-ID order before constructing, migrating, auditing, or recovering any SQLite database. Partial startup releases all resources.
- Use additive, transactional, idempotent SQLite migrations. Do not rewrite existing event/task/run/baseline bytes.
- Keep one hub-wide active-agent lease in V1. The server, not only the browser, enforces it.
- Preserve keyed-session, CSRF, subscription, usage-credit, Sol-readiness, exact-revision, baseline, and repository-delta checks.
- Use red-green-refactor. Do not weaken, skip, or xfail an honest failing test.
- Terra implements. Sol reviews every code, test, documentation, and fix change before acceptance.
- Stage explicit paths only; never use `git add -A`.
- Task reports under `.superpowers/sdd/` are intentionally ignored scratch evidence; verify them with `git check-ignore` and require a clean tracked worktree rather than staging them.

---

### Task 1: Define immutable project configuration and collision-safe state identity

**Files:**
- Create: `src/agent_bridge/projects.py`
- Create: `tests/agent_bridge/test_projects.py`

**Interfaces:**

```python
MAX_PROJECTS = 32

@dataclass(frozen=True, slots=True)
class ProjectSpec:
    project_id: str
    label: str
    repo_root: Path
    branch: str
    state_dir: Path

def project_id_for_root(repo_root: Path) -> str: ...
def parse_project_argument(value: str) -> tuple[str, Path]: ...
def build_project_specs(
    entries: Sequence[tuple[str, Path]],
    *,
    state_root: Path,
    git_executable: Path,
    probe_timeout_seconds: float = 10.0,
) -> tuple[ProjectSpec, ...]: ...
```

`project_id_for_root` is `sha256(os.fsencode(str(repo_root)))[:32]` after exact canonical-root validation. The configured label is never part of state identity.

- [ ] **Step 1: Write failing grammar and identity tests**

Add tests for `alpha=/absolute/root`, ASCII labels matching `[A-Za-z][A-Za-z0-9_-]{0,31}`, rejection of missing `=`, empty labels, relative roots, control characters, case-insensitive duplicate labels, duplicate canonical roots through symlink aliases, more than `MAX_PROJECTS`, non-Git paths, unreadable roots, and Git probe timeouts. Add two same-basename repositories under different parents and assert distinct project IDs/state directories. Rebuild the same root with a different label and assert identical `project_id` and `state_dir`.

- [ ] **Step 2: Run the focused RED**

```bash
python -m pytest -q tests/agent_bridge/test_projects.py
```

Expected: collection fails because `agent_bridge.projects` does not exist.

- [ ] **Step 3: Implement fail-closed project parsing and validation**

Use an injected absolute regular executable for Git, a minimal deterministic Git environment, bounded subprocess output, and a deadline. Resolve each root once, require the probe's `--show-toplevel` result to equal that canonical root, derive the branch with the same sanitized process boundary, reject duplicates before creating directories, sort the returned specs by `project_id`, and create state below `state_root / "projects" / project_id` only after the whole entry set validates.

- [ ] **Step 4: Prove no browser-originated/path-derived authority seam exists**

Add tests that labels such as `../../escape`, roots containing a newline, Git output with multiple records, and hostile inherited `GIT_DIR`, `GIT_WORK_TREE`, `GIT_CONFIG_GLOBAL`, `GIT_CONFIG_SYSTEM`, and provider-secret environment values do not redirect probes or appear in fake-process captures.

- [ ] **Step 5: Run focused tests and commit**

```bash
python -m pytest -q tests/agent_bridge/test_projects.py
git add src/agent_bridge/projects.py tests/agent_bridge/test_projects.py
git diff --cached --check
git commit -m "feat: add immutable project registry inputs"
```

---

### Task 2: Migrate sessions into persistent bounded chats

**Files:**
- Modify: `src/agent_bridge/store.py`
- Modify: `tests/agent_bridge/test_store.py`

**Interfaces:**

```python
MAX_CHAT_TITLE_LENGTH = 80
MAX_CHAT_PAGE_SIZE = 50

@dataclass(frozen=True, slots=True)
class ChatRecord:
    session_id: str
    repo_root: str
    title: str
    created_at: str
    updated_at: str
    latest_sequence: int

@dataclass(frozen=True, slots=True)
class ChatCursor:
    latest_sequence: int
    session_id: str

def create_chat(self, repo_root: str, *, session_id: str | None = None) -> ChatRecord: ...
def list_chats(self, *, before: ChatCursor | None = None, limit: int = 50) -> tuple[ChatRecord, ...]: ...
def chat(self, session_id: str) -> ChatRecord | None: ...
def audit_legacy_project_ownership(self, canonical_repo_root: str) -> None: ...
```

`create_session` remains as a compatibility wrapper. First-message title derivation collapses whitespace, uses Unicode text safely, truncates deterministically to 80 code points, and never invokes a model.

- [ ] **Step 1: Add migration and rollback RED tests**

Create a current-schema fixture with sessions, task revisions, events, runs, baseline settings, and an active-session setting. Reopen it with the new store and assert the exact existing rows/bytes remain, `title == "New chat"`, `updated_at` is conservatively derived, the valid active session becomes the initially selected historical chat, and repeated migration is byte-equivalent. Inject failure after the first DDL/data update and prove the entire migration rolls back.

- [ ] **Step 2: Add chat creation/title/recency/page RED tests**

Assert cryptographically generated IDs are unique and project-bound, empty chats are `New chat`, only the first user `message` derives a title, agent/system messages do not, later messages do not rename it, and lists order by latest event sequence then stable session ID. Assert `limit=0`, `limit=51`, negative sequence cursors, malformed session cursors, and partial cursors reject before SQL. Page through more than 50 chats, including more than one page of empty chats tied at sequence zero, without duplicates or omissions.

- [ ] **Step 3: Add exhaustive legacy ownership RED tests**

Parameterize mixed session roots, missing roots, invalid active-session references, orphan tasks/events/runs, a run pointing at a nonexistent exact revision, malformed baseline setting keys/payloads, baseline task/revision mismatches, and disabled foreign keys with broken rows. Assert the audit is one read transaction and returns no partial adoption signal.

- [ ] **Step 4: Run the focused RED**

```bash
python -m pytest -q \
  tests/agent_bridge/test_store.py -k "chat or migration or legacy_project"
```

Expected: failures for missing columns, DTOs, methods, pagination, and ownership audit.

- [ ] **Step 5: Implement additive migration and bounded queries**

Add `title TEXT NOT NULL DEFAULT 'New chat'` and `updated_at TEXT` to `sessions`, backfill with existing `created_at`, add a recency index driven by session-scoped event sequence, and update session recency/title in the same transaction that commits the first user message. Do not derive ordering from a non-monotonic wall clock. Keep store listener publication after commit.

- [ ] **Step 6: Implement the complete legacy audit**

Validate every session root and active-session pointer, `PRAGMA foreign_key_check`, exact task/event/run joins, latest revisions, and every baseline manifest's task/revision/root identity. Aggregate bounded generic reasons, raise one non-leaking error, and make no mutation during audit.

- [ ] **Step 7: Run store tests and commit**

```bash
python -m pytest -q tests/agent_bridge/test_store.py
git add src/agent_bridge/store.py tests/agent_bridge/test_store.py
git diff --cached --check
git commit -m "feat: persist project chat history"
```

---

### Task 3: Add the hub-only settings store

**Files:**
- Create: `src/agent_bridge/hub_store.py`
- Create: `tests/agent_bridge/test_hub_store.py`

**Interfaces:**

```python
class HubStore:
    def __init__(self, path: Path, *, clock: Callable[[], datetime]) -> None: ...
    def usage_credits_acknowledged(self) -> bool: ...
    def acknowledge_usage_credits(self) -> None: ...
    def close(self) -> None: ...
```

- [ ] **Step 1: Write failing ownership and durability tests**

Assert the hub schema contains only schema metadata and account-level settings, never sessions/tasks/events/runs/baselines/repository paths. Test default false, durable true, idempotent acknowledgement, transactional rollback, malformed stored values failing closed, injected clock use, 0600 file mode, and close idempotence.

- [ ] **Step 2: Run RED, implement, and run GREEN**

```bash
python -m pytest -q tests/agent_bridge/test_hub_store.py
```

Implement with the same SQLite safety conventions as `Store`, but do not generalize or move project data into this database.

- [ ] **Step 3: Commit**

```bash
git add src/agent_bridge/hub_store.py tests/agent_bridge/test_hub_store.py
git diff --cached --check
git commit -m "feat: persist hub account settings"
```

---

### Task 4: Own isolated runtimes and one generation-safe active lease

**Files:**
- Create: `src/agent_bridge/hub.py`
- Create: `tests/agent_bridge/test_hub.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    fable_ready: bool
    fable_status: str
    sol_status: str

class RuntimeReadiness:
    def __init__(
        self, *, initial: RuntimeStatus,
        fable_probe: Callable[[], Awaitable[tuple[bool, str]]],
        sol_probe: Callable[[], Awaitable[str]],
        timeout_seconds: float = 10.0,
    ) -> None: ...
    def snapshot(self) -> RuntimeStatus: ...
    async def require_model_start_ready(
        self, *, usage_credits_acknowledged: bool,
    ) -> RuntimeStatus: ...
    def invalidate_fable_subscription(self) -> None: ...

@dataclass(slots=True)
class InstanceLock:
    path: Path
    descriptor: int
    released: bool = False

    def release(self) -> None: ...

class AppProjectRuntime(Protocol):
    project_id: str
    label: str
    repository: str
    branch: str
    store: SQLiteStore
    coordinator: Coordinator
    broadcaster: EventBroadcaster
    readiness: RuntimeReadiness

@dataclass(slots=True)
class OwnedProjectRuntime:
    spec: ProjectSpec
    store: SQLiteStore
    tracker: RepositoryTracker
    runner: ProcessRunner
    fable: FableAdapter
    sol: SolAdapter
    coordinator: Coordinator
    broadcaster: EventBroadcaster
    readiness: RuntimeReadiness
    lock: InstanceLock

    @property
    def project_id(self) -> str: ...
    @property
    def label(self) -> str: ...
    @property
    def repository(self) -> str: ...
    @property
    def branch(self) -> str: ...
    def close(self) -> None: ...

@dataclass(frozen=True, slots=True)
class LeaseToken:
    generation: int
    project_id: str
    session_id: str
    task_id: str

class ActiveAgentLease:
    def acquire_new(
        self, *, project_id: str, session_id: str, ids: IdFactory,
    ) -> LeaseToken: ...
    def acquire(self, *, project_id: str, session_id: str, task_id: str) -> LeaseToken: ...
    def release(self, token: LeaseToken) -> None: ...
    def snapshot(self) -> LeaseToken | None: ...

class ProjectRegistry:
    def runtime(self, project_id: str) -> AppProjectRuntime: ...
    def projects(self) -> tuple[AppProjectRuntime, ...]: ...
    def close(self) -> None: ...

@dataclass(frozen=True, slots=True)
class PreparedWorkflow:
    token: LeaseToken
    project_id: str
    session_id: str
    task_id: str
    revision: int
    action: str

class HubWorkflowOrchestrator:
    async def prepare_new_request(
        self, *, project_id: str, session_id: str,
        text: str, ids: IdFactory,
    ) -> PreparedWorkflow: ...
    async def prepare_existing_task(
        self, *, project_id: str, session_id: str,
        task_id: str, revision: int, action: str,
    ) -> PreparedWorkflow: ...
    async def run(self, prepared: PreparedWorkflow) -> None: ...
    def abort_prepared(self, prepared: PreparedWorkflow, *, reason: str) -> None: ...
    async def stop(self, *, project_id: str, session_id: str, task_id: str) -> None: ...

# Coordinator preparation methods called only by HubWorkflowOrchestrator.
def prepare_user_request(self, session_id: str, text: str, task_id: str) -> TaskRecord: ...
async def run_prepared_request(self, task_id: str) -> None: ...
def abort_prepared_action(
    self, task_id: str, revision: int, action: str, reason: str,
) -> TaskRecord: ...
```

- [ ] **Step 1: Add registry isolation RED tests**

Construct two app-facing runtimes with deliberately equal session/task IDs. Assert `runtime(project_a)` never queries project B, unknown IDs raise one generic lookup error, project order is stable, and duplicate IDs are rejected. Build owned runtimes separately and verify idempotent close unregisters listeners, closes coordinators/stores, and releases locks in reverse acquisition order even if one close raises. A non-owning compatibility runtime must never close resources supplied by the caller.

- [ ] **Step 2: Add lease race RED tests**

Use threads and barriers to prove only one of two acquisitions succeeds. Assert `acquire_new` creates the task ID inside the lease critical section, stale tokens cannot release a newer generation, double release is harmless, and the active snapshot contains opaque IDs only. Assert a held lease rejects project switch, chat switch, New Chat, approval, Resume, and an answer that would resume a model; Stop bypasses acquisition only for the exact lease-owning task. Force scheduler installation rejection after preparation and assert `abort_prepared` atomically places the exact task/action into `INTERRUPTED` with a resumable pending action, releases only its token, and is idempotent.

- [ ] **Step 3: Run RED and implement**

```bash
python -m pytest -q tests/agent_bridge/test_hub.py
```

Use an in-process lock plus monotonically increasing generation. `HubWorkflowOrchestrator` is the sole lease owner. Each awaited preparation validates route identity and hub acknowledgment without starting a child, then acquires the generation-safe lease (`acquire_new` creates a task ID inside that critical section), and only while holding it calls the route-selected `RuntimeReadiness.require_model_start_ready`. Fresh Claude/Sol probes therefore count inside the one-process boundary. Probe/gate failure releases the token and performs no coordinator/store preparation. After a green gate, `prepare_new_request` calls `Coordinator.prepare_user_request(session_id, text, task_id)` synchronously while holding the lease, releases on preparation failure, and returns `PreparedWorkflow`; `run` calls `Coordinator.run_prepared_request(task_id)` and releases in `finally`. Coordinators and HTTP handlers never reacquire independently. `abort_prepared` is the only preparation-to-scheduler failure path: it calls `Coordinator.abort_prepared_action` with the exact task/revision/action and fixed reason `scheduler_unavailable`, persists an interrupted/resumable pending action, and then releases the exact token. Raw scheduling exceptions never enter persistence. Do not persist a lease as proof of a live process; startup recovery remains per-project and explicit Resume remains required.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest -q tests/agent_bridge/test_projects.py tests/agent_bridge/test_hub.py
git add src/agent_bridge/hub.py tests/agent_bridge/test_hub.py
git diff --cached --check
git commit -m "feat: isolate project runtimes behind a hub lease"
```

---

### Task 5: Assemble all projects safely in the launcher

**Files:**
- Modify: `src/agent_bridge/__main__.py`
- Modify: `tests/agent_bridge/test_main.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class Settings:
    projects: tuple[ProjectSpec, ...]
    hub_state_dir: Path
    host: str
    port: int
    claude_executable: Path
    codex_executable: Path
    git_executable: Path
    bash_executable: Path
    sh_executable: Path

def acquire_instance_lock(path: Path) -> InstanceLock: ...

def assemble_project_runtime(
    spec: ProjectSpec,
    *,
    lock: InstanceLock,
    settings: Settings,
    environment: Mapping[str, str],
    ids: IdFactory,
    clock: Callable[[], datetime],
) -> OwnedProjectRuntime: ...
```

- [ ] **Step 1: Add CLI and layout RED tests**

Assert `--repo` and repeated `--project` are mutually exclusive, at least one is required, `--repo` produces one deterministic label without affecting the digest, and help documents restart-required immutable allowlisting. Assert new projects use `$XDG_STATE_HOME/agent-bridge/projects/<digest>` and hub settings use `$XDG_STATE_HOME/agent-bridge/hub`.

- [ ] **Step 2: Add all-or-nothing startup RED tests**

With three fakes, instrument safe state-path validation, lock acquisition, `SQLiteStore`/`HubStore` construction, migration, legacy audit, and recovery. Assert every state path validates first, then the hub/project locks are acquired in stable project-ID order, and only then may any database constructor run. Make the second lock fail and prove zero database opens/migrations/audits/recoveries, no Uvicorn call, and every acquired lock released. Validated empty private state directories and their 0600 lock files may remain; “no partial state” specifically means no database, migration, recovery, runtime, schema, or artifact state. Make runtime 2 assembly fail after all locks and runtime 1 construction; prove runtime 1 closes and all locks release.

- [ ] **Step 3: Add legacy adoption RED tests**

For both `--repo` and each repeated `--project`, create a legacy basename directory and assert successful exact-root audited adoption keeps that directory in place. Pre-resolve every configured root's digest/legacy candidate without opening SQLite, reject two configured roots claiming one legacy directory, and acquire the selected legacy lock in normal project-ID order before audit. Parameterize mixed root, orphan, invalid active session, corrupt baseline, and failed foreign-key cases; assert startup aborts before recovery. If both legacy and digest state exist for a root, fail as ambiguous.

- [ ] **Step 4: Run RED and implement launcher assembly**

```bash
python -m pytest -q tests/agent_bridge/test_main.py -k "project or lock or legacy or runtime"
```

Reuse the existing safe executable resolution, instance-lock, preflight, repository-context, and foreground Uvicorn seams. Create one shared immutable configuration object, but instantiate all runtime components separately. Each runtime gets a `RuntimeReadiness` with injected bounded fresh Claude-auth and Sol-version probes, a serialization lock, a startup snapshot for bootstrap display, and invalidation on structured auth failure; do not capture one immutable preflight object for the process lifetime. Move usage acknowledgement to `HubStore`; deliberately do not copy an old project acknowledgement.

- [ ] **Step 5: Run launcher tests and commit**

```bash
python -m pytest -q tests/agent_bridge/test_main.py tests/agent_bridge/test_projects.py tests/agent_bridge/test_hub.py
git add src/agent_bridge/__main__.py tests/agent_bridge/test_main.py
git diff --cached --check
git commit -m "feat: launch an allowlisted project hub"
```

---

### Task 6: Add project-aware HTTP and WebSocket boundaries

**Files:**
- Modify: `src/agent_bridge/app.py`
- Modify: `tests/agent_bridge/test_web.py`

**Interfaces:**

```python
def create_hub_app(
    *,
    registry: ProjectRegistry,
    hub_store: HubStore,
    workflows: HubWorkflowOrchestrator,
    static_dir: str | Path,
    session_key: str,
    csrf_token: str,
) -> FastAPI: ...

def create_app(
    *,
    coordinator: Coordinator,
    store: SQLiteStore,
    static_dir: str | Path,
    session_key: str,
    csrf_token: str,
    broadcaster: EventBroadcaster | None = None,
    bootstrap_status: Callable[[], BootstrapStatus] | None = None,
    readiness_check: Callable[[], Awaitable[BootstrapStatus]] | None = None,
) -> FastAPI:
    """Wrap supplied app-facing resources in one non-owning default runtime."""

# Closure defined inside create_hub_app; captures app.state's tracked task set.
def install_prepared_action(
    *, prepared: object,
    coroutine_factory: Callable[[], Coroutine[object, object, None]],
    abort: Callable[[object, str], None],
) -> bool: ...
```

Add exact routes:

```text
GET  /api/projects
GET  /api/projects/{project_id}/chats?before_sequence=...&before_session_id=...&limit=...
POST /api/projects/{project_id}/chats
GET  /api/projects/{project_id}/chats/{session_id}/bootstrap
POST /api/projects/{project_id}/chats/{session_id}/messages
POST /api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/approve
POST /api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/edit
POST /api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/reject
POST /api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/answer
POST /api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/stop
POST /api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/resume
GET  /ws?project_id=...&session_id=...&after=...
```

- [ ] **Step 1: Add cross-project and duplicate-ID RED tests**

Build two runtime fakes with the same session/task IDs. For every mutation route, send project A with identifiers that exist only in B and assert bounded 404/409, zero B queries, zero writes, zero schedules, zero process calls, and no difference in response based on B existence. Repeat for WebSocket replay and bootstrap.

- [ ] **Step 2: Add chat and lease RED tests**

Test bounded project summaries, project-local chat pagination, random New Chat, first-message title, reopening an old idle chat, and no deletion/reset. Hold the hub lease and assert New Chat, project/chat switch bootstrap, approval, Resume, and an answer that would resume a model fail synchronously under a foreign lease; Stop for the exact lease-owning task remains available without acquiring a second token. Assert the foreign-lease rejection launches zero readiness probes. Race two project preparations and prove only the lease winner may run provider probes. Change the fake Claude auth result between two successful lease acquisitions and prove every new model-starting preparation performs a bounded fresh check, updates the snapshot, and fails closed on timeout/malformed/unavailable results while Sol and hub gates remain required; every probe failure releases the exact token and writes no task/action state. Force lifespan scheduler rejection after a committed new/existing preparation; assert the response identifies the durable interrupted/recoverable action, the exact lease releases, no coroutine warning/task appears, and retry uses the persisted action rather than duplicating the user event.

- [ ] **Step 3: Add compatibility RED tests**

Run the current one-project API test harness unchanged through `create_app`. Implement a private non-owning `CompatibilityProjectRuntime` containing only the supplied coordinator, store, broadcaster, a compatibility `RuntimeReadiness` around the existing bootstrap callback plus an injected fresh-check seam, and fixed opaque default project metadata; it does not fabricate tracker/runner/adapter/lock ownership or inspect coordinator private fields. Assert existing paths map only to this sole app-facing runtime, the wrapper never closes caller-owned resources, and a compatibility app cannot contain more than one runtime.

- [ ] **Step 4: Run RED and implement route-selected lookup**

```bash
python -m pytest -q tests/agent_bridge/test_web.py -k "project or chat or lease or websocket"
```

Each handler must call `registry.runtime(project_id)` first, then query only that runtime and verify chat/task/revision ownership before persistence or scheduling. All model-starting handlers await the sole `HubWorkflowOrchestrator` preparation method; it acquires one exact token before any fresh provider probe, performs the complete readiness gate under that token, and commits task/action preparation before scheduling. Define `install_prepared_action` as a closure inside `create_hub_app` so it captures shutdown state and the exact lifespan-tracked task set. It calls the coroutine factory only when installation can proceed; if installation succeeds, ownership transfers to the tracked task; if shutdown/check/create-task installation fails, it calls the supplied synchronous abort callback, closes any just-created coroutine, creates no unobserved task, and returns `False` so the route emits the safe recoverable projection. New/existing workflows pass `workflows.abort_prepared`. Neither app nor coordinator acquires another lease. Preserve the current combined gate—hub acknowledgement, freshly verified Fable subscription readiness, and Sol readiness—for every model-starting action. Stop alone bypasses readiness and lease acquisition, validates the route against the exact current token/task, and causes its owner to release after finalization. Use the selected runtime's broadcaster and session-filtered bounded replay.

- [ ] **Step 5: Run composed backend tests and commit**

```bash
python -m pytest -q \
  tests/agent_bridge/test_hub_store.py \
  tests/agent_bridge/test_hub.py \
  tests/agent_bridge/test_store.py \
  tests/agent_bridge/test_web.py
git add src/agent_bridge/app.py tests/agent_bridge/test_web.py
git diff --cached --check
git commit -m "feat: scope browser APIs to projects and chats"
```

---

### Task 7: Expose project and chat navigation without changing conversation semantics

**Files:**
- Modify: `src/agent_bridge/static/index.html`
- Modify: `src/agent_bridge/static/app.js`
- Modify: `src/agent_bridge/static/styles.css`
- Modify: `tests/agent_bridge/test_static_ui.py`
- Modify: `tests/agent_bridge/test_web.py`

- [ ] **Step 1: Add static structure RED tests**

Require semantic project navigation, bounded chat list, explicit `New Chat`, selected project/chat names, Fable/Sol presence, and accessible labels. Require no path input anywhere. Keep the existing center conversation and right inspector IDs stable for compatibility.

- [ ] **Step 2: Add controller RED tests**

Exercise initial project/chat bootstrap, new chat creation, old chat reopen, project switch, equal session IDs across projects, reconnect cursor reset only on an intentional project/chat selection, and lease-held disabled switching. Assert one socket/timer, safe encoded route components, CSRF on mutations, bounded project/chat/task/message DOM collections, text-only insertion, and no focus stealing on reconnect.

- [ ] **Step 3: Run RED and implement the Phase 1 navigation**

```bash
python -m pytest -q tests/agent_bridge/test_static_ui.py tests/agent_bridge/test_web.py
```

Add project/chat state to the existing controller; abort the old socket and cancel retries before switching. Disable New Chat and navigation while the server lease snapshot is active. Keep the current visual tokens in this phase; the approved warm-copper redesign belongs to Phase 3.

- [ ] **Step 4: Run browser tests and commit**

```bash
python -m pytest -q tests/agent_bridge/test_static_ui.py tests/agent_bridge/test_web.py
git add \
  src/agent_bridge/static/index.html \
  src/agent_bridge/static/app.js \
  src/agent_bridge/static/styles.css \
  tests/agent_bridge/test_static_ui.py \
  tests/agent_bridge/test_web.py
git diff --cached --check
git commit -m "feat: navigate persistent project chats"
```

---

### Task 8: Prove isolation end to end and document Phase 1 operation

**Files:**
- Modify: `tests/agent_bridge/test_e2e_fake_agents.py`
- Modify: `README.md`
- Create: `.superpowers/sdd/2026-08-11-agent-bridge/task-multi-project-foundation-report.md`

- [ ] **Step 1: Add a two-repository fake E2E RED**

Create two temporary Git repositories with equal filenames and intentionally equal session/task IDs. Start the real in-process registry/app stack with absolute fake Claude, Codex, Git, bash, and sh. Create multiple chats, plan in A, verify A's lease blocks B, complete A, plan/approve/execute B, reconnect both WebSockets, and restart. Assert stores, event sequences, provider sessions, baselines, commands, mutations, and recovery remain confined to the owning runtime.

- [ ] **Step 2: Add failure-path E2E coverage**

Prove a partial lock failure starts no server, cross-project hostile IDs produce no oracle, a removed project is inaccessible but its state remains, and successful audited legacy single-project adoption preserves all historical chats/events without copying the usage acknowledgement.

- [ ] **Step 3: Run focused and full fake-only verification**

```bash
python -m pytest -q \
  tests/agent_bridge/test_projects.py \
  tests/agent_bridge/test_hub_store.py \
  tests/agent_bridge/test_hub.py \
  tests/agent_bridge/test_store.py \
  tests/agent_bridge/test_main.py \
  tests/agent_bridge/test_web.py \
  tests/agent_bridge/test_static_ui.py \
  tests/agent_bridge/test_e2e_fake_agents.py
python -m pytest -q tests/agent_bridge
python -m compileall -q src/agent_bridge tests/agent_bridge
node --input-type=module -e "import('./src/agent_bridge/static/app.js')"
git diff --check
```

- [ ] **Step 4: Update the operator guide and report**

Document repeated `--project label=/absolute/root`, compatible `--repo`, immutable/restart-required authority, digest state paths, hub acknowledgement, legacy audit, one active workflow, and Phase 1's lack of directed/intervention behavior. Record exact RED/GREEN commands, counts, warnings, fake-only safety evidence, and known tradeoffs in the report.

- [ ] **Step 5: Request Sol review and resolve findings test-first**

Freeze the diff. Sol reviews project authority, lock/recovery ordering, migration atomicity, cross-project lookup oracles, broadcaster/replay isolation, lease races, compatibility, tests, and docs. For every finding: reproduce with a failing test, implement the narrow fix, rerun focused/full gates, and request re-review. Do not self-approve.

- [ ] **Step 6: Commit the reviewed Phase 1 completion**

```bash
git add \
  README.md \
  tests/agent_bridge/test_e2e_fake_agents.py
git diff --cached --check
git commit -m "test: verify multi-project chat isolation"
git status --short --branch
```

## Phase 1 acceptance gate

Phase 1 is complete only when Sol returns READY with no Critical or Important findings, the full fake-only suite passes, the worktree is clean, and two temporary repositories demonstrate isolated state plus one hub-wide lease. Do not begin Phase 2 on an unreviewed Phase 1 tree.
