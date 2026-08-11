# Intervention and Warm-Copper Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a durable, idempotent Intervene workflow that stops only the owned run and resumes the exact eligible Fable session or Sol thread, then deliver the approved accessible three-pane warm-copper team workspace.

**Architecture:** Persist intervention intent and stop intent atomically before any process signal. Treat Stop and Intervene as separate coordinator operations sharing exact run ownership and stop-wins race guards. Resume from stored provider identity and continuation generation; if Fable planning stopped before an ID existed, expose a deterministic new-session discontinuity rather than inventing continuity. Build the visual workspace on the Phase 1/2 project, chat, conversation, task, and question models without moving security decisions into JavaScript.

**Tech Stack:** Python 3.11+, SQLite immediate transactions, existing ProcessRunner/process-group cancellation, Claude/Codex adapters, FastAPI, browser-native ES modules, semantic HTML/CSS, pytest, Node harnesses, fake executables.

## Prerequisite and constraints

- Begin only from a Sol-reviewed, green Phase 2 tree.
- Work only in the standalone repository and an isolated worktree. Do not access or mention any private source checkout.
- Stop never resumes. Intervene persists guidance and stop intent before signaling, then resumes only through the durable record.
- Endpoint success means the intervention transaction committed, not that process termination or resume completed.
- Repeating one intervention ID cannot duplicate persistence, signal, visible message, or resume-attempt scheduling. Provider execution is never claimed exactly once across an unknowable crash boundary.
- Stop wins all completion, clarification, review, correction, and resume races.
- Fable owns intent/scope. Sol is eligible only for an exact approved nonterminal revision with a valid Sol continuation.
- An intervention never captures a clean baseline around partial work; the original approved baseline remains authoritative.
- Browser recipient choices are display intent. The coordinator validates the effective recipient and exact provider identity.
- Keep structural events collapsed in Activity/Audit. Render all user/agent conversational messages visibly.
- Use the approved warm-copper palette, not Slack names, purple, logos, assets, or copied CSS.
- Preserve text-only DOM insertion, fixed class allowlists, collection bounds, keyed auth, CSRF, readiness, project/chat ownership, and the hub lease.
- Tests are fake/local only. Do not invoke live agents, auth, network, target projects, or a browser server.
- Terra implements. Sol reviews every code, test, documentation, and fix change before acceptance.
- Task reports under `.superpowers/sdd/` are intentionally ignored scratch evidence; verify them with `git check-ignore` and require a clean tracked worktree rather than staging them.

---

### Task 1: Persist intervention and stop intent atomically

**Files:**
- Modify: `src/agent_bridge/store.py`
- Modify: `tests/agent_bridge/test_store.py`

**Interfaces:**

```python
class InterventionStatus(str, Enum):
    PENDING_STOP = "pending_stop"
    READY = "ready"
    RESUMING = "resuming"
    RESUMED = "resumed"
    RESUME_OUTCOME_UNKNOWN = "resume_outcome_unknown"
    CANCELED_BY_STOP = "canceled_by_stop"
    FAILED = "failed"

@dataclass(frozen=True, slots=True)
class InterventionRecord:
    intervention_id: str
    session_id: str
    task_id: str
    revision: int
    addressed_to: ConversationTarget
    routed_to: ConversationTarget
    message: str
    run_id: str
    continuation_state: TaskState
    source_generation: int
    resume_generation: int
    fable_session_id: str | None
    sol_thread_id: str | None
    resume_attempt_id: str | None
    resume_run_id: str | None
    status: InterventionStatus
    created_at: str

def create_intervention_and_request_stop(
    self, *, intervention_id: str, session_id: str, task_id: str,
    revision: int, expected_source_generation: int, message: str,
    addressed_to: ConversationTarget, routed_to: ConversationTarget,
    run_id: str,
) -> InterventionRecord: ...
def mark_intervention_ready(self, intervention_id: str, *, run_id: str) -> InterventionRecord: ...
def claim_intervention_resume(
    self, intervention_id: str, *, expected_resume_generation: int,
    resume_attempt_id: str, resume_run_id: str,
) -> InterventionRecord: ...
def complete_intervention(
    self, intervention_id: str, *, expected_resume_generation: int,
    resume_attempt_id: str, resume_run_id: str,
) -> InterventionRecord: ...
def cancel_intervention_by_stop(
    self, intervention_id: str, *, expected_resume_generation: int,
) -> InterventionRecord: ...
def mark_resume_outcome_unknown(
    self, intervention_id: str, *, resume_attempt_id: str, resume_run_id: str,
) -> InterventionRecord: ...
def authorize_retry_after_unknown(
    self, intervention_id: str, *, expected_resume_generation: int,
    acknowledgment_id: str,
) -> InterventionRecord: ...
```

- [ ] **Step 1: Add additive migration RED tests**

Add an `interventions` table with exact foreign keys to session/task revision/source run, unique intervention ID, status check, bounded text/IDs, provider identity columns, separate source/resume generations, resume-attempt/run identity, user-acknowledgment identity, and timestamps. Reopen current schema, prove old bytes unchanged, repeat migration idempotently, and inject a failure to prove rollback.

- [ ] **Step 2: Add commit-before-signal data RED tests**

The store method must, in one immediate transaction: verify exact session/task/revision/current active run/source generation; derive and snapshot continuation state plus validated Fable/Sol IDs from the authenticated task row rather than caller arguments; validate recipient eligibility; insert or return the idempotent intervention; persist task interruption/stop intent and continuation context; increment/reset the human-direction exchange generation into `resume_generation`; append the user intervention conversation event. Install transaction hooks and prove all rows are visible together or none are.

- [ ] **Step 3: Add idempotency and hostile-binding RED tests**

Retry the same ID/same canonical payload and assert byte-equivalent record and one event. Reuse the ID with different message, recipient, task, revision, run, or generation and fail. Test wrong chat, stale run, terminal task, no active run, provider ID mismatch, pre-approval Sol, malformed provider IDs, and cross-project route selection; none may change task/run/event state.

- [ ] **Step 4: Add crash-state transition RED tests**

Persist and reopen at each status. Claiming resume persists exact attempt/run IDs before invocation and is idempotent only for that owner; stale generations/attempts cannot claim or complete. Stop atomically moves PENDING_STOP/READY/RESUMING to `CANCELED_BY_STOP`, invalidates pending claims, and late workers cannot invoke or complete. Startup retires the stale source run without signaling and moves `PENDING_STOP` to `READY`; existing `READY` remains safely resumable. Startup converts every committed `RESUMING` attempt to `RESUME_OUTCOME_UNKNOWN` without signaling or replaying, including a barrier crash immediately after claim commit and before local process spawn. A new attempt is legal only after an authenticated user supplies a unique acknowledgment that the prior provider call may already have executed; authorization increments `resume_generation` and returns to READY.

- [ ] **Step 5: Run RED and implement**

```bash
python -m pytest -q tests/agent_bridge/test_store.py -k intervention
```

Use strict structural JSON only where existing nested continuation context requires it; intervention identity fields remain queryable columns. Provider IDs are durable coordinator authority but never part of safe browser projections.

- [ ] **Step 6: Run store suite and commit**

```bash
python -m pytest -q tests/agent_bridge/test_store.py
git add src/agent_bridge/store.py tests/agent_bridge/test_store.py
git diff --cached --check
git commit -m "feat: persist durable intervention intent"
```

---

### Task 2: Make ProcessRunner stop ownership and process exit exact

**Files:**
- Modify: `src/agent_bridge/process.py`
- Modify: `tests/agent_bridge/test_process.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class StopReceipt:
    run_id: str
    was_running: bool
    process_exited: bool

async def stop(self, run_id: str, *, timeout_seconds: float) -> StopReceipt: ...
async def wait_process_exit(self, run_id: str, *, timeout_seconds: float) -> None: ...
```

- [ ] **Step 1: Add exact-owned-run RED tests**

Start two fake process groups and stop one run ID. Assert only its process group receives termination, the other completes, PID/PGID are never accepted from browser/store input as authority, and unknown/stale IDs do not fall back to a broad process search. Preserve the current prelaunch stop-intent behavior.

- [ ] **Step 2: Add finalization and race RED tests**

Use barriers for stop-before-launch, stop-during-registration, simultaneous process exit/stop, repeated stop, terminate timeout/escalation, and cancellation. Assert one signal sequence, exact process-completion event identity, bounded wait, and no leaked process map entry. Do not claim this receipt covers adapter parsing or coordinator post-run routing; Task 3 owns that separate completion boundary.

- [ ] **Step 3: Run RED, implement, and run GREEN**

```bash
python -m pytest -q tests/agent_bridge/test_process.py
```

Keep process-group signaling and process-exit observation inside `ProcessRunner`; the coordinator supplies only the run ID it just authenticated from the store. Keep adapter/coordinator completion tracking in the coordinator's exact registered run-completion events. Do not add broad process enumeration or persisted-PID signaling.

- [ ] **Step 4: Commit**

```bash
git add src/agent_bridge/process.py tests/agent_bridge/test_process.py
git diff --cached --check
git commit -m "fix: make owned-run finalization explicit"
```

---

### Task 3: Separate Stop from Intervene in the coordinator

**Files:**
- Modify: `src/agent_bridge/hub.py`
- Modify: `src/agent_bridge/coordinator.py`
- Modify: `tests/agent_bridge/test_hub.py`
- Modify: `tests/agent_bridge/test_coordinator.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class InterventionIntent:
    intervention_id: str
    message: str
    addressed_to: ConversationTarget
    revision: int
    continuation_generation: int

class InterventionLeaseOrigin(str, Enum):
    BORROWED_SOURCE = "borrowed_source"
    RECOVERY_ACQUIRED = "recovery_acquired"

@dataclass(frozen=True, slots=True)
class PreparedIntervention:
    record: InterventionRecord
    lease_token: LeaseToken
    lease_origin: InterventionLeaseOrigin

def prepare_intervention(self, task_id: str, intent: InterventionIntent) -> InterventionRecord: ...
async def continue_intervention(self, intervention_id: str) -> None: ...
async def resume_intervention(
    self, intervention_id: str, *, resume_attempt_id: str, resume_run_id: str,
) -> None: ...

# HubWorkflowOrchestrator methods; this layer alone owns LeaseToken values.
def prepare_intervention(
    self, *, project_id: str, session_id: str, task_id: str,
    intent: InterventionIntent,
) -> PreparedIntervention: ...
async def continue_intervention(self, prepared: PreparedIntervention) -> None: ...
def abort_prepared_intervention(
    self, prepared: PreparedIntervention, *, reason: str,
) -> InterventionRecord: ...
def prepare_recovery_resume(
    self, *, project_id: str, session_id: str, intervention_id: str,
    expected_resume_generation: int,
) -> PreparedIntervention: ...
```

- [ ] **Step 1: Add commit-before-signal RED tests**

Use a fake store transaction hook and runner recorder. Assert the hub orchestrator verifies the exact current lease token, `Coordinator.prepare_intervention` commits and returns an `InterventionRecord`, and the hub wraps it with the sole-owner `LeaseToken` plus `lease_origin=BORROWED_SOURCE` as `PreparedIntervention`; no `runner.stop` occurs in preparation. Make persistence fail and prove no signal/scheduled continuation and the original lease remains owned. Make stop fail in `continue_intervention` and prove intervention remains durable/pending, task remains interrupted, a bounded `stop_error` is emitted without losing guidance, and the borrowed lease is not released while the source process may still be live. It releases only after exact ProcessRunner/coordinator finalization proves the source is gone; otherwise the hub remains blocked until exact Stop retry or process restart.

- [ ] **Step 2: Add Stop-wins matrix RED tests**

At coordinator-owned adapter-completion barriers after process exit/parsing but before each post-run route, race Stop/Intervene with planning, Sol outcome, clarification, Fable review, correction, adapter exception, and session/thread publication. Assert task stays interrupted, intervention remains pending/ready, late sanitized evidence may persist, and no completion/review/correction/new task transition wins. Race a later Stop with PENDING_STOP, READY, and RESUMING; assert atomic `CANCELED_BY_STOP`, exact resume-run signaling when present, and no late invoke/complete.

- [ ] **Step 3: Implement and commit durable preparation/Stop guards**

Implement only synchronous preparation, store transition guards, and post-run Stop/intervention checks first. Do not invoke a resumed agent in this checkpoint. Run the commit-before-signal and Stop-wins selectors, then commit:

```bash
python -m pytest -q tests/agent_bridge/test_hub.py tests/agent_bridge/test_coordinator.py -k "prepare_intervention or stop_wins"
git add \
  src/agent_bridge/hub.py \
  src/agent_bridge/coordinator.py \
  tests/agent_bridge/test_hub.py \
  tests/agent_bridge/test_coordinator.py
git diff --cached --check
git commit -m "feat: prepare stop-safe interventions"
```

- [ ] **Step 4: Add recipient eligibility RED tests**

Fable is an eligible intent target in every active phase, subject to continuation readiness. Sol is rejected during planning/pre-approval, without a valid thread, after terminal state, and after scope has widened. Sol is accepted only under the exact approved revision and valid Sol continuation. Browser addressee and persisted coordinator route are both visible; only routed recipient is invoked.

- [ ] **Step 5: Add recovery/idempotency RED tests**

Crash after commit/before signal, during signal, after adapter/coordinator completion but before READY, after READY but before `claim_intervention_resume` commits, immediately after that claim commits but before local process spawn, and after fake provider acceptance before result persistence. Only a crash before the claim commit leaves READY and permits explicit Resume to create an attempt. Every crash after claim commit—including before local spawn—recovers as `RESUME_OUTCOME_UNKNOWN`, never auto-replays, and requires a new authenticated acknowledgment before another attempt. Uninterrupted flow resumes automatically after READY. Retry the original endpoint and recovery Resume repeatedly; one intervention message, one owned stop, one scheduling record per attempt, and no automatic duplicate provider call. Assert `prepare_recovery_resume` returns `lease_origin=RECOVERY_ACQUIRED` for both PENDING_STOP and READY recovery. Force scheduler installation rejection for both statuses: `abort_prepared_intervention` leaves the exact durable status unchanged, performs no provider probe, claim, signal, or adapter call, and releases exactly the recovery-acquired token so a later explicit Resume can retry. Contrast this with the borrowed-source rejection case, which must retain its live source token.

- [ ] **Step 6: Run RED and implement ordered continuation**

```bash
python -m pytest -q tests/agent_bridge/test_coordinator.py -k "intervene or stop_wins or intervention"
```

Required normal-path order: `HubWorkflowOrchestrator.prepare_intervention` validates route-selected ownership/current token and commits intervention/stop intent/event → the app schedules its returned `continue_intervention` in the lifespan-tracked task set → signal exact source run → await process exit and the coordinator's exact adapter-completion event → re-read exact task/run/source+resume generations/intervention → mark READY → await the route-selected `RuntimeReadiness.require_model_start_ready` fresh combined gate immediately before claim → persist resume-attempt/run IDs → automatically invoke the eligible exact agent while the orchestrator retains the original lease → complete with exact attempt/run CAS → release that token. If the fresh gate fails, leave READY, emit only a fixed safe status, and release the token for later explicit Resume. Pre-claim recovery from PENDING_STOP/READY acquires a new generation-safe lease, returns a `RECOVERY_ACQUIRED` prepared action, and does not probe, claim, signal, or invoke until the app successfully installs its continuation. The installed continuation performs the bounded fresh gate while holding that token. Recovery from any committed RESUMING claim marks outcome unknown and refuses replay until explicit risk acknowledgment. Every await boundary rechecks Stop/intervention/attempt identity.

- [ ] **Step 7: Run hub/coordinator/process/store suites and commit continuation**

```bash
python -m pytest -q \
  tests/agent_bridge/test_store.py \
  tests/agent_bridge/test_process.py \
  tests/agent_bridge/test_hub.py \
  tests/agent_bridge/test_coordinator.py
git add \
  src/agent_bridge/hub.py \
  src/agent_bridge/coordinator.py \
  tests/agent_bridge/test_hub.py \
  tests/agent_bridge/test_coordinator.py
git diff --cached --check
git commit -m "feat: continue durable interventions"
```

---

### Task 4: Resume the exact Fable session or Sol thread

**Files:**
- Modify: `src/agent_bridge/coordinator.py`
- Modify: `tests/agent_bridge/test_claude_cli.py`
- Modify: `tests/agent_bridge/test_codex_cli.py`
- Modify: `tests/agent_bridge/test_coordinator.py`

**Interfaces:**

```python
class InterventionOperation(str, Enum):
    FABLE_PLAN = "fable_plan"
    FABLE_RESUME_PLAN = "fable_resume_plan"
    FABLE_CLARIFY = "fable_clarify"
    FABLE_REVIEW = "fable_review"
    SOL_RESUME = "sol_resume"

@dataclass(frozen=True, slots=True)
class InterventionInvocation:
    operation: InterventionOperation
    run_id: str
    expected_contract: Literal[
        "TaskBrief", "FableClarification", "ReviewVerdict", "SolOutcome",
    ]
```

Do not add a generic adapter method with an ambiguous payload. Map the persisted continuation to existing public protocol methods exactly: early Fable planning → `FableAdapter.plan`/`TaskBrief`; Fable planning with a session → `resume_plan`/`TaskBrief`; clarification or Fable guidance during approved Sol work → `clarify`/`FableClarification`; review → `review`/`ReviewVerdict`; Sol execution/correction → `SolAdapter.resume`/`SolOutcome`. Existing `adapters/base.py` protocols already declare these methods, and the coordinator validates the named contract before routing.

- [ ] **Step 1: Add exact provider-ID RED tests**

Assert Fable `resume_plan`/`clarify`/`review` passes exactly the stored validated session to `--resume`; Sol `resume` passes exactly the stored validated thread and the coordinator prompt contains the original approved brief. Missing, leading-dash, malformed, changed, returned-mismatched, or cross-task IDs fail before invoking/resuming or exposing partial results. Intervention text stays prompt-only and cannot alter argv.

- [ ] **Step 2: Add early-Fable-discontinuity RED tests**

Interrupt planning before Fable emits a session ID. Assert the durable intervention remains READY with no invented ID. Normal automatic continuation—or explicit recovery Resume after a crash before any resume claim commits—starts a fresh Fable planning call from persisted original request, exact task/revision context, current brief if any, partial sanitized evidence, and intervention. Persist and emit a fixed visible system message that continuity could not be preserved and a new Fable session was started; save the new returned ID only after validation.

- [ ] **Step 3: Add same-scope/scope-change RED tests**

Fable same-scope intervention guidance resumes exact Sol thread under the same approved brief. Scope-changing guidance produces revision N+1 and waits for exact approval; Sol never receives the widened instruction beforehand. Preserve the original baseline and all partial changes for final delta comparison.

- [ ] **Step 4: Add nested-continuation RED tests**

Intervene during Fable clarification, Fable review, Sol correction, and an agent-to-agent question. Assert underlying continuation, pending question/exchange, source/resume generations, partial events, Fable session, Sol thread, approved revision, and baseline survive restart. In known pre-invocation and uninterrupted cases exactly one attempt runs; an in-flight crash becomes outcome-unknown and does not replay without acknowledgment.

- [ ] **Step 5: Run RED and implement**

```bash
python -m pytest -q \
  tests/agent_bridge/test_claude_cli.py \
  tests/agent_bridge/test_codex_cli.py \
  tests/agent_bridge/test_coordinator.py -k "intervention or resume_with"
```

Use existing protocol methods, fixed argv, and sanitized environment boundaries. Build each prompt in the coordinator from structurally reconstructed safe context, original approved brief where applicable, exact intervention text, and bounded partial evidence—never raw command/result/stream data. Validate the `expected_contract` named by `InterventionInvocation` before any next transition.

- [ ] **Step 6: Run adapter/coordinator suites and commit**

```bash
python -m pytest -q \
  tests/agent_bridge/test_claude_cli.py \
  tests/agent_bridge/test_codex_cli.py \
  tests/agent_bridge/test_coordinator.py
git add \
  src/agent_bridge/coordinator.py \
  tests/agent_bridge/test_claude_cli.py \
  tests/agent_bridge/test_codex_cli.py \
  tests/agent_bridge/test_coordinator.py
git diff --cached --check
git commit -m "feat: resume exact agents after intervention"
```

---

### Task 5: Add the authenticated project-aware Intervene API

**Files:**
- Modify: `src/agent_bridge/app.py`
- Modify: `tests/agent_bridge/test_web.py`

**Request DTO and route:**

```python
class InterventionRequest(BaseModel):
    intervention_id: StrictStr
    message: StrictStr
    addressed_to: Literal["fable", "sol"]
    revision: StrictInt
    continuation_generation: StrictInt

class InterventionResumeRequest(BaseModel):
    expected_resume_generation: StrictInt

class UnknownOutcomeRetryRequest(BaseModel):
    expected_resume_generation: StrictInt
    acknowledgment_id: StrictStr
    acknowledge_possible_prior_execution: Literal[True]

POST /api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/intervene
POST /api/projects/{project_id}/chats/{session_id}/interventions/{intervention_id}/resume
POST /api/projects/{project_id}/chats/{session_id}/interventions/{intervention_id}/authorize-retry
```

- [ ] **Step 1: Add strict body/auth RED tests**

Require authenticated keyed cookie, exact CSRF, usage acknowledgement, exact project/chat/task/revision/source or resume generation, bounded nonempty message, opaque idempotency/acknowledgment IDs, and enum recipient. Reject extra `routed_to`, run/provider/continuation/path/command/env fields. Stop remains available without model readiness; Intervene validates recipient and can commit while readiness is temporarily unavailable. Automatic continuation and recovery Resume recheck the complete existing combined model gate before claiming an attempt. Unknown-outcome retry additionally requires the literal true risk acknowledgment and never shares the ordinary Resume path.

- [ ] **Step 2: Add acceptance timing RED tests**

Block fake runner stop after store commit and assert the HTTP intervention response is already an accepted safe projection. The route calls only `HubWorkflowOrchestrator.prepare_intervention`, then passes `lambda: workflows.continue_intervention(prepared)` as the factory to Phase 1's app-local `install_prepared_action`, with `workflows.abort_prepared_intervention` as the synchronous abort callback; the coordinator never creates an unobserved task. Force scheduler rejection and assert the response still projects the committed PENDING_STOP intervention and no signal/unobserved coroutine occurs. For `lease_origin=BORROWED_SOURCE`, abort must not release the token early while that source process may be live; the source workflow observes durable stop intent, rejects late routing, and releases its exact token in its existing finalizer. Exercise ordinary recovery Resume from both PENDING_STOP and READY through the same app-local installer: `prepare_recovery_resume` returns `lease_origin=RECOVERY_ACQUIRED`, and forced installation rejection calls the same abort method, leaves the durable status unchanged, starts no probe/claim/signal/coroutine, and releases exactly that acquired token. After restart no in-memory lease survives and recovery can continue the intervention. Persistence failure returns bounded error and no signal/schedule. Subsequent stop/adapter-finalization failure is visible through events/status rather than raw exception data.

- [ ] **Step 3: Add cross-project/idempotency RED tests**

Use identical IDs in two runtimes. Assert the route-selected runtime is the only one queried and a foreign intervention cannot be detected, resumed, signaled, or rewritten. Repeat identical request safely; conflicting reuse returns 409 and does not schedule.

- [ ] **Step 4: Add bootstrap projection RED tests**

Expose only intervention ID, safe message, addressed/routed labels, status, task/revision, source/resume generation, eligibility, visible discontinuity flag, and a fixed `prior resume outcome is unknown and may have executed` warning when applicable. Never expose run/attempt ID, PID/PGID, provider session/thread, pending JSON, baseline path, raw result, or command.

- [ ] **Step 5: Run RED, implement, and run web suite**

```bash
python -m pytest -q tests/agent_bridge/test_web.py -k intervention
python -m pytest -q tests/agent_bridge/test_web.py
```

- [ ] **Step 6: Commit**

```bash
git add src/agent_bridge/app.py tests/agent_bridge/test_web.py
git diff --cached --check
git commit -m "feat: expose durable intervention controls"
```

---

### Task 6: Build the approved semantic three-pane workspace

**Files:**
- Modify: `src/agent_bridge/static/index.html`
- Modify: `tests/agent_bridge/test_static_ui.py`

- [ ] **Step 1: Add semantic desktop RED tests**

Require one application landmark containing: left `nav` with project heading/list, chat heading/list, New Chat, Fable/Sol presence; center `main` with selected project/chat heading, chronological conversation, live region, bound-question/intervention context, composer, Send/Intervene/Stop; right complementary inspector with exact task revision/state/scope/allowed paths/required tests/question budget/controls and collapsed Activity/Audit `details`.

- [ ] **Step 2: Add accessibility RED tests**

Require semantic headings, explicit labels, button types, accessible avatar text, non-color status text, `aria-current` selection, `aria-live` limited to status, dialog/drawer labels, Escape-close controls, inert-background hooks, focus targets, reduced-motion support hooks, and no inline event handlers/styles or unsafe HTML sinks.

- [ ] **Step 3: Add mobile structure RED tests**

Require conversation-first DOM order and two keyboard-accessible drawer dialogs for project/chat navigation and inspector. The same controls must not be duplicated as independently focusable desktop/mobile copies.

- [ ] **Step 4: Run RED and implement HTML structure**

```bash
python -m pytest -q tests/agent_bridge/test_static_ui.py -k "layout or semantic or drawer or accessibility"
```

Use product-owned names Fable, Sol, Projects, Chats, Conversation, and Task inspector. Do not mention Slack in shipped UI or CSS identifiers.

- [ ] **Step 5: Run static tests and commit**

```bash
python -m pytest -q tests/agent_bridge/test_static_ui.py
git add src/agent_bridge/static/index.html tests/agent_bridge/test_static_ui.py
git diff --cached --check
git commit -m "feat: structure the team workspace"
```

---

### Task 7: Apply the warm-copper visual system and responsive behavior

**Files:**
- Modify: `src/agent_bridge/static/styles.css`
- Modify: `src/agent_bridge/static/app.js`
- Modify: `tests/agent_bridge/test_static_ui.py`
- Modify: `tests/agent_bridge/test_web.py`

**Required design tokens:**

```css
:root {
  --ink-950: #211a17;
  --ink-800: #3b302b;
  --cream-50: #fffaf3;
  --cream-100: #f6ecdf;
  --copper-700: #9a4524;
  --copper-600: #b65a31;
  --copper-100: #f4d8c7;
  --green-700: #49634e;
  --green-100: #dce8da;
  --danger-700: #8b2f2f;
  --focus-ring: #176b87;
}
```

Exact contrast values may be adjusted only to improve measured accessibility while preserving ink/cream/copper/muted-green intent.

- [ ] **Step 1: Add visual-token RED tests**

Parse CSS and require named warm-copper tokens, no Slack brand names/assets, no purple primary palette, visible 2px focus indicators, non-color text/icon state, minimum hit targets, responsive grid breakpoints, `prefers-reduced-motion`, and high-contrast-safe borders. Compute relative luminance/contrast for every shipped foreground/background pairing and focus ring against each adjacent surface; require WCAG AA text contrast (4.5:1 normal, 3:1 large) and at least 3:1 for focus/non-text indicators.

- [ ] **Step 2: Add controller/intervention RED tests**

When idle, composer defaults to Fable and Send. While an agent runs, ordinary Send is disabled and explicit Intervene plus separate Stop appear. Intervene opens/binds recipient choices based on safe server eligibility, generates/reuses one idempotency ID until resolved, submits exact revision/source generation, and shows accepted/pending/resume/canceled states. Stop never sends composer text and cancels pending/resuming intervention UI after the server CAS. Reconnect/reload reconstructs pending intervention from bootstrap. `RESUME_OUTCOME_UNKNOWN` never auto-resumes: render a focused warning that the prior call may have executed plus a separate acknowledgment control that generates one acknowledgment ID and submits the literal true risk flag before a new attempt.

- [ ] **Step 3: Add drawer/focus RED tests**

On narrow media, project/chat and inspector drawers trap focus, close on Escape, restore invoking focus, mark background inert, and never steal focus on WebSocket reconnect. On desktop transition, dialogs close and inert state clears. Test no duplicate socket/retry timers and bounded rendering under 200 tasks, 300 messages, 50 chats/page, and project summary limits.

- [ ] **Step 4: Add safe conversation/audit RED tests**

Agent-to-agent messages remain in the center with sender→recipient labels. Structural details render only after opening Activity/Audit. Insert hostile text in project labels, chat titles, messages, questions, intervention, evidence, and errors; assert literal text, fixed classes, no `innerHTML`, no script/event handler execution, and no unbounded JSON dump.

- [ ] **Step 5: Run RED and implement CSS/controller behavior**

```bash
python -m pytest -q tests/agent_bridge/test_static_ui.py tests/agent_bridge/test_web.py
```

Keep server responses authoritative. JavaScript disables controls for clarity, but direct endpoint tests remain the security proof. Use CSS Grid for three panes, plain text avatar initials, fixed actor/target/status class maps, and reduced animation.

- [ ] **Step 6: Run browser tests and commit**

```bash
python -m pytest -q tests/agent_bridge/test_static_ui.py tests/agent_bridge/test_web.py
node --input-type=module -e "import('./src/agent_bridge/static/app.js')"
git add \
  src/agent_bridge/static/styles.css \
  src/agent_bridge/static/app.js \
  tests/agent_bridge/test_static_ui.py \
  tests/agent_bridge/test_web.py
git diff --cached --check
git commit -m "feat: apply the warm-copper team workspace"
```

---

### Task 8: Verify intervention, recovery, isolation, and accessibility end to end

**Files:**
- Modify: `tests/agent_bridge/test_e2e_fake_agents.py`
- Modify: `README.md`
- Create: `.superpowers/sdd/2026-08-11-agent-bridge/task-intervention-workspace-report.md`

- [ ] **Step 1: Add full-stack fake intervention RED tests**

In two temporary Git repos, intervene during Fable planning before session publication, after Fable session publication, during Sol execution with partial allowed edits, during Fable clarification/review, and during Sol correction. Assert commit-before-signal ordering, exact process ownership, visible intervention, explicit new-session discontinuity only for early Fable, exact otherwise, and original baseline coverage.

- [ ] **Step 2: Add crash/restart/race RED tests**

Crash/recreate after every intervention status and race late success/failure/question/review events against Stop/Intervene. Assert latest active revision recovery, all stale runs retired without signals, pre-claim READY Resume visible, no cross-project lease/store/broadcaster mutation, and no stale completion transition. For every crash after claim commit—including before local spawn—and during provider invocation, assert `RESUME_OUTCOME_UNKNOWN`, zero automatic replay after any number of restarts, explicit possible-execution warning, required authenticated acknowledgment, and exactly one newly identified attempt only after that acknowledgment.

- [ ] **Step 3: Add browser full-flow RED tests**

Using the existing Node/fake DOM controller harness, navigate projects/chats, create chat, converse with visible Fable/Sol, answer exact questions, grant +3, approve exact N+1, intervene to each eligible agent, Stop/cancel separately, recover a safe pending Resume, acknowledge an unknown prior resume outcome, reload/reconnect, open Activity, and exercise both drawers by keyboard. Assert warm-copper tokens, computed contrast, and non-color labels without screenshot/image dependencies.

- [ ] **Step 4: Run focused and complete fake-only verification**

```bash
python -m pytest -q \
  tests/agent_bridge/test_store.py \
  tests/agent_bridge/test_process.py \
  tests/agent_bridge/test_claude_cli.py \
  tests/agent_bridge/test_codex_cli.py \
  tests/agent_bridge/test_coordinator.py \
  tests/agent_bridge/test_web.py \
  tests/agent_bridge/test_static_ui.py \
  tests/agent_bridge/test_e2e_fake_agents.py
python -m pytest -q tests/agent_bridge
python -m compileall -q src/agent_bridge tests/agent_bridge
node --input-type=module -e "import('./src/agent_bridge/static/app.js')"
git diff --check
```

- [ ] **Step 5: Update operator documentation and evidence**

Document Stop versus Intervene, eligibility, normal automatic continuation, explicit safe recovery Resume, unknown-outcome no-replay/risk acknowledgment, early-Fable discontinuity, baseline preservation, multi-project/chat startup, directed question budget, accessibility/navigation, and fake-only testing. State plainly that the provider CLIs offer no transactional idempotency key, so the bridge does not claim exactly-once execution across a crash. Record exact test counts, warnings, process provenance, zero live calls, and any consciously retained limitations.

- [ ] **Step 6: Request per-change and final whole-branch Sol review**

Sol reviews all intervention schema/order/races, ProcessRunner ownership, provider continuity, scope/baseline behavior, API/idempotency, browser safety/accessibility, tests, fixtures, and docs. After fixes and focused reruns, request one final review across all three phases against the approved design. No implementer self-approval.

- [ ] **Step 7: Commit reviewed completion**

```bash
git add README.md tests/agent_bridge/test_e2e_fake_agents.py
git diff --cached --check
git commit -m "test: verify intervention and workspace flows"
git status --short --branch
```

## Final acceptance gate

The feature is complete only when Sol returns READY on Phase 3 and the whole three-phase branch with no Critical or Important findings; the complete fake-only suite passes; two temporary repositories prove isolation; race tests prove persistence-before-signal, Stop-wins behavior, and no automatic replay of an unknown resume outcome; exact provider continuity/discontinuity is visible; baseline validation covers all partial work; and the accessible warm-copper workspace supports persistent projects, chats, team questions, Stop, Intervene, and Resume without moving authority into the browser.
