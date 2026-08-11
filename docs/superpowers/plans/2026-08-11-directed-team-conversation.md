# Directed Team Conversation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the project chat stream into a safe directed conversation among the user, Fable, and Sol while keeping Fable as planner/reviewer, Sol as approved executor, user questions human-only, and automatic agent dialogue visibly bounded.

**Architecture:** Add a strict conversational envelope to new persisted events, exact durable question/exchange records to the project store, and one deterministic coordinator routing layer. Browser intent records `addressed_to`; only the coordinator writes `routed_to`. Existing task continuations, provider session/thread IDs, exact revisions, and repository authority remain the execution source of truth. Legacy bytes remain unchanged and receive only a conservative display projection.

**Tech Stack:** Python 3.11+, dataclasses/enums, SQLite, existing Claude/Codex CLI adapters, FastAPI, browser-native ES modules, pytest, Node harnesses, fake agent executables.

## Prerequisite and constraints

- Begin only from a Sol-reviewed, green Phase 1 tree.
- Work only in the standalone repository and an isolated worktree. Do not access or mention any private source checkout.
- Do not allow browser payloads to set `routed_to`, a continuation, provider ID, command, path, executable, or environment value.
- An unbound ordinary user message always starts a new Fable task, even when the chat contains pending questions.
- Sol is an executor only. It cannot receive a model call before exact user approval or after terminal completion.
- Questions routed to the user can be answered only by the authenticated exact-answer endpoint.
- Automatic Fable–Sol dialogue starts with three exchanges per exact revision. A reserved question's one paired answer is always allowed.
- The user may grant exactly three more exchanges per permission action; there is no unlimited mode.
- Emit every conversational question/answer before invoking the next agent. Keep structural events in Activity/Audit.
- Preserve account acknowledgement, subscription/readiness, keyed session, CSRF, hub lease, exact-revision, baseline, and repository checks.
- Tests are fake/local only. Do not call live models, auth, network, or a browser server.
- Terra implements. Sol reviews every code, test, documentation, and fix change before acceptance.
- Task reports under `.superpowers/sdd/` are intentionally ignored scratch evidence; verify them with `git check-ignore` and require a clean tracked worktree rather than staging them.

---

### Task 1: Define the validated conversation envelope

**Files:**
- Modify: `src/agent_bridge/contracts.py`
- Modify: `tests/agent_bridge/test_contracts.py`

**Interfaces:**

```python
class ConversationActor(str, Enum):
    USER = "user"
    FABLE = "fable"
    SOL = "sol"
    SYSTEM = "system"

class ConversationTarget(str, Enum):
    USER = "user"
    FABLE = "fable"
    SOL = "sol"
    TEAM = "team"

class ConversationMessageType(str, Enum):
    STATEMENT = "statement"
    QUESTION = "question"
    ANSWER = "answer"
    APPROVAL = "approval"
    INTERVENTION = "intervention"
    STATUS = "status"

@dataclass(frozen=True, slots=True)
class ConversationEnvelope:
    sender: ConversationActor
    addressed_to: ConversationTarget
    routed_to: ConversationTarget
    message_type: ConversationMessageType
    text: str
    task_id: str | None = None
    revision: int | None = None
    continuation_generation: int | None = None
    question_id: str | None = None
    reply_to_question_id: str | None = None

    def to_dict(self) -> dict[str, object]: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ConversationEnvelope": ...

@dataclass(frozen=True, slots=True)
class UserConversationInput:
    addressed_to: ConversationTarget
    message_type: ConversationMessageType
    text: str
    task_id: str | None = None
    revision: int | None = None
    question_id: str | None = None
    reply_to_question_id: str | None = None
    continuation_generation: int | None = None

@dataclass(frozen=True, slots=True)
class DirectedAgentQuestion:
    addressed_to: Literal["user", "fable", "sol"]
    text: str
    reason: str
```

Project/chat identity remains in the owning event row/runtime and is not accepted inside either contract. `UserConversationInput` deliberately has no `sender` or `routed_to`; the authenticated endpoint fixes `sender=user` and the coordinator constructs the persisted `ConversationEnvelope` after routing.

- [ ] **Step 1: Add strict round-trip RED tests**

Assert exact field sets, enum validation, bounded nonempty text/reason, bounded opaque IDs, positive revisions/generations, and canonical JSON round trips for all three contracts. Reject extra keys, booleans as integers, control characters, missing route, unknown actors/targets/types, question IDs on statements, replies without `message_type=answer`, answers without `reply_to_question_id`, question IDs duplicated into reply fields, and agent-question targets outside user/Fable/Sol.

- [ ] **Step 2: Add semantic pair RED tests**

Require questions to have `question_id`, answers to have `reply_to_question_id`, every bound question/answer to have task/revision/continuation generation, approvals to bind task/revision, and system status to use `sender=system`. Assert `UserConversationInput` accepts only `addressed_to`, `message_type`, text, and exact all-or-none optional binding; attempts to supply `sender` or `routed_to` fail before coordinator routing.

- [ ] **Step 3: Run RED, implement, and run GREEN**

```bash
python -m pytest -q tests/agent_bridge/test_contracts.py -k conversation
python -m pytest -q tests/agent_bridge/test_contracts.py
```

Use the repository's existing bounded-string and exact-field helpers. Do not infer or normalize unknown values.

- [ ] **Step 4: Commit**

```bash
git add src/agent_bridge/contracts.py tests/agent_bridge/test_contracts.py
git diff --cached --check
git commit -m "feat: define directed conversation envelopes"
```

---

### Task 2: Persist exact questions, continuations, and exchange budgets

**Files:**
- Modify: `src/agent_bridge/store.py`
- Modify: `tests/agent_bridge/test_store.py`

**Interfaces:**

```python
INITIAL_INTERNAL_EXCHANGES = 3
EXCHANGE_GRANT_SIZE = 3

@dataclass(frozen=True, slots=True)
class QuestionRecord:
    question_id: str
    session_id: str
    task_id: str
    revision: int
    continuation_generation: int
    asked_by: ConversationActor
    addressed_to: ConversationTarget
    routed_to: ConversationTarget
    text: str
    exchange_id: str | None
    answer_text: str | None
    answered_by: ConversationActor | None

@dataclass(frozen=True, slots=True)
class ExchangeReservation:
    exchange_id: str
    question_id: str
    ordinal: int
    continuation_generation: int

def pause_for_question(
    self, *, session_id: str, task_id: str, revision: int,
    expected_generation: int, question_id: str,
    asked_by: ConversationActor, addressed_to: ConversationTarget,
    routed_to: ConversationTarget, text: str,
    continuation_state: TaskState, pending_action: Mapping[str, object],
    event: ConversationEnvelope,
) -> QuestionRecord: ...
def answer_question_and_prepare_resume(
    self, *, session_id: str, task_id: str, revision: int,
    question_id: str, expected_generation: int, answer_text: str,
    answered_by: ConversationActor, pending_action: Mapping[str, object],
    event: ConversationEnvelope,
) -> QuestionRecord: ...
def reserve_internal_question(
    self, *, session_id: str, task_id: str, revision: int,
    expected_generation: int, question_id: str, request_key: str,
    asked_by: ConversationActor, addressed_to: ConversationTarget,
    routed_to: ConversationTarget, text: str,
    continuation_state: TaskState, pending_action: Mapping[str, object],
    event: ConversationEnvelope,
) -> tuple[ExchangeReservation, QuestionRecord]: ...
def pause_for_exchange_permission(
    self, *, session_id: str, task_id: str, revision: int,
    expected_generation: int, attempted_question: DirectedAgentQuestion,
    continuation_state: TaskState, pending_action: Mapping[str, object],
    event: ConversationEnvelope,
) -> TaskRecord: ...
def grant_internal_exchanges(
    self, *, session_id: str, task_id: str, revision: int,
    expected_generation: int, request_id: str,
) -> int: ...
```

- [ ] **Step 1: Add additive schema/migration RED tests**

Add exact tables/indexes for questions and exchange reservations plus task-level `continuation_generation`, allowance, and consumed count. Reopen current schema and prove every legacy row byte remains. Test idempotent migration and injected transaction failure rollback.

- [ ] **Step 2: Add exact-question CAS RED tests**

Create two tasks with pending questions and assert an answer requires exact session/task/revision/question/generation. Test stale generation, wrong question, wrong chat, wrong revision, already answered, no pending question, and cross-project lookup through the Phase 1 route harness. None may write an event, clear a continuation, or schedule an agent. Enforce at most one unanswered question per task revision, while different tasks may each wait. At the store boundary, derive the only legal `answered_by` from the persisted validated `routed_to`: a user-routed question accepts only `user`, and an agent-routed question only that exact agent.

- [ ] **Step 3: Add atomic exchange RED tests**

With concurrent connections/barriers, reserve the same request key twice and assert one durable exchange ID/one charge/question/event/pending action. Reserve three different questions and assert ordinals 1–3. The fourth initiation atomically persists the attempted question, exact continuation, permission-pending action, and user-addressed event without a reservation. Assert the already-reserved question's paired answer remains legal at the limit. Granting adds exactly three, duplicate grant request keys are idempotent, and no path sets an unlimited/null maximum.

- [ ] **Step 4: Add reset semantics RED tests**

Assert a user answer advances `continuation_generation` and resets remaining allowance to three; expose the same atomic human-direction reset primitive for Phase 3's future intervention transaction. Exact approval of revision N+1 creates a separate three-exchange budget. Retried store operations with the old generation fail. Required planning, execution, review, and correction calls do not consume budget unless they reserve a question.

- [ ] **Step 5: Run RED and implement transactionally**

```bash
python -m pytest -q tests/agent_bridge/test_store.py -k "question or exchange or generation"
```

Each method performs its counter/question or answer mutation, task transition, exact continuation/pending-next-action binding, and conversation-event insert in one immediate transaction. Unique constraints cover request-key idempotency and one unanswered question per exact task revision. `answer_question_and_prepare_resume` compare-and-swaps every identity field plus the derived legal actor. Store listeners snapshot/publish only after commit, so a visible event and its durable next action cannot separate across a crash.

- [ ] **Step 6: Run store suite and commit**

```bash
python -m pytest -q tests/agent_bridge/test_store.py
git add src/agent_bridge/store.py tests/agent_bridge/test_store.py
git diff --cached --check
git commit -m "feat: persist directed questions and exchange budgets"
```

---

### Task 3: Extend fake-safe adapter contracts for directed questions

**Files:**
- Modify: `src/agent_bridge/contracts.py`
- Modify: `src/agent_bridge/adapters/base.py`
- Modify: `src/agent_bridge/adapters/claude_cli.py`
- Modify: `src/agent_bridge/adapters/codex_cli.py`
- Modify: `tests/agent_bridge/fixtures/fake_claude.py`
- Modify: `tests/agent_bridge/fixtures/fake_codex.py`
- Modify: `tests/agent_bridge/test_claude_cli.py`
- Modify: `tests/agent_bridge/test_codex_cli.py`

**Interfaces:**

Extend `SolQuestion`, `FableClarification`, and `ReviewVerdict` with the Task 1 `DirectedAgentQuestion` as an optional strict directed-question projection rather than free-form route text.

Add adapter calls that resume only an exact known provider identity:

```python
# ClaudeCLI: Fable answers a question raised by Sol.
async def answer_sol_question(
    self, *, run_id: str, session_id: str, task_id: str,
    prompt: str, context: str,
) -> AgentRunResult: ...
# CodexCLI: Sol answers a question raised by Fable.
async def answer_fable_question(
    self, *, run_id: str, thread_id: str,
    brief: TaskBrief, prompt: str,
) -> AgentRunResult: ...
```

Extend `FableAdapter` and `SolAdapter` in `adapters/base.py` with these exact keyword-only methods. `answer_sol_question` must return an `AgentRunResult` whose payload validates as `FableClarification` (the existing strict same-scope/scope-changed/revised-brief/escalate schema). `answer_fable_question` must return an `AgentRunResult` whose payload validates as `SolOutcome`. No untyped free-form guidance result is accepted.

- [ ] **Step 1: Add strict schema RED tests**

Assert exact fields, target enum, bounded text/reason, and no `routed_to`, path, command, environment, session, or thread fields in model output. Unknown/extra/malformed structures fail closed. Preserve every current no-question contract byte when the optional field is absent.

- [ ] **Step 2: Add exact-session/thread RED tests**

For Fable answers, assert `--resume` carries the exact validated session ID and no fresh conversation is started. For Sol answers, assert the exact validated thread ID is used and the original approved TaskBrief is included. Missing, malformed, leading-dash, conflicting, or returned mismatched IDs reject before partial result exposure.

- [ ] **Step 3: Add prompt-authority RED tests**

Assert prompts state that Fable owns intent/scope and may produce N+1, while Sol may clarify approved execution but cannot widen scope. Raw user/chat text remains prompt content only; it cannot alter CLI argv. Audit outputs remain allowlisted and bounded.

- [ ] **Step 4: Run RED, implement, and run adapter suites**

```bash
python -m pytest -q \
  tests/agent_bridge/test_contracts.py \
  tests/agent_bridge/test_claude_cli.py \
  tests/agent_bridge/test_codex_cli.py
```

- [ ] **Step 5: Commit**

```bash
git add \
  src/agent_bridge/contracts.py \
  src/agent_bridge/adapters/base.py \
  src/agent_bridge/adapters/claude_cli.py \
  src/agent_bridge/adapters/codex_cli.py \
  tests/agent_bridge/fixtures/fake_claude.py \
  tests/agent_bridge/fixtures/fake_codex.py \
  tests/agent_bridge/test_claude_cli.py \
  tests/agent_bridge/test_codex_cli.py
git diff --cached --check
git commit -m "feat: add directed agent question contracts"
```

---

### Task 4: Implement deterministic coordinator routing

**Files:**
- Modify: `src/agent_bridge/hub.py`
- Modify: `src/agent_bridge/coordinator.py`
- Modify: `tests/agent_bridge/test_hub.py`
- Modify: `tests/agent_bridge/test_coordinator.py`

**Interfaces:**

```python
class RoutingMode(str, Enum):
    NEW_FABLE_TASK = "new_fable_task"
    BOUND_CONTINUATION = "bound_continuation"

@dataclass(frozen=True, slots=True)
class RoutingDecision:
    addressed_to: ConversationTarget
    routed_to: ConversationTarget
    mode: RoutingMode
    task_id: str | None
    revision: int | None
    continuation_generation: int | None

def route_user_intent(
    authenticated_task: TaskRecord | None,
    intent: UserConversationInput,
) -> RoutingDecision: ...

def prepare_continuation_message(
    self, *, session_id: str, task_id: str, revision: int,
    continuation_generation: int, text: str,
    addressed_to: ConversationTarget,
) -> TaskRecord: ...
def prepare_question_answer(
    self, *, session_id: str, task_id: str, revision: int,
    continuation_generation: int, question_id: str, answer: str,
) -> TaskRecord: ...
def prepare_exchange_grant(
    self, *, session_id: str, task_id: str, revision: int,
    continuation_generation: int, request_id: str,
) -> TaskRecord: ...
async def run_prepared_conversation_action(self, task_id: str, action: str) -> None: ...
async def answer_directed_question(self, question: QuestionRecord) -> None: ...

# Add these typed methods to the Phase 1 HubWorkflowOrchestrator.
async def prepare_continuation_message(
    self, *, project_id: str, session_id: str, task_id: str,
    revision: int, continuation_generation: int, text: str,
    addressed_to: ConversationTarget,
) -> PreparedWorkflow: ...
async def prepare_question_answer(
    self, *, project_id: str, session_id: str, task_id: str,
    revision: int, continuation_generation: int,
    question_id: str, answer: str,
) -> PreparedWorkflow: ...
async def prepare_exchange_grant(
    self, *, project_id: str, session_id: str, task_id: str,
    revision: int, continuation_generation: int, request_id: str,
) -> PreparedWorkflow: ...
```

Keep `handle_user_request` and `answer_user_question` as compatibility wrappers that delegate through equivalent prepared state. Each hub preparation validates route/hub acknowledgment, acquires the only lease token before any child probe, performs the fresh complete readiness check under that token, passes every exact value unchanged into the synchronous coordinator/store CAS, releases on probe or preparation failure, and returns a `PreparedWorkflow`. Its asynchronous `run` consumes only the persisted pending action; it never trusts the original request object again.

- [ ] **Step 1: Add routing matrix RED tests**

Cover every sender/addressee/state combination. Required assertions: unbound user message routes Fable/new task; pre-approval `@Sol` persists `addressed_to=sol` but `routed_to=fable`; approved nonterminal valid Sol continuation routes exact Sol; terminal task creates new Fable task; user-addressed agent question pauses; agent attempts to answer it reject; unknown/stale binding does not fall back to a new task or another question.

- [ ] **Step 2: Implement and commit the pure routing decision**

Implement the typed side-effect-free `route_user_intent` above. It returns the exact route/mode/binding or raises one bounded routing error category; it never invokes an adapter or touches the lease. Run the routing-matrix selector and commit only the helper plus its tests:

```bash
python -m pytest -q tests/agent_bridge/test_coordinator.py -k routing_matrix
git add src/agent_bridge/coordinator.py tests/agent_bridge/test_coordinator.py
git diff --cached --check
git commit -m "feat: define deterministic team routing"
```

- [ ] **Step 3: Add visible ordering RED tests**

Install a store listener and fake adapter barrier. Assert question event commits and broadcasts before the answering adapter starts, answer commits before the originating agent resumes, and structural stream events remain separate `agent_event` records. Persist sender/addressed/routed/type/question/reply metadata on every new conversation event.

- [ ] **Step 4: Add budget-exhaustion RED tests**

Run three automatic exchanges and assert the fourth question does not invoke an adapter, transitions durably to user input, and emits a user-addressed permission card. Grant +3, retry, and assert only three more starts. Test crash/retry after reservation and after question event publication; reuse the same exchange ID without double charge or duplicate visible question.

- [ ] **Step 5: Add Fable/Sol authority RED tests**

Sol asks Fable under the exact approved revision. Fable same-scope guidance resumes the same Sol thread. Fable scope-changing response creates revision N+1 in `AWAITING_SCOPE_APPROVAL`; Sol is not invoked until exact approval. Fable may request evidence from Sol only after approval. Existing baseline remains authoritative across all paths.

- [ ] **Step 6: Run RED and implement durable routing/invocation**

```bash
python -m pytest -q tests/agent_bridge/test_coordinator.py -k "directed or exchange or addressed or routed"
```

Keep the pure routing decision limited to store-authenticated state, never raw browser/provider identities. The typed Phase 1 `HubWorkflowOrchestrator` preparation already completed the fresh combined readiness gate and owns one exact lease token for the complete public coordinator action; neither this coordinator code nor an adapter reacquires it. Recheck task state, revision, generation, stop intent, durable pending action, and provider identity after each await before routing onward.

- [ ] **Step 7: Run coordinator suite and commit durable invocation**

```bash
python -m pytest -q tests/agent_bridge/test_coordinator.py tests/agent_bridge/test_store.py
git add \
  src/agent_bridge/hub.py \
  src/agent_bridge/coordinator.py \
  tests/agent_bridge/test_hub.py \
  tests/agent_bridge/test_coordinator.py
git diff --cached --check
git commit -m "feat: run bounded team conversations"
```

---

### Task 5: Expose exact directed-message and answer APIs

**Files:**
- Modify: `src/agent_bridge/app.py`
- Modify: `tests/agent_bridge/test_web.py`

**Request DTOs:**

```python
class MessageRequest(BaseModel):
    text: StrictStr
    addressed_to: Literal["fable", "sol", "team"] = "fable"

class ContinuationMessageRequest(BaseModel):
    text: StrictStr
    addressed_to: Literal["fable", "sol"]
    revision: StrictInt
    continuation_generation: StrictInt

class QuestionAnswerRequest(BaseModel):
    text: StrictStr
    revision: StrictInt
    question_id: StrictStr
    continuation_generation: StrictInt

class ExchangeGrantRequest(BaseModel):
    revision: StrictInt
    continuation_generation: StrictInt
    request_id: StrictStr
```

The ordinary chat message route accepts only `MessageRequest` and always creates a new Fable-planned task, even when addressed to Sol. Add these exact bound routes while retaining Phase 1's answer path identity:

```text
POST /api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/messages
POST /api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/answer
POST /api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/exchanges/grant
```

The first uses `ContinuationMessageRequest`, the second `QuestionAnswerRequest`, and the third `ExchangeGrantRequest`. Pass `ExchangeGrantRequest.request_id` through the app, `HubWorkflowOrchestrator`, coordinator, and store unchanged. `routed_to`, sender, session/thread IDs, and continuation state are forbidden request fields.

- [ ] **Step 1: Add body/route boundary RED tests**

Reject extra fields, wrong scalar types, empty/oversized text, route IDs that differ from body ownership, stale revision/generation, and browser attempts to set `routed_to` or actor. Assert encoded body-only text never appears in URL/log/audit structures.

- [ ] **Step 2: Add direct-bypass RED tests**

Use authenticated CSRF-valid calls to prove a failure of any part of the existing combined model gate—usage acknowledgement, freshly probed exact Fable subscription readiness, or Sol readiness—plus wrong project/chat, held foreign lease, and user-question agent-answer attempts fail before coordinator preparation/scheduling. Stop alone remains ungated. A bound message, answer, or grant awaits its exact typed hub preparation and then uses the Phase 1 scheduler-install helper. A failed gate cannot consume a question/grant/message; forced scheduler rejection leaves the already prepared action interrupted/resumable, releases the token, and creates no unobserved coroutine.

- [ ] **Step 3: Add bounded bootstrap projection RED tests**

Project-chat bootstrap returns safe question cards and exchange allowance for the bounded task list, never continuation JSON, provider IDs, commands, env, raw stderr, or other project state. Include exact revision/generation/question IDs required for an answer and stable event replay metadata.

- [ ] **Step 4: Run RED, implement, and run web tests**

```bash
python -m pytest -q tests/agent_bridge/test_web.py -k "directed or question or exchange"
python -m pytest -q tests/agent_bridge/test_web.py
```

- [ ] **Step 5: Commit**

```bash
git add src/agent_bridge/app.py tests/agent_bridge/test_web.py
git diff --cached --check
git commit -m "feat: expose directed conversation APIs"
```

---

### Task 6: Render a readable addressed team conversation

**Files:**
- Modify: `src/agent_bridge/static/index.html`
- Modify: `src/agent_bridge/static/app.js`
- Modify: `src/agent_bridge/static/styles.css`
- Modify: `tests/agent_bridge/test_static_ui.py`
- Modify: `tests/agent_bridge/test_web.py`

- [ ] **Step 1: Add envelope projection RED tests**

Require visible `User → Fable`, `Sol → Fable`, `Fable → Sol`, and agent `→ You` labels, distinct safe actor avatars, reply association, task/revision context, and compact status rows. Unknown/ambiguous legacy events stay audit-only. Every DOM insertion uses `textContent`; actor/target classes come from fixed maps only.

- [ ] **Step 2: Add bound-question composer RED tests**

Render multiple pending question cards from different tasks. Selecting Reply binds project/chat/task/revision/question/generation, displays the binding, sends the exact answer endpoint, clears only after success, and fails closed on stale 409. Typing an unbound message always uses the new-message endpoint. Explicitly test that it does not answer the newest/first pending question implicitly.

- [ ] **Step 3: Add recipient and exchange-card RED tests**

The ordinary composer defaults to Fable, permits a visible Sol address, and explains when the coordinator routes pre-approval intent through Fable. At hop exhaustion render a user-addressed card with Reply and `Allow 3 more exchanges`; never render unlimited permission. Disable ordinary send under the active lease while preserving applicable answer controls.

- [ ] **Step 4: Run RED and implement**

```bash
python -m pytest -q tests/agent_bridge/test_static_ui.py tests/agent_bridge/test_web.py
```

Keep structural `agent_event` values in the inspector Activity projection, not the center conversation. Preserve existing collection bounds, reconnect generation checks, replay watermark logic, drawer accessibility, and focus behavior.

- [ ] **Step 5: Run GREEN and commit**

```bash
python -m pytest -q tests/agent_bridge/test_static_ui.py tests/agent_bridge/test_web.py
git add \
  src/agent_bridge/static/index.html \
  src/agent_bridge/static/app.js \
  src/agent_bridge/static/styles.css \
  tests/agent_bridge/test_static_ui.py \
  tests/agent_bridge/test_web.py
git diff --cached --check
git commit -m "feat: render directed team conversation"
```

---

### Task 7: Map expired subscription auth to fixed non-leaking guidance

**Files:**
- Modify: `src/agent_bridge/hub.py`
- Modify: `src/agent_bridge/adapters/claude_cli.py`
- Modify: `src/agent_bridge/coordinator.py`
- Modify: `src/agent_bridge/app.py`
- Modify: `tests/agent_bridge/test_hub.py`
- Modify: `tests/agent_bridge/test_claude_cli.py`
- Modify: `tests/agent_bridge/test_coordinator.py`
- Modify: `tests/agent_bridge/test_web.py`

- [ ] **Step 1: Add structured-signal RED tests**

Parameterize the small documented fake error categories that mean login is required. Assert only the fixed conversation text `Fable login expired. Run claude auth login on the host, then Resume.` is emitted. Raw stdout/stderr, account identity, tokens, JSON, command argv, and unmatched provider messages never appear in result, exception, persisted event, browser projection, or representation.

- [ ] **Step 2: Add fail-closed negative RED tests**

Unknown auth-like text remains the generic bounded adapter failure, substring spoofing does not select the login message, and nonzero/malformed auth preflight cannot start a model. Stop remains available and Resume rechecks the complete existing gate: usage acknowledgement, exact Fable subscription readiness, and Sol readiness.

- [ ] **Step 3: Implement an allowlisted error category**

Keep provider parsing in the adapter and user guidance selection in the coordinator. Return a structural enum/category only; never pass raw provider text across the boundary. For the allowlisted login-expired category, one store transaction moves planning, clarification, or review to `INTERRUPTED`, preserves its exact continuation/pending context and validated provider identity, and appends the fixed guidance event. The hub catches that structural category and calls the route-selected `RuntimeReadiness.invalidate_fable_subscription`; bootstrap immediately becomes unavailable. After host login, Resume awaits `prepare_existing_task`, whose bounded fresh Claude auth probe can restore subscription-ready and then rechecks the complete combined gate. Generic unmatched failures retain the existing failure behavior.

- [ ] **Step 4: Run and commit**

```bash
python -m pytest -q \
  tests/agent_bridge/test_hub.py \
  tests/agent_bridge/test_claude_cli.py \
  tests/agent_bridge/test_coordinator.py \
  tests/agent_bridge/test_web.py
git add \
  src/agent_bridge/hub.py \
  src/agent_bridge/adapters/claude_cli.py \
  src/agent_bridge/coordinator.py \
  src/agent_bridge/app.py \
  tests/agent_bridge/test_hub.py \
  tests/agent_bridge/test_claude_cli.py \
  tests/agent_bridge/test_coordinator.py \
  tests/agent_bridge/test_web.py
git diff --cached --check
git commit -m "fix: surface bounded Fable login guidance"
```

---

### Task 8: Verify directed collaboration end to end

**Files:**
- Modify: `tests/agent_bridge/test_e2e_fake_agents.py`
- Modify: `README.md`
- Create: `.superpowers/sdd/2026-08-11-agent-bridge/task-directed-team-conversation-report.md`

- [ ] **Step 1: Add fake-agent workflow RED tests**

Cover: user→Fable plan; pre-approval user→Sol visibly routed through Fable; approval then Sol→Fable question and exact Fable answer; Fable→Sol evidence question; user-addressed question rejecting fake agent answers; three exchanges then pause; +3 then resume; Fable same-scope response; Fable N+1 response awaiting exact approval; multiple task questions with exact binding; terminal Sol address starting a fresh Fable task.

- [ ] **Step 2: Add restart/idempotency RED tests**

Restart after exchange reservation, after visible question, after answer, and while paused at the limit. Assert exact exchange/question IDs, counts, continuation generation, Fable session, Sol thread, and message order survive without duplicate charges/calls/events.

- [ ] **Step 3: Run focused and full fake-only gates**

```bash
python -m pytest -q \
  tests/agent_bridge/test_contracts.py \
  tests/agent_bridge/test_store.py \
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

- [ ] **Step 4: Update docs and the evidence report**

Document recipient semantics, planner/executor authority, exact question replies, three-exchange limit/+3 permission, visible agent dialogue, subscription-login guidance, and that directed messaging never supplies paths/commands. Record exact tests, warnings, fake executable provenance, and zero live calls.

- [ ] **Step 5: Request Sol review and fix findings test-first**

Sol reviews every envelope/schema/routing/API/UI/fixture/doc change, with special attention to agent answers to user questions, Sol-before-approval, stale CAS fallback, budget races, event-before-invocation order, raw error leakage, and exact session/thread use. Freeze between review rounds and do not self-approve.

- [ ] **Step 6: Commit reviewed completion**

```bash
git add README.md tests/agent_bridge/test_e2e_fake_agents.py
git diff --cached --check
git commit -m "test: verify bounded directed collaboration"
git status --short --branch
```

## Phase 2 acceptance gate

Phase 2 is complete only when Sol returns READY with no Critical or Important findings, the complete fake-only suite passes, exact restart tests prove idempotent exchange/question state, and no path can invoke Sol before exact approval or answer a user-directed question as an agent. Do not begin Phase 3 from an unreviewed tree.
