# Conversation-Focused Browser UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Agent Bridge's central workspace behave like a chat by hiding structural telemetry from the conversation, while preserving audit persistence, preventing approval of briefs with unresolved questions, and making repository authority explicit.

**Architecture:** Keep adapters, coordinator event capture, SQLite persistence, WebSocket replay, and task reduction unchanged. Add a fail-closed browser presentation classifier that renders meaningful events as messages, task state changes as compact status rows, and structural agent events only through the existing task activity projection. Add matching client and API approval prechecks for `open_questions` while retaining the coordinator invariant.

**Tech Stack:** Python 3.11+, FastAPI, SQLite, browser-native ES modules, HTML/CSS, pytest, Node test harnesses.

## Global Constraints

- Work only in the standalone `agent-bridge` repository and its isolated feature worktree.
- Do not write any private source-project name, path, branding, state, or filesystem dependency into tracked content.
- Do not access, inspect, modify, run, or target any private source checkout.
- Preserve every persisted sanitized event and the existing SQLite/WebSocket audit stream.
- Do not add runtime repository switching; the sole repository authority remains selected at server startup.
- Preserve CSRF, keyed-session, subscription, usage-credit, Sol-readiness, exact-revision, repository-boundary, and coordinator state-machine checks.
- Unknown browser event kinds must fail closed and must not become conversational content.
- Use text-only DOM insertion and existing safe class allowlists; do not introduce `innerHTML`.
- Tests use fakes and local harnesses only. Do not invoke live agents, authentication, models, network services, or a browser server.
- Use red-green-refactor. Do not weaken, skip, or xfail an honest failing test.
- `gpt-5.6-terra` implements each task. `gpt-5.6-sol` reviews each task and the complete branch before acceptance.
- Stage explicit paths only; never use `git add -A`.

---

### Task 1: Separate conversation content from audit telemetry

**Files:**
- Modify: `src/agent_bridge/static/app.js`
- Modify: `src/agent_bridge/static/index.html`
- Modify: `src/agent_bridge/static/styles.css`
- Test: `tests/agent_bridge/test_static_ui.py`
- Test: `tests/agent_bridge/test_web.py`

**Interfaces:**
- Consumes: persisted event objects shaped as `{sequence, session_id, task_id, actor, kind, payload, created_at}` and the existing `renderMessage(documentRoot, event, associatedRevision)` function.
- Produces: exported `conversationPresentation(event) -> "message" | "status" | "hidden"` and `renderConversationEvent(documentRoot, event, associatedRevision = null) -> Node | null` browser helpers.
- Preserves: `reduceTaskEvent(tasks, event)` continues reducing `agent_event` into the task's bounded `activity` projection and history; persistence and WebSocket delivery do not change.

- [ ] **Step 1: Add failing presentation tests for hidden telemetry and compact task state**

In `tests/agent_bridge/test_static_ui.py`, update the persisted-event behavior harness so reduction still processes every known kind but presentation goes through `renderConversationEvent`. Add explicit assertions equivalent to:

```javascript
const agentEvent = events.find((event) => event.kind === "agent_event");
const taskState = events.find((event) => event.kind === "task_state");
if (bridge.conversationPresentation(agentEvent) !== "hidden") process.exit(20);
if (bridge.conversationPresentation(taskState) !== "status") process.exit(21);
if (bridge.conversationPresentation({kind: "future_event"}) !== "hidden") process.exit(22);

for (const event of events) {
  tasks = bridge.reduceTaskEvent(tasks, event);
  const task = tasks.find((item) => item.task_id === event.task_id);
  bridge.renderConversationEvent(
    documentRoot,
    event,
    bridge.associatedRevisionForEvent(event, task),
  );
}
if (tasks[0].activity.command_sha256 !== "hash-only") process.exit(23);
if (created.some((node) => node.textContent === "Command Execution")) {
  process.exit(24);
}
const statusRows = conversation.children.filter(
  (node) => node.className === "conversation-status",
);
if (statusRows.length !== 1 || !statusRows[0].textContent.includes("Sol Running")) {
  process.exit(25);
}
```

Keep the existing assertions that meaningful event details are rendered safely, but adjust counts and child indexes so `agent_event` has no conversation node and `task_state` has no `<details>` disclosure. Add a direct assertion that a user message followed by any number of structural `agent_event` values and then a TaskBrief produces exactly two message cards in that order.

- [ ] **Step 2: Add failing repository-authority presentation tests**

In `tests/agent_bridge/test_static_ui.py`, extend the semantic layout test to require an element with `id="repository-authority-note"` whose page text says:

```text
Selected at server startup. Messages cannot change repository authority.
```

In the existing controller harness in `tests/agent_bridge/test_web.py`, replace the terse repository assertion with:

```javascript
if (nodes["repository-status"].textContent !== "Repository: /repo · Branch: feat/agent-bridge") {
  process.exit(9);
}
```

The static layout test owns the note itself; the controller harness owns only
the dynamic repository label because it does not load `index.html`.

- [ ] **Step 3: Run the focused tests and verify the intended RED**

Run:

```bash
python -m pytest -q \
  tests/agent_bridge/test_static_ui.py::test_every_persisted_event_kind_reduces_and_renders_full_safe_details \
  tests/agent_bridge/test_static_ui.py::test_index_renders_semantic_option_a_layout_and_accessible_mobile_drawers \
  tests/agent_bridge/test_web.py::test_browser_controller_uses_exact_bootstrap_and_recovers_from_initial_failure
```

Expected: failures because `conversationPresentation`,
`renderConversationEvent`, the authority note, and explicit repository label
do not yet exist.

- [ ] **Step 4: Implement the fail-closed conversation classifier**

In `src/agent_bridge/static/app.js`, add immutable event-kind policy near the existing constants:

```javascript
const MESSAGE_EVENT_KINDS = new Set([
  "message",
  "task_brief",
  "clarification",
  "outcome",
  "review",
  "task_rejected",
  "action_error",
  "stop_error",
  "resume_drift",
]);

export function conversationPresentation(event) {
  if (event?.kind === "task_state") return "status";
  if (MESSAGE_EVENT_KINDS.has(event?.kind)) return "message";
  return "hidden";
}
```

Add a compact status renderer that uses `eventText(event)`, includes safe sequence/task/revision metadata as plain text, applies only the literal `conversation-status` class, removes the empty-state intro, enforces `MAX_CONVERSATION_MESSAGES`, and does not add structured `<details>`. Add:

```javascript
export function renderConversationEvent(documentRoot, event, associatedRevision = null) {
  const presentation = conversationPresentation(event);
  if (presentation === "hidden") return null;
  if (presentation === "status") {
    return renderConversationStatus(documentRoot, event, associatedRevision);
  }
  return renderMessage(documentRoot, event, associatedRevision);
}
```

Change the controller's `handleEvent` to call `renderConversationEvent`. Continue calling `reduceTaskEvent` first so hidden telemetry still updates `activity`, task history, and connection state. Scroll only when a node was rendered.

- [ ] **Step 5: Implement explicit repository authority presentation**

In `src/agent_bridge/static/index.html`, place this static note inside `.brand-block` immediately after `#repository-status`:

```html
<span id="repository-authority-note" class="repository-authority-note">
  Selected at server startup. Messages cannot change repository authority.
</span>
```

Change `renderStatus` in `app.js` to assign:

```javascript
const repository = typeof state.repository === "string" ? state.repository : "checking";
const branch = typeof state.branch === "string" ? state.branch : "checking";
repoNode.textContent = `Repository: ${repository} · Branch: ${branch}`;
```

Add restrained `.repository-authority-note` and `.conversation-status` rules in `styles.css`; the note must wrap on narrow screens and the status row must be visually smaller than `.message` without relying on dynamic class names.

- [ ] **Step 6: Run focused and composed browser tests**

Run:

```bash
python -m pytest -q tests/agent_bridge/test_static_ui.py tests/agent_bridge/test_web.py
```

Expected: all tests pass with only already-established third-party warnings.

- [ ] **Step 7: Commit Task 1 explicitly**

```bash
git add \
  src/agent_bridge/static/app.js \
  src/agent_bridge/static/index.html \
  src/agent_bridge/static/styles.css \
  tests/agent_bridge/test_static_ui.py \
  tests/agent_bridge/test_web.py
git diff --cached --check
git commit -m "fix: keep audit telemetry out of chat"
```

---

### Task 2: Block approval until open questions are resolved

**Files:**
- Modify: `src/agent_bridge/app.py`
- Modify: `src/agent_bridge/static/app.js`
- Test: `tests/agent_bridge/test_web.py`
- Test: `tests/agent_bridge/test_static_ui.py`

**Interfaces:**
- Consumes: `canonicalTaskBrief(task)`, approval-state task snapshots, `TaskBrief.open_questions`, and the existing `Coordinator.approve_task(task_id, revision)` invariant.
- Produces: a client-side disabled approval descriptor for unresolved questions and an HTTP 409 API response with exact detail `resolve the TaskBrief open questions before approval`.
- Preserves: Edit remains subject to the existing readiness gate; Reject remains available without model readiness; coordinator validation remains unchanged.

- [ ] **Step 1: Add a failing API test for direct approval bypass**

In `tests/agent_bridge/test_web.py`, import `replace` from `dataclasses` and add:

```python
def test_approval_rejects_unresolved_open_questions_before_scheduling(
    web_harness: WebHarness,
    valid_brief: TaskBrief,
) -> None:
    unresolved = replace(
        valid_brief,
        revision=2,
        open_questions=("Which path is authoritative?",),
    )
    web_harness.store.save_task(
        SESSION_ID,
        unresolved,
        TaskState.AWAITING_USER_APPROVAL,
    )
    with _authenticated_client(web_harness) as client:
        csrf = _acknowledge_model_usage(client)
        response = client.post(
            "/api/tasks/task-1/approve",
            json={"revision": 2},
            headers={"X-CSRF-Token": csrf},
        )
    assert response.status_code == 409
    assert response.json() == {
        "detail": "resolve the TaskBrief open questions before approval"
    }
    assert not any(call[0] == "approve" for call in web_harness.coordinator.calls)
```

- [ ] **Step 2: Add a failing inspector behavior test**

Extend the existing `renderTaskInspector` Node harness in
`tests/agent_bridge/test_static_ui.py` with an approval-state task whose exact
brief contains `open_questions: ["Which path is authoritative?"]`. Assert:

```javascript
const buttons = created.slice(unresolvedStart).filter((node) => node.tag === "button");
const approve = buttons.find((node) => node.textContent === "Approve & run");
const edit = buttons.find((node) => node.textContent === "Edit");
const reject = buttons.find((node) => node.textContent === "Reject");
if (!approve || !approve.disabled) process.exit(30);
if (!edit || edit.disabled) process.exit(31);
if (!reject || reject.disabled) process.exit(32);
const unresolvedText = created.slice(unresolvedStart).map((node) => node.textContent).join("\n");
if (!unresolvedText.includes("Resolve or remove the open questions in Edit before approval.")) {
  process.exit(33);
}
```

Pass a ready gate to this inspector render so the approval is disabled solely
because of unresolved questions.

- [ ] **Step 3: Run the two selectors and verify the intended RED**

Run:

```bash
python -m pytest -q \
  tests/agent_bridge/test_web.py::test_approval_rejects_unresolved_open_questions_before_scheduling \
  tests/agent_bridge/test_static_ui.py::test_safe_rendering_preserves_untrusted_task_and_message_text
```

Expected: the API schedules approval and the browser enables Approve, so both
regressions fail for the intended reasons.

- [ ] **Step 4: Add the synchronous API approval precheck**

In `src/agent_bridge/app.py`, after exact-revision validation and before
`require_model_start_ready`, add:

```python
if task.brief is None or task.brief.open_questions:
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="resolve the TaskBrief open questions before approval",
    )
```

Do not remove or alter the coordinator's existing `open_questions` check.

- [ ] **Step 5: Disable approval and explain the resolution path in the inspector**

In `renderTaskInspector`, derive:

```javascript
const hasOpenQuestions = brief !== null && brief.open_questions.length > 0;
```

When `approvalState && hasOpenQuestions`, append a text section titled
`Approval blocked` with exact text:

```text
Resolve or remove the open questions in Edit before approval.
```

When constructing action definitions, replace only the approval descriptor
with `control(true, false)` when `hasOpenQuestions`; leave Edit, Reject, Stop,
Resume, and Answer rules unchanged. A missing or identity-mismatched brief must
continue using the existing stronger `hasUnusableBrief` fail-closed behavior.

- [ ] **Step 6: Run focused approval, browser, and coordinator invariants**

Run:

```bash
python -m pytest -q \
  tests/agent_bridge/test_static_ui.py \
  tests/agent_bridge/test_web.py \
  tests/agent_bridge/test_coordinator.py
```

Expected: all tests pass. The coordinator's existing unresolved-question test
must remain green without modification.

- [ ] **Step 7: Run complete fake-only verification and static checks**

Run:

```bash
python -m pytest -q tests/agent_bridge
python -m compileall -q src/agent_bridge tests/agent_bridge
node --experimental-default-type=module -e \
  "import('./src/agent_bridge/static/app.js')"
git diff --check
```

Expected: the complete fake-only suite and all static checks pass. No live CLI,
network, model, server, or target-repository process may be invoked.

- [ ] **Step 8: Commit Task 2 explicitly**

```bash
git add \
  src/agent_bridge/app.py \
  src/agent_bridge/static/app.js \
  tests/agent_bridge/test_static_ui.py \
  tests/agent_bridge/test_web.py
git diff --cached --check
git commit -m "fix: require resolved questions before approval"
```

---

## Final branch gate

After both task-scoped Sol reviews are clean, generate one exact base-to-head
review package and dispatch a fresh `gpt-5.6-sol` whole-branch review. Resolve
every Critical or Important finding test-first and request a scoped re-review.
Only after a clean final verdict should the controller rerun the complete
fake-only suite, inspect the exact committed paths and status, and offer the
user integration/restart choices. Do not push, merge, restart a server, or
change a deployment without separate authorization.
