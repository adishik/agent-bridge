# Conversation-Focused Browser UX Design

## Problem

Agent Bridge currently sends every persisted event through the same browser
message renderer. A single planning run can produce many sanitized structural
`agent_event` records whose useful payload is only an event type. Rendering
each record as a full message card makes the chronological workspace resemble
an audit dump, pushes the completed TaskBrief out of view, and obscures the
actual Fable–user–Sol conversation.

The approval UI also treats every approval-state TaskBrief alike. It can offer
**Approve & run** when `open_questions` is non-empty even though the coordinator
correctly rejects that approval. Because mutations are scheduled
asynchronously, the user then sees a generic action failure instead of a clear
instruction to resolve the questions.

Finally, the existing repository status is visually terse. The server is bound
to exactly one repository at startup, but the browser does not state plainly
that message text cannot change that authority.

## Goals

- Make the central workspace read as a conversation rather than raw telemetry.
- Preserve the complete persisted, sanitized audit stream.
- Keep meaningful lifecycle information visible without allowing it to bury
  user and agent work products.
- Prevent approval while a TaskBrief has unresolved open questions.
- Make the server's active repository authority explicit.
- Preserve every existing security, billing, exact-revision, and repository
  boundary.

## Non-goals

- Changing which adapter events are captured or stored.
- Changing the TaskBrief, coordinator state machine, or clarification protocol.
- Adding runtime repository switching or interpreting a repository name in a
  prompt as authority to access another checkout.
- Migrating existing browser state or SQLite events.
- Calling live agents, starting a server, or changing a deployment as part of
  implementation verification.

## Decision

### Conversation projection

The browser will classify persisted events before rendering them:

- `message`, `task_brief`, `clarification`, `outcome`, `review`,
  `task_rejected`, `action_error`, `stop_error`, and `resume_drift` remain
  meaningful conversation entries.
- `task_state` becomes a compact status row rather than a full actor message.
- `agent_event` never becomes a conversation entry. It continues to update the
  task's bounded latest-activity projection in the inspector and remains in
  SQLite and WebSocket replay for audit and state reconstruction.

Rendered conversation entries retain their safe structured-event disclosure.
Suppressed telemetry is not deleted, transformed, or made less available to
the existing persistence and inspector paths. Filtering occurs only at the
browser presentation boundary, including both bootstrap replay and live
WebSocket delivery.

A completed `task_brief` will therefore appear immediately after the planning
request once intervening telemetry is omitted. Its full contract remains in
the inspector.

### Approval with open questions

For an approval-state task whose exact current brief contains one or more
`open_questions`:

- **Approve & run** is visible but disabled;
- **Edit** and **Reject** retain their existing availability;
- the inspector displays a concise instruction to resolve or remove the open
  questions by editing the next exact revision before approval; and
- the API rejects a direct authenticated approval request synchronously with
  HTTP 409 and a bounded, non-secret explanation.

The coordinator's existing fail-closed validation remains unchanged as the
final invariant. The new API validation is an earlier user-facing check, not a
replacement.

### Repository authority

The existing repository status will use explicit labels such as
`Repository: <path> · Branch: <branch>`. A short adjacent note will state that
the repository is selected at server startup and cannot be changed by a chat
message. The path and branch continue to come only from the trusted bootstrap
status provider; no prompt content is parsed for repository selection.

### Errors and safety

Presentation filtering must fail closed: unknown event kinds are not promoted
to conversational content. Existing bounded client collections, text-only DOM
insertion, safe class allowlists, CSRF checks, readiness checks, session
identity, and exact-revision approval remain intact.

The API approval precheck returns only a fixed explanation. It does not expose
exception text, repository content, commands, environment variables, or agent
output. Other asynchronous failures continue to use the existing sanitized
`action_error` event.

## Alternatives considered

1. **Filter only at the browser presentation boundary (chosen).** This fixes
   the user experience while preserving the complete audit and replay stream.
2. **Discard low-level events before persistence.** This reduces event volume
   but weakens auditability and changes adapter/coordinator behavior.
3. **Build a separate full audit-log screen.** This may be useful later, but it
   adds navigation, storage queries, and another UI surface beyond the observed
   problem. The bounded latest-activity inspector is sufficient for this fix.

## Testing

Implementation will proceed test-first with fake/local tests only:

- browser behavior tests proving structural `agent_event` records do not
  create conversation cards during bootstrap or live replay;
- browser behavior tests proving meaningful events still render and
  `task_state` uses a compact status row;
- inspector tests proving unresolved questions disable approval, retain Edit
  and Reject, and show the resolution guidance;
- API tests proving a direct approval with unresolved questions returns HTTP
  409 without scheduling the coordinator;
- status tests proving the explicit repository/branch labels and startup-bound
  authority note;
- bounded-history and safe-text regression coverage; and
- the complete fake-only `tests/agent_bridge` suite.

No live CLI, authentication, model prompt, network call, browser server, target
repository mutation, or deployment action is permitted during verification.

## Implementation and review

The change is expected to remain within the browser assets, the approval API
boundary, and their focused tests. Terra will implement it in the isolated
feature worktree using red-green-refactor. Sol will review the exact diff and
all fixes before the branch is eligible for integration. Integration and any
operator restart require a separate, explicit completion decision.
