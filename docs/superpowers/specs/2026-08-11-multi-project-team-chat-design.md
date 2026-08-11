# Multi-Project Team Chat Design

## Status

Approved in conversation on 2026-08-11. This document defines the product and
security design only; implementation is split into independently reviewed
phases.

## Problem

Agent Bridge currently binds one repository and one active browser session at
startup. Its browser is task-oriented: a user sends a planning request, Fable
produces a TaskBrief, the user approves an exact revision, and Sol executes it.
The recent conversation projection removes structural telemetry from the main
workspace, but the product still does not feel like an enduring conversation
with a team.

The intended experience has four missing capabilities:

1. one trusted server must expose an explicit startup allowlist of projects;
2. each project must support multiple persistent chats and a clear New Chat
   action;
3. user, Fable, and Sol questions must name their recipient, and visible
   Fable–Sol collaboration must not become an unbounded hidden loop; and
4. a user must be able to stop an active run, direct guidance to the correct
   agent, and resume the exact saved session or thread safely.

These capabilities must preserve the existing authority model. Fable plans and
reviews but does not write the repository. Sol executes but never invents its
own scope. Only an exact user-approved TaskBrief revision grants Sol mutation
authority.

## Goals

- Provide one Slack-style browser workspace for multiple explicitly
  allowlisted projects.
- Keep every project's state, repository authority, agent sessions, tasks, and
  audit stream isolated.
- Support multiple persistent chats per project without discarding old tasks or
  events.
- Render user, Fable, Sol, and system messages as a readable team
  conversation.
- Persist explicit sender, recipient, question, answer, task, and revision
  associations.
- Allow bounded, visible Fable–Sol questions while ensuring questions addressed
  to the user can only be answered by the user.
- Let the user explicitly Stop or Intervene during an active Fable or Sol run.
- Preserve exact validated Fable session and Sol thread continuity across
  questions, interruption, restart, and resume. When interruption happens
  before Fable publishes a session ID, restart planning from deterministic
  persisted context and show the discontinuity instead of guessing an ID.
- Retain every current authentication, subscription, usage-credit,
  exact-revision, repository-delta, CSRF, and audit boundary.
- Give Agent Bridge a distinct warm-copper visual identity rather than copying
  Slack branding.

## Non-goals

- Browser entry of arbitrary repository paths.
- Runtime discovery or automatic scanning of repositories.
- Sol planning, scope definition, or execution before approval.
- Fable repository mutation.
- Simultaneous autonomous workflows across projects in the first release.
- Unlimited agent-to-agent conversation.
- A general-purpose social chat system, direct messages, attachments, search,
  reactions, or notifications.
- Changing provider billing or adding API-key/provider fallbacks.
- Calling live agents, provider authentication, network services, or target
  repositories during implementation tests.

## Alternatives considered

### One trusted multi-project hub — chosen

One foreground server owns an immutable startup project registry. Each project
has an isolated runtime, while one authenticated browser surface presents the
projects and chats. This provides the intended team experience without
allowing the browser to grant filesystem authority.

### Supervisor with one child server per project

A supervisor could start a complete child Agent Bridge process for every
project and reverse-proxy their APIs. This provides stronger process isolation,
but requires port allocation, child health checks, proxy authentication,
multi-process recovery, and more complex deployment. The existing component
boundaries already allow strong in-process isolation without that cost.

### One independent URL per project

Running the existing application separately for each repository has the
simplest backend. It does not provide a shared project/chat sidebar, a coherent
New Chat workflow, or the requested team experience.

## Architecture

### Immutable startup registry

The launcher accepts either:

```text
agent-bridge --repo /absolute/git/root
```

or one or more explicit entries:

```text
agent-bridge \
  --project frontend=/absolute/frontend/root \
  --project backend=/absolute/backend/root
```

`--repo` remains a compatible single-project shortcut. It is mutually
exclusive with `--project`. Project labels must match a small ASCII identifier
grammar, must be unique case-insensitively, and are display labels rather than
filesystem authority. Canonical roots must be unique, absolute, readable Git
top levels. Symlink aliases and duplicate canonical roots are rejected.

The registry is immutable for the process lifetime. Adding or removing a
project requires restarting the server with a new explicit allowlist. Browser
requests contain only opaque project IDs created from the canonical root
identity; a label, chat message, query parameter, or request body can never
introduce a path.

### Project runtime

Every registry entry creates a `ProjectRuntime` with one clear ownership
boundary:

- canonical repository root and safe display metadata;
- state directory and SQLite store;
- repository tracker and external baseline artifacts;
- event broadcaster;
- process runner;
- Fable and Sol adapters bound to that repository;
- coordinator; and
- a readiness service with a safe projection plus bounded fresh provider
  validation before each model-starting action.

Startup preflight seeds the displayed status but is not lifetime authority.
Every model-starting preparation performs a fresh bounded Claude subscription
check and the complete existing Sol-readiness and usage-credit gate. A
structured expired-login result invalidates that project's Fable status
immediately; after the operator logs in, Resume performs a new check rather
than trusting the startup snapshot or requiring a server restart.

New project state lives below:

```text
$XDG_STATE_HOME/agent-bridge/projects/<root-digest>/
```

The directory identity depends only on the exact canonical root. The display
label is stored as metadata and may be renamed without selecting a different
database. The digest prevents same-basename repositories from colliding. Two
labels for the same root are rejected.
Every selected state path is validated first. The hub and project locks are
then acquired in stable project-ID order before any database is opened,
migrated, audited, or recovered. Failure to validate or lock any configured project aborts startup and
releases every acquired resource; the server never starts with a silently
partial authority set.

The current keyed browser session and CSRF boundary remain hub-wide. Project
stores do not share sessions, tasks, events, agent runs, baselines, or
continuations. A small hub-level settings store owns only account-level browser
settings such as the usage-credit acknowledgement. Existing installations ask
for that acknowledgement once after migration rather than inferring it from
one project's database.

### Concurrency policy

The first release has one hub-wide active-agent lease. There may be many chats
and many tasks awaiting approval or user input, but only one Fable or Sol
process may run at a time across the hub. While the lease is held, the browser
disables project switching, chat switching, and New Chat; the server enforces
the same rule so a second tab cannot bypass it. Stop, Intervene, and the pending
question response remain available where applicable.

This intentionally conservative policy keeps provider session ownership,
process cancellation, user attention, and repository authority unambiguous.
Parallel project execution is a future design, not an accidental side effect.

## Projects and chats

Each project has multiple sessions, presented to the user as chats. A New Chat
request creates a cryptographically random session ID permanently bound to that
project. The initial title is `New chat`; after the first user message the store
derives a bounded deterministic title from that message. Manual rename is not
part of the first release.

The sidebar lists chats by latest persisted event sequence. It is bounded and
pageable rather than materializing unlimited history. An old chat may be
reopened and may receive a new planning request when it has no active run.
Creating a new chat never deletes or resets an old one.

Existing single-project session, task, event, run, baseline, and continuation
rows remain in their current project database. Additive migrations provide chat
metadata and indexes. The existing active session becomes the initially
selected historical chat. No event history is rewritten.

## Browser and API boundaries

The authenticated bootstrap returns a bounded list of safe project summaries
and the selected project's bounded chat list. A project summary includes only
its opaque ID, configured label, repository display path, branch, readiness,
and whether an active run holds the hub lease.

Project-aware routes include both project and chat identity, for example:

```text
GET  /api/projects
POST /api/projects/{project_id}/chats
GET  /api/projects/{project_id}/chats/{session_id}/bootstrap
POST /api/projects/{project_id}/chats/{session_id}/messages
POST /api/projects/{project_id}/chats/{session_id}/tasks/{task_id}/intervene
GET  /ws?project_id=...&session_id=...&after=...
```

The route-selected project ID first chooses exactly one runtime; no other store
is queried. Lookup then occurs only inside that runtime, and the returned row
must join to the named chat, task revision, run, and continuation before any
write, scheduling, or process action. Cross-project identifiers therefore fail
without probing another runtime. Existing single-project browser routes may map
to the sole default project for one compatibility release, but multi-project
browser code uses only the project-aware routes.

Each project has its own broadcaster. WebSocket cursors and replay queries are
therefore scoped by both runtime and session, preventing equal or hostile
session IDs from crossing project boundaries. Existing replay pagination,
fresh-page floor, evidence projections, and client collection limits remain
bounded.

## Team conversation model

### Directed envelope

Every new conversational event carries a validated safe envelope:

- `sender`: `user`, `fable`, `sol`, or `system`;
- `addressed_to`: the user-declared or agent-declared addressee (`user`,
  `fable`, `sol`, or `team`);
- `routed_to`: the coordinator-derived effective recipient (`user`, `fable`,
  `sol`, or `team`), which browser input cannot set;
- `message_type`: `statement`, `question`, `answer`, `approval`,
  `intervention`, or `status`;
- project and chat identity supplied by the owning runtime;
- task ID and exact revision when applicable;
- question ID and reply-to question ID when applicable; and
- bounded human-readable text or an existing bounded structured projection.

The envelope is represented inside the existing ordered event stream. Legacy
events keep their stored bytes and receive a conservative presentation-only
projection based on their actor and known event kind. Unknown or ambiguous
legacy events are audit-only rather than guessed into a conversation.

The task's existing durable pending continuation remains the authority for who
may answer and what resumes. Browser envelope fields never select a process or
continuation by themselves.

### Deterministic routing

- An unbound ordinary user message always starts a new Fable-planned task.
  Every message intended for an existing continuation must name the exact
  project, chat, task, revision, and continuation generation and pass a store
  compare-and-swap before routing.
- A user address to Sol before exact approval records `addressed_to=sol` but
  `routed_to=fable`; Sol is not invoked.
- Sol may be routed a message only for an exact approved, nonterminal revision
  in a valid Sol continuation state. Addressing Sol after terminal completion
  begins a new Fable-planned task instead of reusing old approval.
- A reply to a pending question includes exact project, chat, task, revision,
  question ID, and continuation generation. The store compare-and-swaps all of
  them before resuming the agent and continuation that created the question.
- A question with `addressed_to=user` and coordinator-validated
  `routed_to=user` transitions durably to user input and can only be answered
  by the authenticated user endpoint. Neither agent may manufacture that
  answer.
- Agent-to-agent questions reuse the exact persisted Fable session and Sol
  thread; they do not start unrelated model conversations.

Sol may ask Fable about approved intent, ambiguity, or scope. Fable may answer
from the exact approved brief, request evidence from Sol after approval, or
produce revision N+1 when the answer changes scope. Sol remains blocked from
the widened scope until the user approves N+1.

### Bounded internal dialogue

One internal exchange is one agent question plus the other agent's single
answer. Each exchange has a persisted unique ID. Before emitting or invoking an
automatic question, the coordinator atomically reserves one of the revision's
three exchange slots. Retry and recovery reuse that exchange ID, so they cannot
double-charge or bypass the budget. Once a question has reserved a slot, its
single paired answer is always permitted.

At the limit the coordinator pauses only before initiating a fourth automatic
agent-to-agent question and emits a user-addressed card. Required planning,
execution, review, and correction calls do not consume this budget unless they
initiate an agent-to-agent question. The user may answer directly, redirect the
work, or authorize exactly three additional exchanges. There is no unlimited
approval. A user answer or intervention resets the remaining allowance to
three because the human has supplied new direction. User approval of a new
TaskBrief revision also starts a fresh three-exchange budget for that revision.

Every agent question and answer is emitted to the conversation before the next
agent runs. Structural CLI events remain persisted but appear only in the
collapsed Activity/Audit view.

## Stop, intervention, and resume

### Separate actions

`Stop` durably marks the exact active task and run interrupted, terminates the
owned process group, waits for bounded adapter finalization, and does not
resume automatically.

`Intervene` collects a non-empty bounded user message and an eligible recipient.
Before sending any process signal, one immediate transaction creates an
idempotent intervention record containing its ID, message, addressed/effective
recipient, project, chat, task, exact revision, active run, continuation state,
continuation generation, and any validated provider session/thread ID. The same
transaction persists stop intent. The endpoint reports acceptance only after
that transaction commits.

The server then:

1. stops only the recorded owned active run and waits for finalization;
2. marks the intervention ready against the still-exact continuation; and
3. invokes the eligible exact agent session or thread.

Every recovery path can therefore expose the committed intervention as pending
Resume, including a crash before or during process termination. Repeating the
request with the same intervention ID cannot duplicate the message, signal, or
resume-attempt scheduling record.

Before invoking a resumed provider session or thread, the bridge persists an
exact resume-attempt ID and run ID. Claude Code and Codex CLI do not expose a
transactional provider idempotency key, so a process crash after the provider
accepts a resume prompt but before the bridge records the result has an
unknowable outcome. Recovery marks that attempt `resume_outcome_unknown` and
never replays it automatically. The browser explains that the prior attempt
may have executed and requires the user to authorize a new attempt explicitly.
This fail-closed boundary prevents the bridge from claiming impossible
exactly-once provider execution or silently duplicating Sol mutations.

Stop wins all post-run routing races. A late adapter result may be retained as
sanitized evidence but cannot move an interrupted task to completion, review,
or another active agent state.

### Recipient rules

- Fable always has authority eligibility for an intervention because it owns
  intent and scope, but invocation readiness depends on continuation evidence.
  With a persisted validated Fable session ID it resumes that exact session
  with the current brief, partial evidence, and user guidance. If planning was
  interrupted before Fable published an ID, the intervention remains durable
  and Resume starts a new Fable session from the persisted original request,
  task/revision context, and intervention. The conversation displays that
  explicit session discontinuity.
- Sol is eligible only after an exact revision has been approved and a valid
  Sol continuation exists. It resumes the exact thread under the same approved
  brief.
- During pre-approval planning, Sol is visibly unavailable as an intervention
  recipient.
- If Sol determines the requested guidance exceeds approved scope, it must ask
  Fable instead of acting. Repository delta validation remains the final
  fail-closed boundary.
- Fable may return same-scope execution guidance or revision N+1. Revision N+1
  always waits for exact user approval before Sol resumes.

Partial sanitized output, session/thread IDs, baseline identity, question
budget, and nested clarification/review continuations survive intervention and
restart. The baseline continues to cover all changes since approval; an
intervention never creates a new clean baseline around partial work.

## Browser experience

### Desktop layout

The selected layout is a three-pane team workspace:

- the left rail contains allowlisted projects, persistent chats, New Chat, and
  Fable/Sol presence;
- the center contains the chronological human-readable conversation and
  composer; and
- the right inspector contains the selected task's exact revision, state,
  scope, required tests, question budget, controls, and collapsed Activity/Audit
  disclosure.

Fable and Sol have distinct names, avatars, and sender-to-recipient labels.
Agent-to-agent messages look like team conversation rather than telemetry.
Task state changes are compact timeline notices. TaskBrief approval, edit,
questions, corrections, intervention, and extra-hop permission are focused
inline cards.

The ordinary composer identifies Fable as the default recipient. A chat may
contain multiple tasks awaiting user input, with at most one pending question
per task revision. Each question card has its own Reply action. Selecting it
binds the composer to the exact task, revision, question ID, and continuation
generation; a stale answer fails compare-and-swap and does not fall through to
another question. An unbound ordinary message never answers a pending question.
While an agent runs, ordinary sending is disabled and an explicit Intervene
control replaces it; Stop remains separate so interruption cannot happen
accidentally.

### Visual identity and accessibility

The approved palette is warm copper: ink and cream surfaces, copper primary
actions, and muted green status/secondary accents. It preserves the familiar
spatial usefulness of a team workspace without Slack purple, logos, names, or
copied branding.

On narrow screens the conversation stays primary. Project/chat navigation and
the task inspector become keyboard-accessible drawers. The implementation must
retain semantic headings and labels, visible focus, sufficient contrast,
non-color state indicators, reduced-motion respect, safe text-only DOM
insertion, and bounded rendered collections.

## Failure behavior

- Invalid project configuration, duplicate roots, unsafe state paths, or lock
  failure abort startup before recovery or server binding.
- A runtime readiness failure appears on that project and blocks model-starting
  actions without exposing another project's state.
- Stale, unknown, or cross-project IDs are looked up only in the route-selected
  runtime and return bounded 404/409 responses before any write, task
  scheduling, or process action.
- Project/chat switching and New Chat fail synchronously while the active-agent
  lease is held.
- Hop exhaustion pauses durably and asks the user; it never silently drops a
  question or continues past the limit.
- Expired Fable OAuth is mapped from a small allowlist of structured failure
  signals to a fixed instruction to run `claude auth login` on the host. Raw
  provider output, tokens, account identity, and stderr remain excluded.
- Unexpected adapter, process, repository, or persistence errors emit bounded
  system conversation messages and retain safe details in Activity/Audit.
- A crash during a persisted provider resume attempt marks its outcome unknown,
  never auto-replays it, and requires explicit user acknowledgment that the
  prior attempt may already have executed before a new attempt is created.
- Restart recovery interrupts only the latest active revision in each project,
  retires every stale running run, and never sends a signal to persisted PIDs.
- Explicit Resume is required after recovery.

## Security invariants

- The browser never creates repository authority.
- Every project path comes from immutable validated startup configuration.
- Fable uses the saved subscription login; the bridge has no provider/API-key
  fallback.
- Account usage-credit acknowledgement and exact subscription readiness remain
  required for model-starting mutations.
- Sol is an executor only and cannot run before exact TaskBrief approval.
- The exact approved revision, repository baseline, allowed paths, required
  tests, and trusted executable checks remain authoritative.
- Stop remains available when model readiness is unavailable.
- The authenticated route-selected project chooses the only runtime queried;
  session and task ownership are verified from that runtime before scheduling,
  persistence, or process action.
- Conversation text is not command, path, executable, environment, or
  credential input.
- Audit persistence keeps safe structural events; chat projection does not
  weaken auditability.

## Migration and compatibility

SQLite changes are additive and transactional. Migration is idempotent and
tested from the current schema. Existing task revisions, events, agent runs,
session IDs, baseline settings, and continuation JSON keep their exact
semantics. New chat metadata is backfilled conservatively without model calls.

The current single-project launcher remains supported through `--repo`. When a
legacy basename-based state directory already exists, adoption occurs before
recovery in one read transaction. Every persisted session must bind to the
configured canonical root; the active-session setting must reference one of
those sessions; tasks and events must join to sessions in that same database;
runs must join to exact task revisions; baseline settings must reference the
owning task/revision; and SQLite foreign-key plus application-level integrity
checks must pass. It does not move the database or artifacts, because persisted
baseline references may be absolute. Any mixed root, orphan, absent binding,
invalid active-session reference, or ambiguous relationship fails closed.
Newly registered projects use the digest-only layout. The account-level
usage-credit acknowledgement is deliberately requested once after the hub
upgrade rather than copied from an arbitrary project store.

No runtime migration moves repository files, baseline artifacts, provider
state, or CLI session data. A project removed from the next startup allowlist
keeps its external state on disk but is inaccessible until explicitly
allowlisted again.

## Delivery phases

### Phase 1: Project and chat foundation

- immutable project registry and CLI parsing;
- collision-safe project state and all-or-nothing lock acquisition;
- isolated runtime assembly and routing;
- hub-wide active-agent lease and settings;
- persistent chat creation/listing/selection;
- project-aware API and WebSocket boundaries; and
- compatibility/migration tests.

### Phase 2: Directed team conversation

- validated conversation envelope;
- explicit addressed-to, routed-to, and reply association;
- Sol-executor-only routing;
- persisted agent-to-agent exchange budget;
- user permission for three more exchanges;
- visible agent questions/answers and bounded legacy projection; and
- fixed, non-leaking authentication guidance.

### Phase 3: Intervention and workspace

- durable Stop/Intervene distinction;
- exact-session/thread directed resume;
- scope-change return to Fable and revision approval;
- stop/result/restart race handling;
- approved three-pane responsive UI;
- warm-copper visual system; and
- Activity/Audit disclosure and accessibility hardening.

Each phase is independently test-first, reviewed, and mergeable. A later phase
does not weaken an earlier phase's safety gate.

## Testing

All tests use temporary local repositories, fake executables, isolated state,
and deterministic clocks. They do not use live Claude, Codex, provider
authentication, network access, paid services, a deployed browser server, or a
user project.

Required coverage includes:

- project-label grammar, canonical-root duplication, same-basename separation,
  lock ordering, partial-startup cleanup, and path-input rejection;
- two temporary Git repositories proving stores, sessions, events, baselines,
  WebSockets, commands, and mutations never cross projects;
- current-schema migration, idempotence, rollback, preserved legacy rows, chat
  title/recency pagination, and New Chat isolation;
- hub lease enforcement across APIs and multiple browser clients;
- exact sender/addressed-to/routed-to/question/reply validation;
- questions to the user rejecting agent answers;
- Sol never invoked before approval, including explicit user addressing;
- atomic exchange reservation and ID reuse, a guaranteed paired answer, limit
  pause only before a new question, permission for exactly three more, and reset
  on user direction/new approved revision;
- Fable same-scope answers versus revision N+1;
- intervention-record commit before signaling, idempotent retry, Stop and
  Intervene ordering, exact process ownership, partial events, crash recovery,
  early Fable interruption without a session ID, visible new-session fallback,
  unknown resume-outcome acknowledgment, stale completion, nested
  continuations, and baseline preservation;
- expired-auth fixed guidance without raw error leakage;
- project-aware CSRF, keyed session, stale/cross-project ID, replay pagination,
  reconnect, and bounded bootstrap behavior;
- desktop/mobile layout, drawers, keyboard focus, contrast-independent state,
  warm-copper tokens, text-only rendering, and bounded DOM history;
- full fake-agent end-to-end workflows across two projects; and
- the complete existing Agent Bridge suite after every phase.

Implementation defaults to Terra. Sol reviews every implementation, test,
documentation, and fix change and performs the final whole-branch review. No
implementation is accepted on implementer self-review alone.

## Success criteria

The feature is complete when an operator can start one loopback server with two
explicit project entries, open one authenticated warm-copper workspace, create
and revisit multiple chats, converse visibly with Fable and Sol, approve exact
Fable plans, observe bounded agent collaboration, safely intervene in an active
run, and resume the exact continuation—while adversarial fake-only tests prove
that neither the browser nor either agent can cross project, recipient, scope,
revision, process, or credential boundaries.
