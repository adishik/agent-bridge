# Agent Bridge Final Hardening Design

**Status:** Approved

## Goal

Close the three load-bearing findings that remained after the Phase 1 final
review without adding features or changing the product workflow.

## Scope discipline

Before each implementation task, ask: “Am I over-engineering this?” The work
is limited to the three reproduced blockers below. If a fix requires a new
product capability or an architectural decision not described here, stop and
ask Adi rather than expanding the design.

Out of scope:

- the deferred WebSocket cleanup and coroutine-observation Minors;
- the SQL `LIKE` Minor found during the final re-review;
- new UI behavior, provider features, project semantics, or public APIs;
- live provider, authentication, network, or browser-server tests.

## 1. Exact prepared-action authentication

Legacy ownership audit must authenticate every recovery-capable prepared row
against the exact persisted task and pending continuation before startup
recovery may mutate it. Action name alone is insufficient.

The audit will apply the same context-to-state rules used by normal prepared
action creation and Resume:

- Answer continuations must match the task's exact continuation state and the
  corresponding `ScopeApprovalContext`, `SolResumeContext`, or `ReviewContext`.
  Sol continuations may target either `SOL_RUNNING` or the existing valid
  `SOL_CORRECTING` state. A prepared Answer cannot validly target
  `FABLE_CLARIFYING`: the finite state machine has no
  `AWAITING_USER_INPUT -> FABLE_CLARIFYING` edge, so the legacy audit rejects
  such rows rather than adding a new workflow in this hardening task.
- Approval and Resume retain their existing baseline, predecessor, generation,
  and pending-projection checks.
- A mismatched context type, Fable session, Sol thread, baseline, revision, or
  continuation state fails the legacy audit generically before recovery. When
  the referenced agent-run row exists, its task, revision, and CLI session must
  also match. A missing agent-run row remains valid because prepared Store
  workflows legitimately exist before process-run persistence; the audit must
  not invent or require authority that the durable model does not guarantee.
- Prompts and run identifiers remain bounded and typed by the prepared row
  schema. They are compared only where an independent durable copy exists; the
  audit must not falsely claim that the prepared row authenticates itself.

This remains a read-only audit. It must not repair or normalize corrupt rows.

## 2. File-scoped provider schema capability

Sol needs read access to one generated JSON schema, not to the private schema
directory. The launcher will create or open `sol-outcome.json` through its
retained private schema-directory descriptor, then close the directory
descriptor and retain only the regular-file descriptor. The Codex adapter will
therefore pass that inherited, read-only regular-file descriptor whose
child-visible path is
`/proc/self/fd/<fd>`.

The schema bytes are materialized and authenticated before the provider starts.
The inherited descriptor exposes only that file: the child cannot traverse
`..` to reach the project database, locks, artifacts, or other state. No state
directory descriptor is inherited by a provider child.

Descriptor ownership remains explicit:

- the launcher/store assembly owns and closes state-directory capabilities and
  the retained schema-file descriptor;
- the Codex adapter receives only the schema-file capability needed by its child
  process and never closes its caller-owned descriptor;
- success, failure, cancellation, and startup rollback close every duplicate;
- existing fake-provider and schema-path behavior remains compatible.

The current implementation already relies on Linux `/proc/self/fd` semantics;
this design does not introduce a new platform dependency.

## 3. Bounded startup recovery result

Startup recovery will continue updating rows in fixed-size keyset batches inside
one SQLite immediate transaction. It will no longer accumulate every recovered
identifier or materialize every recovered record.

Both active-task and unfinished-prepared-action recovery return the same small
immutable `RecoverySummary(prepared_actions_recovered: int,
tasks_interrupted: int, agent_runs_interrupted: int)`. Fields not affected by a
specific recovery method are zero. Callers that need to verify individual
records query them explicitly after recovery; the launcher needs only the
durable side effects and summary.

This intentionally changes an internal store/coordinator return contract from a
tuple of all records to a constant-size summary. Callback streaming is rejected
because callbacks inside the recovery transaction create reentrancy and failure
semantics. A hard maximum is rejected because legitimate historical growth must
not make startup fail merely due to record count.

Recovery preserves:

- one-transaction rollback for all task, prepared-action, and run mutations;
- deterministic fixed-size batches;
- latest-revision and exact-state checks;
- inert historical PID/process-group handling;
- idempotence on a second recovery call.

Memory must remain bounded when the number of recoverable rows—not merely
unrelated completed history—grows.

## Failure behavior

All three boundaries fail closed with fixed, non-leaking errors. No exception,
audit event, browser projection, or stored payload may expose raw provider
output, filesystem paths outside already-approved operator surfaces, secrets, or
corrupt durable content.

## Verification

Implementation is test-first. Required adversarial coverage includes:

1. An Answer prepared row with a state-incompatible context is rejected before
   recovery, plus positive cases for every valid Answer continuation family.
2. A fake provider can read the inherited schema file but cannot open sibling
   database, lock, artifact, or state files through the descriptor path.
3. Large sets of recoverable active tasks and unfinished prepared actions stay
   under a fixed memory oracle, return accurate summary counts, remain atomic,
   and are idempotent.

After focused RED/GREEN cycles, run the affected store, coordinator, launcher,
Codex adapter, and fake end-to-end tests; then run the complete fake-only
`tests/agent_bridge` suite, compile/import checks, JavaScript syntax check if any
static file changes, and `git diff --check`. Sol independently reviews the exact
implementation diff before acceptance.
