# Agent Bridge Final Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three reviewed Phase 1 blockers with exact prepared-action authentication, a file-only Sol schema capability, and constant-memory startup recovery.

**Architecture:** Keep the existing store, launcher, and adapter boundaries. Strengthen the store's read-only legacy audit using the same typed continuation rules as normal preparation, narrow the provider's inherited authority from a directory to one read-only schema file, and replace recovery result materialization with one immutable scalar summary while retaining fixed-size transactional batches.

**Tech Stack:** Python 3.11+, SQLite, asyncio subprocesses, Linux `/proc/self/fd`, pytest, existing fake Claude/Codex executables.

## Global Constraints

- Source design: `docs/superpowers/specs/2026-08-13-agent-bridge-final-hardening-design.md`.
- Before every task ask: “Am I over-engineering this?” If unsure whether new scope or architecture is required, stop and ask Adi.
- Work only in the standalone `agent-bridge` worktree. Never access or modify another repository.
- Use `/home/adi/agent-bridge/.venv/bin/python` with `PYTHONPATH=<this-worktree>/src`; do not install dependencies.
- Tests and review use fakes only. No live providers, authentication, network, browser server, paid service, or credentials.
- Preserve existing product workflow and compatible direct-adapter construction.
- Do not include the deferred WebSocket/coroutine-observation Minors or SQL `LIKE` Minor.
- Use strict RED/GREEN TDD. Every behavior test must fail for the intended reason before production edits.
- Stage explicit paths only. Do not push, merge, alter remotes, or remove worktrees.
- Terra implements each task; Sol independently reviews each task before acceptance.

---

### Task 1: Authenticate every prepared Answer context before recovery

**Files:**
- Modify: `src/agent_bridge/store.py`
- Modify: `tests/agent_bridge/test_store.py`

**Interfaces:**
- Consumes: `PreparedActionRecord`, `AnswerPayload`, `AnswerContext`, `ScopeApprovalContext`, `SolResumeContext`, `ReviewContext`, `ClarificationContext`, `TaskRecord`, and the existing read-only `audit_legacy_project_ownership()` path.
- Produces: a private, read-only context authentication helper used by `_legacy_prepared_action_is_authenticated()`; no new public API.

- [ ] **Step 1: Add state-incompatible Answer audit tests**

Create valid persisted prepared Answer rows through the real store API, then alter only the persisted context/state pairing to reproduce the reviewed corruption. Parameterize at least these wrong pairs:

```python
sol_context = SolResumeContext(
    sol_thread_id="11111111-1111-4111-8111-111111111111",
    sol_run_id="run-sol",
    prompt="continue exact work",
)
scope_context = ScopeApprovalContext(
    baseline_id="baseline-1",
    approved_revision=1,
    underlying_continuation=sol_context,
)
review_context = ReviewContext(
    fable_session_id="fable-session",
    review_prompt="review exact work",
    completion_allowed=False,
    underlying_continuation=scope_context,
)
clarification_context = ClarificationContext(
    fable_session_id="fable-session",
    clarification_prompt="clarify exact work",
    underlying_continuation=scope_context,
)
wrong_pairs = (
    (TaskState.FABLE_REVIEWING, sol_context),
    (TaskState.FABLE_CLARIFYING, review_context),
    (TaskState.SOL_RUNNING, clarification_context),
)
```

For every row, assert `audit_legacy_project_ownership(canonical_root)` raises the existing generic ownership/audit error and that the row and task bytes/state remain unchanged.

- [ ] **Step 2: Run the tests and verify honest RED**

Run:

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_store.py -k 'legacy_audit_rejects_answer_context'
```

Expected: FAIL because the current Answer audit checks state shape and lineage but not the typed context-to-active-state binding.

- [ ] **Step 3: Add positive context-family coverage**

Use real store transitions to persist valid Answer continuations for
`SOL_RUNNING`, `SOL_CORRECTING`, and review families accepted by
`prepare_answer_action()`. Include the valid no-agent-run `SOL_CORRECTING` case.
Assert the audit accepts each exact row. Expectations must be literal and must
not call the new private helper.

- [ ] **Step 4: Implement the smallest typed authentication helper**

Add a private helper shaped like:

```python
def _legacy_prepared_context_matches_task(
    self,
    *,
    task: TaskRecord,
    active_state: TaskState,
    context: PreparedContinuationContext,
) -> bool:
    if active_state is TaskState.FABLE_REVIEWING:
        return (
            isinstance(context, ReviewContext)
            and context.fable_session_id == task.fable_session_id
            and self._legacy_continuation_identifiers_match_task(
                task, context.underlying_continuation
            )
        )
    return (
        active_state in _SOL_TASK_STATES
        and isinstance(context, (ScopeApprovalContext, SolResumeContext))
        and self._legacy_continuation_identifiers_match_task(task, context)
    )
```

Define the referenced identifier helper in the same task. It must recursively
handle the three continuation families normal Answer preparation accepts. For a
`SolResumeContext`, an existing agent-run row is additional authority; absence
is allowed because the Store supports preparation before run persistence:

```python
def _legacy_continuation_identifiers_match_task(
    self,
    task: TaskRecord,
    context: PreparedContinuationContext,
) -> bool:
    if isinstance(context, SolResumeContext):
        run = self._connection.execute(
            """
            SELECT task_id, revision, cli_session_id FROM agent_runs
            WHERE run_id = ?
            """,
            (context.sol_run_id,),
        ).fetchone()
        return context.sol_thread_id == task.sol_thread_id and (
            run is None
            or (
                run["task_id"] == task.task_id
                and run["revision"] == task.revision
                and run["cli_session_id"] == context.sol_thread_id
            )
        )
    if isinstance(context, ScopeApprovalContext):
        return (
            context.baseline_id == task.baseline_id
            and context.approved_revision == task.revision
            and (
                context.underlying_continuation is None
                or self._legacy_continuation_identifiers_match_task(
                    task, context.underlying_continuation
                )
            )
        )
    if isinstance(context, ReviewContext):
        return (
            context.fable_session_id == task.fable_session_id
            and self._legacy_continuation_identifiers_match_task(
                task, context.underlying_continuation
            )
        )
    return False
```

It must:

- reject `AnswerContext`, which normal `prepare_answer_action()` does not accept
  for any continuation state;
- require `ReviewContext` only for `FABLE_REVIEWING` and scope/Sol continuation
  forms only for the existing `_SOL_TASK_STATES` (`SOL_RUNNING` and
  `SOL_CORRECTING`); reject every prepared Answer targeting
  `FABLE_CLARIFYING`, because normal creation cannot traverse that state edge;
- bind Fable session, Sol thread, baseline, revision, and task fields exactly;
  when the referenced run exists, bind its task/revision/CLI session too, but do
  not require a run row or claim an independent prompt copy exists;
- return `False` rather than repairing, normalizing, or leaking corrupt content;
- leave approval/resume lineage checks intact.

Call it from the Answer branch of `_legacy_prepared_action_is_authenticated()` for both `payload.continuation` and `pending_context` after their existing equality check.

- [ ] **Step 5: Run focused and full Store tests**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_store.py -k 'legacy_audit or prepared_answer'
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_store.py
```

Expected: PASS.

- [ ] **Step 6: Commit explicit Task 1 paths**

```bash
git add src/agent_bridge/store.py tests/agent_bridge/test_store.py
git diff --cached --check
git commit -m "fix: authenticate prepared answer recovery"
```

---

### Task 2: Give Sol only a read-only schema-file capability

**Files:**
- Modify: `src/agent_bridge/adapters/codex_cli.py`
- Modify: `src/agent_bridge/__main__.py`
- Modify: `tests/agent_bridge/test_codex_cli.py`
- Modify: `tests/agent_bridge/test_main.py`
- Modify only if the existing real-stack fixture requires it: `tests/agent_bridge/test_e2e_fake_agents.py`

**Interfaces:**
- Consumes: `_open_private_directory()`, `_OpenedProjectState.release_state_authority()`, `CodexCLI`, `ProcessRunner`'s `pass_fds` parameter, and canonical `SOL_OUTCOME_SCHEMA` serialization.
- Produces: `materialize_sol_schema_file(directory_fd: int) -> int`, returning one caller-owned, read-only regular-file descriptor; the existing `CodexCLI` constructor gains the keyword `schema_file_fd: int | None = None` and passes only that file descriptor to Sol.

- [ ] **Step 1: Add the malicious-child capability test**

Open realistic private state containing `bridge.sqlite3`, a lock, an artifact, and a schemas directory. Construct the real adapter through the launcher seam. The fake child must:

```python
schema_path = Path(argv[argv.index("--output-schema") + 1])
schema = json.loads(schema_path.read_text(encoding="utf-8"))
for sibling in ("bridge.sqlite3", "artifacts", "locks"):
    attempt_open(schema_path / ".." / sibling)
```

Assert it reads the exact schema, every sibling attempt fails, and the recorded `pass_fds` contains a regular file descriptor but no directory descriptor.

- [ ] **Step 2: Run the capability tests and verify honest RED**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_codex_cli.py tests/agent_bridge/test_main.py \
  -k 'schema_file_capability or provider_cannot_traverse_schema'
```

Expected: FAIL because the current child inherits the schemas directory and can traverse to its parent.

- [ ] **Step 3: Implement secure schema-file materialization**

In `codex_cli.py`, add a helper that:

```python
def materialize_sol_schema_file(directory_fd: int) -> int:
    # validate directory fd
    # atomically write canonical SOL_OUTCOME_SCHEMA through dir_fd
    # fsync and replace the leaf without following symlinks
    # reopen the final leaf O_RDONLY | O_NOFOLLOW through dir_fd
    # validate regular file, exact bytes, and non-writable access mode
    # return the read-only fd; close all temporary/write fds on every path
```

Do not pass the directory descriptor to a child.

- [ ] **Step 4: Narrow `CodexCLI` and launcher ownership**

Keep direct adapter construction without an injected descriptor compatible. For launcher assembly:

- create/open the schemas directory through the retained state descriptor;
- call `materialize_sol_schema_file()`;
- close the schemas directory descriptor immediately;
- store only `schema_file_descriptor` on `_OpenedProjectState`;
- give `CodexCLI` `schema_path=Path(f"/proc/self/fd/{schema_file_descriptor}")` through its `schema_file_fd` seam;
- make `CodexCLI` verify the injected descriptor is a read-only regular file containing the exact canonical schema;
- pass only the file FD to `ProcessRunner`;
- close the caller-owned file FD in `release_state_authority()` and every partial-startup rollback path.

No provider child may inherit a state, artifact, schema-directory, or database descriptor.

- [ ] **Step 5: Add lifecycle and compatibility tests**

Cover constructor rejection of directory, writable, closed, and wrong-content FDs; direct temp-directory adapter behavior; start/resume schema reading; startup failure cleanup; runtime close cleanup; and the full fake Codex invocation.

- [ ] **Step 6: Run affected tests**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_codex_cli.py tests/agent_bridge/test_main.py \
  tests/agent_bridge/test_e2e_fake_agents.py
```

Expected: PASS with no live invocation.

- [ ] **Step 7: Commit explicit Task 2 paths**

Stage only files actually changed, verify `git diff --cached --check`, and commit:

```bash
git commit -m "fix: confine Sol schema authority"
```

---

### Task 3: Return constant-size recovery summaries

**Files:**
- Modify: `src/agent_bridge/store.py`
- Modify: `src/agent_bridge/coordinator.py`
- Modify: `tests/agent_bridge/test_store.py`
- Modify: `tests/agent_bridge/test_coordinator.py`
- Modify: `tests/agent_bridge/test_main.py`
- Modify: `tests/agent_bridge/test_e2e_fake_agents.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True, slots=True) class RecoverySummary` with exact integer fields `prepared_actions_recovered`, `tasks_interrupted`, and `agent_runs_interrupted`.
- Changes: `SQLiteStore.recover_unfinished_prepared_actions() -> RecoverySummary`, `SQLiteStore.recover_active_tasks() -> RecoverySummary`, and `Coordinator.recover_unfinished_prepared_actions() -> RecoverySummary`.

- [ ] **Step 1: Add the exact summary contract tests**

For a small mixed fixture, assert literal summaries:

```python
assert store.recover_unfinished_prepared_actions() == RecoverySummary(
    prepared_actions_recovered=2,
    tasks_interrupted=2,
    agent_runs_interrupted=0,
)
assert store.recover_active_tasks() == RecoverySummary(
    prepared_actions_recovered=0,
    tasks_interrupted=3,
    agent_runs_interrupted=4,
)
assert store.recover_active_tasks() == RecoverySummary(0, 0, 0)
```

Query exact tasks, prepared rows, and runs after recovery to preserve all prior behavioral assertions.

- [ ] **Step 2: Add recoverable-row memory tests and verify RED**

Create large bounded fixtures containing thousands of genuinely recoverable active tasks and unfinished prepared actions—not unrelated completed history. Start `tracemalloc` only after fixture creation. Assert accurate scalar counts, durable row state, and a fixed peak-memory ceiling that the current all-record tuple exceeds with clear headroom.

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_store.py -k 'recovery_summary or recoverable_rows_are_bounded'
```

Expected: FAIL because both current methods collect every identity and materialize every record.

- [ ] **Step 3: Implement the immutable summary and counter-only loops**

Add validation in `RecoverySummary.__post_init__`: reject bools, non-integers, and negative values. In each recovery transaction:

- keep only scalar counters and the existing fixed-size batch;
- increment `tasks_interrupted` only when this call transitions a task;
- increment `prepared_actions_recovered` for each PREPARED/CLAIMED row terminalized as RECOVERED;
- take `agent_runs_interrupted` from the exact terminal update row count;
- do not retain IDs or fetch recovered records after commit;
- keep one immediate transaction and existing rollback/CAS behavior.

- [ ] **Step 4: Update callers and behavioral tests**

The launcher may ignore the summary. Coordinator forwards it. Tests and fake E2E code that formerly iterated returned records must query the exact known IDs after recovery and separately assert the summary. Do not add callbacks or a collection compatibility flag.

- [ ] **Step 5: Run affected and full fake-only verification**

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_store.py tests/agent_bridge/test_coordinator.py \
  tests/agent_bridge/test_main.py tests/agent_bridge/test_e2e_fake_agents.py
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m compileall -q src/agent_bridge
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -c \
  'import agent_bridge.store, agent_bridge.coordinator, agent_bridge.__main__'
node --check src/agent_bridge/static/app.js
git diff --check
```

Expected: complete fake-only suite PASS; only the established Starlette/TestClient deprecation warning may remain.

- [ ] **Step 6: Commit explicit Task 3 paths**

Stage only changed Task 3 files, inspect the staged diff, and commit:

```bash
git commit -m "fix: bound startup recovery results"
```

---

## Final acceptance

After each task, generate an exact base-to-head review package and obtain a Sol task review covering both spec compliance and implementation quality. Address Critical/Important findings through the normal scoped fix loop. After all three tasks are clean, run one Sol whole-hardening review over the design-base-to-head range, then perform fresh fake-only verification before presenting integration options. Do not merge or push automatically.
