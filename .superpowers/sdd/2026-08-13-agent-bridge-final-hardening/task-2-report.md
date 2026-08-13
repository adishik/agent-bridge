# Task 2 report — Sol schema-file capability

## Scope and root cause

The inherited implementation retained the private `schemas` **directory**
descriptor in `_OpenedProjectState`, gave Codex a path below
`/proc/self/fd/<directory-fd>`, and passed that directory through
`ProcessRunner.pass_fds`. A provider process could therefore resolve `..` from
the schema path and open sibling private state such as `bridge.sqlite3`,
artifacts, and locks.

The approved smallest fix is implemented: `materialize_sol_schema_file()`
atomically writes the exact canonical JSON schema through the trusted schema
directory descriptor, fsyncs and replaces the leaf, and returns one
caller-owned read-only regular-file descriptor. The launcher closes the schema
directory immediately and retains only the file descriptor. `CodexCLI` checks
the injected descriptor's `F_GETFL` access mode, regular-file type, and exact
canonical bytes, exposes it as `/proc/self/fd/<fd>`, and passes only that FD to
the provider. Direct `CodexCLI` construction remains supported and materializes
the same canonical schema in a supplied temporary directory.

## RED / GREEN evidence

RED (before production edits):

```sh
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_codex_cli.py tests/agent_bridge/test_main.py \
  -k 'schema_file_capability or provider_cannot_traverse_schema'
```

Result: `1 failed`. The real launcher/fake subprocess boundary recorded an
inherited directory descriptor; the assertion `is_regular is True` failed with
`False`. The malicious child had already read the schema path and attempted its
sibling traversal, proving the intended directory-capability regression setup.

Focused GREEN:

```sh
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_codex_cli.py tests/agent_bridge/test_main.py \
  -k 'schema_file_capability or provider_cannot_traverse_schema or materialize_sol_schema_file or noncanonical_schema_file or injected_schema_file or schema_file_descriptor_closes'
```

Result: `8 passed`.

Affected adapter/launcher suites:

```sh
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_codex_cli.py tests/agent_bridge/test_main.py
```

Result: `82 passed`.

Required combined lane was run:

```sh
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_codex_cli.py tests/agent_bridge/test_main.py \
  tests/agent_bridge/test_e2e_fake_agents.py
```

Result: one unrelated, serially reproducible failure in
`test_two_project_http_websocket_workflow_isolated_by_hub_lease`: its mocked
browser-ID iterator returns `beta-restarted` where the test expects
`alpha-restarted`. This fixture directly constructs `CodexCLI` without the
launcher schema-file seam; no Task 2 path participates. The unaffected E2E
remainder passed:

```sh
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_e2e_fake_agents.py \
  -k 'not two_project_http_websocket_workflow_isolated_by_hub_lease'
```

Result: `29 passed`.

Compile/import evidence:

```sh
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m compileall -q src/agent_bridge
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -c 'from agent_bridge.adapters.codex_cli import CodexCLI, materialize_sol_schema_file; import agent_bridge.__main__; print("imports-ok")'
git diff --check
```

Result: all exited zero; import printed `imports-ok`; the diff check was clean.

## FD and lifecycle proof

- The malicious fake child reads exactly `SOL_OUTCOME_SCHEMA` through the
  `/proc/self/fd/<schema-file-fd>` path and cannot open `bridge.sqlite3`,
  `artifacts`, or `locks` using `schema_path / ".." / sibling`.
- The recorded `pass_fds` has exactly one regular descriptor and no directory
  descriptor.
- Constructor tests reject directory, writable, closed, and wrong-content
  descriptors. Materialization returns a regular, `O_RDONLY` descriptor whose
  bytes are exactly the canonical serialized schema.
- The runtime-close assertion verifies its caller-owned schema FD is closed
  after `main()` returns. A forced Codex construction failure verifies startup
  rollback closes the materialized schema FD.
- The adapter never closes an injected FD; `release_state_authority()` owns the
  retained schema-file FD and closes it alongside the other launcher-owned
  authorities.

## Changed files

- `src/agent_bridge/adapters/codex_cli.py`
- `src/agent_bridge/__main__.py`
- `tests/agent_bridge/test_codex_cli.py`
- `tests/agent_bridge/test_main.py`
- `.superpowers/sdd/2026-08-13-agent-bridge-final-hardening/task-2-report.md`

## Commit and concern

Commit: local `fix: confine Sol schema authority` (the exact SHA is supplied
in the handoff after the report is committed).

Concern: the full required three-file test command has the pre-existing/reproducible
unrelated two-project hub-ID failure described above. No Task 2 changes were
made to that out-of-scope fixture.
