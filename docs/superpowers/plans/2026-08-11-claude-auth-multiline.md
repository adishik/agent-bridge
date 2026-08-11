# Claude Multiline Authentication Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept Claude Code's compact or pretty-printed subscription-status JSON without weakening Agent Bridge's fail-closed authentication boundary.

**Architecture:** Keep the existing `ClaudeCLI.preflight()` process and field validation unchanged. Change only `_parse_auth_status` so the captured stdout line tuple is reconstructed into one JSON document before decoding; malformed, concatenated, non-object, or non-subscription responses still raise `SubscriptionAuthError`.

**Tech Stack:** Python 3.11+, pytest, existing fake Claude executable and `ProcessRunner`.

## Global Constraints

- Preserve the exact `loggedIn is True`, `authMethod == "claude.ai"`, `apiProvider == "firstParty"`, and non-empty `subscriptionType` requirements.
- Do not add a provider, API-key, `jq`, shell, or network fallback.
- Do not add output bounding or change environment filtering, browser readiness, interruption behavior, or model invocation arguments.
- Tests use only the existing fake Claude executable; do not invoke live Claude, Codex, authentication, network, browser servers, or paid services.
- Modify only `src/agent_bridge/adapters/claude_cli.py` and `tests/agent_bridge/test_claude_cli.py` for the implementation commit.
- Terra implements test-first; Sol reviews the exact implementation diff before integration.

---

### Task 1: Decode the complete captured Claude auth document

**Files:**
- Modify: `tests/agent_bridge/test_claude_cli.py`
- Modify: `src/agent_bridge/adapters/claude_cli.py:182-206`

**Interfaces:**
- Consumes: `ProcessResult.stdout: tuple[str, ...]`, the captured stdout lines returned by the existing `ProcessRunner`.
- Produces: unchanged `ClaudeCLI._parse_auth_status(result: ProcessResult) -> ClaudeAuthStatus` behavior, extended to valid multiline JSON.

- [ ] **Step 1: Add a failing multiline-success test**

Add this focused test beside the existing subscription preflight tests:

```python
def test_fable_accepts_pretty_printed_subscription_auth_status(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        auth_status = {
            "loggedIn": True,
            "authMethod": "claude.ai",
            "apiProvider": "firstParty",
            "subscriptionType": "max",
        }
        status = await _adapter(
            fake_claude,
            tmp_path,
            FAKE_CLAUDE_AUTH_STATUS=json.dumps(auth_status, indent=2),
        ).preflight()

        assert status.logged_in is True
        assert status.auth_method == "claude.ai"
        assert status.api_provider == "firstParty"
        assert status.subscription_type == "max"

    asyncio.run(scenario())
```

- [ ] **Step 2: Run the new test and verify the honest RED**

After activating the repository development environment, run:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/agent_bridge/test_claude_cli.py::test_fable_accepts_pretty_printed_subscription_auth_status
```

Expected: FAIL with `SubscriptionAuthError` because the current parser rejects `len(result.stdout) != 1`.

- [ ] **Step 3: Add the concatenated-document fail-closed regression**

Add this test beside the multiline-success test:

```python
def test_fable_rejects_multiple_auth_json_documents(
    fake_claude: Path, tmp_path: Path,
) -> None:
    async def scenario() -> None:
        auth_status = json.dumps({
            "loggedIn": True,
            "authMethod": "claude.ai",
            "apiProvider": "firstParty",
            "subscriptionType": "max",
        })
        with pytest.raises(SubscriptionAuthError, match="could not be verified"):
            await _adapter(
                fake_claude,
                tmp_path,
                FAKE_CLAUDE_AUTH_STATUS=f"{auth_status}\n{auth_status}",
            ).preflight()

    asyncio.run(scenario())
```

After activating the repository development environment, run:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/agent_bridge/test_claude_cli.py::test_fable_rejects_multiple_auth_json_documents
```

Expected: PASS under the current fail-closed parser. Retain it to prove the minimal fix does not accept two documents.

- [ ] **Step 4: Add direct fail-closed tests for empty and non-object stdout**

Import `ProcessResult` from `agent_bridge.process`. Add this focused
parameterized test beside the auth preflight tests; it constructs the process
result locally so `stdout=()` is truly empty without changing the fake fixture:

```python
@pytest.mark.parametrize("stdout", ((), ("[]",), ("null",)))
def test_fable_rejects_empty_or_non_object_auth_document(
    fake_claude: Path, tmp_path: Path, stdout: tuple[str, ...],
) -> None:
    result = ProcessResult(
        run_id="auth-result",
        pid=123,
        process_group_id=123,
        exit_code=0,
        stdout=stdout,
        stderr=(),
        interrupted=False,
    )

    with pytest.raises(SubscriptionAuthError, match="could not be verified"):
        _adapter(fake_claude, tmp_path)._parse_auth_status(result)
```

Run:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/agent_bridge/test_claude_cli.py::test_fable_rejects_empty_or_non_object_auth_document
```

Expected: PASS before and after the implementation because empty stdout and
top-level arrays or `null` are never valid authentication objects.

- [ ] **Step 5: Make strict-type cases explicit in the existing fake auth-shape test**

Extend the `auth_status` parameter values in
`test_fable_rejects_every_non_subscription_auth_shape_without_model_call`
with these two exact objects, without modifying the fake Claude fixture:

```python
{"loggedIn": 1, "authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": "max"},
{"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": 1},
```

Run:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/agent_bridge/test_claude_cli.py::test_fable_rejects_every_non_subscription_auth_shape_without_model_call
```

Expected: PASS before and after the implementation, proving `loggedIn=1` is
not accepted as `True` and `subscriptionType` remains string-only.

- [ ] **Step 6: Implement complete-document decoding**

In `ClaudeCLI._parse_auth_status`, replace the one-line condition and one-line decoder input with complete-document reconstruction:

```python
if result.exit_code != 0 or not result.stdout:
    raise SubscriptionAuthError(
        "Claude subscription authentication could not be verified"
    )
auth_document = "\n".join(result.stdout)
try:
    status = json.loads(auth_document)
except (json.JSONDecodeError, TypeError):
    raise SubscriptionAuthError(
        "Claude subscription authentication could not be verified"
    ) from None
```

Leave the existing `Mapping` check and exact subscription-field validation unchanged.

- [ ] **Step 7: Verify focused GREEN and existing failure cases**

After activating the repository development environment, run:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/agent_bridge/test_claude_cli.py
```

Expected: all Claude adapter tests pass, including compact JSON, pretty-printed JSON, malformed JSON, concatenated documents, non-subscription shapes, and nonzero exit status.

- [ ] **Step 8: Run the complete fake-only bridge suite**

After the focused adapter file passes and before committing, run:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q tests/agent_bridge
```

Expected: all Agent Bridge tests pass using only the repository's fake
executables. The known Starlette `TestClient` deprecation warning may remain.

- [ ] **Step 9: Verify the exact diff and commit**

Run:

```bash
git diff --check
git status --short
git diff -- src/agent_bridge/adapters/claude_cli.py tests/agent_bridge/test_claude_cli.py
```

Stage only the two implementation paths and commit:

```bash
git add src/agent_bridge/adapters/claude_cli.py tests/agent_bridge/test_claude_cli.py
git commit -m "fix: accept multiline Claude auth status"
```

The implementation plan ends with the staged implementation commit. Operator
rollout is outside this plan and remains governed by repository policy.
