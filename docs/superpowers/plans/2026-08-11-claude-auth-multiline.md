# Claude Multiline Authentication Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept Claude Code's compact or pretty-printed subscription-status JSON without weakening Agent Bridge's fail-closed authentication boundary.

**Architecture:** Keep the existing `ClaudeCLI.preflight()` process and field validation unchanged. Change only `_parse_auth_status` so the bounded stdout line tuple is reconstructed into one JSON document before decoding; malformed, concatenated, non-object, or non-subscription responses still raise `SubscriptionAuthError`.

**Tech Stack:** Python 3.11+, pytest, existing fake Claude executable and `ProcessRunner`.

## Global Constraints

- Preserve the exact `loggedIn is True`, `authMethod == "claude.ai"`, `apiProvider == "firstParty"`, and non-empty `subscriptionType` requirements.
- Do not add a provider, API-key, `jq`, shell, or network fallback.
- Do not change process bounds, environment filtering, browser readiness, interruption behavior, or model invocation arguments.
- Tests use only the existing fake Claude executable; do not invoke live Claude, Codex, authentication, network, browser servers, or paid services.
- Modify only `src/agent_bridge/adapters/claude_cli.py` and `tests/agent_bridge/test_claude_cli.py` for the implementation commit.
- Terra implements test-first; Sol reviews the exact implementation diff before integration.

---

### Task 1: Decode the complete bounded Claude auth document

**Files:**
- Modify: `tests/agent_bridge/test_claude_cli.py`
- Modify: `src/agent_bridge/adapters/claude_cli.py:182-206`

**Interfaces:**
- Consumes: `ProcessResult.stdout: tuple[str, ...]` from the existing bounded `ProcessRunner`.
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

Run:

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
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

Run:

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_claude_cli.py::test_fable_rejects_multiple_auth_json_documents
```

Expected: PASS under the current fail-closed parser. Retain it to prove the minimal fix does not accept two documents.

- [ ] **Step 4: Implement complete-document decoding**

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

- [ ] **Step 5: Verify focused GREEN and existing failure cases**

Run:

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_claude_cli.py
```

Expected: all Claude adapter tests pass, including compact JSON, pretty-printed JSON, malformed JSON, concatenated documents, non-subscription shapes, and nonzero exit status.

- [ ] **Step 6: Run launcher and complete bridge regression suites**

Run:

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge/test_main.py
```

Expected: all launcher tests pass.

Then run:

```bash
PYTHONPATH="$PWD/src" /home/adi/agent-bridge/.venv/bin/python -m pytest -q \
  tests/agent_bridge
```

Expected: all Agent Bridge tests pass; the known Starlette TestClient deprecation warning may remain.

- [ ] **Step 7: Verify the exact diff and commit**

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

Do not restart the live server or invoke a live model. The controller performs the bounded subscription preflight and deployment restart only after Sol approves this exact commit.
