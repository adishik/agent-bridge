# Claude Multiline Authentication Compatibility Design

## Problem

Agent Bridge verifies Fable's subscription before enabling model actions. The
current parser accepts the Claude Code `auth status --json` response only when
the process runner captured exactly one stdout line. Claude Code 2.1.226 emits
the same valid JSON object in a pretty-printed, multiline form, so startup
incorrectly reports `subscription_unavailable` even though the account is
logged in through `claude.ai`, uses the `firstParty` provider, and has a
non-empty subscription type.

The existing fail-closed subscription requirements remain correct. Only the
assumption about JSON line formatting is wrong.

## Decision

`ClaudeCLI._parse_auth_status` will join the captured stdout lines with newline
characters and decode the complete text as exactly one JSON document. It will
continue to require all of the existing fields and values:

- `loggedIn` is exactly `true`;
- `authMethod` is exactly `claude.ai`;
- `apiProvider` is exactly `firstParty`; and
- `subscriptionType` is a non-empty string.

Exit status, interrupted-run behavior, environment filtering, and browser
readiness gates do not change. Malformed JSON, empty output,
multiple JSON documents, non-object JSON, or any missing/inconsistent
subscription field remains a `SubscriptionAuthError`.

## Alternatives considered

1. **Decode the complete stdout document (chosen).** This directly follows the
   CLI's `--json` contract and accepts both compact and pretty-printed output
   without weakening authentication.
2. **Pipe through `jq -c`.** This adds an executable dependency and another
   process boundary solely to normalize whitespace.
3. **Accept any successful `loggedIn` response.** This would weaken the
   subscription-only billing boundary and is rejected.

## Testing

The adapter tests will first reproduce the failure with a valid multiline auth
object. The minimal implementation must make that test pass while preserving
the existing compact response test. Additional focused fake-only cases will
prove that malformed output, two concatenated JSON documents, top-level `[]`,
top-level `null`, and truly empty captured stdout fail closed. The existing
nonzero-exit coverage remains. The auth-shape parameterization will also
explicitly reject `loggedIn=1` and non-string `subscriptionType` values.
After the focused adapter test file passes, the complete fake-only
`tests/agent_bridge` suite will run before the implementation commit.

No live CLI, model prompt, network call, browser server, or deployment action
is part of the implementation or its verification.

## Scope and rollout

The implementation change is limited to
`src/agent_bridge/adapters/claude_cli.py` and
`tests/agent_bridge/test_claude_cli.py`. Terra implements test-first in the
isolated worktree; Sol reviews the exact diff. Any operator rollout is outside
this implementation plan and remains governed by the repository policy.
