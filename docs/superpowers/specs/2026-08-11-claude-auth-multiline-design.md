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

`ClaudeCLI._parse_auth_status` will join the bounded stdout lines with newline
characters and decode the complete text as exactly one JSON document. It will
continue to require all of the existing fields and values:

- `loggedIn` is exactly `true`;
- `authMethod` is exactly `claude.ai`;
- `apiProvider` is exactly `firstParty`; and
- `subscriptionType` is a non-empty string.

Exit status, interrupted-run behavior, environment filtering, process bounds,
and browser readiness gates do not change. Malformed JSON, empty output,
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
the existing compact response test. Additional focused cases will prove that
malformed output and two concatenated JSON documents still fail closed. The
focused Claude/launcher tests, complete Agent Bridge suite, and a bounded live
startup preflight will then run before deployment is restarted.

No live model prompt is part of the fix verification. The live check is limited
to version and subscription-status preflights until the browser is ready.

## Scope and rollout

The production change is limited to the Claude authentication-status parser
and its focused tests. Terra implements test-first in the isolated worktree;
Sol reviews the exact diff. After review, the fix is integrated into `main`,
pushed through the protected repository workflow, and the foreground server is
restarted against `/home/adi/agent-bridge-demo`.
