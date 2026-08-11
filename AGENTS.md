# Contributor and worktree policy

This repository is a standalone mechanical extraction, not a rewrite. Keep
the extracted behavior intact and preserve compatibility with the approved
design and implementation plan.

## Boundaries

- Do not write private source project names, paths, branding, history, state,
  or filesystem dependencies into this repository.
- External source trees are read-only inputs. Never modify them from this
  repository.
- Use Python >=3.11.
- Tests must use fakes only. Do not use live model or provider logins, API
  keys, paid services, or browser servers.
- Do not configure a remote, push, or publish this repository.
- Stage explicit paths only; never run `git add -A`.

## Model routing and review

- `gpt-5.6-terra` is the default model for implementation and integration
  tasks.
- `gpt-5.6-luna` is reserved for simple, bounded research or mechanical
  tasks.
- `gpt-5.6-sol` must review every implementation, test, documentation, and
  fix change before acceptance, and performs the final whole-branch review.
- Implementers never self-approve their changes.
- Specify the model explicitly for every subagent assignment.
- Never run multiple writing implementers in parallel in a shared worktree.

## Testing and behavior

- Test narrowly first, then run the relevant full suite before acceptance.
- Preserve existing behavior unless the approved plan explicitly requires a
  change.
