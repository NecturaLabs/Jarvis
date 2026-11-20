<!-- Title: type(scope): summary, at most 72 characters (Conventional Commits). One focused change per PR; split a
refactor from a behavior change. Self-review the diff before asking for review, and edit this body whenever the change
does. -->

## Summary
<!-- What changed and why, in a few lines a reader understands without the diff: the problem, the fix, any decision
or shortcoming a reviewer should know. -->

## Issue
<!-- Refs #N. "Closes #N" closes the issue only when this merges into the default branch; after a merge into any other
branch, close it by hand with its evidence. No tracked issue: "None: <why>". -->

## Review focus
<!-- Where to look hardest (risky logic, security, a judgment call) and what is safe to skim (generated or mechanical
changes). -->

## Verification
<!-- What ran and what it showed, written here: counts, results, numbers, the lines that matter. Do not point at
scratch folders: they do not last. Attach screenshots or paste the figures. -->
- Sync: `uv sync --frozen`
- Lint and types: `uv run .lint/check.sh`
- Tests: `uv run pytest -q`
- Local tests (this PC, with the owner for audio): `uv run pytest -m local`

## Not verified
<!-- What was not run or seen, and why. Write "nothing" only when every applicable check above ran. -->
