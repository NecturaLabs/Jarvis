<!--
One focused change per pull request; a refactor goes in its own. The title takes commit-subject
form, `type(scope): summary` (CONTRIBUTING.md, Commit messages). Edit this body whenever the change does.
-->

## Summary

<!-- What was wrong or missing, what this does about it and why. Name any decision or shortcoming
     a reviewer should know about. -->

## Issue

<!-- Closes #N when this completes the issue's acceptance criteria, Refs #N when it does part of
     it, or None: <why>. -->

## Review focus

<!-- Where to look hardest (risky logic, security, a judgment call), and what is safe to skim. -->

## Verification

<!-- What you ran, what you saw, and what you could not check. Write the results here: counts,
     numbers, the lines that matter. "Tests pass" alone is not enough. -->

- Ran:
- Saw:
- Could not check, and why: <!-- "nothing" only when every applicable check ran. Tests marked `local` need
  real hardware and never run in CI; say if your change touches what they cover. -->

## Visible change

<!-- For a change to the chat window, a command's output or the docs a reader sees: paste before and
     after text or a screenshot. Never commit one taken only for a pull request. If nothing visible
     changes, one line is enough:
     No visible change: <why> -->

## Audio

<!-- If the change plays, records or listens to sound, say how it was kept silent (for example
     PIPEWIRE_REMOTE=/nonexistent PULSE_SERVER=unix:/nonexistent), or write "No audio change". -->

## Checks

- [ ] `tools/check.sh` passes (lint, types, tests, REUSE, licences, docs, advisories)
- [ ] Tool operations are tested, including a refused and an unconfirmed case
- [ ] Docs updated where affected: `README.md`, `docs/configuration.md`, `docs/models.md`, `docs/plan.md`,
      an ADR if a decision changed, and `CHANGELOG.md` (Unreleased)
- [ ] New dependency, model or voice: licence checked and compatible with GPL-3.0-or-later
- [ ] No tokens, recordings, personal data or personal paths
- [ ] Every commit is signed off (`git commit -s`)
- [ ] Plain wording: no filler, praise or restatement; the summary leads with what changed and why
