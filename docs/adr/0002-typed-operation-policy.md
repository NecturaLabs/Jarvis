# ADR: Bound every action with typed operations and default deny

**Status:** accepted
**Date:** 2026-11-20

## Context and problem statement

Jarvis acts on the owner's PC and accounts on voice commands interpreted by a local model, and it reads
untrusted text (web pages, files, Spotify metadata). A misheard command, a model error or text injected
by a web page must not become a harmful action. The owner chose "ask for risky" as the default boundary.

## Considered options

1. A generic shell tool guarded by allow and deny lists of command names.
2. Typed operations with validated arguments, each with a class (`safe`, `confirm`, `strong`, `deny`),
   unknown operations refused; generic shell only as an opt-in `strong` operation.

## Decision outcome

Chosen option: "typed operations, default deny", because command-name lists cannot constrain shells,
interpreters, scripts or wrapper commands, so any list-guarded shell is bypassable, while a typed
operation states exactly what can happen and can be checked before it runs. Every message part carries
its provenance; text from outside the owner's speech cannot raise an operation's class, and while such
text is in the session every side-effecting operation needs at least spoken confirmation. A
confirmation is bound to the hash of one operation and its arguments, starts only after Jarvis has
finished speaking, and the `strong` class is confirmed by keybinding, never by voice.

### Consequences

- **Good:** each capability is reviewed and tested as a unit; the policy decision table is plain data
  that tests cover row by row; prompt injection cannot raise its own permissions.
- **Bad:** each new capability needs an operation, a schema and a class; requests outside the set are
  refused until one is added or the owner enables the `strong` shell.

## Status update (2026-11-22)

The keybinding is built: `jarvis confirm`, which the owner binds to a key, reaches the voice service
through its control socket (`src/jarvis/confirmkey.py`, `src/jarvis/control.py`). It allows the one
action Jarvis asked about, bound to that action's hash, once, and only after the question was heard. A
press from a program Jarvis started is refused, and a press with nothing waiting holds the key off for
30 s; a press after a timeout, `hush` or `pause` finds nothing waiting. Refusing programs Jarvis
started is a best effort: any program running as the owner can run `jarvis confirm`, so the key stops
the model acting alone, not malware on the owner's account, and an operation that starts programs
(#19) must not hand the model a way around it. The chat window and `jarvis ask` have no key, so they
refuse `strong` actions. The consent bypass
(`boundaries.bypass_consent`, added later) skips spoken questions only: it never skips the key (#34,
decision 3).

## Related

- `docs/plan.md` — Boundaries, Speaking, barge-in, confirmation
- `docs/adr/0001-own-agent-loop-on-strata.md`
