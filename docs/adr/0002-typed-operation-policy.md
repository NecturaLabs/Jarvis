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

## Related

- `docs/plan.md` — Boundaries, Speaking, barge-in, confirmation
- `docs/adr/0001-own-agent-loop-on-strata.md`
