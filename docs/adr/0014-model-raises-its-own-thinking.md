# ADR: Let the model raise its own thinking within a request

**Status:** accepted
**Date:** 2026-11-25

## Context and problem statement

Since 2026-11-24 every request asks for the same effort, `none` by default, because on 81 everyday spoken
requests thinking made no answer better and every one slower ([models.md](../models.md#thinking)). Some
requests do need it: on 2026-11-25 a misheard artist name led the model through eight quick, wrong steps,
each one playing the next close match. The owner asked for no thinking by default and up to `high` when a
request needs it, decided per request. Who decides, and how, without a list of spoken words?

## Considered options

1. Keep one effort for every request (the owner picks `none` or more in the config).
2. Raise the effort from what the request says, with keywords or phrase patterns.
3. Raise it when something goes wrong: a failed tool, a repeated call.
4. Give the model a typed tool, `think_harder`, that raises its effort for the rest of the request, up to a
   ceiling the config sets.

## Decision outcome

Chosen option 4. The model already decides what a request means ([ADR 0013](0013-free-speech-no-new-shortcuts.md)),
so it is also the one to judge that a request needs careful thought; the tool's description says when.
Option 2 is a phrase list, which ADR 0013 rules out. Option 3 reacts only after a failure and misses a
request that goes wrong without one (every one of the eight steps above succeeded). Option 1 is what the
owner asked to change.

- `[agent]` and `[chat]` gain `thinking_max` (default `high`) and `thinking_raised_tokens` (768 for speech,
  1024 in the chat); `thinking` stays the effort each request starts at. The tool is offered only when
  `thinking_max` is above `thinking`, and its `effort` argument lists only the efforts between them.
- The agent, not the tool, holds the effort: it only rises within a request, never below the starting cap,
  and the next request starts at `thinking` again. Raising is not an action, so the model must still call a
  tool for the request itself, and a claim made after it alone is challenged as one made with no tool.
  Asking again for the same effort is answered, not treated as a stuck loop.
- It is a `safe` operation with no side effect, run through the policy like any other. Outside text that
  talks the model into thinking harder costs time only, within the request's step and time limits.

### Consequences

- **Good:** simple requests stay as quick as before; a hard one gets thinking without the owner changing a
  setting, whatever words the request uses.
- **Bad:** a raise costs one more model call (short: the tool call alone) and up to about 12 s of thinking
  per spoken call after it. On Qwen3.8 a call at `low` or `high` puts the effort's sentence at the start
  of the system prompt, so Strata may read the whole prompt again (about 4.5 s) where `none` and `medium`
  share it; how often Strata still holds it was not measured.
- The context keeps room for the largest thinking cap, so the chat's default `memory_tokens` drops from
  9100 to 7800 to fit the default 24,576-token window.

## Related

- `src/jarvis/agent.py` (`Limits.raised`, `Agent._think_harder`), `src/jarvis/app.py` (`think_operation`),
  `src/jarvis/config.py` (`thinking_max`, `thinking_raised_tokens`), `src/jarvis/operations.py`
  (`Result.effort`)
- [docs/configuration.md](../configuration.md), [docs/models.md](../models.md#thinking)
