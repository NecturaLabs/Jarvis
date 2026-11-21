# ADR: Answer plain commands without the model

**Status:** accepted
**Date:** 2026-11-21

## Context and problem statement

Anything the model answers costs one read of the whole prompt (about 2,100 tokens, about 3.3 s at about 640
tokens/s on the reference machine) before it writes a word, and a model that was freed after idling has to load
first (about 107 s cold). Most of what the owner says to Jarvis is "pause", "next song" or "volume 40", which
needs no reasoning. The owner wants those to feel instant.

## Considered options

1. Send every request to the model.
2. Guess intent from fuzzy matching or a small classifier and run the operation.
3. Match a short, exact list of phrasings per operation, run them through the normal path, and leave everything
   else to the model.

## Decision outcome

Chosen option: "exact phrasings, normal path", because it can only save time, never guess: a request that is
not exactly one of the listed phrasings, however close, goes to the model.

- Each operation may list `Shortcut` phrasings (`operations.py`). `shortcuts.match` lowercases the request, drops
  punctuation, politeness ("please", "can you") and the assistant's name, and then requires a full match.
  Requests over 80 characters, and requests that fit two different calls, match nothing.
- The matched call goes through the same argument validation, policy decision, confirmation and audit log as a
  call the model writes (`Agent._shortcut`). It is added to the conversation as a model-style tool call, so
  when its result has nothing ready to say, or it fails, the model continues from it and words the answer.
- An operation sets `Result.spoken` to a finished sentence ("Paused.", "Back one.", "Volume's at 40.") when its
  result needs no wording. A sentence built from Spotify's words keeps Spotify's origin; a failure sets none.
- Today these operations have shortcuts: pause, resume, next, previous, set volume, louder and quieter, shuffle
  on and off, like, and now playing. "Pause in five minutes", "shuffle my liked songs" and similar go to the
  model.

### Consequences

- **Good:** about 0.3 s for a plain command, with no model load. Boundaries are unchanged, including the PC-audio
  rules in `docs/adr/0003-pc-audio-is-not-the-owner.md`.
- **Bad:** replies on this path are fixed sentences in the operation's code, and a new operation gets the fast
  path only when someone writes its phrasings and a spoken sentence.

## Related

- `src/jarvis/shortcuts.py`, `src/jarvis/agent.py` (`Agent._shortcut`), `src/jarvis/operations.py`,
  `src/jarvis/spotify/ops.py`
- `docs/models.md`: latency
