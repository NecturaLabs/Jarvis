# ADR: Understand free speech with typed tools; add no new shortcuts

**Status:** accepted
**Date:** 2026-11-24

## Context and problem statement

[ADR 0004](0004-model-free-path-for-plain-commands.md) answers a short, exact list of plain commands
("pause", "next song", "volume 40") without the model, so they take about 0.3 s instead of a model request.
On 2026-11-24 the owner asked Jarvis to be quiet and got a full sentence back, and the first plan for the fix was
more shortcut phrasings. The owner turned it down: they never repeat a set phrase and say whatever is on their
mind, so no spoken command is to be hardcoded. Should the shortcuts that exist go too, and how is a new request
served?

## Considered options

1. Remove every shortcut, so the model handles every request.
2. Keep the existing shortcuts, add no new ones, and serve every new request with a typed operation the model
   chooses from its description.
3. Keep adding shortcuts for common requests.

## Decision outcome

Chosen option 2, because the existing shortcuts never decide what a request means: they match only the whole
plain command, and any other wording goes to the model, which reaches the same operation from its description.
Removing them would make the commands said most often several times slower and change nothing about what
Jarvis understands. Option 3 is what the owner ruled out.

- No `Shortcut` phrasing, keyword list or pattern over the owner's words is added. `AGENTS.md` carries the rule.
- Something new Jarvis can do is a typed operation with a description that says when to use it, in any words.
  Being quiet is `stay_quiet`, which ends the request without a reply; a country's time is `time_now` with a
  country, answered from the PC's time zone database.
- Where the model picks the wrong tool or argument, the fix is the operation's description or schema, measured
  against the model, never a phrasing that routes around it.

### Consequences

- **Good:** anything but an exact plain command reaches the model, and no phrase list decides what it
  means. The plain commands keep their speed.
- **Bad:** a new operation is never instant: it costs a model request (2.3 s for the time in Australia,
  measured on 2026-11-24 with the model loaded).

## Related

- `src/jarvis/shortcuts.py`, `src/jarvis/operations.py` (`Shortcut`), `src/jarvis/app.py` (`quiet_operation`),
  `src/jarvis/web/ops.py` (`time_now`), `src/jarvis/web/zones.py`
- `AGENTS.md`: product rules
