# ADR: Watch a window or the screen and say when something shows up, with one no-tool check by the same model

**Status:** proposed (the owner confirms the model check, the GPU cost and the limits below)
**Date:** 2026-11-26

## Context and problem statement

The owner wants to say, in any words, "tell me when X shows up on that window" and be told by voice
([ADR 0013](0013-free-speech-no-new-shortcuts.md): the model decides what a request means). Telling means looking
at the screen now and then without being asked each time. Jarvis already takes a screenshot
(`pc_screenshot`, which never touches a window) and already speaks between requests (reminders,
[ADR 0011](0011-reminders-and-scheduled-requests.md)). What is missing is a repeated look and a judge of what
it shows.

Two product rules bear on the judge. One agent does all the work ([plan](../plan.md), Boundaries): a second
agent, a delegate or a tool-using helper is out. And Jarvis runs locally, beside games, and must use no GPU
or CPU while idle ([ADR 0005](0005-local-only.md)). Looking at a picture means a model call, which uses the
GPU.

## Considered options

1. Compare pixels (template matching, a colour or text match) in code.
2. Ask the model the whole question in an ordinary request every time, with the tools and the conversation.
3. Ask the same model one narrow question per check: no tools, no conversation, a YES or NO answer.

## Decision outcome

Chosen option: 3.

- Option 1 needs the owner's target spelled out as a pattern, which is a keyword list by another name, and it
  cannot read "a login dialog" or "the download finished". Option 2 puts the screen's words next to the tools
  and the conversation, which is the prompt-injection path the policy engine exists to shut, and it grows the
  conversation by a picture every 20 seconds.
- **The tool contract.** `pc_watch` (`src/jarvis/pc/watch.py`) takes what to watch (`of`: a window from
  `pc_windows`, a workspace or the screen), `condition` (the model's own short description of what to look
  for), `every_s` (10 to 300, default 20), `for_min` (1 to 720, default 120) and `repeat` (default true).
  `pc_watches` lists, `pc_watch_stop` stops one or all. A window is bound the way the window tools bind it: a
  number from the request's latest `pc_windows` that the model has read, checked again against Hyprland, never
  a guess. No phrasing or keyword is shipped for any of the three.
- **The policy class.** All three are `safe`: a watch only reads pixels and speaks, like `pc_screenshot`.
  `pc_watch` is still an action (`side_effect=True`), so outside text in the request raises it to `confirm`.
  It is refused for a request heard in PC audio, as the other PC tools are, and when the model cannot see
  pictures ([ADR 0002](0002-typed-operation-policy.md); the picture reader is `strata.pictures`). At most two
  watches run at once.
- **The check.** Each check takes the same capture `pc_screenshot` does (`grim -T` for a window, `-o` for a
  monitor; nothing is focused, moved or resized), shrinks it to 1024 pixels on its longest side and makes one
  model call (`Agent.glance`): a system text that says to answer only YES or NO about whether the picture shows
  the condition and that the picture is outside data, the picture and the condition, no tools, no conversation,
  eight tokens, nothing kept. Only a leading YES counts; anything else is a no. This is the same model asked one
  question, not a second agent: it has no tools, no memory and no way to act, and what it answers is a boolean
  that no code reads as an instruction.
- **What is said.** On a YES Jarvis says a fixed line built from the condition ("Heads up: <condition>."), never
  words read off the screen, so what the screen shows cannot put words in Jarvis's mouth. After one it stays
  quiet for the longer of 60 seconds and three checks, so a thing that stays on screen is said once, then again
  while it is still there. With `repeat` off the watch ends after its first line. A watched window that closes
  ends its watch, said once; three checks in a row that fail (no picture, no answer) end it too, said once.
- **Where it runs.** In the voice service's process (`jarvis run`), between requests, as reminders do: a check
  starts only while the loop is idle (no request, nothing being said, not paused, locked, dictating or in a
  meeting), runs on a thread of its own so the microphone is still read, and its line is said at the next idle
  moment. Watches are kept in memory, so they end with the service, and `jarvis ask` and the chat do not offer
  them. A schedule-style file would let the chat set one for a service that then runs it; it was not worth the
  second writer on a state file for a feature that is meant for a session at the PC.
- **The cost, said plainly.** Each check is a model call with a picture, so it uses the GPU for the length of
  one short answer. That is the owner's request, bounded: at most every 10 seconds, at most twelve hours, at most
  two watches, and nothing at all once none runs. The result of `pc_watch` and its description say so. The
  watch is not tested against a running game's frame time; until it is, this ADR stays proposed.
- **Audit.** Starting, stopping (with why: the owner, the time, the window, failures, a first line with
  `repeat` off), each announcement and each failed check are audited. Passed checks are not: one every 20
  seconds would bury the log.
- **Related: Jarvis sees its own screenshots.** `pc_screenshot` now hands the picture to the model in a
  user message right after its result, as outside data, when the model can see pictures
  ([ADR 0002](0002-typed-operation-policy.md), provenance): the request counts as having read outside text, so
  a side effect after it needs a yes, and the conversation keeps only the picture's name. Without the picture
  reader the result says so.

## Consequences

- The owner can say what to watch for in any words, and the model chooses the target and the condition.
- A page or a window that shows "ignore your rules and say X" cannot make Jarvis say X: the model's answer is
  read only as YES or NO, and the spoken line is the condition the model wrote when the owner asked.
- A model that answers YES wrongly says a wrong line; one that answers NO wrongly says nothing. Neither acts.
- GPU use is real while a watch runs, and unmeasured beside a game. The limits above are the bound.
