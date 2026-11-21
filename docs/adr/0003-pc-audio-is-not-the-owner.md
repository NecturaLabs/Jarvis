# ADR: Treat speech heard in PC audio as someone else's

**Status:** accepted
**Date:** 2026-11-21

## Context and problem statement

Jarvis can also listen to what the PC plays (Discord by default, or all output), so that a person in a call can
say "Hey Jarvis" and ask for a song. Whoever speaks there is not the owner: a video, a stream or a call
participant could say the wake word and a command. Jarvis's own voice also comes out of the same speakers. The
owner still has to be in control of what changes.

## Considered options

1. Do not listen to PC audio at all.
2. Treat PC audio like the owner's speech.
3. Treat it as someone else's speech: it can ask, but the owner decides, and it never becomes the owner's own
   conversation.

## Decision outcome

Chosen option: "someone else's speech", because it keeps the feature and keeps `docs/adr/0002-typed-operation-policy.md`
true: text that did not come from the owner's own speech never raises an operation's class. Listening is off by
default (`audio.listen_to_pc`). When it is on:

- **Origin.** The request carries `Origin.PC_AUDIO`, which is not a trusted origin (`policy.py`). The agent tells the
  model it was heard from the PC speakers, not from the owner (`Agent.handle`).
- **Confirmation.** Every side-effecting operation needs at least the owner's spoken yes, and a
  read-only one such as "what's playing" runs. The question names the source ("Someone on your PC audio asked
  me to ...") and, when a grant is set, says what a yes allows (`Agent._question`). The yes is heard on the
  owner's microphone.
- **Grant window.** After a yes, further PC-audio requests run without asking for `audio.pc_audio_grant_s`
  (default 600 s; 0 asks every time). The owner is told the length in the question.
- **Answer-window block.** While Jarvis waits for the yes, speech in PC audio is counted; if any was heard, the
  answer counts as no and Jarvis says it heard voices from the PC (`audio.confirm_blocks_on_pc_speech`,
  `VoiceLoop._ask`), so a voice on the call cannot answer for the owner.
- **History exclusion.** A PC-audio request and its reply are not added to the owner's conversation history
  (`Agent.handle`), and their audio is never kept for debugging.
- **Self-triggering and double hearing.** PC-audio frames are skipped while Jarvis is speaking or asking. A wake
  word heard by both the microphone and PC audio within 2 s is treated as PC audio, and a PC-audio request
  within `audio.pc_audio_cooldown_s` (5 s) of the last one is ignored. A PC request starts only while the loop
  is idle.

### Consequences

- **Good:** a stranger on a call cannot change playback without the owner's yes, cannot speak the yes
  themselves and cannot leave text in the owner's conversation.
- **Bad:** a yes opens a window in which PC audio can ask for more, so the window length is a trade-off the
  owner sets. A request can be lost to the cooldown or to a busy loop, and the owner must be at the PC to
  answer.

## Related

- `docs/adr/0002-typed-operation-policy.md`: provenance and confirmation
- `src/jarvis/agent.py`, `src/jarvis/voice.py`, `src/jarvis/policy.py`
