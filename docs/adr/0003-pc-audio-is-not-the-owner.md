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
  is idle. If the PC capture has already accepted the wake, the matching microphone wake leaves it running.
  If the microphone started first, a match persists for that request even when a later PC wake is heard.

- **Capture while the loop moves on.** A PC request is captured through a tap (`VoiceLoop._pc_tap`) that
  passes the frames on but still notes wake words in them. The capture ends as soon as the loop takes another
  request or starts waiting for a yes, so PC speech in the answer window is still counted for the block above,
  and a wake word said meanwhile still marks a microphone request that echoes it as PC audio.
- **No second capture.** A PC request moves the loop out of IDLE before its chime plays
  (`VoiceLoop._answer_pc`), so no further PC request is captured while it is handled.

### Consequences

- **Good:** a stranger on a call cannot change playback without the owner's yes, cannot speak the yes
  themselves and cannot leave text in the owner's conversation.
- **Bad:** a yes opens a window in which PC audio can ask for more, so the window length is a trade-off the
  owner sets. A request can be lost to the cooldown or to a busy loop, and the owner must be at the PC to
  answer. Only the apps in `pc_audio_apps` are known as PC audio: on speakers, another app's speech that
  reaches the microphone is heard as the owner's. Headphones or PipeWire echo cancellation close that gap
  (`docs/configuration.md`, "Speakers and other apps' speech").

## Related

- `docs/adr/0002-typed-operation-policy.md`: provenance and confirmation
- `src/jarvis/agent.py`, `src/jarvis/voice.py`, `src/jarvis/policy.py`
