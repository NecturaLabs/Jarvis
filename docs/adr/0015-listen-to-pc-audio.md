# ADR: Listen to PC audio for the owner through the wake-word watcher's frames

**Status:** accepted
**Date:** 2026-11-26

## Context and problem statement

With `audio.listen_to_pc` on, Jarvis scans what the PC plays (Discord by default) for the wake word, so a person
in a call can ask for something ([ADR 0003](0003-pc-audio-is-not-the-owner.md)). The owner also wants Jarvis to
hear what is said or played there when the owner asks ("what are they saying?"), not only to wake words in it.
Before this, the model said it could not hear PC audio at all.

Speech-to-text stays local, through the Voxtype command the voice already runs (`voxtype transcribe`, product
rule). Jarvis runs beside games, so listening must add no CPU work while idle. What other people say on a call is
theirs: it must be held no longer than a request needs, and in memory.

## Considered options

1. A second capture of PC audio, opened for each listen.
2. Hand the frames the PC watcher already reads to a listen, and keep a short recall of the last seconds only
   when the owner sets one.
3. Keep a rolling recording of PC audio at all times, so any past moment can be asked about.

## Decision outcome

Chosen option: 2, because the watcher already holds every frame for the wake word, so a listen opens nothing
new and nothing runs while no one listens. A recall is useful ("what did they just say?") but keeps others'
words before anyone asked, so it is off by default and bounded when on. Option 1 would start a second
`pw-record` per listen and race the watcher for the same streams; option 3 keeps audio nobody asked for.

- **The operation.** `pc_audio_listen` (`src/jarvis/pc/hearing.py`) hears the next `seconds` (3 to 60) of PC
  audio, or with `when=last` the seconds before the request, and returns what was said. It is `safe` and
  read-only, offered only on the voice channel and only while `audio.listen_to_pc` is on. The chat never
  hears PC audio. `jarvis ask` is offered the operation, so `jarvis tools` and `boundaries.classes` know it,
  but every listen there is refused: it has no voice service hearing.
- **Where the frames come from.** `VoiceLoop._heard` passes each frame the PC watcher reads to `PcHearing`
  while it wants one (a listen runs, or a recall is set). A pause drops what is held, and the end of the
  stream the recall (a listen that runs goes on with the next stream's frames); a listen ends at a pause even while no frame comes (it checks every 0.2 s), and at a suspend, so audio from
  both sides of a sleep is never joined. Nothing is taken while Jarvis speaks, so its own voice is never heard
  back when all output is listened to.
- **What is kept.** In memory: the frames of the listen that runs, and with `audio.pc_audio_recall_s` (0 by
  default, else 3 to 60) the last that many seconds, overwritten as they go and timed by the boot clock, so audio
  from before a sleep is dropped. The one place it reaches the disk is the private temporary WAV that
  `voxtype transcribe` reads, deleted straight after, as for every request. The audit log keeps the tool
  result only as a SHA-256 and a length (`audit_text=False`); the reply that follows is audited as written,
  like every reply, and may repeat what was heard. A say-nothing hallucination (`is_phantom`) counts as no
  speech.
- **Speech-to-text.** The unhinted transcriber the voice uses for a yes or no (`voxtype transcribe` with Jarvis's
  own settings). The vocabulary hint names the owner's tools and music, which another person's words need not
  hold.
- **Provenance.** The transcript is `Origin.PC_AUDIO` data, never instructions. As an untrusted origin it puts a
  later side effect behind the owner's yes, and as a private one (`_PRIVATE_ORIGINS` in `agent.py`) it puts a
  search that would send words off the PC behind a yes too, as a file's text does. With
  `boundaries.bypass_consent` on (the default), those questions are skipped as for every other outside text;
  a `strong` action still needs the key, and `stop_process` and `pc_shell` refuse a request holding it.
- **Who may ask.** Only the owner: a request heard in PC audio, or one outside any request, is refused before
  anything is heard.
- **The prompt.** With listening on, the voice prompt says whose audio Jarvis hears, that a request from there is
  that person's, and to use `pc_audio_listen` when asked. Off, it says Jarvis cannot hear PC audio.

### Consequences

- **Good:** the owner can ask what is said on a call or in a video, at no idle cost, with nothing kept on disk.
  While Jarvis asks first, what was heard cannot steer an action or leave the PC without the owner's yes.
- **Bad:** a listen holds the request for as long as it listens (up to 62 seconds), then for its
  transcription. `agent.max_seconds` is checked only before a call starts, so a listen begun late in a request
  can carry it past that time by as much. With `bypass_consent` on, text from a call can steer an action that
  is not `strong` with no question, as web or file text can. With no recall, a question about what was just said can only be answered from now on.
  Speech while Jarvis talks is not heard, in a call too. A listen hears only the apps in `pc_audio_apps`.

## Related

- [ADR 0003](0003-pc-audio-is-not-the-owner.md): PC audio is someone else's speech
- [ADR 0002](0002-typed-operation-policy.md): provenance and confirmation
- `src/jarvis/pc/hearing.py`, `src/jarvis/voice.py`, `src/jarvis/agent.py`, `docs/privacy.md`
