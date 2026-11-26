# ADR: Sing a song the model writes as notes, with a speaking-voice singer and an optional GPU one

**Status:** proposed (the owner confirms the policy class and the off switch below)
**Date:** 2026-11-26

## Context and problem statement

The owner wants Jarvis to sing when asked, in any words ([ADR 0013](0013-free-speech-no-new-shortcuts.md): the
model decides what a request means, and Jarvis ships no stored tune, phrase or trigger). A song is sound Jarvis
makes, not words it says, so it has to play where speech plays: after the words before it, over the same
`pw-play` stream, ended by the wake word or `jarvis hush`, and silent while Jarvis is paused
([ADR 0008](0008-speak-while-writing-and-barge-in.md)).

Jarvis is local and free ([ADR 0005](0005-local-only.md)), runs beside games, and must use no GPU or CPU while
idle. The best singing-voice models need the GPU and a large download, which not every install has, so singing
has to work without them.

## Considered options

1. A single singer: the GPU model only.
2. A single singer: the speaking voice moved to each note on the CPU only.
3. Both, with the GPU model optional and the speaking voice as the fallback.

## Decision outcome

Chosen option: 3. The CPU singer ships ready and always works. The GPU singer sounds more like singing, but it
is 1.3 GB, needs Vulkan, and can fail, so it is an extra the owner installs and a failure falls back to the
CPU singer instead of leaving the owner without a song.

- **The tool contract.** One operation, `sing` (`src/jarvis/sing/ops.py`), with a flat string `notes` and an
  integer `tempo_bpm` (40 to 220). The score is a string because `operations.validate_args` takes only flat
  arguments, and a list of objects was not worth changing that shared validator. Each note is `TEXT PITCH
  BEATS` separated by commas (`hap- G4 1, py E4 1, ~ C5 1, _ 0.5`): a trailing `-` joins syllables into one
  word, `~` holds the sound before it on to a new pitch, `_` is a rest. The score is checked in code before any
  sound is made (`src/jarvis/sing/score.py`): at most 64 notes, 30 seconds and 2000 characters, pitches C2 to
  C6, 0.125 to 8 beats a note, letters only. Every refusal says what to change, so the model can try again.
  The model writes the song; the tool description names no song.
- **The policy class.** `sing` is class `safe` with `side_effect=True`. It changes nothing on the PC, like
  speaking, so it needs no yes of its own. It is still an action: outside text in the request (a web page) raises
  it to `confirm` ([ADR 0002](0002-typed-operation-policy.md)), it takes a slot in `max_actions`, and a reply
  that claims it sang is not challenged as a claim with no action behind it. `side_effect=False` would have
  skipped all three, and a page could then make Jarvis sing with no yes.
- **Where it plays.** The result carries a `Sound`; the voice's `Hooks.play` queues it in the speaker
  (`Speaker.queue_sound`), so it plays like a reply. A channel with no way to play it (the chat, `jarvis ask`,
  `speech-test`) is not offered the tool, and a call that still reaches one fails with a message the model is
  told. The speaker checks a song for its own wake word in chunks of 2.5 seconds, each against its own sound
  and only the words sung in it (the score gives each word's seconds), so only a stretch that sounds like it
  cannot be spoken over: a song that names the assistant once is guarded around that word alone, with an
  Azure keyword model too, which wakes on the name without "Hey". A song plays as it was made, its rests
  included; only a sentence's padding is trimmed. The model may end a request with no words after a song,
  `stay_quiet` included: that ends it in silence without the hush that drops what is queued, so the song plays
  in full. Only the wake word, `jarvis hush` and a pause stop a song once it is queued.
- **The two engines.** `[sing] engine` is `auto`, `speech` or `soulx`.
  - `speech` (`src/jarvis/sing/psola.py`): Kokoro says the words, Praat's overlap-add stretches each vowel to its
    note and sets its pitch. It uses the Kokoro files Jarvis already has and the new `praat-parselmouth`
    dependency (GPL-3.0-or-later).
  - `soulx` (`src/jarvis/sing/soulx.py`): SoulX-Singer, run by audio.cpp's `audiocpp_cli` on the GPU through
    Vulkan, one process per song that exits afterwards, so nothing holds the GPU while Jarvis is idle. The
    reference clip is the same song sung first by the CPU singer, so no recording ships and the song comes out in
    Jarvis's own voice.
  - `auto` is `soulx` when its files are installed, else `speech`. The CPU song is made first in every case, so
    when SoulX-Singer exits with an error, writes a file Jarvis cannot read, a file cut short of its own header
    or a song more than 0.25 s off the score's length, or runs past 20 seconds, that song is sung instead and
    the log says why. The edges of silence Jarvis asks SoulX-Singer for are taken off, so both singers' songs
    start on the score's first beat and keep its rests.
  - **Stopping a song being made.** Whatever would silence a song stops it while it is made: the wake word
    (the voice listens for it then, though nothing plays), `jarvis hush`, a pause, or Jarvis stopping. The CPU
    singer stops before its next phrase; SoulX-Singer is not started, or is killed with every process it
    started and its private folder removed. The call is reported as stopped (`tool_stopped` in the audit,
    "Stopped:" to the model), never as a song. The request's time left bounds it as well: the CPU singer stops
    when it runs out, and SoulX-Singer is given no more than it and is not started with under 5 s left, when
    the CPU song is sung.
- **SoulX-Singer is optional and never downloaded at run time.** `uv run tools/get_singer.py` is the only way
  the files arrive, run by the owner. It fetches each file from its pinned upstream release, checks its size and
  sha256 before keeping it, and takes only the runner, its libraries and licences out of the archive. The pins
  are in `src/jarvis/models.py`, which `jarvis doctor` hashes; it also checks the runner is executable with its
  libraries beside it. Jarvis starts the runner as a separate program and ships none of its files.
- **The off switch is `sing = "deny"`** under `[boundaries.classes]` (also `jarvis tools sing off`). That is
  this repository's convention for any tool ([ADR 0002](0002-typed-operation-policy.md)), so there is no separate
  `[sing] enabled` key. A denied `sing` is not offered, and `jarvis doctor` checks nothing of the singer.
- **A pin younger than the usual wait.** The runner pin is audio.cpp v0.9.6, released 2026-11-14 and 12 days old
  when pinned, under the two weeks this project usually waits for a release. It was the version measured. It stays pinned until the owner decides otherwise: wait for 2026-11-28 and
  re-pin the same release, or test v0.9.5. The archive and runner hashes are in `docs/models.md`.

### Consequences

- **Good:** Jarvis sings any song the model writes, in its own voice, with nothing stored or hardcoded, by voice
  only, and with no cost while idle. Without the 1.3 GB download it still sings.
- **Bad:** the CPU singer sounds buzzy and synthetic, since vowels are stretched up to 20 times. Nobody listened
  to either singer when this was written: pitch and length were measured, how they sound was not. SoulX-Singer's
  VRAM use and its effect on a game's frame time are not measured. Its weights are Apache-2.0, but the sources
  of its training songs are not named. Each song takes a slot in `max_actions`, and a song asked for after web
  text waits for a yes.

## Related

- [ADR 0002](0002-typed-operation-policy.md): classes, provenance and the off switch
- [ADR 0005](0005-local-only.md): local and free
- [ADR 0008](0008-speak-while-writing-and-barge-in.md): speech playback and barge-in
- [ADR 0013](0013-free-speech-no-new-shortcuts.md): no hardcoded phrasings
- `src/jarvis/sing/`, `src/jarvis/audio.py` (`Speaker.queue_sound`, `run_singer`), `tools/get_singer.py`,
  `docs/models.md` (Singing), `docs/configuration.md` (`[sing]`)
