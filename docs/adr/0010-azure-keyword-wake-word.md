# ADR: Offer an Azure custom keyword model as a second wake word, run offline in its own process

**Status:** accepted
**Date:** 2026-11-21

## Context and problem statement

The owner finds that LiveKit's `hey_jarvis` model sometimes wakes Jarvis while they talk to other people or
sing, with no "Hey Jarvis" said. They trained two Azure AI Speech custom keyword models for "Hey Jarvis" and
"Jarvis" in Speech Studio (a Basic model, minutes to train, and an Advanced one, days) and asked for them to
be usable and switchable with the existing wake word in settings, with the licences updated as needed.

Two rules stand in the way of loading them the obvious way:

- Recognising a custom keyword needs Microsoft's Speech SDK (`azure-cognitiveservices-speech`). It is free to
  use but closed source, under the Microsoft Software License Terms: no reverse engineering, limited
  redistribution, and telemetry to Microsoft that can be turned off only in part. Loaded in Jarvis's process
  beside `piper-tts` (GPL-3.0-or-later), it would conflict with Jarvis's licence
  ([ADR 0006](0006-gpl-licence.md)).
- Voice activation stays local (AGENTS.md, [ADR 0005](0005-local-only.md)). The keyword model itself runs on
  the PC, but the SDK's telemetry would not.

## Considered options

- Keep only LiveKit's model and raise `audio.wake_threshold` (fewer false wakes, more misses; no new parts).
- Load the Speech SDK in Jarvis's process (the licence conflict above, and the telemetry).
- Run the SDK in a separate helper process, with its own Python and no network, and talk to it through pipes.
- Train another open model (openWakeWord) on more negative speech (no ready recipe for this voice; the owner's
  models already exist).

## Decision outcome

Chosen: a separate helper process, switchable with `audio.wake_engine` (`"livekit"`, the default, or
`"azure"`) and `jarvis wake livekit|azure [--model FILE]`.

- `src/jarvis/azure_wake_helper.py` is the only file that imports the SDK (ruff's ban has a one-file
  exception, pinned by `tests/test_product_rules.py`). Jarvis never imports it: it runs it with the Python of a
  separate environment (`<data_dir>/azure-wake/venv`, or `audio.azure_python`) that holds the SDK.
- The helper is under the MIT licence, so it may be combined with the closed SDK; Jarvis stays GPL and talks
  to it only through pipes. The SDK is not a Jarvis dependency: it is not in `uv.lock`, and Jarvis's
  licence check never sees it.
- Jarvis always starts the helper as `unshare --map-current-user --net -- <python> -I <helper> <model>`:
  a network namespace whose only interface is a loopback that is down, so the SDK cannot send anything off
  the PC. This is enforced in code (`jarvis.audio.azure_wake_command`) and tested.
- The microphone stays Jarvis's: each 80 ms frame is written to the helper, which answers `wake` when the
  keyword is heard. Writes never block: a helper that falls behind drops audio, not Jarvis.
- The helper tries the model on half a second of silence before it says it is ready, so a model the SDK
  rejects is caught at start. An Azure model that cannot start, or whose helper dies later, falls back to
  LiveKit's, so a switch never leaves Jarvis deaf; `jarvis doctor` runs the same check and names the cause.
- A detection arrives a little after the keyword; the helper reports where the keyword ended, and the
  audio since then (from 0.25 s before it) starts the request, so words said straight after are kept. The PC-audio listener uses the chosen engine too.
- The owner's Azure models also wake on "Jarvis" alone (a model trained with "Containing single keyword"
  does), which LiveKit's model does not, so while one is chosen a
  sentence Jarvis says that holds its own name cannot be spoken over (the own-voice check of
  [ADR 0008](0008-speak-while-writing-and-barge-in.md)).

### Consequences

- Good: the owner can try the Azure models against the false wakes and switch back with one command.
- Good: the SDK cannot reach the network, and Jarvis's own code and licence are unchanged by it.
- Bad: the namespace cuts only the network. The helper runs as the owner, so the SDK can still read the
  owner's files and reach local sockets (the session D-Bus); a mount namespace would be the next step.
- Bad: the own-voice check still uses LiveKit's model for the sound; with an Azure model, only a sentence
  naming Jarvis is held, so another sentence the Azure model mishears could still interrupt Jarvis.
- Bad: one more environment to set up by hand, and one more process (about 51 MiB, measured with Microsoft's
  sample keyword model) using about 2.4 % of one core while it listens, against about 1.4 % for LiveKit's.
- Bad: the model's own threshold decides; `audio.wake_threshold` does not apply to it. Speech Studio stops
  training new custom keyword models on 2027-08-01; existing models keep working.
- Measured on 2026-11-21 with Microsoft's sample model (`kws.table`, keyword "Computer") and Kokoro speech:
  both "Computer" lines in a 16 s clip were heard 0.3 to 0.6 s after the word ended, and "The weather is
  nice today" and "Hey Jarvis, pause the music" were not. The owner's own models were still training, so
  their false wakes against LiveKit's are not measured yet.

## Related

- [configuration.md](../configuration.md#azure-wake-word) (setup), [models.md](../models.md), [privacy.md](../privacy.md)
- [ADR 0005](0005-local-only.md), [ADR 0006](0006-gpl-licence.md), [ADR 0008](0008-speak-while-writing-and-barge-in.md)
