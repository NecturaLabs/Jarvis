# ADR: Hold the wake word to a lower score while Jarvis speaks

**Status:** accepted
**Date:** 2026-11-24

## Context and problem statement

[ADR 0008](0008-speak-while-writing-and-barge-in.md) lets the owner say the wake word over Jarvis to stop
it, and keeps the wake threshold as it is while Jarvis speaks. On the owner's PC (speakers, no echo
cancellation, a game playing), the owner spoke over a 35-second reply and Jarvis did not stop: the only
barge-in in the log came at its end. The log held no score that fell short, so a miss could not be told
from silence.

Jarvis's own voice reaches the microphone and lowers the score of the owner's wake word. Measured on
2026-11-24: the 330 synthetic "Yo Jarvis" clips of the wake benchmark (every second one of 660, 55 voices),
each through a simulated room, were mixed over Jarvis's reply speech (Kokoro `bm_george`, its own simulated
room) at the levels below, and scored with the owner's `yo_jarvis` model. Recall, the share of clips that
woke Jarvis:

| Wake word against Jarvis's voice | at 0.76 (the owner's threshold) | at 0.65 | at 0.5 | at 0.4 |
|---|---|---|---|---|
| Alone | 99.4 % | 100 % | 100 % | 100 % |
| 10 dB louder | 97.3 % | 98.5 % | 99.4 % | 99.7 % |
| 6 dB louder | 92.1 % | 97.3 % | 98.5 % | 99.1 % |
| 3 dB louder | 83.3 % | 93.3 % | 97.9 % | 98.5 % |
| As loud | 71.2 % | 83.3 % | 94.5 % | 97.6 % |
| 3 dB quieter | 51.8 % | 67.3 % | 84.2 % | 90.3 % |
| 6 dB quieter | 33.6 % | 51.8 % | 69.1 % | 79.1 % |

Jarvis's speech alone, through eight simulated rooms, never scored above 0.052.

## Considered options

- Keep the threshold and wait for echo cancellation. PipeWire's echo-cancel module broke the PC's sound
  output when it was tried ([#13](https://github.com/NecturaLabs/Jarvis/issues/13)), and it is the owner's
  system setting, not Jarvis's.
- Cancel the echo inside Jarvis from the sound it plays. Jarvis hands speech to `pw-play` ahead of
  playback and does not know when each sample leaves the speaker, which an echo canceller needs; it would
  have to be measured in the owner's room first.
- Hold the wake word to a lower score only while Jarvis speaks.

## Decision outcome

Chosen option: "a lower score while Jarvis speaks", `audio.barge_in_threshold`, 0.5 by default. A model whose
own threshold is lower keeps it. At 0.5, recall with Jarvis's voice as loud as the owner's is 94.5 % instead
of 71.2 %, and its own voice stays ten times below it.

- `WakeWord.heard_over_speech` is what the barge-in watcher asks; `heard` keeps the idle thresholds.
- The own-voice check (`OwnVoiceCheck`) tests each sentence against the same lower score, so a sentence that
  could stop Jarvis is still not spoken over.
- A stretch of speech that ends without a barge-in logs the closest score heard from 0.2 up, so the next
  miss can be told from silence.

### Consequences

- Good: the owner can stop Jarvis over its own voice most of the time, with no system change.
- Bad: other audio playing while Jarvis speaks wakes it a little more readily. Over the benchmark's media,
  adding `yo_jarvis` at 0.5 woke Jarvis 3.7 more times an hour, against about 0.2 at 0.76
  ([models.md](../models.md)); only the seconds Jarvis speaks count here. A false barge-in only stops the
  reply.
- The measurement is synthetic: the owner's room, microphone and speaker volume decide the real levels.
  The logged closest score is how to check it.

## Related

- `src/jarvis/audio.py` (`WakeWord`, `OwnVoiceCheck`), `src/jarvis/voice.py` (`_listen_while_speaking`)
- [configuration.md](../configuration.md), `audio.barge_in_threshold`
