# ADR: Speak a reply while the model writes it, and let the owner speak over Jarvis

**Status:** proposed
**Date:** 2026-11-21

## Context and problem statement

The owner found that Jarvis takes long to start speaking, and asked for the spoken reply to be streamed and for
the wake word to stop Jarvis mid-sentence so the request can be steered. Until now Jarvis waited for the whole
reply, synthesized all of it, then played it, with the wake word off while it spoke
([plan](../plan.md#speaking-barge-in-confirmation), issue #12).

Measured on 2026-11-21 (Strata 0.1.62, Qwen3.6-35B-A3B, Kokoro `bm_fable`), for a typical command:

| Stage | Time |
|---|---|
| End of speech (`audio.end_silence_s`) | 0.7 s |
| Transcription (`small.en`) | 0.4 to 0.7 s |
| First model call: reading the prompt (about 2,900 tokens: 941 of instructions, 1,970 of tool schemas) | about 4.5 s |
| First model call: writing a tool call | 0.3 to 1 s |
| The tool (Spotify) | 0.3 to 1 s |
| Second model call: reading only the new part, then writing the reply | 0.3 to 1 s |
| Synthesizing the whole reply | 0.3 to 0.4 s short, 1.4 s for four sentences |
| Silence at the start of Kokoro's audio | 0.23 s |

The largest cost is reading the prompt. Strata keeps its cache only for an exact continuation of the last
prompt and what it wrote: a new request that shares the system prompt and tools reused 0 tokens in every
probe, the `strata_prefix` mark Jarvis sends is ignored, there is no raw completions endpoint, and
`continue_final_message` is ignored, so Jarvis cannot prime the cache itself (`docs/models.md`). That
fix belongs to Strata. Within Jarvis, the time spent writing and synthesizing the reply can overlap with
speaking it.

Issue #12 set a constraint: only text that is final may be spoken, never words from a turn whose tool calls
have not resolved. In a stream, words are not known to be final until the message ends: the model can write
text and then call a tool, and Strata ignores `parallel_tool_calls: false`, so a `no_action` answer can be
followed by another call in the same message.

## Considered options

- Wait for each message to finish, then synthesize sentence by sentence (meets #12 to the letter; saves
  only synthesis time, so little for a short reply).
- Speak finished sentences as they are written, with the checks below (saves the writing time of all but the
  first sentence; a few words can come before a tool call in the same message).
- Make every step require a tool call, so each reply is a `no_action` argument (does not help: Strata still
  allows a second call after it).

For speaking over Jarvis:

- PipeWire's echo-cancel module, as the plan foresaw (a system change outside Jarvis, for the owner to make).
- The wake word stays on while Jarvis speaks, and each sentence is first run through a second copy of the
  wake model, so a sentence that would trigger it cannot be spoken over.

## Decision outcome

Proposed: speak finished sentences as they are written, and listen for the wake word while speaking, with the
own-voice check. The rules, all in code:

- Only the message's text, or the `reply` of a `no_action` call that comes first, is spoken; never a tool's
  arguments or reasoning. `no_action` is only streamed while the owner's settings let it run.
- A sentence that says something was done (`claims_action`) is held until the answer is whole, unless an
  action already succeeded in this request; the claim check then decides on the whole reply as before.
  Sentence ends are a subset of the claim check's, so checking the pieces gives the same answer as checking
  the whole.
- Once a tool call follows spoken words, nothing more of that message is spoken, the audio not yet heard is
  dropped, and the audit log records `spoken_before_tool_call`. This relaxes #12's constraint: words that
  passed the claim check may be heard before a call in the same message.
- At most 1,000 characters are spoken before the answer is whole.
- While Jarvis speaks, the wake word stops the speech and the request and starts a new one; a question
  spoken over counts as no. `audio.barge_in = false` turns this off.

Measured: the check costs 6 to 33 ms per sentence. In direct playback, Jarvis's ordinary replies scored 0.001
to 0.035 on the wake model (the threshold is 0.5), and only lines like "Hey, Jarvis here" scored above 0.9.

### Consequences

- Good: a reply starts after its first sentence is written and synthesized instead of after all of it; a
  long research answer starts seconds sooner. Leading silence is cut to 30 ms, and the owner can interrupt.
- Good: one player plays a whole reply, so sentences follow each other without a new player each time.
- Bad: Kokoro now synthesizes while Strata writes; both use CPU threads. Measured: a two-sentence chunk took
  1.11 s instead of 0.63 s, and writing slowed from 45.7 to 41 tokens/s.
- Bad: a few honest words ("Let me look.") can be heard before a tool call that the same message makes.
- Bad: without echo cancellation, loud speakers next to the microphone may still let Jarvis's own voice reach
  the wake model at a level the direct check did not see; the owner's room check (a `local` test) decides.
- The first model call still reads the whole prompt; a Strata engine that honours `strata_prefix` removes
  about 4 s from every request.
