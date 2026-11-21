# ADR: Speak a reply while the model writes it, and let the owner speak over Jarvis

**Status:** accepted
**Date:** 2026-11-21

## Context and problem statement

The owner found that Jarvis takes long to start speaking, and asked for the spoken reply to be streamed and for
the wake word to stop Jarvis mid-sentence so the request can be steered. Until now Jarvis waited for the whole
reply, synthesized all of it, then played it, with the wake word off while it spoke
([plan](../plan.md#speaking-barge-in-confirmation), issue #12).

Measured on 2026-11-21 (Strata 0.1.62, Qwen3.6-35B-A3B, Kokoro `bm_fable`, not the default `bm_george`), for a
typical command:

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

Built as described and accepted on 2026-11-21, after the owner asked for the open decisions to be researched
and settled ("research online, do what u think is best"); see Research below.

Speak finished sentences as they are written, and listen for the wake word while speaking, after checking
that each of Jarvis's own sentences does not trigger it. The rules, all in code:

- Only the message's text, or the `reply` of a `no_action` call that comes first, is spoken; never a tool's
  arguments or reasoning. `no_action` is only streamed while the owner's settings let it run.
- A sentence that says something was done (`claims_action`) is held until the answer is whole, unless an
  action already succeeded in this request; the claim check then decides on the whole reply as before.
  Sentence ends are a subset of the claim check's, so checking the pieces gives the same answer as checking
  the whole.
- Once a tool call follows spoken words, nothing more of that message is spoken and the audit log records
  `spoken_before_tool_call`. Sentences of the message's text already passed on play to their end: they
  passed the claim check, and dropping them at the call cut a word in half. Words of a `no_action` reply
  that turns out to be refused (its arguments are checked only once whole) or not alone are dropped where
  they were not yet heard; an exact copy of the call, which the agent drops, changes nothing. This relaxes
  #12's constraint: words that passed the claim check may be heard before a call in the same message, or
  from a reply call that is then refused.
- When the claim check challenges the whole reply, the sentences already heard stay heard and the corrected
  answer follows them.
- A stream that ends without the server saying it is finished is a model failure, not a reply.
- At most 1,000 characters are spoken before the answer is whole.
- While Jarvis speaks, the wake word stops the speech and the request and starts a new one; a question
  spoken over counts as no, and no further line of that request is spoken. `audio.barge_in = false` turns
  this off.
- With PC audio on, a wake word heard there while Jarvis speaks is noted, so a barge-in that echoes it is
  treated as PC audio, as an idle wake heard on both already was ([ADR 0003](0003-pc-audio-is-not-the-owner.md)).
- Once a sentence about to play sounds like the wake word, the rest of that stretch of speech cannot be
  spoken over (sentences are checked as they are synthesized, ahead of playback). A check that fails counts
  as wake-like.

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
  about 4.5 s from every request.

## Research

Read on 2026-11-21:

- Pipecat and LiveKit Agents both speak text the model writes before a tool call, and on a barge-in keep
  only the words the user heard
  ([Pipecat](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/processors/aggregators/llm_response_universal.py),
  [LiveKit turns](https://docs.livekit.io/agents/logic/turns/)). LiveKit users report a preamble cut to its
  first word when the tool starts as a defect
  ([thread, 2026-05-18](https://community.livekit.io/t/we-need-to-talk-about-pre-tool-call-speech/1152));
  that is why the message's text now plays to its end.
- OpenAI's voice guidance asks for one short sentence before a slow tool and "only say an action was
  completed after the tool call succeeds", in the prompt only
  ([voice prompting guide](https://developers.openai.com/api/docs/guides/voice-prompting)). Jarvis enforces
  the same in code with the claim check. A Retell thread reports an agent that announced a transfer and
  never ran it ([thread](https://community.retellai.com/t/agent-speaks-the-transfer-announcement-but-never-executes-the-transfer-tool/3487)),
  the failure the hold prevents.
- No framework documents wake-word barge-in. Practitioner projects lower the threshold during playback or
  ignore the mic for a while after speech; the per-sentence check with a second wake model is stricter than
  either and does not make a real barge-in harder.
- PipeWire's echo-cancel module removes what the PC plays from the microphone;
  [configuration.md](../configuration.md#speakers-and-other-apps-speech) gives the setup. It stays the
  owner's choice, as a change to the PipeWire setup.
