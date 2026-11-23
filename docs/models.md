# Models and voices

What Jarvis loads, where each file goes, what it costs in time and what licence it carries. `jarvis doctor`
points here when a file is missing. Licences are taken from package metadata, from the licence field Jarvis
pins beside each model's sha256 (`src/jarvis/models.py`), or from a file shipped in the wheel; where nothing
was checked it says so.

## The model

Jarvis serves whichever model Strata's own run config names (`strata.run_config`, the
`strata-<model>.json` that `strata launch` writes): the model, its engine, flags, tokenizer and picture
reader come from that file, and Jarvis adds only its limits ([configuration.md](configuration.md#strata-how-the-jarvis-strata-service-runs-the-model)).
To change models, run `strata launch` for the new one once (it downloads the files and writes the run
config; stop the server it starts), then `jarvis setup --run-config <its file>` and `jarvis restart`.
The files live outside the repository, wherever Strata's data folder puts them. `jarvis setup` refuses a
run config that names a file that is gone. The reference machine for every number below: RX 7900 XT
(20 GB), 64 GB RAM, Strata 0.2.20 (engine 0.1.66), unless a row says otherwise.

| Model | Engine | Files | Licence | Beside a game (8 GiB of VRAM) |
|---|---|---|---|---|
| **Qwen3.8-Flash-Next** IQ2_XS (ISTA-DASLab), 512 experts, about 6B active per token | `strata-engine` | 68 GB, plus its picture reader and DFlash2 drafter | Qwen Community License 1.0 | Works: measured below |
| **Inkling 2 Mini** SQ-15 (Thinking Machines), 128 experts, 3.4B active | `strata-inkling` | 15 GB with its picture reader | Apache-2.0 | Too slow: measured below |
| **Qwen3.6-35B-A3B** UD-Q4_K_XL (Unsloth), 256 experts, 3B active | `strata-engine` | 22.9 GB, plus `mmproj-F16.gguf` | Apache-2.0 | Works: the figures under Latency |

- **Qwen3.8-Flash-Next** with `strata.vram_reserve_mib = 12000` (measured 2026-11-23, Jarvis's own
  profile from Strata's run config): the engine held 7.95 GiB of VRAM and 35.5 GiB of RAM; a request with
  two tool calls took 0.65 to 0.9 s once loaded, writing 60 to 71 tokens per second; the second request
  reused 329 of its 336 prompt tokens, because this engine keeps Jarvis's shared prefix (`strata_prefix`)
  between requests; loading took 13 to 17 s with the files in the OS cache. With Strata's own defaults it
  took all 20 GB of VRAM. Under the Strata unit's limits it needs `strata.memory_max = "44G"`: at the default
  40G the unit sat at the mark with 35.1 GiB of its own memory and the rest file cache trimmed 2,096 times in
  a few requests; at 44G it never reached the mark (peak 38.2 GiB with a picture read). A picture of a red
  square was read as "Red" in 0.7 to 0.9 s. Its licence allows personal and commercial use, asks for the notice to be kept,
  and needs a separate licence only for a hosted model service or an AI work assistant offered to others;
  Jarvis uses it locally and ships none of its files (checked against the model's `LICENSE` on
  2026-11-23).
- **Inkling 2 Mini** with `strata.expert_vram_gib = 4` (measured 2026-11-23): 7.4 GiB of VRAM and 8 GiB of
  RAM, but 6 to 7 tokens per second and 17 s for the same request, and no prompt reuse between requests
  (Strata issue [#6764](https://github.com/Niko1221/Strata/issues/6764)). With all its experts on the card
  (12.3 GB, Strata's default) it wrote 70 tokens per second in 1.6 s, which leaves no room for a game.
  Its engine is three days old in this release, with open issues that touch Jarvis: hidden thinking in a
  reply ([#6675](https://github.com/Niko1221/Strata/issues/6675)) and long prompts failing on RDNA3 cards
  ([#6835](https://github.com/Niko1221/Strata/issues/6835)).
- **Qwen3.6-35B-A3B** was the model until 2026-11-23; its profile used to be built by Jarvis itself. Its
  latency and memory are below.

### Thinking

Since 2026-11-23 Jarvis asks for `low` thinking (`[agent]` and `[chat]` `thinking`), capped at
`thinking_tokens` (192 for speech, 512 in the chat). Strata's engine for the Qwen models takes `low`,
`medium` and `high`. Every request asks for the same effort, so the prompt start Strata keeps
(`strata_prefix`) is the same from one request to the next.

Measured on 2026-11-23, with a game running, through Jarvis's own agent, prompt and tool schemas (27
spoken requests: Spotify, time and weather, sums, PC state, files, programs, reminders, research,
follow-ups and a page with planted instructions; every tool a recording stub):

| Model and thinking | Handled as expected | Median per request | Slowest |
|---|---|---|---|
| Qwen3.8-Flash-Next, `none` | 27 of 27 | 1.4 s | 6.7 s |
| Qwen3.8-Flash-Next, `low` | 26 of 27 (the miss was a right answer the check did not expect) | 2.6 s | 8.0 s |
| Inkling 2 Mini, `none` | 22 of 27 | 68 s | 104 s |
| Inkling 2 Mini, `low` | 16 of 27 | 90 s | 100 s |

On these requests `low` thinking made Qwen3.8 about 1.2 s slower per request without making it more
accurate. It stays on by default for harder requests; `thinking = "none"` in `[agent]` takes the time back.
Inkling 2 Mini, with 4 GiB of its experts in VRAM so a game keeps its room, picked the right tools in
almost every request, but most of its misses were Jarvis's 90 s request limit, and some replies held its
raw tool-call markup (`<|content_invoke_tool_json|>`), which would be spoken. Qwen3.8 stays the model.

### Experts

A mixture-of-experts model keeps some experts on the card and the rest in RAM. On an engine whose run
config starts from a routing profile (`--expert-profile`, Qwen3.8-Flash-Next), `strata.learn_experts`
lets the engine save which experts Jarvis's own requests use (`~/.local/state/jarvis/strata-expert-profile-<model>.bin`, one per model,
when it stops and every 10 minutes between requests) and start from them: those experts are on the card from
the first request, instead of the ones Strata's generic profile ranks first. `jarvis setup` starts that file
from a copy of Strata's own profile, so Jarvis's profile names it from the start and never has to change.
Verified on 2026-11-23: the engine saved it on exit and loaded it at the next start.

Removing experts from the model (REAP pruning) was looked at on 2026-11-23 and not taken: it saves memory,
not time (each token still runs 10 experts), the pruned Flash-Next builds published are 61 to 82 GB (more
than this 68 GB IQ2_XS pack, whose experts take 35.5 GiB of RAM), most are pruned on coding traffic, the
best documented one invents answers more often (25% on its authors' own check, against 0% unpruned), and
none comes as a Strata pack.

### Pictures

- Strata's run config names the model's picture reader, and `jarvis doctor` hashes it. For Qwen3.6 it is
  `mmproj-F16.gguf` (899,283,584 bytes) from `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` at revision
  `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`, Apache-2.0, sha256 `71f3cbc1…0f160`.
  Qwen3.8-Flash-Next's reader is `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (907,543,008 bytes) from
  `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, sha256
  `b1a82259…49bd0`, under Qwen's own licence (the quantiser's card says apache-2.0; the weights are Qwen's).
  Inkling 2 Mini's reader is `Inkling-2-Mini-Vision.strata` (636,526,592 bytes) from `strata-engine/Inkling-2-Mini-Strata`
  at revision `04996a1e2c45009b6a04f2af126cb6bdb2ba3587`, Apache-2.0, sha256 `6589eff4…6fc677`. Each digest was
  checked against the file on this PC, the one Strata recorded at download, and the upstream LFS record
  (2026-11-23).
- On the main engine Strata runs the reader on the processor (`strata-vision`): it starts with the first
  picture (a few seconds) and stops after `strata.vision_idle_s`; a picture becomes at most 300 tokens.
  Inkling's engine reads pictures itself. `strata.pictures = false` drops the reader.

### Latency (Qwen3.6-35B-A3B)

| Request | What happens | Cost |
|---|---|---|
| Plain command: "pause", "next song", "volume 40", "what's playing" | Matched in code, run through the same policy, confirmation and audit as a model call, answered from the operation's own sentence. No model, no load wait. | about 0.3 s |
| Anything else the model answers (a chat reply is one model call: `no_action` carries the reply) | One prompt read of about 2,900 tokens (2,100 before the web tools), then generation; a call after a tool result reads only the new part | about 4.5 s of prompt reading at about 640 tokens/s, plus generation; the first sentence is spoken once it is written ([ADR 0008](adr/0008-speak-while-writing-and-barge-in.md)) |
| Model not loaded | Cold load | about 107 s with a game running; Jarvis says "One moment, waking up". The request gets 180 s more (270 s in all instead of 90 s), and so does its first model call, which waits for the load (`model.request_timeout_s` plus 180 s); later calls keep `model.request_timeout_s` |

Memory while loaded, idle: about 38 GB of RAM and 7.3 GiB of VRAM at an 8,192-token context, no GPU or CPU
use. VRAM grows by about 0.05 GiB per 1,000 tokens of context (the engine's `drm-memory-vram` in
`/proc/<pid>/fdinfo`, measured on 2026-11-21 after each `jarvis restart`): 7.71 GiB at 16,384, 8.00 GiB at
22,528 and 8.10 GiB at 24,576. The default, 24,576 (the owner's choice, slightly over 8 GiB), holds a full research request
(`tests/test_app.py`) with room for Jarvis's limits to grow; 16,384 is the smallest that still holds
one, if VRAM is tight. Speed at 16,384 to 22,528 tokens
(2026-11-21): about 50 tokens/s writing a reply; about 700 tokens/s reading a long prompt, so a request
carrying 10,800 tokens of results waits about 15 s before the first word. RAM stayed between 21 and 37 GB
(`MemoryCurrent` of `jarvis-strata.service`; it varies with how much of the mapped model is paged in). On 2026-11-21
`tools/idle_check.py` measured 30 s of the running services while idle: `jarvis.service` used 1.4 % of one core
(the wake word and end-of-speech models score every 80 ms frame), `jarvis-strata.service` 0.1 %, and neither
grew in memory. The GPU could not be measured that way: Strata's engine holds `/dev/kfd` (ROCm compute) and an
amdgpu DRM client, and amdgpu published no `drm-engine` busy counters for that client in `/proc/<pid>/fdinfo`. The check's default limit of 5 % of one core leaves room for the wake word,
which has to listen. See
[configuration.md](configuration.md#keeping-the-model-loaded).

#### Prompt chunk size and the shared prefix

The prompt chunk (`--prefill`) is whatever Strata's setup wrote into the run config (`auto` for
Qwen3.8-Flash-Next). Qwen3.6's profile, which Jarvis built itself until 2026-11-23, used 512, a size
Strata's own tuner avoids for that model family (smaller chunks read long prompts 11% to 75% slower in its
tests).

Jarvis marks the system prompt and the tools as Strata's shared prefix (`strata_prefix`, one message) on every
request that sends tools. Qwen3.8-Flash-Next's engine keeps it between requests (measured above). The
Qwen3.6 engine in Strata 0.1.62 does not use that mark, so every request still reads the whole prompt. An engine that ignores the mark is harmless; the 4.5 s above is that cost.
Strata's own tracker has it as [issue #5145](https://github.com/Niko1221/Strata/issues/5145): the Qwen3.6
engine drops the mark and keeps no checkpoints between requests, which its author calls a new feature, not
a small fix (still so in 0.2.17).
Probed on 2026-11-21: the engine reuses its cache only when a request continues the last prompt and what the
model wrote after it, token for token (the second call of a request reused about 2,150 tokens); two requests
that share the system prompt but differ after it reused 0. The engine has no raw `/v1/completions` endpoint
and ignores `continue_final_message` and `parallel_tool_calls`, so Jarvis cannot prepare the cache itself.
A second request that repeats the first one and its reply exactly and adds a new question also reused 0,
with the reply as written, with the empty think block written back into it, and with Qwen's
`preserve_thinking` and an empty `reasoning_content` (probed 2026-11-21). Keeping more history would not
help: the cache is reused only between calls within one request. llama.cpp gets reuse on such hybrid models from
saved checkpoints of the recurrent state (`--ctx-checkpoints`), so the fix belongs in the engine.

## Speech to text: Voxtype

Jarvis writes each request to a private temporary WAV file and runs `voxtype transcribe`, never the Voxtype
daemon and never your Voxtype config or recordings. Each run is
`voxtype -q -c <config dir>/voxtype.toml [--model M] [--engine E] [--initial-prompt HINT] transcribe FILE`.

- **Settings file:** Jarvis writes `voxtype.toml` in its own config folder at each start with `threads`
  (`audio.transcribe_threads`, default 8) and `context_window_optimization` (`audio.transcribe_short_window`,
  default on) under `[whisper]`, and `keep_failed_recordings = false` under `[audio]`, so Voxtype keeps no audio of a
  failed transcription from Jarvis's runs. `-q` (errors only) keeps Voxtype's log lines out of the transcript. Whisper pads every clip to a 30 s window, so without the short window a
  two-second request costs as much as a long one; with it a short command took 0.29 s instead of 1.66 s
  (commit `d29d73e`, measured with `small.en`). Whisper.cpp also uses only 4 threads unless told otherwise.
- **Model:** `audio.transcribe_model`, default `small.en`; it must already be downloaded through Voxtype
  (`voxtype setup model`), and `jarvis doctor` checks that it is. In that same measurement `small.en` took
  0.29 s on a short command against 0.49 s for `base.en`, and was more accurate.
- **Hint:** when `audio.vocabulary_hint` is on, `--initial-prompt` carries Jarvis's name, its actions and up to
  `audio.hint_names` of your artist names, shortened, `Jarvis. play, pause, next. Glass Harbor, Night Circuit.`
  It always ends with a full stop: without it Whisper re-decoded short clips and took 1.8 s. A transcript that
  is only a run of the hint's own words is dropped. The answer to a confirmation question is transcribed
  without a hint, so noise cannot turn into words.
- Whisper's stock phrases on silence ("thank you for watching") are ignored.
- Licence: Voxtype is MIT (its Arch package), Whisper's code and weights are MIT (openai/whisper) and the
  `ggml-*.bin` conversions come from whisper.cpp, MIT; checked on 2026-11-21.

## Voices

`audio.voice` picks the voice. Both voices run on the CPU; Kokoro uses at most 8 threads and never spin-waits.
Jarvis synthesizes one sentence at a time while the one before plays, and cuts the silence before the first
word to 0.03 s and after each sentence to 0.2 s. Measured on 2026-11-21 with `bm_fable` (not the default): a short reply takes
0.3 to 0.4 s, a first sentence 0.2 to 0.4 s, four sentences 1.4 s. While Strata writes, a two-sentence chunk
took 1.11 s instead of 0.63 s, and Strata wrote 41 tokens/s instead of 45.7. Before each sentence plays, a
second copy of the wake model checks it (6 to 33 ms), so a sentence that would trigger the wake word cannot be
spoken over while it plays and for a second after (`GUARD_TAIL_S`, the player's buffer and the wake model's lag);
the sentences before and after it can.

### Kokoro (default, `kokoro:<name>`)

- Kokoro-82M on ONNX Runtime, 24 kHz output. Jarvis's own `src/jarvis/kokoro.py` runs it; the default voice is
  `bm_george` (a name starting with `b` uses British English phonemes, otherwise American).
- Files, under `<data_dir>/voices/kokoro/`: `kokoro-v1.0.onnx` and `voices-v1.0.bin` from the kokoro-onnx
  `model-files-v1.1` release. `jarvis doctor` checks their sha256 against the pinned release.
- Phonemes come from espeak-ng, whose library and data ship inside the `piper-tts` wheel, so the Piper package
  is installed and imported even when you use Kokoro. Raw espeak symbols are what the kokoro-onnx package feeds
  too. Kokoro was trained on the symbols of misaki, which rewrites espeak's (for example `eɪ` to `A`, `aɪ` to
  `I`, `tʃ` to `ʧ`, `r` to `ɹ`); `audio.kokoro_phoneme_map` applies that table (misaki 0.9.4, `EspeakFallback`,
  Apache-2.0), off by default until it is heard to help. Over about 2,200 words of these docs the table loses
  no symbol to the model's vocabulary; whether it sounds better is not yet heard.
- Licence: Apache-2.0 for the Kokoro-82M v1.0 weights and voices, rechecked against the
  [model card](https://huggingface.co/hexgrad/Kokoro-82M) on 2026-11-21. The card says it was trained only on
  public-domain audio, audio under permissive licences (Apache, MIT and the like) and synthetic audio from
  closed TTS models; two of its datasets are CC BY: Koniwa `tnc` (CC BY 3.0) and SIWIS (CC BY 4.0). The ONNX
  export and `voices-v1.0.bin` come from [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), MIT. Jarvis
  ships none of these files; you download them.
- Time to first audio, measured on 2026-11-21 on the reference machine (Ryzen 7 9800X3D, desktop load about 2.5): one
  first sentence of a typical reply (one to nine words) took a median 0.24 s to synthesize and trim, 0.28 s at
  the 90th percentile and 0.30 s at most, over 23 sentences, with `bm_george` and with `bm_fable` alike. The first
  sentence after loading the voice took 0.40 s with the files in the page cache and 0.96 s from cold. Opening
  the player and the 0.03 s of kept silence come on top; the time from the end of the owner's speech to the
  first word is in [ADR 0008](adr/0008-speak-while-writing-and-barge-in.md).

### Piper (any other `audio.voice` value)

- `piper-tts` 1.8.0 (OHF-Voice/piper1-gpl). Put a voice at `<data_dir>/voices/<name>.onnx` with
  `<name>.onnx.json` beside it and set `audio.voice = "<name>"`. `audio.speech_speed` sets Piper's length
  scale (the voice's own scale divided by the speed).
- Licence: `piper-tts` is GPL-3.0-or-later (package metadata). Each Piper voice has its own licence: check
  its model card before using it; Jarvis ships none. Checked 2026-11-22 for `en_GB-alan-medium`: its card
  names the dataset `MycroftAI/mimic3-voices` `en_UK/apope_low`, whose LICENSE reads "Copyright 2022 Mycroft
  AI / All Rights Reserved" while the repository's root LICENSE is CC BY-SA 4.0, so that voice's terms are
  unclear. Because Jarvis imports `piper-tts` for both voices, Jarvis is published under GPL-3.0-or-later
  (`docs/adr/0006-gpl-licence.md`).

## Wake word and end of speech

- **Wake word:** the LiveKit `hey_jarvis` model, run by `livekit-wakeword` 2.1.3 on CPU, at
  `<data_dir>/wake/hey_jarvis.onnx`. Threshold: `audio.wake_threshold`, or its own in
  `audio.wake_thresholds`. Licence: Apache-2.0 for the code (package metadata) and the weights (the wheel's
  `NOTICE`); the file is pinned to release 2.1.3 in `models.py`.
- **More wake phrases (optional):** any model `livekit-wakeword train` makes (2.x, standard type), put at
  `<data_dir>/wake/<name>.onnx` and named in `audio.wake_words` (`jarvis wake --words`). The default list
  also names `yo_jarvis`, which you train yourself: Jarvis ships no trained phrase, and the doctor notes it
  as missing until you do or take it off the list. Training needs a CUDA GPU or an Apple silicon Mac and a
  LiveKit Cloud project for the generated speech; a model you trained matches no pinned release, so the
  doctor only shows its digest. Detection stays local and needs no account.
  Measured 2026-11-22 for models trained with `livekit-wakeword` 2.1.3 (`setup --wakeword "<phrase>" --plan
  build --eval`, `train --type standard`, `eval --json`), against the pinned `hey_jarvis` release, as the
  [wake training playbook](playbook/wake-training.md) says. Two measurements:

  **The trainer's reduced test set** (without its gated ESC-50 and GTZAN sets, so music and everyday sounds are
  not in it): 500 spoken phrases and 796 other clips, about 1.01 hours, of which 500 are similar-sounding
  phrases. These rates rest on about an hour: no false wake there still allows about 3 an hour.

  | Model | Recall at 0.5 | False wakes at 0.5 | Similar phrases that woke it at 0.5 |
  |---|---|---|---|
  | `hey_jarvis` (pinned release) | 92.4 % | 0 in 1.01 h | 20.6 % |
  | `hey_jarvis` (retrained) | 92.6 % | 0 in 1.01 h | 18.6 % |
  | `yo_jarvis` | 94.8 % | 0 in 1.01 h | 14.4 % |
  | `jarvis` (bare word) | 96.0 % | 5 in 1.01 h | 3.0 % |

  **`tools/wake_bench.py`**: 3,595 clips from 55 local Kokoro and Piper voices (none of them LiveKit's), each
  clean, quiet, under TV and music at 10, 5 and 0 dB SNR, with room echo and far away; 24 hours of real media
  (10 h TV and YouTube, 4 h music, 10 h parliament speeches); and replays of that media through the models
  together as Jarvis listens (every model reset after a wake, then 4 s not heard). Recall is on the model's own
  phrase; the interval is 95 %.

  | Model, threshold | Clean | 5 dB SNR | 0 dB SNR | Far | False wakes on the media, model alone |
  |---|---|---|---|---|---|
  | `hey_jarvis` (pinned), 0.5 | 96.5 % | 91.2 % | 81.2 % | 97.9 % | 2.83 an hour (68 in 24 h; 2.20 to 3.59) |
  | `hey_jarvis` (retrained), 0.5 | 95.9 % | 90.3 % | 78.6 % | 96.8 % | 1.33 an hour (32 in 24 h; 0.91 to 1.88) |
  | `yo_jarvis`, 0.5 | 100 % | 97.9 % | 91.5 % | 100 % | 4.46 an hour (107 in 24 h; 3.65 to 5.39) |
  | `jarvis` (bare word), 0.85 | 81.5 % | 69.1 % | 46.8 % | 77.7 % | 0.08 an hour (2 in 24 h; 0.01 to 0.30) |

  The retrained `hey_jarvis` hears fewer of the same clips than the pinned one under noise (2.7 points fewer
  at 10 dB SNR, McNemar p = 0.004; 2.6 at 0 dB, p = 0.021), so the pinned release stays. Together as Jarvis
  listens, the pinned `hey_jarvis` at 0.5 woke 2.75 times an hour on the media (66 in 24 h) (5.25 on music, 4.4 on TV, 0.1
  on speeches); adding `yo_jarvis` at 0.5 raised that to 6.46 an hour (155 in 24 h). A threshold picked on one
  half of the media for at most 0.5 false wakes an hour, and checked on the other half: 0.68 for the pinned
  `hey_jarvis` (2 in 12 h, recall 93.8 % clean, 75.8 % at 5 dB SNR) and 0.76 for `yo_jarvis` (2 in 12 h,
  recall 100 % clean, 90.8 % at 5 dB SNR). Replayed at those thresholds, the pinned `hey_jarvis` at 0.68 woke 0.25
  times an hour (6 in 24 h); with `yo_jarvis` at 0.76 beside it, 0.54 an hour (13 in 24 h); and `yo_jarvis` at
  0.76 beside the pinned model at 0.5 added 0.21 an hour (71 against 66 in 24 h). The bare `jarvis` model also woke on "Jervis" (98 % of voices), "Pass
  me a jar of this" (85 %), "Java" (60 %) and "Service" (53 %), so it is not on. No model woke on Jarvis's own
  replies. The trainer's own `eval --media` on the same media agrees with the benchmark for every model: 2.833
  false wakes an hour for the pinned `hey_jarvis`, 1.333 for the retrained one and 4.458 for `yo_jarvis` at
  0.5, and 0.083 for the bare `jarvis` at 0.85. These are synthetic voices and recorded media, not your room: the playbook's desk
  test comes before a default changes.
- **Azure wake word (optional, `audio.wake_engine = "azure"`):** your own Azure custom keyword `.table`
  model, heard by `src/jarvis/azure_wake_helper.py` (MIT) with Microsoft's Speech SDK
  (`azure-cognitiveservices-speech` 1.53.1, Microsoft Software License Terms: closed source, not
  GPL-compatible, telemetry). The SDK is not a Jarvis dependency: it lives in its own environment, and the
  helper runs in a separate process with no network ([ADR 0010](adr/0010-azure-keyword-wake-word.md)).
  Measured with Microsoft's sample model: about 51 MiB and 2.4 % of one core while listening, and a keyword
  heard 0.3 to 0.6 s after the word ended. The `.table` model is your own, trained under Microsoft's terms for the Speech service, and is never
  shipped with Jarvis. Setup: [configuration.md](configuration.md#azure-wake-word).
- **End of speech:** Silero VAD 6.2.3 at `<data_dir>/vad/silero_vad.onnx`, MIT as pinned in `models.py`; not
  rechecked upstream. Threshold and timings: `audio.vad_threshold`, `vad_neg_threshold` (speech goes on
  above it once started; default `vad_threshold` less 0.15, as Silero's own segmenter does),
  `no_speech_timeout_s`, `end_silence_s`, `max_utterance_s`. The defaults for the margin and for the 0.15 s of
  speech a request after the wake word needs were reasoned from the code, not tuned on a voice (2026-11-22):
  the margin is Silero's own (`threshold - 0.15`), and with 0.7 s of silence needed to end a request it only
  keeps a breath or a soft syllable from counting as silence; 0.15 s is two 80 ms frames, the minimum a
  confirmation answer already has (a click is one frame), and less than the 0.3 s asked of a follow-up because
  saying the wake word already shows intent, so a short command such as "stop" still gets through. Silero's own
  250 ms minimum is for cutting speech out of noise with no wake word. Tuning them on your voice is still to do;
  see [configuration.md](configuration.md).
- Jarvis has no downloader yet: place the four files above (wake, VAD and the two Kokoro files) under
  `data_dir` yourself. `jarvis doctor` names any that are missing.

## Python dependencies

| Package | Version | Licence (package metadata) |
|---|---|---|
| httpcore (used directly by the web guard) | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| livekit-wakeword | 2.1.3 | Apache-2.0 |
| numpy | 2.5.5 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| onnxruntime | 1.32.0 | MIT |
| pillow (checks and shrinks pictures for the model) | 13.0.0 | MIT-CMU |
| piper-tts | 1.8.0 | GPL-3.0-or-later |
| pydantic | 2.14.3 | MIT |
| textual (the `jarvis chat` window) | 8.2.8 | MIT |
| textual-image (pictures in the chat window) | 0.14.2 | LGPL-3.0-or-later |
| tomlkit (saves a setting into `config.toml` without touching the rest) | 0.15.2 | MIT |
| trafilatura (the main text of a web page) | 2.3.1 | Apache-2.0 |

What textual pulls in: rich 15.0.0, markdown-it-py 4.3.0, mdit-py-plugins 0.7.0, mdurl 0.1.2 and platformdirs
4.14.8, all MIT (package metadata; the three markdown packages give it as a licence classifier).

textual-image needs only pillow and rich, both above. It is LGPL-3.0-or-later, which a GPL-3.0-or-later
program may use. Version 0.14.2 (2026-10-20) is pinned because 0.14.3 and 0.14.4 were under two weeks old.

tomlkit has no dependencies of its own.

What trafilatura pulls in: lxml 6.1.3 and lxml-html-clean 0.4.6 (BSD-3-Clause), jusText 3.0.2 (BSD-2-Clause,
read from its LICENSE), courlan 1.5.0, htmldate 1.11.0 and tzlocal 5.4.5, dateparser 1.5.1, python-dateutil
4.0.0, pytz and tzdata, and `regex` 2026.11.19 (Apache-2.0 AND CNRI-Python). CNRI-Python is the one licence
outside the GPL-compatible list: the owner accepted it as a named exception in `tools/check_licences.py`
([ADR 0007](adr/0007-outside-services-for-answers.md)). Trafilatura is imported only when a page is first
read.

The SearXNG container (`deploy/searxng/`, AGPL-3.0-or-later) is a separate program Jarvis talks to over HTTP;
its image is pinned by digest.

Every other
installed package, development tools included, is checked by `tools/check_licences.py` in CI: its licence must be
compatible with GPL-3.0-or-later, and `uv.lock` must resolve and download only from PyPI.
