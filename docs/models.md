# Models and voices

What Jarvis loads, where each file goes, what it costs in time and what licence it carries. `jarvis doctor`
points here when a file is missing. Licences are taken from package metadata, from the licence field Jarvis
pins beside each model's sha256 (`src/jarvis/models.py`), or from a file shipped in the wheel; where nothing
was checked it says so.

## The model

- **Qwen3.6-35B-A3B**, Unsloth `UD-Q4_K_XL` quantisation (about 22.9 GB on disk), served by Strata as a
  profile named by `strata.model_id`. It is a mixture of experts (35B parameters, about 3B active per
  token). Strata runs it in hybrid mode: some experts in VRAM (`strata.expert_vram_gib`), the rest in host
  memory.
- Jarvis writes the Strata profile and the `jarvis-strata.service` unit (`jarvis setup`). Strata listens on
  `127.0.0.1` only, loads the model on the first real request and frees it after `strata.idle_unload_s`.
  Health checks (`jarvis status`, `jarvis doctor`) never generate, so they never load it.
- Requests go to Strata's OpenAI-style chat API with thinking off, a token cap per answer (`agent.reply_tokens`, `chat.reply_tokens`) and the
  tools as JSON schemas.
- The model file lives outside the repository; point `jarvis setup --model-file` and `--tokenizer-dir` at it.
- Licence: Apache-2.0 for Qwen3.6-35B-A3B according to `docs/plan.md`; not rechecked against the model card
  here. The Unsloth quantisation and Strata itself: not checked.
- Reference machine for every number below: RX 7900 XT, 64 GB RAM, Strata 0.2.16 (engine build 0.1.62).

### Pictures

- The picture reader is the model's own vision projector, `mmproj-F16.gguf` (899,283,584 bytes) from
  `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` at revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`, the file and
  revision Strata's own setup pins. Licence: Apache-2.0 (the repository's model card, checked 2026-11-21).
  `jarvis doctor` checks its sha256 (`71f3cbc1…0f160`).
- Put it beside the model and set `strata.vision_file` to it, then `jarvis setup` and `jarvis restart`. Strata
  runs it with its CPU reader (`strata-vision`): on a card under 24 GB Strata keeps it off the VRAM. It starts
  with the first picture (a few seconds) and stops after `strata.vision_idle_s`; a picture becomes at most 300
  tokens.

### Latency

| Request | What happens | Cost |
|---|---|---|
| Plain command: "pause", "next song", "volume 40", "what's playing" | Matched in code, run through the same policy, confirmation and audit as a model call, answered from the operation's own sentence. No model, no load wait. | about 0.3 s |
| Anything else the model answers (a chat reply is one model call: `no_action` carries the reply) | One prompt read of about 2,900 tokens (2,100 before the web tools), then generation; a call after a tool result reads only the new part | about 4.5 s of prompt reading at about 640 tokens/s, plus generation; the first sentence is spoken once it is written ([ADR 0008](adr/0008-speak-while-writing-and-barge-in.md)) |
| Model not loaded | Cold load | about 107 s with a game running; Jarvis says "One moment, waking up". The request gets 180 s more (270 s in all instead of 90 s); each model call is still capped by `model.request_timeout_s` |

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

#### Prompt chunk size

`strata.prefill` is 512 tokens. The reason for 512 is not recorded. Strata's own tuner lists 512 for this model
family among the sizes it avoids (smaller chunks read long prompts 11% to 75% slower in its tests), and tries
`auto`, `auto:16384`, `auto:32768`, 2048 and 4096. `strata.prefill = "auto"` is accepted and passed through as
`--prefill auto`. Not measured yet: the owner's run (prompt tokens per second, time to the first word on the
2,900-token request, VRAM, with a game running) is listed in the [playbook](playbook/README.md#owner-runs-for-the-service-units-33).
The default changes only if the numbers favour it. `strata.vram_reserve_mib` (unset: Strata's default, 700 MiB)
is the matching VRAM knob.

Jarvis marks the system prompt and the tools as Strata's shared prefix (`strata_prefix`, one message) on every
request that sends tools. The Qwen3.6 engine in Strata 0.1.62 does not use that mark yet, so every request
still reads the whole prompt. An engine that ignores the mark is harmless; the 4.5 s above is the cost today.
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
spoken over.

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
  build --eval`, `train --type standard`, `eval --json`), on the trainer's held-out test set without its two
  gated sets (ESC-50 and GTZAN, so music and other everyday sounds are not counted): 500 spoken phrases,
  796 other clips (about 1 hour) and 500 similar-sounding phrases. Threshold is the lowest score that gives
  at most 0.5 false wakes an hour, and Recall is measured there; Similar is the share of similar-sounding
  phrases that woke it at that threshold.

  | Model | Threshold | Recall | Similar | Recall at threshold 0.5 |
  |---|---|---|---|---|
  | `hey_jarvis` (pinned release) | 0.47 | 92.8 % | 22 % | 92.4 %, no false wakes |
  | `hey_jarvis` (retrained) | 0.46 | 93.6 % | 19 % | 92.6 %, no false wakes |
  | `yo_jarvis` | 0.43 | 95.8 % | 17 % | 94.8 %, no false wakes |
  | `jarvis` (bare word) | 0.85 | 86.2 % | 1 % | 96.0 %, 4.9 false wakes an hour |

  The pinned `hey_jarvis` hears 2.4 % of bare "jarvis" and 4.2 % of "yo jarvis" phrases, so each phrase
  needs its own model. Give `jarvis` its own threshold (`audio.wake_thresholds`); the shared 0.5 suits the
  two-word phrases.
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
| trafilatura (the main text of a web page) | 2.3.1 | Apache-2.0 |

What textual pulls in: rich 15.0.0, markdown-it-py 4.3.0, mdit-py-plugins 0.7.0, mdurl 0.1.2 and platformdirs
4.14.8, all MIT (package metadata; the three markdown packages give it as a licence classifier).

textual-image needs only pillow and rich, both above. It is LGPL-3.0-or-later, which a GPL-3.0-or-later
program may use. Version 0.14.2 (2026-10-20) is pinned because 0.14.3 and 0.14.4 were under two weeks old.

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
