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
- Requests go to Strata's OpenAI-style chat API with thinking off, a token cap (`model.max_tokens`) and the
  tools as JSON schemas.
- The model file lives outside the repository; point `jarvis setup --model-file` and `--tokenizer-dir` at it.
- Licence: Apache-2.0 for Qwen3.6-35B-A3B according to `docs/plan.md`; not rechecked against the model card
  here. The Unsloth quantisation and Strata itself: not checked.
- Reference machine for every number below: RX 7900 XT, 64 GB RAM, Strata 0.2.16 (engine build 0.1.62).

### Latency

| Request | What happens | Cost |
|---|---|---|
| Plain command: "pause", "next song", "volume 40", "what's playing" | Matched in code, run through the same policy, confirmation and audit as a model call, answered from the operation's own sentence. No model, no load wait. | about 0.3 s |
| Anything else the model answers (a chat reply is one model call: `no_action` carries the reply) | One prompt read of about 2,100 tokens, then generation | about 3.3 s of prompt reading at about 640 tokens/s, plus generation |
| Model not loaded | Cold load | about 107 s with a game running; Jarvis says "One moment, waking up". The request gets 180 s more (270 s in all instead of 90 s); each model call is still capped by `model.request_timeout_s` |

Memory while loaded, idle: about 38 GB of RAM and 7.3 GiB of VRAM, no GPU or CPU use. See
[configuration.md](configuration.md#keeping-the-model-loaded).

Jarvis marks the system prompt and the tools as Strata's shared prefix (`strata_prefix`, one message) on every
request that sends tools. The Qwen3.6 engine in Strata 0.1.62 does not use that mark yet, so every request
still reads the whole prompt. An engine that ignores the mark is harmless; the 3.3 s above is the cost today.

## Speech to text: Voxtype

Jarvis writes each request to a private temporary WAV file and runs `voxtype transcribe`, never the Voxtype
daemon and never your Voxtype config or recordings. Each run is
`voxtype -c <config dir>/voxtype.toml [--model M] [--engine E] [--initial-prompt HINT] transcribe FILE`.

- **Settings file:** Jarvis writes `voxtype.toml` in its own config folder at each start with `threads`
  (`audio.transcribe_threads`, default 8) and `context_window_optimization` (`audio.transcribe_short_window`,
  default on) under `[whisper]`. Whisper pads every clip to a 30 s window, so without the short window a
  two-second request costs as much as a long one; with it a short command took 0.29 s instead of 1.66 s
  (commit `141c2bf`, measured with `small.en`). Whisper.cpp also uses only 4 threads unless told otherwise.
- **Model:** `audio.transcribe_model`, default `small.en`; it must already be downloaded through Voxtype
  (`voxtype setup model`), and `jarvis doctor` checks that it is. In that same measurement `small.en` took
  0.29 s on a short command against 0.49 s for `base.en`, and was more accurate.
- **Hint:** when `audio.vocabulary_hint` is on, `--initial-prompt` carries Jarvis's name, its actions and up to
  `audio.hint_names` of your artist names, shortened, `Jarvis. play, pause, next. Glass Harbor, Night Circuit.`
  It always ends with a full stop: without it Whisper re-decoded short clips and took 1.8 s. A transcript that
  is only a run of the hint's own words is dropped. The answer to a confirmation question is transcribed
  without a hint, so noise cannot turn into words.
- Whisper's stock phrases on silence ("thank you for watching") are ignored.
- Licence: Voxtype and the Whisper models: not checked.

## Voices

`audio.voice` picks the voice. Both voices run on the CPU; Kokoro uses at most 8 threads and never spin-waits.

### Kokoro (default, `kokoro:<name>`)

- Kokoro-82M on ONNX Runtime, 24 kHz output. Jarvis's own `src/jarvis/kokoro.py` runs it; the default voice is
  `bm_george` (a name starting with `b` uses British English phonemes, otherwise American).
- Files, under `<data_dir>/voices/kokoro/`: `kokoro-v1.0.onnx` and `voices-v1.0.bin` from the kokoro-onnx
  `model-files-v1.1` release. `jarvis doctor` checks their sha256 against the pinned release.
- Phonemes come from espeak-ng, whose library and data ship inside the `piper-tts` wheel, so the Piper package
  is installed and imported even when you use Kokoro.
- Licence: Apache-2.0 for the Kokoro weights, as pinned in `models.py` and in the `kokoro.py` header; not
  rechecked against the model card.

### Piper (any other `audio.voice` value)

- `piper-tts` 1.8.0 (OHF-Voice/piper1-gpl). Put a voice at `<data_dir>/voices/<name>.onnx` with
  `<name>.onnx.json` beside it and set `audio.voice = "<name>"`. `audio.speech_speed` sets Piper's length
  scale (the voice's own scale divided by the speed).
- Licence: `piper-tts` is GPL-3.0-or-later (package metadata). Each Piper voice has its own licence: not
  checked. Because Jarvis imports `piper-tts` for both voices, Jarvis is published under GPL-3.0-or-later
  (`docs/adr/0006-gpl-licence.md`).

## Wake word and end of speech

- **Wake word:** the LiveKit `hey_jarvis` model, run by `livekit-wakeword` 2.1.3 on CPU, at
  `<data_dir>/wake/hey_jarvis.onnx`. Threshold: `audio.wake_threshold`. Licence: Apache-2.0 for the code
  (package metadata) and the weights (the wheel's `NOTICE`); the file is pinned to release 2.1.3 in `models.py`.
- **End of speech:** Silero VAD 6.2.3 at `<data_dir>/vad/silero_vad.onnx`, MIT as pinned in `models.py`; not
  rechecked upstream. Threshold and timings: `audio.vad_threshold`, `no_speech_timeout_s`, `end_silence_s`,
  `max_utterance_s`.
- Jarvis has no downloader yet: place the four files above (wake, VAD and the two Kokoro files) under
  `data_dir` yourself. `jarvis doctor` names any that are missing.

## Python dependencies

| Package | Version | Licence (package metadata) |
|---|---|---|
| httpx | 0.28.1 | BSD-3-Clause |
| livekit-wakeword | 2.1.3 | Apache-2.0 |
| numpy | 2.5.5 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| onnxruntime | 1.32.0 | MIT |
| piper-tts | 1.8.0 | GPL-3.0-or-later |
| pydantic | 2.12.5 | MIT |
| textual (the `jarvis chat` window) | 8.2.8 | MIT |

What textual pulls in: rich 15.0.0, markdown-it-py 4.3.0, mdit-py-plugins 0.7.0, mdurl 0.1.2 and platformdirs
4.14.8, all MIT (package metadata; the three markdown packages give it as a licence classifier). The other
packages' own dependencies and the development tools were not checked.
