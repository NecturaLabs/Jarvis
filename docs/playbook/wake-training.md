# Training and benchmarking wake words

How to train a LiveKit wake word model for Jarvis, score it, and decide whether it replaces the one in use.
The results of the last run are in [models.md](../models.md#wake-word-and-end-of-speech).

## What you need

- **A training machine** with a CUDA GPU or an Apple silicon Mac. A 6 GB card is enough: one phrase trains in
  about 13 minutes. Detection on the Jarvis PC needs no GPU: three models in Jarvis's detector, fed 80 ms frames, measured
  1.4 % of one core there (2026-11-22).
- **The trainer**: a Python 3.13 virtual environment with `livekit-wakeword[train]==2.1.3` (the version
  `pyproject.toml` pins for detection), PyTorch with CUDA, and about 20 GB of Hugging Face cache for the noise,
  room and speech sources `setup` downloads. Log in to Hugging Face (`hf auth login`) so the downloads are not
  rate-limited.
- **One LiveKit Cloud project per phrase.** `setup` makes about 2,850 spoken clips per phrase with LiveKit's
  speech service, and a free project's monthly credit runs out after about one phrase. Keep each project's
  `LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` in its own file (mode 0600), outside the
  repository, on the training machine and wherever you keep spares. Only the generated speech uses the
  account; detection never does.

## Train a phrase

From a folder on the training machine holding the trainer's `.venv`:

```sh
TRAINER=$PWD/.venv WORK=$PWD/work tools/wake_train.sh keys/project-1.env "jarvis" --ungated
```

It runs `setup --wakeword "<phrase>" --plan build --eval`, `train --type standard` and `eval --json`, and
writes `work/<phrase_with_underscores>.onnx`, `work/eval-<name>.json` and a log. Run it in the background with
its PID recorded (`nohup … & echo $! > train.pid`): setup takes 8 to 15 minutes, training about 13.

- **The credit runs out:** the trainer logs `TTS credit quota exhausted … (429 Too Many Requests)`
  (`MaxGatewayCredits`), and the script stops before training (exit 3) so no model is trained on half the
  clips. Run it again with another project's key file. Every clip is cached by its request
  (`work/.cache/lkwb/tts`), so only the missing ones are paid for.
- **`GatedRepoError` in setup:** two test-only sources, ESC-50 and GTZAN, need approved Hugging Face access
  (#37). `--ungated` runs the trainer through `tools/wake_ungated.py`, which leaves them out without changing
  the installed package. Training never uses them, so the model is the same; only the test set's music and
  everyday-sound false alarms are missing. Drop `--ungated` once access is granted.
- **One run per GPU at a time.** Queue the next phrase behind the running one (wait on its PID) rather than
  starting both.
- Both gated sets carry research or non-commercial terms: keep them and their clips out of the repository and
  off the Jarvis PC; record only the scores.

## Benchmark before deploying

The trainer's own test set is not enough on its own: it speaks with the same speech service as the training
clips, which favours the new model, and it holds only about an hour of other audio, so a single false wake
there already reads as one an hour. Run all three checks, with the model in use as the baseline.

1. **The trainer's test set**, per model, on its own phrase:
   `eval --wakeword "<phrase>" --model <file>.onnx --json` (through `tools/wake_ungated.py` until #37).
2. **About 24 hours of real media** (TV and YouTube from AudioSet, music with vocals from FMA, parliament
   speeches from VoxPopuli; all test-only, built once): `<trainer venv>/bin/lkwb media -o media/` (5 minutes,
   1.7 GB), then `eval … --media media/` per model for the trainer's false alarms an hour.
3. **Voices the models never heard**, plus the models together as Jarvis runs them, with `tools/wake_bench.py`:

   ```sh
   uv run tools/wake_bench.py synth clips/                       # on the Jarvis PC: 55 local voices, ~4 min
   # copy clips/ and tools/wake_bench.py to the training machine, then there:
   .venv/bin/python -I wake_bench.py score-clips clips/ media/ out/ old=hey_jarvis_old.onnx hey=hey_jarvis.onnx … --workers 8
   .venv/bin/python -I wake_bench.py score-media media/ out/ old=hey_jarvis_old.onnx hey=hey_jarvis.onnx … --workers 8
   # copy out/ back, then:
   uv run tools/wake_bench.py report out/ clips/ old=hey_jarvis:0.5 hey=hey_jarvis:0.5 jarvis=jarvis:0.85
   ```

   `synth` says every phrase in `PHRASES` and every similar phrase in `SIMILAR` with each Kokoro voice
   (English, and the others speaking English with an accent) and the Piper voice. `score-clips` scores each
   clip clean, under TV and music at 10, 5 and 0 dB SNR, and with a synthetic room echo. `score-media` keeps
   every model's score on every 80 ms of the media. `report` gives false wakes an hour per threshold, recall
   at a matched false-wake rate (the fair comparison between two models), recall per voice group, the similar
   phrases that wake each model, and the false wakes of all the models together, each at its own threshold.

Compare models at the same false-wake rate on the media, not at the same threshold: two models' scores are not
on the same scale.

- **ONNX Runtime takes every core per session** unless told otherwise; `OMP_NUM_THREADS` does not limit it.
  `wake_bench.py` loads each model with one thread. The trainer's `eval --media` still uses every core, so
  while it runs beside the benchmark, keep `--workers` to about half the cores.
- `wake_bench.py` runs under the trainer's Python 3.13 and Jarvis's 3.14; keep it free of 3.14-only syntax.
- The owner's own voice is the last check: record a few "wake" attempts and an evening of normal desk audio
  (`docs/plan.md`, the wake test) before changing the defaults.

## Deploy

1. Put the model at `<data_dir>/wake/<name>.onnx` and name it in `audio.wake_words` (`jarvis wake --words`).
2. Give a model that needs a different score its own threshold under `[audio.wake_thresholds]`.
3. `jarvis wake` lists each model with its threshold; `jarvis doctor` checks the files; restart Jarvis.
4. Keep the files you replaced, and record the figures in [models.md](../models.md) with the commands, never
   the machine or account names.
