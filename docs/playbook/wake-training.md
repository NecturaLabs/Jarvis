# Training and benchmarking wake words

How to train a LiveKit wake word model for Jarvis, score it, and decide whether it replaces or joins the ones in
use. The results of the last run are in [models.md](../models.md#wake-word-and-end-of-speech).

## What you need

- **A training machine** with a CUDA GPU or an Apple silicon Mac. A 6 GB card is enough: one phrase trains in
  about 13 minutes.
- **The Jarvis PC** needs no GPU for detection: three models in Jarvis's detector, fed 80 ms frames, used 1.4 %
  of one core (2026-11-22).
- **The trainer**: a Python 3.13 virtual environment with LiveKit's trainer at the version `pyproject.toml` pins
  for detection, and PyTorch with CUDA:

  ```sh
  uv venv --python 3.13 .venv
  uv pip install --python .venv/bin/python "livekit-wakeword[train]==2.1.3"
  ```

  The trainer's `setup` step downloads about 20 GB of noise, room and speech sources into the Hugging Face cache
  in your home folder (`~/.cache/huggingface`), and keeps generated speech under `WORK/.cache/lkwb`. Log in to
  Hugging Face (`hf auth login`) so the downloads are not rate-limited.
- **One LiveKit Cloud project per phrase.** `setup` makes about 2,850 spoken clips per phrase with LiveKit's
  speech service, and a free project's monthly credit runs out after about one phrase. Keep each project's key
  in its own file, outside the repository, mode 0600, holding only three lines:

  ```sh
  LIVEKIT_URL=wss://<project>.livekit.cloud
  LIVEKIT_API_KEY=<key>
  LIVEKIT_API_SECRET=<secret>
  ```

  Only the generated speech uses the account; detection never does.

## Train a phrase

From a folder on the training machine holding the trainer's `.venv` (`--ungated` is explained below):

```sh
TRAINER=.venv WORK=work tools/wake_train.sh keys/project-1.env "jarvis" --ungated
```

It runs `setup --wakeword "<phrase>" --plan build --eval`, `train --type standard` and `eval --json`, and writes
`work/<name>.onnx`, `work/eval-<name>.json` and `work/train-<name>.log`. `<name>` is the trainer's own name for
the phrase: lower case, each run of other characters turned into `_` ("Yo, Jarvis!" is `yo_jarvis`). Relative
paths are taken from where you run it. Run it in the background with its PID recorded
(`nohup … > run.out 2>&1 & echo $! > train.pid`): setup takes 8 to 15 minutes, training about 13.

The key file is read as data, never run: the script refuses one that others can read or that holds anything
but the three lines, and passes the key only to `setup` and `train`. `WORK` is created private.

Two runs of the same recipe do not give identical weights: the trainer seeds its data workers partly from the
clock.

### If it stops

- **Exit codes:** 0 done, 1 a trainer step failed (see the log), 2 bad arguments or key file, 3 the credit ran
  out.
- **The credit runs out:** the trainer logs `TTS credit quota exhausted … (429 Too Many Requests)`
  (`MaxGatewayCredits`), and the script stops before training (exit 3) so no model is trained on half the clips.
  Run it again, same `WORK`, with another project's key file. Every clip is cached by its request, so only the
  missing ones are paid for. Only the current run's output is checked, so an earlier failure in the log does not
  stop the retry.
- **`GatedRepoError` in setup:** two test-only sources, ESC-50 and GTZAN, need approved Hugging Face access
  (#37). `--ungated` runs the trainer through `tools/wake_ungated.py`, which leaves them out of the test set
  without changing the installed package. It checks the trainer is 2.1.3 and that both sources are still listed
  under the names it expects, and stops otherwise. Training uses neither source, so the recipe is the same; only
  the test set's music and everyday-sound false alarms are missing.
- **After access is granted**, train or score the full test set in a fresh `WORK`. A `WORK` used with `--ungated`
  is marked (`WORK/.ungated`) and a full run refuses it, because the trainer would reuse the reduced test set.

### Rules

- One run per GPU at a time. Queue the next phrase behind the running one (wait on its PID) rather than
  starting both.
- Both gated sets carry research or non-commercial terms: keep them and their clips out of the repository and
  off the Jarvis PC; record only the scores.
- The log and eval JSON hold absolute paths, which name your account: copy only the figures into the repository.

## Benchmark before deploying

The trainer's own test set is not enough on its own: it speaks with the same speech service as the training
clips, which may favour the new model, and it holds only about an hour of other audio, so one false wake there
reads as one an hour, and none still cannot rule out about 3 an hour. Run all three checks, with the model in use
as the baseline.

1. **The trainer's test set**, per model, on its own phrase:

   ```sh
   .venv/bin/python tools/wake_ungated.py eval --wakeword "hey jarvis" --model hey_jarvis_retrained.onnx --json
   ```

   (`.venv/bin/livekit-wakeword eval …` once #37 is granted.) `hey_jarvis_retrained.onnx` is
   `work/hey_jarvis.onnx` renamed, so it is not confused with the pinned release.
2. **About 24 hours of real media** (TV and YouTube from AudioSet, music with vocals from FMA, parliament speeches
   from VoxPopuli; all test-only, built once with the trainer's `lkwb` tool): `.venv/bin/lkwb media -o media/`
   (5 minutes, 1.7 GB; step 3 reuses these files), then run the step 1 command with `--media media/` added, for
   each model. It checks the benchmark's own media figures.
3. **Voices the models never heard, and the models together as Jarvis listens**, with `tools/wake_bench.py`.
   `synth` needs Jarvis's Kokoro files and, for the Piper voice, `en_GB-alan-medium.onnx` with its `.onnx.json`
   in `<data_dir>/voices/` ([models.md](../models.md#voices)). On the Jarvis PC:

   ```sh
   uv run tools/wake_bench.py synth clips/          # every Kokoro voice and the Piper voice, about 4 minutes
   ```

   Copy `clips/`, `tools/wake_bench.py` and `<data_dir>/wake/hey_jarvis.onnx` (into `stock/`; check its digest
   with `jarvis doctor` first) to the training machine. There, with each model named once:

   ```sh
   M="stock=stock/hey_jarvis.onnx hey=hey_jarvis_retrained.onnx yo=work/yo_jarvis.onnx jarvis=work/jarvis.onnx"
   .venv/bin/python -I wake_bench.py score-clips clips/ media/ out/ $M --workers 8
   .venv/bin/python -I wake_bench.py score-media media/ out/ $M --workers 8
   .venv/bin/python -I wake_bench.py replay media/ replays/base stock=stock/hey_jarvis.onnx@0.5 --workers 8
   .venv/bin/python -I wake_bench.py replay media/ replays/yo \
     stock=stock/hey_jarvis.onnx@0.5 yo=work/yo_jarvis.onnx@0.5 --workers 8
   ```

   Run one `replay` per combination you might deploy, each with one `hey jarvis` model. ONNX Runtime takes every
   core per session unless told otherwise, and `OMP_NUM_THREADS` does not limit it: `wake_bench.py` loads each
   model with one thread, but the trainer's `eval --media` uses every core, so run it on its own, not beside
   these. Copy `out/` and `replays/` back, then on the Jarvis PC:

   ```sh
   uv run tools/wake_bench.py report out/ clips/ \
     stock=hey_jarvis:0.5 hey=hey_jarvis:0.5 yo=yo_jarvis:0.5 jarvis=jarvis:0.85 \
     --replay replays/base --replay replays/yo
   ```

   Each model is `name=phrase:threshold`; `phrase` is the wake phrase the model is scored on.

What each step does:

- `synth` says each phrase with every voice at three speeds: the wake phrases, similar-sounding phrases
  (homophones apart), and a set of Jarvis's own reply sentences in Jarvis's configured voice. The replies matter
  because the same models listen for barge-in while Jarvis speaks.
- `score-clips` scores each clip clean, quiet (−20 dB), under TV and music at 10, 5 and 0 dB SNR measured over
  the speech, with a room echo (early reflections, direct-to-reverberant ratio 0 to 10 dB) and far away. Each
  clip keeps one noise excerpt across levels. A hit counts only from the start of the speech.
- `score-media` keeps every model's score on every 80 ms of the media, with no resets.
- `replay` streams the media through the models together as Jarvis does: each at its own threshold, every model
  reset after a wake, and nothing heard for 4 s after it (`no_speech_timeout_s`, `--deaf`).
- `report` splits the media into two halves by clip and prints, each with counts and a 95 % interval:
  - thresholds picked on half A, with their false wakes an hour on half B, so a threshold is never checked on
    the audio it was picked on;
  - recall per condition, voice group and text;
  - paired differences between models on the same phrase (exact McNemar, and an interval that resamples whole
    voices);
  - the similar phrases and reply sentences that wake each model;
  - each replay's wakes an hour, overall and per source.

Results carry the sha256 of every model and of the clips; `report` refuses results that are incomplete, mixed
from different files, or partly written.

The voices are local Kokoro and Piper ones, independent of LiveKit's speech service but still synthetic speech
from two engines; the accented Kokoro voices speak from American English phonemes.

## Decide

Compare models at the same false-wake rate, not at the same threshold: two models' scores are not on the same
scale. Write the rule down before you read the results, and apply it as written:

- **Replace the pinned `hey_jarvis`** only when the retrained one, at the threshold where half A gives the same
  false-wake rate as the pinned model at 0.5, has on half B no more false wakes an hour, recall whose
  voice-resampled 95 % interval for the difference stays above −1 percentage point in every condition, and no
  more similar-phrase wakes. Otherwise keep the pinned release.
- **Keep an added phrase on** (`yo_jarvis`, `jarvis`) only when its replay adds at most 0.25 false wakes an hour
  to the baseline replay (`replays/base`), its own clean recall is at least 90 %, and no reply sentence in
  Jarvis's voice wakes it.
- **Last check, with you:** record a few wake attempts and an evening of normal desk audio (`docs/plan.md`, the
  wake test) with the intended microphone and speakers before changing the defaults. The synthetic voices and
  the media are not your room.

## Deploy

1. Keep the files you replace: copy `<data_dir>/wake/` and `config.toml` aside first.
2. Put the model at `<data_dir>/wake/<name>.onnx` and name it in `audio.wake_words` (`jarvis wake --words`).
3. Give a model whose report threshold differs from 0.5 its own threshold under `[audio.wake_thresholds]`, as
   `jarvis = 0.85`.
4. `jarvis wake` lists each model with its threshold. `jarvis doctor` checks each file loads and notes a
   threshold for a name that is not on. Restart Jarvis, then say each phrase once and read the service log:
   each wake names the models that passed, with score and threshold.
5. **Roll back** by putting the saved files and config back and restarting Jarvis.
6. Record the figures in [models.md](../models.md) with the commands, the counts and hours, never the machine or
   account names.
