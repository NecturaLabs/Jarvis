#!/usr/bin/env python3
# SPDX-FileCopyrightText: NecturaLabs and the Jarvis contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure how loud each catalogued Kokoro voice speaks, and the gain that evens them out.

Run with `uv run python tools/voice_loudness.py` (needs the Kokoro files in the data folder;
makes no sound). For each voice it prints the level of its speech as `voices.py` measures it
(the RMS of the 20 ms frames within 40 dB of the loudest, over six sentences, at speed 1.0)
before and after the voice's `gain_db`, and the gain that would bring it to the median.
"""

import numpy as np

from jarvis import config, voices
from jarvis.audio import voice_files

LINES = (
    "Done.",
    "Playing it now on Spotify.",
    "The weather tomorrow is sunny with a high of twenty two degrees and light wind from the west.",
    "Okay, that was actually insane. Let's go, run it back!",
    "I couldn't find that file. Do you want me to look in your downloads instead?",
    "Your meeting starts in ten minutes, and the build finished without errors.",
)
SAMPLE_RATE = 24000  # Kokoro's
FRAME = SAMPLE_RATE // 50  # 20 ms
GATE_DB = 40.0  # frames this far below the loudest count as silence


def speech_level(pcm: np.ndarray) -> float:
    """Return the RMS level in dBFS of the frames of `pcm` that hold speech."""
    audio = pcm.astype(np.float64) / 32768.0
    frames = audio[: len(audio) // FRAME * FRAME].reshape(-1, FRAME)
    power = (frames**2).mean(axis=1)
    spoken = power[power > power.max() * 10 ** (-GATE_DB / 10)]
    return float(10 * np.log10(spoken.mean()))


def main() -> None:
    """Print each voice's level before and after its gain, and the gain it would need."""
    from jarvis.kokoro import Kokoro  # loads the model runtime only when run

    cfg = config.load()
    model, pack = voice_files(cfg.data_dir, voices.KOKORO)
    rows: list[tuple[str, float, float]] = []
    for voice in voices.VOICES:
        speech = Kokoro(model, pack, voice.id, phoneme_map=cfg.audio.kokoro_phoneme_map)
        after = speech_level(np.concatenate([speech.synthesize(line) for line in LINES]))
        rows.append((voice.id, after - voice.gain_db, after))
    median = float(np.median([before for _, before, _ in rows]))
    print(f"{'voice':<12} {'before':>8} {'after':>8} {'gain_db':>8}")
    for name, before, after in rows:
        print(f"{name:<12} {before:8.2f} {after:8.2f} {median - before:+8.1f}")
    afters = [after for _, _, after in rows]
    spread = max(afters) - min(afters)
    print(f"median {median:.2f} dBFS; after the gains the voices span {spread:.2f} dB")


if __name__ == "__main__":
    main()
