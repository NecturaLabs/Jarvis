"""The sound levels of one captured request (`jarvis.levels`)."""

import math
import re

import numpy as np
import pytest

from jarvis.levels import FULL_SCALE, levels

RATE = 16000


def _tone(seconds: float, amplitude: float) -> np.ndarray:
    times = np.arange(int(seconds * RATE)) / RATE
    return np.round(amplitude * np.sin(2 * math.pi * 440 * times)).astype(np.int16)


def _noise(seconds: float, amplitude: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.round(rng.normal(0, amplitude, int(seconds * RATE))).astype(np.int16)


def _dbfs(value: float) -> float:
    return 20 * math.log10(value / FULL_SCALE)


def test_a_steady_tone_has_its_peak_and_level() -> None:
    heard = levels(_tone(1.0, 8000))

    assert heard.peak_dbfs == pytest.approx(_dbfs(8000), abs=0.1)
    assert heard.speech_dbfs == pytest.approx(_dbfs(8000 / math.sqrt(2)), abs=0.2)
    assert heard.clipped == 0


def test_speech_over_a_quiet_room_gives_the_floor_and_the_signal_to_noise_ratio() -> None:
    room = _noise(0.5, 30)
    samples = np.concatenate([room, _tone(1.0, 6000) + _noise(1.0, 30, seed=2), room])

    heard = levels(samples)

    assert heard.floor_dbfs == pytest.approx(_dbfs(30), abs=1.5)
    assert heard.speech_dbfs == pytest.approx(_dbfs(6000 / math.sqrt(2)), abs=0.5)
    assert heard.snr_db == pytest.approx(heard.speech_dbfs - heard.floor_dbfs)
    assert heard.snr_db > 40


def test_a_request_that_starts_on_its_first_sample_shows_a_loud_start() -> None:
    """Speech already loud in the first window means its first sound was cut off."""
    cut = levels(np.concatenate([_tone(1.0, 6000), _noise(0.5, 30)]))
    whole = levels(np.concatenate([_noise(0.3, 30), _tone(1.0, 6000), _noise(0.5, 30)]))

    assert cut.start_db > 30
    assert whole.start_db < 3


def test_samples_at_full_scale_count_as_clipped() -> None:
    times = np.arange(RATE) / RATE
    loud = 60000 * np.sin(2 * math.pi * 440 * times)
    samples = np.clip(np.round(loud), -32768, 32767).astype(np.int16)

    heard = levels(samples)

    assert heard.peak_dbfs == pytest.approx(0, abs=0.01)
    assert 0.3 < heard.clipped < 0.9


def test_digital_silence_has_finite_levels() -> None:
    heard = levels(np.zeros(RATE, dtype=np.int16))

    assert all(
        math.isfinite(value)
        for value in (heard.peak_dbfs, heard.speech_dbfs, heard.floor_dbfs, heard.start_db)
    )
    assert heard.snr_db == 0


def test_a_capture_shorter_than_one_window_is_measured_whole() -> None:
    heard = levels(_tone(0.005, 1000))

    assert heard.speech_dbfs == heard.floor_dbfs


def test_the_log_line_holds_only_numbers() -> None:
    line = levels(np.concatenate([_noise(0.3, 30), _tone(1.0, 6000)])).summary()

    number = r"[-+]?\d+\.\d+"
    assert re.fullmatch(
        rf"peak {number} dBFS, speech {number} dBFS, floor {number} dBFS \(SNR {number} dB\), "
        rf"start {number} dB over the floor, clipped {number}%",
        line,
    ), line


def test_an_empty_capture_has_levels_too() -> None:
    """A diagnostics line must never be what ends a request."""
    heard = levels(np.zeros(0, dtype=np.int16))

    assert heard.snr_db == 0
