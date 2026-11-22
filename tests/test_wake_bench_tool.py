"""tools/wake_bench.py: how wakes are counted, how clips are mixed, and the report it prints."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import wake_bench
from wake_bench import Clip, Model, Results


def _over(*frames: int, length: int = 200) -> np.ndarray:
    over = np.zeros(length, np.bool_)
    over[list(frames)] = True
    return over


def test_a_wake_leaves_the_detector_deaf_for_two_seconds() -> None:
    assert wake_bench.wake_events(_over()) == 0
    assert wake_bench.wake_events(_over(10, 11, 12, 34)) == 1  # 34 is 24 frames on: still deaf
    assert wake_bench.wake_events(_over(10, 35)) == 2  # 25 frames, 2 s, later
    assert wake_bench.wake_events(_over(10, 30, 50)) == 2  # 30 is ignored, so 50 wakes again


def test_a_clean_clip_is_padded_and_left_as_it_is() -> None:
    audio = np.full(800, 0.1, np.float32)

    out = wake_bench.mix(audio, "clean", 1, None)

    assert out.size == 800 + wake_bench.RATE + wake_bench.RATE // 2
    assert not out[: wake_bench.RATE].any()
    np.testing.assert_array_equal(out[wake_bench.RATE : wake_bench.RATE + 800], audio)


@pytest.mark.parametrize("snr_db", [10, 5, 0])
def test_background_audio_is_added_at_the_stated_snr(snr_db: int) -> None:
    rng = np.random.default_rng(0)
    speech = (rng.standard_normal(16000) * 0.05).astype(np.float32)
    noise = (rng.standard_normal(10 * wake_bench.RATE) * 0.2).astype(np.float32)

    out = wake_bench.mix(speech, f"snr{snr_db}", 3, noise)

    added = out[wake_bench.RATE : wake_bench.RATE + speech.size] - speech
    measured = 10 * np.log10(np.mean(speech**2) / np.mean(added**2))
    assert measured == pytest.approx(snr_db, abs=0.3)


def test_a_noise_condition_without_background_audio_is_refused() -> None:
    with pytest.raises(ValueError, match="needs background audio"):
        wake_bench.mix(np.zeros(100, np.float32), "snr5", 0, None)


def test_echo_is_the_same_for_the_same_seed_and_never_clips() -> None:
    audio = np.full(4000, 0.9, np.float32)

    first = wake_bench.mix(audio, "reverb", 7, None)

    np.testing.assert_array_equal(first, wake_bench.mix(audio, "reverb", 7, None))
    assert np.abs(first).max() <= 1.0
    assert not np.array_equal(first, wake_bench.mix(audio, "reverb", 8, None))


@pytest.mark.parametrize(
    ("voice", "group"),
    [
        ("kokoro:af_bella", "us"),
        ("kokoro:am_adam", "us"),
        ("kokoro:bm_george", "gb"),
        ("kokoro:jf_alpha", "accented"),
        ("en_GB-alan-medium", "piper"),
    ],
)
def test_each_voice_falls_in_its_group(voice: str, group: str) -> None:
    assert wake_bench.voice_group(voice) == group


def test_the_lowest_threshold_under_the_target_is_picked() -> None:
    grid = [0.5, 0.6, 0.7]

    assert wake_bench.lowest_threshold(grid, [3.0, 0.4, 0.0], 0.5) == pytest.approx(0.6)
    assert wake_bench.lowest_threshold(grid, [3.0, 2.0, 1.0], 0.5) is None


def test_each_voice_says_every_phrase_at_three_speeds_and_each_similar_phrase_once() -> None:
    plan = wake_bench.clip_plan()

    phrases = sum(len(texts) for texts in wake_bench.PHRASES.values())
    assert len(plan) == phrases * len(wake_bench.SPEEDS) + len(wake_bench.SIMILAR)
    assert {kind for kind, _text, _speed in plan} == {*wake_bench.PHRASES, wake_bench.SIMILAR_KIND}


def test_a_model_spec_names_its_phrase_and_threshold() -> None:
    assert wake_bench.parse_model("jarvis=jarvis:0.85") == Model("jarvis", "jarvis", 0.85)


def _results() -> Results:
    clips = [
        Clip("jarvis", "Jarvis.", 1.0, "kokoro:af_bella"),
        Clip("jarvis", "Jarvis!", 1.0, "en_GB-alan-medium"),
        Clip(wake_bench.SIMILAR_KIND, "Jervis.", 1.0, "kokoro:af_bella"),
        Clip("hey_jarvis", "Hey Jarvis.", 1.0, "kokoro:bm_george"),
    ]
    scores = {
        (name, condition): np.array(values)
        for condition in wake_bench.CONDITIONS
        for name, values in {"hey": [0.1, 0.2, 0.0, 0.8], "jarvis": [0.9, 0.6, 0.7, 0.3]}.items()
    }
    hour = 45000  # frames in one hour
    tv = np.zeros((2, hour), np.float32)
    tv[0, 100] = 0.6  # hey wakes once
    tv[1, 110] = 0.95  # jarvis too, while Jarvis is still busy with the first
    tv[1, 5000] = 0.8  # under jarvis's 0.85
    music = np.zeros((2, hour), np.float32)
    music[1, 300] = 0.9
    media = {"a": ("tv_youtube", tv), "b": ("music", music)}
    return Results(clips, ["hey", "jarvis"], scores, ["hey", "jarvis"], media)


def test_the_models_together_count_a_wake_once_and_each_at_its_own_threshold() -> None:
    results = _results()

    assert results.hours() == pytest.approx(2.0)
    assert results.false_wakes_per_hour({"hey": 0.5, "jarvis": 0.85}) == pytest.approx(1.0)
    assert results.false_wakes_per_hour({"hey": 0.5, "jarvis": 0.85}, "music") == pytest.approx(1.0)
    assert results.false_wakes_per_hour({"jarvis": 0.7}) == pytest.approx(1.5)


def test_recall_counts_the_clips_of_one_phrase_and_voice_group() -> None:
    results = _results()

    assert results.recall("jarvis", "clean", "jarvis", 0.85) == pytest.approx(0.5)
    assert results.recall("jarvis", "clean", "jarvis", 0.5, "piper") == pytest.approx(1.0)
    assert results.recall("jarvis", "clean", wake_bench.SIMILAR_KIND, 0.85) == pytest.approx(0.0)
    assert np.isnan(results.recall("jarvis", "clean", "yo_jarvis", 0.5))  # no clip says it


def test_the_report_has_every_table() -> None:
    models = [Model("hey", "hey_jarvis", 0.5), Model("jarvis", "jarvis", 0.85)]

    text = wake_bench.report(_results(), models)

    assert text.startswith("Media: 2.0 h (music 1.0 h, tv_youtube 1.0 h); clips: 4 from 3 voices")
    for heading in (
        "## False wakes an hour on the media",
        "## Recall on the model's own phrase",
        "## Matched rate: recall at the lowest threshold with at most 0.5 false wakes/h",
        "## Recall per voice group",
        "## Similar phrases that wake each model",
        "## Which phrases wake each model",
        "## All the models together",
    ):
        assert heading in text
    assert "- jarvis (0.85): none" in text
    assert "| all | 1.00 |" in text


def test_the_train_script_without_a_key_and_phrase_prints_its_usage() -> None:
    script = Path(__file__).resolve().parent.parent / "tools" / "wake_train.sh"

    done = subprocess.run([str(script)], capture_output=True, text=True, check=False)

    assert done.returncode == 2  # the usage exit
    assert "wake_train.sh KEY_FILE" in done.stderr
