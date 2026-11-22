"""tools/wake_bench.py: counting wakes, mixing clips, the statistics and the report."""

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import wake_bench
from wake_bench import BenchError, Clip, Model, Results

if TYPE_CHECKING:
    from pathlib import Path


def _over(*frames: int, length: int = 200) -> np.ndarray:
    over = np.zeros(length, np.bool_)
    over[list(frames)] = True
    return over


def test_a_stored_score_wake_leaves_the_detector_deaf() -> None:
    assert wake_bench.wake_events(_over(), 50) == []
    assert wake_bench.wake_events(_over(10, 11, 59), 50) == [10]  # 59 is 49 frames on: still deaf
    assert wake_bench.wake_events(_over(10, 60), 50) == [10, 60]
    assert wake_bench.wake_events(_over(10, 50, 70), 50) == [10, 70]  # 50 is ignored, 70 wakes


class _Scripted:
    """Stands in for LiveKit's model: a score per frame fed."""

    def __init__(self, scores: list[float]) -> None:
        super().__init__()
        self.scores = scores
        self.at = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def predict(self, _frame: np.ndarray) -> dict[str, float]:
        score = self.scores[self.at] if self.at < len(self.scores) else 0.0
        self.at += 1
        return {"m": score}


def test_a_replay_resets_every_model_and_hears_nothing_while_jarvis_captures() -> None:
    hey = _Scripted([0.0, 0.9, 0.0, 0.0, 0.9])
    jarvis = _Scripted([0.0, 0.0, 0.0, 0.0, 0.86])
    audio = np.zeros(8 * wake_bench.FRAME, np.float32)

    wakes = wake_bench.replay_clip(
        {"hey": hey, "jarvis": jarvis}, {"hey": 0.5, "jarvis": 0.85}, audio, 3
    )

    # Frame 1 wakes; frames 2-4 are not fed; the fifth score fed is frame 7's.
    assert [frame for frame, _passed in wakes] == [1, 7]
    assert wakes[1][1] == {"hey": pytest.approx(0.9), "jarvis": pytest.approx(0.86)}
    assert hey.resets == 3  # at the start, and after each wake
    assert hey.at == 5  # frames 0, 1, 5, 6 and 7 were fed


def test_each_model_is_held_to_its_own_threshold_in_a_replay() -> None:
    audio = np.zeros(wake_bench.FRAME, np.float32)
    models = {"hey": _Scripted([0.6]), "jarvis": _Scripted([0.8])}

    wakes = wake_bench.replay_clip(models, {"hey": 0.5, "jarvis": 0.85}, audio, 3)

    assert wakes == [(0, {"hey": pytest.approx(0.6)})]


def test_a_clean_clip_is_padded_and_left_as_it_is() -> None:
    audio = np.full(800, 0.1, np.float32)

    out = wake_bench.mix(audio, "clean", 1, None)

    lead = int(wake_bench.LEAD_S * wake_bench.RATE)
    assert out.size == 800 + lead + int(wake_bench.TAIL_S * wake_bench.RATE)
    assert not out[:lead].any()
    np.testing.assert_allclose(out[lead : lead + 800], audio)


def test_quiet_speech_is_twenty_db_lower() -> None:
    out = wake_bench.mix(np.full(800, 0.5, np.float32), "quiet", 1, None)

    lead = int(wake_bench.LEAD_S * wake_bench.RATE)
    np.testing.assert_allclose(out[lead : lead + 800], 0.05, rtol=1e-5)


def _speech_with_a_pause() -> np.ndarray:
    speech = (np.random.default_rng(0).standard_normal(16000) * 0.05).astype(np.float32)
    speech[4000:10000] = 0.0  # a long pause, which must not count as speech
    return speech


def _noise() -> np.ndarray:
    rng = np.random.default_rng(1)
    return (rng.standard_normal(10 * wake_bench.RATE) * 0.2).astype(np.float32)


@pytest.mark.parametrize("snr_db", [10, 5, 0])
def test_background_audio_is_added_at_the_stated_snr_over_the_speech(snr_db: int) -> None:
    speech = _speech_with_a_pause()

    out = wake_bench.mix(speech, f"snr{snr_db}", 3, _noise())

    lead = int(wake_bench.LEAD_S * wake_bench.RATE)
    added = (out[lead : lead + speech.size] - speech).astype(np.float64)
    active = np.concatenate([speech[:4000], speech[10000:]]).astype(np.float64)
    measured = 10 * math.log10(float(np.mean(active**2)) / float(np.mean(added**2)))
    assert measured == pytest.approx(snr_db, abs=0.5)


def test_one_clip_hears_the_same_background_excerpt_at_every_level() -> None:
    speech = _speech_with_a_pause()

    loud = wake_bench.mix(speech, "snr0", 7, _noise())
    soft = wake_bench.mix(speech, "snr10", 7, _noise())

    lead = int(wake_bench.LEAD_S * wake_bench.RATE)
    ratio = loud[100:lead] / soft[100:lead]  # the noise alone, before the speech
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-3)


def test_a_noise_condition_without_enough_background_audio_is_refused() -> None:
    with pytest.raises(BenchError, match="needs more background audio"):
        wake_bench.mix(np.zeros(100, np.float32), "snr5", 0, None)


def test_the_room_has_a_direct_sound_louder_than_its_tail() -> None:
    start = int(0.02 * wake_bench.RATE)
    for seed in range(20):
        response = wake_bench.room(np.random.default_rng(seed)).astype(np.float64)
        drr = 10 * math.log10(response[0] ** 2 / np.sum(response[start:] ** 2))
        assert -1e-3 <= drr <= 10.0 + 1e-3


def test_echo_is_the_same_for_the_same_seed_and_never_clips() -> None:
    audio = np.full(4000, 0.9, np.float32)

    first = wake_bench.mix(audio, "reverb", 7, None)

    np.testing.assert_array_equal(first, wake_bench.mix(audio, "reverb", 7, None))
    assert np.abs(first).max() <= 0.99 + 1e-6
    assert not np.array_equal(first, wake_bench.mix(audio, "reverb", 8, None))


def test_loud_synthesized_speech_is_scaled_down_rather_than_clipped() -> None:
    pcm = wake_bench.to_int16(np.array([0.0, 1.4, -0.7], np.float32))

    assert pcm[1] == round(wake_bench.HEADROOM * 32767)
    assert pcm[2] == round(-0.7 * wake_bench.HEADROOM / 1.4 * 32767)


@pytest.mark.parametrize(
    ("voice", "group"),
    [
        ("kokoro:af_bella", "us"),
        ("kokoro:bm_george", "gb"),
        ("kokoro:jf_alpha", "accented"),
        ("en_GB-alan-medium", "piper"),
    ],
)
def test_each_voice_falls_in_its_group(voice: str, group: str) -> None:
    assert wake_bench.voice_group(voice) == group


def test_the_chosen_threshold_is_one_from_which_every_higher_one_meets_the_target() -> None:
    grid = [0.5, 0.6, 0.7, 0.8]

    assert wake_bench.lowest_threshold(grid, [3.0, 0.4, 0.0, 0.0], 0.5) == pytest.approx(0.6)
    assert wake_bench.lowest_threshold(grid, [0.0, 0.9, 0.0, 0.0], 0.5) == pytest.approx(0.7)
    assert wake_bench.lowest_threshold(grid, [3.0, 2.0, 1.0, 0.9], 0.5) is None


def test_the_poisson_interval_matches_the_exact_values() -> None:
    assert wake_bench.poisson_interval(0, 1.0) == (0.0, pytest.approx(3.689, abs=0.002))
    low, high = wake_bench.poisson_interval(5, 1.0)
    assert (low, high) == (pytest.approx(1.623, abs=0.002), pytest.approx(11.668, abs=0.003))
    with pytest.raises(BenchError):
        wake_bench.poisson_interval(1, 0.0)


def test_the_wilson_interval_and_mcnemar_match_known_values() -> None:
    low, high = wake_bench.wilson_interval(9, 10)
    assert (low, high) == (pytest.approx(0.596, abs=0.001), pytest.approx(0.982, abs=0.001))
    assert math.isnan(wake_bench.wilson_interval(0, 0)[0])
    assert wake_bench.mcnemar_p(0, 0) == pytest.approx(1.0)
    assert wake_bench.mcnemar_p(33, 16) == pytest.approx(0.0214, abs=0.0005)


def test_the_voice_resampled_difference_brackets_the_observed_one() -> None:
    a = np.array([True, True, False, False, True, True])
    b = np.array([True, True, True, False, True, True])

    diff, low, high = wake_bench.clustered_difference(a, b, ["v1", "v1", "v2", "v2", "v3", "v3"])

    assert diff == pytest.approx(1 / 6)
    assert low <= diff <= high


def test_media_halves_are_stable_and_both_used() -> None:
    assert {wake_bench.media_half(f"clip-{i}") for i in range(50)} == {"A", "B"}
    assert wake_bench.media_half("clip-3") == wake_bench.media_half("clip-3")


def test_each_voice_says_every_phrase_and_only_jarvis_s_own_voice_says_the_replies() -> None:
    plan = wake_bench.clip_plan(own_voice=False)

    phrases = sum(len(texts) for texts in wake_bench.PHRASES.values())
    assert len(plan) == phrases * len(wake_bench.SPEEDS) + len(wake_bench.SIMILAR)
    assert len(wake_bench.clip_plan(own_voice=True)) == len(plan) + len(wake_bench.REPLIES)


def test_a_wake_counts_only_while_the_clip_s_speech_plays() -> None:
    first, last = wake_bench.window(Clip("jarvis", "Jarvis.", 1.0, "v", 8000))

    assert first == int(wake_bench.LEAD_S * wake_bench.RATE) // wake_bench.FRAME
    assert last == math.ceil((16000 + 8000 + 16000) / wake_bench.FRAME)


def test_model_specs_name_their_phrase_or_file_and_threshold() -> None:
    assert wake_bench.parse_model("jarvis=jarvis:0.85") == Model("jarvis", "jarvis", 0.85)
    assert wake_bench.parse_replay_model("j=/m/j.onnx@0.85") == ("j", "/m/j.onnx", 0.85)


CLIPS = [
    Clip("jarvis", "Jarvis.", 1.0, "kokoro:af_bella", 1280),
    Clip("jarvis", "Jarvis!", 1.0, "en_GB-alan-medium", 1280),
    Clip(wake_bench.SIMILAR_KIND, "Service.", 1.0, "kokoro:af_bella", 1280),
    Clip(wake_bench.SIMILAR_KIND, "Jervis.", 1.0, "kokoro:af_bella", 1280),
    Clip(wake_bench.REPLY_KIND, "Opening Spotify.", 1.0, "kokoro:bm_george", 1280),
    Clip("hey_jarvis", "Hey Jarvis.", 1.0, "kokoro:bm_george", 1280),
]
BEST = {"hey": [0.1, 0.2, 0.0, 0.0, 0.0, 0.8], "jarvis": [0.9, 0.6, 0.7, 0.95, 0.9, 0.3]}
HASHES = {"hey": "h1", "jarvis": "j1"}
HOUR = 45000  # frames in one hour


def _clip_arrays() -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for name, values in BEST.items():
        for condition in wake_bench.CONDITIONS:
            rows = [np.array([0.0, v, 0.0], np.float32) for v in values]
            arrays[f"{name}|{condition}|scores"] = np.concatenate(rows).astype(np.float16)
            arrays[f"{name}|{condition}|offsets"] = np.cumsum([0, *(r.size for r in rows)])
    return arrays


def _write_results(tmp_path: Path) -> tuple[Path, Path]:
    """Write a small clip and media result set in the formats the scoring steps write."""
    clip_dir, out = tmp_path / "clips", tmp_path / "out"
    np.savez(clip_dir.mkdir() or clip_dir / "clips.npz", *[np.zeros(10, np.int16)] * len(CLIPS))
    (clip_dir / "clips.json").write_text(json.dumps({"clips": [c.__dict__ for c in CLIPS]}))
    out.mkdir()
    np.savez(out / "clip_frames.npz", **_clip_arrays())
    clip_meta = {
        "models": list(BEST),
        "model_sha256": HASHES,
        "clips": len(CLIPS),
        "clips_sha256": wake_bench.sha256(clip_dir / "clips.npz"),
    }
    (out / "clip_meta.json").write_text(json.dumps(clip_meta))
    tv = np.zeros((2, HOUR), np.float32)
    tv[0, 100] = 0.6  # hey wakes
    tv[1, 110] = 0.95  # jarvis too, within the 4 s after
    music = np.zeros((2, HOUR), np.float32)
    music[1, 300] = 0.9
    np.savez(out / "media_frames.npz", a=tv, b=music)
    clips = {"a": ["tv_youtube", "A"], "b": ["music", "B"]}
    media_meta = {"models": list(BEST), "model_sha256": HASHES, "clips": clips}
    (out / "media_meta.json").write_text(json.dumps(media_meta))
    return out, clip_dir


def test_stored_results_load_and_count_the_models_together(tmp_path: Path) -> None:
    out, clip_dir = _write_results(tmp_path)

    results = wake_bench.load_results(out, clip_dir)

    assert results.hours() == pytest.approx(2.0)
    assert results.hours(half="A") == pytest.approx(1.0)
    assert results.proxy_events({"hey": 0.5, "jarvis": 0.85}) == 2  # a's two are 4 s apart: one
    assert results.proxy_events({"jarvis": 0.85}, half="B") == 1
    assert results.hits("jarvis", "clean", 0.85, results.picked("jarvis")).tolist() == [True, False]


def test_results_from_another_model_file_or_clip_set_are_refused(tmp_path: Path) -> None:
    out, clip_dir = _write_results(tmp_path)
    meta = json.loads((out / "media_meta.json").read_text())
    meta["model_sha256"]["jarvis"] = "other"
    (out / "media_meta.json").write_text(json.dumps(meta))

    with pytest.raises(BenchError, match="different model files"):
        wake_bench.load_results(out, clip_dir)
    (clip_dir / "clips.npz").write_bytes(b"changed")
    with pytest.raises(BenchError, match="were not made from the clips"):
        wake_bench.load_results(out, clip_dir, media=False)


def test_scores_that_miss_a_clip_are_refused(tmp_path: Path) -> None:
    out, clip_dir = _write_results(tmp_path)
    arrays = _clip_arrays()
    arrays["hey|clean|offsets"] = arrays["hey|clean|offsets"][:-1]
    np.savez(out / "clip_frames.npz", **arrays)

    with pytest.raises(BenchError, match="scores for 5 clips, 6 expected"):
        wake_bench.load_results(out, clip_dir)


def test_a_result_file_is_written_in_one_step(tmp_path: Path) -> None:
    target = tmp_path / "out" / "r.json"

    wake_bench.write_atomic(target, b"{}")

    assert target.read_bytes() == b"{}"
    assert list(target.parent.iterdir()) == [target]  # no leftover part file


def test_the_report_has_every_table_and_marks_homophones(tmp_path: Path) -> None:
    out, clip_dir = _write_results(tmp_path)
    results: Results = wake_bench.load_results(out, clip_dir)
    replay = {
        "models": {"jarvis": 0.85},
        "deaf_s": 4.0,
        "hours": {"tv_youtube|A": 1.0, "music|B": 1.0},
        "wakes": [{"id": "b", "kind": "music", "half": "B", "t": 24.0, "passed": {"jarvis": 0.9}}],
    }

    text = wake_bench.report(results, [Model("jarvis", "jarvis", 0.85)], [replay])

    for heading in (
        "## False wakes an hour on all the media",
        "## Threshold chosen on half A for at most 0.5 false wakes/h, checked on half B",
        "## Recall on the model's own phrase at its threshold, per condition",
        "## Recall per voice group",
        "## Recall per text",
        "## Similar phrases that wake each model",
        "## Jarvis's own replies that reach a model's threshold (1, clean)",
        "## Which phrases wake each model",
        "## Replay as Jarvis listens: jarvis 0.85 (reset on a wake, then 4.0 s deaf)",
    ):
        assert heading in text
    assert "Jervis (homophone) 100/100 %" in text
    assert "| B | music | 1.00 | 1.00 [1] (0.03-5.57) |" in text
    assert "- b at 24.0 s (music): jarvis 0.90" in text


def test_the_report_refuses_a_model_it_has_no_scores_for(tmp_path: Path) -> None:
    out, clip_dir = _write_results(tmp_path)

    assert wake_bench.main(["report", str(out), str(clip_dir), "yo=yo_jarvis:0.5"]) == 2
