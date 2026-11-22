"""Kokoro's session settings and its own logic, against a stand-in model."""

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from jarvis import kokoro
from jarvis.kokoro import MAX_PHONEMES, MAX_THREADS, SAMPLE_RATE, Kokoro, session_options

if TYPE_CHECKING:
    from pathlib import Path

VOCAB = {"a": 1, "b": 2, " ": 3}
STYLE_WIDTH = 4


class _Session:
    """What `ort.InferenceSession` gives Kokoro: input names, the vocabulary and `run`."""

    def __init__(self, inputs: dict[str, str]) -> None:
        super().__init__()
        self._inputs = inputs
        self.feeds: list[dict[str, Any]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name, type=kind) for name, kind in self._inputs.items()]

    @staticmethod
    def get_modelmeta() -> SimpleNamespace:
        return SimpleNamespace(custom_metadata_map={"kokoro_config": json.dumps({"vocab": VOCAB})})

    def run(self, _outputs: None, feed: dict[str, Any]) -> list[Any]:
        self.feeds.append(feed)
        return [np.full((1, 100), 0.5, dtype=np.float32)]


def _say(monkeypatch: pytest.MonkeyPatch, sentences: list[str]) -> None:
    """Have espeak phonemize any text into `sentences`."""

    def phonemes(_text: str, _voice: str) -> list[str]:
        return sentences

    monkeypatch.setattr(kokoro, "_espeak_phonemes", phonemes)


def _kokoro(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inputs: dict[str, str] | None = None,
) -> tuple[Kokoro, _Session]:
    session = _Session(
        inputs or {"input_ids": "tensor(int64)", "style": "tensor(float)", "speed": "tensor(float)"}
    )

    def load(*_args: object, **_kwargs: object) -> _Session:
        return session

    monkeypatch.setattr(kokoro.ort, "InferenceSession", load)
    voices = tmp_path / "voices.bin"
    rows = np.arange(MAX_PHONEMES * STYLE_WIDTH, dtype=np.float32)
    with voices.open("wb") as handle:
        np.savez(handle, bm_george=rows.reshape(MAX_PHONEMES, 1, STYLE_WIDTH))
    return Kokoro(tmp_path / "model.onnx", voices, "bm_george"), session


def test_the_voice_model_uses_a_bounded_number_of_threads() -> None:
    options = session_options()

    assert 1 <= options.intra_op_num_threads <= MAX_THREADS
    assert options.inter_op_num_threads == 1


def test_a_model_that_takes_tokens_is_fed_under_that_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the older model build names its input `tokens`; `input_ids` was hard-coded."""
    voice, session = _kokoro(
        tmp_path,
        monkeypatch,
        {"tokens": "tensor(int64)", "style": "tensor(float)", "speed": "tensor(int32)"},
    )
    _say(monkeypatch, ["ab ba"])

    voice.synthesize("ab ba")

    feed = session.feeds[0]
    assert "tokens" in feed
    assert "input_ids" not in feed
    assert feed["speed"].dtype == np.int32  # the graph's own type


def test_a_file_that_is_not_a_kokoro_model_is_refused_when_it_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match=r"model\.onnx.*input_ids"):
        _kokoro(tmp_path, monkeypatch, {"x": "tensor(float)", "style": "tensor(float)"})


def test_synthesis_pads_the_tokens_picks_the_style_by_length_and_joins_with_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    voice, session = _kokoro(tmp_path, monkeypatch)
    _say(monkeypatch, ["ab", "ba b"])

    pcm = voice.synthesize("x")

    first, second = (feed["input_ids"].tolist() for feed in session.feeds)
    assert first == [[0, 1, 2, 0]]
    assert second == [[0, 2, 1, 3, 2, 0]]
    # The style row is the one for the phoneme count (counted without the padding).
    assert session.feeds[0]["style"].tolist() == [[4.0, 5.0, 6.0, 7.0]]
    assert session.feeds[1]["style"].tolist() == [[12.0, 13.0, 14.0, 15.0]]
    assert pcm.dtype == np.int16
    assert pcm.size == 2 * (100 + int(SAMPLE_RATE * 0.12))
    assert pcm[0] == int(0.5 * 32767)


def test_nothing_to_say_is_empty_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    voice, session = _kokoro(tmp_path, monkeypatch)
    _say(monkeypatch, [])

    assert voice.synthesize("").size == 0
    assert not session.feeds


def test_a_sentence_over_the_window_is_split_between_words_and_all_of_it_is_spoken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a sentence was cut at exactly 510 phonemes, in the middle of a word."""
    voice, _session = _kokoro(tmp_path, monkeypatch)
    words = ["ab", "abb", "aab", "bbbb", "aaaaa"] * 120
    _say(monkeypatch, [" ".join(words)])

    chunks = voice.tokens("long")

    assert len(chunks) > 1
    assert all(len(ids) <= MAX_PHONEMES for ids in chunks)
    ids_to_char = {number: char for char, number in VOCAB.items()}
    spoken = [word for ids in chunks for word in "".join(ids_to_char[i] for i in ids).split()]
    assert spoken == words  # no word cut in two, none lost


def test_a_word_longer_than_the_window_is_still_cut_rather_than_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    voice, _session = _kokoro(tmp_path, monkeypatch)
    _say(monkeypatch, ["a" * 1200])

    chunks = voice.tokens("long")

    assert [len(ids) for ids in chunks] == [MAX_PHONEMES, MAX_PHONEMES, 180]


def test_espeak_turns_text_into_ids_of_the_models_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The phonemizer ships in the `piper-tts` wheel, so this needs no model file."""
    voice, _session = _kokoro(tmp_path, monkeypatch)
    phonemes = kokoro._espeak_phonemes("Hello world. Good morning.", "en-gb-x-rp")  # pyright: ignore[reportPrivateUsage]
    voice._vocab = {char: number for number, char in enumerate(sorted(set("".join(phonemes))))}  # pyright: ignore[reportPrivateUsage]

    sentences = voice.tokens("Hello world. Good morning.")

    assert len(phonemes) == 2
    assert len(sentences) == 2
    assert all(sentences)
