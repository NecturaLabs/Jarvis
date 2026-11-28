# ruff: file-ignore[ambiguous-unicode-character-string, ambiguous-unicode-character-comment] - the tests hold Kokoro's IPA phoneme symbols
"""Kokoro's session settings and its own logic, against a stand-in model."""

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, override

import numpy as np
import pytest

from jarvis import kokoro, voices
from jarvis.kokoro import (
    MAX_PHONEMES,
    MAX_THREADS,
    SAMPLE_RATE,
    Kokoro,
    session_options,
)
from jarvis.phonemes import trained_phonemes

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
    *,
    voice: str = "bm_george",
    phoneme_map: bool = False,
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
        packs: dict[str, Any] = {voice: rows.reshape(MAX_PHONEMES, 1, STYLE_WIDTH)}
        np.savez(handle, **packs)
    return Kokoro(tmp_path / "model.onnx", voices, voice, phoneme_map=phoneme_map), session


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
    voice, session = _kokoro(tmp_path, monkeypatch, voice="bf_uncatalogued")  # no level change
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
    phonemes = kokoro._espeak_phonemes("Hello world. Good morning.", "en-gb-x-rp")  # pyright: ignore[reportPrivateUsage] - espeak's phonemizer, used to derive a vocabulary
    voice._vocab = {char: number for number, char in enumerate(sorted(set("".join(phonemes))))}  # pyright: ignore[reportPrivateUsage] - the stand-in model has no public way to set its vocabulary

    sentences = voice.tokens("Hello world. Good morning.")

    assert len(phonemes) == 2
    assert len(sentences) == 2
    assert all(sentences)


# The symbols of the vocabulary in `kokoro-v1.0.onnx`'s own metadata (114 entries).
V1_VOCAB = frozenset(
    ' !"(),.:;?AIOQSTWYabcdefhijklmnopqrstuvwxyzæçðøŋœɐɑɒɔɕɖəɚɛɜɟɡɣɤɥɨɪɯɰɲɳɴɸɹɻɽɾ'
    "ʁʂʃʈʊʋʌʎʒʔʝʣʤʥʦʧʨʰʲˈˌː̃βθχᵊᵝᵻ—“”…→↓↗↘ꭧ"
)

BRITISH = [
    ("kˈæfeɪ", "kˈæfA"),  # eɪ is A
    ("plˈeɪɪŋ ɡlˈɑːs hˈɑːbɐ baɪ nˈaɪt sˈɜːkɪt nˈaʊ.", "plˈAɪŋ ɡlˈɑːs hˈɑːbə bI nˈIt sˈɜːkɪt nˈW."),
    ("ɡˈəʊ", "ɡˈQ"),
    ("tʃˈɜːtʃ dʒˈʌdʒ", "ʧˈɜːʧ ʤˈʌʤ"),
    ("hˈeə nˈiə", "hˈɛː nˈɪə"),
    ("ɔɪl", "Yl"),
    ("rˈɛd", "ɹˈɛd"),
    ("bˈʌʔn̩", "bˈʌtn"),  # a glottal stop is t, and the n after it is not syllabic
    ("ˈɪtl̩", "ˈɪtᵊl"),  # a syllabic mark moves to a ᵊ in front
    ("lˈɒʔ ʔ", "lˈɒt t"),
    ("lˈɒx", "lˈɒk"),
    ("sʲˈuː ʲə ʲo", "sˈuː jə jɔ"),  # a palatal mark goes, or turns into j
]
AMERICAN = [
    ("ɡˈoʊt", "ɡˈOt"),
    ("wˈɔːɾɚ", "wˈɔTəɹ"),  # the long mark goes, a tap is T, ɚ is əɹ
    ("bˈɜːd hˈɑːɹbɚ", "bˈɜɹd hˈɑɹbəɹ"),
    ("nˈɪə", "nˈiə"),
    ("dʒˈʌdʒ tʃ", "ʤˈʌʤ ʧ"),
    ("nˈaɪnti", "nˈInti"),
    ("ˈeɪ", "ˈA"),
]


@pytest.mark.parametrize(("espeak", "trained"), BRITISH)
def test_british_espeak_phonemes_become_the_symbols_kokoro_was_trained_on(
    espeak: str, trained: str
) -> None:
    assert trained_phonemes(espeak, british=True) == trained


@pytest.mark.parametrize(("espeak", "trained"), AMERICAN)
def test_american_espeak_phonemes_become_the_symbols_kokoro_was_trained_on(
    espeak: str, trained: str
) -> None:
    assert trained_phonemes(espeak, british=False) == trained


@pytest.mark.parametrize("british", [True, False])
def test_what_the_table_writes_is_in_the_models_vocabulary_and_is_not_rewritten_again(
    *, british: bool
) -> None:
    for espeak, trained in BRITISH if british else AMERICAN:
        assert set(trained) <= V1_VOCAB, trained
        assert trained_phonemes(trained, british=british) == trained, espeak


def test_a_longer_match_wins_over_a_shorter_one() -> None:
    # `ɜːɹ` is one sound, not `ɜː` followed by `ɹ` (which would be spoken twice).
    assert trained_phonemes("ɜːɹ", british=False) == "ɜɹ"
    assert trained_phonemes("ɔɪ", british=True) == "Y"  # not ɔ then ɪ


def test_the_table_is_off_unless_asked_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _say(monkeypatch, ["kˈæfeɪ"])
    vocab = {char: number for number, char in enumerate(sorted(V1_VOCAB), start=1)}
    plain, _ = _kokoro(tmp_path, monkeypatch)
    plain._vocab = vocab  # pyright: ignore[reportPrivateUsage] - the stand-in model has no public way to set its vocabulary
    mapped, _ = _kokoro(tmp_path, monkeypatch, phoneme_map=True)
    mapped._vocab = vocab  # pyright: ignore[reportPrivateUsage] - the stand-in model has no public way to set its vocabulary

    as_text = {number: char for char, number in vocab.items()}
    assert "".join(as_text[i] for i in plain.tokens("Cafe")[0]) == "kˈæfeɪ"
    assert "".join(as_text[i] for i in mapped.tokens("Cafe")[0]) == "kˈæfA"


def test_the_table_follows_the_voices_dialect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _say(monkeypatch, ["ɡˈoʊt ɡˈəʊ"])
    vocab = {char: number for number, char in enumerate(sorted(V1_VOCAB), start=1)}
    as_text = {number: char for char, number in vocab.items()}
    heard: dict[str, str] = {}
    for name in ("am_adam", "bm_george"):
        voice, _ = _kokoro(tmp_path, monkeypatch, voice=name, phoneme_map=True)
        voice._vocab = vocab  # pyright: ignore[reportPrivateUsage] - the stand-in model has no public way to set its vocabulary
        heard[name] = "".join(as_text[i] for i in voice.tokens("x")[0])

    assert heard == {"am_adam": "ɡˈOt ɡˈəʊ", "bm_george": "ɡˈɔʊt ɡˈQ"}


class _TimedSession(_Session):
    """A model that also gives each token's length, as kokoro-v1.0.onnx does."""

    @override
    def run(self, _outputs: None, feed: dict[str, Any]) -> list[Any]:
        self.feeds.append(feed)
        count = feed["input_ids"].shape[1]
        steps = np.arange(1, count + 1, dtype=np.int64)
        audio = np.zeros(int(steps.sum()) * kokoro.FRAME_SAMPLES, dtype=np.float32)
        return [audio, steps]


def test_timed_speech_gives_each_symbol_and_its_length_without_the_pads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    voice, _ = _kokoro(tmp_path, monkeypatch)
    session = _TimedSession({"input_ids": "", "style": "", "speed": ""})
    voice._session = session  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue] - the stand-in model
    _say(monkeypatch, ["ab ba"])

    (chunk,) = voice.timed("ab ba", speed=0.8)

    assert chunk.symbols == ("a", "b", " ", "b", "a")
    assert chunk.frames.tolist() == [2, 3, 4, 5, 6]  # the pads' 1 and 7 are left out
    assert chunk.lead == kokoro.FRAME_SAMPLES  # the start pad's one step
    assert chunk.audio.size == 28 * kokoro.FRAME_SAMPLES
    assert session.feeds[0]["speed"].tolist() == [pytest.approx(0.8)]


def test_a_model_without_lengths_cannot_give_timed_speech(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    voice, _ = _kokoro(tmp_path, monkeypatch)
    _say(monkeypatch, ["ab"])

    with pytest.raises(ValueError, match="no duration"):
        voice.timed("ab")


def test_the_model_file_is_loaded_once_for_every_voice_that_uses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[str] = []
    session = _Session({"input_ids": "", "style": "", "speed": ""})

    def load(path: str, **_kwargs: object) -> _Session:
        loads.append(path)
        return session

    monkeypatch.setattr(kokoro.ort, "InferenceSession", load)
    voices = tmp_path / "voices.bin"
    with voices.open("wb") as handle:
        np.savez(handle, a=np.zeros((MAX_PHONEMES, 1, STYLE_WIDTH), dtype=np.float32))
    for _ in range(2):
        Kokoro(tmp_path / "shared.onnx", voices, "a")

    assert loads == [str(tmp_path / "shared.onnx")]


@pytest.mark.parametrize("name", ["af_nova", "af_aoede"])
def test_each_catalogued_voice_is_brought_to_the_same_speech_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    voice, _session = _kokoro(tmp_path, monkeypatch, voice=name)
    _say(monkeypatch, ["ab"])
    found = voices.find(name)
    assert found is not None
    gain = 10 ** (found.gain_db / 20)

    pcm = voice.synthesize("x")
    voice._session = _LoudTimedSession({"input_ids": "", "style": "", "speed": ""})  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue] - the stand-in model
    (chunk,) = voice.timed("x")

    assert pcm[0] == int(min(0.5 * gain, 1.0) * 32767)
    assert chunk.audio[0] == pytest.approx(0.5 * gain)  # the singer starts from the same level


class _LoudTimedSession(_TimedSession):
    """A timed model whose speech is all at 0.5."""

    @override
    def run(self, _outputs: None, feed: dict[str, Any]) -> list[Any]:
        audio, steps = super().run(_outputs, feed)
        return [np.full_like(audio, 0.5), steps]
