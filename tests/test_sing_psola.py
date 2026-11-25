# ruff: file-ignore[ambiguous-unicode-character-string] - the tests hold Kokoro's IPA phoneme symbols
"""The speaking-voice singer, against a stand-in voice whose sounds and lengths are known.

The stand-in says each letter of a word as one symbol: a vowel is a buzz at a steady pitch, a
consonant is noise. Like Kokoro, it counts the silence after its last word into the last sound.
The song it sings is then measured: its length, each note's pitch, and its rests.
"""

import math

import numpy as np
import parselmouth
import pytest

from jarvis.kokoro import FRAME_SAMPLES, Timed
from jarvis.sing import psola
from jarvis.sing.psola import RATE, PsolaSinger, octave_shift, syllables
from jarvis.sing.score import Kind, Score, parse

VOICE_HZ = 120.0
LEAD_FRAMES = 4
VOWEL_FRAMES = 4
CONSONANT_FRAMES = 2
SPACE_FRAMES = 2
SILENT_TAIL_FRAMES = 12  # counted into the last sound, as Kokoro does


def _sound(symbol: str, frames: int, rng: np.random.Generator) -> np.ndarray:
    n = frames * FRAME_SAMPLES
    if symbol == " ":
        return np.zeros(n, dtype=np.float32)
    if symbol in "aeiou":
        t = np.arange(n) / RATE
        return (0.3 * (2 * ((t * VOICE_HZ) % 1.0) - 1)).astype(np.float32)
    return (0.05 * rng.standard_normal(n)).astype(np.float32)


class Voice:
    """Says text letter by letter; `merge` says two words as one, so they do not line up."""

    def __init__(self, *, merge: bool = False) -> None:
        super().__init__()
        self.merge = merge
        self.said: list[str] = []

    def timed(self, text: str, *, speed: float = 1.0) -> list[Timed]:
        assert speed > 0
        self.said.append(text)
        if self.merge and text.count(" ") > 1:  # a phrase, not a word said alone
            text = text.replace(" ", "", 1)
        rng = np.random.default_rng(len(text))
        symbols = tuple(text.lower())
        frames = [
            SPACE_FRAMES if s == " " else VOWEL_FRAMES if s in "aeiou" else CONSONANT_FRAMES
            for s in symbols
        ]
        parts = [np.zeros(LEAD_FRAMES * FRAME_SAMPLES, dtype=np.float32)]
        parts += [_sound(s, f, rng) for s, f in zip(symbols, frames, strict=True)]
        parts.append(np.zeros(SILENT_TAIL_FRAMES * FRAME_SAMPLES, dtype=np.float32))
        frames[-1] += SILENT_TAIL_FRAMES
        return [
            Timed(
                np.concatenate(parts),
                symbols,
                np.array(frames, dtype=np.int64),
                LEAD_FRAMES * FRAME_SAMPLES,
            )
        ]


# An original tune, written for these tests: syllables, a hold, a rest and a long last note.
TUNE = "pa C4 1, ta E4 1, ~ G4 1, _ 0.5, ma- D4 0.5, ka C4 2"


def _midi(hz: float) -> float:
    return 69 + 12 * math.log2(hz / 440.0)


def _sung(audio: np.ndarray, score: Score) -> list[tuple[float | None, float]]:
    """Return each sung note's median pitch (MIDI) and the share of it heard voiced.

    Each is read after its first consonants, which are noise, and before its very end. Praat
    makes the noise afresh each time (its overlap-add is not repeatable), so voicing near a
    consonant varies a little between runs.
    """
    track = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=RATE).to_pitch(
        time_step=0.01, pitch_floor=60, pitch_ceiling=800
    )
    hz, times = track.selected_array["frequency"], track.xs()
    out: list[tuple[float | None, float]] = []
    at = 0.0
    for note in score.notes:
        start, end = at + 0.4 * note.seconds, at + 0.9 * note.seconds
        at += note.seconds
        if note.kind is Kind.REST:
            continue
        inside = hz[(times >= start) & (times <= end)]
        voiced = inside[inside > 0]
        share = voiced.size / max(1, inside.size)
        out.append((_midi(float(np.median(voiced))) if voiced.size else None, share))
    return out


def _render(voice: Voice, score: Score) -> np.ndarray:
    pcm, rate = PsolaSinger(voice).render(score)
    assert rate == RATE
    assert pcm.dtype == np.int16
    return pcm.astype(np.float32) / 32767


def test_each_note_is_sung_on_its_pitch_for_its_length() -> None:
    score = parse(TUNE, 120)
    audio = _render(Voice(), score)

    assert audio.size == round(score.seconds * RATE)
    shift = octave_shift(score, _midi(VOICE_HZ))
    sung = [n for n in score.notes if n.kind is not Kind.REST]
    for note, (heard, voiced) in zip(sung, _sung(audio, score), strict=True):
        assert heard is not None, note
        assert abs(heard - (note.midi + shift)) < 0.5, (note, heard)
        assert voiced > 0.9, (note, voiced)


def test_the_last_note_is_held_in_voice_not_in_the_silence_after_the_words() -> None:
    score = parse("la C4 1, lo C4 4", 120)
    audio = _render(Voice(), score)

    _, voiced = _sung(audio, score)[-1]
    assert voiced > 0.9


def test_a_rest_is_silent_and_the_words_are_said_with_one_more_after_them() -> None:
    voice = Voice()
    score = parse(TUNE, 120)
    audio = _render(voice, score)

    rest = audio[round(1.5 * RATE) : round(1.75 * RATE)]  # beats 3 to 3.5 at 120 a minute
    assert not np.any(rest)
    assert voice.said == [f"pa ta {psola.TRAILER}", f"maka {psola.TRAILER}"]


def test_words_that_do_not_line_up_with_the_speech_are_said_one_by_one() -> None:
    voice = Voice(merge=True)
    score = parse("pa C4 1, ta E4 1", 120)
    audio = _render(voice, score)

    assert audio.size == round(score.seconds * RATE)
    assert voice.said[1:] == [f"pa {psola.TRAILER}", f"ta {psola.TRAILER}"]
    assert all(heard is not None for heard, _ in _sung(audio, score))


def test_the_song_is_moved_by_whole_octaves_towards_the_voice() -> None:
    score = parse("la C5 1, la E5 1, la G5 1")

    assert octave_shift(score, 55.0) == -12  # 13 semitones above where this voice sings
    assert octave_shift(score, 47.0) == -24
    assert octave_shift(score, 66.0) == 0
    assert octave_shift(score, None) == 0
    wide = parse("la C2 1, la C5 1, la C6 1")  # three octaves: there is no room to move it down
    assert octave_shift(wide, 40.0) == 0


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (2, [(0, 1, 3, 3), (3, 4, 5, 5)]),  # hap-py: the p opens the second syllable
        (1, [(0, 1, 3, 5)]),  # one note: the stressed vowel is the one held
        (3, [(0, 1, 3, 3), (3, 4, 5, 5)]),  # more notes than vowels: no more syllables
    ],
)
def test_a_word_is_split_into_syllables_at_its_vowels(
    count: int, expected: list[tuple[int, int, int, int]]
) -> None:
    assert syllables(("h", "ˈ", "æ", "p", "ɪ"), count) == expected


def test_a_word_without_a_vowel_is_one_syllable_held_whole() -> None:
    assert syllables(("ʃ", "ʃ"), 2) == [(0, 0, 2, 2)]
