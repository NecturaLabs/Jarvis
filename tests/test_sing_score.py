"""The score the model writes for `sing`: read, checked and refused before any sound is made."""

import pytest

from jarvis.sing.score import (
    MAX_NOTES,
    MAX_SECONDS,
    Kind,
    ScoreError,
    hertz,
    note_name,
    parse,
    pitch,
)

# An original line, written for these tests.
TUNE = "Sun- C4 1, shine E4 0.5, on G4 0.5, the E4 1, wa- D4 1, ter C4 2, ~ D4 1, _ 1"


def test_a_score_is_read_into_notes_words_and_seconds() -> None:
    score = parse(TUNE, 120)

    assert score.words == ("Sunshine", "on", "the", "water")
    assert [n.kind for n in score.notes] == [
        *[Kind.SYLLABLE] * 6,
        Kind.HOLD,
        Kind.REST,
    ]
    assert [n.midi for n in score.notes] == [60, 64, 67, 64, 62, 60, 62, 0]
    assert [n.word for n in score.notes] == [0, 0, 1, 2, 3, 3, 3, -1]
    assert score.seconds == pytest.approx(8 * 0.5)
    assert score.lyrics == "Sunshine on the water"


@pytest.mark.parametrize(
    ("name", "midi"),
    [("C4", 60), ("c4", 60), ("A4", 69), ("F#4", 66), ("Bb3", 58), ("bb3", 58), ("C2", 36),
     ("C6", 84), ("72", 72), ("E♭4", 63)],
)  # fmt: skip
def test_pitches_are_note_names_with_an_octave_or_midi_numbers(name: str, midi: int) -> None:
    assert pitch(name) == midi


@pytest.mark.parametrize("name", ["B1", "C#6", "100", "H4", "C", "do", "\u00b2"])
def test_a_pitch_outside_the_range_or_not_a_note_is_refused(name: str) -> None:
    with pytest.raises(ScoreError):
        pitch(name)


def test_lengths_may_be_fractions_and_the_tempo_sets_the_seconds() -> None:
    score = parse("la C4 1/3, la D4 1.5", 60)

    assert [n.seconds for n in score.notes] == pytest.approx([1 / 3, 1.5])


def test_a_hyphen_on_either_side_joins_syllables_into_one_word() -> None:
    assert parse("hel- C4 1, lo D4 1, -o E4 1").words == ("helloo",)


def test_a_rest_ends_a_word_left_open() -> None:
    assert parse("hel- C4 1, _ 1, lo D4 1").words == ("hel", "lo")


def test_marks_around_a_lyric_are_dropped_and_apostrophes_kept() -> None:
    assert parse("\u201cDon\u2019t! C4 1, (go) D4 1").words == ("Don't", "go")


@pytest.mark.parametrize(
    ("notes", "says"),
    [
        ("la C4", "is not 'TEXT PITCH BEATS'"),
        ("la C4 1 2", "is not 'TEXT PITCH BEATS'"),
        ("~ C4 1, la D4 1", "cannot follow a rest or begin"),
        ("_ 1, ~ C4 1", "cannot follow a rest or begin"),
        ("-lo C4 1", "starts with '-'"),
        ("4 C4 1", "write numbers as words"),
        ("la$ C4 1", "one word or syllable"),
        ("la C4 9", "out of range"),
        ("la C4 0.1", "out of range"),
        ("la C4 x", "not a length in beats"),
        ("la C4 1/0", "not a length in beats"),
        ("la Q4 1", "is not a pitch"),
        ("_ 1, _ 2", "no words to sing"),
        ("supercalifragilisticexpialidocious C4 1", "give each syllable its own note"),
    ],
)
def test_what_cannot_be_sung_is_refused_with_what_to_change(notes: str, says: str) -> None:
    with pytest.raises(ScoreError, match=says.replace("(", r"\(").replace(")", r"\)")):
        parse(notes)


def test_more_than_the_most_notes_is_refused() -> None:
    parse(", ".join(["la C4 0.25"] * MAX_NOTES), 200)
    with pytest.raises(ScoreError, match="too many notes"):
        parse(", ".join(["la C4 0.25"] * (MAX_NOTES + 1)), 200)


def test_a_song_longer_than_the_cap_is_refused() -> None:
    beats = int(MAX_SECONDS)  # at 60 a minute, one second a beat
    parse(", ".join(["la C4 1"] * beats), 60)
    with pytest.raises(ScoreError, match="would last"):
        parse(", ".join(["la C4 1"] * (beats + 1)), 60)


@pytest.mark.parametrize("tempo", [39, 221])
def test_a_tempo_out_of_range_is_refused(tempo: int) -> None:
    with pytest.raises(ScoreError, match="tempo"):
        parse("la C4 1", tempo)


def test_a_score_moves_by_semitones_and_rests_stay_rests() -> None:
    moved = parse("la C4 1, _ 1").transposed(-12)

    assert [n.midi for n in moved.notes] == [48, 0]


def test_pitch_helpers_agree_with_concert_pitch() -> None:
    assert hertz(69) == pytest.approx(440.0)
    assert hertz(57) == pytest.approx(220.0)
    assert note_name(60) == "C4"
    assert note_name(70) == "A#4"
