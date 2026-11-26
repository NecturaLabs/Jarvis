"""The score of a song as the model writes it: notes with words, pitches and lengths, checked.

A score is plain text, one note after another, separated by commas: `TEXT PITCH BEATS`. TEXT is a
word or one syllable of one; a syllable ending in `-` has its word go on in the next note
(`hap- G4 0.75, py G4 0.25`). `~` holds the sound before it on to a new pitch, and a rest is
`_ BEATS`. PITCH is a note name with its octave (`C4` is middle C, `F#4`, `Bb3`) or a MIDI number.
Everything is checked here, before any sound is made: the length, the number of notes and the
range of pitches all have caps.
"""

import math
import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

MIN_TEMPO = 40
MAX_TEMPO = 220
DEFAULT_TEMPO = 100
MAX_NOTES = 64
MAX_SECONDS = 30.0
MIN_BEATS = 0.125
MAX_BEATS = 8.0
LOWEST = 36  # C2
HIGHEST = 84  # C6
MAX_SYLLABLE_CHARS = 24
MAX_SCORE_CHARS = 2000  # the `notes` argument: 64 notes of a long word each, with room

_SEPARATORS = re.compile(r"[,;|\n]")
_PITCH = re.compile(r"([A-Ga-g])([#♯b♭]?)(-?\d)")
_STEPS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_SHARP, _FLAT = "#♯", "b♭"
_DROPPED = re.compile(r"[.!?\"\u201c\u201d\u2018()\[\]:]")  # marks a lyric may carry; not sung
_SYLLABLE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*['\u2019]?")
_HOLD = frozenset({"~", "-", "\u2013"})
_REST = frozenset({"_", "rest"})
_FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)")
_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class ScoreError(ValueError):
    """The score cannot be sung as written; the message says what to change."""


class Kind(Enum):
    """What a note is: a syllable sung, the syllable before it held on, or a rest."""

    SYLLABLE = "syllable"
    HOLD = "hold"
    REST = "rest"


@dataclass(frozen=True, slots=True)
class Note:
    """One note: what is sung on it, its MIDI pitch (0 for a rest) and how long it lasts.

    `word` is the index of the word it belongs to in `Score.words` (-1 for a rest).
    """

    kind: Kind
    text: str
    midi: int
    seconds: float
    word: int


@dataclass(frozen=True, slots=True)
class Score:
    """A checked song: its notes in order and the words they spell."""

    tempo: int
    notes: tuple[Note, ...]
    words: tuple[str, ...]

    @property
    def seconds(self) -> float:
        """How long the song lasts."""
        return sum(note.seconds for note in self.notes)

    @property
    def lyrics(self) -> str:
        """The words sung, in order."""
        return " ".join(self.words)

    @property
    def timed_words(self) -> tuple[tuple[float, float, str], ...]:
        """Each word with the seconds it is sung in: (start, end, word).

        A word runs from its first note's start to its last note's end, its holds included.
        """
        spans: dict[int, list[float]] = {}
        at = 0.0
        for note in self.notes:
            if note.kind is not Kind.REST:
                span = spans.setdefault(note.word, [at, at])
                span[1] = at + note.seconds
            at += note.seconds
        return tuple((start, end, self.words[w]) for w, (start, end) in sorted(spans.items()))

    def transposed(self, semitones: int) -> Score:
        """Return the same song moved by `semitones` (rests stay rests)."""
        notes = tuple(
            Note(n.kind, n.text, n.midi + semitones if n.midi else 0, n.seconds, n.word)
            for n in self.notes
        )
        return Score(self.tempo, notes, self.words)


def hertz(midi: float) -> float:
    """Return the frequency of a MIDI pitch (69 is A4, 440 Hz)."""
    return 440.0 * 2.0 ** ((midi - 69) / 12)


def note_name(midi: int) -> str:
    """Return a MIDI pitch as a note name with its octave (60 is C4)."""
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def pitch(text: str) -> int:
    """Return the MIDI number of a note name (`C4`, `F#3`, `Bb4`) or of a plain MIDI number.

    Raises:
        ScoreError: it is neither, or it lies outside C2 to C6.
    """
    if text.isascii() and text.isdigit():
        midi = int(text)
    elif match := _PITCH.fullmatch(text):
        letter, accidental, octave = match.groups()
        shift = 1 if accidental and accidental in _SHARP else -1 if accidental else 0
        midi = 12 * (int(octave) + 1) + _STEPS[letter.lower()] + shift
    else:
        msg = f"{text!r} is not a pitch: write a note name with its octave, like C4, F#4 or Bb3"
        raise ScoreError(msg)
    if not LOWEST <= midi <= HIGHEST:
        msg = f"{text} is out of range: pitches go from C2 to C6"
        raise ScoreError(msg)
    return midi


def beats(text: str) -> float:
    """Return a length in beats, written as a number (`0.5`, `1.5`) or a fraction (`1/3`).

    Raises:
        ScoreError: it is not a number, or lies outside MIN_BEATS to MAX_BEATS.
    """
    if match := _FRACTION.fullmatch(text):
        top, bottom = int(match[1]), int(match[2])
        exact = Fraction(top, bottom) if bottom else None
        # Compared exactly first: a fraction of long numbers may be too large for a float.
        if exact is not None and not MIN_BEATS <= exact <= MAX_BEATS:
            msg = f"a note of {text} beats is out of range: from {MIN_BEATS} to {MAX_BEATS:g} beats"
            raise ScoreError(msg)
        value = math.nan if exact is None else float(exact)
    else:
        try:
            value = float(text)
        except ValueError:
            value = math.nan
    if not math.isfinite(value):
        msg = f"{text!r} is not a length in beats: write a number such as 1, 0.5 or 1.5"
        raise ScoreError(msg)
    if not MIN_BEATS <= value <= MAX_BEATS:
        msg = f"a note of {text} beats is out of range: from {MIN_BEATS} to {MAX_BEATS:g} beats"
        raise ScoreError(msg)
    return value


def _syllable(text: str) -> tuple[str, bool, bool]:
    """Return a syllable's letters, whether a hyphen joins it to the note before, and to the next.

    Raises:
        ScoreError: it holds anything but letters and apostrophes (a digit, a symbol).
    """
    joins_before = text.startswith("-")
    joins_after = text.endswith("-")
    letters = _DROPPED.sub("", text.strip("-"))
    if not letters or not _SYLLABLE.fullmatch(letters):
        msg = f"{text!r} cannot be sung: a note's text is one word or syllable, in letters"
        if any(ch.isdigit() for ch in text):
            msg += " (write numbers as words)"
        raise ScoreError(msg)
    if len(letters) > MAX_SYLLABLE_CHARS:
        msg = f"{text!r} is too long for one note: give each syllable its own note"
        raise ScoreError(msg)
    return letters.replace("\u2019", "'"), joins_before, joins_after


class _Builder:
    """Collects notes and the words they spell while a score is read."""

    def __init__(self, tempo: int) -> None:
        super().__init__()
        self.beat_s = 60.0 / tempo
        self.notes: list[Note] = []
        self.words: list[str] = []
        self.open = False  # the last word ended in a hyphen: the next syllable joins it

    def rest(self, length: str) -> None:
        self.notes.append(Note(Kind.REST, "", 0, beats(length) * self.beat_s, -1))
        self.open = False

    def hold(self, where: str, length: tuple[str, str]) -> None:
        if not self.notes or self.notes[-1].kind is Kind.REST:
            msg = f"{where}: '~' holds the syllable before it, so it cannot follow a rest or begin"
            raise ScoreError(msg)
        word = self.notes[-1].word
        self.notes.append(
            Note(Kind.HOLD, "", pitch(length[0]), beats(length[1]) * self.beat_s, word)
        )

    def syllable(self, text: str, where: str, length: tuple[str, str]) -> None:
        letters, before, after = _syllable(text)
        midi, seconds = pitch(length[0]), beats(length[1]) * self.beat_s
        joins = (self.open or before) and bool(self.words) and self.notes[-1].kind is not Kind.REST
        if joins:
            self.words[-1] += letters
        else:
            if before:
                msg = f"{where}: {text!r} starts with '-' but no word goes on into it"
                raise ScoreError(msg)
            self.words.append(letters)
        self.notes.append(Note(Kind.SYLLABLE, letters, midi, seconds, len(self.words) - 1))
        self.open = after


def parse(notes: str, tempo: int = DEFAULT_TEMPO) -> Score:
    """Read and check a score (see the module's docstring) at `tempo` beats a minute.

    Raises:
        ScoreError: what cannot be sung, and why, in words the model can act on.
    """
    if not MIN_TEMPO <= tempo <= MAX_TEMPO:
        msg = f"a tempo of {tempo} is out of range: from {MIN_TEMPO} to {MAX_TEMPO} beats a minute"
        raise ScoreError(msg)
    if len(notes) > MAX_SCORE_CHARS:
        msg = f"the score is too long: at most {MAX_NOTES} notes and {MAX_SECONDS:g} seconds"
        raise ScoreError(msg)
    build = _Builder(tempo)
    pieces = [piece.split() for piece in _SEPARATORS.split(notes)]
    for number, parts in enumerate((p for p in pieces if p), start=1):
        where = f"note {number}"
        if len(build.notes) >= MAX_NOTES:
            msg = f"too many notes: at most {MAX_NOTES}"
            raise ScoreError(msg)
        if len(parts) == 2 and parts[0].lower() in _REST:  # ruff: ignore[magic-value-comparison] - a rest and its length
            build.rest(parts[1])
        elif len(parts) == 3 and parts[0] in _HOLD:  # ruff: ignore[magic-value-comparison] - text, pitch, length
            build.hold(where, (parts[1], parts[2]))
        elif len(parts) == 3:  # ruff: ignore[magic-value-comparison] - text, pitch, length
            build.syllable(parts[0], where, (parts[1], parts[2]))
        else:
            msg = (
                f"{where} ({' '.join(parts)!r}) is not 'TEXT PITCH BEATS' or a rest '_ BEATS'; "
                "separate notes with commas"
            )
            raise ScoreError(msg)
    score = Score(tempo, tuple(build.notes), tuple(build.words))
    if not score.words:
        msg = "the score has no words to sing"
        raise ScoreError(msg)
    if score.seconds > MAX_SECONDS + 1e-9:
        msg = (
            f"the song would last {score.seconds:.1f} seconds; sing at most {MAX_SECONDS:g}"
            " (fewer notes, shorter ones or a faster tempo)"
        )
        raise ScoreError(msg)
    return score
