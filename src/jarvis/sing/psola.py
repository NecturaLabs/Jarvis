# ruff: file-ignore[ambiguous-unicode-character-string] - the vowels are Kokoro's IPA symbols
"""Singing with the speaking voice: Kokoro says the words, Praat sets each note's pitch and length.

Kokoro gives how long each phoneme symbol of its speech lasts. Each word is split into its
syllables at its vowels, each syllable is matched to its notes, and Praat's overlap-add
(PSOLA, through `praat-parselmouth`) stretches the vowel to the note's length and moves the voice
to the note's pitch. Consonants keep their own length, so the words stay clear; each note starts
on its beat. The whole song is moved by whole octaves to sit near the voice's own pitch.
It runs on the CPU in a fraction of the song's length, and holds nothing while idle.
"""

import itertools
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from jarvis.sing.score import Kind, Note, Score, hertz

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from jarvis.kokoro import Timed

log = logging.getLogger(__name__)

RATE = 24000
FRAME_SAMPLES = 600  # Kokoro's duration step
SOURCE_SPEED = 0.9  # the words are spoken a little slowly, so a vowel needs less stretching
VOWELS = frozenset("aeiouyæɐɑɒɔəɚɛɜɝɪʊʌɨʉøœɤɯᵊᵻAIOQWYː")
STRESS = frozenset("ˈˌ")
MIN_VOWEL_SHARE = 0.5  # a vowel shrunk below half its length: the whole syllable is shrunk instead
GLIDE_S = 0.04  # how long the voice takes to move from one note's pitch to the next
VIBRATO_AFTER_S = 0.3  # a note held longer than this gets a vibrato, after this long
VIBRATO_RAMP_S = 0.2
VIBRATO_HZ = 5.5
VIBRATO_SEMITONES = 0.25
PITCH_STEP_S = 0.01
FADE_S = 0.005
PEAK = 0.95
SINGING_ABOVE_SPEECH = 8  # semitones: a voice sings a little above where it speaks
MIN_PITCH_HZ, MAX_PITCH_HZ = 60.0, 700.0
MIN_ANALYSIS_SAMPLES = 2400  # 0.1 s: Praat reads a pitch only from three of its lowest periods
EPS_S = 1e-4
PSOLA_ROOM = 3  # Praat's overlap-add output is at most this many times its input
TRAILER = "and"  # said after the words and dropped (`_said`)
TAIL_SAMPLES = 1200  # 50 ms
EARLY_SAMPLES = 480  # Kokoro's sounds come about 20 ms before its durations say (measured)
STEP = 240  # 10 ms, the step the source's loudness and pitch are read in
LOUD_SHARE = 0.1  # of the phrase's loudest step: quieter is no longer the voice
MODAL_LOW, MODAL_HIGH = 0.7, 1.6  # of the phrase's median pitch: the voice's own range


class Speech(Protocol):
    """Speech with the length of each phoneme symbol (`Kokoro.timed`)."""

    def timed(self, text: str, *, speed: float = 1.0) -> list[Timed]:
        """Return the chunks of `text` spoken, with their symbols and lengths."""
        ...


@dataclass(frozen=True, slots=True)
class _Word:
    """The sound of one word: its samples, its symbols and where each symbol starts."""

    audio: NDArray[np.float32]
    symbols: tuple[str, ...]
    starts: tuple[int, ...]  # sample where each symbol starts, then where the word ends


@dataclass(frozen=True, slots=True)
class _Region:
    """A stretch of the source that plays `factor` times as long in the song."""

    src0: float
    src1: float
    factor: float
    out0: float

    @property
    def out1(self) -> float:
        return self.out0 + (self.src1 - self.src0) * self.factor


def _is_vowel(symbol: str) -> bool:
    return symbol in VOWELS


def _nuclei(symbols: Sequence[str]) -> list[tuple[int, int]]:
    """Return each run of vowel symbols, with the stress mark before it, as (first, past last).

    Kokoro's stress mark lasts a while of its own, and what it sounds is the vowel after it.
    """
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(symbols):
        j = i
        while j < len(symbols) and symbols[j] in STRESS:
            j += 1
        if j < len(symbols) and _is_vowel(symbols[j]) and symbols[j] != "ː":
            while j < len(symbols) and _is_vowel(symbols[j]):
                j += 1
            runs.append((i, j))
            i = j
        else:
            i = j + 1 if j == i else j
    return runs


def _onset(symbols: Sequence[str], after: int, vowel: int) -> int:
    """Return where the syllable whose vowel starts at `vowel` begins, after the one before.

    The consonant right before the vowel opens the syllable; any before it close the one before.
    """
    start = vowel
    if start > after and symbols[start - 1] not in STRESS and not _is_vowel(symbols[start - 1]):
        start -= 1
    return start


def syllables(symbols: Sequence[str], count: int) -> list[tuple[int, int, int, int]]:
    """Split a word's symbols into at most `count` syllables at its vowels.

    Each is (start, vowel start, vowel end, end) in symbol indices, covering the word in order.
    With more vowels than `count`, neighbouring ones share a syllable and its stressed vowel (or
    else its first) is the one held. With fewer, fewer syllables come back: the caller sings the
    notes left over on the last one's vowel. A word without a vowel is one syllable, all held.
    """
    runs = _nuclei(symbols)
    if not runs or count < 1:
        return [(0, 0, len(symbols), len(symbols))]
    groups = min(count, len(runs))
    grouped = [runs[len(runs) * g // groups : len(runs) * (g + 1) // groups] for g in range(groups)]
    starts = [0]
    for before, after in itertools.pairwise(grouped):
        starts.append(_onset(symbols, before[-1][1], after[0][0]))
    starts.append(len(symbols))
    out: list[tuple[int, int, int, int]] = []
    for g, group in enumerate(grouped):
        held = next((run for run in group if symbols[run[0]] in STRESS), group[0])
        out.append((starts[g], held[0], held[1], starts[g + 1]))
    return out


def _words_of(chunk: Timed) -> list[_Word]:
    """Split one spoken chunk at its spaces into words, each with its own samples."""
    edges = np.concatenate([[0], np.cumsum(chunk.frames)]) * FRAME_SAMPLES + chunk.lead
    edges = np.maximum(0, edges - EARLY_SAMPLES)
    words: list[_Word] = []
    first = 0
    symbols = chunk.symbols
    for i in range(len(symbols) + 1):
        if i < len(symbols) and symbols[i] != " ":
            continue
        if any(s not in STRESS for s in symbols[first:i]):
            # A word keeps the pause after it, up to the next word's start; the last keeps a
            # little of the model's end padding, where its last sound dies away.
            end = int(edges[i + 1]) if i < len(symbols) else int(edges[i]) + TAIL_SAMPLES
            starts = tuple(int(e) - int(edges[first]) for e in edges[first : i + 1])
            words.append(
                _Word(chunk.audio[int(edges[first]) : end], tuple(symbols[first:i]), starts)
            )
        first = i + 1
    return words


def _said(speech: Speech, words: Sequence[str]) -> list[_Word] | None:
    """Return the sound of each of `words` said together, None if they do not line up.

    One more word is said after them and dropped: said last, a word falls in pitch and often
    ends in a creak, which cannot be sung.
    """
    chunks = speech.timed(" ".join([*words, TRAILER]), speed=SOURCE_SPEED)
    if len(chunks) != 1:
        return None
    said = _words_of(chunks[0])
    return said[: len(words)] if len(said) == len(words) + 1 else None


def _spoken(speech: Speech, words: Sequence[str]) -> list[_Word]:
    """Return the sound of each word: said together, or one by one if they do not line up."""
    together = _said(speech, words)
    if together is not None:
        return together
    log.info("the words of a phrase did not line up with its speech; saying them one by one")
    out: list[_Word] = []
    for word in words:
        alone = _said(speech, [word])
        out.append(
            alone[0]
            if alone
            else _Word(np.zeros(FRAME_SAMPLES, dtype=np.float32), (), (0, FRAME_SAMPLES))
        )
    return out


@dataclass(frozen=True, slots=True)
class _Piece:
    """One syllable and the notes sung on it, with its place in the phrase's source."""

    notes: tuple[Note, ...]
    start: int
    vowel0: int
    vowel1: int
    end: int


def _pieces(phrase: Sequence[Note], words: Sequence[_Word], first_word: int) -> list[_Piece]:
    """Match each syllable of each word to its notes; sources are offsets in the joined words."""
    pieces: list[_Piece] = []
    offset = 0
    for w, word in enumerate(words):
        notes = [n for n in phrase if n.word == first_word + w]
        sung = [n for n in notes if n.kind is Kind.SYLLABLE]
        parts = syllables(word.symbols, len(sung))
        # Each syllable takes its note and the holds after it; notes past the last syllable
        # (more notes than vowels) are sung on the last syllable's vowel.
        groups: list[list[Note]] = []
        for note in notes:
            if note.kind is Kind.SYLLABLE and len(groups) < len(parts):
                groups.append([note])
            else:
                groups[-1].append(note)
        for (s, v0, v1, e), group in zip(parts, groups, strict=True):
            at = word.starts
            end = word.audio.size if e == len(word.symbols) else at[e]
            pieces.append(
                _Piece(tuple(group), offset + at[s], offset + at[v0], offset + at[v1], offset + end)
            )
        offset += word.audio.size
    return pieces


def _sounding(source: NDArray[np.float32]) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return, for each STEP of `source`, whether it is loud, and whether it is clearly sung.

    Clearly sung is loud and voiced at the voice's own pitch: not the creak a voice drops into at
    the end of a sentence, an octave or more below it, which cannot be moved to a note.
    """
    import parselmouth

    steps = source.size // STEP
    if steps == 0:
        empty = np.zeros(0, dtype=np.bool_)
        return empty, empty
    frames = source[: steps * STEP].reshape(steps, STEP)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    loud = rms >= LOUD_SHARE * float(rms.max())
    if source.size < MIN_ANALYSIS_SAMPLES:
        return loud, loud
    sound = parselmouth.Sound(source.astype(np.float64), sampling_frequency=RATE)
    track = sound.to_pitch(
        time_step=STEP / RATE, pitch_floor=MIN_PITCH_HZ, pitch_ceiling=MAX_PITCH_HZ
    )
    hz = np.interp(
        (np.arange(steps) + 0.5) * STEP / RATE,
        track.xs(),
        track.selected_array["frequency"],
        left=0.0,
        right=0.0,
    )
    voiced = hz[hz > 0]
    if voiced.size == 0:
        return loud, loud
    middle = float(np.median(voiced))
    modal = (hz >= MODAL_LOW * middle) & (hz <= MODAL_HIGH * middle)
    return loud, loud & modal


def _longest(mask: NDArray[np.bool_]) -> tuple[int, int] | None:
    """Return the longest run of True in `mask` as (first, past the last), None if there is none."""
    best: tuple[int, int] | None = None
    start = -1
    for i, value in enumerate([*mask.tolist(), False]):
        if value and start < 0:
            start = i
        elif not value and start >= 0:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = -1
    return best


def _trimmed(source: NDArray[np.float32], pieces: Sequence[_Piece]) -> list[_Piece]:
    """Keep only the clearly sung part of each vowel, and end the phrase where its sound ends.

    Kokoro counts the silence after the last word into its last sound, and a vowel can fade in,
    fade out or creak; held on, that would be sung as silence or a rattle. The rest of the vowel
    is kept at its own length.
    """
    loud, sung = _sounding(source)
    out: list[_Piece] = []
    for i, piece in enumerate(pieces):
        v0, v1, end = piece.vowel0, piece.vowel1, piece.end
        first, past = v0 // STEP, -(-v1 // STEP)
        run = _longest(sung[first:past]) or _longest(loud[first:past])
        if run is not None and v1 > v0:
            a, b = (first + run[0]) * STEP, (first + run[1]) * STEP
            v0, v1 = max(v0, a), min(v1, max(b, a + STEP))
        if i == len(pieces) - 1:
            sounding = np.flatnonzero(loud[: -(-end // STEP)])
            if sounding.size:
                end = min(end, max(v1, (int(sounding[-1]) + 1) * STEP + TAIL_SAMPLES))
        out.append(_Piece(piece.notes, piece.start, v0, max(v0, v1), max(v1, end)))
    return out


def _regions(pieces: Sequence[_Piece]) -> list[_Region]:
    """Return how long each part of the source plays so every piece lasts its notes' length."""
    regions: list[_Region] = []
    out = 0.0
    for piece in pieces:
        want = sum(n.seconds for n in piece.notes) * RATE
        start, v0, v1, end = piece.start, piece.vowel0, piece.vowel1, piece.end
        if v1 <= v0:
            v0, v1 = start, end
        fixed = (v0 - start) + (end - v1)
        vowel = (want - fixed) / (v1 - v0) if v1 > v0 else 0.0
        if vowel < MIN_VOWEL_SHARE:
            # Too short for its consonants: the whole syllable is shrunk alike.
            factor = want / max(1, end - start)
            parts = [(start, end, factor)]
        else:
            parts = [(start, v0, 1.0), (v0, v1, vowel), (v1, end, 1.0)]
        for a, b, factor in parts:
            if b > a:
                regions.append(_Region(a, b, factor, out))
                out = regions[-1].out1
    return regions


def _source_time(regions: Sequence[_Region], out_s: float) -> float:
    """Return the source time (s) heard at `out_s` in the song (s)."""
    out = out_s * RATE
    for region in regions:
        if out <= region.out1 or region is regions[-1]:
            return (region.src0 + (out - region.out0) / region.factor) / RATE
    return 0.0


def _pitch_points(notes: Sequence[Note]) -> list[tuple[float, float]]:
    """Return (song time, Hz) points for the notes, with glides between them and a vibrato."""
    points: list[tuple[float, float]] = []
    at = 0.0
    for note in notes:
        length = note.seconds
        half = min(GLIDE_S / 2, length / 4)
        t = at + half
        while t <= at + length - half + EPS_S:
            into = t - at
            depth = 0.0
            if length > VIBRATO_AFTER_S and into > VIBRATO_AFTER_S / 2:
                depth = VIBRATO_SEMITONES * min(1.0, (into - VIBRATO_AFTER_S / 2) / VIBRATO_RAMP_S)
            wobble = depth * math.sin(2 * math.pi * VIBRATO_HZ * into)
            points.append((t, hertz(note.midi + wobble)))
            t += PITCH_STEP_S
        at += length
    return points


def _sing_phrase(source: NDArray[np.float32], pieces: Sequence[_Piece]) -> NDArray[np.float32]:
    """Return the phrase sung: each vowel stretched to its notes and the voice on their pitch."""
    import parselmouth
    from parselmouth.praat import call

    regions = _regions(pieces)
    notes = [n for piece in pieces for n in piece.notes]
    want = round(sum(n.seconds for n in notes) * RATE)
    # Praat's overlap-add makes at most three times its input's length, and is silent past it:
    # silence after the phrase (played at its own length) gives a long one the room.
    room = max(0, (want - PSOLA_ROOM * source.size) // 2) + TAIL_SAMPLES
    padded = np.concatenate([source, np.zeros(room, dtype=np.float32)])
    sound = parselmouth.Sound(padded.astype(np.float64), sampling_frequency=RATE)
    manipulation = call(sound, "To Manipulation", PITCH_STEP_S, 75, 600)
    durations = call("Create DurationTier", "durations", 0, sound.xmax)
    for region in regions:
        a, b = region.src0 / RATE, region.src1 / RATE
        call(durations, "Add point", a + EPS_S, region.factor)
        call(durations, "Add point", max(a + EPS_S, b - EPS_S), region.factor)
    pitches = call("Create PitchTier", "pitches", 0, sound.xmax)
    for t, hz in _pitch_points(notes):
        call(pitches, "Add point", _source_time(regions, t), hz)
    call([manipulation, pitches], "Replace pitch tier")
    call([manipulation, durations], "Replace duration tier")
    sung = call(manipulation, "Get resynthesis (overlap-add)")
    audio = np.asarray(sung.values[0], dtype=np.float32)
    audio = audio[:want] if audio.size >= want else np.pad(audio, (0, want - audio.size))
    fade = min(int(FADE_S * RATE), audio.size // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    return audio


def _phrases(score: Score) -> list[list[Note]]:
    """Split the notes at the rests: each run of sung notes, and each rest alone."""
    out: list[list[Note]] = []
    for note in score.notes:
        if note.kind is Kind.REST or not out or out[-1][0].kind is Kind.REST:
            out.append([note])
        else:
            out[-1].append(note)
    return out


def speaking_pitch(audio: NDArray[np.float32]) -> float | None:
    """Return the median pitch (MIDI) the voice speaks at in `audio`, None if none is heard."""
    import parselmouth

    if audio.size < MIN_ANALYSIS_SAMPLES:
        return None
    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=RATE)
    track = sound.to_pitch(pitch_floor=MIN_PITCH_HZ, pitch_ceiling=MAX_PITCH_HZ)
    hz = track.selected_array["frequency"]
    voiced = hz[hz > 0]
    if voiced.size == 0:
        return None
    return 69 + 12 * math.log2(float(np.median(voiced)) / 440.0)


def octave_shift(score: Score, voice_midi: float | None) -> int:
    """Return the whole octaves to move the song by so its middle sits where the voice sings."""
    if voice_midi is None:
        return 0
    pitches = sorted(n.midi for n in score.notes if n.midi)
    middle = pitches[len(pitches) // 2]
    return 12 * round((voice_midi + SINGING_ABOVE_SPEECH - middle) / 12)


class PsolaSinger:
    """Sings a score with the speaking voice (see the module's docstring)."""

    name = "speech"

    def __init__(self, speech: Speech) -> None:
        """Sing with `speech`, the voice that says the words."""
        super().__init__()
        self._speech = speech

    def render(self, score: Score) -> tuple[NDArray[np.int16], int]:
        """Return the song as int16 samples at RATE, and RATE."""
        pcm, _ = self.sing(score)
        return pcm, RATE

    def sing(self, score: Score) -> tuple[NDArray[np.int16], Score]:
        """Return the song as int16 samples at RATE, and the score as sung (moved by octaves)."""
        phrases = _phrases(score)
        spoken: list[list[_Word]] = []
        for phrase in phrases:
            if phrase[0].kind is Kind.REST:
                spoken.append([])
                continue
            first, last = phrase[0].word, phrase[-1].word
            spoken.append(_spoken(self._speech, score.words[first : last + 1]))
        said = [w.audio for words in spoken for w in words]
        shift = octave_shift(score, speaking_pitch(np.concatenate(said)) if said else None)
        if shift:
            log.info("singing %+d semitones from the score, to suit the voice", shift)
            score = score.transposed(shift)
            phrases = _phrases(score)
        parts: list[NDArray[np.float32]] = []
        for phrase, words in zip(phrases, spoken, strict=True):
            if not words:
                parts.append(np.zeros(round(phrase[0].seconds * RATE), dtype=np.float32))
                continue
            source = np.concatenate([w.audio for w in words])
            pieces = _trimmed(source, _pieces(phrase, words, phrase[0].word))
            parts.append(_sing_phrase(source[: pieces[-1].end], pieces))
        audio = np.concatenate(parts)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > PEAK:
            audio *= PEAK / peak
        return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16), score
