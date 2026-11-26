"""The `sing` operation: the model writes a song as notes, Jarvis checks it and sings it.

The song is the model's own score (`jarvis.sing.score`), never a stored tune or phrase. It is
checked before any sound is made, rendered by the configured singer, and handed back as the
result's `sound`, which the voice plays like a reply: after the words before it, spoken over by
the wake word, ended by `jarvis hush` and silent while Jarvis is paused. The same signals stop a
song while it is still being made, and the voice listens for the wake word meanwhile.
"""

import logging
import threading
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Protocol

from jarvis.operations import (
    Operation,
    OperationError,
    OperationStoppedError,
    Result,
    Sound,
    current_request,
)
from jarvis.sing.budget import Budget, OutOfTimeError, SongStoppedError
from jarvis.sing.score import (
    DEFAULT_TEMPO,
    MAX_NOTES,
    MAX_SCORE_CHARS,
    MAX_SECONDS,
    MAX_TEMPO,
    MIN_TEMPO,
    Score,
    ScoreError,
    parse,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray

    from jarvis.operations import Request

log = logging.getLogger(__name__)

MAX_SUMMARY_WORDS = 6


class Singer(Protocol):
    """Turns a checked score into sound."""

    name: str

    def render(self, score: Score, budget: Budget, /) -> tuple[NDArray[np.int16], int]:
        """Return the song as int16 mono samples and their sample rate.

        It checks `budget` between its pieces of work and stops as soon as it says so.

        Raises:
            SongStoppedError: the song was stopped while it was being made.
            OutOfTimeError: the request ran out of time first.
        """
        ...


STOPPED = "the song was stopped before it played (the owner spoke over Jarvis, hushed or paused it)"


def _budget(request: Request | None) -> Budget:
    """Return what a song made in `request` may spend: it stops once nothing more can be heard."""
    if request is None:
        return Budget()
    return Budget(
        stopped=lambda: request.cancelled() or request.hushed(), time_left=request.time_left
    )


def _summary(args: dict[str, Any]) -> str:
    """Return the confirmation's words: sing, and how the song starts."""
    try:
        words = parse(str(args.get("notes", "")), int(args.get("tempo_bpm", DEFAULT_TEMPO))).words
    except ScoreError, ValueError:
        return "sing a song"
    start = " ".join(words[:MAX_SUMMARY_WORDS])
    return f"sing {start}{'…' if len(words) > MAX_SUMMARY_WORDS else ''}"


DESCRIPTION = (
    "Sing to the owner when they ask you to sing, however they put it. You write the song "
    "yourself as notes: a melody you know or make up, and its words, at most "
    f"{MAX_SECONDS:g} seconds and {MAX_NOTES} notes. It plays in your voice after what you have "
    "said; afterwards say a few words at most, and never the lyrics again."
)
NOTES = (
    "The notes in order, separated by commas, each 'TEXT PITCH BEATS'. TEXT is one word, or one "
    "syllable ending in '-' when its word goes on in the next note (hap- G4 1, py E4 1). '~' as "
    "TEXT holds the sound before it on to a new pitch (day C5 1, ~ B4 1). A rest is '_ BEATS'. "
    "PITCH is a note name with its octave, C2 to C6 (C4 is middle C; F#4, Bb3). BEATS is a "
    "length in beats, 0.125 to 8 (0.5, 1, 1.5, 1/3). Write numbers in words."
)


class SingOps:
    """The `sing` operation over a singer that is made on the first song and kept."""

    def __init__(self, singer: Callable[[], Singer]) -> None:
        """Sing with what `singer` returns; it raises OperationError when no singer can run."""
        super().__init__()
        self._make = singer
        self._singer: Singer | None = None
        self._lock = threading.Lock()

    def _get(self) -> Singer:
        with self._lock:
            if self._singer is None:
                self._singer = self._make()
            return self._singer

    def _sing(self, args: dict[str, Any]) -> Result:
        """Check the score, render it and hand the song back to be played.

        Raises:
            OperationError: the score cannot be sung as written (the model can fix it), no
                singer can run, or the request ran out of time.
            OperationStoppedError: the song was stopped before it could be played.
        """
        try:
            score = parse(str(args["notes"]), int(args.get("tempo_bpm", DEFAULT_TEMPO)))
        except ScoreError as exc:
            msg = f"Nothing was sung: {exc}."
            raise OperationError(msg) from exc
        request = current_request()
        budget = _budget(request)
        if budget.stopped():
            raise OperationStoppedError(STOPPED)
        singer = self._get()
        try:
            with request.making_sound() if request is not None else nullcontext():
                pcm, rate = singer.render(score, budget)
        except SongStoppedError as exc:
            raise OperationStoppedError(STOPPED) from exc
        except OutOfTimeError as exc:
            msg = "I couldn't sing that: the request ran out of time while the song was made."
            raise OperationError(msg) from exc
        except OperationError:
            raise
        except Exception as exc:
            log.exception("the %s singer failed", singer.name)
            msg = "I couldn't sing that: the singing voice failed."
            raise OperationError(msg) from exc
        if budget.stopped():  # stopped as it was finished: it is not played
            raise OperationStoppedError(STOPPED)
        seconds = pcm.size / rate
        return Result(
            f"Singing now ({len(score.notes)} notes, {seconds:.0f} seconds): it plays after what"
            " you have said. Do not repeat the lyrics; reply in a few words or not at all.",
            sound=Sound(pcm.tobytes(), rate, score.lyrics),
        )

    def operations(self) -> list[Operation]:
        """Return the `sing` operation."""
        return [
            Operation(
                "sing",
                DESCRIPTION,
                {
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_SCORE_CHARS,
                            "description": NOTES,
                        },
                        "tempo_bpm": {
                            "type": "integer",
                            "minimum": MIN_TEMPO,
                            "maximum": MAX_TEMPO,
                            "default": DEFAULT_TEMPO,
                            "description": "Beats a minute.",
                        },
                    },
                    "required": ["notes"],
                    "additionalProperties": False,
                },
                self._sing,
                spoken_summary=_summary,
            )
        ]
