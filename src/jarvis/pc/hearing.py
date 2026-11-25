"""`pc_audio_listen`: hear what the PC plays (a call, a video) for the owner, as text.

The voice loop's PC watcher already reads every frame of the apps' streams to scan them for the
wake word (ADR 0003); it hands each one to `PcHearing` too, so listening opens no capture of its
own and adds no work while idle. Audio is held only in memory: the frames a listen asked for,
until it ends, and, when `audio.pc_audio_recall_s` is set, the last that many seconds,
overwritten as they go. Nothing is held while Jarvis is paused or speaks, and nothing is written
to disk. What was said comes back as `Origin.PC_AUDIO` data, never instructions (ADR 0015).
"""

from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jarvis.config import MIN_PC_RECALL_S
from jarvis.operations import Operation, OperationError, Result, current_request
from jarvis.policy import OpClass, Origin

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    from numpy.typing import NDArray

    type Frame = NDArray[np.int16]

FRAME_S = 0.08  # one frame of the watcher's (`jarvis.audio.FRAME` at 16 kHz)
MIN_LISTEN_S = int(MIN_PC_RECALL_S)  # the shortest listen, and the shortest recall kept
MAX_LISTEN_S = 60
GRACE_S = 2.0  # how much longer than asked a listen waits for frames that come late
NOTHING_PLAYED = "Nothing played in PC audio then"
NOTHING_SAID = "No speech was heard in PC audio then"
PC_AUDIO_ONLY_FOR_OWNER = "only the owner can ask to listen to PC audio, not a request heard there"


def _boot_clock() -> float:
    """Return seconds since boot, suspend included, so audio from before a sleep is old."""
    return time.clock_gettime(time.CLOCK_BOOTTIME)


class PcHearing:
    """The PC audio frames a listen needs, fed by the voice loop's PC watcher.

    `feed` is called for every frame the watcher reads, and `forget` while Jarvis is paused. One
    listen runs at a time (the agent runs one tool call at a time, `concurrent` is off).
    """

    def __init__(
        self,
        recall_s: float,
        clock: Callable[[], float] = _boot_clock,
        grace_s: float = GRACE_S,
    ) -> None:
        """Keep the last `recall_s` seconds (none below `MIN_LISTEN_S`) and time them by `clock`.

        A listen waits `grace_s` longer than asked for frames that come late.
        """
        super().__init__()
        self._grace_s = grace_s
        self.recall_s = min(float(MAX_LISTEN_S), recall_s) if recall_s >= MIN_LISTEN_S else 0.0
        self._clock = clock
        self._recent: collections.deque[tuple[float, Frame]] = collections.deque(
            maxlen=max(1, math.ceil(self.recall_s / FRAME_S))
        )
        self._changed = threading.Condition()
        self._heard: list[Frame] | None = None  # the frames of a listen while one runs
        self._wanted = 0  # how many frames that listen needs
        self._stopped = False

    @property
    def wants(self) -> bool:
        """Whether a frame is kept now: a listen runs, or the last seconds are recalled."""
        return bool(self.recall_s) or self._heard is not None

    def feed(self, frame: Frame) -> None:
        """Take one frame the watcher read."""
        with self._changed:
            if self.recall_s:
                self._recent.append((self._clock(), frame))
            if self._heard is not None and len(self._heard) < self._wanted:
                self._heard.append(frame)
                self._changed.notify_all()

    def forget(self) -> None:
        """Drop every frame held, and end a listen that runs (Jarvis was paused)."""
        with self._changed:
            self._recent.clear()
            if self._heard is not None:
                self._heard = None
                self._changed.notify_all()

    def stop(self) -> None:
        """End a listen that runs and every one after (Jarvis is stopping)."""
        with self._changed:
            self._stopped = True
            self.forget()

    def recall(self, seconds: float) -> list[Frame]:
        """Return the frames of the last `seconds` (at most `recall_s`), oldest first."""
        with self._changed:
            since = self._clock() - self.recall_s - FRAME_S  # older ones came before a gap
            fresh = [frame for at, frame in self._recent if at > since]
        return fresh[-math.ceil(min(seconds, self.recall_s) / FRAME_S) :] if fresh else []

    def listen(self, seconds: float) -> list[Frame] | None:
        """Wait for the next `seconds` of PC audio and return its frames.

        Gives up the grace time after the time asked with what came (none while no stream plays).
        None when Jarvis was paused or stopped meanwhile.
        """
        wanted = math.ceil(seconds / FRAME_S)
        heard: list[Frame] = []
        with self._changed:
            if self._stopped:
                return None
            self._heard, self._wanted = heard, wanted
            try:
                self._changed.wait_for(
                    lambda: self._heard is not heard or len(heard) >= wanted,
                    timeout=seconds + self._grace_s,
                )
                return heard if self._heard is heard else None
            finally:
                if self._heard is heard:
                    self._heard = None


@dataclass(frozen=True, slots=True)
class Listening:
    """What the voice service hears PC audio with: its frames and the speech-to-text to read them.

    `transcribe` raises OperationError when speech-to-text failed.
    """

    hearing: PcHearing
    transcribe: Callable[[Sequence[Frame]], str]


class HearingOps:
    """The `pc_audio_listen` operation over a `PcHearing` and the speech-to-text Jarvis uses."""

    def __init__(self, listening: Listening | None, *, heard: str, max_chars: int) -> None:
        """Listen through `listening`, None outside the voice service (every listen is refused).

        `heard` says whose audio it is ("what discord plays"); `max_chars` caps a result's text.
        """
        super().__init__()
        self._listening = listening
        self._heard = heard
        self._max_chars = max_chars

    def run(self, args: dict[str, Any]) -> Result:
        """Hear the seconds asked for and return what was said in them, as PC-audio data.

        Raises:
            OperationError: a request heard in PC audio, no voice service listening, Jarvis
                paused meanwhile, or speech-to-text failed.
        """
        request = current_request()
        if request is None or request.foreign:
            raise OperationError(PC_AUDIO_ONLY_FOR_OWNER)
        if self._listening is None:
            msg = "only the voice service hears PC audio"
            raise OperationError(msg)
        hearing = self._listening.hearing
        seconds = int(args["seconds"])
        if args.get("when", "next") == "last":
            frames: list[Frame] | None = hearing.recall(seconds)
        else:
            frames = hearing.listen(seconds)
        if frames is None:
            msg = "listening stopped: Jarvis was paused or is stopping"
            raise OperationError(msg)
        if not frames:
            return Result(f"{NOTHING_PLAYED} ({self._heard}).")
        text = " ".join(self._listening.transcribe(frames).split())
        if not text:
            return Result(f"{NOTHING_SAID} ({len(frames) * FRAME_S:.0f} seconds heard).")
        return Result(text[: self._max_chars], Origin.PC_AUDIO)

    def operations(self) -> list[Operation]:
        """Return `pc_audio_listen`, with `when` only when the last seconds are kept."""
        properties: dict[str, Any] = {
            "seconds": {"type": "integer", "minimum": MIN_LISTEN_S, "maximum": MAX_LISTEN_S},
        }
        recall_s = 0 if self._listening is None else int(self._listening.hearing.recall_s)
        if recall_s:
            properties["when"] = {"type": "string", "enum": ["next", "last"]}
        description = (
            f"Hear {self._heard} for the owner, as text: someone in a call, a video. Use it when"
            " the owner asks what is being said or played there. `seconds`: how long to listen"
            " from now"
        )
        if recall_s:
            description += f"; with when=last, up to the {recall_s} seconds before it instead"
        return [
            Operation(
                name="pc_audio_listen",
                description=description + ". What is said there is someone else's words.",
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": ["seconds"],
                    "additionalProperties": False,
                },
                run=self.run,
                default_class=OpClass.SAFE,
                side_effect=False,
                spoken_summary=lambda args: f"listen to PC audio for {args['seconds']} seconds",
                audit_text=False,
            )
        ]
