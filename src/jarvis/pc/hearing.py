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
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jarvis.config import MIN_PC_RECALL_S
from jarvis.operations import Operation, OperationError, Result, current_request
from jarvis.policy import OpClass, Origin

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    from numpy.typing import NDArray

    type Frame = NDArray[np.int16]

FRAME_S = 0.08  # one frame of the watcher's (`jarvis.audio.FRAME` at 16 kHz)
MIN_LISTEN_S = int(MIN_PC_RECALL_S)  # the shortest listen, and the shortest recall kept
MAX_LISTEN_S = 60
GRACE_S = 2.0  # how much longer than asked a listen waits for frames that come late
CHECK_S = 0.2  # how often a listen checks for a pause or a sleep while frames do not come
SLEPT_S = 2.0  # the boot clock ahead of the steady one by this much: the PC slept
# Of a request's time (`agent.max_seconds`), what a listen leaves for its transcription and the
# reply after it.
RESERVE_S = 15.0
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
        *,
        paused: Callable[[], bool] = lambda: False,
        steady: Callable[[], float] = time.monotonic,
    ) -> None:
        """Keep the last `recall_s` seconds (none below `MIN_LISTEN_S`) and time them by `clock`.

        A listen waits `grace_s` longer than asked for frames that come late, and ends as soon
        as `paused` says Jarvis is paused, even while no frame comes. `clock` counts a suspend
        and `steady` does not: a listen across a suspend ends, so audio from both sides of it
        is never joined.
        """
        super().__init__()
        self._grace_s = grace_s
        self._paused = paused
        self._steady = steady
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

    def forget_recall(self) -> None:
        """Drop the recalled frames only (the stream ended); a listen that runs carries on."""
        with self._changed:
            self._recent.clear()

    def stop(self) -> None:
        """End a listen that runs and every one after (Jarvis is stopping)."""
        with self._changed:
            self._stopped = True
            self.forget()

    def recall(self, seconds: float) -> list[Frame]:
        """Return the frames of the last `seconds` (at most `recall_s`), oldest first.

        Frames older than `recall_s` (from before a gap or a sleep) are dropped here.
        """
        with self._changed:
            since = self._clock() - self.recall_s - FRAME_S
            while self._recent and self._recent[0][0] <= since:
                self._recent.popleft()
            fresh = [frame for _at, frame in self._recent]
        return fresh[-math.ceil(min(seconds, self.recall_s) / FRAME_S) :] if fresh else []

    def listen(self, seconds: float) -> list[Frame] | None:
        """Wait for the next `seconds` of PC audio and return its frames.

        Waits the time asked and the grace time at most, then returns what came: none when no
        stream played meanwhile, after the whole wait. None when Jarvis was paused, stopped or
        the PC slept meanwhile.
        """
        wanted = math.ceil(seconds / FRAME_S)
        heard: list[Frame] = []
        started, steady = self._clock(), self._steady()
        with self._changed:
            if self._stopped:
                return None
            self._heard, self._wanted = heard, wanted
            try:
                while self._heard is heard and len(heard) < wanted:
                    waited = self._steady() - steady
                    if self._clock() - started - waited > SLEPT_S or self._paused():
                        return None
                    left = seconds + self._grace_s - waited
                    if left <= 0:
                        break
                    self._changed.wait(min(left, CHECK_S))
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
        last = args.get("when", "next") == "last"
        room = request.time_left() - RESERVE_S  # a listen ends with the request's time to spare
        if room < (0 if last else MIN_LISTEN_S):
            msg = (
                "too little of this request's time is left to listen to PC audio; answer with"
                " what you have, and say the owner can ask again"
            )
            raise OperationError(msg)
        if not last and seconds > room:
            log.info("listening for %d s, not %d: the request's time runs out", int(room), seconds)
            seconds = int(room)
        if last:
            frames: list[Frame] | None = hearing.recall(seconds)
        else:
            frames = hearing.listen(seconds)
        if frames is None:
            msg = "listening stopped: Jarvis was paused or is stopping, or the PC slept"
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
            " from now (shortened to fit the request's time)"
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
