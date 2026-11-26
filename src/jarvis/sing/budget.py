"""What making one song may spend: when to stop, and how much of the request's time is left."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class SongStoppedError(Exception):
    """The song was stopped while it was being made (hushed, spoken over, paused or stopping)."""


class OutOfTimeError(Exception):
    """The request's time ran out while the song was being made."""


@dataclass(frozen=True, slots=True)
class Budget:
    """What a song may spend, which a singer checks between its pieces of work.

    `stopped` turns true when nothing of the song may be heard any more, and `time_left` is the
    request's seconds left.
    """

    stopped: Callable[[], bool] = lambda: False
    time_left: Callable[[], float] = lambda: math.inf

    def check(self) -> None:
        """Return only while the song may still be worked on.

        Raises:
            SongStoppedError: it was stopped.
            OutOfTimeError: the request has no time left.
        """
        if self.stopped():
            raise SongStoppedError
        if self.time_left() <= 0:
            raise OutOfTimeError
