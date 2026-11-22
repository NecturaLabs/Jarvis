"""The cap on side-effecting actions: at most `limit` in any `window_s`, whatever asked for them.

The voice service, the chat window and `jarvis ask` each run their own agent, so the count lives in
one private file in the state folder (`actions.json`, mode 0600 in a 0700 folder, opened without
following a link and locked while it is read and written). When that file cannot be used, each
process still keeps its own count, so the cap never lifts. The times are wall-clock seconds, the
only clock the processes share; one later than now (the clock was set back) counts as now, so it
holds for one window and no longer.
"""

import contextlib
import fcntl
import json
import logging
import os
import stat
import threading
import time
from typing import TYPE_CHECKING, cast

from jarvis.jsonutil import obj

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

log = logging.getLogger(__name__)

MAX_ACTIONS = 30
WINDOW_S = 600.0
_PRIVATE = 0o600


class ActionRate:
    """Counts the side-effecting actions taken in the last `window_s` seconds."""

    def __init__(
        self,
        path: Path | None,
        *,
        limit: int = MAX_ACTIONS,
        window_s: float = WINDOW_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Share the count through `path`; with None, count for this process only."""
        super().__init__()
        self._path = path
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._lock = threading.Lock()
        self._own: list[float] = []

    def take(self) -> bool:
        """Count one more action now if the window has room; return whether it had."""
        with self._lock:
            now = self._clock()
            self._own = self._recent(self._own, now)
            allowed = len(self._own) < self.limit
            if self._path is not None:
                try:
                    allowed = self._take_shared(self._path, now)
                except OSError as exc:
                    log.warning("the action count in %s could not be used: %s", self._path, exc)
            if allowed:
                self._own.append(now)
            return allowed

    def _recent(self, times: Iterable[float], now: float) -> list[float]:
        """Return the times still inside the window, a time later than now taken as now."""
        return [moment for t in times if now - (moment := min(t, now)) < self.window_s]

    def _take_shared(self, path: Path, now: float) -> bool:
        """Count one more action in the shared file if its window has room.

        Raises:
            OSError: the file cannot be opened, is a link or not a regular file, or the write fails.
        """
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(path, flags, _PRIVATE)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            status = os.fstat(fd)
            if not stat.S_ISREG(status.st_mode):
                msg = f"{path} is not a regular file"
                raise OSError(msg)
            if stat.S_IMODE(status.st_mode) != _PRIVATE:
                os.fchmod(fd, _PRIVATE)
            taken = self._recent(self._read(fd, path), now)
            allowed = len(taken) < self.limit
            if allowed:
                taken.append(now)
            data = json.dumps({"taken": taken}).encode("ascii")
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, data)
            return allowed
        finally:
            with contextlib.suppress(OSError):
                os.close(fd)

    @staticmethod
    def _read(fd: int, path: Path) -> list[float]:
        """Return the times in the shared file; a damaged file counts as empty."""
        chunks: list[bytes] = []
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
        raw = b"".join(chunks)
        if not raw:
            return []
        try:
            taken = obj(json.loads(raw)).get("taken")
        except ValueError:
            taken = None
        if not isinstance(taken, list):
            log.warning("the action count in %s was damaged; it starts again", path)
            return []
        values = cast("list[object]", taken)
        return [float(t) for t in values if isinstance(t, int | float) and not isinstance(t, bool)]
