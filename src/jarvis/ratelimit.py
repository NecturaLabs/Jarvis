"""The cap on side-effecting actions: at most `limit` in any `window_s`, whatever asked for them.

The voice service, the chat window and `jarvis ask` each run their own agent, so the count lives in
one private file in the state folder (`actions.json`, mode 0600 in a 0700 folder, opened without
following a link and locked while it is read and written). The lock is waited for `lock_wait_s` at
most. When that file cannot be used (it is locked too long, or cannot be opened or written), each
process still keeps its own count, so the cap never lifts. A damaged file counts as empty and is
written again. The new count is written over the old one before the file is cut to its length, so a
write that fails leaves the old count. The times are wall-clock seconds, the only clock the
processes share; one later than now (the clock was set back) counts as now, so it holds for one
window and no longer.
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
LOCK_WAIT_S = 2.0
_LOCK_POLL_S = 0.01
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
        lock_wait_s: float = LOCK_WAIT_S,
    ) -> None:
        """Share the count through `path`; with None, count for this process only."""
        super().__init__()
        self._path = path
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._lock_wait_s = lock_wait_s
        self._lock = threading.Lock()
        self._own: list[float] = []

    def full(self) -> bool:
        """Return whether the window has no room for one more action; count nothing."""
        return not self._count(take=False)

    def take(self) -> bool:
        """Count one more action now if the window has room; return whether it had."""
        return self._count(take=True)

    def _count(self, *, take: bool) -> bool:
        """Return whether the window has room, and with `take` count one action in it."""
        with self._lock:
            now = self._clock()
            self._own = self._recent(self._own, now)
            allowed = len(self._own) < self.limit
            if self._path is not None:
                try:
                    allowed = self._shared(self._path, now, take=take)
                except OSError as exc:
                    log.warning("the action count in %s could not be used: %s", self._path, exc)
            if allowed and take:
                self._own.append(now)
            return allowed

    def _recent(self, times: Iterable[float], now: float) -> list[float]:
        """Return the times still inside the window, a time later than now taken as now."""
        return [moment for t in times if now - (moment := min(t, now)) < self.window_s]

    def _shared(self, path: Path, now: float, *, take: bool) -> bool:
        """Return whether the shared file's window has room, and with `take` count one in it.

        Raises:
            OSError: the file cannot be opened, is a link or not a regular file, stays locked
                longer than `lock_wait_s` (`TimeoutError`), or the write fails.
        """
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(path, flags, _PRIVATE)
        try:
            self._lock_file(fd, path)
            status = os.fstat(fd)
            if not stat.S_ISREG(status.st_mode):
                msg = f"{path} is not a regular file"
                raise OSError(msg)
            if stat.S_IMODE(status.st_mode) != _PRIVATE:
                os.fchmod(fd, _PRIVATE)
            read = self._read(fd, path)
            taken = self._recent(read or [], now)
            allowed = len(taken) < self.limit
            if allowed and take:
                taken.append(now)
            if taken != read:  # a place taken, times aged out or set back to now, or damage
                _overwrite(fd, taken)
            return allowed
        finally:
            with contextlib.suppress(OSError):
                os.close(fd)

    def _lock_file(self, fd: int, path: Path) -> None:
        """Lock the open file for this process, waiting `lock_wait_s` at most.

        Raises:
            TimeoutError: another process held the lock all that time.
        """
        deadline = time.monotonic() + self._lock_wait_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    msg = f"{path} stayed locked for {self._lock_wait_s} s"
                    raise TimeoutError(msg) from None
                time.sleep(_LOCK_POLL_S)
            else:
                return

    @staticmethod
    def _read(fd: int, path: Path) -> list[float] | None:
        """Return the times in the shared file, [] for none; None when it is damaged."""
        chunks: list[bytes] = []
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
        raw = b"".join(chunks)
        if not raw:
            return []
        try:
            taken = obj(json.loads(raw)).get("taken")
            values = cast("list[object]", taken) if isinstance(taken, list) else None
            times = (
                None
                if values is None
                else [
                    float(t)
                    for t in values
                    if isinstance(t, int | float) and not isinstance(t, bool)
                ]
            )
        except ValueError, OverflowError, RecursionError:
            times = None
        if times is None:
            log.warning("the action count in %s was damaged; it starts again", path)
        return times


def _overwrite(fd: int, taken: list[float]) -> None:
    """Write the count over the file from its start, then cut the file to that length.

    A write that fails part way leaves the old count, or old bytes behind new ones (damaged: the
    next reader starts again), never an empty file the other processes take for no actions.
    """
    data = json.dumps({"taken": taken}).encode("ascii")
    os.lseek(fd, 0, os.SEEK_SET)
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]
    os.ftruncate(fd, len(data))
