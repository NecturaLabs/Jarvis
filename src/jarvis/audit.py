"""Append-only JSONL audit log of requests, tool calls, decisions and outcomes.

The log holds what was said (the owner's requests and, with PC audio on, other people's), so the
file is private (mode 0600 in a 0700 folder) and bounded: when it passes `max_bytes` it is moved to
`<name>.1` (the older ones to `.2` and on, up to `generations`, the oldest dropped) and started
again. What a write puts in a file is not in it: the agent records such arguments as a SHA-256 and
a length (`Operation.hashed`). Recording never fails the caller: an event that cannot be written
goes to the service log instead.
"""

import contextlib
import fcntl
import itertools
import json
import logging
import os
import stat
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

log = logging.getLogger(__name__)

MAX_BYTES = 5 * 1024 * 1024
GENERATIONS = 5  # older logs kept beside the current one
_PRIVATE = 0o600


def _line(entry: dict[str, object]) -> str:
    """Return one ASCII JSON line for `entry`, whatever its values are."""
    try:
        return json.dumps(entry, default=str, ensure_ascii=True) + "\n"
    except TypeError, ValueError:
        return json.dumps({k: repr(v) for k, v in entry.items()}, ensure_ascii=True) + "\n"


def _is_current(path: Path, status: os.stat_result) -> bool:
    """Whether the open file with `status` is still the one at `path`."""
    try:
        current = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (current.st_dev, current.st_ino) == (status.st_dev, status.st_ino)


class AuditLog:
    """One JSON object per line; each line has a UTC timestamp and an event name."""

    def __init__(
        self,
        path: Path | None,
        max_bytes: int = MAX_BYTES,
        *,
        tags: Mapping[str, str] | None = None,
        generations: int = GENERATIONS,
    ) -> None:
        """Log to `path`; with None, keep the events in `events` instead (tests).

        `tags` go on every event (`{"via": "chat"}`), over any field of the same name. A full log
        is kept as `generations` older files (at least one).
        """
        super().__init__()
        self._path = path
        self._max_bytes = max_bytes
        self._generations = max(1, generations)
        self._tags = dict(tags or {})
        self._lock = threading.Lock()
        self.events: list[dict[str, object]] = []

    def record(self, event: str, **fields: object) -> None:
        """Append one event; a write failure is logged, never raised."""
        entry: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
            **self._tags,
        }
        with self._lock:
            if self._path is None:
                self.events.append(entry)
                return
            line = _line(entry)
            try:
                self._append(self._path, line)
            except OSError:
                log.exception("the audit log could not be written; event: %s", line.strip())

    def _append(self, path: Path, line: str) -> None:
        """Write `line` under an exclusive lock on the log, rotating it first when it is full.

        Other Jarvis processes write the same log, so the lock is `flock` on the open file: the
        size check, the rotation and the write happen while no other writer holds it. A writer
        that waited while the log was rotated finds its file is no longer the log and opens the
        new one, so a full log is rotated once and no event is lost. The new log may have filled
        in the meantime, so its size is checked again before the write.
        """
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        data = line.encode("ascii")
        fd = self._open_locked(path)
        try:
            # Checked again after each rotation: other writers may have filled the new log
            # between this one letting go of the old log and taking the new one.
            while (size := os.fstat(fd).st_size) and size + len(data) > self._max_bytes:
                self._rotate(path)
                old, fd = fd, -1
                with contextlib.suppress(OSError):
                    os.close(old)  # releases the old log's lock once the new one is in place
                fd = self._open_locked(path)
            os.write(fd, data)
        finally:
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)

    @staticmethod
    def _open_locked(path: Path) -> int:
        """Open the log, private, holding its lock; retry while another writer replaces it."""
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
        while True:
            fd = os.open(path, flags, _PRIVATE)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                status = os.fstat(fd)
            except BaseException:
                os.close(fd)
                raise
            if _is_current(path, status):
                if stat.S_IMODE(status.st_mode) != _PRIVATE:
                    os.fchmod(fd, _PRIVATE)
                return fd
            os.close(fd)  # rotated while waiting: the log is a new file now

    def _rotate(self, path: Path) -> None:
        """Move each older log one place on (`.1` to `.2`...), the oldest dropped, then the log."""
        older = [path.with_name(f"{path.name}.{n}") for n in range(1, self._generations + 1)]
        for target, source in itertools.pairwise(reversed(older)):
            with contextlib.suppress(FileNotFoundError):
                source.replace(target)
        path.replace(older[0])
