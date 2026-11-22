"""Append-only JSONL audit log of requests, tool calls, decisions and outcomes.

The log holds what was said (the owner's requests and, with PC audio on, other people's), so the
file is private (mode 0600 in a 0700 folder) and bounded: when it passes `max_bytes` it is moved to
`<name>.1` (the older ones to `.2` and on, up to `generations`, the oldest dropped) and started
again. What a write puts in a file is not in it: the agent records such arguments as a SHA-256 and
a length (`Operation.hashed`). Recording never fails the caller: an event that cannot be written
goes to the service log instead.
"""

import contextlib
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
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        data = line.encode("ascii")
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(path, flags, _PRIVATE)
        try:
            status = os.fstat(fd)
            if stat.S_IMODE(status.st_mode) != _PRIVATE:
                os.fchmod(fd, _PRIVATE)
            if status.st_size and status.st_size + len(data) > self._max_bytes:
                os.close(fd)
                fd = -1
                self._rotate(path)
                fd = os.open(path, flags | os.O_EXCL, _PRIVATE)
            os.write(fd, data)
        finally:
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)

    def _rotate(self, path: Path) -> None:
        """Move each older log one place on (`.1` to `.2`...), the oldest dropped, then the log."""
        older = [path.with_name(f"{path.name}.{n}") for n in range(1, self._generations + 1)]
        for target, source in itertools.pairwise(reversed(older)):
            with contextlib.suppress(FileNotFoundError):
                source.replace(target)
        path.replace(older[0])
