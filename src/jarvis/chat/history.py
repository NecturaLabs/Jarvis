"""What the owner typed in the chat, kept between sessions so ↑ brings it back.

The file is one JSON string a line, readable by the owner only, and holds the last `size` entries.
A line that is not a JSON string is skipped, so a damaged file loses only that line.
"""

import json
import logging
from typing import TYPE_CHECKING

from jarvis.files import write_atomic

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


class InputHistory:
    """The typed lines, oldest first, saved after each one."""

    def __init__(self, path: Path | None, size: int) -> None:
        """Read the history kept at `path` (none kept with None), up to `size` entries."""
        super().__init__()
        self._path = path
        self._size = size
        self.entries: list[str] = self._load()

    def _load(self) -> list[str]:
        if self._path is None:
            return []
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except (OSError, UnicodeDecodeError) as exc:
            log.warning("the chat history could not be read: %s", exc)
            return []
        entries: list[str] = []
        for line in text.splitlines():
            try:
                value: object = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, str) and value.strip():
                entries.append(value)
        return entries[-self._size :] if self._size else []

    def add(self, text: str) -> None:
        """Keep `text` as the newest entry (once, when it repeats the last) and save."""
        if not text.strip() or self._size <= 0:
            return
        if not self.entries or self.entries[-1] != text:
            self.entries = [*self.entries, text][-self._size :]
        if self._path is None:
            return
        try:
            lines = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in self.entries)
            write_atomic(self._path, lines, mode=0o600)
        except OSError as exc:
            log.warning("the chat history could not be saved: %s", exc)
