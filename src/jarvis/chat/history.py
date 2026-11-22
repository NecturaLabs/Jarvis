"""What the owner typed in the chat, kept between sessions so ↑ brings it back.

The file is one JSON value a line, readable by the owner only, and holds the last `size` entries.
A typed line is a JSON string; one with text pasted into it is an object that also keeps what was
pasted (`{"text": …, "pasted": […]}`), so a long paste brought back with ↑ still counts as pasted.
A line that is neither is skipped, so a damaged file loses only that line.
"""

import json
import logging
from typing import TYPE_CHECKING, cast

from jarvis.files import write_atomic

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

log = logging.getLogger(__name__)


class InputHistory:
    """The typed lines, oldest first, with what was pasted into each, saved after each one."""

    def __init__(self, path: Path | None, size: int) -> None:
        """Read the history kept at `path` (none kept with None), up to `size` entries."""
        super().__init__()
        self._path = path
        self._size = size
        self._kept: list[tuple[str, tuple[str, ...]]] = self._load()

    @property
    def entries(self) -> list[str]:
        """The kept lines, oldest first."""
        return [text for text, _ in self._kept]

    def pasted(self, index: int) -> tuple[str, ...]:
        """Return what was pasted into the entry at `index` (as in `entries`), once per paste."""
        return self._kept[index][1]

    def _load(self) -> list[tuple[str, tuple[str, ...]]]:
        if self._path is None:
            return []
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except (OSError, UnicodeDecodeError) as exc:
            log.warning("the chat history could not be read: %s", exc)
            return []
        kept: list[tuple[str, tuple[str, ...]]] = []
        for line in text.splitlines():
            try:
                value: object = json.loads(line)
            except ValueError:
                continue
            entry = _entry(value)
            if entry is not None and entry[0].strip():
                kept.append(entry)
        return kept[-self._size :] if self._size else []

    def add(self, text: str, pasted: Iterable[str] = ()) -> None:
        """Keep `text` as the newest entry (once, when it repeats the last) and save.

        `pasted` is what was pasted into it. A repeat keeps the more pasted of the two, so a line
        sent once as a paste is never brought back as typed.
        """
        if not text.strip() or self._size <= 0:
            return
        marks = tuple(pasted)
        if self._kept and self._kept[-1][0] == text:
            before = self._kept[-1][1]
            if sum(map(len, before)) >= sum(map(len, marks)):
                marks = before
            self._kept = self._kept[:-1]
        self._kept = [*self._kept, (text, marks)][-self._size :]
        if self._path is None:
            return
        try:
            lines = "".join(
                json.dumps(
                    {"text": line, "pasted": list(marks)} if marks else line, ensure_ascii=False
                )
                + "\n"
                for line, marks in self._kept
            )
            write_atomic(self._path, lines, mode=0o600)
        except OSError as exc:
            log.warning("the chat history could not be saved: %s", exc)


def _entry(value: object) -> tuple[str, tuple[str, ...]] | None:
    """Return one line of the file as (text, pasted), or None when it is neither form."""
    if isinstance(value, str):
        return value, ()
    if not isinstance(value, dict):
        return None
    fields = cast("dict[str, object]", value)
    text, pasted = fields.get("text"), fields.get("pasted", [])
    if not isinstance(text, str) or not isinstance(pasted, list):
        return None
    marks = cast("list[object]", pasted)
    if not all(isinstance(mark, str) for mark in marks):
        return None
    return text, tuple(cast("list[str]", marks))
