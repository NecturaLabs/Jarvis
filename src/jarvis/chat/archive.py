"""Every chat conversation, kept for reading back and for `/resume`.

Each conversation has two files in the archive folder (0700), both 0600: `<id>.memory.json`, the
conversation as `Conversation.save` writes it (its turns, summary and their provenance flags, so a
resumed one still treats outside text as data), and `<id>.json`, its name, when it started and was
last written, and the full log of the window's own events (requests, steps, results, replies, each
with its time). Only the newest `keep` conversations are kept.
"""

import json
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from jarvis.files import write_atomic
from jarvis.jsonutil import obj, objs
from jarvis.secretfile import read_private

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from jarvis.memory import Conversation

log = logging.getLogger(__name__)
FILE_VERSION = 1
MAX_NAME = 60  # characters of a name, given or taken from the first request
MEMORY_SUFFIX = ".memory.json"


@dataclass(frozen=True, slots=True)
class Archived:
    """One kept conversation, as `/resume` lists it."""

    id: str
    name: str
    created: datetime
    updated: datetime
    requests: int
    named: bool  # the owner named it (`/rename`), rather than its first request


def _now() -> datetime:
    return datetime.now(UTC)


def _short(text: str) -> str:
    words = " ".join(text.split())
    return words if len(words) <= MAX_NAME else words[: MAX_NAME - 1].rstrip() + "…"


class ChatArchive:
    """The archive folder: the conversation being written, and the ones kept before it."""

    def __init__(self, folder: Path, keep: int) -> None:
        """Keep at most `keep` conversations in `folder`."""
        super().__init__()
        self._folder = folder
        self._keep = keep
        self.current = self._new_id()
        self._log: list[dict[str, object]] = []
        self._name = ""
        self._named = False
        self._created = _now()

    @staticmethod
    def _new_id() -> str:
        return f"{_now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"

    def _meta(self, conversation_id: str) -> Path:
        return self._folder / f"{conversation_id}.json"

    def _memory(self, conversation_id: str) -> Path:
        return self._folder / f"{conversation_id}{MEMORY_SUFFIX}"

    @property
    def name(self) -> str:
        """The current conversation's name."""
        return self._name or f"{self._created.astimezone():%b %d %H:%M}"

    def record(self, entry: dict[str, object]) -> None:
        """Add one of the window's own events to the current conversation's log."""
        self._log.append(entry)
        if not self._name and entry.get("event") == "request":
            self._name = _short(str(entry.get("text", "")))

    def rename(self, name: str) -> None:
        """Name the current conversation; it keeps the name when it is resumed."""
        self._name = _short(name)
        self._named = True

    def save(self, memory: Conversation) -> None:
        """Write the current conversation, if it holds anything, and drop the oldest past `keep`."""
        if not self._log and not memory.turns and not memory.summary:
            return
        self._folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        data = {
            "version": FILE_VERSION,
            "id": self.current,
            "name": self.name,
            "named": self._named,
            "created": self._created.isoformat(),
            "updated": _now().isoformat(),
            "log": self._log,
        }
        try:
            memory.save(self._memory(self.current))
            write_atomic(
                self._meta(self.current),
                json.dumps(data, ensure_ascii=False, default=str),
                mode=0o600,
            )
        except OSError as exc:
            log.warning("the chat archive could not be written: %s", exc)
            return
        self._prune()

    def start(self) -> None:
        """Begin a new conversation; the one before stays as it was last saved."""
        self.current = self._new_id()
        self._log = []
        self._name = ""
        self._named = False
        self._created = _now()

    def _read(self, conversation_id: str) -> dict[str, object] | None:
        try:
            text = read_private(self._meta(conversation_id))
            data = obj(json.loads(text)) if text is not None else None
        except (OSError, ValueError) as exc:
            log.warning("skipped an archived conversation that could not be read: %s", exc)
            return None
        return data if data is not None and data.get("version") == FILE_VERSION else None

    def listed(self) -> list[Archived]:
        """Return the kept conversations, the most recently written first."""
        if not self._folder.is_dir():
            return []
        kept: list[Archived] = []
        for path in self._folder.glob("*.json"):
            if path.name.endswith(MEMORY_SUFFIX):
                continue
            data = self._read(path.stem)
            if data is None:
                continue
            try:
                created = datetime.fromisoformat(str(data.get("created")))
                updated = datetime.fromisoformat(str(data.get("updated")))
            except ValueError:
                continue
            requests = sum(e.get("event") == "request" for e in objs(data.get("log")))
            kept.append(
                Archived(
                    path.stem,
                    str(data.get("name", "")),
                    created,
                    updated,
                    requests,
                    data.get("named") is True,
                )
            )
        return sorted(kept, key=lambda a: a.updated, reverse=True)

    def resume(self, conversation_id: str, load: Callable[[Path], bool]) -> bool:
        """Continue an archived conversation; return whether `load` took its conversation file.

        `load` (the agent's) reads the conversation, provenance and all. The log, name and start
        carry on, so it stays one conversation in the archive.
        """
        data = self._read(conversation_id)
        if data is None or not load(self._memory(conversation_id)):
            return False
        self.current = conversation_id
        self._log = [dict(e) for e in objs(data.get("log"))]
        self._name = str(data.get("name", ""))
        self._named = data.get("named") is True
        try:
            self._created = datetime.fromisoformat(str(data.get("created")))
        except ValueError:
            self._created = _now()
        return True

    def forget(self) -> int:
        """Delete every kept conversation; return how many there were."""
        gone = 0
        if self._folder.is_dir():
            for path in self._folder.glob("*.json"):
                path.unlink(missing_ok=True)
                gone += not path.name.endswith(MEMORY_SUFFIX)
        self.start()
        return gone

    def _prune(self) -> None:
        for old in self.listed()[self._keep :]:
            self._meta(old.id).unlink(missing_ok=True)
            self._memory(old.id).unlink(missing_ok=True)
