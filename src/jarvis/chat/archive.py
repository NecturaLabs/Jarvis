"""Every chat conversation, kept for reading back and for `/resume`.

Each conversation has three files in the archive folder (0700), all 0600: `<id>.memory.json`, the
conversation as `Conversation.save` writes it (its turns, summary and their provenance flags, so a
resumed one still treats outside text as data); `<id>.log.jsonl`, the full log of the window's own
events (requests, steps, results, replies, each with its time), one line each, added to at every
save and never rewritten; and `<id>.json`, its name, when it started and was last written, and how
many requests it holds, which is all that listing and the cap read. Only the newest `keep`
conversations are kept.

A conversation is written only once something happened in it: a request, or a change to one
already in the archive. A chat that opens with the conversation it remembers and closes again
writes no copy of it.
"""

import json
import logging
import os
import secrets
import threading
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
FILE_VERSION = 2
FIRST_VERSION = 1  # the log inside the meta file: read, and moved to its own file
MAX_NAME = 60  # characters of a name, given or taken from the first request
MEMORY_SUFFIX = ".memory.json"
LOG_SUFFIX = ".log.jsonl"


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


def _lines(entries: list[dict[str, object]]) -> str:
    return "".join(json.dumps(e, ensure_ascii=False, default=str) + "\n" for e in entries)


def _append(path: Path, text: str) -> None:
    """Add `text` to the end of the private file `path`, made 0600 when it is new."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
    with os.fdopen(os.open(path, flags, 0o600), "a", encoding="utf-8") as handle:
        handle.write(text)


class ChatArchive:
    """The archive folder: the conversation being written, and the ones kept before it.

    The window records events and the request's worker may save at close, so every method
    holds the archive's lock.
    """

    def __init__(self, folder: Path, keep: int) -> None:
        """Keep at most `keep` conversations in `folder`."""
        super().__init__()
        self._folder = folder
        self._keep = keep
        self._lock = threading.Lock()
        self.current = self._new_id()
        self._unwritten: list[dict[str, object]] = []  # log entries not yet in its file
        self._requests = 0
        self._name = ""
        self._named = False
        self._created = _now()
        self._kept = False  # the current conversation is in the archive already
        self._changed = False  # something happened in it since it was last written

    @staticmethod
    def _new_id() -> str:
        return f"{_now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"

    def _meta(self, conversation_id: str) -> Path:
        return self._folder / f"{conversation_id}.json"

    def _memory(self, conversation_id: str) -> Path:
        return self._folder / f"{conversation_id}{MEMORY_SUFFIX}"

    def _log(self, conversation_id: str) -> Path:
        return self._folder / f"{conversation_id}{LOG_SUFFIX}"

    @property
    def name(self) -> str:
        """The current conversation's name."""
        return self._name or f"{self._created.astimezone():%b %d %H:%M}"

    @property
    def kept(self) -> bool:
        """Whether the current conversation is in the archive already."""
        return self._kept

    def record(self, entry: dict[str, object]) -> None:
        """Add one of the window's own events to the current conversation's log."""
        with self._lock:
            self._unwritten.append(entry)
            request = entry.get("event") == "request"
            self._requests += request
            self._changed = self._changed or request or self._kept
            if not self._name and request:
                self._name = _short(str(entry.get("text", "")))

    def rename(self, name: str) -> None:
        """Name the current conversation; it keeps the name when it is resumed.

        One not in the archive yet is written under that name once it has a request.
        """
        with self._lock:
            self._name = _short(name)
            self._named = True
            self._changed = self._changed or self._kept

    def touch(self) -> None:
        """Note that the current conversation changed (a request ended): the next save writes it."""
        with self._lock:
            self._changed = True

    def save(self, memory: Conversation) -> None:
        """Write the current conversation if something happened in it, then apply the cap.

        Something happened once a request was recorded in it or it was `touch`ed, or, once it is
        in the archive, any change to it.
        """
        with self._lock:
            if not self._changed or not (self._requests or memory.turns or memory.summary):
                return
            try:
                self._write(memory)
            except OSError as exc:
                log.warning("the chat archive could not be written: %s", exc)
                return
            self._kept, self._changed = True, False
            self._prune()

    def _write(self, memory: Conversation) -> None:
        """Write the conversation, add the new log entries to its log, then its meta data."""
        self._folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        memory.save(self._memory(self.current))
        if self._unwritten:
            _append(self._log(self.current), _lines(self._unwritten))
            self._unwritten = []
        data = {
            "version": FILE_VERSION,
            "id": self.current,
            "name": self.name,
            "named": self._named,
            "created": self._created.isoformat(),
            "updated": _now().isoformat(),
            "requests": self._requests,
        }
        write_atomic(self._meta(self.current), json.dumps(data, ensure_ascii=False), mode=0o600)

    def start(self) -> None:
        """Begin a new conversation; the one before stays as it was last saved."""
        with self._lock:
            self._begin()

    def _begin(self) -> None:
        self.current = self._new_id()
        self._unwritten = []
        self._requests = 0
        self._name = ""
        self._named = False
        self._created = _now()
        self._kept = self._changed = False

    def _read(self, conversation_id: str) -> dict[str, object] | None:
        """Return a conversation's meta data, its log moved out of a file in the first format."""
        try:
            text = read_private(self._meta(conversation_id))
            data = obj(json.loads(text)) if text is not None else None
        except (OSError, ValueError, RecursionError) as exc:
            log.warning("skipped an archived conversation that could not be read: %s", exc)
            return None
        if data is not None and data.get("version") == FIRST_VERSION:
            return self._upgrade(conversation_id, data)
        return data if data is not None and data.get("version") == FILE_VERSION else None

    def _upgrade(self, conversation_id: str, data: dict[str, object]) -> dict[str, object] | None:
        """Move a first-format conversation's log into its own file; return its new meta data."""
        entries = objs(data.get("log"))
        upgraded = {key: value for key, value in data.items() if key != "log"}
        upgraded["version"] = FILE_VERSION
        upgraded["requests"] = sum(e.get("event") == "request" for e in entries)
        try:
            if not self._log(conversation_id).exists():
                _append(self._log(conversation_id), _lines(entries))
            write_atomic(
                self._meta(conversation_id), json.dumps(upgraded, ensure_ascii=False), mode=0o600
            )
        except OSError as exc:
            log.warning("an archived conversation could not be moved to the new format: %s", exc)
            return None
        return upgraded

    def listed(self) -> list[Archived]:
        """Return the kept conversations, the most recently written first."""
        with self._lock:
            return self._listed()

    def _listed(self) -> list[Archived]:
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
            requests = data.get("requests")
            kept.append(
                Archived(
                    path.stem,
                    str(data.get("name", "")),
                    created,
                    updated,
                    requests if isinstance(requests, int) else 0,
                    data.get("named") is True,
                )
            )
        return sorted(kept, key=lambda a: a.updated, reverse=True)

    def resume(self, conversation_id: str, load: Callable[[Path], bool]) -> bool:
        """Continue an archived conversation; return whether `load` took its conversation file.

        `load` (the agent's) reads the conversation, provenance and all. The log, name and start
        carry on, so it stays one conversation in the archive.
        """
        with self._lock:
            data = self._read(conversation_id)
            if data is None or not load(self._memory(conversation_id)):
                return False
            self.current = conversation_id
            self._unwritten = []
            requests = data.get("requests")
            self._requests = requests if isinstance(requests, int) else 0
            self._name = str(data.get("name", ""))
            self._named = data.get("named") is True
            try:
                self._created = datetime.fromisoformat(str(data.get("created")))
            except ValueError:
                self._created = _now()
            self._kept, self._changed = True, False
            return True

    def forget(self) -> int:
        """Delete every kept conversation and begin a new one; return how many there were."""
        with self._lock:
            gone = 0
            if self._folder.is_dir():
                for path in [*self._folder.glob("*.json"), *self._folder.glob(f"*{LOG_SUFFIX}")]:
                    path.unlink(missing_ok=True)
                    gone += path.suffix == ".json" and not path.name.endswith(MEMORY_SUFFIX)
            self._begin()
            return gone

    def _prune(self) -> None:
        for old in self._listed()[self._keep :]:
            for path in (self._meta(old.id), self._memory(old.id), self._log(old.id)):
                path.unlink(missing_ok=True)
