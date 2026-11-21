"""What the chat window shows of Jarvis's audit log: new events as they are written, one entry each.

The log holds other people's words (Spotify names, PC-audio speech), so every text is shown as
plain characters: nothing in it can draw on the terminal or style the window.
"""

import contextlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, BinaryIO, cast

if TYPE_CHECKING:
    from pathlib import Path

MAX_SHOWN_CHARS = 400
READ_CHUNK = 1 << 16


class AuditTail:
    """Reads the events appended to one audit log since the last read.

    It starts at the log's end, waits for a log that does not exist yet, finishes the old file
    when the log is rotated (moved aside and started again) and reads a truncated log from its
    start again.
    """

    def __init__(self, path: Path) -> None:
        """Follow the log at `path`, from what is written after now."""
        super().__init__()
        self._path = path
        self._file: BinaryIO | None = None
        self._rest = b""
        self._open(at_end=True)

    def _open(self, *, at_end: bool) -> None:
        try:
            file = self._path.open("rb")
        except OSError:
            return
        if at_end:
            file.seek(0, os.SEEK_END)
        self._file = file
        self._rest = b""

    def _drain(self) -> list[bytes]:
        file = self._file
        if file is None:
            return []
        if os.fstat(file.fileno()).st_size < file.tell():
            file.seek(0)
            self._rest = b""
        data = self._rest
        while chunk := file.read(READ_CHUNK):
            data += chunk
        *lines, self._rest = data.split(b"\n")
        return lines

    def _replaced(self) -> bool:
        """Return whether the path now names another file than the one being read."""
        file = self._file
        try:
            now = self._path.stat()
        except OSError:
            return False
        if file is None:
            return True
        held = os.fstat(file.fileno())
        return (now.st_dev, now.st_ino) != (held.st_dev, held.st_ino)

    def read(self) -> list[dict[str, object]]:
        """Return the complete events written since the last read, oldest first."""
        lines = self._drain()
        if self._replaced():
            if self._file is not None:
                with contextlib.suppress(OSError):
                    self._file.close()
            self._file = None
            self._open(at_end=False)
            lines += self._drain()
        return [entry for line in lines if (entry := _entry(line)) is not None]

    def close(self) -> None:
        """Let go of the file."""
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None


def _entry(line: bytes) -> dict[str, object] | None:
    try:
        value: object = json.loads(line.decode("utf-8"))
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    fields = cast("dict[object, object]", value)
    return {str(key): item for key, item in fields.items()}


@dataclass(frozen=True, slots=True)
class Shown:
    """One event as the window shows it.

    `kind` picks its mark and colour; a `nested` event belongs to the step above it (a result, an
    answer); `note` is said after the words, quietly (how a request came in, how long a reply
    took); a `detail` is shown only when the owner asks for every detail.
    """

    time: str
    kind: str
    text: str
    nested: bool = False
    note: str = ""

    @property
    def detail(self) -> bool:
        """Whether the event is shown only with every detail on."""
        return self.kind == "detail"

    @property
    def starts_turn(self) -> bool:
        """Whether the event opens a request: someone asked Jarvis something."""
        return self.kind in {"you", "pc"}


type _Line = tuple[str, str, bool, str]  # kind, text, nested, note


def _plain(value: object) -> str:
    """Return outside text as one line of printable characters, cut to a readable length."""
    text = " ".join("".join(c if c.isprintable() else " " for c in str(value)).split())
    return text if len(text) <= MAX_SHOWN_CHARS else text[: MAX_SHOWN_CHARS - 1] + "…"


def _local_time(stamp: object) -> str:
    try:
        return datetime.fromisoformat(str(stamp)).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return ""


def _action(op: object) -> str:
    return _plain(str(op).removeprefix("spotify_").replace("_", " "))


def _request(entry: dict[str, object]) -> _Line:
    words = _plain(entry.get("text", ""))
    if entry.get("origin") == "pc_audio":
        return "pc", words, False, "PC audio"
    if entry.get("via") in {"chat", "ask"}:
        return "you", words, False, "you, typed"
    return "you", words, False, "you, by voice"


def _tool_call(entry: dict[str, object]) -> _Line:
    action = _action(entry.get("op", ""))
    verdict = entry.get("verdict")
    if verdict == "confirm":
        return "ask", action, False, "needs your yes"
    if verdict in {"deny", "strong"}:
        return "warn", f"{action} refused: {_plain(entry.get('reason', ''))}", False, ""
    return "step", action, False, ""


def _confirmation(entry: dict[str, object]) -> _Line:
    if entry.get("approved") is True:
        return "result", "you said yes", True, ""
    if entry.get("asked") is False:
        return "warn", "no: not asked again after an earlier no", True, ""
    return "warn", "no yes, so nothing was done", True, ""


def _tool_result(entry: dict[str, object]) -> _Line:
    words = _plain(entry.get("text", ""))
    return (
        ("result", words, True, "") if entry.get("ok") else ("warn", f"not done: {words}", True, "")
    )


def _seconds(entry: dict[str, object]) -> str:
    value = entry.get("seconds")
    return f"{value} s" if isinstance(value, int | float) else "?"


def _reply(entry: dict[str, object]) -> _Line:
    return "jarvis", _plain(entry.get("text", "")), False, _seconds(entry)


def _simple(entry: dict[str, object]) -> _Line | None:
    op = _action(entry.get("op", ""))
    lines: dict[str, _Line] = {
        "shortcut": ("step", op, False, "without the model"),
        "tool_failed": ("warn", f"{op} failed: {_plain(entry.get('error', ''))}", True, ""),
        "tool_uncertain": ("warn", f"{op}: not sure it went through", True, ""),
        "tool_rejected": ("warn", f"{op} rejected: {_plain(entry.get('reason', ''))}", False, ""),
        "tool_denied": ("warn", f"{op} refused: {_plain(entry.get('reason', ''))}", False, ""),
        "tool_repeated": ("detail", f"{op} already done in this request", True, ""),
        "tool_budget": ("warn", "too many actions in one request", True, ""),
        "tool_duplicates_dropped": ("detail", "dropped copies of the same call", True, ""),
        "model_call": ("detail", f"model answered in {_seconds(entry)}", True, ""),
        "model_error": ("error", f"the model failed: {_plain(entry.get('error', ''))}", False, ""),
        "claimed_without_action": ("detail", "the model claimed an action; asked again", True, ""),
        "claim_replaced": ("warn", "the model claimed an action that was not done", True, ""),
        "pc_audio_permission": (
            "result",
            f"PC audio may ask again for {_seconds(entry)}",
            True,
            "",
        ),
    }
    return lines.get(str(entry.get("event")))


_EVENTS = {
    "request": _request,
    "tool_call": _tool_call,
    "confirmation": _confirmation,
    "tool_result": _tool_result,
    "reply": _reply,
}


def describe(entry: dict[str, object]) -> Shown | None:
    """Return how the window shows one audit event, or None for an entry with no event name."""
    event = entry.get("event")
    if not isinstance(event, str) or not event:
        return None
    render = _EVENTS.get(event)
    line = (render(entry) if render else _simple(entry)) or ("detail", _plain(event), True, "")
    kind, text, nested, note = line
    return Shown(_local_time(entry.get("ts")), kind, text, nested=nested, note=note)
