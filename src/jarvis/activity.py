"""What the chat window shows of Jarvis's audit log: new events as they are written, one entry each.

The log holds other people's words (Spotify names, PC-audio speech, web pages), so every text is
shown as plain characters: nothing in it can draw on the terminal or style the window. A request
and a reply keep all their words and lines (the window renders a reply as Markdown); a step is one
line, cut to a readable length.
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
            # What was written to the old log after the read above and before it was moved
            # aside is still in it: nothing writes to it once it is replaced.
            lines += self._drain()
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
    def level(self) -> str:
        """The least verbosity that shows it: "replies" (always), "steps" or "everything"."""
        if self.kind == "detail":
            return "everything"
        if self.kind in {"step", "result", "ask"} or (self.kind == "warn" and self.nested):
            return "steps"
        return "replies"

    @property
    def starts_turn(self) -> bool:
        """Whether the event opens a request: someone asked Jarvis something."""
        return self.kind in {"you", "pc"}


type _Line = tuple[str, str, bool, str]  # kind, text, nested, note


def _plain(value: object) -> str:
    """Return outside text as one line of printable characters, cut to a readable length."""
    text = " ".join("".join(c if c.isprintable() else " " for c in str(value)).split())
    return text if len(text) <= MAX_SHOWN_CHARS else text[: MAX_SHOWN_CHARS - 1] + "…"


def full_text(value: object) -> str:
    """Return outside text whole, as printable characters and line breaks only."""
    lines = str(value).replace("\r\n", "\n").replace("\t", "    ").split("\n")
    kept = ["".join(c if c.isprintable() else " " for c in line).rstrip() for line in lines]
    return "\n".join(kept).strip("\n")


def _local_time(stamp: object) -> str:
    try:
        return datetime.fromisoformat(str(stamp)).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return ""


def _action(op: object) -> str:
    words = _plain(str(op).removeprefix("spotify_").replace("_", " "))
    return words[:1].upper() + words[1:]


def _request(entry: dict[str, object]) -> _Line:
    words = full_text(entry.get("text", ""))
    if entry.get("origin") == "pc_audio":
        return "pc", words, False, "PC audio"
    if entry.get("via") in {"chat", "ask"}:
        return "you", words, False, "typed"
    return "you", words, False, "by voice"


MAX_ARGUMENT_CHARS = 60


def _arguments(value: object) -> str:
    """Return the gist of a call's arguments: its first words, as `"words"`."""
    if not isinstance(value, dict):
        return ""
    for item in cast("dict[object, object]", value).values():
        if isinstance(item, str) and item.strip():
            words = _plain(item)
            if len(words) > MAX_ARGUMENT_CHARS:
                words = words[: MAX_ARGUMENT_CHARS - 1] + "…"
            return f'"{words}"'
    return ""


def _tool_call(entry: dict[str, object]) -> _Line:
    action = _action(entry.get("op", ""))
    gist = _arguments(entry.get("args"))
    verdict = entry.get("verdict")
    if verdict == "confirm":
        return "ask", action, False, f"{gist} · needs your yes" if gist else "needs your yes"
    if verdict == "strong":
        if "refused" in entry:
            return (
                "warn",
                f"{action} refused: it needs the confirm key, which only spoken requests have",
                False,
                "",
            )
        return (
            "ask",
            action,
            False,
            f"{gist} · needs your confirm key" if gist else "needs your confirm key",
        )
    if verdict == "deny":
        return "warn", f"{action} refused: {_plain(entry.get('reason', ''))}", False, ""
    return "step", action, False, gist


def _confirmation(entry: dict[str, object]) -> _Line:
    if entry.get("method") == "key":
        if entry.get("approved") is True:
            return "result", "you pressed the confirm key", True, ""
        if entry.get("asked") is False:
            return "warn", "no: not asked again after an earlier no", True, ""
        why = NO_REASON_WORDS.get(str(entry.get("reason", "")), "")
        return "warn", f"no key press{why}, so nothing was done", True, ""
    if entry.get("approved") is True:
        return "result", "you said yes", True, ""
    if entry.get("asked") is False:
        return "warn", "no: not asked again after an earlier no", True, ""
    why = NO_REASON_WORDS.get(str(entry.get("reason", "")), "")
    return "warn", f"no yes{why}, so nothing was done", True, ""


NO_REASON_WORDS = {  # the confirmation's `reason`, as the window says it after "no yes"
    "said_no": ": you answered, not with a yes",
    "not_heard": ": the question was not heard out",
    "key_held_off": ": the confirm key would not arm",
    "no_answer": ": no answer in time",
    "too_short": ": too little speech for a yes",
    "untranscribed": ": the answer could not be transcribed",
    "pc_speech": ": PC audio spoke while you answered",
    "spoken_over": ": you spoke over the question",
    "slept": ": the PC slept with the question open",
    "mic_lost": ": the microphone stopped",
    "closed": ": the window closed",
    "failed": ": asking failed",
}


def _tool_result(entry: dict[str, object]) -> _Line:
    if "text" not in entry and "text_sha256" in entry:  # kept only as a digest (`audit_text`)
        words = f"{entry.get('op', 'a tool')} finished; its output is not kept in the log"
    else:
        words = _plain(entry.get("text", ""))
    if entry.get("ok"):
        return ("result", words, True, "")
    if entry.get("acted"):  # it ran, but did not end cleanly
        return ("warn", f"done, not cleanly: {words}", True, "")
    return ("warn", f"not done: {words}", True, "")


def _seconds(entry: dict[str, object]) -> str:
    value = entry.get("seconds")
    return f"{value} s" if isinstance(value, int | float) else "?"


def _reply(entry: dict[str, object]) -> _Line:
    return "jarvis", full_text(entry.get("text", "")), False, _seconds(entry)


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
        "memory_summarized": (
            "note",
            f"Compacted the conversation: {entry.get('turns', '?')} earlier turns summarized",
            False,
            "",
        ),
        "memory_trimmed": (
            "note",
            f"Forgot the {entry.get('turns', '?')} oldest turns to make room",
            False,
            "",
        ),
        "memory_expired": ("note", "Started a new conversation after a long pause", False, ""),
        "memory_cleared": ("detail", "the conversation was cleared", True, ""),
        "reply_limit": ("detail", "the reply reached its length limit", True, ""),
        "result_cleared": ("detail", "cleared an earlier result to make room", True, ""),
        "spoken_before_tool_call": ("detail", "spoke before a tool call", True, ""),
        "control": (
            "note",
            _CONTROL_LINES.get(
                str(entry.get("command")), f"jarvis {_plain(entry.get('command', ''))}"
            ),
            False,
            "",
        ),
        "wake_ignored": (
            "note",
            "Dropped a request: you started dictating"
            if entry.get("reason") == "dictation started mid-request"
            else "Ignored the wake word: you were dictating",
            False,
            "",
        ),
        "control_refused": (
            "warn",
            "refused the confirm key from a program Jarvis started"
            if entry.get("reason") == "started by Jarvis"
            else "refused a control command from another user",
            False,
            "",
        ),
        "confirm_key": _KEY_LINES.get(
            str(entry.get("outcome")),
            ("warn", f"confirm key: {_plain(entry.get('outcome', ''))}", True, ""),
        ),
        "consent_changed": (
            "note",
            "Questions turned off: Jarvis acts without asking"
            if entry.get("bypass_consent") is True
            else "Questions turned on: Jarvis asks before acting",
            False,
            "",
        ),
        "service_started": (
            "note",
            "The voice service started"
            + (", acting without asking" if entry.get("bypass_consent") is True else ""),
            False,
            "",
        ),
        "pc_audio_permission": (
            "result",
            f"PC audio may ask again for {_seconds(entry)}",
            True,
            "",
        ),
    }
    return lines.get(str(entry.get("event")))


_KEY_LINES: dict[str, _Line] = {
    "pressed": ("detail", "the confirm key was pressed", True, ""),
    "nothing waiting": ("warn", "the confirm key was pressed when nothing was waiting", False, ""),
    "too soon": ("warn", "the confirm key came too soon to answer the question", True, ""),
    "held off": ("warn", "held off: a stray confirm key press came less than 30 s ago", True, ""),
    "busy": ("warn", "another action was already waiting for the confirm key", True, ""),
    "cancelled": ("detail", "the wait for the confirm key was cancelled", True, ""),
    "ended": ("detail", "stopped waiting for the confirm key", True, ""),
}

_CONTROL_LINES = {
    "pause": "Paused: the microphone is let go (jarvis pause)",
    "resume": "Listening again (jarvis resume)",
    "hush": "Hushed (jarvis hush)",
}

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


_LABELS = {
    "web_search": "Searching the web",
    "web_read": "Reading a page",
    "reddit_search": "Searching Reddit",
    "reddit_read": "Reading a Reddit thread",
    "weather": "Checking the weather",
    "time_now": "Checking the time",
    "read_skill": "Reading its notes",
    "os_skill": "Reading Omarchy's guide",
    "no_action": "Writing",
}


def working_label(entry: dict[str, object]) -> str | None:
    """Return what Jarvis is doing after this event, for the working line; None to keep it."""
    event = entry.get("event")
    if event == "tool_call":
        op = str(entry.get("op", ""))
        if op.startswith("spotify_"):
            return "Talking to Spotify"
        return _LABELS.get(op, f"Running {_action(op)}")
    if event in {"tool_result", "tool_failed", "confirmation", "model_call"}:
        return "Thinking"
    if event == "memory_summarized":
        return "Compacting the conversation"
    return None
