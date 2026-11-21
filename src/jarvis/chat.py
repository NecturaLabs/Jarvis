"""The chat window (`jarvis chat`): type requests and watch everything Jarvis does, live.

Typed requests go to an agent of the window's own, as with `jarvis ask`; what the voice service
and that agent do is read from the shared audit log, so both show in one place. A question the
agent asks is answered by typing, and anything but a clear yes is a no, as by voice.
"""

import contextlib
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Protocol, override

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog, Static

from jarvis.activity import Shown, describe
from jarvis.agent import Hooks
from jarvis.policy import is_yes

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


POLL_S = 0.5
STATUS_S = 10.0
HELP = (
    "Type a request as you would say it. /clear empties this window, /help shows this, "
    "/quit leaves (so does Ctrl+Q)."
)
STYLES = {
    "you": "bold",
    "pc": "magenta",
    "jarvis": "green",
    "step": "cyan",
    "warn": "yellow",
    "error": "bold red",
    "detail": "dim",
    "question": "bold yellow",
    "note": "italic",
}


class Tail(Protocol):
    """The audit events written since the last read."""

    def read(self) -> list[dict[str, object]]:
        """Return the new events, oldest first."""
        ...

    def close(self) -> None:
        """Let go of the log."""
        ...


type Ask = Callable[[str, Hooks], str]


@dataclass(slots=True)
class _Question:
    text: str
    answered: threading.Event = field(default_factory=threading.Event)
    answer: bool = False


class ChatApp(App[None]):
    """Shows Jarvis's activity and takes typed requests, one at a time."""

    TITLE = "Jarvis"
    CSS = """
    #activity { height: 1fr; border: none; padding: 0 1; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    Input { dock: bottom; }
    """
    BINDINGS: ClassVar = [("ctrl+q", "quit", "Quit")]

    def __init__(  # ruff: ignore[too-many-arguments] - collaborators and keyword-only settings
        self,
        tail: Tail,
        ask: Ask | None,
        *,
        unavailable: str = "",
        status: Callable[[], str] = lambda: "",
        names: Sequence[str] = (),
        poll_s: float = POLL_S,
        status_s: float = STATUS_S,
    ) -> None:
        """Show what `tail` reads and send typed requests to `ask`.

        Without `ask`, typing says why (`unavailable`). `status` is the line under the activity,
        refreshed every `status_s`; `names` are what the assistant is called ("yes, Jarvis").
        """
        super().__init__()
        self._tail = tail
        self._ask = ask
        self._unavailable = unavailable
        self._status = status
        self._names = tuple(names)
        self._poll_s = poll_s
        self._status_s = status_s
        self._lock = threading.Lock()
        self._busy = False
        self._closing = False
        self._question: _Question | None = None

    @override
    def compose(self) -> ComposeResult:
        """Lay out the activity, the status line and the input."""
        yield RichLog(id="activity", wrap=True, markup=False, highlight=False)
        yield Static("", id="status", markup=False)
        yield Input(placeholder="Type a request, or /help")

    def on_mount(self) -> None:
        """Start reading the log and the status, with the cursor in the input."""
        self.query_one(Input).focus()
        self._note(HELP)
        self.set_interval(self._poll_s, self._poll)
        self.set_interval(self._status_s, self._refresh_status)
        self._refresh_status()

    def on_unmount(self) -> None:
        """Answer a waiting question with no and let go of the log."""
        with self._lock:
            self._closing = True
            question, self._question = self._question, None
        if question is not None:
            question.answered.set()
        self._tail.close()

    def _write(self, shown: Shown) -> None:
        line = Text()
        if shown.time:
            line.append(f"{shown.time}  ", style="dim")
        line.append(shown.text, style=STYLES.get(shown.kind, ""))
        self.query_one("#activity", RichLog).write(line)

    def _note(self, text: str, kind: str = "note") -> None:
        self._write(Shown("", kind, text))

    def _poll(self) -> None:
        for entry in self._tail.read():
            shown = describe(entry)
            if shown is not None:
                self._write(shown)

    def _refresh_status(self) -> None:
        self.run_worker(self._read_status, thread=True, group="status", exclusive=True)

    def _read_status(self) -> None:
        try:
            line = self._status()
        except Exception as exc:  # ruff: ignore[blind-except] - a status that fails says so
            line = f"status unknown: {type(exc).__name__}"
        self.call_from_thread(self.query_one("#status", Static).update, line)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Answer a waiting question, run a command or send a request."""
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        with self._lock:
            question, self._question = self._question, None
        if question is not None:
            question.answer = is_yes(text, self._names)
            self._note(f"You: {text}", "you")
            question.answered.set()
            return
        if text.startswith("/"):
            self._command(text)
            return
        self._request(text)

    def _command(self, text: str) -> None:
        match text.lower():
            case "/quit" | "/exit":
                self.exit()
            case "/clear":
                self.query_one("#activity", RichLog).clear()
            case "/help":
                self._note(HELP)
            case _:
                self._note(f"No command {text}. {HELP}", "warn")

    def _request(self, text: str) -> None:
        if self._ask is None:
            self._note(f"Typed requests are off: {self._unavailable}", "warn")
            return
        with self._lock:
            if self._busy:
                self._note("I'm still working on the last request.", "warn")
                return
            self._busy = True
        self.run_worker(lambda: self._handle(text), thread=True, group="request")

    def _handle(self, text: str) -> None:
        ask = self._ask
        hooks = Hooks(
            confirm=self._confirm,
            waking=lambda: self._from_thread("Waking the model; the first answer takes longer."),
            still_waiting=lambda: self._from_thread("Still loading the model."),
            cancelled=lambda: self._closing,
        )
        try:
            if ask is not None:
                ask(text, hooks)
        except Exception as exc:  # ruff: ignore[blind-except] - shown in the window, never raised
            self._from_thread(f"Something went wrong: {type(exc).__name__}", "error")
        finally:
            with self._lock:
                self._busy = False

    def _from_thread(self, text: str, kind: str = "note") -> None:
        with contextlib.suppress(RuntimeError):  # the window has closed
            self.call_from_thread(self._note, text, kind)

    def _confirm(self, text: str) -> bool:
        """Ask the owner in the window and wait for the typed answer (no when it closes)."""
        question = _Question(text)
        with self._lock:
            if self._closing:
                return False
            self._question = question
        self._from_thread(f"{text} (type yes or no)", "question")
        question.answered.wait()
        return question.answer
