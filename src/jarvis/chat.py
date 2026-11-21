"""The chat window (`jarvis chat`): type requests and watch everything Jarvis does, live.

Typed requests go to an agent of the window's own, as with `jarvis ask`; what the voice service
and that agent do is read from the shared audit log, so both show in one place, one request at a
time: what was asked, each step Jarvis took (`●`) with its result under it (`⎿`), and the
reply. A question the agent asks opens a yes-or-no choice above the input; anything but a chosen
yes is a no, as by voice. The layout follows Claude Code's terminal interface, and the colours
are the terminal's own, so the window takes the Omarchy theme.
"""

import contextlib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Protocol, override

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from jarvis.activity import Shown, describe
from jarvis.agent import Hooks

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import RenderableType
    from textual.timer import Timer


POLL_S = 0.5
STATUS_S = 10.0
TICK_S = 0.12
ARM_S = 0.4  # a choice made sooner than this after a question appears was meant for something else
MAX_KEPT = 2000  # entries kept for redrawing the transcript
SPINNER = "·✢✳✶✻✽✻✶✳✢"
DOT = "\N{BLACK CIRCLE}"  # a step or a reply
CHEVRON = "\N{HEAVY RIGHT-POINTING ANGLE QUOTATION MARK ORNAMENT}"  # what was asked


class Command(NamedTuple):
    """A slash command and what it does, as the command menu lists it."""

    name: str
    about: str


COMMANDS = (
    Command("/help", "what this window can do"),
    Command("/details", "show or hide every step (also Ctrl+O)"),
    Command("/status", "check the voice service and the model now"),
    Command("/clear", "empty the window (also Ctrl+L)"),
    Command("/quit", "leave (also Ctrl+Q)"),
)
HELP = (
    "Type a request as you would say it and press Enter. What you say to Jarvis out loud shows "
    "here too. Esc stops a request, ↑ and ↓ bring back what you typed, / lists the commands and "
    "? the shortcuts."
)
SHORTCUTS = (
    ("enter", "send"),
    ("esc", "stop the request, or close a menu"),
    ("↑ ↓", "earlier requests"),
    ("/", "commands, tab completes"),
    ("ctrl+o", "every step"),
    ("ctrl+l", "empty the window"),
    ("ctrl+q", "leave"),
)

# How each kind of entry is marked: the mark, its colour, and the colour of the words. Colour only
# reinforces what the words and marks say. ANSI names, so the terminal's theme picks the colours.
MARKS: dict[str, tuple[str, str, str]] = {
    "you": (CHEVRON, "bold", "bold"),
    "pc": (CHEVRON, "bold magenta", "magenta"),
    "jarvis": (DOT, "bold", ""),
    "step": (DOT, "green", ""),
    "ask": (DOT, "yellow", ""),
    "result": ("⎿", "dim", "dim"),
    "warn": (DOT, "yellow", "yellow"),
    "error": (DOT, "red", "red"),
    "detail": ("⎿", "dim", "dim italic"),
    "note": ("✻", "cyan", "dim"),
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
    shown_at: float = 0.0
    answered: threading.Event = field(default_factory=threading.Event)
    answer: bool = False


@dataclass(slots=True)
class _Work:
    started: float
    label: str = "Working"
    stop: threading.Event = field(default_factory=threading.Event)


def render(shown: Shown) -> RenderableType:
    """One transcript entry: its mark in a gutter, then the words, wrapped under themselves."""
    mark, mark_style, style = MARKS.get(shown.kind, MARKS["detail"])
    nested = shown.nested or shown.kind in {"result", "detail"}
    gutter = Text(f"  {mark}  " if nested else f"{mark} ", style=mark_style)
    words = Text()
    if shown.kind == "pc":
        words.append("PC audio: ", style="bold magenta")
    words.append(shown.text, style=style)
    note = shown.note if shown.kind != "pc" else ""
    if shown.starts_turn and shown.time:
        note = f"{note} · {shown.time}" if note else shown.time
    if note:
        words.append(f"  {note}", style="dim")
    grid = Table.grid(padding=0)
    grid.add_column(width=gutter.cell_len, no_wrap=True)
    grid.add_column(ratio=1)
    grid.add_row(gutter, words)
    return grid


def welcome(name: str, unavailable: str) -> Panel:
    """The card the window opens with: who it is for, and how to use it."""
    lines: list[RenderableType] = [
        Text.assemble(
            ("✻ ", "cyan"),
            (name, "bold"),
            (f"  typed requests, and everything {name} does", "dim"),
        ),
        Text(""),
        Text(HELP, style="dim"),
    ]
    if unavailable:
        lines += [Text(""), Text(f"Typed requests are off: {unavailable}", style="yellow")]
    return Panel(Group(*lines), box=box.ROUNDED, border_style="cyan", padding=(0, 1), expand=False)


class PromptInput(Input):
    """The input line; ↑ and ↓ and Tab go to the window (history and the command menu)."""

    BINDINGS: ClassVar = [
        Binding("up", "app.older", show=False),
        Binding("down", "app.newer", show=False),
        Binding("tab", "app.complete", show=False),
    ]


class ChoiceList(OptionList):
    """The yes-or-no choice; 1 or y chooses yes, 2 or n no."""

    BINDINGS: ClassVar = [
        Binding("1,y", "app.yes", show=False),
        Binding("2,n", "app.no", show=False),
    ]


class ChatApp(App[None]):
    """Shows Jarvis's activity and takes typed requests, one at a time."""

    TITLE = "Jarvis"
    CSS = """
    Screen { layout: vertical; }
    #activity {
        height: 1fr; border: none; padding: 0 1; scrollbar-size-vertical: 1;
        scrollbar-background: ansi_default; scrollbar-color: ansi_bright_black;
    }
    #working { height: auto; padding: 0 1; color: ansi_cyan; }
    #question {
        height: auto; margin: 0 1; padding: 0 1;
        border: round ansi_yellow; border-title-color: ansi_yellow;
    }
    #question-text { height: auto; }
    #choices { height: auto; border: none; padding: 0; background: ansi_default; }
    #prompt { height: auto; margin: 0 1; border: round ansi_bright_black; }
    #prompt:focus-within { border: round ansi_cyan; }
    #caret { width: 3; padding: 0 1; color: ansi_cyan; text-style: bold; }
    #prompt Input { border: none; padding: 0; height: 1; background: ansi_default; }
    #menu { height: auto; max-height: 8; margin: 0 1; border: none; background: ansi_default; }
    #question-hint { height: auto; color: ansi_bright_black; }
    OptionList > .option-list--option-highlighted {
        background: ansi_default; color: ansi_cyan; text-style: bold;
    }
    #shortcuts { height: auto; margin: 0 2; color: ansi_bright_black; }
    #footer { height: 1; margin: 0 2; }
    #hints { width: 1fr; color: ansi_bright_black; }
    #status { width: auto; color: ansi_bright_black; }
    """
    BINDINGS: ClassVar = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("escape", "escape", show=False, priority=True),
        Binding("ctrl+o", "details", show=False, priority=True),
        Binding("ctrl+l", "clear", show=False, priority=True),
    ]

    def __init__(  # ruff: ignore[too-many-arguments] - collaborators and keyword-only settings
        self,
        tail: Tail,
        ask: Ask | None,
        *,
        unavailable: str = "",
        status: Callable[[], str] = lambda: "",
        name: str = "Jarvis",
        poll_s: float = POLL_S,
        status_s: float = STATUS_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Show what `tail` reads and send typed requests to `ask`.

        Without `ask`, typing says why (`unavailable`). `status` is the line under the input,
        refreshed every `status_s`; `name` is what the assistant is called. `clock` times the
        working line and the question's arming.
        """
        super().__init__()
        self.theme = "ansi-dark"
        self._tail = tail
        self._ask = ask
        self._unavailable = unavailable
        self._status = status
        self._name = name
        self._welcome = True  # the welcome card is shown until the window is emptied
        self._poll_s = poll_s
        self._status_s = status_s
        self._clock = clock
        self._lock = threading.Lock()
        self._work: _Work | None = None
        self._closing = False
        self._question: _Question | None = None
        self._kept: list[Shown] = []
        self._details = False
        self._history: list[str] = []
        self._recall = 0  # how far back ↑ has gone; 0 is the line being written
        self._draft = ""
        self._ticker: Timer | None = None
        self._echoed: list[str] = []  # typed requests shown at once, whose log entry is skipped

    @override
    def compose(self) -> ComposeResult:
        """Lay out the transcript, the working line, the question, the input and the footer."""
        yield RichLog(id="activity", wrap=True, markup=False, highlight=False)
        yield Static("", id="working", markup=False)
        with Vertical(id="question"):
            yield Static("", id="question-text", markup=False)
            yield ChoiceList(id="choices")
            yield Static("enter or 1 or 2 to choose · esc is no", id="question-hint", markup=False)
        with Horizontal(id="prompt"):
            yield Static(CHEVRON, id="caret")
            yield PromptInput(placeholder='Try "what\'s playing?"  (/ for commands)')
        yield OptionList(id="menu")
        yield Static("", id="shortcuts", markup=False)
        with Horizontal(id="footer"):
            yield Static("? for shortcuts", id="hints", markup=False)
            yield Static("", id="status", markup=False)

    def on_mount(self) -> None:
        """Open with the welcome card, then follow the log and the status."""
        for widget in ("#working", "#question", "#menu", "#shortcuts"):
            self.query_one(widget).display = False
        self.query_one("#question").border_title = f"{self._name} asks"
        self._transcript().write(welcome(self._name, self._unavailable))
        self.query_one(PromptInput).focus()
        self.set_interval(self._poll_s, self._poll)
        self.set_interval(self._status_s, self._refresh_status)
        self._refresh_status()

    def on_unmount(self) -> None:
        """Answer a waiting question with no, stop a request and let go of the log."""
        with self._lock:
            self._closing = True
            question, self._question = self._question, None
            work = self._work
        if question is not None:
            question.answered.set()
        if work is not None:
            work.stop.set()
        self._tail.close()

    # The transcript

    def _transcript(self) -> RichLog:
        return self.query_one("#activity", RichLog)

    def _show(self, shown: Shown) -> None:
        if shown.starts_turn and self._kept:
            self._transcript().write(Text(""))
        self._kept = [*self._kept[-(MAX_KEPT - 1) :], shown]
        if self._details or not shown.detail:
            self._transcript().write(render(shown))

    def _note(self, text: str, kind: str = "note") -> None:
        self._show(Shown("", kind, text, nested=kind in {"result", "detail"}))

    def _redraw(self) -> None:
        log = self._transcript()
        log.clear()
        if self._welcome:
            log.write(welcome(self._name, self._unavailable))
        first = True
        for shown in self._kept:
            if shown.starts_turn and not first:
                log.write(Text(""))
            if self._details or not shown.detail:
                log.write(render(shown))
            first = False

    def _poll(self) -> None:
        for entry in self._tail.read():
            if self._echo_of(entry):
                continue
            shown = describe(entry)
            if shown is not None:
                self._show(shown)

    def _echo_of(self, entry: dict[str, object]) -> bool:
        """Whether `entry` logs a request this window already showed as it was sent."""
        if entry.get("event") != "request" or entry.get("via") != "chat":
            return False
        text = entry.get("text")
        if text not in self._echoed:
            return False
        self._echoed.remove(str(text))
        return True

    # The footer and the working line

    def _refresh_status(self) -> None:
        self.run_worker(self._read_status, thread=True, group="status", exclusive=True)

    def _read_status(self) -> None:
        try:
            line = self._status()
        except Exception as exc:  # ruff: ignore[blind-except] - a status that fails says so
            line = f"status unknown: {type(exc).__name__}"
        with contextlib.suppress(RuntimeError):  # the window has closed
            self.call_from_thread(self.query_one("#status", Static).update, line)

    def _tick(self) -> None:
        work = self._work
        working = self.query_one("#working", Static)
        hints = self.query_one("#hints", Static)
        if work is None:
            working.display = False
            hints.update("? for shortcuts" if self._question is None else "")
            return
        seconds = int(self._clock() - work.started)
        frame = SPINNER[int(self._clock() / TICK_S) % len(SPINNER)]
        stopping = work.stop.is_set()
        label = "Stopping" if stopping else work.label
        working.update(f"{frame} {label}… ({seconds} s{'' if stopping else ' · esc to stop'})")
        working.display = True
        hints.update("esc to stop" if self._question is None else "")

    # Typing

    def on_input_changed(self, event: Input.Changed) -> None:
        """Open the command menu for a slash, and the shortcuts for a lone question mark."""
        value = event.value
        if value == "?":
            event.input.value = ""
            shortcuts = self.query_one("#shortcuts", Static)
            shortcuts.update(self._shortcuts())
            shortcuts.display = not shortcuts.display
            return
        self._menu(value)

    @staticmethod
    def _shortcuts() -> Text:
        text = Text()
        for i, (keys, what) in enumerate(SHORTCUTS):
            if i:
                text.append("   ")
            text.append(keys, style="bold")
            text.append(f" {what}")
        return text

    @staticmethod
    def _matches(value: str) -> list[Command]:
        value = value.lower()
        if not value.startswith("/") or " " in value:
            return []
        return [c for c in COMMANDS if c.name.startswith(value)]

    def _menu(self, value: str) -> None:
        menu = self.query_one("#menu", OptionList)
        matches = self._matches(value)
        if not matches or [c.name for c in matches] == [value.lower()]:
            menu.display = False
            return
        menu.clear_options()
        for command in matches:
            menu.add_option(
                Option(Text.assemble((f"{command.name:<10}", "bold"), (command.about, "dim")))
            )
        menu.highlighted = 0
        menu.display = True

    def _menu_choice(self) -> str | None:
        menu = self.query_one("#menu", OptionList)
        if not menu.display or menu.highlighted is None:
            return None
        matches = self._matches(self.query_one(PromptInput).value)
        return matches[menu.highlighted].name if menu.highlighted < len(matches) else None

    def action_complete(self) -> None:
        """Tab: take the command highlighted in the menu."""
        choice = self._menu_choice()
        if choice is not None:
            prompt = self.query_one(PromptInput)
            prompt.value = choice
            prompt.cursor_position = len(choice)
            self.query_one("#menu").display = False

    def action_older(self) -> None:
        """↑: move up the command menu, or bring back an earlier request."""
        menu = self.query_one("#menu", OptionList)
        if menu.display:
            menu.action_cursor_up()
            return
        if self._recall >= len(self._history):
            return
        prompt = self.query_one(PromptInput)
        if self._recall == 0:
            self._draft = prompt.value
        self._recall += 1
        prompt.value = self._history[-self._recall]
        prompt.cursor_position = len(prompt.value)

    def action_newer(self) -> None:
        """↓: move down the command menu, or forward to the line being written."""
        menu = self.query_one("#menu", OptionList)
        if menu.display:
            menu.action_cursor_down()
            return
        if self._recall == 0:
            return
        self._recall -= 1
        prompt = self.query_one(PromptInput)
        prompt.value = self._history[-self._recall] if self._recall else self._draft
        prompt.cursor_position = len(prompt.value)

    def action_escape(self) -> None:
        """Esc: answer a question no, else close a menu, else stop the request."""
        if self._question is not None:
            self._answer(yes=False)
            return
        for widget in ("#menu", "#shortcuts"):
            if self.query_one(widget).display:
                self.query_one(widget).display = False
                return
        work = self._work
        if work is not None and not work.stop.is_set():
            work.stop.set()

    def action_details(self) -> None:
        """Ctrl+O: show or hide every step (model calls, dropped copies)."""
        self._details = not self._details
        self._redraw()
        self.notify(f"Every step {'shown' if self._details else 'hidden'} (ctrl+o)", timeout=2)

    def action_clear(self) -> None:
        """Ctrl+L: empty the window."""
        self._kept = []
        self._welcome = False
        self._transcript().clear()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Run a command or send a request."""
        choice = self._menu_choice()
        text = (choice or event.value).strip()
        event.input.value = ""
        self.query_one("#menu").display = False
        self.query_one("#shortcuts").display = False
        self._recall = 0
        if not text:
            return
        if text.startswith("/"):
            self._command(text)
            return
        if not self._history or self._history[-1] != text:
            self._history.append(text)
        self._request(text)

    def _command(self, text: str) -> None:
        match text.lower():
            case "/quit" | "/exit":
                self.exit()
            case "/clear":
                self.action_clear()
            case "/details":
                self.action_details()
            case "/status":
                self._note("Checking the voice service and the model…")
                self._refresh_status()
            case "/help":
                self._note(HELP)
                for command in COMMANDS:
                    self._note(f"{command.name}  {command.about}", "result")
            case _:
                self._note(f"No command {text}; / lists them.", "warn")

    # A request

    def _request(self, text: str) -> None:
        if self._ask is None:
            self._note(f"Typed requests are off: {self._unavailable}", "warn")
            return
        with self._lock:
            if self._work is not None:
                self._note("I'm still working on the last request; esc stops it.", "warn")
                return
            work = self._work = _Work(self._clock())
        self._echoed = [*self._echoed[-9:], text]
        now = datetime.now().astimezone().strftime("%H:%M:%S")
        self._show(Shown(now, "you", text, note="you, typed"))
        self._ticker = self.set_interval(TICK_S, self._tick)
        self._tick()
        self.run_worker(lambda: self._handle(text, work), thread=True, group="request")

    def _handle(self, text: str, work: _Work) -> None:
        ask = self._ask
        hooks = Hooks(
            confirm=self._confirm,
            waking=lambda: self._relabel(work, "Waking the model; the first answer takes longer"),
            still_waiting=lambda: self._relabel(work, "Still loading the model"),
            cancelled=lambda: self._closing or work.stop.is_set(),
        )
        try:
            if ask is not None:
                ask(text, hooks)
        except Exception as exc:  # ruff: ignore[blind-except] - shown in the window, never raised
            self._from_thread(f"Something went wrong: {type(exc).__name__}", "error")
        finally:
            if work.stop.is_set() and not self._closing:
                self._from_thread("stopped", "result")
            with self._lock:
                self._work = None
            with contextlib.suppress(RuntimeError):
                self.call_from_thread(self._idle)

    def _idle(self) -> None:
        if self._ticker is not None:
            self._ticker.stop()
            self._ticker = None
        self._tick()

    @staticmethod
    def _relabel(work: _Work, label: str) -> None:
        work.label = label

    def _from_thread(self, text: str, kind: str = "note") -> None:
        with contextlib.suppress(RuntimeError):  # the window has closed
            self.call_from_thread(self._note, text, kind)

    # A question

    def _confirm(self, text: str) -> bool:
        """Ask the owner in the window and wait for the choice (no when it closes)."""
        question = _Question(text)
        with self._lock:
            if self._closing:
                return False
            self._question = question
        try:
            self.call_from_thread(self._open_question, question)
        except RuntimeError:  # the window has closed
            return False
        question.answered.wait()
        return question.answer

    def _open_question(self, question: _Question) -> None:
        question.shown_at = self._clock()
        self.query_one("#question-text", Static).update(Text(question.text, style="bold"))
        choices = self.query_one("#choices", OptionList)
        choices.clear_options()
        choices.add_options([Option("1. Yes", id="yes"), Option("2. No", id="no")])
        choices.highlighted = 0
        self.query_one("#question").display = True
        self.query_one("#prompt").display = False
        choices.focus()
        self._tick()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Enter on a choice: answer the question, or run the highlighted command."""
        if event.option_list.id == "choices":
            self._answer(yes=event.option.id == "yes")
        elif event.option_list.id == "menu":
            self.action_complete()
            self.query_one(PromptInput).focus()

    def action_yes(self) -> None:
        """1 or y: answer the open question yes."""
        self._answer(yes=True)

    def action_no(self) -> None:
        """2 or n: answer the open question no."""
        self._answer(yes=False)

    def _answer(self, *, yes: bool) -> None:
        with self._lock:
            question = self._question
            if question is None:
                return
            if yes and self._clock() - question.shown_at < ARM_S:
                return  # a key meant for the input, pressed as the question opened
            self._question = None
        question.answer = yes
        self.query_one("#question").display = False
        self.query_one("#prompt").display = True
        self.query_one(PromptInput).focus()
        question.answered.set()
        self._tick()
