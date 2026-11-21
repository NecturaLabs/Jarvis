"""The chat window (`jarvis chat`): talk to Jarvis in writing and watch what it does, live.

Typed requests go to the window's own agent; its replies are rendered as Markdown as they are
written. What the voice service does is read from the shared audit log, so spoken requests and
their replies show here too. How much of the work in between shows is the verbosity: by default
only requests and replies, with a working line that says what Jarvis is doing; `steps` adds each
tool call and its result, `everything` the model calls and other details.

A question the agent asks opens a yes-or-no choice above the input; anything but a chosen yes is a
no, as by voice. A picture is attached as Claude Code does it: Ctrl+V pastes a copied picture (or
text), a file dropped on the terminal or `/image` attaches that file, and each shows as an
`[Image #N]` tag in the input; deleting the tag leaves the picture out. The layout follows Claude
Code's terminal interface, and the colours are the terminal's own, so the window takes the Omarchy
theme. Everything lines up on two columns: the marks of the transcript, the input's border and the
working line on the first, and the words, the caret and the footer two cells in.
"""

import contextlib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, override

from rich.cells import cell_len
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import OptionList, Static

from jarvis.activity import Shown, describe, full_text, working_label
from jarvis.agent import PICTURES_OFF, Hooks
from jarvis.chat import commands
from jarvis.chat.clipboard import ClipboardError
from jarvis.chat.commands import VERBOSITY
from jarvis.chat.widgets import (
    CHEVRON,
    STAR,
    ChoiceList,
    Entry,
    Note,
    PictureView,
    PointerList,
    PromptArea,
    Reply,
    Request,
    meter,
    tokens_short,
    toolbar,
    welcome,
    worked_line,
)
from jarvis.config import ChatConfig
from jarvis.pictures import PictureError, newest, pasted_path, prepare, read_file

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rich.console import RenderableType
    from textual.timer import Timer
    from textual.widget import Widget

    from jarvis.agent import ContextUse
    from jarvis.chat.clipboard import Clip
    from jarvis.chat.history import InputHistory
    from jarvis.memory import Conversation
    from jarvis.pictures import Picture, PictureCache
    from jarvis.sysinfo import Stats


POLL_S = 0.25
STATUS_S = 10.0
STATS_S = 2.0
TICK_S = 0.12
ARM_S = 0.4  # a choice made sooner than this after a question appears was meant for something else
QUIT_WINDOW_S = 2.0  # a second Ctrl+C this soon after the first leaves
MAX_ENTRIES = 600  # transcript widgets kept; older ones are removed
RESUMED_TURNS = 12  # remembered turns shown when the window opens on an earlier conversation
PASTED_NAME = "pasted picture"  # what a picture from the clipboard is called
MAX_CAPTION = 200
SPINNER = "·✢✳✶✻✽✻✶✳✢"
PLACEHOLDER = 'Ask anything, or try "what\'s playing?"  (/ for commands)'
PLACEHOLDER_OFF = "Typed requests are off (see above); / for commands"
MENU_HINT = "tab completes · enter runs · esc closes"
MENU_ROWS = 12  # the command menu's most rows; it keeps one height while it is open
FOOTER_MARGIN = 6  # the footer's margins, left and right
STATUS_GAP = 2  # between the footer's hint and the services' status
SHORTCUTS = (
    ("enter", "send"),
    ("shift+enter", "new line (or \\ then enter)"),
    ("esc", "stop the request, or close a menu"),
    ("↑ ↓", "earlier requests"),
    ("/", "commands, tab completes"),
    ("ctrl+v", "paste text or a picture"),
    ("ctrl+o", "show more or less of the work"),
    ("ctrl+l", "redraw"),
    ("ctrl+c", "clear the input; twice to leave"),
)
VERBOSITY_ABOUT = {
    "replies": "requests and replies only",
    "steps": "every step and its result",
    "everything": "every step, model call and detail",
}


class Tail(Protocol):
    """The audit events written since the last read."""

    def read(self) -> list[dict[str, object]]:
        """Return the new events, oldest first."""
        ...

    def close(self) -> None:
        """Let go of the log."""
        ...


class Session(Protocol):
    """What the window needs of its agent."""

    def handle(self, text: str, hooks: Hooks, *, pictures: Sequence[Picture] = ()) -> str:
        """Answer one request, with any pictures the owner attached, and return the reply."""
        ...

    def forget(self) -> None:
        """Start a new conversation."""
        ...

    def last_rate(self) -> float | None:
        """Return how fast the model wrote in the last request, in tokens per second."""
        ...

    def compact(self, *, everything: bool = False) -> bool:
        """Summarize the older turns now; return whether anything was folded."""
        ...

    def context_use(self) -> ContextUse:
        """Return how full the model's context is."""
        ...

    @property
    def memory(self) -> Conversation:
        """The remembered conversation."""
        ...


@dataclass(frozen=True, slots=True)
class Services:
    """What the window reads besides the agent; each part is optional."""

    status: Callable[[], str] = lambda: ""  # the voice service and model, for the toolbar
    details: Callable[[], list[tuple[str, str]]] = list  # more for /status: (name, value)
    stats: Callable[[], Stats] | None = None  # the PC's load
    save_settings: Callable[[ChatConfig], None] | None = None
    history: InputHistory | None = None
    export_dir: Path = field(default_factory=Path.home)
    paste: Callable[[], Clip] | None = None  # reads the clipboard, for Ctrl+V
    pictures: bool = False  # the model can see pictures
    screenshots: tuple[Path, ...] = ()  # where `/image` looks for the newest screenshot
    home: Path = field(default_factory=Path.home)  # what `~/` means in a pasted path
    shown: PictureCache | None = None  # where the pictures Jarvis shows are kept
    twelve_hour: bool = True  # "done 7:37 PM" under a reply, or "done 19:37"


@dataclass(slots=True)
class _Question:
    text: str
    shown_at: float = 0.0
    answered: threading.Event = field(default_factory=threading.Event)
    answer: bool = False


@dataclass(slots=True)
class _Work:
    started: float
    label: str = "Thinking"
    stop: threading.Event = field(default_factory=threading.Event)
    draft: str = ""


@dataclass(frozen=True, slots=True)
class _Setting:
    key: str
    label: str
    values: tuple[object, ...]
    live: bool  # takes effect now, not the next time the chat opens


SETTINGS = (
    _Setting("verbosity", "Verbosity", VERBOSITY, live=True),
    _Setting("toolbar", "Toolbar", (True, False), live=True),
    _Setting("resume", "Resume the last conversation", (True, False), live=False),
    _Setting("reply_tokens", "Longest reply (tokens)", (1024, 2048, 4096), live=False),
    _Setting("max_steps", "Model calls per request", (15, 30, 60), live=False),
)


def _joined(*parts: str) -> str:
    """Return the parts that say something, between dots."""
    return " · ".join(part for part in parts if part)


def _shown_value(value: object) -> str:
    if value is True:
        return "on"
    if value is False:
        return "off"
    return str(value)


class ChatApp(App[None]):
    """Shows Jarvis's conversation and work, and takes typed requests, one at a time."""

    TITLE = "Jarvis"
    CSS = """
    Screen { layout: vertical; background: ansi_default; }
    VerticalScroll, OptionList {
        scrollbar-size-vertical: 1; scrollbar-background: ansi_default;
        scrollbar-background-hover: ansi_default; scrollbar-background-active: ansi_default;
        scrollbar-color: ansi_bright_black; scrollbar-color-hover: ansi_white;
        scrollbar-color-active: ansi_white;
    }
    #transcript { width: 100%; height: 1fr; padding: 0 0 0 1; scrollbar-gutter: stable; }
    #welcome { height: auto; margin: 1 0 0 0; }
    .entry { height: auto; }
    .entry.spaced { margin: 1 0 0 0; }
    .note { margin: 1 0 0 0; }
    .request { height: auto; margin: 1 0 0 0; padding: 0 2 0 0; }
    .reply { height: auto; margin: 1 0 0 0; }
    .reply-mark { width: 2; color: ansi_default; text-style: bold; }
    .reply.draft .reply-mark { color: ansi_bright_black; }
    .reply-body { height: auto; width: 1fr; }
    .reply-note { height: auto; margin: 1 0 0 0; color: ansi_bright_black; }
    .reply Markdown { margin: 0; padding: 0; background: ansi_default; }
    .reply Markdown > *:first-child { margin-top: 0; }
    .reply Markdown > *:last-child { margin-bottom: 0; }
    .reply Markdown > MarkdownHeader:first-child { margin: 0; }
    .reply MarkdownH1, .reply MarkdownH2, .reply MarkdownH3,
    .reply MarkdownH4, .reply MarkdownH5, .reply MarkdownH6 {
        width: 1fr; margin: 1 0 0 0; content-align: left top;
        color: ansi_default; background: ansi_default; text-style: bold;
    }
    .reply MarkdownH1 { text-style: bold italic underline; }
    .reply MarkdownBlock > .code_inline { color: ansi_blue; }
    .reply MarkdownBullet { color: ansi_default; }
    .reply MarkdownFence {
        margin: 1 0; padding: 0 0 0 1; background: ansi_default;
        border-left: outer ansi_bright_black;
    }
    .reply MarkdownFence > Label { padding: 0; }
    .reply MarkdownBlockQuote {
        margin: 1 0; padding: 0 0 0 1; background: ansi_default;
        border-left: outer ansi_bright_black; color: ansi_bright_black; text-style: italic;
    }
    .reply MarkdownBlockQuote > MarkdownParagraph { margin: 0; }
    .reply MarkdownTable { width: auto; background: ansi_default; }
    .reply MarkdownTableContent { keyline: thin ansi_bright_black; }
    .reply MarkdownTableContent > .header { color: ansi_default; text-style: bold; }
    .reply MarkdownHorizontalRule { border-bottom: solid ansi_bright_black; }
    .picture { height: auto; margin: 1 0 0 2; }
    .picture-image { height: auto; }
    .picture-caption { height: auto; color: ansi_bright_black; }
    .level-steps, .level-everything { display: none; }
    Screen.-steps .level-steps { display: block; }
    Screen.-everything .level-steps, Screen.-everything .level-everything { display: block; }

    #working { height: auto; margin: 1 1 0 1; color: ansi_cyan; }
    #question, #settings {
        height: auto; margin: 1 1 0 1; padding: 0 1;
        border: round ansi_yellow; border-title-color: ansi_yellow;
    }
    #settings { border: round ansi_bright_black; border-title-color: ansi_default; }
    #question-text { height: auto; margin: 0 0 1 0; }
    #choices, #settings-list { height: auto; border: none; padding: 0; background: ansi_default; }
    .panel-hint { height: auto; margin: 1 0 0 0; color: ansi_bright_black; }
    #prompt { height: auto; margin: 1 1 0 1; padding: 0 1; border: round ansi_bright_black; }
    #caret { width: 2; padding: 0; color: ansi_bright_black; text-style: bold; }
    #prompt:focus-within #caret { color: ansi_cyan; }
    #prompt PromptArea {
        height: auto; max-height: 12; border: none; padding: 0; background: ansi_default;
    }
    #prompt PromptArea:focus { border: none; }
    #pictures { height: auto; margin: 0 3 0 5; color: ansi_bright_black; }
    #menu {
        margin: 0 3; border: none; background: ansi_default;
        text-wrap: nowrap; text-overflow: ellipsis;
    }
    OptionList > .option-list--option-highlighted,
    OptionList:focus > .option-list--option-highlighted {
        background: ansi_default; color: ansi_default; text-style: none;
    }
    OptionList > .option-list--option-hover { background: ansi_default; }
    #shortcuts { height: auto; margin: 0 3; color: ansi_bright_black; }
    #footer { height: 1; margin: 0 3; }
    #hints { width: 1fr; color: ansi_bright_black; text-wrap: nowrap; text-overflow: ellipsis; }
    #status {
        width: auto; max-width: 50%; padding: 0 0 0 2; color: ansi_bright_black;
        text-wrap: nowrap; text-overflow: ellipsis;
    }
    #toolbar { height: 1; margin: 0 3; text-wrap: nowrap; text-overflow: ellipsis; }
    """
    BINDINGS: ClassVar = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "interrupt", show=False, priority=True),
        Binding("escape", "escape", show=False, priority=True),
        Binding("ctrl+o", "verbosity", show=False, priority=True),
        Binding("ctrl+l", "redraw", show=False, priority=True),
    ]

    def __init__(  # ruff: ignore[too-many-arguments] - collaborators and keyword-only settings
        self,
        tail: Tail,
        session: Session | None,
        *,
        unavailable: str = "",
        name: str = "Jarvis",
        settings: ChatConfig | None = None,
        services: Services | None = None,
        session_id: str = "",
        poll_s: float = POLL_S,
        status_s: float = STATUS_S,
        stats_s: float = STATS_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Show what `tail` reads and send typed requests to `session`.

        Without a session, typing says why (`unavailable`). `session_id` marks the audit events
        of this window's own agent, which the window shows as they happen instead. `clock` times
        the working line, the question's arming and the second Ctrl+C.
        """
        super().__init__()
        self.theme = "ansi-dark"
        self._tail = tail
        self._session = session
        self._unavailable = unavailable
        self._name = name
        self._settings = settings or ChatConfig()
        self._services = services or Services()
        self._session_id = session_id
        self._poll_s = poll_s
        self._status_s = status_s
        self._stats_s = stats_s
        self._clock = clock
        self._lock = threading.Lock()
        self._work: _Work | None = None
        self._closing = False
        self._question: _Question | None = None
        self._recall = 0  # how far back ↑ has gone; 0 is the line being written
        self._draft = ""
        self._ticker: Timer | None = None
        self._status = ""
        self._stats: Stats | None = None
        self._quit_armed = float("-inf")
        self._exchanges: list[tuple[str, str]] = []  # ("you" or "jarvis", words), for /export
        self._history = self._services.history
        self._stick = True  # follow new entries down until the owner scrolls up
        self._attached: dict[int, Picture] = {}  # by the number in their `[Image #N]` tag
        self._picture_number = 0

    @override
    def compose(self) -> ComposeResult:
        """Lay out the transcript, the working line, the panels, the input and the footer."""
        yield VerticalScroll(id="transcript")
        yield Static("", id="working", markup=False)
        with Vertical(id="question"):
            yield Static("", id="question-text", markup=False)
            yield ChoiceList(id="choices")
            yield Static("enter, 1 or y: yes · 2, n or esc: no", classes="panel-hint")
        with Vertical(id="settings"):
            yield PointerList(id="settings-list")
            yield Static(
                "enter changes · esc closes · * applies the next time the chat opens",
                classes="panel-hint",
            )
        with Horizontal(id="prompt"):
            yield Static(CHEVRON, id="caret")
            yield PromptArea(
                placeholder=PLACEHOLDER if self._session is not None else PLACEHOLDER_OFF,
                compact=True,
                highlight_cursor_line=False,
                id="input",
            )
        yield Static("", id="pictures", markup=False)
        yield PointerList(id="menu")
        yield Static("", id="shortcuts", markup=False)
        with Horizontal(id="footer"):
            yield Static("", id="hints", markup=False)
            yield Static("", id="status", markup=False)
        yield Static("", id="toolbar", markup=False)

    def on_mount(self) -> None:
        """Open with the welcome card and the remembered conversation, then follow the log."""
        for widget in ("#working", "#question", "#settings", "#menu", "#shortcuts", "#pictures"):
            self.query_one(widget).display = False
        self.query_one("#question").border_title = f"{self._name} asks"
        self.query_one("#settings").border_title = "Settings"
        self.query_one("#menu").styles.height = min(len(commands.COMMANDS), MENU_ROWS)
        self._apply_verbosity()
        self.query_one("#toolbar").display = self._settings.toolbar
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(Static(self._welcome(), id="welcome"))
        self.watch(transcript, "scroll_y", self._scrolled, init=False)
        self._show_remembered()
        self._input().focus()
        self.set_interval(self._poll_s, self._poll)
        self.set_interval(self._status_s, self._refresh_status)
        self.set_interval(self._stats_s, self._refresh_stats)
        self._refresh_status()
        self._refresh_stats()
        self._tick()

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

    def _input(self) -> PromptArea:
        return self.query_one("#input", PromptArea)

    def _welcome(self) -> RenderableType:
        lines = [
            Text(
                "Ask anything, or tell me what to do. /help lists the commands, ? the shortcuts.",
                style="bright_black",
            )
        ]
        memory = self._session.memory if self._session is not None else None
        if memory is not None and (memory.turns or memory.summary):
            when = (
                datetime.fromtimestamp(memory.saved_at).astimezone().strftime("%b %d, %H:%M")
                if memory.saved_at
                else "earlier"
            )
            count = len(memory.turns)
            lines.append(
                Text.assemble(
                    ("Continuing the conversation from ", "bright_black"),
                    (when, ""),
                    (f" · {count} turn{'' if count == 1 else 's'}", "bright_black"),
                    (" and a summary" if memory.summary else "", "bright_black"),
                    (" · /clear starts a new one", "bright_black"),
                )
            )
        if self._unavailable:
            lines.append(Text(f"Typed requests are off: {self._unavailable}", style="yellow"))
        return welcome(self._name, lines)

    def _show_remembered(self) -> None:
        if self._session is None:
            return
        memory = self._session.memory
        if memory.summary:
            self._mount(Note(self._labelled("Earlier, in short", memory.summary)))
        for turn in memory.turns[-RESUMED_TURNS:]:
            self._mount(Request(turn.request, "earlier"))
            self._mount(Reply(turn.reply))
            self._exchanges += [("you", turn.request), ("jarvis", turn.reply)]

    @staticmethod
    def _labelled(label: str, text: str) -> Table:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_row(Text(f"{STAR} {label}", style="cyan"), Text(text, style="bright_black"))
        return grid

    def _mount(self, widget: Widget) -> None:
        """Add `widget` to the transcript; follow it down when the owner was at the bottom."""
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.mount(widget)
        if self._stick:
            widget.call_after_refresh(self._follow_down)
        children = transcript.children
        excess = len(children) - MAX_ENTRIES
        if excess > 0:
            for old in list(children)[1 : excess + 1]:  # the welcome card stays first
                old.remove()

    def _follow_down(self) -> None:
        """Keep the newest entry in view while the owner has not scrolled up to read.

        It also runs from timers and after a refresh, so it does nothing once the window is gone.
        """
        with contextlib.suppress(NoMatches):
            transcript = self.query_one("#transcript", VerticalScroll)
            if self._stick and transcript.scroll_y < transcript.max_scroll_y:
                transcript.scroll_end(animate=False)

    def _scrolled(self, scroll_y: float) -> None:
        with contextlib.suppress(NoMatches):
            transcript = self.query_one("#transcript", VerticalScroll)
            self._stick = scroll_y >= transcript.max_scroll_y - 1

    def _note(self, body: RenderableType | str, kind: str = "note") -> None:
        if isinstance(body, str):
            self._mount(Entry(Shown("", kind, body, nested=kind in {"result", "detail"})))
        else:
            self._mount(Note(body))

    def _poll(self) -> None:
        self._follow_down()  # a reply's Markdown grows after it is mounted
        for entry in self._tail.read():
            own = bool(self._session_id) and entry.get("session") == self._session_id
            if own:
                self._follow(entry)
                if entry.get("event") in {"request", "reply"}:
                    continue  # shown as it was sent, and as it was written
                if entry.get("event") == "tool_result" and entry.get("picture"):
                    self._show_shown(entry)
            shown = describe(entry)
            if shown is None:
                continue
            if shown.kind in {"you", "pc"}:
                pc = shown.kind == "pc"  # the request itself says "PC audio"
                self._mount(
                    Request(shown.text, _joined("" if pc else shown.note, shown.time), pc=pc)
                )
                self._exchanges.append(("you", shown.text))
            elif shown.kind == "jarvis":
                self._mount(Reply(shown.text, self._worked_note(entry)))
                self._exchanges.append(("jarvis", shown.text))
            else:
                self._mount(Entry(shown))

    def _show_shown(self, entry: dict[str, object]) -> None:
        """Draw a picture this window's agent showed, if it is in the chat's picture folder."""
        cache = self._services.shown
        path = Path(str(entry.get("picture")))
        if cache is None or not cache.holds(path):
            return
        caption = " ".join(full_text(entry.get("caption", "")).split())[:MAX_CAPTION]
        try:
            self._mount(PictureView(path, caption))
        except OSError:
            self._note("The picture couldn't be drawn.", "warn")

    def _worked_note(self, entry: dict[str, object]) -> str:
        """Return the line under a reply read from the log, as under one typed here."""
        seconds = entry.get("seconds")
        try:
            done = datetime.fromisoformat(str(entry.get("ts"))).astimezone()
        except ValueError:
            done = datetime.now().astimezone()
        if not isinstance(seconds, int | float):
            return ""
        line = worked_line(float(seconds), done, None, twelve_hour=self._services.twelve_hour)
        return f"{STAR} {line}"

    def _follow(self, entry: dict[str, object]) -> None:
        """Keep the working line on what this window's request is doing."""
        work = self._work
        label = working_label(entry)
        if work is not None and label is not None:
            work.label = label

    # The footer, the toolbar and the working line

    def _refresh_status(self) -> None:
        self.run_worker(self._read_status, thread=True, group="status", exclusive=True)

    def _read_status(self) -> None:
        try:
            line = self._services.status()
        except Exception as exc:  # ruff: ignore[blind-except] - a status that fails says so
            line = f"status unknown: {type(exc).__name__}"
        self._from_worker(self._set_status, line)

    def _set_status(self, line: str) -> None:
        self._status = line
        self.query_one("#status", Static).update(line)
        self._tick()

    def _refresh_stats(self) -> None:
        if self._services.stats is not None and self._settings.toolbar:
            self.run_worker(self._read_stats, thread=True, group="stats", exclusive=True)
        else:
            self._draw_toolbar()

    def _read_stats(self) -> None:
        sample = self._services.stats
        if sample is None:
            return
        try:
            stats = sample()
        except Exception:  # ruff: ignore[blind-except] - the toolbar leaves the figures out
            stats = None
        self._from_worker(self._set_stats, stats)

    def _set_stats(self, stats: Stats | None) -> None:
        self._stats = stats
        self._draw_toolbar()

    def _draw_toolbar(self) -> None:
        use = self._session.context_use() if self._session is not None else None
        width = max(0, self.size.width - FOOTER_MARGIN)
        self.query_one("#toolbar", Static).update(toolbar("", self._stats, use, width))

    def on_resize(self) -> None:
        """Fit the footer and the toolbar to the new width."""
        self._tick()
        self._draw_toolbar()

    def _from_worker(self, update: Callable[..., object], *args: object) -> None:
        """Run `update` on the window's thread, from a worker; nothing once the window is gone.

        A worker can finish while the window closes: its widgets may be gone by then.
        """

        def run() -> None:
            with contextlib.suppress(NoMatches):
                update(*args)

        with contextlib.suppress(RuntimeError):  # the window has closed
            self.call_from_thread(run)

    def _bottom(self) -> None:
        """While the command menu or the shortcuts are open they take the footer's place.

        As in Claude Code, the menu replaces the hints and the toolbar instead of squeezing the
        transcript, and the newest entry stays in view.
        """
        menu = self.query_one("#menu").display or self.query_one("#shortcuts").display
        footer, toolbar = self.query_one("#footer"), self.query_one("#toolbar")
        wanted = (not menu, self._settings.toolbar and not menu)
        if (footer.display, toolbar.display) != wanted:
            footer.display, toolbar.display = wanted
            self.call_after_refresh(self._follow_down)

    def _tick(self) -> None:
        self._bottom()
        work = self._work
        working = self.query_one("#working", Static)
        hints = self.query_one("#hints", Static)
        level = self._settings.verbosity
        if work is None:
            working.display = False
            idle = f"? for shortcuts · {VERBOSITY_ABOUT[level]} (ctrl+o)"
            room = self.size.width - FOOTER_MARGIN - len(self._status) - STATUS_GAP
            if len(idle) > room:
                idle = f"? for shortcuts · ctrl+o: {level}"
            if self.query_one("#menu").display:
                idle = MENU_HINT
            hints.update("" if self._question is not None else idle)
            return
        seconds = int(self._clock() - work.started)
        frame = SPINNER[int(self._clock() / TICK_S) % len(SPINNER)]
        stopping = work.stop.is_set()
        asking = self._question is not None
        label = "Stopping" if stopping else "Waiting for your answer" if asking else work.label
        line = Text.assemble((f"{frame} {label}…", "cyan"), (f" ({seconds}s", "bright_black"))
        if not stopping and not asking:
            line.append(" · esc to interrupt", style="bright_black")
        line.append(")", style="bright_black")
        working.update(line)
        working.display = True
        hints.update("")

    # Typing

    def on_text_area_changed(self, event: PromptArea.Changed) -> None:
        """Open the command menu for a slash, and the shortcuts for a lone question mark."""
        if event.text_area.id != "input":
            return
        value = event.text_area.text
        if self._attached:
            self._show_pictures()
        if value == "?":
            self._input().set_text("")
            shortcuts = self.query_one("#shortcuts", Static)
            shortcuts.update(self._shortcuts())
            shortcuts.display = not shortcuts.display
            self._tick()
            return
        self._menu(value)

    @staticmethod
    def _shortcuts() -> Table:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(no_wrap=True)
        grid.add_column()
        grid.add_column(no_wrap=True)
        grid.add_column()
        rows = list(SHORTCUTS)
        half = (len(rows) + 1) // 2
        right_column = rows[half:] + [("", "")] * (half - len(rows[half:]))
        for left, right in zip(rows[:half], right_column, strict=True):
            grid.add_row(
                Text(left[0], style="bold"), Text(left[1]), Text(right[0], style="bold"), right[1]
            )
        return grid

    def _menu(self, value: str) -> None:
        menu = self.query_one("#menu", PointerList)
        found = commands.suggest(value)
        shown = bool(found) and [s.value.strip() for s in found] != [value.strip().lower()]
        if shown:
            menu.show(
                [(s.label, s.about) for s in found],
                ("bold", "bright_black"),
                widths=(max(cell_len(every.label) for every in commands.suggest("/")),),
            )
        if menu.display != shown:
            menu.display = shown
            self._tick()

    def _menu_choice(self) -> str | None:
        menu = self.query_one("#menu", OptionList)
        if not menu.display or menu.highlighted is None:
            return None
        found = commands.suggest(self._input().text)
        return found[menu.highlighted].value if menu.highlighted < len(found) else None

    def action_complete(self) -> None:
        """Tab: take the command highlighted in the menu."""
        choice = self._menu_choice()
        if choice is not None:
            self._input().set_text(choice)
            self._menu(choice)

    def on_prompt_area_leave(self, event: PromptArea.Leave) -> None:
        """↑ or ↓ past the input's edge: move in the menu, else through earlier requests."""
        menu = self.query_one("#menu", OptionList)
        if menu.display:
            if event.up:
                menu.action_cursor_up()
            else:
                menu.action_cursor_down()
            return
        if event.up:
            self._older()
        else:
            self._newer()

    def _entries(self) -> list[str]:
        return self._history.entries if self._history is not None else []

    def _older(self) -> None:
        entries = self._entries()
        if self._recall >= len(entries):
            return
        prompt = self._input()
        if self._recall == 0:
            self._draft = prompt.text
        self._recall += 1
        prompt.set_text(entries[-self._recall])
        prompt.move_cursor((0, len(prompt.document.lines[0])))

    def _newer(self) -> None:
        if self._recall == 0:
            return
        self._recall -= 1
        entries = self._entries()
        self._input().set_text(entries[-self._recall] if self._recall else self._draft)

    def action_escape(self) -> None:
        """Esc: answer a question no, else close a panel or menu, else stop the request."""
        if self._question is not None:
            self._answer(yes=False)
            return
        if self.query_one("#settings").display:
            self._close_settings()
            return
        for widget in ("#menu", "#shortcuts"):
            if self.query_one(widget).display:
                self.query_one(widget).display = False
                self._tick()
                return
        work = self._work
        if work is not None and not work.stop.is_set():
            work.stop.set()
            self._tick()

    def action_interrupt(self) -> None:
        """Ctrl+C: clear what is typed; on an empty input, leave when pressed twice."""
        prompt = self._input()
        if prompt.text:
            prompt.set_text("")
            return
        now = self._clock()
        if now - self._quit_armed <= QUIT_WINDOW_S:
            self.exit()
            return
        self._quit_armed = now
        self.notify("Press ctrl+c again to leave", timeout=QUIT_WINDOW_S)

    def action_verbosity(self) -> None:
        """Ctrl+O: show more of the work, then less again."""
        level = VERBOSITY[(VERBOSITY.index(self._settings.verbosity) + 1) % len(VERBOSITY)]
        self._set_verbosity(level)

    def _set_verbosity(self, level: str) -> None:
        self._settings = self._settings.model_copy(update={"verbosity": level})
        self._apply_verbosity()
        self._save_settings()
        self.notify(f"Showing {VERBOSITY_ABOUT[level]}", timeout=2)

    def _apply_verbosity(self) -> None:
        level = self._settings.verbosity
        self.screen.set_class(level == "steps", "-steps")
        self.screen.set_class(level == "everything", "-everything")
        self._tick()

    def action_redraw(self) -> None:
        """Ctrl+L: draw the whole window again."""
        self.refresh(layout=True)
        self.screen.refresh(layout=True)

    def on_prompt_area_submitted(self, event: PromptArea.Submitted) -> None:
        """Run a command or send a request."""
        choice = self._menu_choice()
        text = event.text.strip()
        pictures = self._sent_pictures(text)
        if choice is not None and choice.strip() != text:
            self.action_complete()
            if not choice.endswith(" "):
                text = choice.strip()
            else:
                return  # a command that takes an argument: the menu now offers them
        self._input().set_text("")
        self.query_one("#menu").display = False
        self.query_one("#shortcuts").display = False
        self._tick()
        self._recall = 0
        if not text:
            return
        if self._history is not None:
            self._history.add(text)
        if text.startswith("/"):
            self._command(text)
            return
        self._request(text, pictures)

    # Pictures

    @staticmethod
    def _tag(number: int) -> str:
        return f"[Image #{number}]"

    def _sent_pictures(self, text: str) -> list[Picture]:
        """Return the attached pictures whose tag is still in `text`, and let go of them all."""
        pictures = [p for n, p in sorted(self._attached.items()) if self._tag(n) in text]
        self._attached = {}
        self._show_pictures()
        return pictures

    def _show_pictures(self) -> None:
        """List the attached pictures under the input; one whose tag was deleted is dropped."""
        text = self._input().text
        self._attached = {n: p for n, p in self._attached.items() if self._tag(n) in text}
        shown = self.query_one("#pictures", Static)
        lines = Text()
        for number, picture in sorted(self._attached.items()):
            if lines:
                lines.append("\n")
            lines.append(f"{self._tag(number)} ", style="cyan")
            lines.append(f"{picture.name} · {picture.width}x{picture.height}")
        if lines:
            lines.append("\ndelete a tag to leave its picture out", style="dim")
        shown.update(lines)
        shown.display = bool(self._attached)

    def _attach_picture(self, picture: Picture) -> None:
        self._picture_number += 1
        self._attached[self._picture_number] = picture
        self._input().insert_text(f"{self._tag(self._picture_number)} ")
        self._show_pictures()

    def _load_picture(self, load: Callable[[], Picture]) -> None:
        """Read and check a picture off the window's thread, then attach it."""
        if not self._services.pictures:
            self._note(PICTURES_OFF, "warn")
            return

        def run() -> None:
            try:
                picture = load()
            except PictureError as exc:
                self._from_worker(self._note, str(exc), "warn")
            else:
                self._from_worker(self._attach_picture, picture)

        self.run_worker(run, thread=True, group="picture")

    def action_paste(self) -> None:
        """Ctrl+V: attach a copied picture, or put copied text in, as Claude Code does."""
        paste = self._services.paste
        if paste is None:
            self._note("Pasting from the clipboard isn't set up here.", "warn")
            return

        def run() -> None:
            try:
                clip = paste()
            except ClipboardError as exc:
                self._from_worker(self._note, str(exc), "warn")
            else:
                self._from_worker(self._pasted_clip, clip)

        self.run_worker(run, thread=True, group="paste", exclusive=True)

    def _pasted_clip(self, clip: Clip) -> None:
        if not clip:
            self._note("Nothing is copied.", "warn")
        elif clip.image:
            data = clip.image
            self._load_picture(lambda: prepare(data, PASTED_NAME))
        else:
            self._pasted_text(clip.text)

    def on_prompt_area_pasted(self, event: PromptArea.Pasted) -> None:
        """The terminal's paste: a dropped picture file is attached, anything else typed in."""
        self._pasted_text(event.text)

    def _pasted_text(self, text: str) -> None:
        path = pasted_path(text, self._services.home) if self._services.pictures else None
        if path is None:
            self._input().insert_text(text)
            return
        self._load_picture(lambda: read_file(path))

    def _image_command(self, argument: str) -> None:
        """`/image [path]`: attach that picture file, or the newest screenshot."""
        if argument:
            value = argument.strip("'\"")
            home = self._services.home
            path = home / value[2:] if value.startswith("~/") else Path(value)
            self._load_picture(lambda: read_file(path))
            return
        folders = self._services.screenshots
        shot = newest(folders)
        if shot is None:
            where = ", ".join(str(folder) for folder in folders) or "no screenshot folder"
            self._note(f"No screenshot in {where}; name a file: /image ~/Pictures/shot.png", "warn")
            return
        self._load_picture(lambda: read_file(shot))

    # Commands

    def _command(self, text: str) -> None:
        head, _, argument = text.partition(" ")
        command = commands.find(head)
        if command is None:
            self._note(f"No command {head}; / lists them.", "warn")
            return
        handlers: dict[str, Callable[[str], None]] = {
            "/help": lambda _a: self._help(),
            "/clear": lambda _a: self._clear(),
            "/compact": lambda _a: self._compact(),
            "/context": lambda _a: self._context_note(),
            "/verbosity": self._verbosity_command,
            "/config": lambda _a: self._open_settings(),
            "/status": lambda _a: self._status_note(),
            "/copy": lambda _a: self._copy(),
            "/export": lambda _a: self._export(),
            "/image": self._image_command,
            "/exit": lambda _a: self.exit(),
        }
        handlers[command.name](argument.strip())

    def _help(self) -> None:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(no_wrap=True)
        grid.add_column()
        for command in commands.COMMANDS:
            grid.add_row(
                Text(f"{command.name} {command.hint}".strip(), style="bold"),
                Text(command.about, style="bright_black"),
            )
        self._note(
            self._panel(
                "Commands",
                grid,
                "Type a request as you would say it. What you say to Jarvis out loud shows here "
                "too. ? lists the keyboard shortcuts.",
            )
        )

    @staticmethod
    def _panel(title: str, body: RenderableType, footer: str = "") -> Table:
        grid = Table.grid()
        grid.add_column(width=2)
        grid.add_column()
        grid.add_row(Text(STAR, style="cyan"), Text(title, style="bold"))
        grid.add_row("", body)
        if footer:
            grid.add_row("", "")
            grid.add_row("", Text(footer, style="bright_black"))
        return grid

    def _clear(self) -> None:
        if self._work is not None:
            self._note("Wait for the request to finish, or press esc to stop it.", "warn")
            return
        if self._session is not None:
            self._session.forget()
        transcript = self.query_one("#transcript", VerticalScroll)
        for child in list(transcript.children)[1:]:
            child.remove()
        self._exchanges = []
        self._picture_number = 0
        self._input().set_text("")
        self.query_one("#welcome", Static).update(self._welcome())
        self._note("Started a new conversation.")
        self._draw_toolbar()

    def _compact(self) -> None:
        session = self._session
        if session is None:
            self._note(f"Typed requests are off: {self._unavailable}", "warn")
            return
        if not self._start("Compacting the conversation"):
            return
        work = self._work

        def run() -> None:
            try:
                folded = session.compact(everything=True)
            except Exception as exc:  # ruff: ignore[blind-except] - shown in the window
                self._from_worker(self._note, f"Couldn't compact: {type(exc).__name__}", "error")
            else:
                note = (
                    "Compacted the conversation into a summary."
                    if folded
                    else "Nothing to compact yet, or the model couldn't summarize it."
                )
                self._from_worker(self._note, note)
            finally:
                self._finish(work)

        self.run_worker(run, thread=True, group="request")

    def _context_note(self) -> None:
        if self._session is None:
            self._note(f"Typed requests are off: {self._unavailable}", "warn")
            return
        use = self._session.context_use()
        memory = self._session.memory
        free = max(0, use.window - use.fixed - use.memory)
        grid = Table.grid(padding=(0, 2))
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True, justify="right")
        grid.add_column()
        grid.add_row(
            meter(use.used / use.window if use.window else 0, 20),
            Text(f"{tokens_short(use.used)}/{tokens_short(use.window)}", style="bold"),
            Text(f"{round(100 * use.used / use.window) if use.window else 0}% used", "dim"),
        )
        grid.add_row(Text("prompt and tools"), Text(tokens_short(use.fixed)), Text(""))
        summary = " and a summary" if memory.summary else ""
        grid.add_row(
            Text("conversation"),
            Text(tokens_short(use.memory)),
            Text(
                f"{use.turns} turn{'' if use.turns == 1 else 's'}{summary}; compacts past "
                f"{tokens_short(use.memory_budget)}",
                style="bright_black",
            ),
        )
        grid.add_row(Text("free"), Text(tokens_short(free)), Text(""))
        self._note(self._panel("Context", grid, "/compact summarizes now; /clear starts anew."))

    def _verbosity_command(self, argument: str) -> None:
        argument = argument.lower()
        if argument in VERBOSITY:
            self._set_verbosity(argument)
        elif argument:
            self._note(f"Verbosity is one of {', '.join(VERBOSITY)}.", "warn")
        else:
            level = self._settings.verbosity
            self._note(f"Showing {VERBOSITY_ABOUT[level]} ({level}). ctrl+o changes it.")

    def _status_note(self) -> None:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(no_wrap=True, style="bright_black")
        grid.add_column()
        grid.add_row("services", self._status or "checking…")
        try:
            details = self._services.details()
        except Exception as exc:  # ruff: ignore[blind-except] - a status that fails says so
            details = [("details", f"unknown: {type(exc).__name__}")]
        for key, value in details:
            grid.add_row(key, value)
        self._note(self._panel("Status", grid))
        self._refresh_status()

    def _copy(self) -> None:
        last = next((words for who, words in reversed(self._exchanges) if who == "jarvis"), "")
        if not last:
            self._note("There is no reply to copy yet.", "warn")
            return
        self.copy_to_clipboard(last)
        self._note("Copied the last reply.")

    def _export(self) -> None:
        if not self._exchanges:
            self._note("There is nothing to save yet.", "warn")
            return
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d-%H%M%S")
        folder = self._services.export_dir
        path = folder / f"jarvis-chat-{stamp}.md"
        parts = [f"# {self._name} chat, {stamp}\n"]
        for who, words in self._exchanges:
            parts.append(f"**You:** {words}\n" if who == "you" else f"{words}\n")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                handle.write("\n".join(parts))
        except OSError as exc:
            self._note(f"Couldn't save it: {exc.strerror or type(exc).__name__}", "error")
            return
        self._note(f"Saved the conversation to {path}")

    # Settings

    def _open_settings(self) -> None:
        self._fill_settings()
        self.query_one("#settings").display = True
        self.query_one("#prompt").display = False
        self.query_one("#settings-list", OptionList).focus()

    def _fill_settings(self, highlighted: int = 0) -> None:
        rows = [
            (
                setting.label,
                _shown_value(getattr(self._settings, setting.key)),
                "" if setting.live else "*",
            )
            for setting in SETTINGS
        ]
        self.query_one("#settings-list", PointerList).show(
            rows, ("", "bold", "bright_black"), [s.key for s in SETTINGS], highlighted
        )

    def _close_settings(self) -> None:
        self.query_one("#settings").display = False
        self.query_one("#prompt").display = True
        self._input().focus()

    def _change_setting(self, key: str) -> None:
        setting = next(s for s in SETTINGS if s.key == key)
        current = getattr(self._settings, key)
        values = setting.values
        following = (
            values[(values.index(current) + 1) % len(values)] if current in values else values[0]
        )
        self._settings = self._settings.model_copy(update={key: following})
        if key == "verbosity":
            self._apply_verbosity()
        if key == "toolbar":
            self.query_one("#toolbar").display = bool(following)
            self._refresh_stats()
        self._save_settings()
        self._fill_settings(SETTINGS.index(setting))

    def _save_settings(self) -> None:
        save = self._services.save_settings
        if save is None:
            return
        try:
            save(self._settings)
        except Exception as exc:  # ruff: ignore[blind-except] - the change still holds for now
            self.notify(f"Couldn't save the setting: {type(exc).__name__}", severity="warning")

    # A request

    def _start(self, label: str) -> bool:
        with self._lock:
            if self._work is not None:
                busy = True
            else:
                busy = False
                self._work = _Work(self._clock(), label)
        if busy:
            self._note("I'm still working on the last request; esc stops it.", "warn")
            return False
        self._ticker = self.set_interval(TICK_S, self._tick)
        self._tick()
        return True

    def _finish(self, work: _Work | None) -> None:
        with self._lock:
            if self._work is work:
                self._work = None
        self._from_worker(self._idle)

    def _request(self, text: str, pictures: Sequence[Picture] = ()) -> None:
        if self._session is None:
            self._note(f"Typed requests are off: {self._unavailable}", "warn")
            return
        if not self._start("Thinking"):
            return
        work = self._work
        if work is None:
            return
        now = datetime.now().astimezone().strftime("%H:%M:%S")
        count = len(pictures)
        shared = "" if not count else "a picture" if count == 1 else f"{count} pictures"
        self._mount(Request(text, _joined("typed", now, shared)))
        self._exchanges.append(("you", text))
        self.run_worker(lambda: self._handle(text, work, pictures), thread=True, group="request")

    def _handle(self, text: str, work: _Work, pictures: Sequence[Picture] = ()) -> None:
        session = self._session
        hooks = Hooks(
            confirm=self._confirm,
            waking=lambda: self._relabel(work, "Waking the model; the first answer takes longer"),
            still_waiting=lambda: self._relabel(work, "Still loading the model"),
            cancelled=lambda: self._closing or work.stop.is_set(),
            draft=lambda text: self._from_worker(self._show_draft, work, text),
            hush=lambda: self._from_worker(self._drop_draft, work),
        )
        reply = ""
        rate: float | None = None
        try:
            if session is not None:
                reply = session.handle(text, hooks, pictures=pictures)
                rate = session.last_rate()
        except Exception as exc:  # ruff: ignore[blind-except] - shown in the window, never raised
            self._from_worker(self._note, f"Something went wrong: {type(exc).__name__}", "error")
        finally:
            seconds = self._clock() - work.started
            stopped = work.stop.is_set() and not self._closing
            self._from_worker(
                partial(self._settle, stopped=stopped, rate=rate), work, full_text(reply), seconds
            )
            self._finish(work)

    def _show_draft(self, work: _Work, text: str) -> None:
        work.draft = full_text(text)
        try:
            draft = self.query_one(".reply.draft", Reply)
        except NoMatches:
            self._mount(Reply(work.draft, draft=True))
            return
        draft.write(work.draft)

    def _drop_draft(self, work: _Work) -> None:
        work.draft = ""
        for draft in self.query(".reply.draft"):
            draft.remove()

    def _settle(
        self, work: _Work, reply: str, seconds: float, *, stopped: bool, rate: float | None = None
    ) -> None:
        self._drop_draft(work)
        done = datetime.now().astimezone()
        line = worked_line(
            seconds, done, rate, twelve_hour=self._services.twelve_hour, stopped=stopped
        )
        note = f"{STAR} {line}"
        if reply:
            self._mount(Reply(reply, note))
            self._exchanges.append(("jarvis", reply))
        elif stopped:
            self._note("Stopped.")
        self._draw_toolbar()

    def _idle(self) -> None:
        if self._ticker is not None:
            self._ticker.stop()
            self._ticker = None
        self._tick()

    @staticmethod
    def _relabel(work: _Work, label: str) -> None:
        work.label = label

    # A question

    def _confirm(self, text: str) -> bool:
        """Ask the owner in the window and wait for the choice (no when it closes)."""
        question = _Question(text)
        with self._lock:
            if self._closing:
                return False
            self._question = question
        self._from_worker(self._open_question, question)
        question.answered.wait()  # answered no when the window closes
        return question.answer

    def _open_question(self, question: _Question) -> None:
        question.shown_at = self._clock()
        self.query_one("#question-text", Static).update(Text(question.text, style="bold"))
        choices = self.query_one("#choices", ChoiceList)
        choices.show([("1. Yes",), ("2. No",)], ("",), ["yes", "no"])
        self.query_one("#question").display = True
        self.query_one("#prompt").display = False
        choices.focus()
        self._tick()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Enter on a choice, a command in the menu, or a setting."""
        match event.option_list.id:
            case "choices":
                self._answer(yes=event.option.id == "yes")
            case "menu":
                self.action_complete()
                self._input().focus()
            case "settings-list":
                if event.option.id is not None:
                    self._change_setting(event.option.id)
            case _:
                pass

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
        self._input().focus()
        question.answered.set()
        self._tick()
