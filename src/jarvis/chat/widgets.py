"""The chat window's pieces: transcript entries, replies in Markdown, the input and the toolbar.

Every colour is a terminal colour name (`ansi_*` in CSS, plain names in Rich styles), so the window
takes the terminal's palette and with it the Omarchy theme. Colour only repeats what words and
marks say.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, override

from PIL import Image as PillowImage
from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events  # ruff: ignore[typing-only-third-party-import] - Textual reads handler annotations
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Markdown, OptionList, Static, TextArea
from textual_image.widget import Image

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import RenderableType
    from textual.app import ComposeResult

    from jarvis.activity import Shown
    from jarvis.agent import ContextUse
    from jarvis.sysinfo import Stats

DOT = "\N{BLACK CIRCLE}"  # a step or a reply
CHEVRON = "\N{HEAVY RIGHT-POINTING ANGLE QUOTATION MARK ORNAMENT}"  # the input
PROMPT = ">"  # what was asked, as Claude Code marks it
ELBOW = "\N{DENTISTRY SYMBOL LIGHT VERTICAL AND BOTTOM RIGHT}"  # a result under its step
STAR = "\N{TEARDROP-SPOKED ASTERISK}"
BAR_FULL = "\N{BOX DRAWINGS HEAVY HORIZONTAL}"
BAR_EMPTY = "\N{BOX DRAWINGS HEAVY HORIZONTAL}"
BAR_CELLS = 10
PICTURE_CELLS = 64  # the widest a shown picture is drawn, in terminal cells
PICTURE_ROWS = 24  # and the tallest
CELL_SHAPE = 2  # a terminal cell is about twice as tall as it is wide
GIB = 1024**3
WARN_AT = 0.7
ALARM_AT = 0.9

# How each kind of entry is marked: the mark, its colour, and the colour of the words.
MARKS: dict[str, tuple[str, str, str]] = {
    "jarvis": (DOT, "bold", ""),
    "step": (DOT, "green", ""),
    "ask": (DOT, "yellow", ""),
    "result": (ELBOW, "bright_black", "bright_black"),
    "warn": (DOT, "yellow", "yellow"),
    "error": (DOT, "red", "red"),
    "detail": (ELBOW, "bright_black", "bright_black italic"),
    "note": (STAR, "cyan", "bright_black"),
}


def gutter_line(mark: str, mark_style: str, words: Text, *, nested: bool = False) -> Table:
    """Return a line with its mark in a gutter and the words wrapped under themselves."""
    gutter = Text(f"  {mark}  " if nested else f"{mark} ", style=mark_style)
    grid = Table.grid(padding=0)
    grid.add_column(width=gutter.cell_len, no_wrap=True)
    grid.add_column(ratio=1)
    grid.add_row(gutter, words)
    return grid


def entry_renderable(shown: Shown) -> RenderableType:
    """Return a step, result, warning or note as one marked line."""
    mark, mark_style, style = MARKS.get(shown.kind, MARKS["detail"])
    nested = shown.nested or shown.kind in {"result", "detail"}
    words = Text(shown.text, style=f"{style} bold".strip() if shown.kind == "step" else style)
    if shown.note:
        words.append(f"  {shown.note}", style="bright_black")
    return gutter_line(mark, mark_style, words, nested=nested)


class Entry(Static):
    """One step, result, warning or note; its level decides at which verbosity it shows."""

    def __init__(self, shown: Shown) -> None:
        """Show `shown`, marked for its verbosity level."""
        spaced = " spaced" if not shown.nested and shown.kind not in {"result", "detail"} else ""
        super().__init__(entry_renderable(shown), classes=f"entry level-{shown.level}{spaced}")
        self.shown = shown


class Note(Static):
    """Something the window itself says (help, context, status): always shown."""

    def __init__(self, body: RenderableType) -> None:
        """Show `body`."""
        super().__init__(body, classes="entry note")


class Request(Static):
    """What was asked: the owner's words on a shaded band, with how and when they came in."""

    def __init__(self, text: str, note: str, *, pc: bool = False) -> None:
        """Show `text` with `note` after it; `pc` marks speech heard from the PC's audio."""
        words = Text()
        if pc:
            words.append("PC audio: ", style="bold magenta")
        words.append(text, style="magenta" if pc else "")
        if note:
            words.append(f"  {note}", style="bright_black")
        super().__init__(
            gutter_line(PROMPT, "bold magenta" if pc else "bright_black", words),
            classes="request",
        )
        self.text = text


class Reply(Horizontal):
    """A reply: the dot, then the words rendered as Markdown, then how long it took."""

    def __init__(self, text: str, note: str = "", *, draft: bool = False) -> None:
        """Show `text`; a `draft` is still being written and is replaced when it is whole."""
        super().__init__(classes="reply draft" if draft else "reply")
        self.text = text
        self._note = note

    @override
    def compose(self) -> ComposeResult:
        """The mark and the body."""
        yield Static(DOT, classes="reply-mark")
        with Vertical(classes="reply-body"):
            yield Markdown(self.text)
            if self._note:
                yield Static(self._note, classes="reply-note", markup=False)

    def write(self, text: str) -> None:
        """Show `text` in place of what the draft held."""
        self.text = text
        self.query_one(Markdown).update(text)


class PictureView(Vertical):
    """A picture Jarvis shows, drawn with the terminal's own graphics, with its caption.

    In kitty it is drawn with the terminal graphics protocol, in a Sixel terminal with Sixel, and
    elsewhere in coloured half cells (textual-image picks when the window starts).
    """

    def __init__(self, path: Path, caption: str) -> None:
        """Show the picture file at `path`, which the chat's own picture folder holds.

        A file that cannot be read as a picture raises `OSError` here, not while it is drawn.
        """
        super().__init__(classes="picture")
        self.path = path
        self._caption = caption
        self._cells = self.cells(path)

    @staticmethod
    def cells(path: Path) -> int:
        """Return how many cells wide to draw the picture so it is at most `PICTURE_ROWS` tall."""
        with PillowImage.open(path) as picture:
            width, height = picture.size
        fitting = (PICTURE_ROWS * CELL_SHAPE * width) // max(1, height)
        return max(1, min(PICTURE_CELLS, fitting))

    @override
    def compose(self) -> ComposeResult:
        """The picture, then its caption."""
        picture = Image(self.path, classes="picture-image")
        picture.styles.width = self._cells
        yield picture
        yield Static(f"{STAR} {self._caption}", classes="picture-caption", markup=False)


def welcome(name: str, lines: list[Text]) -> Panel:
    """The card the window opens with."""
    body: list[RenderableType] = [
        Text.assemble((f"{STAR} ", "cyan"), ("Welcome to ", ""), (name, "bold")),
        Text(""),
        *lines,
    ]
    return Panel(Group(*body), box=box.ROUNDED, border_style="cyan", padding=(0, 1), expand=False)


class PromptArea(TextArea):
    """The input: Enter sends; Shift+Enter, Ctrl+J or a backslash then Enter start a new line.

    ↑ on the first line and ↓ on the last go to the window (history, or the command menu); Tab
    takes the highlighted command. Ctrl+V and the terminal's own paste go to the window too, which
    attaches a picture or puts the text in (`insert_text`).
    """

    BINDINGS: ClassVar = [
        Binding("enter", "submit", show=False, priority=True),
        Binding("shift+enter,ctrl+j", "newline", show=False, priority=True),
        Binding("tab", "app.complete", show=False, priority=True),
        Binding("up", "up", show=False, priority=True),
        Binding("down", "down", show=False, priority=True),
        Binding("ctrl+v", "app.paste", show=False, priority=True),
    ]

    @dataclass
    class Submitted(Message):
        """The owner pressed Enter on what they typed."""

        text: str

    @dataclass
    class Leave(Message):
        """↑ on the first line (`up`) or ↓ on the last (not `up`): for the window to handle."""

        up: bool

    @dataclass
    class Pasted(Message):
        """The terminal pasted `text` (or a file was dropped on it): for the window to place."""

        text: str

    @override
    async def _on_paste(self, event: events.Paste) -> None:
        """Hand a paste to the window, which knows a dropped picture from text."""
        event.prevent_default()
        event.stop()
        if not self.read_only:
            self.post_message(self.Pasted(event.text))

    def insert_text(self, text: str) -> None:
        """Put `text` where the cursor is, in place of any selection."""
        result = self.replace(text, *self.selection)
        self.move_cursor(result.end_location)
        self.focus()

    def action_submit(self) -> None:
        """Enter: send the text, or start a new line after a trailing backslash."""
        row, column = self.cursor_location
        if column > 0 and self.document.lines[row][column - 1 : column] == "\\":
            self.replace("\n", (row, column - 1), (row, column))
            return
        self.post_message(self.Submitted(self.text))

    def action_newline(self) -> None:
        """Shift+Enter: a new line."""
        self.insert("\n")

    def action_up(self) -> None:
        """↑: the line above, or the window's ↑ from the first line."""
        if self.cursor_location[0] == 0:
            self.post_message(self.Leave(up=True))
        else:
            self.action_cursor_up()

    def action_down(self) -> None:
        """↓: the line below, or the window's ↓ from the last line."""
        if self.cursor_location[0] >= self.document.line_count - 1:
            self.post_message(self.Leave(up=False))
        else:
            self.action_cursor_down()

    def set_text(self, text: str) -> None:
        """Replace what is typed and put the cursor at its end."""
        self.text = text
        self.move_cursor(self.document.end)


def _gib(value: int) -> str:
    return f"{value / GIB:.1f}"


def _level(fraction: float) -> str:
    if fraction >= ALARM_AT:
        return "red"
    if fraction >= WARN_AT:
        return "yellow"
    return "green"


def meter(fraction: float, cells: int = BAR_CELLS) -> Text:
    """Return a bar of `cells` cells, filled to `fraction` and coloured by how full it is."""
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * cells)
    return Text.assemble(
        (BAR_FULL * filled, _level(fraction)), (BAR_EMPTY * (cells - filled), "bright_black")
    )


def tokens_short(count: int) -> str:
    """Return a token count the way Claude Code shows it: 950, 7.6k, 24.6k."""
    thousand = 1000
    return f"{count}" if count < thousand else f"{count / thousand:.1f}k"


def _figure(text: Text, label: str, value: str, style: str = "") -> None:
    if text:
        text.append("  ·  ", style="bright_black")
    text.append(f"{label} ", style="bright_black")
    text.append(value, style=style)


def toolbar(status: str, stats: Stats | None, use: ContextUse | None) -> Text:
    """Return the toolbar: services, then the PC's load, then how full the context is."""
    text = Text()
    if status:
        text.append(status, style="bright_black")
    if stats is not None:
        _load(text, stats)
    if use is not None and use.window:
        _figure(text, "context", "")
        text.append_text(meter(use.used / use.window))
        text.append(f" {tokens_short(use.used)}/{tokens_short(use.window)}", style="bright_black")
        if use.memory_budget and use.memory >= WARN_AT * use.memory_budget:
            text.append("  compacts soon", style="yellow")
    return text


def _load(text: Text, stats: Stats) -> None:
    """Add the processor, graphics card and memory figures that are known."""
    if stats.cpu_percent is not None:
        _figure(text, "CPU", f"{stats.cpu_percent:.0f}%", _level(stats.cpu_percent / 100))
    if stats.cpu_temperature_c is not None:
        text.append(f" {stats.cpu_temperature_c:.0f}°", style="bright_black")
    gpu = stats.gpu
    if gpu is not None and gpu.busy_percent is not None:
        _figure(text, "GPU", f"{gpu.busy_percent}%", _level(gpu.busy_percent / 100))
        if gpu.temperature_c is not None:
            text.append(f" {gpu.temperature_c:.0f}°", style="bright_black")
    if gpu is not None and gpu.vram_used is not None and gpu.vram_total:
        share = gpu.vram_used / gpu.vram_total
        _figure(text, "VRAM", f"{_gib(gpu.vram_used)}/{_gib(gpu.vram_total)}G", _level(share))
    if stats.ram_used is not None and stats.ram_total:
        share = stats.ram_used / stats.ram_total
        _figure(text, "RAM", f"{_gib(stats.ram_used)}/{_gib(stats.ram_total)}G", _level(share))


class ChoiceList(OptionList):
    """The yes-or-no choice; 1 or y chooses yes, 2 or n no."""

    BINDINGS: ClassVar = [
        Binding("1,y", "app.yes", show=False),
        Binding("2,n", "app.no", show=False),
    ]
