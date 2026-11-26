"""The chat window's pieces: transcript entries, replies in Markdown, the input and the toolbar.

Every colour is a terminal colour name (`ansi_*` in CSS, plain names in Rich styles), so the window
takes the terminal's palette and with it the Omarchy theme. Colour only repeats what words and
marks say, and nothing is drawn on black: Omarchy's terminals make colour 0 the background, so
structure comes from marks, rules and spacing instead of shaded bands. Cyan is the one accent: the
caret, the owner's requests, and the option the keyboard is on.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Protocol, cast, override
from urllib.parse import urlsplit

from PIL import Image as PillowImage
from rich import box
from rich.cells import cell_len
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events  # ruff: ignore[typing-only-third-party-import] - Textual reads handler annotations
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Markdown, OptionList, Static, TextArea

# Textual exports no public way to change how a list is numbered, so these two come from its
# private module; tests/test_chat.py checks the numbers, and would catch a Textual change.
from textual.widgets._markdown import (  # ruff: ignore[import-private-name] - see above
    MarkdownBullet,
    MarkdownOrderedList,
)
from textual.widgets.option_list import Option
from textual_image.widget import Image

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import datetime
    from pathlib import Path

    from rich.console import RenderableType
    from textual.app import ComposeResult
    from textual.widgets.markdown import MarkdownBlock

    from jarvis.activity import Shown
    from jarvis.agent import ContextUse
    from jarvis.sysinfo import Stats

DOT = "\N{BLACK CIRCLE}"  # a step or a reply
CHEVRON = "\N{HEAVY RIGHT-POINTING ANGLE QUOTATION MARK ORNAMENT}"  # the input, a request, a choice
ACCENT = "cyan"
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
    words = Text.assemble((shown.text, f"{style} bold".strip() if shown.kind == "step" else style))
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


MAX_CHAIN_TARGETS = 3  # the different targets a chain's row names before it says "and N more"


class Chain(Vertical):
    """Calls of one tool in a row, as one step: how many, on what, and the newest result.

    With every detail shown, each call's result is listed under it too.
    """

    def __init__(self, first: Shown, op: str, target: str, session: object) -> None:
        """Start the chain with its first call."""
        super().__init__(classes="chain")
        self.op = op
        self.session = session
        self.shown = first  # the step, as an `Entry` has it
        self._targets = [target]
        self._results: list[Shown] = []
        self._earlier: list[Static] = []  # earlier results waiting to be composed
        self._head = Static(classes=f"entry level-{first.level} spaced")
        self._newest = Static(classes=f"entry level-{first.level}")
        self._newest.display = False

    @override
    def compose(self) -> ComposeResult:
        """The step's line, then its newest result."""
        yield self._head
        yield from self._earlier
        self._earlier = []
        yield self._newest
        self._draw()

    def add_call(self, target: str) -> None:
        """Count one more call of the tool."""
        self._targets.append(target)
        self._draw()

    def add_result(self, result: Shown) -> None:
        """Show `result` as the newest; it is listed with every detail shown."""
        self._results.append(result)
        if len(self._results) > 1:  # the one before moves into the list of every result
            earlier = Static(entry_renderable(self._results[-2]), classes="entry level-everything")
            if self._newest.parent is None:  # not composed yet: it comes with the rest
                self._earlier.append(earlier)
            else:
                self.mount(earlier, before=self._newest)
        self._draw()

    def _draw(self) -> None:
        calls = len(self._targets)
        named = list(dict.fromkeys(t for t in self._targets if t))
        quoted = ", ".join(f'"{t}"' for t in named[:MAX_CHAIN_TARGETS])
        if len(named) > MAX_CHAIN_TARGETS:
            quoted += f" and {len(named) - MAX_CHAIN_TARGETS} more"
        note = quoted if calls == 1 else _joined_note(quoted, f"{calls} calls")
        self._head.update(entry_renderable(replace(self.shown, note=note)))
        if self._results:
            newest = self._results[-1]
            done = len(self._results)
            if done > 1:
                newest = replace(newest, note=_joined_note(newest.note, f"newest of {done}"))
            self._newest.update(entry_renderable(newest))
            self._newest.display = True


def _joined_note(*parts: str) -> str:
    return " · ".join(part for part in parts if part)


class Note(Static):
    """Something the window itself says (help, context, status): always shown."""

    def __init__(self, body: RenderableType) -> None:
        """Show `body`."""
        super().__init__(body, classes="entry note")


class Request(Static):
    """What was asked: the owner's words after the accent's chevron, how and when at the right."""

    def __init__(self, text: str, note: str, *, pc: bool = False) -> None:
        """Show `text` with `note` after it; `pc` marks speech heard from the PC's audio."""
        words = Text()
        if pc:
            words.append("PC audio: ", style="bold magenta")
        words.append(text, style="magenta" if pc else "")
        grid = Table.grid(padding=0, expand=True)
        grid.add_column(width=2, no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_column(no_wrap=True, justify="right")
        grid.add_row(
            Text(f"{CHEVRON} ", style=f"bold {'magenta' if pc else ACCENT}"),
            words,
            Text(f"  {note}", style="bright_black") if note else Text(""),
        )
        super().__init__(grid, classes="request")
        self.text = text


class ReplyOrderedList(MarkdownOrderedList):
    """A numbered list whose numbers start at the text's edge, as Claude Code sets them.

    Textual pads each number one space more than the widest needs; that space is taken off, so
    "1." lines up with a dash and "10." still lines up with "9.".
    """

    def on_mount(self) -> None:
        """Take the extra space off this list's own numbers (not a nested list's)."""
        for row in self.query_children(Horizontal):
            for bullet in row.query_children(MarkdownBullet):
                bullet.symbol = bullet.symbol.removeprefix(" ")


class _Opener(Protocol):
    def open_url(self, url: str) -> None: ...


def link_opens(href: str) -> bool:
    """Whether a clicked link in a reply may be opened: web addresses only.

    A reply carries words from pages and files, so a `file:`, `javascript:` or app link in it is
    someone else's, and clicking it does nothing.
    """
    parts = urlsplit(href.strip())
    return parts.scheme in {"http", "https"} and bool(parts.netloc)


class ReplyMarkdown(Markdown):
    """A reply's Markdown, set as Claude Code sets it: dashes for lists, numbers at the edge."""

    BULLETS: ClassVar[list[str]] = ["- "]

    def __init__(self, markdown: str | None = None, *, classes: str | None = None) -> None:
        """Set `markdown`; Textual's own link opening is off, `on_markdown_link_clicked` opens."""
        super().__init__(markdown, classes=classes, open_links=False)

    @override
    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Open a clicked web link in the browser; any other kind of link stays shut."""
        href = event.href.strip()
        if link_opens(href):
            app = cast("_Opener", self.app)  # pyright: ignore[reportUnknownMemberType] - App[Unknown]
            app.open_url(href)

    @override
    def get_block_class(self, block_name: str) -> type[MarkdownBlock]:
        """Number lists with `ReplyOrderedList`; every other block is Textual's own."""
        if block_name == "ordered_list_open":
            return ReplyOrderedList
        return super().get_block_class(block_name)


class Reply(Vertical):
    """A reply: the dot, then the words rendered as Markdown; under them, how long it took."""

    def __init__(self, text: str, note: str = "", *, draft: bool = False) -> None:
        """Show `text`; a `draft` is still being written and is replaced when it is whole."""
        super().__init__(classes="reply draft" if draft else "reply")
        self.text = text
        self._note = note

    @override
    def compose(self) -> ComposeResult:
        """The mark and the body, then the note in the mark's column, as Claude Code's is."""
        with Horizontal(classes="reply-line"):
            yield Static(DOT, classes="reply-mark")
            yield ReplyMarkdown(self.text, classes="reply-body")
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


MINUTE = 60
HOUR = 3600
NOON = 12


def duration(seconds: float) -> str:
    """Return how long something took the way Claude Code says it: 34s, 4m 41s, 1h 2m."""
    whole = max(1, round(seconds))
    if whole < MINUTE:
        return f"{whole}s"
    if whole < HOUR:
        return f"{whole // MINUTE}m {whole % MINUTE}s"
    return f"{whole // HOUR}h {whole % HOUR // MINUTE}m"


def _clock_time(moment: datetime, *, twelve_hour: bool) -> str:
    if not twelve_hour:
        return f"{moment.hour}:{moment.minute:02d}"
    return f"{moment.hour % 12 or 12}:{moment.minute:02d} {'AM' if moment.hour < NOON else 'PM'}"


def worked_line(
    seconds: float,
    done: datetime,
    rate: float | None,
    *,
    twelve_hour: bool,
    stopped: bool = False,
) -> str:
    """Return the line under a reply: "Worked for 4m 41s · done 7:37 PM · 50 TPS".

    `rate` is the model's own writing speed in tokens per second, left out when unknown.
    """
    took = f"{'Stopped after' if stopped else 'Worked for'} {duration(seconds)}"
    parts = [took, f"done {_clock_time(done, twelve_hour=twelve_hour)}"]
    if rate:
        parts.append(f"{round(rate)} TPS")
    return " · ".join(parts)


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


SEPARATOR = " · "  # between the items of a line under the input, dim
GAP = "  "  # the least between a line's left part and its right part


def _gib(value: int, *, tenths: bool = True) -> str:
    return f"{value / GIB:.1f}" if tenths else f"{value / GIB:.0f}"


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


def _joined_parts(parts: list[Text]) -> Text:
    """Join items with the one quiet separator."""
    line = Text()
    for part in parts:
        if line:
            line.append(SEPARATOR, style="bright_black")
        line.append_text(part)
    return line


def _item(label: str, value: str, style: str = "", extra: str = "") -> Text:
    """One figure: a dim label, its value coloured by how loaded it is, a dim extra after."""
    text = Text()
    text.append(f"{label} ", style="bright_black")
    text.append(value, style=style)
    if extra:
        text.append(f" {extra}", style="bright_black")
    return text


@dataclass(frozen=True, slots=True)
class _Fit:
    """How much the toolbar says; it says less in a narrower window."""

    temperatures: bool = True
    meter: int = BAR_CELLS
    tenths: bool = True  # memory in tenths of a gigabyte
    cpu: bool = True
    gpu: bool = True
    label: str = "context"


# From the widest window down, each step drops the least useful thing left: the temperatures,
# half the context meter, the tenths of a gigabyte, the processor, the graphics card's load, the
# meter itself. Whole items go, never part of one, so the line never wraps or ends mid-word.
FITS = (
    _Fit(),
    _Fit(temperatures=False),
    _Fit(temperatures=False, meter=BAR_CELLS // 2),
    _Fit(temperatures=False, meter=BAR_CELLS // 2, tenths=False),
    _Fit(temperatures=False, meter=BAR_CELLS // 2, tenths=False, cpu=False),
    _Fit(temperatures=False, meter=BAR_CELLS // 2, tenths=False, cpu=False, gpu=False),
    _Fit(temperatures=False, meter=0, tenths=False, cpu=False, gpu=False, label="ctx"),
)


def toolbar(
    status: str, stats: Stats | None, use: ContextUse | None, width: int | None = None
) -> Text:
    """Return the toolbar: the PC's load at the left, how full the context is at the right.

    Given a `width`, it says as much as fits on one line (`FITS`) and the context's right edge
    is on the last cell, as the status above it ends. Figures have a fixed width, so the line
    does not move as they change.
    """
    line = Text()
    for fit in FITS:
        line = _toolbar(status, stats, use, fit, width)
        if width is None or line.cell_len <= width:
            break
    return line


def _toolbar(
    status: str, stats: Stats | None, use: ContextUse | None, fit: _Fit, width: int | None
) -> Text:
    left = Text()
    if status:
        left.append(status, style="bright_black")
    if stats is not None:
        load = _load(stats, fit)
        if load:
            if left:
                left.append(SEPARATOR, style="bright_black")
            left.append_text(load)
    right = _context(use, fit) if use is not None and use.window else Text()
    if not right:
        return left
    if width is None or not left:
        gap = GAP if left else ""
        return Text.assemble(left, gap, right)
    pad = max(len(GAP), width - left.cell_len - right.cell_len)
    return Text.assemble(left, " " * pad, right)


def _context(use: ContextUse, fit: _Fit) -> Text:
    text = Text()
    text.append(f"{fit.label} ", style="bright_black")
    if fit.meter:
        text.append_text(meter(use.used / use.window, fit.meter))
        text.append(" ")
    window = tokens_short(use.window)
    text.append(f"{tokens_short(use.used):>{len(window)}}/{window}", style="bright_black")
    if use.memory_budget and use.memory >= WARN_AT * use.memory_budget:
        text.append(SEPARATOR, style="bright_black")
        text.append("compacts soon", style="yellow")
    return text


def _load(stats: Stats, fit: _Fit) -> Text:
    """Return the processor, graphics card and memory figures that are known."""
    parts: list[Text] = []
    gpu = stats.gpu
    if fit.cpu and stats.cpu_percent is not None:
        temperature = stats.cpu_temperature_c
        parts.append(
            _item(
                "CPU",
                f"{stats.cpu_percent:>3.0f}%",
                _level(stats.cpu_percent / 100),
                _degrees(temperature, fit),
            )
        )
    if fit.gpu and gpu is not None and gpu.busy_percent is not None:
        parts.append(
            _item(
                "GPU",
                f"{gpu.busy_percent:>3}%",
                _level(gpu.busy_percent / 100),
                _degrees(gpu.temperature_c, fit),
            )
        )
    if gpu is not None and gpu.vram_used is not None and gpu.vram_total:
        parts.append(_memory("VRAM", gpu.vram_used, gpu.vram_total, fit))
    if stats.ram_used is not None and stats.ram_total:
        parts.append(_memory("RAM", stats.ram_used, stats.ram_total, fit))
    return _joined_parts(parts)


def _degrees(temperature: float | None, fit: _Fit) -> str:
    """Return a temperature in three cells and the sign, or as many blanks when unknown."""
    if not fit.temperatures:
        return ""
    return f"{temperature:>3.0f}°" if temperature is not None else "    "


def _memory(label: str, used: int, total: int, fit: _Fit) -> Text:
    whole = _gib(total, tenths=fit.tenths)
    value = f"{_gib(used, tenths=fit.tenths):>{len(whole)}}/{whole}G"
    return _item(label, value, _level(used / total))


_FOOTER_STEPS = ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2))


def footer_line(hints: Sequence[str], status: str, width: int) -> tuple[str, str]:
    """Return the hint and the status that fit side by side in `width` cells, as most as can.

    `hints` run from the longest wording to the shortest. The status is "voice inactive · model
    server not running": its first part is kept when the whole does not fit. What goes first is
    the status's second part, then the longer hint, then the rest of the status; whole items
    only, so nothing is cut mid-word.
    """
    parts = [part for part in status.split(SEPARATOR) if part]
    statuses = [status, parts[0] if parts else "", ""]
    for hint_at, status_at in _FOOTER_STEPS:
        hint = hints[min(hint_at, len(hints) - 1)]
        said = statuses[status_at]
        if cell_len(hint) + (cell_len(said) + len(GAP) if said else 0) <= width:
            return hint, said
    return hints[-1], ""


class PointerList(OptionList):
    """Options in aligned columns, the highlighted one pointed at, as in Claude Code's menus.

    The highlighted row gets the chevron and the accent; the others keep each column's own style.
    """

    _rows: tuple[tuple[str, ...], ...] = ()
    _styles: tuple[str, ...] = ()
    _widths: tuple[int, ...] = ()
    _pointed: int | None = None
    _marks: tuple[tuple[int, ...], ...] | None = None

    def show(  # ruff: ignore[too-many-arguments] - the rows and how to draw them
        self,
        rows: Sequence[tuple[str, ...]],
        styles: tuple[str, ...],
        ids: Sequence[str | None] = (),
        highlighted: int = 0,
        *,
        widths: Sequence[int] = (),
        marks: Sequence[tuple[int, ...]] | None = None,
    ) -> None:
        """List `rows` (their cells styled by `styles`, column by column) and point at one.

        `widths` are the columns' least widths, so a list that is filtered keeps its columns
        where they were. `marks` makes it a filtered list, as Claude Code draws its command
        menu: each row's typed letters (places in its first cell) in bold, the chosen row lit
        whole in the accent and the others dim, no name in bold.
        """
        self._marks = None if marks is None else tuple(marks)
        self._rows = tuple(rows)
        self._styles = styles
        columns = max((len(row) for row in rows), default=0)
        least = tuple(widths) + (0,) * columns
        self._widths = tuple(
            max([least[column], *(cell_len(row[column]) for row in rows if column < len(row))])
            for column in range(columns)
        )
        self.clear_options()
        self._pointed = highlighted
        self.add_options(
            Option(self._prompt(index, chosen=index == highlighted), id=ids[index] if ids else None)
            for index in range(len(self._rows))
        )
        self.highlighted = highlighted if self._rows else None

    def _prompt(self, index: int, *, chosen: bool) -> Text:
        row = self._rows[index]
        text = Text(no_wrap=True, overflow="ellipsis")
        text.append(f"{CHEVRON} " if chosen else "  ", style=f"bold {ACCENT}")
        filtered = self._marks is not None
        for column, cell in enumerate(row):
            last = column == len(row) - 1
            padded = cell if last else cell + " " * (self._widths[column] - cell_len(cell) + 3)
            if chosen:
                style = f"bold {ACCENT}" if column == 0 and not filtered else ACCENT
            else:
                style = self._styles[column] if column < len(self._styles) else ""
            start = len(text)
            text.append(padded, style=style)
            if column == 0 and self._marks is not None and index < len(self._marks):
                for place in self._marks[index]:
                    text.stylize(
                        "bold" if chosen else "bold default", start + place, start + place + 1
                    )
        return text

    @override
    def watch_highlighted(self, highlighted: int | None) -> None:
        """Move the pointer to the highlighted row."""
        super().watch_highlighted(highlighted)
        for index in {self._pointed, highlighted} - {None}:
            if index is not None and index < self.option_count:
                self.replace_option_prompt_at_index(
                    index, self._prompt(index, chosen=index == highlighted)
                )
        self._pointed = highlighted


_CHOICE_KEYS = frozenset({"1", "2", "n"})


class ChoiceList(PointerList):
    """The yes-or-no choice; 1 chooses yes, 2 or n no, and Enter the one pointed at.

    `on_typed` is called for any other printable key: the owner is typing words meant for the
    input, so a 1 among them is not a yes.
    """

    BINDINGS: ClassVar = [
        Binding("1", "app.yes", show=False),
        Binding("2,n", "app.no", show=False),
    ]
    on_typed: Callable[[], None] | None = None

    def on_key(self, event: events.Key) -> None:
        """Note a printable key that is not a choice."""
        if self.on_typed is not None and event.is_printable and event.character not in _CHOICE_KEYS:
            self.on_typed()
