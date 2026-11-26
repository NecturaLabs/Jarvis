"""The chat's `/config` screen, set as Claude Code sets its settings.

Sections are tabs across the top and a search line sits under them; typing anywhere filters the
settings of every section. Each setting is one line: its name at the left, its value in an aligned
column, the one the keyboard is on marked with the chevron. What the selected setting does and when
a change applies are said once, dimmed, in a footer under the list, so the list stays short enough
to read at a glance.

Enter or space turns an on-or-off setting over; on one with more values it opens a small list of
them (↑↓ choose, Enter picks, Esc leaves it as it was), and ← → still step through the values. A
risky setting (acting without asking, taking requests from PC audio) is marked while it is on, and
turning it on asks first: the footer says what it will do, and only a second Enter turns it on.
"""

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, override

from rich.cells import cell_len
from rich.console import Console
from rich.table import Table
from rich.text import Text
from textual import events  # ruff: ignore[typing-only-third-party-import] - Textual reads handler annotations
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from jarvis.chat.widgets import ACCENT, CHEVRON

if TYPE_CHECKING:
    from collections.abc import Sequence

    from textual.app import ComposeResult

    from jarvis.toolswitch import ToolSetting

WARN = "yellow"
DANGER = "bold red"
DIM = "bright_black"
SIGN = "\N{WARNING SIGN}"
MORE_BEFORE = "\N{SINGLE LEFT-POINTING ANGLE QUOTATION MARK}"  # tabs out of view at the left
MORE_AFTER = "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK}"  # ... at the right
SEARCH = "⌕"  # the search line's magnifier
NOW = "now"  # when a change takes effect: at once
NEXT_CHAT = "next chat"  # ... when the chat opens again
ON_CLOSE = "on close"  # ... when the screen closes (the voice service restarts)
NOW_AND_ON_CLOSE = "now, on close"  # ... in this chat at once, and for the voice on close
APPLIES = {
    NOW: "Applies at once.",
    NEXT_CHAT: "Applies the next time the chat opens.",
    ON_CLOSE: "Applies to the voice when you close settings.",
    NOW_AND_ON_CLOSE: "Applies at once, and to the voice when you close settings.",
}
RESTART_NOTE = "The voice service restarts when you close settings."
RESTART_NOTE_SHORT = "Restarts when you close settings."
FOOTER_LINES = 8  # the most the footer may take; `#settings-footer` has the same `max-height`
_MEASURE = Console(width=80, file=io.StringIO())  # to count the lines a text wraps to
TOOL_ROW = "tool:"  # the start of a row's id for one tool
TOOLS = "Tools"
HINTS = (
    "↑↓ select · enter change · ←→ value · tab section · type to search · esc close",
    "↑↓ select · enter change · tab section · esc close",
    "↑↓ · enter · tab · esc",
)
TAB_HINTS = ("←→ section · ↓ settings · type to search · esc close", "←→ section · esc close")
PICKER_HINTS = ("↑↓ choose · enter pick · esc back", "↑↓ · enter · esc")
SEARCH_HINTS = (
    "↑↓ select · enter change · backspace erases · esc clears the search",
    "↑↓ · enter · esc clears",
)
PLACEHOLDER = "Search settings…"
_TOOL_STYLES = {"on": "green", "asks first": WARN, "needs the confirm key": "magenta", "off": DIM}
MINUTE = 60
HOUR = 3600


def _duration(seconds: float) -> str:
    whole = round(seconds)
    if whole < MINUTE:
        return f"{whole} s"
    if whole < HOUR:
        return f"{whole / MINUTE:g} min"
    return f"{whole / HOUR:g} h"


def shown(value: object, unit: str = "", zero: str = "") -> str:
    """Return a value as the settings show it: on and off for a yes or no, a length of time short.

    A `unit` of "seconds" is shown as 30 s, 10 min or 4 h, and `zero` is the word for 0 in a
    setting where 0 means something ("never", "none").
    """
    named = {True: "on", False: "off"}
    if isinstance(value, bool):
        return named[value]
    if zero and value == 0:
        return zero
    if unit == "seconds" and isinstance(value, int | float):
        return _duration(value)
    return str(value)


@dataclass(frozen=True, slots=True)
class Row:
    """One setting: its id, name, what it does, its choices and value, and when it applies.

    `risky` is what turning it on means, said before it is turned on; while it is on, the row is
    marked. `unit` and `zero` say how a number is shown (`shown`). A `tool` row has no choices of
    its own: Enter moves it to the next stricter setting, and `note` is what is said after it.
    """

    id: str
    name: str
    about: str
    choices: tuple[object, ...]
    value: object
    applies: str = NOW
    risky: str = ""
    unit: str = ""
    zero: str = ""
    tool: bool = False
    note: str = ""

    @property
    def toggle(self) -> bool:
        """Whether it is on or off, which Enter turns over."""
        return self.choices in {(False, True), (True, False)}

    @property
    def text(self) -> str:
        """The value as it is shown."""
        return self.text_of(self.value)

    def text_of(self, value: object) -> str:
        """Return `value` as this setting shows its values."""
        return str(value) if self.tool else shown(value, self.unit, self.zero)

    @property
    def dangerous(self) -> bool:
        """Whether a risky setting is on now: at any value but its first, which is off."""
        return bool(self.risky) and bool(self.choices) and self.value != self.choices[0]

    def turns_on(self, value: object) -> bool:
        """Whether moving to `value` turns a risky setting on from off."""
        return bool(self.risky) and not self.dangerous and value != self.choices[0]

    def step(self, by: int) -> object:
        """Return the value `by` places on from the one in force, round the choices."""
        if self.value not in self.choices:
            return self.choices[0]
        return self.choices[(self.choices.index(self.value) + by) % len(self.choices)]


@dataclass(frozen=True, slots=True)
class Section:
    """A tab and its rows."""

    title: str
    rows: tuple[Row, ...]


@dataclass(frozen=True, slots=True)
class Pending:
    """A risky setting asked to be turned on to `value` by a step `by` (1 on, -1 back)."""

    key: str
    value: object
    by: int

    def hint(self, row: Row) -> str:
        """Return how to confirm it: the same step again, to the same value.

        A value picked from the list has no step to repeat, so only Enter confirms it.
        """
        keys = "enter again" if self.by == 0 else "enter or → again" if self.by > 0 else "← again"
        what = "turn it on" if row.toggle else f"set {row.text_of(self.value)}"
        return f"{keys}: {what} · esc: keep it off"


def tool_rows(tools: Sequence[ToolSetting]) -> tuple[Row, ...]:
    """Return a row for each tool: its name, its setting, and what it does."""
    rows: list[Row] = []
    for tool in tools:
        note = f"Ships {tool.shipped_label}." if tool.state is not tool.shipped else ""
        rows.append(
            Row(
                f"{TOOL_ROW}{tool.name}",
                tool.name,
                f"{tool.summary.rstrip('.')}. Enter or → moves it on, ← back, never below "
                "how it ships.",
                (),
                tool.label,
                ON_CLOSE,
                tool=True,
                note=note,
            )
        )
    return tuple(rows)


def tool_style(label: object) -> str:
    """Return the colour a tool's setting is shown in."""
    return _TOOL_STYLES.get(str(label), "")


class SettingList(OptionList):
    """The rows of one section, or of a search, or the values of one setting being picked."""

    BINDINGS: ClassVar = [
        Binding("enter", "activate", "Change", show=False),
        Binding("space", "space", "Change, or a space in the search", show=False),
        Binding("right", "step(1)", "Next value", show=False),
        Binding("left", "step(-1)", "Previous value", show=False),
        Binding("tab", "section(1)", "Next section", show=False),
        Binding("shift+tab", "section(-1)", "Previous section", show=False),
        Binding("backspace", "erase", "Erase", show=False),
    ]

    class Activated(Message):
        """Enter or space on the row `row_id`."""

        def __init__(self, row_id: str) -> None:
            """Say that the row `row_id` was asked to change."""
            super().__init__()
            self.row_id = row_id

    class Stepped(Message):
        """← or → on the row `row_id`, `by` places (1 on, -1 back)."""

        def __init__(self, row_id: str, by: int) -> None:
            """Say that the row `row_id` moved `by` places."""
            super().__init__()
            self.row_id = row_id
            self.by = by

    class Go(Message):
        """Tab, or ← → on the tab row (`by`), a key typed, Backspace, or ↑ ↓ (`vertical`)."""

        def __init__(
            self, by: int = 0, typed: str = "", *, erase: bool = False, vertical: int = 0
        ) -> None:
            """Say where to go, or what was typed."""
            super().__init__()
            self.by = by
            self.typed = typed
            self.erase = erase
            self.vertical = vertical

    @override
    def action_cursor_up(self) -> None:
        """↑: the panel decides, as the first row's ↑ goes to the tab row."""
        self.post_message(self.Go(vertical=-1))

    @override
    def action_cursor_down(self) -> None:
        """↓: the panel decides, as ↓ on the tab row goes into the list."""
        self.post_message(self.Go(vertical=1))

    def move(self, by: int) -> None:
        """Move the pointer one row, as the keys do."""
        if by < 0:
            super().action_cursor_up()
        else:
            super().action_cursor_down()

    def _on_row(self) -> str | None:
        if self.highlighted is None or not self.option_count:
            return None
        return self.get_option_at_index(self.highlighted).id

    def action_activate(self) -> None:
        """Enter or space."""
        row = self._on_row()
        if row is not None:
            self.post_message(self.Activated(row))

    def action_space(self) -> None:
        """Space: a space in the search when one is begun, else Enter. The panel decides."""
        self.post_message(self.Go(typed=" "))

    def action_step(self, by: int) -> None:
        """← →: the value of the row moves; on the tab row, the section does."""
        row = self._on_row()
        if row is not None:
            self.post_message(self.Stepped(row, by))

    def action_section(self, by: int) -> None:
        """Tab and Shift+Tab."""
        self.post_message(self.Go(by))

    def action_erase(self) -> None:
        """Backspace."""
        self.post_message(self.Go(erase=True))

    def on_key(self, event: events.Key) -> None:
        """Typing anywhere searches: a printable key that is no command goes to the search line."""
        if event.is_printable and event.character and event.key != "space":
            event.stop()
            event.prevent_default()
            self.post_message(self.Go(typed=event.character))


class SettingsPanel(Vertical):
    """The settings screen: tabs, a search line, the list, and a footer on the selected setting."""

    class Activated(Message):
        """Enter or space on a setting, or a value picked: `value` when one was."""

        def __init__(self, row_id: str, value: object = None, *, picked: bool = False) -> None:
            """Say that the setting `row_id` was changed, to `value` when `picked`."""
            super().__init__()
            self.row_id = row_id
            self.value = value
            self.picked = picked

    class Stepped(Message):
        """← or → on a setting."""

        def __init__(self, row_id: str, by: int) -> None:
            """Say that the setting `row_id` moved `by` places."""
            super().__init__()
            self.row_id = row_id
            self.by = by

    class Moved(Message):
        """The keyboard moved to the setting `row_id`."""

        def __init__(self, row_id: str | None) -> None:
            """Say which setting the keyboard is on (none: it left the list)."""
            super().__init__()
            self.row_id = row_id

    class Redrawn(Message):
        """The panel drew again: its keys may have changed (a search, a picker, the tab row)."""

    def __init__(self, *, id: str | None = None) -> None:  # ruff: ignore[builtin-argument-shadowing] - Textual's own name
        """An empty panel; `show` fills it."""
        super().__init__(id=id)
        self._sections: tuple[Section, ...] = ()
        self._tab = 0
        self._search = ""
        self._tabs_focus = False  # the keyboard is on the tab row
        self._picking: Row | None = None
        self._pending: Pending | None = None
        self._restart = False
        self._pointed: int | None = None
        self._label_width = 0

    @override
    def compose(self) -> ComposeResult:
        """The tab row, the search line, the list and the footer."""
        yield Static("", id="settings-tabs", markup=False)
        yield Static("", id="settings-search", markup=False)
        yield SettingList(id="settings-list")
        yield Static("", id="settings-footer", markup=False)

    @property
    def options(self) -> SettingList:
        """The list of settings."""
        return self.query_one("#settings-list", SettingList)

    @property
    def sections(self) -> tuple[Section, ...]:
        """The tabs, in order."""
        return self._sections

    @property
    def tab(self) -> Section | None:
        """The section on show."""
        return self._sections[self._tab] if self._sections else None

    @property
    def searched(self) -> str:
        """What is typed in the search line."""
        return self._search

    @property
    def rows(self) -> list[Row]:
        """The settings on show: the section's, or whatever the search finds in every section."""
        if self._search:
            words = self._search.casefold().split()
            return [
                row
                for section in self._sections
                for row in section.rows
                if all(w in f"{row.name} {section.title}".casefold() for w in words)
            ]
        return list(self._sections[self._tab].rows) if self._sections else []

    @property
    def row(self) -> Row | None:
        """The setting the keyboard is on."""
        found = self._row_at(self.options.highlighted)
        return None if self._picking is not None else found

    @property
    def hint(self) -> tuple[str, ...]:
        """The keys that work now, from the longest wording to the shortest."""
        if self._picking is not None:
            return PICKER_HINTS
        if self._tabs_focus:
            return TAB_HINTS
        return SEARCH_HINTS if self._search else HINTS

    def open(self) -> None:
        """Start on the first tab, with nothing searched for."""
        self._tab = 0
        self._search = ""
        self._tabs_focus = False
        self._picking = None
        self._pending = None
        self._restart = False

    def show(
        self,
        sections: Sequence[Section],
        highlighted: str | None = None,
        pending: Pending | None = None,
        *,
        restart: bool = False,
    ) -> None:
        """List `sections`, pointing at the setting `highlighted` (else the first of the tab)."""
        self._sections = tuple(sections)
        self._tab = min(self._tab, max(0, len(self._sections) - 1))
        self._pending = pending
        self._restart = restart
        self._draw(highlighted)

    def pick_from(self, row: Row) -> None:
        """Open the list of the values `row` can take, with the one in force pointed at."""
        self._picking = row
        self._draw(None)

    def escape(self) -> bool:
        """Esc: close a picker, else clear the search; whether it did either."""
        if self._picking is not None:
            row, self._picking = self._picking, None
            self._draw(row.id)
            return True
        if self._search:
            self._search = ""
            self._draw(None)
            return True
        if self._tabs_focus:
            self._tabs_focus = False
            self._draw(None)
            return True
        return False

    def go(self, event: SettingList.Go) -> None:
        """Move to another section, or edit the search."""
        if event.vertical:
            self._vertical(event.vertical)
        elif event.typed == " " and not self._search:
            row_id = self._row_id_at(self.options.highlighted)
            if row_id is not None:
                self._activate(row_id)
        elif event.typed or event.erase:
            if self._picking is not None:
                return
            self._search = self._search[:-1] if event.erase else self._search + event.typed
            self._tabs_focus = False
            self._draw(None)
            self.post_message(self.Moved(None))
        elif event.by and self._sections and not self._search and self._picking is None:
            self._tab = (self._tab + event.by) % len(self._sections)
            self._draw(None)
            self.post_message(self.Moved(None))

    def _vertical(self, by: int) -> None:
        """↑ or ↓: move in the list; ↑ from the first setting goes to the tab row, ↓ back."""
        if self._tabs_focus:
            if by > 0:
                self._tabs_focus = False
                self._draw(None)
            return
        if by < 0 and not self.options.highlighted and not self._search and self._picking is None:
            self._tabs_focus = True
            self._draw(None)
            self.post_message(self.Moved(None))
            return
        self.options.move(by)

    def on_setting_list_stepped(self, event: SettingList.Stepped) -> None:
        """← → on a setting moves its value; on the tab row, the section."""
        event.stop()
        if self._tabs_focus:
            self.go(SettingList.Go(event.by))
        elif self._picking is None:
            self.post_message(self.Stepped(event.row_id, event.by))

    def on_setting_list_activated(self, event: SettingList.Activated) -> None:
        """Enter on a setting: a value picked, a setting turned over, or a picker opened."""
        event.stop()
        self._activate(event.row_id)

    def _row_id_at(self, index: int | None) -> str | None:
        if index is None or index >= self.options.option_count:
            return None
        return self.options.get_option_at_index(index).id

    def _activate(self, row_id: str) -> None:
        if self._tabs_focus:
            self._vertical(1)
            return
        if self._picking is not None:
            index = int(row_id.removeprefix("pick:"))
            row, self._picking = self._picking, None
            self.post_message(self.Activated(row.id, row.choices[index], picked=True))
            return
        row = next((r for r in self.rows if r.id == row_id), None)
        if row is None:
            return
        if row.tool or row.toggle or self._pending is not None:
            self.post_message(self.Activated(row.id))
        else:
            self.pick_from(row)

    def on_setting_list_go(self, event: SettingList.Go) -> None:
        """Tab, Shift+Tab, a typed letter or Backspace."""
        event.stop()
        self.go(event)

    # Drawing

    def _row_at(self, index: int | None) -> Row | None:
        if index is None or index >= self.options.option_count:
            return None
        row_id = self.options.get_option_at_index(index).id
        return next((r for r in self.rows if r.id == row_id), None)

    def _draw(self, highlighted: str | None) -> None:
        listed = self.options
        self._pointed = None
        listed.clear_options()
        rows = self._picking_rows() if self._picking is not None else self.rows
        names = (
            [self._picking.text_of(c) for c in self._picking.choices]
            if self._picking is not None
            else [r.name for r in rows]
        )
        self._label_width = max((cell_len(name) for name in names), default=0) + 3
        options: list[Option] = []
        for index, row_id in enumerate(self._ids(rows)):
            options.append(Option(self._line(rows, index, chosen=False), id=row_id))
        listed.add_options(options)
        if options:
            ids = [str(option.id) for option in options]
            if self._picking is not None:
                current = (
                    self._picking.choices.index(self._picking.value)
                    if (self._picking.value in self._picking.choices)
                    else 0
                )
                listed.highlighted = current
            else:
                listed.highlighted = ids.index(highlighted) if highlighted in ids else 0
        self._lit()
        self._redraw_header()
        self.post_message(self.Redrawn())

    def _picking_rows(self) -> list[Row]:
        picking = self._picking
        return [] if picking is None else [picking] * len(picking.choices)

    def _ids(self, rows: Sequence[Row]) -> list[str]:
        if self._picking is not None:
            return [f"pick:{index}" for index in range(len(self._picking.choices))]
        return [row.id for row in rows]

    def _line(self, rows: Sequence[Row], index: int, *, chosen: bool) -> Table:
        """One line: pointer, name, value; a picker line is a value alone."""
        chosen = chosen and not self._tabs_focus
        line = Table.grid(expand=True)
        line.add_column(width=2, min_width=2, no_wrap=True)
        line.add_column(width=self._label_width, min_width=self._label_width, no_wrap=True)
        line.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        mark = Text(f"{CHEVRON} " if chosen else "  ", style=f"bold {ACCENT}")
        row = rows[index]
        if self._picking is not None:
            value = row.choices[index]
            label = Text(row.text_of(value), style=f"bold {ACCENT}" if chosen else "")
            line.add_row(mark, label, Text("(now)" if value == row.value else "", style=DIM))
            return line
        name = Text(row.name, style=f"bold {ACCENT}" if chosen else "")
        line.add_row(mark, name, self._value(row, chosen=chosen))
        return line

    @staticmethod
    def _value(row: Row, *, chosen: bool) -> Text:
        text = Text(no_wrap=True, overflow="ellipsis")
        if row.dangerous:
            text.append(f"{SIGN} ", style=DANGER)
        if row.tool:
            text.append(row.text, style=tool_style(row.value))
            if row.note:
                text.append(f"  {row.note.removesuffix('.')}", style=DIM)
            return text
        if chosen:
            style = f"bold {ACCENT}"
        elif row.value is False or (row.value == row.choices[0] if row.risky else False):
            style = DIM
        else:
            style = ""
        text.append(row.text, style=style)
        return text

    def _lit(self) -> None:
        """Point at the highlighted row and fill the footer."""
        listed = self.options
        if self._pointed is not None and self._pointed < listed.option_count:
            self._set_line(self._pointed, chosen=False)
        index = listed.highlighted
        if index is not None and index < listed.option_count:
            self._set_line(index, chosen=True)
        self._pointed = index
        self._redraw_footer()

    def _set_line(self, index: int, *, chosen: bool) -> None:
        rows = self._picking_rows() if self._picking is not None else self.rows
        if index < len(rows):
            self.options.replace_option_prompt_at_index(
                index, self._line(rows, index, chosen=chosen)
            )

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Moving to another row points at it and says what it does."""
        event.stop()
        self._lit()
        if self._picking is None and event.option.id is not None:
            self.post_message(self.Moved(event.option.id))

    def on_resize(self) -> None:
        """A new width may need other tabs in view and a shorter footer."""
        self._redraw_header()
        self._redraw_footer()

    def _visible_tabs(self) -> tuple[int, int]:
        """Return the first and last tab to draw: all of them, or a run round the active one.

        The run grows from the active tab, a tab at a time on either side, while it fits with
        a mark at each end that has more.
        """
        count = len(self._sections)
        room = self.size.width
        widths = [cell_len(section.title) + 3 for section in self._sections]
        if not room or sum(widths) <= room:
            return 0, count - 1
        first = last = self._tab
        used = widths[first] + 4  # the two marks
        while True:
            grown = False
            for side in (1, -1):
                at = last + 1 if side > 0 else first - 1
                if 0 <= at < count and used + widths[at] <= room:
                    used += widths[at]
                    first, last = (first, at) if side > 0 else (at, last)
                    grown = True
            if not grown:
                return first, last

    def _redraw_header(self) -> None:
        tabs = Text(no_wrap=True, overflow="ellipsis")
        first, last = self._visible_tabs() if self._sections else (0, -1)
        if first > 0:
            tabs.append(f"{MORE_BEFORE} ", style=DIM)
        for number in range(first, last + 1):
            section = self._sections[number]
            if number > first:
                tabs.append(" ")
            if self._search:
                style = DIM
            elif number == self._tab:
                style = f"bold reverse {ACCENT}" if self._tabs_focus else "bold reverse"
            else:
                style = ACCENT if self._tabs_focus else ""
            tabs.append(f" {section.title} ", style=style)
        if last < len(self._sections) - 1:
            tabs.append(f" {MORE_AFTER}", style=DIM)
        self.query_one("#settings-tabs", Static).update(tabs)
        search = Text(no_wrap=True, overflow="ellipsis")
        if self._picking is not None:
            search.append(self._picking.name, style="bold")
            search.append("  choose a value", style=DIM)
        elif self._search:
            search.append(f"{SEARCH} ", style=DIM)
            search.append(self._search)
            search.append("▏", style=ACCENT)
            count = len(self.rows)
            search.append(f"   {count} found" if count else "   nothing found", style=DIM)
        else:
            search.append(f"{SEARCH} {PLACEHOLDER}", style=DIM)
        self.query_one("#settings-search", Static).update(search)

    def _redraw_footer(self) -> None:
        """Say what the selected setting does and when, in one place; a warning if it has one.

        What the owner must read whole goes last and always shows: the danger prompt's key hint,
        when a change applies, and that the voice service restarts. If all of it is more than the
        footer may hold (`FOOTER_LINES`), the text on what the setting does is cut short with an
        ellipsis, never what follows it.
        """
        head = Text()
        tail: list[Text] = []
        row = self.row
        picking = self._picking
        if picking is not None:
            head.append(picking.about, style=DIM)
            tail.append(Text(APPLIES[picking.applies], style=DIM))
        elif row is not None:
            pending = self._pending if self._pending and self._pending.key == row.id else None
            if pending is not None:
                head.append(f"{SIGN} Turn on? ", style=f"bold {WARN}")
                head.append(row.risky, style=WARN)
                tail.append(Text(pending.hint(row), style=DIM))
            else:
                if row.dangerous:
                    head.append(f"{SIGN} ", style=DANGER)
                    head.append(row.about, style="red")
                else:
                    head.append(row.about, style=DIM)
                tail.append(Text(APPLIES[row.applies], style=DIM))
            if self._search:
                section = next((s.title for s in self._sections if row in s.rows), "")
                tail[-1].append(f"  In {section}.", style=DIM)
        width = self.query_one("#settings-footer", Static).size.width or self.size.width
        if self._restart:
            note = Text(RESTART_NOTE, style=WARN)
            if _lines(note, width) > 1:
                note = Text(RESTART_NOTE_SHORT, style=WARN)
            tail.append(note)
        footer = _fitted(head, tail, width)
        self.query_one("#settings-footer", Static).update(footer)


def _lines(text: Text, width: int) -> int:
    """How many lines `text` takes when wrapped to `width` cells."""
    if width <= 0:
        return text.plain.count("\n") + 1
    return len(text.wrap(_MEASURE, width))


def _fitted(head: Text, tail: list[Text], width: int) -> Text:
    """Return `head` and then each of `tail` on a line of its own, `head` cut to leave room."""
    after = Text("\n").join(tail) if tail else Text()
    room = FOOTER_LINES - _lines(after, width) - (1 if tail else 0)
    kept = len(head.plain)
    shown = head
    while kept > 0 and _lines(shown, width) > max(1, room):
        kept = max(0, kept - max(1, width // 2))
        shown = head[:kept]
        shown.rstrip()
        shown.append("…", style=DIM)
    return Text("\n").join([shown, after]) if tail else shown
