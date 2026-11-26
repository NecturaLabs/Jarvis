"""The chat's `/config` screen: settings in sections, each with its value, what it does and when.

Each row shows its name, every value it can take with the one in force lit (or the value between
arrows when there are many), when a change takes effect, and a line on what it does. The row the
keyboard is on gets the pointer. A risky setting (acting without asking, taking requests from PC
audio) is marked while it is on, and turning it on asks first: the row says what it will do, and
only a second Enter turns it on. The tools have a list of their own, one line each.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, override

from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from textual.binding import Binding
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from jarvis.chat.widgets import ACCENT, CHEVRON

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jarvis.toolswitch import ToolSetting

WARN = "yellow"
DANGER = "bold red"
DIM = "bright_black"
NOW = "now"  # when a change takes effect: at once
NEXT_CHAT = "next chat"  # ... when the chat opens again
ON_CLOSE = "on close"  # ... when the screen closes (the voice service restarts)
NOW_AND_ON_CLOSE = "now, on close"  # ... in this chat at once, and for the voice on close
SETTINGS_HINT = "↑↓ move · enter or → next value · ← previous · esc closes"
SETTINGS_LEGEND = (
    "now: at once · next chat: when the chat opens again · on close: the voice service restarts"
)
TOOLS_HINT = "↑↓ move · enter: the next setting · esc: back to the settings"
TOOLS_LEGEND = "Enter goes from how a tool ships to stricter; `jarvis tools NAME on` lowers one."
_TOOL_STYLES = {"on": "green", "asks first": WARN, "needs the confirm key": "magenta", "off": DIM}


def shown(value: object) -> str:
    """Return a value as the settings show it: on and off for a yes or no."""
    if value is True:
        return "on"
    if value is False:
        return "off"
    return str(value)


@dataclass(frozen=True, slots=True)
class Row:
    """One setting: its id, name, what it does, its choices and value, and when it applies.

    `risky` is what turning it on means, said before it is turned on; while it is on, the row is
    marked. `summary` replaces the choices for a row that opens a list of its own.
    """

    id: str
    name: str
    about: str
    choices: tuple[object, ...]
    value: object
    applies: str = NOW
    risky: str = ""
    summary: str = ""

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
    """A heading and its rows."""

    title: str
    rows: tuple[Row, ...]


def _choices(row: Row, *, chosen: bool) -> Text:
    """Return every value the row can take, the one in force lit and the others dim."""
    if row.summary:
        return Text(row.summary, style=f"bold {ACCENT}" if chosen else "bold")
    lit = DANGER if row.dangerous else f"bold {ACCENT}" if chosen else "bold"
    text = Text()
    for number, choice in enumerate(row.choices):
        if number:
            text.append("  ")
        if choice == row.value:
            text.append(f"⚠ {shown(choice)}" if row.dangerous else shown(choice), style=lit)
        else:
            text.append(shown(choice), style=DIM)
    return text


@dataclass(frozen=True, slots=True)
class Pending:
    """A risky setting asked to be turned on to `value` by a step `by` (1 on, -1 back)."""

    key: str
    value: object
    by: int

    def hint(self, row: Row) -> str:
        """Return how to confirm it: the same step again, to the same value."""
        keys = "enter or → again" if self.by > 0 else "← again"
        what = "turn it on" if set(row.choices) == {False, True} else f"set {shown(self.value)}"
        return f"  {keys}: {what} · esc: keep it off"


def row_prompt(
    row: Row, widths: tuple[int, int], *, chosen: bool, pending: Pending | None
) -> Group:
    """Draw one setting: pointer, name, choices and when it applies, then what it does.

    The name and when-it-applies columns keep their width on every row, so the choices line up;
    in a narrow window the choices wrap rather than push the rest out. A `pending` risky
    setting says what turning it on means, and how to confirm the value asked about.
    """
    line = Table.grid(expand=True)
    line.add_column(width=2, min_width=2, no_wrap=True)
    line.add_column(width=widths[0], min_width=widths[0], no_wrap=True)
    line.add_column(ratio=1, overflow="fold")
    line.add_column(width=widths[1], min_width=widths[1], justify="right", no_wrap=True)
    line.add_row(
        Text(f"{CHEVRON} " if chosen else "  ", style=f"bold {ACCENT}"),
        Text(row.name, style=f"bold {ACCENT}" if chosen else ""),
        _choices(row, chosen=chosen),
        Text(row.applies, style=DIM),
    )
    if pending is not None:
        about = Text.assemble(
            ("⚠ Turn on? ", f"bold {WARN}"),
            (row.risky, WARN),
            (pending.hint(row), DIM),
        )
    else:
        about = Text(row.about, style="red" if row.dangerous else DIM)
    return Group(line, Padding(about, (0, 0, 0, 4)))


def _heading(title: str, *, first: bool) -> RenderableType:
    rule = Rule(Text(f" {title} ", style="bold"), align="left", style=DIM, characters="─")
    return rule if first else Group(Text(""), rule)


class SettingsList(OptionList):
    """The settings in sections; Enter, → and space move a value on, ← moves it back."""

    BINDINGS: ClassVar = [
        Binding("right,space", "step(1)", "Next value", show=False),
        Binding("left", "step(-1)", "Previous value", show=False),
    ]

    class Stepped(Message):
        """A setting's value was moved on (`by` 1) or back (`by` -1) with ← → or space."""

        def __init__(self, row_id: str, by: int) -> None:
            """Say that the setting `row_id` moved `by` places."""
            super().__init__()
            self.row_id = row_id
            self.by = by

    def __init__(self, *, id: str | None = None) -> None:  # ruff: ignore[builtin-argument-shadowing] - Textual's own name
        """An empty list; `show` fills it."""
        super().__init__(id=id)
        self._rows: dict[str, Row] = {}
        self._widths = (0, 0)
        self._pending: Pending | None = None
        self._pointed: int | None = None

    def show(
        self,
        sections: Sequence[Section],
        highlighted: str | None = None,
        pending: Pending | None = None,
    ) -> None:
        """List `sections`, pointing at the row `highlighted` (else the first one)."""
        self._rows = {row.id: row for section in sections for row in section.rows}
        self._pending = pending
        rows = list(self._rows.values())
        self._widths = (
            max((len(row.name) for row in rows), default=0) + 3,
            max((len(row.applies) for row in rows), default=0) + 1,
        )
        self.clear_options()
        options: list[Option] = []
        for number, section in enumerate(sections):
            options.append(Option(_heading(section.title, first=number == 0), disabled=True))
            options += [Option(self._prompt(row, chosen=False), id=row.id) for row in section.rows]
        self.add_options(options)
        target = highlighted if highlighted in self._rows else (rows[0].id if rows else None)
        if target is not None:
            self.highlighted = self.get_option_index(target)

    @property
    def rows(self) -> list[Row]:
        """Every setting listed, in order."""
        return list(self._rows.values())

    @property
    def row(self) -> Row | None:
        """The setting the keyboard is on."""
        if self.highlighted is None:
            return None
        option = self.get_option_at_index(self.highlighted)
        return None if option.id is None else self._rows.get(option.id)

    def _prompt(self, row: Row, *, chosen: bool) -> Group:
        pending = self._pending if self._pending and self._pending.key == row.id else None
        return row_prompt(row, self._widths, chosen=chosen, pending=pending)

    @override
    def watch_highlighted(self, highlighted: int | None) -> None:
        """Point at the highlighted setting and redraw the one left."""
        previous = self._pointed
        super().watch_highlighted(highlighted)
        for index in {previous, highlighted} - {None}:
            if index is None or index >= self.option_count:
                continue
            option = self.get_option_at_index(index)
            row = None if option.id is None else self._rows.get(option.id)
            if row is not None:
                self.replace_option_prompt_at_index(
                    index, self._prompt(row, chosen=index == highlighted)
                )
        self._pointed = highlighted

    def action_step(self, by: int) -> None:
        """Move the highlighted setting's value on or back."""
        row = self.row
        if row is not None:
            self.post_message(self.Stepped(row.id, by))


def tools_rows(tools: Sequence[ToolSetting]) -> list[tuple[str, tuple[Text, Text, Text]]]:
    """Return each tool's cells: name, setting in colour (with how it ships, if changed), use."""
    lines: list[tuple[str, tuple[Text, Text, Text]]] = []
    for tool in tools:
        state = Text(tool.label, style=_TOOL_STYLES.get(tool.label, ""))
        if tool.state is not tool.shipped:
            state.append(f" (ships {tool.shipped_label})", style=WARN)
        lines.append((tool.name, (Text(tool.name), state, Text(tool.summary, style=DIM))))
    return lines


def tools_summary(tools: Sequence[ToolSetting]) -> str:
    """Return how many tools are at each setting: "30 on · 4 ask first · 1 off"."""
    counts: dict[str, int] = {}
    for tool in tools:
        counts[tool.label] = counts.get(tool.label, 0) + 1
    order = list(_TOOL_STYLES)
    ranked = sorted(
        counts.items(), key=lambda item: order.index(item[0]) if item[0] in order else 9
    )
    return " · ".join(f"{count} {label}" for label, count in ranked)


class ToolsList(OptionList):
    """Every tool on one line: name, setting in colour, what it does; the pointed one lit."""

    def __init__(self, *, id: str | None = None) -> None:  # ruff: ignore[builtin-argument-shadowing] - Textual's own name
        """An empty list; `show` fills it."""
        super().__init__(id=id)
        self._cells: dict[str, tuple[Text, Text, Text]] = {}
        self._widths = (0, 0)
        self._pointed: int | None = None

    def show(self, tools: Sequence[ToolSetting], highlighted: str | None = None) -> None:
        """List `tools` under a heading, pointing at the one named `highlighted` or the first."""
        lines = tools_rows(tools)
        self._cells = dict(lines)
        self._widths = (
            max((cells[0].cell_len for _name, cells in lines), default=0) + 3,
            max((cells[1].cell_len for _name, cells in lines), default=0) + 3,
        )
        heading = (
            Text("Tool", style=DIM),
            Text("Setting", style=DIM),
            Text("What it does", style=DIM),
        )
        self.clear_options()
        self.add_option(Option(self._line(("  ", ""), heading), disabled=True))
        self.add_options(
            Option(self._prompt(name, chosen=False), id=name) for name, _cells in lines
        )
        names = [name for name, _cells in lines]
        if names:
            first = names.index(highlighted) if highlighted in names else 0
            self.highlighted = first + 1  # after the heading

    def _line(self, pointer: tuple[str, str], cells: tuple[Text, Text, Text]) -> Table:
        line = Table.grid(expand=True)
        line.add_column(width=2, min_width=2, no_wrap=True)
        line.add_column(width=self._widths[0], min_width=self._widths[0], no_wrap=True)
        line.add_column(width=self._widths[1], min_width=self._widths[1], no_wrap=True)
        line.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        line.add_row(Text(*pointer), *cells)
        return line

    def _prompt(self, name: str, *, chosen: bool) -> Table:
        cells = self._cells[name]
        if chosen:
            cells = (Text(name, style=f"bold {ACCENT}"), cells[1], cells[2])
        return self._line((f"{CHEVRON} " if chosen else "  ", f"bold {ACCENT}"), cells)

    @override
    def watch_highlighted(self, highlighted: int | None) -> None:
        """Point at the highlighted tool."""
        previous = self._pointed
        super().watch_highlighted(highlighted)
        for index in {previous, highlighted} - {None}:
            if index is not None and index < self.option_count:
                name = self.get_option_at_index(index).id
                if name is not None and name in self._cells:
                    self.replace_option_prompt_at_index(
                        index, self._prompt(name, chosen=index == highlighted)
                    )
        self._pointed = highlighted
