"""`pc_windows`, `pc_window` and `pc_window_close`: the owner's windows, through Hyprland.

`pc_windows` lists the open windows by number; `pc_window` and `pc_window_close` act only on a
number from the latest listing of the same request, and only once the model has read it (a call
written beside the listing, in the same model message, is refused: the number would be a guess).
The number is bound to that window's Hyprland address and title before any question, and
checked again when the call runs. Everything goes through Hyprland's socket (`pc/hyprland.py`):
no program is started. A window list Hyprland does not give is an error, never "no windows".

Hyprland answers `ok` to some dispatches aimed at a window that is gone, so every action is
read back, and said as done only once the window shows the change.

A fullscreen window is left alone: a program such as a game can crash when it is moved,
resized, closed or covered, or when a window is moved onto, laid out beside or focused over it.
So nothing but focusing the fullscreen window itself is done to it, nothing is done to another
window on its workspace, no window is moved there, and no special workspace is opened over a
monitor that shows one. Fullscreen is Hyprland's live state (`fullscreen` or `fullscreenClient`
above 0), read when acting, on any workspace.
"""

import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from jarvis.jsonutil import obj
from jarvis.operations import (
    Operation,
    OperationError,
    Request,
    Result,
    current_request,
    inert,
)
from jarvis.pc import hyprland
from jarvis.pc.apps import BACKGROUND, SPECIAL, WORKSPACES
from jarvis.policy import OpClass, Origin

if TYPE_CHECKING:
    from collections.abc import Callable

MAX_LISTED = 30
MAX_TITLE = 80
MAX_APP = 40
VERIFY_S = 1.5  # how long an action is read back before it is said not to have shown
CLOSE_WAIT_S = 3.0
POLL_S = 0.1
_ADDRESS = re.compile(r"0x[0-9a-f]{1,16}")
STABLE_ID = re.compile(r"[0-9a-f]{1,16}")
ACTIONS = ("focus", "move", "fullscreen_on", "float_on", "float_off")
UNREADABLE = "Hyprland did not give its window list, so nothing was done"


@dataclass(frozen=True, slots=True)
class Win:
    """One mapped window as Hyprland lists it, with where it is and how it is shown."""

    address: str
    stable: str  # Hyprland's stable id, what `grim -T` captures it by ("" when it gave none)
    klass: str
    title: str
    workspace: str  # its name: "3", or "special:jarvis" for the background
    monitor: int
    x: int
    y: int
    width: int
    height: int
    focused: bool = False
    fullscreen: bool = False
    floating: bool = False
    pid: int = 0

    @property
    def app(self) -> str:
        """The program's class as it may be said or shown: printable and bounded."""
        return printable(self.klass, MAX_APP) or "unnamed"


@dataclass(frozen=True, slots=True)
class Monitor:
    """One monitor: its place in the layout, its scale and the workspaces it shows now."""

    id: int
    name: str
    x: int
    y: int
    width: int  # in layout units (pixels divided by the scale)
    height: int
    focused: bool
    workspace: str
    special: str  # the special workspace shown over it, "" when none
    scale: float = 1.0


class Desktop(Protocol):
    """What the window tools need of the compositor."""

    def windows(self) -> list[Win]:
        """Return the mapped windows; raise `OperationError` when they cannot be read."""
        ...

    def monitors(self) -> list[Monitor]:
        """Return the monitors; raise `OperationError` when they cannot be read."""
        ...

    def run(self, lua: str) -> bool:
        """Run one Lua statement; True when the compositor answered `ok`."""
        ...


def _pair(value: object) -> tuple[int, int] | None:
    match value:
        case [int() as first, int() as second]:
            return first, second
        case _:
            return None


def _above_zero(value: object) -> bool:
    return isinstance(value, int) and value > 0


def _win(client: dict[str, object]) -> Win | None:
    """Return one listed window, or None for one that cannot be acted on.

    That is one unmapped (not shown at all) or without an address or workspace. Any other
    field Hyprland leaves out takes a harmless default rather than dropping the
    window, so a fullscreen window is not missed by the guard for an odd field.
    """
    address = client.get("address")
    workspace = obj(client.get("workspace")).get("name")
    if not (
        isinstance(address, str)
        and _ADDRESS.fullmatch(address)
        and isinstance(workspace, str)
        and client.get("mapped") is True
    ):
        return None
    stable, klass, title = client.get("stableId"), client.get("class"), client.get("title")
    monitor, pid = client.get("monitor"), client.get("pid")
    x, y = _pair(client.get("at")) or (0, 0)
    width, height = _pair(client.get("size")) or (0, 0)
    return Win(
        address,
        stable if isinstance(stable, str) and STABLE_ID.fullmatch(stable) else "",
        klass if isinstance(klass, str) else "",
        title if isinstance(title, str) else "",
        workspace,
        monitor if isinstance(monitor, int) else -1,
        x,
        y,
        width,
        height,
        focused=client.get("focusHistoryID") == 0,
        fullscreen=_above_zero(client.get("fullscreen"))
        or _above_zero(client.get("fullscreenClient")),
        floating=client.get("floating") is True,
        pid=pid if isinstance(pid, int) else 0,
    )


def _monitor(found: dict[str, object]) -> Monitor | None:
    ident, name, scale = found.get("id"), found.get("name"), found.get("scale")
    numbers = [found.get(key) for key in ("x", "y", "width", "height")]
    if not (
        isinstance(ident, int)
        and isinstance(name, str)
        and isinstance(scale, int | float)
        and scale > 0
        and all(isinstance(n, int) for n in numbers)
    ):
        return None
    x, y, width, height = (int(n) for n in numbers if isinstance(n, int))
    transform = found.get("transform")
    if isinstance(transform, int) and transform % 2 == 1:  # turned a quarter: sides swap
        width, height = height, width
    return Monitor(
        ident,
        name,
        x,
        y,
        round(width / scale),
        round(height / scale),
        found.get("focused") is True,
        str(obj(found.get("activeWorkspace")).get("name", "")),
        str(obj(found.get("specialWorkspace")).get("name", "")),
        float(scale),
    )


class HyprlandDesktop:
    """Hyprland, through its socket."""

    @staticmethod
    def windows() -> list[Win]:
        """Return Hyprland's mapped windows.

        Raises:
            OperationError: Hyprland did not answer, so what is open is unknown.
        """
        found = hyprland.listing(b"j/clients")
        if found is None:
            raise OperationError(UNREADABLE)
        return [w for w in map(_win, found) if w is not None]

    @staticmethod
    def monitors() -> list[Monitor]:
        """Return Hyprland's monitors.

        Raises:
            OperationError: Hyprland did not answer, or described a monitor it can't be read
                from, so what is on screen is unknown.
        """
        found = hyprland.listing(b"j/monitors")
        monitors = [m for m in map(_monitor, found or []) if m is not None]
        if not found or len(monitors) != len(found):
            msg = "Hyprland did not say what its monitors show, so nothing was done"
            raise OperationError(msg)
        return monitors

    @staticmethod
    def run(lua: str) -> bool:
        """Run `lua` through `hyprctl eval`'s socket request."""
        return hyprland.run_lua(lua)


def place(workspace: str) -> str:
    """Return how the owner names a workspace: its number, or the background."""
    return BACKGROUND if workspace == SPECIAL else workspace


def where(workspace: str) -> str:
    """Return `workspace` in words: "on workspace 2" or "in the background"."""
    named = place(workspace)
    return "in the background" if named == BACKGROUND else f"on workspace {named}"


def shown(window: Win, monitors: list[Monitor]) -> bool:
    """Whether `window` is on screen now: its workspace is one a monitor shows."""
    return any(
        window.workspace in {m.workspace, m.special} for m in monitors if m.id == window.monitor
    )


def fullscreen_on(workspace: str, windows: list[Win], *, but: str = "") -> Win | None:
    """Return the fullscreen window on `workspace` other than the one at `but`, if any."""
    return next(
        (w for w in windows if w.workspace == workspace and w.fullscreen and w.address != but),
        None,
    )


def printable(text: str, limit: int) -> str:
    """Return `text` with control characters blanked, cut to `limit` characters."""
    clean = " ".join("".join(ch if ch.isprintable() else " " for ch in text).split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _order(window: Win) -> tuple[int, int, int, int]:
    named = place(window.workspace)
    number = int(named) if named.isdigit() else 99
    return number, window.monitor, window.y, window.x


@dataclass(slots=True)
class _Listed:
    """The latest windows `pc_windows` listed in one request, by number, and in which message."""

    request: Request | None = None
    message: int = 0
    windows: dict[int, Win] = field(default_factory=dict[int, Win])


class WindowOps:
    """Builds `pc_windows`, `pc_window` and `pc_window_close` over the compositor."""

    def __init__(
        self,
        desktop: Desktop | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Act through `desktop` (Hyprland by default)."""
        super().__init__()
        self.desktop: Desktop = desktop or HyprlandDesktop()
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._listed = _Listed()

    # --- pc_windows -------------------------------------------------------------------------

    def list_windows(self, args: dict[str, Any]) -> Result:
        """List the open windows by number, for the other window tools and `pc_screenshot`.

        Each listing replaces the one before it in the request: its numbers are the ones that
        count.
        """
        wanted = args.get("workspace")
        windows = sorted(self.desktop.windows(), key=_order)
        if wanted is not None:
            target = SPECIAL if wanted == BACKGROUND else str(wanted)
            windows = [w for w in windows if w.workspace == target]
        monitors = self.desktop.monitors()
        listed = dict(enumerate(windows[:MAX_LISTED], start=1))
        request = current_request()
        with self._lock:
            message = request.message if request is not None else 0
            self._listed = record = _Listed(request, message, listed)
        if not listed:
            scope = f" {where(SPECIAL if wanted == BACKGROUND else str(wanted))}" if wanted else ""
            return Result(f"No windows are open{scope}.", Origin.PROCESS)
        lines = ["Open windows (pc_window, pc_window_close and pc_screenshot take the number):"]
        for number, w in listed.items():
            marks = [where(w.workspace), "on screen" if shown(w, monitors) else "not on screen"]
            flags = ((w.focused, "focused"), (w.fullscreen, "fullscreen"), (w.floating, "floating"))
            marks += [word for on, word in flags if on]
            title = printable(w.title, MAX_TITLE)
            lines.append(f'{number}. {w.app}: "{title}" ({", ".join(marks)})')
        rows = dict(zip(listed, lines[1:], strict=True))
        if len(windows) > len(listed):
            lines.append(f"{len(windows) - len(listed)} more are not listed; name a workspace.")
        return Result("\n".join(lines), Origin.PROCESS, delivered=self._reached(record, rows))

    def _reached(self, record: _Listed, rows: dict[int, str]) -> Callable[[str], None]:
        """Return what keeps only the numbers whose whole row reached the model.

        The agent cuts a long result to its cap: a number in a row it cut, or cut short, was
        never read, so no window tool takes it.
        """

        def reached(body: str) -> None:
            with self._lock:
                for number, row in rows.items():
                    if f"\n{inert(row)}" not in body:
                        record.windows.pop(number, None)

        return reached

    def listed(self, number: int) -> Win:
        """Return the window the latest `pc_windows` of this request listed as `number`.

        Raises:
            OperationError: no listing in this request holds the number, or the model has not
                read it yet (the call was written beside the listing).
        """
        request = current_request()
        with self._lock:
            mine = request is not None and self._listed.request is request
            found = self._listed.windows.get(number) if mine else None
            unread = mine and request is not None and request.message <= self._listed.message
        if found is None:
            msg = (
                f"Window {number} was not listed by pc_windows in this request. Call pc_windows "
                "first and use a number it lists."
            )
            raise OperationError(msg)
        if unread:
            msg = (
                "The window list was not read yet: call pc_windows alone, read it, then use a "
                "number from it."
            )
            raise OperationError(msg)
        return found

    def now(self, window: Win) -> tuple[Win, list[Win]]:
        """Return `window` as it is now, with every window.

        Raises:
            OperationError: it is closed, its address now holds another program's window, or
                Hyprland did not give its window list.
        """
        windows = self.desktop.windows()
        current = next((w for w in windows if w.address == window.address), None)
        if current is None or current.klass != window.klass:
            msg = f"The {window.app} window that was listed is no longer open; nothing was done"
            raise OperationError(msg)
        return current, windows

    # --- pc_window --------------------------------------------------------------------------

    @staticmethod
    def _owner_request() -> Request:
        request = current_request()
        if request is None or request.foreign:
            msg = "a window is changed only on the owner's own request, never from PC audio"
            raise OperationError(msg)
        return request

    def _spared(self, window: Win, windows: list[Win], action: str, target: str) -> None:
        """Refuse what could disturb a fullscreen window: see the module's docstring.

        Raises:
            OperationError: the action would touch a fullscreen window, its workspace, or the
                monitor that shows it.
        """
        if window.fullscreen and action != "focus":
            msg = (
                f"The {window.app} window is fullscreen, and a fullscreen program such as a game "
                "can crash when it is changed; nothing was done"
            )
            raise OperationError(msg)
        beside = fullscreen_on(window.workspace, windows, but=window.address)
        if beside is not None:
            msg = (
                f"The {window.app} window shares its workspace with the fullscreen {beside.app}, "
                "which can crash when the windows around it change; nothing was done"
            )
            raise OperationError(msg)
        there = fullscreen_on(target, windows, but=window.address) if target else None
        if there is not None:
            msg = (
                f"{where(target).capitalize()} has the fullscreen {there.app}, which can crash "
                "when a window joins it; nothing was moved"
            )
            raise OperationError(msg)
        if action == "focus" and window.workspace.startswith("special:"):
            showing = {m.workspace for m in self.desktop.monitors()}
            covered = next((w for w in windows if w.fullscreen and w.workspace in showing), None)
            if covered is not None:
                msg = (
                    f"Showing the {window.app} window would cover the fullscreen {covered.app}, "
                    "which can crash; nothing was done"
                )
                raise OperationError(msg)

    @staticmethod
    def _target(action: str, workspace: object) -> str:
        if action != "move":
            return ""
        return SPECIAL if workspace == BACKGROUND else str(workspace)

    def bind(self, args: dict[str, Any]) -> dict[str, Any]:
        """Bind the window number to the window it named, checked against what it may do.

        Raises:
            OperationError: not the owner's own request, a number not listed (or not yet read)
                in it, a window closed since, a move without a workspace, or one a fullscreen
                window forbids.
        """
        self._owner_request()
        action = str(args["action"])
        workspace = args.get("workspace")
        if action == "move" and workspace is None:
            msg = "Say which workspace to move the window to."
            raise OperationError(msg)
        target = self._target(action, workspace)
        window, windows = self.now(self.listed(int(args["window"])))
        self._spared(window, windows, action, target)
        return {
            "window": int(args["window"]),
            "action": action,
            **({"workspace": workspace} if action == "move" else {}),
            "address": window.address,
            "app": window.app,
            "title": printable(window.title, MAX_TITLE),
        }

    @staticmethod
    def _lua(action: str, address: str, target: str) -> str:
        window = hyprland.lua_string(f"address:{address}")
        if action == "focus":
            return f"hl.dispatch(hl.dsp.focus({{ window = {window} }}))"
        if action == "move":
            place_lua = hyprland.lua_string(target) if target == SPECIAL else target
            return (
                f"hl.dispatch(hl.dsp.window.move({{ workspace = {place_lua}, follow = false, "
                f"window = {window} }}))"
            )
        if action == "fullscreen_on":
            return (
                'hl.dispatch(hl.dsp.window.fullscreen({ mode = "fullscreen", action = "set", '
                f"window = {window} }}))"
            )
        how = "on" if action == "float_on" else "off"
        return f'hl.dispatch(hl.dsp.window.float({{ action = "{how}", window = {window} }}))'

    @staticmethod
    def _done(window: Win, action: str, target: str) -> bool:
        return {
            "focus": window.focused,
            "move": window.workspace == target,
            "fullscreen_on": window.fullscreen,
            "float_on": window.floating,
            "float_off": not window.floating,
        }[action]

    def _shows(self, address: str, check: Callable[[Win | None], bool], wait_s: float) -> bool:
        """Read the window back until `check` holds for it, for at most `wait_s`."""
        deadline = self._clock() + wait_s
        while True:
            try:
                windows: list[Win] | None = self.desktop.windows()
            except OperationError:
                windows = None  # an unreadable list proves nothing either way
            if windows is not None and check(
                next((w for w in windows if w.address == address), None)
            ):
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(POLL_S)

    def act(self, args: dict[str, Any]) -> Result:
        """Do the bound action, if the window is still the one listed, and read it back.

        Raises:
            OperationError: what `bind` refuses, checked again now, or Hyprland did not take it.
        """
        bound = args if "address" in args else self.bind(args)
        action = str(bound["action"])
        target = self._target(action, bound.get("workspace"))
        window, windows = self.now(self.listed(int(bound["window"])))
        if window.address != bound["address"]:
            msg = "That window number now names another window; nothing was done"
            raise OperationError(msg)
        self._spared(window, windows, action, target)
        said = _said(action, window.app, target)
        if self._done(window, action, target):
            return Result(f"The {window.app} window is already {said[1]}.", Origin.PROCESS)
        if not self.desktop.run(self._lua(action, window.address, target)):
            msg = f"Hyprland did not take the command, so the {window.app} window was not changed"
            raise OperationError(msg)
        if self._shows(
            window.address, lambda w: w is not None and self._done(w, action, target), VERIFY_S
        ):
            return Result(f"{said[0]}.", Origin.PROCESS)
        return Result(
            f"Asked Hyprland to {said[2]}, but the window does not show it yet.",
            Origin.PROCESS,
            ok=False,
            acted=True,
        )

    # --- pc_window_close --------------------------------------------------------------------

    def bind_close(self, args: dict[str, Any]) -> dict[str, Any]:
        """Bind the number to the window it named; only on the owner's own words.

        Raises:
            OperationError: not the owner's own request, outside text in it, a number not
                listed (or not yet read) in it, a closed window, a fullscreen window, or one
                beside a fullscreen window.
        """
        request = self._owner_request()
        outside = request.outside - {Origin.PROCESS.value}
        if outside:
            msg = (
                "a window is closed only on the owner's own words, and this request also holds "
                f"outside text ({', '.join(sorted(outside))}); nothing was closed. Ask the owner "
                "to say which window to close in a request of its own."
            )
            raise OperationError(msg)
        window, windows = self.now(self.listed(int(args["window"])))
        if window.fullscreen:
            msg = (
                f"The {window.app} window is fullscreen, and a fullscreen program such as a game "
                "must not be interrupted; nothing was closed"
            )
            raise OperationError(msg)
        beside = fullscreen_on(window.workspace, windows, but=window.address)
        if beside is not None:
            msg = (
                f"The {window.app} window shares its workspace with the fullscreen {beside.app}, "
                "which can crash when the windows around it change; nothing was closed"
            )
            raise OperationError(msg)
        return {
            "window": int(args["window"]),
            "address": window.address,
            "app": window.app,
            "title": printable(window.title, MAX_TITLE),
        }

    def close(self, args: dict[str, Any]) -> Result:
        """Ask the bound window to close, as its close button does, and say whether it went.

        Raises:
            OperationError: what `bind_close` refuses, checked again now, a window that shows
                another title than the one asked about, or Hyprland did not take it.
        """
        bound = args if "address" in args else self.bind_close(args)
        fresh = self.bind_close({"window": bound["window"]})
        if fresh["address"] != bound["address"]:
            msg = "That window number now names another window; nothing was closed"
            raise OperationError(msg)
        if fresh["title"] != bound["title"]:
            msg = (
                f'The {fresh["app"]} window now shows "{fresh["title"]}", not what the owner was '
                "asked about; nothing was closed. Ask again."
            )
            raise OperationError(msg)
        address, app = str(bound["address"]), str(bound["app"])
        window = hyprland.lua_string(f"address:{address}")
        if not self.desktop.run(f"hl.dispatch(hl.dsp.window.close({{ window = {window} }}))"):
            msg = f"Hyprland did not take the command, so the {app} window was not closed"
            raise OperationError(msg)
        if self._shows(address, lambda w: w is None, CLOSE_WAIT_S):
            return Result(f"Closed the {app} window.", Origin.PROCESS)
        return Result(
            f"Asked the {app} window to close, but it is still open after {CLOSE_WAIT_S:g} "
            "seconds; it may be asking to save. Tell the owner.",
            Origin.PROCESS,
            ok=False,
            acted=True,
        )

    def operations(self) -> list[Operation]:
        """Return `pc_windows`, `pc_window` and `pc_window_close`."""
        number = {"type": "integer", "minimum": 1, "maximum": MAX_LISTED}
        places = {"type": "string", "enum": [*WORKSPACES, BACKGROUND]}
        return [
            Operation(
                "pc_windows",
                "List the open windows by number: program, title, workspace, whether it shows.",
                {
                    "type": "object",
                    "properties": {"workspace": places},
                    "additionalProperties": False,
                },
                self.list_windows,
                side_effect=False,
                spoken_summary=lambda _a: "list the open windows",
            ),
            Operation(
                "pc_window",
                "Change a window pc_windows listed in this request, as the owner asks: focus "
                "shows it, move sends it to workspace.",
                {
                    "type": "object",
                    "properties": {
                        "window": number,
                        "action": {"type": "string", "enum": list(ACTIONS)},
                        "workspace": places,
                    },
                    "required": ["window", "action"],
                    "additionalProperties": False,
                },
                self.act,
                spoken_summary=_summary,
                prepare=self.bind,
            ),
            Operation(
                "pc_window_close",
                "Close a window pc_windows listed in this request, as its close button does.",
                {
                    "type": "object",
                    "properties": {"window": number},
                    "required": ["window"],
                    "additionalProperties": False,
                },
                self.close,
                default_class=OpClass.CONFIRM,
                spoken_summary=lambda a: (
                    f"close the {a.get('app', 'listed')} window"
                    + (f' "{a["title"]}"' if a.get("title") else "")
                ),
                prepare=self.bind_close,
            ),
        ]


def _said(action: str, app: str, target: str) -> tuple[str, str, str]:
    """Return what was done, the state it leaves, and what was asked, in words."""
    return {
        "focus": (f"Showed the {app} window", "focused", f"show the {app} window"),
        "move": (
            f"Moved the {app} window {where(target)}",
            where(target),
            f"move the {app} window {where(target)}",
        ),
        "fullscreen_on": (
            f"Made the {app} window fullscreen",
            "fullscreen",
            f"make the {app} window fullscreen",
        ),
        "float_on": (f"Made the {app} window float", "floating", f"make the {app} window float"),
        "float_off": (f"Tiled the {app} window", "tiled", f"tile the {app} window"),
    }[action]


def _summary(args: dict[str, Any]) -> str:
    action, app = str(args.get("action", "")), str(args.get("app", "listed"))
    target = (
        ""
        if action != "move"
        else SPECIAL
        if args.get("workspace") == BACKGROUND
        else str(args.get("workspace"))
    )
    said = _said(action, app, target)[2] if action in ACTIONS else "change a window"
    title = args.get("title")
    return f'{said} ("{title}")' if title else said
