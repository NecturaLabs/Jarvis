"""`open_app`: open one program the owner asked for, on the workspace the owner chose.

Jarvis never starts a program on its own: only `open_app` does, only for a program the owner's
`[apps]` lists, and only in a request of the owner's own (never one heard in PC audio). Where it
opens is the owner's to say, a workspace from 1 to 10 or the background (Hyprland's special
workspace `jarvis`, out of sight), so the operation takes no default and the model must have asked.

The program is started by Hyprland itself (`hl.exec_cmd` with a `[workspace N silent]` rule, sent
through its socket), so it opens there without taking focus. Jarvis then reads the windows back
until a new window of the program's (known by its class) shows, moves it if it landed elsewhere,
looks again a moment later (some programs replace a first window) and says where it is only once
it saw it there. A window that was open before is never moved, and nothing opens on a workspace
that has a fullscreen window: a game there can crash when a window joins it.
"""

import re
import shlex
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from jarvis.jsonutil import obj
from jarvis.operations import Operation, OperationError, Result, current_request
from jarvis.pc import hyprland
from jarvis.policy import OpClass, Origin

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jarvis.config import AppConfig

WORKSPACES = tuple(str(number) for number in range(1, 11))
BACKGROUND = "background"
SPECIAL = "special:jarvis"  # the background: a special workspace, shown only when toggled
OPEN_WAIT_S = 15.0
SETTLE_S = 1.0
POLL_S = 0.25
UNREADABLE = "Hyprland did not give its window list, so nothing was opened"


@dataclass(frozen=True, slots=True)
class Placed:
    """One window as Hyprland lists it: its address, workspace name, class and title."""

    address: str
    workspace: str
    klass: str
    title: str
    fullscreen: bool = False


class Compositor(Protocol):
    """What `open_app` needs of the compositor."""

    def windows(self) -> list[Placed]:
        """Return the windows now open; raise `OperationError` when they cannot be read."""
        ...

    def showing(self) -> set[str]:
        """Return the workspaces the monitors show now; raise `OperationError` if unknown."""
        ...

    def run(self, lua: str) -> bool:
        """Run one Lua statement; True when the compositor took it."""
        ...


class HyprlandCompositor:
    """Hyprland, through its socket (`pc/hyprland.py`)."""

    @staticmethod
    def windows() -> list[Placed]:
        """Return Hyprland's windows.

        Raises:
            OperationError: Hyprland did not answer, so what is open is unknown.
        """
        clients = hyprland.listing(b"j/clients")
        if clients is None:
            raise OperationError(UNREADABLE)
        placed: list[Placed] = []
        for client in clients:
            address, klass, title = client.get("address"), client.get("class"), client.get("title")
            name = obj(client.get("workspace")).get("name")
            full = [client.get(key) for key in ("fullscreen", "fullscreenClient")]
            # An odd field takes a harmless default: a fullscreen window is never missed.
            placed.append(
                Placed(
                    address if isinstance(address, str) else "",
                    name if isinstance(name, str) else "",
                    klass if isinstance(klass, str) else "",
                    title if isinstance(title, str) else "",
                    fullscreen=any(isinstance(f, int) and f > 0 for f in full),
                )
            )
        return placed

    @staticmethod
    def showing() -> set[str]:
        """Return the workspaces Hyprland's monitors show now, special ones included.

        Raises:
            OperationError: Hyprland did not answer.
        """
        monitors = hyprland.listing(b"j/monitors")
        if not monitors:
            raise OperationError(UNREADABLE)
        shown: set[str] = set()
        for monitor in monitors:
            for key in ("activeWorkspace", "specialWorkspace"):
                name = obj(monitor.get(key)).get("name")
                if isinstance(name, str) and name:
                    shown.add(name)
        return shown

    @staticmethod
    def run(lua: str) -> bool:
        """Run `lua` through `hyprctl eval`'s socket request."""
        return hyprland.run_lua(lua)


def _spare_fullscreen(windows: list[Placed], workspace: str, outcome: str) -> None:
    """Refuse to put a window on a workspace with a fullscreen one (a game there can crash).

    Raises:
        OperationError: `workspace` has a fullscreen window.
    """
    game = next((w for w in windows if w.workspace == workspace and w.fullscreen), None)
    if game is not None:
        msg = (
            f"{_where(_place_named(workspace)).capitalize()} has the fullscreen {game.klass}, "
            f"which can crash when a window opens beside it; {outcome}. Ask the owner for another "
            "workspace"
        )
        raise OperationError(msg)


def _where(place: str) -> str:
    return "in the background" if place == BACKGROUND else f"on workspace {place}"


def _place_named(workspace: str) -> str:
    """Return how the owner names a Hyprland workspace: its number, or the background."""
    return BACKGROUND if workspace == SPECIAL else workspace


class AppOps:
    """Builds `open_app` over the owner's `[apps]` list and the compositor."""

    def __init__(
        self,
        apps: Mapping[str, AppConfig],
        compositor: Compositor | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Open only what `apps` lists, through `compositor` (Hyprland by default)."""
        super().__init__()
        self._apps = dict(apps)
        self._compositor = compositor or HyprlandCompositor()
        self._clock = clock
        self._sleep = sleep

    def _mine(self, name: str) -> Callable[[Placed], bool]:
        pattern = re.compile(self._apps[name].window, re.IGNORECASE)
        return lambda w: bool(pattern.search(w.klass))

    def open_app(self, args: dict[str, Any]) -> Result:
        """Open the program on the place the owner chose, and say where it is once seen there.

        Raises:
            OperationError: not the owner's own request, or Hyprland did not take the command.
        """
        request = current_request()
        if request is None or request.foreign:
            msg = "a program is opened only on the owner's own request, never from PC audio"
            raise OperationError(msg)
        name, place = str(args["app"]), str(args["where"])
        mine = self._mine(name)
        windows = self._compositor.windows()
        already = [w for w in windows if mine(w)]
        if already:  # by class: a browser tab named after the program is not the program
            where = _where(_place_named(already[0].workspace))
            text = f"{name} is already open {where}; nothing was opened."
            return Result(text, Origin.TOOL_STATUS, spoken=f"{name} is already open {where}.")
        workspace = SPECIAL if place == BACKGROUND else place
        _spare_fullscreen(windows, workspace, f"{name} was not opened")
        # A program can ignore where it was sent and open where the owner is looking.
        for shown in sorted(self._compositor.showing()):
            _spare_fullscreen(windows, shown, f"{name} was not opened, as it could open there")
        rule = f"[workspace {workspace} silent] {shlex.join(self._apps[name].command)}"
        if not self._compositor.run(f"hl.exec_cmd({hyprland.lua_string(rule)})"):
            msg = f"Hyprland did not take the command, so {name} was not opened"
            raise OperationError(msg)
        new = mine  # none of its windows was open, so each one that shows is the new program
        if self._wait(new) is None:
            text = (
                f"Started {name}, but no window of it showed within {OPEN_WAIT_S:g} seconds; it "
                "may still be starting."
            )
            return Result(text, Origin.TOOL_STATUS, ok=False, acted=True)
        self._gather(new, workspace)
        self._sleep(SETTLE_S)
        strays = self._gather(new, workspace)
        if strays is None:
            text = f"Started {name}, but Hyprland did not say where its window is now."
            return Result(text, Origin.TOOL_STATUS, ok=False, acted=True)
        if strays:
            seen = ", ".join(sorted({_where(_place_named(w.workspace)) for w in strays}))
            text = f"Opened {name}, but a window of it stayed {seen}, not {_where(place)}."
            return Result(text, Origin.TOOL_STATUS, ok=False, acted=True)
        sentence = f"Opened {name} {_where(place)}."
        return Result(sentence, Origin.TOOL_STATUS, spoken=sentence)

    def _wait(self, mine: Callable[[Placed], bool]) -> Placed | None:
        """Return the program's first window once it shows, None after `OPEN_WAIT_S`."""
        deadline = self._clock() + OPEN_WAIT_S
        while True:
            found = [w for w in self._read() if mine(w) and not w.fullscreen]
            if found:
                return found[0]
            if self._clock() >= deadline:
                return None
            self._sleep(POLL_S)

    def _read(self) -> list[Placed]:
        """Return the windows, [] while Hyprland does not answer (the program was started)."""
        try:
            return self._compositor.windows()
        except OperationError:
            return []

    def _gather(self, mine: Callable[[Placed], bool], workspace: str) -> list[Placed] | None:
        """Move the program's windows that are elsewhere to `workspace`; return those still out.

        A fullscreen window, and any window on a workspace that has one, is never moved, and
        nothing is moved onto a workspace that has one now. None: Hyprland did not answer.
        """
        target = hyprland.lua_string(workspace) if workspace == SPECIAL else workspace
        windows = self._read()
        held = {w.workspace for w in windows if w.fullscreen}
        for window in [w for w in windows if mine(w)]:
            moving = window.workspace != workspace and not window.fullscreen
            if moving and window.workspace not in held and workspace not in held:
                address = hyprland.lua_string(f"address:{window.address}")
                self._compositor.run(
                    "hl.dispatch(hl.dsp.window.move({ workspace = "
                    f"{target}, follow = false, window = {address} }}))"
                )
        try:
            after = self._compositor.windows()
        except OperationError:
            return None
        return [w for w in after if mine(w) and w.workspace != workspace]

    def operations(self) -> list[Operation]:
        """Return `open_app` over the listed programs; none when the list is empty."""
        if not self._apps:
            return []
        return [
            Operation(
                "open_app",
                "Open a program, only when the owner asked to open it in this request; never to "
                "do something else (if Spotify is closed, say so and ask first). `where` is the "
                "workspace the owner named, or background (running, out of sight): if the owner "
                "did not say where, ask; never choose yourself.",
                {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "enum": sorted(self._apps)},
                        "where": {"type": "string", "enum": [*WORKSPACES, BACKGROUND]},
                    },
                    "required": ["app", "where"],
                    "additionalProperties": False,
                },
                self.open_app,
                default_class=OpClass.SAFE,
                spoken_summary=lambda a: (
                    f"open {a.get('app', 'a program')} {_where(str(a.get('where', '')))}"
                ),
            ),
        ]
