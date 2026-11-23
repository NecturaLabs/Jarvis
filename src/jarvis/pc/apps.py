"""`open_app`: open one program the owner asked for, on the workspace the owner chose.

Jarvis never starts a program on its own: only `open_app` does, only for a program the owner's
`[apps]` lists, and only in a request of the owner's own (never one heard in PC audio). Where it
opens is the owner's to say, a workspace from 1 to 10 or the background (Hyprland's special
workspace `jarvis`, out of sight), so the operation takes no default and the model must have asked.

The program is started by Hyprland itself (`hl.exec_cmd` with a `[workspace N silent]` rule, sent
through its socket), so it opens there without taking focus. Jarvis then reads the windows back
until a new window of the program's (known by its class) shows, moves it if it landed elsewhere,
looks again a moment later (some programs replace a first window) and says where it is only once
it saw it there. A window that was open before is never moved.
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


@dataclass(frozen=True, slots=True)
class Placed:
    """One window as Hyprland lists it: its address, workspace name, class and title."""

    address: str
    workspace: str
    klass: str
    title: str


class Compositor(Protocol):
    """What `open_app` needs of the compositor."""

    def windows(self) -> list[Placed]:
        """Return the windows now open."""
        ...

    def run(self, lua: str) -> bool:
        """Run one Lua statement; True when the compositor took it."""
        ...


class HyprlandCompositor:
    """Hyprland, through its socket (`pc/hyprland.py`)."""

    @staticmethod
    def windows() -> list[Placed]:
        """Return Hyprland's windows; [] when it cannot answer."""
        placed: list[Placed] = []
        for client in hyprland.clients():
            address, klass, title = client.get("address"), client.get("class"), client.get("title")
            name = obj(client.get("workspace")).get("name")
            if all(isinstance(v, str) for v in (address, klass, title, name)):
                placed.append(Placed(str(address), str(name), str(klass), str(title)))
        return placed

    @staticmethod
    def run(lua: str) -> bool:
        """Run `lua` through `hyprctl eval`'s socket request."""
        return hyprland.run_lua(lua)


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
        already = [w for w in self._compositor.windows() if mine(w)]
        if already:  # by class: a browser tab named after the program is not the program
            where = _where(_place_named(already[0].workspace))
            text = f"{name} is already open {where}; nothing was opened."
            return Result(text, Origin.TOOL_STATUS, spoken=f"{name} is already open {where}.")
        workspace = SPECIAL if place == BACKGROUND else place
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
            found = [w for w in self._compositor.windows() if mine(w)]
            if found:
                return found[0]
            if self._clock() >= deadline:
                return None
            self._sleep(POLL_S)

    def _gather(self, mine: Callable[[Placed], bool], workspace: str) -> list[Placed]:
        """Move the program's windows that are elsewhere to `workspace`; return those still out."""
        target = hyprland.lua_string(workspace) if workspace == SPECIAL else workspace
        for window in [w for w in self._compositor.windows() if mine(w)]:
            if window.workspace != workspace:
                address = hyprland.lua_string(f"address:{window.address}")
                self._compositor.run(
                    "hl.dispatch(hl.dsp.window.move({ workspace = "
                    f"{target}, follow = false, window = {address} }}))"
                )
        return [w for w in self._compositor.windows() if mine(w) and w.workspace != workspace]

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
