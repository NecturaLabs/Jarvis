"""`pc_watch`, `pc_watches` and `pc_watch_stop`: tell the owner when something shows on a screen.

The owner asks, in any words, to be told when something shows up in a window, a workspace or on
the screen; the model starts a watch with its own short description of what to look for. While
the voice service is idle it takes a picture now and then (the same capture as `pc_screenshot`:
`grim` only reads frames, nothing is focused, moved or resized) and asks the model one question
about it: no tools, no conversation, the picture marked as outside data, an answer of YES or NO
(`Judge.glance`). Anything but a leading YES is a no. On a YES Jarvis says a fixed line built from
the condition, never words read off the screen.

Each check uses the GPU, so a watch is bounded: it checks at most every 10 seconds, ends after
`for_min` minutes (at most 12 hours), and at most two run at once. A watch lives in the voice
service's process (`jarvis run`) and ends with it; the other channels do not offer it. Checks are
made only between requests, never while Jarvis speaks or is paused (`voice.py` decides when).
"""

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from jarvis.agent import PICTURES_OFF
from jarvis.operations import Operation, OperationError, Result, current_request
from jarvis.pc.apps import BACKGROUND, SPECIAL, WORKSPACES
from jarvis.pc.windows import MAX_LISTED, Win, WindowOps, printable, where
from jarvis.pictures import SHOWN_SIDE, PictureError, prepare
from jarvis.policy import Origin

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.pc.screenshot import ScreenshotOps
    from jarvis.pictures import Picture

EVERY_DEFAULT_S = 20
EVERY_MIN_S = 10
EVERY_MAX_S = 300
FOR_DEFAULT_MIN = 120
FOR_MAX_MIN = 720
MAX_WATCHES = 2
COOL_DOWN_MIN_S = (
    60.0  # after an announcement: at least this long, or three checks, whichever is more
)
COOL_DOWN_CHECKS = 3
MAX_CONDITION = 120
FAILURES_END = 3  # checks that fail one after another end a watch
KINDS = ("window", "workspace", "screen")
NEEDS_PICTURES = f"I can't watch the screen. {PICTURES_OFF}"
NOT_HERE = "A watch runs only in the voice service, so it can't be set from here."


class Judge(Protocol):
    """The model's side of a watch: can it see pictures, and does one show a condition."""

    def sees(self) -> bool:
        """Whether the model can see pictures now."""
        ...

    def glance(self, picture: Picture, condition: str) -> bool:
        """Return whether `picture` shows `condition`, from one model call without tools."""
        ...


class _Audit(Protocol):
    def record(self, event: str, **fields: object) -> None: ...


@dataclass(slots=True)
class Watch:
    """One running watch: what it looks at, for what, how often and until when."""

    ident: int
    kind: str
    condition: str
    every_s: float
    repeat: bool
    ends_at: float
    next_at: float
    window: Win | None = None
    workspace: str = ""
    quiet_until: float = 0.0
    failures: int = 0
    announced: int = 0
    active: bool = True

    def of(self) -> str:
        """Return what it looks at, in words."""
        if self.kind == "window" and self.window is not None:
            return f"the {self.window.app} window"
        if self.kind == "workspace":
            return f"the windows {where(self.workspace)}"
        return "the screen"


def _condition(text: str) -> str:
    """Return the model's description of what to look for as one short line."""
    return printable(text, MAX_CONDITION).rstrip(" .!?,;:")


@dataclass(slots=True)
class _Shared:
    watches: dict[int, Watch] = field(default_factory=dict[int, Watch])
    free: int = 1
    checking: bool = False


class ScreenWatch:
    """Builds the watch operations and makes the checks the voice service asks for."""

    def __init__(
        self,
        windows: WindowOps,
        shots: ScreenshotOps,
        *,
        clock: Callable[[], float] = time.monotonic,
        audit: _Audit | None = None,
    ) -> None:
        """Capture through `shots`, find windows through `windows`, read time from `clock`."""
        super().__init__()
        self._windows = windows
        self._shots = shots
        self._clock = clock
        self._audit = audit
        self._judge: Judge | None = None
        self._lock = threading.Lock()
        self._shared = _Shared()

    @property
    def windows(self) -> WindowOps:
        """The window tools the watches bind their windows through (shared with `pc_windows`)."""
        return self._windows

    def attach(self, judge: Judge) -> None:
        """Let the model judge the pictures (it is built after the operations)."""
        self._judge = judge

    def _record(self, event: str, watch: Watch, **fields: object) -> None:
        if self._audit is not None:
            self._audit.record(
                event, watch=watch.ident, of=watch.kind, condition=watch.condition, **fields
            )

    # --- the operations ---------------------------------------------------------------------

    def start(self, args: dict[str, Any]) -> Result:
        """Start a watch on a window, a workspace or the screen.

        Raises:
            OperationError: not the owner's own request, the model cannot see pictures, two
                watches already run, or the window or workspace is not given (or not listed in
                this request, as for the window tools).
        """
        request = current_request()
        if request is None or request.foreign:
            msg = "a watch is set only on the owner's own request, never from PC audio"
            raise OperationError(msg)
        if self._judge is None or not self._judge.sees():
            raise OperationError(NEEDS_PICTURES)
        with self._lock:
            if len(self._shared.watches) >= MAX_WATCHES:
                msg = "Two watches already run. Stop one first (pc_watches lists them)."
                raise OperationError(msg)
        kind = str(args["of"])
        window: Win | None = None
        workspace = ""
        if kind == "window":
            if args.get("window") is None:
                msg = "Say which window: a number pc_windows listed in this request."
                raise OperationError(msg)
            window, _ = self._windows.now(self._windows.listed(int(args["window"])))
        elif kind == "workspace":
            if args.get("workspace") is None:
                msg = "Say which workspace to watch."
                raise OperationError(msg)
            wanted = args["workspace"]
            workspace = SPECIAL if wanted == BACKGROUND else str(wanted)
        condition = _condition(str(args["condition"]))
        if not condition:
            msg = "Say what to look for."
            raise OperationError(msg)
        now = self._clock()
        every_s = float(args["every_s"])
        with self._lock:
            if len(self._shared.watches) >= MAX_WATCHES:
                msg = "Two watches already run. Stop one first (pc_watches lists them)."
                raise OperationError(msg)
            watch = Watch(
                self._shared.free,
                kind,
                condition,
                every_s,
                bool(args["repeat"]),
                now + float(args["for_min"]) * 60,
                now,  # the first look is at once
                window,
                workspace,
            )
            self._shared.free += 1
            self._shared.watches[watch.ident] = watch
        self._record(
            "watch_started", watch, every_s=every_s, for_min=args["for_min"], repeat=watch.repeat
        )
        again = "each time it shows" if watch.repeat else "the first time it shows"
        return Result(
            f"Watching {watch.of()} for: {condition}. Checking every {every_s:g} s for "
            f"{args['for_min']} min (each check uses the GPU); I'll say {again}. "
            f"Stop it with pc_watch_stop (watch {watch.ident}).",
            Origin.PROCESS if kind == "window" else Origin.TOOL_STATUS,
        )

    @staticmethod
    def _left(watch: Watch, now: float) -> str:
        minutes = max(round((watch.ends_at - now) / 60), 1)
        return f"{minutes} min left"

    def list_watches(self, _args: dict[str, Any]) -> Result:
        """List the running watches."""
        now = self._clock()
        with self._lock:
            watches = list(self._shared.watches.values())
        if not watches:
            return Result("No watches are running.")
        lines = [
            f"{w.ident}. {w.of()} for: {w.condition} (every {w.every_s:g} s, {self._left(w, now)}, "
            f"{'each time' if w.repeat else 'once'}, said {w.announced}x)"
            for w in watches
        ]
        return Result("\n".join(lines), Origin.PROCESS)

    def stop(self, args: dict[str, Any]) -> Result:
        """Stop one watch, or all when none is named."""
        wanted = args.get("watch")
        with self._lock:
            found = [
                w for w in self._shared.watches.values() if wanted is None or w.ident == wanted
            ]
            for watch in found:
                self._end(watch)
        if not found:
            return Result(
                "No such watch." if wanted is not None else "No watches are running.", ok=False
            )
        for watch in found:
            self._record("watch_stopped", watch, reason="owner")
        names = ", ".join(str(w.ident) for w in found)
        return Result(f"Stopped watch {names}.")

    def _end(self, watch: Watch) -> None:
        """Take `watch` out (the lock is held)."""
        watch.active = False
        self._shared.watches.pop(watch.ident, None)

    def operations(self) -> list[Operation]:
        """Return `pc_watch`, `pc_watches` and `pc_watch_stop`."""
        return [
            Operation(
                "pc_watch",
                "Tell the owner by voice when something shows up in a window pc_windows listed, "
                "a workspace or the screen. condition: what to look for, in a few words. Each "
                "check uses the GPU; it ends after for_min.",
                {
                    "type": "object",
                    "properties": {
                        "of": {"type": "string", "enum": list(KINDS)},
                        "window": {"type": "integer", "minimum": 1, "maximum": MAX_LISTED},
                        "workspace": {"type": "string", "enum": [*WORKSPACES, BACKGROUND]},
                        "condition": {"type": "string", "minLength": 1, "maxLength": MAX_CONDITION},
                        "every_s": {
                            "type": "integer",
                            "minimum": EVERY_MIN_S,
                            "maximum": EVERY_MAX_S,
                            "default": EVERY_DEFAULT_S,
                        },
                        "for_min": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": FOR_MAX_MIN,
                            "default": FOR_DEFAULT_MIN,
                        },
                        "repeat": {"type": "boolean", "default": True},
                    },
                    "required": ["of", "condition"],
                    "additionalProperties": False,
                },
                self.start,
                spoken_summary=lambda a: f"watch for {a.get('condition', 'something')}",
            ),
            Operation(
                "pc_watches",
                "List the running screen watches.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                self.list_watches,
                side_effect=False,
                spoken_summary=lambda _a: "list the watches",
            ),
            Operation(
                "pc_watch_stop",
                "Stop a screen watch by its number, or all of them.",
                {
                    "type": "object",
                    "properties": {"watch": {"type": "integer", "minimum": 1}},
                    "additionalProperties": False,
                },
                self.stop,
                spoken_summary=lambda _a: "stop watching",
            ),
        ]

    # --- the checks, for the voice service --------------------------------------------------

    def ready(self) -> bool:
        """Whether a watch is due a check or its end now (cheap: asked between audio frames)."""
        now = self._clock()
        with self._lock:
            return not self._shared.checking and any(
                w.next_at <= now or w.ends_at <= now for w in self._shared.watches.values()
            )

    def check_due(self) -> list[str]:
        """Check every watch that is due; return the lines Jarvis is to say, in order.

        Each check takes a picture and makes one model call. The caller makes sure nothing else
        is happening (no request, nothing spoken, not paused).
        """
        now = self._clock()
        with self._lock:
            if self._shared.checking:
                return []
            due = [w for w in self._shared.watches.values() if w.next_at <= now or w.ends_at <= now]
            self._shared.checking = True
        try:
            lines = [line for watch in due if (line := self._check(watch)) is not None]
        finally:
            with self._lock:
                self._shared.checking = False
        return lines

    def _finish(self, watch: Watch, reason: str) -> None:
        with self._lock:
            self._end(watch)
        self._record("watch_stopped", watch, reason=reason)

    def _check(self, watch: Watch) -> str | None:
        """Make one check of `watch`; return what to say, or None."""
        now = self._clock()
        if now >= watch.ends_at:
            self._finish(watch, "expired")
            return f"My watch for {watch.condition} is over."
        watch.next_at = now + watch.every_s
        gone = self._gone(watch)
        if gone:
            self._finish(watch, "window_gone")
            return f"The window I was watching for {watch.condition} is gone, so I've stopped."
        try:
            shown = self._look(watch)
        except Exception as exc:  # ruff: ignore[blind-except] - a failed capture or model call is a failed check, never a crash of the voice loop
            return self._failed(watch, str(exc))
        watch.failures = 0
        if not shown or not watch.active:
            return None
        if now < watch.quiet_until:
            return None  # said not long ago
        watch.announced += 1
        watch.quiet_until = now + max(COOL_DOWN_MIN_S, COOL_DOWN_CHECKS * watch.every_s)
        self._record("watch_announced", watch, count=watch.announced)
        if not watch.repeat:
            self._finish(watch, "done")
        return f"Heads up: {watch.condition}."

    def _gone(self, watch: Watch) -> bool:
        """Whether the watched window is closed (or now another program's): its watch ends."""
        window = watch.window
        if window is None:
            return False
        try:
            windows = self._windows.desktop.windows()
        except OperationError:
            return False  # not read just now: the capture fails the same way, and is counted
        current = next((w for w in windows if w.address == window.address), None)
        if current is None or current.klass != window.klass:
            return True
        watch.window = current
        return False

    def _look(self, watch: Watch) -> bool:
        """Take the picture and ask the model about it; any failure is the caller's to count.

        Raises:
            PictureError: the model cannot see pictures now.
        """
        judge = self._judge
        if judge is None or not judge.sees():
            msg = "the picture reader is off"
            raise PictureError(msg)
        data = self._shots.capture(watch.kind, workspace=watch.workspace, window=watch.window)
        picture = prepare(data, "watch.png", max_side=SHOWN_SIDE)
        return judge.glance(picture, watch.condition)

    def _failed(self, watch: Watch, why: str) -> str | None:
        """Count a check that did not finish; three in a row end the watch, said once."""
        watch.failures += 1
        self._record("watch_check_failed", watch, error=why[:200], failures=watch.failures)
        if watch.failures < FAILURES_END:
            return None
        self._finish(watch, "failed")
        return f"I can't check the screen for {watch.condition} any more, so I've stopped."
