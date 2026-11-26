"""`pc_watch`, `pc_watches` and `pc_watch_stop`: tell the owner when something shows on a screen.

The owner asks, in any words, to be told when something shows up in a window, a workspace or on
the screen; the model starts a watch with its own short description of what to look for. While
the voice service is idle it takes a picture now and then (the same capture as `pc_screenshot`:
`grim` only reads frames, nothing is focused, moved or resized) and asks the model one question
about it: no conversation, nothing run, the picture marked as outside data, an answer of YES or NO
(`Judge.glance`; the tools are sent only so the server keeps its pinned prompt, and no call is ever
run). Anything but a leading YES is a no. On a YES Jarvis says a fixed line built from the
condition (or the model's own short `say` line, written when the watch began), never words read off
the screen, and only when the condition starts to show: a line that stays up is said once.

The watch hashes the capture and asks the model only when the picture changed. A window watch
can also look for a freeze using the same capture and the process's state and CPU time.
Each watch is bounded: it checks at most every 10
seconds, ends after `for_min` minutes (at most 12 hours), and at most two run at once. A watch lives
in the voice service's process (`jarvis run`) and ends with it; the other channels do not offer it.
Checks are made only between requests, never while Jarvis speaks or is paused, and a check is
dropped when a request begins (`voice.py` decides when). Times are the wall clock, which runs on
through a suspend: a watch whose time passed while the PC slept ends at the first look after it.
"""

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from jarvis.agent import PICTURES_OFF
from jarvis.operations import Operation, OperationError, Result, current_request
from jarvis.pc.apps import BACKGROUND, SPECIAL, WORKSPACES
from jarvis.pc.processes import ProcessOps
from jarvis.pc.windows import MAX_LISTED, Win, WindowOps, printable, shown, where
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
MAX_SAY = 60  # the spoken line the model may give instead of the condition
FAILURES_END = 3  # checks that fail one after another end a watch
KINDS = ("window", "workspace", "screen", "process")
MAX_PROCESS_WATCHES = 4  # read from /proc: no picture, no model call
STUCK_S = 30.0  # waiting on a disk or device this long, at every check, is stuck
STUCK_CHECKS = 3  # ... and at least this many checks in a row: two looks prove no wait between
_STOPPED = frozenset({"T", "t"})  # stopped by a signal, or held by a debugger
_ENDED = frozenset({"Z", "X", "x"})  # exited; a zombie only waits for its parent
FREEZE_CHANGES = 2
FREEZE_CHECKS = 3
FREEZE_S = 60.0
FREEZE_CPU = 0.02  # fraction of one core
FREEZE_LONG_S = 180.0
NEEDS_PICTURES = f"I can't watch the screen. {PICTURES_OFF}"
NOT_HERE = "A watch runs only in the voice service, so it can't be set from here."
_NO_PID = "Say which process: a PID find_processes listed in this request."
_NO_WINDOW = "Say which window: a number pc_windows listed in this request."
_NO_WORKSPACE = "Say which workspace to watch."
_NO_CONDITION = "Say what to look for."


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
class _Freeze:
    frame: bytes = b""
    changes: int = 0
    identical: int = 0
    elapsed: float = 0.0
    cpu_ticks: int = 0
    last_sample: tuple[float, int] | None = None
    alarmed: bool = False
    suppressed: bool = False  # a process fact has already explained this streak

    def observe(self, frame: bytes, now: float, ticks: int, ticks_per_s: int) -> str:
        """Update the shown streak and return the new freeze or recovery state, if any."""
        previous, self.last_sample = self.last_sample, (now, ticks)
        if self.frame != frame:
            if self.frame:
                self.changes = min(self.changes + 1, FREEZE_CHANGES)
            self.frame = frame
            self.identical = 0
            self.elapsed = 0.0
            self.cpu_ticks = 0
            self.suppressed = False
            if self.alarmed:
                self.alarmed = False
                return "moving"
            return ""
        self.identical += 1
        if previous is not None:
            self.elapsed += max(now - previous[0], 0)
            self.cpu_ticks += max(ticks - previous[1], 0)
        if self.alarmed or self.suppressed or self.changes < FREEZE_CHANGES:
            return ""
        if self.elapsed >= FREEZE_LONG_S or (
            self.elapsed >= FREEZE_S
            and self.identical >= FREEZE_CHECKS
            and self.cpu_ticks < self.elapsed * (FREEZE_CPU * ticks_per_s)
        ):
            self.alarmed = True
            self.changes = 0
            return "frozen"
        return ""


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
    pid: int = 0  # a process watch's process: its PID, program and start time
    program: str = ""
    started: int = 0
    trouble: str = ""  # what was last said about it: "", "stopped" or "stuck"
    waiting_since: float = 0.0  # the first check of this wait on a disk or device (0: none)
    waits: int = 0  # checks in a row that found it waiting
    showing: bool = False  # the last check saw the condition: the next YES is not new
    say: str = ""  # the model's own short line for a YES; empty says the condition
    condition_frame: bytes = b""
    condition_pending: bool = False  # a new YES observed during the announcement cool-down
    freeze: _Freeze | None = None

    def of(self) -> str:
        """Return what it looks at, in words."""
        if self.kind == "window" and self.window is not None:
            return f"the {self.program or self.window.app} window"
        if self.kind == "workspace":
            return f"the windows {where(self.workspace)}"
        if self.kind == "process":
            return self.process_name()
        return "the screen"

    def process_name(self) -> str:
        """Return the process's name and PID for factual announcements."""
        return f"{self.program} (PID {self.pid})"

    def looking_for(self) -> str:
        """Return the condition, a freeze, or both."""
        return " and ".join(
            part for part in (self.condition, "a freeze" if self.freeze else "") if part
        )


def _over(watch: Watch) -> str:
    """Return what Jarvis says when a watch's time is up."""
    if watch.kind == "process":
        return f"My watch on {watch.of()} is over."
    return f"My watch for {watch.looking_for()} is over."


def _summary(args: dict[str, Any]) -> str:
    """Return what the owner is asked to allow: what is watched and, when given, the line said."""
    if args.get("of") == "process":
        return f"watch process {args.get('pid', '')} and say if it exits, stops or gets stuck"
    condition = _condition(str(args.get("condition", "")))
    if args.get("frozen"):
        program = args.get("program")
        target = f"the {program} window" if program else f"window {args.get('window')}"
        asked = f"watch {target} for a freeze"
        if condition:
            asked += f" and for {condition}"
    elif app := args.get("app"):
        asked = f"watch the {app} window for {condition or 'something'}"
    else:
        asked = f"watch for {condition or 'something'}"
    say = printable(str(args.get("say", "")), MAX_SAY).rstrip(" .!?,;:")
    if condition and say:
        asked += f' and then say "Heads up: {say}."'
    return f"{asked} ({_screen_cost(condition)})"


def _screen_cost(condition: str) -> str:
    if condition:
        return "a model call with a picture only when the frame changes"
    return "a capture and a hash, with no model call"


def _window_identity(window: Win) -> tuple[str, str, str, int]:
    return window.address, window.stable, window.klass, window.pid


def _watches_freeze(args: dict[str, Any]) -> bool:
    """Whether this is a freeze watch, which requires a window target.

    Raises:
        OperationError: frozen was requested for another kind of watch.
    """
    frozen = bool(args.get("frozen"))
    if frozen and args["of"] != "window":
        msg = "A freeze watch needs a window."
        raise OperationError(msg)
    return frozen


def _heads_up(watch: Watch) -> str:
    """Return the line said when a screen watch's condition starts to show."""
    return f"Heads up: {watch.say or watch.condition}."


def _owner_only() -> None:
    """Refuse unless the owner's own request is being handled (not PC audio, not a job).

    Raises:
        OperationError: it is not.
    """
    request = current_request()
    if request is None or request.foreign or request.scheduled:
        msg = "a watch is set only on the owner's own request, never from PC audio or a job"
        raise OperationError(msg)


def _condition(text: str) -> str:
    """Return the model's description of what to look for as one short line."""
    return printable(text, MAX_CONDITION).rstrip(" .!?,;:")


class _AbandonedError(Exception):
    """A check was dropped because the voice service stopped being idle; it is not a failure."""


@dataclass(slots=True)
class _Shared:
    watches: dict[int, Watch] = field(default_factory=dict[int, Watch])
    free: int = 1
    checking: bool = False


class ScreenWatch:
    """Builds the watch operations and makes the checks the voice service asks for."""

    def __init__(  # ruff: ignore[too-many-arguments] - wall time for expiry and awake time for frames
        self,
        windows: WindowOps,
        shots: ScreenshotOps,
        processes: ProcessOps | None = None,
        *,
        clock: Callable[[], float] = time.time,
        freeze_clock: Callable[[], float] = time.monotonic,
        audit: _Audit | None = None,
    ) -> None:
        """Capture through `shots`, find windows through `windows`, read time from `clock`.

        Wall time runs through suspend. The freeze clock counts only awake time.
        """
        super().__init__()
        self._windows = windows
        self._shots = shots
        self._processes = ProcessOps() if processes is None else processes
        self._clock = clock
        self._freeze_clock = freeze_clock
        self._audit = audit
        self._judge: Judge | None = None
        self._lock = threading.Lock()
        self._shared = _Shared()

    @property
    def windows(self) -> WindowOps:
        """The window tools the watches bind their windows through (shared with `pc_windows`)."""
        return self._windows

    @property
    def processes(self) -> ProcessOps:
        """The process tools a process watch binds its PID through (`find_processes`'s)."""
        return self._processes

    def attach(self, judge: Judge) -> None:
        """Let the model judge the pictures (it is built after the operations)."""
        self._judge = judge

    def _record(self, event: str, watch: Watch, **fields: object) -> None:
        if self._audit is not None:
            self._audit.record(
                event,
                watch=watch.ident,
                of=watch.kind,
                condition=watch.condition,
                frozen=watch.freeze is not None,
                **fields,
            )

    # --- the operations ---------------------------------------------------------------------

    def start(self, args: dict[str, Any]) -> Result:
        """Start a watch on a window, a workspace or the screen.

        Raises:
            OperationError: not the owner's own request, the model cannot see pictures, two
                watches already run, or the window or workspace is not given (or not listed in
                this request, as for the window tools).
        """
        _owner_only()
        frozen = _watches_freeze(args)
        if str(args["of"]) == "process":
            return self._start_process(args)
        condition = self._screen_condition(args, frozen=frozen)
        kind = str(args["of"])
        window: Win | None = None
        workspace = ""
        if kind == "window":
            if "window_identity" not in args:
                args = self._bind_window(args, frozen=frozen)
            window = self._bound_window(args)
        elif kind == "workspace":
            wanted = args["workspace"]
            workspace = SPECIAL if wanted == BACKGROUND else str(wanted)
        say = printable(str(args.get("say", "")), MAX_SAY).rstrip(" .!?,;:") if condition else ""
        now = self._clock()
        every_s = float(args["every_s"])
        with self._lock:
            if self._screens() >= MAX_WATCHES:
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
                say=say,
                freeze=_Freeze() if frozen else None,
                pid=window.pid if frozen and window is not None else 0,
                program=str(args.get("program", "")),
                started=int(args.get("started", 0)),
            )
            self._shared.free += 1
            self._shared.watches[watch.ident] = watch
        self._record(
            "watch_started",
            watch,
            every_s=every_s,
            for_min=args["for_min"],
            repeat=watch.repeat,
            say=say,
            **({"pid": watch.pid} if frozen else {}),
        )
        again = "each time it starts to show" if watch.repeat else "the first time it shows"
        announcement = ""
        if condition:
            announcement = f' I\'ll say "{_heads_up(watch)}" {again}.'
        return Result(
            f"Watching {watch.of()} for: {watch.looking_for()}. Checking every {every_s:g} s for "
            f"{args['for_min']} min ({_screen_cost(condition)}).{announcement} "
            f"Stop it with pc_watch_stop (watch {watch.ident}).",
            Origin.PROCESS if kind == "window" else Origin.TOOL_STATUS,
        )

    def _prepare(self, args: dict[str, Any]) -> dict[str, Any]:
        """Bind a listed target before the owner can be asked about the watch.

        Raises:
            OperationError: the request cannot set this watch or its target is unavailable.
        """
        _owner_only()
        frozen = _watches_freeze(args)
        if args["of"] == "process":
            if args.get("pid") is None:
                raise OperationError(_NO_PID)
            self._processes.watched(int(args["pid"]))
            return args
        self._screen_condition(args, frozen=frozen)
        return self._bind_window(args, frozen=frozen) if args["of"] == "window" else args

    def _screen_condition(self, args: dict[str, Any], *, frozen: bool) -> str:
        """Return the condition after checking whether the requested screen watch can run.

        Raises:
            OperationError: the watch has no condition, model vision or workspace when required.
        """
        condition = _condition(str(args.get("condition", "")))
        if (condition or not frozen) and (self._judge is None or not self._judge.sees()):
            raise OperationError(NEEDS_PICTURES)
        if args["of"] == "workspace" and args.get("workspace") is None:
            raise OperationError(_NO_WORKSPACE)
        if not condition and not frozen:
            raise OperationError(_NO_CONDITION)
        return condition

    def _bind_window(self, args: dict[str, Any], *, frozen: bool) -> dict[str, Any]:
        """Bind a window from the read listing and, for a freeze watch, its process.

        Raises:
            OperationError: the request has no window number or its target is unavailable.
        """
        if args.get("window") is None:
            raise OperationError(_NO_WINDOW)
        listed = self._windows.listed(int(args["window"]))
        bound = {**args, "window_identity": _window_identity(listed), "app": listed.app}
        window = self._bound_window(bound)
        if frozen:
            program, started = self._processes.window_target(window.pid)
            bound.update(program=program, started=started)
        return bound

    def _bound_window(self, args: dict[str, Any]) -> Win:
        """Find the exact window the owner was asked about, even if the listing changed.

        Raises:
            OperationError: the bound window or process is no longer there.
        """
        for window in self._windows.desktop.windows():
            if _window_identity(window) == args["window_identity"]:
                if "started" not in args or self._processes.sample(window.pid, args["started"]):
                    return window
                break
        msg = "The window that was listed is no longer open; no watch was set."
        raise OperationError(msg)

    def _screens(self) -> int:
        """Return how many watches look at the screen (the lock held)."""
        return sum(w.kind != "process" for w in self._shared.watches.values())

    def _start_process(self, args: dict[str, Any]) -> Result:
        """Start watching a process `find_processes` listed in this request.

        Raises:
            OperationError: no PID, not listed, no longer that process, already watched, or
                `MAX_PROCESS_WATCHES` already run.
        """
        if args.get("pid") is None:
            raise OperationError(_NO_PID)
        pid = int(args["pid"])
        program, started = self._processes.watched(pid)
        now = self._clock()
        every_s = float(args["every_s"])
        with self._lock:
            mine = [w for w in self._shared.watches.values() if w.kind == "process"]
            if any(w.pid == pid and w.started == started for w in mine):
                msg = f"{program} (PID {pid}) is already watched."
                raise OperationError(msg)
            if len(mine) >= MAX_PROCESS_WATCHES:
                msg = f"{MAX_PROCESS_WATCHES} process watches already run. Stop one first."
                raise OperationError(msg)
            watch = Watch(
                ident=self._shared.free,
                kind="process",
                condition=f"{program} exits, is stopped or gets stuck",
                every_s=every_s,
                repeat=True,
                ends_at=now + float(args["for_min"]) * 60,
                next_at=now,
                pid=pid,
                program=program,
                started=started,
            )
            self._shared.free += 1
            self._shared.watches[watch.ident] = watch
        self._record("watch_started", watch, every_s=every_s, for_min=args["for_min"], pid=pid)
        return Result(
            f"Watching {watch.of()}: I'll say if it exits, is stopped or gets stuck, checking "
            f"every {every_s:g} s for {args['for_min']} min (watch {watch.ident}).",
            origin=Origin.PROCESS,
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
            f"{w.ident}. {w.of()} for: {w.looking_for()} "
            f"(every {w.every_s:g} s, {self._left(w, now)}, "
            + (f"{'each time' if w.repeat else 'once'}, " if w.condition else "")
            + f"said {w.announced}x"
            + (f', saying "{_heads_up(w)}"' if w.say else "")
            + (f", {_screen_cost(w.condition)}" if w.kind != "process" else "")
            + ")"
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
            found = [w for w in found if self._end(w)]
        if not found:
            return Result(
                "No such watch." if wanted is not None else "No watches are running.", ok=False
            )
        for watch in found:
            self._record("watch_stopped", watch, reason="owner")
        names = ", ".join(str(w.ident) for w in found)
        return Result(f"Stopped watch {names}.")

    def _end(self, watch: Watch) -> bool:
        """Take `watch` out (the lock is held); False when it was out already."""
        watch.active = False
        return self._shared.watches.pop(watch.ident, None) is not None

    def operations(self) -> list[Operation]:
        """Return `pc_watch`, `pc_watches` and `pc_watch_stop`."""
        return [
            Operation(
                "pc_watch",
                "Watch a listed window, workspace or screen for a condition (a model call with "
                "a picture only when it changes; optional say line). Use frozen=true on a window "
                "to report a freeze. Omit condition for a freeze watch without the model. "
                "Use of=process with a PID find_processes listed to report exits, stops or stalls.",
                {
                    "type": "object",
                    "properties": {
                        "of": {"type": "string", "enum": list(KINDS)},
                        "window": {"type": "integer", "minimum": 1, "maximum": MAX_LISTED},
                        "pid": {"type": "integer", "minimum": 2},
                        "workspace": {"type": "string", "enum": [*WORKSPACES, BACKGROUND]},
                        "condition": {"type": "string", "minLength": 1, "maxLength": MAX_CONDITION},
                        "frozen": {"type": "boolean", "default": False},
                        "say": {"type": "string", "minLength": 1, "maxLength": MAX_SAY},
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
                    "required": ["of"],
                    "additionalProperties": False,
                },
                self.start,
                spoken_summary=_summary,
                prepare=self._prepare,
                hashed=frozenset({"window_identity"}),
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

    def check_due(self, stop: Callable[[], bool] = lambda: False) -> list[str]:
        """Check every watch that is due; return the lines Jarvis is to say, in order.

        `stop` says the voice service is no longer idle (the wake word, speech, a request).
        It is asked before each picture and
        model call, and a check it stops is dropped, not counted as a failure; the watch is due
        again at once.
        """
        now = self._clock()
        with self._lock:
            if self._shared.checking:
                return []
            due = [w for w in self._shared.watches.values() if w.next_at <= now or w.ends_at <= now]
            self._shared.checking = True
        lines: list[str] = []
        try:
            for watch in due:
                if stop():
                    break
                lines.extend(self._check(watch, stop))
        finally:
            with self._lock:
                self._shared.checking = False
        return lines

    def _finish(self, watch: Watch, reason: str) -> bool:
        """End `watch` and audit why; False when it had already ended (nothing is audited)."""
        with self._lock:
            ended = self._end(watch)
        if ended:
            self._record("watch_stopped", watch, reason=reason)
        return ended

    def _check(self, watch: Watch, stop: Callable[[], bool]) -> list[str]:
        """Make one check of `watch` and return the lines to say.

        Whatever goes wrong in it counts as a failed check, never as a crash of the voice loop.
        """
        if not watch.active:  # stopped while an earlier watch was being checked
            return []
        due_at = watch.next_at
        lines: list[str] = []
        try:
            self._checked(watch, stop, lines)
        except _AbandonedError:
            watch.next_at = due_at
        except Exception as exc:  # ruff: ignore[blind-except] - a failed capture or model call is a failed check, never a crash of the voice loop
            if watch.freeze is not None:
                watch.freeze.last_sample = None
            if (line := self._failed(watch, str(exc))) is not None:
                lines.append(line)
        return lines

    def _checked(self, watch: Watch, stop: Callable[[], bool], lines: list[str]) -> None:
        now = self._clock()
        if now >= watch.ends_at:
            if self._finish(watch, "expired"):
                lines.append(_over(watch))
            return
        watch.next_at = now + watch.every_s
        if watch.kind == "process":
            state = self._processes.state_of(watch.pid, watch.started)
            if (line := self._process_check(watch, now, state)) is not None:
                lines.append(line)
            return

        ticks = 0
        if watch.freeze is not None:
            line, ticks = self._freeze_process_check(watch, watch.freeze, now)
            if line is not None:
                lines.append(line)
                if not watch.active or not watch.condition:
                    return
        if self._gone(watch):
            if self._finish(watch, "window_gone") and not lines:
                lines.append(
                    f"The window I was watching for {watch.looking_for()} is gone, so I've stopped."
                )
            return
        lines.extend(self._screen_check(watch, now, ticks, stop, visual_due=not lines))

    def _freeze_process_check(
        self, watch: Watch, freeze: _Freeze, now: float
    ) -> tuple[str | None, int]:
        try:
            sample = self._processes.sample(watch.pid, watch.started)
        except OSError:
            watch.waiting_since, watch.waits = 0.0, 0  # the wait was not confirmed at this check
            raise
        state, ticks = sample if sample is not None else (None, 0)
        line = self._process_check(watch, now, state)
        if watch.trouble:
            freeze.suppressed = True
            freeze.last_sample = None
        return line, ticks

    def _screen_check(
        self, watch: Watch, now: float, ticks: int, stop: Callable[[], bool], *, visual_due: bool
    ) -> list[str]:
        """Use one capture for the checks due on this screen watch."""
        freeze, window = watch.freeze, watch.window
        if freeze is not None and (
            not visual_due
            or watch.trouble
            or window is None
            or not shown(window, self._windows.desktop.monitors())
        ):
            freeze.last_sample = None
            freeze = None
        condition_due = bool(watch.condition) and (
            watch.freeze is not None or now >= watch.quiet_until
        )
        if freeze is None and not condition_due:
            return []
        awake_at = self._freeze_clock() if freeze is not None else 0.0
        frame, verdict = self._look(watch, condition_due=condition_due, stop=stop)
        if not watch.active:
            return []
        watch.failures = 0
        lines: list[str] = []
        if freeze is not None and (
            state := freeze.observe(frame, awake_at, ticks, self._processes.ticks_per_s)
        ):
            lines.append(self._freeze_line(watch, freeze, state))
        if condition_due:
            watch.condition_frame = frame
            if line := self._condition_line(watch, now, verdict=verdict):
                lines.append(line)
        return lines

    def _condition_line(self, watch: Watch, now: float, *, verdict: bool) -> str | None:
        was, watch.showing = watch.showing, verdict
        watch.condition_pending = verdict and (not was or watch.condition_pending)
        if not watch.condition_pending or now < watch.quiet_until:
            return None
        watch.condition_pending = False
        watch.announced += 1
        watch.quiet_until = now + max(COOL_DOWN_MIN_S, COOL_DOWN_CHECKS * watch.every_s)
        self._record("watch_announced", watch, count=watch.announced)
        if not watch.repeat:
            self._finish(watch, "done")
        return _heads_up(watch)

    def _freeze_line(self, watch: Watch, freeze: _Freeze, state: str) -> str:
        watch.announced += 1
        self._record("watch_announced", watch, count=watch.announced, state=state)
        if state == "moving":
            return f"{watch.program} is moving again."
        return (
            f"Heads up: {watch.program} seems frozen: its picture has not changed "
            f"for {freeze.elapsed:.0f} seconds."
        )

    def _process_check(self, watch: Watch, now: float, state: str | None) -> str | None:
        """Say once when the sampled process exits, stops, gets stuck or runs again."""
        if not watch.active:
            return None
        name = watch.process_name()
        if state is None or state in _ENDED:
            line = f"Heads up: {name} has exited: it closed, crashed or was killed."
            return line if self._finish(watch, "process_ended") else None
        trouble = ""
        if state == "D":
            watch.waiting_since = watch.waiting_since or now
            watch.waits += 1
            waited = now - watch.waiting_since >= STUCK_S and watch.waits >= STUCK_CHECKS
            # Still stuck, newly stuck, or not yet: a stopped program waiting again was resumed.
            trouble = "stuck" if waited or watch.trouble == "stuck" else ""
        else:
            watch.waiting_since, watch.waits = 0.0, 0  # any other state breaks the wait
            trouble = "stopped" if state in _STOPPED else ""
        if trouble == watch.trouble:
            return None
        watch.trouble = trouble
        watch.announced += 1
        self._record("watch_announced", watch, count=watch.announced, state=trouble or "running")
        if trouble == "stopped":
            return f"Heads up: {name} is stopped, frozen until something resumes it."
        if trouble == "stuck":
            return (
                f"Heads up: {name} has been stuck for {now - watch.waiting_since:.0f} seconds, "
                "waiting on a disk or a device at every check."
            )
        return f"{name} is running again."

    def _gone(self, watch: Watch) -> bool:
        """Whether the watched window is closed (or now another program's): its watch ends.

        Hyprland not giving its windows raises, which the check counts as failed.
        """
        window = watch.window
        if window is None:
            return False
        windows = self._windows.desktop.windows()
        current = next((w for w in windows if w.address == window.address), None)
        if current is None or _window_identity(current) != _window_identity(window):
            return True
        watch.window = current
        return False

    def _look(
        self, watch: Watch, *, condition_due: bool, stop: Callable[[], bool]
    ) -> tuple[bytes, bool]:
        """Hash the captured bytes and judge a changed frame if its condition check is due.

        Raises:
            PictureError: the model cannot see pictures now.
            _AbandonedError: `stop` says the service is busy; the answer, if any, is dropped.
        """
        judge = self._judge if condition_due else None
        if condition_due and (judge is None or not judge.sees()):
            msg = "the picture reader is off"
            raise PictureError(msg)
        if stop():
            raise _AbandonedError
        data = self._shots.capture(watch.kind, workspace=watch.workspace, window=watch.window)
        frame = hashlib.blake2b(data).digest()
        if stop():
            raise _AbandonedError
        if judge is None or frame == watch.condition_frame:
            return frame, watch.showing
        picture = prepare(data, "watch.png", max_side=SHOWN_SIDE)
        verdict = judge.glance(picture, watch.condition)
        if stop():
            raise _AbandonedError  # a request began meanwhile: this answer is not worth saying
        return frame, verdict

    def _failed(self, watch: Watch, why: str) -> str | None:
        """Count a check that did not finish; three in a row end the watch, said once."""
        if not watch.active:
            return None
        watch.failures += 1
        self._record("watch_check_failed", watch, error=why[:200], failures=watch.failures)
        if watch.failures < FAILURES_END:
            return None
        if not self._finish(watch, "failed"):
            return None
        return f"I can't check the screen for {watch.looking_for()} any more, so I've stopped."
