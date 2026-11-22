"""`find_processes` and `stop_process`: stop one of the owner's programs, chosen by its PID.

Never by name or pattern (docs/plan.md, Boundaries). `find_processes` lists the owner's own
processes whose names match a spoken name, with PIDs, and `stop_process` stops only a PID that
listing returned in the same request, after the confirm key (`strong`, which no owner setting can
lower). Before the question, the PID is bound to its program and start time, so the key allows
that exact process; just before the signal, the process is held by a pidfd and checked again, so a
PID that ended and was reused in between is never signalled. A request heard in PC audio may stop
nothing.

Processes are read from /proc, and window titles from Hyprland's own socket, so finding starts no
program. Program names and window titles are other programs' words: the listing carries the
`process` origin and reaches the model as data.
"""

import contextlib
import json
import os
import re
import select
import signal
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Protocol

from jarvis.jsonutil import objs
from jarvis.operations import Operation, OperationError, Result, current_request
from jarvis.policy import OpClass, Origin
from jarvis.services import JARVIS_UNIT, STRATA_UNIT

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.operations import Request

MAX_SHOWN = 8
MAX_PROTECTED_SHOWN = 5
MAX_NAME = 80
MAX_TITLE = 80
MAX_PROGRAM = 60
MAX_PID = 4_194_304  # PID_MAX_LIMIT on 64-bit Linux
STOP_WAIT_S = 3.0
HYPR_TIMEOUT_S = 0.5
MAX_HYPR_BYTES = 4 * 1024 * 1024
PF_KTHREAD = 0x00200000
KTHREADD = 2
STAT_PPID, STAT_FLAGS, STAT_START = 1, 6, 19  # fields 4, 9 and 22, counted after the comm
MINUTE_S, HOUR_S, DAY_S = 60, 3600, 86400

# Programs never listed or stopped, by the casefolded name of their command, executable or
# argv[0], with why.
PROTECTED_PROGRAMS = {
    "systemd": "the service manager: stopping it ends the session",
    "(sd-pam)": "part of the user's service manager",
    "dbus-daemon": "the message bus the desktop's programs talk over",
    "dbus-broker": "the message bus the desktop's programs talk over",
    "dbus-broker-launch": "the message bus the desktop's programs talk over",
    "pipewire": "the sound server: every program's sound and Jarvis's microphone",
    "pipewire-pulse": "the sound server's PulseAudio side, which most programs play through",
    "wireplumber": "the sound server's session manager",
    "hyprland": "the compositor: stopping it ends the desktop session",
    "start-hyprland": "starts and watches the compositor",
    "xwayland": "shows every X11 program at once, Wine and Proton games among them",
    "uwsm": "the session manager that runs the compositor",
    "waybar": "the bar at the top of the screen",
    "quickshell": "Omarchy's shell: the bar and its widgets",
    "hyprlock": "the lock screen: stopping it would unlock the session",
    "hypridle": "locks the screen when the PC is left alone",
    "voxtype": "Omarchy's dictation, which Jarvis listens through",
    "voxtype-audio-bridge": "part of Omarchy's dictation",
    "voxtype-osd-quickshell": "part of Omarchy's dictation",
    "strata": "the model server Jarvis thinks with",
    "jarvis": "Jarvis itself",
}

# Systemd units whose processes are never listed or stopped, by the last part of their cgroup.
PROTECTED_UNITS = {
    JARVIS_UNIT: "Jarvis's voice service",
    STRATA_UNIT: "the model server Jarvis thinks with",
}

# Processes never listed or stopped for what they are, whatever their name.
PROTECTED_RULES = {
    "init": "PID 1, the system's service manager",
    "kernel": "a kernel thread, part of Linux itself",
    "other_user": "it runs as another user, or with another user's rights",
    "jarvis": "Jarvis itself, or a program Jarvis runs inside (its terminal or shell)",
    "session": "it started the desktop session",
    "login_shell": "the login shell of a session",
}

_FILLER = frozenset({"the", "process", "processes", "app", "application", "program", "game"})


@dataclass(frozen=True, slots=True)
class Window:
    """One Hyprland window: the PID that owns it, its title and its class."""

    pid: int
    title: str
    klass: str


@dataclass(frozen=True, slots=True)
class _Proc:
    pid: int
    ppid: int
    uids: tuple[int, ...]
    comm: str
    start: int
    kernel: bool
    state: str


@dataclass(frozen=True, slots=True)
class _Found:
    proc: _Proc
    program: str
    window: Window | None
    by_name: bool
    children: int = 0


@dataclass(slots=True)
class _Shown:
    """The PIDs `find_processes` listed in one request, with each one's program and start."""

    request: Request | None = None
    targets: dict[int, tuple[str, int]] = field(default_factory=dict[int, tuple[str, int]])


class Stopper(Protocol):
    """How a process is held, signalled and waited on."""

    def open(self, pid: int) -> int:
        """Return a handle on the process; raises ProcessLookupError when it is gone."""
        ...

    def send(self, handle: int, signum: int) -> None:
        """Send a signal to the held process."""
        ...

    def wait(self, handle: int, timeout_s: float) -> bool:
        """Return whether the held process ended within `timeout_s`."""
        ...

    def close(self, handle: int) -> None:
        """Let go of the process."""
        ...


class PidfdStopper:
    """Holds a process by a pidfd, so a signal reaches that process and never a reused PID."""

    @staticmethod
    def open(pid: int) -> int:
        """Return a pidfd for `pid`."""
        return os.pidfd_open(pid)

    @staticmethod
    def send(handle: int, signum: int) -> None:
        """Signal the process the pidfd holds."""
        signal.pidfd_send_signal(handle, signum)

    @staticmethod
    def wait(handle: int, timeout_s: float) -> bool:
        """Return whether the process ended: its pidfd turns readable when it does."""
        poller = select.poll()
        poller.register(handle, select.POLLIN)
        return bool(poller.poll(int(timeout_s * 1000)))

    @staticmethod
    def close(handle: int) -> None:
        """Close the pidfd."""
        os.close(handle)


def _ask_hyprland(path: Path, request: bytes, timeout_s: float) -> bytes:
    """Return Hyprland's whole answer to one socket request, up to a size cap."""
    chunks: list[bytes] = []
    size = 0
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout_s)
        conn.connect(str(path))
        conn.sendall(request)
        while size <= MAX_HYPR_BYTES and (chunk := conn.recv(65536)):
            chunks.append(chunk)
            size += len(chunk)
    return b"".join(chunks)


def hyprland_windows(timeout_s: float = HYPR_TIMEOUT_S) -> list[Window]:
    """Return Hyprland's windows, read from its socket (`j/clients`); [] when it cannot answer."""
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    signature = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE", "")
    if not runtime or not signature or "/" in signature or signature in {".", ".."}:
        return []
    path = Path(runtime) / "hypr" / signature / ".socket.sock"
    try:
        clients: object = json.loads(_ask_hyprland(path, b"j/clients", timeout_s))
    except OSError, ValueError:
        return []
    windows: list[Window] = []
    for client in objs(clients):
        pid, title, klass = client.get("pid"), client.get("title"), client.get("class", "")
        if isinstance(pid, int) and isinstance(title, str) and isinstance(klass, str):
            windows.append(Window(pid, title, klass))
    return windows


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w]+", " ", text.casefold()).replace("_", " ").split()


def spoken_words(name: str) -> list[str]:
    """Return a spoken program name as bare words: no "the", "process", "app" or punctuation."""
    return [word for word in _words(name) if word not in _FILLER]


def _name_words(name: str) -> list[str]:
    words = _words(name)
    return words[:-1] if len(words) > 1 and words[-1] == "exe" else words


def _matches(spoken: list[str], name: str) -> bool:
    """Return whether `name` is the spoken name: whole words in order, or the same letters."""
    words = _name_words(name)
    if not words:
        return False
    if "".join(words) == "".join(spoken):
        return True
    size = len(spoken)
    return any(words[i : i + size] == spoken for i in range(len(words) - size + 1))


def _basename(path: str) -> str:
    """Return the last part of a path, a Windows one (a Wine game's argv[0]) included."""
    return PureWindowsPath(path).name if "\\" in path else Path(path).name


def _printable(text: str, limit: int) -> str:
    return "".join(ch for ch in text if ch.isprintable()).strip()[:limit]


class ProcessOps:
    """`find_processes` and `stop_process` over /proc and Hyprland's windows."""

    def __init__(  # ruff: ignore[too-many-arguments] - the readers, each replaceable in tests
        self,
        *,
        proc: Path = Path("/proc"),
        uid: int | None = None,
        self_pid: int | None = None,
        windows: Callable[[], list[Window]] = hyprland_windows,
        stopper: Stopper | None = None,
        clock: Callable[[], float] = time.time,
        ticks_per_s: int | None = None,
    ) -> None:
        """Read processes under `proc` as the owner `uid`; Jarvis's own PID is `self_pid`."""
        super().__init__()
        self._proc = proc
        self._uid = os.getuid() if uid is None else uid
        self._self = os.getpid() if self_pid is None else self_pid
        self._windows = windows
        self._stopper: Stopper = stopper or PidfdStopper()
        self._clock = clock
        self._ticks = ticks_per_s or os.sysconf("SC_CLK_TCK")
        self._lock = threading.Lock()
        self._shown = _Shown()

    # --- reading /proc --------------------------------------------------------------------

    def _text(self, pid: int, name: str) -> str:
        try:
            return (self._proc / str(pid) / name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _read(self, pid: int) -> _Proc | None:
        stat = self._text(pid, "stat")
        head, sep, tail = stat.rpartition(")")
        fields = tail.split()
        if not sep or len(fields) <= STAT_START:
            return None
        uids: tuple[int, ...] = ()
        for line in self._text(pid, "status").splitlines():
            if line.startswith("Uid:"):
                with contextlib.suppress(ValueError):
                    uids = tuple(int(value) for value in line.split()[1:])
        if not uids:
            return None
        try:
            ppid, flags, start = (int(fields[i]) for i in (STAT_PPID, STAT_FLAGS, STAT_START))
        except ValueError:
            return None
        comm = head.partition("(")[2]
        kernel = bool(flags & PF_KTHREAD) or KTHREADD in {pid, ppid}
        return _Proc(pid, ppid, uids, comm, start, kernel, fields[0])

    def _table(self) -> dict[int, _Proc]:
        table: dict[int, _Proc] = {}
        try:
            entries = [entry.name for entry in self._proc.iterdir() if entry.name.isdigit()]
        except OSError:
            return table
        for name in entries:
            found = self._read(int(name))
            if found is not None:
                table[found.pid] = found
        return table

    def _argv(self, pid: int) -> list[str]:
        try:
            raw = (self._proc / str(pid) / "cmdline").read_bytes()
        except OSError:
            return []
        return [os.fsdecode(word) for word in raw.split(b"\0") if word]

    def _exe(self, pid: int) -> str:
        try:
            target = (self._proc / str(pid) / "exe").readlink()
        except OSError:
            return ""
        return target.name.removesuffix(" (deleted)")

    def _names(self, proc: _Proc) -> list[str]:
        """Return the names a process goes by: its command, executable and argv[0]."""
        argv = self._argv(proc.pid)
        names = [proc.comm, self._exe(proc.pid)]
        if argv:
            names.append(_basename(argv[0].removeprefix("-")))
        return [name for name in names if name]

    def _program(self, proc: _Proc) -> str:
        """Return the name a process is shown and asked about by."""
        argv = self._argv(proc.pid)
        first = _basename(argv[0]) if argv else ""
        exe = self._exe(proc.pid)
        if first.casefold().endswith(".exe"):
            name = first[: -len(".exe")]
        elif exe and exe != proc.comm and exe.startswith(proc.comm):
            name = exe  # the command name was cut at 15 characters
        else:
            name = proc.comm
        return _printable(name, MAX_PROGRAM) or str(proc.pid)

    # --- what is protected ----------------------------------------------------------------

    @staticmethod
    def _chain(pid: int, table: dict[int, _Proc]) -> list[int]:
        """Return `pid` and its ancestors, nearest first."""
        chain: list[int] = []
        current = table.get(pid)
        while current is not None and current.pid not in chain:
            chain.append(current.pid)
            current = table.get(current.ppid)
        return chain

    def _guarded(self, table: dict[int, _Proc]) -> dict[int, str]:
        """Return the PIDs protected for who they are in the tree: Jarvis's and the session's."""
        guarded = dict.fromkeys(self._chain(self._self, table), PROTECTED_RULES["jarvis"])
        for proc in table.values():
            if proc.comm.casefold() == "hyprland":
                for pid in self._chain(proc.ppid, table):
                    guarded.setdefault(pid, PROTECTED_RULES["session"])
        return guarded

    def _protection(self, proc: _Proc, guarded: dict[int, str]) -> str | None:
        """Return why a process may never be stopped, or None when it may."""
        return self._protected_kind(proc, guarded) or self._protected_name(proc)

    def _protected_kind(self, proc: _Proc, guarded: dict[int, str]) -> str | None:
        """Return why a process is protected for what it is, whatever its name."""
        if proc.pid == 1:
            return PROTECTED_RULES["init"]
        if proc.kernel:
            return PROTECTED_RULES["kernel"]
        if not proc.uids or any(uid != self._uid for uid in proc.uids):
            return PROTECTED_RULES["other_user"]
        if proc.pid in guarded:
            return guarded[proc.pid]
        argv = self._argv(proc.pid)
        return PROTECTED_RULES["login_shell"] if argv and argv[0].startswith("-") else None

    def _protected_name(self, proc: _Proc) -> str | None:
        """Return why a process is protected for its unit or its program's name."""
        unit = self._text(proc.pid, "cgroup").strip().rpartition("/")[2]
        if unit in PROTECTED_UNITS:
            return PROTECTED_UNITS[unit]
        for name in self._names(proc):
            reason = PROTECTED_PROGRAMS.get(name.casefold())
            if reason is not None:
                return reason
        return None

    # --- find_processes -------------------------------------------------------------------

    def _windows_by_pid(self) -> dict[int, Window]:
        try:
            windows = self._windows()
        except Exception:  # ruff: ignore[blind-except] - window titles are a nicety; names stay
            return {}
        by_pid: dict[int, Window] = {}
        for window in windows:
            by_pid.setdefault(window.pid, window)
        return by_pid

    def _match(self, spoken: list[str], table: dict[int, _Proc]) -> tuple[list[_Found], list[str]]:
        """Return the stoppable matches, and the protected ones as "name (why)"."""
        guarded = self._guarded(table)
        windows = self._windows_by_pid()
        found: list[_Found] = []
        protected: list[str] = []
        for proc in table.values():
            if proc.state in {"Z", "X"}:
                continue
            window = windows.get(proc.pid)
            by_name = any(_matches(spoken, name) for name in self._names(proc)) or (
                window is not None and _matches(spoken, window.klass)
            )
            by_title = window is not None and _matches(spoken, window.title)
            if not by_name and not by_title:
                continue
            reason = self._protection(proc, guarded)
            if reason is not None:
                protected.append(f"{_printable(proc.comm, MAX_PROGRAM)} ({reason})")
                continue
            found.append(_Found(proc, self._program(proc), window, by_name=by_name))
        if any(item.by_name for item in found):
            found = [item for item in found if item.by_name]
        return self._fold(found, table), sorted(set(protected))[:MAX_PROTECTED_SHOWN]

    def _fold(self, found: list[_Found], table: dict[int, _Proc]) -> list[_Found]:
        """Keep the topmost of each tree of matches, counting the matches below it."""
        pids = {item.proc.pid for item in found}
        below: dict[int, int] = {}
        for item in found:
            above = [pid for pid in self._chain(item.proc.ppid, table) if pid in pids]
            if above:
                below[above[-1]] = below.get(above[-1], 0) + 1
        kept = [
            _Found(item.proc, item.program, item.window, item.by_name, below.get(item.proc.pid, 0))
            for item in found
            if not any(pid in pids for pid in self._chain(item.proc.ppid, table))
        ]
        return sorted(kept, key=lambda item: (-item.proc.start, item.proc.pid))

    def _age(self, start: int) -> str:
        boot = 0.0
        for line in (self._proc / "stat").read_text(encoding="utf-8").splitlines():
            if line.startswith("btime "):
                boot = float(line.split()[1])
        seconds = max(0, int(self._clock() - (boot + start / self._ticks)))
        for size, unit in ((DAY_S, "day"), (HOUR_S, "hour"), (MINUTE_S, "minute")):
            if seconds >= size:
                count = seconds // size
                return f"started {count} {unit}{'' if count == 1 else 's'} ago"
        return "started just now"

    def _line(self, item: _Found) -> str:
        parts = [f"PID {item.proc.pid}: {item.program}"]
        if item.window is not None and item.window.title.strip():
            parts.append(f'window "{_printable(item.window.title, MAX_TITLE)}"')
        with contextlib.suppress(OSError, ValueError):
            parts.append(self._age(item.proc.start))
        if item.children:
            many = "es" if item.children != 1 else ""
            parts.append(f"with {item.children} child process{many}")
        return "- " + ", ".join(parts)

    def find(self, args: dict[str, Any]) -> Result:
        """List the owner's stoppable processes that match the spoken name."""
        name = str(args["name"])
        spoken = spoken_words(name)
        if len("".join(spoken)) < 2:  # ruff: ignore[magic-value-comparison] - one letter is no name
            return Result("Say which program: that is not a program name.", ok=False)
        table = self._table()
        found, protected = self._match(spoken, table)
        shown = found[:MAX_SHOWN]
        with self._lock:
            request = current_request()
            if self._shown.request is not request:
                self._shown = _Shown(request)
            for item in shown:
                self._shown.targets[item.proc.pid] = (item.program, item.proc.start)
        lines: list[str] = []
        if not found and protected:
            lines.append(
                f'Only programs that are never stopped match "{name}". Tell the owner why; do not '
                "search again with a wider name or guess another program."
            )
        elif not found:
            lines.append(
                f'No program of the owner\'s matches "{name}". Tell the owner; do not search '
                "again with a wider name or guess another program."
            )
        else:
            lines.append(f'The owner\'s programs matching "{name}" (stop_process takes one PID):')
            lines.extend(self._line(item) for item in shown)
            if len(found) > len(shown):
                lines.append(
                    f"{len(found) - len(shown)} more match but are not listed; ask the owner for "
                    "a more exact name."
                )
            if len(shown) > 1:
                lines.append(
                    "More than one matches: read these to the owner and ask which one. Never guess."
                )
        if protected:
            lines.append("Also matching, but never stopped: " + "; ".join(protected) + ".")
        missing = not found and not protected
        spoken_reply = f"I can't find a program called {name} running." if missing else ""
        return Result("\n".join(lines), Origin.PROCESS, spoken=spoken_reply)

    # --- stop_process ---------------------------------------------------------------------

    @staticmethod
    def _owner_request() -> Request:
        request = current_request()
        if request is None:
            msg = "a process can be stopped only while handling the owner's request"
            raise OperationError(msg)
        if request.foreign:
            msg = "a request heard in PC audio cannot stop a process; only the owner can"
            raise OperationError(msg)
        return request

    def _listed(self, request: Request, pid: int) -> tuple[str, int]:
        with self._lock:
            target = self._shown.targets.get(pid) if self._shown.request is request else None
        if target is None:
            msg = (
                f"PID {pid} was not listed by find_processes in this request. Call "
                "find_processes with the name the owner said first, and stop only a PID it lists."
            )
            raise OperationError(msg)
        return target

    def _check(self, pid: int, program: str, start: int) -> None:
        """Refuse unless `pid` is still the listed process and still not protected.

        Raises:
            OperationError: the process ended, is another one now, or is protected.
        """
        table = self._table()
        proc = table.get(pid)
        if proc is None or proc.start != start or self._program(proc) != program:
            msg = f"PID {pid} is no longer the {program} that was listed; nothing was stopped"
            raise OperationError(msg)
        reason = self._protection(proc, self._guarded(table))
        if reason is not None:
            msg = f"PID {pid} is never stopped: {reason}"
            raise OperationError(msg)

    def prepare(self, args: dict[str, Any]) -> dict[str, Any]:
        """Bind the PID to the program and start time `find_processes` listed in this request.

        It refuses (`OperationError`) when no owner's request is being handled or it was heard in
        PC audio, when `find_processes` did not list the PID in it, and when the process is no
        longer the one listed or is protected.
        """
        pid = int(args["pid"])
        program, start = self._listed(self._owner_request(), pid)
        self._check(pid, program, start)
        return {"pid": pid, "force": bool(args["force"]), "program": program, "started": start}

    def stop(self, args: dict[str, Any]) -> Result:
        """Stop the bound process, if it is still that process, and say whether it ended.

        Raises:
            OperationError: what `prepare` refuses, checked again with the process held, or the
                process ended or could not be reached before the signal.
        """
        bound = args if "started" in args else self.prepare(args)
        pid, program, start = int(bound["pid"]), str(bound["program"]), int(bound["started"])
        self._listed(self._owner_request(), pid)
        try:
            handle = self._stopper.open(pid)
        except ProcessLookupError as exc:
            msg = f"{program} (PID {pid}) has already ended"
            raise OperationError(msg) from exc
        except OSError as exc:
            msg = f"{program} (PID {pid}) could not be reached: {exc.strerror}"
            raise OperationError(msg) from exc
        try:
            self._check(pid, program, start)
            signum = signal.SIGKILL if bound["force"] else signal.SIGTERM
            try:
                self._stopper.send(handle, signum)
            except ProcessLookupError as exc:
                msg = f"{program} (PID {pid}) has already ended"
                raise OperationError(msg) from exc
            ended = self._stopper.wait(handle, STOP_WAIT_S)
        finally:
            self._stopper.close(handle)
        if ended:
            return Result(f"{program} (PID {pid}) has ended.", spoken=f"Stopped {program}.")
        return Result(
            f"Asked {program} (PID {pid}) to stop, but it is still running after "
            f"{STOP_WAIT_S:g} seconds; it may be asking to save. Only if the owner asks, stop it "
            "again with force=true.",
            ok=False,
        )

    def operations(self) -> list[Operation]:
        """Return `find_processes` and `stop_process`."""
        return [
            Operation(
                "find_processes",
                "Find the owner's running programs by the name the owner said, before stopping "
                "any. Lists each match's PID for stop_process. Several: read them to the owner "
                "and ask which; never guess. None: say so; never try a wider name.",
                {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_NAME,
                            "description": "As the owner said it.",
                        }
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
                self.find,
                side_effect=False,
                spoken_summary=lambda a: f"look for {a.get('name', 'a program')}",
            ),
            Operation(
                "stop_process",
                "Stop one program the owner named, by a PID find_processes listed in this "
                "request; never any other. force=true only if the owner asks to force it.",
                {
                    "type": "object",
                    "properties": {
                        "pid": {"type": "integer", "minimum": 2, "maximum": MAX_PID},
                        "force": {"type": "boolean", "default": False},
                    },
                    "required": ["pid"],
                    "additionalProperties": False,
                },
                self.stop,
                default_class=OpClass.STRONG,
                lowest_class=OpClass.STRONG,
                spoken_summary=_stop_summary,
                prepare=self.prepare,
            ),
        ]


def _stop_summary(args: dict[str, Any]) -> str:
    verb = "force-stop" if args.get("force") else "stop"
    return f"{verb} {args.get('program', 'a program')} (PID {args.get('pid')})"
