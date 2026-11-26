"""`find_processes` and `stop_process`: stop one of the owner's programs, chosen by its PID.

Never by name or pattern (docs/plan.md, Boundaries). `find_processes` lists the owner's own
processes whose names match a spoken name, with PIDs, and `stop_process` stops only a PID that
listing returned in the same request, after the confirm key (`strong`, which no owner setting can
lower). Before the question, the PID is bound to its program and start time, so the key allows
that exact process; just before the signal, the process is held by a pidfd and checked again, so a
PID that ended and was reused in between is never signalled. A request heard in PC audio may stop
nothing.

Without a name, `find_processes` lists every running program of the owner's instead, with its
processes' resident memory and CPU added up (`usage` gives the same to `pc_stats`). That listing
shows no PID and binds none: a program in it is stopped only after a by-name find.

Processes are read from /proc, and window titles from Hyprland's own socket, so finding starts no
program. Program names and window titles are other programs' words: the listing carries the
`process` origin and reaches the model as data.
"""

import contextlib
import os
import re
import select
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Protocol

from jarvis.operations import (
    REMEMBERED_LISTING,
    Operation,
    OperationError,
    Result,
    current_request,
    inert,
)
from jarvis.pc import hyprland
from jarvis.pc.hyprland import HYPR_TIMEOUT_S
from jarvis.policy import OpClass, Origin
from jarvis.services import JARVIS_UNIT, STRATA_UNIT

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.operations import Request

MAX_SHOWN = 8
LIST_COUNT, MAX_LISTED = 12, 40  # programs a listing without a name shows: by default, at most
CPU_SAMPLE_S = 0.5  # how long a listing without a name measures CPU use
SORTS = ("memory", "cpu", "newest")
GIB, MIB = 1 << 30, 1 << 20
MAX_PROTECTED_SHOWN = 5
MAX_NAME = 80
MAX_TITLE = 80
MAX_PROGRAM = 60
MAX_PID = 4_194_304  # PID_MAX_LIMIT on 64-bit Linux
STOP_WAIT_S = 3.0
COMM_MAX = 15  # a command name is cut to this many characters
PF_KTHREAD = 0x00200000
KTHREADD = 2
STAT_PPID, STAT_FLAGS, STAT_START = 1, 6, 19  # fields 4, 9 and 22, counted after the comm
STAT_UTIME, STAT_STIME = 11, 12  # fields 14 and 15
MINUTE_S, HOUR_S, DAY_S = 60, 3600, 86400

# Programs never stopped (never matched by name, marked in a listing of all), by the casefolded
# name of their command, executable or argv[0], with why.
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
    "xdg-desktop-portal": "the desktop portal: file pickers, screen sharing and screenshots",
    "xdg-desktop-portal-hyprland": "the desktop portal's Hyprland side: screen sharing",
    "xdg-desktop-portal-gtk": "the desktop portal's GTK side: file pickers and settings",
    "xdg-document-portal": "gives sandboxed apps the files they were handed",
    "xdg-permission-store": "keeps the permissions the desktop portal granted",
    "gnome-keyring-daemon": "the keyring that holds saved passwords and logins",
    "polkit-gnome-authentication-agent-1": "asks for the password when a program needs rights",
    "hyprpolkitagent": "asks for the password when a program needs rights",
    "dconf-service": "stores the desktop's and GTK programs' settings",
    "at-spi-bus-launcher": "the accessibility bus: screen readers and assistive tools",
    "at-spi2-registryd": "the accessibility registry: screen readers and assistive tools",
    "mako": "shows the desktop's notifications",
    "swayosd-server": "shows the volume and brightness pop-ups",
    "hyprsunset": "the night light",
    "walker": "the app launcher",
    "elephant": "the app launcher's data service",
    "fcitx5": "the input method: typing in other scripts and languages",
}

# Programs never stopped (as above), by the start of their casefolded name, with why.
PROTECTED_PREFIXES = {
    "gvfs": "GNOME's virtual file systems: mounts, the trash and network shares",
    "omarchy-": "one of Omarchy's own desktop helpers: its watchers and launchers",
}

# Systemd units whose processes are never stopped (as above), by the last part of their cgroup.
PROTECTED_UNITS = {
    JARVIS_UNIT: "Jarvis's voice service",
    STRATA_UNIT: "the model server Jarvis thinks with",
}

# Processes never stopped (as above) for what they are, whatever their name.
PROTECTED_RULES = {
    "init": "PID 1, the system's service manager",
    "kernel": "a kernel thread, part of Linux itself",
    "other_user": "it runs as another user, or with another user's rights",
    "jarvis": "Jarvis itself, or a program Jarvis runs inside (its terminal or shell)",
    "session": "it started the desktop session",
    "login_shell": "the login shell of a session",
    "session_scope": "it belongs to the login session itself: stopping it can end the desktop",
    "foreign_parent": "another user started it (or its parent cannot be seen): a session's leader",
}

_SESSION_SCOPE = re.compile(r"session-\d+\.scope")

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
    cpu_ticks: int


@dataclass(frozen=True, slots=True)
class _Found:
    proc: _Proc
    program: str
    window: Window | None
    by_name: bool
    children: int = 0


@dataclass(frozen=True, slots=True)
class Usage:
    """One program's use of the PC: its processes, resident memory (bytes) and CPU.

    `cpu_percent` is its share of the whole processor, every thread counted, as `pc_stats` gives
    the processor's own load.
    """

    name: str
    processes: int
    rss: int
    cpu_percent: float


@dataclass(slots=True)
class _Program:
    """The owner's processes of one program, added up, for a listing without a name."""

    name: str
    pids: list[int] = field(default_factory=list[int])
    rss: int = 0  # bytes
    ticks: int = 0  # CPU ticks over the sample
    start: int = 0  # the newest process's
    never: str | None = None  # why some or all of its processes are never stopped
    stoppable: int = 0  # how many of its processes may be stopped


@dataclass(slots=True)
class _Shown:
    """The PIDs `find_processes` listed in one request.

    Each with its program, start time and, for a match by a window title alone, that title.
    """

    request: Request | None = None
    targets: dict[int, tuple[str, int, str]] = field(
        default_factory=dict[int, tuple[str, int, str]]
    )
    message: int = 0  # the model message whose call listed them last (a watch needs it read)


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


def hyprland_windows(timeout_s: float = HYPR_TIMEOUT_S) -> list[Window]:
    """Return Hyprland's windows, read from its socket (`j/clients`); [] when it cannot answer."""
    windows: list[Window] = []
    for client in hyprland.clients(timeout_s):
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
        sleep: Callable[[float], None] = time.sleep,
        timer: Callable[[], float] = time.monotonic,
        page_size: int | None = None,
        cpus: int | None = None,
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
        self._sleep = sleep
        self._timer = timer
        self._page = page_size or os.sysconf("SC_PAGE_SIZE")
        self._cpus = cpus or os.cpu_count() or 1
        self._lock = threading.Lock()
        self._shown = _Shown()

    # --- reading /proc --------------------------------------------------------------------

    def _text(self, pid: int, name: str, *, strict: bool = False) -> str:
        try:
            return (self._proc / str(pid) / name).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            if strict and not isinstance(exc, FileNotFoundError):
                raise
            return ""

    def _read(self, pid: int, *, strict: bool = False) -> _Proc | None:
        stat = self._text(pid, "stat", strict=strict)
        head, sep, tail = stat.rpartition(")")
        fields = tail.split()
        if not sep or len(fields) <= STAT_START:
            return None
        uids: tuple[int, ...] = ()
        for line in self._text(pid, "status", strict=strict).splitlines():
            if line.startswith("Uid:"):
                with contextlib.suppress(ValueError):
                    uids = tuple(int(value) for value in line.split()[1:])
        if not uids:
            return None
        try:
            ppid, flags, start = (int(fields[i]) for i in (STAT_PPID, STAT_FLAGS, STAT_START))
            cpu_ticks = int(fields[STAT_UTIME]) + int(fields[STAT_STIME])
        except ValueError:
            return None
        comm = head.partition("(")[2]
        kernel = bool(flags & PF_KTHREAD) or KTHREADD in {pid, ppid}
        return _Proc(pid, ppid, uids, comm, start, kernel, fields[0], cpu_ticks)

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
        if len(argv) > 1 and not argv[1].startswith("-"):
            script = _basename(argv[1])
            if script[:COMM_MAX] == proc.comm:  # a script, which the kernel names after its file
                names.append(script)
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

    def _protection(
        self, proc: _Proc, table: dict[int, _Proc], guarded: dict[int, str]
    ) -> str | None:
        """Return why a process may never be stopped, or None when it may."""
        return (
            self._protected_kind(proc, table)
            or self._protected_place(proc, guarded)
            or self._protected_name(proc)
        )

    def _owned(self, proc: _Proc) -> bool:
        return bool(proc.uids) and all(uid == self._uid for uid in proc.uids)

    def _protected_kind(self, proc: _Proc, table: dict[int, _Proc]) -> str | None:
        """Return why a process is protected for what it is and who started it."""
        if proc.pid == 1:
            return PROTECTED_RULES["init"]
        if proc.kernel:
            return PROTECTED_RULES["kernel"]
        if not self._owned(proc):
            return PROTECTED_RULES["other_user"]
        parent = table.get(proc.ppid)
        if proc.ppid > 0 and (parent is None or not self._owned(parent)):
            return PROTECTED_RULES["foreign_parent"]
        return None

    def _protected_place(self, proc: _Proc, guarded: dict[int, str]) -> str | None:
        """Return why a process is protected for where it sits in the session."""
        if proc.pid in guarded:
            return guarded[proc.pid]
        argv = self._argv(proc.pid)
        if argv and argv[0].startswith("-"):
            return PROTECTED_RULES["login_shell"]
        unit = self._text(proc.pid, "cgroup").strip().rpartition("/")[2]
        if _SESSION_SCOPE.fullmatch(unit):
            return PROTECTED_RULES["session_scope"]
        return PROTECTED_UNITS.get(unit)

    def _protected_name(self, proc: _Proc) -> str | None:
        """Return why a process is protected for its program's name, full or cut."""
        for name in (name.casefold() for name in self._names(proc)):
            reason = PROTECTED_PROGRAMS.get(name) or next(
                (why for start, why in PROTECTED_PREFIXES.items() if name.startswith(start)), None
            )
            if reason is not None:
                return reason
        comm = proc.comm.casefold()
        if len(comm) == COMM_MAX:  # cut: a longer protected name may start with it
            for name, reason in PROTECTED_PROGRAMS.items():
                if name.startswith(comm):
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
            reason = self._protection(proc, table, guarded)
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
        if not item.by_name:
            parts.append("matched only by a window title: stopping it closes all its windows")
        return "- " + ", ".join(parts)

    def _rss(self, pid: int) -> int:
        """Return the resident memory of a process in bytes (0 when it cannot be read)."""
        fields = self._text(pid, "statm").split()
        try:
            return int(fields[1]) * self._page
        except IndexError, ValueError:
            return 0

    def _programs(self) -> tuple[list[_Program], float]:
        """Return the owner's running programs, each with its processes added up, and the sample.

        CPU use is the processes' CPU ticks before and after a wait of `CPU_SAMPLE_S`, over the
        time that really passed between the two readings (never less than the wait).
        """
        before = self._table()
        began = self._timer()
        self._sleep(CPU_SAMPLE_S)
        table = self._table()
        sample_s = max(CPU_SAMPLE_S, self._timer() - began)
        guarded = self._guarded(table)
        programs: dict[str, _Program] = {}
        for proc in table.values():
            if proc.state in {"Z", "X"} or proc.kernel or not self._owned(proc):
                continue
            never = self._protection(proc, table, guarded)
            name = self._program(self._top_of_same_program(proc, table))
            item = programs.setdefault(name, _Program(name))
            if never is None:
                item.stoppable += 1
            elif item.never is None:
                item.never = never
            item.pids.append(proc.pid)
            item.rss += self._rss(proc.pid)
            item.start = max(item.start, proc.start)
            old = before.get(proc.pid)
            if old is not None and old.start == proc.start:
                item.ticks += max(0, proc.cpu_ticks - old.cpu_ticks)
        return list(programs.values()), sample_s

    def _top_of_same_program(self, proc: _Proc, table: dict[int, _Proc]) -> _Proc:
        """Return the topmost ancestor running the same executable: a browser's for its tabs."""
        exe = self._exe(proc.pid)
        top = proc
        seen = {proc.pid}
        while exe and (parent := table.get(top.ppid)) is not None and parent.pid not in seen:
            if not self._owned(parent) or self._exe(parent.pid) != exe:
                break
            seen.add(parent.pid)
            top = parent
        return top

    def _cpu_scale(self, sample_s: float) -> float:
        """Return what one CPU tick in the sample is as a share of the whole processor (percent)."""
        return 100 / self._ticks / sample_s / self._cpus

    def usage(self) -> list[Usage]:
        """Return every running program of the owner's with its memory and CPU use now."""
        programs, sample_s = self._programs()
        scale = self._cpu_scale(sample_s)
        return [Usage(item.name, len(item.pids), item.rss, item.ticks * scale) for item in programs]

    def _program_line(self, item: _Program, windows: dict[int, Window], scale: float) -> str:
        size = f"{item.rss / GIB:.1f} GB" if item.rss >= GIB else f"{item.rss // MIB} MB"
        count = len(item.pids)
        parts = [
            f"- {item.name}: {count} process{'es' if count != 1 else ''}",
            f"{size} of memory",
            f"{item.ticks * scale:.0f}% CPU",
        ]
        window = next((windows[pid] for pid in sorted(item.pids) if pid in windows), None)
        if window is not None and window.title.strip():
            parts.append(f'window "{_printable(window.title, MAX_TITLE)}"')
        with contextlib.suppress(OSError, ValueError):
            parts.append(self._age(item.start).replace("started", "newest started"))
        if item.never is not None and item.stoppable:
            parts.append(f"some of its processes are never stopped ({item.never})")
        elif item.never is not None:
            parts.append(f"never stopped ({item.never})")
        return ", ".join(parts)

    def _list_all(self, sort: str, count: int) -> Result:
        """List every running program of the owner's, sorted by `sort`; it binds no PID."""
        programs, sample_s = self._programs()
        scale = self._cpu_scale(sample_s)
        keys: dict[str, Callable[[_Program], tuple[int, str]]] = {
            "memory": lambda p: (-p.rss, p.name),
            "cpu": lambda p: (-p.ticks, p.name),
            "newest": lambda p: (-p.start, p.name),
        }
        programs.sort(key=keys[sort])
        shown = programs[:count]
        windows = self._windows_by_pid()
        processes = sum(len(item.pids) for item in programs)
        head = (
            f"The owner's running programs ({len(programs)}, {processes} processes), by {sort}; "
            f"CPU over {CPU_SAMPLE_S:g} s:"
        )
        lines = [head]
        lines.extend(self._program_line(item, windows, scale) for item in shown)
        if len(programs) > len(shown):
            lines.append(f"{len(programs) - len(shown)} more are not listed.")
        lines.append(
            "No PID here can be stopped: to stop one, call find_processes with its name first."
        )
        return Result("\n".join(lines), Origin.PROCESS)

    def find(self, args: dict[str, Any]) -> Result:
        """List the owner's stoppable processes that match the spoken name, or every program."""
        if "name" not in args:
            return self._list_all(
                str(args.get("sort", "memory")), int(args.get("count", LIST_COUNT))
            )
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
            self._shown.message = request.message if request is not None else 0
            for item in shown:
                title = "" if item.by_name or item.window is None else item.window.title
                self._shown.targets[item.proc.pid] = (
                    item.program,
                    item.proc.start,
                    _printable(title, MAX_TITLE),
                )
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
        rows = {item.proc.pid: inert(self._line(item)) for item in shown}
        return Result("\n".join(lines), Origin.PROCESS, delivered=self._reached(request, rows))

    def _reached(self, request: Request | None, rows: dict[int, str]) -> Callable[[str], None]:
        """Return what keeps only the PIDs whose whole row reached the model.

        The agent cuts a long result to its cap: a PID in a row it cut, or cut short, was never
        read, so it may not be stopped or watched.
        """

        def reached(body: str) -> None:
            with self._lock:
                if self._shown.request is not request:
                    return
                for pid, row in rows.items():
                    if f"\n{row}" not in body:
                        self._shown.targets.pop(pid, None)

        return reached

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
        outside = request.outside - {Origin.PROCESS.value, REMEMBERED_LISTING}
        if outside:
            msg = (
                "a process is stopped only on the owner's own words, and this request also holds "
                f"outside text ({', '.join(sorted(outside))}); nothing was stopped. Ask the owner "
                "to say which program to stop in a request of its own."
            )
            raise OperationError(msg)
        return request

    def _listed(self, request: Request, pid: int, verb: str = "stop") -> tuple[str, int, str]:
        """Return what `find_processes` listed for `pid` in `request`, read by the model.

        A PID written beside its listing, in the same model message, is a guess: refused, as a
        window number is.

        Raises:
            OperationError: the PID was not listed in this request, or not read yet.
        """
        with self._lock:
            target = self._shown.targets.get(pid) if self._shown.request is request else None
            unread = target is not None and request.message <= self._shown.message
        if target is None:
            msg = (
                f"PID {pid} was not listed by find_processes in this request. Call find_processes "
                f"with the name the owner said first, and {verb} only a PID it lists."
            )
            raise OperationError(msg)
        if unread:
            msg = (
                "The process list was not read yet: call find_processes alone, read it, then use "
                "a PID from it."
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
        reason = self._protection(proc, table, self._guarded(table))
        if reason is not None:
            msg = f"PID {pid} is never stopped: {reason}"
            raise OperationError(msg)

    def watched(self, pid: int) -> tuple[str, int]:
        """Return the program and start time of a PID `find_processes` listed in this request.

        The listing must have been read: a PID written beside it, in the same model message, is
        a guess, as for the window tools.

        Raises:
            OperationError: no owner's request is being handled, it holds outside text, the PID
                was not listed in it or not read yet, or it is no longer the process listed.
        """
        request = current_request()
        if request is None or request.foreign or request.scheduled:
            msg = (
                "a process is watched only on the owner's own request, never from PC audio or a job"
            )
            raise OperationError(msg)
        outside = request.outside - {Origin.PROCESS.value, REMEMBERED_LISTING}
        if outside:
            msg = (
                "a process is watched only on the owner's own words, and this request also holds "
                f"outside text ({', '.join(sorted(outside))}); no watch was set. Ask the owner to "
                "say which program to watch in a request of its own."
            )
            raise OperationError(msg)
        program, start, _ = self._listed(request, pid, "watch")
        proc = self._read(pid)
        if proc is None or proc.start != start:
            msg = f"PID {pid} is no longer the {program} that was listed"
            raise OperationError(msg)
        return program, start

    def state_of(self, pid: int, start: int) -> str | None:
        """Return the state letter (`R`, `S`, `D`, `T`, `Z`...) of the process started at `start`.

        None once it is gone: another process may have its PID by then.
        """
        sample = self.sample(pid, start, strict=False)
        return sample[0] if sample is not None else None

    def window_target(self, pid: int) -> tuple[str, int]:
        """Return the program and start time of a PID from a bound window.

        Raises:
            OperationError: the PID cannot be read or does not belong to the owner.
        """
        proc = self._read(pid) if pid > 1 else None
        if proc is None or not self._owned(proc):
            msg = "I can't read the process for that window."
            raise OperationError(msg)
        return self._program(proc), proc.start

    @property
    def ticks_per_s(self) -> int:
        """The clock tick rate used by the process stat counters."""
        return self._ticks

    def sample(self, pid: int, start: int, *, strict: bool = True) -> tuple[str, int] | None:
        """Return the process state and CPU ticks, or None if that process ended.

        In strict mode an I/O error other than a missing file is a failed check.
        """
        proc = self._read(pid, strict=strict)
        if proc is None or proc.start != start:
            return None
        state = proc.state
        if proc.state == "Z" and self._threads(pid, strict=strict) > 1:
            state = "S"  # its first thread ended, the others run: the program has not exited
        return state, proc.cpu_ticks

    def _threads(self, pid: int, *, strict: bool) -> int:
        """Return how many threads the process has, from its `status` (1 when it does not say)."""
        for line in self._text(pid, "status", strict=strict).splitlines():
            if line.startswith("Threads:"):
                with contextlib.suppress(ValueError, IndexError):
                    return int(line.split()[1])
        return 1

    def prepare(self, args: dict[str, Any]) -> dict[str, Any]:
        """Bind the PID to the program and start time `find_processes` listed in this request.

        It refuses (`OperationError`) when no owner's request is being handled or it was heard in
        PC audio, when `find_processes` did not list the PID in it, and when the process is no
        longer the one listed or is protected.
        """
        pid = int(args["pid"])
        program, start, window = self._listed(self._owner_request(), pid)
        self._check(pid, program, start)
        return {
            "pid": pid,
            "force": bool(args["force"]),
            "program": program,
            "started": start,
            "window": window,
        }

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
            return Result(f"{program} (PID {pid}) has ended.")
        return Result(
            f"Asked {program} (PID {pid}) to stop, but it is still running after "
            f"{STOP_WAIT_S:g} seconds; it may be asking to save. Only if the owner asks, stop it "
            "again with force=true.",
            ok=False,
            acted=True,
        )

    def operations(self) -> list[Operation]:
        """Return `find_processes` and `stop_process`."""
        return [
            Operation(
                "find_processes",
                "Find the owner's running programs by the name the owner said, before stopping "
                "any. Lists each match's PID for stop_process. Several: read them to the owner "
                "and ask which; never guess. None: say so; never try a wider name. Without a "
                "name: every running program with its memory and CPU, for what is running or "
                "using the most.",
                {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_NAME,
                            "description": "As the owner said it; leave out to list every program.",
                        },
                        "sort": {"type": "string", "enum": list(SORTS), "default": "memory"},
                        "count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_LISTED,
                            "default": LIST_COUNT,
                        },
                    },
                    "additionalProperties": False,
                },
                self.find,
                side_effect=False,
                spoken_summary=lambda a: f"look for {a.get('name', 'a program')}",
            ),
            Operation(
                "stop_process",
                "Stop one program the owner named, by a PID find_processes listed in this "
                "request; never any other. Refused when the request also read outside text (a "
                "page, a file, a picture). force=true only if the owner asks to force it.",
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
    target = f"{verb} {args.get('program', 'a program')} (PID {args.get('pid')})"
    window = args.get("window")
    if window:
        return f'{target}, all its windows, including the one titled "{window}"'
    return target
