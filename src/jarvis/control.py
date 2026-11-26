"""The control socket: `jarvis status`, `pause`, `resume`, `hush` and `confirm` reach Jarvis.

`jarvis run` serves a Unix socket in `$XDG_RUNTIME_DIR/jarvis/` (folder 0700, socket 0600) and
answers only processes of the same user (`SO_PEERCRED`). Each connection carries one JSON line
(`{"command": "status"}`) and gets one JSON line back. The socket does nothing while nobody
connects: its thread sleeps in `accept`.

`confirm` is the confirm key (`confirmkey`), which allows a `strong` action. As a best effort it
is refused from a program Jarvis started (`started_by_jarvis`: its descendants and, under systemd,
its control group). That is no guarantee: a program can leave both (a `systemd-run --user` unit,
`hyprctl dispatch exec`), and any program running as the owner can run `jarvis confirm`. An
operation that starts programs must not let the model start them that way (#19).

`subscribe` is the one request that keeps its connection: the service answers `{"ok": true,
"subscribed": true}` and then writes one JSON line for each announcement (`jarvis.announce`: a
reminder, a watch alert) as it is made, `{"announcement": {"id", "at", "kind", "text"}}`, until the
client closes. With `"after": N` in the request, the remembered announcements numbered above N
come first. A subscriber that cannot keep up loses its oldest lines (a `{"dropped": n}` line says
how many) and never slows the voice loop; at most `MAX_SUBSCRIBERS` stream at once. The same
same-user rule as every request applies.

The public state, which the Omarchy bar reads (#14), is one of `STATES`.
"""

import contextlib
import fcntl
import json
import logging
import os
import select
import socket
import stat
import struct
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from jarvis import announce
from jarvis.jsonutil import obj
from jarvis.paths import runtime_dir

log = logging.getLogger(__name__)

STATES = ("idle", "listening", "thinking", "speaking", "paused", "model loading", "unavailable")
COMMANDS = ("status", "pause", "resume", "hush", "confirm")
SUBSCRIBE = "subscribe"  # the one request that stays open and streams announcements
MAX_SUBSCRIBERS = 8
STREAM_POLL_S = 1.0  # how often a stream looks whether its client has gone
MAX_LINE = 1024  # bytes of one request
TIMEOUT_S = 2.0  # how long one connection may take to send its line
MAX_STREAM_LINE = 64 * 1024  # bytes of one line of a stream
_UCRED = struct.Struct("3i")  # pid, uid, gid

# The voice loop's states (voice.State values) as the bar shows them.
_LOOP_STATES = {
    "idle": "idle",
    "capture": "listening",
    "follow-up": "listening",
    "confirm": "listening",
    "transcribe": "thinking",
    "think": "thinking",
    "speak": "speaking",
}

NO_MIC = "no-mic"  # voice.State.NO_MIC: the loop waits to reopen a lost microphone
IDLE_WAIT_S = 120.0  # how long a pause waits for the request in flight before unloading
IDLE_POLL_S = 0.25

type Handler = Callable[[str], Mapping[str, Any]]


class _Loop(Protocol):
    @property
    def state(self) -> str: ...
    def hush(self) -> None: ...


class _Mic(Protocol):
    @property
    def paused(self) -> bool: ...
    def pause(self) -> bool: ...
    def resume(self) -> bool: ...


class _Key(Protocol):
    def press(self) -> dict[str, str | None]: ...
    def cancel(self) -> None: ...


NO_KEY = "this Jarvis has no confirm key"
FROM_JARVIS = "the confirm key cannot come from a program Jarvis started"


class _Audit(Protocol):
    def record(self, event: str, **fields: object) -> None: ...


class _NoAudit:
    def record(self, event: str, **fields: object) -> None:
        pass


class _Model(Protocol):
    def status(self) -> str | None: ...
    def file_missing(self) -> bool: ...
    def unload(self) -> bool: ...
    def warm(self, *, keep_going: Callable[[], bool] = ...) -> bool: ...
    def is_loaded(self) -> bool | None: ...
    def load(self) -> bool: ...


def _in_background(work: Callable[[], object]) -> None:
    threading.Thread(target=work, name="jarvis-control-work", daemon=True).start()


def socket_path() -> Path:
    """Return the control socket's path."""
    return runtime_dir() / "control.sock"


def _lock_path(path: Path) -> Path:
    return path.with_suffix(".lock")


def public_state(loop_state: str, *, paused: bool, model: str | None) -> str:
    """Map the loop's state, the pause and the model server's state to one of `STATES`.

    A request in flight shows what the loop does whatever the model's state; at rest, a model
    server that cannot be reached is `unavailable`, and a model on its way into memory is
    `model loading`. A loop waiting to reopen a lost microphone is `unavailable` too.
    """
    if paused:
        return "paused"
    if loop_state == NO_MIC:
        return "unavailable"
    shown = _LOOP_STATES.get(loop_state, "idle")
    if shown != "idle":
        return shown
    if model is None or model == "failed":
        return "unavailable"
    if model == "loading":
        return "model loading"
    return "idle"


def unreachable_model(model: _Model) -> str:
    """Name an unreachable model server's state: `missing` when the model file is not there."""
    return "missing" if model.file_missing() else "unreachable"


class Controls:
    """Carries out the control commands on the running voice loop (the server's handler).

    `pause` stops the speech, lets go of the microphone and, once the request in flight has
    ended, unloads the model; `resume` opens the microphone again and loads the model. Unloading
    and loading take turns, so a quick resume cannot end with the model unloaded. Each command
    that changes something is audited.
    """

    def __init__(  # ruff: ignore[too-many-arguments] - collaborators, each a test seam
        self,
        loop: _Loop,
        mic: _Mic,
        model: _Model,
        audit: _Audit | None = None,
        *,
        keys: _Key | None = None,
        prime: Callable[[], object] = lambda: None,
        background: Callable[[Callable[[], object]], None] = _in_background,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Act on `loop`, `mic` and `model`; slow work (unloading, loading) goes to `background`.

        `keys` is the confirm key `confirm` presses; it audits each press itself. `prime` has
        the model read its prompt once it is loaded at start or on resume (`Agent.prime`), so the
        first request does not wait for that.
        """
        super().__init__()
        self._keys = keys
        self._prime = prime
        self._loop = loop
        self._mic = mic
        self._model = model
        self._audit = audit or _NoAudit()
        self._model_turn = threading.Lock()  # an unload and a load never overlap
        self._commands = threading.Lock()
        self._background = background
        self._clock = clock
        self._sleep = sleep
        self._pauses = 0

    def __call__(self, command: str) -> dict[str, Any]:
        """Run one of `COMMANDS` and return what to answer."""
        match command:
            case "pause":
                return self._pause()
            case "resume":
                return self._resume()
            case "hush":
                if self._keys is not None:
                    self._keys.cancel()  # first: a press from now on finds nothing waiting
                self._loop.hush()
                self._audit.record("control", command="hush")
                return {}
            case "confirm":
                if self._keys is None:
                    return {"confirmed": None, "reason": NO_KEY}
                return dict(self._keys.press())
            case _:
                return self._status()

    def _status(self) -> dict[str, Any]:
        model = self._model.status()
        paused = self._mic.paused
        return {
            "state": public_state(str(self._loop.state), paused=paused, model=model),
            "paused": paused,
            "model": model or unreachable_model(self._model),
        }

    def _pause(self) -> dict[str, Any]:
        if self._keys is not None:
            self._keys.cancel()
        self._loop.hush()
        with self._commands:  # a pause and a resume arriving together are taken in turn
            changed = self._mic.pause()
            self._pauses += changed
            pause = self._pauses
        self._audit.record("control", command="pause", changed=changed)
        if changed:
            self._background(lambda: self._unload_when_idle(pause))
        return {"paused": True, "changed": changed}

    def _unload_when_idle(self, pause: int) -> None:
        deadline = self._clock() + IDLE_WAIT_S
        while self._in_flight() and self._clock() < deadline:
            self._sleep(IDLE_POLL_S)
        with self._model_turn:
            if pause == self._pauses and self._mic.paused:
                log.info("paused: unloading the model")
                self._model.unload()

    def _in_flight(self) -> bool:
        return str(self._loop.state) not in {"idle", NO_MIC}

    def warm_at_start(self) -> None:
        """Load the model in the background as the service starts, taking turns with a pause."""
        with self._commands:
            resume = self._pauses
        self._background(lambda: self._warm(resume))

    def prewarm(self) -> None:
        """Load the model ahead of a request a wake word starts, unless paused or already in.

        Called on its own thread; it takes turns with a pause's unload, so a pause that comes
        meanwhile still frees the model once the load is over.
        """
        with self._commands:
            resume = self._pauses
        with self._model_turn:
            if resume == self._pauses and not self._mic.paused and not self._model.is_loaded():
                self._model.load()

    def after_sleep(self) -> None:
        """After a suspend: check the model server, and load the model again if its load failed.

        A model that is unloaded stays so (`jarvis unload` before a game, or the idle unload),
        and so does a paused one. Called on the voice loop's thread; the check runs in the
        background.
        """
        with self._commands:
            resume = self._pauses
        self._background(lambda: self._recheck(resume))

    def _recheck(self, resume: int) -> None:
        state = self._model.status()
        log.info("back from sleep: the model is %s", state or unreachable_model(self._model))
        if state == "failed":
            self._warm(resume)  # skipped while paused

    def _warm(self, resume: int) -> None:
        def wanted() -> bool:  # not paused again meanwhile
            return resume == self._pauses and not self._mic.paused

        with self._model_turn:
            if not (wanted() and self._model.warm(keep_going=wanted)):
                return
            # A load still under way on the server is left alone: the first request reads then.
            if wanted() and self._model.is_loaded():
                self._prime()

    def _resume(self) -> dict[str, Any]:
        with self._commands:
            changed = self._mic.resume()
            self._pauses += changed  # a pause's unload still waiting is now stale
            resume = self._pauses
        self._audit.record("control", command="resume", changed=changed)
        if changed:
            self._background(lambda: self._warm(resume))
        return {"paused": False, "changed": changed}


def _peer(conn: socket.socket) -> tuple[int, int] | None:
    """Return the connecting process's PID and user ID."""
    try:
        raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _UCRED.size)
    except OSError:
        return None
    pid, uid, _gid = _UCRED.unpack(raw)
    return pid, uid


_PROC = Path("/proc")
_MAX_DEPTH = 4096  # more parents than this is a loop in a broken /proc: refuse


def _parent(pid: int, proc: Path) -> int:
    stat_line = (proc / str(pid) / "stat").read_text()
    # The name sits in parentheses and may hold anything, ") S 1 (" too: the fields after the
    # last ")" are the state, then the parent's PID.
    return int(stat_line.rsplit(")", 1)[1].split()[1])


def _cgroup(pid: int, proc: Path) -> str:
    return (proc / str(pid) / "cgroup").read_text().strip()


def started_by_jarvis(pid: int, *, me: int | None = None, proc: Path = _PROC) -> bool:
    """Return True if `pid` is Jarvis (`me`), runs under it, or cannot be checked.

    Under it means a descendant, or, when Jarvis runs as a systemd service, any process in the
    service's control group: a child that forked again and was handed to another parent stays
    there. A process that is gone before it can be checked counts as started by Jarvis.
    """
    me = os.getpid() if me is None else me
    try:
        return _in_own_service(pid, me, proc) or _descends(pid, me, proc)
    except OSError, ValueError, IndexError:
        return True


def _in_own_service(pid: int, me: int, proc: Path) -> bool:
    own = _cgroup(me, proc)
    return own.endswith(".service") and _cgroup(pid, proc) == own


def _descends(pid: int, me: int, proc: Path) -> bool:
    current = pid
    for _ in range(_MAX_DEPTH):
        if current == me:
            return True
        if current <= 1:
            return False
        current = _parent(current, proc)
    return True  # a loop of parents: refuse


def _read_line(conn: socket.socket) -> bytes | None:
    data = b""
    while b"\n" not in data:
        chunk = conn.recv(MAX_LINE + 1 - len(data))
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_LINE:
            return None
    return data.split(b"\n", 1)[0]


def _command(line: bytes | None) -> str | None:
    if line is None:
        return None
    try:
        command = obj(json.loads(line)).get("command")
    except ValueError:
        return None
    return command if isinstance(command, str) else None


def _after(line: bytes | None) -> int | None:
    """Return the announcement number a `subscribe` asks to continue after, or None."""
    if line is None:
        return None
    try:
        after = obj(json.loads(line)).get("after")
    except ValueError:
        return None
    return after if isinstance(after, int) and not isinstance(after, bool) and after >= 0 else None


def _client_gone(conn: socket.socket) -> bool:
    """Whether the client of a streaming connection closed it (what it sends is ignored)."""
    if not select.select([conn], [], [], 0)[0]:
        return False
    try:
        return not conn.recv(4096)
    except OSError:
        return True


class ControlServer:
    """Serves the control socket on a thread of its own until `close`."""

    def __init__(
        self,
        handler: Handler,
        path: Path | None = None,
        audit: _Audit | None = None,
        *,
        started_by_jarvis: Callable[[int], bool] = started_by_jarvis,
        hub: announce.Hub | None = None,
    ) -> None:
        """Answer each known command with `handler(command)`; `path` defaults to `socket_path`.

        `subscribe` streams the announcements of `hub` (this process's `announce.HUB` by default).

        A connection refused for coming from another user, or a `confirm` from a program
        `started_by_jarvis`, is recorded in `audit`.
        """
        super().__init__()
        self._handler = handler
        self._started_by_jarvis = started_by_jarvis
        self.path = path or socket_path()
        self._audit = audit or _NoAudit()
        self._hub = hub or announce.HUB
        self._streams = threading.BoundedSemaphore(MAX_SUBSCRIBERS)
        self._sock: socket.socket | None = None
        self._lock_fd: int | None = None
        self._thread: threading.Thread | None = None
        self._closed = threading.Event()

    def start(self) -> None:
        """Bind the socket (replacing a stale one) and start answering.

        Raises:
            OSError: the folder or socket could not be made, or another Jarvis is serving it.
        """
        folder = self.path.parent
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = folder.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            msg = f"{folder} is not a folder of this user"
            raise OSError(msg)
        folder.chmod(0o700)
        # Held while serving: a second Jarvis stops here rather than replacing the socket.
        lock = os.open(_lock_path(self.path), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(lock)
            msg = f"another Jarvis is already serving {self.path}"
            raise OSError(msg) from None
        try:
            sock = self._bind()
        except OSError:
            os.close(lock)
            raise
        self._lock_fd = lock
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, name="jarvis-control", daemon=True)
        self._thread.start()

    def _bind(self) -> socket.socket:
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()  # left by a run that was killed
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(self.path))  # in a 0700 folder, so nobody else can reach it meanwhile
            self.path.chmod(0o600)
            sock.listen(8)
        except OSError:
            sock.close()
            raise
        return sock

    def _serve(self) -> None:
        sock = self._sock
        assert sock is not None  # ruff: ignore[assert] - only started after binding
        while not self._closed.is_set():
            try:
                conn, _ = sock.accept()
            except OSError:
                return  # closed
            # Each connection on its own thread: a slow `status` never holds up a `hush`.
            threading.Thread(
                target=self._connection, args=(conn,), name="jarvis-control-conn", daemon=True
            ).start()

    def _connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                self._answer(conn)
            except OSError as exc:
                log.info("control connection failed: %s", exc)
            except Exception:
                log.exception("control connection failed")

    def _answer(self, conn: socket.socket) -> None:
        peer = _peer(conn)
        if peer is None or peer[1] != os.getuid():
            log.warning("refused a control connection from another user")
            self._audit.record("control_refused", peer_uid=None if peer is None else peer[1])
            return
        conn.settimeout(TIMEOUT_S)
        line = _read_line(conn)
        command = _command(line)
        if command == SUBSCRIBE:
            self._stream(conn, _after(line))
            return
        if command is None:
            reply: Mapping[str, Any] = {"ok": False, "error": "send one JSON line with a command"}
        elif command not in COMMANDS:
            reply = {"ok": False, "error": f"unknown command {command!r}"}
        elif command == "confirm" and self._started_by_jarvis(peer[0]):
            log.warning("refused the confirm key from a program Jarvis started (PID %d)", peer[0])
            self._audit.record(
                "control_refused", command="confirm", reason="started by Jarvis", peer_pid=peer[0]
            )
            reply = {"ok": False, "error": FROM_JARVIS}
        else:
            try:
                reply = {"ok": True, **self._handler(command)}
            except Exception:
                log.exception("control command %s failed", command)
                reply = {"ok": False, "error": f"{command} failed; see the log"}
        conn.sendall(json.dumps(reply).encode() + b"\n")

    def _stream(self, conn: socket.socket, after: int | None) -> None:
        """Write each announcement as one JSON line until the client goes, or this closes."""
        if not self._streams.acquire(blocking=False):
            refusal = {"ok": False, "error": f"at most {MAX_SUBSCRIBERS} subscribers at once"}
            conn.sendall(json.dumps(refusal).encode() + b"\n")
            return
        try:
            with self._hub.subscribe(after) as feed:
                conn.sendall(json.dumps({"ok": True, "subscribed": True}).encode() + b"\n")
                while not self._closed.is_set() and not _client_gone(conn):
                    if lost := feed.take_dropped():
                        conn.sendall(json.dumps({"dropped": lost}).encode() + b"\n")
                    item = feed.get(STREAM_POLL_S)
                    if item is not None:
                        line = json.dumps({"announcement": item.to_json()})
                        conn.sendall(line.encode() + b"\n")  # a client that stalls times out
        finally:
            self._streams.release()

    def close(self) -> None:
        """Stop answering and remove the socket."""
        self._closed.set()
        sock, self._sock = self._sock, None
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()  # this run's own: the lock says no other run serves it
        lock, self._lock_fd = self._lock_fd, None
        if lock is not None:
            os.close(lock)
        if self._thread is not None:
            self._thread.join(TIMEOUT_S)


def send(command: str, path: Path | None = None, timeout: float = 10.0) -> dict[str, Any] | None:
    """Send `command` to the running Jarvis; return its reply, or None when none is running.

    A Jarvis that is running but does not answer in time gets an `ok: false` reply.
    """
    target = path or socket_path()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout)
        try:
            conn.connect(str(target))
        except OSError:
            return None
        try:
            conn.sendall(json.dumps({"command": command}).encode() + b"\n")
            line = _read_reply(conn)
        except OSError:
            return {"ok": False, "error": "Jarvis did not answer in time"}
    try:
        return obj(json.loads(line))
    except ValueError:
        return {"ok": False, "error": "Jarvis's answer could not be read"}


class Feed:
    """A subscription to the running Jarvis's announcements: iterate it, and `close` it."""

    def __init__(self, conn: socket.socket, pending: bytes) -> None:
        """Read announcements from `conn`, which already read `pending` past its first line."""
        super().__init__()
        self._conn = conn
        self._buffer = pending
        self._lock = threading.Lock()
        self.dropped = 0  # announcements the service dropped for being read too slowly

    def __iter__(self) -> Feed:
        """Iterate over the announcements as they come."""
        return self

    def __next__(self) -> announce.Announcement:
        """Return the next announcement, waiting for it.

        Raises:
            StopIteration: the service ended the stream, or this was closed.
        """
        while True:
            line = self._line()
            if line is None:
                raise StopIteration
            found = _json_object(line)
            if isinstance(found.get("dropped"), int):
                self.dropped += found["dropped"]
                continue
            item = announce.Announcement.from_json(found.get("announcement"))
            if item is not None:
                return item

    def _line(self) -> bytes | None:
        while b"\n" not in self._buffer:
            if len(self._buffer) > MAX_STREAM_LINE:
                return None  # not what the service writes
            try:
                chunk = self._conn.recv(4096)
            except OSError:
                return None
            if not chunk:
                return None
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line

    def close(self) -> None:
        """Close the connection; a read waiting on it ends."""
        with contextlib.suppress(OSError):
            self._conn.shutdown(socket.SHUT_RDWR)
        self._conn.close()


def _json_object(line: bytes) -> dict[str, Any]:
    try:
        return obj(json.loads(line))
    except ValueError:
        return {}


def _subscribed(conn: socket.socket, request: dict[str, object]) -> bytes | None:
    """Ask for the stream; return what followed the answer's line, or None if it was refused."""
    conn.sendall(json.dumps(request).encode() + b"\n")
    data = b""
    while b"\n" not in data:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    first, _, rest = data.partition(b"\n")
    return rest if _json_object(first).get("subscribed") is True else None


def subscribe(
    path: Path | None = None, *, after: int | None = None, timeout: float = 10.0
) -> Feed | None:
    """Follow the running Jarvis's announcements; None when none runs or it refuses.

    With `after`, the announcements it remembers above that number come first.
    """
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    request: dict[str, object] = {"command": SUBSCRIBE}
    if after is not None:
        request["after"] = after
    try:
        conn.connect(str(path or socket_path()))
        rest = _subscribed(conn, request)
    except OSError:
        rest = None
    if rest is None:
        conn.close()
        return None
    conn.settimeout(None)  # waits for the next announcement for as long as it takes
    return Feed(conn, rest)


def _read_reply(conn: socket.socket) -> bytes:
    data = b""
    while b"\n" not in data:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    return data.split(b"\n", 1)[0]
