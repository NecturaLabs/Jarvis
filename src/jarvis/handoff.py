"""Handing a spoken instruction to an open chat window, over a local owner-only socket.

The voice service and the chat are two configurations that work apart: a spoken request is the
voice agent's own and never shows in the chat. When the owner asks, in speech, for something to be
done in or added to the chat's own work, the voice agent calls `chat_hand_off` and the words go to
the window as the owner's own input: it joins the request that is running (steering) or starts a
new one, and marks the words as spoken.

Each chat window serves a Unix socket of its own in `$XDG_RUNTIME_DIR/jarvis/chat/` (folder
0700, socket 0600) and answers only processes of the same user (`SO_PEERCRED`), as the control
socket does (`control`). A connection carries one JSON line (`{"spoken": "..."}`) and gets one
back (`{"ok": true, "state": "steered"}`). The newest window that answers is the open chat.
"""

import contextlib
import json
import logging
import os
import socket
import stat
import threading
from enum import StrEnum
from typing import TYPE_CHECKING

from jarvis import paths
from jarvis.jsonutil import obj
from jarvis.operations import REMEMBERED, REMEMBERED_LISTING, Operation, OperationError, Result
from jarvis.operations import current_request as request_now

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

log = logging.getLogger(__name__)

NAME = "chat_hand_off"
MAX_LINE = 8192  # bytes of one line: the spoken words are at most 4000 characters
MAX_CHARS = 4000
TIMEOUT_S = 5.0  # how long a window may take to take the words


class State(StrEnum):
    """What the chat did with handed-over words."""

    STEERED = "steered"  # joined the request it is running, before the model's next call
    QUEUED = "queued"  # waits for the running request to end
    STARTED = "started"  # began as a new request


class HandoffError(Exception):
    """The words did not reach a chat window, or it refused them; the message says why."""


def chats_dir() -> Path:
    """Return the folder of the chat windows' sockets."""
    return paths.runtime_dir() / "chat"


def socket_path(session: str) -> Path:
    """Return the socket of the chat window `session`."""
    return chats_dir() / f"{session}.sock"


def _sockets(folder: Path) -> list[Path]:
    """Return the sockets in `folder`, newest first."""
    try:
        found = [p for p in folder.glob("*.sock") if p.is_socket()]
    except OSError:
        return []
    return sorted(found, key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)


def _connect(path: Path, timeout: float) -> tuple[socket.socket | None, bool]:
    """Connect to a window's socket; return it (or None) and whether nobody listens on it."""
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    try:
        conn.connect(str(path))
    except OSError as exc:
        conn.close()
        return None, isinstance(exc, ConnectionRefusedError | FileNotFoundError)
    return conn, False


def open_chats(folder: Path | None = None) -> list[Path]:
    """Return the sockets of the chat windows that answer, newest first.

    A socket nobody answers on was left by a window that was killed; it is removed.
    """
    found: list[Path] = []
    for path in _sockets(folder or chats_dir()):
        conn, dead = _connect(path, 1.0)
        if conn is None:
            if dead:
                with contextlib.suppress(OSError):
                    path.unlink()
            continue
        conn.close()
        found.append(path)
    return found


def chat_open(folder: Path | None = None) -> bool:
    """Whether a chat window is open."""
    return bool(open_chats(folder))


def send(words: str, folder: Path | None = None, timeout: float = TIMEOUT_S) -> State:
    """Hand `words` to the newest open chat window and return what it did with them.

    Raises:
        HandoffError: no chat window is open, or it did not take the words.
    """
    for path in open_chats(folder):
        conn, _dead = _connect(path, timeout)
        if conn is None:
            continue
        with conn:
            try:
                conn.sendall(json.dumps({"spoken": words}).encode() + b"\n")
                line = _read_line(conn)
                reply = obj(json.loads(line))
            except (OSError, ValueError) as exc:
                msg = "the chat window did not answer; it may have the words, so do not resend"
                raise HandoffError(msg) from exc
        if reply.get("ok") is True:
            return State(str(reply.get("state")))
        raise HandoffError(str(reply.get("error", "the chat window refused")))
    msg = "no chat window is open"
    raise HandoffError(msg)


def _read_line(conn: socket.socket) -> bytes:
    data = b""
    while b"\n" not in data:
        chunk = conn.recv(MAX_LINE + 1 - len(data))
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_LINE:
            msg = "line too long"
            raise ValueError(msg)
    return data.split(b"\n", 1)[0]


class ChatListener:
    """Serves one chat window's socket on a thread of its own until `close`.

    `take` is called with the spoken words (on the connection's thread) and returns what the
    window did with them; it raises `HandoffError` to refuse them. A connection that sends
    nothing (a probe: is this window open?) is closed without a call.
    """

    def __init__(
        self,
        session: str,
        take: Callable[[str], State],
        path: Path | None = None,
        *,
        started_by_jarvis: Callable[[int], bool] | None = None,
    ) -> None:
        """Answer on `path` (default: this `session`'s socket) with `take`.

        A connection from a program this window started (`started_by_jarvis`, as for the control
        socket's `confirm`) is refused: its words would pass as the owner's.
        """
        super().__init__()
        self._take = take
        self._started_by_jarvis = started_by_jarvis
        self.path = path or socket_path(session)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._closed = threading.Event()

    def start(self) -> None:
        """Bind the socket and start answering.

        Raises:
            OSError: the folder or socket could not be made.
        """
        folder = self.path.parent
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = folder.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            msg = f"{folder} is not a folder of this user"
            raise OSError(msg)
        folder.chmod(0o700)
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(self.path))
            self.path.chmod(0o600)
            sock.listen(4)
        except OSError:
            sock.close()
            raise
        self._sock = sock
        self._thread = threading.Thread(target=self._serve, name="jarvis-chat-handoff", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        sock = self._sock
        assert sock is not None  # ruff: ignore[assert] - only started after binding
        while not self._closed.is_set():
            try:
                conn, _ = sock.accept()
            except OSError:
                return  # closed
            threading.Thread(
                target=self._connection, args=(conn,), name="jarvis-chat-conn", daemon=True
            ).start()

    def _connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                self._answer(conn)
            except OSError as exc:
                log.info("chat hand-off connection failed: %s", exc)
            except Exception:
                log.exception("chat hand-off connection failed")

    def _answer(self, conn: socket.socket) -> None:
        from jarvis.control import peer_credentials, started_by_jarvis  # ruff: ignore[import-outside-top-level] - the socket server, kept out of programs that only ask

        creds = peer_credentials(conn)
        if creds is None or creds[1] != os.getuid():
            log.warning("refused a chat hand-off from another user")
            return
        if (self._started_by_jarvis or started_by_jarvis)(creds[0]):
            log.warning(
                "refused a chat hand-off from a program this window started (PID %d)", creds[0]
            )
            conn.sendall(b'{"ok": false, "error": "not from a program the chat started"}\n')
            return
        conn.settimeout(TIMEOUT_S)
        try:
            line = _read_line(conn)
        except ValueError:
            conn.sendall(b'{"ok": false, "error": "too long"}\n')
            return
        if not line:
            return  # a probe
        reply: dict[str, object]
        try:
            words = obj(json.loads(line)).get("spoken")
        except ValueError:
            words = None
        if not isinstance(words, str) or not words.strip():
            reply = {"ok": False, "error": "send one JSON line with the spoken words"}
        else:
            try:
                reply = {"ok": True, "state": self._take(words.strip()[:MAX_CHARS]).value}
            except HandoffError as exc:
                reply = {"ok": False, "error": str(exc)}
        conn.sendall(json.dumps(reply).encode() + b"\n")

    def close(self) -> None:
        """Stop answering and remove the socket."""
        self._closed.set()
        sock, self._sock = self._sock, None
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()
        if self._thread is not None:
            self._thread.join(TIMEOUT_S)


ABOUT = (
    "Call this alone when the owner asks, by voice, for something to be done in or added to the "
    "ongoing work or conversation of the chat window they have open: a request that belongs to "
    "that chat's own work (go on with it, add this to it, tell it something). Their words go to "
    "the chat as their own and the chat does it and shows it. Never for something to do yourself "
    "right now, however it is put: playing music, the volume, the time, a question or a task of "
    "the PC are yours to do, and they do not show in the chat."
)


def operation(deliver: Callable[[str], State] = send) -> Operation:
    """Return `chat_hand_off`: the voice agent's way to pass the owner's words to the open chat.

    It sends the owner's own words of this request, never words the model wrote, so what the
    chat receives is what the owner said. It is refused for a request heard in PC audio, one run
    by a schedule, and one that read outside text, whose words are not the owner's alone.
    """

    def run(_args: dict[str, object]) -> Result:
        request = request_now()
        if request is None or request.foreign or request.scheduled:
            msg = "Only a request the owner speaks just now goes to the chat."
            raise OperationError(msg)
        if request.outside - {REMEMBERED, REMEMBERED_LISTING}:
            msg = "This request read outside text: answer it yourself instead."
            raise OperationError(msg)
        words = request.words
        if not words:
            msg = "There are no spoken words to hand over."
            raise OperationError(msg)
        try:
            state = deliver(words)
        except HandoffError as exc:
            msg = f"It did not go to the chat: {exc}."
            raise OperationError(msg) from exc
        line = {
            State.STEERED: "Added to what the chat is working on.",
            State.QUEUED: "Queued in the chat.",
            State.STARTED: "Started in the chat.",
        }[state]
        return Result(f"Handed to the open chat ({state.value}).", spoken=line, ends=True)

    return Operation(
        "chat_hand_off",  # NAME
        ABOUT,
        {"type": "object", "properties": {}, "additionalProperties": False},
        run,
        side_effect=False,
        spoken_summary=lambda _args: "send that to the chat",
        offered=chat_open,
    )
