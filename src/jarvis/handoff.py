"""Handing a spoken instruction to an open chat window, over a local owner-only socket.

The voice service and the chat are two configurations: a spoken request is the voice agent's own
(with `[channels] shared` on it shows in the chat and joins its conversation, ADR 0030). When the
owner asks, in speech, for something to be done in or added to the chat's own work, the voice agent
calls `chat_hand_off` and the words go to the window as the owner's own input: it joins the
request that is running (steering) or starts a new one, and marks the words as spoken.

Each chat window serves a Unix socket of its own in `$XDG_RUNTIME_DIR/jarvis/chat/` (folders
0700, socket 0600) and takes words only from the voice service process itself (`SO_PEERCRED`: its
PID is the one serving the control socket, never a program it started). A connection carries one
JSON line (`{"spoken": "..."}`) and gets one back (`{"ok": true, "state": "steered"}`). The newest
window that answers is the open chat.
"""

import contextlib
import json
import logging
import os
import socket
import stat
import threading
import time
import unicodedata
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

from jarvis import paths
from jarvis.jsonutil import obj
from jarvis.operations import (
    REMEMBERED,
    REMEMBERED_LISTING,
    Operation,
    OperationError,
    Result,
    UncertainOutcomeError,
)
from jarvis.operations import current_request as request_now

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

log = logging.getLogger(__name__)

NAME = "chat_hand_off"
MAX_CHARS = 4000  # of the spoken words
MAX_LINE = 4 * MAX_CHARS + 512  # bytes of one line: UTF-8 takes at most 4 bytes a character
TIMEOUT_S = 5.0  # how long a window may take to answer
PROBE_S = 0.2  # how long a probe for an open window waits for a connection
CACHE_S = 3.0  # how long "is a chat open?" is remembered: it is asked on every model call


class State(StrEnum):
    """What the chat did with handed-over words."""

    STEERED = "steered"  # joined the request it is running, before the model's next call
    QUEUED = "queued"  # waits for the running request to end
    STARTED = "started"  # began as a new request


class Sent(NamedTuple):
    """What a hand-off did, and how many chat windows were open (the newest took the words)."""

    state: State
    windows: int = 1


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
        return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []


def _connect(path: Path, timeout: float) -> socket.socket | None:
    """Connect to a window's socket; None when nothing answers there."""
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    try:
        conn.connect(str(path))
    except OSError:
        conn.close()
        return None
    return conn


def open_chats(folder: Path | None = None) -> list[Path]:
    """Return the sockets of the chat windows that answer, newest first.

    A socket nobody answers on is skipped, never removed here: a window that is between `bind`
    and `listen` would lose its socket. A window removes its own stale one when it starts.
    """
    found: list[Path] = []
    for path in _sockets(folder or chats_dir()):
        if (conn := _connect(path, PROBE_S)) is not None:
            conn.close()
            found.append(path)
    return found


_seen: list[float | bool] = [float("-inf"), False]  # when chat_open last looked, and what it saw


def chat_open(folder: Path | None = None, *, clock: Callable[[], float] = time.monotonic) -> bool:
    """Whether a chat window is open, as it was a few seconds ago at most.

    The voice agent asks on every model call, so a window that does not answer costs one probe
    in `CACHE_S`, not one in each call.
    """
    now = clock()
    if folder is None and 0 <= now - float(_seen[0]) < CACHE_S:
        return bool(_seen[1])
    found = bool(open_chats(folder))
    if folder is None:
        _seen[0], _seen[1] = now, found
    return found


def forget_chat_probe() -> None:
    """Drop what `chat_open` remembers (a window that opened or closed just now)."""
    _seen[0], _seen[1] = float("-inf"), False


def send(words: str, folder: Path | None = None, timeout: float = TIMEOUT_S) -> Sent:
    """Hand `words` to the newest open chat window and return what it did with them.

    Raises:
        HandoffError: no chat window is open, or it refused the words: they were not taken.
        UncertainOutcomeError: the words were sent and the window did not answer in time or
            sent nothing readable, so it may or may not have them.
    """
    windows = open_chats(folder)
    for path in windows:
        if (conn := _connect(path, timeout)) is None:
            continue
        line = (
            json.dumps({"spoken": _plain(words)[:MAX_CHARS]}, ensure_ascii=False).encode() + b"\n"
        )
        with conn:
            try:
                conn.sendall(line)
                reply = obj(json.loads(_read_line(conn)))
                sent = (
                    Sent(State(str(reply.get("state"))), len(windows))
                    if reply.get("ok") is True
                    else None
                )
            except (OSError, ValueError) as exc:
                msg = "the chat window did not answer in time, so it may or may not have the words"
                raise UncertainOutcomeError(msg) from exc
        if sent is not None:
            return sent
        raise HandoffError(str(reply.get("error", "the chat window refused")))
    msg = "no chat window is open"
    raise HandoffError(msg)


def _plain(words: str) -> str:
    """Return `words` with each control character but the line break a space.

    JSON writes one as a six-byte escape, which would let a transcript of them outgrow the line.
    """
    return "".join(" " if unicodedata.category(ch) == "Cc" and ch != "\n" else ch for ch in words)


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


def _private_folders(folder: Path) -> None:
    """Make `folder` and the one above it (Jarvis's runtime folder), owner-only.

    Raises:
        OSError: one of them is not a folder of this user.
    """
    for path in (folder.parent, folder):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            msg = f"{path} is not a folder of this user"
            raise OSError(msg)
        path.chmod(0o700)


def _from_the_voice_service(pid: int) -> bool:
    """Whether `pid` is the voice service process itself, not a program it started."""
    from jarvis.control import serving_pid  # ruff: ignore[import-outside-top-level] - the socket server, kept out of programs that only ask

    return pid == serving_pid()


class ChatListener:
    """Serves one chat window's socket on a thread of its own until `close`.

    `take` is called with the spoken words (on the connection's thread) and returns what the
    window did with them; it raises `HandoffError` to refuse them. A connection that sends
    nothing (a probe: is this window open?) is closed without a call. Only a peer for which
    `allowed` says yes is served (default: the voice service process); `refused` is told why
    each other one was not, and why `take` refused.
    """

    def __init__(
        self,
        session: str,
        take: Callable[[str], State],
        path: Path | None = None,
        *,
        allowed: Callable[[int], bool] | None = None,
        refused: Callable[[str], None] = lambda _why: None,
    ) -> None:
        """Answer on `path` (default: this `session`'s socket) with `take`."""
        super().__init__()
        self._take = take
        self._allowed = allowed or _from_the_voice_service
        self._refused = refused
        self.path = path or socket_path(session)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._closed = threading.Event()

    def start(self) -> None:
        """Bind the socket (replacing this window's own stale one) and start answering.

        Raises:
            OSError: the folders or the socket could not be made.
        """
        _private_folders(self.path.parent)
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
        forget_chat_probe()

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
        from jarvis.control import peer_credentials  # ruff: ignore[import-outside-top-level] - the socket server, kept out of programs that only ask

        conn.settimeout(TIMEOUT_S)
        too_long = False
        try:
            line = _read_line(conn)
        except ValueError:
            line, too_long = b"", True
        if not line and not too_long:
            return  # a probe
        creds = peer_credentials(conn)
        if creds is None or creds[1] != os.getuid():
            self._refuse(conn, "another user")
            return
        if not self._allowed(creds[0]):
            self._refuse(conn, f"not the voice service (PID {creds[0]})")
            return
        if too_long:
            self._refused("too long")
            conn.sendall(b'{"ok": false, "error": "too long"}\n')
            return
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
                self._refused(str(exc))
                reply = {"ok": False, "error": str(exc)}
        conn.sendall(json.dumps(reply).encode() + b"\n")

    def _refuse(self, conn: socket.socket, why: str) -> None:
        log.warning("refused a chat hand-off: %s", why)
        self._refused(why)
        conn.sendall(b'{"ok": false, "error": "only the voice service hands words over"}\n')

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
        forget_chat_probe()


ABOUT = (
    "Call this alone when the owner asks, by voice, for something to be done in or added to the "
    "ongoing work or conversation of the chat window they have open: a request that belongs to "
    "that chat's own work (go on with it, add this to it, tell it something). Their words go to "
    "the chat as their own and the chat does it and shows it, and the request ends there with one "
    "short line spoken. Only when the whole request belongs to the chat: if it also holds "
    "something for you to do, do that yourself and say the chat part is for a request of its "
    "own. Never for something to do yourself right now, however it is put: playing music, the "
    "volume, the time, a question or a task of the PC are yours to do, and they do not show in "
    "the chat."
)


def operation(deliver: Callable[[str], Sent] = send) -> Operation:
    """Return `chat_hand_off`: the voice agent's way to pass the owner's words to the open chat.

    It sends the owner's own words of this request, never words the model wrote, so what the
    chat receives is what the owner said. It is refused for a request heard in PC audio, one run
    by a schedule, one that read outside text, one that holds other calls or already tried an
    action (its words would carry that work into the chat too), and runs as the request's action,
    so what the owner hears is not taken for a claim of work that was not done.
    """

    def run(_args: dict[str, object]) -> Result:
        request = request_now()
        if request is None or request.foreign or request.scheduled:
            msg = "Only a request the owner speaks just now goes to the chat."
            raise OperationError(msg)
        if request.outside - {REMEMBERED, REMEMBERED_LISTING}:
            msg = "This request read outside text: answer it yourself instead."
            raise OperationError(msg)
        if request.beside > 1 or request.acted:
            msg = "The request holds other work, so it cannot go to the chat whole: do it yourself."
            raise OperationError(msg)
        words = request.words
        if not words:
            msg = "There are no spoken words to hand over."
            raise OperationError(msg)
        try:
            sent = deliver(words)
        except HandoffError as exc:
            msg = f"It did not go to the chat: {exc}."
            raise OperationError(msg) from exc
        line = {
            State.STEERED: "Added to what the chat is working on.",
            State.QUEUED: "Queued in the chat.",
            State.STARTED: "Started in the chat.",
        }[sent.state]
        if sent.windows > 1:
            line = f"{line.removesuffix('.')}, in the newest chat window."
        return Result(f"Handed to the open chat ({sent.state.value}).", spoken=line, ends=True)

    return Operation(
        "chat_hand_off",  # NAME
        ABOUT,
        {"type": "object", "properties": {}, "additionalProperties": False},
        run,
        spoken_summary=lambda _args: "send that to the chat",
        offered=chat_open,
    )
