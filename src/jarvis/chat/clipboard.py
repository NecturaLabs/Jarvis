"""Reading the Wayland clipboard for Ctrl+V in the chat: a copied picture, else copied text.

The owner asked for Ctrl+V to paste a picture as it does in Claude Code; the terminal's own paste
(Ctrl+Shift+V) carries only text. `wl-paste` (wl-clipboard) is started with a fixed argument list:
the only thing that varies is the media type, and that is one of the names below, never one the
clipboard offered. Its output is read up to a cap and within a timeout, and a reader that runs
over either is stopped.
"""

import contextlib
import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import IO

PROGRAM = "wl-paste"
TIMEOUT_S = 3.0
MAX_BYTES = 20 * 1024 * 1024
MAX_ERROR_CHARS = 200
READ_CHUNK = 1 << 16
IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp")
TEXT_TYPES = ("text/plain;charset=utf-8", "UTF8_STRING", "text/plain", "STRING", "TEXT")
URI_TYPE = "text/uri-list"
_EMPTY = ("Nothing is copied", "No selection", "No suitable type")


class ClipboardError(Exception):
    """The clipboard could not be read; the message says why in plain words."""


@dataclass(frozen=True, slots=True)
class Clip:
    """What is copied: a picture's bytes and media type, or text (nothing for an empty one)."""

    text: str = ""
    image: bytes = b""
    mime: str = ""

    def __bool__(self) -> bool:
        """Whether anything was copied."""
        return bool(self.text or self.image)


def _collect(
    stdout: IO[bytes], stderr: IO[bytes], timeout_s: float, max_bytes: int
) -> tuple[bytes, str]:
    """Return what was printed on `stdout` and the start of `stderr`, within the limits.

    Raises:
        ClipboardError: more than `max_bytes` was printed, or it took longer than `timeout_s`.
    """
    deadline = time.monotonic() + timeout_s
    out: list[bytes] = []
    size = 0
    err = b""
    with selectors.DefaultSelector() as selector:
        selector.register(stdout, selectors.EVENT_READ, "out")
        selector.register(stderr, selectors.EVENT_READ, "err")
        while selector.get_map():
            left = deadline - time.monotonic()
            if left <= 0:
                msg = "The clipboard didn't answer in time"
                raise ClipboardError(msg)
            for key, _ in selector.select(left):
                chunk = os.read(key.fd, READ_CHUNK)
                if not chunk:
                    selector.unregister(key.fileobj)
                elif key.data == "out":
                    size += len(chunk)
                    if size > max_bytes:
                        msg = (
                            f"What's copied is too big to paste (over {max_bytes // 1_048_576} MB)"
                        )
                        raise ClipboardError(msg)
                    out.append(chunk)
                else:
                    err = (err + chunk)[:MAX_ERROR_CHARS]
    return b"".join(out), err.decode("utf-8", errors="replace")


def _stop(proc: subprocess.Popen[bytes]) -> None:
    """Stop the reader this module started, and anything it started in turn."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)


def _run(argv: list[str], timeout_s: float, max_bytes: int) -> tuple[int, bytes, str]:
    """Return the exit status, output and error text of `argv`.

    Raises:
        ClipboardError: the program is missing, ran over the limits, or could not start.
    """
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        msg = "Pasting a picture needs wl-paste, from the wl-clipboard package"
        raise ClipboardError(msg) from exc
    except OSError as exc:
        msg = f"The clipboard couldn't be read: {exc.strerror or type(exc).__name__}"
        raise ClipboardError(msg) from exc
    with proc:
        stdout, stderr = proc.stdout, proc.stderr
        if stdout is None or stderr is None:  # both are pipes: never, but the types allow it
            _stop(proc)
            msg = "The clipboard couldn't be read"
            raise ClipboardError(msg)
        try:
            out, err = _collect(stdout, stderr, timeout_s, max_bytes)
        except ClipboardError:
            _stop(proc)
            raise
        try:
            code = proc.wait(timeout=max(0.1, timeout_s))
        except subprocess.TimeoutExpired as exc:
            _stop(proc)
            msg = "The clipboard didn't answer in time"
            raise ClipboardError(msg) from exc
    return code, out, err


def _failure(err: str) -> str | None:
    """Return why a failed `wl-paste` failed, or None when it failed because nothing is copied."""
    line = err.strip().splitlines()[0] if err.strip() else ""
    if not line or line.startswith(_EMPTY):
        return None
    return f"The clipboard couldn't be read: {line}"


def read(
    program: str = PROGRAM, *, timeout_s: float = TIMEOUT_S, max_bytes: int = MAX_BYTES
) -> Clip:
    """Return what is copied: a picture when one is offered, else text, else an empty clip.

    Raises:
        ClipboardError: `wl-paste` is missing or failed, or the copy is too big or too slow.
    """
    code, listing, err = _run([program, "--list-types"], timeout_s, max_bytes)
    if code != 0:
        problem = _failure(err)
        if problem is not None:
            raise ClipboardError(problem)
        return Clip()
    offered = set(listing.decode("utf-8", errors="replace").split())
    kind = next((t for t in (*IMAGE_TYPES, *TEXT_TYPES, URI_TYPE) if t in offered), None)
    if kind is None:
        return Clip()
    code, data, err = _run([program, "--no-newline", "--type", kind], timeout_s, max_bytes)
    if code != 0:
        problem = _failure(err)
        if problem is not None:
            raise ClipboardError(problem)
        return Clip()
    if kind in IMAGE_TYPES:
        return Clip(image=data, mime=kind) if data else Clip()
    return Clip(text=data.decode("utf-8", errors="replace"))
