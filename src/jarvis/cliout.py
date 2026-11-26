"""How the `jarvis` commands write their answers: one look for people, one-line JSON for agents.

A check or a state is a line that starts with its status in one column (`ok`, `FAIL`, `note`),
then what it is about, then for a failure the fix after an arrow. Colour marks the status only,
and only on a terminal: never when the output is piped or read by a program, nor with
`NO_COLOR` set or a dumb terminal. For an agent (`--agent`, or `JARVIS_AGENT_FORMAT=1` in the
environment), a command writes one JSON object on one line instead, always with `command`, `ok`
and `exit`, and the same facts under stable keys.
"""

import contextlib
import json
import os
import re
import sys
import tempfile
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    import io
    from collections.abc import Generator, Sequence

_RESET = "\x1b[0m"
_COLOURS = {"ok": "\x1b[32m", "FAIL": "\x1b[1;31m", "note": "\x1b[36m", "warn": "\x1b[33m"}
_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_NOTE = re.compile(r"^note:?\s+", re.IGNORECASE)
WIDTH = 4  # the status column: "FAIL" and a space
AGENT_ENV = "JARVIS_AGENT_FORMAT"  # set to 1 to have every command answer in the agent format
_YES = frozenset({"1", "true", "yes", "on"})


def agent_mode(*, asked: bool = False) -> bool:
    """Whether to answer in the agent format: `--agent` was given or `AGENT_ENV` says so."""
    return asked or os.environ.get(AGENT_ENV, "").strip().casefold() in _YES


def colour(stream: TextIO | None = None) -> bool:
    """Whether `stream` (stdout by default) takes colour: a terminal, no NO_COLOR, not dumb."""
    out = sys.stdout if stream is None else stream
    if agent_mode() or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    isatty = getattr(out, "isatty", None)
    return bool(isatty and isatty())


def status(word: str, stream: TextIO | None = None) -> str:
    """Return a status word padded to its column, coloured when `stream` takes colour."""
    padded = f"{word:<{WIDTH}}"
    if word in _COLOURS and colour(stream):
        return f"{_COLOURS[word]}{padded}{_RESET}"
    return padded


def check(what: str, fix: str = "", *, ok: bool, stream: TextIO | None = None) -> str:
    """Return a check's line: `ok   what`, or `FAIL what  -> fix`."""
    line = f"{status('ok' if ok else 'FAIL', stream)} {what}"
    return line if ok or not fix else f"{line}  -> {fix}"


def note(text: str, stream: TextIO | None = None) -> str:
    """Return a note's line, `note …`, however `text` began (`note `, `note:` or neither)."""
    return f"{status('note', stream)} {_NOTE.sub('', text, count=1)}"


def strip(text: str) -> str:
    """Return `text` without its colour."""
    return _ESCAPE.sub("", text)


def summary(checks: Sequence[tuple[bool, str, str]], *, extra: int = 0) -> str:
    """Return the closing line: how many checks ran (with `extra` passed ones), and which failed."""
    total = len(checks) + extra
    failed = [what for ok, what, _ in checks if not ok]
    counted = f"{total} check{'' if total == 1 else 's'}"
    if not failed:
        return f"{counted}, none failed"
    return f"{counted}, {len(failed)} failed: {', '.join(failed)}"


def emit(data: dict[str, Any] | list[Any], stream: TextIO | None = None) -> None:
    """Write `data` as one line of JSON, its words as written (not escaped)."""
    out = sys.stdout if stream is None else stream
    out.write(json.dumps(data, ensure_ascii=False) + "\n")


@contextlib.contextmanager
def captured(out: io.StringIO, err: io.StringIO) -> Generator[None]:
    """Collect into `out` and `err` all a command writes, its child programs' output included.

    Python's own streams are swapped, and file descriptors 1 and 2 point at temporary files
    meanwhile, so a `systemctl` or `journalctl` the command runs cannot write past the answer.
    """
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    with (
        tempfile.TemporaryFile("w+", encoding="utf-8") as fd_out,
        tempfile.TemporaryFile("w+", encoding="utf-8") as fd_err,
    ):
        saved = os.dup(1), os.dup(2)
        os.dup2(fd_out.fileno(), 1)
        os.dup2(fd_err.fileno(), 2)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                yield
        finally:
            for fd, old in zip((1, 2), saved, strict=True):
                os.dup2(old, fd)
                os.close(old)
            for file, into in ((fd_out, out), (fd_err, err)):
                file.seek(0)
                into.write(file.read())
