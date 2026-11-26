"""Private files (mode 0600, written atomically) for tokens and keys."""

import errno
import json
import os
import re
import secrets
import stat
from typing import TYPE_CHECKING

from jarvis.files import write_atomic

if TYPE_CHECKING:
    from pathlib import Path


def write_private(path: Path, data: str) -> None:
    """Write `data` to `path` atomically with mode 0600, creating a 0700 parent."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_atomic(path, data, 0o600)


def read_private(path: Path) -> str | None:
    """Return the file's text, or None when it does not exist.

    The file is opened once and checked through that descriptor, so what is checked is what is
    read.

    Raises:
        PermissionError: the file is a symbolic link, is not a regular file, belongs to another
            user or is readable by group or others.
        OSError: the file could not be opened for another reason.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno != errno.ELOOP:
            raise
        msg = f"{path} must not be a symbolic link"
        raise PermissionError(msg) from exc
    with os.fdopen(fd, "r", encoding="utf-8") as handle:
        status = os.fstat(handle.fileno())
        if not stat.S_ISREG(status.st_mode) or status.st_uid != os.getuid():
            msg = f"{path} must be a regular file of your own"
            raise PermissionError(msg)
        if status.st_mode & 0o077:
            msg = f"{path} must be private (chmod 600)"
            raise PermissionError(msg)
        return handle.read()


def write_private_json(path: Path, value: dict[str, object]) -> None:
    """Write a JSON object privately."""
    write_private(path, json.dumps(value, indent=1, sort_keys=True))


KEY_VARIABLE = "STRATA_API_KEY"
_BEARER_TOKEN = re.compile(r"[-a-zA-Z0-9._~+/]+=*")
_MAX_BEARER_LENGTH = 4096
MAX_KEY_INPUT = 65536  # the most of a key file `jarvis setup --remote` reads


def remote_key(text: str | None) -> str | None:
    """Read one bearer token, supplied raw or as the backend's environment line."""
    value = (text or "").strip().removeprefix(f"{KEY_VARIABLE}=")
    return value if len(value) <= _MAX_BEARER_LENGTH and _BEARER_TOKEN.fullmatch(value) else None


def _key_line(text: str | None) -> str | None:
    """Return the value of the first `STRATA_API_KEY=...` line of an environment file's text.

    A value in one pair of single or double quotes is read without them, as systemd reads the
    same file for Strata, so both send the same key.
    """
    for line in (text or "").splitlines():
        name, sep, value = line.partition("=")
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if sep and name.strip() == KEY_VARIABLE and value:
            return value
    return None


def read_api_key(path: Path) -> str | None:
    """Return the key from a private `STRATA_API_KEY=...` environment file, or None."""
    return _key_line(read_private(path))


def remote_key_input(text: str) -> str | None:
    """Read the key from what `jarvis setup --remote` is given: a whole backend key file or a token.

    The backend keeps its key in an environment file whose `STRATA_API_KEY=` line can sit among
    other lines, so that line is found wherever it is. Without one, a lone token is the key.
    """
    if len(text) > MAX_KEY_INPUT:
        return None
    value = _key_line(text)
    if value is not None:
        return remote_key(value)
    lines = [line for line in text.splitlines() if line.strip()]
    return remote_key(lines[0]) if len(lines) == 1 else None


def ensure_api_key(path: Path) -> str:
    """Return the key in the environment file at `path`, adding a random one if there is none.

    Other lines of the file (the owner's own variables and comments) are kept.
    """
    existing = read_api_key(path)
    if existing:
        return existing
    key = secrets.token_urlsafe(32)
    kept = [
        line
        for line in (read_private(path) or "").splitlines()
        if line.partition("=")[0].strip() != KEY_VARIABLE
    ]
    write_private(path, "\n".join([*kept, f"{KEY_VARIABLE}={key}"]) + "\n")
    return key
