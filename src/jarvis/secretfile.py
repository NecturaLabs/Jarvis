"""Private files (mode 0600, written atomically) for tokens and keys."""

import errno
import json
import os
import secrets
import stat
import tempfile
from pathlib import Path


def _fill(fd: int, data: str) -> None:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_private(path: Path, data: str) -> None:
    """Write `data` to `path` atomically with mode 0600, creating a 0700 parent."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        _fill(fd, data)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


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


def read_api_key(path: Path) -> str | None:
    """Return the key from a private `STRATA_API_KEY=...` environment file, or None."""
    text = read_private(path)
    for line in (text or "").splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == KEY_VARIABLE and value.strip():
            return value.strip()
    return None


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
