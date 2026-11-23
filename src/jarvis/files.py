"""Writing a file so that readers see the old content or the new, never a part of either."""

import os
import tempfile
from pathlib import Path


def _fill(fd: int, data: bytes, mode: int) -> None:
    os.fchmod(fd, mode)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_atomic(path: Path, text: str, mode: int = 0o644) -> None:
    """Write `text` to `path` through an exclusive temp file in the same folder and a rename.

    The temp file has a fresh random name, so nothing left at a predictable name (a stale file, a
    link) is ever written through.
    """
    write_atomic_bytes(path, text.encode(), mode)


def write_atomic_bytes(path: Path, data: bytes, mode: int = 0o644) -> None:
    """Write `data` to `path` the way `write_atomic` writes text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        _fill(fd, data, mode)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
