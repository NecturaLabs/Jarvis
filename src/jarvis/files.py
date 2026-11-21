"""Writing a file so that readers see the old content or the new, never a part of either."""

import os
import tempfile
from pathlib import Path


def _fill(fd: int, text: str, mode: int) -> None:
    os.fchmod(fd, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def write_atomic(path: Path, text: str, mode: int = 0o644) -> None:
    """Write `text` to `path` through an exclusive temp file in the same folder and a rename.

    The temp file has a fresh random name, so nothing left at a predictable name (a stale file, a
    link) is ever written through.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        _fill(fd, text, mode)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
