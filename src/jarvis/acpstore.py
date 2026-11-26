"""The folder `jarvis acp` keeps its sessions in (ADR 0023), for the server and the chat window.

The chat's `/forget all` deletes what is kept here, so these helpers sit apart from the server
(`jarvis.acp`) and the window does not load it.
"""

import logging
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

MAX_KEPT = 200  # sessions kept on disk for `session/load`, the newest
SESSION_ID = re.compile(r"jarvis-[0-9a-f]{12}")  # what `session/new` makes, and nothing else
STALE_TEMP_S = 3600.0  # a temp file this old is no writer's: `write_atomic` renames within moments
# What `write_atomic` leaves when a crash stops it: ".<file>.<8 random characters>.tmp".
_TEMP = re.compile(r"\.jarvis-[0-9a-f]{12}\..+\.[a-z0-9_]{8}\.tmp")


def forget_kept(store: Path) -> int:
    """Delete every session kept in `store`; return how many there were (an `OSError` passes)."""
    if not store.is_dir():
        return 0
    gone = 0
    for path in store.glob("jarvis-*"):
        if SESSION_ID.fullmatch(path.name.split(".", 1)[0]):
            gone += path.name.count(".") == 1  # the meta file, one per session
            path.unlink(missing_ok=True)
    _remove_temps(store, None)
    return gone


def prune(store: Path) -> None:
    """Keep only the newest `MAX_KEPT` sessions' files in `store`."""
    try:
        metas = sorted(
            (p for p in store.glob("jarvis-*.json") if SESSION_ID.fullmatch(p.stem)),
            key=lambda p: p.stat().st_mtime,
        )
        for old in metas[: max(0, len(metas) - MAX_KEPT)]:
            for path in store.glob(f"{old.stem}.*"):
                path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("old sessions could not be removed: %s", exc)
    try:
        _remove_temps(store, STALE_TEMP_S)
    except OSError as exc:
        log.warning("stale temp files could not be removed: %s", exc)


def _remove_temps(store: Path, older_than_s: float | None) -> None:
    """Delete the temp files `write_atomic` left in `store`, all or those older than `older_than_s`.

    A writer renames its temp file within moments, so the age keeps a live one safe.
    """
    now = time.time()
    for path in store.glob(".jarvis-*"):
        if not _TEMP.fullmatch(path.name):
            continue
        try:
            if older_than_s is None or now - path.stat().st_mtime >= older_than_s:
                path.unlink(missing_ok=True)
        except FileNotFoundError:  # its writer finished meanwhile
            continue
