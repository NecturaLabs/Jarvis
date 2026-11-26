"""Jarvis's own folder (`files.workspace`): making it, checking it and remembering its files.

The folder is the owner's and Jarvis's meeting place: everything Jarvis makes without a path
goes there, and writing in it never waits on a question for a new file (ADR 0025). That only
holds while the folder is really the owner's alone, so it is made mode 0700 and, when it already
exists, used only if it is a real folder (not a link) owned by this user that no one else can
write to. A folder the owner made is never changed to fit: it is reported, not repaired.

`Ledger` remembers which files Jarvis made there (by device, inode and change time), so it can
tell its own file from one the owner put in the folder.
"""

import contextlib
import json
import os
import stat
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from jarvis.files import write_atomic

if TYPE_CHECKING:
    from jarvis.config import FilesConfig

FOLDER_MODE = 0o700
_OTHERS_WRITE = stat.S_IWGRP | stat.S_IWOTH
MAX_REMEMBERED = 5000


def workspace_path(config: FilesConfig, home: Path) -> Path:
    """Return the workspace as written in the config, with `~` made home but no link followed."""
    text = config.workspace
    return Path(str(home) + text[1:]) if text.startswith("~") else Path(text)


def workspace_problem(path: Path, *, create: bool) -> str:
    """Return why the folder at `path` cannot be the workspace, or "" when it can.

    With `create`, a missing folder is made, mode 0700 (its missing parents get the usual mode).
    A folder that exists is never changed. `path` is as the config writes it: a link at the
    folder's own name is refused, though a link higher up (`~/Documents` on another disk) is not.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return _made(path) if create else ""
    except OSError as exc:
        return f"cannot be read ({exc.strerror or exc})"
    return _judged(info)


def _made(path: Path) -> str:
    """Make the folder, mode 0700 (the umask may only narrow it), and judge what was made."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(mode=FOLDER_MODE, exist_ok=True)
        return _judged(os.lstat(path))
    except OSError as exc:
        return f"cannot be made ({exc.strerror or exc})"


def _judged(info: os.stat_result) -> str:
    """Return why a folder that exists, as `info` describes it, is not the owner's alone."""
    if stat.S_ISLNK(info.st_mode):
        return "is a link; point files.workspace at the folder itself"
    if not stat.S_ISDIR(info.st_mode):
        return "is not a folder"
    if info.st_uid != os.geteuid():
        return "belongs to another user"
    if info.st_mode & _OTHERS_WRITE:
        return "can be written by other users (remove group and other write with chmod go-w)"
    return ""


class Ledger:
    """The files Jarvis made in its workspace: path, device, inode and change time, in one file.

    A file the owner replaced, or edited in any way (a rename over it, a write, a chmod), has
    another inode or change time and so is no longer Jarvis's. A file the owner made under a
    deleted file's name, which the filesystem may give the same inode, differs in change time,
    which no one can set back. Another process may write the file at the same
    moment; a lost entry only means Jarvis asks about that file as it does for the owner's.
    """

    def __init__(self, file: Path) -> None:
        """Keep the list in `file`."""
        super().__init__()
        self._file = file
        self._lock = threading.Lock()

    @staticmethod
    def _identity(info: os.stat_result) -> list[int]:
        return [info.st_dev, info.st_ino, info.st_ctime_ns]

    def _read(self) -> dict[str, list[int]]:
        try:
            loaded = cast("dict[str, Any]", json.loads(self._file.read_text(encoding="utf-8")))
            entries = cast("dict[str, list[int]]", loaded["files"])
            return {
                key: [int(device), int(inode), int(changed)]
                for key, (device, inode, changed) in entries.items()
            }
        except OSError, ValueError, KeyError, TypeError, AttributeError:
            return {}

    def made(self, path: Path, info: os.stat_result) -> bool:
        """Whether the file at `path`, as `info` describes it, is one Jarvis made."""
        with self._lock:
            return self._read().get(str(path)) == self._identity(info)

    def record(self, path: Path, info: os.stat_result) -> None:
        """Note that the file at `path`, as `info` describes it, is Jarvis's."""
        with self._lock:
            entries = self._read()
            entries.pop(str(path), None)
            entries[str(path)] = self._identity(info)
            for old in list(entries)[: max(0, len(entries) - MAX_REMEMBERED)]:
                del entries[old]
            with contextlib.suppress(OSError):  # not remembering only means asking next time
                write_atomic(self._file, json.dumps({"version": 1, "files": entries}), 0o600)
