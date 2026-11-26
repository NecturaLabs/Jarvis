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
        return _made(path) if create else _parents(path)
    except OSError as exc:
        return f"cannot be read ({exc.strerror or exc})"
    return folder_problem(info) or _parents(path)


def workspace_in_use(config: FilesConfig, home: Path) -> bool:
    """Whether the workspace is a folder Jarvis may use now (a missing one is made on use)."""
    return not workspace_problem(workspace_path(config, home), create=False)


def _parents(path: Path) -> str:
    """Return why a folder above `path` lets someone else move or replace the workspace, or "".

    A folder that group or others may write to (without the sticky bit, which keeps others from
    renaming what is not theirs), or one that belongs to another user who is not root, can have
    the workspace renamed away and another put in its place. A group-writable folder is refused
    even when you are the group's only member (a private group, umask 002): Jarvis cannot tell.
    Missing folders are skipped: they are made by this user. Links are followed first.
    """
    for folder in Path(os.path.realpath(path)).parents:
        try:
            info = os.lstat(folder)
        except OSError:
            continue
        if info.st_uid not in {0, os.geteuid()}:
            return (
                f"has a folder above it, {folder}, that belongs to another user, who could "
                "move the workspace away and put another in its place"
            )
        if info.st_mode & _OTHERS_WRITE and not info.st_mode & stat.S_ISVTX:
            return (
                f"has a folder above it, {folder}, that group or other users can write to, so "
                "they could move the workspace away and put another in its place. Fix it with "
                f"`chmod go-w {folder}` (this holds for a group of only you too, which Jarvis "
                "cannot tell from a shared one)"
            )
    return ""


def _made(path: Path) -> str:
    """Make the folder, mode 0700 (the umask may only narrow it), and judge what was made."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(mode=FOLDER_MODE, exist_ok=True)
        return folder_problem(os.lstat(path)) or _parents(path)
    except OSError as exc:
        return f"cannot be made ({exc.strerror or exc})"


def folder_problem(info: os.stat_result) -> str:
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
    """The files Jarvis made in its workspace: path, device, inode, times and size, in one file.

    A file the owner replaced, or edited in any way (a rename over it, a write, a chmod), has
    another inode, change time, modification time or size and so is no longer Jarvis's. A file
    the owner made under a deleted file's name, which the filesystem may give the same inode,
    differs in change time, which no one can set back. Another process may write the file at the
    same moment; a lost entry only means Jarvis asks about that file as it does for the owner's.
    """

    def __init__(self, file: Path) -> None:
        """Keep the list in `file`."""
        super().__init__()
        self._file = file
        self._lock = threading.Lock()

    @staticmethod
    def _identity(info: os.stat_result) -> list[int]:
        """Return what tells this file from any other, or from itself changed.

        The change time cannot be set back, but a filesystem may keep it coarsely (a second, even
        two), so a rewrite inside that tick would pass it; the modification time and the size
        are kept too, and a file only stays Jarvis's if every one still matches.
        """
        return [info.st_dev, info.st_ino, info.st_ctime_ns, info.st_mtime_ns, info.st_size]

    def _read(self) -> dict[str, list[int]]:
        try:
            loaded = cast("dict[str, Any]", json.loads(self._file.read_text(encoding="utf-8")))
            entries = cast("dict[str, list[int]]", loaded["files"])
            return {key: [int(part) for part in identity] for key, identity in entries.items()}
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
