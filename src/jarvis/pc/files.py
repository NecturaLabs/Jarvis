"""The owner's files: listed and read inside the read roots, written inside the write roots.

Every path is resolved through its links before it is checked, so a link inside a root cannot
lead out of it, and a link to a secret is refused as the secret. The checked path is then opened
one name at a time from `/`, following no link, so a folder swapped for a link after the check is
refused rather than followed. A file under home with a second name on disk (a hard link) is never
read or added to, since that name may be a secret's. Some things are never read or
written whatever the roots say: keys, tokens, password stores, browser profiles and Jarvis's own
state (`SECRET_NAMES`, `SECRET_PATHS`, plus the owner's `files.deny`).

What a file holds, and the names in a folder, were written by someone (a download, a program), so
they reach the model as `file` data: they never raise a class, side effects need a yes while they
are in the conversation, and so does sending text off the PC (`policy.decide`), unless the owner
turned questions off. Hidden files and folders under home stay shut unless `files.hidden_allow`
opens them.

Jarvis's workspace (`files.workspace`) is always readable and writable, and a path with no
folder (a bare name, a relative path) is written there. Inside it a hidden name is no bar (only
secrets and version-control folders are), and a call that makes a new file, or writes one Jarvis
made, needs no question (`FileOps.unasked`, ADR 0025).

Writing is `confirm`: it asks first unless the owner turned questions off
(`boundaries.bypass_consent`). The question names the resolved path, the folders the write would
make and, for `replace`, the old file's size; the write runs only if that path still resolves to
itself, needs no folder the question did not name and, for `replace`, finds the size it named. A
name that turned into a link after the question is refused. A write never touches a hidden file
or folder (no `~/.bashrc`, `~/.config/…`), never replaces a file by accident (`create` refuses an
existing file) and never makes a file executable.
"""

import contextlib
import errno
import fnmatch
import glob
import itertools
import os
import re
import secrets
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from jarvis.config import SYSTEM_FOLDER_NAMES, ConfigError, deny_pattern, workspace_refusal
from jarvis.operations import (
    DATA_WRAP_CHARS,
    UNASKED_ARG,
    Operation,
    OperationError,
    Result,
    cut_inert,
    inert_chars,
)
from jarvis.pc.workspace import folder_problem, workspace_path, workspace_problem
from jarvis.policy import OpClass, Origin

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from jarvis.config import FilesConfig
    from jarvis.pc.workspace import Ledger

NEW_FILE = "a new file in the workspace"  # `FileOps.unasked`: the reasons a write needs no yes
MADE_FILE = "a file Jarvis made in the workspace"
MAX_PATH = 1000
MAX_CONTENT = 20000
MAX_ENTRIES = 200
MAX_SCANNED = 10_000  # entries read from one folder; a bigger one is listed from these
MAX_READ_BYTES = 4 * 1024 * 1024
BINARY_PROBE = 8192
KIB = 1024
# A result's header without its path, its closing line and the agent's wrapper around it.
HEADROOM = 200 + DATA_WRAP_CHARS
# What one result always holds, however long its path: a few spelled-out brackets, so reading
# in parts always moves on. Only a path near MAX_PATH at the lowest cap makes a result overshoot.
MIN_ROOM = 32
FILE_MODE = 0o644
FOLDER_MODE = 0o755
_CARRIED_MODE = 0o666  # what a replace keeps of the old file's mode: never execute or special bits
PREVIEW_LINES = 3  # of what a write puts in, shown under the question in the chat
PREVIEW_WIDTH = 160
_FOLDER_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_SWAPPED = frozenset({errno.ELOOP, errno.ENOTDIR})  # a name on the way became a link or a file

# Names never read, listed or written, wherever they are (shell-style, matched case-blind).
SECRET_NAMES = (
    ".ssh",
    ".gnupg",
    ".password-store",
    ".aws",
    ".azure",
    ".kube",
    ".docker",
    ".netrc",
    ".git-credentials",
    ".pgpass",
    ".npmrc",
    ".pypirc",
    ".vault-token",
    ".pki",
    ".mozilla",
    ".thunderbird",
    ".claude",
    ".codex",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.kdbx",
    "*.keystore",
    "*.gpg",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    "*credential*",
    "*secret*",
    "*password*",
    "*.token",
    "*_history",
    ".*history",
    ".histfile",
    "cookies",
    "cookies.sqlite*",
    "login data*",
)
# Folders under the home folder that hold secrets: keyrings, logins and browser profiles.
SECRET_PATHS = (
    ".config/gh",
    ".config/hub",
    ".config/rclone",
    ".config/Bitwarden*",
    ".config/1Password",
    ".config/keepassxc",
    ".config/chromium",
    ".config/google-chrome*",
    ".config/BraveSoftware",
    ".config/vivaldi*",
    ".config/microsoft-edge*",
    ".config/Signal",
    ".config/discord",
    ".local/share/keyrings",
    ".local/share/kwalletd",
    ".cargo/credentials*",
)


SYSTEM_FOLDERS = tuple(Path(name) for name in SYSTEM_FOLDER_NAMES)
# Folders whose contents run programs (hooks): never written, even in the workspace.
_VERSION_CONTROL = frozenset({".git", ".hg", ".svn"})


class FileRefusalError(OperationError):
    """A refused or failed file operation, whose words reach the model as `file` data.

    It names paths found on disk (a link's target, a folder's entries), which anyone can choose.
    """

    def __init__(self, message: str) -> None:
        """Keep the message, marked as file data."""
        super().__init__(message, origin=Origin.FILE)


def _content(line: str) -> str:
    """Return a line without its line ending."""
    return line.splitlines()[0] if line else ""


def _size(count: int) -> str:
    """Return a size the way people say it: 512 B, 1.2 KB, 3.4 MB."""
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < KIB or unit == "GB":
            return f"{count} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= KIB
    return f"{count} B"


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


_WILDCARD = re.compile(r"[*?\[]")


def _anchored(pattern: str, home: Path) -> str:
    """Return a `files.deny` path as an absolute pattern: under home unless written absolute.

    The folders before the first wildcard are resolved through links, as the paths it is matched
    with are, and taken literally.
    """
    written = PurePosixPath(pattern)
    base, parts = (
        (written.anchor, written.parts[1:]) if written.is_absolute() else (home, written.parts)
    )
    fixed = next((i for i, part in enumerate(parts) if _WILDCARD.search(part)), len(parts))
    head = glob.escape(os.path.realpath(PurePosixPath(base, *parts[:fixed])))
    return str(PurePosixPath(head, *parts[fixed:]))


def _starts_with(path: Path, pattern: str) -> bool:
    """Return whether a resolved path is what an absolute pattern names, or inside it."""
    depth = len(PurePosixPath(pattern).parts)
    return len(path.parts) >= depth and fnmatch.fnmatch(str(Path(*path.parts[:depth])), pattern)


def _listed(names: list[str]) -> str:
    """Return names as they are said in a list: "a", "a and b", "a, b and c"."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _printable(line: str) -> str:
    """Return a line with every control character shown as a replacement mark."""
    text = "".join(ch if ch.isprintable() else "\ufffd" for ch in line.replace("\t", " "))
    return text if len(text) <= PREVIEW_WIDTH else text[: PREVIEW_WIDTH - 1] + "\u2026"


def _open_folder(
    folder: Path, *, create: int = 0, vet: tuple[int, Callable[[int], None]] | None = None
) -> int:
    """Open a resolved folder one name at a time from `/`, following no link on the way.

    The last `create` names may be missing; they are made (mode `FOLDER_MODE`) as they are met.
    `vet` is a name's place on the way and a check to run on that folder as opened, which raises
    to stop (the descriptor is then closed).
    What `os.open` and `os.mkdir` raise passes through: `FileNotFoundError` for a missing folder
    that may not be made, `ELOOP` for a link on the way and `ENOTDIR` for a name that is no folder.
    """
    names = folder.parts[1:]
    fd = os.open("/", _FOLDER_FLAGS)
    for index, name in enumerate(names):
        try:
            inner = _open_inner(fd, name, make=index >= len(names) - create)
        finally:
            os.close(fd)
        if vet is not None and vet[0] == index:
            try:
                vet[1](inner)
            except BaseException:
                os.close(inner)
                raise
        fd = inner
    return fd


def _open_inner(fd: int, name: str, *, make: bool) -> int:
    """Open the folder `name` inside the open folder `fd` without following a link.

    With `make`, a missing one is made first.

    Raises:
        FileNotFoundError: it is missing and `make` is off.
    """
    try:
        return os.open(name, _FOLDER_FLAGS, dir_fd=fd)
    except FileNotFoundError:
        if not make:
            raise
    os.mkdir(name, FOLDER_MODE, dir_fd=fd)
    return os.open(name, _FOLDER_FLAGS, dir_fd=fd)


@dataclass(frozen=True, slots=True)
class Boundary:
    """Where Jarvis may read and write, resolved through links once when it is made."""

    home: Path
    read_roots: tuple[Path, ...]
    write_roots: tuple[Path, ...]
    deny: tuple[str, ...] = ()  # the owner's names, matched against every part of a path
    deny_paths: tuple[str, ...] = ()  # the owner's paths, absolute (under home when written so)
    protected: tuple[Path, ...] = ()
    hidden_allow: tuple[str, ...] = ()
    workspace: Path | None = None  # Jarvis's own folder: in both roots, hidden names allowed
    workspace_given: Path | None = None  # ... as the config writes it, links not followed

    @classmethod
    def of(
        cls,
        config: FilesConfig,
        home: Path,
        *,
        protected: Iterable[Path] = (),
        workspace: bool = True,
    ) -> Boundary:
        """Return the boundary a `[files]` section sets; `protected` folders are never touched.

        The workspace is added to both roots unless `workspace` is off (its folder is unsafe).

        Raises:
            ConfigError: the workspace is, holds or lies inside a protected folder.
        """
        home = Path(os.path.realpath(home))
        shut = (*protected, *SYSTEM_FOLDERS)
        refusal = workspace_refusal(config.workspace, home, shut)
        if refusal:
            msg = f"files.workspace {refusal}"
            raise ConfigError(msg)

        def expand(root: str) -> Path:
            text = str(home) + root[1:] if root.startswith("~") else root
            return Path(os.path.realpath(text))

        deny = [deny_pattern(entry) for entry in config.deny]
        folder = expand(config.workspace) if workspace else None
        extra = () if folder is None else (folder,)
        return cls(
            home,
            tuple(dict.fromkeys((*(expand(root) for root in config.read_roots), *extra))),
            tuple(dict.fromkeys((*(expand(root) for root in config.write_roots), *extra))),
            tuple(d for d in deny if "/" not in d),
            tuple(_anchored(d, home) for d in deny if "/" in d),
            tuple(Path(os.path.realpath(path)) for path in shut),
            tuple(config.hidden_allow),
            folder,
            workspace_path(config, home) if workspace else None,
        )

    def workspace_trouble(self) -> str:
        """Return why the workspace is not, now, a folder of the owner's alone, or "".

        A missing one is no trouble (it is made, mode 0700, on the next write).
        """
        given = self.workspace_given
        if given is None:
            return "is not in use"
        problem = workspace_problem(given, create=False)
        if not problem and Path(os.path.realpath(given)) != self.workspace:
            problem = "is not where it was when Jarvis started"
        return problem

    def hidden_in_workspace(self, path: Path) -> bool:
        """Return whether a path in the workspace has a hidden name (a dot) below the folder."""
        if self.workspace is None or not self.in_workspace(path):
            return False
        return any(part.startswith(".") for part in path.relative_to(self.workspace).parts)

    def in_workspace(self, path: Path) -> bool:
        """Return whether a resolved path is inside the workspace (the folder itself is not)."""
        return (
            self.workspace is not None
            and path.is_relative_to(self.workspace)
            and path != self.workspace
        )

    def show(self, path: Path) -> str:
        """Return a path as the owner says it: under home as `~/…`."""
        if path == self.home:
            return "~"
        return f"~/{path.relative_to(self.home)}" if path.is_relative_to(self.home) else str(path)

    def _resolve(self, raw: str, *, write: bool = False) -> Path:
        """Return what a path the model gave means, through every link that exists.

        A path with no folder (`notes.md`, `drafts/a.md`) is in the workspace. To read, it is
        there only if something of that name is, else under home, as before the workspace.

        Raises:
            FileRefusalError: the path is empty or names another user's home.
        """
        text = raw.strip()
        if not text or "\x00" in text:
            msg = "That is not a path."
            raise FileRefusalError(msg)
        if text == "~" or text.startswith("~/"):
            text = str(self.home) + text[1:]
        elif text.startswith("~"):
            msg = f"{text} is outside the folders I may use."
            raise FileRefusalError(msg)
        elif not text.startswith("/"):
            text = str(self._relative(text, write=write))
        return Path(os.path.realpath(text))

    def _relative(self, text: str, *, write: bool) -> Path:
        """Return where a relative path is: the workspace for a write, or a read that finds it."""
        if self.workspace is None:
            return self.home / text
        inside = self.workspace / text
        return inside if write or os.path.lexists(inside) else self.home / text

    def _base(self, path: Path) -> Path | None:
        """Return the folder a path is judged from: home, else the deepest root holding it."""
        if path.is_relative_to(self.home):
            return self.home
        holding = [r for r in (*self.read_roots, *self.write_roots) if path.is_relative_to(r)]
        return max(holding, key=lambda root: len(root.parts), default=None)

    def secret(self, path: Path) -> bool:
        """Return whether a resolved path is, or is inside, something never read or written."""
        if _inside(path, self.protected):
            return True
        base = self._base(path)
        below = path.relative_to(base).parts if base is not None else path.parts
        names = [part.casefold() for part in below]
        for pattern in (*SECRET_NAMES, *self.deny):
            if any(fnmatch.fnmatchcase(name, pattern.casefold()) for name in names):
                return True
        if any(_starts_with(path, pattern) for pattern in self.deny_paths):
            return True
        if base != self.home:
            return False
        for pattern in SECRET_PATHS:
            depth = len(Path(pattern).parts)
            if len(below) >= depth and fnmatch.fnmatch("/".join(below[:depth]), pattern):
                return True
        return False

    def hidden_closed(self, path: Path, *, folder: bool = False) -> bool:
        """Return whether a path is hidden under home and `files.hidden_allow` keeps it shut.

        A `folder` on the way to an opened one (`~/.config` for `.config/hypr`) may be listed;
        its entries are judged one by one.
        """
        if self._base(path) != self.home:
            return False
        below = path.relative_to(self.home).parts
        if not below or not below[0].startswith("."):
            return False
        for allowed in self.hidden_allow:
            pattern = allowed.strip().rstrip("/")
            parts = Path(pattern).parts
            if len(below) >= len(parts):
                if fnmatch.fnmatch("/".join(below[: len(parts)]), pattern):
                    return False
            elif folder and fnmatch.fnmatch("/".join(below), "/".join(parts[: len(below)])):
                return False
        return True

    def readable(self, raw: str, *, folder: bool = False) -> Path:
        """Return the resolved path to read, or to list when `folder`.

        Raises:
            FileRefusalError: it is outside the read roots, private, or hidden and not opened.
        """
        path = self._resolve(raw)
        if not _inside(path, self.read_roots):
            msg = f"{self.show(path)} is outside the folders I may read."
            raise FileRefusalError(msg)
        if self.secret(path):
            msg = f"{self.show(path)} is private (keys, tokens or logins), so I don't read it."
            raise FileRefusalError(msg)
        if self.hidden_closed(path, folder=folder):
            msg = (
                f"{self.show(path)} is hidden, and hidden files under home stay shut unless "
                "files.hidden_allow opens them, so I don't read it."
            )
            raise FileRefusalError(msg)
        return path

    def writable(self, raw: str) -> Path:
        """Return the resolved path to write.

        Raises:
            FileRefusalError: it is outside the write roots, hidden or private.
        """
        path = self._resolve(raw, write=True)
        if path == self.workspace:
            msg = f"{self.show(path)} is my workspace folder, not a file."
            raise FileRefusalError(msg)
        roots = [root for root in self.write_roots if path.is_relative_to(root) and path != root]
        if not roots:
            msg = f"{self.show(path)} is outside the folders I may write to."
            raise FileRefusalError(msg)
        root = max(roots, key=lambda root: len(root.parts))
        below = path.relative_to(root).parts
        if root == self.workspace:
            if _VERSION_CONTROL.intersection(part.casefold() for part in below):
                msg = f"{self.show(path)} is inside a version-control folder, so I don't write it."
                raise FileRefusalError(msg)
        elif any(part.startswith(".") for part in below):
            msg = f"{self.show(path)} is a hidden file or folder, so I don't write it."
            raise FileRefusalError(msg)
        if self.secret(path):
            msg = f"{self.show(path)} looks private (keys, tokens or logins), so I don't write it."
            raise FileRefusalError(msg)
        return path


def _open_regular(path: Path, shown: str, *, single: bool) -> tuple[int, os.stat_result]:
    """Open a file for reading without following any link or blocking on a pipe.

    With `single`, a file with more than one name on disk (a hard link) is refused.

    Raises:
        FileRefusalError: it is missing, a folder, not a regular file, reached through a link,
            or hard-linked when `single`.
    """
    if not path.name:
        msg = f"{shown} is a folder; list it instead."
        raise FileRefusalError(msg)
    try:
        folder = _open_folder(path.parent)
    except FileNotFoundError as exc:
        msg = f"There's no file at {shown}."
        raise FileRefusalError(msg) from exc
    except OSError as exc:
        msg = (
            f"The way to {shown} changed (a folder became a link), so I didn't read it."
            if exc.errno in _SWAPPED
            else f"I couldn't open {shown}: {exc.strerror}."
        )
        raise FileRefusalError(msg) from exc
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        fd = os.open(path.name, flags, dir_fd=folder)
    except FileNotFoundError as exc:
        msg = f"There's no file at {shown}."
        raise FileRefusalError(msg) from exc
    except IsADirectoryError as exc:
        msg = f"{shown} is a folder; list it instead."
        raise FileRefusalError(msg) from exc
    except OSError as exc:
        msg = f"I couldn't open {shown}: {exc.strerror}."
        raise FileRefusalError(msg) from exc
    finally:
        os.close(folder)
    info = os.fstat(fd)
    if stat.S_ISDIR(info.st_mode):
        os.close(fd)
        msg = f"{shown} is a folder; list it instead."
        raise FileRefusalError(msg)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        msg = f"{shown} is not a regular file, so I don't read it."
        raise FileRefusalError(msg)
    if single and info.st_nlink > 1:
        os.close(fd)
        msg = f"{shown} has another name on disk (a hard link), so I don't read it."
        raise FileRefusalError(msg)
    return fd, info


def _when(timestamp: float) -> str:
    return f"{datetime.fromtimestamp(timestamp).astimezone():%Y-%m-%d %H:%M}"


class FileOps:
    """`list_dir`, `read_file` and `write_file` within a `Boundary`.

    `max_chars` is the room for one result, so a long file is read in parts the model can ask for.
    `part_chars` is the most text one write carries, so a long file is written in parts too: one
    answer of the model holds the call, and a whole file in one would be cut off at its cap.
    `made` remembers the files Jarvis made in its workspace; without it only a new file there
    is written unasked, never a later write to it.
    """

    def __init__(
        self,
        boundary: Boundary,
        *,
        max_chars: int,
        part_chars: int = MAX_CONTENT,
        made: Ledger | None = None,
    ) -> None:
        """Work within `boundary`: at most `max_chars` characters a result, `part_chars` a write."""
        super().__init__()
        self._boundary = boundary
        self._max_chars = max_chars
        self._part_chars = max(1, min(part_chars, MAX_CONTENT))
        self._made = made

    # --- reading ------------------------------------------------------------------------------

    def _room(self, shown: str) -> int:
        """Return how much of a file or listing one result holds, measured as the agent sends it."""
        return max(MIN_ROOM, self._max_chars - HEADROOM - inert_chars(shown))

    def read_file(self, args: dict[str, Any]) -> Result:
        """Return a text file's lines from `from_line` on, as much as fits one result.

        A line longer than a whole result is given in pieces: `from_char` reads on inside it.

        Raises:
            FileRefusalError: the path is not allowed, not a readable regular file, or not text.
        """
        path = self._boundary.readable(str(args["path"]))
        shown = self._boundary.show(path)
        fd, info = _open_regular(path, shown, single=path.is_relative_to(self._boundary.home))
        with os.fdopen(fd, "rb") as handle:
            data = handle.read(MAX_READ_BYTES)
        if b"\x00" in data[:BINARY_PROBE]:
            msg = f"{shown} is not text ({_size(info.st_size)}), so I can't show it."
            raise FileRefusalError(msg)
        lines = data.decode("utf-8", errors="replace").splitlines(keepends=True)
        span, body, more = self._part(
            lines,
            max(1, int(args.get("from_line", 1))),
            max(1, int(args.get("from_char", 1))),
            shown,
        )
        head = f"File {shown} ({_size(info.st_size)}, changed {_when(info.st_mtime)}, {span}):"
        tail = ""
        if more:
            tail = f"\n(More follows: read_file with {more}.)"
        elif len(data) >= MAX_READ_BYTES and info.st_size > MAX_READ_BYTES:
            tail = f"\n(Only the first {_size(MAX_READ_BYTES)} can be read.)"
        return Result(f"{head}\n{body.rstrip(chr(10))}{tail}", Origin.FILE)

    def _part(self, lines: list[str], first: int, column: int, shown: str) -> tuple[str, str, str]:
        """Return the span read, the text and where the next part starts ("" at the end).

        Raises:
            FileRefusalError: `first` or `column` is past the end of the file or line.
        """
        total = len(lines)
        if total and first > total:
            msg = f"{shown} has only {total} lines."
            raise FileRefusalError(msg)
        pending = lines[first - 1 :]
        length = len(_content(pending[0])) if pending else 0
        if column > 1:
            if column > length:
                msg = f"Line {first} of {shown} has only {length} characters."
                raise FileRefusalError(msg)
            pending[0] = pending[0][column - 1 :]
        room = self._room(shown)
        if pending and inert_chars(_content(pending[0])) > room:
            # A line longer than a whole result: this piece of it, and where the next one starts.
            piece = cut_inert(pending[0], room)
            end = column - 1 + len(piece)
            span = f"line {first} of {total}, characters {column} to {end} of {length}"
            return span, piece, f"from_line={first}, from_char={end + 1}"
        body, used, last = "", 0, first - 1
        for line in pending:
            if body and used + inert_chars(line) > room:
                break
            body += line
            used += inert_chars(line)
            last += 1
        span = f"lines {first} to {last} of {total}" if total else "empty"
        if column > 1:
            span = f"lines {first} (from character {column}) to {last} of {total}"
        return span, body, f"from_line={last + 1}" if last < total else ""

    def list_dir(self, args: dict[str, Any]) -> Result:
        """List a folder: folders first, then files, with sizes and when they changed.

        Raises:
            FileRefusalError: the path is not allowed or not a folder.
        """
        path = self._boundary.readable(str(args["path"]), folder=True)
        shown = self._boundary.show(path)
        if not path.is_dir():
            msg = f"{shown} is not a folder." if path.exists() else f"There's no folder {shown}."
            raise FileRefusalError(msg)
        hidden = args.get("hidden") is True
        pattern = str(args.get("pattern", "")).casefold()
        try:
            folder = _open_folder(path)
        except OSError as exc:
            msg = (
                f"The way to {shown} changed (a folder became a link), so I didn't list it."
                if exc.errno in _SWAPPED or isinstance(exc, FileNotFoundError)
                else f"I couldn't list {shown}: {exc.strerror}."
            )
            raise FileRefusalError(msg) from exc
        try:  # an entry reads its details through the folder's descriptor, so it stays open
            rows, more = self._rows(folder, path, shown, hidden=hidden, pattern=pattern)
        finally:
            os.close(folder)
        rows.sort(key=lambda row: (not row[0], row[1].casefold()))
        lines: list[str] = []
        used, room = 0, max(MIN_ROOM, self._room(shown) - inert_chars(pattern))
        for _folder, _name, line in rows[:MAX_ENTRIES]:
            if lines and used + inert_chars(line) + 1 > room:  # the first always shows
                break
            lines.append(line)
            used += inert_chars(line) + 1
        what = (
            f"more than {len(rows)} items"
            if more
            else f"{len(rows)} item{'' if len(rows) == 1 else 's'}"
        )
        text = f"Folder {shown} ({what}{', matching ' + pattern if pattern else ''}):"
        text += "\n" + "\n".join(lines) if lines else "\n(nothing to show)"
        if len(lines) < len(rows):
            text += f"\n… and {len(rows) - len(lines)} more."
        return Result(text, Origin.FILE)

    def _rows(
        self, folder: int, path: Path, shown: str, *, hidden: bool, pattern: str
    ) -> tuple[list[tuple[bool, str, str]], bool]:
        """Return the rows of an open folder that may be shown, and whether it holds more.

        Raises:
            FileRefusalError: the folder could not be read.
        """
        try:
            with os.scandir(folder) as found:
                entries = list(itertools.islice(found, MAX_SCANNED + 1))
        except OSError as exc:
            msg = f"I couldn't list {shown}: {exc.strerror}."
            raise FileRefusalError(msg) from exc
        rows: list[tuple[bool, str, str]] = []
        for entry in entries[:MAX_SCANNED]:
            if (entry.name.startswith(".") and not hidden) or (
                pattern and not fnmatch.fnmatchcase(entry.name.casefold(), pattern)
            ):
                continue
            real = Path(os.path.realpath(path / entry.name))
            if self._boundary.secret(real) or self._boundary.hidden_closed(
                real, folder=entry.is_dir()
            ):
                continue
            rows.append(self._row(entry))
        return rows, len(entries) > MAX_SCANNED

    @staticmethod
    def _row(entry: os.DirEntry[str]) -> tuple[bool, str, str]:
        try:
            folder = entry.is_dir()
            info = entry.stat(follow_symlinks=False)
        except OSError:
            return False, entry.name, f"- {entry.name} (unreadable)"
        link = " (link)" if entry.is_symlink() else ""
        if folder:
            return True, entry.name, f"- {entry.name}/{link}"
        if link:
            return False, entry.name, f"- {entry.name}{link}"
        size, when = _size(info.st_size), _when(info.st_mtime)
        return False, entry.name, f"- {entry.name}{link}  {size}  {when}"

    # --- writing ------------------------------------------------------------------------------

    def bind_write(self, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve the path before the question, so the yes is for that exact file.

        The bound arguments also say how many folders the write would make (`new_folders`) and,
        for `replace`, the old file's size (`old_size`, None when there is none) and which file it
        is (`old_file`: its inode and its change and modification times in nanoseconds), so the
        question names them and the write refuses when they no longer hold.

        Raises:
            FileRefusalError: the path may not be written, `create` meets an existing file, or
                the file is read-only for this user (a replace would get past that, as it
                renames a new file over the old one).
        """
        path = self._boundary.writable(str(args["path"]))
        mode = str(args.get("mode", "create"))
        shown = self._boundary.show(path)
        if path.exists() and not path.is_file():
            msg = f"{shown} is not a regular file, so I don't write it."
            raise FileRefusalError(msg)
        if mode == "create" and path.exists():
            msg = f"{shown} already exists; say whether to replace it or add to it."
            raise FileRefusalError(msg)
        if path.exists() and not _writable(path.stat(follow_symlinks=False)):
            msg = f"{shown} is read-only, so I don't change it."
            raise FileRefusalError(msg)
        new_folders = 0
        while not path.parents[new_folders].exists():
            new_folders += 1
        if args.get(UNASKED_ARG) == NEW_FILE and os.path.lexists(path):
            msg = f"{shown} appeared just now, so I didn't write it without asking."
            raise FileRefusalError(msg)
        bound = {**args, "path": str(path), "mode": mode, "new_folders": new_folders}
        if mode in {"replace", "edit"}:
            _bind_old(bound, path, shown)
        return bound

    def unasked(self, args: dict[str, Any]) -> str:
        """Return why a write needs no question, or "" when it asks as usual (ADR 0025).

        In the workspace a write that makes a new file, or changes one Jarvis made there, is the
        owner's own request done in Jarvis's own folder: nothing of theirs is lost. Replacing or
        adding to a file the owner put there, or one Jarvis made that was swapped, changed or
        given a second name since, asks. So does a hidden name (`.envrc`, `.vscode`: files that
        act on their own) and any write while the workspace is no longer the owner's alone.
        The policy still asks after outside text, whatever this says; the write itself holds to
        what is said here (`bind_write`, `write_file`).
        """
        try:
            path = self._boundary.writable(str(args["path"]))
        except FileRefusalError, KeyError:
            return ""
        if (
            not self._boundary.in_workspace(path)
            or self._boundary.hidden_in_workspace(path)
            or self._boundary.workspace_trouble()
        ):
            return ""
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            return NEW_FILE
        except OSError:
            return ""
        return MADE_FILE if self._ours(path, info) else ""

    def _ours(self, path: Path, info: os.stat_result) -> bool:
        """Whether `info` describes a regular file with one name that Jarvis made at `path`."""
        return (
            self._made is not None
            and stat.S_ISREG(info.st_mode)
            and info.st_nlink == 1
            and self._made.made(path, info)
        )

    def _remembered(self, path: Path, folder: int, claim: object) -> bool:
        """Return whether the file at `path`, before a write, is new or one Jarvis made.

        A write that was let through unasked (`claim`) must still find what was judged.

        Raises:
            FileRefusalError: the write was bound as to a file Jarvis made, and that file is not
                the one now there (swapped, changed or given a second name since).
        """
        if not self._boundary.in_workspace(path) or self._made is None:
            return False
        try:
            info = os.stat(path.name, dir_fd=folder, follow_symlinks=False)
        except FileNotFoundError:
            ours = True
        except OSError:
            ours = False
        else:
            ours = self._ours(path, info)
        if claim == MADE_FILE and not ours:
            msg = f"{self._boundary.show(path)} changed after you were asked, so I didn't write it."
            raise FileRefusalError(msg)
        return ours

    def _remember(self, path: Path, folder: int) -> None:
        """Note that the file just written at `path` is Jarvis's."""
        if self._made is None:
            return
        with contextlib.suppress(OSError):
            self._made.record(path, os.stat(path.name, dir_fd=folder, follow_symlinks=False))

    def _ready_workspace(self, path: Path, shown: str) -> tuple[int, Callable[[int], None]] | None:
        """Before a write into the workspace: judge the folder again, and make it if it is gone.

        Returns the check to run on the workspace folder as `_open_folder` opens it, so the
        folder written through is the one judged here, or None for a path outside it.

        Raises:
            FileRefusalError: the workspace is no longer a folder of the owner's alone, or cannot
                be made.
        """
        boundary, given = self._boundary, self._boundary.workspace_given
        if not boundary.in_workspace(path) or given is None or boundary.workspace is None:
            return None
        problem = boundary.workspace_trouble() or workspace_problem(given, create=True)
        judged = (0, 0)
        if not problem:
            try:
                info = os.lstat(given)
            except OSError as exc:
                problem = f"cannot be read ({exc.strerror or exc})"
            else:
                judged = (info.st_dev, info.st_ino)
        if problem:
            msg = f"My workspace {problem}, so I didn't write {shown}."
            raise FileRefusalError(msg)

        def vet(fd: int) -> None:
            """Judge the workspace folder as opened: the one judged just now, still the owner's.

            Raises:
                FileRefusalError: it is another folder than the one judged, or no longer safe.
            """
            info = os.fstat(fd)
            reason = (
                "is not the folder it was a moment ago"
                if (info.st_dev, info.st_ino) != judged
                else folder_problem(info)
            )
            if reason:
                msg = f"My workspace {reason}, so I didn't write {shown}."
                raise FileRefusalError(msg)

        return len(boundary.workspace.parts) - 2, vet

    def _verifier(self, path: Path, changed: str) -> Callable[[os.stat_result], None]:
        """Return the check that a file opened for a "made by Jarvis" write is still that file."""

        def verify(info: os.stat_result) -> None:
            if not self._ours(path, info):
                raise FileRefusalError(changed)

        return verify

    def write_summary(self, args: dict[str, Any]) -> str:
        """Return the confirmation question's words: how, which file, how much, what is made."""
        path = Path(str(args.get("path", "")))
        folder = self._boundary.show(path.parent).removeprefix("~/") if path.is_absolute() else ""
        where = f"{path.name} in {'your home folder' if folder == '~' else folder}"
        count = len(str(args.get("content", "")))
        if str(args.get("mode")) == "edit":
            found = len(str(args.get("find", "")))
            target = where if folder else str(path)
            plural = "" if count == 1 else "s"
            return f"change {target}: {found} characters there become {count} character{plural}"
        verb = {"replace": "replace", "append": "add to"}.get(str(args.get("mode")), "create")
        what = "the file " if verb == "create" else ""
        plural = "" if count == 1 else "s"
        target = where if folder else str(path)
        old = args.get("old_size")
        if verb == "replace" and isinstance(old, int):
            text = f"replace {target} ({_size(old)} now) with {count} character{plural}"
        else:
            text = f"{verb} {what}{target} ({count} character{plural})"
        new = int(args.get("new_folders") or 0) if path.is_absolute() else 0
        if new:
            made = [self._boundary.show(p).removeprefix("~/") for p in path.parents[:new]]
            noun = "folder" if new == 1 else "folders"
            text += f", making the new {noun} {_listed(made[::-1])}"
        return text

    @staticmethod
    def write_preview(args: dict[str, Any]) -> str:
        """Return the first lines of what a write puts in, for a window to show under the question.

        Control characters are shown as marks, so the text cannot steer the terminal.
        """
        lines = str(args.get("content", "")).splitlines()
        shown = [_printable(line) for line in lines[:PREVIEW_LINES]]
        rest = len(lines) - PREVIEW_LINES
        if rest > 0:
            shown.append(f"(and {rest} more line{'' if rest == 1 else 's'})")
        return "\n".join(shown)

    def write_file(self, args: dict[str, Any]) -> Result:
        """Write the confirmed file, if what was asked about still holds.

        Raises:
            FileRefusalError: the path, its folders or the old file changed since the question,
                the file is hard-linked under home (for `append`), or the write failed.
        """
        path = Path(str(args["path"]))
        again = self._boundary.writable(str(path))
        shown = self._boundary.show(path)
        changed = f"{shown} changed after you were asked, so I didn't write it."
        if again != path:
            raise FileRefusalError(changed)
        vet = self._ready_workspace(path, shown)
        claim = args.get(UNASKED_ARG)
        mode, content = _how_written(args, changed)
        try:
            folder = _open_folder(path.parent, create=int(args.get("new_folders") or 0), vet=vet)
        except FileRefusalError:
            raise
        except OSError as exc:
            if exc.errno in _SWAPPED or isinstance(exc, FileNotFoundError):
                raise FileRefusalError(changed) from exc
            msg = f"I couldn't write {shown}: {exc.strerror or exc}."
            raise FileRefusalError(msg) from exc
        target = _Target(
            folder,
            path.name,
            shown,
            single=path.is_relative_to(self._boundary.home),
            old=_named(args.get("old_size"), args.get("old_file")),
            verify=self._verifier(path, changed) if claim == MADE_FILE else None,
        )
        try:
            ours = self._remembered(path, folder, claim)
            content = _in_full(target, path, mode, content, args)
            _write(target, content.encode("utf-8"), "replace" if mode == "edit" else mode)
            if ours:
                self._remember(path, folder)
        except FileExistsError as exc:
            msg = f"{shown} already exists; say whether to replace it or add to it."
            raise FileRefusalError(msg) from exc
        except FileRefusalError:
            raise
        except OSError as exc:
            msg = f"I couldn't write {shown}: {exc.strerror or exc}."
            raise FileRefusalError(msg) from exc
        finally:
            os.close(folder)
        sentence = _written(mode, shown, content, args)
        # The full path and its link for the model, which otherwise guesses what "~" is.
        where = f" Its full path is {path}; as a link: {path.as_uri()}"
        return Result(sentence + where, spoken=sentence)

    # --- the operations -----------------------------------------------------------------------

    def operations(self) -> list[Operation]:
        """Return `list_dir`, `read_file` and `write_file`."""
        part = self._part_chars
        path = {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_PATH,
            "description": "Such as ~/Documents/notes.md",
        }
        return [
            Operation(
                "list_dir",
                "List a folder on the owner's PC, with sizes and dates.",
                {
                    "type": "object",
                    "properties": {
                        "path": path,
                        "pattern": {
                            "type": "string",
                            "maxLength": 100,
                            "description": "Only names like this, such as *.md",
                        },
                        "hidden": {
                            "type": "boolean",
                            "default": False,
                            "description": "Also list hidden (dot) files",
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                self.list_dir,
                side_effect=False,
                spoken_summary=lambda a: f"list {a.get('path', 'that folder')}",
            ),
            Operation(
                "read_file",
                "Read a text file on the owner's PC, in parts if long.",
                {
                    "type": "object",
                    "properties": {
                        "path": path,
                        "from_line": {"type": "integer", "minimum": 1, "maximum": 10_000_000},
                        "from_char": {"type": "integer", "minimum": 1, "maximum": MAX_READ_BYTES},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                self.read_file,
                side_effect=False,
                spoken_summary=lambda a: f"read {a.get('path', 'that file')}",
            ),
            Operation(
                "write_file",
                "Create, replace, add to or edit a file (mode; edit: find, found exactly once, "
                "becomes content).",
                {
                    "type": "object",
                    "properties": {
                        "path": path,
                        "content": {"type": "string", "maxLength": part},
                        "mode": {
                            "type": "string",
                            "enum": ["create", "replace", "append", "edit"],
                        },
                        "find": {"type": "string", "maxLength": part},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                self.write_file,
                default_class=OpClass.CONFIRM,
                spoken_summary=self.write_summary,
                prepare=self.bind_write,
                preview=self.write_preview,
                hashed=frozenset({"content", "find"}),
                verbatim=frozenset({"content", "find"}),
                unasked=self.unasked,
            ),
        ]


@dataclass(frozen=True, slots=True)
class _Target:
    """The file a confirmed write goes to: its folder's descriptor and what was asked about."""

    folder: int
    name: str
    shown: str
    single: bool  # refuse a file with a second name on disk (under home)
    old: tuple[int, ...] | None  # for `replace`: the file the question named (`_state`), or none
    verify: Callable[[os.stat_result], None] | None = None  # for the file as opened (a made one)


def _identity(info: os.stat_result) -> list[int]:
    """Return what tells one file apart from another of the same size, or from itself rewritten.

    A file swapped in by a rename has another inode; one rewritten in place has a new change
    time, which, unlike the modification time, no one can set back.
    """
    return [info.st_ino, info.st_mtime_ns, info.st_ctime_ns]


def _writable(info: os.stat_result) -> bool:
    """Whether the file's write bit for this user is set, read from its mode, root or not."""
    if info.st_uid == os.geteuid():
        return bool(info.st_mode & stat.S_IWUSR)
    if info.st_gid in {os.getegid(), *os.getgroups()}:
        return bool(info.st_mode & stat.S_IWGRP)
    return bool(info.st_mode & stat.S_IWOTH)


def _state(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_size, *_identity(info))


def _named(size: object, identity: object) -> tuple[int, ...] | None:
    """Return the bound old file as `_state` gives it, None when the question named none."""
    if not isinstance(size, int) or not isinstance(identity, list):
        return None
    return (size, *(int(part) for part in cast("list[int]", identity)))


def _write(target: _Target, data: bytes, mode: str) -> None:
    """Write `data` into the folder's own descriptor, never through a link at the last name.

    `create` makes a new file only; `append` adds to the end; `replace` writes a fresh file
    beside it, with the old file's read and write permissions, and renames it over, so a reader
    sees the old content or the new.

    Raises:
        FileRefusalError: the old file is not the one the question named, or (for `append`) it
            is not a regular file or is hard-linked when `target.single`.
    """
    folder, name = target.folder, target.name
    if mode == "replace":
        old = _check_replaced(target)
        if old is not None and target.verify is not None:
            target.verify(old)
        carried = None if old is None else stat.S_IMODE(old.st_mode) & _CARRIED_MODE
        temp = f".{name}.{secrets.token_hex(6)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(temp, flags, FILE_MODE if carried is None else 0o600, dir_fd=folder)
        try:
            _fill(fd, data, mode=carried)
            os.replace(temp, name, src_dir_fd=folder, dst_dir_fd=folder)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temp, dir_fd=folder)
            raise
        return
    extra = os.O_EXCL if mode == "create" else os.O_APPEND
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC | extra
    fd = os.open(name, flags, FILE_MODE, dir_fd=folder)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or (target.single and info.st_nlink > 1):
        os.close(fd)
        msg = f"{target.shown} has another name on disk (a hard link), so I don't write it."
        if not stat.S_ISREG(info.st_mode):
            msg = f"{target.shown} is not a regular file, so I don't write it."
        raise FileRefusalError(msg)
    if mode == "append" and target.verify is not None:
        try:  # the file as opened, not as looked at a moment ago, is what was judged
            target.verify(info)
        except FileRefusalError:
            os.close(fd)
            raise
    _fill(fd, data)


def _how_written(args: dict[str, Any], changed: str) -> tuple[str, str]:
    """Return how a confirmed write writes (its mode) and the text it puts in.

    A write let through as "a new file" makes the file exclusively, whatever its mode: one that
    appeared since is the owner's, and was never asked about. A whole file ends in a line break;
    an edit's text is put in as it is.

    Raises:
        FileRefusalError: an edit was judged as making a new file (there was none to change).
    """
    claim = args.get(UNASKED_ARG)
    asked = str(args.get("mode", "create"))
    if asked == "edit" and claim == NEW_FILE:
        raise FileRefusalError(changed)
    content = str(args["content"])
    if asked != "edit" and not content.endswith("\n"):
        content += "\n"
    return ("create" if claim == NEW_FILE else asked), content


def _in_full(target: _Target, path: Path, mode: str, content: str, args: dict[str, Any]) -> str:
    """Return the whole new text of the file: `content`, or for an edit the file with it in.

    An edit reads the file through the folder as opened, and only the file the question named;
    one that changed, or no longer holds the passage once, is refused (`FileRefusalError`).
    """
    if mode != "edit":
        return content
    return _edited(_read_text(path, target.shown, target.folder, target.old), args, target.shown)


def _written(mode: str, shown: str, content: str, args: dict[str, Any]) -> str:
    """Return the sentence that says what a write did."""
    if mode == "edit":
        found, put = len(str(args["find"])), len(str(args["content"]))
        return f"Changed {shown}: {found} characters became {put}."
    done = {"replace": "Replaced", "append": "Added to"}.get(mode, "Created")
    return f"{done} {shown} ({len(content)} characters)."


def _bind_old(bound: dict[str, Any], path: Path, shown: str) -> None:
    """Bind the file a `replace` or `edit` changes: its size and identity, None when it is absent.

    Raises:
        FileRefusalError: an `edit` meets no file, or a passage missing or not unique (said now,
            before any question).
    """
    try:
        old = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        bound["old_size"] = bound["old_file"] = None
    else:
        bound["old_size"], bound["old_file"] = old.st_size, _identity(old)
    if bound["mode"] != "edit":
        return
    if bound["old_size"] is None:
        msg = f"There is no {shown} to change; create it instead."
        raise FileRefusalError(msg)
    _edited(_read_text(path, shown), bound, shown)


def _read_text(
    path: Path, shown: str, folder: int | None = None, named: tuple[int, ...] | None = None
) -> str:
    """Return the text of the file an `edit` changes, never through a link at its last name.

    With `folder` it is read by name from that open folder, and must be the file `named` (its
    `_state`), the one the question was about.

    Raises:
        FileRefusalError: it is not a regular file, not that file, too long or not UTF-8 text.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        fd = (
            os.open(path.name, flags, dir_fd=folder) if folder is not None else os.open(path, flags)
        )
    except OSError as exc:
        msg = f"I couldn't read {shown}: {exc.strerror or exc}."
        raise FileRefusalError(msg) from exc
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or (folder is not None and _state(info) != named):
            msg = f"{shown} changed after you were asked, so I didn't write it."
            raise FileRefusalError(msg)
        if info.st_size > MAX_READ_BYTES:
            msg = f"{shown} is too long to edit in place; replace it instead."
            raise FileRefusalError(msg)
        data = handle.read(MAX_READ_BYTES + 1)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"{shown} is not UTF-8 text, so I don't edit it."
        raise FileRefusalError(msg) from exc


def _edited(text: str, args: dict[str, Any], shown: str) -> str:
    """Return `text` with the one passage `find` names changed to `content`.

    Raises:
        FileRefusalError: `find` is missing, or not found exactly once.
    """
    find, new = str(args.get("find", "")), str(args.get("content", ""))
    if not find:
        msg = "An edit needs find: the exact text to change."
        raise FileRefusalError(msg)
    count = text.count(find)
    if count != 1:
        msg = (
            f"{shown} does not contain that text; read it again and copy the passage exactly."
            if count == 0
            else f"{shown} contains that text {count} times; give more of the lines around it."
        )
        raise FileRefusalError(msg)
    return text.replace(find, new, 1)


def _check_replaced(target: _Target) -> os.stat_result | None:
    """Return the old file of a `replace`, refused unless it is the one the question named.

    Raises:
        FileRefusalError: there is a file where the question named none, none where it named
            one, another file or this one changed (size, inode, times), or something other than
            a regular file.
    """
    try:
        info = os.stat(target.name, dir_fd=target.folder, follow_symlinks=False)
    except FileNotFoundError:
        info = None
    found = None if info is None else _state(info)
    if (info is not None and not stat.S_ISREG(info.st_mode)) or found != target.old:
        msg = f"{target.shown} changed after you were asked, so I didn't write it."
        raise FileRefusalError(msg)
    return info


def _fill(fd: int, data: bytes, *, mode: int | None = None) -> None:
    """Write `data` to the open file and close it, first setting its `mode` when one is given."""
    with os.fdopen(fd, "wb") as handle:
        if mode is not None:
            os.fchmod(handle.fileno(), mode)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
