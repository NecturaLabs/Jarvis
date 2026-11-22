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
import itertools
import os
import secrets
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.operations import Operation, OperationError, Result
from jarvis.policy import OpClass, Origin

if TYPE_CHECKING:
    from collections.abc import Iterable

    from jarvis.config import FilesConfig

MAX_PATH = 1000
MAX_CONTENT = 20000
MAX_ENTRIES = 200
MAX_SCANNED = 10_000  # entries read from one folder; a bigger one is listed from these
MAX_READ_BYTES = 4 * 1024 * 1024
BINARY_PROBE = 8192
KIB = 1024
HEADROOM = 120  # the header and the closing line of a result
FILE_MODE = 0o644
FOLDER_MODE = 0o755
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


SYSTEM_FOLDERS = (Path("/proc"), Path("/sys"), Path("/dev"), Path("/run"))


class FileRefusalError(OperationError):
    """A refused or failed file operation, whose words reach the model as `file` data.

    It names paths found on disk (a link's target, a folder's entries), which anyone can choose.
    """

    def __init__(self, message: str) -> None:
        """Keep the message, marked as file data."""
        super().__init__(message, origin=Origin.FILE)


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


def _listed(names: list[str]) -> str:
    """Return names as they are said in a list: "a", "a and b", "a, b and c"."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _printable(line: str) -> str:
    """Return a line with every control character shown as a replacement mark."""
    text = "".join(ch if ch.isprintable() else "\ufffd" for ch in line.replace("\t", " "))
    return text if len(text) <= PREVIEW_WIDTH else text[: PREVIEW_WIDTH - 1] + "\u2026"


def _open_folder(folder: Path, *, create: int = 0) -> int:
    """Open a resolved folder one name at a time from `/`, following no link on the way.

    The last `create` names may be missing; they are made (mode `FOLDER_MODE`) as they are met.
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
    deny: tuple[str, ...] = ()
    protected: tuple[Path, ...] = ()
    hidden_allow: tuple[str, ...] = ()

    @classmethod
    def of(cls, config: FilesConfig, home: Path, *, protected: Iterable[Path] = ()) -> Boundary:
        """Return the boundary a `[files]` section sets; `protected` folders are never touched."""
        home = Path(os.path.realpath(home))

        def expand(root: str) -> Path:
            text = str(home) + root[1:] if root.startswith("~") else root
            return Path(os.path.realpath(text))

        return cls(
            home,
            tuple(expand(root) for root in config.read_roots),
            tuple(expand(root) for root in config.write_roots),
            tuple(config.deny),
            tuple(Path(os.path.realpath(path)) for path in (*protected, *SYSTEM_FOLDERS)),
            tuple(config.hidden_allow),
        )

    def show(self, path: Path) -> str:
        """Return a path as the owner says it: under home as `~/…`."""
        if path == self.home:
            return "~"
        return f"~/{path.relative_to(self.home)}" if path.is_relative_to(self.home) else str(path)

    def _resolve(self, raw: str) -> Path:
        """Return what a path the model gave means, through every link that exists.

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
            text = str(self.home / text)
        return Path(os.path.realpath(text))

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
        for pattern in (*SECRET_NAMES, *(d for d in self.deny if "/" not in d)):
            if any(fnmatch.fnmatchcase(name, pattern.casefold()) for name in names):
                return True
        if base != self.home:
            return False
        for pattern in (*SECRET_PATHS, *(d for d in self.deny if "/" in d)):
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
        path = self._resolve(raw)
        roots = [root for root in self.write_roots if path.is_relative_to(root) and path != root]
        if not roots:
            msg = f"{self.show(path)} is outside the folders I may write to."
            raise FileRefusalError(msg)
        below = path.relative_to(max(roots, key=lambda root: len(root.parts))).parts
        if any(part.startswith(".") for part in below):
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
    """

    def __init__(self, boundary: Boundary, *, max_chars: int) -> None:
        """Work within `boundary`, giving at most `max_chars` characters a result."""
        super().__init__()
        self._boundary = boundary
        self._room = max(200, max_chars - HEADROOM)

    # --- reading ------------------------------------------------------------------------------

    def read_file(self, args: dict[str, Any]) -> Result:
        """Return a text file's lines from `from_line` on, as much as fits one result.

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
        total = len(lines)
        first = max(1, int(args.get("from_line", 1)))
        if total and first > total:
            msg = f"{shown} has only {total} lines."
            raise FileRefusalError(msg)
        body, last = "", first - 1
        for line in lines[first - 1 :]:
            if body and len(body) + len(line) > self._room:
                break
            body += line[: self._room]
            last += 1
        span = f"lines {first} to {last} of {total}" if total else "empty"
        head = f"File {shown} ({_size(info.st_size)}, changed {_when(info.st_mtime)}, {span}):"
        tail = ""
        if last < total:
            tail = f"\n(More follows: read_file with from_line={last + 1}.)"
        elif len(data) >= MAX_READ_BYTES and info.st_size > MAX_READ_BYTES:
            tail = f"\n(Only the first {_size(MAX_READ_BYTES)} can be read.)"
        return Result(f"{head}\n{body.rstrip(chr(10))}{tail}", Origin.FILE)

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
        used = 0
        for _folder, _name, line in rows[:MAX_ENTRIES]:
            if used + len(line) + 1 > self._room:
                break
            lines.append(line)
            used += len(line) + 1
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
        for `replace`, the old file's size (`old_size`, None when there is none), so the question
        names them and the write refuses when they no longer hold.

        Raises:
            FileRefusalError: the path may not be written, or `create` meets an existing file.
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
        new_folders = 0
        while not path.parents[new_folders].exists():
            new_folders += 1
        bound = {**args, "path": str(path), "mode": mode, "new_folders": new_folders}
        if mode == "replace":
            bound["old_size"] = path.stat().st_size if path.exists() else None
        return bound

    def write_summary(self, args: dict[str, Any]) -> str:
        """Return the confirmation question's words: how, which file, how much, what is made."""
        path = Path(str(args.get("path", "")))
        folder = self._boundary.show(path.parent).removeprefix("~/") if path.is_absolute() else ""
        where = f"{path.name} in {'your home folder' if folder == '~' else folder}"
        count = len(str(args.get("content", "")))
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
        content = str(args["content"])
        if not content.endswith("\n"):
            content += "\n"
        mode = str(args.get("mode", "create"))
        try:
            folder = _open_folder(path.parent, create=int(args.get("new_folders") or 0))
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
            old_size=args.get("old_size"),
        )
        try:
            _write(target, content.encode("utf-8"), mode)
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
        done = {"replace": "Replaced", "append": "Added to"}.get(mode, "Created")
        sentence = f"{done} {shown} ({len(content)} characters)."
        return Result(sentence, spoken=sentence)

    # --- the operations -----------------------------------------------------------------------

    def operations(self) -> list[Operation]:
        """Return `list_dir`, `read_file` and `write_file`."""
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
                "Read a text file on the owner's PC; a long one in parts (from_line reads on).",
                {
                    "type": "object",
                    "properties": {
                        "path": path,
                        "from_line": {"type": "integer", "minimum": 1, "maximum": 10_000_000},
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
                "Create a text file on the owner's PC, or replace or add to one (mode). Jarvis "
                "asks the owner first when they want to be asked.",
                {
                    "type": "object",
                    "properties": {
                        "path": path,
                        "content": {"type": "string", "maxLength": MAX_CONTENT},
                        "mode": {
                            "type": "string",
                            "enum": ["create", "replace", "append"],
                            "default": "create",
                        },
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                self.write_file,
                default_class=OpClass.CONFIRM,
                spoken_summary=self.write_summary,
                prepare=self.bind_write,
                preview=self.write_preview,
                hashed=frozenset({"content"}),
            ),
        ]


@dataclass(frozen=True, slots=True)
class _Target:
    """The file a confirmed write goes to: its folder's descriptor and what was asked about."""

    folder: int
    name: str
    shown: str
    single: bool  # refuse a file with a second name on disk (under home)
    old_size: int | None  # for `replace`: the size the question named, None for no file


def _write(target: _Target, data: bytes, mode: str) -> None:
    """Write `data` into the folder's own descriptor, never through a link at the last name.

    `create` makes a new file only; `append` adds to the end; `replace` writes a fresh file
    beside it and renames it over, so a reader sees the old content or the new.

    Raises:
        FileRefusalError: the old file is not the one the question named, or (for `append`) it
            is not a regular file or is hard-linked when `target.single`.
    """
    folder, name = target.folder, target.name
    if mode == "replace":
        _check_replaced(target)
        temp = f".{name}.{secrets.token_hex(6)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(temp, flags, FILE_MODE, dir_fd=folder)
        try:
            _fill(fd, data)
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
    _fill(fd, data)


def _check_replaced(target: _Target) -> None:
    """Refuse a `replace` when the old file is not the size the question named.

    Raises:
        FileRefusalError: there is a file where the question named none, none where it named
            one, a different size, or something other than a regular file.
    """
    try:
        info = os.stat(target.name, dir_fd=target.folder, follow_symlinks=False)
    except FileNotFoundError:
        info = None
    found = None if info is None else info.st_size
    if (info is not None and not stat.S_ISREG(info.st_mode)) or found != target.old_size:
        msg = f"{target.shown} changed after you were asked, so I didn't write it."
        raise FileRefusalError(msg)


def _fill(fd: int, data: bytes) -> None:
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
