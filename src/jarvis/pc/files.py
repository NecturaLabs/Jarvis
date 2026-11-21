"""The owner's files: listed and read inside the read roots, written inside the write roots.

Every path is resolved through its links before it is checked, so a link inside a root cannot
lead out of it, and a link to a secret is refused as the secret. Some things are never read or
written whatever the roots say: keys, tokens, password stores, browser profiles and Jarvis's own
state (`SECRET_NAMES`, `SECRET_PATHS`, plus the owner's `files.deny`).

What a file holds, and the names in a folder, were written by someone (a download, a program), so
they reach the model as `file` data: they never raise a class, side effects need a yes while they
are in the conversation, and so does sending text off the PC (`policy.decide`).

Writing always asks first (`confirm`). The question names the resolved path, and the write runs
only if that path still resolves to itself: a name that turned into a link after the question is
refused. A write never touches a hidden file or folder (no `~/.bashrc`, `~/.config/…`), never
replaces a file by accident (`create` refuses an existing file) and never makes a file executable.
"""

import contextlib
import fnmatch
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
MAX_READ_BYTES = 4 * 1024 * 1024
BINARY_PROBE = 8192
KIB = 1024
HEADROOM = 120  # the header and the closing line of a result
FILE_MODE = 0o644
FOLDER_MODE = 0o755

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


@dataclass(frozen=True, slots=True)
class Boundary:
    """Where Jarvis may read and write, resolved through links once when it is made."""

    home: Path
    read_roots: tuple[Path, ...]
    write_roots: tuple[Path, ...]
    deny: tuple[str, ...] = ()
    protected: tuple[Path, ...] = ()

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

    def readable(self, raw: str) -> Path:
        """Return the resolved path to read or list.

        Raises:
            FileRefusalError: it is outside the read roots or private.
        """
        path = self._resolve(raw)
        if not _inside(path, self.read_roots):
            msg = f"{self.show(path)} is outside the folders I may read."
            raise FileRefusalError(msg)
        if self.secret(path):
            msg = f"{self.show(path)} is private (keys, tokens or logins), so I don't read it."
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


def _open_regular(path: Path, shown: str) -> tuple[int, os.stat_result]:
    """Open a file for reading without following a last link or blocking on a pipe.

    Raises:
        FileRefusalError: it is missing, a folder or not a regular file.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError as exc:
        msg = f"There's no file at {shown}."
        raise FileRefusalError(msg) from exc
    except IsADirectoryError as exc:
        msg = f"{shown} is a folder; list it instead."
        raise FileRefusalError(msg) from exc
    except OSError as exc:
        msg = f"I couldn't open {shown}: {exc.strerror}."
        raise FileRefusalError(msg) from exc
    info = os.fstat(fd)
    if stat.S_ISDIR(info.st_mode):
        os.close(fd)
        msg = f"{shown} is a folder; list it instead."
        raise FileRefusalError(msg)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        msg = f"{shown} is not a regular file, so I don't read it."
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
        fd, info = _open_regular(path, shown)
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
        path = self._boundary.readable(str(args["path"]))
        shown = self._boundary.show(path)
        if not path.is_dir():
            msg = f"{shown} is not a folder." if path.exists() else f"There's no folder {shown}."
            raise FileRefusalError(msg)
        hidden = args.get("hidden") is True
        pattern = str(args.get("pattern", "")).casefold()
        rows: list[tuple[bool, str, str]] = []
        try:
            entries = list(os.scandir(path))
        except OSError as exc:
            msg = f"I couldn't list {shown}: {exc.strerror}."
            raise FileRefusalError(msg) from exc
        for entry in entries:
            if (entry.name.startswith(".") and not hidden) or (
                pattern and not fnmatch.fnmatchcase(entry.name.casefold(), pattern)
            ):
                continue
            if self._boundary.secret(Path(os.path.realpath(entry.path))):
                continue
            rows.append(self._row(entry))
        rows.sort(key=lambda row: (not row[0], row[1].casefold()))
        lines: list[str] = []
        used = 0
        for _folder, _name, line in rows[:MAX_ENTRIES]:
            if used + len(line) + 1 > self._room:
                break
            lines.append(line)
            used += len(line) + 1
        what = f"{len(rows)} item{'' if len(rows) == 1 else 's'}"
        text = f"Folder {shown} ({what}{', matching ' + pattern if pattern else ''}):"
        text += "\n" + "\n".join(lines) if lines else "\n(nothing to show)"
        if len(lines) < len(rows):
            text += f"\n… and {len(rows) - len(lines)} more."
        return Result(text, Origin.FILE)

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
        return {**args, "path": str(path), "mode": mode}

    def write_summary(self, args: dict[str, Any]) -> str:
        """Return the confirmation question's words: how, which file, how much."""
        path = Path(str(args.get("path", "")))
        folder = self._boundary.show(path.parent).removeprefix("~/") if path.is_absolute() else ""
        where = f"{path.name} in {'your home folder' if folder == '~' else folder}"
        count = len(str(args.get("content", "")))
        verb = {"replace": "replace", "append": "add to"}.get(str(args.get("mode")), "create")
        what = "the file " if verb == "create" else ""
        plural = "" if count == 1 else "s"
        return f"{verb} {what}{where if folder else path} ({count} character{plural})"

    def write_file(self, args: dict[str, Any]) -> Result:
        """Write the confirmed file, if its path still resolves to what was asked about.

        Raises:
            FileRefusalError: the path changed since the question, or the write failed.
        """
        path = Path(str(args["path"]))
        again = self._boundary.writable(str(path))
        shown = self._boundary.show(path)
        if again != path:
            msg = f"{shown} changed after you were asked, so I didn't write it."
            raise FileRefusalError(msg)
        content = str(args["content"])
        if not content.endswith("\n"):
            content += "\n"
        mode = str(args.get("mode", "create"))
        try:
            self._make_parents(path.parent)
            _write(path, content.encode("utf-8"), mode)
        except FileExistsError as exc:
            msg = f"{shown} already exists; say whether to replace it or add to it."
            raise FileRefusalError(msg) from exc
        except OSError as exc:
            msg = f"I couldn't write {shown}: {exc.strerror or exc}."
            raise FileRefusalError(msg) from exc
        done = {"replace": "Replaced", "append": "Added to"}.get(mode, "Created")
        sentence = f"{done} {shown} ({len(content)} characters)."
        return Result(sentence, spoken=sentence)

    @staticmethod
    def _make_parents(folder: Path) -> None:
        """Create the missing folders up to `folder` (all below an existing, resolved one)."""
        missing: list[Path] = []
        while not folder.exists():
            missing.append(folder)
            folder = folder.parent
        for part in reversed(missing):
            part.mkdir(mode=FOLDER_MODE)

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
                "Create a text file on the owner's PC, or replace or add to one (mode). The owner "
                "is asked first.",
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
            ),
        ]


def _write(path: Path, data: bytes, mode: str) -> None:
    """Write `data` into the folder's own descriptor, never through a link at the last name.

    `create` makes a new file only; `append` adds to the end; `replace` writes a fresh file
    beside it and renames it over, so a reader sees the old content or the new.
    """
    folder = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        if mode == "replace":
            temp = f".{path.name}.{secrets.token_hex(6)}.tmp"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
            fd = os.open(temp, flags, FILE_MODE, dir_fd=folder)
            try:
                _fill(fd, data)
                os.replace(temp, path.name, src_dir_fd=folder, dst_dir_fd=folder)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(temp, dir_fd=folder)
                raise
            return
        extra = os.O_EXCL if mode == "create" else os.O_APPEND
        flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | extra
        _fill(os.open(path.name, flags, FILE_MODE, dir_fd=folder), data)
    finally:
        os.close(folder)


def _fill(fd: int, data: bytes) -> None:
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
