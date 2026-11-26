"""Push announcements into T3 Code threads that run Jarvis, so T3 Code shows them at once.

T3 Code drops message chunks that arrive outside a prompt, so an announcement written into a
session waits for the owner's next message. T3 Code's own command line can start a turn instead:
`t3 thread send <thread> -` hands a message to a thread, and "a busy thread gets it as its next
turn; nothing is interrupted". The turn reaches `jarvis acp` as a prompt, and `jarvis acp` answers
it with the announcement itself, without the model or any tool (`acp.py`), so the owner sees the
line with T3 Code's own turn-finished notification.

The message carries a tag only Jarvis can make: a keyed hash (the key is a private file in
Jarvis's state folder) of the announcement's number, time and words. A prompt without a valid tag
is the owner's ordinary input. Which threads: those of the project a session was opened in whose
provider is a Jarvis ACP provider, found from T3 Code's settings (`providerInstances` whose
command arguments are `acp` and whose program is named `jarvis`) unless `acp.t3_provider` names
the provider; the newest that is not settled. Nothing happens, and nothing is raised, when `t3` is
not there, the settings name no Jarvis provider or the thread cannot be found.

The program is started with an argument list, never a shell; the words go to its standard input.
"""

import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jarvis import announce, paths
from jarvis.jsonutil import obj, objs
from jarvis.secretfile import read_private, write_private

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from jarvis.announce import Announcement

MIN_KEY = 32  # characters of a usable key
KEY_LOCK_MODE = 0o600
MAX_WORDS = 500  # characters of an announcement passed on
MAX_SETTINGS_BYTES = 1024 * 1024
# The arguments T3 Code runs Jarvis with: the regular provider, and the coding one.
ACP_ARGS = (["acp"], ["acp", "--coding"])
TIMEOUT_S = 20.0
KEPT_TAGS = 200  # pushed announcements remembered, so two Jarvis processes send one
MARK = "⟲ Jarvis: "
_TAG = re.compile(r"\n\[announcement ([0-9]{1,18})-([0-9]{1,18})-([0-9a-f]{16})\]\Z")
_THREAD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")  # a thread id as `t3` prints it


@dataclass(frozen=True, slots=True)
class Ran:
    """What a run of `t3` came to: its exit code and what it wrote."""

    code: int
    out: str


type Runner = Callable[[list[str], str], Ran]


def run_t3(args: list[str], words: str) -> Ran:
    """Run `args` (an argument list, no shell) with `words` on its input; never raises."""
    import subprocess  # ruff: ignore[import-outside-top-level] - only a Jarvis that pushes to T3 Code starts a program

    try:
        done = subprocess.run(
            args,
            input=words,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return Ran(-1, "")
    return Ran(done.returncode, done.stdout)


def clean(text: str) -> str:
    """Return `text` as one plain line of at most `MAX_WORDS` characters."""
    return announce.plain(text)[:MAX_WORDS]


def _key_file() -> Path:
    return paths.state_dir() / "t3-push.key"


def _usable(key: str | None) -> bool:
    return key is not None and len(key.strip()) >= MIN_KEY


def _make_key(path: Path) -> None:
    """Make the key file when it is missing or broken, once, whichever Jarvis process gets here.

    Under an exclusive lock on a file beside it, so two processes never each replace a broken key
    with their own; the new key is written whole and private to a temporary file and moved into
    place, so a reader sees the old file or the complete new one, and no hard link is needed.

    Raises:
        PermissionError: the lock file is not regular, private and owned by this user.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = os.open(
        path.with_name(f"{path.name}.lock"),
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
        KEY_LOCK_MODE,
    )
    try:
        status = os.fstat(lock)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) != KEY_LOCK_MODE
        ):
            msg = "the announcement key lock must be a regular file of your own with mode 0600"
            raise PermissionError(msg)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _usable(read_private(path)):  # another process made it while this waited
            return
        mine = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}")
        try:
            write_private(mine, secrets.token_hex(32))
            mine.replace(path)
        finally:
            mine.unlink(missing_ok=True)
    finally:
        os.close(lock)  # releases the lock


def _key() -> bytes:
    """Return the private key that tags what Jarvis pushes, making it on first use.

    Raises:
        OSError: the key file cannot be read or made.
    """
    path = _key_file()
    found = read_private(path)
    if not _usable(found):
        _make_key(path)
        found = read_private(path)
    if found is None or not _usable(found):
        msg = f"no usable key in {path}"
        raise OSError(msg)
    return found.strip().encode()


def _mac(key: bytes, number: int, at: int, words: str) -> str:
    return hmac.new(key, f"{number}\n{at}\n{words}".encode(), hashlib.sha256).hexdigest()[:16]


def message(item: Announcement) -> str:
    """Return the message that pushes `item`: its words, then a tag only Jarvis can make."""
    words = clean(item.text)
    at = int(item.at)
    return f"{MARK}{words}\n[announcement {item.id}-{at}-{_mac(_key(), item.id, at, words)}]"


@dataclass(frozen=True, slots=True)
class Delivery:
    """The identity and words authenticated by an announcement's tag."""

    id: int
    at: int
    text: str


def verified_delivery(prompt: str) -> Delivery | None:
    """Return the authenticated announcement when `prompt` is a message Jarvis made."""
    if not prompt.startswith(MARK):
        return None
    tag = _TAG.search(prompt)
    if tag is None:
        return None
    words = prompt[len(MARK) : tag.start()]
    if not words or words != clean(words):
        return None
    try:
        number, at, given = int(tag.group(1)), int(tag.group(2)), tag.group(3)
        key = _key()
    except OSError, ValueError:
        return None
    return (
        Delivery(number, at, words)
        if hmac.compare_digest(given, _mac(key, number, at, words))
        else None
    )


def verified(prompt: str) -> str | None:
    """Return the words of an authenticated announcement, when there is one."""
    delivery = verified_delivery(prompt)
    return None if delivery is None else delivery.text


def t3_home() -> Path:
    """Return T3 Code's data folder: `T3CODE_HOME`, else `~/.t3`."""
    return Path(os.environ.get("T3CODE_HOME") or Path.home() / ".t3")


def jarvis_providers() -> frozenset[str]:
    """Return the ids of T3 Code's providers that run `jarvis acp` (or `jarvis acp --coding`)."""
    path = t3_home() / "userdata" / "settings.json"
    try:
        if path.stat().st_size > MAX_SETTINGS_BYTES:
            return frozenset()
        settings = obj(json.loads(path.read_text(encoding="utf-8")))
    except OSError, ValueError:
        return frozenset()
    found: set[str] = set()
    for name, entry in obj(settings.get("providerInstances")).items():
        config = obj(obj(entry).get("config"))
        program = config.get("commandPath")
        if (
            config.get("commandArgs") in ACP_ARGS
            and isinstance(program, str)
            and Path(program).name.startswith("jarvis")
        ):
            found.add(str(name))
    return frozenset(found)


class T3Push:
    """Sends an announcement into the T3 Code thread of each project a session is open in."""

    def __init__(
        self,
        command: str = "t3",
        provider: str = "",
        runner: Runner = run_t3,
        providers: Callable[[], frozenset[str]] = jarvis_providers,
        *,
        enabled: bool = True,
    ) -> None:
        """Run `command`; `provider` is T3 Code's id for Jarvis, else it is looked up.

        Without `enabled` nothing is pushed (`push` returns at once); `verified` and `marked`
        still tell a message of Jarvis's from one that only looks like it.
        """
        super().__init__()
        self._enabled = enabled
        self._command = command
        self._provider = provider
        self._runner = runner
        self._providers = providers

    @staticmethod
    def marked(prompt: str) -> bool:
        """Whether `prompt` starts as a message Jarvis pushes does, valid or not."""
        return prompt.lstrip().startswith(MARK.rstrip())

    @staticmethod
    def verified(prompt: str) -> str | None:
        """Return the announcement's words when `prompt` is a message Jarvis pushed, else None."""
        return verified(prompt)

    @staticmethod
    def verified_delivery(prompt: str) -> Delivery | None:
        """Return the authenticated identity and words from a pushed prompt."""
        return verified_delivery(prompt)

    def available(self) -> bool:
        """Whether the program is there to run."""
        return shutil.which(self._command) is not None

    def _wanted(self) -> frozenset[str]:
        """The providers to push into: those T3 Code runs as `jarvis acp`, or the one picked."""
        runs_jarvis = self._providers()
        return runs_jarvis & {self._provider} if self._provider else runs_jarvis

    def provider_problem(self) -> str | None:
        """Say why nothing can be pushed when `t3_provider` is not a provider that runs Jarvis."""
        if self._provider and self._provider not in self._providers():
            return (
                f"acp.t3_provider {self._provider!r} is not a T3 Code provider that runs "
                "`jarvis acp` in its settings, so nothing is pushed"
            )
        return None

    def push(self, item: Announcement, projects: Iterable[str]) -> set[str]:
        """Send `item` into the newest open Jarvis thread of each of `projects`.

        Returns the projects it was sent for, or that another Jarvis already sent it for.
        """
        done: set[str] = set()
        if not self._enabled or not self.available():
            return done
        wanted = self._wanted()
        if not wanted:
            return done
        text = message(item)
        for project in sorted(set(projects)):
            if not _claim(f"{item.id}-{int(item.at)}@{project}"):
                done.add(project)  # another Jarvis process sends it
            elif self._send(project, wanted, text):
                done.add(project)
            else:
                _release(f"{item.id}-{int(item.at)}@{project}")
        return done

    def _send(self, project: str, wanted: frozenset[str], text: str) -> bool:
        listed = self._runner([self._command, "thread", "list", "--project", project, "--json"], "")
        if listed.code != 0:
            return False
        try:
            threads = objs(json.loads(listed.out))
        except ValueError:
            return False
        for thread in threads:  # newest first
            ident = thread.get("threadId")
            if (
                isinstance(ident, str)
                and _THREAD.fullmatch(ident)
                and thread.get("providerInstanceId") in wanted
                and thread.get("settled") is not True
            ):
                sent = self._runner([self._command, "thread", "send", ident, "-"], text)
                return sent.code == 0
        return False


def _tags_file() -> Path:
    return paths.state_dir() / "t3-pushed.json"


def _claim(tag: str) -> bool:
    """Record that `tag` is being pushed; False when it already was (by any Jarvis process)."""
    path = _tags_file()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            tags = [str(t) for t in json.loads(handle.read() or "[]")]
        except ValueError:
            tags = []
        if tag in tags:
            return False
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps([*tags, tag][-KEPT_TAGS:]))
    return True


def _release(tag: str) -> None:
    """Forget that `tag` was pushed, after a send that failed, so another may try."""
    path = _tags_file()
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    except OSError:
        return
    with os.fdopen(fd, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            tags = [str(t) for t in json.loads(handle.read() or "[]") if str(t) != tag]
        except ValueError:
            tags = []
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(tags))
