"""The conversation the voice service and the chat window share (`[channels] shared`, ADR 0030).

Each of the two agents keeps its own conversation, tools and limits. With sharing on, each also
writes every request the owner made of it, with its reply, to one private file, and reads the
other's new ones into its own conversation before its next request, marked with how the owner
said them ("said aloud", "typed in the chat"). A spoken request is answered aloud and a typed
one in text, as before; each channel just knows what was said in the other.

What a turn came with travels with it: the outside text its reply drew on (`Turn.outside`) and
whether it read the owner's files (`Turn.private`), so the other agent treats it as it treats
its own remembered turns, and an action after a page read in the chat still needs the owner's
yes by voice. A request heard in PC audio is never shared, and nor are the tool calls and
results a turn kept (`Turn.work`): only the request, the reply, what was done and looked up.

The file holds the newest `MAX_TURNS` turns, and an agent reads only turns of the last
`FRESH_S` seconds, the voice conversation's own idle time, so an old talk does not come back:
not in a process that starts, nor in one that sat idle. Every row names the file it was written
to (`epoch`), so a file that was removed and written again is read from its first turn. Asking
Jarvis to forget the conversation clears the file too (`clear`).
"""

import contextlib
import errno
import fcntl
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from jarvis.jsonutil import obj
from jarvis.memory import Turn, origin_name
from jarvis.secretfile import read_private, write_private

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

type Channel = Literal["voice", "chat"]

MAX_TURNS = 40  # the newest turns the file keeps
MAX_FIELD = 4000  # the most characters of one request, reply, actions or looked-up list kept
FRESH_S = 900.0  # how old a turn an agent still reads
LOCK_WAIT_S = 2.0  # how long a request waits for the other process to finish with the file
HOW = {"voice": "said aloud", "chat": "typed in the chat"}  # how the owner made the request
LEGACY_EPOCH = "legacy"  # keep old rows' identity when the next write adds the epoch field


class SharedBusyError(OSError):
    """The other process held the shared file's lock for longer than `LOCK_WAIT_S`."""


@dataclass(frozen=True, slots=True)
class SharedTurn:
    """One request the owner made of a channel, its reply and what came with it."""

    number: int
    channel: Channel  # the channel that answered it, and wrote it here
    at: float  # wall-clock seconds
    request: str
    reply: str
    actions: str = ""
    outside: str = ""
    private: bool = False
    looked: str = ""
    said: Channel | None = None  # how the owner made it, when not the way `channel` takes words
    epoch: str = ""  # which file it was written to

    def turn(self, at: float) -> Turn:
        """Return it as a turn of the other channel's conversation, at that channel's `at`."""
        return Turn(
            f"({HOW[self.said or self.channel]}) {self.request}",
            self.reply,
            self.actions,
            self.outside,
            at,
            private=self.private,
            looked=self.looked,
        )


def _channel(value: object) -> Channel | None:
    if value == "voice":
        return "voice"
    return "chat" if value == "chat" else None


def _parse(line: str) -> SharedTurn | None:
    try:
        row = obj(json.loads(line))
    except ValueError, RecursionError:
        return None
    channel, number = _channel(row.get("channel")), row.get("number")
    if channel is None or not isinstance(number, int):
        return None
    at = row.get("at")
    return SharedTurn(
        number,
        channel,
        float(at) if isinstance(at, (int, float)) else 0.0,
        str(row.get("request", ""))[:MAX_FIELD],
        str(row.get("reply", ""))[:MAX_FIELD],
        str(row.get("actions", ""))[:MAX_FIELD],
        origin_name(row.get("outside", "")),
        row.get("private") is True,
        str(row.get("looked", ""))[:MAX_FIELD],
        _channel(row.get("said")),
        str(row.get("epoch") or LEGACY_EPOCH),
    )


class SharedTurns:
    """One channel's side of the shared conversation file."""

    def __init__(
        self, path: Path, channel: Channel, *, clock: Callable[[], float] = time.time
    ) -> None:
        """Share through `path` as `channel`; read the other channel's turns of the last FRESH_S."""
        super().__init__()
        self._path = path
        self._channel: Channel = channel
        self._clock = clock
        self._epoch, self._seen = "", 0
        with contextlib.suppress(OSError):  # an unreadable file is reported when it is used
            turns = self._read()
            self._epoch = turns[-1].epoch if turns else ""

    @property
    def mark(self) -> str:
        """Where this channel has read up to, for a saved conversation to start from again."""
        return f"{self._epoch}:{self._seen}" if self._epoch else ""

    def resume(self, mark: str) -> None:
        """Start after `mark`, from a saved conversation that already holds the turns before it."""
        epoch, _, number = mark.partition(":")
        if epoch and epoch == self._epoch and number.isdigit():
            self._seen = max(self._seen, int(number))

    @contextlib.contextmanager
    def _locked(self) -> Generator[None]:
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = self._path.with_name(f"{self._path.name}.lock")
        fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            deadline = time.monotonic() + LOCK_WAIT_S
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise SharedBusyError(
                            errno.EWOULDBLOCK, "the shared file is busy"
                        ) from None
                    time.sleep(0.02)
            yield
        finally:
            os.close(fd)

    def _read(self) -> list[SharedTurn]:
        """Return the file's turns; none for a missing file. An unsafe or unreadable one raises."""
        text = read_private(self._path) or ""
        return [turn for line in text.splitlines() if (turn := _parse(line)) is not None]

    def publish(self, turn: Turn, *, said: Channel | None = None) -> None:
        """Write a finished turn of this channel for the other one to read.

        `said` is how the owner made the request when it differs from this channel's own way:
        spoken words handed to the chat are written as said aloud.
        """
        with self._locked():
            kept = self._read()
            epoch = kept[-1].epoch if kept else secrets.token_hex(8)
            number = max((t.number for t in kept), default=0) + 1
            row = {
                "number": number,
                "channel": self._channel,
                "at": round(self._clock(), 3),
                "request": turn.request[:MAX_FIELD],
                "reply": turn.reply[:MAX_FIELD],
                "actions": turn.actions[:MAX_FIELD],
                "outside": turn.outside,
                "private": turn.private,
                "looked": turn.looked[:MAX_FIELD],
                "said": said if said is not None and said != self._channel else None,
                "epoch": epoch,
            }
            lines = [json.dumps(_row(t, epoch)) for t in kept[-(MAX_TURNS - 1) :]]
            write_private(self._path, "\n".join([*lines, json.dumps(row)]) + "\n")

    def take(self) -> list[SharedTurn]:
        """Return the other channel's turns of the last FRESH_S written since the last call."""
        turns = self._read()
        epoch = turns[-1].epoch if turns else self._epoch
        if epoch != self._epoch:
            self._epoch, self._seen = epoch, 0  # the file was removed and written again
        fresh = self._clock() - FRESH_S
        new = [
            t
            for t in turns
            if t.number > self._seen and t.channel != self._channel and t.at >= fresh
        ]
        self._seen = max((t.number for t in turns), default=self._seen)
        return new

    def clear(self) -> None:
        """Forget every shared turn: the owner asked to start a new conversation."""
        with self._locked(), contextlib.suppress(FileNotFoundError):
            self._path.unlink()
        self._epoch, self._seen = "", 0


def _row(turn: SharedTurn, epoch: str) -> dict[str, object]:
    return {
        "number": turn.number,
        "channel": turn.channel,
        "at": turn.at,
        "request": turn.request,
        "reply": turn.reply,
        "actions": turn.actions,
        "outside": turn.outside,
        "private": turn.private,
        "looked": turn.looked,
        "said": turn.said,
        "epoch": epoch,
    }
