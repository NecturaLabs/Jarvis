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

The file holds the newest `MAX_TURNS` turns. A process that starts reads only the turns of the
last `FRESH_S` seconds, the voice conversation's own idle time, so an old talk does not come
back.
"""

import contextlib
import fcntl
import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from jarvis.jsonutil import obj
from jarvis.memory import Turn
from jarvis.secretfile import read_private, write_private

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

type Channel = Literal["voice", "chat"]

MAX_TURNS = 40  # the newest turns the file keeps
MAX_FIELD = 4000  # the most characters of one request or reply kept
FRESH_S = 900.0  # how old a turn a newly started process still reads
HOW = {"voice": "said aloud", "chat": "typed in the chat"}  # how the owner made the request


@dataclass(frozen=True, slots=True)
class SharedTurn:
    """One request the owner made of a channel, its reply and what came with it."""

    number: int
    channel: Channel
    at: float  # wall-clock seconds
    request: str
    reply: str
    actions: str = ""
    outside: str = ""
    private: bool = False
    looked: str = ""

    def turn(self, at: float) -> Turn:
        """Return it as a turn of the other channel's conversation, at that channel's `at`."""
        return Turn(
            f"({HOW[self.channel]}) {self.request}",
            self.reply,
            self.actions,
            self.outside,
            at,
            private=self.private,
            looked=self.looked,
        )


def _parse(line: str) -> SharedTurn | None:
    try:
        row = obj(json.loads(line))
    except ValueError, RecursionError:
        return None
    channel, number = row.get("channel"), row.get("number")
    if channel not in HOW or not isinstance(number, int):
        return None
    at = row.get("at")
    return SharedTurn(
        number,
        "voice" if channel == "voice" else "chat",
        float(at) if isinstance(at, (int, float)) else 0.0,
        str(row.get("request", "")),
        str(row.get("reply", "")),
        str(row.get("actions", "")),
        str(row.get("outside", "")),
        row.get("private") is True,
        str(row.get("looked", "")),
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
        fresh = clock() - FRESH_S
        self._seen = max((t.number for t in self._read() if t.at < fresh), default=0)

    @contextlib.contextmanager
    def _locked(self) -> Generator[None]:
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = self._path.with_name(f"{self._path.name}.lock")
        fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _read(self) -> list[SharedTurn]:
        try:
            text = read_private(self._path) or ""
        except OSError:
            return []
        return [turn for line in text.splitlines() if (turn := _parse(line)) is not None]

    def publish(self, turn: Turn) -> None:
        """Write a finished turn of this channel for the other one to read."""
        with self._locked():
            kept = self._read()
            number = max((t.number for t in kept), default=0) + 1
            row = {
                "number": number,
                "channel": self._channel,
                "at": round(self._clock(), 3),
                "request": turn.request[:MAX_FIELD],
                "reply": turn.reply[:MAX_FIELD],
                "actions": turn.actions,
                "outside": turn.outside,
                "private": turn.private,
                "looked": turn.looked,
            }
            lines = [json.dumps(_row(t)) for t in kept[-(MAX_TURNS - 1) :]]
            write_private(self._path, "\n".join([*lines, json.dumps(row)]) + "\n")

    def take(self) -> list[SharedTurn]:
        """Return the other channel's turns written since the last call, oldest first."""
        turns = self._read()
        if turns and max(t.number for t in turns) < self._seen:
            self._seen = 0  # the file was removed and started again
        new = [t for t in turns if t.number > self._seen and t.channel != self._channel]
        self._seen = max((t.number for t in turns), default=self._seen)
        return new


def _row(turn: SharedTurn) -> dict[str, object]:
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
    }
