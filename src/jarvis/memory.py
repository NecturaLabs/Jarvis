"""The conversation Jarvis remembers: its turns, a summary of older ones, and what they cost.

The model reads the whole conversation with every request, so what is kept is measured in tokens
against a budget that fits the model's context window. When the turns outgrow it, the oldest are
folded into a summary (the agent asks the model to write it); when that fails, they are dropped.

A turn whose reply drew on outside text (a web page, a song name, a file) is kept as data, never
as the model's own words, and the conversation counts as holding outside text until that turn is
summarized away or forgotten. A turn that read one of the owner's files is also marked private, so
what Jarvis sends off the PC needs a yes while it is remembered.
"""

import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from jarvis.files import write_atomic
from jarvis.jsonutil import obj, objs
from jarvis.secretfile import read_private

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)
_ORIGIN_NAME = re.compile(r"[a-z_]{1,32}")  # what Origin values look like

CHARS_PER_TOKEN = 3.0  # Qwen's tokenizer averages about 3.5 to 4 characters of English a token
MESSAGE_OVERHEAD = 8  # the chat template's role markers around each message
NO_ACTIONS = "none (nothing was changed)"
KEEP_RECENT = 2  # turns never folded into the summary, so the latest exchange stays word for word
FILE_VERSION = 1


def tokens(text: str) -> int:
    """Return a cautious estimate of how many tokens the model reads for `text`."""
    return math.ceil(len(text) / CHARS_PER_TOKEN) + MESSAGE_OVERHEAD


def _inert(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


@dataclass(slots=True)
class Turn:
    """One request and the reply to it.

    `outside` names where outside text in the reply came from ("web", "spotify"...), empty when the
    reply is the model's own words over trusted results only. `actions` lists what was done.
    `private` marks a turn that read text from the owner's files.
    """

    request: str
    reply: str
    actions: str = ""
    outside: str = ""
    at: float = 0.0
    private: bool = False

    def messages(self) -> list[dict[str, Any]]:
        """Return the turn as the user and assistant messages the model reads."""
        reply = (
            f'<data origin="{self.outside}">{_inert(self.reply)}</data>'
            if self.outside
            else self.reply
        )
        done = self.actions or NO_ACTIONS
        return [
            {"role": "user", "content": self.request},
            {"role": "assistant", "content": f"[actions taken: {done}] {reply}"},
        ]

    def cost(self) -> int:
        """Return the tokens this turn takes in the conversation."""
        return sum(tokens(str(m["content"])) for m in self.messages())


@dataclass(slots=True)
class Conversation:
    """What Jarvis remembers of the talk so far, within a token budget.

    `budget` is how many tokens the turns and the summary may take together; `idle_s` forgets
    everything after that long without a request (0 never forgets).
    """

    budget: int
    idle_s: float = 0.0
    turns: list[Turn] = field(default_factory=list[Turn])
    summary: str = ""
    summary_outside: bool = False  # the summary was written from turns that held outside text
    summary_private: bool = False  # ... and from turns that read the owner's files
    last: float = 0.0  # on the agent's clock; not kept across sessions
    saved_at: float = 0.0  # wall-clock time of the save a loaded conversation came from

    def expire(self, now: float) -> bool:
        """Forget everything after a long silence; return whether something was forgotten."""
        if self.idle_s <= 0 or not self.last or now - self.last <= self.idle_s:
            return False
        had = bool(self.turns or self.summary)
        self.clear()
        return had

    def clear(self) -> None:
        """Forget the whole conversation."""
        self.turns.clear()
        self.summary = ""
        self.summary_outside = False
        self.summary_private = False

    def add(self, turn: Turn) -> None:
        """Remember one finished turn."""
        self.turns.append(turn)
        self.last = turn.at

    @property
    def holds_outside_text(self) -> bool:
        """Whether anything remembered came from outside text."""
        return self.summary_outside or any(turn.outside for turn in self.turns)

    @property
    def holds_private_text(self) -> bool:
        """Whether anything remembered came from the owner's files."""
        return self.summary_private or any(turn.private for turn in self.turns)

    def summary_messages(self) -> list[dict[str, Any]]:
        """Return the summary as messages, or none when there is no summary."""
        if not self.summary:
            return []
        text = (
            f'<data origin="memory">{_inert(self.summary)}</data>'
            if self.summary_outside
            else self.summary
        )
        return [
            {"role": "user", "content": f"(A summary of our conversation so far.) {text}"},
            {"role": "assistant", "content": f"[actions taken: {NO_ACTIONS}] Noted."},
        ]

    def messages(self) -> list[dict[str, Any]]:
        """Return everything remembered as messages, oldest first."""
        out = self.summary_messages()
        for turn in self.turns:
            out.extend(turn.messages())
        return out

    def cost(self) -> int:
        """Return the tokens the remembered conversation takes."""
        return sum(tokens(str(m["content"])) for m in self.messages())

    def over_budget(self) -> bool:
        """Return whether the conversation has outgrown its budget."""
        return self.cost() > self.budget

    def oldest_to_fold(self) -> list[Turn]:
        """Return the oldest turns to fold into the summary: enough to fit, but never the latest."""
        foldable = self.turns[: max(0, len(self.turns) - KEEP_RECENT)]
        if not foldable:
            return []
        excess = self.cost() - self.budget // 2  # fold down to half, so it is not redone each turn
        chosen: list[Turn] = []
        for turn in foldable:
            chosen.append(turn)
            excess -= turn.cost()
            if excess <= 0:
                break
        return chosen

    def fold(self, folded: list[Turn], summary: str) -> None:
        """Replace `folded` (the oldest turns) with a new summary that covers them."""
        self.summary_outside = self.summary_outside or any(turn.outside for turn in folded)
        self.summary_private = self.summary_private or any(turn.private for turn in folded)
        self.summary = summary.strip()
        del self.turns[: len(folded)]

    def drop_to_fit(self) -> int:
        """Drop the oldest turns, then the summary, until the conversation fits; return how many."""
        dropped = 0
        while self.turns and self.over_budget():
            del self.turns[0]
            dropped += 1
        if self.over_budget():
            self.summary = ""
            self.summary_outside = False
            self.summary_private = False
        return dropped

    # --- keeping it across sessions -----------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write the conversation to `path`, readable by the owner only."""
        data = {
            "version": FILE_VERSION,
            "summary": self.summary,
            "summary_outside": self.summary_outside,
            "summary_private": self.summary_private,
            "saved_at": time.time(),
            "turns": [asdict(turn) for turn in self.turns],
        }
        write_atomic(path, json.dumps(data, ensure_ascii=False), mode=0o600)

    def load(self, path: Path) -> bool:
        """Read a conversation saved by `save`; return whether one was read.

        A missing, unreadable or foreign file leaves the conversation empty.
        """
        try:
            saved = _read(path)
        except (OSError, ValueError, TypeError) as exc:
            if not isinstance(exc, FileNotFoundError):
                log.warning("the saved conversation could not be read: %s", exc)
            return False
        if saved is None:
            return False
        turns, summary, (summary_outside, summary_private), saved_at = saved
        self.turns, self.summary = turns, summary
        self.summary_outside, self.summary_private = summary_outside, summary_private
        self.saved_at, self.last = saved_at, 0.0
        return bool(turns or summary)


def _origin(value: object) -> str:
    """Return a saved turn's outside mark, or `outside` for one that is not a plain origin name."""
    text = str(value)
    return text if not text or _ORIGIN_NAME.fullmatch(text) else "outside"


def _read(path: Path) -> tuple[list[Turn], str, tuple[bool, bool], float] | None:
    """Return the turns, summary, its outside and private marks and save time, or None if foreign.

    The file must be the owner's own, readable by nobody else and not a link (`read_private`).
    What it, `json.loads`, `obj` and `float` raise for such a file, or an unreadable, non-JSON or
    malformed one, passes through.

    Raises:
        FileNotFoundError: there is no saved conversation.
    """
    text = read_private(path)
    if text is None:
        raise FileNotFoundError(path)
    data = obj(json.loads(text))
    if data.get("version") != FILE_VERSION:
        return None
    turns = [
        Turn(
            str(t.get("request", "")),
            str(t.get("reply", "")),
            str(t.get("actions", "")),
            _origin(t.get("outside", "")),
            float(t.get("at", 0.0)),
            private=t.get("private") is True,
        )
        for t in objs(data.get("turns"))
    ]
    marks = (data.get("summary_outside") is True, data.get("summary_private") is True)
    return turns, str(data.get("summary", "")), marks, float(data.get("saved_at", 0.0))
