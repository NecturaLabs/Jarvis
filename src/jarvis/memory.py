"""The conversation Jarvis remembers: its turns, a summary of older ones, and what they cost.

The model reads the whole conversation with every request, so what is kept is measured in tokens
against a budget that fits the model's context window. When the turns outgrow it, the oldest are
folded into a summary (the agent asks the model to write it); when that fails, they are dropped.
A turn may also keep its work: the tool calls the model made for it and what they returned, as the
model last read them (`memory_detail = "full"`). That goes first when room runs short, oldest
first, before anything is summarized.

A turn whose reply drew on outside text (a web page, a song name, a file) is kept as data, never
as the model's own words, and the conversation counts as holding outside text until that turn is
summarized away or forgotten. A turn that read one of the owner's files is also marked private, so
what Jarvis sends off the PC needs a yes while it is remembered.
"""

import copy
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

from jarvis.files import write_atomic
from jarvis.jsonutil import obj, objs
from jarvis.policy import Origin
from jarvis.secretfile import read_private

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)
_ORIGIN_NAME = re.compile(r"[a-z_]{1,32}")  # what Origin values look like

CHARS_PER_TOKEN = 3.0  # Qwen's tokenizer averages about 3.5 to 4 characters of English a token
MESSAGE_OVERHEAD = 8  # the chat template's role markers around each message
NO_ACTIONS = "none (nothing was changed)"
KEEP_RECENT = 2  # turns never folded into the summary, so the latest exchange stays word for word
SHED_TO = 3 / 4  # of the budget: work is shed down to this, so it is not shed again every request
FILE_VERSION = 1
_ROLES = frozenset({"assistant", "tool", "user"})
# What starts the owner's words typed while a request ran, as the request's work holds them.
STEERED = "(The owner adds, while you work on this request:)"
_WRAPPED_ORIGIN = re.compile(r'<data origin="([a-z_]{1,32})">')
# Results that may hold the owner's own data: the turn is marked private, so text that leaves the PC
# waits for a yes. A shell command's output can be any file the owner's user can read, and PC audio
# heard for the owner (`pc_audio_listen`) what people said in a call.
PRIVATE_ORIGINS = frozenset({Origin.FILE, Origin.SHELL, Origin.PC_AUDIO})


def tokens(text: str) -> int:
    """Return a cautious estimate of how many tokens the model reads for `text`."""
    return math.ceil(len(text) / CHARS_PER_TOKEN) + MESSAGE_OVERHEAD


def message_tokens(message: dict[str, Any]) -> int:
    """Return the tokens one message takes: its words and the calls it makes, with their arguments.

    A call's arguments can be most of a message (a whole file being written), so they count.
    """
    calls = message.get("tool_calls")
    called = json.dumps(calls, ensure_ascii=False) if calls else ""
    return tokens(f"{message.get('content') or ''}{called}")


def _inert(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


@dataclass(slots=True)
class Turn:
    """One request and the reply to it.

    `outside` names where outside text in the reply came from ("web", "spotify"...), empty when the
    reply is the model's own words over trusted results only. `actions` lists what was done.
    `private` marks a turn that read text from the owner's files. `work` is what the model did
    for the request between it and the reply: its tool calls and their results as it last read
    them (outside text in them already wrapped as data), empty when they are not kept.
    """

    request: str
    reply: str
    actions: str = ""
    outside: str = ""
    at: float = 0.0
    private: bool = False
    looked: str = ""  # what the look-ups asked for: queries, pages, places
    work: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    def messages(self, *, work: bool = True) -> list[dict[str, Any]]:
        """Return the turn as the messages the model reads: the request, its work, the reply.

        The work comes as the model read it while it worked, so the server can reuse what it
        already read of them. What was looked up goes with the reply, inside its data when the
        reply drew on outside text (a page's address is that page's words), so a follow-up can
        read the same page again rather than search for it. Without `work`, only the request
        and the reply.
        """
        said = f"[looked up: {self.looked}] {self.reply}" if self.looked else self.reply
        reply = f'<data origin="{self.outside}">{_inert(said)}</data>' if self.outside else said
        done = self.actions or NO_ACTIONS
        return [
            {"role": "user", "content": self.request},
            *(copy.deepcopy(self.work) if work else []),
            {"role": "assistant", "content": f"[actions taken: {done}] {reply}"},
        ]

    def shed(self) -> None:
        """Leave the work out; words the owner typed while it ran join the request instead."""
        steered = [
            str(m["content"])
            for m in self.work
            if m.get("role") == "user" and str(m.get("content", "")).startswith(STEERED)
        ]
        self.request = "\n".join([self.request, *steered])
        self.work = []

    def cost(self, *, work: bool = True) -> int:
        """Return the tokens this turn takes in the conversation."""
        return sum(message_tokens(m) for m in self.messages(work=work))


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

    def holds_outside_text_but(self, origin: str) -> bool:
        """Whether anything remembered came from outside text of another origin than `origin`.

        A summary of outside text counts, whatever its origin.
        """
        return self.summary_outside or any(t.outside not in {"", origin} for t in self.turns)

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
        return sum(message_tokens(m) for m in self.messages())

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

    def fitted_messages(self) -> list[dict[str, Any]]:
        """Return what `drop_to_fit` would leave, as messages, and change nothing."""
        view = replace(self, turns=[replace(turn) for turn in self.turns])
        view.drop_to_fit()
        return view.messages()

    def shed_work(self, target: int) -> int:
        """Leave out the oldest turns' work until the conversation costs `target` at most.

        The latest turn keeps its work while the conversation fits its budget with it. A turn
        keeps its request, reply, actions, look-ups and marks. Return how many turns shed work.
        """
        shed = 0
        for index, turn in enumerate(self.turns):
            if self.cost() <= target:
                break
            latest = index == len(self.turns) - 1
            if not turn.work or (latest and not self.over_budget()):
                continue
            turn.shed()
            shed += 1
        return shed

    def drop_to_fit(self) -> int:
        """Shed work, then drop the oldest turns, then the summary, until it fits; return drops.

        Only the turns dropped are counted; work shed on the way is not.
        """
        self.shed_work(self.budget)
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
            looked=str(t.get("looked", "")),
            work=_work(t.get("work")),
        )
        for t in objs(data.get("turns"))
    ]
    for turn in turns:  # the marks never say less than the work shows
        origins = _wrapped_origins(turn.work)
        listing = Origin.PROCESS.value
        other = next((origin for origin in origins if origin != listing), "")
        if turn.outside in {"", listing}:  # a listing alone may still be stopped by its PID
            turn.outside = other or turn.outside or (listing if origins else "")
        turn.private = turn.private or not PRIVATE_ORIGINS.isdisjoint(origins)
    marks = (data.get("summary_outside") is True, data.get("summary_private") is True)
    return turns, str(data.get("summary", "")), marks, float(data.get("saved_at", 0.0))


def _work(value: object) -> list[dict[str, Any]]:
    """Return a saved turn's work: its tool calls and results, with nothing a message lacks.

    Only assistant, tool and user messages with words are read, and only their own fields: a
    saved file cannot add a system message or a field the model server would act on.
    """
    work: list[dict[str, Any]] = []
    for message in objs(value):
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or role not in _ROLES or not isinstance(content, str):
            continue
        kept: dict[str, Any] = {"role": role, "content": content}
        if role == "tool":
            kept["tool_call_id"] = str(message.get("tool_call_id", ""))
        calls = objs(message.get("tool_calls")) if role == "assistant" else []
        if calls:
            kept["tool_calls"] = [_call(call) for call in calls]
        work.append(kept)
    return answered_calls(work)


def answered_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return `messages` up to the end of the last call whose results all came back.

    A model server may refuse a call without its result, or a result without its call.
    """
    end, waiting = 0, set[str]()
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            waiting = {str(call.get("id", "")) for call in objs(message.get("tool_calls"))}
        elif message.get("role") == "tool":
            if str(message.get("tool_call_id", "")) not in waiting:
                break
            waiting.discard(str(message.get("tool_call_id", "")))
        if not waiting:
            end = index + 1
    return messages[:end]


def _call(call: dict[str, Any]) -> dict[str, Any]:
    """Return a saved tool call with only its id, type, name and arguments."""
    function = obj(call.get("function"))
    return {
        "id": str(call.get("id", "")),
        "type": "function",
        "function": {
            "name": str(function.get("name", "")),
            "arguments": str(function.get("arguments", "")),
        },
    }


def _wrapped_origins(work: list[dict[str, Any]]) -> list[str]:
    """Return the origins of the results in `work` wrapped as outside data, first seen first."""
    found = (
        _WRAPPED_ORIGIN.findall(str(message["content"]))
        for message in work
        if message["role"] == "tool"
    )
    return list(dict.fromkeys(origin for origins in found for origin in origins))
