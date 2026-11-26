"""Typed operations: a name, a JSON schema for its arguments, a class and an executor."""

import math
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from jarvis.policy import OpClass, Origin

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping

    from jarvis.shortcuts import Shortcut

MAX_STRING = 200
MAX_INTEGER = 2**31 - 1
# The agent's `<data origin="…">` and `</data>` around outside text, at the longest origin's name.
DATA_WRAP_CHARS = len('<data origin=""></data>') + max(len(origin.value) for origin in Origin)


def inert(text: str) -> str:
    """Return outside text with its angle brackets spelled out, so it cannot close its wrapper."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def inert_chars(text: str) -> int:
    """Return how long `text` is once `inert` spells out its angle brackets."""
    return len(text) + 3 * (text.count("<") + text.count(">"))


def cut_inert(text: str, room: int) -> str:
    """Return the longest start of `text` that `inert` keeps within `room` characters.

    A result read in parts is sized with this, so the agent's own cut at a tool result's cap
    never takes the line that says where the next part starts.
    """
    if inert_chars(text) <= room:
        return text
    used = 0
    for index, char in enumerate(text):
        used += 4 if char in "<>" else 1
        if used > room:
            return text[:index]
    return text


class UncertainOutcomeError(Exception):
    """An operation may or may not have taken effect (a timeout after the request was sent)."""


class ArgumentError(ValueError):
    """A tool call's arguments do not match the operation's schema."""


class OperationStoppedError(Exception):
    """An operation stopped part way because its request was cut short or silenced (`sing`).

    Nothing it would have done was done; the message says what was stopped.
    """


class OperationError(Exception):
    """An operation could not do what was asked; the message is what the model is told.

    `origin` says where the message's words come from: Jarvis's own status text unless the
    operation passes on text an outside service wrote.
    """

    def __init__(self, message: str, *, origin: Origin = Origin.TOOL_STATUS) -> None:
        """Keep the message and the origin of its words."""
        super().__init__(message)
        self.origin = origin


REMEMBERED = "memory"  # the origin of outside text a request remembers from earlier turns
REMEMBERED_LISTING = "memory-process"  # ... when all of it was process listings


@dataclass(frozen=True, eq=False, slots=True)
class Request:
    """The request the agent is handling, while it handles it.

    Each request is its own object, equal only to itself, so an operation can keep what it showed
    the owner in one request and refuse to act on it in another. `foreign` says the request was
    heard in PC audio, not from the owner. `outside` holds the origins of the outside text that
    came into the request so far (`"memory"` for outside text it remembers). `words` are the
    owner's own words of the request: none for one heard in PC audio or handed over with outside
    text behind it, and without the pastes that count as outside data; with the words the owner
    typed into it while it ran (`steered`). `time_left` says how many seconds the request has left
    before its time limit (`agent.max_seconds`), for an operation that waits as long as it is asked
    (`pc_audio_listen`). `cancelled` turns true when the request is cut short (the owner spoke over
    Jarvis, Jarvis was paused or is stopping), for an operation that runs long, and `hushed` when
    nothing more of it can be heard (`jarvis hush` too), for one that makes a sound (`sing`). An
    operation making a sound runs inside `making_sound()`, so the voice listens for the wake word
    meanwhile, as it does while it speaks. `scheduled` marks a request run when a job fell due: the
    owner did not ask for it just now. `message` counts the model's messages whose calls have run,
    so an operation can tell a result the model has read (an earlier message) from one written
    beside the call in the same message (`pc_window` acts only on a listing the model has seen).
    """

    foreign: bool
    outside: set[str] = field(default_factory=set[str])
    typed: str = ""
    steered: list[str] = field(default_factory=list[str])
    time_left: Callable[[], float] = field(default=lambda: math.inf)
    cancelled: Callable[[], bool] = field(default=lambda: False)
    hushed: Callable[[], bool] = field(default=lambda: False)
    making_sound: Callable[[], AbstractContextManager[None]] = field(default=nullcontext)
    scheduled: bool = False
    _messages: list[int] = field(default_factory=lambda: [0])  # held in a list: the rest is frozen

    @property
    def message(self) -> int:
        """How many of the model's messages have had their calls run, the running one included."""
        return self._messages[0]

    def next_message(self) -> None:
        """Note that the calls of the model's next message are about to run."""
        self._messages[0] += 1

    @property
    def words(self) -> str:
        """The owner's own words of the request, with those typed into it while it ran."""
        return "\n".join(part for part in (self.typed, *self.steered) if part)

    def saw(self, origin: str) -> None:
        """Note that text of `origin` came into the request."""
        self.outside.add(origin)

    def added(self, words: str) -> None:
        """Note words the owner typed into the request while it ran, as their own."""
        self.steered.append(words)


_CURRENT: ContextVar[Request | None] = ContextVar("jarvis_request", default=None)


def current_request() -> Request | None:
    """Return the request being handled, or None outside one."""
    return _CURRENT.get()


@contextmanager
def request_scope(  # ruff: ignore[too-many-arguments] - the request's fields, keyword-only
    *,
    foreign: bool,
    words: str = "",
    time_left: Callable[[], float] = lambda: math.inf,
    cancelled: Callable[[], bool] = lambda: False,
    scheduled: bool = False,
    hushed: Callable[[], bool] = lambda: False,
    making_sound: Callable[[], AbstractContextManager[None]] = nullcontext,
) -> Generator[Request]:
    """Mark everything run inside as part of one new request; `words` are the owner's own."""
    request = Request(
        foreign,
        typed=words,
        time_left=time_left,
        cancelled=cancelled,
        scheduled=scheduled,
        hushed=hushed,
        making_sound=making_sound,
    )
    token = _CURRENT.set(request)
    try:
        yield request
    finally:
        _CURRENT.reset(token)


class Lyric(NamedTuple):
    """A word a sound says, and the seconds from the sound's start it is heard in."""

    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class Sound:
    """A sound an operation made for the owner to hear, a song: int16 mono samples at `rate`.

    `lyrics` are the words it says and when, so the voice's check for its own wake word reads
    each stretch of it with the words heard there.
    """

    pcm: bytes
    rate: int
    lyrics: tuple[Lyric, ...] = ()

    @property
    def words(self) -> str:
        """The words it says, in order."""
        return " ".join(lyric.text for lyric in self.lyrics)


@dataclass(frozen=True, slots=True)
class Result:
    """What an operation returns to the model, with the origin of its text.

    `spoken` is a finished sentence for the owner. An operation sets it when its result needs no
    wording by the model, which lets a shortcut request be answered without the model. `picture`
    is a picture file the chat shows (with its `caption`), from Jarvis's own picture folder.
    `acted` marks a result that is not `ok` although the operation did act: a command ran but
    exited non-zero, a program was asked to stop but is still running, a program started but its
    window did not show where asked. A reply may then say what was done. `quiet` ends the
    request with nothing said: the owner asked for silence. `effort` asks the agent to let the
    model think that hard for the rest of the request (`think_harder`); the agent decides.
    `sound` is played to the owner in the reply, after the words spoken before it; a channel
    that cannot play sound turns the result into a failure the model is told about.
    `mode` asks the agent to go on in that mode, on that mode's model (`switch_model`).
    """

    text: str
    origin: Origin = Origin.TOOL_STATUS
    ok: bool = True
    spoken: str = ""
    picture: str = ""
    caption: str = ""
    acted: bool = False
    quiet: bool = False
    effort: str = ""
    sound: Sound | None = None
    mode: str = ""


@dataclass(frozen=True, slots=True)
class Operation:
    """One thing Jarvis can do. `parameters` is a JSON schema object with only flat properties.

    `prepare` runs on the validated arguments once the policy has not refused the call, before
    any confirmation question. It returns the arguments the call will run with, so an operation
    that acts on "whatever is current" can bind the exact target the owner is asked about.
    `shortcuts` are the phrasings that mean this operation without the model's help.
    `outbound` marks an operation that sends words the model wrote (a search, a place) to a
    service off this PC; while file text is in the conversation it needs a yes. With
    `outbound_args`, only a call that gives one of those arguments sends any (`sends_words`).
    `preview` returns the start of the text a confirmed call would write, which a window shows
    under the question (speech says only how much). `hashed` names the arguments the audit log
    keeps only as a SHA-256 and a length, never as text.
    `lowest_class` is as low as the owner's `boundaries.classes` may set it. With
    `audit_text=False` the audit log keeps a result's text only as a SHA-256 and a length (a
    shell command's output can hold anything the owner's user can read).
    `concurrent` lets the agent run the call beside the other `concurrent` calls of the same
    model message; only a read-only operation whose `run` is safe in several threads at once
    may set it. With `always_ask`, a call that needs a yes is asked about even while the owner
    has consent bypassed (`boundaries.bypass_consent`): it can send the owner's words to whoever
    wrote a page, so a page must not be able to make it happen unseen.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any]], Result]
    default_class: OpClass = OpClass.SAFE
    lowest_class: OpClass = OpClass.SAFE
    side_effect: bool = True
    outbound: bool = False
    outbound_args: frozenset[str] = frozenset()
    spoken_summary: Callable[[dict[str, Any]], str] = field(default=lambda _args: "do that")
    prepare: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    shortcuts: tuple[Shortcut, ...] = ()
    preview: Callable[[dict[str, Any]], str] | None = None
    hashed: frozenset[str] = frozenset()
    audit_text: bool = True
    concurrent: bool = False
    always_ask: bool = False

    def __post_init__(self) -> None:
        """Refuse an operation that acts yet claims to run beside others.

        Raises:
            ValueError: `concurrent` is set on an operation with a side effect.
        """
        if self.concurrent and self.side_effect:
            msg = f"{self.name}: only a read-only operation may run beside others"
            raise ValueError(msg)

    def sends_words(self, args: Mapping[str, Any]) -> bool:
        """Whether this call sends words the model wrote off the PC.

        An `outbound` operation with `outbound_args` sends them only in those arguments (a place
        for the weather): a call without any of them sends nothing the model wrote.
        """
        if not self.outbound:
            return False
        return not self.outbound_args or any(
            str(args.get(name) or "").strip() for name in self.outbound_args
        )

    def tool_schema(self) -> dict[str, Any]:
        """Return the OpenAI tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _check_string(name: str, spec: dict[str, Any], value: object) -> str:
    if not isinstance(value, str):
        raise ArgumentError(f"{name} must be a string")
    text = value.strip()
    least = spec.get("minLength", 0)
    if not text and least > 0:
        raise ArgumentError(f"{name} must not be empty")
    if len(text) < least:
        raise ArgumentError(f"{name} must be at least {least} characters")
    if len(text) > spec.get("maxLength", MAX_STRING):
        raise ArgumentError(f"{name} is too long")
    if "enum" in spec and text not in spec["enum"]:
        raise ArgumentError(f"{name} must be one of {spec['enum']}")
    return text


def _check_integer(name: str, spec: dict[str, Any], value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArgumentError(f"{name} must be an integer")
    if isinstance(value, float) and not (math.isfinite(value) and value.is_integer()):
        raise ArgumentError(f"{name} must be an integer")
    number = int(value)
    low, high = spec.get("minimum", -MAX_INTEGER), spec.get("maximum", MAX_INTEGER)
    if not low <= number <= high:
        raise ArgumentError(f"{name} must be between {low} and {high}")
    return number


def _check_value(name: str, spec: dict[str, Any], value: object) -> object:
    kind = spec.get("type")
    if kind == "string":
        return _check_string(name, spec, value)
    if kind == "integer":
        return _check_integer(name, spec, value)
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ArgumentError(f"{name} must be true or false")
        return value
    raise ArgumentError(f"{name} has an unsupported type")


def validate_args(op: Operation, args: object) -> dict[str, Any]:
    """Return the arguments cleaned against the operation's schema, with its defaults filled in.

    Raises:
        ArgumentError: the arguments are not an object, miss a required key, have an unknown key or
            a value of the wrong type or range.
    """
    if not isinstance(args, dict):
        raise ArgumentError("arguments must be a JSON object")
    props: dict[str, dict[str, Any]] = op.parameters.get("properties", {})
    required: list[str] = op.parameters.get("required", [])
    cleaned: dict[str, Any] = {}
    for key, value in cast("dict[str, object]", args).items():
        if key not in props:
            raise ArgumentError(f"unknown argument {key!r}")
        cleaned[key] = _check_value(key, props[key], value)
    missing = [key for key in required if key not in cleaned]
    if missing:
        raise ArgumentError(f"missing argument(s): {', '.join(missing)}")
    for key, spec in props.items():
        if key not in cleaned and "default" in spec:
            cleaned[key] = spec["default"]
    return cleaned


class Registry:
    """The operations offered to the model; anything else is refused."""

    def __init__(
        self, ops: Iterable[Operation], *, resets: Iterable[Callable[[], None]] = ()
    ) -> None:
        """Index the operations by name; `resets` clear what they remember of a conversation."""
        super().__init__()
        self._ops = {op.name: op for op in ops}
        self._resets = tuple(resets)

    def reset(self) -> None:
        """Make the operations forget what they keep from the conversation (search results)."""
        for reset in self._resets:
            reset()

    def get(self, name: str) -> Operation | None:
        """Return the operation or None."""
        return self._ops.get(name)

    def without(self, names: Iterable[str]) -> Registry:
        """Return the same registry without the named operations, which are then not offered."""
        dropped = set(names)
        kept = [op for name, op in self._ops.items() if name not in dropped]
        return Registry(kept, resets=self._resets)

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return every operation as an OpenAI tool."""
        return [op.tool_schema() for op in self._ops.values()]

    def names(self) -> list[str]:
        """Return the operation names."""
        return list(self._ops)

    def shortcuts(self) -> list[tuple[str, Shortcut]]:
        """Return every shortcut phrasing with the name of the operation it means."""
        return [(op.name, shortcut) for op in self._ops.values() for shortcut in op.shortcuts]
