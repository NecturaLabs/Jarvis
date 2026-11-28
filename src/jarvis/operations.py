"""Typed operations: a name, a JSON schema for its arguments, a class and an executor."""

import math
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from jarvis.jsonutil import unbroken
from jarvis.policy import OpClass, Origin
from jarvis.toolgroups import CORE_NAME, GROUPS, ToolGroup, group_name

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Generator, Iterable, Mapping

    from jarvis.pictures import Picture
    from jarvis.shortcuts import Shortcut

MAX_STRING = 200
MAX_INTEGER = 2**31 - 1
# The agent's `<data origin="…">` and `</data>` around outside text, at the longest origin's name.
DATA_WRAP_CHARS = len('<data origin=""></data>') + max(len(origin.value) for origin in Origin)


def inert(text: str) -> str:
    """Return outside text with its angle brackets spelled out, so it cannot close its wrapper."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def one_line(text: str) -> str:
    """Return model-written text for a window: one line, each control character a space.

    A terminal acts on escape sequences, so words the model wrote for the owner to read carry none.
    """
    return " ".join("".join(ch if ch.isprintable() else " " for ch in text).split())


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


UNASKED_ARG = "unasked"  # the argument that hands an operation its own `unasked` reason back

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
    `noted` says which note began the request ("answer", "timed_out", "closed"), none otherwise:
    `owner_asked` is what the operations that act only on the owner's own request check.
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
    noted: str = ""
    # Held in lists: the rest is frozen. Messages run, calls in the running message, actions tried.
    _messages: list[int] = field(default_factory=lambda: [0, 1, 0])
    _waited: list[float] = field(default_factory=lambda: [0.0])

    @property
    def owner_asked(self) -> bool:
        """Whether the owner asked for this request just now: not PC audio, not begun by a note.

        A request begun by the note that a question went unanswered (`noted` "timed_out" or
        "closed") has no words of the owner's: Jarvis goes on by itself. One begun by the owner's
        pick (`noted` "answer") is their choice, so it counts.
        """
        return not self.foreign and self.noted not in {"timed_out", "closed"}

    @property
    def message(self) -> int:
        """How many of the model's messages have had their calls run, the running one included."""
        return self._messages[0]

    @property
    def beside(self) -> int:
        """How many calls the running message holds, this one included."""
        return self._messages[1]

    @property
    def acted(self) -> bool:
        """Whether a call that could change something was already tried in this request."""
        return self._messages[2] > 0

    def next_message(self, calls: int = 1) -> None:
        """Note that the `calls` of the model's next message are about to run."""
        self._messages[0] += 1
        self._messages[1] = calls

    def note_action(self) -> None:
        """Note that a call that could change something was tried."""
        self._messages[2] += 1

    def waited(self, seconds: float) -> None:
        """Note that a call waited `seconds` for the owner, time the request's limit leaves out."""
        self._waited[0] += max(0.0, seconds)

    def take_waited(self) -> float:
        """Return the seconds waited for the owner since the last call, and start counting again."""
        seconds, self._waited[0] = self._waited[0], 0.0
        return seconds

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


@dataclass(frozen=True, slots=True)
class Notice:
    """Words that join a request in progress and are not the owner's own typed words.

    An answer the owner picked from options the model wrote, or Jarvis's own note that a question
    went unanswered. `origin` says whose words they are (Jarvis's own status text unless it
    passes on outside text) and `kind` names what happened, for the audit log. The agent shows
    them to the model but never counts them as the owner's words (`Request.words`), except `words`:
    what the owner typed under Other, which is theirs.
    """

    text: str
    kind: str
    origin: Origin = Origin.TOOL_STATUS
    words: str = ""


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
    noted: str = "",
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
        noted=noted,
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
    each stretch of it with the words heard there. It plays `times` times over, back to back.
    """

    pcm: bytes
    rate: int
    lyrics: tuple[Lyric, ...] = ()
    times: int = 1

    @property
    def words(self) -> str:
        """The words it says, in order."""
        return " ".join(lyric.text for lyric in self.lyrics)


@dataclass(frozen=True, slots=True)
class Result:
    """What an operation returns to the model, with the origin of its text.

    `spoken` is a finished sentence for the owner. An operation sets it when its result needs no
    wording by the model, which lets a shortcut request be answered without the model; `written`
    is that answer laid out in Markdown, for a window that shows the reply instead. `picture`
    is a picture file the chat shows (with its `caption`), from Jarvis's own picture folder.
    `acted` marks a result that is not `ok` although the operation did act: a command ran but
    exited non-zero, a program was asked to stop but is still running, a program started but its
    window did not show where asked. A reply may then say what was done. `quiet` ends the
    request with nothing said: the owner asked for silence. `effort` asks the agent to let the
    model think that hard for the rest of the request (`think_harder`); the agent decides.
    `sound` is played to the owner in the reply, after the words spoken before it; a channel
    that cannot play sound turns the result into a failure the model is told about.
    `mode` asks the agent to go on in that mode, on that mode's model (`switch_model`).
    `groups` asks the agent to offer those tool groups for the rest of the request (`use_tools`).
    `look` is a picture the model is to see next (a screenshot just taken), when it can see
    pictures; the agent adds it to the conversation as outside data, and says in the result's
    text whether it did. `delivered` is told the result as the model got it, cut to the agent's
    cap (and wrapped, for outside text), so an operation can hold to what was really read.
    `ends` closes the request with `spoken` as its reply, without another model call, when
    the call ran alone in its message (`chat_hand_off`: the work went to the chat).
    """

    text: str
    origin: Origin = Origin.TOOL_STATUS
    ok: bool = True
    spoken: str = ""
    written: str = ""
    picture: str = ""
    caption: str = ""
    acted: bool = False
    quiet: bool = False
    effort: str = ""
    sound: Sound | None = None
    mode: str = ""
    ends: bool = False
    groups: tuple[str, ...] = ()
    look: Picture | None = None
    delivered: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        """Keep no lone surrogate (a non-UTF-8 file name): the text is sent, saved and logged."""
        for name in ("text", "spoken", "written", "caption"):
            value = getattr(self, name)
            if (clean := unbroken(value)) != value:
                object.__setattr__(self, name, clean)


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
    keeps only as a SHA-256 and a length, never as text. `verbatim` names the string arguments
    kept exactly as written, not trimmed: file text, whose leading spaces and line breaks matter.
    `lowest_class` is as low as the owner's `boundaries.classes` may set it. With
    `audit_text=False` the audit log keeps a result's text only as a SHA-256 and a length (a
    shell command's output can hold anything the owner's user can read).
    `concurrent` lets the agent run the call beside the other `concurrent` calls of the same
    model message; only a read-only operation whose `run` is safe in several threads at once
    may set it. With `always_ask`, a call that needs a yes is asked about even while the owner
    has consent bypassed (`boundaries.bypass_consent`): it can send the owner's words to whoever
    wrote a page, so a page must not be able to make it happen unseen.
    `offered` says whether the model is offered the operation now (`chat_hand_off`, only while a
    chat window is open); without it the operation is always offered.
    `unasked` says, for one call's validated arguments, why that call needs no question although
    the operation is `confirm` ("a new file in the workspace"), or "" to ask as usual. The owner's
    own class for the operation, and outside text in the conversation, still win over it.
    A call that runs on that reason gets it back in its arguments (`UNASKED_ARG`) for `prepare`
    and `run`, which must then do only what the reason said (a "new file" fails if one appeared).
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
    verbatim: frozenset[str] = frozenset()
    audit_text: bool = True
    concurrent: bool = False
    always_ask: bool = False
    offered: Callable[[], bool] | None = None
    unasked: Callable[[dict[str, Any]], str] | None = None

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


def _check_string(name: str, spec: dict[str, Any], value: object, *, verbatim: bool = False) -> str:
    if not isinstance(value, str):
        raise ArgumentError(f"{name} must be a string")
    text = value if verbatim else value.strip()
    least = spec.get("minLength", 0)
    if not text.strip() and least > 0:
        raise ArgumentError(f"{name} must not be empty")
    if len(text) < least:
        raise ArgumentError(f"{name} must be at least {least} characters")
    if len(text) > (most := spec.get("maxLength", MAX_STRING)):
        raise ArgumentError(f"{name} is too long: at most {most} characters")
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


def _check_array(name: str, spec: dict[str, Any], value: object) -> list[object]:
    # A list of scalars or flat objects, each checked against `items`, of at most `maxItems` (10).
    if not isinstance(value, list):
        raise ArgumentError(f"{name} must be a list")
    items = cast("list[object]", value)
    if len(items) < spec.get("minItems", 0):
        raise ArgumentError(f"{name} needs at least {spec['minItems']} item(s)")
    if len(items) > spec.get("maxItems", 10):
        raise ArgumentError(f"{name} has too many items")
    item_spec: dict[str, Any] = spec.get("items", {})
    if item_spec.get("type") not in {"string", "integer", "boolean", "object"}:
        raise ArgumentError(f"{name} has an unsupported type")
    return [_check_value(name, item_spec, item) for item in items]


def _check_object(name: str, spec: dict[str, Any], value: object) -> dict[str, object]:
    # An item of an array: only the keys its `properties` name, `required` ones present.
    if not isinstance(value, dict):
        raise ArgumentError(f"{name} must be an object")
    props: dict[str, dict[str, Any]] = spec.get("properties", {})
    cleaned: dict[str, object] = {}
    for key, item in cast("dict[str, object]", value).items():
        if key not in props:
            raise ArgumentError(f"{name} has an unknown key {key!r}")
        cleaned[key] = _check_value(f"{name}.{key}", props[key], item)
    missing = [key for key in spec.get("required", []) if key not in cleaned]
    if missing:
        raise ArgumentError(f"{name} is missing {', '.join(missing)}")
    return cleaned


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
    if kind == "array":
        return _check_array(name, spec, value)
    if kind == "object":
        return _check_object(name, spec, value)
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
        if key in op.verbatim and props[key].get("type") == "string":
            cleaned[key] = _check_string(key, props[key], value, verbatim=True)
        else:
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
        self._by_group: dict[str, list[Operation]] = {}
        for op in self._ops.values():
            if (group := group_name(op.name)) is not None:
                self._by_group.setdefault(group, []).append(op)

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

    def tool_schemas(
        self, loaded: Collection[str] | None = None, *, every: bool = False
    ) -> list[dict[str, Any]]:
        """Return the operations offered now as OpenAI tools: all, or the core and `loaded` groups.

        Given `loaded`, the order is fixed whatever order the groups were loaded in: the core
        tools, then each loaded group's in the order the groups first appear in this registry, so
        calls with the same groups share their whole start. An operation no group names is never
        offered that way (a test makes sure there is none). An operation whose `offered` says no
        is left out, unless `every` asks for all of them (what the context budget counts).
        """
        if loaded is None:
            ops = list(self._ops.values())
        else:
            wanted = [CORE_NAME, *(g.name for g in self.groups() if g.name in loaded)]
            ops = [op for name in wanted for op in self._by_group.get(name, [])]
        return [op.tool_schema() for op in ops if every or op.offered is None or op.offered()]

    def groups(self) -> list[ToolGroup]:
        """Return the tool groups this registry has, apart from the core, as it has them.

        Each lists only the operations present here, in order, and the groups come in the order
        their first operation appears: the canonical order the tools are offered in.
        """
        declared = {group.name: group for group in GROUPS}
        return [
            ToolGroup(name, declared[name].summary, tuple(op.name for op in ops))
            for name, ops in self._by_group.items()
            if name in declared
        ]

    def core_names(self) -> list[str]:
        """Return the names of the core tools this registry has, which every request is offered."""
        return [op.name for op in self._by_group.get(CORE_NAME, [])]

    def ungrouped(self) -> list[str]:
        """Return the operations no tool group names."""
        return [name for name in self._ops if group_name(name) is None]

    def names(self) -> list[str]:
        """Return the operation names."""
        return list(self._ops)

    def shortcuts(self) -> list[tuple[str, Shortcut]]:
        """Return every shortcut phrasing with the name of the operation it means."""
        return [(op.name, shortcut) for op in self._ops.values() for shortcut in op.shortcuts]
