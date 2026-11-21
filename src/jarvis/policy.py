"""Boundaries enforced in code: typed operations, default deny, provenance, bound confirmation.

The decision table is plain data (`decide`), so tests cover it row by row. Text that did not come
from the owner's own speech never raises an operation's class, and while such text is in the
conversation every side-effecting operation needs at least spoken confirmation. While text read
from one of the owner's files is in the conversation, an operation that sends words the model
wrote to a service off this PC (a web search, a place name) needs that confirmation too, so the
file's contents cannot be carried out of the PC without the owner seeing it.
"""

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class OpClass(StrEnum):
    """What it takes for an operation to run."""

    SAFE = "safe"
    CONFIRM = "confirm"
    STRONG = "strong"
    DENY = "deny"


_RANK = {OpClass.SAFE: 0, OpClass.CONFIRM: 1, OpClass.STRONG: 2, OpClass.DENY: 3}


class Origin(StrEnum):
    """Where a message part came from."""

    SYSTEM = "system"
    OWNER_SPEECH = "owner_speech"
    PC_AUDIO = "pc_audio"
    MODEL = "model"
    TOOL_STATUS = "tool_status"
    SPOTIFY = "spotify"
    WEB = "web"
    FILE = "file"
    PICTURE = "picture"  # a picture the owner shared: what it shows was made by anyone


TRUSTED_ORIGINS = frozenset({Origin.SYSTEM, Origin.OWNER_SPEECH, Origin.TOOL_STATUS})


class Classified(Protocol):
    """What the policy reads from an operation."""

    @property
    def name(self) -> str:
        """The operation's name."""
        ...

    @property
    def default_class(self) -> OpClass:
        """The class it ships with."""
        ...

    @property
    def side_effect(self) -> bool:
        """Whether it changes anything."""
        ...

    @property
    def outbound(self) -> bool:
        """Whether it sends words the model wrote to a service off this PC."""
        ...


class Verdict(StrEnum):
    """The outcome of a policy decision."""

    RUN = "run"
    CONFIRM = "confirm"
    STRONG = "strong"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class Decision:
    """A verdict and the reason for it, as recorded in the audit log."""

    verdict: Verdict
    reason: str
    op_class: OpClass | None


def action_hash(op_name: str, args: Mapping[str, object]) -> str:
    """Return the identity of one exact action: the operation and its canonical arguments."""
    canonical = json.dumps({"op": op_name, "args": args}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def effective_class(op: Classified, overrides: Mapping[str, str]) -> OpClass:
    """Return the owner's class for the operation if set, else its shipped default."""
    override = overrides.get(op.name)
    return OpClass(override) if override in OpClass._value2member_map_ else op.default_class


def decide(
    op: Classified | None,
    name: str,
    overrides: Mapping[str, str],
    *,
    untrusted_in_context: bool,
    private_in_context: bool = False,
) -> Decision:
    """Decide whether a validated tool call may run, needs confirmation or is refused.

    `private_in_context` says text read from the owner's files is in the conversation.
    """
    if op is None:
        return Decision(Verdict.DENY, f"unknown operation {name!r}", None)
    cls = effective_class(op, overrides)
    below_confirm = _RANK[cls] < _RANK[OpClass.CONFIRM]
    if untrusted_in_context and op.side_effect and below_confirm:
        cls = OpClass.CONFIRM
        reason = "side effect while untrusted text is in the conversation"
    elif private_in_context and op.outbound and below_confirm:
        cls = OpClass.CONFIRM
        reason = "sends text off the PC while file text is in the conversation"
    else:
        reason = f"class {cls.value}"
    verdict = {
        OpClass.SAFE: Verdict.RUN,
        OpClass.CONFIRM: Verdict.CONFIRM,
        OpClass.STRONG: Verdict.STRONG,
        OpClass.DENY: Verdict.DENY,
    }[cls]
    return Decision(verdict, reason, cls)


_YES = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "ok",
        "okay",
        "confirm",
        "confirmed",
        "absolutely",
        "definitely",
        "do it",
        "go ahead",
        "go for it",
        "please do",
        "of course",
    }
)
_BESIDE_A_YES = frozenset({"please", "thanks", "thank", "you", "then", "and", "that"})
_ALLOWED = _BESIDE_A_YES | {word for phrase in _YES for word in phrase.split()}
_LONGEST_YES = 6


def _spoken_words(text: str) -> list[str]:
    return "".join(ch for ch in text.casefold() if ch.isalpha() or ch.isspace()).split()


def is_yes(transcript: str, names: Iterable[str] = ()) -> bool:
    """Return True only for a clear spoken yes; anything else (or silence) is no.

    An answer counts only when it holds a yes and nothing but words that go with one, so "yes"
    followed by any word this does not know ("yes, don't", "yeah, nah", "sure, not now") is a no.
    `names` are what the assistant is called: "yes, Jarvis" is still a yes.
    """
    called = {word for name in names for word in _spoken_words(name)}
    words = [word for word in _spoken_words(transcript) if word not in called]
    if not words or len(words) > _LONGEST_YES or not _ALLOWED.issuperset(words):
        return False
    phrase = f" {' '.join(words)} "
    return any(f" {yes} " in phrase for yes in _YES)
