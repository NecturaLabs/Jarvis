"""Boundaries enforced in code: typed operations, default deny, provenance, bound confirmation.

The decision table is plain data (`decide`), so tests cover it row by row. Text that did not come
from the owner's own speech never raises an operation's class, and while such text is in the
conversation every side-effecting operation needs at least spoken confirmation.
"""

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping


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
) -> Decision:
    """Decide whether a validated tool call may run, needs confirmation or is refused."""
    if op is None:
        return Decision(Verdict.DENY, f"unknown operation {name!r}", None)
    cls = effective_class(op, overrides)
    if untrusted_in_context and op.side_effect and _RANK[cls] < _RANK[OpClass.CONFIRM]:
        cls = OpClass.CONFIRM
        reason = "side effect while untrusted text is in the conversation"
    else:
        reason = f"class {cls.value}"
    verdict = {
        OpClass.SAFE: Verdict.RUN,
        OpClass.CONFIRM: Verdict.CONFIRM,
        OpClass.STRONG: Verdict.STRONG,
        OpClass.DENY: Verdict.DENY,
    }[cls]
    return Decision(verdict, reason, cls)


YES_WORDS = frozenset(
    {"yes", "yeah", "yep", "yup", "sure", "confirm", "confirmed", "do it", "go ahead", "please do"}
)


_SHORT_ANSWER_WORDS = 3


def is_yes(transcript: str) -> bool:
    """Return True only for a clear spoken yes; anything else (or silence) is no."""
    words = "".join(ch for ch in transcript.lower() if ch.isalpha() or ch.isspace()).split()
    phrase = " ".join(words)
    return bool(phrase) and (
        phrase in YES_WORDS
        or (len(words) <= _SHORT_ANSWER_WORDS and words[0] in YES_WORDS and "no" not in words)
    )
