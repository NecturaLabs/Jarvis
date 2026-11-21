"""Requests simple enough to run without the model: "pause", "next song", "volume 40".

An operation may list the exact phrasings that mean it (`Shortcut`). A request that is nothing but
one of them goes straight to that operation, through the same policy, confirmation and audit as a
call the model proposes, and skips the model: no load, no thinking time. Anything else, however
close, is left to the model, so a shortcut can only ever save time, never guess.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

MAX_REQUEST_CHARS = 80
_OPENERS = re.compile(
    r"^(?:(?:hey |ok |okay )?(?:please |now |just )*"
    r"(?:(?:can|could|would|will) you (?:please |just )?)?)"
)
_CLOSERS = re.compile(r"(?: please| now| for me| thanks| thank you)+$")
_KEPT = re.compile(r"[^a-z0-9% ]+")


def normalise(text: str, names: Iterable[str] = ()) -> str:
    """Return a request as bare lowercase words: no punctuation, politeness or assistant's name."""
    words = _KEPT.sub(" ", text.casefold().replace("'", "").replace("\u2019", ""))
    words = " ".join(words.split())
    for name in names:
        bare = " ".join(_KEPT.sub(" ", name.casefold()).split())
        if bare:
            words = re.sub(rf"^(?:hey |ok |okay )?{re.escape(bare)} ", "", words)
            words = re.sub(rf" {re.escape(bare)}$", "", words)
    for _ in range(2):
        words = _CLOSERS.sub("", _OPENERS.sub("", words)).strip()
    return words


@dataclass(frozen=True, slots=True)
class Shortcut:
    """Phrasings that mean one operation call.

    `pattern` is a regular expression over the normalised request and must match all of it.
    `arguments` builds the call's arguments from the match (named groups arrive as strings).
    """

    pattern: str
    arguments: Callable[[dict[str, str]], dict[str, Any]] = field(default=lambda _groups: {})


@dataclass(frozen=True, slots=True)
class Matched:
    """The operation a request maps to, with its arguments (not yet validated)."""

    operation: str
    arguments: dict[str, Any]


def match(
    text: str, shortcuts: Iterable[tuple[str, Shortcut]], names: Iterable[str] = ()
) -> Matched | None:
    """Return the one operation `text` plainly asks for, or None to leave it to the model.

    `shortcuts` pairs operation names with their phrasings. A request that fits two different
    calls is ambiguous and matches nothing.
    """
    if len(text) > MAX_REQUEST_CHARS:
        return None
    words = normalise(text, names)
    if not words:
        return None
    found: list[Matched] = []
    for operation, shortcut in shortcuts:
        hit = re.fullmatch(shortcut.pattern, words)
        if hit is None:
            continue
        groups = {k: v for k, v in hit.groupdict().items() if v is not None}
        candidate = Matched(operation, shortcut.arguments(groups))
        if candidate not in found:
            found.append(candidate)
    return found[0] if len(found) == 1 else None
