"""Where a sentence of a reply ends, so speech can start before the whole reply is written.

A sentence ends at `.`, `!` or `?` followed by white space, or at a line break. Every end found
here is also a sentence end for `jarvis.agent.claims_action`, so a reply cut at these points is
checked the same in pieces as whole. An abbreviation ("Dr.", "e.g.", "a.m."), an initial ("J.") or
a list number ("1.") does not end a sentence, except "a.m." and "p.m." at the end of a line.
"""

import re

MAX_SPOKEN_CHARS = 2000

_END = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_BEFORE = re.compile(r"(\S+)$")
_ABBREVIATIONS = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "st.",
        "sr.",
        "jr.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
        "approx.",
        "no.",
        "fig.",
        "ft.",
        "mt.",
        "inc.",
        "ltd.",
        "co.",
        "a.m.",
        "p.m.",
    }
)
_TIMES = frozenset({"a.m.", "p.m."})  # a line may end on them


def _real_end(text: str, end: int, *, line_end: bool) -> bool:
    """Return False when the full stop before `end` belongs to an abbreviation or a number.

    `line_end` says a line break follows, where "a.m." and "p.m." do end the sentence.
    """
    if text[end - 1] != ".":
        return True
    found = _WORD_BEFORE.search(text, 0, end)
    word = found.group(1).lower() if found else ""
    word = word.lstrip("(\"'")
    if word in _ABBREVIATIONS:
        return line_end and word in _TIMES
    stem = word.rstrip(".")
    return not (stem.isdigit() or (len(stem) == 1 and stem.isalpha()))


def sentence_end(text: str, start: int = 0) -> int | None:
    """Return the index just past the first complete sentence at or after `start`, or None.

    The index points at the white space that follows the sentence, so `text[start:end]` is the
    sentence with no trailing space. A sentence is complete only once the white space after it
    has arrived, so a reply still being written never splits "3." from ".5".
    """
    for match in _END.finditer(text, start):
        end = len(
            text[: match.start()].rstrip()
        )  # a line break after spaces: they are not the sentence's
        if (
            end > start
            and text[start:end].strip()
            and _real_end(text, end, line_end="\n" in match.group())
        ):
            return end
    return None


def split_sentences(text: str) -> list[str]:
    """Return `text` as its sentences, each without surrounding white space."""
    pieces: list[str] = []
    start = 0
    while (end := sentence_end(text, start)) is not None:
        pieces.append(text[start:end].strip())
        start = end
    rest = text[start:].strip()
    return [*pieces, rest] if rest else pieces


MAX_SAME_SENTENCE = (
    2  # a sentence said this often is kept ("Knock knock. Knock knock."); more is a loop
)
MIN_LOOP_CHARS = 8  # shorter sentences ("Ha.", "No.") may come back on purpose


def looped_at(text: str) -> int | None:
    """Return where a sentence comes back once more than `MAX_SAME_SENTENCE` times, or None.

    A small model can fall into writing one sentence again and again until its reply limit;
    the reply is cut before the first sentence that repeats past the limit (#70).
    """
    counts: dict[str, int] = {}
    start, whole = 0, f"{text} "
    while (end := sentence_end(whole, start)) is not None:
        key = " ".join(whole[start:end].split()).casefold()
        if len(key) >= MIN_LOOP_CHARS:
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > MAX_SAME_SENTENCE:
                return len(whole[:start].rstrip())
        start = end
    return None
