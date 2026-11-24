"""Measure speech-to-text on your own voice: record a private set of requests, then score them.

`record` shows one card at a time ("Ask Jarvis to play music by …", "Ask the time somewhere"),
you say it in your own words, and the capture is saved with what you said, which you confirm or
type. The capture is Jarvis's own: the same microphone, voice detection and lead-in as a request.
`score` transcribes every saved request again and reports the word error rate, the names heard
right and the time taken; `compare` says whether two scores differ beyond chance, pairing them
request by request (an exact McNemar test on the names, a sign test on the words).

Recordings stay in the folder chosen, readable by its owner only, and are never sent anywhere.
"""

import itertools
import json
import math
import os
import re
import statistics
import time
import wave
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from jarvis.audio import RATE, AudioError, write_wav
from jarvis.levels import levels

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    from numpy.typing import NDArray

MANIFEST = "set.jsonl"
SKIP, STOP = "s", "q"
# What the owner is asked to say. A card with a name asks for it; one with `asks_name` has the
# owner type the name they chose, so it can be checked in what was heard.
_ARTIST = "Ask Jarvis to play music by {name}, in your own words."
_GENERAL = (
    ("other-artist", "Ask Jarvis to play an artist or song you rarely listen to.", True),
    ("time", "Ask the time in a city, state or country of your choice.", True),
    ("weather", "Ask about the weather, here or somewhere else.", False),
    ("player", "Pause, skip, change the volume or ask what is playing, in your own words.", False),
    ("quiet", "Tell Jarvis to be quiet, in your own words.", False),
    ("free", "Ask Jarvis anything you would normally ask it.", False),
)
_ARTIST_SHARE = 0.5  # half the cards name one of your artists, when you have enough of them
_CHANCE = "A p under 0.05 is a difference beyond chance; above it, the set cannot tell them apart."
_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC


@dataclass(frozen=True, slots=True)
class Card:
    """One thing to say: its kind, the prompt shown, and the name it is about (if any)."""

    kind: str
    prompt: str
    name: str = ""
    asks_name: bool = False


def cards(artists: list[str], count: int, rng: random.Random) -> list[Card]:
    """Return `count` cards: about half name one of `artists`, the rest are open requests."""
    names = list(dict.fromkeys(_printable(name) for name in artists if _printable(name)))
    named = min(round(count * _ARTIST_SHARE), len(names))
    out = [Card("artist", _ARTIST.format(name=name), name) for name in rng.sample(names, named)]
    for kind, prompt, asks_name in (rng.choice(_GENERAL) for _ in range(count - named)):
        out.append(Card(kind, prompt, asks_name=asks_name))
    rng.shuffle(out)
    return out


class FolderError(Exception):
    """The folder given holds something other than a speech test set."""


class ScoreError(ValueError):
    """A score file does not hold the rows `score` saves."""


_ROW_KEYS = frozenset({"id", "kind", "music", "errors", "words", "name_heard", "seconds"})


def _rows(score: object) -> list[dict[str, Any]]:
    if not isinstance(score, list) or not all(
        isinstance(row, dict) and row.keys() >= _ROW_KEYS for row in cast("list[object]", score)
    ):
        msg = "not a score saved by `jarvis speech-test score --out`"
        raise ScoreError(msg)
    return cast("list[dict[str, Any]]", score)


def _printable(name: str) -> str:
    """Return a name from Spotify with what a terminal would act on left out."""
    return " ".join("".join(ch for ch in name if ch.isprintable()).split())


def _set_folder(folder: Path) -> None:
    """Make the set's folder (readable by its owner only), or check an existing one is a set.

    An existing folder is left as it is: only an empty one or one holding a set is used.

    Raises:
        FolderError: the folder holds other things.
    """
    try:
        folder.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        ours = (folder / MANIFEST).is_file() or all(
            path.suffix == ".wav" and path.stem.isdigit() for path in folder.iterdir()
        )
        if not ours:
            msg = f"{folder} is not a speech test set; give an empty or a new folder"
            raise FolderError(msg) from None


def _next_id(folder: Path) -> int:
    """Return the next free number: after every saved request and every recording left behind."""
    saved = (int(entry["id"]) for entry in read_set(folder))
    left = (int(path.stem) for path in folder.glob("*.wav") if path.stem.isdigit())
    return 1 + max(itertools.chain(saved, left), default=0)


def read_set(folder: Path) -> list[dict[str, Any]]:
    """Return the saved requests in `folder`, oldest first (none when there is no set yet)."""
    try:
        lines = (folder / MANIFEST).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines if line.strip()]


def read_wav(path: Path) -> NDArray[np.int16]:
    """Return a saved request's samples (16 kHz mono 16-bit, as `record` writes them).

    Raises:
        ValueError: the file holds some other format.
    """
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, RATE):
            msg = f"{path.name} is not 16 kHz mono 16-bit audio"
            raise ValueError(msg)
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.int16)


def _save(folder: Path, entry: dict[str, Any], samples: NDArray[np.int16]) -> None:
    wav = folder / f"{entry['id']}.wav"
    with os.fdopen(os.open(wav, _FLAGS | os.O_EXCL, 0o600), "wb") as handle:
        write_wav(handle, samples)
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with os.fdopen(os.open(folder / MANIFEST, _FLAGS | os.O_APPEND, 0o600), "a") as handle:
        handle.write(line)


def record(  # ruff: ignore[too-many-arguments] - the capture and the terminal are injected
    folder: Path,
    to_say: Iterable[Card],
    listen: Callable[[], NDArray[np.int16] | None],
    transcribe: Callable[[NDArray[np.int16]], str],
    *,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
    music: bool = False,
    heard_with: str = "",
) -> int:
    """Record one request per card into `folder` until the cards run out or the owner stops.

    Each request is saved with what the owner said: what was heard, confirmed, or the words they
    typed instead, and `heard_with`, the transcriber that heard it first (a score by that one
    starts from its own words). The end of input stops the session like `q`. Returns how many
    were saved. A `folder` holding something other than a set raises `FolderError`.
    """
    _set_folder(folder)
    saved = 0
    try:
        for _entry in _session(
            folder,
            to_say,
            listen,
            transcribe,
            ask=ask,
            show=show,
            music=music,
            heard_with=heard_with,
        ):
            saved += 1
    except EOFError:
        show("")
    return saved


def _session(  # ruff: ignore[too-many-arguments] - the parts of one session
    folder: Path,
    to_say: Iterable[Card],
    listen: Callable[[], NDArray[np.int16] | None],
    transcribe: Callable[[NDArray[np.int16]], str],
    *,
    ask: Callable[[str], str],
    show: Callable[[str], None],
    music: bool,
    heard_with: str,
) -> Iterator[dict[str, Any]]:
    """Yield each request saved, one card at a time, until the cards run out or `q`."""
    number = _next_id(folder)
    for card in to_say:
        show(f"\n[{number:04d}] {card.prompt}")
        start = ask("  Press Enter, then speak (s to skip, q to stop): ").strip().lower()
        if start == STOP:
            return
        if start == SKIP:
            continue
        samples = listen()
        if samples is None:
            show("  Nothing was heard; moving on.")
            continue
        try:
            heard = transcribe(samples)
        except AudioError as exc:
            heard = ""
            show(f"  Speech-to-text failed ({exc}); type what you said.")
        show(f"  Heard: {heard!r}")
        answer = ask("  Enter if that is what you said, or type what you said (s to skip): ")
        if answer.strip().lower() == SKIP or not (answer.strip() or heard):
            continue
        name = card.name
        if card.asks_name:
            name = ask("  Which name or place did you say? ").strip()
        entry = {
            "id": f"{number:04d}",
            "kind": card.kind,
            "prompt": card.prompt,
            "name": name,
            "said": answer.strip() or heard,
            "heard_with": heard_with,
            "music": music,
            "seconds": round(len(samples) / RATE, 2),
            "levels": asdict(levels(samples)),
        }
        _save(folder, entry, samples)
        number += 1
        yield entry


def words(text: str) -> list[str]:
    """Return a transcript's words, ignoring case and punctuation."""
    return re.findall(r"[\w']+", text.casefold())


def word_errors(said: list[str], heard: list[str]) -> int:
    """Return the fewest words substituted, dropped or added to turn `said` into `heard`."""
    row = list(range(len(heard) + 1))
    for i, word in enumerate(said, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(heard, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (word != other))
    return row[-1]


def _letters(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def name_heard(name: str, heard: str) -> bool:
    """Whether `name` is in `heard` as whole words, spaced or run together ("glassharbor")."""
    wanted = _letters(name)
    said = [_letters(word) for word in words(heard)]
    return any(
        "".join(said[start:end]) == wanted
        for start in range(len(said))
        for end in range(start + 1, len(said) + 1)
    )


def score(
    folder: Path, transcribe: Callable[[NDArray[np.int16]], str], show: Callable[[str], None]
) -> list[dict[str, Any]]:
    """Transcribe every saved request again; return one result per request."""
    results: list[dict[str, Any]] = []
    entries = read_set(folder)
    for count, entry in enumerate(entries, 1):
        samples = read_wav(folder / f"{entry['id']}.wav")
        started = time.perf_counter()
        heard = transcribe(samples)
        seconds = time.perf_counter() - started
        said = words(entry["said"])
        name = str(entry.get("name", ""))
        results.append(
            {
                "id": entry["id"],
                "kind": entry["kind"],
                "music": entry.get("music", False),
                "heard": heard,
                "errors": word_errors(said, words(heard)),
                "words": len(said),
                "name_heard": name_heard(name, heard) if _letters(name) else None,
                "seconds": round(seconds, 3),
            }
        )
        show(f"  {count}/{len(entries)}")
    return results


def _summary(label: str, results: list[dict[str, Any]]) -> str:
    total = sum(r["words"] for r in results)
    errors = sum(r["errors"] for r in results)
    named = [r["name_heard"] for r in results if r["name_heard"] is not None]
    times = sorted(r["seconds"] for r in results)
    rate = f"{errors / total:.1%}" if total else "n/a"
    names = f"names heard {sum(named)} of {len(named)}" if named else "no names"
    p90 = times[min(len(times) - 1, math.ceil(0.9 * len(times)) - 1)]
    return (
        f"{label}: {len(results)} requests, word error rate {rate} ({errors} of {total} words), "
        f"{names}, transcription {statistics.median(times):.2f} s median, {p90:.2f} s at p90"
    )


def report(results: list[dict[str, Any]]) -> str:
    """Return the scores as text: all requests, then each kind of card, then with music."""
    if not results:
        return "No requests recorded yet."
    kinds = sorted({r["kind"] for r in results})
    lines = [_summary("all", results)]
    lines.extend("  " + _summary(k, [r for r in results if r["kind"] == k]) for k in kinds)
    playing = [r for r in results if r["music"]]
    if playing:
        lines.append("  " + _summary("with music playing", playing))
    return "\n".join(lines)


def two_sided_p(wins: int, losses: int) -> float:
    """Return the exact two-sided p of `wins` against `losses` under a fair coin (sign test)."""
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(min(wins, losses) + 1)) / 2**trials
    return min(1.0, 2 * tail)


def compare(first: object, second: object) -> str:
    """Say whether two scores of the same set differ beyond chance, request by request.

    Either score not being a list of the rows `score` saves raises `ScoreError`.
    """
    first, second = _rows(first), _rows(second)
    by_id = {r["id"]: r for r in first}
    pairs = [(by_id[r["id"]], r) for r in second if r["id"] in by_id]
    if not pairs:
        return "The two scores share no requests."
    named = [(a["name_heard"], b["name_heard"]) for a, b in pairs if a["name_heard"] is not None]
    only_first = sum(1 for a, b in named if a and not b)
    only_second = sum(1 for a, b in named if b and not a)
    fewer_first = sum(1 for a, b in pairs if a["errors"] < b["errors"])
    fewer_second = sum(1 for a, b in pairs if b["errors"] < a["errors"])
    names = (
        f"names: {only_first} heard only by the first, {only_second} only by the second "
        f"(p = {two_sided_p(only_first, only_second):.3f}, exact McNemar)"
    )
    errors = (
        f"words: fewer errors from the first on {fewer_first} requests, from the second on "
        f"{fewer_second} (p = {two_sided_p(fewer_first, fewer_second):.3f}, sign test)"
    )
    return "\n".join(
        (
            _summary("first", [a for a, _ in pairs]),
            _summary("second", [b for _, b in pairs]),
            names,
            errors,
            _CHANCE,
        )
    )
