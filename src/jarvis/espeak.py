"""Espeak-ng, the phonemizer inside the `piper-tts` wheel, for everything in Jarvis that uses it.

Espeak keeps one global state (its voice and its buffers) and is not safe to call from two
threads at once. Every call here holds the lock Piper's own voices hold, so Kokoro, a Piper voice
and name matching never phonemize at the same time.
"""

import functools
import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading

AMERICAN = "en-us"
_LANG_SWITCH = re.compile(r"\([^)]+\)")  # "(fr)" ... "(en)" around a word read in another language


def lock() -> threading.Lock:
    """Return the one lock every espeak call in the process holds: Piper's own."""
    from piper import voice

    # Piper's voices take this lock; sharing it keeps their calls and ours apart.
    return voice._ESPEAK_PHONEMIZER_LOCK  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access] - Piper's own lock


def clauses(text: str, voice: str) -> list[tuple[str, str, bool]]:
    """Return espeak's phonemes for `text` in `voice`, clause by clause.

    Each clause comes with its terminator and whether a sentence ends there.
    """
    from piper import espeakbridge
    from piper.phonemize_espeak import ESPEAK_DATA_DIR

    with lock():
        espeakbridge.initialize(str(ESPEAK_DATA_DIR))
        espeakbridge.set_voice(voice)
        return espeakbridge.get_phonemes(text)


def _sound(symbol: str) -> bool:
    """Whether a phoneme symbol is a sound, not stress, length or a modifier."""
    return unicodedata.category(symbol).startswith("L") and unicodedata.category(symbol) != "Lm"


@functools.lru_cache(maxsize=16384)
def sounds(text: str) -> str:
    """Return how `text` sounds in American English: its phonemes alone, as one string.

    Names that sound the same have the same sounds ("Kneel Moons" and "Neil Moons"), and a
    number is read as words. Results are kept, since the same library names are compared again
    on every request.
    """
    if not any(ch.isalnum() for ch in text):
        return ""
    phonemes = "".join(clause for clause, _end, _sentence in clauses(text, AMERICAN))
    return "".join(filter(_sound, unicodedata.normalize("NFC", _LANG_SWITCH.sub("", phonemes))))
