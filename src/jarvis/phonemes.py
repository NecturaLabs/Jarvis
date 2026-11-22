# ruff: file-ignore[ambiguous-unicode-character-string, ambiguous-unicode-character-comment]
"""Espeak's English phonemes, rewritten into the symbols Kokoro was trained on.

The IPA symbols here look like ASCII letters to the linter, hence the one exception above.
"""

import re

# How misaki, the phonemizer Kokoro was trained with, rewrites espeak's phonemes for English (its
# `EspeakFallback`, Apache-2.0, misaki 0.9.4): longest match first, as it does. Misaki takes a word
# from its own dictionary when it has it and falls back to this for the rest; Jarvis has no such
# dictionary, so every word goes through the rewrite. Misaki reads espeak's output with a tie mark
# inside each diphthong and affricate (`a^ɪ`); the bridge used here prints none (`aɪ`), so the keys
# here have no tie.
_SYLLABIC = re.compile(r"(\S)\u0329")
_SPOKEN_TWICE_BRITISH = {"eə": "ɛː"}  # before `e` becomes A, or it never matches (misaki's order)
_ESPEAK_TO_TRAINED = {
    "ʔˌn\u0329": "ʔn",
    "ʔn\u0329": "ʔn",
    "aɪ": "I",
    "aʊ": "W",
    "dʒ": "ʤ",
    "eɪ": "A",
    "e": "A",
    "tʃ": "ʧ",
    "ɔɪ": "Y",
    "ʲo": "jo",
    "ʲə": "jə",
    "ʲ": "",
    "ɚ": "əɹ",
    "r": "ɹ",
    "x": "k",
    "ç": "k",
    "ɐ": "ə",
    "ɬ": "l",
    "\u0303": "",
}
_BRITISH = {"iə": "ɪə", "əʊ": "Q"}
_AMERICAN = {"oʊ": "O", "ɜːɹ": "ɜɹ", "ɜː": "ɜɹ", "ɪə": "iə", "ː": ""}
_AFTER = {"o": "ɔ", "ɾ": "T", "ʔ": "t"}


def _rewrite(phonemes: str, rules: dict[str, str]) -> str:
    for old, new in sorted(rules.items(), key=lambda rule: -len(rule[0])):
        phonemes = phonemes.replace(old, new)
    return phonemes


def trained_phonemes(phonemes: str, *, british: bool) -> str:
    """Rewrite espeak's English phonemes into the symbols Kokoro was trained on (see above)."""
    if british:
        phonemes = _rewrite(phonemes, _SPOKEN_TWICE_BRITISH)
    phonemes = _rewrite(phonemes, _ESPEAK_TO_TRAINED)
    phonemes = _SYLLABIC.sub("ᵊ\\1", phonemes).replace("\u0329", "")
    phonemes = _rewrite(phonemes, _BRITISH if british else _AMERICAN)
    return _rewrite(phonemes, _AFTER)
