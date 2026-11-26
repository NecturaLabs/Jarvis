# ruff: file-ignore[ambiguous-unicode-character-string] - the keys are espeak's IPA symbols
"""A word's sounds as ARPAbet phones (`HH`, `AH0`), read from espeak's American English IPA.

SoulX-Singer takes English words as ARPAbet, the CMU dictionary's phone set: consonants, and
vowels carrying their stress (1 main, 2 secondary, 0 none). Espeak already knows how every word
is said, so its phonemes are mapped symbol by symbol, longest match first; nothing is guessed
from the spelling.
"""

from jarvis import espeak

VOICE = "en-us"
_STRESS = {"ˈ": "1", "ˌ": "2"}
_VOWELS = {
    "aɪə": "AY",
    "aʊə": "AW",
    "aɪ": "AY",
    "aʊ": "AW",
    "eɪ": "EY",
    "oʊ": "OW",
    "əʊ": "OW",
    "ɔɪ": "OY",
    "ɪə": "IH",
    "ɛə": "EH",
    "ʊə": "UH",
    "iː": "IY",
    "uː": "UW",
    "ɑː": "AA",
    "ɔː": "AO",
    "ɜː": "ER",
    "i": "IY",
    "ɪ": "IH",
    "ᵻ": "IH",
    "e": "EH",
    "ɛ": "EH",
    "æ": "AE",
    "a": "AA",
    "ɑ": "AA",
    "ɒ": "AA",
    "ɔ": "AO",
    "o": "OW",
    "ʊ": "UH",
    "u": "UW",
    "ʌ": "AH",
    "ə": "AH",
    "ɐ": "AH",
    "ɚ": "ER",
    "ɝ": "ER",
    "ɜ": "ER",
}
_CONSONANTS = {
    "tʃ": "CH",
    "dʒ": "JH",
    "p": "P",
    "b": "B",
    "t": "T",
    "d": "D",
    "k": "K",
    "ɡ": "G",
    "g": "G",
    "f": "F",
    "v": "V",
    "θ": "TH",
    "ð": "DH",
    "s": "S",
    "z": "Z",
    "ʃ": "SH",
    "ʒ": "ZH",
    "h": "HH",
    "m": "M",
    "n": "N",
    "ŋ": "NG",
    "l": "L",
    "ɹ": "R",
    "r": "R",
    "j": "Y",
    "w": "W",
    "ɾ": "T",  # the flap of "water"
    "ʔ": "T",
    "x": "K",
    "ç": "HH",
}
_LONGEST = max(len(key) for key in (*_VOWELS, *_CONSONANTS))
_IGNORED = frozenset("ː̩ ̃-‿")


def from_ipa(ipa: str) -> list[str]:
    """Return the ARPAbet phones of espeak IPA, vowels with their stress digit.

    A syllabic consonant (`l̩`, `n̩`) is said with a weak vowel before it, as the dictionary has
    it ("bottle" is `B AA1 T AH0 L`). Symbols with no phone (length marks, ties) are dropped.
    """
    phones: list[str] = []
    stress = "0"
    i = 0
    while i < len(ipa):
        if ipa[i] in _STRESS:
            stress = _STRESS[ipa[i]]
            i += 1
            continue
        for size in range(_LONGEST, 0, -1):
            piece = ipa[i : i + size]
            if piece in _VOWELS:
                phones.append(_VOWELS[piece] + stress)
                stress = "0"
                break
            if piece in _CONSONANTS:
                if ipa[i + size : i + size + 1] == "̩":
                    phones.append("AH0")
                phones.append(_CONSONANTS[piece])
                break
        else:
            size = 1  # a symbol with no phone
        i += size
    if phones and not any(p[-1].isdigit() for p in phones):
        phones.insert(_first_vowel_slot(phones), "AH0")  # a word must be sung on a vowel
    return phones


def _first_vowel_slot(phones: list[str]) -> int:
    return 1 if len(phones) > 1 else 0


def word(text: str) -> list[str]:
    """Return how one word is said, as ARPAbet phones (espeak's American English)."""
    ipa = "".join(phonemes for phonemes, _, _ in espeak.clauses(text, VOICE))
    return from_ipa("".join(ch for ch in ipa if ch not in _IGNORED or ch == "̩"))
