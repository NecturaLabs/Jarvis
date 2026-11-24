"""Espeak-ng shared by everything in Jarvis that phonemizes (`jarvis.espeak`)."""

import pytest

from jarvis import espeak


def test_every_caller_holds_the_lock_pipers_own_voices_hold() -> None:
    """Espeak is not safe from two threads: a Piper voice and Jarvis must share one lock."""
    from piper import voice as piper_voice

    lock = piper_voice._ESPEAK_PHONEMIZER_LOCK  # pyright: ignore[reportPrivateUsage] - the lock Piper's voices take

    assert espeak.lock() is lock


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ("flocks days", "Phlox Daze"),
        ("7 Nights Sale", "Seven Knights Sail"),
        ("Kneel Moons", "Neil Moons"),
    ],
)
def test_names_that_sound_the_same_have_the_same_sounds(one: str, other: str) -> None:
    assert espeak.sounds(one) == espeak.sounds(other)


def test_sounds_hold_neither_stress_nor_spaces_nor_punctuation() -> None:
    sounds = espeak.sounds("Glass-Harbor, live!")

    assert sounds
    assert not set(sounds) & set("ˈˌː ,-!()")


def test_nothing_to_say_has_no_sounds() -> None:
    assert not espeak.sounds("")
    assert not espeak.sounds("!?")
