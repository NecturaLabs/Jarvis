"""Espeak-ng shared by everything in Jarvis that phonemizes (`jarvis.espeak`)."""

import threading
from typing import TYPE_CHECKING

import pytest

from jarvis import espeak

if TYPE_CHECKING:
    from collections.abc import Callable


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


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: espeak.sounds("Held Lock Name"), id="sounds"),
        pytest.param(lambda: espeak.clauses("Hi", "en-us"), id="clauses"),
    ],
)
def test_espeak_waits_while_another_caller_holds_the_lock(call: Callable[[], object]) -> None:
    espeak.sounds.cache_clear()
    done = threading.Event()

    def run() -> None:
        call()
        done.set()

    with espeak.lock():
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        assert not done.wait(0.3)  # espeak takes milliseconds; it is waiting for the lock
    assert done.wait(5)


@pytest.mark.parametrize("name", ["Glass\x00 Harbor", "Bad\ud800Name", "Tab\tName"])
def test_a_name_with_characters_espeak_cannot_take_still_has_sounds(name: str) -> None:
    """Spotify's JSON can carry a NUL or a lone surrogate; espeak raised on both."""
    assert espeak.sounds(name)


def test_a_very_long_name_is_cut_before_espeak_reads_it() -> None:
    long = "glass harbor " * 1000

    assert espeak.sounds(long) == espeak.sounds(long[: espeak.MAX_CHARS])


@pytest.mark.parametrize("name", ["東京", "椎名林檎", "القاهرة", "Tokio 東京"])
def test_a_name_in_another_script_has_no_english_sounds(name: str) -> None:
    """An English voice reads every such letter by its script's name, so all would sound alike."""
    assert not espeak.sounds(name)


def test_latin_letters_with_accents_still_have_sounds() -> None:
    assert espeak.sounds("Måneskin Café")
