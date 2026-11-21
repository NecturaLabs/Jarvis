"""Requests that skip the model must be exact; everything else is left to it."""

import pytest

from jarvis.shortcuts import Matched, Shortcut, match, normalise

SHORTCUTS = [
    ("pause", Shortcut(r"(?:pause|stop)(?: it| the music)?")),
    ("next", Shortcut(r"(?:next|skip)(?: (?:this )?(?:song|track))?")),
    (
        "set_volume",
        Shortcut(r"volume (?:to )?(?P<percent>\d{1,3})", lambda g: {"percent": int(g["percent"])}),
    ),
    ("stop_everything", Shortcut(r"halt")),
    ("also_halt", Shortcut(r"halt")),
]


@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("Pause.", Matched("pause", {})),
        ("Hey Jarvis, could you please pause the music?", Matched("pause", {})),
        ("Stop it, Jarvis.", Matched("pause", {})),
        ("Skip this song please", Matched("next", {})),
        ("Volume to 40", Matched("set_volume", {"percent": 40})),
    ],
)
def test_a_plain_request_maps_to_its_operation(said: str, expected: Matched) -> None:
    assert match(said, SHORTCUTS, names=["Jarvis"]) == expected


@pytest.mark.parametrize(
    "said",
    [
        "",
        "Pause after this song",
        "Don't pause",
        "Play the song Skip",
        "What does pause do?",
        "Pause. And then play Glass Harbor",
        "halt",
        "pause " * 30,
    ],
)
def test_anything_else_is_left_to_the_model(said: str) -> None:
    assert match(said, SHORTCUTS, names=["Jarvis"]) is None


def test_normalise_strips_politeness_and_the_name_only_at_the_edges() -> None:
    assert normalise("OK Jarvis, can you just skip, please?", ["Jarvis"]) == "skip"
    assert normalise("Play Jarvis Cocker", ["Jarvis"]) == "play jarvis cocker"
