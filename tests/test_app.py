import pytest

from jarvis.app import MAX_HINT_CHARS, echoes_hint, no_action_operation, speech_hint
from jarvis.config import AudioConfig, Config, PersonaConfig
from jarvis.operations import Operation, Registry, Result


def _op(name: str) -> Operation:
    return Operation(name, name, {"type": "object", "properties": {}}, lambda _a: Result("ok"))


def test_speech_hint_comes_from_config_tools_and_library() -> None:
    cfg = Config(persona=PersonaConfig(name="Friday"), audio=AudioConfig(hint_names=2))
    registry = Registry([_op("spotify_play"), _op("spotify_set_volume"), no_action_operation()])

    hint = speech_hint(cfg, registry, ["Artist A", "Artist B", "Artist C"])

    assert hint.startswith("Friday.")
    assert "play" in hint
    assert "set volume" in hint
    assert "no action" not in hint
    assert "Artist B" in hint
    assert "Artist C" not in hint


def test_speech_hint_without_library_still_names_the_tools() -> None:
    hint = speech_hint(Config(), Registry([_op("spotify_pause")]), [])

    assert "pause" in hint


def test_speech_hint_keeps_only_printable_bounded_names() -> None:
    cfg = Config(audio=AudioConfig(hint_names=50))
    names = ["Glass\x00Harbor", "Night\nCircuit", "\ud800Low", "A" * 101, "Second Sun", ""]

    hint = speech_hint(cfg, Registry([_op("spotify_pause")]), names)

    assert "Second Sun" in hint
    for refused in ("Glass", "Night", "Low", "A" * 101):
        assert refused not in hint
    hint.encode("utf-8")  # never an unencodable argument for the transcriber


def test_speech_hint_names_stop_at_the_length_cap() -> None:
    cfg = Config(audio=AudioConfig(hint_names=200))
    names = [f"Band Number {i:03d}" for i in range(200)]

    hint = speech_hint(cfg, Registry([_op("spotify_pause")]), names)

    assert len(hint) <= MAX_HINT_CHARS
    assert "Band Number 000" in hint


@pytest.mark.parametrize(
    ("transcript", "echo"),
    [
        ("Glass Harbor, Night Circuit, Second Sun, Low Tide.", True),
        ("pause, resume, next, previous", True),
        ("play Glass Harbor please", False),
        ("Night Circuit", False),  # too short to tell from a real request
        ("", False),
    ],
)
def test_a_transcript_that_only_repeats_the_hint_is_an_echo(transcript: str, echo: bool) -> None:
    hint = (
        "Jarvis. pause, resume, next, previous. Glass Harbor, Night Circuit, Second Sun, Low Tide"
    )

    assert echoes_hint(transcript, hint) is echo
