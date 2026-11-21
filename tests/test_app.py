from jarvis.app import no_action_operation, speech_hint
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
