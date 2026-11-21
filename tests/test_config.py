import tomllib
from typing import TYPE_CHECKING

import pytest

from jarvis import config

if TYPE_CHECKING:
    from pathlib import Path


def test_missing_file_gives_defaults(tmp_path: Path) -> None:
    assert config.load(tmp_path / "none.toml") == config.Config()


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    original = config.Config(
        spotify=config.SpotifyConfig(client_id="abc"),
        persona=config.PersonaConfig(addendum='Say "sir".\nKeep it short.'),
        boundaries=config.BoundariesConfig(classes={"spotify_play": "confirm"}),
        audio=config.AudioConfig(listen_to_pc=True, pc_audio_apps=["discord", "teamspeak"]),
    )
    path = tmp_path / "config.toml"

    config.save(original, path)

    assert config.load(path) == original


def test_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("schema_version = 99\n", encoding="utf-8")
    with pytest.raises(config.ConfigError):
        config.load(path)


def test_invalid_toml_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[audio\n", encoding="utf-8")
    with pytest.raises(config.ConfigError):
        config.load(path)


def test_control_characters_in_strings_survive_save_and_load(tmp_path: Path) -> None:
    nasty = 'tab\t cr\r bell\x07 del\x7f quote" backslash\\ newline\n bs\b ff\f nul\x00 esc\x1b'
    original = config.Config(
        spotify=config.SpotifyConfig(client_id=nasty),
        persona=config.PersonaConfig(name=nasty, tone=nasty, addendum=nasty),
        boundaries=config.BoundariesConfig(classes={nasty: "deny"}),
    )
    text = config.dumps(original)

    assert tomllib.loads(text)["persona"]["addendum"] == nasty
    path = tmp_path / "config.toml"
    config.save(original, path)
    assert config.load(path) == original


def test_dumped_text_holds_no_raw_control_character() -> None:
    text = config.dumps(config.Config(persona=config.PersonaConfig(addendum="a\r\x07\x7fb")))

    assert not {"\r", "\x07", "\x7f"} & set(text)
