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
