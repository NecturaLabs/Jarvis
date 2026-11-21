"""Where Jarvis keeps its files: XDG folders, with the spec's fallbacks."""

from typing import TYPE_CHECKING

import pytest

from jarvis import app, paths

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_folders_default_under_the_home_directory(home: Path) -> None:
    assert paths.config_dir() == home / ".config" / "jarvis"
    assert paths.state_dir() == home / ".local" / "state" / "jarvis"
    assert paths.default_data_dir() == home / ".local" / "share" / "jarvis"
    assert paths.config_file() == home / ".config" / "jarvis" / "config.toml"


def test_an_absolute_xdg_value_is_used(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "data"))

    assert paths.xdg_config_home() == home / "cfg"
    assert paths.config_dir() == home / "cfg" / "jarvis"
    assert paths.state_dir() == home / "state" / "jarvis"
    assert paths.xdg_data_home() == home / "data"
    assert paths.default_data_dir() == home / "data" / "jarvis"


@pytest.mark.parametrize("value", ["relative/dir", ".", "..", "~/cfg", "cfg"])
def test_a_relative_xdg_value_is_ignored(
    home: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(name, value)

    assert paths.xdg_config_home() == home / ".config"
    assert paths.xdg_data_home() == home / ".local" / "share"
    assert paths.config_dir() == home / ".config" / "jarvis"
    assert paths.state_dir() == home / ".local" / "state" / "jarvis"
    assert paths.default_data_dir() == home / ".local" / "share" / "jarvis"


def test_an_empty_xdg_value_is_ignored(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", "")

    assert paths.state_dir() == home / ".local" / "state" / "jarvis"


def test_the_private_files_live_in_the_state_folder(home: Path) -> None:
    state = home / ".local" / "state" / "jarvis"

    assert paths.strata_key_file() == state / "strata.env"
    assert paths.spotify_token_file() == state / "spotify-token.json"
    assert paths.audit_file() == state / "audit.jsonl"


def test_the_app_module_still_offers_the_moved_helpers() -> None:
    assert app.strata_key_file is paths.strata_key_file
    assert app.spotify_token_file is paths.spotify_token_file
    assert app.audit_file is paths.audit_file
