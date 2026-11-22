"""Where Jarvis keeps its config, state and downloaded files (XDG base directories)."""

import os
from pathlib import Path


def _xdg(name: str, fallback: str) -> Path:
    value = os.environ.get(name, "")
    return Path(value) if value and Path(value).is_absolute() else Path.home() / fallback


def xdg_config_home() -> Path:
    """Return `$XDG_CONFIG_HOME` (default `~/.config`)."""
    return _xdg("XDG_CONFIG_HOME", ".config")


def xdg_data_home() -> Path:
    """Return `$XDG_DATA_HOME` (default `~/.local/share`)."""
    return _xdg("XDG_DATA_HOME", ".local/share")


def config_dir() -> Path:
    """Return `$XDG_CONFIG_HOME/jarvis`."""
    return _xdg("XDG_CONFIG_HOME", ".config") / "jarvis"


def state_dir() -> Path:
    """Return `$XDG_STATE_HOME/jarvis`, which holds tokens, keys and the audit log."""
    return _xdg("XDG_STATE_HOME", ".local/state") / "jarvis"


def runtime_dir() -> Path:
    """Return `$XDG_RUNTIME_DIR/jarvis` (the user's runtime folder, emptied at logout)."""
    value = os.environ.get("XDG_RUNTIME_DIR", "")
    base = Path(value) if value and Path(value).is_absolute() else Path(f"/run/user/{os.getuid()}")
    return base / "jarvis"


def default_data_dir() -> Path:
    """Return `$XDG_DATA_HOME/jarvis`, the default home of wake, VAD and voice models."""
    return _xdg("XDG_DATA_HOME", ".local/share") / "jarvis"


def config_file() -> Path:
    """Return the path of `config.toml`."""
    return config_dir() / "config.toml"


def strata_key_file() -> Path:
    """Return the file holding the Strata API key (shared by Jarvis and the Strata unit)."""
    return state_dir() / "strata.env"


def spotify_token_file() -> Path:
    """Return the Spotify token file."""
    return state_dir() / "spotify-token.json"


def audit_file() -> Path:
    """Return the audit log path."""
    return state_dir() / "audit.jsonl"


def actions_file() -> Path:
    """Return the file that counts recent side-effecting actions across Jarvis's processes."""
    return state_dir() / "actions.json"
