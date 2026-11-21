"""Builds Jarvis from its config: model, operations, prompt, audit log and agent."""

from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING

from jarvis import paths
from jarvis.agent import Agent
from jarvis.audit import AuditLog
from jarvis.model import StrataModel
from jarvis.operations import Operation, Registry, Result
from jarvis.secretfile import read_api_key
from jarvis.spotify.api import SpotifyApi
from jarvis.spotify.auth import TokenStore
from jarvis.spotify.ops import SpotifyOps

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from jarvis.config import Config


def strata_key_file() -> Path:
    """Return the file holding the Strata API key (shared by Jarvis and the Strata unit)."""
    return paths.state_dir() / "strata.env"


def spotify_token_file() -> Path:
    """Return the Spotify token file."""
    return paths.state_dir() / "spotify-token.json"


def audit_file() -> Path:
    """Return the audit log path."""
    return paths.state_dir() / "audit.jsonl"


def no_action_operation() -> Operation:
    """Return the explicit "nothing to do" choice the model makes for questions and chat."""
    return Operation(
        "no_action",
        "Call this when the request needs no Spotify action (a greeting, a general question or a "
        "request you can't do). Then answer in words. Never use it when the owner asks you to "
        "play, pause, resume, skip, change volume, shuffle, repeat, queue or like something.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        lambda _args: Result("No action taken. Answer the owner briefly."),
        side_effect=False,
        idempotent=True,
        spoken_summary=lambda _args: "do nothing",
    )


def system_prompt(config: Config) -> str:
    """Render the shipped core prompt with the owner's persona and addendum."""
    core = files("jarvis").joinpath("prompts/core.md").read_text(encoding="utf-8")
    text = core.replace("{name}", config.persona.name).replace("{tone}", config.persona.tone)
    if config.persona.addendum.strip():
        text += f"\nThe owner's own instructions:\n{config.persona.addendum.strip()}\n"
    return text


def speech_hint(config: Config, registry: Registry, artist_names: list[str]) -> str:
    """Return words speech-to-text should expect.

    The assistant's name, what its tools do and the owner's artist names, all taken from the
    config, the registry and the owner's library, so the hint follows what is installed and played.
    """
    actions = [
        name.removeprefix("spotify_").replace("_", " ")
        for name in registry.names()
        if name != "no_action"
    ]
    names = artist_names[: config.audio.hint_names]
    parts = [f"{config.persona.name}.", ", ".join(actions) + ".", ", ".join(names)]
    return " ".join(p for p in parts if p.strip(". "))


@dataclass(frozen=True, slots=True)
class Jarvis:
    """The assembled agent and what it needs."""

    agent: Agent
    model: StrataModel
    audit: AuditLog
    ops: SpotifyOps
    hint: Callable[[], str]


class SetupError(Exception):
    """Something setup provides is missing."""


def build(config: Config) -> Jarvis:
    """Assemble the agent from the config.

    Raises:
        SetupError: the Spotify client ID or the Strata API key is missing.
    """
    if not config.spotify.client_id:
        msg = "no Spotify client ID in the config; run `jarvis setup`"
        raise SetupError(msg)
    key = read_api_key(strata_key_file())
    if not key:
        msg = f"no Strata API key at {strata_key_file()}; run `jarvis setup`"
        raise SetupError(msg)
    model = StrataModel(config.model.model_copy(update={"base_url": config.model_base_url()}), key)
    tokens = TokenStore(config.spotify.client_id, spotify_token_file())
    ops = SpotifyOps(SpotifyApi(tokens), config.spotify)
    audit = AuditLog(audit_file())
    registry = Registry([*ops.operations(), no_action_operation()])
    agent = Agent(
        model,
        registry,
        system_prompt(config),
        audit,
        config.boundaries.classes,
        grant_s=config.audio.pc_audio_grant_s,
    )
    return Jarvis(
        agent, model, audit, ops, lambda: speech_hint(config, registry, ops.artist_names())
    )
