"""Builds Jarvis from its config: model, operations, prompt, audit log and agent."""

from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING

from jarvis.agent import Agent
from jarvis.audit import AuditLog
from jarvis.model import StrataModel
from jarvis.operations import Operation, Registry, Result
from jarvis.paths import audit_file, spotify_token_file, strata_key_file  # re-exported for callers
from jarvis.secretfile import read_api_key
from jarvis.spotify.api import SpotifyApi
from jarvis.spotify.auth import TokenStore
from jarvis.spotify.ops import SpotifyOps

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.config import Config


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
        spoken_summary=lambda _args: "do nothing",
    )


def system_prompt(config: Config) -> str:
    """Render the shipped core prompt with the owner's persona and addendum."""
    core = files("jarvis").joinpath("prompts/core.md").read_text(encoding="utf-8")
    text = core.replace("{name}", config.persona.name).replace("{tone}", config.persona.tone)
    if config.persona.addendum.strip():
        text += f"\nThe owner's own instructions:\n{config.persona.addendum.strip()}\n"
    return text


MAX_HINT_NAME_CHARS = 100
MAX_HINT_CHARS = 800
MIN_ECHO_WORDS = 4


def _fits_hint(name: str) -> bool:
    """Return whether a library name may go into the hint (it reaches the transcriber's argv)."""
    if not name.strip() or len(name) > MAX_HINT_NAME_CHARS or not name.isprintable():
        return False
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def speech_hint(config: Config, registry: Registry, artist_names: list[str]) -> str:
    """Return words speech-to-text should expect.

    The assistant's name, what its tools do and the owner's artist names, all taken from the
    config, the registry and the owner's library, so the hint follows what is installed and played.
    Library names are someone else's text: only short printable ones are used, up to a length cap.
    """
    actions = [
        name.removeprefix("spotify_").replace("_", " ")
        for name in registry.names()
        if name != "no_action"
    ]
    head = " ".join(
        p for p in (f"{config.persona.name}.", ", ".join(actions) + ".") if p.strip(". ")
    )
    names: list[str] = []
    room = MAX_HINT_CHARS - len(head) - 1
    for name in filter(_fits_hint, artist_names[: config.audio.hint_names]):
        cost = len(name) + (2 if names else 0)
        if cost > room:
            break
        names.append(name)
        room -= cost
    return f"{head} {', '.join(names)}" if names else head


def _words(text: str) -> list[str]:
    return "".join(ch if ch.isalnum() else " " for ch in text.lower()).split()


def echoes_hint(transcript: str, hint: str) -> bool:
    """Return whether a transcript is only a run of the hint's own words.

    On silence or noise, a transcriber primed with a hint can return the hint itself; that is
    not a request.
    """
    heard = _words(transcript)
    if len(heard) < MIN_ECHO_WORDS:
        return False
    primed = _words(hint)
    return any(primed[i : i + len(heard)] == heard for i in range(len(primed) - len(heard) + 1))


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
        names=(config.persona.name,),
    )
    return Jarvis(
        agent, model, audit, ops, lambda: speech_hint(config, registry, ops.artist_names())
    )
