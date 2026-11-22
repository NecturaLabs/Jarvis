"""Builds Jarvis from its config: model, operations, prompt, audit log and agent."""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from jarvis import calc, paths, skills
from jarvis.agent import Agent, Limits
from jarvis.audit import AuditLog
from jarvis.model import StrataModel
from jarvis.operations import Operation, Registry, Result
from jarvis.paths import audit_file, spotify_token_file, strata_key_file  # re-exported for callers
from jarvis.pc.files import Boundary, FileOps
from jarvis.pc.processes import ProcessOps
from jarvis.pc.stats import StatsOps
from jarvis.pictures import PictureCache
from jarvis.policy import effective_class
from jarvis.secretfile import read_api_key
from jarvis.spotify.api import SpotifyApi
from jarvis.spotify.auth import TokenStore
from jarvis.spotify.ops import SpotifyOps
from jarvis.sysinfo import Sampler
from jarvis.web.arctic import ArcticShift
from jarvis.web.guard import public_client
from jarvis.web.openmeteo import OpenMeteo
from jarvis.web.ops import Clients, WebOps
from jarvis.web.page import PageReader
from jarvis.web.searxng import Searxng

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.config import Config


MAX_REPLY_CHARS = 4000
CHARS_PER_REPLY_TOKEN = 8  # generous: the token cap, not this, keeps a reply to length
WORDS_PER_TOKEN = 0.5  # a cautious count of English words a reply token holds

type Channel = Literal["voice", "chat"]


def no_action_operation(max_reply_chars: int = MAX_REPLY_CHARS) -> Operation:
    """Return the explicit "nothing to do" choice the model makes for questions and chat.

    Its `reply` is the model's own answer, so a greeting or a question takes one model call.
    """
    return Operation(
        "no_action",
        "Call this when the request needs no tool (a greeting, chat, or a request you can't do), "
        "with your spoken answer in `reply`, or to answer from what earlier tools returned. Never "
        "use it when the owner asks you to play, pause, resume, skip, change volume, shuffle, "
        "repeat, queue or like something, or for the time, the weather or facts you would have "
        "to look up.",
        {
            "type": "object",
            "properties": {
                "reply": {
                    "type": "string",
                    "maxLength": max_reply_chars,
                    "description": "Your reply to the owner, as they will hear or read it.",
                }
            },
            "additionalProperties": False,
        },
        lambda args: Result(
            "No action taken. Answer the owner briefly.", spoken=str(args.get("reply", ""))
        ),
        side_effect=False,
        spoken_summary=lambda _args: "do nothing",
    )


def system_prompt(config: Config, channel: Channel = "voice") -> str:
    """Render the shipped prompt for a channel with the owner's persona, skills and addendum.

    The voice prompt asks for short spoken replies; the chat prompt for written Markdown that fits
    the chat's reply budget.
    """
    prompts = files("jarvis").joinpath("prompts")
    core = prompts.joinpath("core.md").read_text(encoding="utf-8")
    style = prompts.joinpath(f"{channel}.md").read_text(encoding="utf-8")
    words = int(round(config.chat.reply_tokens * WORDS_PER_TOKEN, -1))
    fill = {
        "skills": skills.prompt_section(skills.shipped()),
        "name": config.persona.name,
        "tone": config.persona.tone,
        "words": str(words),
        "numbers": _number_rule(config.format.thousands_separator),
    }
    text = core + "\n" + style
    for key, value in fill.items():
        text = text.replace("{" + key + "}", value)
    if config.persona.addendum.strip():
        text += f"\nThe owner's own instructions:\n{config.persona.addendum.strip()}\n"
    return text


PAGE_HEADROOM = 200  # a read page's title line, within the channel's tool_text_chars
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
    return f"{head} {', '.join(names)}." if names else head


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


def web_operations(config: Config, shown: PictureCache | None = None) -> list[Operation]:
    """Return the time, weather, web and Reddit operations, with clients made on first use.

    With `shown`, where the chat keeps the pictures it shows, the picture operations come too.
    """
    return WebOps(
        config,
        Clients(
            search=Searxng(config.research.searxng_url),
            weather=OpenMeteo(factory=public_client),
            reddit=ArcticShift(factory=public_client),
            pages=PageReader(factory=public_client),
        ),
        shown=shown,
    ).operations()


def pc_operations(config: Config, limits: Limits, home: Path | None = None) -> list[Operation]:
    """Return `pc_stats`, the file operations, within the owner's `[files]` folders, and processes.

    Jarvis's own state (its tokens and keys) is never read or written, whatever the roots say.
    """
    boundary = Boundary.of(config.files, home or Path.home(), protected=(paths.state_dir(),))
    return [
        *StatsOps(Sampler()).operations(),
        *FileOps(boundary, max_chars=limits.tool_text_chars).operations(),
        *ProcessOps().operations(),
    ]


def _number_rule(separator: str) -> str:
    """Say how to write a number of four or more digits, as `[format]` sets it."""
    if not separator:
        return "Write numbers without separators between digit groups: 1488, 25000."
    sample = f"{1488:,}".replace(",", separator), f"{25000:,}".replace(",", separator)
    mark = {" ": "a space", "'": "an apostrophe"}.get(separator, f"{separator!r}")
    return (
        f"Write numbers of four or more digits with {mark} between groups of three:"
        f" {sample[0]}, {sample[1]}."
    )


class SetupError(Exception):
    """Something setup provides is missing."""


def channel_limits(config: Config, channel: Channel) -> Limits:
    """Return the limits of a channel: the chat's own, or the voice service's."""
    section = config.chat if channel == "chat" else config.agent
    return Limits.of(section, config.strata.context, remember_outside=channel == "chat")


def channel_registry(
    config: Config, spotify: list[Operation], limits: Limits, channel: Channel
) -> Registry:
    """Return every operation a channel's agent is offered, sized to its limits.

    A read page fills the chat's room for one result; spoken requests keep the configured length.
    """
    room = limits.tool_text_chars - PAGE_HEADROOM
    pages = room if channel == "chat" else min(config.research.page_chars, room)
    web = config.model_copy(
        update={"research": config.research.model_copy(update={"page_chars": pages})}
    )
    shipped = skills.shipped()
    return Registry(
        [
            *spotify,
            *web_operations(web, chat_pictures() if channel == "chat" else None),
            *pc_operations(config, limits),
            *calc.operations(config.format),
            *([skills.skill_operation(shipped)] if shipped else []),
            no_action_operation(limits.reply_tokens * CHARS_PER_REPLY_TOKEN),
        ]
    )


def unknown_overrides(config: Config, spotify: list[Operation]) -> list[str]:
    """Return the names in `boundaries.classes` that no operation of either channel has."""
    known: set[str] = set()
    for channel in ("voice", "chat"):
        registry = channel_registry(config, spotify, channel_limits(config, channel), channel)
        known.update(registry.names())
    return sorted(set(config.boundaries.classes) - known)


def overrides_below_floor(config: Config, spotify: list[Operation]) -> list[str]:
    """Return the names `boundaries.classes` sets below the lowest class their operation allows."""
    classes = config.boundaries.classes
    low: set[str] = set()
    for channel in ("voice", "chat"):
        registry = channel_registry(config, spotify, channel_limits(config, channel), channel)
        for name, override in classes.items():
            op = registry.get(name)
            if op is not None and effective_class(op, {name: override}).value != override:
                low.add(name)
    return sorted(low)


def chat_pictures() -> PictureCache:
    """Return where the chat keeps the pictures it shows (0600, the newest 50)."""
    return PictureCache(paths.state_dir() / "chat-pictures")


def chat_memory_file() -> Path:
    """Return where the chat keeps its conversation between sessions."""
    return paths.state_dir() / "chat-conversation.json"


def strata_model(config: Config, key: str) -> StrataModel:
    """Return the client for this config's Strata server, knowing its model file when set."""
    model_file = config.strata.model_file
    return StrataModel(
        config.model.model_copy(update={"base_url": config.model_base_url()}),
        key,
        model_file=model_file if model_file != Path() else None,
    )


def build(config: Config, *, via: str | None = None, session: str = "") -> Jarvis:
    """Assemble the agent from the config.

    `via` marks every audit event of this agent with where its requests come from ("chat",
    "ask"); the voice service leaves it out. The chat gets the chat's limits and written replies,
    and keeps its conversation across sessions when `chat.resume` is on. `session` marks the
    agent's audit events too, so a window can tell its own from others'.

    Raises:
        SetupError: the Spotify client ID or the Strata API key is missing, or
            `boundaries.classes` names an operation that does not exist (a typo would otherwise
            leave it at its default class) or sets one below its lowest class.
    """
    if not config.spotify.client_id:
        msg = "no Spotify client ID in the config; run `jarvis setup`"
        raise SetupError(msg)
    key = read_api_key(strata_key_file())
    if not key:
        msg = f"no Strata API key at {strata_key_file()}; run `jarvis setup`"
        raise SetupError(msg)
    model = strata_model(config, key)
    tokens = TokenStore(config.spotify.client_id, spotify_token_file())
    ops = SpotifyOps(
        SpotifyApi(tokens, library_ttl_s=config.spotify.library_refresh_s), config.spotify
    )
    tags = {"via": via} if via else {}
    if session:
        tags["session"] = session
    audit = AuditLog(audit_file(), tags=tags or None)
    if unknown := unknown_overrides(config, ops.operations()):
        msg = (
            f"boundaries.classes names no operation: {', '.join(unknown)}; "
            "fix the name or remove it"
        )
        raise SetupError(msg)
    if low := overrides_below_floor(config, ops.operations()):
        msg = (
            f"boundaries.classes sets {', '.join(low)} below what it allows; "
            'stop_process takes only "strong" or "deny"'
        )
        raise SetupError(msg)
    channel: Channel = "chat" if via == "chat" else "voice"
    limits = channel_limits(config, channel)
    registry = channel_registry(config, ops.operations(), limits, channel)
    remembered = chat_memory_file() if channel == "chat" else None
    if remembered is not None and not config.chat.resume:
        remembered.unlink(missing_ok=True)
    agent = Agent(
        model,
        registry,
        system_prompt(config, channel),
        audit,
        config.boundaries.classes,
        grant_s=config.audio.pc_audio_grant_s,
        names=(config.persona.name,),
        limits=limits,
        memory_file=remembered,
        pictures=config.strata.pictures,
        bypass_consent=config.boundaries.bypass_consent,
    )
    return Jarvis(
        agent, model, audit, ops, lambda: speech_hint(config, registry, ops.artist_names())
    )
