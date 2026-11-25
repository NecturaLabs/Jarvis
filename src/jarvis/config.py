"""`config.toml`: versioned, validated settings with documented defaults."""

import re
import tomllib
from collections.abc import MutableMapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self, cast
from urllib.parse import urlsplit

import tomlkit
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from jarvis import paths
from jarvis.files import write_atomic

SCHEMA_VERSION = 1

MIN_SPEECH_SPEED = 0.5  # how slow Jarvis may speak: half the voice's own pace
MAX_SPEECH_SPEED = 3.0  # how fast: three times it
OpClassName = Literal["safe", "confirm", "strong", "deny"]

type Effort = Literal["none", "low", "medium", "high"]
EFFORTS: tuple[Effort, ...] = ("none", "low", "medium", "high")  # least to most thinking


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _no_control(text: str) -> str:
    if _CONTROL.search(text):
        msg = "must not contain control characters (newlines included)"
        raise ValueError(msg)
    return text


def _no_control_path(path: Path) -> Path:
    _no_control(str(path))
    return path


# Paths and names end up in systemd unit files, so a newline or other control character in one could
# add a directive; refuse them at the config, where the error can name the key.
SafePath = Annotated[Path, AfterValidator(_no_control_path)]
SafeText = Annotated[str, AfterValidator(_no_control)]

VRAM_RESERVE_MIN_MIB = (
    100  # our floor, not the engine's: a smaller reserve leaves a game no real room
)
VRAM_RESERVE_MAX_MIB = 65536


def _reserve_or_unset(mib: int) -> int:
    if mib != 0 and mib < VRAM_RESERVE_MIN_MIB:
        msg = f"must be 0 (unset) or at least {VRAM_RESERVE_MIN_MIB}"
        raise ValueError(msg)
    return mib


KV_RESIDENT_MIN_CELLS = 20480  # the engine's own floor for `--kv-resident`


def _streamed_or_unset(cells: int) -> int:
    if cells != 0 and cells < KV_RESIDENT_MIN_CELLS:
        msg = f"must be 0 (unset) or at least {KV_RESIDENT_MIN_CELLS}"
        raise ValueError(msg)
    return cells


KvResident = Annotated[int, Field(ge=0, le=262144), AfterValidator(_streamed_or_unset)]

VramReserve = Annotated[
    int, Field(ge=0, le=VRAM_RESERVE_MAX_MIB), AfterValidator(_reserve_or_unset)
]


def _check_endpoint(url: str) -> str:
    """Return `url` when it is an `http` or `https` URL on this machine; empty means unset.

    Jarvis runs locally (docs/plan.md): the model server is on this machine, so a request and the
    key it carries never leave it.

    Raises:
        ValueError: the URL has control characters, a password, another scheme or another host.
    """
    if not url:
        return url
    _no_control(url)
    parts = urlsplit(url)
    if parts.username is not None or parts.password is not None:
        msg = "must not hold a user name or password (keys never live in the config)"
        raise ValueError(msg)
    if parts.scheme not in {"http", "https"} or parts.hostname not in _LOOPBACK_HOSTS:
        msg = "must be an http or https URL on this machine (127.0.0.1, localhost or ::1)"
        raise ValueError(msg)
    return url


Endpoint = Annotated[str, AfterValidator(_check_endpoint)]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Section):
    """How Jarvis talks to its model, the local Strata server."""

    base_url: Endpoint = ""  # empty: the local Strata server (see `Config.model_base_url`)
    model: str = "jarvis"
    request_timeout_s: float = Field(default=120.0, gt=0, le=600)
    temperature: float = Field(default=0.2, ge=0, le=2)


class AgentConfig(_Section):
    """How much one spoken request may do, and how much of the talk Jarvis remembers.

    `reply_tokens` caps what the model writes in one go, so a reply is kept short by the model,
    never cut afterwards. `memory_tokens` is the room the remembered conversation may take of the
    model's context; older turns are summarized to stay within it. `thinking` is how hard the
    model thinks before each answer at the start of a request (`none` answers at once), and
    `thinking_tokens` the most it may think then, on top of the reply's room. The model may raise
    its own effort for the rest of a request, up to `thinking_max` (the same as `thinking`: never);
    a raised call thinks up to `thinking_raised_tokens`.
    """

    max_steps: int = Field(default=8, ge=1, le=100)  # model calls in one request
    max_tool_calls: int = Field(default=12, ge=1, le=200)
    max_seconds: float = Field(default=90.0, ge=10, le=3600)
    reply_tokens: int = Field(default=400, ge=64, le=8192)
    tool_text_chars: int = Field(default=3000, ge=500, le=50000)  # of one tool result
    memory_tokens: int = Field(default=6000, ge=0, le=200000)
    memory_idle_s: float = Field(default=1800.0, ge=0)  # forget after this long unused; 0 never
    thinking: Effort = "none"
    thinking_tokens: int = Field(default=192, ge=32, le=8192)
    thinking_max: Effort = "high"
    thinking_raised_tokens: int = Field(default=768, ge=32, le=8192)

    @model_validator(mode="after")
    def _ceiling_above_start(self) -> Self:
        if EFFORTS.index(self.thinking_max) < EFFORTS.index(self.thinking):
            msg = "thinking_max must be at least thinking (the same switches raising off)"
            raise ValueError(msg)
        return self


class ChatConfig(AgentConfig):
    """The same limits for typed requests in `jarvis chat`, with room for longer, written work."""

    max_steps: int = Field(default=30, ge=1, le=100)
    max_tool_calls: int = Field(default=40, ge=1, le=200)
    max_seconds: float = Field(default=600.0, ge=10, le=3600)
    reply_tokens: int = Field(default=2048, ge=64, le=8192)
    tool_text_chars: int = Field(default=8000, ge=500, le=50000)
    memory_tokens: int = Field(default=7000, ge=0, le=200000)
    memory_idle_s: float = Field(default=0.0, ge=0)
    thinking_tokens: int = Field(default=512, ge=32, le=8192)
    thinking_raised_tokens: int = Field(default=1024, ge=32, le=8192)
    resume: bool = True  # open the chat on the conversation it had last time
    verbosity: Literal["replies", "steps", "everything"] = "replies"
    toolbar: bool = True  # the line of PC load, model and context under the input
    # What Enter does while a request runs: steer it (the words join it before the model's next
    # call) or queue them as the next request.
    while_working: Literal["steer", "queue"] = "steer"
    history_size: int = Field(default=1000, ge=0, le=100000)  # typed lines ↑ brings back


# Keys the model's own Strata run config replaced (its model, tokenizer, name, prompt chunk and
# picture reader).
_REPLACED_BY_RUN_CONFIG = frozenset(
    {"model_file", "tokenizer_dir", "model_id", "prefill", "vision_file"}
)


class StrataConfig(_Section):
    """How the `jarvis-strata` user service runs Strata (written into its profile by setup)."""

    home: SafePath = Field(default_factory=lambda: paths.xdg_data_home() / "strata")
    components_parent: SafePath = Field(default_factory=paths.xdg_data_home)
    # Strata's own run config for the model (`strata-<model>.json`, written by `strata launch`):
    # the model, its engine and its flags all come from it.
    run_config: SafePath = Path()
    port: int = Field(default=8080, ge=1024, le=65535)
    context: int = Field(default=24576, ge=2048, le=262144)
    # Expert weights kept in VRAM, for an engine that takes a size (`--expert-vram`,
    # `--vram-experts-gb`); the rest stay in RAM.
    expert_vram_gib: float = Field(default=1.0, ge=0, le=64)
    idle_unload_s: int = Field(default=86400, ge=30, le=86400)  # stays loaded (#34, decision 2)
    reasoning_budget_tokens: int = Field(default=512, gt=0, le=8192)
    # Where Strata writes a raised effort (`effort_position`); empty: the run config's own.
    effort_position: Literal["", "start", "end"] = ""
    memory_max: str = Field(default="40G", pattern=r"^\d+[KMGT]?$")
    # Keep the model server's memory in RAM: its unit may lock memory and gets no swap. Off: the
    # kernel may swap the model out while the PC is short of memory and back in on the next
    # request, pausing the GPU each time, for little room: quantized weights barely compress.
    keep_in_ram: bool = True
    # Keep a control vector the run config loads (Strata's experimental speed projection, which
    # lowers how often the model declines). Off: Jarvis's profile leaves those options out.
    speed_projection: bool = True
    # VRAM in MiB the engine leaves free for other programs (`--vram-reserve-mib`); 0 passes
    # nothing, so Strata's own default (700) applies.
    vram_reserve_mib: VramReserve = 0
    # Context cells kept in VRAM (`--kv-resident`); the whole context then lives in pinned RAM, so
    # `context` can grow past them without more VRAM. 0: all of the context in VRAM.
    kv_resident: KvResident = 0
    # Let an engine that keeps a routing profile (`--expert-profile`) learn which experts Jarvis's
    # requests use, and start from them: those experts are on the card from the first request.
    learn_experts: bool = True
    # Read pictures with the reader the run config names; off: the profile drops it.
    pictures: bool = True
    vision_idle_s: int = Field(default=120, ge=10, le=86400)  # the reader stops after this idle

    @model_validator(mode="before")
    @classmethod
    def _replaced_keys(cls, data: object) -> object:
        """Refuse the keys the run config replaced, saying what to do instead.

        Raises:
            ValueError: one of those keys is set.
        """
        if not isinstance(data, dict):
            return data
        fields = cast("dict[str, object]", data)
        old = sorted(set(fields) & _REPLACED_BY_RUN_CONFIG)
        if old:
            msg = (
                f"strata.{', strata.'.join(old)} came from before strata.run_config: run "
                "`strata launch` for the model, set strata.run_config to the "
                "strata-<model>.json it writes (`jarvis setup --run-config`), and remove "
                "the old keys"
            )
            raise ValueError(msg)
        return fields


class SpotifyConfig(_Section):
    """The owner's own Spotify developer app (PKCE, no client secret)."""

    client_id: str = ""
    redirect_port: int = Field(default=8765, ge=1024, le=65535)
    device_name: str = ""
    close_match: float = Field(default=0.8, gt=0, le=1)
    loose_match: float = Field(default=0.6, gt=0, le=1)
    queue_max: int = Field(default=20, ge=1, le=100)
    search_pages: int = Field(default=8, ge=1, le=20)
    library_refresh_s: float = Field(default=6 * 3600.0, ge=60)

    @model_validator(mode="after")
    def _loose_is_looser(self) -> Self:
        if self.loose_match > self.close_match:
            msg = "loose_match must not exceed close_match"
            raise ValueError(msg)
        return self


VAD_EXIT_MARGIN = 0.15  # how far below the entry threshold speech goes on, as Silero's own does
VAD_EXIT_FLOOR = 0.01

# A wake word model's name: its file `<data_dir>/wake/<name>.onnx`, so never a path.
WAKE_WORD_PATTERN = r"^[a-z0-9][a-z0-9_]{0,39}$"
WakeWordName = Annotated[str, StringConstraints(pattern=WAKE_WORD_PATTERN)]
WakeThreshold = Annotated[float, Field(gt=0, lt=1)]


class AudioConfig(_Section):
    """Microphone, wake word, end-of-speech and voice settings."""

    source: str = ""
    # The LiveKit wake word models listened for, each `<data_dir>/wake/<name>.onnx`.
    wake_words: list[WakeWordName] = Field(
        default_factory=lambda: ["hey_jarvis", "yo_jarvis"], min_length=1, max_length=8
    )
    wake_threshold: WakeThreshold = 0.5
    # A wake word's own score needed, by model name, in place of `wake_threshold`.
    wake_thresholds: dict[WakeWordName, WakeThreshold] = Field(
        default_factory=dict[str, float], max_length=32
    )
    vad_threshold: float = Field(default=0.5, gt=0, lt=1)
    # Once speech has started it goes on while the probability stays above this; 0 means
    # `vad_threshold` less VAD_EXIT_MARGIN (Silero's own rule), never below VAD_EXIT_FLOOR.
    vad_neg_threshold: float = Field(default=0.0, ge=0, lt=1)
    no_speech_timeout_s: float = Field(default=4.0, gt=0, le=30)
    end_silence_s: float = Field(default=0.7, gt=0, le=5)
    max_utterance_s: float = Field(default=20.0, gt=1, le=120)
    confirm_window_s: float = Field(default=8.0, gt=1, le=30)
    # How long a `strong` action waits for the confirm key (`jarvis confirm`) once asked about.
    confirm_key_s: float = Field(default=30.0, ge=5, le=120)
    barge_in: bool = True  # the wake word stops Jarvis mid-sentence and starts a new request
    # While Jarvis speaks, a LiveKit wake word needs at most this score: its voice in the
    # microphone lowers the owner's. Jarvis's own sentences are checked against it too.
    barge_in_threshold: WakeThreshold = 0.5
    # Off: a wake word or PC-audio request is ignored while the Omarchy lock screen is up.
    listen_when_locked: bool = True
    # Seconds after a spoken reply in which the next request needs no wake word; 0 turns it off.
    follow_up_s: float = Field(default=6.0, ge=0, le=15)
    follow_up_over_audio: bool = False  # open the window while another program plays audio too
    wake_engine: Literal["livekit", "azure"] = "livekit"
    azure_keyword_model: str = ""  # the Azure custom keyword `.table` file
    azure_python: str = ""  # the Python with the Speech SDK; empty: <data_dir>/azure-wake/venv
    voice: str = "kokoro:bm_george"
    speech_speed: float = Field(default=1.0, ge=MIN_SPEECH_SPEED, le=MAX_SPEECH_SPEED)
    # Rewrite espeak's phonemes into the symbols Kokoro was trained on; off until heard to help.
    kokoro_phoneme_map: bool = False
    transcribe_engine: str = ""
    transcribe_model: str = "small.en"  # measured faster and more accurate than base.en here
    transcribe_threads: int = Field(default=8, ge=1, le=256)
    # Where the gguf engine runs (Voxtype's `gguf_backend`); empty: Voxtype's own choice.
    transcribe_backend: Literal["", "auto", "cpu", "cpu_accel", "vulkan"] = ""
    transcribe_short_window: bool = True  # size Whisper's window to the clip: ~5x faster
    vocabulary_hint: bool = True
    hint_names: int = Field(default=40, ge=0, le=200)
    listen_to_pc: bool = False
    pc_audio_apps: list[str] = Field(default_factory=lambda: ["discord"])
    pc_audio_grant_s: float = Field(default=600.0, ge=0, le=86400)
    confirm_blocks_on_pc_speech: bool = True
    pc_audio_cooldown_s: float = Field(default=5.0, ge=0, le=600)

    @model_validator(mode="after")
    def _wake_words_are_distinct(self) -> Self:
        if len(set(self.wake_words)) != len(self.wake_words):
            msg = "wake_words must not name a model twice"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _exit_is_not_above_entry(self) -> Self:
        if self.vad_neg_threshold > self.vad_threshold:
            msg = "vad_neg_threshold must not exceed vad_threshold"
            raise ValueError(msg)
        return self

    @property
    def vad_exit_threshold(self) -> float:
        """The probability above which speech that has started goes on."""
        if self.vad_neg_threshold:
            return self.vad_neg_threshold
        return max(self.vad_threshold - VAD_EXIT_MARGIN, VAD_EXIT_FLOOR)


class PersonaConfig(_Section):
    """Who Jarvis is and how it speaks."""

    name: str = "Jarvis"
    tone: str = "Calm, warm and to the point, like a capable butler with a light touch of dry wit."
    addendum: str = ""


class SingConfig(_Section):
    """How Jarvis sings (the `sing` tool; `sing = "deny"` under `[boundaries.classes]` is off).

    `engine` "speech" sings with the Kokoro voice, its words stretched and moved to each note on
    the CPU. "soulx" sings with SoulX-Singer, a singing model run on the GPU for each song, and
    falls back to "speech" when its files are missing or it fails. "auto" is "soulx" when its
    files are installed, else "speech".
    """

    engine: Literal["auto", "speech", "soulx"] = "auto"
    device: int = Field(default=0, ge=0, le=15)  # the Vulkan device SoulX-Singer runs on


class ClockConfig(_Section):
    """How Jarvis says the time."""

    twelve_hour: bool = True


class FormatConfig(_Section):
    """How Jarvis writes numbers, in every reply and tool answer."""

    # Between groups of three digits in a whole part: 1,488. "" writes none.
    thousands_separator: Literal[",", " ", "'", ""] = ","


POINT = 2  # a latitude and a longitude
MAX_LATITUDE = 90.0
MAX_LONGITUDE = 180.0


class WeatherConfig(_Section):
    """The owner's home for weather and time questions that name no place (Open-Meteo)."""

    place: SafeText = ""
    coordinates: list[float] = Field(default_factory=list[float])
    temperature_unit: Literal["celsius", "fahrenheit"] = "celsius"
    wind_unit: Literal["kmh", "mph", "ms", "kn"] = "kmh"

    @model_validator(mode="after")
    def _point(self) -> Self:
        if not self.coordinates:
            return self
        if len(self.coordinates) != POINT:
            msg = "coordinates must be [latitude, longitude] or empty"
            raise ValueError(msg)
        latitude, longitude = self.coordinates
        if abs(latitude) > MAX_LATITUDE or abs(longitude) > MAX_LONGITUDE:
            msg = "coordinates must be a latitude within 90 and a longitude within 180 degrees"
            raise ValueError(msg)
        return self


class ResearchConfig(_Section):
    """Web search through the local SearXNG, page reading and Reddit."""

    searxng_url: Endpoint = "http://127.0.0.1:8888"
    results: int = Field(default=5, ge=1, le=8)
    page_chars: int = Field(default=2800, ge=500, le=50000)  # within the channel's tool_text_chars
    reddit_comments: int = Field(default=6, ge=1, le=10)


def _check_root(root: str) -> str:
    """Return `root` when it is `~`, a path under `~/` or an absolute path.

    Raises:
        ValueError: anything else, or a root with a control character.
    """
    if not root or _CONTROL.search(root) or not (root == "~" or root.startswith(("~/", "/"))):
        msg = "a root must be ~, a path under ~/ or an absolute path, on one line"
        raise ValueError(msg)
    return root


RootPath = Annotated[str, AfterValidator(_check_root)]


# Hidden folders under home that Jarvis may read by default: the desktop's own settings.
DESKTOP_SETTINGS = (
    ".config/hypr",
    ".config/omarchy",
    ".config/waybar",
    ".config/walker",
    ".config/mako",
    ".config/swayosd",
    ".config/kitty",
    ".config/alacritty",
    ".config/ghostty",
    ".config/foot",
    ".config/btop",
    ".config/fastfetch",
    ".config/jarvis",
)


def deny_pattern(entry: str) -> str:
    """Return a `files.deny` entry in one form: a name, a path under home, or an absolute path.

    `~/notes`, `./notes` and `notes/` are all the path `notes` under home.

    Raises:
        ValueError: the entry could never match anything.
    """
    text = entry.strip().removeprefix("~/")
    text = str(PurePosixPath(text)) if text else ""
    if text in {"", "."} or text.startswith("~") or ".." in PurePosixPath(text).parts:
        msg = (
            f"files.deny entry {entry!r} can never match: write a name (diary*), a path under "
            "home (~/Documents/private) or an absolute path, without '..'"
        )
        raise ValueError(msg)
    return text


def _check_deny(entry: str) -> str:
    deny_pattern(entry)
    return entry


DenyEntry = Annotated[SafeText, AfterValidator(_check_deny)]


class FilesConfig(_Section):
    """The folders Jarvis may read and write (`~` is the owner's home).

    Secrets (keys, tokens, browser profiles, Jarvis's own state) are never read or written,
    whatever the roots; `deny` adds names, paths under home or absolute paths to that list
    (shell-style patterns such as `diary*`, `~/.config/app` or `/data/finance`). Hidden files
    and folders under home (shell history, app logins and caches live there) are not read
    unless `hidden_allow` names them; by default it opens the desktop's own settings. Writing
    asks first, unless `boundaries.bypass_consent` is on, and never touches a hidden file or
    folder.
    """

    read_roots: list[RootPath] = Field(default_factory=lambda: ["~"])
    write_roots: list[RootPath] = Field(default_factory=lambda: ["~/Documents"])
    deny: list[DenyEntry] = Field(default_factory=list[str])
    hidden_allow: list[SafeText] = Field(default_factory=lambda: list(DESKTOP_SETTINGS))


def _check_window(pattern: str) -> str:
    try:
        re.compile(pattern)
    except re.error as exc:
        msg = f"not a regular expression: {exc}"
        raise ValueError(msg) from exc
    return pattern


AppName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,31}$")]
WindowPattern = Annotated[SafeText, StringConstraints(min_length=1, max_length=200)]


class AppConfig(_Section):
    """One program `open_app` may start: its command line and how its window is known.

    `command` is the program and its arguments, run as written (no shell). `window` is a regular
    expression matched, case-blind, against a window's class (never its title, which a browser tab
    can copy), to see that the program opened and on which workspace.
    """

    command: list[Annotated[SafeText, StringConstraints(min_length=1, max_length=400)]] = Field(
        min_length=1, max_length=32
    )
    window: Annotated[WindowPattern, AfterValidator(_check_window)]


class AppsConfig(_Section):
    """The programs Jarvis may open when the owner asks, by name; nothing else is ever started."""

    programs: dict[AppName, AppConfig] = Field(
        default_factory=lambda: {
            "spotify": AppConfig(command=["uwsm-app", "--", "spotify"], window=r"\bspotify\b")
        }
    )


class ShellConfig(_Section):
    """The opt-in shell operation, `pc_shell` (ADR 0002): off unless `enabled` is set.

    When on, every command is `strong` (the confirm key at the PC) and runs once in a user scope
    limited to `memory_max_mib` of memory with no swap, `cpu_percent` of one core (200 is two),
    `tasks_max` processes and threads, and `timeout_s` seconds; never with raised rights.
    """

    enabled: bool = False
    timeout_s: float = Field(default=30.0, ge=1, le=600)
    memory_max_mib: int = Field(default=1024, ge=64, le=65536)
    cpu_percent: int = Field(default=100, ge=10, le=6400)
    tasks_max: int = Field(default=256, ge=8, le=4096)


class BoundariesConfig(_Section):
    """Per-operation class overrides set by the owner, and whether Jarvis asks before acting.

    `bypass_consent` (the owner's choice, on by default) runs every action that would need a
    spoken yes without asking: `confirm` actions, actions after outside or file text, and requests
    heard in PC audio. A `strong` action still waits for the confirm key, and an operation set to
    `deny` stays refused. Whatever the consent mode, at most `max_actions` side-effecting actions
    run in any `actions_window_s` seconds, counted across the voice service, the chat and
    `jarvis ask` together.
    """

    classes: dict[str, OpClassName] = Field(default_factory=dict[str, OpClassName])
    bypass_consent: bool = True
    max_actions: int = Field(default=30, ge=1, le=10000)
    actions_window_s: float = Field(default=600.0, ge=1, le=86400)


class Config(_Section):
    """The whole file."""

    schema_version: int = SCHEMA_VERSION
    data_dir: SafePath = Field(default_factory=paths.default_data_dir)
    model: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)
    strata: StrataConfig = Field(default_factory=StrataConfig)
    spotify: SpotifyConfig = Field(default_factory=SpotifyConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    sing: SingConfig = Field(default_factory=SingConfig)
    clock: ClockConfig = Field(default_factory=ClockConfig)
    format: FormatConfig = Field(default_factory=FormatConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    files: FilesConfig = Field(default_factory=FilesConfig)
    apps: AppsConfig = Field(default_factory=AppsConfig)
    shell: ShellConfig = Field(default_factory=ShellConfig)
    boundaries: BoundariesConfig = Field(default_factory=BoundariesConfig)

    def strata_url(self) -> str:
        """Return the URL of the local Strata server, on its configured port."""
        return f"http://127.0.0.1:{self.strata.port}/v1"

    def model_base_url(self) -> str:
        """Return the local model server's URL: the explicit setting, else Strata's own port."""
        return self.model.base_url or self.strata_url()


class ConfigError(Exception):
    """The config file is unreadable or invalid."""


def load(path: Path | None = None) -> Config:
    """Read and validate the config; a missing file gives the defaults.

    Raises:
        ConfigError: the file is not valid TOML, has an unknown schema version or fails validation.
    """
    path = path or paths.config_file()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Config()
    except tomllib.TOMLDecodeError as exc:
        msg = f"{path}: {exc}"
        raise ConfigError(msg) from exc
    version = raw.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        msg = f"{path}: schema_version {version} is not supported (expected {SCHEMA_VERSION})"
        raise ConfigError(msg)
    _drop_retired(raw)
    try:
        cfg = Config.model_validate(raw)
    except ValueError as exc:
        msg = f"{path}: {exc}"
        raise ConfigError(msg) from exc
    return _without_derived_url(cfg)


# Keys older versions wrote that now live elsewhere; a saved file still holding one stays valid.
_RETIRED = {"model": ("max_tokens",)}  # now [agent] and [chat] reply_tokens


def _drop_retired(raw: dict[str, object]) -> None:
    for section, keys in _RETIRED.items():
        table = raw.get(section)
        if isinstance(table, dict):
            for key in keys:
                cast("dict[str, object]", table).pop(key, None)


def _without_derived_url(cfg: Config) -> Config:
    """Drop a saved model URL that only repeats the one derived from Strata's port.

    Older versions saved the derived URL; kept, it would pin the model to that port when the
    Strata port is changed later.
    """
    if cfg.model.base_url != cfg.strata_url():
        return cfg
    return cfg.model_copy(update={"model": cfg.model.model_copy(update={"base_url": ""})})


def dumps(config: Config, text: str = "") -> str:
    """Return the config as TOML: `text` (a config file) changed to say `config`.

    Only what differs from the file is written. A key the file lacks is added only when its value
    is not the default, and a key whose value goes back to the default is taken out, so a later
    release's defaults reach the install; everything else in the file, comments included, stays
    as the owner wrote it.
    """
    doc = tomlkit.parse(text) if text.strip() else tomlkit.document()
    if "schema_version" not in doc:
        doc["schema_version"] = SCHEMA_VERSION
    _merge(doc, config, Config(), keep=frozenset({"schema_version"}))
    return tomlkit.dumps(doc)


def _merge(
    table: MutableMapping[str, Any],
    model: BaseModel,
    default: BaseModel,
    *,
    keep: frozenset[str] = frozenset(),
) -> None:
    """Make the TOML `table` say `model`, writing as little as `dumps` says.

    A section is merged key by key, since loading fills what it lacks from the defaults. Any
    other value, a mapping included (`apps.programs`), is written whole, since it is read whole.
    """
    values = model.model_dump(mode="json")
    defaults = default.model_dump(mode="json")
    for key in [key for key in table if key not in values]:
        del table[key]  # a key this version no longer has
    for key, value in values.items():
        section = getattr(model, key)
        if isinstance(section, BaseModel):
            if key not in table and value == defaults[key]:
                continue
            if not isinstance(table.get(key), MutableMapping):
                table[key] = tomlkit.table()
            _merge(table[key], section, cast("BaseModel", getattr(default, key)))
        elif key in table:
            if _plain(table[key]) == value:
                continue
            if value == defaults[key] and key not in keep:
                del table[key]
            else:
                table[key] = value
        elif value != defaults[key]:
            table[key] = value


def _plain(item: object) -> object:
    """Return a TOML item's plain value."""
    unwrap = getattr(item, "unwrap", None)
    return unwrap() if callable(unwrap) else item


def save(config: Config, path: Path | None = None) -> None:
    """Write the config file (not secret) in one step, so a reader never sees half of it."""
    target = path or paths.config_file()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    write_atomic(target, dumps(config, text))
