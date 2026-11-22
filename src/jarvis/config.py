"""`config.toml`: versioned, validated settings with documented defaults."""

import re
import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self, cast
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from jarvis import paths
from jarvis.files import write_atomic

SCHEMA_VERSION = 1

OpClassName = Literal["safe", "confirm", "strong", "deny"]


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
    model's context; older turns are summarized to stay within it.
    """

    max_steps: int = Field(default=8, ge=1, le=100)  # model calls in one request
    max_tool_calls: int = Field(default=12, ge=1, le=200)
    max_seconds: float = Field(default=90.0, ge=10, le=3600)
    reply_tokens: int = Field(default=400, ge=64, le=8192)
    tool_text_chars: int = Field(default=3000, ge=500, le=50000)  # of one tool result
    memory_tokens: int = Field(default=6000, ge=0, le=200000)
    memory_idle_s: float = Field(default=1800.0, ge=0)  # forget after this long unused; 0 never


class ChatConfig(AgentConfig):
    """The same limits for typed requests in `jarvis chat`, with room for longer, written work."""

    max_steps: int = Field(default=30, ge=1, le=100)
    max_tool_calls: int = Field(default=40, ge=1, le=200)
    max_seconds: float = Field(default=600.0, ge=10, le=3600)
    reply_tokens: int = Field(default=2048, ge=64, le=8192)
    tool_text_chars: int = Field(default=8000, ge=500, le=50000)
    memory_tokens: int = Field(default=11500, ge=0, le=200000)
    memory_idle_s: float = Field(default=0.0, ge=0)
    resume: bool = True  # open the chat on the conversation it had last time
    verbosity: Literal["replies", "steps", "everything"] = "replies"
    toolbar: bool = True  # the line of PC load, model and context under the input
    history_size: int = Field(default=1000, ge=0, le=100000)  # typed lines ↑ brings back


class StrataConfig(_Section):
    """How the `jarvis-strata` user service runs Strata (written into its profile by setup)."""

    home: SafePath = Field(default_factory=lambda: paths.xdg_data_home() / "strata")
    components_parent: SafePath = Field(default_factory=paths.xdg_data_home)
    model_file: SafePath = Path()
    tokenizer_dir: SafePath = Path()
    model_id: SafeText = "qwen3.6-35b-a3b-ud-q4_k_xl"
    port: int = Field(default=8080, ge=1024, le=65535)
    context: int = Field(default=24576, ge=2048, le=262144)
    expert_vram_gib: float = Field(default=1.0, ge=0, le=64)
    prefill: int = Field(default=512, ge=128, le=8192)
    idle_unload_s: int = Field(default=86400, ge=30, le=86400)  # stays loaded (#34, decision 2)
    reasoning_budget_tokens: int = Field(default=512, gt=0, le=8192)
    memory_max: str = Field(default="40G", pattern=r"^\d+[KMGT]?$")
    # The picture reader (the model's mmproj file, beside it); unset: Jarvis reads no pictures.
    vision_file: SafePath = Path()
    vision_idle_s: int = Field(default=120, ge=10, le=86400)  # the reader stops after this idle

    @property
    def pictures(self) -> bool:
        """Whether the model is served with its picture reader."""
        return self.vision_file != Path()


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


class AudioConfig(_Section):
    """Microphone, wake word, end-of-speech and voice settings."""

    source: str = ""
    wake_threshold: float = Field(default=0.5, gt=0, lt=1)
    vad_threshold: float = Field(default=0.5, gt=0, lt=1)
    no_speech_timeout_s: float = Field(default=4.0, gt=0, le=30)
    end_silence_s: float = Field(default=0.7, gt=0, le=5)
    max_utterance_s: float = Field(default=20.0, gt=1, le=120)
    confirm_window_s: float = Field(default=8.0, gt=1, le=30)
    barge_in: bool = True  # the wake word stops Jarvis mid-sentence and starts a new request
    # Off: a wake word or PC-audio request is ignored while the Omarchy lock screen is up.
    listen_when_locked: bool = True
    # Seconds after a spoken reply in which the next request needs no wake word; 0 turns it off.
    follow_up_s: float = Field(default=6.0, ge=0, le=15)
    wake_engine: Literal["livekit", "azure"] = "livekit"
    azure_keyword_model: str = ""  # the Azure custom keyword `.table` file
    azure_python: str = ""  # the Python with the Speech SDK; empty: <data_dir>/azure-wake/venv
    voice: str = "kokoro:bm_george"
    speech_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    transcribe_engine: str = ""
    transcribe_model: str = "small.en"  # measured faster and more accurate than base.en here
    transcribe_threads: int = Field(default=8, ge=1, le=256)
    transcribe_short_window: bool = True  # size Whisper's window to the clip: ~5x faster
    vocabulary_hint: bool = True
    hint_names: int = Field(default=40, ge=0, le=200)
    listen_to_pc: bool = False
    pc_audio_apps: list[str] = Field(default_factory=lambda: ["discord"])
    pc_audio_grant_s: float = Field(default=600.0, ge=0, le=86400)
    confirm_blocks_on_pc_speech: bool = True
    pc_audio_cooldown_s: float = Field(default=5.0, ge=0, le=600)


class PersonaConfig(_Section):
    """Who Jarvis is and how it speaks."""

    name: str = "Jarvis"
    tone: str = "Calm, warm and to the point, like a capable butler with a light touch of dry wit."
    addendum: str = ""


class ClockConfig(_Section):
    """How Jarvis says the time."""

    twelve_hour: bool = True


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


class FilesConfig(_Section):
    """The folders Jarvis may read and write (`~` is the owner's home).

    Secrets (keys, tokens, browser profiles, Jarvis's own state) are never read or written,
    whatever the roots; `deny` adds names or paths under home to that list (shell-style
    patterns such as `diary*` or `.config/app`). Hidden files and folders under home (shell
    history, app logins and caches live there) are not read unless `hidden_allow` names them;
    by default it opens the desktop's own settings. Writing asks first, unless
    `boundaries.bypass_consent` is on, and never touches a hidden file or folder.
    """

    read_roots: list[RootPath] = Field(default_factory=lambda: ["~"])
    write_roots: list[RootPath] = Field(default_factory=lambda: ["~/Documents"])
    deny: list[SafeText] = Field(default_factory=list[str])
    hidden_allow: list[SafeText] = Field(default_factory=lambda: list(DESKTOP_SETTINGS))


class BoundariesConfig(_Section):
    """Per-operation class overrides set by the owner, and whether Jarvis asks before acting.

    `bypass_consent` (the owner's choice, on by default) runs every action that would need a yes
    without asking: `confirm` and `strong` actions, actions after outside or file text, and
    requests heard in PC audio. An operation set to `deny` stays refused.
    """

    classes: dict[str, OpClassName] = Field(default_factory=dict[str, OpClassName])
    bypass_consent: bool = True


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
    clock: ClockConfig = Field(default_factory=ClockConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    files: FilesConfig = Field(default_factory=FilesConfig)
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


_TOML_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}
_DEL = 0x7F


def _toml_char(char: str) -> str:
    if char in _TOML_ESCAPES:
        return _TOML_ESCAPES[char]
    if ord(char) < ord(" ") or ord(char) == _DEL:
        return f"\\u{ord(char):04X}"
    return char


def _toml_value(value: object) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in cast("list[object]", value)) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return '"' + "".join(_toml_char(c) for c in str(value)) + '"'


def dumps(config: Config) -> str:
    """Render the config as TOML."""
    data = config.model_dump(mode="json")
    lines = [
        f"schema_version = {data.pop('schema_version')}",
        f"data_dir = {_toml_value(data.pop('data_dir'))}",
    ]
    for section, values in data.items():
        lines.append(f"\n[{section}]")
        nested: dict[str, dict[str, object]] = {}
        for key, value in values.items():
            if isinstance(value, dict):
                nested[key] = value
            else:
                lines.append(f"{key} = {_toml_value(value)}")
        for key, table in nested.items():
            lines.append(f"\n[{section}.{key}]")
            lines.extend(f"{_toml_value(k)} = {_toml_value(v)}" for k, v in table.items())
    return "\n".join(lines) + "\n"


def save(config: Config, path: Path | None = None) -> None:
    """Write the config file (not secret) in one step, so a reader never sees half of it."""
    write_atomic(path or paths.config_file(), dumps(config))
