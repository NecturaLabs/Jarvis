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
    """Return `url` when it is `https`, or `http` to this machine; empty means unset.

    Raises:
        ValueError: the URL has control characters, a password or an insecure scheme or host.
    """
    if not url:
        return url
    _no_control(url)
    parts = urlsplit(url)
    if parts.username is not None or parts.password is not None:
        msg = "must not hold a user name or password (keys never live in the config)"
        raise ValueError(msg)
    secure = parts.scheme == "https" or (
        parts.scheme == "http" and parts.hostname in _LOOPBACK_HOSTS
    )
    if not secure or not parts.hostname:
        msg = "must be an https URL (http only for 127.0.0.1, localhost or ::1)"
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
    max_tokens: int = Field(default=200, gt=0, le=4096)
    temperature: float = Field(default=0.2, ge=0, le=2)


class StrataConfig(_Section):
    """How the `jarvis-strata` user service runs Strata (written into its profile by setup)."""

    home: SafePath = Field(default_factory=lambda: paths.xdg_data_home() / "strata")
    components_parent: SafePath = Field(default_factory=paths.xdg_data_home)
    model_file: SafePath = Path()
    tokenizer_dir: SafePath = Path()
    model_id: SafeText = "qwen3.6-35b-a3b-ud-q4_k_xl"
    port: int = Field(default=8080, ge=1024, le=65535)
    context: int = Field(default=8192, ge=2048, le=262144)
    expert_vram_gib: float = Field(default=1.0, ge=0, le=64)
    prefill: int = Field(default=512, ge=128, le=8192)
    idle_unload_s: int = Field(default=300, ge=30, le=86400)
    reasoning_budget_tokens: int = Field(default=512, gt=0, le=8192)
    memory_max: str = Field(default="40G", pattern=r"^\d+[KMGT]?$")


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
    voice: str = "kokoro:bm_george"
    speech_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    transcribe_engine: str = ""
    transcribe_model: str = "base.en"
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


class BoundariesConfig(_Section):
    """Per-operation class overrides set by the owner (default profile: ask for risky)."""

    classes: dict[str, OpClassName] = Field(default_factory=dict[str, OpClassName])


class Config(_Section):
    """The whole file."""

    schema_version: int = SCHEMA_VERSION
    data_dir: SafePath = Field(default_factory=paths.default_data_dir)
    model: ModelConfig = Field(default_factory=ModelConfig)
    strata: StrataConfig = Field(default_factory=StrataConfig)
    spotify: SpotifyConfig = Field(default_factory=SpotifyConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    boundaries: BoundariesConfig = Field(default_factory=BoundariesConfig)

    def model_base_url(self) -> str:
        """Return the local model server's URL: the explicit setting, else Strata's own port."""
        return self.model.base_url or f"http://127.0.0.1:{self.strata.port}/v1"


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
    try:
        return Config.model_validate(raw)
    except ValueError as exc:
        msg = f"{path}: {exc}"
        raise ConfigError(msg) from exc


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
