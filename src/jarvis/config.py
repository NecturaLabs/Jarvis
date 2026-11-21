"""`config.toml`: versioned, validated settings with documented defaults."""

import tomllib
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from jarvis import paths

SCHEMA_VERSION = 1

OpClassName = Literal["safe", "confirm", "strong", "deny"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Section):
    """The Strata server Jarvis talks to."""

    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "jarvis"
    request_timeout_s: float = Field(default=120.0, gt=0, le=600)
    max_tokens: int = Field(default=300, gt=0, le=4096)
    temperature: float = Field(default=0.2, ge=0, le=2)


class StrataConfig(_Section):
    """How the `jarvis-strata` user service runs Strata (written into its profile by setup)."""

    home: Path = Field(default_factory=lambda: paths.xdg_data_home() / "strata")
    components_parent: Path = Field(default_factory=paths.xdg_data_home)
    model_file: Path = Path()
    tokenizer_dir: Path = Path()
    model_id: str = "qwen3.6-35b-a3b-ud-q4_k_xl"
    port: int = Field(default=8080, ge=1024, le=65535)
    context: int = Field(default=8192, ge=2048, le=262144)
    expert_vram_gib: float = Field(default=1.0, ge=0, le=64)
    prefill: int = Field(default=512, ge=128, le=8192)
    idle_unload_s: int = Field(default=300, ge=30, le=86400)
    reasoning_budget_tokens: int = Field(default=512, gt=0, le=8192)
    memory_max: str = "40G"


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
    data_dir: Path = Field(default_factory=paths.default_data_dir)
    model: ModelConfig = Field(default_factory=ModelConfig)
    strata: StrataConfig = Field(default_factory=StrataConfig)
    spotify: SpotifyConfig = Field(default_factory=SpotifyConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    boundaries: BoundariesConfig = Field(default_factory=BoundariesConfig)


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


def _toml_value(value: object) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in cast("list[object]", value)) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


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
    """Write the config file (not secret; mode follows the umask)."""
    path = path or paths.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(dumps(config), encoding="utf-8")
    tmp.replace(path)
