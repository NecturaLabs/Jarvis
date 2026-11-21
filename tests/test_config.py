import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from jarvis import config

if TYPE_CHECKING:
    from collections.abc import Callable


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


def test_control_characters_in_strings_survive_save_and_load(tmp_path: Path) -> None:
    nasty = 'tab\t cr\r bell\x07 del\x7f quote" backslash\\ newline\n bs\b ff\f nul\x00 esc\x1b'
    original = config.Config(
        spotify=config.SpotifyConfig(client_id=nasty),
        persona=config.PersonaConfig(name=nasty, tone=nasty, addendum=nasty),
        boundaries=config.BoundariesConfig(classes={nasty: "deny"}),
    )
    text = config.dumps(original)

    assert tomllib.loads(text)["persona"]["addendum"] == nasty
    path = tmp_path / "config.toml"
    config.save(original, path)
    assert config.load(path) == original


def test_dumped_text_holds_no_raw_control_character() -> None:
    text = config.dumps(config.Config(persona=config.PersonaConfig(addendum="a\r\x07\x7fb")))

    assert not {"\r", "\x07", "\x7f"} & set(text)


def _load_text(tmp_path: Path, text: str) -> config.Config:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return config.load(path)


def test_shipped_defaults_are_the_safe_ones() -> None:
    cfg = config.Config()

    assert cfg.audio.listen_to_pc is False
    assert cfg.audio.vocabulary_hint is True
    assert cfg.audio.pc_audio_grant_s == 600
    assert cfg.audio.confirm_blocks_on_pc_speech is True
    assert cfg.audio.pc_audio_cooldown_s == pytest.approx(5.0)
    assert not cfg.model.base_url
    assert cfg.model.max_tokens == 200


def test_an_unknown_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(config.ConfigError, match="surprise"):
        _load_text(tmp_path, "[audio]\nsurprise = 1\n")


@pytest.mark.parametrize(
    "text",
    [
        "[model]\nmax_tokens = 0\n",
        "[model]\nmax_tokens = 5000\n",
        "[model]\ntemperature = 3.0\n",
        "[strata]\nport = 80\n",
        "[audio]\nwake_threshold = 1.0\n",
        "[audio]\npc_audio_cooldown_s = -1\n",
        "[audio]\npc_audio_cooldown_s = 601\n",
        "[spotify]\nclose_match = 1.5\n",
    ],
)
def test_out_of_range_values_are_refused(tmp_path: Path, text: str) -> None:
    with pytest.raises(config.ConfigError):
        _load_text(tmp_path, text)


def test_the_model_url_is_derived_from_the_strata_port_unless_set() -> None:
    assert config.Config().model_base_url() == "http://127.0.0.1:8080/v1"
    derived = config.Config(strata=config.StrataConfig(port=9100))
    assert derived.model_base_url() == "http://127.0.0.1:9100/v1"
    explicit = config.Config(model=config.ModelConfig(base_url="https://llm.example.org/v1"))
    assert explicit.model_base_url() == "https://llm.example.org/v1"


@pytest.mark.parametrize(
    "url",
    [
        "https://llm.example.org/v1",
        "http://127.0.0.1:9000/v1",
        "http://localhost:9000/v1",
        "http://[::1]:9000/v1",
    ],
)
def test_https_or_loopback_http_urls_are_accepted(url: str) -> None:
    assert config.ModelConfig(base_url=url).base_url == url


@pytest.mark.parametrize(
    "url",
    [
        "http://llm.example.org/v1",
        "http://192.168.1.5:8080/v1",
        "ftp://127.0.0.1/v1",
        "file:///etc/passwd",
        "llm.example.org/v1",
        "https:///v1",
        "https://user:pass@llm.example.org/v1",
        "https://llm.example.org/v1\nHost: other",
    ],
)
def test_other_urls_are_refused_naming_the_key(url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        config.ModelConfig(base_url=url)


@pytest.mark.parametrize("value", ["40G", "512M", "8", "2T", "100K"])
def test_memory_max_accepts_sizes(value: str) -> None:
    assert config.StrataConfig(memory_max=value).memory_max == value


@pytest.mark.parametrize("value", ["", "40GB", "G", "4 0G", "40G\n", "-1G", "40g", "infinity"])
def test_memory_max_refuses_everything_else(value: str) -> None:
    with pytest.raises(ValueError, match="memory_max"):
        config.StrataConfig(memory_max=value)


_PATH_FIELDS: list[tuple[str, Callable[[str], Any]]] = [
    ("data_dir", lambda v: config.Config(data_dir=Path(v))),
    ("home", lambda v: config.StrataConfig(home=Path(v))),
    ("components_parent", lambda v: config.StrataConfig(components_parent=Path(v))),
    ("model_file", lambda v: config.StrataConfig(model_file=Path(v))),
    ("tokenizer_dir", lambda v: config.StrataConfig(tokenizer_dir=Path(v))),
    ("model_id", lambda v: config.StrataConfig(model_id=v)),
]


@pytest.mark.parametrize("bad", ["/a\nb", "/a\rb", "/a\x00b", "/a\x07b", "/a\x7fb", "/a\tb"])
@pytest.mark.parametrize(("name", "make"), _PATH_FIELDS)
def test_paths_and_unit_strings_refuse_control_characters(
    name: str, make: Callable[[str], Any], bad: str
) -> None:
    with pytest.raises(ValueError, match=name):
        make(bad)


def test_a_path_with_a_space_is_fine() -> None:
    cfg = config.StrataConfig(model_file=Path("/data/my models/q4.gguf"))

    assert cfg.model_file.name == "q4.gguf"


def test_control_characters_in_a_loaded_path_are_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(config.ConfigError, match="model_file"):
        _load_text(tmp_path, '[strata]\nmodel_file = "/a\\nb"\n')


def test_loose_match_may_not_exceed_close_match() -> None:
    with pytest.raises(ValueError, match="loose_match"):
        config.SpotifyConfig(close_match=0.5, loose_match=0.7)
    assert config.SpotifyConfig(close_match=0.6, loose_match=0.6).loose_match == pytest.approx(0.6)


def test_a_non_default_config_round_trips_with_the_new_keys(tmp_path: Path) -> None:
    original = config.Config(
        data_dir=Path("/data/with space/jarvis"),
        model=config.ModelConfig(
            base_url="http://localhost:9000/v1",
            max_tokens=123,
        ),
        strata=config.StrataConfig(memory_max="12G", port=9000),
        audio=config.AudioConfig(confirm_blocks_on_pc_speech=False, pc_audio_cooldown_s=12.5),
    )
    path = tmp_path / "config.toml"

    config.save(original, path)

    assert config.load(path) == original
    assert "confirm_blocks_on_pc_speech = false" in path.read_text(encoding="utf-8")


def test_save_does_not_write_through_a_link_left_at_a_temp_name(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched", encoding="utf-8")
    target = tmp_path / "config.toml"
    target.with_suffix(".toml.tmp").symlink_to(victim)

    config.save(config.Config(), target)

    assert victim.read_text(encoding="utf-8") == "untouched"
    assert config.load(target) == config.Config()
