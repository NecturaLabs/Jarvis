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
    assert cfg.agent.reply_tokens == 400
    assert cfg.chat.reply_tokens == 2048
    assert cfg.chat.resume is True


def test_a_saved_reply_cap_from_an_older_version_still_loads(tmp_path: Path) -> None:
    cfg = _load_text(tmp_path, "[model]\nmax_tokens = 200\n")

    assert cfg.agent.reply_tokens == 400


@pytest.mark.parametrize("section", ["agent", "chat"])
def test_the_reply_and_memory_limits_must_fit_their_ranges(tmp_path: Path, section: str) -> None:
    for text in ("reply_tokens = 10", "max_steps = 0", "memory_tokens = -1"):
        with pytest.raises(config.ConfigError):
            _load_text(tmp_path, f"[{section}]\n{text}\n")


def test_an_unknown_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(config.ConfigError, match="surprise"):
        _load_text(tmp_path, "[audio]\nsurprise = 1\n")


@pytest.mark.parametrize(
    "text",
    [
        "[agent]\nreply_tokens = 0\n",
        "[chat]\nreply_tokens = 9000\n",
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
    explicit = config.Config(model=config.ModelConfig(base_url="http://127.0.0.1:9000/v1"))
    assert explicit.model_base_url() == "http://127.0.0.1:9000/v1"


def test_a_saved_url_that_only_repeats_the_strata_port_follows_it_later(tmp_path: Path) -> None:
    """Older versions saved the derived URL; it must not pin the model to the old port."""
    cfg = _load_text(
        tmp_path, '[model]\nbase_url = "http://127.0.0.1:8080/v1"\n[strata]\nport = 8080\n'
    )
    moved = cfg.model_copy(update={"strata": cfg.strata.model_copy(update={"port": 9100})})

    assert not cfg.model.base_url
    assert moved.model_base_url() == "http://127.0.0.1:9100/v1"


def test_a_saved_url_for_another_port_is_kept(tmp_path: Path) -> None:
    cfg = _load_text(tmp_path, '[model]\nbase_url = "http://127.0.0.1:9000/v1"\n')

    assert cfg.model.base_url == "http://127.0.0.1:9000/v1"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:9000/v1",
        "http://localhost:9000/v1",
        "http://[::1]:9000/v1",
        "https://127.0.0.1:9443/v1",
        "https://localhost/v1",
    ],
)
def test_only_urls_on_this_machine_are_accepted(url: str) -> None:
    assert config.ModelConfig(base_url=url).base_url == url


@pytest.mark.parametrize(
    "url",
    [
        "https://llm.example.org/v1",
        "https://192.168.1.5:8443/v1",
        "https://127.0.0.1.example.org/v1",
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
        ),
        agent=config.AgentConfig(reply_tokens=123, memory_idle_s=0),
        chat=config.ChatConfig(max_steps=50, resume=False),
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


def test_transcription_runs_fast_by_default() -> None:
    audio = config.AudioConfig()
    assert audio.transcribe_threads == 8
    assert audio.transcribe_short_window is True


def test_weather_and_research_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = config.Config(
        clock=config.ClockConfig(twelve_hour=True),
        weather=config.WeatherConfig(
            place="Lisbon", coordinates=[38.72, -9.14], temperature_unit="fahrenheit"
        ),
        research=config.ResearchConfig(searxng_url="http://127.0.0.1:9999", results=3),
    )

    config.save(original, path)

    assert config.load(path) == original


@pytest.mark.parametrize("point", [[1.0], [1.0, 2.0, 3.0], [91.0, 0.0], [0.0, -181.0]])
def test_coordinates_are_a_real_point_or_nothing(point: list[float]) -> None:
    with pytest.raises(ValueError, match="coordinates"):
        config.WeatherConfig(coordinates=point)


def test_search_runs_only_on_this_machine() -> None:
    with pytest.raises(ValueError, match="on this machine"):
        config.ResearchConfig(searxng_url="https://searx.example.org")
    assert config.ResearchConfig().searxng_url == "http://127.0.0.1:8888"


EXAMPLE = Path(__file__).resolve().parent.parent / "config.example.toml"


def test_the_example_config_holds_exactly_the_shipped_defaults() -> None:
    raw = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))

    assert config.Config.model_validate(raw) == config.Config()


def test_the_example_config_names_no_machine_or_person() -> None:
    text = EXAMPLE.read_text(encoding="utf-8")

    assert str(Path.home()) not in text
    assert "/home/" not in text
    raw = tomllib.loads(text)
    assert not raw["spotify"]["client_id"]
    assert not raw["weather"]["place"]
    assert raw["weather"]["coordinates"] == []


def test_consent_is_bypassed_unless_the_owner_turns_asking_on() -> None:
    assert config.Config().boundaries.bypass_consent is True
    assert (
        config.Config.model_validate(
            {"boundaries": {"bypass_consent": False}}
        ).boundaries.bypass_consent
        is False
    )
