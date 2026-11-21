"""The Strata profile and systemd units: their text, their quoting and how they are written."""

import json
import stat
import sys
from pathlib import Path

import pytest

from jarvis import services
from jarvis.config import Config, ConfigError, StrataConfig


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


pytestmark = pytest.mark.usefixtures("home")


def _config(tmp_path: Path, **strata: object) -> Config:
    settings: dict[str, object] = {
        "home": tmp_path / "strata-home",
        "components_parent": tmp_path / "components",
        "model_file": tmp_path / "models" / "q4.gguf",
        "tokenizer_dir": tmp_path / "tokenizer",
        **strata,
    }
    return Config(strata=StrataConfig.model_validate(settings))


def _lines(text: str) -> list[str]:
    return text.splitlines()


def test_the_strata_unit_for_a_normal_config(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    key = tmp_path / "state" / "jarvis" / "strata.env"

    text = services.strata_unit(cfg, key)

    lines = _lines(text)
    assert f"ConditionPathExists={tmp_path}/models/q4.gguf" in lines
    assert f"Environment=STRATA_DATA_DIR={tmp_path}/strata-home" in lines
    assert f"Environment=XDG_DATA_HOME={tmp_path}/components" in lines
    assert f"EnvironmentFile={key}" in lines
    assert "Environment=PIPEWIRE_REMOTE=/nonexistent" in lines
    assert "MemoryMax=40G" in lines
    assert any(line.startswith("ExecStart=") and " serve --profile " in line for line in lines)
    assert text.endswith("WantedBy=graphical-session.target\n")


def test_paths_with_spaces_are_quoted_for_systemd(tmp_path: Path) -> None:
    spaced = tmp_path / "my models"
    cfg = _config(tmp_path, home=spaced / "strata home", model_file=spaced / "q4 k.gguf")
    key = tmp_path / "my keys" / "strata.env"

    lines = _lines(services.strata_unit(cfg, key))

    assert f"ConditionPathExists={spaced}/q4 k.gguf" in lines
    assert f'Environment="STRATA_DATA_DIR={spaced}/strata home"' in lines
    assert f"EnvironmentFile={key}" in lines


def test_the_exec_line_quotes_a_program_path_with_a_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program = tmp_path / "my apps" / "venv" / "bin" / "python"
    monkeypatch.setattr(sys, "executable", str(program))

    assert f'ExecStart="{program.parent}/jarvis" run' in _lines(services.jarvis_unit())


@pytest.mark.parametrize(
    ("raw", "written"),
    [
        ("/a b", '"/a b"'),
        ('/a"b', '"/a\\"b"'),
        ("/a\\b", '"/a\\\\b"'),
        ("/a$HOME", "/a$$HOME"),
        ("/a%h", "/a%%h"),
        ("/a;b", '"/a;b"'),
        ("/plain/path", "/plain/path"),
    ],
)
def test_exec_words_are_escaped_per_systemd_rules(raw: str, written: str) -> None:
    assert services.exec_word(raw) == written


@pytest.mark.parametrize(
    "program", ["-/bin/x", "@/bin/x", "+/bin/x", "!/bin/x", ":/bin/x", "bin/x", ""]
)
def test_a_program_that_is_not_an_absolute_path_never_starts_a_unit_line(program: str) -> None:
    with pytest.raises(ConfigError, match="absolute path"):
        services.exec_program(program, "program")


def test_an_absolute_program_is_written_as_an_exec_word() -> None:
    assert services.exec_program("/my apps/jarvis", "program") == '"/my apps/jarvis"'


@pytest.mark.parametrize("bad", ["/a\nExecStartPre=/bin/evil", "/a\rb", "/a\x00b", "/a\x7fb"])
def test_a_control_character_never_reaches_a_unit(tmp_path: Path, bad: str) -> None:
    cfg = _config(tmp_path)
    unchecked = cfg.model_copy(
        update={"strata": cfg.strata.model_copy(update={"model_file": Path(bad)})}
    )

    with pytest.raises(ConfigError, match="control character"):
        services.strata_unit(unchecked, tmp_path / "strata.env")
    with pytest.raises(ConfigError, match="control character"):
        services.strata_unit(cfg, Path(bad))


def test_a_refused_value_writes_no_unit_at_all(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    unchecked = cfg.model_copy(
        update={"strata": cfg.strata.model_copy(update={"memory_max": "40G\nExecStart=/bin/evil"})}
    )

    with pytest.raises(ConfigError, match="memory_max"):
        services.write_units(unchecked, tmp_path / "strata.env")

    assert not services.unit_dir().exists() or list(services.unit_dir().iterdir()) == []


def test_a_value_ending_in_a_backslash_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="backslash"):
        services.strata_unit(_config(tmp_path), Path("/keys/strata.env\\"))


def test_the_model_condition_is_omitted_when_no_model_is_configured(tmp_path: Path) -> None:
    cfg = Config(strata=StrataConfig(home=tmp_path / "s"))

    text = services.strata_unit(cfg, tmp_path / "strata.env")

    assert "ConditionPathExists" not in text


def test_a_setup_error_is_not_restarted() -> None:
    lines = _lines(services.jarvis_unit())

    assert "RestartPreventExitStatus=2" in lines
    assert "Restart=on-failure" in lines


def test_the_strata_profile_content(tmp_path: Path) -> None:
    cfg = _config(tmp_path, port=9100, context=4096, idle_unload_s=120)

    profile = json.loads(services.write_strata_profile(cfg).read_text(encoding="utf-8"))

    assert profile["port"] == 9100
    assert profile["host"] == "127.0.0.1"
    assert profile["lazy_load"] is True
    assert profile["idle_unload_s"] == 120
    assert profile["model_name"] == cfg.strata.model_id
    assert profile["tokenizer"] == str(tmp_path / "tokenizer")
    assert profile["args"][:2] == ["--native", str(tmp_path / "models" / "q4.gguf")]
    assert profile["args"][profile["args"].index("--max-context") + 1] == "4096"
    assert "api_key" not in json.dumps(profile)


def test_units_are_written_atomically_and_leave_no_temp_file(tmp_path: Path) -> None:
    written = services.write_units(_config(tmp_path), tmp_path / "strata.env")

    assert [p.name for p in written] == [services.STRATA_UNIT, services.JARVIS_UNIT]
    assert sorted(p.name for p in services.unit_dir().iterdir()) == sorted(
        [services.JARVIS_UNIT, services.STRATA_UNIT]
    )
    assert stat.S_IMODE(written[0].stat().st_mode) == 0o644
    assert written[1].read_text(encoding="utf-8") == services.jarvis_unit()


def test_rewriting_a_unit_replaces_it_whole(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = services.write_units(cfg, tmp_path / "strata.env")[0]
    first.write_text("stale\n", encoding="utf-8")

    services.write_units(cfg, tmp_path / "strata.env")

    assert "stale" not in first.read_text(encoding="utf-8")


def test_a_failed_write_leaves_the_old_file_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "unit" / "x.service"
    target.parent.mkdir()
    target.write_text("old\n", encoding="utf-8")

    def refuse(_self: Path, _target: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(Path, "replace", refuse)

    with pytest.raises(OSError, match="disk full"):
        services.write_atomic(target, "new\n")

    assert target.read_text(encoding="utf-8") == "old\n"
    assert [p.name for p in target.parent.iterdir()] == ["x.service"]


def test_the_profile_is_written_atomically(tmp_path: Path) -> None:
    path = services.write_strata_profile(_config(tmp_path))

    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]
