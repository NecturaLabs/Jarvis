"""The `jarvis` command, run against a throw-away home: never the owner's config or services."""

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from jarvis import __version__, cli, config
from jarvis.cli import main

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every XDG folder at `tmp_path` and stand in for systemd."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    def inactive(_unit: str) -> str:
        return "inactive"

    def systemctl(*_args: str) -> int:
        return 0

    monkeypatch.setattr(cli, "_unit_state", inactive)
    monkeypatch.setattr(cli, "_systemctl", systemctl)
    return tmp_path


pytestmark = pytest.mark.usefixtures("home")


def test_version_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"jarvis {__version__}"


def test_no_arguments_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "Jarvis is not running" in capsys.readouterr().out


def test_pc_audio_toggle_saves_the_setting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    restarts: list[bool] = []
    monkeypatch.setattr(cli, "_restart_if_running", lambda: restarts.append(True))

    assert main(["pc-audio", "on"]) == 0
    assert config.load().audio.listen_to_pc
    assert config.load().audio.pc_audio_apps == ["discord"]

    assert main(["pc-audio", "--apps", "discord,teamspeak"]) == 0
    assert config.load().audio.pc_audio_apps == ["discord", "teamspeak"]

    assert main(["pc-audio", "off"]) == 0
    assert not config.load().audio.listen_to_pc
    assert len(restarts) == 3
    assert "off" in capsys.readouterr().out.splitlines()[-1]


def test_pc_audio_without_arguments_changes_nothing(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_restart_if_running", lambda: pytest.fail("no restart"))

    assert main(["pc-audio"]) == 0
    assert not (home / "config" / "jarvis" / "config.toml").exists()


def test_pc_audio_apps_are_one_comma_separated_value() -> None:
    assert main(["pc-audio", "--apps", " discord , Firefox "]) == 0

    assert config.load().audio.pc_audio_apps == ["discord", "Firefox"]


def test_listening_to_all_pc_output_needs_the_explicit_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["pc-audio", "on"]) == 0

    assert main(["pc-audio", "--all-output"]) == 0

    saved = config.load().audio
    assert saved.pc_audio_apps == []
    assert saved.listen_to_pc
    assert "all PC audio" in capsys.readouterr().out.splitlines()[-1]


def test_a_second_app_name_is_not_taken_as_a_list(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["pc-audio", "--apps", "discord", "firefox"])

    assert exit_info.value.code == 2
    assert "invalid choice: 'firefox'" in capsys.readouterr().err
    assert not config.load().audio.listen_to_pc


@pytest.mark.parametrize("value", ["", " ", ",", "discord,", ",discord", "a,,b"])
def test_an_empty_apps_value_is_refused(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["pc-audio", "--apps", value])

    assert exit_info.value.code == 2
    assert "--all-output" in capsys.readouterr().err
    assert config.load().audio.pc_audio_apps == ["discord"]


def test_apps_and_all_output_together_are_refused(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["pc-audio", "--apps", "discord", "--all-output"])

    assert exit_info.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_the_model_client_follows_the_strata_port(monkeypatch: pytest.MonkeyPatch) -> None:
    config.save(config.Config(strata=config.StrataConfig(port=9100)))
    seen: list[str] = []

    def fake_model(model: config.ModelConfig, _key: str) -> SimpleNamespace:
        seen.append(model.base_url)
        return SimpleNamespace(status=lambda: "unloaded")

    monkeypatch.setattr(cli, "StrataModel", fake_model)

    assert main(["status"]) == 0
    assert seen == ["http://127.0.0.1:9100/v1"]
