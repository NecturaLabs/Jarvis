from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from jarvis import __version__, cli, config
from jarvis.cli import main

if TYPE_CHECKING:
    from pathlib import Path


def test_version_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"jarvis {__version__}"


def test_no_arguments_exits_cleanly() -> None:
    assert main([]) == 0


def test_pc_audio_toggle_saves_the_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    restarts: list[bool] = []
    monkeypatch.setattr(cli, "_restart_if_running", lambda: restarts.append(True))

    assert main(["pc-audio", "on"]) == 0
    assert config.load().audio.listen_to_pc
    assert config.load().audio.pc_audio_apps == ["discord"]

    assert main(["pc-audio", "--apps", "discord", "teamspeak"]) == 0
    assert config.load().audio.pc_audio_apps == ["discord", "teamspeak"]

    assert main(["pc-audio", "off"]) == 0
    assert not config.load().audio.listen_to_pc
    assert len(restarts) == 3
    assert "off" in capsys.readouterr().out.splitlines()[-1]


def test_pc_audio_without_arguments_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_restart_if_running", lambda: pytest.fail("no restart"))

    assert main(["pc-audio"]) == 0
    assert not (tmp_path / "jarvis" / "config.toml").exists()


def test_the_model_client_follows_the_strata_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config.save(config.Config(strata=config.StrataConfig(port=9100)))
    seen: list[str] = []

    def fake_model(model: config.ModelConfig, _key: str) -> SimpleNamespace:
        seen.append(model.base_url)
        return SimpleNamespace(status=lambda: "unloaded")

    monkeypatch.setattr(cli, "StrataModel", fake_model)

    assert main(["status"]) == 0
    assert seen == ["http://127.0.0.1:9100/v1"]
