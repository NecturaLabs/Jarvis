"""The `jarvis` command, run against a throw-away home: never the owner's config or services."""

import hashlib
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from jarvis import __version__, cli, config, models
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


def _known(role: str, content: bytes) -> models.KnownModel:
    return models.KnownModel(
        role=role,
        project=f"Example {role}",
        version="9.9",
        sha256=hashlib.sha256(content).hexdigest(),
        licence="MIT",
    )


def _doctor_config(data: Path) -> None:
    config.save(config.Config(data_dir=data))


def _doctor_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().out.splitlines()


def test_doctor_names_the_release_of_a_model_file_it_recognises(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = home / "models"
    (data / "wake").mkdir(parents=True)
    (data / "wake" / "hey_jarvis.onnx").write_bytes(b"wake bytes")
    _doctor_config(data)
    monkeypatch.setattr(models, "KNOWN_MODELS", (_known("wake", b"wake bytes"),))

    cli.main(["doctor"])

    out = _doctor_lines(capsys)
    assert any(
        line.startswith("ok ") and "Example wake 9.9" in line and "MIT" in line for line in out
    )
    assert not any("not a known release" in line for line in out)


def test_doctor_notes_an_unknown_digest_without_failing_the_file(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = home / "models"
    (data / "vad").mkdir(parents=True)
    (data / "vad" / "silero_vad.onnx").write_bytes(b"other build")
    _doctor_config(data)
    monkeypatch.setattr(models, "KNOWN_MODELS", (_known("vad", b"the release"),))
    digest = hashlib.sha256(b"other build").hexdigest()

    cli.main(["doctor"])

    out = _doctor_lines(capsys)
    assert any(f"not a known release (sha256 {digest})" in line for line in out)
    assert not any("FAIL" in line and "silero_vad.onnx" in line for line in out)
    assert any(line.startswith("ok ") and "silero_vad.onnx" in line for line in out)


def test_doctor_still_fails_a_missing_file_and_points_at_the_models_doc(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _doctor_config(home / "models")

    assert cli.main(["doctor"]) == 1

    out = _doctor_lines(capsys)
    assert any(
        line.startswith("FAIL") and "hey_jarvis.onnx" in line and "see docs/models.md" in line
        for line in out
    )
    assert not any("docs/playbook" in line for line in out)
    assert not any("not a known release" in line for line in out)


def test_doctor_checks_the_kokoro_files_only_for_a_kokoro_voice(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = home / "models"
    kokoro = data / "voices" / "kokoro"
    kokoro.mkdir(parents=True)
    (kokoro / "kokoro-v1.0.onnx").write_bytes(b"model")
    (kokoro / "voices-v1.0.bin").write_bytes(b"voices")
    config.save(config.Config(data_dir=data))
    monkeypatch.setattr(
        models,
        "KNOWN_MODELS",
        (_known("kokoro-model", b"model"), _known("kokoro-voices", b"voices")),
    )

    cli.main(["doctor"])

    out = _doctor_lines(capsys)
    assert any("Example kokoro-model 9.9" in line for line in out)
    assert any("Example kokoro-voices 9.9" in line for line in out)

    piper = data / "voices"
    (piper / "en_GB-alan-medium.onnx").write_bytes(b"piper")
    (piper / "en_GB-alan-medium.onnx.json").write_bytes(b"{}")
    config.save(config.Config(data_dir=data, audio=config.AudioConfig(voice="en_GB-alan-medium")))

    cli.main(["doctor"])

    out = _doctor_lines(capsys)
    assert not any("Example kokoro" in line or "kokoro-v1.0" in line for line in out)
    assert not any("not a known release" in line and "alan" in line for line in out)


def test_doctor_fails_a_model_file_it_cannot_read(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = home / "models"
    (data / "wake").mkdir(parents=True)
    wake = data / "wake" / "hey_jarvis.onnx"
    wake.write_bytes(b"wake")
    wake.chmod(0)
    _doctor_config(data)

    assert cli.main(["doctor"]) == 1

    assert any(
        line.startswith("FAIL read ") and "hey_jarvis.onnx" in line
        for line in _doctor_lines(capsys)
    )


def _stub_subprocess(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(cli.subprocess, "run", run)
    return calls


def test_setup_writes_the_config_profile_and_units_under_the_home(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_subprocess(monkeypatch)
    model = home / "models" / "q4.gguf"

    assert main(["setup", "--model-file", str(model), "--spotify-client-id", "abc"]) == 0

    assert config.load().strata.model_file == model
    assert (home / "config" / "systemd" / "user" / "jarvis.service").is_file()
    assert (home / "config" / "jarvis" / "strata-profile.json").is_file()
    assert (home / "state" / "jarvis" / "strata.env").is_file()
    assert calls == [["systemctl", "--user", "daemon-reload"]]
    assert "next: jarvis spotify-login" in capsys.readouterr().out


def test_setup_refuses_a_value_that_cannot_go_into_a_unit(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_subprocess(monkeypatch)

    assert main(["setup", "--model-file", "/models/a\nExecStart=/bin/evil"]) == 2

    assert "model_file" in capsys.readouterr().err
    assert calls == []
    assert not (home / "config").exists()
