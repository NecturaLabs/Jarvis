"""The `jarvis` command, run against a throw-away home: never the owner's config or services."""

import contextlib
import hashlib
import json
import signal
import subprocess
import threading
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Self, cast

import numpy as np
import pytest
from strata_fixture import engine_run_config, install_engine

from jarvis import __version__, app, cli, config, models, paths, services
from jarvis import audio as audio_module
from jarvis import voice as voice_module
from jarvis.agent import Hooks
from jarvis.audit import AuditLog
from jarvis.chat.app import ChatApp
from jarvis.cli import main
from jarvis.confirmkey import ConfirmKey
from jarvis.secretfile import write_private_json
from jarvis.services import strata_profile_file

if TYPE_CHECKING:
    import argparse
    from pathlib import Path


real_systemctl = cli._systemctl  # pyright: ignore[reportPrivateUsage] - the `home` fixture replaces it in `cli`


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


pytestmark = pytest.mark.usefixtures("home", "voxtype_lists_no_models")


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

    def fake_model(model: config.ModelConfig, _key: str, **_known: object) -> SimpleNamespace:
        seen.append(model.base_url)
        return SimpleNamespace(status=lambda: "unloaded")

    monkeypatch.setattr(app, "StrataModel", fake_model)  # the client `jarvis status` builds

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


def _cannot_say(_engine: str) -> frozenset[str] | None:
    return None


@pytest.fixture
def voxtype_lists_no_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """`jarvis doctor` asks Voxtype for its models; here it cannot say, so the doctor falls back
    to the file check (tests of the Voxtype answer patch it themselves)."""
    monkeypatch.setattr(audio_module, "installed_models", _cannot_say)


def _doctor_config(data: Path) -> None:
    config.save(config.Config(data_dir=data))


def _doctor_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().out.splitlines()


def test_doctor_lines_up_its_checks_and_ends_with_a_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main(["doctor"])
    lines = _doctor_lines(capsys)

    checks = [line for line in lines if line[:5] in {"ok   ", "FAIL ", "note "}]
    assert checks
    assert all(line[4] == " " and line[5] != " " for line in checks)
    failed = sum(line.startswith("FAIL") for line in lines)
    total = sum(line[:5] in {"ok   ", "FAIL "} for line in lines)
    assert not any(line.startswith("ok  ") and line[4] != " " for line in lines)
    assert lines[-1] == f"{total} checks, {failed} failed"


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
        line.startswith("ok   hey_jarvis.onnx: ") and "Example wake 9.9" in line and "MIT" in line
        for line in out
    )
    assert not any("not a known release" in line for line in out)
    assert out[-1].endswith(" failed")
    total = sum(line[:5] in {"ok   ", "FAIL "} for line in out)
    assert out[-1].startswith(f"{total} checks, ")


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


def test_doctor_verifies_the_picture_reader_when_pictures_are_on(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = engine_run_config(home)
    (home / "models" / "mmproj.gguf").write_bytes(b"reader bytes")
    strata = config.StrataConfig(run_config=run)
    config.save(config.Config(data_dir=home / "models", strata=strata))
    monkeypatch.setattr(models, "KNOWN_MODELS", (_known("vision", b"reader bytes"),))

    cli.main(["doctor"])

    out = _doctor_lines(capsys)
    assert any(line.startswith("ok ") and "mmproj.gguf" in line for line in out)


@pytest.mark.parametrize(
    ("sha256", "licence"),
    [
        ("71f3cbc1f7cc0f30d09d41cfa924c0060827ebc33bf15ace7e86661e856f0160", "Apache-2.0"),
        (
            "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0",
            "LicenseRef-Qwen-Community-License-1.0",
        ),
        ("6589eff47f9c20bc3a032faa3d449f10bf4f04a84cc6b2ee8a44d51d2b6fc677", "Apache-2.0"),
    ],
)
def test_each_picture_reader_pin_is_the_release_strata_names(sha256: str, licence: str) -> None:
    (pin,) = [m for m in models.KNOWN_MODELS if m.role == "vision" and m.sha256 == sha256]
    assert pin.licence == licence


def test_doctor_still_fails_a_missing_file_and_points_at_the_models_doc(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _doctor_config(home / "models")

    assert cli.main(["doctor"]) == 1

    out = _doctor_lines(capsys)
    assert any(
        line.startswith("FAIL wake word files: none found in ") and "see docs/models.md" in line
        for line in out
    )
    assert not any("docs/playbook" in line for line in out)
    assert not any("not a known release" in line for line in out)


@pytest.mark.parametrize(
    ("state", "line"),
    [(None, "FAIL lock screen state (unknown)"), (False, "ok   lock screen state (unlocked)")],
)
def test_doctor_checks_the_lock_screen_can_be_read_when_jarvis_must_not_listen_there(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    state: bool | None,
    line: str,
) -> None:
    config.save(config.Config(data_dir=home, audio=config.AudioConfig(listen_when_locked=False)))
    monkeypatch.setattr(audio_module, "lock_state", lambda: state)

    cli.main(["doctor"])

    assert any(shown.startswith(line) for shown in _doctor_lines(capsys))


def test_doctor_leaves_the_lock_screen_alone_while_jarvis_listens_there(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def asked() -> bool | None:
        pytest.fail("the lock screen was asked about")

    _doctor_config(home)
    monkeypatch.setattr(audio_module, "lock_state", asked)

    cli.main(["doctor"])

    assert not any("lock screen" in shown for shown in _doctor_lines(capsys))


_MIC_NODE = "alsa_input.usb-Example_Mic-00.analog-stereo"


def _fake_pw_dump(monkeypatch: pytest.MonkeyPatch, names: list[str]) -> None:
    """Make `pw-dump` list one microphone node per name (the real tool is never run)."""
    nodes = [
        {
            "id": 10 + number,
            "type": "PipeWire:Interface:Node",
            "info": {
                "props": {
                    "media.class": "Audio/Source",
                    "node.name": name,
                    "object.serial": 100 + number,
                }
            },
        }
        for number, name in enumerate(names)
    ]
    monkeypatch.setattr(audio_module, "pw_dump", lambda: nodes)


@pytest.mark.parametrize(
    ("source", "listed", "line"),
    [
        (_MIC_NODE, [_MIC_NODE], "ok "),
        ("100", [_MIC_NODE], "ok "),
        ("alsa_input.usb-Typo_Mic-00.analog-stereo", [_MIC_NODE], "FAIL"),
        (_MIC_NODE, [], "FAIL"),
    ],
)
def test_doctor_checks_that_the_configured_source_is_a_microphone_node(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    source: str,
    listed: list[str],
    line: str,
) -> None:
    config.save(config.Config(data_dir=home, audio=config.AudioConfig(source=source)))
    _fake_pw_dump(monkeypatch, listed)

    cli.main(["doctor"])

    (shown,) = [row for row in _doctor_lines(capsys) if f"audio.source {source}" in row]
    assert shown.startswith(line)
    assert ("audio.source" in shown and "pw-dump" in shown) is (line == "FAIL")


def test_doctor_says_when_pipewire_cannot_be_read_rather_than_blaming_the_source(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config.save(config.Config(data_dir=home, audio=config.AudioConfig(source=_MIC_NODE)))
    monkeypatch.setattr(audio_module, "pw_dump", list)  # what `pw_dump` returns when it fails

    cli.main(["doctor"])

    (shown,) = [row for row in _doctor_lines(capsys) if f"audio.source {_MIC_NODE}" in row]
    assert shown.startswith("FAIL")
    assert "PipeWire" in shown
    assert "matches no microphone" not in shown


def test_doctor_leaves_the_default_microphone_alone(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def listed() -> object:
        pytest.fail("the PipeWire graph was read")

    _doctor_config(home)
    monkeypatch.setattr(audio_module, "pw_dump", listed)

    cli.main(["doctor"])

    assert not any("audio.source" in shown for shown in _doctor_lines(capsys))


def test_the_default_speech_model_is_small_en() -> None:
    assert config.AudioConfig().transcribe_model == "small.en"


def test_doctor_asks_voxtype_whether_the_speech_model_is_installed(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installed: frozenset[str] = frozenset()
    asked: list[str] = []

    def models(engine: str) -> frozenset[str]:
        asked.append(engine)
        return installed

    monkeypatch.setattr(audio_module, "installed_models", models)
    _doctor_config(home / "models")

    cli.main(["doctor"])

    assert any(
        line.startswith("FAIL") and "small.en" in line and "voxtype setup model" in line
        for line in _doctor_lines(capsys)
    )
    installed = frozenset({"small.en"})

    cli.main(["doctor"])

    assert any(
        line.startswith("ok ") and "speech model small.en (Voxtype reports it installed)" in line
        for line in _doctor_lines(capsys)
    )
    assert asked == ["whisper", "whisper"]


def test_doctor_checks_the_whisper_model_is_downloaded(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A Voxtype that cannot list its models (the fixture above): the doctor looks for the file.
    _doctor_config(home / "models")

    cli.main(["doctor"])

    out = _doctor_lines(capsys)
    assert any(
        line.startswith("FAIL") and "ggml-small.en.bin" in line and "voxtype setup model" in line
        for line in out
    )
    models_dir = home / "data" / "voxtype" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "ggml-small.en.bin").write_bytes(b"model")

    cli.main(["doctor"])

    assert any(
        line.startswith("ok ") and "ggml-small.en.bin" in line for line in _doctor_lines(capsys)
    )


def test_doctor_leaves_other_speech_engines_to_voxtype(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = config.AudioConfig(
        transcribe_engine="parakeet", transcribe_model="parakeet-tdt-0.6b-v3"
    )
    config.save(config.Config(data_dir=home / "models", audio=audio))

    cli.main(["doctor"])

    assert not any("ggml-" in line for line in _doctor_lines(capsys))


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
    run = engine_run_config(home)
    install_engine(home / "data")
    config.save(config.Config(strata=config.StrataConfig(vram_reserve_mib=12000)))

    assert main(["setup", "--run-config", str(run), "--spotify-client-id", "abc"]) == 0

    assert config.load().strata.run_config == run
    assert (home / "config" / "systemd" / "user" / "jarvis.service").is_file()
    assert (home / "config" / "jarvis" / "strata-profile.json").is_file()
    assert (home / "state" / "jarvis" / "strata.env").is_file()
    assert calls == [["systemctl", "--user", "daemon-reload"]]
    assert "next: jarvis spotify-login" in capsys.readouterr().out


def test_setup_with_enable_clears_a_failed_state_before_starting_the_units(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_subprocess(monkeypatch)
    run = engine_run_config(home)
    install_engine(home / "data")
    config.save(config.Config(strata=config.StrataConfig(vram_reserve_mib=12000)))

    assert main(["setup", "--run-config", str(run), "--enable"]) == 0

    units = ["jarvis-strata.service", "jarvis.service"]
    assert calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "reset-failed", *units],
        ["systemctl", "--user", "enable", "--now", *units],
    ]


def test_a_config_command_clears_a_failed_state_before_restarting_jarvis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The start limit counts restarts too: five config changes in 15 minutes must not leave the
    # voice service stopped.
    calls: list[tuple[str, ...]] = []

    def systemctl(*args: str) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(cli, "_systemctl", systemctl)

    def active(_unit: str) -> str:
        return "active"

    monkeypatch.setattr(cli, "_unit_state", active)

    cli._restart_if_running()  # pyright: ignore[reportPrivateUsage] - the helper three commands share

    assert calls == [("reset-failed", "jarvis.service"), ("restart", "jarvis.service")]


def test_setup_refuses_a_value_that_cannot_go_into_a_unit(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_subprocess(monkeypatch)

    assert main(["setup", "--run-config", "/models/a\nExecStart=/bin/evil"]) == 2

    assert "run_config" in capsys.readouterr().err
    assert calls == []
    assert not (home / "config").exists()


def _login(authorized_days_ago: float | None) -> None:
    record: dict[str, object] = {
        "access_token": "example-access",
        "refresh_token": "example-refresh",
        "expires_at": time.time() + 3600,
        "scope": "",
    }
    if authorized_days_ago is not None:
        record["authorized_at"] = time.time() - authorized_days_ago * 86400
    write_private_json(paths.spotify_token_file(), record)


def _login_line(capsys: pytest.CaptureFixture[str]) -> str:
    cli.main(["doctor"])
    (line,) = [line for line in _doctor_lines(capsys) if "Spotify login" in line]
    return line


@pytest.mark.parametrize(
    ("days_ago", "ok", "shown"),
    [
        (9.5, True, "173 days"),
        (174.5, True, "8 days"),
        (200, False, "expired"),
        (None, True, "unknown"),
    ],
)
def test_doctor_says_how_long_the_spotify_login_lasts(
    home: Path,
    capsys: pytest.CaptureFixture[str],
    days_ago: float | None,
    *,
    ok: bool,
    shown: str,
) -> None:
    _doctor_config(home / "models")
    _login(days_ago)

    line = _login_line(capsys)

    assert line.startswith("ok ") is ok
    assert shown in line


def test_doctor_without_a_spotify_login_fails_that_line(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _doctor_config(home / "models")

    assert _login_line(capsys).startswith("FAIL")


def _opened_chat(monkeypatch: pytest.MonkeyPatch) -> list[ChatApp]:
    opened: list[ChatApp] = []

    def run(self: ChatApp, *_args: object, **_kwargs: object) -> None:
        opened.append(self)

    monkeypatch.setattr(ChatApp, "run", run)
    return opened


def test_chat_opens_the_window_with_typing_off_until_setup_is_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = _opened_chat(monkeypatch)

    assert main(["chat"]) == 0

    (app,) = opened
    assert app._session is None  # pyright: ignore[reportPrivateUsage] - the chat app's session, held until the first request
    assert "jarvis setup" in app._unavailable  # pyright: ignore[reportPrivateUsage] - why the chat app is unavailable, shown in its window


def test_chat_sends_typed_requests_to_an_agent_that_marks_its_audit_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = _opened_chat(monkeypatch)
    built: list[tuple[str | None, str]] = []

    def build(_config: object, *, via: str | None = None, session: str = "") -> object:
        built.append((via, session))

        def handle(text: str, _hooks: Hooks) -> str:
            return f"heard {text}"

        return SimpleNamespace(agent=SimpleNamespace(handle=handle))

    monkeypatch.setattr(cli, "build", build)

    assert main(["chat"]) == 0

    (app,) = opened
    ((via, session),) = built
    assert via == "chat"
    assert session
    assert app._session_id == session  # pyright: ignore[reportPrivateUsage] - the chat app's session id, which no public name shows
    agent = app._session  # pyright: ignore[reportPrivateUsage] - the chat app's agent, built on the first request
    assert agent is not None
    assert agent.handle("pause", Hooks(confirm=lambda _q: False)) == "heard pause"


def _transcribed(monkeypatch: pytest.MonkeyPatch, text: str) -> list[tuple[str, Path | None]]:
    heard: list[tuple[str, Path | None]] = []

    def transcribe(
        _samples: object, _engine: str, _model: str, hint: str, *, settings: Path | None = None
    ) -> str:
        heard.append((hint, settings))
        return text

    monkeypatch.setattr(audio_module, "transcribe", transcribe)
    return heard


SAMPLES = np.zeros(1600, dtype=np.int16)


def test_requests_are_transcribed_with_the_hint_and_answers_without(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heard = _transcribed(monkeypatch, "pause")
    request, answer = cli._transcribers(config.Config(), lambda: "Jarvis. play, pause.")  # pyright: ignore[reportPrivateUsage] - the helper that builds both transcribers

    assert request(SAMPLES) == "pause"
    assert answer(SAMPLES) == "pause"

    settings = paths.config_dir() / "voxtype.toml"
    assert [hint for hint, _ in heard] == ["Jarvis. play, pause.", ""]
    assert {used for _, used in heard} == {settings}
    assert settings.read_text() == audio_module.voxtype_settings(8, short_window=True)


def test_a_transcript_that_only_repeats_the_hint_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hint = "Jarvis. play, pause, skip, queue songs."
    _transcribed(monkeypatch, "play, pause, skip, queue songs.")
    request, _ = cli._transcribers(config.Config(), lambda: hint)  # pyright: ignore[reportPrivateUsage] - the helper that builds both transcribers

    assert not request(SAMPLES)


def test_without_the_vocabulary_hint_requests_get_none(monkeypatch: pytest.MonkeyPatch) -> None:
    heard = _transcribed(monkeypatch, "pause")
    cfg = config.Config(audio=config.AudioConfig(vocabulary_hint=False))
    request, _ = cli._transcribers(cfg, lambda: "Jarvis.")  # pyright: ignore[reportPrivateUsage] - the helper that builds both transcribers

    request(SAMPLES)

    assert not heard[0][0]


class _Part:
    """A stand-in for a device or listener: records that it was started and stopped."""

    def __init__(self, *_args: object, **kwargs: object) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.started = False
        self.stopped = threading.Event()

    def start(self) -> Self:
        self.started = True
        return self

    def stop(self) -> None:
        self.stopped.set()

    @staticmethod
    def dictating() -> bool | None:
        return False

    @staticmethod
    def meeting() -> bool:
        return False


@dataclass
class _Run:
    """What `cmd_run` built and did, with a voice loop scripted by the test."""

    handlers: dict[int, Callable[[int, object], None]] = field(
        default_factory=dict[int, Callable[[int, object], None]]
    )
    parts: list[_Part] = field(default_factory=list[_Part])
    kinds: dict[str, list[_Part]] = field(default_factory=dict[str, list[_Part]])
    refresh: list[threading.Event] = field(default_factory=list[threading.Event])
    warmed: threading.Event = field(default_factory=threading.Event)
    loops: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    audit: AuditLog = field(default_factory=lambda: AuditLog(None))

    def part(self, *args: object, **kwargs: object) -> _Part:
        made = _Part(*args, **kwargs)
        self.parts.append(made)
        return made

    def maker(self, kind: str) -> Callable[..., _Part]:
        def make(*args: object, **kwargs: object) -> _Part:
            made = self.part(*args, **kwargs)
            self.kinds.setdefault(kind, []).append(made)
            return made

        return make


def _cmd_run(
    monkeypatch: pytest.MonkeyPatch,
    body: Callable[[_Run, Callable[[], bool]], None],
    cfg: config.Config | None = None,
    run: _Run | None = None,
) -> tuple[_Run, int]:
    run = run or _Run()
    for name in ("Microphone", "Speaker", "Vad", "WakeWord", "AppAudio", "VoxtypeWatch"):
        monkeypatch.setattr(audio_module, name, run.maker(name))

    def no_voice(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(audio_module, "make_synthesizer", no_voice)
    _transcribed(monkeypatch, "")

    def signal_handler(signum: int, handler: Callable[[int, object], None]) -> None:
        run.handlers[signum] = handler

    monkeypatch.setattr(cli.signal, "signal", signal_handler)
    monkeypatch.setattr(cli, "_load", lambda: cfg or config.Config())
    model = SimpleNamespace(
        warm=lambda **_kw: run.warmed.set(),  # pyright: ignore[reportUnknownLambdaType] - a stand-in
        is_loaded=lambda: True,
        load=lambda: True,
        status=lambda: "failed",  # what a suspend can leave behind
        file_missing=lambda: False,
    )
    ops = SimpleNamespace(start_refresh=run.refresh.append)
    jarvis = SimpleNamespace(agent=object(), model=model, ops=ops, hint=lambda: "", audit=run.audit)

    def build(_cfg: config.Config) -> SimpleNamespace:
        return jarvis

    monkeypatch.setattr(cli, "build", build)

    class Loop:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            super().__init__()
            run.loops.append(kwargs)

        @staticmethod
        def run(should_stop: Callable[[], bool]) -> None:
            body(run, should_stop)

    monkeypatch.setattr(voice_module, "VoiceLoop", Loop)
    return run, main(["run"])


def test_a_stop_signal_only_raises_the_flag_and_the_stopper_ends_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def body(run: _Run, should_stop: Callable[[], bool]) -> None:
        assert not should_stop()
        run.handlers[signal.SIGTERM](signal.SIGTERM, None)
        assert should_stop()
        assert all(part.stopped.wait(5) for part in run.parts if part.started)

    cfg = config.Config(audio=config.AudioConfig(listen_to_pc=True))
    run, code = _cmd_run(monkeypatch, body, cfg)

    assert code == 0
    assert set(run.handlers) == {signal.SIGTERM, signal.SIGINT}
    (refresh,) = run.refresh
    assert refresh.wait(5)
    assert run.warmed.wait(5)
    (mic,), (speaker,), (pc_audio,) = (run.kinds[k] for k in ("Microphone", "Speaker", "AppAudio"))
    assert mic.started
    assert all(part.stopped.is_set() for part in (mic, speaker, pc_audio))
    assert run.loops[0]["pc"] is not None


def test_a_loop_failure_after_a_stop_ends_the_run_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def body(run: _Run, _should_stop: Callable[[], bool]) -> None:
        run.handlers[signal.SIGINT](signal.SIGINT, None)
        msg = "the microphone stream ended"
        raise audio_module.AudioError(msg)

    _, code = _cmd_run(monkeypatch, body)

    assert code == 0


def test_a_loop_failure_without_a_stop_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def body(_run: _Run, _should_stop: Callable[[], bool]) -> None:
        msg = "the microphone stream ended"
        raise audio_module.AudioError(msg)

    with pytest.raises(audio_module.AudioError):
        _cmd_run(monkeypatch, body)


def test_the_run_reopens_a_lost_microphone_and_notices_one_gone_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _ = _cmd_run(monkeypatch, lambda _run, _should_stop: None)

    (mic,) = run.kinds["Microphone"]
    assert mic.kwargs["stall_s"] == audio_module.MIC_STALL_S
    assert callable(run.loops[0]["reopen"])


def test_the_run_follows_voxtypes_status_off_the_voice_thread_and_stops_it_with_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def body(run: _Run, should_stop: Callable[[], bool]) -> None:
        run.handlers[signal.SIGTERM](signal.SIGTERM, None)
        assert should_stop()
        assert run.kinds["VoxtypeWatch"][0].stopped.wait(5)

    run, code = _cmd_run(monkeypatch, body)

    assert code == 0
    (watch,) = run.kinds["VoxtypeWatch"]
    assert watch.started
    kwargs = run.loops[0]
    assert (kwargs["dictating"], kwargs["meeting"]) == (watch.dictating, watch.meeting)
    assert kwargs["audit"] is run.audit


def test_a_failure_before_the_loop_runs_leaves_no_status_reader_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def serve(*_args: object, **_kwargs: object) -> object:
        msg = "the control socket is taken"
        raise OSError(msg)

    monkeypatch.setattr(cli, "_serve_controls", serve)
    run = _Run()

    with pytest.raises(OSError, match="socket is taken"):
        _cmd_run(monkeypatch, lambda _run, _should_stop: None, run=run)

    (watch,) = run.kinds["VoxtypeWatch"]
    assert not watch.started or watch.stopped.is_set()


@pytest.mark.parametrize("phoneme_map", [False, True])
def test_the_speaker_is_built_with_the_kokoro_phoneme_map_setting(
    monkeypatch: pytest.MonkeyPatch, *, phoneme_map: bool
) -> None:
    made: list[tuple[object, ...]] = []

    def synthesizer(*args: object, **kwargs: object) -> None:
        made.append((*args, kwargs))

    monkeypatch.setattr(audio_module, "make_synthesizer", synthesizer)

    def speaker(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(audio_module, "Speaker", speaker)
    cfg = config.Config(audio=config.AudioConfig(barge_in=False, kokoro_phoneme_map=phoneme_map))

    cli._speaker(cfg)  # pyright: ignore[reportPrivateUsage] - the helper that builds the speaker from config

    ((_data, voice, speed, options),) = made
    assert (voice, speed, options) == ("kokoro:bm_george", 1.0, {"phoneme_map": phoneme_map})


def test_the_lock_screen_is_not_asked_about_while_listening_there_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def asked() -> bool:
        pytest.fail("the lock screen was asked about")

    monkeypatch.setattr(audio_module, "screen_locked", asked)
    run, _ = _cmd_run(monkeypatch, lambda _run, _should_stop: None)

    locked = run.loops[0]["locked"]
    assert callable(locked)
    assert locked() is False


def test_with_listening_at_the_lock_screen_off_the_run_asks_the_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_module, "screen_locked", lambda: True)
    cfg = config.Config(audio=config.AudioConfig(listen_when_locked=False))
    run, _ = _cmd_run(monkeypatch, lambda _run, _should_stop: None, cfg)

    locked = run.loops[0]["locked"]
    assert callable(locked)
    assert locked() is True


def test_after_a_suspend_the_run_loads_a_failed_model_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def body(run: _Run, _should_stop: Callable[[], bool]) -> None:
        assert run.warmed.wait(5)  # the load at start
        run.warmed.clear()
        after_sleep = run.loops[0]["after_sleep"]
        assert callable(after_sleep)
        after_sleep()
        assert run.warmed.wait(5)

    _, code = _cmd_run(monkeypatch, body)

    assert code == 0


def test_pc_audio_is_not_followed_unless_it_is_switched_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _ = _cmd_run(monkeypatch, lambda _run, _should_stop: None)

    assert run.loops[0]["pc"] is None


def _restart_calls(
    monkeypatch: pytest.MonkeyPatch, started_at: float | None
) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def systemctl(*args: str) -> int:
        calls.append(args)
        return 0

    def started(_unit: str) -> float | None:
        return started_at

    monkeypatch.setattr(cli, "_systemctl", systemctl)
    monkeypatch.setattr(cli, "_started_at", started)
    assert cli.main(["restart"]) == 0
    return calls


def test_restart_restarts_strata_when_its_settings_changed_after_it_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = strata_profile_file()
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text("{}")

    calls = _restart_calls(monkeypatch, profile.stat().st_mtime - 60)

    assert ("restart", "jarvis-strata.service") in calls
    assert ("restart", "jarvis.service") in calls


def test_restart_leaves_a_running_strata_alone_when_nothing_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = strata_profile_file()
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text("{}")

    calls = _restart_calls(monkeypatch, profile.stat().st_mtime + 60)

    assert ("start", "jarvis-strata.service") in calls
    assert ("restart", "jarvis-strata.service") not in calls


def test_restart_writes_a_context_changed_in_the_config_to_strata_and_restarts_it(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: only `jarvis setup` wrote Strata's profile, so a context changed in the config
    # never reached the server, which kept its old window while Jarvis planned for the new one.
    run = engine_run_config(home)
    install_engine(home / "data")
    cfg = config.Config(
        strata=config.StrataConfig(run_config=run, context=24576, vram_reserve_mib=12000)
    )
    services.write_strata_profile(cfg)
    services.write_units(cfg, cli.strata_key_file())
    written = strata_profile_file().stat().st_mtime
    config.save(cfg.model_copy(update={"strata": cfg.strata.model_copy(update={"context": 49152})}))

    calls = _restart_calls(monkeypatch, written)  # Strata started on the old profile

    args = json.loads(strata_profile_file().read_text(encoding="utf-8"))["args"]
    assert args[args.index("--max-context") + 1] == "49152"
    assert ("daemon-reload",) in calls
    assert ("restart", "jarvis-strata.service") in calls


def test_restart_leaves_strata_loaded_when_the_config_holds_what_it_runs(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = engine_run_config(home)
    install_engine(home / "data")
    cfg = config.Config(strata=config.StrataConfig(run_config=run, vram_reserve_mib=12000))
    config.save(cfg)
    services.write_strata_profile(cfg)
    services.write_units(cfg, cli.strata_key_file())

    calls = _restart_calls(monkeypatch, time.time() + 1)

    assert ("restart", "jarvis-strata.service") not in calls


def test_restart_starts_a_stopped_strata(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _restart_calls(monkeypatch, None)

    assert ("start", "jarvis-strata.service") in calls


def _states(monkeypatch: pytest.MonkeyPatch, **states: str) -> None:
    """Answer `systemctl is-active` per unit (`jarvis`, `strata`), "inactive" otherwise."""
    names = {"jarvis.service": "jarvis", "jarvis-strata.service": "strata"}

    def state(unit: str) -> str:
        return states.get(names[unit], "inactive")

    monkeypatch.setattr(cli, "_unit_state", state)


def test_start_clears_a_failed_state_before_starting_both_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A unit that hit its start limit stays `failed` until reset (#33).
    calls: list[tuple[str, ...]] = []

    def systemctl(*args: str) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(cli, "_systemctl", systemctl)
    _states(monkeypatch, jarvis="failed", strata="failed")

    assert cli.main(["start"]) == 0
    assert calls == [
        ("reset-failed", "jarvis-strata.service", "jarvis.service"),
        ("start", "jarvis-strata.service", "jarvis.service"),
    ]


def test_start_resets_only_a_unit_that_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: `reset-failed` on a stopped unit systemd had unloaded printed
    # "Unit jarvis.service not loaded" on every `jarvis start`.
    calls: list[tuple[str, ...]] = []

    def systemctl(*args: str) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(cli, "_systemctl", systemctl)
    _states(monkeypatch, jarvis="inactive", strata="active")

    assert cli.main(["start"]) == 0
    assert calls == [("start", "jarvis-strata.service", "jarvis.service")]


def test_a_failed_reset_does_not_stop_start(monkeypatch: pytest.MonkeyPatch) -> None:
    # `reset-failed` fails for a unit that is not loaded; `start` then says what is wrong.
    calls: list[tuple[str, ...]] = []

    def systemctl(*args: str) -> int:
        calls.append(args)
        return 1 if args[0] == "reset-failed" else 0

    monkeypatch.setattr(cli, "_systemctl", systemctl)
    _states(monkeypatch, jarvis="failed")

    assert cli.main(["start"]) == 0
    assert calls[-1][0] == "start"


def test_restart_clears_a_failed_state_before_starting_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _states(monkeypatch, jarvis="failed", strata="failed")
    calls = _restart_calls(monkeypatch, None)

    assert calls[0] == ("reset-failed", "jarvis-strata.service", "jarvis.service")


def test_a_strata_start_time_is_read_to_the_microsecond() -> None:
    assert cli.parse_systemd_time("Sat 2026-11-21 04:40:30.157367 UTC") == pytest.approx(
        1795236030.157367
    )
    assert cli.parse_systemd_time("") is None
    assert cli.parse_systemd_time("n/a") is None


def test_files_written_just_before_strata_started_in_the_same_second_are_no_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = strata_profile_file()
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text("{}")

    calls = _restart_calls(monkeypatch, profile.stat().st_mtime + 0.001)

    assert ("restart", "jarvis-strata.service") not in calls


def test_doctor_fails_a_saved_model_url_that_misses_the_strata_port(
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths.config_file().parent.mkdir(parents=True, exist_ok=True)
    paths.config_file().write_text(
        '[model]\nbase_url = "http://127.0.0.1:8080/v1"\n[strata]\nport = 9100\n', encoding="utf-8"
    )

    assert cli.main(["doctor"]) == 1

    assert any(
        line.startswith("FAIL") and "model.base_url" in line and "9100" in line
        for line in _doctor_lines(capsys)
    )


def test_doctor_takes_a_full_path_as_the_speech_model(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = home / "whisper" / "ggml-custom.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    audio = config.AudioConfig(transcribe_model=str(model))
    config.save(config.Config(data_dir=home / "models", audio=audio))

    cli.main(["doctor"])

    assert any(line.startswith("ok ") and str(model) in line for line in _doctor_lines(capsys))


def test_screenshots_are_looked_for_where_omarchy_saves_them_then_in_pictures(
    tmp_path: Path,
) -> None:
    home = tmp_path
    env = {"OMASNAP_SCREENSHOT_DIR": str(tmp_path / "Shots")}

    assert cli._screenshot_folders(home, env) == (  # pyright: ignore[reportPrivateUsage] - the helper that finds the screenshot folders
        tmp_path / "Shots",
        home / "Pictures" / "Screenshots",
        home / "Pictures",
    )
    assert cli._screenshot_folders(home, {}) == (  # pyright: ignore[reportPrivateUsage] - the helper that finds the screenshot folders
        home / "Pictures" / "Screenshots",
        home / "Pictures",
    )


def test_consent_toggle_saves_the_setting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    restarts: list[bool] = []
    monkeypatch.setattr(cli, "_restart_if_running", lambda: restarts.append(True))

    assert main(["consent"]) == 0
    assert "bypassed" in capsys.readouterr().out
    assert main(["consent", "ask"]) == 0
    assert config.load().boundaries.bypass_consent is False
    assert "asks" in capsys.readouterr().out
    assert main(["consent", "bypass"]) == 0
    assert config.load().boundaries.bypass_consent is True
    assert len(restarts) == 2


def test_a_consent_change_is_audited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_restart_if_running", lambda: None)

    main(["consent", "ask"])
    main(["consent"])  # only shows the setting
    main(["consent", "bypass"])

    events = [json.loads(line) for line in paths.audit_file().read_text().splitlines()]
    assert [(e["event"], e["bypass_consent"]) for e in events] == [
        ("consent_changed", False),
        ("consent_changed", True),
    ]


def test_the_voice_service_records_its_consent_mode_when_it_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, code = _cmd_run(monkeypatch, lambda _run, _should_stop: None)

    assert code == 0
    assert [(e["event"], e["bypass_consent"]) for e in run.audit.events][:1] == [
        ("service_started", True)
    ]


def test_a_systemctl_call_that_hangs_is_given_up(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    timeouts: list[object] = []

    def hangs(cmd: list[str], **kwargs: object) -> object:
        timeouts.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(cmd, 1.0)

    monkeypatch.setattr(cli.subprocess, "run", hangs)

    assert real_systemctl("stop", "jarvis.service") == 1
    assert "did not finish" in capsys.readouterr().out
    assert timeouts == [cli.SYSTEMCTL_S]


def test_doctor_points_an_unreachable_strata_at_jarvis_start(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # No Strata key here, so the server counts as unreachable. `jarvis start` also clears a start
    # limit, which a bare `systemctl --user start` would be refused by.
    _doctor_config(home / "models")

    cli.main(["doctor"])

    line = next(shown for shown in _doctor_lines(capsys) if "Strata server" in shown)
    assert line.startswith("FAIL")
    assert line.endswith("-> jarvis start (it clears a start limit first)")


def test_the_run_shares_one_confirm_key_between_the_loop_and_the_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: list[object] = []

    def serve(handler: object, _audit: object) -> object:
        served.append(handler)
        return types.SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(cli, "_serve_controls", serve)
    run, _ = _cmd_run(monkeypatch, lambda _run, _should_stop: None)

    key = run.loops[0]["keys"]
    assert isinstance(key, ConfirmKey)
    (handler,) = served
    assert callable(handler)
    waiting = key.arm("hash-a", "stop process 42", 30.0)
    assert waiting is not None
    waiting.armed_at -= 1  # heard a second ago
    assert handler("confirm") == {"confirmed": "stop process 42"}


def test_jarvis_ask_with_yes_has_no_confirm_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `--yes` answers spoken questions only: a `strong` action is refused, as there is no key here.
    seen: list[Hooks] = []

    def handle(_text: str, hooks: Hooks) -> str:
        seen.append(hooks)
        return "I can't."

    agent = types.SimpleNamespace(handle=handle)

    def build(_cfg: object, **_kwargs: object) -> object:
        return types.SimpleNamespace(agent=agent)

    monkeypatch.setattr(cli, "build", build)

    assert main(["ask", "--yes", "stop", "it"]) == 0

    assert seen[0].confirm_key is None
    assert "I can't." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("typed", "allowed"),
    [
        ("y", True),
        ("Y", True),
        (" y ", True),
        ("yes", True),
        ("n", False),
        ("", False),
        ("yn", False),
    ],
)
def test_jarvis_ask_takes_a_typed_y_as_yes(
    monkeypatch: pytest.MonkeyPatch, typed: str, *, allowed: bool
) -> None:
    answers: list[object] = []

    def handle(_text: str, hooks: Hooks) -> str:
        answers.append(hooks.confirm("Should I pause?"))
        return "ok"

    agent = types.SimpleNamespace(handle=handle)

    def build(_cfg: object, **_kwargs: object) -> object:
        return types.SimpleNamespace(agent=agent)

    def typing(_prompt: str) -> str:
        return typed

    monkeypatch.setattr(cli, "build", build)
    monkeypatch.setattr("builtins.input", typing)

    assert main(["ask", "pause"]) == 0

    assert answers == [allowed]


def test_help_lists_the_commands_as_dash_dash_help_does(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["help"]) == 0
    shown = capsys.readouterr().out

    cli.build_parser().print_help()
    assert shown == capsys.readouterr().out
    assert "doctor" in shown


def _help(capsys: pytest.CaptureFixture[str], *args: str) -> str:
    """Return a command's help as a wide colour terminal would get it."""
    with contextlib.suppress(SystemExit):
        cli.main([*args, "--help"])
    return capsys.readouterr().out


def test_help_lists_every_command_once_in_groups_within_80_columns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    shown = _help(capsys)
    lines = shown.splitlines()

    assert lines[0] == "usage: jarvis [<command> [options]]"
    assert "{" not in shown  # no list of every command's name in a brace
    assert "positional arguments" not in shown
    for command in cli.COMMANDS:
        if command != "overview":
            assert sum(line.split()[:1] == [command] for line in lines) == 1, command
    assert all(len(line) <= 80 for line in lines)
    for group in ("Everyday", "Running Jarvis", "Settings", "First-time setup"):
        assert f"  {group}" in lines
    assert "Run `jarvis <command> --help` for a command's options." in shown


@pytest.mark.parametrize("command", ["pause", "doctor", "unload", "chat", "logs", "status"])
def test_each_command_says_what_it_does_in_its_own_help(
    capsys: pytest.CaptureFixture[str], command: str
) -> None:
    lines = _help(capsys, command).splitlines()

    description = lines[2]
    assert description[:1].isupper()
    assert description.endswith(".")


def test_every_argument_of_every_command_is_explained(capsys: pytest.CaptureFixture[str]) -> None:
    parser = cli.build_parser()
    actions = parser._actions
    commands = next(a.choices for a in actions if a.dest == "command")
    for name, command in cast("dict[str, argparse.ArgumentParser]", commands).items():
        for action in command._actions:
            assert action.help, f"{name} {action.dest}"
        assert "_" not in _help(capsys, name).split("options:")[0], name


@pytest.mark.parametrize("command", [None, *[n for _g, c in cli.COMMAND_GROUPS for n, _s in c]])
def test_every_help_page_is_plain_text_within_80_columns_on_a_wide_colour_terminal(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, command: str | None
) -> None:
    monkeypatch.setenv("COLUMNS", "160")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    shown = _help(capsys, *([command] if command else []))

    assert "\x1b[" not in shown
    assert all(len(line) <= 80 for line in shown.splitlines()), command


@pytest.mark.parametrize(
    ("command", "said"),
    [
        ("consent", "confirm key"),
        ("ask", "confirm key"),
        ("restart", "model server"),
        ("spotify-login", "waits"),
        ("pc-audio", "with no argument"),
        ("wake", "with no argument"),
    ],
)
def test_help_says_what_each_option_really_does(
    capsys: pytest.CaptureFixture[str], command: str, said: str
) -> None:
    assert said in " ".join(_help(capsys, command).split()).casefold()


def test_the_overview_asks_systemd_for_each_unit_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked: list[str] = []

    def state(unit: str) -> str:
        asked.append(unit)
        return "inactive"

    monkeypatch.setattr(cli, "_unit_state", state)

    assert main([]) == 0
    capsys.readouterr()
    assert len(asked) <= 4, asked  # each one can take seconds when systemd is slow


def test_a_mistyped_command_gets_a_suggestion(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["stauts"])

    assert "maybe you meant 'status'?" in capsys.readouterr().err


def test_no_arguments_names_the_units_in_plain_words_and_points_at_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == "jarvis: stopped   model server: stopped"
    assert not any(line.startswith("commands:") for line in lines)
    assert "All commands: jarvis help" in lines


def test_setup_without_a_run_config_saves_the_config_and_says_what_comes_next(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_subprocess(monkeypatch)

    assert main(["setup", "--spotify-client-id", "abc"]) == 0

    assert config.load().spotify.client_id == "abc"
    assert "strata launch" in capsys.readouterr().out
    assert not (home / "config" / "systemd" / "user" / "jarvis-strata.service").exists()
    assert calls == []


def test_doctor_fails_a_run_config_it_cannot_use_and_says_how_to_fix_it(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config.save(config.Config(strata=config.StrataConfig(run_config=home / "gone.json")))

    assert cli.main(["doctor"]) == 1

    out = _doctor_lines(capsys)
    assert any(line.startswith("FAIL") and "gone.json" in line for line in out)


def test_doctor_checks_the_learned_expert_profile_when_the_engine_learns(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = engine_run_config(home)
    data = json.loads(run.read_text(encoding="utf-8"))
    data["args"] += ["--expert-profile", str(home / "models" / "q4.gguf")]
    run.write_text(json.dumps(data), encoding="utf-8")
    config.save(config.Config(strata=config.StrataConfig(run_config=run)))

    cli.main(["doctor"])

    out = _doctor_lines(capsys)
    assert any(line.startswith("FAIL") and "expert profile" in line for line in out)


def test_setup_refuses_an_engine_that_would_fill_the_card_and_says_how_to_fix_it(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_subprocess(monkeypatch)
    run = engine_run_config(home)
    install_engine(home / "data")

    assert main(["setup", "--run-config", str(run)]) == 2

    assert "strata.vram_reserve_mib" in capsys.readouterr().err


def test_doctor_names_a_missing_model_drive_and_fails_vram_left_for_games(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = engine_run_config(home)
    (home / "models" / "q4.gguf").unlink()
    config.save(config.Config(strata=config.StrataConfig(run_config=run)))

    cli.main(["doctor"])

    out = _doctor_lines(capsys)
    assert any(
        line.startswith("FAIL") and "q4.gguf" in line and "mount its drive" in line for line in out
    )
    assert any(line.startswith("FAIL") and "VRAM left for games" in line for line in out)


def test_doctor_fails_an_empty_learned_expert_profile(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = engine_run_config(home)
    data = json.loads(run.read_text(encoding="utf-8"))
    data["args"] += ["--expert-profile", str(home / "models" / "q4.gguf")]
    run.write_text(json.dumps(data), encoding="utf-8")
    cfg = config.Config(strata=config.StrataConfig(run_config=run, vram_reserve_mib=12000))
    config.save(cfg)
    learned = services.expert_profile_file(cfg)
    learned.parent.mkdir(parents=True, exist_ok=True)
    learned.write_bytes(b"")

    cli.main(["doctor"])

    out = _doctor_lines(capsys)
    assert any(line.startswith("FAIL") and "expert profile" in line for line in out)
