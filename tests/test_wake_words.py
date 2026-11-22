"""Choosing which LiveKit wake word models Jarvis listens for (`audio.wake_words`)."""

import logging
from typing import TYPE_CHECKING

import pydantic
import pytest

from jarvis import cli, config
from jarvis.cli import main

if TYPE_CHECKING:
    from pathlib import Path


class _LiveKit:
    """Stands in for LiveKit's model, which needs its files."""

    def __init__(self, paths: list[Path], threshold: float) -> None:
        super().__init__()
        self.paths = paths
        self.threshold = threshold


def _cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **audio: object) -> config.Config:
    import jarvis.audio  # ruff: ignore[import-outside-top-level] - patched where cli reads it

    monkeypatch.setattr(jarvis.audio, "WakeWord", _LiveKit)
    cfg = config.Config(data_dir=tmp_path)
    return cfg.model_copy(update={"audio": cfg.audio.model_copy(update=audio)})


def _models(folder: Path, *names: str) -> None:
    (folder / "wake").mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / "wake" / f"{name}.onnx").write_bytes(b"model")


def test_hey_jarvis_and_yo_jarvis_are_on_by_default() -> None:
    assert config.AudioConfig().wake_words == ["hey_jarvis", "yo_jarvis"]


@pytest.mark.parametrize(
    "words",
    [[], ["../hey_jarvis"], ["hey jarvis"], ["Hey_Jarvis"], ["hey_jarvis", "hey_jarvis"], [""]],
)
def test_a_wake_word_list_that_is_empty_a_path_or_repeats_is_refused(words: list[str]) -> None:
    with pytest.raises(pydantic.ValidationError):
        config.AudioConfig(wake_words=words)


def test_only_the_wake_words_that_are_on_are_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _models(tmp_path, "hey_jarvis", "jarvis", "yo_jarvis")
    cfg = _cfg(monkeypatch, tmp_path, wake_words=["yo_jarvis", "hey_jarvis"], wake_threshold=0.6)

    wake = cli.wake_word(cfg)

    assert isinstance(wake, _LiveKit)
    wake_dir = tmp_path / "wake"
    assert wake.paths == [wake_dir / "yo_jarvis.onnx", wake_dir / "hey_jarvis.onnx"]
    assert wake.threshold == pytest.approx(0.6)


def test_a_wake_word_with_no_model_file_is_left_out_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _models(tmp_path, "hey_jarvis")
    cfg = _cfg(monkeypatch, tmp_path)

    with caplog.at_level(logging.WARNING):
        wake = cli.livekit_wake(cfg)

    assert isinstance(wake, _LiveKit)
    assert wake.paths == [tmp_path / "wake" / "hey_jarvis.onnx"]
    assert "not listening for yo_jarvis" in caplog.text


def test_with_no_model_file_at_all_every_file_is_passed_on_so_loading_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wake = cli.livekit_wake(_cfg(monkeypatch, tmp_path))

    assert isinstance(wake, _LiveKit)
    assert [path.name for path in wake.paths] == ["hey_jarvis.onnx", "yo_jarvis.onnx"]


def test_the_doctor_checks_the_file_of_each_wake_word_that_is_on(tmp_path: Path) -> None:
    cfg = config.Config(data_dir=tmp_path)
    cfg = cfg.model_copy(update={"audio": cfg.audio.model_copy(update={"wake_words": ["jarvis"]})})

    roles = cli._model_files(cfg)  # pyright: ignore[reportPrivateUsage] - the doctor's file list

    assert ("wake", tmp_path / "wake" / "jarvis.onnx") in roles
    assert not any(path.name == "hey_jarvis.onnx" for _role, path in roles)


def test_the_wake_command_turns_wake_words_on_and_lists_every_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    restarts: list[bool] = []
    monkeypatch.setattr(cli, "_restart_if_running", lambda: restarts.append(True))
    _models(config.load().data_dir, "hey_jarvis", "jarvis", "yo_jarvis")

    assert main(["wake", "--words", "jarvis, hey_jarvis.onnx"]) == 0

    assert config.load().audio.wake_words == ["jarvis", "hey_jarvis"]
    assert restarts == [True]
    out = capsys.readouterr().out
    assert "LiveKit jarvis, hey_jarvis" in out
    assert "  on  jarvis\n" in out
    assert "  on  hey_jarvis\n" in out
    assert "  off yo_jarvis\n" in out


def test_the_wake_command_shows_a_wake_word_that_is_on_but_has_no_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_restart_if_running", lambda: pytest.fail("no restart"))
    _models(config.load().data_dir, "hey_jarvis")

    assert main(["wake"]) == 0

    out = capsys.readouterr().out
    assert "  on  hey_jarvis\n" in out
    assert "  on  yo_jarvis (no file " in out


@pytest.mark.parametrize("words", ["hey_jarvis,hey_jarvis", "../secret"])
def test_the_wake_command_refuses_a_bad_list_and_keeps_the_setting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], words: str
) -> None:
    monkeypatch.setattr(cli, "_restart_if_running", lambda: pytest.fail("no restart"))
    before = config.load().audio.wake_words

    assert main(["wake", "--words", words]) == 2

    assert "wake value refused" in capsys.readouterr().err
    assert config.load().audio.wake_words == before


def test_the_wake_command_refuses_an_empty_name() -> None:
    with pytest.raises(SystemExit):
        main(["wake", "--words", "hey_jarvis,,yo_jarvis"])
