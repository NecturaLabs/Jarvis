"""Choosing which LiveKit wake word models Jarvis listens for (`audio.wake_words`)."""

from typing import TYPE_CHECKING

import pydantic
import pytest

from jarvis import cli, config
from jarvis.cli import main

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class _LiveKit:
    """Stands in for LiveKit's model, which needs its files."""

    def __init__(
        self, paths: list[Path], threshold: float, thresholds: Mapping[str, float]
    ) -> None:
        super().__init__()
        self.paths = paths
        self.threshold = threshold
        self.thresholds = thresholds


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
    assert wake.thresholds == {}


def test_each_wake_word_s_own_threshold_reaches_the_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _cfg(monkeypatch, tmp_path, wake_words=["jarvis"], wake_thresholds={"jarvis": 0.85})

    wake = cli.livekit_wake(cfg)

    assert isinstance(wake, _LiveKit)
    assert wake.thresholds == {"jarvis": pytest.approx(0.85)}


@pytest.mark.parametrize(
    "thresholds", [{"jarvis": 1.0}, {"jarvis": 0.0}, {"../jarvis": 0.8}, {"Jarvis": 0.8}]
)
def test_a_wake_word_threshold_out_of_range_or_for_a_bad_name_is_refused(
    thresholds: dict[str, float],
) -> None:
    with pytest.raises(pydantic.ValidationError):
        config.AudioConfig(wake_thresholds=thresholds)


def test_wake_word_thresholds_round_trip_through_the_config_file(tmp_path: Path) -> None:
    original = config.Config(
        audio=config.AudioConfig(
            wake_words=["hey_jarvis", "jarvis"], wake_thresholds={"jarvis": 0.85}
        )
    )
    path = tmp_path / "config.toml"

    config.save(original, path)

    assert "[audio.wake_thresholds]" in path.read_text(encoding="utf-8")
    assert config.load(path) == original


def test_the_wake_listing_shows_a_wake_word_s_own_threshold(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_restart_if_running", lambda: pytest.fail("no restart"))
    cfg = config.load()
    _models(cfg.data_dir, "hey_jarvis", "jarvis")
    audio = cfg.audio.model_copy(
        update={"wake_words": ["hey_jarvis", "jarvis"], "wake_thresholds": {"jarvis": 0.85}}
    )
    config.save(cfg.model_copy(update={"audio": audio}))

    assert main(["wake"]) == 0

    out = capsys.readouterr().out
    assert "LiveKit hey_jarvis, jarvis (threshold 0.5)" in out
    assert "  on  hey_jarvis\n" in out
    assert "  on  jarvis (threshold 0.85)\n" in out


def test_every_wake_word_that_is_on_is_passed_on_even_without_a_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _models(tmp_path, "hey_jarvis")

    wake = cli.livekit_wake(_cfg(monkeypatch, tmp_path))

    assert isinstance(wake, _LiveKit)  # WakeWord leaves out what it cannot load
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


def test_the_wake_listing_leaves_out_files_wake_words_cannot_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_restart_if_running", lambda: pytest.fail("no restart"))
    data = config.load().data_dir
    _models(data, "hey_jarvis", "Odd name")
    (data / "wake" / "folder.onnx").mkdir()

    assert main(["wake"]) == 0

    out = capsys.readouterr().out
    assert "Odd name" not in out
    assert "folder" not in out


def test_the_doctor_passes_with_one_wake_word_file_and_notes_the_untrained_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = config.load()
    _models(cfg.data_dir, "hey_jarvis")

    check, notes = cli._wake_checks(cfg)  # pyright: ignore[reportPrivateUsage] - the doctor's wake check

    assert check[0]
    assert check[1] == "wake word files (hey_jarvis)"
    assert len(notes) == 1
    assert notes[0].startswith("note: wake word yo_jarvis is on but has no file ")
    assert "jarvis wake --words" in notes[0]
    assert not capsys.readouterr().out


def test_the_doctor_fails_when_no_wake_word_that_is_on_has_a_file() -> None:
    check, _notes = cli._wake_checks(config.load())  # pyright: ignore[reportPrivateUsage] - the doctor's wake check

    assert not check[0]
    assert check[1] == f"wake word files: none found in {config.load().data_dir / 'wake'}"
    assert check[2] == "see docs/models.md"


def test_the_speakers_own_voice_check_listens_with_the_same_wake_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import jarvis.audio  # ruff: ignore[import-outside-top-level] - patched where cli reads it

    detectors: list[config.Config] = []
    checks: list[object] = []

    def livekit_wake(cfg: config.Config) -> str:
        detectors.append(cfg)
        return "the wake models"

    def own_voice_check(wake: object, _names: object) -> None:
        checks.append(wake)

    def synthesizer(*_args: object, **_kwargs: object) -> object:
        return object()

    def speaker(_synthesizer: object, own_wake: object) -> object:
        return own_wake

    monkeypatch.setattr(cli, "livekit_wake", livekit_wake)
    monkeypatch.setattr(jarvis.audio, "OwnVoiceCheck", own_voice_check)
    monkeypatch.setattr(jarvis.audio, "make_synthesizer", synthesizer)
    monkeypatch.setattr(jarvis.audio, "Speaker", speaker)
    cfg = _cfg(monkeypatch, tmp_path, wake_words=["yo_jarvis"], barge_in=True)

    cli._speaker(cfg)  # pyright: ignore[reportPrivateUsage] - builds the barge-in check

    assert detectors == [cfg]
    assert checks == ["the wake models"]


def test_azure_that_cannot_start_falls_back_to_the_wake_words_that_are_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _cfg(
        monkeypatch,
        tmp_path,
        wake_engine="azure",
        azure_keyword_model=str(tmp_path / "missing.table"),
        wake_words=["jarvis", "yo_jarvis"],
    )

    wake = cli.wake_word(cfg)

    assert isinstance(wake, _LiveKit)
    assert [path.name for path in wake.paths] == ["jarvis.onnx", "yo_jarvis.onnx"]
