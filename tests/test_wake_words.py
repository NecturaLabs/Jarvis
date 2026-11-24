"""Choosing which LiveKit wake word models Jarvis listens for (`audio.wake_words`)."""

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pydantic
import pytest

from jarvis import audio, cli, config
from jarvis.cli import main

if TYPE_CHECKING:
    from collections.abc import Mapping

    import onnxruntime as ort


class _LiveKit:
    """Stands in for LiveKit's model, which needs its files."""

    def __init__(
        self,
        paths: list[Path],
        threshold: float,
        thresholds: Mapping[str, float],
        over_speech: float | None = None,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.over_speech = over_speech
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


def test_the_barge_in_threshold_reaches_the_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert config.AudioConfig().barge_in_threshold == pytest.approx(0.5)
    cfg = _cfg(monkeypatch, tmp_path, barge_in_threshold=0.4)

    wake = cli.livekit_wake(cfg)

    assert isinstance(wake, _LiveKit)
    assert wake.over_speech == pytest.approx(0.4)


class _Scored:
    """Stands in for LiveKit's model: every frame scores what `score` holds."""

    score: ClassVar[float] = 0.0
    resets: ClassVar[int] = 0

    def __init__(self, *, sess_options: ort.SessionOptions) -> None:
        super().__init__()
        del sess_options

    @staticmethod
    def load_model(path: Path) -> None:
        del path

    @classmethod
    def predict(cls, _frame: object) -> dict[str, float]:
        return {"yo_jarvis": cls.score}

    @classmethod
    def reset(cls) -> None:
        cls.resets += 1


def _scored(monkeypatch: pytest.MonkeyPatch, score: float) -> audio.WakeWord:
    monkeypatch.setattr(_Scored, "score", score)
    monkeypatch.setattr("livekit.wakeword.WakeWordModel", _Scored)
    return audio.WakeWord([Path("yo_jarvis.onnx")], 0.5, {"yo_jarvis": 0.76}, over_speech=0.5)


FRAME_OF_SOUND = np.zeros(audio.FRAME, dtype=np.int16)


def test_while_jarvis_speaks_the_wake_word_needs_only_the_barge_in_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Measured: Jarvis's voice as loud as the owner's drops recall at 0.76 from 99 % to 71 %,
    # while its voice alone never scored above 0.06 (ADR 0012).
    wake = _scored(monkeypatch, 0.6)

    assert not wake.heard(FRAME_OF_SOUND)  # idle, the model's own 0.76 holds
    assert wake.heard_over_speech(FRAME_OF_SOUND)


def test_the_barge_in_score_never_raises_a_lower_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_Scored, "score", 0.45)
    monkeypatch.setattr("livekit.wakeword.WakeWordModel", _Scored)
    wake = audio.WakeWord([Path("yo_jarvis.onnx")], 0.4, over_speech=0.5)

    assert wake.heard_over_speech(FRAME_OF_SOUND)


def test_the_best_score_that_fell_short_over_speech_is_kept_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wake = _scored(monkeypatch, 0.3)

    assert not wake.heard_over_speech(FRAME_OF_SOUND)
    monkeypatch.setattr(_Scored, "score", 0.42)
    assert not wake.heard_over_speech(FRAME_OF_SOUND)
    monkeypatch.setattr(_Scored, "score", 0.25)
    assert not wake.heard_over_speech(FRAME_OF_SOUND)

    assert wake.closest_miss() == "yo_jarvis score 0.42 (threshold 0.50)"
    assert wake.closest_miss() is None


def test_a_score_like_jarvis_s_own_voice_is_no_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    wake = _scored(monkeypatch, audio.NEAR_MISS / 2)

    assert not wake.heard_over_speech(FRAME_OF_SOUND)
    assert not wake.heard(FRAME_OF_SOUND)
    assert wake.closest_miss() is None


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
    assert "LiveKit hey_jarvis, jarvis (threshold 0.5; jarvis 0.85)" in out
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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = config.load()
    _models(cfg.data_dir, "hey_jarvis")

    _fake_loader(monkeypatch)
    (check, *_), notes = cli._wake_checks(cfg)  # pyright: ignore[reportPrivateUsage] - the doctor's wake check

    assert check[0]
    assert check[1] == "wake word files (hey_jarvis)"
    assert len(notes) == 1
    assert notes[0].startswith("note: wake word yo_jarvis is on but has no file ")
    assert "jarvis wake --words" in notes[0]
    assert not capsys.readouterr().out


def test_the_doctor_fails_when_no_wake_word_that_is_on_has_a_file() -> None:
    (check, *_), _notes = cli._wake_checks(config.load())  # pyright: ignore[reportPrivateUsage] - the doctor's wake check

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


class _Loader:
    """Stands in for `livekit.wakeword.WakeWordModel`: loads what the test allows."""

    bad: ClassVar[dict[str, Exception]] = {}
    loaded: ClassVar[list[Path]] = []

    def __init__(self, *, sess_options: ort.SessionOptions) -> None:
        super().__init__()
        # The doctor loads with the runtime's quiet options, not one thread per core.
        assert sess_options.intra_op_num_threads == 1
        assert sess_options.inter_op_num_threads == 1

    def load_model(self, path: Path) -> None:
        if path.name in self.bad:
            raise self.bad[path.name]
        self.loaded.append(path)


def _fake_loader(monkeypatch: pytest.MonkeyPatch, **bad: Exception) -> None:
    monkeypatch.setattr(_Loader, "bad", {f"{name}.onnx": exc for name, exc in bad.items()})
    monkeypatch.setattr(_Loader, "loaded", [])
    monkeypatch.setattr("livekit.wakeword.WakeWordModel", _Loader)


def test_the_doctor_fails_a_wake_word_file_that_does_not_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config.load()
    _models(cfg.data_dir, "hey_jarvis", "yo_jarvis")
    _fake_loader(monkeypatch, yo_jarvis=ValueError("not a streaming model"))

    checks, _notes = cli._wake_checks(cfg)  # pyright: ignore[reportPrivateUsage] - the doctor's wake check

    assert [c[0] for c in checks] == [True, False]
    assert checks[1][1].startswith("wake word yo_jarvis does not load: ")
    assert "not a streaming model" in checks[1][1]
    assert _Loader.loaded == [cfg.data_dir / "wake" / "hey_jarvis.onnx"]


def test_the_doctor_does_not_try_to_load_a_wake_word_that_has_no_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config.load()
    _fake_loader(monkeypatch)

    checks, _notes = cli._wake_checks(cfg)  # pyright: ignore[reportPrivateUsage] - the doctor's wake check

    assert len(checks) == 1
    assert not _Loader.loaded


def test_the_doctor_notes_a_threshold_for_a_name_that_is_not_a_wake_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = config.load()
    audio = cfg.audio.model_copy(
        update={"wake_words": ["jarvis"], "wake_thresholds": {"jarvis": 0.85, "jarivs": 0.8}}
    )
    _fake_loader(monkeypatch)

    _checks, notes = cli._wake_checks(cfg.model_copy(update={"audio": audio}))  # pyright: ignore[reportPrivateUsage] - the doctor's wake check

    flagged = [n for n in notes if "jarivs" in n]
    assert len(flagged) == 1
    assert flagged[0].startswith("note: wake_thresholds names jarivs")
    assert not any("names jarvis" in n for n in notes)


def test_the_wake_header_shows_each_own_threshold(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_restart_if_running", lambda: pytest.fail("no restart"))
    cfg = config.load()
    audio = cfg.audio.model_copy(
        update={
            "wake_words": ["hey_jarvis", "jarvis", "yo_jarvis"],
            "wake_thresholds": {"jarvis": 0.85, "off_word": 0.9},
        }
    )
    config.save(cfg.model_copy(update={"audio": audio}))

    assert main(["wake"]) == 0

    assert "LiveKit hey_jarvis, jarvis, yo_jarvis (threshold 0.5; jarvis 0.85)" in (
        capsys.readouterr().out
    )


def _voxtype_cannot_say(_engine: str) -> frozenset[str] | None:
    return None


def test_the_doctor_prints_a_wake_word_file_that_does_not_load_and_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = config.load()
    config.save(cfg)
    _models(cfg.data_dir, "hey_jarvis", "yo_jarvis")
    _fake_loader(monkeypatch, yo_jarvis=ValueError("not a streaming model"))
    monkeypatch.setattr(audio, "installed_models", _voxtype_cannot_say)

    status = main(["doctor"])

    out = capsys.readouterr().out.splitlines()
    assert status == 1
    assert any(
        line.startswith("FAIL wake word yo_jarvis does not load: not a streaming model")
        for line in out
    )


def test_the_load_check_names_a_missing_file_and_a_file_that_is_not_a_model(
    tmp_path: Path,
) -> None:
    junk = tmp_path / "junk.onnx"
    junk.write_bytes(b"not a model")
    missing = tmp_path / "missing.onnx"

    errors = audio.wake_load_errors([junk, missing])

    assert set(errors) == {junk, missing}
    assert "not found" in errors[missing]
    assert "not a valid ONNX model" in errors[junk]
