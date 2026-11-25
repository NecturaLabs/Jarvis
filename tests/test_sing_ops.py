"""The `sing` operation: its arguments, its policy class, its singer and where it is offered."""

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from test_sing_psola import Voice

from jarvis import app
from jarvis.config import Config
from jarvis.operations import ArgumentError, Operation, OperationError, validate_args
from jarvis.policy import OpClass
from jarvis.sing import engines, soulx
from jarvis.sing.engines import kokoro_voice, singer_for
from jarvis.sing.ops import SingOps
from jarvis.sing.psola import PsolaSinger

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.sing.score import Score

# An original line, written for these tests.
LINE = "Morn- E4 1, ing G4 1, light C5 2"


class Singer:
    """Renders one second of quiet sound per song and keeps the scores it was given."""

    name = "test"

    def __init__(self, *, fails: bool = False) -> None:
        super().__init__()
        self.fails = fails
        self.scores: list[Score] = []

    def render(self, score: Score) -> tuple[np.ndarray, int]:
        if self.fails:
            msg = "no GPU"
            raise RuntimeError(msg)
        self.scores.append(score)
        return np.ones(16000, dtype=np.int16), 16000


def _op(singer: Singer, made: list[int] | None = None) -> Operation:
    def make() -> Singer:
        if made is not None:
            made.append(1)
        return singer

    (op,) = SingOps(make).operations()
    return op


def test_a_song_is_rendered_and_handed_back_as_sound_with_its_words() -> None:
    singer = Singer()
    op = _op(singer)

    result = op.run(validate_args(op, {"notes": LINE, "tempo_bpm": 90}))

    assert result.ok
    assert result.sound is not None
    assert result.sound.rate == 16000
    assert len(result.sound.pcm) == 32000
    assert result.sound.words == "Morning light"
    assert singer.scores[0].tempo == 90
    assert "Do not repeat the lyrics" in result.text


def test_the_tempo_defaults_and_the_singer_is_made_once_on_the_first_song() -> None:
    made: list[int] = []
    singer = Singer()
    op = _op(singer, made)
    assert not made  # nothing loads until Jarvis sings

    op.run(validate_args(op, {"notes": LINE}))
    op.run(validate_args(op, {"notes": LINE}))

    assert made == [1]
    assert singer.scores[0].tempo == 100


@pytest.mark.parametrize(
    "args",
    [{}, {"notes": LINE, "key": "C"}, {"notes": LINE, "tempo_bpm": 400}, {"notes": ""}],
)
def test_arguments_outside_the_schema_are_refused_before_anything_runs(
    args: dict[str, Any],
) -> None:
    singer = Singer()
    op = _op(singer)

    with pytest.raises(ArgumentError):
        validate_args(op, args)
    assert not singer.scores


def test_a_score_that_cannot_be_sung_is_an_error_the_model_can_fix() -> None:
    singer = Singer()
    op = _op(singer)

    with pytest.raises(OperationError, match=r"Nothing was sung: .*out of range"):
        op.run(validate_args(op, {"notes": "la C9 1"}))
    assert not singer.scores


def test_a_failing_singer_is_an_error_not_a_crash() -> None:
    op = _op(Singer(fails=True))

    with pytest.raises(OperationError, match="singing voice failed"):
        op.run(validate_args(op, {"notes": LINE}))


def test_singing_is_safe_and_counts_as_an_action() -> None:
    op = _op(Singer())

    assert op.default_class is OpClass.SAFE
    assert op.side_effect
    assert op.spoken_summary({"notes": LINE}) == "sing Morning light"


@pytest.mark.parametrize(
    ("engine", "installed", "chosen"),
    [
        ("auto", False, "speech"),
        ("auto", True, "soulx"),
        ("soulx", False, "speech"),  # said in the log
        ("soulx", True, "soulx"),
        ("speech", True, "speech"),
    ],
)
def test_the_engine_is_soulx_when_set_up_and_else_the_speaking_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str, installed: bool, chosen: str
) -> None:
    cfg = Config.model_validate({"data_dir": str(tmp_path), "sing": {"engine": engine}})
    speech = PsolaSinger(Voice())

    def speech_singer(_cfg: Config) -> PsolaSinger:
        return speech

    monkeypatch.setattr(engines, "speech_singer", speech_singer)
    if installed:
        found = soulx.files(tmp_path)
        for path in (found.binary, found.model):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")

    assert singer_for(cfg)().name == chosen


def test_sing_is_offered_by_voice_and_not_in_the_chat_and_deny_switches_it_off() -> None:
    cfg = Config()
    voice = app.channel_registry(cfg, [], app.channel_limits(cfg, "voice"), "voice")
    chat = app.channel_registry(cfg, [], app.channel_limits(cfg, "chat"), "chat")

    assert voice.get("sing") is not None
    assert chat.get("sing") is None
    off = Config.model_validate({"boundaries": {"classes": {"sing": "deny"}}})
    assert "sing" in app.switched_off(off)
    assert not app.unknown_overrides(off, [])


def test_the_singing_voice_is_the_speaking_kokoro_voice_or_the_default() -> None:
    assert kokoro_voice(Config()) == "bm_george"
    kokoro = Config.model_validate({"audio": {"voice": "kokoro:af_heart"}})
    assert kokoro_voice(kokoro) == "af_heart"
    piper = Config.model_validate({"audio": {"voice": "en_US-lessac-medium"}})
    assert kokoro_voice(piper) == engines.DEFAULT_KOKORO_VOICE


def test_without_the_kokoro_files_singing_says_what_is_missing(tmp_path: Path) -> None:
    cfg = Config.model_validate({"data_dir": str(tmp_path)})

    with pytest.raises(OperationError, match="Kokoro voice is not installed"):
        singer_for(cfg)()
