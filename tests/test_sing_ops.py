"""The `sing` operation: its arguments, its policy class, its singer and where it is offered."""

import contextlib
import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from test_sing_psola import Voice

from jarvis import app, models
from jarvis.config import Config
from jarvis.operations import (
    ArgumentError,
    Operation,
    OperationError,
    OperationStoppedError,
    request_scope,
    validate_args,
)
from jarvis.policy import OpClass
from jarvis.sing import engines, soulx
from jarvis.sing.budget import Budget, OutOfTimeError, SongStoppedError
from jarvis.sing.engines import kokoro_voice, singer_for
from jarvis.sing.ops import SingOps
from jarvis.sing.psola import PsolaSinger

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from jarvis.sing.score import Score

# An original line, written for these tests.
LINE = "Morn- E4 1, ing G4 1, light C5 2"


class Singer:
    """Renders one second of quiet sound per song and keeps the scores it was given.

    `raises` is raised instead of a song; `during` runs while the song is being made.
    """

    name = "test"

    def __init__(
        self, *, raises: Exception | None = None, during: Callable[[], object] = lambda: None
    ) -> None:
        super().__init__()
        self.raises = raises
        self.during = during
        self.scores: list[Score] = []
        self.budgets: list[Budget] = []

    def render(self, score: Score, budget: Budget, /) -> tuple[np.ndarray, int]:
        self.budgets.append(budget)
        self.during()
        if self.raises is not None:
            raise self.raises
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
    beat = 60 / 90
    assert result.sound.lyrics == ((0.0, 2 * beat, "Morning"), (2 * beat, 4 * beat, "light"))
    assert singer.scores[0].tempo == 90
    assert "Do not repeat the lyrics" in result.text


@pytest.mark.parametrize("signal", ["cancelled", "hushed"])
def test_the_song_stops_when_its_request_is_cut_short_or_hushed(signal: str) -> None:
    singer = Singer()
    op = _op(singer)
    stopped = [False]

    with request_scope(foreign=False, **{signal: lambda: stopped[0]}):  # pyright: ignore[reportArgumentType]
        op.run(validate_args(op, {"notes": LINE}))
    (budget,) = singer.budgets
    before = budget.stopped()
    stopped[0] = True

    assert (before, budget.stopped()) == (False, True)


def test_the_song_is_given_the_requests_time_left() -> None:
    singer = Singer()
    op = _op(singer)

    with request_scope(foreign=False, time_left=lambda: 12.5):
        op.run(validate_args(op, {"notes": LINE}))

    assert singer.budgets[0].time_left() == pytest.approx(12.5)


def test_a_song_outside_a_request_cannot_be_stopped() -> None:
    singer = Singer()
    op = _op(singer)

    op.run(validate_args(op, {"notes": LINE}))

    assert not singer.budgets[0].stopped()


def test_a_song_is_made_while_the_voice_listens_for_the_wake_word() -> None:
    inside: list[bool] = []
    listening = [False]

    @contextlib.contextmanager
    def making_sound() -> Generator[None]:
        listening[0] = True
        try:
            yield
        finally:
            listening[0] = False

    op = _op(Singer(during=lambda: inside.append(listening[0])))

    with request_scope(foreign=False, making_sound=making_sound):
        op.run(validate_args(op, {"notes": LINE}))

    assert inside == [True]
    assert not listening[0]


def test_a_song_stopped_while_it_is_made_is_stopped_not_sung() -> None:
    op = _op(Singer(raises=SongStoppedError()))

    with pytest.raises(OperationStoppedError, match="stopped before it played"):
        op.run(validate_args(op, {"notes": LINE}))


def test_a_song_stopped_as_it_was_finished_is_not_handed_back() -> None:
    hushed = [False]
    singer = Singer(during=lambda: hushed.__setitem__(0, True))
    op = _op(singer)

    with (
        request_scope(foreign=False, hushed=lambda: hushed[0]),
        pytest.raises(OperationStoppedError),
    ):
        op.run(validate_args(op, {"notes": LINE}))


def test_a_song_already_stopped_is_not_made_at_all() -> None:
    made: list[int] = []
    singer = Singer()
    op = _op(singer, made)

    with (
        request_scope(foreign=False, cancelled=lambda: True),
        pytest.raises(OperationStoppedError),
    ):
        op.run(validate_args(op, {"notes": LINE}))

    assert not made  # the voice was not even loaded
    assert not singer.budgets


def test_a_song_whose_request_ran_out_of_time_is_an_error() -> None:
    op = _op(Singer(raises=OutOfTimeError()))

    with pytest.raises(OperationError, match="ran out of time"):
        op.run(validate_args(op, {"notes": LINE}))


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
    op = _op(Singer(raises=RuntimeError("no GPU")))

    with pytest.raises(OperationError, match="singing voice failed"):
        op.run(validate_args(op, {"notes": LINE}))


def test_singing_is_safe_and_counts_as_an_action() -> None:
    op = _op(Singer())

    assert op.default_class is OpClass.SAFE
    assert op.side_effect


def test_the_question_before_a_song_holds_all_of_its_words() -> None:
    op = _op(Singer())
    # The same harmless start; what follows differs.
    first = "Sun- C4 1, ny E4 1, days G4 1, are C4 1, here D4 1, a- E4 1, gain F4 1, friend G4 2"
    second = "Sun- C4 1, ny E4 1, days G4 1, are C4 1, here D4 1, a- E4 1, gain F4 1, foe G4 2"

    asked = [op.spoken_summary({"notes": first}), op.spoken_summary({"notes": second})]

    assert asked[0] == "sing a 5-second song with these words: Sunny days are here again friend"
    assert asked[1].endswith("again foe")


def test_a_score_that_cannot_be_sung_is_refused_before_anyone_is_asked() -> None:
    op = _op(Singer())
    assert op.prepare is not None

    with pytest.raises(OperationError, match=r"Nothing was sung: .*out of range"):
        op.prepare(validate_args(op, {"notes": "la C4 " + "9" * 400 + "/1"}))
    args = validate_args(op, {"notes": LINE})
    assert op.prepare(args) == args


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


def _installed(
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    executable: bool = True,
    libraries: bool = True,
) -> None:
    """Put a runner bundle in place and pin it as the release: the runner and one library."""
    bundle = {soulx.BINARY: b"the runner", "libggml-vulkan.so": b"a library"}
    monkeypatch.setattr(
        models,
        "SOULX_RUNNER",
        tuple(
            models.RunnerFile(n, len(d), hashlib.sha256(d).hexdigest()) for n, d in bundle.items()
        ),
    )
    found = soulx.files(data_dir)
    found.binary.parent.mkdir(parents=True, exist_ok=True)
    found.binary.write_bytes(bundle[soulx.BINARY])
    found.binary.chmod(0o755 if executable else 0o644)
    found.model.write_bytes(b"")
    if libraries:
        (found.binary.parent / "libggml-vulkan.so").write_bytes(bundle["libggml-vulkan.so"])


def _singer_checks(data_dir: Path, engine: str = "auto") -> dict[str, bool]:
    cfg = Config.model_validate({"data_dir": str(data_dir), "sing": {"engine": engine}})
    checks, _notes = engines.doctor_checks(cfg)
    return {label.split(" ")[0]: ok for ok, label, _fix in checks}


def test_the_doctor_passes_an_installed_runner_that_can_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _installed(tmp_path, monkeypatch)

    checks = _singer_checks(tmp_path)

    assert checks["executable"]
    assert checks["runner"]


@pytest.mark.parametrize(
    ("flaw", "failed"),
    [({"executable": False}, "executable"), ({"libraries": False}, "runner")],
    ids=["not-executable", "no-libraries"],
)
def test_the_doctor_fails_a_runner_that_cannot_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flaw: dict[str, bool], failed: str
) -> None:
    _installed(tmp_path, monkeypatch, **flaw)

    checks = _singer_checks(tmp_path)

    assert not checks[failed]


def test_the_doctor_fails_a_bundle_with_a_library_that_is_not_the_pinned_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _installed(tmp_path, monkeypatch)
    (soulx.files(tmp_path).binary.parent / "libggml-vulkan.so").write_bytes(b"another build")

    assert not _singer_checks(tmp_path)["runner"]


def test_the_doctor_leaves_the_runner_alone_when_soulx_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _installed(tmp_path, monkeypatch, executable=False, libraries=False)

    assert "executable" not in _singer_checks(tmp_path, "speech")


def test_the_phoneme_cache_is_bounded() -> None:
    assert soulx._phones.cache_info().maxsize is not None  # pyright: ignore[reportPrivateUsage]
