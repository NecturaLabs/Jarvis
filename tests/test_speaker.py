"""The speaker's queue: sentences in order through one player, spoken over at once on a hush,
failures reported at the wait, and a sentence that sounds like the wake word guarded."""

import contextlib
import os
import signal
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest

from jarvis.audio import (
    FRAME,
    LEAD_KEEP_S,
    TAIL_KEEP_S,
    AudioError,
    OwnVoiceCheck,
    Speaker,
    to_16k,
    trim_silence,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

RATE = 24000
WAIT_S = 5.0


def _sound(level: int, seconds: float = 0.1) -> bytes:
    return np.full(int(RATE * seconds), level, dtype=np.int16).tobytes()


def _voice(text: str) -> tuple[bytes, int]:
    """A distinct, loud sound per text, so the order of what was played can be read back."""
    return _sound(1000 + len(text)), RATE


def _recorder(folder: Path) -> tuple[Callable[[int], list[str]], list[Path]]:
    """A player that writes what it was given to a new file in `folder` per start."""
    files: list[Path] = []

    def player(_rate: int) -> list[str]:
        target = folder / f"played-{len(files)}.raw"
        files.append(target)
        return ["sh", "-c", f"cat > {target}"]

    return player, files


def _endless(pid_file: Path) -> Callable[[int], list[str]]:
    return lambda _rate: ["sh", "-c", f"echo $$ > {pid_file}; cat > /dev/null; exec sleep 600"]


def _stuck(pid_file: Path) -> Callable[[int], list[str]]:
    """A player that takes nothing: once the pipe is full, handing it sound blocks, as with a
    real player that plays at the speed of speech."""
    return lambda _rate: ["sh", "-c", f"echo $$ > {pid_file}; exec sleep 600"]


def _end(pid_file: Path) -> None:
    with contextlib.suppress(OSError, ValueError):
        os.kill(int(pid_file.read_text(encoding="ascii")), signal.SIGKILL)


def _until(condition: Callable[[], bool]) -> None:
    pause = threading.Event()
    for _ in range(int(WAIT_S / 0.005)):
        if condition():
            return
        pause.wait(0.005)
    pytest.fail("a condition never came true")


def test_queued_sentences_play_in_order_through_one_player(tmp_path: Path) -> None:
    player, files = _recorder(tmp_path)
    speaker = Speaker(_voice, player=player)

    for text in ("a", "bb", "ccc"):
        speaker.queue(text)
    speaker.wait()

    assert len(files) == 1
    assert files[0].read_bytes() == _voice("a")[0] + _voice("bb")[0] + _voice("ccc")[0]
    assert not speaker.busy


def test_a_sentence_after_a_pause_gets_a_new_player(tmp_path: Path) -> None:
    player, files = _recorder(tmp_path)
    speaker = Speaker(_voice, player=player, linger_s=0.01)

    speaker.say("a")
    speaker.say("bb")

    assert [f.read_bytes() for f in files] == [_voice("a")[0], _voice("bb")[0]]


def test_a_hush_drops_what_is_queued_and_ends_the_sentence_playing(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    synthesized: list[str] = []

    def voice(text: str) -> tuple[bytes, int]:
        synthesized.append(text)
        return _sound(1000, seconds=2.0), RATE  # more than a pipe holds

    speaker = Speaker(voice, player=_stuck(pid_file))
    try:
        for text in ("first", "second", "third"):
            speaker.queue(text)
        _until(pid_file.exists)
        speaker.hush()
        waited = threading.Thread(target=speaker.wait, daemon=True)
        waited.start()
        waited.join(WAIT_S)

        assert not waited.is_alive()
        assert not speaker.busy
        assert "third" not in synthesized
    finally:
        _end(pid_file)


def test_speech_queued_after_a_hush_is_spoken(tmp_path: Path) -> None:
    player, files = _recorder(tmp_path)
    speaker = Speaker(_voice, player=player)

    speaker.hush()
    speaker.say("again")

    assert files[-1].read_bytes() == _voice("again")[0]


def test_a_failing_voice_is_reported_at_the_wait_and_the_next_sentence_still_plays(
    tmp_path: Path,
) -> None:
    player, files = _recorder(tmp_path)

    def voice(text: str) -> tuple[bytes, int]:
        if text == "bad":
            msg = "voice broke"
            raise RuntimeError(msg)
        return _voice(text)

    speaker = Speaker(voice, player=player)
    with pytest.raises(AudioError, match="voice broke"):
        speaker.say("bad")
    speaker.say("ok")

    assert files[-1].read_bytes() == _voice("ok")[0]


def test_a_sentence_that_sounds_like_the_wake_word_cannot_be_spoken_over(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "pid"
    speaker = Speaker(_voice, player=_endless(pid_file), own_wake=lambda _pcm, _rate: True)
    try:
        assert speaker.interruptible
        speaker.queue("hey jarvis")
        _until(pid_file.exists)
        assert not speaker.interruptible
        speaker.hush()
        _until(lambda: not speaker.busy)
        assert speaker.interruptible
    finally:
        _end(pid_file)


def test_ordinary_speech_can_be_spoken_over(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    speaker = Speaker(_voice, player=_endless(pid_file), own_wake=lambda _pcm, _rate: False)
    try:
        speaker.queue("paused")
        _until(pid_file.exists)
        assert speaker.busy
        assert speaker.interruptible
    finally:
        speaker.hush()
        _end(pid_file)


def test_silence_before_the_first_word_and_after_each_sentence_is_trimmed() -> None:
    lead, tail = int(0.5 * RATE), int(0.6 * RATE)
    pcm = np.concatenate([np.zeros(lead), np.full(RATE // 10, 1000), np.zeros(tail)]).astype(
        np.int16
    )

    trimmed = np.frombuffer(
        trim_silence(pcm.tobytes(), RATE, lead_s=LEAD_KEEP_S, tail_s=TAIL_KEEP_S), dtype=np.int16
    )
    kept_lead = np.frombuffer(
        trim_silence(pcm.tobytes(), RATE, lead_s=None, tail_s=TAIL_KEEP_S), dtype=np.int16
    )

    assert trimmed.size == int(LEAD_KEEP_S * RATE) + RATE // 10 + int(TAIL_KEEP_S * RATE)
    assert kept_lead.size == lead + RATE // 10 + int(TAIL_KEEP_S * RATE)
    assert trim_silence(np.zeros(RATE, dtype=np.int16).tobytes(), RATE, lead_s=0, tail_s=0) == b""


class _MarkWake:
    """A wake model that fires on a frame holding the marker level."""

    def __init__(self) -> None:
        super().__init__()
        self.resets = 0

    @staticmethod
    def heard(frame: NDArray[np.int16]) -> bool:
        return bool((frame == 7777).sum() > FRAME // 2)

    def reset(self) -> None:
        self.resets += 1


def test_the_own_voice_check_runs_the_sentence_through_its_own_wake_model() -> None:
    wake = _MarkWake()
    check = OwnVoiceCheck(wake)

    assert check(_sound(7777, seconds=0.5), RATE)
    assert not check(_sound(1000, seconds=0.5), RATE)
    assert wake.resets == 4  # fresh before and after each check, so checks never mix


def test_resampling_to_16k_keeps_the_length_in_time() -> None:
    assert to_16k(_sound(1000, seconds=1.0), RATE).size == 16000


def test_a_sentence_being_synthesized_when_hushed_is_never_played(tmp_path: Path) -> None:
    player, files = _recorder(tmp_path)
    started, release = threading.Event(), threading.Event()

    def voice(text: str) -> tuple[bytes, int]:
        started.set()
        release.wait(WAIT_S)
        return _voice(text)

    speaker = Speaker(voice, player=player)
    speaker.queue("slow")
    assert started.wait(WAIT_S)
    speaker.hush()
    release.set()
    speaker.wait()

    assert files == []


def test_a_sentence_right_after_a_hush_gets_a_fresh_player(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    record, files = _recorder(tmp_path)
    starts: list[int] = []

    def player(rate: int) -> list[str]:
        starts.append(rate)
        return _stuck(pid_file)(rate) if len(starts) == 1 else record(rate)

    speaker = Speaker(lambda _t: (_sound(1000, seconds=2.0), RATE), player=player)
    try:
        speaker.queue("first")
        _until(pid_file.exists)
        speaker.hush()
        speaker.say("again")  # within the linger: the hushed player must not be reused

        assert len(starts) == 2
        assert files[-1].read_bytes() == _sound(1000, seconds=2.0)
    finally:
        _end(pid_file)
