"""`pc_audio_listen` and the PC audio it hears, with fake frames and a fake speech-to-text."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from jarvis.audio import FRAME
from jarvis.config import AudioConfig
from jarvis.operations import (
    ArgumentError,
    Operation,
    OperationError,
    Registry,
    request_scope,
    validate_args,
)
from jarvis.pc.hearing import (
    MAX_LISTEN_S,
    NOTHING_PLAYED,
    NOTHING_SAID,
    PC_AUDIO_ONLY_FOR_OWNER,
    RESERVE_S,
    HearingOps,
    Listening,
    PcHearing,
)
from jarvis.policy import OpClass, Origin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

SPEECH = np.ones(FRAME, dtype=np.int16)
WAIT_S = 5.0


class Clock:
    def __init__(self) -> None:
        super().__init__()
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Transcriber:
    def __init__(self, text: str = "they said hello") -> None:
        super().__init__()
        self.text = text
        self.heard: list[int] = []

    def __call__(self, frames: Sequence[NDArray[np.int16]]) -> str:
        self.heard.append(len(frames))
        return self.text


def _op(
    hearing: PcHearing | None, transcriber: Transcriber | None = None, max_chars: int = 3000
) -> Operation:
    listening = None if hearing is None else Listening(hearing, transcriber or Transcriber())
    (op,) = HearingOps(listening, heard="what discord plays", max_chars=max_chars).operations()
    return op


def _feed_while_listening(hearing: PcHearing, frames: int) -> threading.Thread:
    """Feed `frames` frames once a listen has started, from another thread like the watcher."""

    def feed() -> None:
        pause = threading.Event()
        for _ in range(int(WAIT_S / 0.002)):
            if hearing.wants:
                break
            pause.wait(0.002)
        for _ in range(frames):
            hearing.feed(SPEECH)

    thread = threading.Thread(target=feed, daemon=True)
    thread.start()
    return thread


def test_a_listen_returns_the_next_seconds_of_pc_audio() -> None:
    hearing = PcHearing(0)
    feeder = _feed_while_listening(hearing, 100)

    frames = hearing.listen(3)

    feeder.join(WAIT_S)
    assert frames is not None
    assert len(frames) == 38  # 3 s of 80 ms frames, rounded up; nothing past it is kept


def test_a_listen_with_no_stream_playing_gives_up_with_nothing() -> None:
    assert PcHearing(0, grace_s=0).listen(0.16) == []


def test_nothing_is_kept_while_no_listen_runs_and_recall_is_off() -> None:
    hearing = PcHearing(0)

    assert not hearing.wants
    hearing.feed(SPEECH)
    assert hearing.recall(10) == []


@pytest.mark.parametrize("ends", ["forget", "stop"])
def test_a_pause_or_a_stop_ends_a_listen_with_none(ends: str) -> None:
    hearing = PcHearing(0)  # without recall, `wants` says a listen runs
    done: list[list[NDArray[np.int16]] | None] = []
    listener = threading.Thread(target=lambda: done.append(hearing.listen(30)), daemon=True)
    listener.start()
    for _ in range(int(WAIT_S / 0.002)):
        if hearing.wants:
            break
        threading.Event().wait(0.002)

    getattr(hearing, ends)()

    listener.join(WAIT_S)
    assert done == [None]


def test_after_a_stop_every_listen_ends_at_once() -> None:
    hearing = PcHearing(0)
    hearing.stop()

    assert hearing.listen(30) is None


def test_recall_keeps_only_the_last_seconds_asked_for_and_set() -> None:
    clock = Clock()
    hearing = PcHearing(5, clock=clock)
    for _ in range(200):  # 16 s of audio: only the last 5 s are held
        clock.now += 0.08
        hearing.feed(SPEECH)

    assert len(hearing.recall(3)) == 38  # 3 s of 80 ms frames, rounded up, as a listen
    assert len(hearing.recall(30)) == 63  # never more than the 5 s kept


def test_recall_drops_audio_from_before_a_gap_such_as_a_sleep() -> None:
    clock = Clock()
    hearing = PcHearing(10, clock=clock)
    hearing.feed(SPEECH)
    clock.now += 60  # the boot clock counts a suspend

    assert hearing.recall(10) == []


def test_a_pause_drops_the_recalled_audio() -> None:
    hearing = PcHearing(10, clock=Clock())
    hearing.feed(SPEECH)

    hearing.forget()

    assert hearing.recall(10) == []


def test_a_recall_shorter_than_the_shortest_listen_keeps_nothing() -> None:
    hearing = PcHearing(2)

    assert hearing.recall_s == 0
    assert not hearing.wants


def test_the_operation_is_a_safe_read_whose_text_the_audit_keeps_only_as_a_digest() -> None:
    op = _op(PcHearing(0))

    assert op.name == "pc_audio_listen"
    assert op.default_class is OpClass.SAFE
    assert not op.side_effect
    assert not op.audit_text
    assert not op.outbound
    assert [s["function"]["name"] for s in Registry([op]).tool_schemas()] == ["pc_audio_listen"]


@pytest.mark.parametrize(
    "args",
    [{}, {"seconds": 2}, {"seconds": MAX_LISTEN_S + 1}, {"seconds": 10, "loud": True}],
)
def test_arguments_out_of_bounds_are_refused_before_anything_runs(args: dict[str, Any]) -> None:
    with pytest.raises(ArgumentError):
        validate_args(_op(PcHearing(0)), args)


def test_the_last_seconds_can_be_asked_for_only_when_they_are_kept() -> None:
    with pytest.raises(ArgumentError):
        validate_args(_op(PcHearing(0)), {"seconds": 10, "when": "last"})

    op = _op(PcHearing(30))
    assert validate_args(op, {"seconds": 10, "when": "last"})["when"] == "last"
    assert "30 seconds" in op.description


def test_a_listen_returns_what_was_said_as_pc_audio_data() -> None:
    hearing = PcHearing(0)
    transcriber = Transcriber("  they said\nhello  ")
    op = _op(hearing, transcriber)
    feeder = _feed_while_listening(hearing, 100)

    with request_scope(foreign=False, words="what are they saying"):
        result = op.run(validate_args(op, {"seconds": 3}))

    feeder.join(WAIT_S)
    assert (result.text, result.origin) == ("they said hello", Origin.PC_AUDIO)
    assert transcriber.heard == [38]


def test_the_last_seconds_are_read_from_what_was_kept() -> None:
    clock = Clock()
    hearing = PcHearing(10, clock=clock)
    for _ in range(50):
        clock.now += 0.08
        hearing.feed(SPEECH)
    transcriber = Transcriber()
    op = _op(hearing, transcriber)

    with request_scope(foreign=False):
        result = op.run(validate_args(op, {"seconds": 3, "when": "last"}))

    assert result.origin is Origin.PC_AUDIO
    assert transcriber.heard == [38]


def test_a_transcript_longer_than_a_result_may_be_is_cut() -> None:
    clock = Clock()
    hearing = PcHearing(10, clock=clock)
    clock.now += 0.08
    hearing.feed(SPEECH)
    op = _op(hearing, Transcriber("word " * 1000), max_chars=500)

    with request_scope(foreign=False):
        result = op.run(validate_args(op, {"seconds": 3, "when": "last"}))

    assert len(result.text) == 500


def test_silence_and_no_stream_are_said_as_jarvis_own_words() -> None:
    clock = Clock()
    hearing = PcHearing(10, clock=clock)
    op = _op(hearing, Transcriber(""))

    with request_scope(foreign=False):
        nothing = op.run(validate_args(op, {"seconds": 3, "when": "last"}))
        clock.now += 0.08
        hearing.feed(SPEECH)
        silent = op.run(validate_args(op, {"seconds": 3, "when": "last"}))

    assert nothing.text.startswith(NOTHING_PLAYED)
    assert "what discord plays" in nothing.text
    assert silent.text.startswith(NOTHING_SAID)
    assert {nothing.origin, silent.origin} == {Origin.TOOL_STATUS}


def test_a_request_heard_in_pc_audio_cannot_listen_to_it() -> None:
    transcriber = Transcriber()
    op = _op(PcHearing(10, clock=Clock()), transcriber)

    with request_scope(foreign=True), pytest.raises(OperationError, match=PC_AUDIO_ONLY_FOR_OWNER):
        op.run(validate_args(op, {"seconds": 3, "when": "last"}))
    with pytest.raises(OperationError, match=PC_AUDIO_ONLY_FOR_OWNER):
        op.run(validate_args(op, {"seconds": 3, "when": "last"}))  # outside any request

    assert transcriber.heard == []


def test_without_the_voice_service_every_listen_is_refused() -> None:
    op = _op(None)

    with request_scope(foreign=False), pytest.raises(OperationError, match="voice service"):
        op.run(validate_args(op, {"seconds": 3}))


def test_a_pause_while_listening_is_said() -> None:
    paused = threading.Event()
    paused.set()
    op = _op(PcHearing(0, paused=paused.is_set))

    with request_scope(foreign=False), pytest.raises(OperationError, match="paused"):
        op.run(validate_args(op, {"seconds": 3}))


def test_a_pause_ends_a_listen_while_no_pc_frame_comes() -> None:
    paused = threading.Event()
    hearing = PcHearing(0, paused=paused.is_set)
    threading.Timer(0.05, paused.set).start()
    started = time.monotonic()

    frames = hearing.listen(MAX_LISTEN_S)

    assert frames is None
    assert time.monotonic() - started < WAIT_S  # within a check, not after a minute


def test_a_listen_across_a_suspend_ends_without_joining_both_sides() -> None:
    boot = Clock()
    hearing = PcHearing(0, clock=boot, steady=lambda: 0.0)
    feeder = _feed_while_listening(hearing, 5)

    def sleep() -> None:
        feeder.join(WAIT_S)
        boot.now += 600  # the boot clock counts the suspend, the steady one does not

    threading.Thread(target=sleep, daemon=True).start()

    assert hearing.listen(MAX_LISTEN_S) is None


def test_the_config_takes_recall_off_or_from_three_seconds() -> None:
    assert AudioConfig().pc_audio_recall_s == 0
    assert AudioConfig(pc_audio_recall_s=3).pc_audio_recall_s == 3
    with pytest.raises(ValueError, match="pc_audio_recall_s"):
        AudioConfig(pc_audio_recall_s=1)
    with pytest.raises(ValueError, match="pc_audio_recall_s"):
        AudioConfig(pc_audio_recall_s=61)


def test_a_listen_is_shortened_to_end_with_the_requests_time_to_spare() -> None:
    hearing = PcHearing(0)
    transcriber = Transcriber()
    op = _op(hearing, transcriber)
    feeder = _feed_while_listening(hearing, 200)

    with request_scope(foreign=False, time_left=lambda: RESERVE_S + 5):
        result = op.run(validate_args(op, {"seconds": MAX_LISTEN_S}))

    feeder.join(WAIT_S)
    assert result.origin is Origin.PC_AUDIO
    assert transcriber.heard == [63]  # 5 s of 80 ms frames, not a minute


@pytest.mark.parametrize(("when", "left"), [("next", RESERVE_S + 2), ("last", RESERVE_S - 1)])
def test_a_listen_with_too_little_of_the_requests_time_left_is_refused(
    when: str, left: float
) -> None:
    transcriber = Transcriber()
    op = _op(PcHearing(10, clock=Clock()), transcriber)

    with (
        request_scope(foreign=False, time_left=lambda: left),
        pytest.raises(OperationError, match="too little of this request's time"),
    ):
        op.run(validate_args(op, {"seconds": 3, "when": when}))

    assert transcriber.heard == []
