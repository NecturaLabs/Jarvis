"""The voice loop with scripted fakes: owner requests, failures, confirmation and PC audio."""

import logging
import stat
import threading
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast, override

import httpx
import numpy as np
import pytest

from jarvis import voice
from jarvis.agent import MODEL_MISSING, Declined, NoReason
from jarvis.agent import Agent as RealAgent
from jarvis.app import no_action_operation
from jarvis.audio import FRAME, AudioError, PausableMic
from jarvis.audit import AuditLog
from jarvis.config import AudioConfig, Config, ModelConfig
from jarvis.confirmkey import MIN_PRESS_S, NOTHING_WAITING, STRAY_HOLD_S, ConfirmKey
from jarvis.model import Delta, StrataModel
from jarvis.operations import Operation, Registry, Result
from jarvis.pc.hearing import PcHearing
from jarvis.policy import OpClass, Origin
from jarvis.schedule import Due, Job
from jarvis.voice import PcListener, State, VoiceLoop

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Generator, Iterable, Iterator, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from jarvis.agent import Hooks
    from jarvis.model import ChatModel

SILENCE = np.zeros(FRAME, dtype=np.int16)
SPEECH = np.ones(FRAME, dtype=np.int16)
WAKE = np.full(FRAME, 2, dtype=np.int16)
REQUEST = [WAKE, SPEECH, SPEECH, SPEECH, *[SILENCE] * 5]
ANSWER = [SPEECH, SPEECH, *[SILENCE] * 5]
# Follow-up listening is off here, so each scripted request starts with the wake word; the window
# has its own tests below.
CFG = AudioConfig(end_silence_s=0.3, no_speech_timeout_s=1.0, confirm_window_s=2.0, follow_up_s=0)
WAIT_S = 5.0


@dataclass(frozen=True, slots=True)
class Idle:
    """A script step that streams silence until `condition` is true."""

    condition: Callable[[], bool]


type Step = NDArray[np.int16] | Idle | Callable[[], object]


def _steps(steps: Iterable[Step]) -> Iterator[NDArray[np.int16]]:
    """Yield the frames of a script, running its other steps (waits, clock moves) in between."""
    pause = threading.Event()
    for step in steps:
        if isinstance(step, np.ndarray):
            yield step
        elif isinstance(step, Idle):
            for _ in range(int(WAIT_S / 0.002)):
                if step.condition():
                    break
                pause.wait(0.002)
                yield SILENCE
            else:
                pytest.fail("a scripted condition never came true")
        else:
            step()


def until(condition: Callable[[], bool]) -> Callable[[], None]:
    """A script step that waits for `condition`, failing the test when it never comes."""

    def wait() -> None:
        done = threading.Event()
        for _ in range(int(WAIT_S / 0.002)):
            if condition():
                return
            done.wait(0.002)
        pytest.fail("a scripted condition never came true")

    return wait


@dataclass
class Clock:
    now: float = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> Callable[[], None]:
        def step() -> None:
            self.now += seconds

        return step


class Mic:
    """One shared stream: every frames() call carries on where the last one stopped."""

    def __init__(self, *steps: Step) -> None:
        super().__init__()
        self._frames = _steps(steps)
        self.drained = 0

    def frames(self) -> Iterator[NDArray[np.int16]]:
        # A plain loop, not `yield from`: closing one abandoned reader must not end the stream.
        for frame in self._frames:  # ruff: ignore[yield-in-for-loop] - `yield from` would end the stream when a reader closes
            yield frame

    def drain(self) -> None:
        self.drained += 1


class PcAudio:
    """PC audio whose single stream plays a script, then silence until the test ends."""

    def __init__(self, *steps: Step) -> None:
        super().__init__()
        self._frames = _steps(steps)
        self.ended = threading.Event()

    def _session(self) -> Generator[NDArray[np.int16]]:
        for frame in self._frames:  # ruff: ignore[yield-in-for-loop] - `yield from` would end the script when a session closes
            yield frame
        self.ended.set()
        pause = threading.Event()
        while True:
            pause.wait(0.002)
            yield SILENCE

    def sessions(self) -> Generator[Generator[NDArray[np.int16]]]:
        yield self._session()

    def frames(self) -> Iterator[NDArray[np.int16]]:
        yield from self._session()

    def drain(self) -> None:
        pass


@dataclass
class Wake:
    failures: int = 0

    def heard(self, frame: NDArray[np.int16]) -> bool:
        if self.failures:
            self.failures -= 1
            msg = "wake model failed"
            raise RuntimeError(msg)
        return bool(frame[0] == WAKE[0])

    def reset(self) -> None:
        pass


@dataclass
class Vad:
    @staticmethod
    def speech_probability(frame: NDArray[np.int16]) -> float:
        return 1.0 if frame[0] == SPEECH[0] else 0.0

    def reset(self) -> None:
        pass


@dataclass
class Speaker:
    """Speaks at once; `broken` fails each speech the way the real speaker does, at its wait."""

    said: list[str] = field(default_factory=list[str])
    chimes: int = 0
    broken: bool = False
    during_chime: Callable[[], object] = lambda: None
    busy: bool = False
    interruptible: bool = True
    hushed: int = 0
    _failed: bool = False

    def say(self, text: str) -> None:
        self.queue(text)
        self.wait()

    def queue(self, text: str) -> None:
        if self.broken:
            self._failed = True
            return
        self.said.append(text)

    def wait(self) -> None:
        failed, self._failed = self._failed, False
        if failed:
            msg = "no audio output"
            raise AudioError(msg)

    def hush(self) -> None:
        self.hushed += 1

    def chime(self) -> None:
        self.chimes += 1
        self.during_chime()


@dataclass
class Hearing:
    """Takes every PC frame it is fed; `wants` as `pc_audio_listen`'s while one listens."""

    wants: bool = True
    fed: list[int] = field(default_factory=list[int])
    forgot_after: list[int] = field(default_factory=list[int])  # frames fed before each forget
    recall_dropped_after: list[int] = field(default_factory=list[int])

    def feed(self, frame: NDArray[np.int16]) -> None:
        self.fed.append(int(frame[0]))

    def forget(self) -> None:
        self.forgot_after.append(len(self.fed))

    def forget_recall(self) -> None:
        self.recall_dropped_after.append(len(self.fed))


type Reply = Callable[[str, Hooks, Origin], str]


@dataclass
class Agent:
    reply: Reply = lambda _text, _hooks, _origin: "Done."
    calls: list[tuple[str, Origin]] = field(default_factory=list[tuple[str, Origin]])
    called: threading.Event = field(default_factory=threading.Event)
    steered: list[frozenset[str]] = field(default_factory=list[frozenset[str]])
    scheduled: list[bool] = field(default_factory=list[bool])

    def handle(
        self,
        text: str,
        hooks: Hooks,
        origin: Origin = Origin.OWNER_SPEECH,
        *,
        steered_by: Collection[str] = (),
        scheduled: bool = False,
    ) -> str:
        """Answer like the real agent: the reply goes to `hooks.speak` unless cancelled."""
        self.calls.append((text, origin))
        self.steered.append(frozenset(steered_by))
        self.scheduled.append(scheduled)
        self.called.set()
        reply = self.reply(text, hooks, origin)
        if hooks.speak is not None and reply and not hooks.cancelled():
            hooks.speak(reply)
        return reply


def _loop(
    agent: Agent | RealAgent,
    mic: Mic,
    speaker: Speaker,
    *,
    heard: str = "pause the music",
    answer: Callable[[NDArray[np.int16]], str] = lambda _s: "yes",
    pc: PcAudio | EndingPcAudio | None = None,
    cfg: AudioConfig = CFG,
    clock: Clock | None = None,
    pc_wake: Wake | None = None,
    wake: Wake | None = None,
    dictating: Callable[[], bool | None] = lambda: False,
    meeting: Callable[[], bool] = lambda: False,
    media: Callable[[], bool | None] = lambda: False,
    paused: Callable[[], bool] = lambda: False,
    audit: AuditLog | None = None,
    transcribe: Callable[[NDArray[np.int16]], str] | None = None,
    locked: Callable[[], bool] = lambda: False,
    asleep_s: Callable[[], float] = lambda: 0.0,
    after_sleep: Callable[[], None] = lambda: None,
    due: Callable[[], Sequence[Due]] = tuple,
    hearing: Hearing | PcHearing | None = None,
) -> VoiceLoop:
    listener = None if pc is None else PcListener(pc, pc_wake or Wake(), Vad(), hearing)
    return VoiceLoop(
        agent,
        mic,
        wake or Wake(),
        Vad(),
        speaker,
        cfg,
        transcribe or (lambda _samples: heard),
        pc=listener,
        transcribe_answer=answer,
        names=("Jarvis",),
        clock=clock or Clock(),
        pc_retry_s=0,
        dictating=dictating,
        meeting=meeting,
        media_playing=media,
        paused=paused,
        audit=audit,
        locked=locked,
        asleep_s=asleep_s,
        after_sleep=after_sleep,
        due=due,
    )


def _run(loop: VoiceLoop) -> None:
    """Run until the mic script ends, then stop the PC watcher too."""
    stop = threading.Event()
    try:
        loop.run(stop.is_set)
    finally:
        stop.set()


def _asking(question: str) -> Reply:
    """An agent that asks one question and says whether it was approved."""

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        return "approved" if hooks.confirm(question) else "declined"

    return reply


def test_an_owner_request_is_answered_and_the_loop_goes_back_to_idle() -> None:
    agent = Agent(lambda _t, _h, _o: "Paused.")
    speaker = Speaker()
    loop = _loop(agent, Mic(*REQUEST), speaker)

    _run(loop)

    assert agent.calls == [("pause the music", Origin.OWNER_SPEECH)]
    assert speaker.said == ["Paused."]
    assert speaker.chimes == 1
    assert loop.state is State.IDLE


@dataclass
class Probabilities:
    """A VAD that rates the frames it is given in order, with one scripted probability each."""

    scores: list[float]
    seen: int = 0

    def speech_probability(self, frame: NDArray[np.int16]) -> float:
        del frame
        score = self.scores[self.seen]
        self.seen += 1
        return score

    def reset(self) -> None:
        pass


def _capture(scores: list[float], cfg: AudioConfig) -> voice.Utterance:
    return voice.capture_utterance(iter([SILENCE] * len(scores)), Probabilities(scores), cfg)


HYSTERESIS = AudioConfig(
    vad_threshold=0.5, vad_neg_threshold=0.35, end_silence_s=0.3, no_speech_timeout_s=1.0
)


def test_speech_goes_on_while_the_probability_stays_above_the_exit_threshold() -> None:
    # Two frames of speech, six that dip between the thresholds (a breath, a trailing syllable),
    # then four of silence: 0.32 s, over the 0.3 s that ends an utterance.
    heard = _capture([0.9] * 2 + [0.4] * 6 + [0.1] * 4 + [0.1], HYSTERESIS)

    assert heard.reason == "end of speech"
    assert heard.samples is not None
    assert len(heard.samples) == 12 * FRAME
    assert heard.speech_s == pytest.approx(2 * voice.FRAME_S)  # only the entry threshold counts


def test_equal_thresholds_end_speech_as_soon_as_it_dips() -> None:
    cfg = HYSTERESIS.model_copy(update={"vad_neg_threshold": 0.5})

    heard = _capture([0.9] * 2 + [0.4] * 6 + [0.1] * 4 + [0.1], cfg)

    assert heard.samples is not None
    assert len(heard.samples) == 6 * FRAME  # the first four dipping frames are the silence


def test_continuous_speech_is_cut_at_the_length_cap() -> None:
    cfg = HYSTERESIS.model_copy(update={"max_utterance_s": 2.0})
    cap_frames = round(cfg.max_utterance_s / voice.FRAME_S)
    scores = Probabilities([0.9] * (cap_frames + 15))
    frames = iter([SILENCE] * len(scores.scores))

    heard = voice.capture_utterance(frames, scores, cfg)

    assert heard.reason == "length cap"
    assert heard.samples is not None
    assert len(heard.samples) == cap_frames * FRAME
    assert scores.seen == cap_frames  # the rest of the stream is left for the loop to read
    assert len(list(frames)) == 15


def test_speech_that_ends_just_under_the_length_cap_is_not_cut() -> None:
    cfg = HYSTERESIS.model_copy(update={"max_utterance_s": 2.0})
    cap_frames = round(cfg.max_utterance_s / voice.FRAME_S)

    heard = _capture([0.9] * (cap_frames - 5) + [0.0] * 4, cfg)

    assert heard.reason == "end of speech"


def test_a_probability_between_the_thresholds_does_not_start_an_utterance() -> None:
    heard = _capture([0.4] * 14, HYSTERESIS)

    assert heard.samples is None
    assert heard.reason == "no speech"


@pytest.mark.parametrize("live", [1, 3])
def test_the_wake_words_own_tail_is_kept_but_not_counted_as_speech(live: int) -> None:
    scores = [0.9] * 3 + [0.9] * live + [0.0] * 5

    heard = voice.capture_utterance(
        iter([SILENCE] * len(scores)), Probabilities(scores), HYSTERESIS, lead_frames=3
    )

    assert heard.samples is not None
    assert len(heard.samples) == (3 + live + 4) * FRAME  # the tail and the request both kept
    assert heard.speech_s == pytest.approx(live * voice.FRAME_S)


def test_the_wake_words_own_tail_alone_starts_no_request() -> None:
    scores = [0.9] * 4 + [0.0] * 8

    heard = voice.capture_utterance(
        iter([SILENCE] * len(scores)), Probabilities(scores), HYSTERESIS, lead_frames=4
    )

    assert heard.samples is None  # still waiting for the request when the stream ended


@pytest.mark.parametrize(("gap", "tail_kept"), [(0, True), (1, True), (2, False), (3, False)])
def test_the_tail_is_kept_only_when_the_request_follows_within_the_pre_roll(
    gap: int, *, tail_kept: bool
) -> None:
    scores = [0.9] * 4 + [0.0] * gap + [0.9] * 3 + [0.0] * 4

    heard = voice.capture_utterance(
        iter([SILENCE] * len(scores)), Probabilities(scores), HYSTERESIS, lead_frames=4
    )

    assert heard.samples is not None
    # Within 160 ms the tail is the request's own start; later, none of it is kept, however
    # far back the lead-in before speech reaches.
    before = 4 + gap if tail_kept else gap
    assert len(heard.samples) == (before + 3 + 4) * FRAME


def test_a_third_of_a_second_before_the_first_speech_is_kept() -> None:
    """Speech starts softly before the voice detector is sure of it: its first sound is kept."""
    scores = [0.0] * 10 + [0.9] * 3 + [0.0] * 4
    frames = [np.full(FRAME, number, dtype=np.int16) for number in range(len(scores))]

    heard = voice.capture_utterance(iter(frames), Probabilities(scores), HYSTERESIS)

    assert heard.samples is not None
    first = int(heard.samples[0])  # the number of the earliest frame kept
    assert (10 - first) * voice.FRAME_S >= 0.3


def test_the_wait_for_a_request_starts_after_the_wake_words_tail() -> None:
    # 12 frames of silence (0.96 s) after the tail is within the 1 s no-speech timeout.
    scores = [0.9] * 4 + [0.0] * 12 + [0.9] * 3 + [0.0] * 4

    heard = voice.capture_utterance(
        iter([SILENCE] * len(scores)), Probabilities(scores), HYSTERESIS, lead_frames=4
    )

    assert heard.speech_s == pytest.approx(3 * voice.FRAME_S)


def test_a_pause_after_the_wake_word_does_not_end_the_request() -> None:
    # The keyword's tail, then a pause longer than the trailing silence that ends a request (the
    # owner draws breath after "Hey Jarvis"), then the request itself.
    scores = [0.9] * 4 + [0.0] * 6 + [0.9] * 3 + [0.0] * 4

    heard = voice.capture_utterance(
        iter([SILENCE] * len(scores)), Probabilities(scores), HYSTERESIS, lead_frames=4
    )

    assert heard.reason == "end of speech"
    assert heard.speech_s == pytest.approx(3 * voice.FRAME_S)
    assert heard.samples is not None
    pre_roll = voice.PRE_ROLL_FRAMES - 1
    assert len(heard.samples) == (pre_roll + 3 + 4) * FRAME  # not the tail: the pause came first


def test_a_click_after_the_wake_word_is_not_a_request() -> None:
    agent = Agent()
    speaker = Speaker()

    def fail(_samples: NDArray[np.int16]) -> str:
        pytest.fail("a click was sent to the transcriber")

    _run(_loop(agent, Mic(WAKE, SPEECH, *[SILENCE] * 5), speaker, transcribe=fail))

    assert agent.calls == []
    assert speaker.said == []  # no apology either: nothing was asked


def test_two_frames_of_speech_after_the_wake_word_are_a_request() -> None:
    agent = Agent()

    _run(_loop(agent, Mic(WAKE, SPEECH, SPEECH, *[SILENCE] * 5), Speaker()))

    assert len(agent.calls) == 1


class LateWake(Wake):
    """A wake word that is heard late (Azure), so the loop is handed frames from its tail."""

    def __init__(self, lead: list[NDArray[np.int16]]) -> None:
        super().__init__()
        self._lead = lead

    def after_wake(self) -> list[NDArray[np.int16]]:
        lead, self._lead = self._lead, []
        return lead


@pytest.mark.parametrize(("live", "calls"), [(1, 0), (2, 1)])
def test_the_minimum_speech_is_asked_of_the_request_not_the_keywords_tail(
    live: int, calls: int
) -> None:
    agent = Agent()
    mic = Mic(WAKE, *[SPEECH] * live, *[SILENCE] * 5)

    _run(_loop(agent, mic, Speaker(), wake=LateWake([SPEECH] * 3)))

    assert len(agent.calls) == calls


def test_a_short_request_heard_before_a_late_wake_word_was_reported_is_kept() -> None:
    # Azure reports the keyword 0.3-0.6 s after it ended: a short "stop" said at once is all in
    # the frames the wake word kept, none of it in the live microphone frames after them.
    agent = Agent()
    mic = Mic(WAKE, *[SILENCE] * 8)

    _run(_loop(agent, mic, Speaker(), wake=LateWake([SPEECH] * 9)))

    assert len(agent.calls) == 1


def test_a_click_after_the_wake_word_in_pc_audio_is_not_a_request() -> None:
    agent = Agent()
    pc = PcAudio(SILENCE, WAKE, SPEECH, *[SILENCE] * 5)
    mic = Mic(Idle(pc.ended.is_set), *[SILENCE] * 3)

    _run(_loop(agent, mic, Speaker(), pc=pc))

    assert agent.calls == []


@pytest.mark.parametrize(
    ("after_wake", "why"),
    [
        ([SILENCE] * 20, "no speech"),
        ([SPEECH, *[SILENCE] * 5], "0.08 s of speech is too little"),
    ],
)
def test_a_pc_wake_without_a_request_logs_why_it_was_dropped(
    caplog: pytest.LogCaptureFixture, after_wake: list[NDArray[np.int16]], why: str
) -> None:
    caplog.set_level(logging.INFO, logger="jarvis.voice")
    agent = Agent()
    pc = PcAudio(SILENCE, WAKE, *after_wake, *[SILENCE] * 3)
    mic = Mic(Idle(pc.ended.is_set), *[SILENCE] * 3)

    _run(_loop(agent, mic, Speaker(), pc=pc))

    assert agent.calls == []
    assert f"dropped a PC audio request: {why}" in caplog.messages


def test_a_pc_wake_while_the_loop_is_busy_logs_why_it_was_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="jarvis.voice")
    pc_done = threading.Event()
    holder: list[VoiceLoop] = []

    agent = Agent()
    pc = PcAudio(
        until(lambda: bool(holder) and holder[0].state is State.CAPTURE),
        WAKE,
        pc_done.set,
    )
    mic = Mic(WAKE, until(pc_done.is_set), SPEECH, SPEECH, *[SILENCE] * 5)
    loop = _loop(agent, mic, Speaker(), pc=pc)
    holder.append(loop)

    _run(loop)

    assert "ignored a wake word in PC audio: the loop was busy (capture)" in caplog.messages


def test_every_pc_frame_the_watcher_reads_is_heard_for_pc_audio_listen() -> None:
    hearing = Hearing()
    pc = PcAudio(SILENCE, WAKE, SPEECH, SPEECH, *[SILENCE] * 5)
    agent = Agent()
    mic = Mic(Idle(lambda: pc.ended.is_set() and agent.called.is_set()), *[SILENCE] * 3)

    _run(_loop(agent, mic, Speaker(), pc=pc, hearing=hearing))

    # The frames a PC request was captured from are heard too.
    assert hearing.fed[:9] == [0, 2, 1, 1, 0, 0, 0, 0, 0]


def test_no_pc_frame_is_heard_while_nothing_listens_or_jarvis_speaks() -> None:
    for hearing, speaker in [(Hearing(wants=False), Speaker()), (Hearing(), Speaker(busy=True))]:
        pc = PcAudio(SILENCE, SPEECH, *[SILENCE] * 3)
        mic = Mic(Idle(pc.ended.is_set), *[SILENCE] * 3)

        _run(_loop(Agent(), mic, speaker, pc=pc, hearing=hearing))

        assert hearing.fed == []


def test_a_pause_drops_what_pc_audio_listen_holds_and_hears_nothing_meanwhile() -> None:
    hearing = Hearing()
    paused = threading.Event()
    pc = PcAudio(SPEECH, paused.set, SPEECH, SPEECH, paused.clear, WAKE, *[SILENCE] * 2)
    mic = Mic(Idle(pc.ended.is_set), *[SILENCE] * 3)

    _run(_loop(Agent(), mic, Speaker(), pc=pc, hearing=hearing, paused=paused.is_set))

    assert hearing.forgot_after[0] == 1  # once for the pause, not once a frame
    assert hearing.forgot_after.count(1) == 1
    assert hearing.fed[:2] == [1, 2]


class EndingPcAudio:
    """PC audio whose one stream plays a script and ends, as a call that is over."""

    def __init__(self, *frames: NDArray[np.int16]) -> None:
        super().__init__()
        self._frames = frames
        self.ended = threading.Event()

    def sessions(self) -> Generator[Generator[NDArray[np.int16]]]:
        yield self._session()
        self.ended.set()
        pause = threading.Event()
        while True:
            pause.wait(0.002)
            yield from ()

    def _session(self) -> Generator[NDArray[np.int16]]:
        yield from self._frames

    def drain(self) -> None:
        pass


def test_the_end_of_a_pc_stream_drops_the_recall_but_not_a_listen_that_runs() -> None:
    hearing = Hearing()
    pc = EndingPcAudio(SPEECH, SPEECH)
    mic = Mic(Idle(pc.ended.is_set), *[SILENCE] * 3)

    _run(_loop(Agent(), mic, Speaker(), pc=pc, hearing=hearing))

    assert hearing.recall_dropped_after[:1] == [2]
    assert hearing.forgot_after == []


def test_a_listen_running_when_the_pc_stream_ends_keeps_what_it_heard() -> None:
    hearing = PcHearing(10, grace_s=0)
    result: list[list[NDArray[np.int16]] | None] = []
    listener = threading.Thread(target=lambda: result.append(hearing.listen(1.0)), daemon=True)
    listener.start()
    until(lambda: hearing.wants and listener.is_alive())()
    pc = EndingPcAudio(SPEECH, SPEECH, SPEECH)
    mic = Mic(Idle(lambda: pc.ended.is_set() and not listener.is_alive()), *[SILENCE] * 3)

    _run(_loop(Agent(), mic, Speaker(), pc=pc, hearing=hearing))

    listener.join(WAIT_S)
    frames = result[0]
    assert frames is not None
    assert len(frames) == 3  # the stream's frames, not None for a pause


def test_a_pc_capture_cut_short_by_a_pause_logs_why_it_was_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="jarvis.voice")
    paused = threading.Event()
    agent = Agent()
    pc = PcAudio(WAKE, SPEECH, SPEECH, paused.set, *[SILENCE] * 5)
    mic = Mic(Idle(pc.ended.is_set), *[SILENCE] * 3)

    _run(_loop(agent, mic, Speaker(), pc=pc, paused=paused.is_set))

    assert agent.calls == []
    assert "dropped a PC audio request: Jarvis was paused meanwhile" in caplog.messages


def test_a_pc_request_found_with_another_still_waiting_logs_why_it_was_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Only the watcher queues, and only while the queue is empty, so this is the guard's own test.
    caplog.set_level(logging.INFO, logger="jarvis.voice")
    pc = PcAudio()
    loop = _loop(Agent(), Mic(), Speaker(), pc=pc)
    waiting = loop._pc_heard  # pyright: ignore[reportPrivateUsage] - the one-place queue
    waiting.put_nowait(voice._PcRequest(np.concatenate([SPEECH] * 3), 1.0))  # pyright: ignore[reportPrivateUsage]
    listener = PcListener(pc, Wake(), Vad())

    loop._capture_pc(listener, iter([SPEECH, SPEECH, *[SILENCE] * 5]), 2.0)  # pyright: ignore[reportPrivateUsage]

    assert waiting.qsize() == 1
    assert "dropped a PC audio request: another one is still waiting" in caplog.messages


def test_a_pc_request_queued_as_jarvis_pauses_is_dropped_with_a_log_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="jarvis.voice")
    agent = Agent()
    holder: list[VoiceLoop] = []

    def queued() -> bool:
        return bool(holder) and not holder[0]._pc_heard.empty()  # pyright: ignore[reportPrivateUsage] - the request is waiting

    def pause() -> None:
        until(queued)()
        mic.pause()  # before the loop's next frame, so it never takes the request

    def resume_soon() -> None:
        threading.Thread(target=mic.resume, daemon=True).start()

    script = Capture(SILENCE, pause, resume_soon, *[SILENCE] * 5)
    mic = PausableMic(script)
    pc = PcAudio(WAKE, SPEECH, SPEECH, *[SILENCE] * 5)
    loop = VoiceLoop(
        agent,
        mic,
        Wake(),
        Vad(),
        Speaker(),
        CFG,
        lambda _samples: "pause the music",
        pc=PcListener(pc, Wake(), Vad()),
        paused=lambda: mic.paused,
    )
    holder.append(loop)

    _run(loop)

    assert agent.calls == []
    assert "dropped a PC audio request: Jarvis was paused meanwhile" in caplog.messages


def test_a_failing_request_is_apologised_for_and_the_loop_keeps_listening() -> None:
    def reply(_text: str, _hooks: Hooks, _origin: Origin) -> str:
        if len(agent.calls) == 1:
            msg = "boom"
            raise RuntimeError(msg)
        return "Skipped."

    agent = Agent(reply)
    speaker = Speaker()

    _run(_loop(agent, Mic(*REQUEST, *REQUEST), speaker))

    assert len(agent.calls) == 2
    assert speaker.said == [voice.SORRY_INTERNAL, "Skipped."]


def test_an_apology_that_cannot_be_spoken_does_not_end_the_loop() -> None:
    def reply(_text: str, _hooks: Hooks, _origin: Origin) -> str:
        msg = "boom"
        raise RuntimeError(msg)

    agent = Agent(reply)

    _run(_loop(agent, Mic(*REQUEST, *REQUEST), Speaker(broken=True)))

    assert len(agent.calls) == 2


def test_a_confirmation_that_fails_counts_as_no() -> None:
    def broken(_samples: NDArray[np.int16]) -> str:
        msg = "transcriber crashed"
        raise RuntimeError(msg)

    agent = Agent(_asking("Should I?"))
    speaker = Speaker()

    _run(_loop(agent, Mic(*REQUEST, *ANSWER), speaker, answer=broken))

    assert speaker.said == ["Should I?", "declined"]


def test_the_answer_is_transcribed_without_the_vocabulary_hint() -> None:
    requests: list[int] = []
    answers: list[int] = []

    def request(_samples: NDArray[np.int16]) -> str:
        requests.append(1)
        return "like this song"

    def answer(_samples: NDArray[np.int16]) -> str:
        answers.append(1)
        return "yes"

    agent = Agent(_asking("Should I?"))
    speaker = Speaker()
    loop = VoiceLoop(
        agent,
        Mic(*REQUEST, *ANSWER),
        Wake(),
        Vad(),
        speaker,
        CFG,
        request,
        transcribe_answer=answer,
        clock=Clock(),
    )

    _run(loop)

    assert (len(requests), len(answers)) == (1, 1)
    assert speaker.said[-1] == "approved"


def test_a_yes_heard_in_a_blip_of_noise_is_a_no() -> None:
    """Whisper hears "Okay." in a short noise the VAD let through: too little speech for a yes."""
    agent = Agent(_asking("Should I?"))
    speaker = Speaker()

    _run(_loop(agent, Mic(*REQUEST, SPEECH, *[SILENCE] * 5), speaker, answer=lambda _s: "Okay."))

    assert speaker.said == ["Should I?", "declined"]


def test_yes_with_the_assistants_name_is_a_yes() -> None:
    agent = Agent(_asking("Should I?"))
    speaker = Speaker()

    _run(_loop(agent, Mic(*REQUEST, *ANSWER), speaker, answer=lambda _s: "Yes, Jarvis."))

    assert speaker.said[-1] == "approved"


def test_silence_after_the_question_is_a_no() -> None:
    agent = Agent(_asking("Should I?"))
    speaker = Speaker()

    _run(_loop(agent, Mic(*REQUEST, *[SILENCE] * 40), speaker))

    assert speaker.said[-1] == "declined"


def test_the_confirmation_question_is_spoken_with_nothing_dropped() -> None:
    agent = Agent(_asking("Should I play [Low Tide](Second Sun) by *Glass Harbor*?"))
    speaker = Speaker()

    _run(_loop(agent, Mic(*REQUEST, *ANSWER), speaker))

    question = speaker.said[0]
    for words in ("Low Tide", "Second Sun", "Glass Harbor"):
        assert words in question
    assert not set("[]()*") & set(question)


@pytest.mark.parametrize(("blocks", "approved"), [(True, False), (False, True)])
def test_speech_in_pc_audio_during_the_answer_window_blocks_the_yes(
    *, blocks: bool, approved: bool
) -> None:
    pc_spoke = threading.Event()
    agent = Agent(_asking("Should I?"))
    speaker = Speaker()
    holder: list[VoiceLoop] = []
    pc = PcAudio(
        until(lambda: bool(holder) and holder[0].state is State.CONFIRM),
        SPEECH,
        SPEECH,
        pc_spoke.set,
    )
    mic = Mic(*REQUEST, until(pc_spoke.is_set), *ANSWER)
    loop = _loop(
        agent,
        mic,
        speaker,
        pc=pc,
        cfg=CFG.model_copy(update={"confirm_blocks_on_pc_speech": blocks}),
    )
    holder.append(loop)

    _run(loop)

    assert speaker.said[-1] == ("approved" if approved else "declined")
    assert (voice.PC_SPEECH_BLOCKED in speaker.said) is blocks


def test_a_wake_heard_in_pc_audio_while_capturing_makes_the_request_pc_audio() -> None:
    """Regression: the owner's request was also heard through PC audio and acted on twice."""
    pc_woke = threading.Event()
    holder: list[VoiceLoop] = []
    pc = PcAudio(
        until(lambda: bool(holder) and holder[0].state is State.CAPTURE), WAKE, pc_woke.set
    )
    agent = Agent()
    mic = Mic(WAKE, until(pc_woke.is_set), SPEECH, SPEECH, *[SILENCE] * 5, *[SILENCE] * 5)
    loop = _loop(agent, mic, Speaker(), pc=pc)
    holder.append(loop)

    _run(loop)

    assert agent.calls == [("pause the music", Origin.PC_AUDIO)]


def test_a_pc_capture_of_a_wake_the_loop_already_took_is_dropped() -> None:
    pc_woke = threading.Event()
    pc_queued = threading.Event()
    holder: list[VoiceLoop] = []
    pc = PcAudio(
        WAKE,
        pc_woke.set,
        until(lambda: bool(holder) and holder[0].state is State.CAPTURE),
        SPEECH,
        SPEECH,
        *[SILENCE] * 5,
        SILENCE,
        pc_queued.set,
    )
    agent = Agent()
    mic = Mic(
        until(pc_woke.is_set),
        WAKE,
        SPEECH,
        SPEECH,  # two frames: one is too little for a request
        until(pc_queued.is_set),
        *[SILENCE] * 5,
        *[SILENCE] * 5,
    )
    loop = _loop(agent, mic, Speaker(), pc=pc)
    holder.append(loop)

    _run(loop)

    assert agent.calls == [("pause the music", Origin.PC_AUDIO)]


@pytest.mark.parametrize(("cooldown_s", "handled"), [(5.0, 1), (0.0, 2)])
def test_pc_requests_too_soon_after_the_last_one_are_dropped(
    cooldown_s: float, handled: int
) -> None:
    agent = Agent()
    clock = Clock()
    pc_request = [WAKE, SPEECH, SPEECH, *[SILENCE] * 5]
    pc = PcAudio(
        *pc_request,
        until(lambda: len(agent.calls) == 1 and holder[0].state is State.IDLE),
        clock.advance(1.0),
        *pc_request,
    )
    holder: list[VoiceLoop] = []
    mic = Mic(Idle(lambda: pc.ended.is_set() and len(agent.calls) >= handled), *[SILENCE] * 3)
    loop = _loop(
        agent,
        mic,
        Speaker(),
        pc=pc,
        clock=clock,
        cfg=CFG.model_copy(update={"pc_audio_cooldown_s": cooldown_s}),
    )
    holder.append(loop)

    _run(loop)

    assert len(agent.calls) == handled
    assert {origin for _text, origin in agent.calls} == {Origin.PC_AUDIO}


def test_the_pc_watcher_survives_an_error_and_keeps_listening() -> None:
    agent = Agent()
    pc = PcAudio(SILENCE, WAKE, SPEECH, SPEECH, *[SILENCE] * 5)
    mic = Mic(Idle(agent.called.is_set), *[SILENCE] * 3)

    _run(_loop(agent, mic, Speaker(), pc=pc, pc_wake=Wake(failures=1)))

    assert agent.calls == [("pause the music", Origin.PC_AUDIO)]


def test_a_pc_capture_is_dropped_when_the_loop_takes_another_request_meanwhile(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="jarvis.voice")
    clock = Clock()
    pc_woke = threading.Event()
    pc_done = threading.Event()
    agent = Agent()
    pc = PcAudio(
        WAKE,
        SPEECH,
        pc_woke.set,
        until(agent.called.is_set),
        SPEECH,
        *[SILENCE] * 5,
        SILENCE,
        pc_done.set,
    )
    mic = Mic(
        until(pc_woke.is_set), clock.advance(3.0), *REQUEST, until(pc_done.is_set), *[SILENCE] * 3
    )
    loop = _loop(agent, mic, Speaker(), pc=pc, clock=clock)

    _run(loop)

    assert agent.calls == [("pause the music", Origin.OWNER_SPEECH)]
    assert "dropped a PC audio request: the loop took another one meanwhile" in caplog.messages


def test_no_reply_is_spoken_once_jarvis_is_stopping() -> None:
    stop = threading.Event()
    cancelled: list[bool] = []

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        stop.set()
        cancelled.append(hooks.cancelled())
        return "Paused."

    speaker = Speaker()
    loop = _loop(Agent(reply), Mic(*REQUEST), speaker)

    loop.run(stop.is_set)

    assert cancelled == [True]
    assert speaker.said == []


def test_the_sound_levels_of_each_request_are_logged_as_numbers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="jarvis.voice")
    agent = Agent()

    _run(_loop(agent, Mic(*REQUEST), Speaker()))

    assert len(agent.calls) == 1
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("levels: ")]
    assert len(lines) == 1
    assert "dBFS" in lines[0]


def test_owner_utterances_are_kept_privately_when_debugging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "utterances"
    monkeypatch.setenv("JARVIS_DEBUG_AUDIO_DIR", str(folder))

    _run(_loop(Agent(), Mic(*REQUEST, *REQUEST), Speaker()))

    kept = list(folder.iterdir())
    assert len(kept) == 2
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert {stat.S_IMODE(path.stat().st_mode) for path in kept} == {0o600}


def test_a_debug_folder_that_cannot_be_written_does_not_stop_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("")
    monkeypatch.setenv("JARVIS_DEBUG_AUDIO_DIR", str(blocker))
    agent = Agent()

    _run(_loop(agent, Mic(*REQUEST), Speaker()))

    assert len(agent.calls) == 1


def test_pc_audio_is_never_kept_when_debugging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "utterances"
    monkeypatch.setenv("JARVIS_DEBUG_AUDIO_DIR", str(folder))
    agent = Agent()
    pc = PcAudio(WAKE, SPEECH, SPEECH, *[SILENCE] * 5)
    mic = Mic(Idle(agent.called.is_set), *[SILENCE] * 3)

    _run(_loop(agent, mic, Speaker(), pc=pc))

    assert agent.calls == [("pause the music", Origin.PC_AUDIO)]
    assert not folder.exists() or not list(folder.iterdir())


def test_the_wake_chime_plays_while_the_request_is_already_being_heard() -> None:
    """Words said right after the wake word were lost while the chime played and was drained."""
    capturing = threading.Event()
    speaker = Speaker(during_chime=until(capturing.is_set))
    drains_before_capture: list[int] = []

    def first_capture_frame() -> None:
        drains_before_capture.append(mic.drained)
        capturing.set()

    agent = Agent()
    mic = Mic(WAKE, first_capture_frame, *REQUEST[1:])

    _run(_loop(agent, mic, speaker))

    assert drains_before_capture == [0]
    assert agent.calls == [("pause the music", Origin.OWNER_SPEECH)]


def test_pc_speech_blocks_the_owners_yes_while_a_pc_capture_was_running() -> None:
    """PC audio says the wake word and keeps talking; the owner's confirm window opens 3 s later."""
    clock = Clock()
    pc_woke = threading.Event()
    pc_spoke = threading.Event()
    holder: list[VoiceLoop] = []
    agent = Agent(_asking("Should I?"))
    speaker = Speaker()
    pc = PcAudio(
        WAKE,
        SPEECH,
        pc_woke.set,
        until(lambda: bool(holder) and holder[0].state is State.CONFIRM),
        SPEECH,
        SPEECH,
        pc_spoke.set,
    )
    mic = Mic(
        until(pc_woke.is_set),
        clock.advance(3.0),
        *REQUEST,
        until(pc_spoke.is_set),
        *ANSWER,
        *[SILENCE] * 3,
    )
    loop = _loop(agent, mic, speaker, pc=pc, clock=clock)
    holder.append(loop)

    _run(loop)

    assert speaker.said[-1] == "declined", speaker.said
    assert voice.PC_SPEECH_BLOCKED in speaker.said


def test_pc_audio_cannot_approve_its_own_request_after_a_wake_during_the_chime() -> None:
    """PC audio asks, then says the wake word during the chime, then answers its own question."""
    pc_woke2 = threading.Event()
    pc_spoke = threading.Event()
    holder: list[VoiceLoop] = []
    agent = Agent(_asking("Someone on your PC audio asked me to pause. Should I?"))
    speaker = Speaker(during_chime=until(pc_woke2.is_set))
    pc = PcAudio(
        WAKE,
        SPEECH,
        SPEECH,
        *[SILENCE] * 5,
        until(lambda: speaker.chimes >= 1),
        WAKE,
        pc_woke2.set,
        SPEECH,
        until(lambda: bool(holder) and holder[0].state is State.CONFIRM),
        SPEECH,
        SPEECH,
        pc_spoke.set,
    )
    mic = Mic(
        Idle(agent.called.is_set),
        until(pc_spoke.is_set),
        *ANSWER,
        *[SILENCE] * 3,
    )
    loop = _loop(agent, mic, speaker, pc=pc)
    holder.append(loop)

    _run(loop)

    assert agent.calls[0][1].value == "pc_audio"
    assert "approved" not in speaker.said, speaker.said


def test_a_pc_wake_during_a_pc_capture_still_marks_the_mic_request_as_pc_audio() -> None:
    """PC audio: "Hey Jarvis ... (3 s) ... Hey Jarvis, play X"; the mic hears the second wake."""
    clock = Clock()
    pc_woke = threading.Event()
    pc_second = threading.Event()
    holder: list[VoiceLoop] = []
    agent = Agent()
    pc = PcAudio(
        WAKE,
        SPEECH,
        pc_woke.set,
        until(lambda: bool(holder) and holder[0].state is State.CAPTURE),
        WAKE,  # the same wake word the mic is hearing through the speakers right now
        pc_second.set,
        SPEECH,
        SPEECH,
        *[SILENCE] * 5,
    )
    mic = Mic(
        until(pc_woke.is_set),
        clock.advance(3.0),
        WAKE,
        until(pc_second.is_set),
        SPEECH,
        SPEECH,
        *[SILENCE] * 5,
        *[SILENCE] * 5,
    )
    loop = _loop(agent, mic, Speaker(), pc=pc, clock=clock)
    holder.append(loop)

    _run(loop)

    assert agent.calls[0][1].value == "pc_audio", agent.calls


def test_a_pc_request_said_during_the_chime_of_another_is_not_taken() -> None:
    """Jarvis takes no PC audio request while it is busy with one, its chime included."""
    second_said = threading.Event()
    agent = Agent()
    speaker = Speaker(during_chime=until(second_said.is_set))
    pc = PcAudio(
        WAKE,
        SPEECH,
        SPEECH,
        *[SILENCE] * 5,
        until(lambda: speaker.chimes >= 1),
        WAKE,
        SPEECH,
        SPEECH,
        *[SILENCE] * 5,
        second_said.set,
    )
    mic = Mic(Idle(agent.called.is_set), *[SILENCE] * 30)
    loop = _loop(agent, mic, speaker, pc=pc)

    _run(loop)

    assert [origin for _text, origin in agent.calls] == [Origin.PC_AUDIO]


@dataclass
class WindowWake:
    """Like the real wake model: fires on the third of three wake frames; reset clears them."""

    seen: list[int] = field(default_factory=list[int])

    def heard(self, frame: NDArray[np.int16]) -> bool:
        self.seen = [*self.seen[-2:], int(frame[0])]
        return self.seen == [3, 4, 5]

    def reset(self) -> None:
        self.seen = []


def test_a_pc_wake_heard_after_the_mics_still_marks_the_request_pc_audio() -> None:
    """The PC stream finishes the second wake word two frames after the mic does."""
    first, second, third = (np.full(FRAME, value, dtype=np.int16) for value in (3, 4, 5))
    clock = Clock()
    pc_woke = threading.Event()
    pc_second = threading.Event()
    holder: list[VoiceLoop] = []
    agent = Agent()
    pc = PcAudio(
        first,
        second,
        third,
        SPEECH,
        pc_woke.set,
        first,
        until(lambda: bool(holder) and holder[0].state is State.CAPTURE),
        second,
        third,
        pc_second.set,
        SPEECH,
        SPEECH,
        *[SILENCE] * 5,
    )
    mic = Mic(
        until(pc_woke.is_set),
        clock.advance(3.0),
        WAKE,
        until(pc_second.is_set),
        SPEECH,
        SPEECH,
        *[SILENCE] * 10,
    )
    loop = VoiceLoop(
        agent,
        mic,
        Wake(),
        Vad(),
        Speaker(),
        CFG,
        lambda _samples: "pause the music",
        pc=PcListener(pc, WindowWake(), Vad()),
        transcribe_answer=lambda _samples: "yes",
        names=("Jarvis",),
        clock=clock,
        pc_retry_s=0,
    )
    holder.append(loop)

    _run(loop)

    assert [origin for _text, origin in agent.calls] == [Origin.PC_AUDIO]


@dataclass
class Playing(Speaker):
    """A speaker whose first `holds` speeches keep playing until spoken over or `release`d."""

    holds: int = 1
    release: threading.Event = field(default_factory=threading.Event)
    play_s: float = WAIT_S  # how long a held speech plays when nobody speaks over it
    queued: list[str] = field(default_factory=list[str])
    on_hush: Callable[[], object] = lambda: None
    _over: threading.Event = field(default_factory=threading.Event)

    @override
    def queue(self, text: str) -> None:
        super().queue(text)
        self.queued.append(text)
        self.busy = True

    @override
    def wait(self) -> None:
        if self.busy and self.holds > 0:
            self.holds -= 1
            for _ in range(int(self.play_s / 0.002)):
                if self._over.is_set() or self.release.is_set():
                    break
                self._over.wait(0.002)
            else:
                if self.play_s >= WAIT_S:
                    pytest.fail("the speech was never spoken over")
        self.busy = False
        self._over.clear()
        super().wait()

    @override
    def hush(self) -> None:
        super().hush()
        self.on_hush()
        self.busy = False
        self._over.set()


def _replies(*texts: str) -> tuple[Reply, list[Hooks]]:
    """An agent reply per request, keeping each request's hooks."""
    seen: list[Hooks] = []
    remaining = list(texts)

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        seen.append(hooks)
        return remaining.pop(0)

    return reply, seen


def test_the_wake_word_while_jarvis_speaks_stops_it_and_takes_the_new_request() -> None:
    reply, hooks = _replies("Here is a long answer.", "Skipped.")
    agent = Agent(reply)
    cancelled_after: list[bool] = []
    speaker = Playing(on_hush=lambda: cancelled_after.append(hooks[0].cancelled()))
    mic = Mic(
        *REQUEST,
        until(lambda: speaker.busy),
        SILENCE,
        WAKE,
        SPEECH,
        SPEECH,
        SPEECH,
        *[SILENCE] * 5,
    )

    _run(_loop(agent, mic, speaker))

    assert speaker.hushed == 1
    assert cancelled_after == [True]  # the first request was told to stop
    assert len(agent.calls) == 2
    assert speaker.said == ["Here is a long answer.", "Skipped."]
    assert speaker.chimes == 2  # the new request is greeted like any other


@dataclass
class BargeWake(Wake):
    """Hears the wake frame only at the lower score a barge-in needs, and keeps one miss."""

    miss: str | None = None
    over_speech: int = 0

    @override
    def heard(self, frame: NDArray[np.int16]) -> bool:
        return False if frame[0] == WAKE[0] and self.over_speech else super().heard(frame)

    def heard_over_speech(self, frame: NDArray[np.int16]) -> bool:
        self.over_speech += 1
        return bool(frame[0] == WAKE[0])

    def closest_miss(self) -> str | None:
        miss, self.miss = self.miss, None
        return miss


def test_the_wake_word_over_jarvis_s_speech_is_held_to_the_barge_in_score() -> None:
    reply, _ = _replies("Here is a long answer.", "Skipped.")
    speaker = Playing()
    wake = BargeWake()
    mic = Mic(*REQUEST, until(lambda: speaker.busy), SILENCE, WAKE, *REQUEST[1:])

    _run(_loop(Agent(reply), mic, speaker, wake=wake))

    assert speaker.hushed == 1  # only the lower score heard it
    assert wake.over_speech > 0


def test_a_wake_score_that_fell_short_over_jarvis_s_speech_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="jarvis.voice")
    speaker = Playing(play_s=0.05)
    wake = BargeWake(miss="yo_jarvis score 0.42 (threshold 0.50)")

    _run(
        _loop(Agent(lambda _t, _h, _o: "Here is a long answer."), Mic(*REQUEST), speaker, wake=wake)
    )

    assert (
        "no wake word while speaking; the closest was yo_jarvis score 0.42 (threshold 0.50)"
        in caplog.text
    )


def test_the_owner_can_switch_speaking_over_jarvis_off() -> None:
    agent = Agent(lambda _t, _h, _o: "Here is a long answer.")
    speaker = Playing(play_s=0.05)
    mic = Mic(*REQUEST, WAKE)  # with nobody listening, the wake is only heard after the speech

    _run(_loop(agent, mic, speaker, cfg=CFG.model_copy(update={"barge_in": False})))

    assert speaker.hushed == 0


def test_a_sentence_that_would_wake_jarvis_itself_is_not_spoken_over() -> None:
    agent = Agent(lambda _t, _h, _o: "Hey, Jarvis here.")
    speaker = Playing(interruptible=False)
    mic = Mic(*REQUEST, until(lambda: speaker.busy), WAKE, speaker.release.set, SILENCE)

    _run(_loop(agent, mic, speaker))

    assert speaker.hushed == 0
    assert len(agent.calls) == 1  # the wake frame went to the watcher, which ignored it


def test_speaking_over_a_confirmation_question_is_a_no_and_starts_a_new_request() -> None:
    answers: list[bool | Declined] = []

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        if answers:
            return "Fine."
        answers.append(hooks.confirm("Should I buy it?"))
        return "Okay, I'll leave it."

    agent = Agent(reply)
    speaker = Playing()
    mic = Mic(*REQUEST, until(lambda: speaker.busy), WAKE, *REQUEST[1:])

    _run(_loop(agent, mic, speaker))

    assert answers == [Declined(NoReason.SPOKEN_OVER)]
    assert speaker.said == ["Should I buy it?", "Fine."]


def test_streamed_parts_of_a_reply_are_queued_sentence_by_sentence() -> None:
    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        assert hooks.speak is not None
        hooks.speak("**First** part. Second part!")
        return ""

    speaker = Speaker()
    _run(_loop(Agent(reply), Mic(*REQUEST), speaker))

    assert speaker.said == ["First part.", "Second part!"]


@dataclass
class NotingWake(Wake):
    fired: threading.Event = field(default_factory=threading.Event)

    @override
    def heard(self, frame: NDArray[np.int16]) -> bool:
        found = super().heard(frame)
        if found:
            self.fired.set()
        return found


def test_a_wake_said_over_jarvis_that_pc_audio_also_held_is_pc_audio() -> None:
    reply, _ = _replies("Here is a long answer.", "Skipped.")
    agent = Agent(reply)
    speaker = Playing()
    pc_wake = NotingWake()
    pc = PcAudio(Idle(lambda: speaker.busy), WAKE)
    mic = Mic(*REQUEST, until(pc_wake.fired.is_set), WAKE, *REQUEST[1:])
    cfg = CFG.model_copy(update={"listen_to_pc": True})

    _run(_loop(agent, mic, speaker, pc=pc, cfg=cfg, pc_wake=pc_wake))

    assert [origin for _text, origin in agent.calls] == [Origin.OWNER_SPEECH, Origin.PC_AUDIO]


def test_jarvis_own_voice_is_drained_after_speech_but_not_after_a_barge_in() -> None:
    plain = Mic(*REQUEST)
    _run(_loop(Agent(lambda _t, _h, _o: "Paused."), plain, Speaker()))
    reply, _ = _replies("Here is a long answer.", "Skipped.")
    speaker = Playing()
    barged = Mic(*REQUEST, until(lambda: speaker.busy), WAKE, *REQUEST[1:])
    _run(_loop(Agent(reply), barged, speaker))

    # Once after the speech and once as the loop goes back to waiting. With a barge-in there are
    # two speeches but the first is not drained, so the owner's words after the wake are kept.
    assert plain.drained == 2
    assert barged.drained == 2


def test_nothing_more_is_said_once_the_owner_spoke_over_jarvis() -> None:
    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        if hooks.speak is None:
            return ""
        hooks.speak("Here is a long answer.")
        until(hooks.cancelled)()
        hooks.waking()  # a line that comes after the barge-in must not be spoken
        hooks.speak("And more.")
        return ""

    calls: list[int] = []

    def counted(text: str, hooks: Hooks, origin: Origin) -> str:
        calls.append(1)
        return reply(text, hooks, origin) if len(calls) == 1 else ""

    speaker = Playing()
    mic = Mic(*REQUEST, until(lambda: speaker.busy), WAKE, *REQUEST[1:])
    _run(_loop(Agent(counted), mic, speaker))

    assert speaker.said == ["Here is a long answer."]


def test_dropping_unheard_words_puts_the_loop_back_to_thinking() -> None:
    states: list[State] = []
    loops: list[VoiceLoop] = []

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        assert hooks.speak is not None
        hooks.speak("Let me look.")
        hooks.hush()
        states.append(loops[0].state)
        return ""

    loop = _loop(Agent(reply), Mic(*REQUEST), Speaker())
    loops.append(loop)
    _run(loop)

    assert states == [State.THINK]


# Follow-up requests (#11): after a spoken reply, `follow_up_s` of speech needs no wake word.
FOLLOW_UP = CFG.model_copy(update={"follow_up_s": 1.0})
FOLLOW_UP_REQUEST = [SPEECH, SPEECH, SPEECH, SPEECH, *[SILENCE] * 5]  # 0.32 s of speech
WINDOW_FRAMES = 13  # 1.0 s of 80 ms frames: the window ends after the 13th silent frame
# A request whose capture uses up every frame (0.3 s of trailing silence is 4 frames), so the
# window starts at the next frame.
EXACT_REQUEST = [WAKE, SPEECH, SPEECH, SPEECH, *[SILENCE] * 4]


def _texts(*texts: str) -> Callable[[NDArray[np.int16]], str]:
    """A transcriber that hears `texts` in turn, then the last one again."""
    heard = list(texts) or ["pause the music"]

    def transcribe(_samples: NDArray[np.int16]) -> str:
        return heard.pop(0) if len(heard) > 1 else heard[0]

    return transcribe


def _follow_loop(
    agent: Agent,
    mic: Mic,
    speaker: Speaker,
    *texts: str,
    cfg: AudioConfig = FOLLOW_UP,
    dictating: Callable[[], bool | None] = lambda: False,
    meeting: Callable[[], bool] = lambda: False,
    paused: Callable[[], bool] = lambda: False,
    media: Callable[[], bool | None] = lambda: False,
) -> VoiceLoop:
    return _loop(
        agent,
        mic,
        speaker,
        cfg=cfg,
        dictating=dictating,
        meeting=meeting,
        media=media,
        paused=paused,
        transcribe=_texts(*texts),
    )


def test_follow_up_listening_is_on_by_default_for_six_seconds() -> None:
    """The owner wants it on (#11, 2026-11-21); 0 turns it off."""
    assert AudioConfig().follow_up_s == pytest.approx(6.0)
    agent = Agent()
    cfg = CFG.model_copy(update={"follow_up_s": AudioConfig().follow_up_s})

    _run(_follow_loop(agent, Mic(*REQUEST, *FOLLOW_UP_REQUEST), Speaker(), "a", "b", cfg=cfg))

    assert [text for text, _ in agent.calls] == ["a", "b"]


def test_a_window_of_0_turns_follow_up_listening_off() -> None:
    agent = Agent()

    _run(_loop(agent, Mic(*REQUEST, *FOLLOW_UP_REQUEST), Speaker()))

    assert len(agent.calls) == 1


def _counting(text: str, transcribed: list[int]) -> Callable[[NDArray[np.int16]], str]:
    def transcribe(samples: NDArray[np.int16]) -> str:
        transcribed.append(len(samples))
        return text

    return transcribe


@pytest.mark.parametrize("frames", [1, 2, 3])
def test_too_little_speech_in_the_window_is_dropped_untranscribed(frames: int) -> None:
    """A cough, a click or one word from the room is not a request."""
    agent = Agent()
    speaker = Speaker()
    transcribed: list[int] = []
    states: list[State] = []
    mic = Mic(
        *EXACT_REQUEST,
        *[SPEECH] * frames,
        *[SILENCE] * WINDOW_FRAMES,
        lambda: states.append(loop.state),
        *FOLLOW_UP_REQUEST,
    )
    loop = _loop(agent, mic, speaker, cfg=FOLLOW_UP, transcribe=_counting("okay", transcribed))

    _run(loop)

    assert len(agent.calls) == 1
    assert len(transcribed) == 1
    assert speaker.said == ["Done."]
    assert states == [State.IDLE]


def test_the_window_listens_on_after_too_little_speech() -> None:
    agent = Agent()
    transcribed: list[int] = []
    mic = Mic(*EXACT_REQUEST, SPEECH, *[SILENCE] * 5, *FOLLOW_UP_REQUEST)

    _run(_loop(agent, mic, Speaker(), cfg=FOLLOW_UP, transcribe=_counting("a", transcribed)))

    assert len(agent.calls) == 2
    assert len(transcribed) == 2


def test_the_window_keeps_its_length_across_too_little_speech() -> None:
    """Speech dropped as too short does not restart the window's time."""
    agent = Agent()
    mic = Mic(*EXACT_REQUEST, SPEECH, *[SILENCE] * (WINDOW_FRAMES - 1), *FOLLOW_UP_REQUEST)

    _run(_follow_loop(agent, mic, Speaker()))

    assert len(agent.calls) == 1


def test_speech_right_after_a_reply_is_the_next_request_without_the_wake_word() -> None:
    agent = Agent()
    speaker = Speaker()
    mic = Mic(*REQUEST, *FOLLOW_UP_REQUEST)

    _run(_follow_loop(agent, mic, speaker, "pause the music", "and turn it down"))

    assert agent.calls == [
        ("pause the music", Origin.OWNER_SPEECH),
        ("and turn it down", Origin.OWNER_SPEECH),
    ]
    assert speaker.said == ["Done.", "Done."]
    assert speaker.chimes == 1  # only the wake word is greeted


def test_a_follow_up_opens_another_window() -> None:
    agent = Agent()

    _run(
        _follow_loop(
            agent, Mic(*REQUEST, *FOLLOW_UP_REQUEST, *FOLLOW_UP_REQUEST), Speaker(), "a", "b", "c"
        )
    )

    assert [text for text, _ in agent.calls] == ["a", "b", "c"]


def test_a_request_answered_with_silence_opens_no_follow_up_window() -> None:
    """Told to be quiet, Jarvis says nothing and stops listening without the wake word."""
    agent = Agent(lambda _t, _h, _o: "")

    _run(_follow_loop(agent, Mic(*REQUEST, *FOLLOW_UP_REQUEST), Speaker(), "be quiet", "b"))

    assert [text for text, _ in agent.calls] == ["be quiet"]


def test_quiet_asked_for_after_words_were_spoken_still_opens_no_follow_up_window() -> None:
    """Regression: words spoken before the quiet call made the silent reply look answered."""

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        assert hooks.speak is not None
        hooks.speak("Understood.")
        hooks.hush()
        hooks.quiet()
        return ""

    agent = Agent(reply)
    speaker = Speaker()

    _run(_follow_loop(agent, Mic(*REQUEST, *FOLLOW_UP_REQUEST), speaker, "be quiet", "b"))

    assert [text for text, _ in agent.calls] == ["be quiet"]
    assert speaker.said == ["Understood."]


def test_a_hushed_reply_opens_no_follow_up_window() -> None:
    """`jarvis hush` ends the exchange: speech right after it needs the wake word again."""
    agent = Agent()

    def reply(_text: str, _hooks: Hooks, _origin: Origin) -> str:
        loop.hush()  # `jarvis hush` from the control thread
        return "Done."

    agent.reply = reply
    loop = _follow_loop(agent, Mic(*REQUEST, *FOLLOW_UP_REQUEST), Speaker(), "a", "b")

    _run(loop)

    assert [text for text, _ in agent.calls] == ["a"]


def test_a_reply_ended_by_a_pause_opens_no_follow_up_window_even_if_it_is_lifted_at_once() -> None:
    """The pause is lifted by the time the loop looks for the next request; the reply stays over."""
    agent = Agent()
    loop = _loop(
        agent,
        Mic(*REQUEST, *FOLLOW_UP_REQUEST),
        Speaker(),
        cfg=FOLLOW_UP,
        paused=lambda: loop.state is State.THINK,  # paused while the agent works, not after
        transcribe=_texts("a", "b"),
    )

    _run(loop)

    assert [text for text, _ in agent.calls] == ["a"]


def test_silence_through_the_window_goes_back_to_waiting_for_the_wake_word() -> None:
    agent = Agent()
    states: list[State] = []
    mic = Mic(
        *EXACT_REQUEST,
        *[SILENCE] * WINDOW_FRAMES,
        lambda: states.append(loop.state),
        *FOLLOW_UP_REQUEST,
    )
    loop = _follow_loop(agent, mic, Speaker())

    _run(loop)

    assert len(agent.calls) == 1
    assert states == [State.IDLE]


def test_speech_just_inside_the_window_is_taken() -> None:
    agent = Agent()
    mic = Mic(*EXACT_REQUEST, *[SILENCE] * (WINDOW_FRAMES - 1), *FOLLOW_UP_REQUEST)

    _run(_follow_loop(agent, mic, Speaker()))

    assert len(agent.calls) == 2


def test_the_loop_is_in_its_follow_up_state_while_the_window_is_open() -> None:
    states: list[State] = []
    mic = Mic(*REQUEST, SILENCE, lambda: states.append(loop.state), *[SILENCE] * WINDOW_FRAMES)
    loop = _follow_loop(Agent(), mic, Speaker())

    _run(loop)

    assert states == [State.FOLLOW_UP]


def test_the_wake_word_inside_the_window_starts_one_request() -> None:
    agent = Agent()
    speaker = Speaker()

    _run(_follow_loop(agent, Mic(*REQUEST, SILENCE, *REQUEST), speaker, "a", "b"))

    assert [text for text, _ in agent.calls] == ["a", "b"]
    assert speaker.chimes == 2  # greeted like any wake


def test_no_window_opens_while_the_owner_is_dictating() -> None:
    agent = Agent()
    # The owner starts dictating once the request is taken: a wake word said into dictation is
    # ignored, so it cannot be dictating from the start.

    _run(
        _follow_loop(
            agent,
            Mic(*REQUEST, *FOLLOW_UP_REQUEST),
            Speaker(),
            dictating=lambda: bool(agent.calls),
        )
    )

    assert len(agent.calls) == 1


def test_speech_in_the_window_is_dropped_when_dictation_started_meanwhile() -> None:
    agent = Agent()
    dictating = [False]
    mic = Mic(*REQUEST, SPEECH, lambda: dictating.__setitem__(0, True), *FOLLOW_UP_REQUEST[1:])

    _run(_follow_loop(agent, mic, Speaker(), dictating=lambda: dictating[0]))

    assert len(agent.calls) == 1


def test_no_window_opens_after_a_reply_that_ends_in_a_pause() -> None:
    # A regression guard for the whole loop: `_next_request` already ends on a pause, so this
    # does not isolate the guard inside `_follow_up`.
    paused = [False]
    states: list[State] = []

    def reply(_text: str, _hooks: Hooks, _origin: Origin) -> str:
        paused[0] = True  # `jarvis pause` while the reply streams
        return "Done."

    agent = Agent(reply)
    mic = Mic(*REQUEST, SILENCE, lambda: states.append(loop.state), *FOLLOW_UP_REQUEST)
    loop = _follow_loop(agent, mic, Speaker(), paused=lambda: paused[0])

    _run(loop)

    assert len(agent.calls) == 1
    assert State.FOLLOW_UP not in states


def test_speech_in_the_window_is_dropped_when_jarvis_was_paused_meanwhile() -> None:
    agent = Agent()
    paused = [False]
    mic = Mic(*REQUEST, SPEECH, lambda: paused.__setitem__(0, True), *FOLLOW_UP_REQUEST[1:])

    _run(_follow_loop(agent, mic, Speaker(), paused=lambda: paused[0]))

    assert len(agent.calls) == 1


def test_a_pause_in_the_open_window_takes_no_request_and_listening_resumes() -> None:
    # The real microphone ends the window at the pause (a `MicPausedError` at its next frame);
    # what was said before the resume is not heard, and a wake word after it is.
    agent = Agent()

    def resume_soon() -> None:
        def later() -> None:
            until(lambda: loop.state is State.IDLE)()
            assert mic.resume()

        threading.Thread(target=later, daemon=True).start()

    def pause() -> None:
        mic.pause()

    script = Capture(*REQUEST, SILENCE, pause, resume_soon, *FOLLOW_UP_REQUEST, *REQUEST)
    mic = PausableMic(script)
    loop = VoiceLoop(
        agent,
        mic,
        Wake(),
        Vad(),
        Speaker(),
        FOLLOW_UP,
        _texts("a", "b", "c"),
        paused=lambda: mic.paused,
    )
    stop = threading.Event()

    def stop_after_answers() -> None:
        until(lambda: len(agent.calls) == 2 and loop.state is State.IDLE)()
        stop.set()

    threading.Thread(target=stop_after_answers, daemon=True).start()
    try:
        loop.run(stop.is_set)
    except AudioError:
        pass  # the script ended
    finally:
        stop.set()

    assert [text for text, _ in agent.calls] == ["a", "b"]  # the wake word's request, not "speech"


def test_an_unclear_follow_up_is_dropped_without_an_apology() -> None:
    agent = Agent()
    speaker = Speaker()

    _run(_follow_loop(agent, Mic(*REQUEST, *FOLLOW_UP_REQUEST), speaker, "pause the music", ""))

    assert len(agent.calls) == 1
    assert speaker.said == ["Done."]


def test_a_confirmation_answer_is_not_taken_for_a_follow_up() -> None:
    agent = Agent(_asking("Delete it?"))
    speaker = Speaker()
    mic = Mic(*REQUEST, until(lambda: speaker.said != []), *ANSWER, *FOLLOW_UP_REQUEST)

    _run(_follow_loop(agent, mic, speaker))

    assert len(agent.calls) == 2  # the request, then the follow-up; the yes was the answer
    assert speaker.said[1] == "approved"


def test_no_window_follows_a_request_heard_in_pc_audio() -> None:
    agent = Agent()
    pc = PcAudio(*REQUEST)
    mic = Mic(Idle(lambda: len(agent.calls) == 1), *FOLLOW_UP_REQUEST)

    _run(_loop(agent, mic, Speaker(), pc=pc, cfg=FOLLOW_UP, clock=Clock()))

    assert [origin for _, origin in agent.calls] == [Origin.PC_AUDIO]


def test_no_window_opens_once_jarvis_is_stopping() -> None:
    agent = Agent()
    stop = threading.Event()
    loop = _follow_loop(agent, Mic(*REQUEST, stop.set, *FOLLOW_UP_REQUEST), Speaker())

    loop.run(stop.is_set)

    assert len(agent.calls) == 1


def test_a_wake_without_a_request_opens_no_window() -> None:
    agent = Agent()

    _run(_follow_loop(agent, Mic(WAKE, *[SILENCE] * 14, *FOLLOW_UP_REQUEST), Speaker()))

    assert agent.calls == []


def test_the_window_does_not_even_open_while_the_owner_is_dictating() -> None:
    states: list[State] = []
    agent = Agent()
    mic = Mic(*EXACT_REQUEST, SILENCE, lambda: states.append(loop.state), SILENCE)
    loop = _follow_loop(agent, mic, Speaker(), dictating=lambda: bool(agent.calls))

    _run(loop)

    assert agent.calls  # the request was taken, then dictation began
    assert states == [State.IDLE]


def test_speech_in_the_window_is_dropped_when_jarvis_starts_stopping() -> None:
    agent = Agent()
    stop = threading.Event()
    loop = _follow_loop(agent, Mic(*EXACT_REQUEST, SPEECH, SPEECH, stop.set, SPEECH), Speaker())

    loop.run(stop.is_set)

    assert len(agent.calls) == 1


def test_a_follow_up_is_dropped_when_pc_audio_held_speech_in_the_window() -> None:
    """A voice on a call reaching the mic must not become a request without the wake word."""
    holder: list[VoiceLoop] = []
    pc_spoke = threading.Event()
    agent = Agent()
    pc = PcAudio(
        until(lambda: bool(holder) and holder[0].state is State.FOLLOW_UP),
        SPEECH,
        SPEECH,
        pc_spoke.set,
    )
    mic = Mic(*EXACT_REQUEST, until(pc_spoke.is_set), *FOLLOW_UP_REQUEST)
    loop = _loop(agent, mic, Speaker(), pc=pc, cfg=FOLLOW_UP, transcribe=_texts("a", "b"))
    holder.append(loop)

    _run(loop)

    assert [text for text, _ in agent.calls] == ["a"]


def test_a_follow_up_is_taken_when_pc_audio_was_quiet() -> None:
    agent = Agent()
    pc = PcAudio(SILENCE)
    mic = Mic(*EXACT_REQUEST, until(pc.ended.is_set), *FOLLOW_UP_REQUEST)

    _run(_loop(agent, mic, Speaker(), pc=pc, cfg=FOLLOW_UP, transcribe=_texts("a", "b")))

    assert [text for text, _ in agent.calls] == ["a", "b"]


class Capture:
    """A capture process stand-in that plays one shared script and ends when stopped."""

    def __init__(self, *steps: Step) -> None:
        super().__init__()
        self._frames = _steps(steps)
        self._ended = threading.Event()
        self.starts = 0
        self.stops = 0

    @property
    def pid(self) -> int | None:
        return None

    def start(self) -> None:
        self.starts += 1
        self._ended = threading.Event()

    def stop(self) -> None:
        self.stops += 1
        self._ended.set()

    def end(self) -> None:
        """End on its own, the way `pw-record` does when PipeWire restarts."""
        self._ended.set()

    def drain(self) -> None:
        pass

    def frames(self) -> Iterator[NDArray[np.int16]]:
        ended = self._ended
        while not ended.is_set():  # a stopped capture reads nothing more from the script
            frame = next(self._frames, None)
            if frame is None:
                return
            yield frame
        msg = "the microphone stream ended"
        raise AudioError(msg)


def test_a_pause_ends_the_request_in_flight_silently_and_listening_resumes() -> None:
    agent = Agent()
    speaker = Speaker()
    resumed = threading.Event()

    def resume_soon() -> None:
        def later() -> None:
            until(lambda: loop.state is State.IDLE)()
            assert mic.resume()
            resumed.set()

        threading.Thread(target=later, daemon=True).start()

    def pause() -> None:
        mic.pause()  # the mic made below; a bound method taken now would be unset

    script = Capture(WAKE, SPEECH, pause, resume_soon, SPEECH, *REQUEST)
    mic = PausableMic(script)
    loop = VoiceLoop(
        agent,
        mic,
        Wake(),
        Vad(),
        speaker,
        CFG,
        lambda _samples: "pause the music",
        paused=lambda: mic.paused,
    )
    stop = threading.Event()

    def stop_after_answer() -> None:
        until(lambda: bool(agent.calls) and loop.state is State.IDLE)()
        stop.set()

    threading.Thread(target=stop_after_answer, daemon=True).start()
    try:
        loop.run(stop.is_set)
    except AudioError:
        pass  # the script ended
    finally:
        stop.set()

    assert resumed.is_set()
    assert agent.calls == [("pause the music", Origin.OWNER_SPEECH)]
    assert speaker.said == ["Done."]  # no apology for the request the pause ended
    assert script.stops == 1
    assert script.starts == 1


def test_hush_silences_the_rest_of_a_streamed_reply_but_the_request_carries_on() -> None:
    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        assert hooks.speak is not None
        hooks.speak("One.")
        loop.hush()  # `jarvis hush` from the control thread
        hooks.speak("Two.")
        return "Three."

    agent = Agent(reply)
    speaker = Speaker()
    loop = _loop(agent, Mic(*REQUEST, *REQUEST), speaker)

    _run(loop)

    assert len(agent.calls) == 2  # the hushed request ran to its end
    assert speaker.said == ["One.", "One."]  # each request is hushed after its first sentence
    assert speaker.hushed >= 2


def test_a_hush_lasts_only_for_the_request_it_came_in() -> None:
    hushed = [False]

    def reply(_text: str, _hooks: Hooks, _origin: Origin) -> str:
        if not hushed[0]:
            hushed[0] = True
            loop.hush()
            return "Hushed."
        return "Heard."

    agent = Agent(reply)
    speaker = Speaker()
    loop = _loop(agent, Mic(*REQUEST, *REQUEST), speaker)

    _run(loop)

    assert speaker.said == ["Heard."]


def test_a_paused_loop_says_nothing_more() -> None:
    paused = [False]

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        assert hooks.speak is not None
        hooks.speak("One.")
        paused[0] = True  # `jarvis pause` while the reply streams
        hooks.speak("Two.")
        return "Three."

    agent = Agent(reply)
    speaker = Speaker()
    loop = VoiceLoop(
        agent,
        Mic(*REQUEST),
        Wake(),
        Vad(),
        speaker,
        CFG,
        lambda _samples: "pause the music",
        paused=lambda: paused[0],
    )

    _run(loop)

    assert speaker.said == ["One."]


def _reopening_loop(
    agent: Agent, mic: PausableMic, reopened: list[State]
) -> tuple[VoiceLoop, Callable[[], bool]]:
    holder: list[VoiceLoop] = []

    def reopen() -> bool:
        reopened.append(holder[0].state)
        return mic.recover()

    loop = VoiceLoop(
        agent,
        mic,
        Wake(),
        Vad(),
        Speaker(),
        CFG,
        lambda _samples: "pause the music",
        paused=lambda: mic.paused,
        until_resumed=mic.wait_resumed,
        reopen=reopen,
    )
    holder.append(loop)
    return loop, reopen


def _no_waits(_event: threading.Event, _seconds: float) -> bool:
    return False


def test_a_microphone_that_ends_on_its_own_is_reopened_and_requests_are_heard_again() -> None:
    # PipeWire restarting ends `pw-record`; on 2026-11-21 that ended `jarvis run` with an error.
    agent = Agent()
    reopened: list[State] = []

    def end() -> None:
        script.end()

    script = Capture(SILENCE, end, SILENCE, *REQUEST)
    mic = PausableMic(script, wait=_no_waits)
    loop, _ = _reopening_loop(agent, mic, reopened)

    _run(loop)  # the script's end ends the loop

    assert reopened == [State.NO_MIC]  # the bar shows Jarvis unavailable meanwhile
    assert script.starts == 1  # the reopen started a new capture
    assert agent.calls == [("pause the music", Origin.OWNER_SPEECH)]
    assert loop.state is State.IDLE


def test_a_reopen_refused_by_a_stopped_microphone_ends_the_loop() -> None:
    agent = Agent()

    def end() -> None:
        script.end()  # the capture made below; a bound method taken now would be unset

    script = Capture(SILENCE, end, SILENCE)
    mic = PausableMic(script, wait=_no_waits)
    reopened: list[State] = []
    loop, _ = _reopening_loop(agent, mic, reopened)
    mic.stop()  # `jarvis stop`: the reader ends and the reopen answers False

    loop.run(lambda: False)  # returns instead of reopening for ever

    assert reopened == [State.NO_MIC]
    assert script.starts == 0
    assert loop.state is State.IDLE


def test_a_microphone_ended_by_stopping_jarvis_is_not_reopened() -> None:
    reopened: list[State] = []
    stopping = [False]

    def stop() -> None:
        stopping[0] = True
        mic.stop()

    script = Capture(SILENCE, stop, SILENCE)
    mic = PausableMic(script, wait=_no_waits)
    loop, _ = _reopening_loop(Agent(), mic, reopened)

    loop.run(lambda: stopping[0])

    assert reopened == []


def test_without_a_reopen_an_ended_microphone_still_ends_the_loop() -> None:
    def end() -> None:
        script.end()  # the capture made below; a bound method taken now would be unset

    script = Capture(SILENCE, end, SILENCE)
    loop = VoiceLoop(Agent(), script, Wake(), Vad(), Speaker(), CFG, lambda _s: "")

    with pytest.raises(AudioError):
        loop.run(lambda: False)


def test_a_wake_word_is_ignored_while_the_screen_is_locked() -> None:
    agent = Agent()
    asked: list[bool] = []

    def locked() -> bool:
        asked.append(True)
        return True

    _run(_loop(agent, Mic(*REQUEST), Speaker(), locked=locked))

    assert asked == [True]  # asked once, when the wake word came, not per frame
    assert agent.calls == []


def test_a_wake_word_is_answered_while_unlocked() -> None:
    agent = Agent()

    _run(_loop(agent, Mic(*REQUEST), Speaker(), locked=lambda: False))

    assert agent.calls == [("pause the music", Origin.OWNER_SPEECH)]


def test_a_pc_request_is_ignored_while_the_screen_is_locked() -> None:
    agent = Agent()
    asked: list[bool] = []

    def locked() -> bool:
        asked.append(True)
        return True

    pc = PcAudio(SILENCE, WAKE, SPEECH, SPEECH, *[SILENCE] * 5)
    mic = Mic(Idle(lambda: bool(asked)), *[SILENCE] * 3)

    _run(_loop(agent, mic, Speaker(), pc=pc, locked=locked))

    assert agent.calls == []


def test_after_a_suspend_the_loop_listens_afresh_and_says_so() -> None:
    agent = Agent()
    asleep = [0.0]
    woke: list[bool] = []

    def suspend() -> None:
        asleep[0] += 600.0  # ten minutes asleep: the boot clock ran on, the monotonic did not

    drained: list[int] = []
    mic = Mic(SILENCE, suspend, SILENCE, SILENCE, lambda: drained.append(mic.drained), *REQUEST)
    loop = _loop(
        agent, mic, Speaker(), asleep_s=lambda: asleep[0], after_sleep=lambda: woke.append(True)
    )

    _run(loop)

    assert woke == [True]
    assert drained == [1]  # frames from before the sleep were dropped
    assert agent.calls == [("pause the music", Origin.OWNER_SPEECH)]


def test_no_suspend_is_seen_when_the_clocks_agree() -> None:
    woke: list[bool] = []
    asleep = [0.0]

    def drift() -> None:
        asleep[0] += 0.5  # clock noise, not a sleep

    _run(
        _loop(
            Agent(),
            Mic(SILENCE, drift, SILENCE, *REQUEST),
            Speaker(),
            asleep_s=lambda: asleep[0],
            after_sleep=lambda: woke.append(True),
        )
    )

    assert woke == []


def test_the_lock_screen_is_asked_about_only_once_a_follow_up_is_heard() -> None:
    asked: list[bool] = []

    def locked() -> bool:
        asked.append(True)
        return False

    _run(_loop(Agent(), Mic(*REQUEST), Speaker(), locked=locked))  # no follow-up window here

    assert asked == [True]  # the wake word's question only


def test_no_follow_up_is_taken_once_the_screen_locked() -> None:
    agent = Agent()

    def locked() -> bool:
        return bool(agent.calls)  # the screen locked while Jarvis answered

    mic = Mic(*REQUEST, *FOLLOW_UP_REQUEST)
    loop = _loop(agent, mic, Speaker(), cfg=FOLLOW_UP, transcribe=_texts("a", "b"), locked=locked)

    _run(loop)

    assert [text for text, _ in agent.calls] == ["a"]


def test_a_request_spoken_over_jarvis_at_the_lock_screen_is_ignored() -> None:
    reply, _ = _replies("Here is a long answer.", "Skipped.")
    agent = Agent(reply)
    speaker = Playing()
    mic = Mic(*REQUEST, until(lambda: speaker.busy), SILENCE, WAKE, *REQUEST[1:])

    _run(_loop(agent, mic, speaker, locked=lambda: bool(agent.calls)))

    assert speaker.hushed == 1  # the wake word still stops the speech
    assert len(agent.calls) == 1  # but starts no request


def test_a_microphone_lost_mid_request_ends_it_without_an_apology() -> None:
    # Saying sorry to a microphone that is gone helps nobody; the loop reopens it next.
    agent = Agent()
    speaker = Speaker()

    def end() -> None:
        script.end()

    script = Capture(WAKE, SPEECH, end, SILENCE, *REQUEST)
    mic = PausableMic(script, wait=_no_waits)
    loop = VoiceLoop(
        agent,
        mic,
        Wake(),
        Vad(),
        speaker,
        CFG,
        lambda _samples: "pause the music",
        reopen=mic.recover,
    )

    _run(loop)

    assert speaker.said == ["Done."]  # the second request's answer, and no apology
    assert agent.calls == [("pause the music", Origin.OWNER_SPEECH)]


def test_a_wake_word_said_while_the_owner_dictates_starts_no_request() -> None:
    agent = Agent()
    speaker = Speaker()
    audit = AuditLog(None)

    def fail(_samples: NDArray[np.int16]) -> str:
        pytest.fail("dictation was sent to the transcriber")

    _run(_loop(agent, Mic(*REQUEST), speaker, dictating=lambda: True, audit=audit, transcribe=fail))

    assert agent.calls == []
    assert speaker.chimes == 0
    assert speaker.said == []
    assert [(e["event"], e["reason"]) for e in audit.events] == [("wake_ignored", "dictating")]


def test_dictation_that_starts_mid_request_drops_it_without_a_word() -> None:
    # The owner starts their own dictation while Jarvis is capturing: what is said from here on
    # is going into it. Jarvis lets go of its capture; the conftest guard fails the test should
    # it run `voxtype` to stop or start anything, and the loop has no other way to reach it.
    agent = Agent()
    speaker = Speaker()
    audit = AuditLog(None)
    dictating = [False]

    def fail(_samples: NDArray[np.int16]) -> str:
        pytest.fail("dictation was sent to the transcriber")

    mic = Mic(WAKE, SPEECH, lambda: dictating.__setitem__(0, True), *REQUEST[2:])

    _run(
        _loop(
            agent,
            mic,
            speaker,
            dictating=lambda: dictating[0],
            audit=audit,
            transcribe=fail,
        )
    )

    assert agent.calls == []
    assert speaker.said == []
    assert [(e["event"], e["reason"]) for e in audit.events] == [
        ("wake_ignored", "dictation started mid-request")
    ]


class CountingMic(Mic):
    """A `Mic` that counts the frames handed out."""

    def __init__(self, *steps: Step) -> None:
        super().__init__(*steps)
        self.read = 0

    @override
    def frames(self) -> Iterator[NDArray[np.int16]]:
        for frame in super().frames():
            self.read += 1
            yield frame


def test_the_capture_stops_at_the_frame_where_dictation_starts() -> None:
    # The capture would run on through the speech and the silence after it; it must stop at once.
    # The loop goes on listening afterwards, so the frame count is read when dictation is looked at.
    first_dictated_frame = 4
    mic = CountingMic(WAKE, *[SPEECH] * 20, *[SILENCE] * 5)
    looked_at: list[int] = []

    def dictating() -> bool:
        looked_at.append(mic.read)
        return mic.read >= first_dictated_frame

    _run(_loop(Agent(), mic, Speaker(), dictating=dictating, audit=AuditLog(None)))

    # The wake word, then one look per frame of the capture up to the one that showed it.
    assert looked_at == [1, 2, 3, 4]


def test_dictation_over_before_the_capture_ends_still_drops_the_request() -> None:
    # Dictation that is on only for the look after the last frame (never during the capture) is
    # still seen: the post-capture re-check cuts the request, as the speech may have gone into it.
    agent = Agent()
    mic = CountingMic(WAKE, SPEECH, SPEECH, SPEECH, *[SILENCE] * 5)
    looked_at: list[int] = []

    def dictating() -> bool:
        repeat = bool(looked_at) and looked_at[-1] == mic.read
        looked_at.append(mic.read)
        return repeat  # true only for a look that follows no new frame: the re-check

    _run(_loop(agent, mic, Speaker(), dictating=dictating, audit=AuditLog(None)))

    assert agent.calls == []


def test_a_request_is_taken_again_after_dictation_that_cut_one_short() -> None:
    agent = Agent()
    audit = AuditLog(None)
    dictating = [False]
    mic = Mic(
        WAKE,
        SPEECH,
        lambda: dictating.__setitem__(0, True),
        *REQUEST[2:],
        lambda: dictating.__setitem__(0, False),
        *REQUEST,
    )

    _run(_loop(agent, mic, Speaker(), dictating=lambda: dictating[0], audit=audit))

    assert agent.calls == [("pause the music", Origin.OWNER_SPEECH)]
    assert [event["event"] for event in audit.events] == ["wake_ignored"]


def test_a_wake_word_is_taken_again_once_the_dictation_is_over() -> None:
    agent = Agent()
    audit = AuditLog(None)
    dictating = [True]
    mic = Mic(*REQUEST, lambda: dictating.__setitem__(0, False), *REQUEST)

    _run(_loop(agent, mic, Speaker(), dictating=lambda: dictating[0], audit=audit))

    assert len(agent.calls) == 1
    assert len(audit.events) == 1


def test_a_wake_word_is_taken_while_the_status_is_unknown() -> None:
    # Voxtype's status cannot be read (an older Voxtype, no daemon to ask): Jarvis stays usable.
    agent = Agent()

    _run(_loop(agent, Mic(*REQUEST), Speaker(), dictating=lambda: None))

    assert len(agent.calls) == 1


def test_a_wake_word_is_taken_while_a_meeting_runs() -> None:
    agent = Agent()

    _run(_loop(agent, Mic(*REQUEST), Speaker(), meeting=lambda: True))

    assert len(agent.calls) == 1


def test_a_wake_word_said_over_jarvis_while_the_owner_dictates_starts_no_request() -> None:
    reply, _ = _replies("Here is a long answer.", "Skipped.")
    agent = Agent(reply)
    speaker = Playing()
    audit = AuditLog(None)
    mic = Mic(*REQUEST, until(lambda: speaker.busy), SILENCE, WAKE, *REQUEST[1:])

    _run(_loop(agent, mic, speaker, dictating=lambda: bool(agent.calls), audit=audit))

    assert len(agent.calls) == 1
    assert [event["event"] for event in audit.events] == ["wake_ignored"]


def test_no_window_opens_while_a_meeting_runs() -> None:
    agent = Agent()

    _run(_follow_loop(agent, Mic(*REQUEST, *FOLLOW_UP_REQUEST), Speaker(), meeting=lambda: True))

    assert len(agent.calls) == 1


def test_speech_in_the_window_is_dropped_when_a_meeting_started_meanwhile() -> None:
    agent = Agent()
    meeting = [False]
    mic = Mic(*REQUEST, SPEECH, lambda: meeting.__setitem__(0, True), *FOLLOW_UP_REQUEST[1:])

    _run(_follow_loop(agent, mic, Speaker(), meeting=lambda: meeting[0]))

    assert len(agent.calls) == 1


def test_no_window_opens_while_other_audio_plays() -> None:
    """Lyrics, a video or a call are not the owner's next request."""
    agent = Agent()

    _run(_follow_loop(agent, Mic(*REQUEST, *FOLLOW_UP_REQUEST), Speaker(), media=lambda: True))

    assert len(agent.calls) == 1


def test_the_owner_can_keep_the_window_while_other_audio_plays() -> None:
    agent = Agent()
    cfg = FOLLOW_UP.model_copy(update={"follow_up_over_audio": True})
    mic = Mic(*REQUEST, *FOLLOW_UP_REQUEST)

    _run(_follow_loop(agent, mic, Speaker(), cfg=cfg, media=lambda: True))

    assert len(agent.calls) == 2


def test_no_window_opens_while_it_cannot_be_told_whether_audio_plays() -> None:
    agent = Agent()

    _run(_follow_loop(agent, Mic(*REQUEST, *FOLLOW_UP_REQUEST), Speaker(), media=lambda: None))

    assert len(agent.calls) == 1


def test_speech_in_the_window_is_dropped_when_audio_started_meanwhile() -> None:
    agent = Agent()
    playing = [False]
    mic = Mic(*REQUEST, SPEECH, lambda: playing.__setitem__(0, True), *FOLLOW_UP_REQUEST[1:])

    _run(_follow_loop(agent, mic, Speaker(), media=lambda: playing[0]))

    assert len(agent.calls) == 1


def test_no_window_opens_while_the_status_is_unknown() -> None:
    agent = Agent()

    _run(_follow_loop(agent, Mic(*REQUEST, *FOLLOW_UP_REQUEST), Speaker(), dictating=lambda: None))

    assert len(agent.calls) == 1


def _asking_for_the_key(
    question: str = "Should I stop process 42?", answers: list[bool | Declined] | None = None
) -> Reply:
    """An agent that waits for the confirm key about one action and says whether it came."""

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        if hooks.confirm_key is None:
            return "no key here"
        approved = hooks.confirm_key(question, "hash-a")
        if answers is not None:
            answers.append(approved)
        return "approved" if approved else "declined"

    return reply


def _key_loop(  # ruff: ignore[too-many-positional-arguments] - the parts of a loop, each a test seam
    agent: Agent,
    mic: Mic,
    speaker: Speaker,
    key: ConfirmKey,
    clock: Clock,
    wake: Wake | None = None,
) -> VoiceLoop:
    return VoiceLoop(
        agent,
        mic,
        wake or Wake(),
        Vad(),
        speaker,
        CFG,
        lambda _samples: "stop process 42",
        names=("Jarvis",),
        clock=clock,
        keys=key,
    )


def test_the_confirm_key_allows_the_action_jarvis_asked_about() -> None:
    clock, key_clock = Clock(), Clock()
    key = ConfirmKey(clock=key_clock)
    agent = Agent(_asking_for_the_key())
    speaker = Speaker()
    states: list[State] = []
    loop = _key_loop(
        agent,
        Mic(
            *REQUEST,
            SILENCE,
            lambda: states.append(loop.state),
            key_clock.advance(1),
            key.press,
            SILENCE,
        ),
        speaker,
        key,
        clock,
    )

    _run(loop)

    assert speaker.said == [f"Should I stop process 42? {voice.PRESS_KEY}", "approved"]
    assert states == [State.CONFIRM]


def test_no_press_before_the_wait_ends_is_a_no() -> None:
    clock, key_clock = Clock(), Clock()
    key = ConfirmKey(clock=key_clock)
    agent = Agent(_asking_for_the_key())
    speaker = Speaker()
    loop = _key_loop(
        agent,
        Mic(
            *REQUEST,
            SILENCE,
            key_clock.advance(CFG.confirm_key_s),
            key.press,  # late: in the same moment as the wait ran out, before the next frame
            SILENCE,
        ),
        speaker,
        key,
        clock,
    )

    _run(loop)

    assert speaker.said[-1] == "declined"


def test_the_wait_for_the_key_ends_when_its_time_is_up() -> None:
    clock, key_clock = Clock(), Clock()
    key = ConfirmKey(clock=key_clock)
    answers: list[bool | Declined] = []
    agent = Agent(_asking_for_the_key(answers=answers))
    mic = Mic(
        *REQUEST,
        SILENCE,
        key_clock.advance(CFG.confirm_key_s),
        Idle(lambda: bool(answers)),  # the loop gives its answer without waiting for more
    )

    _run(_key_loop(agent, mic, Speaker(), key, clock))

    assert answers == [False]


def test_the_wake_word_while_waiting_for_the_key_is_a_no() -> None:
    clock, key_clock = Clock(), Clock()
    key = ConfirmKey(clock=key_clock)
    agent = Agent(_asking_for_the_key())
    speaker = Speaker()
    loop = _key_loop(
        agent,
        Mic(*REQUEST, SILENCE, WAKE, key_clock.advance(1), key.press, SILENCE),
        speaker,
        key,
        clock,
    )

    _run(loop)

    assert speaker.said[-1] == "declined"


def test_hush_while_waiting_for_the_key_is_a_no() -> None:
    clock, key_clock = Clock(), Clock()
    key = ConfirmKey(clock=key_clock)
    answers: list[bool | Declined] = []
    agent = Agent(_asking_for_the_key(answers=answers))
    speaker = Speaker()

    def hush() -> None:
        loop.hush()

    loop = _key_loop(
        agent,
        Mic(*REQUEST, SILENCE, hush, SILENCE, key_clock.advance(1), key.press, SILENCE),
        speaker,
        key,
        clock,
    )

    _run(loop)

    assert answers == [False]


class PressedWithTheWakeWord(Wake):
    """A wake word heard in the same frame as a press of the confirm key."""

    def __init__(self, key: ConfirmKey) -> None:
        super().__init__()
        self.key = key
        self.wakes = 0

    @override
    def heard(self, frame: NDArray[np.int16]) -> bool:
        woke = super().heard(frame)
        self.wakes += woke
        # The first wake word starts the request; the second comes while the key is awaited.
        if woke and self.wakes == 2 and self.key.press()["confirmed"] is None:
            pytest.fail("the press came when nothing waited")
        return woke


def test_a_press_the_key_took_before_the_wake_word_still_counts() -> None:
    # `jarvis confirm` already said "confirmed": the first of the two to reach Jarvis decides.
    clock, key_clock = Clock(), Clock()
    key = ConfirmKey(clock=key_clock)
    answers: list[bool | Declined] = []
    agent = Agent(_asking_for_the_key(answers=answers))
    mic = Mic(*REQUEST, SILENCE, key_clock.advance(1), WAKE, SILENCE)
    loop = _key_loop(agent, mic, Speaker(), key, clock, PressedWithTheWakeWord(key))

    _run(loop)

    assert answers == [True]


def test_speaking_over_the_confirm_key_question_is_a_no_and_arms_nothing() -> None:
    clock, key_clock = Clock(), Clock()
    key = ConfirmKey(clock=key_clock)
    answers: list[bool | Declined] = []
    agent = Agent(_asking_for_the_key(answers=answers))
    speaker = Playing()
    mic = Mic(*REQUEST, until(lambda: speaker.busy), WAKE, *REQUEST[1:])

    _run(_key_loop(agent, mic, speaker, key, clock))

    assert answers[0] == Declined(NoReason.SPOKEN_OVER)
    key_clock.advance(STRAY_HOLD_S)()
    assert key.press()["confirmed"] is None  # nothing was left waiting


def test_a_pause_while_waiting_for_the_key_is_a_no() -> None:
    clock, key_clock = Clock(), Clock()
    key = ConfirmKey(clock=key_clock)
    answers: list[bool | Declined] = []
    paused = threading.Event()
    mic = Mic(*REQUEST, SILENCE, paused.set, SILENCE, key_clock.advance(1), key.press, SILENCE)
    loop = VoiceLoop(
        Agent(_asking_for_the_key(answers=answers)),
        mic,
        Wake(),
        Vad(),
        Speaker(),
        CFG,
        lambda _samples: "stop process 42",
        clock=clock,
        paused=paused.is_set,
        keys=key,
    )

    _run(loop)

    assert answers == [False]


class PressingWhileSpeaking(Speaker):
    """A speaker during whose question the confirm key is pressed."""

    def __init__(self, key: ConfirmKey) -> None:
        super().__init__()
        self.key = key

    @override
    def queue(self, text: str) -> None:
        super().queue(text)
        if text.endswith(voice.PRESS_KEY):
            self.key.press()


def test_a_press_before_the_question_was_heard_holds_the_key_off() -> None:
    clock, key_clock = Clock(), Clock()
    key = ConfirmKey(clock=key_clock)
    agent = Agent(_asking_for_the_key())
    speaker = PressingWhileSpeaking(key)
    loop = _key_loop(
        agent, Mic(*REQUEST, SILENCE, key_clock.advance(1), key.press, SILENCE), speaker, key, clock
    )

    _run(loop)

    assert speaker.said[1:] == [voice.KEY_HELD_OFF, "declined"]


def test_a_key_that_will_not_arm_says_why() -> None:
    clock, key_clock = Clock(), Clock()
    key = ConfirmKey(clock=key_clock)
    answers: list[bool | Declined] = []
    speaker = PressingWhileSpeaking(key)
    mic = Mic(*REQUEST, SILENCE, key_clock.advance(1), key.press, SILENCE)
    loop = _key_loop(Agent(_asking_for_the_key(answers=answers)), mic, speaker, key, clock)

    _run(loop)

    assert answers == [Declined(NoReason.KEY_HELD_OFF)]


def test_without_a_confirm_key_the_agent_is_told_there_is_none() -> None:
    agent = Agent(_asking_for_the_key())
    speaker = Speaker()

    _run(_loop(agent, Mic(*REQUEST), speaker))

    assert speaker.said == ["no key here"]


# --- session transitions the loop meets (docs/playbook/session-transitions.md) -----------------


def test_a_wake_word_ignored_at_the_lock_screen_is_taken_after_the_unlock(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="jarvis.voice")
    agent = Agent()
    speaker = Speaker()
    locked = [True]
    mic = Mic(*REQUEST, SILENCE, lambda: locked.__setitem__(0, False), SILENCE, *REQUEST)

    _run(_loop(agent, mic, speaker, locked=lambda: locked[0]))

    assert "the wake word was ignored: the screen is locked" in caplog.text
    assert agent.calls == [("pause the music", Origin.OWNER_SPEECH)]  # only the second wake word
    assert speaker.said == ["Done."]
    assert speaker.chimes == 1


def _lost_mic_loop(
    agent: Agent,
    script: Capture,
    speaker: Speaker,
    *,
    keys: ConfirmKey | None = None,
    clock: Callable[[], float] | None = None,
    asleep_s: Callable[[], float] = lambda: 0.0,
    after_sleep: Callable[[], None] = lambda: None,
) -> tuple[VoiceLoop, list[State]]:
    """A loop on a microphone that reopens when its capture ends, with the states it reopened in."""
    reopened: list[State] = []
    mic = PausableMic(script, wait=_no_waits)

    def reopen() -> bool:
        reopened.append(loop.state)
        return mic.recover()

    loop = VoiceLoop(
        agent,
        mic,
        Wake(),
        Vad(),
        speaker,
        CFG,
        lambda _samples: "stop process 42",
        transcribe_answer=lambda _samples: "yes",
        names=("Jarvis",),
        clock=clock or Clock(),
        reopen=reopen,
        keys=keys,
        asleep_s=asleep_s,
        after_sleep=after_sleep,
    )
    return loop, reopened


def test_a_microphone_lost_while_waiting_for_a_spoken_yes_is_a_no_and_is_reopened() -> None:
    answers: list[bool | Declined] = []

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        answers.append(hooks.confirm("Should I stop process 42?"))
        return "done" if answers[-1] else "declined"

    def end() -> None:
        script.end()

    # The first request is asked about and loses the microphone; the second, after the reopen,
    # is asked about again and answered.
    script = Capture(*REQUEST, SILENCE, end, SILENCE, *REQUEST, *ANSWER)
    speaker = Speaker()
    loop, reopened = _lost_mic_loop(Agent(reply), script, speaker)

    _run(loop)

    assert answers == [
        Declined(NoReason.MIC_LOST),
        True,
    ]  # the lost microphone is a no, never a yes
    assert reopened == [State.NO_MIC]
    assert script.starts == 1
    assert speaker.said == [
        "Should I stop process 42?",
        "declined",
        "Should I stop process 42?",
        "done",
    ]
    assert loop.state is State.IDLE


def test_a_microphone_lost_while_waiting_for_the_confirm_key_is_a_no_and_the_next_wait_works() -> (
    None
):
    clock, key_clock = Clock(), Clock()
    key = ConfirmKey(clock=key_clock)
    answers: list[bool | Declined] = []

    def end() -> None:
        script.end()

    script = Capture(
        *REQUEST,
        SILENCE,
        end,
        SILENCE,
        *REQUEST,
        SILENCE,
        key_clock.advance(1),
        key.press,
        SILENCE,
    )
    loop, reopened = _lost_mic_loop(
        Agent(_asking_for_the_key(answers=answers)), script, Speaker(), keys=key, clock=clock
    )

    _run(loop)

    assert answers == [Declined(NoReason.MIC_LOST), True]  # a no, then asked afresh
    assert reopened == [State.NO_MIC]
    assert key.press()["confirmed"] is None  # nothing is left waiting


def test_a_suspend_while_a_request_is_in_flight_is_noticed_once_it_is_answered() -> None:
    asleep = [0.0]
    woke: list[bool] = []
    slept_before: list[int] = []

    def reply(_text: str, _hooks: Hooks, _origin: Origin) -> str:
        slept_before.append(len(woke))
        if len(slept_before) == 1:
            asleep[0] += 600.0  # the PC suspends while the agent works
        return "Done."

    speaker = Speaker()
    mic = Mic(*REQUEST, SILENCE, SILENCE, *REQUEST)
    loop = _loop(
        Agent(reply),
        mic,
        speaker,
        asleep_s=lambda: asleep[0],
        after_sleep=lambda: woke.append(True),
    )

    _run(loop)

    assert speaker.said == ["Done.", "Done."]  # the request in flight is answered, not dropped
    assert woke == [True]
    assert slept_before == [0, 1]  # the loop listened afresh before the second request


def test_a_yes_spoken_after_a_suspend_is_not_a_yes_to_the_question_asked_before_it() -> None:
    asleep = [0.0]
    woke: list[bool] = []
    answers: list[bool | Declined] = []

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        answers.append(hooks.confirm("Should I stop process 42?"))
        return "done" if answers[-1] else "declined"

    def suspend() -> None:
        asleep[0] += 600.0  # the PC sleeps with the question open

    speaker = Speaker()
    mic = Mic(*REQUEST, SILENCE, suspend, *ANSWER, SILENCE)
    loop = _loop(
        Agent(reply),
        mic,
        speaker,
        asleep_s=lambda: asleep[0],
        after_sleep=lambda: woke.append(True),
    )

    _run(loop)

    assert answers == [Declined(NoReason.SLEPT)]
    assert speaker.said == ["Should I stop process 42?", "declined"]
    assert woke == [True]  # and the loop still listens afresh


def test_a_yes_that_began_before_a_suspend_is_not_a_yes_either() -> None:
    # Speech already kept when the PC wakes is what ending the stream hands back: it must not count.
    asleep = [0.0]
    answers: list[bool | Declined] = []

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        answers.append(hooks.confirm("Should I stop process 42?"))
        return "done" if answers[-1] else "declined"

    def suspend() -> None:
        asleep[0] += 600.0

    mic = Mic(*REQUEST, SPEECH, SPEECH, SPEECH, suspend, SPEECH, *[SILENCE] * 5)
    loop = _loop(Agent(reply), mic, Speaker(), asleep_s=lambda: asleep[0])

    _run(loop)

    assert answers == [Declined(NoReason.SLEPT)]


def test_a_confirm_key_wait_open_across_a_suspend_has_run_out_when_the_pc_wakes() -> None:
    # The key's own clock counts the time asleep, so a press after waking finds the wait over.
    clock, key_clock = Clock(), Clock()
    key = ConfirmKey(clock=key_clock)
    asleep = [0.0]
    woke: list[bool] = []
    answers: list[bool | Declined] = []
    presses: list[dict[str, str | None]] = []

    def suspend() -> None:
        asleep[0] += 600.0
        key_clock.advance(600.0)()

    mic = Mic(
        *REQUEST,
        SILENCE,
        # Past the too-soon guard and well inside the wait: only the jump of the suspend ends it.
        key_clock.advance(max(1.0, 2 * MIN_PRESS_S)),
        suspend,
        lambda: presses.append(key.press()),
        SILENCE,
        SILENCE,
    )
    loop = VoiceLoop(
        Agent(_asking_for_the_key(answers=answers)),
        mic,
        Wake(),
        Vad(),
        Speaker(),
        CFG,
        lambda _samples: "stop process 42",
        clock=clock,
        keys=key,
        asleep_s=lambda: asleep[0],
        after_sleep=lambda: woke.append(True),
    )

    _run(loop)

    assert answers == [False]
    assert presses == [{"confirmed": None, "reason": NOTHING_WAITING}]  # not "too soon"
    assert woke == [True]


@pytest.mark.parametrize("frame_between", [True, False], ids=["a frame between", "none between"])
def test_a_microphone_lost_right_after_a_suspend_is_reopened_and_listening_is_afresh(
    *, frame_between: bool
) -> None:
    # PipeWire often restarts on resume: the capture that ended may be the first thing seen.
    asleep = [0.0]
    woke: list[bool] = []
    agent = Agent()

    def suspend() -> None:
        asleep[0] += 600.0

    def end() -> None:
        script.end()

    script = Capture(
        SILENCE,
        suspend,
        *([SILENCE] if frame_between else []),
        end,
        SILENCE,
        SILENCE,
        *REQUEST,
    )
    speaker = Speaker()
    loop, reopened = _lost_mic_loop(
        agent,
        script,
        speaker,
        asleep_s=lambda: asleep[0],
        after_sleep=lambda: woke.append(True),
    )

    _run(loop)

    assert reopened == [State.NO_MIC]
    assert woke == [True]  # once, whichever came first
    assert speaker.said == ["Done."]  # a request after both is heard
    assert loop.state is State.IDLE


def test_a_follow_up_window_open_across_a_suspend_is_over_when_the_pc_wakes() -> None:
    # The window counts audio frames, which stand still while the PC sleeps.
    asleep = [0.0]
    woke: list[bool] = []

    def suspend() -> None:
        asleep[0] += 600.0

    agent = Agent()
    mic = Mic(*EXACT_REQUEST, *[SILENCE] * 3, suspend, *FOLLOW_UP_REQUEST)
    loop = _loop(
        agent,
        mic,
        Speaker(),
        cfg=FOLLOW_UP,
        transcribe=_texts("a", "b"),
        asleep_s=lambda: asleep[0],
        after_sleep=lambda: woke.append(True),
    )

    _run(loop)

    assert [text for text, _ in agent.calls] == ["a"]  # nothing said after waking is a request
    assert woke == [True]
    assert loop.state is State.IDLE


def test_follow_up_speech_heard_before_a_suspend_is_not_taken_when_the_pc_wakes() -> None:
    # Ending the stream returns the speech kept so far; a half-heard request must not be answered.
    asleep = [0.0]

    def suspend() -> None:
        asleep[0] += 600.0

    agent = Agent()
    mic = Mic(*EXACT_REQUEST, SPEECH, SPEECH, SPEECH, SPEECH, suspend, SPEECH)
    loop = _loop(
        agent,
        mic,
        Speaker(),
        cfg=FOLLOW_UP,
        transcribe=_texts("a", "b"),
        asleep_s=lambda: asleep[0],
    )

    _run(loop)

    assert [text for text, _ in agent.calls] == ["a"]


def _request_open_across_a_suspend(
    before: list[NDArray[np.int16]], after: list[NDArray[np.int16]]
) -> tuple[Agent, Speaker, list[bool], VoiceLoop]:
    """A wake word, `before`, a suspend, `after`, then one whole request."""
    asleep = [0.0]
    woke: list[bool] = []

    def suspend() -> None:
        asleep[0] += 600.0

    agent = Agent()
    speaker = Speaker()
    mic = Mic(WAKE, *before, suspend, *after, *REQUEST)
    loop = _loop(
        agent, mic, speaker, asleep_s=lambda: asleep[0], after_sleep=lambda: woke.append(True)
    )
    return agent, speaker, woke, loop


def test_a_request_spoken_after_the_pc_woke_is_not_one_for_a_wake_word_heard_before_the_sleep() -> (
    None
):
    agent, speaker, woke, loop = _request_open_across_a_suspend(
        [], [SPEECH, SPEECH, SPEECH, *[SILENCE] * 5]
    )

    _run(loop)

    assert woke == [True]
    assert [text for text, _ in agent.calls] == ["pause the music"]  # only the request after it
    assert speaker.chimes == 2
    assert loop.state is State.IDLE


def test_a_request_begun_before_a_suspend_is_dropped_when_the_pc_wakes() -> None:
    # Ending the stream returns the speech kept so far; a half-heard request must not be answered.
    agent, speaker, woke, loop = _request_open_across_a_suspend([SPEECH, SPEECH, SPEECH], [SPEECH])

    _run(loop)

    assert woke == [True]
    assert [text for text, _ in agent.calls] == ["pause the music"]
    assert speaker.chimes == 2
    assert loop.state is State.IDLE


def _strata_without_its_file(tmp_path: Path, naps: list[float]) -> StrataModel:
    return StrataModel(
        ModelConfig(base_url=Config().model_base_url()),
        "key",
        transport=httpx.MockTransport(_refuse),
        sleep=naps.append,
        model_file=tmp_path / "gone.gguf",
    )


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def test_a_spoken_request_with_the_model_drive_missing_is_apologised_for_at_once_by_voice(
    tmp_path: Path,
) -> None:
    # systemd skipped Strata (`ConditionPathExists=`): the real agent says why; the loop speaks it.
    naps: list[float] = []
    model = _strata_without_its_file(tmp_path, naps)
    agent = RealAgent(model, Registry([no_action_operation()]), "system", AuditLog(None), {})
    speaker = Speaker()
    loop = VoiceLoop(
        agent, Mic(*REQUEST), Wake(), Vad(), speaker, CFG, lambda _samples: "stop process 42"
    )

    _run(loop)

    assert " ".join(speaker.said) == MODEL_MISSING  # heard in full, and no second apology
    assert naps == []  # no waiting for a server that is not coming
    assert loop.state is State.IDLE


class _AsksOnce:
    """A model that asks to run `stop` once, then says what it got back."""

    def __init__(self) -> None:
        super().__init__()
        self._turn = 0

    def chat(
        self,
        _messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]],
        *,
        on_delta: Callable[[Delta], None] | None = None,
        **_options: object,
    ) -> dict[str, Any]:
        self._turn += 1
        if self._turn == 1:
            call = {"id": "c1", "type": "function", "function": {"name": "stop", "arguments": "{}"}}
            return {"role": "assistant", "content": "", "tool_calls": [call]}
        if on_delta is not None:
            on_delta(Delta(text="Okay."))
        return {"role": "assistant", "content": "Okay."}

    @staticmethod
    def is_loaded() -> bool | None:
        return True


def _confirming_agent(audit: AuditLog, runs: list[str]) -> RealAgent:
    def run(_args: dict[str, Any]) -> Result:
        runs.append("stop")
        return Result("stopped", Origin.TOOL_STATUS)

    stop = Operation(
        "stop",
        "stop",
        {"type": "object", "properties": {}},
        run,
        default_class=OpClass.CONFIRM,
        spoken_summary=lambda _args: "stop process 42",
    )
    model = cast("ChatModel", _AsksOnce())
    return RealAgent(model, Registry([stop, no_action_operation()]), "system", audit, {})


def _confirmation(audit: AuditLog) -> dict[str, object]:
    return next(e for e in audit.events if e["event"] == "confirmation")


def test_a_suspend_with_the_question_open_is_audited_as_the_reason_for_the_no() -> None:
    asleep = [0.0]
    audit = AuditLog(None)
    runs: list[str] = []

    def suspend() -> None:
        asleep[0] += 600.0

    mic = Mic(*REQUEST, SILENCE, suspend, *ANSWER, SILENCE)
    loop = _loop(_confirming_agent(audit, runs), mic, Speaker(), asleep_s=lambda: asleep[0])

    _run(loop)

    assert runs == []
    confirmation = _confirmation(audit)
    assert confirmation["approved"] is False
    assert confirmation["reason"] == "slept"


def _answering(answers: list[bool | Declined]) -> Reply:
    """An agent that asks one question and keeps the hook's answer as it came back."""

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        answers.append(hooks.confirm("Should I?"))
        return "approved" if answers[-1] is True else "declined"

    return reply


def _transcriber_failing(error: Exception) -> Callable[[NDArray[np.int16]], str]:
    def transcribe(_samples: NDArray[np.int16]) -> str:
        raise error

    return transcribe


def _saying(words: str) -> Callable[[NDArray[np.int16]], str]:
    def transcribe(_samples: NDArray[np.int16]) -> str:
        return words

    return transcribe


@pytest.mark.parametrize(
    ("after", "answer", "reason"),
    [
        ([*[SILENCE] * 40], _saying("yes"), NoReason.NO_ANSWER),
        ([SPEECH, *[SILENCE] * 5], _saying("Okay."), NoReason.TOO_SHORT),
        (ANSWER, _saying("no, leave it"), NoReason.SAID_NO),
        (ANSWER, _transcriber_failing(AudioError("whisper stopped")), NoReason.UNTRANSCRIBED),
        (ANSWER, _transcriber_failing(RuntimeError("bug")), NoReason.FAILED),
    ],
    ids=["no-answer", "too-short", "said-no", "untranscribed", "failed"],
)
def test_a_spoken_no_says_why(
    after: list[Step], answer: Callable[[NDArray[np.int16]], str], reason: NoReason
) -> None:
    answers: list[bool | Declined] = []

    _run(_loop(Agent(_answering(answers)), Mic(*REQUEST, *after), Speaker(), answer=answer))

    assert answers == [Declined(reason)]


def test_a_yes_blocked_by_pc_speech_says_why() -> None:
    pc_spoke = threading.Event()
    answers: list[bool | Declined] = []
    holder: list[VoiceLoop] = []
    pc = PcAudio(
        until(lambda: bool(holder) and holder[0].state is State.CONFIRM),
        SPEECH,
        SPEECH,
        pc_spoke.set,
    )
    mic = Mic(*REQUEST, until(pc_spoke.is_set), *ANSWER)
    loop = _loop(Agent(_answering(answers)), mic, Speaker(), pc=pc)
    holder.append(loop)

    _run(loop)

    assert answers == [Declined(NoReason.PC_SPEECH)]


def _hushing_then_asking(loop: list[VoiceLoop], answers: list[bool | Declined]) -> Reply:
    """An agent whose request is hushed (`jarvis hush`) before it asks its one question."""

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        loop[0].hush()
        answers.append(hooks.confirm("Should I?"))
        return "approved" if answers[-1] is True else "declined"

    return reply


def test_a_yes_to_a_question_that_was_hushed_before_it_was_heard_is_a_no() -> None:
    # Regression: the question was cut by a hush, and a yes said next still approved it.
    holder: list[VoiceLoop] = []
    answers: list[bool | Declined] = []
    loop = _loop(Agent(_hushing_then_asking(holder, answers)), Mic(*REQUEST, *ANSWER), Speaker())
    holder.append(loop)

    _run(loop)

    assert answers == [Declined(NoReason.NOT_HEARD)]


def test_a_question_that_cannot_be_played_is_a_no() -> None:
    answers: list[bool | Declined] = []

    _run(_loop(Agent(_answering(answers)), Mic(*REQUEST, *ANSWER), Speaker(broken=True)))

    assert answers == [Declined(NoReason.NOT_HEARD)]


def test_an_answer_transcribed_as_nothing_says_so() -> None:
    answers: list[bool | Declined] = []

    _run(_loop(Agent(_answering(answers)), Mic(*REQUEST, *ANSWER), Speaker(), answer=_saying(" ")))

    assert answers == [Declined(NoReason.UNTRANSCRIBED)]


def _job(ident: int, action: str, text: str) -> Job:
    return Job(ident, cast("Any", action), text, "none", "", (), 0, 1000.0)


@dataclass
class DueJobs:
    """The jobs file's due jobs: given once, at the first look after `ready` turns true."""

    items: list[Due]
    ready: Callable[[], bool] = lambda: True
    looks: int = 0

    def __call__(self) -> list[Due]:
        self.looks += 1
        if not self.ready():
            return []
        items, self.items = self.items, []
        return items


def test_a_due_reminder_is_said_after_a_chime_and_audited() -> None:
    agent = Agent()
    speaker = Speaker()
    audit = AuditLog(None)
    jobs = DueJobs([Due(_job(3, "remind", "take the bread out"), 5.0)])

    _run(_loop(agent, Mic(*[SILENCE] * 3), speaker, audit=audit, due=jobs))

    assert speaker.chimes == 1
    assert speaker.said == ["Reminder: take the bread out"]
    assert agent.calls == []
    assert [(e["event"], e["id"], e["action"]) for e in audit.events] == [
        ("schedule_due", 3, "remind")
    ]


def test_a_late_reminder_says_how_late_it_is() -> None:
    speaker = Speaker()
    jobs = DueJobs([Due(_job(1, "remind", "call the garage"), 25 * 60.0)])

    _run(_loop(Agent(), Mic(SILENCE), speaker, due=jobs))

    assert speaker.said == ["Reminder from 25 minutes ago: call the garage"]


def test_a_due_request_goes_to_the_agent_as_the_owners_and_its_reply_is_spoken() -> None:
    agent = Agent(lambda _t, _h, _o: "Paused.")
    speaker = Speaker()
    jobs = DueJobs([Due(_job(2, "do", "pause the music"), 1.0)])

    _run(_loop(agent, Mic(SILENCE), speaker, due=jobs))

    assert len(agent.calls) == 1
    text, origin = agent.calls[0]
    assert text.endswith("pause the music")
    assert origin is Origin.OWNER_SPEECH
    assert speaker.said == ["Paused."]
    assert speaker.chimes == 0


def test_a_due_request_added_beside_outside_text_runs_with_that_text_counted() -> None:
    agent = Agent()
    audit = AuditLog(None)
    job = replace(_job(2, "do", "pause the music"), outside=("web",))

    _run(_loop(agent, Mic(SILENCE), Speaker(), audit=audit, due=DueJobs([Due(job, 1.0)])))

    assert agent.calls[0][1] is Origin.OWNER_SPEECH
    assert agent.steered == [frozenset({"web"})]
    assert agent.scheduled == [True]  # its words are the model's, not the owner's own
    due = next(e for e in audit.events if e["event"] == "schedule_due")
    assert (due["foreign"], due["outside"]) == (False, ["web"])


def test_a_due_request_first_heard_in_pc_audio_runs_as_pc_audio() -> None:
    agent = Agent()
    job = replace(_job(2, "do", "pause the music"), foreign=True)

    _run(_loop(agent, Mic(SILENCE), Speaker(), due=DueJobs([Due(job, 1.0)])))

    assert agent.calls[0][1] is Origin.PC_AUDIO


def test_a_due_request_the_owner_alone_asked_for_runs_as_theirs() -> None:
    agent = Agent()

    _run(_loop(agent, Mic(SILENCE), Speaker(), due=DueJobs([Due(_job(2, "do", "x"), 1.0)])))

    assert agent.steered == [frozenset()]


def test_a_job_that_came_too_late_is_not_done_but_audited_as_missed() -> None:
    agent = Agent()
    speaker = Speaker()
    audit = AuditLog(None)
    jobs = DueJobs(
        [
            Due(_job(4, "do", "pause the music"), 3600.0),
            Due(_job(5, "remind", "stretch"), 7 * 3600.0),
        ]
    )

    _run(_loop(agent, Mic(SILENCE), speaker, audit=audit, due=jobs))

    assert agent.calls == []
    assert speaker.said == []
    assert [(e["event"], e["id"], e["late_s"]) for e in audit.events] == [
        ("schedule_missed", 4, 3600),
        ("schedule_missed", 5, 7 * 3600),
    ]


def test_due_jobs_wait_while_the_owner_dictates_or_a_meeting_runs() -> None:
    speaker = Speaker()
    dictating = threading.Event()
    dictating.set()
    meeting = threading.Event()
    jobs = DueJobs([Due(_job(1, "remind", "stretch"), 0.0)])
    mic = Mic(
        *[SILENCE] * 3,
        dictating.clear,
        meeting.set,
        *[SILENCE] * 3,
        lambda: _no_word_yet(speaker),
        meeting.clear,
        *[SILENCE] * 3,
    )

    _run(_loop(Agent(), mic, speaker, dictating=dictating.is_set, meeting=meeting.is_set, due=jobs))

    assert speaker.said == ["Reminder: stretch"]


def test_a_request_held_back_by_dictation_for_too_long_is_missed() -> None:
    agent = Agent()
    audit = AuditLog(None)
    clock = Clock()
    dictating = threading.Event()
    dictating.set()
    jobs = DueJobs([Due(_job(6, "do", "pause the music"), 60.0)])
    mic = Mic(SILENCE, clock.advance(15 * 60), dictating.clear, *[SILENCE] * 3)

    _run(
        _loop(agent, mic, Speaker(), clock=clock, dictating=dictating.is_set, audit=audit, due=jobs)
    )

    assert agent.calls == []
    assert [(e["event"], e["late_s"]) for e in audit.events] == [("schedule_missed", 16 * 60)]


def _no_word_yet(speaker: Speaker) -> None:
    assert speaker.said == []


def test_no_follow_up_is_taken_after_a_reminder() -> None:
    # Nobody asked for the reminder: speech right after it is not taken as a request.
    agent = Agent()
    speaker = Speaker()
    cfg = CFG.model_copy(update={"follow_up_s": 2.0})
    jobs = DueJobs([Due(_job(1, "remind", "stretch"), 0.0)])

    _run(_loop(agent, Mic(SILENCE, *ANSWER, *[SILENCE] * 5), speaker, cfg=cfg, due=jobs))

    assert agent.calls == []
    assert speaker.said == ["Reminder: stretch"]


def test_a_jobs_file_that_cannot_be_read_is_logged_once_and_listening_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = Agent()

    def broken() -> list[Due]:
        msg = "permission denied"
        raise OSError(msg)

    with caplog.at_level(logging.ERROR, logger="jarvis.voice"):
        _run(_loop(agent, Mic(*[SILENCE] * 3, *REQUEST), Speaker(), due=broken))

    assert agent.calls == [("pause the music", Origin.OWNER_SPEECH)]
    assert len([r for r in caplog.records if "scheduled requests" in r.getMessage()]) == 1
