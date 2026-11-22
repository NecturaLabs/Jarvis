"""The voice loop with scripted fakes: owner requests, failures, confirmation and PC audio."""

import stat
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import numpy as np
import pytest

from jarvis import voice
from jarvis.audio import FRAME, AudioError, PausableMic
from jarvis.config import AudioConfig
from jarvis.policy import Origin
from jarvis.voice import PcListener, State, VoiceLoop

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator
    from pathlib import Path

    from numpy.typing import NDArray

    from jarvis.agent import Hooks

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
        for frame in self._frames:  # ruff: ignore[yield-in-for-loop]
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
        for frame in self._frames:  # ruff: ignore[yield-in-for-loop]
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


type Reply = Callable[[str, Hooks, Origin], str]


@dataclass
class Agent:
    reply: Reply = lambda _text, _hooks, _origin: "Done."
    calls: list[tuple[str, Origin]] = field(default_factory=list[tuple[str, Origin]])
    called: threading.Event = field(default_factory=threading.Event)

    def handle(self, text: str, hooks: Hooks, origin: Origin = Origin.OWNER_SPEECH) -> str:
        """Answer like the real agent: the reply goes to `hooks.speak` unless cancelled."""
        self.calls.append((text, origin))
        self.called.set()
        reply = self.reply(text, hooks, origin)
        if hooks.speak is not None and reply and not hooks.cancelled():
            hooks.speak(reply)
        return reply


def _loop(
    agent: Agent,
    mic: Mic,
    speaker: Speaker,
    *,
    heard: str = "pause the music",
    answer: Callable[[NDArray[np.int16]], str] = lambda _s: "yes",
    pc: PcAudio | None = None,
    cfg: AudioConfig = CFG,
    clock: Clock | None = None,
    pc_wake: Wake | None = None,
    dictating: Callable[[], bool] = lambda: False,
    transcribe: Callable[[NDArray[np.int16]], str] | None = None,
) -> VoiceLoop:
    listener = None if pc is None else PcListener(pc, pc_wake or Wake(), Vad())
    return VoiceLoop(
        agent,
        mic,
        Wake(),
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


def test_a_pc_capture_is_dropped_when_the_loop_takes_another_request_meanwhile() -> None:
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
    answers: list[bool] = []

    def reply(_text: str, hooks: Hooks, _origin: Origin) -> str:
        if answers:
            return "Fine."
        answers.append(hooks.confirm("Should I buy it?"))
        return "Okay, I'll leave it."

    agent = Agent(reply)
    speaker = Playing()
    mic = Mic(*REQUEST, until(lambda: speaker.busy), WAKE, *REQUEST[1:])

    _run(_loop(agent, mic, speaker))

    assert answers == [False]
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
    dictating: Callable[[], bool] = lambda: False,
) -> VoiceLoop:
    return _loop(agent, mic, speaker, cfg=cfg, dictating=dictating, transcribe=_texts(*texts))


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

    _run(_follow_loop(agent, Mic(*REQUEST, *FOLLOW_UP_REQUEST), Speaker(), dictating=lambda: True))

    assert len(agent.calls) == 1


def test_speech_in_the_window_is_dropped_when_dictation_started_meanwhile() -> None:
    agent = Agent()
    dictating = [False]
    mic = Mic(*REQUEST, SPEECH, lambda: dictating.__setitem__(0, True), *FOLLOW_UP_REQUEST[1:])

    _run(_follow_loop(agent, mic, Speaker(), dictating=lambda: dictating[0]))

    assert len(agent.calls) == 1


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
    mic = Mic(*EXACT_REQUEST, SILENCE, lambda: states.append(loop.state), SILENCE)
    loop = _follow_loop(Agent(), mic, Speaker(), dictating=lambda: True)

    _run(loop)

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
