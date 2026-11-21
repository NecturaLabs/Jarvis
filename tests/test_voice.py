"""The voice loop with scripted fakes: owner requests, failures, confirmation and PC audio."""

import stat
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pytest

from jarvis import voice
from jarvis.audio import FRAME, AudioError
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
CFG = AudioConfig(end_silence_s=0.3, no_speech_timeout_s=1.0, confirm_window_s=2.0)
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
    said: list[str] = field(default_factory=list[str])
    chimes: int = 0
    broken: bool = False
    during_chime: Callable[[], object] = lambda: None

    def say(self, text: str) -> None:
        if self.broken:
            msg = "no audio output"
            raise AudioError(msg)
        self.said.append(text)

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
        self.calls.append((text, origin))
        self.called.set()
        return self.reply(text, hooks, origin)


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
) -> VoiceLoop:
    listener = None if pc is None else PcListener(pc, pc_wake or Wake(), Vad())
    return VoiceLoop(
        agent,
        mic,
        Wake(),
        Vad(),
        speaker,
        cfg,
        lambda _samples: heard,
        pc=listener,
        transcribe_answer=answer,
        names=("Jarvis",),
        clock=clock or Clock(),
        pc_retry_s=0,
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
    """PC audio: "Hey Jarvis ... (3 s) ... Hey Jarvis, play X"; the mic hears the second wake too."""
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
