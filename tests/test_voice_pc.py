"""The voice loop taking a request from PC audio, with fakes for audio, wake word and VAD."""

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from jarvis.audio import FRAME
from jarvis.config import AudioConfig
from jarvis.policy import Origin
from jarvis.voice import PcListener, VoiceLoop

if TYPE_CHECKING:
    from collections.abc import Collection, Generator

    from numpy.typing import NDArray

    from jarvis.agent import Hooks

SILENCE = np.zeros(FRAME, dtype=np.int16)
SPEECH = np.ones(FRAME, dtype=np.int16)


@dataclass
class Frames:
    script: list[NDArray[np.int16]]
    forever: bool = False
    drained: int = 0

    def frames(self) -> Generator[NDArray[np.int16]]:
        yield from self.script
        while self.forever:
            threading.Event().wait(0.005)
            yield SILENCE

    def sessions(self) -> Generator[Generator[NDArray[np.int16]]]:
        yield self.frames()

    def drain(self) -> None:
        self.drained += 1


@dataclass
class WakeOnFirstSpeech:
    @staticmethod
    def heard(frame: NDArray[np.int16]) -> bool:
        return bool(frame.any())

    def reset(self) -> None:
        pass


@dataclass
class Vad:
    @staticmethod
    def speech_probability(frame: NDArray[np.int16]) -> float:
        return 1.0 if frame.any() else 0.0

    def reset(self) -> None:
        pass


@dataclass
class NeverWake:
    @staticmethod
    def heard(frame: NDArray[np.int16]) -> bool:
        del frame
        return False

    def reset(self) -> None:
        pass


@dataclass
class Speaker:
    said: list[str] = field(default_factory=list[str])
    busy: bool = False
    interruptible: bool = True

    def say(self, text: str) -> None:
        self.said.append(text)

    def queue(self, text: str) -> None:
        self.said.append(text)

    def wait(self) -> None:
        pass

    def hush(self) -> None:
        pass

    def chime(self) -> None:
        pass


@dataclass
class Agent:
    calls: list[tuple[str, Origin]] = field(default_factory=list[tuple[str, Origin]])
    done: threading.Event = field(default_factory=threading.Event)

    def handle(
        self,
        text: str,
        hooks: Hooks,
        origin: Origin = Origin.OWNER_SPEECH,
        *,
        steered_by: Collection[str] = (),
        scheduled: bool = False,
    ) -> str:
        assert not steered_by  # a request heard now carries nothing from before
        assert not scheduled
        self.calls.append((text, origin))
        self.done.set()
        if hooks.speak is not None:
            hooks.speak("Skipped it.")
        return "Skipped it."


def test_wake_word_in_pc_audio_hands_the_request_over_as_pc_audio() -> None:
    agent = Agent()
    speaker = Speaker()
    pc_frames = Frames([SILENCE, SPEECH, SPEECH, SPEECH, *[SILENCE] * 20], forever=True)
    loop = VoiceLoop(
        agent,
        Frames([], forever=True),
        NeverWake(),
        Vad(),
        speaker,
        AudioConfig(end_silence_s=0.3),
        lambda _samples: "skip this song",
        pc=PcListener(pc_frames, WakeOnFirstSpeech(), Vad()),
    )
    stop = threading.Event()
    runner = threading.Thread(target=loop.run, args=(stop.is_set,), daemon=True)
    runner.start()

    assert agent.done.wait(5)
    stop.set()
    runner.join(5)

    assert agent.calls == [("skip this song", Origin.PC_AUDIO)]
    assert speaker.said == ["Skipped it."]
