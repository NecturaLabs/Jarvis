"""The voice loop: IDLE -> WAKE -> CAPTURE -> TRANSCRIBE -> THINK -> CONFIRM? -> SPEAK -> IDLE.

One thread owns the microphone. Wake detection is off while Jarvis speaks (its own voice is
drained from the capture), so it cannot trigger itself. Every state has a timeout.
"""

import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from jarvis.agent import Hooks
from jarvis.audio import FRAME, RATE, AudioError, is_phantom, write_wav
from jarvis.policy import Origin, is_yes

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray

    from jarvis.config import AudioConfig

log = logging.getLogger(__name__)

_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_EMPHASIS = re.compile(r"(\*{1,3}|`+|(?<!\w)_{1,3})(\S(?:.*?\S)?)\1(?!\w)")
_MD_LINE_MARK = re.compile(r"^\s{0,3}(?:#{1,6}|[-*+]|\d+[.)]|>)\s+", re.MULTILINE)
_NOT_SPOKEN = re.compile(r"[\U0001F000-\U0001FAFF☀-➿️‍]")


def speakable(text: str) -> str:
    """Return a reply as plain words: no markdown marks or emoji, lines joined as sentences."""
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_LINE_MARK.sub("", text)
    for _ in range(3):
        text = _MD_EMPHASIS.sub(r"\2", text)
    text = _NOT_SPOKEN.sub("", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    joined = " ".join(line if line[-1] in ".!?:;," else f"{line}." for line in lines[:-1])
    return f"{joined} {lines[-1]}".strip() if lines else ""


FRAME_S = FRAME / RATE
PRE_ROLL_FRAMES = 3


class State(StrEnum):
    """Where the loop is."""

    IDLE = "idle"
    CAPTURE = "capture"
    TRANSCRIBE = "transcribe"
    THINK = "think"
    CONFIRM = "confirm"
    SPEAK = "speak"


class _Mic(Protocol):
    def frames(self) -> Iterator[NDArray[np.int16]]: ...
    def drain(self) -> None: ...


class _Wake(Protocol):
    def heard(self, frame: NDArray[np.int16]) -> bool: ...
    def reset(self) -> None: ...


class _Vad(Protocol):
    def speech_probability(self, frame: NDArray[np.int16]) -> float: ...
    def reset(self) -> None: ...


class _Responder(Protocol):
    def handle(self, text: str, hooks: Hooks, origin: Origin = Origin.OWNER_SPEECH) -> str: ...


class _Speaker(Protocol):
    def say(self, text: str) -> None: ...
    def chime(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PcListener:
    """A second ear on what the PC plays (Discord by default), with its own wake word and VAD."""

    audio: _Mic
    wake: _Wake
    vad: _Vad


@dataclass(frozen=True, slots=True)
class Utterance:
    """Captured speech, or why there is none."""

    samples: NDArray[np.int16] | None
    reason: str


def capture_utterance(
    frames: Iterator[NDArray[np.int16]], vad: _Vad, cfg: AudioConfig
) -> Utterance:
    """Collect speech until trailing silence, a no-speech timeout or the length cap."""
    vad.reset()
    kept: list[NDArray[np.int16]] = []
    recent: list[NDArray[np.int16]] = []
    started = False
    silent_s = 0.0
    waited_s = 0.0
    for frame in frames:
        speech = vad.speech_probability(frame) >= cfg.vad_threshold
        if not started:
            recent = [*recent[-(PRE_ROLL_FRAMES - 1) :], frame]
            waited_s += FRAME_S
            if speech:
                started = True
                kept.extend(recent)
            elif waited_s >= cfg.no_speech_timeout_s:
                return Utterance(None, "no speech")
            continue
        kept.append(frame)
        silent_s = 0.0 if speech else silent_s + FRAME_S
        if silent_s >= cfg.end_silence_s:
            return Utterance(np.concatenate(kept), "end of speech")
        if len(kept) * FRAME_S >= cfg.max_utterance_s:
            return Utterance(np.concatenate(kept), "length cap")
    return Utterance(np.concatenate(kept) if kept else None, "stream ended")


class VoiceLoop:
    """Runs the listen-think-speak cycle forever."""

    def __init__(  # ruff: ignore[too-many-arguments, too-many-positional-arguments] - collaborators, each a test seam
        self,
        agent: _Responder,
        mic: _Mic,
        wake: _Wake,
        vad: _Vad,
        speaker: _Speaker,
        cfg: AudioConfig,
        transcribe: Callable[[NDArray[np.int16]], str],
        prewarm: Callable[[], object] = lambda: None,
        pc: PcListener | None = None,
    ) -> None:
        """Wire the loop to its parts; `pc` also takes requests heard in PC audio."""
        super().__init__()
        self._agent = agent
        self._mic = mic
        self._wake = wake
        self._vad = vad
        self._speaker = speaker
        self._cfg = cfg
        self._transcribe = transcribe
        self._prewarm = prewarm
        self._pc = pc
        self._pc_heard: queue.Queue[NDArray[np.int16]] = queue.Queue(maxsize=1)
        self.state = State.IDLE

    def _say(self, text: str) -> None:
        self.state = State.SPEAK
        self._speaker.say(speakable(text))
        self._mic.drain()
        self._wake.reset()
        if self._pc is not None:
            self._pc.audio.drain()

    def _confirm(self, question: str) -> bool:
        self._say(question)
        self.state = State.CONFIRM
        cfg = self._cfg.model_copy(update={"no_speech_timeout_s": self._cfg.confirm_window_s})
        heard = capture_utterance(self._mic.frames(), self._vad, cfg)
        if heard.samples is None:
            return False
        try:
            answer = self._transcribe(heard.samples)
        except AudioError:
            return False
        log.info("confirmation answer: %r", answer)
        return is_yes(answer)

    def handle_wake(self) -> None:
        """Capture one request after the wake word and answer it."""
        threading.Thread(target=self._prewarm, daemon=True).start()
        self._speaker.chime()
        self._mic.drain()
        self.state = State.CAPTURE
        heard = capture_utterance(self._mic.frames(), self._vad, self._cfg)
        if heard.samples is None:
            log.info("wake without a request (%s)", heard.reason)
            self.state = State.IDLE
            return
        self._answer(heard.samples, Origin.OWNER_SPEECH)

    def _answer(self, samples: NDArray[np.int16], origin: Origin) -> None:
        """Transcribe one captured request and speak the agent's reply."""
        from_owner = origin is Origin.OWNER_SPEECH
        self.state = State.TRANSCRIBE
        debug_dir = os.environ.get("JARVIS_DEBUG_AUDIO_DIR", "")
        if debug_dir and from_owner:  # never keep other people's voices from PC audio
            path = Path(debug_dir) / f"utterance-{int(time.time())}.wav"
            write_wav(path, samples)
            log.info("saved the utterance to %s", path)
        try:
            text = self._transcribe(samples)
        except AudioError as exc:
            log.warning("transcription failed: %s", exc)
            if from_owner:
                self._say("Sorry, I couldn't make that out.")
            self.state = State.IDLE
            return
        if is_phantom(text) or (not text and not from_owner):
            log.info("ignored a phantom or empty transcript: %r", text)
            self.state = State.IDLE
            return
        if not text:
            self._say("Sorry, I didn't catch that.")
            self.state = State.IDLE
            return
        log.info("heard (%s): %r", origin.value, text)
        self.state = State.THINK
        reply = self._agent.handle(
            text,
            Hooks(confirm=self._confirm, waking=lambda: self._say("One moment, waking up.")),
            origin,
        )
        self._say(reply)
        self.state = State.IDLE

    def _watch_pc(self, should_stop: Callable[[], bool]) -> None:
        """Listen to PC audio for the wake word and hand each request after it to the loop.

        Jarvis's own voice also plays through the PC, so frames are skipped while it is busy.
        """
        pc = self._pc
        assert pc is not None  # ruff: ignore[assert] - only started with a listener
        frames = pc.audio.frames()
        busy = False
        for frame in frames:
            if should_stop():
                return
            if self.state is not State.IDLE or not self._pc_heard.empty():
                busy = True
                continue
            if busy:
                pc.wake.reset()
                busy = False
            if not pc.wake.heard(frame):
                continue
            log.info("wake word in PC audio at %.1f", time.time())
            threading.Thread(target=self._prewarm, daemon=True).start()
            heard = capture_utterance(frames, pc.vad, self._cfg)
            pc.wake.reset()
            if heard.samples is not None:
                self._pc_heard.put(heard.samples)

    def run(self, should_stop: Callable[[], bool] = lambda: False) -> None:
        """Listen for the wake word and handle requests until `should_stop` is true."""
        log.info("listening for the wake word")
        if self._pc is not None:
            log.info("also listening to PC audio")
            threading.Thread(target=self._watch_pc, args=(should_stop,), daemon=True).start()
        for frame in self._mic.frames():
            if should_stop():
                return
            try:
                pc_request = self._pc_heard.get_nowait()
            except queue.Empty:
                pass
            else:
                self._speaker.chime()
                self._answer(pc_request, Origin.PC_AUDIO)
                self._mic.drain()
                continue
            if self._wake.heard(frame):
                log.info("wake word at %.1f", time.time())
                self.handle_wake()
                self._mic.drain()
