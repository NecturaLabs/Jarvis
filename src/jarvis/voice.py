"""The voice loop: IDLE -> WAKE -> CAPTURE -> TRANSCRIBE -> THINK -> CONFIRM? -> SPEAK -> IDLE.

One thread owns the microphone. Wake detection is off while Jarvis speaks (its own voice is
drained from the capture), so it cannot trigger itself. Every state has a timeout, and a request
that fails is apologised for without ending the loop.
"""

import contextlib
import functools
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
    from collections.abc import Callable, Generator, Iterator, Sequence

    from numpy.typing import NDArray

    from jarvis.config import AudioConfig

log = logging.getLogger(__name__)

MAX_SPOKEN_CHARS = 2000
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_EMPHASIS = re.compile(r"(\*{1,3}|`+|(?<!\w)_{1,3})(\S(?:.*?\S)?)\1(?!\w)")
_MD_LINE_MARK = re.compile(r"^\s{0,3}(?:#{1,6}|[-*+]|\d+[.)]|>)\s+", re.MULTILINE)
_NOT_SPOKEN = re.compile(r"[\U0001F000-\U0001FAFF☀-➿️‍]")
_QUESTION_MARKS = re.compile(r"[\[\]()`*_]")


def speakable(text: str) -> str:
    """Return a reply as plain words: no markdown marks or emoji, lines joined as sentences."""
    text = _MD_LINK.sub(r"\1", text[:MAX_SPOKEN_CHARS])
    text = _MD_LINE_MARK.sub("", text)
    for _ in range(3):
        text = _MD_EMPHASIS.sub(r"\2", text)
    text = _NOT_SPOKEN.sub("", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    joined = " ".join(line if line[-1] in ".!?:;," else f"{line}." for line in lines[:-1])
    return f"{joined} {lines[-1]}".strip() if lines else ""


def spoken_question(text: str) -> str:
    """Return a confirmation question as words, keeping every word of it.

    Unlike a reply, nothing is dropped: a markdown link's target is spoken too, so a name that
    hides other words behind link syntax cannot hide them from the owner being asked.
    """
    text = _QUESTION_MARKS.sub(" ", text[:MAX_SPOKEN_CHARS])
    return " ".join(_NOT_SPOKEN.sub("", text).split())


FRAME_S = FRAME / RATE
MIN_ANSWER_SPEECH_S = (
    0.15  # two frames: a spoken "yes" holds about 0.3 s of speech, a click one frame
)
PRE_ROLL_FRAMES = 3
WAKE_DEDUPE_S = 2.0
SORRY_INTERNAL = "Sorry, something went wrong on my side."
PC_SPEECH_BLOCKED = "I heard voices from the PC over that, so I didn't count it."
WAKING = "One moment, waking up."
STILL_WAITING = "Still waking up, nearly there."


class State(StrEnum):
    """Where the loop is."""

    IDLE = "idle"
    CAPTURE = "capture"
    TRANSCRIBE = "transcribe"
    THINK = "think"
    CONFIRM = "confirm"
    SPEAK = "speak"


# While the loop is in these states, a wake word in PC audio is still noted (it may be the same
# request heard twice); a PC request is only captured while the loop is idle.
_NOTING_WAKES = frozenset({State.IDLE, State.CAPTURE, State.TRANSCRIBE})


class _Mic(Protocol):
    def frames(self) -> Iterator[NDArray[np.int16]]: ...
    def drain(self) -> None: ...


class _PcAudio(Protocol):
    def sessions(
        self,
    ) -> Generator[Generator[NDArray[np.int16]]]: ...
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

    audio: _PcAudio
    wake: _Wake
    vad: _Vad


@dataclass(frozen=True, slots=True)
class Utterance:
    """Captured speech, or why there is none."""

    samples: NDArray[np.int16] | None
    reason: str
    speech_s: float = 0.0  # how long the VAD rated the audio as speech


@dataclass(frozen=True, slots=True)
class _PcRequest:
    samples: NDArray[np.int16]
    wake_at: float


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
    speech_s = 0.0
    for frame in frames:
        speech = vad.speech_probability(frame) >= cfg.vad_threshold
        speech_s += FRAME_S if speech else 0.0
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
            return Utterance(np.concatenate(kept), "end of speech", speech_s)
        if len(kept) * FRAME_S >= cfg.max_utterance_s:
            return Utterance(np.concatenate(kept), "length cap", speech_s)
    return Utterance(np.concatenate(kept) if kept else None, "stream ended", speech_s)


def keep_for_debugging(samples: NDArray[np.int16]) -> None:
    """Save the owner's utterance when `JARVIS_DEBUG_AUDIO_DIR` is set, readable by the owner only.

    Debugging never gets in the way of a request: a folder that cannot be written is logged.
    """
    folder = os.environ.get("JARVIS_DEBUG_AUDIO_DIR", "")
    if not folder:
        return
    path = Path(folder) / f"utterance-{time.time_ns()}.wav"
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        with os.fdopen(os.open(path, flags, 0o600), "wb") as handle:
            write_wav(handle, samples)
    except OSError as exc:
        log.warning("could not keep the utterance for debugging: %s", exc)
        return
    log.info("saved the utterance to %s", path)


class VoiceLoop:
    """Runs the listen-think-speak cycle until stopped."""

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
        *,
        transcribe_answer: Callable[[NDArray[np.int16]], str] | None = None,
        names: Sequence[str] = (),
        clock: Callable[[], float] = time.monotonic,
        pc_retry_s: float = 1.0,
    ) -> None:
        """Wire the loop to its parts.

        `pc` also takes requests heard in PC audio. `transcribe_answer` hears a yes or no to a
        confirmation (by default `transcribe`; give one without a vocabulary hint, so the
        hint cannot turn noise into words). `names` are the assistant's names, allowed beside
        a yes. `clock` times wake words for telling one request heard twice from two.
        """
        super().__init__()
        self._agent = agent
        self._mic = mic
        self._wake = wake
        self._vad = vad
        self._speaker = speaker
        self._cfg = cfg
        self._transcribe = transcribe
        self._transcribe_answer = transcribe_answer or transcribe
        self._prewarm = prewarm
        self._pc = pc
        self._names = tuple(names)
        self._clock = clock
        self._pc_retry_s = pc_retry_s
        self._should_stop: Callable[[], bool] = lambda: False
        self._pc_heard: queue.Queue[_PcRequest] = queue.Queue(maxsize=1)
        self._pc_wake_at: float | None = None  # when PC audio last held the wake word
        self._pc_done_at: float | None = None  # when the last PC-audio request ended
        self._pc_speech = 0  # PC speech frames heard while waiting for an answer
        self._requests = 0  # requests the loop has taken
        self._chiming: threading.Thread | None = None
        self.state = State.IDLE

    def _chime(self) -> None:
        try:
            self._speaker.chime()
        except Exception:
            log.exception("could not play the chime")

    def _speak(self, words: str) -> None:
        chiming, self._chiming = self._chiming, None
        if chiming is not None:
            chiming.join()
        self.state = State.SPEAK
        self._speaker.say(words)
        self._mic.drain()
        self._wake.reset()
        if self._pc is not None:
            self._pc.audio.drain()

    def _say(self, text: str) -> None:
        self._speak(speakable(text))

    def _try_say(self, text: str) -> None:
        try:
            self._say(text)
        except Exception:
            log.exception("could not say %r", text)

    def _cue(self, text: str) -> None:
        """Say a line while the agent works, then let the loop carry on thinking."""
        self._try_say(text)
        if self.state is State.SPEAK:
            self.state = State.THINK

    def _confirm(self, question: str) -> bool:
        """Ask the owner; anything but a clear spoken yes, a failure included, is no."""
        try:
            return self._ask(question)
        except Exception:
            log.exception("the confirmation failed; counting it as no")
            return False

    def _ask(self, question: str) -> bool:
        self._speak(spoken_question(question))
        pc_speech = self._pc_speech
        self.state = State.CONFIRM
        cfg = self._cfg.model_copy(update={"no_speech_timeout_s": self._cfg.confirm_window_s})
        heard = capture_utterance(self._mic.frames(), self._vad, cfg)
        if heard.samples is None:
            return False
        if heard.speech_s < MIN_ANSWER_SPEECH_S:
            # Whisper turns a short noise into "Okay." or "Yes."; a spoken yes is longer.
            log.info("the answer held %.2f s of speech; too little for a yes", heard.speech_s)
            return False
        try:
            answer = self._transcribe_answer(heard.samples)
        except AudioError:
            return False
        log.info("confirmation answer: %r", answer)
        if not is_yes(answer, self._names):
            return False
        blocks = self._pc is not None and self._cfg.confirm_blocks_on_pc_speech
        if blocks and self._pc_speech != pc_speech:
            log.info("PC audio held speech while waiting for the answer; not counting it")
            self._try_say(PC_SPEECH_BLOCKED)
            return False
        return True

    def _serve(self, request: Callable[[], None]) -> None:
        """Run one request; a failure is logged and apologised for, and the loop carries on."""
        try:
            request()
        except Exception:
            log.exception("the request failed")
            if not self._should_stop():
                self._try_say(SORRY_INTERNAL)
        finally:
            self.state = State.IDLE

    def handle_wake(self, wake_at: float | None = None) -> None:
        """Capture one request after the wake word (heard at `wake_at`) and answer it."""
        wake_at = self._clock() if wake_at is None else wake_at
        self._requests += 1
        threading.Thread(target=self._prewarm, daemon=True).start()
        # The chime plays while the request is already being heard: words said right after the
        # wake word are kept. Measured: Silero rates the chime far below speech, and it does not
        # change what Whisper hears even at full level over the first words.
        self._chiming = threading.Thread(target=self._chime, name="jarvis-chime", daemon=True)
        self._chiming.start()
        self.state = State.CAPTURE
        started = time.perf_counter()
        heard = capture_utterance(self._mic.frames(), self._vad, self._cfg)
        if heard.samples is None:
            log.info("wake without a request (%s)", heard.reason)
            self.state = State.IDLE
            return
        self._answer(
            heard.samples,
            Origin.OWNER_SPEECH,
            wake_at=wake_at,
            capture_s=time.perf_counter() - started,
        )

    def _heard_on_pc(self, wake_at: float) -> bool:
        pc_wake_at = self._pc_wake_at
        return pc_wake_at is not None and abs(pc_wake_at - wake_at) <= WAKE_DEDUPE_S

    def _answer(
        self,
        samples: NDArray[np.int16],
        origin: Origin,
        *,
        wake_at: float | None = None,
        capture_s: float = 0.0,
    ) -> None:
        """Transcribe one captured request and speak the agent's reply."""
        try:
            self._reply(samples, origin, wake_at=wake_at, capture_s=capture_s)
        finally:
            if origin is Origin.PC_AUDIO or (wake_at is not None and self._heard_on_pc(wake_at)):
                self._pc_done_at = self._clock()

    def _reply(
        self, samples: NDArray[np.int16], origin: Origin, *, wake_at: float | None, capture_s: float
    ) -> None:
        self.state = State.TRANSCRIBE
        started = time.perf_counter()
        try:
            text = self._transcribe(samples)
        except AudioError as exc:
            log.warning("transcription failed: %s", exc)
            if origin is Origin.OWNER_SPEECH:
                self._say("Sorry, I couldn't make that out.")
            return
        stt_s = time.perf_counter() - started
        if origin is Origin.OWNER_SPEECH and wake_at is not None and self._heard_on_pc(wake_at):
            log.info("the wake word was also in PC audio; treating the request as PC audio")
            origin = Origin.PC_AUDIO
        from_owner = origin is Origin.OWNER_SPEECH
        if from_owner:  # never keep other people's voices from PC audio
            keep_for_debugging(samples)
        if is_phantom(text) or (not text and not from_owner):
            log.info("ignored a phantom or empty transcript: %r", text)
            return
        if not text:
            self._say("Sorry, I didn't catch that.")
            return
        log.info("heard (%s): %r", origin.value, text)
        self.state = State.THINK
        started = time.perf_counter()
        reply = self._agent.handle(
            text,
            Hooks(
                confirm=self._confirm,
                waking=lambda: self._cue(WAKING),
                still_waiting=lambda: self._cue(STILL_WAITING),
                cancelled=self._should_stop,
            ),
            origin,
        )
        agent_s = time.perf_counter() - started
        if self._should_stop():
            log.info("stopping; the reply is not spoken")
            return
        started = time.perf_counter()
        self._say(reply)
        log.info(
            "timing: capture %.2f s, transcription %.2f s, agent %.2f s, speech %.2f s",
            capture_s,
            stt_s,
            agent_s,
            time.perf_counter() - started,
        )

    def _watch_pc(self, should_stop: Callable[[], bool]) -> None:
        """Listen to PC audio for the wake word, restarting after a failure, until stopped."""
        while not should_stop():
            try:
                self._watch_sessions(should_stop)
            except Exception:
                log.exception("listening to PC audio failed; starting again")
                threading.Event().wait(self._pc_retry_s)
            else:
                return

    def _watch_sessions(self, should_stop: Callable[[], bool]) -> None:
        pc = self._pc
        assert pc is not None  # ruff: ignore[assert] - only started with a listener
        with contextlib.closing(pc.audio.sessions()) as sessions:
            for session in sessions:
                with contextlib.closing(session):
                    if self._watch_session(pc, session, should_stop):
                        return

    def _watch_session(
        self, pc: PcListener, frames: Iterator[NDArray[np.int16]], should_stop: Callable[[], bool]
    ) -> bool:
        """Hand each request heard in one PC audio stream to the loop; True once stopping.

        Jarvis's own voice also plays through the PC, so frames are skipped while it is busy. While
        it waits for a yes or no, speech in PC audio is counted, so a voice there cannot answer.
        """
        busy = False
        for frame in frames:
            if should_stop():
                return True
            state = self.state
            if state is State.CONFIRM:
                busy = True
                if pc.vad.speech_probability(frame) >= self._cfg.vad_threshold:
                    self._pc_speech += 1
                continue
            if state not in _NOTING_WAKES or not self._pc_heard.empty():
                busy = True
                continue
            if busy:
                pc.wake.reset()
                busy = False
            if not pc.wake.heard(frame):
                continue
            wake_at = self._clock()
            self._pc_wake_at = wake_at
            log.info("wake word in PC audio")
            if state is not State.IDLE or self._too_soon(wake_at):
                pc.wake.reset()
                continue
            self._capture_pc(pc, frames, wake_at)
        return False

    def _too_soon(self, wake_at: float) -> bool:
        done_at = self._pc_done_at
        if done_at is not None and wake_at - done_at < self._cfg.pc_audio_cooldown_s:
            log.info("a PC audio request came too soon after the last one; ignored")
            return True
        return False

    def _capture_pc(
        self, pc: PcListener, frames: Iterator[NDArray[np.int16]], wake_at: float
    ) -> None:
        requests = self._requests
        threading.Thread(target=self._prewarm, daemon=True).start()
        heard = capture_utterance(self._pc_tap(pc, frames, requests), pc.vad, self._cfg)
        if self._requests == requests and self.state is not State.CONFIRM:
            # Ended by itself. A capture the tap cut short keeps the wake model as it is, so a
            # wake word still being heard on the PC is not lost (it marks the mic's request).
            pc.wake.reset()
        if heard.samples is None:
            return
        if self._requests != requests:
            log.info("dropped a PC audio request: the loop took another one meanwhile")
            return
        with contextlib.suppress(queue.Full):
            self._pc_heard.put_nowait(_PcRequest(heard.samples, wake_at))

    def _pc_tap(
        self, pc: PcListener, frames: Iterator[NDArray[np.int16]], requests: int
    ) -> Iterator[NDArray[np.int16]]:
        """Pass on a PC capture's frames while still noting wake words in them.

        The capture ends once the loop takes another request or waits for a yes, so the watcher
        counts PC speech in the answer window again, and a wake word said meanwhile still marks a
        mic request that echoes it as PC audio. A plain loop, so closing this leaves `frames` open.
        """
        for frame in frames:
            if pc.wake.heard(frame):
                self._pc_wake_at = self._clock()
            if self._requests != requests or self.state is State.CONFIRM:
                return
            yield frame

    def _answer_pc(self, request: _PcRequest) -> None:
        self.state = State.TRANSCRIBE  # out of IDLE before the chime: no PC capture starts now
        self._speaker.chime()
        self._answer(request.samples, Origin.PC_AUDIO, wake_at=request.wake_at)

    def run(self, should_stop: Callable[[], bool] = lambda: False) -> None:
        """Listen for the wake word and handle requests until `should_stop` is true.

        A request that fails is apologised for and the loop carries on; a microphone that stops
        delivering ends the loop with its error.
        """
        self._should_stop = should_stop
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
                self._requests += 1
                self._serve(functools.partial(self._answer_pc, pc_request))
                self._mic.drain()
                continue
            if self._wake.heard(frame):
                wake_at = self._clock()
                log.info("wake word")
                self._serve(functools.partial(self.handle_wake, wake_at))
                self._mic.drain()
