"""The voice loop: IDLE -> CAPTURE -> TRANSCRIBE -> THINK -> CONFIRM? -> SPEAK -> IDLE.

One thread owns the microphone and listens for the wake word while IDLE. While Jarvis speaks, a
watcher listens for it too (`audio.barge_in`): the owner saying it stops the speech and the
request, and the loop hears the new one. A sentence that would trigger the wake word by itself
cannot be spoken over, and Jarvis's own voice is drained from the capture once it is done.
Every state has a timeout, and a request that fails is apologised for without ending the loop.
"""

import contextlib
import functools
import itertools
import logging
import math
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from jarvis import announce, modes
from jarvis.agent import Declined, Hooks, NoReason
from jarvis.audio import (
    FRAME,
    RATE,
    AudioError,
    MicPausedError,
    OverSpeech,
    is_phantom,
    keyword_tail,
    write_wav,
)
from jarvis.levels import levels
from jarvis.policy import Origin, is_yes
from jarvis.schedule import reminder_words, scheduled_request
from jarvis.sentences import split_sentences

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Generator, Iterator, Sequence

    from numpy.typing import NDArray

    from jarvis.config import AudioConfig
    from jarvis.confirmkey import ConfirmKey, Waiting
    from jarvis.operations import Sound
    from jarvis.schedule import Due

log = logging.getLogger(__name__)

MAX_SPOKEN_CHARS = 2000
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_EMPHASIS = re.compile(r"(\*{1,3}|`+|(?<!\w)_{1,3})(\S(?:.*?\S)?)\1(?!\w)")
_MD_LINE_MARK = re.compile(r"^\s{0,3}(?:#{1,6}|[-*+]|\d+[.)]|>)\s+", re.MULTILINE)
_NOT_SPOKEN = re.compile(r"[\U0001F000-\U0001FAFF☀-➿️‍]")
_WEB_ADDRESS = re.compile(
    r"https?://(?:www\.)?([^\s/?#)\]>]+)[^\s)\]>]*?(?=[.,;:!?]*(?:[\s)\]>]|$))"
)
_FENCE = re.compile(r"^[ \t]*(?:```|~~~).*$", re.MULTILINE)
_TABLE_RULE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$", re.MULTILINE
)
_TABLE_ROW = re.compile(r"^[ \t]*\|(.+?)\|?[ \t]*$", re.MULTILINE)
# An en dash read "to": unspaced between words or figures (Mon to Fri, 9am to 5pm, $5 to $10), or
# spaced between a figure, a time or a unit sign and a figure (-5 to -3, 9 a.m. to 5 p.m.).
_RANGE = re.compile(
    r"(?<=[\w°%.])[\u2012\u2013](?=[\w$€£\u2212+-])"
    r"|(?:(?<=[\d%°])|(?<=[ap]\.m\.)|(?<=\d[ap]m))\s?[\u2012\u2013]\s?(?=[$€£\u2212+-]?\d)"
)
# A long dash anywhere, or a hyphen set off by spaces between words (between numbers it is a minus).
_DASH = re.compile(r"\s*,?\s*[\u2012-\u2015]\s*|(?<=[^\d\s]),?\s+-{1,2}\s+(?=\D)")
_ARROW = re.compile(r"\s*(?:\u2192|->)\s*")
_ABOUT = re.compile(r"(?<![\w/])~(?=\d)")
# The unit signs Jarvis's own tools write after a number, as the words a voice should say: the
# speech engine spells the signs out letter by letter ("k m slash h").
_UNIT_WORDS = {
    "°C": "degrees Celsius",
    "°F": "degrees Fahrenheit",
    "°": "degrees",
    "km/h": "kilometres per hour",
    "m/s": "metres per second",
    "mph": "miles per hour",
    "mm": "millimetres",
    "km": "kilometres",
    "KB": "kilobytes",
    "MB": "megabytes",
    "GB": "gigabytes",
    "TB": "terabytes",
}
_UNIT = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)*)\s?("
    + "|".join(re.escape(sign) for sign in sorted(_UNIT_WORDS, key=len, reverse=True))
    + r")(?![\w/°])"
)
_SINGULAR = re.compile(
    r"^(degree|kilometre|metre|mile|millimetre|kilobyte|megabyte|gigabyte|terabyte)s"
)
_QUESTION_MARKS = re.compile(r"[\[\]()`*_]")


SLEPT_S = 2.0  # the boot clock running this far ahead of the monotonic one is a suspend


def time_asleep() -> float:
    """Return how long the PC has been suspended since it booted, in seconds.

    The boot clock counts the time asleep and the monotonic clock does not (Linux).
    """
    return time.clock_gettime(time.CLOCK_BOOTTIME) - time.monotonic()


def _unit_words(found: re.Match[str]) -> str:
    number, words = found.group(1), _UNIT_WORDS[found.group(2)]
    return f"{number} {_SINGULAR.sub(r'\1', words) if number == '1' else words}"


def _table_row(found: re.Match[str]) -> str:
    return ", ".join(cell.strip() for cell in found.group(1).split("|") if cell.strip())


def speakable(text: str) -> str:
    """Return a reply as the words a voice can say, lines joined as sentences.

    Markdown marks, emoji, code fences and a table's rules are left out; a table row is read
    cell by cell and a web address as its site. A dash becomes a pause (between numbers, "to"),
    and a unit sign after a number its word, since the speech engine runs on over the first and
    spells out the second.
    """
    text = _MD_LINK.sub(r"\1", text[:MAX_SPOKEN_CHARS])
    text = _WEB_ADDRESS.sub(r"\1", text)
    text = _TABLE_RULE.sub("", _FENCE.sub("", text))
    text = _TABLE_ROW.sub(_table_row, text)
    text = _MD_LINE_MARK.sub("", text)
    for _ in range(3):
        text = _MD_EMPHASIS.sub(r"\2", text)
    text = _NOT_SPOKEN.sub("", text)
    text = _UNIT.sub(_unit_words, _RANGE.sub(" to ", text))  # a range before its unit signs
    text = _ARROW.sub(" to ", text)
    text = _ABOUT.sub("about ", _DASH.sub(", ", text))
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
MIN_ANSWER_SPEECH_S = 0.15  # two 80 ms frames; a click or knock is usually one
# The same minimum for a request after the wake word: saying it already shows the intent, so only
# what Whisper turns into words from a click or a knock (one frame) is dropped, and a short command
# such as "stop" is not.
MIN_REQUEST_SPEECH_S = MIN_ANSWER_SPEECH_S
# A request heard without the wake word must hold more speech than a cough, a click or a word
# from the room: four 80 ms frames.
MIN_FOLLOW_UP_SPEECH_S = 0.3
# The frames kept from before speech is detected, the detected one included: 320 ms of lead, so
# a soft first sound the detector was not yet sure of is not cut off.
PRE_ROLL_FRAMES = 5
# The wake word's tail is kept only when speech follows it within this many frames (160 ms):
# then the request was said straight after the wake word and may start inside the tail.
TAIL_GAP_FRAMES = 2
WAKE_DEDUPE_S = 2.0
SCREEN_GRACE_S = 10.0  # no screen check starts this soon after a request, speech or a line
SCREEN_LOCK_CACHE_S = 5.0  # how long the answer to "is the screen locked" is kept
SORRY_INTERNAL = "Sorry, something went wrong on my side."
NO_SPEECH = "no speech"  # why a capture that timed out before any speech ended
PC_SPEECH_BLOCKED = "I heard voices from the PC over that, so I didn't count it."
WAKING = "One moment, waking up."
STILL_WAITING = "Still waking up, nearly there."
PRESS_KEY = "Press the confirm key to allow it."
KEY_HELD_OFF = (
    "The confirm key was pressed too early, so I'll leave it. Ask me again in half a minute."
)


class State(StrEnum):
    """Where the loop is."""

    IDLE = "idle"
    CAPTURE = "capture"
    TRANSCRIBE = "transcribe"
    THINK = "think"
    CONFIRM = "confirm"
    SPEAK = "speak"
    FOLLOW_UP = "follow-up"  # listening for a request without the wake word after a reply
    NO_MIC = "no-mic"  # the microphone's capture ended on its own; waiting to reopen it


# While the loop is in these states, a wake word in PC audio is still noted (it may be the same
# request heard twice); a PC request is only captured while the loop is idle.
_NOTING_WAKES = frozenset({State.IDLE, State.CAPTURE, State.TRANSCRIBE, State.FOLLOW_UP})


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


@runtime_checkable
class _Leads(Protocol):
    """A wake word heard late (the Azure helper), which keeps the audio up to its detection.

    That audio starts `audio.AZURE_LEAD_BACK` before the keyword ended.
    """

    def after_wake(self) -> list[NDArray[np.int16]]: ...


def _with_lead(
    wake: object, frames: Iterator[NDArray[np.int16]]
) -> tuple[Iterator[NDArray[np.int16]], int]:
    """Return `frames` after those a late wake word kept, and how many of those are its keyword."""
    lead = wake.after_wake() if isinstance(wake, _Leads) else []
    return itertools.chain(lead, frames), keyword_tail(lead)


class _Hearing(Protocol):
    @property
    def wants(self) -> bool: ...
    def feed(self, frame: NDArray[np.int16]) -> None: ...
    def forget(self) -> None: ...
    def forget_recall(self) -> None: ...


class _Vad(Protocol):
    def speech_probability(self, frame: NDArray[np.int16]) -> float: ...
    def reset(self) -> None: ...


class _Responder(Protocol):
    def handle(
        self,
        text: str,
        hooks: Hooks,
        origin: Origin = Origin.OWNER_SPEECH,
        *,
        steered_by: Collection[str] = (),
        scheduled: bool = False,
    ) -> str: ...


class _Audit(Protocol):
    def record(self, event: str, **fields: object) -> None: ...


class Watching(Protocol):
    """Screen watches (`pc/watch.py`): whether one is due, and the checks that make lines to say."""

    def ready(self) -> bool:
        """Whether a check is due now (cheap: asked between audio frames)."""
        ...

    def check_due(self, stop: Callable[[], bool]) -> list[str]:
        """Make the checks that are due, dropping one when `stop()` turns true; return the lines."""
        ...


@dataclass(slots=True)
class _Screens:
    """The screen watches' checks as the loop runs them: what they said, and the one in progress."""

    watching: Watching
    lines: list[str] = field(default_factory=list[str])
    thread: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    activity: int = 0  # counts what is not idleness: a wake word, a request, a line said
    active_at: float = -math.inf  # when the last of them happened (the loop's clock)
    locked_at: float = -math.inf  # when the screen lock was last asked about, and what it said
    locked: bool = False


class _Speaker(Protocol):
    @property
    def busy(self) -> bool: ...
    @property
    def interruptible(self) -> bool: ...
    def say(self, text: str) -> None: ...
    def queue(self, text: str) -> None: ...
    def queue_sound(
        self, pcm: bytes, rate: int, lyrics: Sequence[tuple[float, float, str]] = ()
    ) -> None: ...
    def wait(self) -> None: ...
    def hush(self) -> None: ...
    def chime(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PcListener:
    """A second ear on what the PC plays (Discord by default), with its own wake word and VAD."""

    audio: _PcAudio
    wake: _Wake
    vad: _Vad
    hearing: _Hearing | None = None  # what `pc_audio_listen` hears, fed every frame read


@dataclass(frozen=True, slots=True)
class Utterance:
    """Captured speech, or why there is none."""

    samples: NDArray[np.int16] | None
    reason: str
    speech_s: float = 0.0  # how long the VAD rated the audio as speech


@dataclass(slots=True)
class _Window:
    """What the follow-up window has seen so far."""

    woke_at: float | None = None  # when the wake word was heard in it
    slept: bool = False  # the PC slept meanwhile
    read: int = 0  # frames read since it opened


@dataclass(frozen=True, slots=True)
class _PcRequest:
    samples: NDArray[np.int16]
    wake_at: float


def capture_utterance(
    frames: Iterator[NDArray[np.int16]], vad: _Vad, cfg: AudioConfig, *, lead_frames: int = 0
) -> Utterance:
    """Collect speech until trailing silence, a no-speech timeout or the length cap.

    Speech starts at a probability of `vad_threshold` and goes on while it stays above
    `vad_exit_threshold`, so a breath or a soft syllable inside a sentence is not silence. Only
    frames at the entry threshold count toward `speech_s`. The first `lead_frames` frames are the
    wake word's own tail: they neither start the request nor count as its speech, and are kept
    only when the request starts within `TAIL_GAP_FRAMES` after them, so a pause after the wake
    word is waited out like the silence before any other request. Otherwise up to
    `PRE_ROLL_FRAMES` from before speech are kept, none of them the tail's.
    """
    vad.reset()
    kept: list[NDArray[np.int16]] = []
    recent: list[tuple[int, NDArray[np.int16]]] = []
    started = False
    silent_s = 0.0
    waited_s = 0.0
    speech_s = 0.0
    exit_threshold = cfg.vad_exit_threshold
    for number, frame in enumerate(frames):
        probability = vad.speech_probability(frame)
        speech = probability >= cfg.vad_threshold and number >= lead_frames
        if speech:
            speech_s += FRAME_S
        if not started:
            recent.append((number, frame))
            if number >= lead_frames + TAIL_GAP_FRAMES:  # too late for the tail to be kept
                del recent[:-PRE_ROLL_FRAMES]
            if number >= lead_frames:  # the wait starts once the wake word has ended
                waited_s += FRAME_S
            if speech:
                started = True
                tail_kept = number < lead_frames + TAIL_GAP_FRAMES
                first = 0 if tail_kept else max(lead_frames, number + 1 - PRE_ROLL_FRAMES)
                kept.extend(f for n, f in recent if n >= first)
            elif waited_s >= cfg.no_speech_timeout_s:
                return Utterance(None, NO_SPEECH)
            continue
        kept.append(frame)
        silent_s = 0.0 if probability >= exit_threshold else silent_s + FRAME_S
        if silent_s >= cfg.end_silence_s:
            return Utterance(np.concatenate(kept), "end of speech", speech_s)
        if len(kept) * FRAME_S >= cfg.max_utterance_s:
            return Utterance(np.concatenate(kept), "length cap", speech_s)
    return Utterance(np.concatenate(kept) if kept else None, "stream ended", speech_s)


def _no_request(heard: Utterance) -> str:
    """Say why a capture after the wake word holds no request."""
    if heard.samples is None:
        return heard.reason
    return f"{heard.speech_s:.2f} s of speech is too little"


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
        dictating: Callable[[], bool | None] = lambda: False,
        meeting: Callable[[], bool] = lambda: False,
        media_playing: Callable[[], bool | None] = lambda: False,
        audit: _Audit | None = None,
        paused: Callable[[], bool] = lambda: False,
        until_resumed: Callable[[], None] = lambda: None,
        reopen: Callable[[], bool] | None = None,
        locked: Callable[[], bool] = lambda: False,
        asleep_s: Callable[[], float] = time_asleep,
        after_sleep: Callable[[], None] = lambda: None,
        keys: ConfirmKey | None = None,
        due: Callable[[], Sequence[Due]] = tuple,
        schedule_ready: Callable[[], bool] = lambda: True,
        watching: Watching | None = None,
    ) -> None:
        """Wire the loop to its parts.

        `pc` also takes requests heard in PC audio. `transcribe_answer` hears a yes or no to a
        confirmation (by default `transcribe`; give one without a vocabulary hint, so the
        hint cannot turn noise into words). `names` are the assistant's names, allowed beside
        a yes. `clock` times wake words for telling one request heard twice from two.
        `dictating` says whether the owner's own dictation is recording or transcribing, and None
        when that is not known: a wake word heard meanwhile is ignored (and said to `audit`), and
        no follow-up is taken (also while the answer is unknown). `meeting` says whether a Voxtype
        meeting is running, which blocks follow-ups too. Both must answer at once: they are asked
        on this thread. `media_playing` says whether another program plays audio (None when not
        known); no follow-up is taken then either, as what it plays is not the owner speaking to
        Jarvis. `paused` says whether
        `jarvis pause` let go of the microphone: a request in flight then ends without an
        apology, and PC audio is not listened to; `until_resumed` returns once it is over.
        `reopen` starts the microphone again after its capture ended on its own (PipeWire
        restarted, the microphone unplugged) and answers False once Jarvis is stopping; without
        it, that end ends the loop. While `locked` is true, a wake word or a PC request is
        ignored (it is asked only then). `asleep_s` is how long the PC has slept since boot: when
        it grows, the loop listens afresh and calls `after_sleep`. `keys` is the confirm key a
        `strong` action waits for (`jarvis confirm`); without it such an action is refused.
        `due` takes the reminders and scheduled requests whose time came (`JobStore.take_due`);
        it is asked between every two frames, so it must be cheap. What it gives waits while the
        owner dictates or a meeting runs. `watching` makes the screen watches' checks, one at a
        time on a thread of their own, only while the loop is idle (nothing said or heard, not
        dictating or paused); the lines they make are said between requests.
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
        self._dictating = dictating
        self._meeting = meeting
        self._media_playing = media_playing
        self._audit = audit
        self._paused = paused
        self._until_resumed = until_resumed
        self._reopen = reopen
        self._locked = locked
        self._keys = keys
        self._due = due
        self._schedule_ready = schedule_ready
        self._screens = None if watching is None else _Screens(watching)
        self._due_failed = False  # the last look at the due jobs failed (logged once)
        self._held: list[tuple[Due, float]] = []  # due jobs waiting for dictation to end, and when
        self._asleep_s = asleep_s
        self._after_sleep = after_sleep
        self._asleep = asleep_s()  # the sleep seen so far
        self._muted = False  # `jarvis hush` came during this request: say nothing more
        self._answered = False  # the last request was the owner's and was answered by voice
        self._said_nothing = False  # the last reply had no words, or the owner asked for quiet
        self._should_stop: Callable[[], bool] = lambda: False
        self._pc_heard: queue.Queue[_PcRequest] = queue.Queue(maxsize=1)
        self._pc_wake_at: float | None = None  # when PC audio last held the wake word
        self._pc_done_at: float | None = None  # when the last PC-audio request ended
        self._pc_speech = 0  # PC speech frames heard while waiting for an answer
        self._requests = 0  # requests the loop has taken
        self._chiming: threading.Thread | None = None
        # Listens while Jarvis speaks, with the event that ends it (one per watcher).
        self._watcher: tuple[threading.Thread, threading.Event] | None = None
        self._watch_lock = threading.Lock()
        self._barged_at: float | None = None  # when the owner spoke over Jarvis
        self._making = False  # an operation is making a sound: listen for the wake word
        self._first_words: float | None = None
        self.state = State.IDLE

    def _chime(self) -> None:
        try:
            self._speaker.chime()
        except Exception:
            log.exception("could not play the chime")

    def _begin_speaking(self) -> None:
        chiming, self._chiming = self._chiming, None
        if chiming is not None:
            chiming.join()
        self.state = State.SPEAK
        self._watch()

    def _watch(self) -> None:
        """Start listening for the wake word while Jarvis speaks, unless barge-in is off."""
        if self._cfg.barge_in:
            with self._watch_lock:
                if self._watcher is None:
                    done = threading.Event()
                    watcher = threading.Thread(
                        target=self._watch_speech, args=(done,), name="jarvis-barge-in", daemon=True
                    )
                    self._watcher = (watcher, done)
                    watcher.start()

    @contextlib.contextmanager
    def _making_sound(self) -> Generator[None]:
        """Listen for the wake word while an operation makes a sound (the `making_sound` hook).

        A song takes a while to make with nothing playing meanwhile; spoken over, the request
        is cancelled, which stops it being made, and the owner's new request is taken.
        """
        self._making = True
        self._watch()
        try:
            yield
        finally:
            self._making = False

    def _watch_speech(self, done: threading.Event) -> None:
        """Listen for the wake word while Jarvis speaks or makes a sound; on it, stop speaking.

        Frames are read only while speech is playing or a sound is being made, so the loop's
        own reads (an answer to a question, the next request) never lose one to this thread.
        """
        try:
            self._listen_while_speaking(done)
        except AudioError:
            log.warning("the microphone stopped while Jarvis was speaking")
        except Exception:
            log.exception("listening while speaking failed")

    def _listen_while_speaking(self, done: threading.Event) -> None:
        self._wake.reset()
        frames = self._mic.frames()
        while not done.is_set():
            speaking = self._speaker.busy
            if not speaking and not self._making:
                done.wait(0.02)
                continue
            frame = next(frames, None)
            if frame is None:
                return
            if not self._speaker.interruptible:
                continue
            if self._heard_over_speech(frame) if speaking else self._wake.heard(frame):
                self._barged_at = self._clock()
                log.info("wake word while speaking: stopping to listen")
                self._speaker.hush()
                return

    def _heard_over_speech(self, frame: NDArray[np.int16]) -> bool:
        wake = self._wake
        return wake.heard_over_speech(frame) if isinstance(wake, OverSpeech) else wake.heard(frame)

    def _note_closest_miss(self) -> None:
        """Log the best wake score Jarvis's speech drowned, so a missed barge-in can be told."""
        if not isinstance(self._wake, OverSpeech):
            return
        missed = self._wake.closest_miss()
        if missed is not None and self._barged_at is None:
            log.info("no wake word while speaking; the closest was %s", missed)

    def _end_speaking(self) -> None:
        """Wait for the speech to finish (or be spoken over), then put the mic back to rest.

        A failed speech raises the speaker's `AudioError` once the mic is back to rest.
        """
        try:
            self._speaker.wait()
        finally:
            with self._watch_lock:
                watching, self._watcher = self._watcher, None
            if watching is not None:
                watcher, done = watching
                done.set()
                watcher.join()
                self._note_closest_miss()
            if self._barged_at is None:
                # Jarvis's own voice is in what the mics heard meanwhile; after a barge-in the
                # owner's request follows the wake word, so nothing is dropped then.
                self._mic.drain()
                self._wake.reset()
                if self._pc is not None:
                    self._pc.audio.drain()

    def _silent(self) -> bool:
        """Whether this request may say nothing more: spoken over, hushed or paused."""
        return self._barged_at is not None or self._muted or self._paused()

    def _speak(self, words: str) -> None:
        if self._silent():
            return
        self._begin_speaking()
        self._speaker.queue(words)
        if self._silent():  # spoken over or hushed while it was being queued
            self._speaker.hush()
        self._end_speaking()

    def _utter(self, text: str) -> None:
        """Queue part of a reply while the agent works on (the agent's `speak` hook)."""
        words = speakable(text)
        if not words or self._silent():
            return
        if self._first_words is None:
            self._first_words = time.perf_counter()
        self._begin_speaking()
        for sentence in split_sentences(words):
            self._speaker.queue(sentence)
        if self._silent():  # spoken over or hushed while these were being queued
            self._speaker.hush()

    def _play(self, sound: Sound) -> None:
        """Queue a sound an operation made (a song) after the reply's words (the `play` hook).

        It plays like speech: spoken over with the wake word, ended by `jarvis hush`, and not
        played at all while paused or after a hush.
        """
        if self._silent():
            return
        if self._first_words is None:
            self._first_words = time.perf_counter()
        self._begin_speaking()
        self._speaker.queue_sound(sound.pcm, sound.rate, sound.lyrics)
        if self._silent():  # spoken over or hushed while it was being queued
            self._speaker.hush()

    def _hush(self) -> None:
        """Drop the reply's words not heard yet (the agent's `hush` hook)."""
        self._speaker.hush()
        if self.state is State.SPEAK:
            self.state = State.THINK

    def _asked_quiet(self) -> None:
        """Note that the owner asked for silence (the agent's `quiet` hook): no follow-up."""
        self._said_nothing = True

    def _cancelled(self) -> bool:
        return self._should_stop() or self._paused() or self._barged_at is not None

    def hush(self) -> None:
        """Stop speaking now (`jarvis hush`, from another thread).

        The request in flight carries on but says nothing more; the next one speaks again. The
        loop's state is left to the loop's own thread.
        """
        self._muted = True
        self._speaker.hush()

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

    def _confirm(self, question: str) -> bool | Declined:
        """Ask the owner; anything but a clear spoken yes, a failure included, is no."""
        try:
            return self._ask(question)
        except AudioError as exc:
            log.warning(
                "the microphone failed while the answer was awaited (%s); counting it as no", exc
            )
            return Declined(NoReason.MIC_LOST)
        except Exception:
            log.exception("the confirmation failed; counting it as no")
            return Declined(NoReason.FAILED)

    def _ask(self, question: str) -> bool | Declined:
        heard_out = self._said_in_full(spoken_question(question))
        if self._barged_at is not None:
            log.info("the owner spoke over the question; counting it as no")
            return Declined(NoReason.SPOKEN_OVER)
        if not heard_out:
            log.info(
                "the question was not heard out (hushed, paused or not played); counting it as no"
            )
            return Declined(NoReason.NOT_HEARD)
        return self._hear_answer()

    def _said_in_full(self, words: str) -> bool:
        """Speak a question; True only if it played to its end, unhushed and unpaused."""
        try:
            self._speak(words)
        except AudioError:
            log.warning("the question could not be played; counting it as no", exc_info=True)
            return False
        return not self._silent()

    def _capture_answer(self, cfg: AudioConfig) -> Utterance | None:
        """Listen for the answer to a question; None when the PC slept meanwhile (a no)."""
        slept = [False]

        def awake() -> Iterator[NDArray[np.int16]]:
            for frame in self._mic.frames():
                if self._slept():
                    slept[0] = True
                    return
                yield frame

        heard = capture_utterance(awake(), self._vad, cfg)
        if slept[0]:
            log.info("the PC slept while the answer was awaited; counting it as no")
            return None
        return heard

    def _hear_answer(self) -> bool | Declined:
        pc_speech = self._pc_speech
        self.state = State.CONFIRM
        answer = self._answer_text()
        if isinstance(answer, Declined):
            return answer
        log.info("confirmation answer: %r", answer)
        if not is_yes(answer, self._names):
            return Declined(NoReason.SAID_NO)
        blocks = self._pc is not None and self._cfg.confirm_blocks_on_pc_speech
        if blocks and self._pc_speech != pc_speech:
            log.info("PC audio held speech while waiting for the answer; not counting it")
            self._try_say(PC_SPEECH_BLOCKED)
            return Declined(NoReason.PC_SPEECH)
        return True

    def _answer_text(self) -> str | Declined:
        """Hear the answer and transcribe it, or say why there is nothing that could be a yes."""
        cfg = self._cfg.model_copy(update={"no_speech_timeout_s": self._cfg.confirm_window_s})
        heard = self._capture_answer(cfg)
        if heard is None:
            return Declined(NoReason.SLEPT)
        if heard.samples is None:
            # Only a timeout ends a capture with nothing heard; otherwise the mic stream ended.
            gone = heard.reason != NO_SPEECH
            return Declined(NoReason.MIC_LOST if gone else NoReason.NO_ANSWER)
        if heard.speech_s < MIN_ANSWER_SPEECH_S:
            # Whisper turns a short noise into "Okay." or "Yes."; a spoken yes is longer.
            log.info("the answer held %.2f s of speech; too little for a yes", heard.speech_s)
            return Declined(NoReason.TOO_SHORT)
        try:
            words = self._transcribe_answer(heard.samples)
        except AudioError:
            return Declined(NoReason.UNTRANSCRIBED)
        return words if words.strip() else Declined(NoReason.UNTRANSCRIBED)

    def _confirm_key(self, question: str, action: str) -> bool | Declined:
        """Ask about one exact action and wait for the confirm key; anything else is no."""
        try:
            return self._await_key(question, action)
        except AudioError as exc:
            log.warning("the microphone failed while waiting for the confirm key (%s); no", exc)
            return Declined(NoReason.MIC_LOST)
        except Exception:
            log.exception("waiting for the confirm key failed; counting it as no")
            return Declined(NoReason.FAILED)

    def _await_key(self, question: str, action: str) -> bool | Declined:
        """Speak the question, then wait for the key while still listening for the wake word.

        Whatever ends the wait first decides: a press the key took before it (so what `jarvis
        confirm` printed holds), else no. The wait's own time runs on the key's clock.
        """
        keys = self._keys
        if keys is None:
            return Declined(NoReason.NO_KEY)
        heard_out = self._said_in_full(f"{spoken_question(question)} {PRESS_KEY}")
        if self._barged_at is not None:
            log.info("the owner spoke over the confirm key question; counting it as no")
            return Declined(NoReason.SPOKEN_OVER)
        if not heard_out:
            log.info("the confirm key question was not heard out; counting it as no")
            return Declined(NoReason.NOT_HEARD)
        waiting = keys.arm(action, question, self._cfg.confirm_key_s)
        if waiting is None:
            self._try_say(KEY_HELD_OFF)
            return Declined(NoReason.KEY_HELD_OFF)
        self.state = State.CONFIRM
        why = "failed"
        try:
            why = self._wait_for_key(keys, waiting)
        finally:
            approved = keys.disarm(waiting, why)
        self._wake.reset()
        return approved

    def _wait_for_key(self, keys: ConfirmKey, waiting: Waiting) -> str:
        """Read the mic until the key is pressed or the wait ends; return what ended it."""
        self._wake.reset()
        for frame in self._mic.frames():
            if waiting.pressed.is_set():
                return "pressed"
            if self._silent():
                return "hushed, paused or spoken over"
            if not keys.open(waiting):
                return "no longer waiting (timed out, pressed too soon or cancelled)"
            if self._wake.heard(frame):
                log.info("the wake word while waiting for the confirm key; counting it as no")
                return "the wake word"
        return "the microphone ended"

    def _serve(self, request: Callable[[], None] | None) -> None:
        """Run one request, then each one the owner starts by speaking over Jarvis or follows on.

        A failure is logged and apologised for, and the loop carries on; one that came from a
        lost microphone or speaker is not apologised for.
        """
        while request is not None:
            self._answered = False
            self._muted = False
            self._note_activity()
            try:
                request()
            except MicPausedError:
                log.info("the request ended: Jarvis was paused")
            except AudioError as exc:  # the microphone or speaker is gone: an apology helps nobody
                log.warning("the request ended: %s", exc)
            except Exception:
                log.exception("the request failed")
                if not self._cancelled():
                    self._try_say(SORRY_INTERNAL)
            finally:
                self.state = State.IDLE
                self._note_activity()
            request = self._next_request()

    def _next_request(self) -> Callable[[], None] | None:
        """Return the request that follows the last one: spoken over it, or in the follow-up."""
        barged_at, self._barged_at = self._barged_at, None
        if self._should_stop() or self._paused():
            return None
        if barged_at is not None:
            if self._ignored_while_locked("a request spoken over Jarvis"):
                return None
            if self._ignored_while_dictating("a request spoken over Jarvis"):
                return None
            return functools.partial(self.handle_wake, barged_at)
        follow_up = self._follow_up()
        if follow_up is not None and self._ignored_while_locked("a follow-up request"):
            return None  # asked once one was heard, so a lock during the window counts too
        return follow_up

    def _follow_up(self) -> Callable[[], None] | None:
        """Listen for `follow_up_s` after a spoken reply; return the request heard, if any.

        Only after a reply to the owner's own voice has finished playing, and never while the
        owner dictates (speech in the window then is dictation, not a request) or another
        program plays audio (sung lyrics, a video, a call). A pause ends the
        window, and nothing heard in it is taken once Jarvis is paused. Speech heard while
        PC audio also held speech is dropped, as it may be a voice from the PC. Speech too short
        for a request (a cough, a click) is dropped untranscribed and the window listens on for
        the rest of its time. The wake word inside the window starts a request the usual way.
        """
        window_s = self._cfg.follow_up_s
        if window_s <= 0 or not self._answered:
            return None
        if self._using_voxtype():
            log.info("no follow-up: the owner is dictating or in a meeting")
            return None
        if self._over_audio():
            log.info("no follow-up: other audio is playing, or that is not known")
            return None
        self.state = State.FOLLOW_UP
        pc_speech = self._pc_speech
        started = time.perf_counter()
        try:
            woke_at, samples = self._listen_for_follow_up(window_s)
        finally:
            self.state = State.IDLE
        if woke_at is not None:
            log.info("wake word in the follow-up window")
            return functools.partial(self.handle_wake, woke_at)
        if samples is None or not self._keeps_follow_up(pc_speech):
            return None
        log.info("follow-up request without the wake word")
        self._requests += 1
        return functools.partial(
            self._answer,
            samples,
            Origin.OWNER_SPEECH,
            capture_s=time.perf_counter() - started,
            follow_up=True,
        )

    def _listen_for_follow_up(
        self, window_s: float
    ) -> tuple[float | None, NDArray[np.int16] | None]:
        """Listen through the window; return when the wake word was heard, or the request heard.

        Speech too short for a request is dropped, and the rest of the window listened to. A
        sleep of the PC ends the window: it counts audio frames, which stand still meanwhile.
        """
        window = _Window()
        self._wake.reset()
        while (left_s := window_s - window.read * FRAME_S) > 0:
            cfg = self._cfg.model_copy(update={"no_speech_timeout_s": left_s})
            heard = capture_utterance(self._window_frames(window), self._vad, cfg)
            if window.slept:
                log.info("the PC slept in the follow-up window; it is over")
                return None, None
            if window.woke_at is not None:
                return window.woke_at, None
            if heard.samples is None:
                return None, None
            if heard.speech_s >= MIN_FOLLOW_UP_SPEECH_S:
                return None, heard.samples
            log.info("dropped %.2f s of speech in the follow-up window: too short", heard.speech_s)
        return None, None

    def _window_frames(self, window: _Window) -> Iterator[NDArray[np.int16]]:
        """The mic's frames until the window is stopped, paused, slept through or woken."""
        for frame in self._mic.frames():
            if self._should_stop() or self._paused():
                return
            if self._slept():  # the frame count stood still while the PC slept
                window.slept = True
                return
            if self._wake.heard(frame):
                window.woke_at = self._clock()
                return
            window.read += 1
            yield frame

    def _keeps_follow_up(self, pc_speech: int) -> bool:
        """Whether speech heard in the follow-up window may be taken as a request.

        `pc_speech` is the PC speech count when the window opened.
        """
        if self._should_stop():
            return False
        if self._paused():
            log.info("dropped a follow-up: Jarvis was paused meanwhile")
            return False
        if self._using_voxtype():
            log.info("dropped a follow-up: the owner started dictating or a meeting")
            return False
        if self._over_audio():
            log.info("dropped a follow-up: other audio started playing")
            return False
        if self._pc_speech != pc_speech:
            log.info("dropped a follow-up: PC audio held speech meanwhile")
            return False
        return True

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
        frames, lead = _with_lead(self._wake, self._mic.frames())
        dictated = [False]
        slept = [False]

        def until_dictation() -> Iterator[NDArray[np.int16]]:
            for frame in frames:
                if self._dictating() is True:
                    dictated[0] = True
                    return
                if self._slept():
                    slept[0] = True
                    return
                yield frame

        heard = capture_utterance(until_dictation(), self._vad, self._cfg, lead_frames=lead)
        if dictated[0] or self._dictating() is True:
            # The owner's own dictation started meanwhile: what was said is going into it.
            self._note_dictation_ignored(
                "a request being captured", reason="dictation started mid-request"
            )
            self.state = State.IDLE
            return
        if slept[0]:
            log.info("the PC slept after the wake word; dropping the request and listening afresh")
            self.state = State.IDLE
            return
        if heard.samples is None or heard.speech_s < MIN_REQUEST_SPEECH_S:
            log.info("wake without a request (%s)", _no_request(heard))
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
        follow_up: bool = False,
    ) -> None:
        """Transcribe one captured request and speak the agent's reply.

        A `follow_up` request (heard without the wake word) that cannot be made out is dropped
        without a word: it may not have been meant for Jarvis.
        """
        try:
            self._reply(samples, origin, wake_at=wake_at, capture_s=capture_s, follow_up=follow_up)
        finally:
            if origin is Origin.PC_AUDIO or (wake_at is not None and self._heard_on_pc(wake_at)):
                self._pc_done_at = self._clock()

    def _reply(
        self,
        samples: NDArray[np.int16],
        origin: Origin,
        *,
        wake_at: float | None,
        capture_s: float,
        follow_up: bool = False,
    ) -> None:
        self.state = State.TRANSCRIBE
        log.info("levels: %s", levels(samples).summary())
        started = time.perf_counter()
        try:
            text = self._transcribe(samples)
        except AudioError as exc:
            log.warning("transcription failed: %s", exc)
            if origin is Origin.OWNER_SPEECH and not follow_up:
                self._say("Sorry, I couldn't make that out.")
                self._answered = True
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
            if not follow_up:
                self._say("Sorry, I didn't catch that.")
                self._answered = True
            return
        log.info("heard (%s): %r", origin.value, text)
        started = time.perf_counter()
        agent_s = self._think(text, origin)
        # A reply that said nothing (the owner asked for quiet) opens no follow-up window.
        self._answered = (
            from_owner and not self._cancelled() and not self._muted and not self._said_nothing
        )
        first = self._first_words
        log.info(
            "timing: capture %.2f s, transcription %.2f s, first words after %s, agent %.2f s, "
            "done speaking after %.2f s",
            capture_s,
            stt_s,
            "-" if first is None else f"{first - started:.2f} s",
            agent_s,
            time.perf_counter() - started,
        )

    def _think(
        self,
        text: str,
        origin: Origin,
        steered_by: Collection[str] = (),
        *,
        scheduled: bool = False,
    ) -> float:
        """Hand `text` to the agent and speak its reply; return the seconds the agent took.

        `steered_by` is what outside text could have steered it before, and `scheduled` says it
        is a job's (`Agent.handle`).
        """
        self.state = State.THINK
        started = time.perf_counter()
        self._first_words = None
        self._said_nothing = False
        try:
            reply = self._agent.handle(
                text,
                Hooks(
                    confirm=self._confirm,
                    confirm_key=None if self._keys is None else self._confirm_key,
                    waking=lambda: self._cue(WAKING),
                    still_waiting=lambda: self._cue(STILL_WAITING),
                    cancelled=self._cancelled,
                    speak=self._utter,
                    play=self._play,
                    hushed=self._silent,
                    making_sound=self._making_sound,
                    hush=self._hush,
                    quiet=self._asked_quiet,
                ),
                origin,
                steered_by=steered_by,
                scheduled=scheduled,
            )
            if scheduled:
                announce.publish(reply, announce.SCHEDULE)  # the request ran on its own: say so
            if not reply.strip() and self._first_words is None:
                self._said_nothing = True
            return time.perf_counter() - started
        finally:
            self._end_speaking()

    def _take_due(self) -> list[Due]:
        """Return the jobs whose time came, how late each is now.

        While the owner dictates or a meeting runs, they are held and given once it is over.
        """
        try:
            taken = self._due()
        except Exception:
            if not self._due_failed:
                log.exception("could not read the reminders and scheduled requests")
            self._due_failed = True
            taken = ()
        else:
            self._due_failed = False
        now = self._clock()
        self._held.extend((item, now) for item in taken)
        if not self._held or self._dictating() is True or self._meeting():
            return []
        held, self._held = self._held, []
        return [replace(item, late_s=item.late_s + now - at) for item, at in held]

    def _served_due(self) -> bool:
        """Serve each job whose time came, if any; return whether there was one."""
        if not self._held and not self._schedule_ready():
            return False
        with modes.voice_request() as admitted:
            if not admitted:
                return False
            due = self._take_due()
            for item in due:
                self._requests += 1
                self._serve(functools.partial(self._run_due, item))
        if not due:
            return False
        self._mic.drain()
        self._wake.reset()
        return True

    def _note_activity(self) -> None:
        """Note that something other than idleness happened: a screen check in progress drops."""
        screens = self._screens
        if screens is not None:
            screens.activity += 1
            screens.active_at = self._clock()

    def _screen_locked(self, screens: _Screens) -> bool:
        """Whether the screen is locked, asked at most every few seconds (it starts a program)."""
        now = self._clock()
        if now - screens.locked_at >= SCREEN_LOCK_CACHE_S:
            screens.locked, screens.locked_at = self._locked(), now
        return screens.locked

    def _screen_quiet(self, screens: _Screens, *, grace: bool) -> bool:
        """Whether nothing else is going on; asked only when a line or a check is about to start.

        Jarvis is not speaking or paused, with `grace` nothing happened in the last few seconds,
        the owner is not dictating, no meeting runs and the screen is not locked (asked last: it
        is the dearest).
        """
        if self._speaker.busy or self._paused():
            return False
        if grace and self._clock() - screens.active_at < SCREEN_GRACE_S:
            return False
        if self._dictating() is not False or self._meeting():
            return False
        return not self._screen_locked(screens)

    def _screen_check(self, screens: _Screens) -> None:
        """Make the checks that are due (on the check's own thread) and keep what they say.

        A check is dropped as soon as the loop is no longer idle: something was heard, a request
        began or Jarvis speaks.
        """
        seen = screens.activity

        def busy() -> bool:
            return screens.activity != seen or self.state is not State.IDLE or self._speaker.busy

        try:
            lines = screens.watching.check_due(busy)
        except Exception:
            log.exception("a screen watch's check failed")
            return
        with screens.lock:
            screens.lines.extend(lines)

    def _served_watch(self) -> bool:
        """Say what the screen watches found, and start the next check when one is due.

        Asked between every two audio frames, so with nothing to say and nothing due it answers
        at once, without asking whether the owner dictates or the screen is locked. A check starts
        only while the loop is idle (not within a few seconds of anything else); a line is said
        between requests, as a reminder is. Returns whether something was said.
        """
        screens = self._screens
        if screens is None:
            return False
        with screens.lock:
            waiting = bool(screens.lines)
        running = screens.thread is not None and screens.thread.is_alive()
        if not waiting and (running or not screens.watching.ready()):
            return False
        if not self._screen_quiet(screens, grace=not waiting):
            return False
        if waiting:
            return self._said_watch(screens)
        screens.thread = threading.Thread(
            target=self._screen_check, args=(screens,), name="jarvis-screen-watch", daemon=True
        )
        screens.thread.start()
        return False

    def _said_watch(self, screens: _Screens) -> bool:
        """Say the lines the checks made; they stay for later when no request may start now."""
        with screens.lock:
            lines, screens.lines = screens.lines, []
        with modes.voice_request() as admitted:
            if not admitted:
                with screens.lock:
                    screens.lines[:0] = lines
                return False
            for line in lines:
                self._requests += 1
                self._serve(functools.partial(self._announce, line))
        self._mic.drain()
        self._wake.reset()
        return True

    def _announce(self, line: str) -> None:
        """Say a line a screen watch made, after a chime: the one way a watch line is said."""
        self._note_activity()
        self.state = State.SPEAK
        self._chime()
        self._say(line)

    def _run_due(self, due: Due) -> None:
        """Say a due reminder, or carry out a due request as if it had just been asked.

        It is asked again by whoever asked for it: the owner, or PC audio for a request first
        heard there, with the outside text its request had seen counted as in the
        conversation, so what needed a yes then needs one now. One that came too late is not
        done, only audited as missed.
        """
        job = due.job
        fields = {
            "id": job.id,
            "action": job.action,
            "late_s": round(due.late_s),
            "foreign": job.foreign,  # where it was asked for, so the audit shows why it asks
            "outside": list(job.outside),
        }
        if due.missed:
            log.info("job %d is %.0f s late: not done", job.id, due.late_s)
            if self._audit is not None:
                self._audit.record("schedule_missed", **fields)
            return
        log.info("job %d is due (%s)", job.id, job.action)
        if self._audit is not None:
            self._audit.record("schedule_due", **fields)
        if job.action == "remind":
            self.state = State.SPEAK
            words = reminder_words(due)
            announce.publish(words, announce.REMINDER)  # a text front end shows it too
            self._chime()
            self._say(words)
            return
        origin = Origin.PC_AUDIO if job.foreign else Origin.OWNER_SPEECH
        self._think(scheduled_request(job), origin, job.outside, scheduled=True)

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
                    frames = session if pc.hearing is None else self._heard(pc.hearing, session)
                    if self._watch_session(pc, frames, should_stop):
                        return

    def _heard(
        self, hearing: _Hearing, frames: Iterator[NDArray[np.int16]]
    ) -> Iterator[NDArray[np.int16]]:
        """Pass on each PC frame, handing it to `hearing` too unless paused or Jarvis speaks.

        Jarvis's own voice plays through the PC (all of it, when every output is listened to),
        so nothing is heard while it speaks. A pause drops whatever `hearing` holds, and the end
        of the stream its recall, so a call's last words are not kept once it is over. A plain
        loop, so closing this leaves `frames` open.
        """
        paused = False
        try:
            for frame in frames:
                if self._paused():
                    if not paused:
                        hearing.forget()
                        paused = True
                else:
                    paused = False
                    if hearing.wants and not self._speaker.busy:
                        hearing.feed(frame)
                yield frame
        finally:
            hearing.forget_recall()  # a listen that runs goes on with the next stream's frames

    def _watch_session(
        self, pc: PcListener, frames: Iterator[NDArray[np.int16]], should_stop: Callable[[], bool]
    ) -> bool:
        """Hand each request heard in one PC audio stream to the loop; True once stopping.

        Jarvis's own voice also plays through the PC, so no request is captured while it is busy;
        while it speaks or waits for a yes, `_pc_busy` notes what the frames mean.
        """
        busy = False
        for frame in frames:
            if should_stop():
                return True
            if self._paused():
                busy = True
                continue
            state = self.state
            if self._pc_busy(pc, frame, state):
                busy = True
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
            if state is not State.IDLE:
                log.info("ignored a wake word in PC audio: the loop was busy (%s)", state.value)
                pc.wake.reset()
                continue
            if self._too_soon(wake_at):
                pc.wake.reset()
                continue
            self._capture_pc(pc, frames, wake_at)
        return False

    def _pc_busy(self, pc: PcListener, frame: NDArray[np.int16], state: State) -> bool:
        """Note what a PC frame means while Jarvis speaks or waits for a yes; True if it did.

        A wake word said over Jarvis may come from the call: it is noted, so a barge-in that
        echoes it is the call's request, not the owner's (see `_reply`). While Jarvis waits for
        a yes or a follow-up, speech in PC audio is counted, so a voice there cannot answer or
        make a request without the wake word.
        """
        if state is State.SPEAK:
            if pc.wake.heard(frame):
                self._pc_wake_at = self._clock()
                log.info("wake word in PC audio while Jarvis speaks")
            return True
        if state in {State.CONFIRM, State.FOLLOW_UP}:
            if pc.vad.speech_probability(frame) >= self._cfg.vad_threshold:
                self._pc_speech += 1
            return True
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
        tap, lead = _with_lead(pc.wake, self._pc_tap(pc, frames, requests))
        heard = capture_utterance(tap, pc.vad, self._cfg, lead_frames=lead)
        if self._requests == requests and self.state is not State.CONFIRM:
            # Ended by itself. A capture the tap cut short keeps the wake model as it is, so a
            # wake word still being heard on the PC is not lost (it marks the mic's request).
            pc.wake.reset()
        if self._requests != requests:
            log.info("dropped a PC audio request: the loop took another one meanwhile")
            return
        if self._paused():
            log.info("dropped a PC audio request: Jarvis was paused meanwhile")
            return
        if heard.samples is None or heard.speech_s < MIN_REQUEST_SPEECH_S:
            log.info("dropped a PC audio request: %s", _no_request(heard))
            return
        try:
            self._pc_heard.put_nowait(_PcRequest(heard.samples, wake_at))
        except queue.Full:
            log.info("dropped a PC audio request: another one is still waiting")

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
            if self._requests != requests or self.state is State.CONFIRM or self._paused():
                return
            yield frame

    def _answer_pc(self, request: _PcRequest) -> None:
        self.state = State.TRANSCRIBE  # out of IDLE before the chime: no PC capture starts now
        self._speaker.chime()
        self._answer(request.samples, Origin.PC_AUDIO, wake_at=request.wake_at)

    def run(self, should_stop: Callable[[], bool] = lambda: False) -> None:
        """Listen for the wake word and handle requests until `should_stop` is true.

        A request that fails is apologised for and the loop carries on. A microphone whose
        capture ends on its own is reopened by `reopen` (the bar shows Jarvis unavailable
        meanwhile). `jarvis pause` lets the microphone go until `resume`: the request in flight
        ends, and listening starts afresh.

        Raises:
            AudioError: the microphone's capture ended and the loop has no `reopen`.
        """
        self._should_stop = should_stop
        log.info("listening for the wake word")
        if self._pc is not None:
            log.info("also listening to PC audio")
            threading.Thread(target=self._watch_pc, args=(should_stop,), daemon=True).start()
        while True:
            try:
                self._listen(should_stop)
            except MicPausedError:
                log.info("paused: the microphone is let go")
                self.state = State.IDLE
                self._wake.reset()
                self._until_resumed()
                # A PC request heard before the pause, or queued as it came, is dropped.
                with contextlib.suppress(queue.Empty):
                    self._pc_heard.get_nowait()
                    log.info("dropped a PC audio request: Jarvis was paused meanwhile")
                if should_stop():
                    return
                log.info("resumed: listening for the wake word")
                continue
            except AudioError as exc:
                if self._reopen is None:
                    raise
                if should_stop():
                    return
                log.warning("the microphone stopped (%s); opening it again", exc)
                self.state = State.NO_MIC
                self._wake.reset()
                reopened = self._reopen()
                self.state = State.IDLE
                if not reopened or should_stop():
                    return
                continue
            return

    def _slept(self) -> bool:
        """Whether the PC slept since the last frame; if so, start listening afresh."""
        asleep = self._asleep_s()
        slept, self._asleep = asleep - self._asleep, asleep
        if slept < SLEPT_S:
            return False
        log.info("back from sleep after %.0f s: listening afresh", slept)
        self._wake.reset()
        self._mic.drain()  # frames from before the sleep
        self._after_sleep()
        return True

    def _over_audio(self) -> bool:
        """Whether another program plays audio (or that is not known) and that shuts follow-ups."""
        return not self._cfg.follow_up_over_audio and self._media_playing() is not False

    def _using_voxtype(self) -> bool:
        """Whether speech may be the owner's dictation or a meeting's (or that is not known)."""
        return self._dictating() is not False or self._meeting()

    def _ignored_while_dictating(self, what: str) -> bool:
        """Whether the owner's dictation is running, so `what` was said into it, not to Jarvis.

        An answer that is not known counts as no: the wake word is then taken, so a Voxtype whose
        status cannot be read does not leave Jarvis deaf.
        """
        if self._dictating() is not True:
            return False
        self._note_dictation_ignored(what)
        return True

    def _note_dictation_ignored(self, what: str, reason: str = "dictating") -> None:
        log.info("%s was ignored: the owner is dictating", what)
        if self._audit is not None:
            self._audit.record("wake_ignored", reason=reason)

    def _ignored_while_locked(self, what: str) -> bool:
        if not self._locked():
            return False
        log.info("%s was ignored: the screen is locked (audio.listen_when_locked is off)", what)
        return True

    def _listen(self, should_stop: Callable[[], bool]) -> None:
        """Hear wake words and requests until stopping; `MicPausedError` once paused."""
        for frame in self._mic.frames():
            if should_stop():
                return
            if self._slept():
                continue
            try:
                pc_request = self._pc_heard.get_nowait()
            except queue.Empty:
                pass
            else:
                self._note_activity()
                if self._ignored_while_locked("a PC-audio request"):
                    continue
                self._requests += 1
                self._serve_if_admitted(functools.partial(self._answer_pc, pc_request))
                self._mic.drain()
                continue
            if self._served_due() or self._served_watch():
                continue
            if self._wake.heard(frame):
                self._note_activity()  # a screen check under way gives way
                if self._ignored_while_locked("the wake word") or self._ignored_while_dictating(
                    "the wake word"
                ):
                    self._wake.reset()
                    continue
                wake_at = self._clock()
                log.info("wake word")
                self._serve_if_admitted(functools.partial(self.handle_wake, wake_at))
                self._mic.drain()

    def _serve_if_admitted(self, request: Callable[[], None]) -> None:
        with modes.voice_request() as admitted:
            if admitted:
                self._serve(request)
