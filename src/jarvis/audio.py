"""Microphone capture, wake word, end of speech, transcription and speech output.

Capture and playback go through PipeWire's own tools (`pw-record`, `pw-play`); transcription goes
through Voxtype's documented `voxtype transcribe FILE` contract (path B in docs/plan.md), so
Jarvis never touches Voxtype's daemon, config or the owner's recordings.
"""

import collections
import contextlib
import json
import logging
import os
import queue
import re
import select
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import onnxruntime as ort

from jarvis.jsonutil import obj, objs

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from typing import IO

    from numpy.typing import NDArray

RATE = 16000
FRAME = 1280  # 80 ms, the wake model's step
VAD_CHUNK = 512
VAD_CONTEXT = 64
STOP_WAIT_S = 5.0  # how long a capture process gets to exit before it is killed
# Session-manager (WirePlumber) properties that keep a stream on its target and end it otherwise.
PINNED_PROPERTIES = ("node.dont-reconnect", "node.dont-fallback", "node.dont-move")


def quiet_session_options() -> ort.SessionOptions:
    """Return ONNX Runtime options for a small streaming model: one thread, no spin-waiting.

    The defaults start a thread per core that spins between calls, which kept about three cores
    busy while Jarvis only waited for the wake word.
    """
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


class AudioError(Exception):
    """The microphone, transcriber or speaker failed."""


def pw_record_command(
    target: str = "", *, capture_output: bool = False, pinned: bool = False
) -> list[str]:
    """Return `pw-record` writing raw 16 kHz mono int16 to stdout.

    `target` is a node name or serial (a microphone, or one app's playback stream); with
    `capture_output`, it records what the default output plays instead of a microphone.
    `pinned` ties the recorder to `target`: the session manager then ends it when the target is
    missing or goes away, where it would otherwise move it to the default microphone.
    """
    cmd = ["pw-record", "--raw", "--rate", str(RATE), "--channels", "1", "--format", "s16"]
    cmd += ["--latency", "80ms"]
    if target:
        cmd += ["--target", target]
    properties = ["stream.capture.sink = true"] if capture_output else []
    if pinned:
        properties += [f"{key} = true" for key in PINNED_PROPERTIES]
    if properties:
        cmd += ["-P", "{ " + " ".join(properties) + " }"]
    return [*cmd, "-"]


def app_record_command(target: str) -> list[str]:
    """Return the capture command for one app's stream (pinned to it), or all PC output if empty."""
    if target:
        return pw_record_command(target, pinned=True)
    return pw_record_command(capture_output=True)


def _nodes(objects: object) -> Iterator[tuple[object, dict[str, object], dict[str, object]]]:
    """Yield (id, info, props) for every object with properties in `pw-dump` output."""
    for item in objs(objects):
        info = obj(item.get("info"))
        props = obj(info.get("props"))
        if props:
            yield item.get("id"), info, props


def match_app_streams(objects: object, apps: list[str]) -> list[str]:
    """Return the serials of playback streams (in `pw-dump` output) that belong to one of `apps`.

    An app matches by a case-insensitive part of its process binary, application or node name,
    so "discord" finds Discord's voice stream (its application name is "WEBRTC VoiceEngine").
    Streams that are playing come first: an app may also hold idle ones.
    """
    wanted = [a.casefold() for a in apps if a.strip()]
    playing: list[str] = []
    idle: list[str] = []
    for _id, info, props in _nodes(objects):
        if props.get("media.class") != "Stream/Output/Audio" or props.get("object.serial") is None:
            continue
        names = " ".join(
            str(props.get(key, ""))
            for key in ("application.process.binary", "application.name", "node.name")
        ).casefold()
        if any(app in names for app in wanted):
            group = playing if info.get("state", "running") == "running" else idle
            group.append(str(props["object.serial"]))
    return playing + idle


def capture_sources(objects: object, pid: int) -> list[str]:
    """Return the serials of the nodes that feed process `pid`'s streams (in `pw-dump` output)."""
    clients = {
        item_id
        for item_id, _info, props in _nodes(objects)
        if str(props.get("pipewire.sec.pid")) == str(pid)
    }
    serials: dict[object, str] = {}
    mine: set[object] = set()
    for item_id, _info, props in _nodes(objects):
        serials[item_id] = str(props.get("object.serial", ""))
        if props.get("client.id") in clients:
            mine.add(item_id)
    sources: list[str] = []
    for item in objs(objects):
        link = obj(item.get("info"))
        if link.get("input-node-id") in mine and link.get("output-node-id") in serials:
            source = serials[link.get("output-node-id")]
            if source not in sources:
                sources.append(source)
    return sources


def pw_dump() -> object:
    """Return PipeWire's object graph, or an empty one when it cannot be read."""
    try:
        done = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5, check=False)
        return json.loads(done.stdout or "[]")
    except OSError, subprocess.TimeoutExpired, ValueError:
        return []


def app_stream_serials(apps: list[str]) -> list[str]:
    """Return the serials of the given apps' playback streams playing now (empty if none)."""
    return match_app_streams(pw_dump(), apps)


def capture_source_serials(pid: int) -> list[str]:
    """Return the serials of the nodes feeding process `pid`'s capture stream now."""
    return capture_sources(pw_dump(), pid)


def push_frame(buffer: queue.Queue[NDArray[np.int16]], frame: NDArray[np.int16]) -> None:
    """Add `frame` to `buffer`, dropping the oldest frame when it is full."""
    while True:
        try:
            buffer.put_nowait(frame)
        except queue.Full:
            # A reader may empty the buffer between the two calls; then there is room again.
            with contextlib.suppress(queue.Empty):
                buffer.get_nowait()
        else:
            return


class AppAudio:
    """Streams frames of what some apps play (Discord by default), or of all PC output.

    The apps' streams come and go (a call starts, ends), so this waits for one to appear,
    records it until it goes away and then waits again. With no apps named, it records the
    default output (everything the speakers play).
    """

    def __init__(
        self,
        apps: list[str],
        find: Callable[[list[str]], list[str]] = app_stream_serials,
        poll_s: float = 2.0,
        command_for: Callable[[str], list[str]] = app_record_command,
        feeds: Callable[[int], list[str]] = capture_source_serials,
    ) -> None:
        """Follow `apps` (by process or app name), checking for their streams every `poll_s`.

        `command_for(target)` builds the capture command, all PC output for "" (a test seam).
        """
        super().__init__()
        self._apps = apps
        self._find = find
        self._poll_s = poll_s
        self._command_for = command_for
        self._feeds = feeds
        self._stopping = threading.Event()
        self._mic: Microphone | None = None

    def _target(self) -> str | None:
        """Return what to record now: a stream's serial, "" for all PC output, None for nothing."""
        if not self._apps:
            return ""
        serials = self._find(self._apps)
        return serials[0] if serials else None

    def _misplaced(self, target: str, pid: int | None) -> str:
        """Return why the recorder no longer hears `target` (empty while it does).

        The stream may have gone, or the recorder may have been linked to something else (the
        default microphone, on a session manager that ignores the pinning properties). Either
        way its frames are not the app's any more and must not be passed on.
        """
        if target not in self._find(self._apps):
            return "the stream is gone"
        if pid is not None and any(source != target for source in self._feeds(pid)):
            return "the recorder was linked to another source"
        return ""

    def _record(self, target: str) -> Iterator[NDArray[np.int16]]:
        """Yield frames from `target` until it ends or stops being what the recorder hears."""
        mic = Microphone(command=self._command_for(target))
        mic.start()
        self._mic = mic
        try:
            checked = time.monotonic()
            for frame in mic.frames():
                if target and time.monotonic() - checked >= self._poll_s:
                    checked = time.monotonic()
                    reason = self._misplaced(target, mic.pid)
                    if reason:
                        log.info("PC audio recorder dropped: %s", reason)
                        return
                yield frame
        finally:
            self._mic = None
            mic.stop()

    def _session(self, target: str) -> Generator[NDArray[np.int16]]:
        try:
            yield from self._record(target)
        except AudioError as exc:
            log.info("PC audio stream ended (%s); waiting for it to come back", exc)

    def sessions(self) -> Generator[Generator[NDArray[np.int16]]]:
        """Yield one frame stream per matching stream that plays, each ending with it.

        A consumer that reads speech from one session cannot run on into the next stream. Runs
        until stopped.
        """
        while not self._stopping.is_set():
            target = self._target()
            if target is not None:
                yield self._session(target)
            self._stopping.wait(self._poll_s)

    def frames(self) -> Iterator[NDArray[np.int16]]:
        """Yield frames from whichever matching stream is playing, until stopped."""
        for session in self.sessions():
            yield from session

    def drain(self) -> None:
        """Drop frames captured so far."""
        mic = self._mic
        if mic is not None:
            mic.drain()

    def stop(self) -> None:
        """Stop following the apps."""
        self._stopping.set()
        mic = self._mic
        if mic is not None:
            mic.stop()


class Microphone:
    """Streams 16 kHz mono int16 frames of 80 ms from a PipeWire source."""

    def __init__(self, source: str = "", command: list[str] | None = None) -> None:
        """Capture from `source` (a node name or serial), or PipeWire's default when empty.

        `command` replaces `pw-record` (tests feed raw PCM from a file this way).
        """
        super().__init__()
        self._source = source
        self._command = command
        self._proc: subprocess.Popen[bytes] | None = None
        self._frames: queue.Queue[NDArray[np.int16]] = queue.Queue(maxsize=400)
        self._ended = threading.Event()
        self._stop_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def pid(self) -> int | None:
        """The capture process's ID while it runs."""
        proc = self._proc
        return proc.pid if proc is not None else None

    def start(self) -> None:
        """Start `pw-record` in raw mode.

        Raises:
            AudioError: the capture tool could not be started.
        """
        cmd = self._command or pw_record_command(self._source)
        # Each capture has its own end: a capture stopped earlier (`jarvis pause`) that finishes
        # reading after a new one started must not end the new one.
        ended = threading.Event()
        self._ended = ended
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError as exc:
            ended.set()
            msg = f"could not start {cmd[0]}: {exc}"
            raise AudioError(msg) from exc
        self._proc = proc
        self._thread = threading.Thread(target=self._read, args=(proc.stdout, ended), daemon=True)
        self._thread.start()

    def _read(self, stream: IO[bytes] | None, ended: threading.Event) -> None:
        need = FRAME * 2
        while stream is not None and not ended.is_set():
            data = stream.read(need)
            if len(data) < need:
                break
            push_frame(self._frames, np.frombuffer(data, dtype=np.int16).copy())
        ended.set()

    def frames(self) -> Iterator[NDArray[np.int16]]:
        """Yield frames until the capture ends.

        Raises:
            AudioError: the capture process stopped.
        """
        while True:
            try:
                yield self._frames.get(timeout=0.2)
            except queue.Empty:
                if self._ended.is_set():
                    msg = "the microphone stream ended"
                    raise AudioError(msg) from None

    def wait_ended(self, timeout: float) -> None:
        """Wait until the capture process's output has been read to its end."""
        if self._thread is not None:
            self._thread.join(timeout)

    def drain(self) -> None:
        """Drop frames captured so far (e.g. Jarvis's own voice)."""
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        """Stop the capture process."""
        self._ended.set()
        with self._stop_lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(STOP_WAIT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


class MicPausedError(AudioError):
    """The microphone was let go by `jarvis pause`; frames come again after `resume`."""


class _Capture(Protocol):
    @property
    def pid(self) -> int | None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def drain(self) -> None: ...
    def frames(self) -> Iterator[NDArray[np.int16]]: ...


class PausableMic:
    """A `Microphone` that `jarvis pause` can let go of (its capture ends) and `resume` reopen.

    A frames reader opened before a pause raises `MicPausedError` at its next frame, so a request
    that was listening ends; a reader opened while paused waits, using no CPU, for `resume`.
    """

    def __init__(self, mic: _Capture) -> None:
        """Wrap `mic` (a `Microphone`), which this object starts and stops from now on."""
        super().__init__()
        self._mic = mic
        self._lock = threading.Lock()
        self._running = threading.Event()  # clear while paused
        self._running.set()
        self._pauses = 0  # pauses so far; a reader opened before one ends
        self._stopped = False

    @property
    def pid(self) -> int | None:
        """The capture process's ID while it runs."""
        return self._mic.pid

    @property
    def paused(self) -> bool:
        """Whether the microphone is let go."""
        return not self._running.is_set()

    def start(self) -> None:
        """Start capturing."""
        self._mic.start()

    def pause(self) -> bool:
        """Let go of the microphone; return False when already paused or stopped."""
        with self._lock:
            if self._stopped or self.paused:
                return False
            self._pauses += 1
            self._running.clear()
            self._mic.stop()
        return True

    def resume(self) -> bool:
        """Open the microphone again; return False when not paused or stopped.

        A capture tool that cannot be started raises `Microphone.start`'s `AudioError`, and the
        microphone stays paused.
        """
        with self._lock:
            if self._stopped or not self.paused:
                return False
            self._mic.drain()
            self._mic.start()
            self._running.set()
        return True

    def stop(self) -> None:
        """Stop for good: every reader, a waiting one too, ends with `AudioError`."""
        with self._lock:
            self._stopped = True
            self._running.set()
            self._mic.stop()

    def drain(self) -> None:
        """Drop frames captured so far."""
        self._mic.drain()

    def frames(self) -> Iterator[NDArray[np.int16]]:
        """Yield frames until the capture ends, waiting first while paused.

        Raises:
            MicPausedError: a pause came while this reader was open.
            AudioError: the capture ended, or the microphone was stopped.
        """
        self._running.wait()
        pauses = self._pauses
        if self._stopped:
            msg = "the microphone was stopped"
            raise AudioError(msg)
        for frame in self._frames_until_ended():
            if self._pauses != pauses:
                raise MicPausedError
            yield frame

    def _frames_until_ended(self) -> Iterator[NDArray[np.int16]]:
        pauses = self._pauses
        try:
            yield from self._mic.frames()
        except AudioError:
            if self._pauses != pauses and not self._stopped:
                raise MicPausedError from None
            raise


class WakeWord:
    """LiveKit's `hey_jarvis` streaming model."""

    def __init__(self, model_path: Path, threshold: float) -> None:
        """Load the model file."""
        super().__init__()
        from livekit.wakeword import WakeWordModel  # ruff: ignore[import-outside-top-level] - heavy import, only when listening

        self._model = WakeWordModel(models=[model_path], sess_options=quiet_session_options())
        self._threshold = threshold

    def heard(self, frame: NDArray[np.int16]) -> bool:
        """Feed one frame; return True when the wake phrase was detected."""
        scores = self._model.predict(frame)
        score = max(scores.values(), default=0.0)
        if score >= self._threshold:
            log.info("wake word score %.2f (threshold %.2f)", score, self._threshold)
            self._model.reset()
            return True
        return False

    def reset(self) -> None:
        """Forget audio heard so far."""
        self._model.reset()


AZURE_HELPER = Path(__file__).with_name("azure_wake_helper.py")
AZURE_READY_S = 30.0  # how long the helper may take to load and try its model
AZURE_CLOSE_S = 1.0  # how long the helper may take to end once its input closes
AZURE_HISTORY = 40  # frames kept (3.2 s), so the words right after the keyword are not lost
AZURE_LEAD_BACK = RATE // 4  # start a request 0.25 s before the keyword's reported end
AZURE_ERR_LINES = 20


def azure_wake_command(python: Path, model: Path, *, check: bool = False) -> list[str]:
    """Return the command that runs the Azure keyword helper with no network.

    `unshare --net` gives the helper a network namespace with only a loopback that is down, so the
    closed-source Speech SDK it loads cannot reach anything over the network (AGENTS.md: voice
    activation stays local). It still runs as the owner, so local files and sockets are not
    walled off; ADR 0010 says what that leaves. `-I` stops the helper's Python reading environment
    variables, the user site folder or the current folder. With `check` it only tries the model.
    """
    cmd = ["unshare", "--map-current-user", "--net", "--", str(python), "-I", str(AZURE_HELPER)]
    return [*cmd, str(model), *(["--check"] if check else [])]


def azure_env() -> dict[str, str]:
    """Return the helper's environment: only what it needs, none of Jarvis's own."""
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


class AzureWake:
    """An Azure custom keyword model (a `.table` file), heard by a helper process.

    The helper (`jarvis.azure_wake_helper`) runs in its own Python, with no network; frames are
    written to it as they come, and each `wake <sample>` line it prints is one detection that
    ended at that sample. The model's own threshold decides, so `audio.wake_threshold` does not
    apply. A detection arrives a few frames after the keyword, so the frames since its end are
    kept and handed to the request (`after_wake`).
    """

    def __init__(self, command: list[str], ready_s: float = AZURE_READY_S) -> None:
        """Start the helper with `command` and wait until it has tried its model.

        Raises:
            AudioError: the helper could not start, rejected the model, or was not ready in time.
        """
        super().__init__()
        self._lock = threading.Lock()
        self._wakes: list[int | None] = []
        self._dropped = False
        self._written = 0  # samples handed to the helper
        self._history: collections.deque[tuple[int, NDArray[np.int16]]] = collections.deque(
            maxlen=AZURE_HISTORY
        )
        self._lead: list[NDArray[np.int16]] = []
        self._errors: collections.deque[str] = collections.deque(maxlen=AZURE_ERR_LINES)
        self._ready = False
        try:
            self._proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=azure_env(),
            )
        except OSError as exc:
            msg = f"could not start the Azure wake helper: {exc}"
            raise AudioError(msg) from exc
        stdout, stdin, stderr = self._proc.stdout, self._proc.stdin, self._proc.stderr
        if stdout is None or stdin is None or stderr is None:  # pragma: no cover - Popen gave pipes
            msg = "the Azure wake helper has no pipes"
            raise AudioError(msg)
        self._stdin = stdin.fileno()
        self._stdin_file = stdin
        answered = threading.Event()  # "ready", or the helper's output ended first
        threading.Thread(
            target=self._read, args=(stdout, answered), name="jarvis-azure-wake", daemon=True
        ).start()
        self._err_reader = threading.Thread(
            target=self._read_errors, args=(stderr,), name="jarvis-azure-wake-err", daemon=True
        )
        self._err_reader.start()
        if not answered.wait(ready_s) or not self._ready:
            self.close()
            msg = f"the Azure wake helper did not start: {self._why()}"
            raise AudioError(msg)
        os.set_blocking(self._stdin, False)  # a stalled helper drops frames, never Jarvis

    def _read(self, stdout: IO[bytes], answered: threading.Event) -> None:
        try:
            for line in stdout:
                word, _, value = line.strip().partition(b" ")
                if word == b"ready":
                    self._ready = True
                    answered.set()
                elif word == b"wake":
                    with self._lock:
                        self._wakes.append(int(value) if value.isdigit() else None)
        finally:
            answered.set()

    def _read_errors(self, stderr: IO[bytes]) -> None:
        for line in stderr:  # drained, so a chatty helper never blocks on a full pipe
            text = line.decode(errors="replace").rstrip()
            if text:
                self._errors.append(text[:300])
                log.debug("azure wake helper: %s", text)

    def _why(self) -> str:
        self._err_reader.join(timeout=AZURE_CLOSE_S)
        found = [line for line in self._errors if not line.startswith(("/", "["))]
        if found:
            return found[-1]
        code = self._proc.poll()
        return "no answer in time" if code is None else f"exit code {code}"

    def heard(self, frame: NDArray[np.int16]) -> bool:
        """Feed one frame; return True when the helper heard the keyword since the last call.

        Raises:
            AudioError: the helper has stopped.
        """
        if self._proc.poll() is not None:
            msg = f"the Azure wake helper stopped: {self._why()}"
            raise AudioError(msg)
        pcm = frame.astype(np.int16)
        try:
            os.write(self._stdin, pcm.tobytes())  # one frame is below PIPE_BUF: all or nothing
        except BlockingIOError:
            if not self._dropped:
                log.warning("the Azure wake helper is behind; dropping audio until it catches up")
            self._dropped = True
        except OSError as exc:
            msg = f"the Azure wake helper stopped: {exc.strerror or exc}"
            raise AudioError(msg) from exc
        else:
            self._dropped = False
            self._history.append((self._written, pcm))
            self._written += pcm.size
        with self._lock:
            wakes, self._wakes = self._wakes, []
        if not wakes:
            return False
        self._lead = self._since(wakes[-1])
        return True

    def _since(self, end: int | None) -> list[NDArray[np.int16]]:
        if end is None:
            return []
        start = max(0, end - AZURE_LEAD_BACK)
        lead: list[NDArray[np.int16]] = []
        for first, pcm in self._history:
            if first + pcm.size > start:
                lead.append(pcm[max(0, start - first) :])
        return lead

    def after_wake(self) -> list[NDArray[np.int16]]:
        """Return, once, the frames heard since the last detected keyword ended."""
        lead, self._lead = self._lead, []
        return lead

    def reset(self) -> None:
        """Forget a detection not yet reported and the audio kept for it."""
        with self._lock:
            self._wakes = []
        self._lead = []
        self._history.clear()

    def close(self) -> None:
        """End the helper (its input closes; it is killed if it does not end at once)."""
        with contextlib.suppress(OSError, ValueError):
            self._stdin_file.close()
        try:
            self._proc.wait(timeout=AZURE_CLOSE_S)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


class FallbackWake:
    """Listens with an Azure model until its helper fails, then with LiveKit's for good.

    A helper that dies mid-run (a model the SDK rejects later, a crash) is said in the log once,
    and Jarvis keeps listening instead of stopping.
    """

    def __init__(self, primary: AzureWake, fallback: Callable[[], _WakeModel]) -> None:
        """Use `primary`; on its failure, the wake word `fallback` makes."""
        super().__init__()
        self._primary: AzureWake | None = primary
        self._make = fallback
        self._backup: _WakeModel | None = None

    def heard(self, frame: NDArray[np.int16]) -> bool:
        """Feed one frame to whichever model listens; True when it heard the wake word."""
        if self._primary is not None:
            try:
                return self._primary.heard(frame)
            except AudioError as exc:
                log.warning("%s; listening with LiveKit's wake word from now on", exc)
                self._primary.close()
                self._primary = None
        if self._backup is None:
            self._backup = self._make()
        return self._backup.heard(frame)

    def after_wake(self) -> list[NDArray[np.int16]]:
        """Return the frames heard since the keyword ended (Azure only)."""
        return self._primary.after_wake() if self._primary is not None else []

    def reset(self) -> None:
        """Forget audio heard so far."""
        if self._primary is not None:
            self._primary.reset()
        if self._backup is not None:
            self._backup.reset()

    def close(self) -> None:
        """End the helper, if it still runs."""
        if self._primary is not None:
            self._primary.close()


class Vad:
    """Silero VAD (ONNX) over 32 ms chunks with its 64-sample context."""

    def __init__(self, model_path: Path) -> None:
        """Load the model on one CPU thread."""
        super().__init__()
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=quiet_session_options(),
            providers=["CPUExecutionProvider"],
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, VAD_CONTEXT), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        """Clear the recurrent state."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, VAD_CONTEXT), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def speech_probability(self, frame: NDArray[np.int16]) -> float:
        """Return the highest speech probability among the chunks this frame completes."""
        self._pending = np.concatenate([self._pending, frame.astype(np.float32) / 32768.0])
        best = 0.0
        while len(self._pending) >= VAD_CHUNK:
            chunk, self._pending = self._pending[:VAD_CHUNK], self._pending[VAD_CHUNK:]
            x = np.concatenate([self._context, chunk.reshape(1, -1)], axis=1)
            out, self._state = self._session.run(
                None, {"input": x, "state": self._state, "sr": np.array(RATE, dtype=np.int64)}
            )
            self._context = x[:, -VAD_CONTEXT:]
            best = max(best, float(np.asarray(out).reshape(-1)[0]))
        return best


def write_wav(target: Path | IO[bytes], samples: NDArray[np.int16]) -> None:
    """Write 16 kHz mono 16-bit PCM to a file path or an open binary file."""
    with wave.open(str(target) if isinstance(target, Path) else target, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(samples.astype("<i2").tobytes())


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_DIAGNOSTIC = re.compile(
    r"^(Loading audio file:|Audio format:|Processing \d+ samples|\d{4}-\d\d-\d\dT|whisper_)"
)


def parse_transcript(stdout: str) -> str:
    """Return the transcript from `voxtype transcribe` output without its progress and log lines."""
    lines = [_ANSI.sub("", line).strip() for line in stdout.splitlines()]
    kept = [line for line in lines if line and not _DIAGNOSTIC.match(line)]
    return " ".join(kept).strip()


_PHANTOMS = frozenset({
    "thank you for watching", "thanks for watching", "thank you", "thank you so much for watching",
    "please subscribe", "you", "bye", "subtitles by the amaraorg community",
})  # fmt: skip


def is_phantom(text: str) -> bool:
    """Return True for Whisper's well-known phrases on silence or noise (not a real request)."""
    words = "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).split()
    return " ".join(words) in _PHANTOMS


def voxtype_settings(threads: int, *, short_window: bool) -> str:
    """Return Jarvis's own Voxtype config (TOML), read in place of the owner's for its runs.

    Whisper pads every clip to 30 s, so a two-second request costs as much as a long one;
    `short_window` (Voxtype's `context_window_optimization`) sizes the window to the clip, about
    five times faster for a spoken command. Whisper.cpp also uses only 4 threads unless told.
    """
    return (
        "[whisper]\n"
        f"threads = {int(threads)}\n"
        f"context_window_optimization = {'true' if short_window else 'false'}\n"
    )


def voxtype_command(
    wav: Path, engine: str = "", model: str = "", prompt: str = "", settings: Path | None = None
) -> list[str]:
    """Return the `voxtype transcribe` command; options override the owner's config for one run.

    `settings` is a config file of Jarvis's own (see `voxtype_settings`); the owner's Voxtype
    config is never changed.
    """
    cmd = ["voxtype"]
    if settings is not None:
        cmd += ["-c", str(settings)]
    # The `--option=value` form keeps a value that starts with a dash (a song name in the hint)
    # from being read as another option.
    if model:
        cmd.append(f"--model={model}")
    if engine:
        cmd.append(f"--engine={engine}")
    if prompt:
        cmd.append(f"--initial-prompt={prompt}")
    return [*cmd, "transcribe", str(wav)]


# The states `voxtype status` reports in which the owner's dictation is not running (Voxtype's
# docs/INTEGRATIONS.md lists idle, recording, streaming, transcribing and stopped); any other
# state, one added later included, counts as dictation.
_NOT_DICTATING = frozenset({"idle", "stopped"})


def dictation_active(
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run, timeout_s: float = 2.0
) -> bool:
    """Whether the owner's Voxtype dictation is recording or transcribing.

    Read from `voxtype status --format json`, part of Voxtype's integration contract. A daemon
    that is not running (a non-zero exit) is not dictating; a status that cannot be run or read
    counts as dictating, so Jarvis never takes the owner's dictation for a request.
    """
    try:
        done = run(
            ["voxtype", "status", "--format", "json"],
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        log.warning("could not read Voxtype's status: %s", exc)
        return True
    if done.returncode != 0:
        return False
    try:
        state = obj(json.loads(done.stdout)).get("class")
    except ValueError:
        state = None
    if not isinstance(state, str):
        log.warning("could not read Voxtype's status: %r", done.stdout[:200])
        return True
    return state not in _NOT_DICTATING


def transcribe(  # ruff: ignore[too-many-arguments] - Voxtype's options plus a test seam
    samples: NDArray[np.int16],
    engine: str = "",
    model: str = "",
    prompt: str = "",
    timeout_s: float = 60.0,
    *,
    settings: Path | None = None,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> str:
    """Transcribe an utterance with `voxtype transcribe` on a private temporary WAV.

    `prompt` hints Whisper with words it should expect (the owner's artist names); `settings` is
    Jarvis's own Voxtype config file. `run` replaces `subprocess.run` in tests.

    Raises:
        AudioError: Voxtype is missing, could not be started with these options, failed or timed
            out.
    """
    with tempfile.TemporaryDirectory(prefix="jarvis-") as tmp:
        wav = Path(tmp) / "utterance.wav"
        write_wav(wav, samples)
        try:
            done = run(
                voxtype_command(wav, engine, model, prompt, settings),
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            msg = "transcription timed out"
            raise AudioError(msg) from exc
        except (OSError, ValueError) as exc:
            # ValueError: an argument the system cannot pass on (a NUL in the hint).
            msg = f"voxtype could not be started: {exc}"
            raise AudioError(msg) from exc
    if done.returncode != 0:
        detail = done.stderr.decode("utf-8", errors="replace").strip()[-200:]
        msg = f"voxtype transcribe failed ({done.returncode}): {detail}"
        raise AudioError(msg)
    return parse_transcript(done.stdout.decode("utf-8", errors="replace"))


KOKORO_PREFIX = "kokoro:"


def voice_files(data_dir: Path, voice: str) -> list[Path]:
    """Return the files a voice setting needs (`kokoro:<name>` or a Piper voice name)."""
    if voice.startswith(KOKORO_PREFIX):
        base = data_dir / "voices" / "kokoro"
        return [base / "kokoro-v1.0.onnx", base / "voices-v1.0.bin"]
    return [data_dir / "voices" / f"{voice}.onnx", data_dir / "voices" / f"{voice}.onnx.json"]


def make_synthesizer(
    data_dir: Path, voice: str, speed: float
) -> Callable[[str], tuple[bytes, int]]:
    """Return text -> (int16 PCM bytes, sample rate) for the configured voice."""
    files = voice_files(data_dir, voice)
    if voice.startswith(KOKORO_PREFIX):
        from jarvis.kokoro import SAMPLE_RATE, Kokoro  # ruff: ignore[import-outside-top-level] - heavy

        kokoro = Kokoro(files[0], files[1], voice.removeprefix(KOKORO_PREFIX), speed=speed)
        return lambda text: (kokoro.synthesize(text).tobytes(), SAMPLE_RATE)
    from piper import PiperVoice, SynthesisConfig  # ruff: ignore[import-outside-top-level] - heavy

    piper = PiperVoice.load(files[0])
    # Piper's length scale is the time per sound, so a faster speed is a shorter one.
    settings = SynthesisConfig(length_scale=piper.config.length_scale / speed)

    def run(text: str) -> tuple[bytes, int]:
        chunks = list(piper.synthesize(text, settings))
        rate = chunks[0].sample_rate if chunks else 22050
        return b"".join(chunk.audio_int16_bytes for chunk in chunks), rate

    return run


def pw_play_command(rate: int) -> list[str]:
    """Return the `pw-play` command for raw mono 16-bit PCM on standard input."""
    return ["pw-play", "--raw", "--rate", str(rate), "--channels", "1", "--format", "s16", "-"]


PLAY_GRACE_S = 10.0  # how long playback may run past the sound's own length before it is ended
LINGER_S = 0.3  # how long an open player waits for the next sentence before it is closed
WRITE_STEP = 4096  # bytes handed to the player at a time (one pipe page; about 0.085 s at 24 kHz)
QUIET = 300  # samples at or below this level count as silence at the ends of a sentence
LEAD_KEEP_S = 0.03  # silence kept before the first word Jarvis says
TAIL_KEEP_S = 0.2  # silence kept after each sentence (the voice pads about 0.35 s)


def trim_silence(pcm: bytes, rate: int, *, lead_s: float | None, tail_s: float) -> bytes:
    """Return int16 `pcm` with its leading and trailing silence cut to `lead_s` and `tail_s`.

    `lead_s` None keeps the leading silence (the pause between two sentences). A clip with no
    sound at all comes back empty.
    """
    samples = np.frombuffer(pcm, dtype=np.int16)
    loud = np.flatnonzero(np.abs(samples.astype(np.int32)) > QUIET)
    if loud.size == 0:
        return b""
    start = 0 if lead_s is None else max(0, int(loud[0]) - int(lead_s * rate))
    end = min(samples.size, int(loud[-1]) + 1 + int(tail_s * rate))
    return samples[start:end].tobytes()


def to_16k(pcm: bytes, rate: int) -> NDArray[np.int16]:
    """Return int16 `pcm` at `rate` resampled to 16 kHz (linear; enough for the wake model)."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if rate == RATE or samples.size == 0:
        return samples.astype(np.int16)
    count = int(samples.size * RATE / rate)
    points = np.linspace(0, samples.size - 1, count)
    return np.interp(points, np.arange(samples.size), samples).astype(np.int16)


class _WakeModel(Protocol):
    def heard(self, frame: NDArray[np.int16]) -> bool: ...
    def reset(self) -> None: ...


class OwnVoiceCheck:
    """Tells whether a sentence Jarvis is about to say would itself trigger the wake word.

    While such a sentence plays, the owner cannot speak over Jarvis (it would hear itself).
    It uses its own copy of the wake model, so the listening one keeps its state.
    """

    def __init__(self, wake: _WakeModel, names: re.Pattern[str] | None = None) -> None:
        """Check with `wake`, a wake model used for nothing else.

        `names` matches a sentence whose words alone would wake Jarvis: an Azure keyword model
        also wakes on "Jarvis" without "Hey", and only LiveKit's model checks the sound here.
        """
        super().__init__()
        self._wake = wake
        self._names = names

    def __call__(self, pcm: bytes, rate: int, text: str = "") -> bool:
        """Return True when `text` names the wake word or the wake model hears it in `pcm`."""
        if self._names is not None and self._names.search(text):
            return True
        samples = np.concatenate([to_16k(pcm, rate), np.zeros(RATE // 2, dtype=np.int16)])
        self._wake.reset()
        try:
            return any(
                self._wake.heard(samples[i : i + FRAME])
                for i in range(0, samples.size - FRAME + 1, FRAME)
            )
        finally:
            self._wake.reset()


class Speaker:
    """Speaks through `pw-play` with a synthesizer from `make_synthesizer`.

    Sentences are queued and spoken in order by one worker thread, each synthesized while the
    one before it plays, through one player for as long as sentences keep coming. `hush` drops
    what has not been heard yet, so the owner can speak over Jarvis.
    """

    def __init__(
        self,
        synthesize: Callable[[str], tuple[bytes, int]],
        *,
        player: Callable[[int], list[str]] = pw_play_command,
        grace_s: float = PLAY_GRACE_S,
        own_wake: Callable[[bytes, int, str], bool] | None = None,
        linger_s: float = LINGER_S,
    ) -> None:
        """Use `synthesize` for text to speech; `player` builds the playback command for a rate.

        `own_wake` tells whether a synthesized sentence (its sound, its rate and its text) would
        trigger the wake word itself.
        """
        super().__init__()
        self._synthesize = synthesize
        self._player = player
        self._grace_s = grace_s
        self._own_wake = own_wake
        self._linger_s = linger_s
        self._lock = threading.Condition()
        self._items: queue.Queue[tuple[int, str]] = queue.Queue()
        self._pending = 0  # sentences queued, being synthesized or being handed to the player
        self._run = 0  # raised by `hush`: sentences queued before it are dropped
        self._error: AudioError | None = None
        self._guarded = False  # the speech playing now holds the wake phrase
        self._worker: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None  # the open player
        self._rate = 0
        self._player_run = 0  # the run the open player was started for
        self._opened = 0.0
        self._written_s = 0.0
        self._chime: subprocess.Popen[bytes] | None = None
        self._stopped = threading.Event()

    @property
    def busy(self) -> bool:
        """True while sentences are waiting, being synthesized or playing (a hushed one is not)."""
        with self._lock:
            return self._busy()

    def _busy(self) -> bool:
        playing = self._proc is not None and self._player_run == self._run
        return self._pending > 0 or playing

    @property
    def interruptible(self) -> bool:
        """False while the speech playing would trigger the wake word by itself."""
        with self._lock:
            return not self._guarded

    def say(self, text: str) -> None:
        """Speak `text` after anything already queued; return once it has all played.

        A player that cannot be started or does not finish is an `AudioError`.
        """
        self.queue(text)
        self.wait()

    def queue(self, text: str) -> None:
        """Have `text` spoken after what is already queued, and return at once."""
        if self._stopped.is_set() or not text.strip():
            return
        with self._lock:
            self._pending += 1
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._work, name="jarvis-speech", daemon=True
                )
                self._worker.start()
            self._items.put((self._run, text))

    def wait(self) -> None:
        """Return once everything queued has played or was hushed.

        A failure of synthesis or of the player since the last wait is raised here, as the
        `AudioError` the worker kept.
        """
        with self._lock:
            while self._busy():
                self._lock.wait()
            error, self._error = self._error, None
        if error is not None:
            raise error

    def hush(self) -> None:
        """Drop every sentence not yet heard and end the one playing, at once."""
        with self._lock:
            self._run += 1
            self._guarded = False  # what held the wake phrase is no longer playing
            while True:
                try:
                    self._items.get_nowait()
                except queue.Empty:
                    break
                self._pending -= 1
            proc = self._proc
            self._lock.notify_all()
        if proc is not None:
            with contextlib.suppress(OSError):
                proc.terminate()

    def _work(self) -> None:
        while True:
            try:
                run, text = self._items.get(timeout=self._linger_s if self._proc else None)
            except queue.Empty:
                self._close()
                continue
            try:
                self._speak(run, text)
            except AudioError as exc:
                self._failed(exc)
            except Exception as exc:
                log.exception("speech synthesis failed")
                self._failed(AudioError(f"speech synthesis failed: {exc}"))
            finally:
                with self._lock:  # taken off the queue, so a hush did not count it
                    self._pending -= 1
                    self._lock.notify_all()

    def _failed(self, error: AudioError) -> None:
        self._end_player()
        with self._lock:
            self._error = self._error or error

    def _current(self, run: int) -> bool:
        return run == self._run and not self._stopped.is_set()

    def _speak(self, run: int, text: str) -> None:
        if not self._current(run):
            return
        pcm, rate = self._synthesize(text)
        if not self._current(run):
            return
        if self._proc is not None and (rate != self._rate or self._player_run != run):
            self._close()  # another rate, or a player a hush has ended
        first = self._proc is None
        pcm = trim_silence(pcm, rate, lead_s=LEAD_KEEP_S if first else None, tail_s=TAIL_KEEP_S)
        if not pcm:
            return
        if self._sounds_like_wake(pcm, rate, text):
            with self._lock:
                if run != self._run:  # hushed during the check: it will not be played
                    return
                self._guarded = True
            log.info("a sentence sounds like the wake word; it cannot be spoken over")
        proc = self._proc or self._open(rate, run)
        if proc is not None:
            self._write(proc, pcm, run)

    def _sounds_like_wake(self, pcm: bytes, rate: int, text: str) -> bool:
        """Return whether `pcm` would trigger the wake word; a check that fails counts as yes."""
        if self._own_wake is None:
            return False
        try:
            return self._own_wake(pcm, rate, text)
        except Exception:
            log.exception("the own-voice wake check failed")
            return True  # the safe answer: this sentence cannot be spoken over

    def _open(self, rate: int, run: int) -> subprocess.Popen[bytes] | None:
        cmd = self._player(rate)
        if not self._current(run):
            return None
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError as exc:
            msg = f"could not start {cmd[0]}: {exc}"
            raise AudioError(msg) from exc
        if proc.stdin is not None:
            os.set_blocking(proc.stdin.fileno(), False)  # a stuck player cannot block a write
        with self._lock:
            self._proc = proc
            self._rate = rate
            self._player_run = run
            self._opened = time.monotonic()
            self._written_s = 0.0
        if not self._current(run):  # a hush or stop that came while the player was starting
            with contextlib.suppress(OSError):
                proc.terminate()
        return proc

    def _write(self, proc: subprocess.Popen[bytes], pcm: bytes, run: int) -> None:
        """Hand `pcm` to the player as fast as it takes it.

        Raises:
            AudioError: the player stopped taking sound without a hush or stop.
        """
        stdin = proc.stdin
        assert stdin is not None  # ruff: ignore[assert] - started with a pipe
        fd = stdin.fileno()
        view = memoryview(pcm)
        progress = time.monotonic()
        while view:
            if not self._current(run):
                return
            if time.monotonic() - progress > self._grace_s:
                msg = "the player stopped taking sound"
                raise AudioError(msg)
            _, ready, _ = select.select([], [fd], [], self._grace_s)
            try:
                if not ready:
                    msg = "the player stopped taking sound"
                    raise AudioError(msg)
                view = view[os.write(fd, view[:WRITE_STEP]) :]
                progress = time.monotonic()
            except BlockingIOError:
                continue  # the pipe filled between select and write
            except OSError as exc:
                if not self._current(run):
                    return
                msg = f"the player stopped: {exc}"
                raise AudioError(msg) from exc
        with self._lock:
            self._written_s += len(pcm) / (2 * self._rate)

    def _close(self) -> None:
        """Let the open player finish what it was given, then end it.

        A player that does not finish in time is killed and reported at the next `wait`.
        """
        proc = self._proc
        if proc is None:
            return
        with contextlib.suppress(OSError):
            if proc.stdin is not None:
                proc.stdin.close()
        left_s = max(0.0, self._written_s - (time.monotonic() - self._opened))
        try:
            proc.wait(timeout=left_s + self._grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            with self._lock:
                self._error = self._error or AudioError(
                    f"{self._player(self._rate)[0]} did not finish playing"
                )
        self._release()

    def _end_player(self) -> None:
        proc = self._proc
        if proc is not None:
            with contextlib.suppress(OSError):
                proc.kill()
            with contextlib.suppress(OSError):
                if proc.stdin is not None:
                    proc.stdin.close()
            proc.wait()
            self._release()

    def _release(self) -> None:
        with self._lock:
            self._proc = None
            self._guarded = False
            self._lock.notify_all()

    def _play(self, pcm: bytes, rate: int) -> None:
        cmd = self._player(rate)
        if self._stopped.is_set():
            return
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError as exc:
            msg = f"could not start {cmd[0]}: {exc}"
            raise AudioError(msg) from exc
        self._chime = proc
        if self._stopped.is_set():  # a stop that came while the player was starting
            with contextlib.suppress(OSError):
                proc.terminate()
        try:
            proc.communicate(pcm, timeout=len(pcm) / (2 * rate) + self._grace_s)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.communicate()
            msg = f"{cmd[0]} did not finish playing"
            raise AudioError(msg) from exc
        finally:
            self._chime = None

    def chime(self) -> None:
        """Play a short two-note chime (about 0.15 s) that says "I'm listening"."""
        rate = 24000
        notes: list[NDArray[np.float64]] = []
        for freq, seconds in ((880.0, 0.06), (1320.0, 0.09)):
            t = np.arange(int(rate * seconds)) / rate
            fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)
            notes.append(0.25 * np.sin(2 * np.pi * freq * t) * fade)
        pcm = (np.concatenate(notes) * 32767).astype(np.int16).tobytes()
        self._play(pcm, rate)

    def stop(self) -> None:
        """Stop speaking now and for good (Jarvis is shutting down); safe from any thread.

        Nothing is spoken after it, and the player running now is ended.
        """
        self._stopped.set()
        self.hush()
        chime = self._chime
        if chime is not None:
            with contextlib.suppress(OSError):
                chime.terminate()
