"""Microphone capture, wake word, end of speech, transcription and speech output.

Capture and playback go through PipeWire's own tools (`pw-record`, `pw-play`); transcription goes
through Voxtype's documented `voxtype transcribe FILE` contract (path B in docs/plan.md), so
Jarvis never touches Voxtype's daemon, config or the owner's recordings.
"""

import contextlib
import json
import logging
import queue
import re
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from jarvis.jsonutil import obj, objs

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
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

    def frames(self) -> Iterator[NDArray[np.int16]]:
        """Yield frames from whichever matching stream is playing, until stopped."""
        while not self._stopping.is_set():
            target = self._target()
            if target is not None:
                try:
                    yield from self._record(target)
                except AudioError as exc:
                    log.info("PC audio stream ended (%s); waiting for it to come back", exc)
            self._stopping.wait(self._poll_s)

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
        self._ended.clear()
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError as exc:
            self._ended.set()
            msg = f"could not start {cmd[0]}: {exc}"
            raise AudioError(msg) from exc
        self._proc = proc
        self._thread = threading.Thread(target=self._read, args=(proc.stdout,), daemon=True)
        self._thread.start()

    def _read(self, stream: IO[bytes] | None) -> None:
        need = FRAME * 2
        while stream is not None:
            data = stream.read(need)
            if len(data) < need:
                break
            push_frame(self._frames, np.frombuffer(data, dtype=np.int16).copy())
        self._ended.set()

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
        if max(scores.values(), default=0.0) >= self._threshold:
            self._model.reset()
            return True
        return False

    def reset(self) -> None:
        """Forget audio heard so far."""
        self._model.reset()


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


def write_wav(path: Path, samples: NDArray[np.int16]) -> None:
    """Write 16 kHz mono 16-bit PCM."""
    with wave.open(str(path), "wb") as handle:
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


def voxtype_command(wav: Path, engine: str = "", model: str = "", prompt: str = "") -> list[str]:
    """Return the `voxtype transcribe` command; options override the owner's config for one run."""
    cmd = ["voxtype"]
    if model:
        cmd += ["--model", model]
    if engine:
        cmd += ["--engine", engine]
    if prompt:
        cmd += ["--initial-prompt", prompt]
    return [*cmd, "transcribe", str(wav)]


def transcribe(  # ruff: ignore[too-many-arguments] - Voxtype's options plus a test seam
    samples: NDArray[np.int16],
    engine: str = "",
    model: str = "",
    prompt: str = "",
    timeout_s: float = 60.0,
    *,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> str:
    """Transcribe an utterance with `voxtype transcribe` on a private temporary WAV.

    `prompt` hints Whisper with words it should expect (the owner's artist names). `run` replaces
    `subprocess.run` in tests.

    Raises:
        AudioError: Voxtype is missing, could not be started with these options, failed or timed
            out.
    """
    with tempfile.TemporaryDirectory(prefix="jarvis-") as tmp:
        wav = Path(tmp) / "utterance.wav"
        write_wav(wav, samples)
        try:
            done = run(
                voxtype_command(wav, engine, model, prompt),
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
    from piper import PiperVoice  # ruff: ignore[import-outside-top-level] - heavy

    piper = PiperVoice.load(files[0])

    def run(text: str) -> tuple[bytes, int]:
        chunks = list(piper.synthesize(text))
        rate = chunks[0].sample_rate if chunks else 22050
        return b"".join(chunk.audio_int16_bytes for chunk in chunks), rate

    return run


def pw_play_command(rate: int) -> list[str]:
    """Return the `pw-play` command for raw mono 16-bit PCM on standard input."""
    return ["pw-play", "--raw", "--rate", str(rate), "--channels", "1", "--format", "s16", "-"]


PLAY_GRACE_S = 10.0  # how long playback may run past the sound's own length before it is ended


class Speaker:
    """Speaks through `pw-play` with a synthesizer from `make_synthesizer`."""

    def __init__(
        self,
        synthesize: Callable[[str], tuple[bytes, int]],
        *,
        player: Callable[[int], list[str]] = pw_play_command,
        grace_s: float = PLAY_GRACE_S,
    ) -> None:
        """Use `synthesize` for text to speech; `player` builds the playback command for a rate."""
        super().__init__()
        self._synthesize = synthesize
        self._player = player
        self._grace_s = grace_s
        self._proc: subprocess.Popen[bytes] | None = None

    def say(self, text: str) -> None:
        """Speak `text` and return once playback has drained (or was stopped).

        A player that cannot be started or does not finish is an `AudioError`.
        """
        pcm, rate = self._synthesize(text)
        if pcm:
            self._play(pcm, rate)

    def _play(self, pcm: bytes, rate: int) -> None:
        cmd = self._player(rate)
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError as exc:
            msg = f"could not start {cmd[0]}: {exc}"
            raise AudioError(msg) from exc
        self._proc = proc
        try:
            proc.communicate(pcm, timeout=len(pcm) / (2 * rate) + self._grace_s)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.communicate()
            msg = f"{cmd[0]} did not finish playing"
            raise AudioError(msg) from exc
        finally:
            self._proc = None

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
        """Stop speaking now.

        Takes no lock, so it is safe from any thread, a signal handler included: it only signals
        the player that is running at this moment, if there is one.
        """
        proc = self._proc
        if proc is not None:
            with contextlib.suppress(OSError):
                proc.terminate()
