"""Microphone capture, wake word, end of speech, transcription and speech output.

Capture and playback go through PipeWire's own tools (`pw-record`, `pw-play`); transcription goes
through Voxtype's documented `voxtype transcribe FILE` contract (path B in docs/plan.md), so
Jarvis never touches Voxtype's daemon, config or the owner's recordings.
"""

import collections
import contextlib
import functools
import json
import logging
import math
import os
import queue
import re
import select
import stat
import struct
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, cast, runtime_checkable

import numpy as np
import onnxruntime as ort

from jarvis.confirmkey import boot_clock
from jarvis.jsonutil import obj, objs

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
    from typing import IO

    from numpy.typing import NDArray

RATE = 16000
FRAME = 1280  # 80 ms, the wake model's step
VAD_CHUNK = 512
VAD_CONTEXT = 64
STOP_WAIT_S = 5.0  # how long a capture process gets to exit before it is killed
READER_JOIN_S = 0.5  # how long the reader gets to see the pipe end once the process is gone
MIC_STALL_S = 5.0  # a microphone capture that sends nothing this long has lost its source
REOPEN_FIRST_S = 1.0  # the wait before reopening a microphone whose capture ended on its own
REOPEN_MAX_S = 30.0  # each failed reopen doubles the wait, up to this
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


class ControlTimeoutError(AudioError):
    """A sound or media command was sent and did not answer in time: it may have taken effect."""


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


def _app_streams(objects: object, apps: list[str]) -> tuple[list[str], list[str]]:
    """Return the serials of the apps' playback streams that play and those that are idle."""
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
    return playing, idle


def match_app_streams(objects: object, apps: list[str]) -> list[str]:
    """Return the serials of playback streams (in `pw-dump` output) that belong to one of `apps`.

    An app matches by a case-insensitive part of its process binary, application or node name,
    so "discord" finds Discord's voice stream (its application name is "WEBRTC VoiceEngine").
    Streams that are playing come first: an app may also hold idle ones.
    """
    playing, idle = _app_streams(objects, apps)
    return playing + idle


def playing_app_streams(objects: object, apps: list[str]) -> list[str]:
    """Return the serials of every stream of `apps` that plays, or of one idle one if none does.

    The apps are matched as in `match_app_streams`. All playing streams are recorded together,
    so an app that plays through several is heard whole.
    """
    playing, idle = _app_streams(objects, apps)
    return playing or idle[:1]


def others_playing(objects: object, ours: Callable[[int], bool]) -> bool:
    """Whether a playback stream (in `pw-dump` output) is running that is not Jarvis's own.

    `ours` says whether a process id is Jarvis or one of its players.
    """
    for _id, info, props in _nodes(objects):
        if props.get("media.class") != "Stream/Output/Audio" or info.get("state") != "running":
            continue
        pid = str(props.get("application.process.id", ""))
        if not (pid.isdigit() and ours(int(pid))):
            return True
    return False


def _ours(pid: int) -> bool:
    """Whether `pid` is this process or a child of it (a `pw-play` speaking a reply)."""
    if pid == os.getpid():
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    fields = stat.rpartition(")")[2].split()  # the name before it may hold spaces
    return len(fields) > 1 and fields[1] == str(os.getpid())


def audio_playing_elsewhere() -> bool | None:
    """Whether another program plays audio now, or None when PipeWire cannot be read."""
    try:
        done = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5, check=False)
        objects: object = json.loads(done.stdout)
    except OSError, subprocess.TimeoutExpired, ValueError:
        return None
    if done.returncode != 0:
        return None
    return others_playing(objects, _ours)


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


def source_found(objects: object, source: str) -> bool:
    """Whether `source` (a node name or serial) names a microphone node in `pw-dump` output.

    `pw-record --target` takes either; a microphone is any `Audio/Source` node, a virtual or an
    internal one included. A graph that could not be read has none.
    """
    return any(
        str(props.get("media.class", "")).startswith("Audio/Source")
        and source in {str(props.get("node.name")), str(props.get("object.serial"))}
        for _id, _info, props in _nodes(objects)
    )


CONTROL_TIMEOUT_S = 5.0


def _control(program: str, args: Sequence[str], who: str) -> str:
    """Run `program` with a fixed argument list and return its output; `who` names it in errors.

    Raises:
        ControlTimeoutError: the program did not answer in time; it may have done what it was told.
        AudioError: the program is missing or failed.
    """
    try:
        done = subprocess.run(
            [program, *args], capture_output=True, text=True, timeout=CONTROL_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"{who} did not answer in time"
        raise ControlTimeoutError(msg) from exc
    except OSError as exc:
        msg = f"{who} did not answer ({type(exc).__name__})"
        raise AudioError(msg) from exc
    if done.returncode != 0:
        msg = f"{who} refused that ({done.stderr.strip()[:200] or done.returncode})"
        raise AudioError(msg)
    return done.stdout


def wpctl(args: Sequence[str]) -> str:
    """Run PipeWire's `wpctl` with `pc_volume`'s arguments and return its output.

    An `AudioError` says wpctl is missing, timed out or failed.
    """
    return _control("wpctl", args, "the sound server")


def playerctl(args: Sequence[str]) -> str:
    """Run `playerctl` (MPRIS media players) with `pc_media`'s arguments and return its output.

    An `AudioError` says playerctl is missing, timed out or failed; its own words come back in
    it, so "No players found" can be told apart.
    """
    return _control("playerctl", args, "the media players")


def pw_dump() -> object:
    """Return PipeWire's object graph, or an empty one when it cannot be read."""
    try:
        done = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5, check=False)
        return json.loads(done.stdout or "[]")
    except OSError, subprocess.TimeoutExpired, ValueError:
        return []


def app_stream_serials(apps: list[str]) -> list[str]:
    """Return the serials of the given apps' streams to record now (`playing_app_streams`)."""
    return playing_app_streams(pw_dump(), apps)


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
    records it until it goes away and then waits again. Every stream the apps play is recorded,
    each by its own recorder pinned to it, and heard mixed. With no apps named, it records the
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

        `find` returns the serials of the streams to record (`playing_app_streams`).
        `command_for(target)` builds the capture command, all PC output for "" (a test seam).
        """
        super().__init__()
        self._apps = apps
        self._find = find
        self._poll_s = poll_s
        self._command_for = command_for
        self._feeds = feeds
        self._stopping = threading.Event()
        self._mics: list[Microphone] = []

    def _targets(self) -> list[str] | None:
        """Return what to record now: the streams' serials, [""] for all PC output, or None."""
        if not self._apps:
            return [""]
        return list(dict.fromkeys(self._find(self._apps))) or None

    def _misplaced(self, targets: list[str], mics: list[Microphone]) -> str:
        """Return why the recorders no longer hear just `targets` (empty while they do).

        A stream may have gone or a new one started, or a recorder may have been linked to
        something else (the default microphone, on a session manager that ignores the pinning
        properties). Either way the frames are not the apps' as they play now.
        """
        found = set(self._find(self._apps))
        if not found.issuperset(targets):
            return "the stream is gone"
        if found.difference(targets):
            return "another stream started"
        for target, mic in zip(targets, mics, strict=True):
            pid = mic.pid
            if pid is not None and any(source != target for source in self._feeds(pid)):
                return "the recorder was linked to another source"
        return ""

    def _record(self, targets: list[str]) -> Iterator[NDArray[np.int16]]:
        """Yield frames from `targets` until one ends or the recorders stop hearing just them."""
        mics = [Microphone(command=self._command_for(target)) for target in targets]
        try:
            for mic in mics:
                mic.start()
                self._mics.append(mic)
            frames = mics[0].frames() if len(mics) == 1 else _mixed(mics)
            checked = time.monotonic()
            for frame in frames:
                if targets != [""] and time.monotonic() - checked >= self._poll_s:
                    checked = time.monotonic()
                    reason = self._misplaced(targets, mics)
                    if reason:
                        log.info("PC audio recorder dropped: %s", reason)
                        return
                yield frame
        finally:
            self._mics = []
            for mic in mics:
                mic.stop()

    def _session(self, targets: list[str]) -> Generator[NDArray[np.int16]]:
        try:
            yield from self._record(targets)
        except AudioError as exc:
            log.info("PC audio stream ended (%s); waiting for it to come back", exc)

    def sessions(self) -> Generator[Generator[NDArray[np.int16]]]:
        """Yield one frame stream per set of matching streams that play, each ending with them.

        A consumer that reads speech from one session cannot run on into the next stream. Runs
        until stopped.
        """
        while not self._stopping.is_set():
            targets = self._targets()
            if targets is not None:
                yield self._session(targets)
            self._stopping.wait(self._poll_s)

    def frames(self) -> Iterator[NDArray[np.int16]]:
        """Yield frames from whichever matching streams are playing, until stopped."""
        for session in self.sessions():
            yield from session

    def drain(self) -> None:
        """Drop frames captured so far."""
        for mic in list(self._mics):
            mic.drain()

    def stop(self) -> None:
        """Stop following the apps."""
        self._stopping.set()
        for mic in list(self._mics):
            mic.stop()


_ACTIVE_S = 0.5  # a recorder that sent nothing for this long is not waited for in the mix


def _mixed(mics: list[Microphone]) -> Iterator[NDArray[np.int16]]:
    """Yield the sum of one frame from each recorder that is sending, until all have ended.

    Each recorder is read on its own thread. A frame is let out once every recorder heard
    lately has given one, or one gives a second before the rest: a paused stream that sends
    nothing holds up the others for no longer than `_ACTIVE_S`.

    Raises:
        AudioError: every recorder's capture ended.
    """
    shared: queue.Queue[tuple[int, NDArray[np.int16]]] = queue.Queue(maxsize=400 * len(mics))
    readers = [
        threading.Thread(
            target=_forward, args=(n, mic, shared), name="jarvis-pc-audio-mix", daemon=True
        )
        for n, mic in enumerate(mics)
    ]
    for reader in readers:
        reader.start()
    pending: dict[int, NDArray[np.int16]] = {}
    heard: dict[int, float] = {}
    while True:
        try:
            index, frame = shared.get(timeout=0.2)
        except queue.Empty:
            if not any(reader.is_alive() for reader in readers):
                msg = "the PC audio streams ended"
                raise AudioError(msg) from None
            continue
        if index in pending:
            yield _mix(list(pending.values()))
            pending.clear()
        now = time.monotonic()
        pending[index], heard[index] = frame, now
        lately = {n for n, at in heard.items() if now - at < _ACTIVE_S and readers[n].is_alive()}
        if lately.issubset(pending):
            yield _mix(list(pending.values()))
            pending.clear()


def _forward(
    index: int, mic: Microphone, shared: queue.Queue[tuple[int, NDArray[np.int16]]]
) -> None:
    """Pass the recorder's frames on, tagged with `index`, the oldest dropped when full."""
    with contextlib.suppress(AudioError):
        for frame in mic.frames():
            while True:
                try:
                    shared.put_nowait((index, frame))
                    break
                except queue.Full:
                    with contextlib.suppress(queue.Empty):
                        shared.get_nowait()


def _mix(frames: list[NDArray[np.int16]]) -> NDArray[np.int16]:
    """Return the frames added together, clipped to 16 bits."""
    total = np.sum(np.stack(frames).astype(np.int32), axis=0)
    return np.clip(total, -32768, 32767).astype(np.int16)


class Microphone:
    """Streams 16 kHz mono int16 frames of 80 ms from a PipeWire source."""

    def __init__(
        self, source: str = "", command: list[str] | None = None, *, stall_s: float | None = None
    ) -> None:
        """Capture from `source` (a node name or serial), or PipeWire's default when empty.

        `command` replaces `pw-record` (tests feed raw PCM from a file this way). With `stall_s`,
        a capture that sends no frame for that long counts as ended: `pw-record` keeps running,
        silent, when its source goes away and there is none to move to.
        """
        super().__init__()
        self._source = source
        self._command = command
        self._stall_s = stall_s
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
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError as exc:
            ended.set()
            self._ended = ended
            msg = f"could not start {cmd[0]}: {exc}"
            raise AudioError(msg) from exc
        thread = threading.Thread(target=self._read, args=(proc.stdout, ended), daemon=True)
        # A stop() sees the process with its own reader thread or neither.
        with self._stop_lock:
            self._ended = ended
            self._proc = proc
            self._thread = thread
            thread.start()

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
            AudioError: the capture process stopped, or sent nothing for `stall_s`.
        """
        # The monotonic clock stands still while the PC sleeps, so a suspend is no stall.
        last = time.monotonic()
        while True:
            try:
                frame = self._frames.get(timeout=0.2)
            except queue.Empty:
                if self._ended.is_set():
                    msg = "the microphone stream ended"
                    raise AudioError(msg) from None
                if self._stall_s is not None and time.monotonic() - last > self._stall_s:
                    msg = f"no audio from the microphone for {self._stall_s:g} s"
                    raise AudioError(msg) from None
                continue
            yield frame
            last = time.monotonic()  # the time the reader spent elsewhere is no stall

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
            thread = self._thread
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(STOP_WAIT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        # The reader sees the pipe end at once when the process is gone. One that does not is held
        # by a child the capture command left behind, which will not let go soon: do not wait on
        # it (`PausableMic` stops under its lock). Closing the pipe while it still reads would
        # wait for it, so that reader keeps the pipe.
        if thread is not None:
            thread.join(READER_JOIN_S)
        if proc.stdout is not None and (thread is None or not thread.is_alive()):
            proc.stdout.close()


class MicPausedError(AudioError):
    """The microphone was let go by `jarvis pause`; frames come again after `resume`."""


class _Capture(Protocol):
    @property
    def pid(self) -> int | None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def drain(self) -> None: ...
    def frames(self) -> Iterator[NDArray[np.int16]]: ...


def _wait(event: threading.Event, seconds: float) -> bool:
    return event.wait(seconds)


class PausableMic:
    """A `Microphone` that `jarvis pause` can let go of (its capture ends) and `resume` reopen.

    A frames reader open at a pause raises `MicPausedError` at its next frame, and one opened while
    paused raises it at once, so a request that was listening or about to ends; the loop then
    waits in `wait_resumed`, using no CPU. A capture that ends on its own (PipeWire restarted,
    the microphone unplugged) is started again by `recover`.
    """

    def __init__(
        self, mic: _Capture, *, wait: Callable[[threading.Event, float], bool] = _wait
    ) -> None:
        """Wrap `mic` (a `Microphone`), which this object starts and stops from now on.

        `wait` waits on an event for some seconds and says whether it was set (a test seam).
        """
        super().__init__()
        self._mic = mic
        self._lock = threading.Lock()
        self._running = threading.Event()  # clear while paused
        self._running.set()
        self._pauses = 0  # pauses so far; a reader opened before one ends
        self._stopped = False
        self._interrupt = threading.Event()  # a pause or stop ends a reopen wait
        self._wait = wait
        self._failures = 0  # reopens since the last frame

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
            self._interrupt.set()
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
            self._interrupt.set()
            self._mic.stop()

    def drain(self) -> None:
        """Drop frames captured so far."""
        self._mic.drain()

    def recover(self) -> bool:
        """Start a new capture after the last one ended on its own; False once stopped.

        Waits first: a second, doubled after each reopen that brought no frame, up to 30 s. A
        pause during the wait ends it and leaves the microphone closed for `resume`. A capture
        tool that cannot be started is logged, and its reader ends at once for the next try.
        """
        delay = min(REOPEN_MAX_S, REOPEN_FIRST_S * 2.0**self._failures)
        self._failures += 1
        with self._lock:
            if self._stopped:
                return False
            if self.paused:  # `resume` starts the next capture
                return True
            pauses = self._pauses
            self._interrupt.clear()
            self._mic.stop()  # the ended capture's process is reaped
        if self._wait(self._interrupt, delay) and self._stopped:
            return False
        with self._lock:
            if self._stopped:
                return False
            if self._pauses != pauses:  # paused meanwhile; a resume has started its own capture
                return True
            self._mic.drain()
            try:
                self._mic.start()
            except AudioError as exc:
                log.warning("the microphone could not be reopened: %s", exc)
        return True

    def wait_resumed(self) -> None:
        """Return once the microphone is open again, or stopped; waits using no CPU."""
        self._running.wait()

    def frames(self) -> Iterator[NDArray[np.int16]]:
        """Yield frames until the capture ends.

        Raises:
            MicPausedError: the microphone is paused, or a pause came while this reader was open.
            AudioError: the capture ended, or the microphone was stopped.
        """
        if self._stopped:
            msg = "the microphone was stopped"
            raise AudioError(msg)
        if self.paused:  # a request or watcher that would start now ends instead
            raise MicPausedError
        pauses = self._pauses
        for frame in self._frames_until_ended():
            if self._pauses != pauses:
                raise MicPausedError
            self._failures = 0
            yield frame

    def _frames_until_ended(self) -> Iterator[NDArray[np.int16]]:
        pauses = self._pauses
        try:
            yield from self._mic.frames()
        except AudioError:
            if self._pauses != pauses and not self._stopped:
                raise MicPausedError from None
            raise


def wake_load_errors(paths: Sequence[Path]) -> dict[Path, str]:
    """Return, for each of `paths` that LiveKit cannot load as the runtime does, the reason.

    Loading reads the file into an ONNX Runtime session; it plays and records nothing.
    """
    from livekit.wakeword import WakeWordModel

    model = WakeWordModel(sess_options=quiet_session_options())
    errors: dict[Path, str] = {}
    for path in paths:
        try:
            model.load_model(path)
        except (OSError, ValueError) as exc:
            errors[path] = str(exc)
    return errors


# A wake score over Jarvis's speech that falls short but is logged: its own voice stays near 0.05.
NEAR_MISS = 0.2


class WakeWord:
    """LiveKit streaming wake word models, one per phrase, heard side by side."""

    def __init__(
        self,
        model_paths: Sequence[Path],
        threshold: float,
        thresholds: Mapping[str, float] | None = None,
        over_speech: float | None = None,
    ) -> None:
        """Load the model files, leaving out with a warning any that is missing or unusable.

        Each model is held to its own score in `thresholds`, by the name of its file, or else to
        `threshold`. While Jarvis speaks (`heard_over_speech`) a model needs at most
        `over_speech`: Jarvis's own voice in the microphone lowers the owner's score.

        One bad file (missing, half copied, not a streaming model) must not leave Jarvis deaf to
        the others; only when none loads is the last error (an `OSError` or `ValueError`) raised,
        as it would be for a single model.
        """
        super().__init__()
        from livekit.wakeword import WakeWordModel

        self._model = WakeWordModel(sess_options=quiet_session_options())
        error: OSError | ValueError | None = None
        loaded = 0
        for path in model_paths:
            try:
                self._model.load_model(path)
            except (OSError, ValueError) as exc:
                log.warning(
                    "wake word model %s cannot be used (%s); not listening for it", path, exc
                )
                error = exc
            else:
                loaded += 1
        if not loaded and error is not None:
            raise error
        self._threshold = threshold
        self._thresholds = dict(thresholds or {})
        self._over_speech = over_speech
        self._closest: tuple[str, float] | None = None  # the best miss while Jarvis spoke

    def heard(self, frame: NDArray[np.int16]) -> bool:
        """Feed one frame; return True when any of the wake phrases was detected."""
        return self._heard(frame, speaking=False)

    def heard_over_speech(self, frame: NDArray[np.int16]) -> bool:
        """Feed one frame heard while Jarvis speaks; True when a wake phrase was detected.

        A score that falls short is kept for `closest_miss`.
        """
        return self._heard(frame, speaking=True)

    def closest_miss(self) -> str | None:
        """Return, once, the highest score over Jarvis's speech that fell short, from NEAR_MISS."""
        closest, self._closest = self._closest, None
        if closest is None:
            return None
        name, score = closest
        return f"{name} score {score:.2f} (threshold {self._needed(name, speaking=True):.2f})"

    def _heard(self, frame: NDArray[np.int16], *, speaking: bool) -> bool:
        scores = self._model.predict(frame)
        over = {
            name: score
            for name, score in scores.items()
            if score >= self._needed(name, speaking=speaking)
        }
        if not over:
            if speaking and scores:
                name = max(scores, key=scores.__getitem__)
                best = self._closest[1] if self._closest is not None else NEAR_MISS
                if scores[name] >= best:
                    self._closest = (name, scores[name])
            return False
        heard = ", ".join(
            f"{name} score {over[name]:.2f} (threshold {self._needed(name, speaking=speaking):.2f})"
            for name in sorted(over, key=over.__getitem__, reverse=True)
        )
        log.info("wake word %s", heard)
        self._closest = None
        self._model.reset()
        return True

    def _needed(self, name: str, *, speaking: bool = False) -> float:
        needed = self._thresholds.get(name, self._threshold)
        if speaking and self._over_speech is not None:
            return min(needed, self._over_speech)
        return needed

    def reset(self) -> None:
        """Forget audio heard so far."""
        self._model.reset()


AZURE_HELPER = Path(__file__).with_name("azure_wake_helper.py")
AZURE_READY_S = 30.0  # how long the helper may take to load and try its model
AZURE_CLOSE_S = 1.0  # how long the helper may take to end once its input closes
AZURE_HISTORY = 40  # frames kept (3.2 s), so the words right after the keyword are not lost
AZURE_LEAD_BACK = RATE // 4  # start a request 0.25 s before the keyword's reported end
AZURE_ERR_LINES = 20


def keyword_tail(lead: list[NDArray[np.int16]]) -> int:
    """Count the frames of a late wake word's `lead` that start before the keyword ended.

    The lead begins `AZURE_LEAD_BACK` before the keyword's reported end and runs to the
    detection, which comes 0.3 to 0.6 s later: everything after that first quarter second may
    already be the request.
    """
    tail = 0
    start = 0
    for pcm in lead:
        if start >= AZURE_LEAD_BACK:
            break
        tail += 1
        start += pcm.size
    return tail


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
        for stream in (self._proc.stdout, self._proc.stderr):
            if stream is not None:
                with contextlib.suppress(OSError, ValueError):
                    stream.close()


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

    def heard_over_speech(self, frame: NDArray[np.int16]) -> bool:
        """Like `heard`, for a frame heard while Jarvis speaks (LiveKit's model needs less)."""
        if self._primary is None and isinstance(self._backup, OverSpeech):
            return self._backup.heard_over_speech(frame)
        return self.heard(frame)

    def closest_miss(self) -> str | None:
        """Return the best score that fell short over Jarvis's speech, once (LiveKit's only)."""
        return self._backup.closest_miss() if isinstance(self._backup, OverSpeech) else None

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
    """Return the transcript from `voxtype transcribe` output without its progress and log lines.

    `-q` already leaves out the log lines; they are filtered too in case a Voxtype release prints
    them anyway.
    """
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


# Engines that read their model file from a section of Voxtype's config named after them, not
# from `--model`.
FILE_ENGINES = frozenset({"parakeet", "cohere", "gguf"})
_GGUF_SCALARS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_STRING, _GGUF_ARRAY = 8, 9
# llama.cpp writes general.architecture first; the files seen have it at byte 56. Only this much
# of a file is read, so a huge or hostile one costs one bounded read and parse.
_GGUF_HEAD = 1 << 20
_GGUF_MAX_KEYS = 64


def gguf_architecture(path: Path) -> str:
    """Return a GGUF checkpoint's `general.architecture`, or "" when it cannot be read.

    Only a regular file is read (a named pipe would block), and only its first `_GGUF_HEAD`
    bytes.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError, ValueError:  # ValueError: a NUL in the path
        return ""
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return ""
        head = os.read(fd, _GGUF_HEAD)
    except OSError:
        return ""
    finally:
        os.close(fd)
    try:
        return _architecture(_Head(head))
    except ValueError, struct.error:  # UnicodeDecodeError is a ValueError
        return ""


class _Head:
    """The bytes read from the start of a GGUF file, taken in order."""

    def __init__(self, data: bytes) -> None:
        super().__init__()
        self._data = data
        self._at = 0

    def left(self) -> int:
        return len(self._data) - self._at

    def take(self, size: int) -> bytes:
        if size > self.left():
            msg = "the metadata runs past what was read"
            raise ValueError(msg)
        part = self._data[self._at : self._at + size]
        self._at += size
        return part

    def numbers(self, layout: str) -> tuple[int, ...]:
        return struct.unpack(layout, self.take(struct.calcsize(layout)))

    def text(self) -> str:
        (length,) = self.numbers("<Q")
        return self.take(length).decode()


def _skip(head: _Head, kind: int) -> None:
    """Read past one value; arrays nest without recursion, and every step reads some bytes.

    Raises:
        ValueError: the value is of no kind GGUF defines, or longer than the bytes read.
    """
    pending = [[kind, 1]]  # [kind, how many still to skip], innermost last
    while pending:
        top = pending[-1]
        item, count = top
        if not count:
            pending.pop()
        elif item in _GGUF_SCALARS:
            head.take(_GGUF_SCALARS[item] * count)
            pending.pop()
        elif item == _GGUF_STRING:
            top[1] -= 1
            head.text()
        elif item == _GGUF_ARRAY:
            top[1] -= 1
            inner, items = head.numbers("<IQ")
            if items > head.left():  # every item takes at least a byte
                msg = "an array longer than the metadata"
                raise ValueError(msg)
            pending.append([inner, items])
        else:
            msg = f"a value of unknown kind {item}"
            raise ValueError(msg)


def _architecture(head: _Head) -> str:
    if head.take(4) != b"GGUF":
        return ""
    _version, _tensors, keys = head.numbers("<IQQ")
    for _ in range(min(keys, _GGUF_MAX_KEYS)):
        key = head.text()
        (kind,) = head.numbers("<I")
        if key == "general.architecture" and kind == _GGUF_STRING:
            return head.text()
        _skip(head, kind)
    return ""


_TOML_ESCAPES = {'"': '\\"', "\\": "\\\\"}


def _toml_string(text: str) -> str:
    """Return `text` as a TOML basic string.

    Raises:
        ValueError: `text` holds a lone surrogate (a path whose bytes are not UTF-8), which a
            TOML file cannot hold.
    """
    try:
        text.encode()
    except UnicodeEncodeError as exc:
        msg = f"{text!r} is not UTF-8 text, which Voxtype's config must be"
        raise ValueError(msg) from exc
    escaped = "".join(
        _TOML_ESCAPES.get(ch) or (f"\\u{ord(ch):04X}" if ch < " " or ch == "\x7f" else ch)
        for ch in text
    )
    return f'"{escaped}"'


def model_in_settings(engine: str, model: str) -> bool:
    """Whether `model` goes in Jarvis's Voxtype config rather than on the command line."""
    return engine in FILE_ENGINES and model.startswith("/")


def voxtype_settings(
    threads: int, *, short_window: bool, engine: str = "", model: str = "", backend: str = ""
) -> str:
    """Return Jarvis's own Voxtype config (TOML), read in place of the owner's for its runs.

    Whisper pads every clip to 30 s, so a two-second request costs as much as a long one;
    `short_window` (Voxtype's `context_window_optimization`) sizes the window to the clip, about
    five times faster for a spoken command. Whisper.cpp also uses only 4 threads unless told; the
    GGUF engine is told the same count (measured 2026-11-25 on its CPU build: 2 threads took 10 to
    13 s for a request its default took 6.7 s for).
    `keep_failed_recordings` off means Voxtype never leaves the owner's voice in its
    `failed-audio` folder after one of Jarvis's runs. A model file for an engine in
    `FILE_ENGINES` goes in that engine's own section, which is where Voxtype reads it; the
    GGUF engine is also told the file's architecture, which it refuses to guess. A model path
    that is not UTF-8 text cannot go in the file, and raises ValueError. `backend` is where
    the GGUF engine runs (its `gguf_backend`): on Voxtype's GPU build, "cpu" keeps it off the
    graphics card (measured 2026-11-25: no VRAM instead of about 3 GB per request, 7 to 8
    CPU-seconds instead of 0.35, the same transcript).
    """
    text = (
        "[audio]\n"
        "keep_failed_recordings = false\n"
        "\n"
        "[whisper]\n"
        f"threads = {int(threads)}\n"
        f"context_window_optimization = {'true' if short_window else 'false'}\n"
    )
    if model_in_settings(engine, model):
        text += f"\n[{engine}]\nmodel = {_toml_string(model)}\n"
        architecture = gguf_architecture(Path(model)) if engine == "gguf" else ""
        if architecture:
            text += f"architecture = {_toml_string(architecture)}\n"
        elif engine == "gguf":
            log.warning(
                "could not read the architecture of %r; Voxtype refuses a GGUF file without one",
                model,
            )
        if engine == "gguf":
            text += f"threads = {int(threads)}\n"
    if engine == "gguf" and backend:
        if not model_in_settings(engine, model):
            text += "\n[gguf]\n"
        text += f"gguf_backend = {_toml_string(backend)}\n"
    return text


# A file larger than this is not read ahead: no speech model is, and a wrong path may be.
MAX_WARM_BYTES = 8 << 30
# Measured on Linux 7.2 with ext4 (2026-11-25): read 2 MiB or more at a time, only half of a 2 GB
# file was still in the cache afterwards; read 1 MiB or less at a time, all of it was.
_WARM_CHUNK = 1 << 20
_CACHE_SAMPLES = 64
# `preadv2`'s flag for "do not wait for the disk" (Linux 4.14). A Python built against older
# headers does not name it; the number is the kernel's own.
RWF_NOWAIT: int = getattr(os, "RWF_NOWAIT", 8)


def _in_memory(fd: int, size: int) -> bool:
    """Whether a file's data is in the system's file cache, judged from samples spread over it."""
    sample = bytearray(1)
    for number in range(_CACHE_SAMPLES):
        try:
            os.preadv(fd, [sample], size * number // _CACHE_SAMPLES, RWF_NOWAIT)
        except OSError, NotImplementedError:
            # BlockingIOError: that part is on the disk. Anything else: this system or this
            # Python cannot say, and the file is read.
            return False
    return True


def read_into_memory(path: Path, clock: Callable[[], float] = time.monotonic) -> float | None:
    """Read a file into the system's file cache; return the seconds it took, None when not read.

    The program that opens the file next then finds it in memory instead of waiting for the
    disk. Nothing is read when it is there already (so this costs nothing then), when it is not a
    regular file (a named pipe would block), is empty or larger than `MAX_WARM_BYTES`, or cannot
    be opened or read.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError, ValueError:  # ValueError: a NUL in the path
        return None
    try:
        return _read_ahead(fd, clock)
    except OSError:
        return None
    finally:
        os.close(fd)


def _read_ahead(fd: int, clock: Callable[[], float]) -> float | None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_WARM_BYTES:
        return None
    if _in_memory(fd, info.st_size):
        return None
    began = clock()
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
    chunk = bytearray(_WARM_CHUNK)
    while os.readv(fd, [chunk]):
        pass
    return clock() - began


WARM_COOL_S = 300.0  # after a whole read the file is in memory still; no new read sooner


class FileWarmer:
    """Reads one file into memory on a thread of its own, one read at a time.

    Voxtype loads the speech model anew for every request. With the file out of the file cache
    that load waits for the disk; started at the wake word, the read is over, or nearly, by the
    time the request has been said.
    """

    def __init__(
        self,
        path: Path | None,
        read: Callable[[Path], float | None] = read_into_memory,
        *,
        clock: Callable[[], float] = boot_clock,
    ) -> None:
        """Keep the file to read; without one, `start` does nothing.

        `clock` counts time asleep too: a suspend may have pushed the file out of memory.
        """
        super().__init__()
        self._path = path
        self._read = read
        self._clock = clock
        self._reading = threading.Lock()
        self._read_at = float("-inf")  # when the last whole read ended

    def start(self) -> threading.Thread | None:
        """Start a read unless one is under way or ended within `WARM_COOL_S`.

        Return its thread, None when none started. The rest after a whole read matters where the
        cache cannot be asked without waiting: there every read reads all of the file again.
        """
        path = self._path
        if path is None or not self._reading.acquire(blocking=False):
            return None
        if self._clock() - self._read_at < WARM_COOL_S:
            self._reading.release()
            return None
        thread = threading.Thread(
            target=self._run, args=(path,), name="jarvis-speech-warm", daemon=True
        )
        try:
            thread.start()
        except RuntimeError:  # no more threads: the request loads the model itself
            self._reading.release()
            return None
        return thread

    def _run(self, path: Path) -> None:
        try:
            seconds = self._read(path)
            if seconds is not None:
                self._read_at = self._clock()
        finally:
            self._reading.release()
        if seconds is not None:
            log.info("read the speech model into memory in %.1f s", seconds)


def voxtype_command(
    wav: Path, engine: str = "", model: str = "", prompt: str = "", settings: Path | None = None
) -> list[str]:
    """Return the `voxtype transcribe` command; options override the owner's config for one run.

    `settings` is a config file of Jarvis's own (see `voxtype_settings`); the owner's Voxtype
    config is never changed. `-q` (errors only) keeps Voxtype's log lines off the transcript.
    """
    cmd = ["voxtype", "-q"]
    if settings is not None:
        cmd += ["-c", str(settings)]
    # The `--option=value` form keeps a value that starts with a dash (a song name in the hint)
    # from being read as another option.
    if model and not model_in_settings(engine, model):
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


LOCK_QUERY = ("omarchy-shell", "lock", "isLocked")  # the Omarchy shell owns the lock screen


def lock_state(
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run, timeout_s: float = 2.0
) -> bool | None:
    """Whether the Omarchy lock screen is up, asked of the running Omarchy shell; None if unknown.

    (The shell does not tell logind, so `LockedHint` stays false while locked.)
    """
    try:
        done = run(list(LOCK_QUERY), capture_output=True, timeout=timeout_s, check=False)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        log.warning("could not ask whether the screen is locked: %s", exc)
        return None
    answer = done.stdout.decode(errors="replace").strip()
    if done.returncode == 0 and answer in {"true", "false"}:
        return answer == "true"
    log.warning("could not read whether the screen is locked: %r", answer)
    return None


def screen_locked(
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run, timeout_s: float = 2.0
) -> bool:
    """Whether to treat the screen as locked: an answer that cannot be had counts as locked.

    Asked only when `audio.listen_when_locked` is off, at each wake word, so a fault keeps
    Jarvis deaf at a locked screen rather than listening.
    """
    return lock_state(run, timeout_s) is not False


def installed_models(
    engine: str,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    timeout_s: float = 10.0,
) -> frozenset[str] | None:
    """Return the models Voxtype reports installed for `engine`; None when it cannot say.

    Read from `voxtype info models --json --engine <engine>`, which checks each installed model's
    size and header (#34, decision 1). An older Voxtype without the command says nothing, and
    `jarvis doctor` then looks for the file itself.
    """
    try:
        done = run(
            ["voxtype", "info", "models", "--json", "--engine", engine],
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        log.warning("could not ask Voxtype for its models: %s", exc)
        return None
    if done.returncode != 0:
        return None
    try:
        listed = obj(obj(obj(json.loads(done.stdout)).get("engines")).get(engine)).get("models")
    except ValueError:
        return None
    if not isinstance(listed, list):
        return None
    return frozenset(
        str(model["name"])
        for model in objs(cast("object", listed))
        if model.get("installed") is True and isinstance(model.get("name"), str)
    )


@dataclass(frozen=True, slots=True)
class VoxtypeState:
    """What Voxtype's status says about the owner's own use of it."""

    dictating: bool  # dictation is recording or transcribing
    meeting: bool  # a `voxtype meeting` is running


# One long-lived reader (it costs nothing while Voxtype's state is the same) in place of a call
# every time the loop asks; `--extended` adds the meeting fields.
VOXTYPE_FOLLOW = ("voxtype", "status", "--follow", "--format", "json", "--extended")
VOXTYPE_RETRY_S = 1.0  # the wait before the status reader is started again, doubled each failure
VOXTYPE_RETRY_MAX_S = 30.0
VOXTYPE_STOP_S = 2.0  # how long the reader gets to end before it is killed


def parse_voxtype_status(line: bytes) -> VoxtypeState | None:
    """Read one JSON line of `voxtype status`; None when it cannot be read.

    Any state but idle and stopped counts as dictation, one added later included. A meeting is
    any `meeting_status` but idle, or `meeting_active`; a status without those fields has none.
    """
    try:
        fields = obj(json.loads(line))
    except ValueError:
        return None
    state = fields.get("class")
    if not isinstance(state, str):
        return None
    meeting = fields.get("meeting_status")
    return VoxtypeState(
        dictating=state not in _NOT_DICTATING,
        meeting=fields.get("meeting_active") is True
        or (isinstance(meeting, str) and meeting != "idle"),
    )


def next_retry_wait(wait: float, *, ran_s: float, first: float, longest: float) -> float:
    """Return how long to wait after a run of the status command that waited `wait` before it.

    A run that lasted `longest` or more was healthy and starts over at `first`; a shorter one
    (one that prints a line and dies, say) doubles the wait, up to `longest`.
    """
    return first if ran_s >= longest else min(wait * 2, longest)


def _end_process(proc: subprocess.Popen[bytes], wait_s: float) -> None:
    """Ask `proc` to end, kill it after `wait_s`, and reap it."""
    proc.terminate()
    try:
        proc.wait(wait_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class VoxtypeWatch:
    """Follows `voxtype status` in the background, so asking is a read of what it last said.

    The state comes from `voxtype status`, a command AGENTS.md allows, read-only: Jarvis
    never starts or stops anything with it. A daemon that is not running is reported by the
    command itself (`stopped`: not dictating). A status that is not known yet, cannot be read, or
    cannot be had (the command will not start, or ends: an older Voxtype without these options,
    or a crash) is `None`. The command is started again after it ends.
    """

    def __init__(
        self,
        command: Sequence[str] = VOXTYPE_FOLLOW,
        *,
        retry_s: float = VOXTYPE_RETRY_S,
        max_retry_s: float = VOXTYPE_RETRY_MAX_S,
    ) -> None:
        """Follow the status that `command` prints; the reader starts with `start`."""
        super().__init__()
        self._command = list(command)
        self._retry_s = retry_s
        self._max_retry_s = max_retry_s
        self._state: VoxtypeState | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._complained = False

    @property
    def pid(self) -> int | None:
        """The status command's process ID while it runs."""
        with self._lock:
            return self._proc.pid if self._proc is not None else None

    def start(self) -> Self:
        """Start following the status on a thread of its own, which never blocks the caller."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jarvis-voxtype-status", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop following and end the status command this started."""
        self._stop.set()
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None:
            _end_process(proc, VOXTYPE_STOP_S)
        if self._thread is not None:
            self._thread.join(VOXTYPE_STOP_S)

    def dictating(self) -> bool | None:
        """Whether the owner's dictation is recording or transcribing; None when not known."""
        state = self._state
        return None if state is None else state.dictating

    def meeting(self) -> bool:
        """Whether a Voxtype meeting is running; one that cannot be seen counts as none."""
        state = self._state
        return state is not None and state.meeting

    def _run(self) -> None:
        wait = self._retry_s / 2  # so the first failure waits `retry_s`
        while not self._stop.is_set():
            started = time.monotonic()
            self._follow()
            ran_s = time.monotonic() - started
            wait = next_retry_wait(
                wait, ran_s=ran_s, first=self._retry_s, longest=self._max_retry_s
            )
            self._stop.wait(wait)

    def _follow(self) -> None:
        """Run the status command once, keeping the state it prints."""
        try:
            proc = subprocess.Popen(
                self._command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except OSError as exc:
            self._unreadable(f"cannot run {self._command[0]}: {exc.strerror or exc}")
            return
        with self._lock:
            if self._stop.is_set():
                proc.terminate()
            self._proc = proc
        try:
            with proc:  # closes the pipe and waits for the command to end
                try:
                    self._read_status(proc)
                except BaseException:
                    # The command still runs, and the wait that closes this block would be for it.
                    _end_process(proc, VOXTYPE_STOP_S)
                    raise
        finally:
            with self._lock:
                self._proc = None
            if not self._stop.is_set():
                self._state = None  # nothing follows the status until the command runs again

    def _read_status(self, proc: subprocess.Popen[bytes]) -> None:
        for line in proc.stdout or ():
            state = parse_voxtype_status(line)
            if state is None:
                self._unreadable(f"unreadable status {line[:200]!r}")
                continue
            self._complained = False
            self._state = state

    def _unreadable(self, why: str) -> None:
        self._state = None
        if not self._complained:
            log.warning("could not read Voxtype's status: %s", why)
            self._complained = True


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
    data_dir: Path, voice: str, speed: float, *, phoneme_map: bool = False
) -> Callable[[str], tuple[bytes, int]]:
    """Return text -> (int16 PCM bytes, sample rate) for the configured voice.

    `phoneme_map` is for a Kokoro voice only (`audio.kokoro_phoneme_map`).
    """
    files = voice_files(data_dir, voice)
    if voice.startswith(KOKORO_PREFIX):
        from jarvis.kokoro import SAMPLE_RATE, Kokoro

        kokoro = Kokoro(
            files[0],
            files[1],
            voice.removeprefix(KOKORO_PREFIX),
            speed=speed,
            phoneme_map=phoneme_map,
        )
        return lambda text: (kokoro.synthesize(text).tobytes(), SAMPLE_RATE)
    from piper import PiperVoice, SynthesisConfig

    piper = PiperVoice.load(files[0])
    # Piper's length scale is the time per sound, so a faster speed is a shorter one.
    settings = SynthesisConfig(length_scale=piper.config.length_scale / speed)

    def run(text: str) -> tuple[bytes, int]:
        chunks = list(piper.synthesize(text, settings))
        rate = chunks[0].sample_rate if chunks else 22050
        return b"".join(chunk.audio_int16_bytes for chunk in chunks), rate

    return run


def run_singer(cmd: list[str], timeout_s: float) -> tuple[int, str]:
    """Run a singing program (`jarvis.sing.soulx`) to its end; return its exit code and stderr.

    It reads nothing and makes no sound itself: it writes a file the speaker then plays.

    Raises:
        AudioError: it could not start, or ran past `timeout_s` and was killed.
    """
    try:
        done = subprocess.run(
            cmd, capture_output=True, timeout=timeout_s, check=False, stdin=subprocess.DEVNULL
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"{Path(cmd[0]).name} ran past {timeout_s:g} s and was stopped"
        raise AudioError(msg) from exc
    except OSError as exc:
        msg = f"could not start {cmd[0]}: {exc}"
        raise AudioError(msg) from exc
    return done.returncode, done.stderr.decode("utf-8", errors="replace")


def pw_play_command(rate: int) -> list[str]:
    """Return the `pw-play` command for raw mono 16-bit PCM on standard input."""
    return ["pw-play", "--raw", "--rate", str(rate), "--channels", "1", "--format", "s16", "-"]


PLAY_GRACE_S = 10.0  # how long playback may run past the sound's own length before it is ended
LINGER_S = 0.3  # how long an open player waits for the next sentence before it is closed
WRITE_STEP = 4096  # bytes handed to the player at a time (one pipe page; about 0.085 s at 24 kHz)
QUIET = 300  # samples at or below this level count as silence at the ends of a sentence
LEAD_KEEP_S = 0.03  # silence kept before the first word Jarvis says
TAIL_KEEP_S = 0.2  # silence kept after each sentence (the voice pads about 0.35 s)
# How long after a wake-like sentence ends the wake word stays ignored: the player's buffer and
# the wake model's own lag, so the end of the sentence heard late does not wake Jarvis.
GUARD_TAIL_S = 1.0
# A song is checked for its own wake word in chunks, so a hit guards only its own stretch; each
# chunk is checked together with the second before it, so a phrase across two is still heard whole.
SONG_CHUNK_S = 2.5
SONG_OVERLAP_S = 1.0


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


RESAMPLE_ZERO_CROSSINGS = 32  # sinc lobes each side of a sample: a transition band about 1 kHz wide
RESAMPLE_BETA = 8.6  # Kaiser window shape: stopband about 80 dB down
RESAMPLE_BLOCK = 8000  # output samples worked on at a time


@functools.lru_cache(maxsize=4)
def _resampling_filter(rate: int) -> tuple[int, int, int, NDArray[np.float32]]:
    """Return (up, down, half, weights) to resample `rate` to RATE, a windowed-sinc low-pass.

    Output sample n sits at input time n * down / up. `weights[r]` holds the `2 * half` taps for
    the outputs whose position is r / up past an input sample, each row summing to 1.
    """
    divisor = math.gcd(RATE, rate)
    up, down = RATE // divisor, rate // divisor
    cutoff = 0.45 * min(1.0, RATE / rate)  # cycles per input sample: 90% of the lower Nyquist rate
    half = math.ceil(RESAMPLE_ZERO_CROSSINGS / (2 * cutoff))
    t = np.arange(1 - half, half + 1)[None, :] - (np.arange(up)[:, None] / up)
    window = np.i0(RESAMPLE_BETA * np.sqrt(np.clip(1 - (t / half) ** 2, 0, None)))
    weights = np.sinc(2 * cutoff * t) * window / np.i0(RESAMPLE_BETA)
    weights /= weights.sum(axis=1, keepdims=True)
    return up, down, half, weights.astype(np.float32)


def to_16k(pcm: bytes, rate: int) -> NDArray[np.int16]:
    """Return int16 `pcm` at `rate` resampled to 16 kHz with a low-pass first.

    Whatever the voice puts above 8 kHz is filtered out (a plain interpolation would fold it into
    the band the wake model listens to), so the check hears what the microphone would.
    """
    samples = np.frombuffer(pcm, dtype=np.int16)
    if rate == RATE or samples.size == 0:
        return samples.copy()
    up, down, half, weights = _resampling_filter(rate)
    count = samples.size * up // down
    padded = np.pad(samples.astype(np.float32), (half, half + 1))
    taps = np.arange(2 * half)
    out = np.empty(count, dtype=np.float32)
    for first in range(0, count, RESAMPLE_BLOCK):
        n = np.arange(first, min(first + RESAMPLE_BLOCK, count)) * down
        window = padded[(n // up)[:, None] + taps[None, :] + 1]
        out[first : first + n.size] = np.einsum("nk,nk->n", window, weights[n % up])
    return np.clip(np.rint(out), -32768, 32767).astype(np.int16)


class _WakeModel(Protocol):
    def heard(self, frame: NDArray[np.int16]) -> bool: ...
    def reset(self) -> None: ...


@runtime_checkable
class OverSpeech(Protocol):
    """A wake word held to a lower score while Jarvis speaks (`audio.barge_in_threshold`)."""

    def heard_over_speech(self, frame: NDArray[np.int16]) -> bool:
        """Feed one frame heard while Jarvis speaks; True when a wake phrase was detected."""
        ...

    def closest_miss(self) -> str | None:
        """Return, once, the best score over Jarvis's speech that fell short, if one was kept."""
        ...


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
        # The score a barge-in needs, since that is what the sentence could trigger.
        wake = self._wake
        heard = wake.heard_over_speech if isinstance(wake, OverSpeech) else wake.heard
        wake.reset()
        try:
            return any(
                heard(samples[i : i + FRAME]) for i in range(0, samples.size - FRAME + 1, FRAME)
            )
        finally:
            if isinstance(wake, OverSpeech):
                wake.closest_miss()  # its own sentences are no miss of the owner's
            wake.reset()


class Speaker:
    """Speaks through `pw-play` with a synthesizer from `make_synthesizer`.

    Sentences are queued and spoken in order by one worker thread, each synthesized while the
    one before it plays, through one player for as long as sentences keep coming. A sound made
    elsewhere (a song) takes its place in the same queue (`queue_sound`) and plays the same way.
    `hush` drops what has not been heard yet, so the owner can speak over Jarvis.
    """

    def __init__(  # ruff: ignore[too-many-arguments] - the player's settings, keyword-only
        self,
        synthesize: Callable[[str], tuple[bytes, int]],
        *,
        player: Callable[[int], list[str]] = pw_play_command,
        grace_s: float = PLAY_GRACE_S,
        own_wake: Callable[[bytes, int, str], bool] | None = None,
        linger_s: float = LINGER_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Use `synthesize` for text to speech; `player` builds the playback command for a rate.

        `own_wake` tells whether a synthesized sentence (its sound, its rate and its text) would
        trigger the wake word itself. `clock` times when each sentence plays (tests).
        """
        super().__init__()
        self._synthesize = synthesize
        self._player = player
        self._grace_s = grace_s
        self._own_wake = own_wake
        self._linger_s = linger_s
        self._clock = clock
        self._lock = threading.Condition()
        self._items: queue.Queue[tuple[int, str, tuple[bytes, int] | None]] = queue.Queue()
        self._pending = 0  # sentences queued, being synthesized or being handed to the player
        self._run = 0  # raised by `hush`: sentences queued before it are dropped
        self._error: AudioError | None = None
        # When each wake-like sentence plays (start, end plus GUARD_TAIL_S), on `_clock`.
        self._guards: list[tuple[float, float]] = []
        self._ends = 0.0  # when the sound handed to the open player will have played
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
        """False while a sentence that would trigger the wake word by itself is playing."""
        with self._lock:
            now = self._clock()
            self._guards = [guard for guard in self._guards if guard[1] > now]
            return not any(start <= now for start, _ in self._guards)

    def say(self, text: str) -> None:
        """Speak `text` after anything already queued; return once it has all played.

        A player that cannot be started or does not finish is an `AudioError`.
        """
        self.queue(text)
        self.wait()

    def queue(self, text: str) -> None:
        """Have `text` spoken after what is already queued, and return at once."""
        if not text.strip():
            return
        self._put(text, None)

    def queue_sound(self, pcm: bytes, rate: int, words: str = "") -> None:
        """Have int16 mono `pcm` at `rate` played after what is already queued; return at once.

        `words` are what it says, for the check that it would not trigger the wake word itself.
        """
        if pcm:
            self._put(words, (pcm, rate))

    def _put(self, text: str, sound: tuple[bytes, int] | None) -> None:
        if self._stopped.is_set():
            return
        with self._lock:
            self._pending += 1
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._work, name="jarvis-speech", daemon=True
                )
                self._worker.start()
            self._items.put((self._run, text, sound))

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
            self._guards.clear()  # what held the wake phrase is no longer playing
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
                run, text, sound = self._items.get(timeout=self._linger_s if self._proc else None)
            except queue.Empty:
                self._close()
                continue
            try:
                self._speak(run, text, sound)
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

    def _speak(self, run: int, text: str, sound: tuple[bytes, int] | None = None) -> None:
        if not self._current(run):
            return
        pcm, rate = sound if sound is not None else self._synthesize(text)
        if not self._current(run):
            return
        if self._proc is not None and (rate != self._rate or self._player_run != run):
            self._close()  # another rate, or a player a hush has ended
        first = self._proc is None
        pcm = trim_silence(pcm, rate, lead_s=LEAD_KEEP_S if first else None, tail_s=TAIL_KEEP_S)
        if not pcm:
            return
        spans = self._wake_spans(pcm, rate, text, run, chunked=sound is not None)
        with self._lock:
            if run != self._run:  # hushed during the check: it will not be played
                return
            # It plays once what the open player already holds has played; a new player starts
            # it now. Kept before the player opens, so the guard is never late.
            start = max(self._clock(), self._ends if self._proc is not None else 0.0)
            self._ends = start + len(pcm) / (2 * rate)
            self._guards.extend((start + a, start + b + GUARD_TAIL_S) for a, b in spans)
        if spans:
            log.info("a sentence sounds like the wake word; it cannot be spoken over")
        proc = self._proc or self._open(rate, run)
        if proc is not None:
            self._write(proc, pcm, run)

    def _wake_spans(
        self, pcm: bytes, rate: int, text: str, run: int, *, chunked: bool
    ) -> list[tuple[float, float]]:
        """Return the seconds of `pcm` (start, end) that would trigger the wake word.

        A sentence is checked whole; a song (`chunked`) in chunks of `SONG_CHUNK_S`.
        """
        total = len(pcm) // 2
        step = int(SONG_CHUNK_S * rate) if chunked else total
        spans: list[tuple[float, float]] = []
        for first in range(0, total, step):
            if not self._current(run):
                break
            end = min(total, first + step)
            back = max(0, first - int(SONG_OVERLAP_S * rate)) if chunked else first
            if self._sounds_like_wake(pcm[2 * back : 2 * end], rate, text):
                spans.append((back / rate, end / rate))
        return spans

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
            self._proc = None  # a guard still running out keeps its end: the tail may be heard
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
