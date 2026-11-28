"""Owner-controlled recordings of the PCM sent to Jarvis's player."""

import contextlib
import logging
import os
import time
import wave
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

from jarvis.operations import Operation, OperationError, Result, current_request
from jarvis.policy import OpClass

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

NAME = "voice_record"
RECORD_RATE = 24_000
MAX_RECORD_S = 300
log = logging.getLogger(__name__)


@dataclass(slots=True)
class _Capture:
    parts: list[tuple[int, bytearray]] = field(default_factory=list[tuple[int, bytearray]])
    seconds: float = 0.0


class OutputRecorder:
    """Buffer the player input in memory and save it only when told to finish."""

    def __init__(
        self, folder: Path, *, on_saved: Callable[[Path], None] = lambda _path: None
    ) -> None:
        """Keep WAV files in `folder` and report each completed file to `on_saved`."""
        super().__init__()
        self._folder = folder
        self._on_saved = on_saved
        self._lock = Lock()
        self._capture: _Capture | None = None
        self._saving = False

    def start(self) -> None:
        """Start a new recording without overwriting a current one.

        Raises:
            OperationError: A recording is already active or being saved.
        """
        with self._lock:
            if self._capture is not None or self._saving:
                msg = "There is already a recording. Stop it to save it first."
                raise OperationError(msg)
            self._capture = _Capture()

    def feed(self, pcm: bytes | memoryview, rate: int) -> None:
        """Keep samples accepted by the player, up to the recording's length limit."""
        with self._lock:
            capture = self._capture
            if capture is None:
                return
            room = max(0, int((MAX_RECORD_S - capture.seconds) * rate))
            samples = min(room, len(pcm) // 2)
            if not samples:
                return
            if not capture.parts or capture.parts[-1][0] != rate:
                capture.parts.append((rate, bytearray()))
            capture.parts[-1][1].extend(pcm[: samples * 2])
            capture.seconds += samples / rate

    def finish(self) -> Path | None:
        """Stop and save the current recording, or return None when it holds no audio.

        Raises:
            OperationError: A recording is already being saved.
        """
        with self._lock:
            if self._saving:
                msg = "The recording is already being saved."
                raise OperationError(msg)
            capture, self._capture = self._capture, None
            if capture is None or not capture.parts:
                return None
            self._saving = True
        try:
            path = self._save(capture)
        except Exception:
            with self._lock:
                self._capture = capture
            raise
        finally:
            with self._lock:
                self._saving = False
        try:
            self._on_saved(path)
        except Exception:
            log.exception("the recording was saved but could not be recorded in the audit log")
        return path

    def close(self) -> None:
        """Save any outstanding recording on a graceful service shutdown."""
        try:
            self.finish()
        except OSError, OperationError:
            log.exception("could not save the closing voice recording")

    def _save(self, capture: _Capture) -> Path:
        self._folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        folder = os.open(self._folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(folder)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                msg = "the recordings folder must be owned by you and accessible only to you"
                raise PermissionError(msg)
            name = f"output-{time.time_ns()}.wav"
            partial = f".{name}.part"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
            fd = os.open(partial, flags, 0o600, dir_fd=folder)
            try:
                with os.fdopen(fd, "wb") as handle, wave.open(handle, "wb") as output:
                    _write(output, capture)
                os.link(partial, name, src_dir_fd=folder, dst_dir_fd=folder, follow_symlinks=False)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(partial, dir_fd=folder)
        finally:
            os.close(folder)
        return self._folder / name

    def _run(self, args: dict[str, Any]) -> Result:
        _owner_only(args)
        if args["action"] == "start":
            self.start()
            return Result(
                "Recording Jarvis's own voice and singing. Ask to stop to save the WAV. "
                "It can hold up to five minutes of output."
            )
        try:
            path = self.finish()
        except OSError as exc:
            msg = "Could not save the WAV. The audio is still in memory. Ask to stop again."
            raise OperationError(msg) from exc
        if path is None:
            return Result("There is no recorded output to save.")
        return Result(f"Saved the recording at {path}; link: {path.as_uri()}")

    def operation(self) -> Operation:
        """Return the recording control for the running voice service."""
        return Operation(
            "voice_record",  # NAME, written out for the product-rule scan
            "Start recording Jarvis's own spoken and sung audio when the owner asks. "
            "Start before making the sound, and stop to save the recording as a private WAV file.",
            {
                "type": "object",
                "properties": {"action": {"type": "string", "enum": ["start", "stop"]}},
                "required": ["action"],
                "additionalProperties": False,
            },
            self._run,
            default_class=OpClass.CONFIRM,
            prepare=_owner_only,
            spoken_summary=lambda args: f"{args['action']} recording my output",
        )


def _owner_only(args: dict[str, Any]) -> dict[str, Any]:
    request = current_request()
    if request is None or not request.owner_asked or request.scheduled or request.outside:
        msg = "Only the owner's own request can start or stop voice recordings."
        raise OperationError(msg)
    return args


def _write(output: wave.Wave_write, capture: _Capture) -> None:
    from jarvis.audio import resample_pcm  # ruff: ignore[import-outside-top-level] - avoids loading numpy until audio is saved

    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(RECORD_RATE)
    for rate, pcm in capture.parts:
        if rate == RECORD_RATE:
            output.writeframesraw(pcm)
        else:
            output.writeframesraw(resample_pcm(pcm, rate, RECORD_RATE).astype("<i2").tobytes())
