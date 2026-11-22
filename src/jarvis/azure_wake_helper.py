# SPDX-FileCopyrightText: NecturaLabs and the Jarvis contributors
# SPDX-License-Identifier: MIT
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false
"""Listen for an Azure custom keyword in raw audio: the helper behind `audio.wake_engine = "azure"`.

Jarvis never imports this file. It runs as its own program, with the Python of a separate
environment that holds Microsoft's Speech SDK, inside a namespace with no network
(`jarvis.audio.azure_wake_command`), so the SDK cannot reach anything over the network. It is under
the MIT licence, not Jarvis's GPL, so it may be combined with that closed-source SDK; Jarvis only
talks to it through pipes.

Standard input: raw 16 kHz mono signed 16-bit audio. Standard output: the line `ready` once the
model has been tried on a moment of silence, then `wake <sample>` each time the keyword is heard,
where <sample> counts the input's samples up to the keyword's end. It ends when its input does.
The keyword engine wants silence before a keyword, so the helper puts 1.5 s of it into the
engine's stream before the first sample and again after each keyword; the samples it reports leave
that silence out, so they stay positions in the input.
A model the SDK cannot use is said on standard error, with exit code 1. With `--check` it only
tries the model and exits.
"""

import sys
import threading
from typing import TYPE_CHECKING, BinaryIO, TextIO

if TYPE_CHECKING:
    from types import ModuleType

RATE = 16000
CHUNK = 2560  # 80 ms: Jarvis writes frames of this size
PROBE_BYTES = RATE  # half a second of silence, enough for the SDK to load the model
TICKS_PER_S = 10_000_000  # the SDK's offsets and durations are in 100 ns ticks
CHECK_ARGS = 2  # the model file and --check
SILENCE_S = 1.5  # Microsoft's keyword engine asks for at least this much before a keyword
SILENCE_SAMPLES = int(SILENCE_S * RATE)
SILENCE_BYTES = SILENCE_SAMPLES * 2


def _failure(sdk: ModuleType, result: object) -> str | None:
    """Return why a recognition ended in an error, or None when it ended normally."""
    if result.reason != sdk.ResultReason.Canceled:
        return None
    details = result.cancellation_details
    if details.reason != sdk.CancellationReason.Error:
        return None
    lines = str(details.error_details).strip().splitlines()
    return lines[0] if lines else "the Speech SDK failed"


def _fmt(sdk: ModuleType) -> object:
    return sdk.audio.AudioStreamFormat(samples_per_second=RATE, bits_per_sample=16, channels=1)


class PushedStream:
    """The engine's push stream, fed by the input and with silence put in after each keyword.

    It counts the samples written and where each silence went, so that a position the engine
    reports in its stream can be told as a position in the input.
    """

    def __init__(self, stream: object) -> None:
        """Wrap the SDK's push stream."""
        super().__init__()
        self._stream = stream
        self._lock = threading.Lock()
        self._samples = 0
        self._silences: list[int] = []  # the stream position of each silence
        self._closed = False

    def write(self, chunk: bytes) -> None:
        """Write audio from the input."""
        with self._lock:
            self._stream.write(chunk)
            self._samples += len(chunk) // 2

    def silence(self) -> None:
        """Write SILENCE_S of silence, unless the stream is closed."""
        with self._lock:
            if self._closed:
                return
            self._silences.append(self._samples)
            self._stream.write(b"\0" * SILENCE_BYTES)
            self._samples += SILENCE_SAMPLES

    def close(self) -> None:
        """End the stream: the input has ended."""
        with self._lock:
            self._closed = True
            self._stream.close()

    def input_position(self, end: int) -> int:
        """Return the stream position `end` as a position in the input, leaving out the silence."""
        with self._lock:
            return end - sum(min(SILENCE_SAMPLES, end - at) for at in self._silences if at < end)


def probe(sdk: ModuleType, model: object) -> str | None:
    """Try `model` on half a second of silence; return why it cannot be used, or None."""
    stream = sdk.audio.PushAudioInputStream(stream_format=_fmt(sdk))
    recognizer = sdk.KeywordRecognizer(sdk.audio.AudioConfig(stream=stream))
    future = recognizer.recognize_once_async(model)
    stream.write(b"\0" * PROBE_BYTES)
    stream.close()
    return _failure(sdk, future.get())


def listen(sdk: ModuleType, model_file: str, source: BinaryIO, out: TextIO, err: TextIO) -> int:
    """Feed `source` to the keyword model in `model_file`; write a line to `out` per keyword."""
    model = sdk.KeywordRecognitionModel(model_file)
    why = probe(sdk, model)
    if why is not None:
        err.write(f"the keyword model cannot be used: {why}\n")
        return 1
    pushed = sdk.audio.PushAudioInputStream(stream_format=_fmt(sdk))
    recognizer = sdk.KeywordRecognizer(sdk.audio.AudioConfig(stream=pushed))
    stream = PushedStream(pushed)
    stream.silence()  # before the first sample: Jarvis writes nothing until it has read `ready`

    def pump() -> None:
        while chunk := source.read(CHUNK):
            stream.write(chunk)
        stream.close()

    threading.Thread(target=pump, name="azure-wake-input", daemon=True).start()
    out.write("ready\n")
    out.flush()
    while True:
        result = recognizer.recognize_once_async(model).get()
        if result.reason == sdk.ResultReason.RecognizedKeyword:
            end = stream.input_position((result.offset + result.duration) * RATE // TICKS_PER_S)
            out.write(f"wake {end}\n")
            out.flush()
            stream.silence()  # the next keyword is to be heard after silence too
            continue
        why = _failure(sdk, result)
        if why is not None:
            err.write(f"the keyword recognizer stopped: {why}\n")
            return 1
        return 0  # the input ended: nothing more to hear


def check(sdk: ModuleType, model_file: str, out: TextIO, err: TextIO) -> int:
    """Only try the model (`jarvis doctor`); return 0 when it can be used."""
    why = probe(sdk, sdk.KeywordRecognitionModel(model_file))
    if why is not None:
        err.write(f"the keyword model cannot be used: {why}\n")
        return 1
    out.write("ok\n")
    return 0


def main() -> int:
    """Run on standard input and output with the model file named on the command line."""
    args = sys.argv[1:]
    if not args or len(args) > CHECK_ARGS or args[1:] not in ([], ["--check"]):
        sys.stderr.write("usage: azure_wake_helper.py <keyword model .table> [--check]\n")
        return 2
    # The one import of the Speech SDK in the tree (ADR 0010); only the helper's own Python has it.
    import azure.cognitiveservices.speech as sdk  # ruff: ignore[import-outside-top-level, banned-api]

    if args[1:]:
        return check(sdk, args[0], sys.stdout, sys.stderr)
    return listen(sdk, args[0], sys.stdin.buffer, sys.stdout, sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
