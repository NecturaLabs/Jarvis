# SPDX-FileCopyrightText: NecturaLabs and the Jarvis contributors
# SPDX-License-Identifier: MIT
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingTypeStubs=false
"""Listen for an Azure custom keyword in raw audio: the helper behind `audio.wake_engine = "azure"`.

Jarvis never imports this file. It runs as its own program, with the Python of a separate
environment that holds Microsoft's Speech SDK, inside a namespace with no network
(`jarvis.audio.azure_wake_command`), so the SDK cannot send anything off the PC. It is under the MIT
licence, not Jarvis's GPL, so it may be combined with that closed-source SDK; Jarvis only talks to
it through pipes.

Standard input: raw 16 kHz mono signed 16-bit audio. Standard output: the line `ready` once the
model is loaded, then `wake` each time the keyword is heard. It ends when its input does.
"""

import sys
import threading
from typing import TYPE_CHECKING, BinaryIO, TextIO

if TYPE_CHECKING:
    from types import ModuleType

RATE = 16000
CHUNK = 2560  # 80 ms: Jarvis writes frames of this size


def listen(sdk: ModuleType, model_file: str, source: BinaryIO, out: TextIO) -> int:
    """Feed `source` to the keyword model in `model_file`; write a line to `out` per keyword."""
    model = sdk.KeywordRecognitionModel(model_file)
    fmt = sdk.audio.AudioStreamFormat(samples_per_second=RATE, bits_per_sample=16, channels=1)
    stream = sdk.audio.PushAudioInputStream(stream_format=fmt)
    recognizer = sdk.KeywordRecognizer(sdk.audio.AudioConfig(stream=stream))

    def pump() -> None:
        while chunk := source.read(CHUNK):
            stream.write(chunk)
        stream.close()

    threading.Thread(target=pump, name="azure-wake-input", daemon=True).start()
    out.write("ready\n")
    out.flush()
    while True:
        result = recognizer.recognize_once_async(model).get()
        if result.reason != sdk.ResultReason.RecognizedKeyword:
            return 0  # the input ended (or the recognizer stopped): nothing more to hear
        out.write("wake\n")
        out.flush()


def main() -> int:
    """Run on standard input and output with the model file named on the command line."""
    if len(sys.argv) != 2:  # ruff: ignore[magic-value-comparison] - the program and the model file
        sys.stderr.write("usage: azure_wake_helper.py <keyword model .table>\n")
        return 2
    import azure.cognitiveservices.speech as sdk  # ruff: ignore[import-outside-top-level] - only the helper's own Python has it

    return listen(sdk, sys.argv[1], sys.stdin.buffer, sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
