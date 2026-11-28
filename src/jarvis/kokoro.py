"""Kokoro-82M text to speech (Apache-2.0 weights) on ONNX Runtime, phonemized by espeak-ng.

Kokoro takes espeak IPA phonemes mapped through its vocabulary (stored in the model's own
metadata), a 256-value style vector chosen by phoneme count, and a speed; it returns 24 kHz audio.
The espeak-ng library and data come from the `piper-tts` wheel already installed.
"""

import functools
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import onnxruntime as ort

from jarvis import espeak, voices
from jarvis.phonemes import trained_phonemes

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

SAMPLE_RATE = 24000
FRAME_SAMPLES = 600  # the audio one step of the model's `duration` output lasts (25 ms)
MAX_PHONEMES = 510
MAX_THREADS = 8
TOKEN_INPUTS = ("input_ids", "tokens")  # what the v1.0 and the older model builds call the phonemes
_LANG_SWITCH = re.compile(r"\([^)]+\)")


def _espeak_phonemes(text: str, voice: str) -> list[str]:
    sentences: list[str] = []
    current = ""
    for phonemes, terminator, end_of_sentence in espeak.clauses(text, voice):
        current += _LANG_SWITCH.sub("", phonemes) + terminator
        if terminator in {",", ":", ";"}:
            current += " "
        if end_of_sentence:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return [unicodedata.normalize("NFC", s) for s in sentences if s]


def _chunks(phonemes: str) -> list[str]:
    """Split `phonemes` into pieces of at most MAX_PHONEMES, at a space where there is one."""
    out: list[str] = []
    rest = phonemes
    while len(rest) > MAX_PHONEMES:
        cut = rest.rfind(" ", 1, MAX_PHONEMES + 1)
        if cut < 0:
            cut = MAX_PHONEMES  # one word longer than the window
        out.append(rest[:cut])
        rest = rest[cut:].lstrip()
    out.append(rest)
    return [chunk for chunk in out if chunk.strip()]


def session_options() -> ort.SessionOptions:
    """Return ONNX Runtime options for the voice model: bounded threads that never spin-wait.

    Measured on a 16-thread CPU: a short reply takes 0.1 s and a long one 0.5 s with 8 threads,
    the same as with one thread per core, so more threads only take cores from the language model.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(MAX_THREADS, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


@functools.cache
def _session(model_path: str) -> ort.InferenceSession:
    """Return the ONNX Runtime session of a model file, loaded once per process.

    The speaking voice and the singing one share it, so the model is in memory once; a session
    may run from several threads at once.
    """
    return ort.InferenceSession(
        model_path, sess_options=session_options(), providers=["CPUExecutionProvider"]
    )


@dataclass(frozen=True, slots=True)
class Timed:
    """One chunk of speech with how long each of its phoneme symbols lasts.

    `frames[i]` is how many `FRAME_SAMPLES` steps `symbols[i]` lasts in `audio`, which begins
    with `lead` samples of the model's own start padding.
    """

    audio: NDArray[np.float32]
    symbols: tuple[str, ...]
    frames: NDArray[np.int64]
    lead: int


class Kokoro:
    """One Kokoro model and voice pack; `synthesize` returns int16 PCM at 24 kHz."""

    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        voice: str,
        *,
        speed: float = 1.0,
        phoneme_map: bool = False,
    ) -> None:
        """Load the model and pick a voice (e.g. `bm_george`).

        With `phoneme_map`, espeak's phonemes are rewritten into the symbols the model was trained
        on (`trained_phonemes`) before they are looked up; off, they are looked up as espeak gives
        them.

        Raises:
            ValueError: the file is not a Kokoro model, or the voice pack has no voice of that
                name.
        """
        super().__init__()
        self._session = _session(str(model_path))
        inputs = {item.name: str(item.type) for item in self._session.get_inputs()}
        names = [name for name in TOKEN_INPUTS if name in inputs]
        if not names or not {"style", "speed"} <= inputs.keys():
            msg = (
                f"{model_path} is not a Kokoro model Jarvis can run: it takes {sorted(inputs)}, "
                f"not one of {TOKEN_INPUTS} with style and speed"
            )
            raise ValueError(msg)
        self._tokens_input = names[0]
        self._speed_type = np.int32 if inputs["speed"] == "tensor(int32)" else np.float32
        meta = self._session.get_modelmeta().custom_metadata_map
        self._vocab = cast("dict[str, int]", json.loads(meta["kokoro_config"])["vocab"])
        with np.load(voices_path) as pack:
            if voice not in pack.files:
                msg = f"unknown Kokoro voice {voice!r}; available: {', '.join(sorted(pack.files))}"
                raise ValueError(msg)
            self._style = np.asarray(pack[voice], dtype=np.float32)
        self._gain = voices.gain(voice)  # each listed voice to the same speech level
        self._british = voice.startswith("b")
        self._espeak_voice = "en-gb-x-rp" if self._british else "en-us"
        self._speed = speed
        self._phoneme_map = phoneme_map

    def tokens(self, text: str) -> list[list[int]]:
        """Return the token ids of each sentence, split to fit the model's window."""
        out: list[list[int]] = []
        for sentence in _espeak_phonemes(text, self._espeak_voice):
            spoken = (
                trained_phonemes(sentence, british=self._british) if self._phoneme_map else sentence
            )
            known = "".join(ch for ch in spoken if ch in self._vocab)
            out.extend([self._vocab[ch] for ch in chunk] for chunk in _chunks(known))
        return [ids for ids in out if ids]

    def _feed(self, ids: list[int], speed: float) -> dict[str, Any]:
        style = self._style[min(len(ids), len(self._style)) - 1]
        return {
            self._tokens_input: np.array([[0, *ids, 0]], dtype=np.int64),
            "style": style.reshape(1, -1),
            "speed": np.array([speed], dtype=self._speed_type),
        }

    def timed(self, text: str, *, speed: float = 1.0) -> list[Timed]:
        """Return each chunk of `text` spoken at `speed`, with its symbols and their lengths.

        Raises:
            ValueError: the model gives no duration per symbol (an older build).
        """
        symbols = {index: symbol for symbol, index in self._vocab.items()}
        out: list[Timed] = []
        for ids in self.tokens(text):
            outputs = self._session.run(None, self._feed(ids, speed))
            if len(outputs) < 2:  # ruff: ignore[magic-value-comparison] - audio, then durations
                msg = "this Kokoro model gives no duration per phoneme"
                raise ValueError(msg)
            audio = np.asarray(outputs[0], dtype=np.float32).reshape(-1) * self._gain
            steps = np.asarray(outputs[1], dtype=np.int64).reshape(-1)
            if steps.size != len(ids) + 2:  # the two pads around the ids
                msg = f"Kokoro gave {steps.size} durations for {len(ids) + 2} tokens"
                raise ValueError(msg)
            out.append(
                Timed(
                    audio,
                    tuple(symbols[i] for i in ids),
                    steps[1:-1].copy(),
                    int(steps[0]) * FRAME_SAMPLES,
                )
            )
        return out

    def synthesize(self, text: str) -> NDArray[np.int16]:
        """Return the whole utterance as int16 PCM at SAMPLE_RATE."""
        parts: list[NDArray[np.float32]] = []
        for ids in self.tokens(text):
            audio = self._session.run(None, self._feed(ids, self._speed))[0].reshape(-1)
            parts.extend((audio, np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32)))
        if not parts:
            return np.zeros(0, dtype=np.int16)
        audio = np.clip(np.concatenate(parts) * self._gain, -1.0, 1.0)
        return (audio * 32767).astype(np.int16)
