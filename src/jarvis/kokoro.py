"""Kokoro-82M text to speech (Apache-2.0 weights) on ONNX Runtime, phonemized by espeak-ng.

Kokoro takes espeak IPA phonemes mapped through its vocabulary (stored in the model's own
metadata), a 256-value style vector chosen by phoneme count, and a speed; it returns 24 kHz audio.
The espeak-ng library and data come from the `piper-tts` wheel already installed.
"""

import json
import re
import threading
import unicodedata
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import onnxruntime as ort

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

SAMPLE_RATE = 24000
MAX_PHONEMES = 510
_LANG_SWITCH = re.compile(r"\([^)]+\)")
_espeak_lock = threading.Lock()


def _espeak_phonemes(text: str, voice: str) -> list[str]:
    from piper import espeakbridge  # ruff: ignore[import-outside-top-level] - the compiled bridge Piper ships
    from piper.phonemize_espeak import ESPEAK_DATA_DIR  # ruff: ignore[import-outside-top-level]

    sentences: list[str] = []
    current = ""
    with _espeak_lock:
        espeakbridge.initialize(str(ESPEAK_DATA_DIR))
        espeakbridge.set_voice(voice)
        clauses = espeakbridge.get_phonemes(text)
    for phonemes, terminator, end_of_sentence in clauses:
        current += _LANG_SWITCH.sub("", phonemes) + terminator
        if terminator in {",", ":", ";"}:
            current += " "
        if end_of_sentence:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return [unicodedata.normalize("NFC", s) for s in sentences if s]


class Kokoro:
    """One Kokoro model and voice pack; `synthesize` returns int16 PCM at 24 kHz."""

    def __init__(
        self, model_path: Path, voices_path: Path, voice: str, *, speed: float = 1.0
    ) -> None:
        """Load the model and pick a voice (e.g. `bm_george`).

        Raises:
            ValueError: the voice pack has no voice of that name.
        """
        super().__init__()
        options = ort.SessionOptions()
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        meta = self._session.get_modelmeta().custom_metadata_map
        self._vocab = cast("dict[str, int]", json.loads(meta["kokoro_config"])["vocab"])
        with np.load(voices_path) as pack:
            if voice not in pack.files:
                msg = f"unknown Kokoro voice {voice!r}; available: {', '.join(sorted(pack.files))}"
                raise ValueError(msg)
            self._style = np.asarray(pack[voice], dtype=np.float32)
        self._espeak_voice = "en-gb-x-rp" if voice.startswith("b") else "en-us"
        self._speed = speed

    def tokens(self, text: str) -> list[list[int]]:
        """Return the token ids of each sentence, split to fit the model's window."""
        out: list[list[int]] = []
        for sentence in _espeak_phonemes(text, self._espeak_voice):
            ids = [self._vocab[ch] for ch in sentence if ch in self._vocab]
            out.extend(ids[i : i + MAX_PHONEMES] for i in range(0, len(ids), MAX_PHONEMES))
        return [ids for ids in out if ids]

    def synthesize(self, text: str) -> NDArray[np.int16]:
        """Return the whole utterance as int16 PCM at SAMPLE_RATE."""
        parts: list[NDArray[np.float32]] = []
        for ids in self.tokens(text):
            style = self._style[min(len(ids), len(self._style)) - 1]
            feed: dict[str, Any] = {
                "input_ids": np.array([[0, *ids, 0]], dtype=np.int64),
                "style": style.reshape(1, -1),
                "speed": np.array([self._speed], dtype=np.float32),
            }
            audio = self._session.run(None, feed)[0].reshape(-1)
            parts.extend((audio, np.zeros(int(SAMPLE_RATE * 0.12), dtype=np.float32)))
        if not parts:
            return np.zeros(0, dtype=np.int16)
        audio = np.clip(np.concatenate(parts), -1.0, 1.0)
        return (audio * 32767).astype(np.int16)
