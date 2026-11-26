"""Which singer sings: made from the config on the first song (`singer_for`)."""

import logging
import os
from typing import TYPE_CHECKING

from jarvis.operations import OperationError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from jarvis.config import Config
    from jarvis.sing.ops import Singer
    from jarvis.sing.psola import PsolaSinger

log = logging.getLogger(__name__)

KOKORO_PREFIX = "kokoro:"
DEFAULT_KOKORO_VOICE = "bm_george"


def kokoro_voice(cfg: Config) -> str:
    """Return the Kokoro voice that sings: the speaking one if it is Kokoro's, else the default."""
    voice = cfg.audio.voice
    return (
        voice.removeprefix(KOKORO_PREFIX)
        if voice.startswith(KOKORO_PREFIX)
        else DEFAULT_KOKORO_VOICE
    )


def speech_singer(cfg: Config) -> PsolaSinger:
    """Return the singer that uses the Kokoro voice (`jarvis.sing.psola`).

    Raises:
        OperationError: the Kokoro files are not installed.
    """
    from jarvis.audio import voice_files  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, only when Jarvis first sings
    from jarvis.kokoro import Kokoro
    from jarvis.sing.psola import PsolaSinger  # ruff: ignore[import-outside-top-level] - loads Praat, as above

    model, voices = voice_files(cfg.data_dir, KOKORO_PREFIX + kokoro_voice(cfg))
    if not model.is_file() or not voices.is_file():
        msg = f"I can't sing: the Kokoro voice is not installed ({model.parent})."
        raise OperationError(msg)
    try:
        voice = Kokoro(model, voices, kokoro_voice(cfg), phoneme_map=cfg.audio.kokoro_phoneme_map)
    except ValueError as exc:
        msg = f"I can't sing: {exc}."
        raise OperationError(msg) from exc
    return PsolaSinger(voice)


def singer_for(cfg: Config) -> Callable[[], Singer]:
    """Return what makes the configured singer (`[sing] engine`), called on the first song.

    "soulx" with its files missing sings with the speaking voice, and says so in the log.
    """

    def make() -> Singer:
        from jarvis.sing import soulx  # ruff: ignore[import-outside-top-level] - only when Jarvis first sings

        speech = speech_singer(cfg)
        found = soulx.files(cfg.data_dir)
        if cfg.sing.engine == "speech":
            return speech
        if not found.installed:
            if cfg.sing.engine == "soulx":
                log.warning(
                    "SoulX-Singer is not installed (%s); singing with the speaking voice",
                    found.model.parent,
                )
            return speech
        return soulx.SoulxSinger(found, speech, device=cfg.sing.device)

    return make


def doctor_checks(cfg: Config) -> tuple[list[tuple[bool, str, str]], list[str]]:
    """Return `jarvis doctor`'s checks of the singer that is set up, and its notes.

    Nothing is checked while `sing` is switched off (`deny`).
    """
    from jarvis.audio import voice_files  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need

    if cfg.boundaries.classes.get("sing") == "deny":
        return [], ['note: singing is off (sing = "deny" in [boundaries.classes])']
    from jarvis.sing import soulx  # ruff: ignore[import-outside-top-level] - as above

    model, voices = voice_files(cfg.data_dir, KOKORO_PREFIX + kokoro_voice(cfg))
    fix = "see docs/models.md (Singing)"
    checks = [
        (model.is_file() and voices.is_file(), f"singing voice kokoro:{kokoro_voice(cfg)}", fix),
    ]
    found = soulx.files(cfg.data_dir)
    notes: list[str] = []
    if cfg.sing.engine == "soulx":
        checks += [(f.is_file(), f"file {f}", fix) for f in (found.binary, found.model)]
    if cfg.sing.engine != "speech" and found.binary.is_file():
        checks += [
            (
                os.access(found.binary, os.X_OK),
                f"executable {found.binary}",
                f"chmod +x {found.binary}",
            ),
            (
                any(found.binary.parent.glob("libggml*")),
                f"libraries beside {found.binary.name}",
                "run tools/get_singer.py again",
            ),
        ]
    elif cfg.sing.engine == "auto" and not found.installed:
        notes.append(
            "note: Jarvis sings with its speaking voice; SoulX-Singer, a singing voice on the GPU, "
            f"is not installed in {found.model.parent} (docs/models.md, Singing)"
        )
    return checks, notes


def singer_files(cfg: Config) -> list[tuple[str, Path]]:
    """Return the (role, file) pairs of the singer `jarvis doctor` hashes against its releases.

    The Kokoro files are listed only when the speaking voice is not Kokoro's (it lists them then).
    """
    from jarvis.audio import voice_files  # ruff: ignore[import-outside-top-level] - loads numpy and ONNX Runtime, which most commands never need
    from jarvis.sing import soulx  # ruff: ignore[import-outside-top-level] - as above

    if cfg.boundaries.classes.get("sing") == "deny":
        return []
    out: list[tuple[str, Path]] = []
    if not cfg.audio.voice.startswith(KOKORO_PREFIX):
        model, voices = voice_files(cfg.data_dir, KOKORO_PREFIX + kokoro_voice(cfg))
        out += [("kokoro-model", model), ("kokoro-voices", voices)]
    if cfg.sing.engine != "speech":
        found = soulx.files(cfg.data_dir)
        out += [("soulx-binary", found.binary), ("soulx-model", found.model)]
    return out
