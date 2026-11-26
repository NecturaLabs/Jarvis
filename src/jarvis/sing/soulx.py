"""Singing with SoulX-Singer, a singing-voice model, run by audio.cpp's `audiocpp_cli` on the GPU.

SoulX-Singer sings a score in the voice of a reference clip of singing, given the score of that
clip too. The reference is the same song sung first by the speaking-voice singer
(`jarvis.sing.psola`), whose timing and pitches are known exactly because it made them: no
recording ships with Jarvis, and the song comes out in Jarvis's own voice. Both scores are
written as SoulX-Singer's metadata (ARPAbet words, `jarvis.sing.arpabet`), and one process runs
per song and exits, so nothing holds the GPU or the CPU while Jarvis is idle. When the process
fails or runs too long, or the request has too little time left to start it, the speaking-voice
song made for the reference is sung instead; a song that is stopped ends the process at once.
"""

import hashlib
import json
import logging
import os
import stat
import struct
import tempfile
import threading
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from jarvis import models, paths
from jarvis.audio import AudioError, SingerStoppedError, run_singer
from jarvis.sing import arpabet
from jarvis.sing.budget import SongStoppedError
from jarvis.sing.score import Kind, Score

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from jarvis.sing.budget import Budget
    from jarvis.sing.psola import PsolaSinger

log = logging.getLogger(__name__)

FOLDER = "sing"  # under data_dir
BINARY = "audiocpp_cli"  # in FOLDER/audiocpp, beside the libraries it loads
MODEL = "soulx-singer-q8_0.gguf"
RATE = 24000
EDGE_S = 0.2  # silence before and after the song in both clips, a rest of SoulX's own
TIMEOUT_S = 20.0  # a song of at most 30 s takes about 3 s on the reference GPU
MIN_GPU_S = 5.0  # the least of the request's time left that SoulX-Singer is started with
SEED = 2026
MAX_ERROR_CHARS = 400
PHONES_CACHED = 1024  # words whose phones are kept; a song has at most 64
MIN_RATE, MAX_RATE = 8000, 192000
LENGTH_SLACK_S = 0.25  # how far a sung song may be from its metadata's length (it is on it)


@dataclass(frozen=True, slots=True)
class Files:
    """Where SoulX-Singer's runner and model live."""

    binary: Path
    model: Path

    @property
    def installed(self) -> bool:
        """Whether both are there (`verify` says whether they may run)."""
        return self.binary.is_file() and self.model.is_file()


def files(data_dir: Path) -> Files:
    """Return where SoulX-Singer's files go under `data_dir`."""
    folder = data_dir / FOLDER
    return Files(folder / "audiocpp" / BINARY, folder / MODEL)


_hashed: dict[
    Path, tuple[tuple[int, ...], str]
] = {}  # a file's stat when it was hashed, and its sha256
_hashed_lock = threading.Lock()


def _signature(st: os.stat_result) -> tuple[int, ...]:
    """What changes when a file is written, replaced or moved: it is hashed again then."""
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_mode)


def _sha256(path: Path) -> tuple[os.stat_result, str]:
    """Return a regular file's stat and sha256, read through no link; kept while it is unchanged.

    A file that cannot be read, or is a link (ELOOP), raises what `os.open` raises.
    """
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        with _hashed_lock:
            kept = _hashed.get(path)
        if kept is not None and kept[0] == _signature(st):
            return st, kept[1]
        digest = hashlib.sha256()
        while chunk := handle.read(models.CHUNK_SIZE):
            digest.update(chunk)
        sha = digest.hexdigest()
        if _signature(os.fstat(handle.fileno())) == _signature(st):  # unchanged while it was read
            with _hashed_lock:
                _hashed[path] = (_signature(st), sha)
    return st, sha


def _file_problem(folder: Path, pinned: models.RunnerFile) -> str | None:
    """Return why `folder`'s copy of one file of the bundle is not the pinned one, or None."""
    path = folder / pinned.name
    try:
        st = path.lstat()
        whole = stat.S_ISREG(st.st_mode) and st.st_size == pinned.size
        sha = _sha256(path)[1] if whole else ""
    except FileNotFoundError:
        return f"{pinned.name} is missing"
    except OSError as exc:
        return f"{pinned.name} cannot be read: {exc.strerror or exc}"
    if not stat.S_ISREG(st.st_mode):
        return f"{pinned.name} is not a regular file"
    if st.st_size != pinned.size:
        return f"{pinned.name} is {st.st_size} bytes, not {pinned.size}"
    if pinned.name == BINARY and not st.st_mode & stat.S_IXUSR:
        return f"{pinned.name} is not executable"
    return None if sha == pinned.sha256 else f"{pinned.name} is not the pinned build"


def runner_problem(folder: Path) -> str | None:
    """Return why `folder` does not hold the pinned runner bundle (`models.SOULX_RUNNER`), or None.

    Every file of the bundle must be there as a regular file, not a link, with its pinned size
    and sha256, and the runner must be executable. A file is hashed again only once it changed.
    """
    if folder.is_symlink() or not folder.is_dir():
        return f"{folder} is not a folder"
    return next(
        (p for pinned in models.SOULX_RUNNER if (p := _file_problem(folder, pinned)) is not None),
        None,
    )


def verify(found: Files) -> str | None:
    """Return why SoulX-Singer may not run from `found`, or None when its files are the pinned ones.

    The runner bundle is checked whole (`runner_problem`), and the model by its pinned size
    (`jarvis doctor` hashes it; 1.2 GB is too much to hash for every song).
    """
    if found.binary.parent / BINARY != found.binary:
        return f"{found.binary} is not where the runner goes"
    problem = runner_problem(found.binary.parent)
    if problem is not None:
        return problem
    try:
        size = found.model.stat().st_size
    except OSError as exc:
        return f"the model cannot be read: {exc.strerror or exc}"
    if size != models.SOULX_MODEL_BYTES:
        return f"the model is {size} bytes, not {models.SOULX_MODEL_BYTES}"
    return None


@lru_cache(maxsize=PHONES_CACHED)
def _phones(word: str) -> str:
    return "en_" + "-".join(arpabet.word(word))


def metadata(score: Score) -> list[dict[str, Any]]:
    """Return a score as SoulX-Singer's metadata: one segment of notes, rests and holds.

    Each note carries its whole word, as SoulX-Singer's own scores do: the word's first note is
    sung (type 2), and its later syllables and holds go on from it (type 3, a slur).
    """
    phonemes, texts, lengths, pitches, kinds = ["<SP>"], ["<SP>"], [EDGE_S], ["0"], ["1"]
    begun: set[int] = set()
    for note in score.notes:
        lengths.append(note.seconds)
        if note.kind is Kind.REST:
            phonemes.append("<SP>")
            texts.append("<SP>")
            pitches.append("0")
            kinds.append("1")
            continue
        word = score.words[note.word]
        phonemes.append(_phones(word))
        texts.append(word)
        pitches.append(str(note.midi))
        kinds.append("3" if note.word in begun else "2")
        begun.add(note.word)
    phonemes.append("<SP>")
    texts.append("<SP>")
    lengths.append(EDGE_S)
    pitches.append("0")
    kinds.append("1")
    return [
        {
            "index": "song",
            "language": "English",
            "time": [0, round(sum(lengths) * 1000)],
            "duration": " ".join(f"{length:.3f}" for length in lengths),
            "text": " ".join(texts),
            "phoneme": " ".join(phonemes),
            "note_pitch": " ".join(pitches),
            "note_type": " ".join(kinds),
        }
    ]


def _write_wav(path: Path, pcm: NDArray[np.int16]) -> None:
    edge = np.zeros(round(EDGE_S * RATE), dtype=np.int16)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(np.concatenate([edge, pcm, edge]).tobytes())


def _samples(sung: wave.Wave_read, seconds: float) -> tuple[NDArray[np.int16], int]:
    """Return an open WAV's samples and rate, checked to hold a song of `seconds` with its edges.

    Raises:
        ValueError: it is not mono 16-bit sound at a usable rate, its length is not the song's,
            or it holds fewer samples than its header says (a file cut off).
    """
    if sung.getnchannels() != 1 or sung.getsampwidth() != 2:  # ruff: ignore[magic-value-comparison] - mono, 16-bit
        msg = "the singer wrote something other than mono 16-bit sound"
        raise ValueError(msg)
    rate = sung.getframerate()
    if not MIN_RATE <= rate <= MAX_RATE:
        msg = f"the singer wrote sound at {rate} Hz"
        raise ValueError(msg)
    frames = sung.getnframes()
    asked = seconds + 2 * EDGE_S
    if abs(frames / rate - asked) > LENGTH_SLACK_S:
        msg = f"the singer wrote {frames / rate:.2f} s of sound for a song of {asked:.2f} s"
        raise ValueError(msg)
    data = sung.readframes(frames)
    if len(data) != 2 * frames:
        msg = f"the singer's sound file is cut off ({len(data) // 2} of {frames} samples)"
        raise ValueError(msg)
    return np.frombuffer(data, dtype=np.int16).copy(), rate


def _read_wav(path: Path, seconds: float) -> tuple[NDArray[np.int16], int]:
    """Return the song of `seconds` in a mono 16-bit WAV, without the edges of silence around it.

    The song starts EDGE_S in, as its metadata asked, and is cut or padded to `seconds`, so it
    lines up with the score as the speaking voice's song does.

    Raises:
        ValueError: it is not one (another format, a damaged or cut-off file, another length).
    """
    try:
        with wave.open(str(path), "rb") as sung:
            pcm, rate = _samples(sung, seconds)
    except (wave.Error, EOFError, struct.error) as exc:
        msg = f"the singer wrote a sound file that cannot be read: {exc}"
        raise ValueError(msg) from exc
    edge, length = round(EDGE_S * rate), round(seconds * rate)
    song = pcm[edge : edge + length]
    return np.pad(song, (0, length - song.size)), rate


def command(found: Files, folder: Path, device: int) -> list[str]:
    """Return the `audiocpp_cli` command that sings `folder`'s target in its reference's voice."""
    return [
        str(found.binary),
        "--task",
        "gen",
        "--family",
        "soulx_singer",
        "--model",
        str(found.model),
        "--backend",
        "vulkan",
        "--device",
        str(device),
        "--voice-ref",
        str(folder / "reference.wav"),
        "--request-option",
        f"reference_metadata_file={folder / 'reference.json'}",
        "--request-option",
        f"target_metadata_file={folder / 'target.json'}",
        "--request-option",
        "control=score",
        "--request-option",
        f"seed={SEED}",
        "--out",
        str(folder / "sung.wav"),
    ]


type Runner = Callable[[list[str], float, Callable[[], bool]], tuple[int, str]]


class SoulxSinger:
    """Sings with SoulX-Singer, falling back to the speaking-voice song (see the docstring)."""

    name = "soulx"

    def __init__(  # ruff: ignore[too-many-arguments] - the runner's settings, keyword-only
        self,
        found: Files,
        speech: PsolaSinger,
        *,
        device: int = 0,
        timeout_s: float = TIMEOUT_S,
        run: Runner = run_singer,
        check: Callable[[Files], str | None] = verify,
    ) -> None:
        """Sing with the files `found`, the reference sung by `speech`, on GPU `device`.

        `check` says why the files may not run (`verify`); it is asked before every song.
        """
        super().__init__()
        self._files = found
        self._speech = speech
        self._device = device
        self._timeout_s = timeout_s
        self._runner = run
        self._check = check

    def render(self, score: Score, budget: Budget, /) -> tuple[NDArray[np.int16], int]:
        """Return the song sung by SoulX-Singer, or by the speaking voice if that fails.

        SoulX-Singer is not started when the request has less than MIN_GPU_S left, and is given
        no more than the time left. When `budget` is stopped, the program is ended at once. The
        reference raises `OutOfTimeError` when the time runs out while it is made.

        Raises:
            SongStoppedError: the song was stopped while it was being made.
        """
        reference, sung = self._speech.sing(score, budget)
        if budget.stopped():
            raise SongStoppedError
        problem = self._check(self._files)
        if problem is not None:
            log.warning(
                "SoulX-Singer's files are not the pinned release (%s); singing with the speaking"
                " voice (run tools/get_singer.py again)",
                problem,
            )
            return reference, RATE
        left = min(self._timeout_s, budget.time_left())
        if left < MIN_GPU_S:
            log.info(
                "%.1f s left: too little for SoulX-Singer; singing with the speaking voice", left
            )
            return reference, RATE
        try:
            return self._sing(reference, sung, left, budget.stopped)
        except SingerStoppedError as exc:
            raise SongStoppedError from exc
        except (OSError, ValueError, AudioError) as exc:
            log.warning("SoulX-Singer failed (%s); singing with the speaking voice", exc)
            return reference, RATE

    def _sing(
        self,
        reference: NDArray[np.int16],
        sung: Score,
        timeout_s: float,
        stopped: Callable[[], bool],
    ) -> tuple[NDArray[np.int16], int]:
        """Run SoulX-Singer once on the song, for at most `timeout_s`.

        An OSError (a file that could not be written) and the runner's AudioError (it could not
        start, ran past its time and was killed, or was stopped as `stopped` turned true) come
        through as they are.

        Raises:
            ValueError: it exited with an error or wrote no usable sound.
        """
        runtime = paths.runtime_dir()
        runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sing-", dir=runtime) as temp:
            folder = Path(temp)
            _write_wav(folder / "reference.wav", reference)
            meta = json.dumps(metadata(sung))
            (folder / "reference.json").write_text(meta, encoding="utf-8")
            (folder / "target.json").write_text(meta, encoding="utf-8")
            code, errors = self._runner(
                command(self._files, folder, self._device), timeout_s, stopped
            )
            if code != 0:
                msg = f"exit {code}: {errors.strip()[-MAX_ERROR_CHARS:]}"
                raise ValueError(msg)
            pcm, rate = _read_wav(folder / "sung.wav", sung.seconds)
        return pcm, rate
