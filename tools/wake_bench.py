#!/usr/bin/env python3
"""Benchmark LiveKit wake word models on voices they were not trained with and on hours of media.

The trainer's own test set (`livekit-wakeword eval`) speaks with the same speech service the new
models were trained on, and holds about an hour of other audio, so a single false wake there
already reads as one an hour. These steps add checks without those limits:

- `synth` says each wake phrase (four texts, three speeds), a list of similar-sounding phrases
  and a set of Jarvis's own reply sentences, with every local Kokoro voice and the Piper voice,
  through Jarvis's own synthesizer. Runs in Jarvis's environment; plays nothing.
- `score-clips` keeps each model's score on every 80 ms frame while a clip's speech plays (from
  its start to 1 s after its end), in seven conditions: clean, quiet (-20 dB), TV and music at
  10, 5 and 0 dB SNR measured over the speech, a room echo, and the echo with quiet speech.
- `score-media` keeps every model's score on every 80 ms frame of an `lkwb media` set (about
  24 h of TV, music and speeches), without resets. It is used to choose thresholds.
- `replay` streams the media through a set of models with their thresholds the way Jarvis
  listens: a wake resets every model and the detector hears nothing for the capture that
  follows (`--deaf`, at least Jarvis's 4 s no-speech wait). It lists every wake.
- `report` prints the tables: false wakes an hour (with counts and 95 % intervals), thresholds
  chosen on half A of the media at a false-wake target and checked on half B, recall at each
  model's threshold per condition, voice group and text, paired tests between models, similar
  phrases, Jarvis's own voice, latency, and the replays.

The trainer's media and its training sources are separate named corpora, but nothing here proves
that no recording is in both; the voices are the local Kokoro and Piper ones, not the speech
service the models were trained with.

`score-clips`, `score-media` and `replay` need `pyarrow` and `soundfile` beside
`livekit-wakeword`, as the trainer's environment (`livekit-wakeword[train]`, Python 3.13) has
them. Models are `name=path.onnx`; a replay takes `name=path.onnx@threshold`; `report` takes each
model as `name=phrase:threshold`.

    uv run tools/wake_bench.py synth clips/
    python tools/wake_bench.py score-clips clips/ media/ out/ hey=hey_jarvis.onnx yo=yo_jarvis.onnx
    python tools/wake_bench.py score-media media/ out/ hey=hey_jarvis.onnx yo=yo_jarvis.onnx
    python tools/wake_bench.py replay media/ out/today hey=hey_jarvis.onnx@0.5 yo=yo_jarvis.onnx@0.5
    uv run tools/wake_bench.py report out/ clips/ hey=hey_jarvis:0.5 --replay out/today
"""

# The trainer's environment runs Python 3.13, which evaluates annotations when it defines them.
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import math
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

RATE = 16000
FRAME = 1280  # 80 ms, the model's step and what Jarvis feeds it
FRAME_S = FRAME / RATE
LEAD_S = 1.0  # silence (or noise) before each clip's speech
TAIL_S = 1.0  # after it: a wake may come just after the word
DEAF_S = 4.0  # Jarvis's no-speech wait after a wake: the shortest time it is not listening
CONDITIONS = ("clean", "quiet", "snr10", "snr5", "snr0", "reverb", "far")
QUIET_GAIN = 0.1  # -20 dB
NOISE_KINDS = ("tv_youtube", "music")
NOISE_S_PER_KIND = 900  # background audio kept for mixing, per kind
CLIP_S = 60  # an `lkwb media` clip
CLIP_BATCH = 300
DEFAULT_WORKERS = 6
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
GRID = tuple(round(0.05 + 0.01 * i, 2) for i in range(95))
TARGETS = (0.5, 0.2, 0.1)
SPEEDS = (0.8, 1.0, 1.2)
BOOTSTRAP = 2000
TOP_WAKES = 15
PHRASES: dict[str, tuple[str, ...]] = {
    "hey_jarvis": (
        "Hey Jarvis.",
        "Hey Jarvis, what time is it?",
        "Hey, Jarvis!",
        "Hey Jarvis play some music",
    ),
    "jarvis": ("Jarvis.", "Jarvis, what time is it?", "Jarvis!", "Jarvis turn the volume down"),
    "yo_jarvis": (
        "Yo Jarvis.",
        "Yo Jarvis, what's the weather?",
        "Yo, Jarvis!",
        "Yo Jarvis next song",
    ),
}
BARE = frozenset({"Hey Jarvis.", "Hey, Jarvis!", "Jarvis.", "Jarvis!", "Yo Jarvis.", "Yo, Jarvis!"})
# Near-homophones of the name: waking on them may be acceptable, so they are reported apart.
HOMOPHONES = ("Jervis.", "Jarvik.", "Marvis.")
SIMILAR = (
    *HOMOPHONES,
    "Hey Travis.",
    "Hey Marvin.",
    "Hey Charlie.",
    "Hey Harvey.",
    "Service.",
    "Nervous.",
    "Garbage.",
    "Hey jar.",
    "Yo Davis.",
    "Yo Travis.",
    "Hey Siri.",
    "Hey Google.",
    "Alexa.",
    "Davis.",
    "Mavis.",
    "Carcass.",
    "Jarred.",
    "Yo Jarred.",
    "Hey Jason.",
    "Charges.",
    "Harvest.",
    "Java.",
    "Pass me a jar of this.",
    "The car is here.",
    "Our vision is clear.",
    "Hey Jack, sit down.",
)
# What Jarvis itself says, in its own voice, while it listens for a barge-in.
REPLIES = (
    "It's half past seven.",
    "Playing your music now.",
    "The weather today is sunny with a high of twenty degrees.",
    "I couldn't find that file.",
    "Sorry, I didn't catch that.",
    "Done. The volume is at forty percent.",
    "Your service is ready.",
    "I've started the Java build for you.",
    "There's a jar of coffee on the list.",
    "Should I go ahead?",
    "Okay, skipping to the next song.",
    "Here is what I found about Harvey Davis.",
    "The car service opens at nine.",
    "That's nervous energy, I suppose.",
    "I can't do that without your confirmation.",
    "Timer set for ten minutes.",
    "The meeting starts in five minutes.",
    "It looks like the garbage collection runs on Tuesday.",
    "Your package has been delivered.",
    "Opening Spotify.",
)
SIMILAR_KIND = "similar"
REPLY_KIND = "reply"
PIPER_VOICE = "en_GB-alan-medium"
ENGLISH_PREFIXES = {"af": "us", "am": "us", "bf": "gb", "bm": "gb"}
VOICE_GROUPS = ("us", "gb", "accented", "piper")
HEADROOM = 0.9  # synthesized clips are scaled under this peak instead of clipped

type Audio = NDArray[np.float32]
type Scores = NDArray[np.float32]


@dataclass(frozen=True)
class Clip:
    """One synthesized clip: what it says and who says it."""

    kind: str  # a phrase key of PHRASES, SIMILAR_KIND or REPLY_KIND
    text: str
    speed: float
    voice: str
    samples: int = 0  # length of the speech, at RATE


@dataclass(frozen=True)
class Model:
    """A model under test in `report`: its own phrase and the threshold it would run at."""

    name: str
    phrase: str
    threshold: float


class BenchError(Exception):
    """An input or result set that cannot give honest numbers."""


# --- statistics -----------------------------------------------------------------------------------


def _poisson_cdf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0
    return math.fsum(math.exp(i * math.log(lam) - lam - math.lgamma(i + 1)) for i in range(k + 1))


def _solve_rate(target: float, k: int) -> float:
    """Return the mean at which P(X <= k) equals `target`."""
    lo, hi = 0.0, 10.0 * (k + 10)
    for _ in range(200):
        mid = (lo + hi) / 2
        if _poisson_cdf(k, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def poisson_interval(events: int, hours: float, alpha: float = 0.05) -> tuple[float, float]:
    """Return the exact (Garwood) 95 % interval, per hour, for `events` in `hours`.

    Raises:
        BenchError: no hours of audio.
    """
    if hours <= 0:
        msg = "no hours of audio"
        raise BenchError(msg)
    lower = 0.0 if events == 0 else _solve_rate(1 - alpha / 2, events - 1)
    return lower / hours, _solve_rate(alpha / 2, events) / hours


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval for `hits` of `n`."""
    if n == 0:
        return math.nan, math.nan
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_p(only_a: int, only_b: int) -> float:
    """Return the exact two-sided McNemar p-value for the clips only one of two models caught."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = math.fsum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def clustered_difference(
    a: NDArray[np.bool_], b: NDArray[np.bool_], voices: Sequence[str], seed: int = 0
) -> tuple[float, float, float]:
    """Return b's recall minus a's and its 95 % interval, resampling whole voices."""
    names = sorted(set(voices))
    index = {name: np.array([i for i, v in enumerate(voices) if v == name]) for name in names}
    rng = np.random.default_rng(seed)
    diffs = np.empty(BOOTSTRAP)
    for k in range(BOOTSTRAP):
        draw = rng.integers(0, len(names), len(names))
        picked = np.concatenate([index[names[int(j)]] for j in draw])
        diffs[k] = b[picked].mean() - a[picked].mean()
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(b.mean() - a.mean()), float(low), float(high)


# --- counting and mixing --------------------------------------------------------------------------


def voice_group(voice: str) -> str:
    """Name a voice's group: American or British Kokoro English, accented Kokoro, or Piper.

    The accented voices speak from the same English phonemes; only the voice differs.
    """
    if not voice.startswith("kokoro:"):
        return "piper"
    return ENGLISH_PREFIXES.get(voice.removeprefix("kokoro:")[:2], "accented")


def wake_events(over: NDArray[np.bool_], deaf_frames: int) -> list[int]:
    """Return the frames that start a wake: each leaves the detector deaf for `deaf_frames`.

    This is the proxy used on stored scores, which carry no reset; `replay` is the real count.
    """
    starts: list[int] = []
    ready = 0
    for frame in np.flatnonzero(over):
        if frame >= ready:
            starts.append(int(frame))
            ready = int(frame) + deaf_frames
    return starts


def speech_power(audio: Audio) -> float:
    """Mean power over the active part of a clip: 10 ms blocks within 40 dB of the loudest."""
    block = RATE // 100
    n = audio.size // block
    if n == 0:
        return float(np.mean(audio.astype(np.float64) ** 2)) if audio.size else 0.0
    power = np.mean(audio[: n * block].astype(np.float64).reshape(n, block) ** 2, axis=1)
    return float(power[power >= power.max() * 1e-4].mean())


def room(rng: np.random.Generator) -> Audio:
    """Return a room response: the direct sound, early reflections and a tail at 0 to 10 dB DRR."""
    rt60 = rng.uniform(0.3, 0.6)
    n = int(rt60 * RATE)
    response = np.zeros(n, np.float64)
    response[0] = 1.0
    for _ in range(6):
        tap = int(rng.uniform(0.003, 0.02) * RATE)
        response[tap] += rng.uniform(0.2, 0.5) * float(rng.choice((-1.0, 1.0)))
    start = int(0.02 * RATE)
    tail = rng.standard_normal(n - start) * np.exp(-6.9 * np.arange(n - start) / RATE / rt60)
    drr_db = rng.uniform(0.0, 10.0)
    tail *= math.sqrt(10 ** (-drr_db / 10) / float(np.sum(tail**2)))
    response[start:] += tail
    return response.astype(np.float32)


def mix(audio: Audio, condition: str, seed: int, noise: Audio | None) -> Audio:
    """Pad a clip with LEAD_S before and TAIL_S after, then apply `condition`.

    `snrN` adds background audio at N dB below the speech's active power; the excerpt depends
    on `seed` only, so one clip hears the same excerpt at every level. `reverb` convolves a room
    response; `far` is that with quiet speech; `quiet` lowers the speech by 20 dB.

    Raises:
        BenchError: a noise condition without enough background audio.
    """
    speech = audio * QUIET_GAIN if condition in {"quiet", "far"} else audio
    lead, tail = int(LEAD_S * RATE), int(TAIL_S * RATE)
    out = np.concatenate([np.zeros(lead, np.float32), speech, np.zeros(tail, np.float32)]).astype(
        np.float64
    )
    if condition in {"reverb", "far"}:
        out = np.convolve(out, room(np.random.default_rng(seed)))[: out.size]
    elif condition.startswith("snr"):
        if noise is None or noise.size <= out.size:
            msg = f"{condition} needs more background audio"
            raise BenchError(msg)
        start = int(np.random.default_rng(seed).integers(0, noise.size - out.size))
        part = noise[start : start + out.size].astype(np.float64)
        snr_db = float(condition.removeprefix("snr"))
        noise_power = (float(np.mean(part**2)) + 1e-12) * 10 ** (snr_db / 10)
        out += part * math.sqrt(speech_power(speech) / noise_power)
    peak = float(np.abs(out).max(initial=0.0))
    if peak > 0.99:  # ruff: ignore[magic-value-comparison] - full scale
        out *= 0.99 / peak
    return out.astype(np.float32)


def lowest_threshold(grid: Sequence[float], rates: Sequence[float], target: float) -> float | None:
    """Return the lowest threshold from which every higher one stays at or under `target`."""
    best: float | None = None
    for threshold, rate in zip(reversed(grid), reversed(rates), strict=True):
        if rate > target:
            break
        best = threshold
    return best


def media_half(clip_id: str) -> str:
    """Put a media clip in half A (for choosing thresholds) or B (for reporting), stably."""
    return "A" if zlib.crc32(clip_id.encode()) % 2 == 0 else "B"


def sha256(path: Path) -> str:
    """Return a file's SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    """Write a file in one step, so a half-written result never looks complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.part")
    temp.write_bytes(data)
    Path(temp).replace(path)


def _write_json(path: Path, value: object) -> None:
    write_atomic(path, json.dumps(value).encode())


def _write_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    buffer = io.BytesIO()
    values: dict[str, Any] = dict(arrays)
    np.savez_compressed(buffer, **values)
    write_atomic(path, buffer.getvalue())


def _spec(spec: str) -> tuple[str, Path]:
    name, path = spec.split("=", 1)
    return name, Path(path)


# --- synth ----------------------------------------------------------------------------------------


def clip_plan(*, own_voice: bool) -> list[tuple[str, str, float]]:
    """Every (kind, text, speed) one voice says; Jarvis's own voice also says the replies."""
    plan = [
        (kind, text, speed) for kind, texts in PHRASES.items() for text in texts for speed in SPEEDS
    ]
    plan += [(SIMILAR_KIND, text, 1.0) for text in SIMILAR]
    return plan + ([(REPLY_KIND, text, 1.0) for text in REPLIES] if own_voice else [])


def to_int16(audio: Audio) -> NDArray[np.int16]:
    """Convert to int16, scaled under HEADROOM when it would clip."""
    peak = float(np.abs(audio).max(initial=0.0))
    if peak > HEADROOM:
        return np.round(audio * (HEADROOM / peak) * 32767).astype(np.int16)
    return np.round(audio * 32767).astype(np.int16)


def _voice_clips(job: tuple[Path, str, bool]) -> list[tuple[Clip, NDArray[np.int16]]]:
    from livekit.wakeword._resample import Resampler  # ruff: ignore[import-private-name] - the library's own resampler

    from jarvis.audio import make_synthesizer  # ruff: ignore[import-outside-top-level] - loads ONNX voices in the worker

    data_dir, voice, own = job
    synths = {speed: make_synthesizer(data_dir, voice, speed) for speed in SPEEDS}
    out: list[tuple[Clip, NDArray[np.int16]]] = []
    for kind, text, speed in clip_plan(own_voice=own):
        pcm, rate = synths[speed](text)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if rate != RATE:
            audio = Resampler(rate)(audio)
        out.append((Clip(kind, text, speed, voice, int(audio.size)), to_int16(audio)))
    return out


def voices(data_dir: Path) -> list[str]:
    """Every Kokoro voice in the voice pack, and the Piper voice if its file is there."""
    with np.load(data_dir / "voices" / "kokoro" / "voices-v1.0.bin") as pack:
        names = [f"kokoro:{name}" for name in sorted(pack.files)]
    if (data_dir / "voices" / f"{PIPER_VOICE}.onnx").is_file():
        names.append(PIPER_VOICE)
    return names


def synth(data_dir: Path, own_voice: str, out: Path, workers: int) -> int:
    """Write `clips.npz` and `clips.json` to `out`; return the number of clips."""
    clips: list[dict[str, object]] = []
    audio: dict[str, NDArray[np.int16]] = {}
    jobs = [(data_dir, voice, voice == own_voice) for voice in voices(data_dir)]
    with ProcessPoolExecutor(workers) as pool:
        for n, rows in enumerate(pool.map(_voice_clips, jobs), 1):
            for clip, pcm in rows:
                audio[f"arr_{len(clips)}"] = pcm
                clips.append(asdict(clip))
            print(f"voice {n}/{len(jobs)} done", flush=True)
    _write_npz(out / "clips.npz", audio)
    _write_json(out / "clips.json", {"own_voice": own_voice, "clips": clips})
    return len(clips)


def load_clips(folder: Path) -> list[Clip]:
    """Read what `synth` wrote about each clip.

    Raises:
        BenchError: the folder holds no clips.
    """
    raw: Any = json.loads((folder / "clips.json").read_text(encoding="utf-8"))
    rows = cast("list[dict[str, Any]]", raw["clips"] if isinstance(raw, dict) else raw)
    clips = [
        Clip(
            str(row["kind"]),
            str(row["text"]),
            float(row["speed"]),
            str(row["voice"]),
            int(row.get("samples", 0)),
        )
        for row in rows
    ]
    if not clips:
        msg = f"{folder}: no clips"
        raise BenchError(msg)
    return clips


# --- media and detectors -------------------------------------------------------------------------


def _module(name: str) -> Any:  # ruff: ignore[any-type] - a package Jarvis does not depend on
    return importlib.import_module(name)


def shards(media: Path) -> list[Path]:
    """Return the media set's shards.

    Raises:
        BenchError: the set has none.
    """
    found = sorted((media / "data").glob("media-*.parquet"))
    if not found:
        msg = f"{media}: no media-*.parquet shards"
        raise BenchError(msg)
    return found


def media_index(media: Path) -> list[tuple[int, int, str, str]]:
    """Return (shard, row, id, kind) for every media clip, reading no audio.

    Raises:
        BenchError: two clips share an id.
    """
    parquet = _module("pyarrow.parquet")
    index: list[tuple[int, int, str, str]] = []
    for s, shard in enumerate(shards(media)):
        table = parquet.read_table(shard, columns=["id", "noise_type"]).to_pylist()
        index += [(s, r, str(row["id"]), str(row["noise_type"])) for r, row in enumerate(table)]
    ids = [i for _s, _r, i, _k in index]
    if len(set(ids)) != len(ids):
        msg = f"{media}: repeated clip ids"
        raise BenchError(msg)
    return index


def decode(data: bytes, clip_id: str) -> Audio:
    """Decode one media clip.

    Raises:
        BenchError: it is not mono at RATE.
    """
    soundfile = _module("soundfile")
    audio, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if rate != RATE or audio.shape[1] != 1:
        msg = f"{clip_id}: {rate} Hz, {audio.shape[1]} channels; expected mono at {RATE}"
        raise BenchError(msg)
    return np.asarray(audio[:, 0], np.float32)


def _rows(shard: Path) -> list[dict[str, Any]]:
    parquet = _module("pyarrow.parquet")
    columns = ["id", "noise_type", "audio"]
    return cast("list[dict[str, Any]]", parquet.read_table(shard, columns=columns).to_pylist())


def shard_clips(shard: Path) -> Iterator[tuple[str, str, Audio]]:
    """Yield (id, kind, audio) for every clip of one shard."""
    for row in _rows(shard):
        clip_id = str(row["id"])
        yield clip_id, str(row["noise_type"]), decode(row["audio"]["bytes"], clip_id)


def background(media: Path, seed: int = 0) -> tuple[Audio, dict[str, float]]:
    """Return TV and music for mixing, drawn at random in equal parts, and the hours of each.

    Raises:
        BenchError: the set holds no audible TV or music.
    """
    index = media_index(media)
    rng = np.random.default_rng(seed)
    wanted: dict[int, set[int]] = {}
    for kind in NOISE_KINDS:
        rows = [(s, r) for s, r, _i, k in index if k == kind]
        for j in rng.permutation(len(rows))[: math.ceil(NOISE_S_PER_KIND / CLIP_S) + 2]:
            s, r = rows[int(j)]
            wanted.setdefault(s, set()).add(r)
    parts: dict[str, list[Audio]] = {kind: [] for kind in NOISE_KINDS}
    for s, shard in enumerate(shards(media)):
        if s not in wanted:
            continue
        table = _rows(shard)
        for r in sorted(wanted[s]):
            audio = decode(table[r]["audio"]["bytes"], str(table[r]["id"]))
            if float(np.sqrt(np.mean(audio**2))) > 1e-3:  # ruff: ignore[magic-value-comparison] - a silent clip
                parts[str(table[r]["noise_type"])].append(audio)
    used = {kind: sum(a.size for a in parts[kind]) / RATE / 3600 for kind in NOISE_KINDS}
    pool = [a for kind in NOISE_KINDS for a in parts[kind]]
    if not pool:
        msg = f"{media}: no TV or music to mix"
        raise BenchError(msg)
    order = np.random.default_rng(seed + 1).permutation(len(pool))
    return np.concatenate([pool[int(i)] for i in order]).astype(np.float32), used


def _session_options() -> ort.SessionOptions:
    """One thread per session: ONNX Runtime otherwise takes every core for each one."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def _detectors(specs: Sequence[str]) -> dict[str, Any]:
    from livekit.wakeword import WakeWordModel

    out: dict[str, Any] = {}
    for spec in specs:
        name, path = _spec(spec)
        model = WakeWordModel()
        model.load_model(path, model_name="m", sess_options=_session_options())
        out[name] = model
    return out


def model_hashes(specs: Sequence[str]) -> dict[str, str]:
    """Return each model's SHA-256 by name, so results from different files cannot mix."""
    return {name: sha256(path) for name, path in map(_spec, specs)}


def frame_scores(detector: Any, audio: Audio) -> Scores:  # ruff: ignore[any-type] - LiveKit's model
    """Score every 80 ms frame of `audio` from a fresh start."""
    detector.reset()
    frames = audio.size // FRAME
    out = np.zeros(frames, np.float32)
    for j in range(frames):
        out[j] = detector.predict(audio[j * FRAME : (j + 1) * FRAME])["m"]
    return out


# --- score-clips ----------------------------------------------------------------------------------


def window(clip: Clip) -> tuple[int, int]:
    """Return the frames of a mixed clip in which a wake counts: its speech, then TAIL_S."""
    first = int(LEAD_S * RATE) // FRAME
    last = math.ceil((LEAD_S * RATE + clip.samples + TAIL_S * RATE) / FRAME)
    return first, last


def _score_clips(job: tuple[Path, Path, str, tuple[str, ...], int, int]) -> dict[str, list[Scores]]:
    clips_dir, media, condition, specs, start, end = job
    noise = background(media)[0] if condition.startswith("snr") else None
    detectors = _detectors(specs)
    meta = load_clips(clips_dir)
    out: dict[str, list[Scores]] = {name: [] for name in detectors}
    with np.load(clips_dir / "clips.npz") as data:
        for i in range(start, end):
            audio = mix(data[f"arr_{i}"].astype(np.float32) / 32768, condition, i, noise)
            first, last = window(meta[i])
            for name, detector in detectors.items():
                out[name].append(frame_scores(detector, audio)[first:last])
    return out


def score_clips(clips: Path, media: Path, out: Path, specs: Sequence[str], workers: int) -> None:
    """Write `clip_frames.npz` (each model's scores in each clip's window) and `clip_meta.json`.

    Raises:
        BenchError: a model's scores do not cover every clip.
    """
    n = len(load_clips(clips))
    arrays: dict[str, NDArray[Any]] = {}
    with ProcessPoolExecutor(workers) as pool:
        for condition in CONDITIONS:
            jobs = [
                (clips, media, condition, tuple(specs), s, min(s + CLIP_BATCH, n))
                for s in range(0, n, CLIP_BATCH)
            ]
            parts: dict[str, list[Scores]] = {}
            for part in pool.map(_score_clips, jobs):
                for name, rows in part.items():
                    parts.setdefault(name, []).extend(rows)
            for name, rows in parts.items():
                if len(rows) != n:
                    msg = f"{name} {condition}: {len(rows)} clips scored of {n}"
                    raise BenchError(msg)
                arrays[f"{name}|{condition}|scores"] = np.concatenate(rows).astype(np.float16)
                arrays[f"{name}|{condition}|offsets"] = np.cumsum([0, *(r.size for r in rows)])
            print(f"condition {condition} done", flush=True)
    _write_npz(out / "clip_frames.npz", arrays)
    meta = {
        "models": [_spec(s)[0] for s in specs],
        "model_sha256": model_hashes(specs),
        "clips_sha256": sha256(clips / "clips.npz"),
        "clips": n,
        "noise_hours": background(media)[1],
    }
    _write_json(out / "clip_meta.json", meta)


# --- score-media and replay -----------------------------------------------------------------------


def _score_media(job: tuple[Path, tuple[str, ...]]) -> dict[str, tuple[str, Scores]]:
    shard, specs = job
    detectors = list(_detectors(specs).values())
    return {
        clip_id: (kind, np.stack([frame_scores(d, audio) for d in detectors]))
        for clip_id, kind, audio in shard_clips(shard)
    }


def score_media(media: Path, out: Path, specs: Sequence[str], workers: int) -> None:
    """Write `media_frames.npz` (scores per model and frame, no resets) and `media_meta.json`.

    Raises:
        BenchError: not every media clip was scored.
    """
    index = media_index(media)
    results: dict[str, tuple[str, Scores]] = {}
    jobs = [(shard, tuple(specs)) for shard in shards(media)]
    with ProcessPoolExecutor(workers) as pool:
        for k, part in enumerate(pool.map(_score_media, jobs), 1):
            results.update(part)
            print(f"media shard {k}/{len(jobs)} done", flush=True)
    if len(results) != len(index):
        msg = f"{len(results)} media clips scored of {len(index)}"
        raise BenchError(msg)
    _write_npz(out / "media_frames.npz", {cid: scores for cid, (_kind, scores) in results.items()})
    meta = {
        "models": [_spec(s)[0] for s in specs],
        "model_sha256": model_hashes(specs),
        "clips": {cid: [kind, media_half(cid)] for cid, (kind, _s) in results.items()},
    }
    _write_json(out / "media_meta.json", meta)


def parse_replay_model(spec: str) -> tuple[str, str, float]:
    """Read `name=path.onnx@threshold`."""
    name, rest = spec.split("=", 1)
    path, threshold = rest.rsplit("@", 1)
    return name, path, float(threshold)


def replay_clip(
    detectors: Mapping[str, Any], thresholds: Mapping[str, float], audio: Audio, deaf_frames: int
) -> list[tuple[int, dict[str, float]]]:
    """Stream one clip as Jarvis listens; return each wake's frame and the models that passed.

    A wake resets every model and the next `deaf_frames` frames are not heard, as Jarvis
    captures the request then; listening starts again from a reset.
    """
    for detector in detectors.values():
        detector.reset()
    wakes: list[tuple[int, dict[str, float]]] = []
    j, frames = 0, audio.size // FRAME
    while j < frames:
        frame = audio[j * FRAME : (j + 1) * FRAME]
        scores = {name: float(d.predict(frame)["m"]) for name, d in detectors.items()}
        passed = {name: s for name, s in scores.items() if s >= thresholds[name]}
        if passed:
            wakes.append((j, passed))
            for detector in detectors.values():
                detector.reset()
            j += deaf_frames
        j += 1
    return wakes


def _replay(
    job: tuple[Path, tuple[str, ...], int],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    shard, specs, deaf_frames = job
    parsed = [parse_replay_model(s) for s in specs]
    detectors = _detectors([f"{n}={p}" for n, p, _t in parsed])
    thresholds = {n: t for n, _p, t in parsed}
    wakes: list[dict[str, object]] = []
    hours: dict[str, float] = {}
    for clip_id, kind, audio in shard_clips(shard):
        half = media_half(clip_id)
        hours[f"{kind}|{half}"] = hours.get(f"{kind}|{half}", 0.0) + audio.size / RATE / 3600
        for frame, passed in replay_clip(detectors, thresholds, audio, deaf_frames):
            wake: dict[str, object] = {"id": clip_id, "kind": kind, "half": half}
            wakes.append({**wake, "t": frame * FRAME_S, "passed": passed})
    return wakes, hours


def replay(media: Path, out: Path, specs: Sequence[str], deaf_s: float, workers: int) -> None:
    """Write `replay.json`: every wake of the models together, and the hours per kind and half."""
    media_index(media)  # refuses repeated ids
    deaf_frames = round(deaf_s / FRAME_S)
    wakes: list[dict[str, object]] = []
    hours: dict[str, float] = {}
    jobs = [(shard, tuple(specs), deaf_frames) for shard in shards(media)]
    with ProcessPoolExecutor(workers) as pool:
        for part, part_hours in pool.map(_replay, jobs):
            wakes += part
            for key, value in part_hours.items():
                hours[key] = hours.get(key, 0.0) + value
    parsed = [parse_replay_model(s) for s in specs]
    result = {
        "models": {n: t for n, _p, t in parsed},
        "model_sha256": model_hashes([f"{n}={p}" for n, p, _t in parsed]),
        "deaf_s": deaf_s,
        "hours": hours,
        "wakes": wakes,
    }
    _write_json(out / "replay.json", result)


# --- report ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Results:
    """The clip and media score sets, loaded and checked."""

    clips: list[Clip]
    clip_models: list[str]
    clip_scores: dict[tuple[str, str], NDArray[np.float32]]  # (model, condition) -> best per clip
    windows: dict[tuple[str, str], list[Scores]]  # (model, condition) -> window scores per clip
    media_models: list[str]
    media: dict[str, tuple[str, str, Scores]]  # clip id -> (kind, half, scores per model, frame)

    def hours(self, kind: str | None = None, half: str | None = None) -> float:
        """Hours of media of one kind and half, or all."""
        frames = sum(
            s.shape[1] for k, h, s in self.media.values() if kind in {None, k} and half in {None, h}
        )
        return frames * FRAME_S / 3600

    def proxy_events(
        self, thresholds: Mapping[str, float], half: str | None = None, kind: str | None = None
    ) -> int:
        """Count wakes on the stored scores (no reset, 4 s deaf): the proxy for choosing."""
        deaf = round(DEAF_S / FRAME_S)
        count = 0
        for k, h, scores in self.media.values():
            if kind not in {None, k} or half not in {None, h}:
                continue
            over = np.zeros(scores.shape[1], np.bool_)
            for name, threshold in thresholds.items():
                over |= scores[self.media_models.index(name)] >= threshold
            count += len(wake_events(over, deaf))
        return count

    def picked(self, kind: str, group: str | None = None, text: str | None = None) -> list[int]:
        """Indexes of the clips of `kind`, optionally of one voice group or text."""
        return [
            i
            for i, c in enumerate(self.clips)
            if c.kind == kind and group in {None, voice_group(c.voice)} and text in {None, c.text}
        ]

    def hits(
        self, name: str, condition: str, threshold: float, picked: Sequence[int]
    ) -> NDArray[np.bool_]:
        """Whether `name` reached `threshold` in each picked clip's window."""
        return self.clip_scores[name, condition][list(picked)] >= threshold


def _unpack(frames: Mapping[str, NDArray[Any]], name: str, condition: str, n: int) -> list[Scores]:
    key = f"{name}|{condition}"
    if f"{key}|scores" not in frames:
        msg = f"{name} {condition}: no clip scores"
        raise BenchError(msg)
    scores = frames[f"{key}|scores"].astype(np.float32)
    offsets = frames[f"{key}|offsets"]
    if offsets.size != n + 1:
        msg = f"{name} {condition}: scores for {offsets.size - 1} clips, {n} expected"
        raise BenchError(msg)
    return [scores[offsets[i] : offsets[i + 1]] for i in range(n)]


def _media(
    out: Path, clip_hashes: Mapping[str, str]
) -> tuple[list[str], dict[str, tuple[str, str, Scores]]]:
    meta = cast("dict[str, Any]", json.loads((out / "media_meta.json").read_text(encoding="utf-8")))
    hashes = cast("dict[str, str]", meta.get("model_sha256", {}))
    for name in clip_hashes.keys() & hashes.keys():
        if clip_hashes[name] != hashes[name]:
            msg = f"{name}: the clip and media scores come from different model files"
            raise BenchError(msg)
    rows: dict[str, tuple[str, str, Scores]] = {}
    with np.load(out / "media_frames.npz") as frames:
        for cid, info in cast("dict[str, list[str]]", meta["clips"]).items():
            half = info[1] if len(info) > 1 else media_half(cid)
            rows[cid] = (info[0], half, frames[cid])
    if not rows:
        msg = "no media scores"
        raise BenchError(msg)
    return cast("list[str]", meta["models"]), rows


def load_results(out: Path, clips: Path, *, media: bool = True) -> Results:
    """Read what `score-clips` and `score-media` wrote.

    Raises:
        BenchError: missing models or clips, or results from different model or clip files.
    """
    meta = load_clips(clips)
    clip_meta = cast(
        "dict[str, Any]", json.loads((out / "clip_meta.json").read_text(encoding="utf-8"))
    )
    if clip_meta["clips"] != len(meta) or clip_meta.get("clips_sha256") not in {
        None,
        sha256(clips / "clips.npz"),
    }:
        msg = f"the clip scores in {out} were not made from the clips in {clips}"
        raise BenchError(msg)
    names = cast("list[str]", clip_meta["models"])
    best: dict[tuple[str, str], NDArray[np.float32]] = {}
    windows: dict[tuple[str, str], list[Scores]] = {}
    with np.load(out / "clip_frames.npz") as frames:
        for name in names:
            for condition in CONDITIONS:
                rows = _unpack(frames, name, condition, len(meta))
                windows[name, condition] = rows
                best[name, condition] = np.array([r.max(initial=0.0) for r in rows], np.float32)
    media_names: list[str] = []
    media_rows: dict[str, tuple[str, str, Scores]] = {}
    if media:
        media_names, media_rows = _media(
            out, cast("dict[str, str]", clip_meta.get("model_sha256", {}))
        )
    return Results(meta, names, best, windows, media_names, media_rows)


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows)]


def pct(hits: int, n: int) -> str:
    """Render a share with its 95 % Wilson interval, as `98.1 (97-99)`."""
    if n == 0:
        return "n/a"
    low, high = wilson_interval(hits, n)
    return f"{hits / n * 100:.1f} ({low * 100:.0f}-{high * 100:.0f})"


def rate(events: int, hours: float) -> str:
    """Render a rate an hour with its count and 95 % interval, as `0.25 [3] (0.05-0.73)`."""
    low, high = poisson_interval(events, hours)
    return f"{events / hours:.2f} [{events}] ({low:.2f}-{high:.2f})"


def choose_thresholds(
    results: Results, models: Sequence[Model], target: float
) -> dict[str, float | None]:
    """Return each model's lowest threshold at or under `target` false wakes an hour on half A."""
    hours = results.hours(half="A")
    return {
        m.name: lowest_threshold(
            GRID, [results.proxy_events({m.name: t}, half="A") / hours for t in GRID], target
        )
        for m in models
    }


def latency(results: Results, model: Model, condition: str) -> str:
    """Median and 90th percentile seconds from the end of a bare phrase to the wake."""
    lags: list[float] = []
    for i in results.picked(model.phrase):
        clip = results.clips[i]
        if clip.text not in BARE:
            continue
        over = np.flatnonzero(results.windows[model.name, condition][i] >= model.threshold)
        if over.size:
            lags.append((int(over[0]) + 1 - clip.samples / FRAME) * FRAME_S)
    if not lags:
        return "n/a"
    p50, p90 = np.percentile(lags, [50, 90])
    return f"{p50:+.2f} / {p90:+.2f}"


def _mean(values: NDArray[np.bool_]) -> float:
    """Return the share of True, or nan for no values (printed as nan)."""
    return float(values.mean()) if values.size else math.nan


def _share(results: Results, m: Model, condition: str, picked: Sequence[int]) -> float:
    return float(_mean(results.hits(m.name, condition, m.threshold, picked)))


def _recall_cell(results: Results, m: Model, condition: str, picked: Sequence[int]) -> str:
    return pct(int(results.hits(m.name, condition, m.threshold, picked).sum()), len(picked))


def _media_tables(results: Results, models: Sequence[Model]) -> list[str]:
    kinds = sorted({k for k, _h, _s in results.media.values()})
    lines = [
        f"Media: {results.hours():.2f} h (half A {results.hours(half='A'):.2f} h, half B "
        f"{results.hours(half='B'):.2f} h; "
        + ", ".join(f"{k} {results.hours(k):.2f} h" for k in kinds)
        + ")",
        "",
        "## False wakes an hour on all the media, per model (stored scores, 4 s deaf, no reset)",
        "",
        *_table(
            ["Model", *map(str, THRESHOLDS)],
            [
                [
                    m.name,
                    *(rate(results.proxy_events({m.name: t}), results.hours()) for t in THRESHOLDS),
                ]
                for m in models
            ],
        ),
    ]
    similar = results.picked(SIMILAR_KIND)
    for target in TARGETS:
        chosen = choose_thresholds(results, models, target)
        rows: list[list[object]] = []
        for m in models:
            t = chosen[m.name]
            if t is None:
                rows.append([m.name, "none", "", "", "", "", ""])
                continue
            at = Model(m.name, m.phrase, t)
            own = results.picked(m.phrase)
            rows.append(
                [
                    m.name,
                    f"{t:.2f}",
                    rate(results.proxy_events({m.name: t}, half="B"), results.hours(half="B")),
                    *(_recall_cell(results, at, c, own) for c in ("clean", "snr5", "far")),
                    _recall_cell(results, at, "clean", similar),
                ]
            )
        lines += [
            "",
            f"## Threshold chosen on half A for at most {target} false wakes/h, checked on half B",
            "",
            *_table(
                [
                    "Model",
                    "Threshold",
                    "Half B wakes/h",
                    "Recall clean",
                    "snr5",
                    "far",
                    "Similar woke",
                ],
                rows,
            ),
        ]
    return lines


def _clip_tables(results: Results, models: Sequence[Model]) -> list[str]:
    lines = ["", "## Recall on the model's own phrase at its threshold, per condition", ""]
    lines += _table(
        ["Model (threshold)", *CONDITIONS],
        [
            [
                f"{m.name} ({m.threshold})",
                *(_recall_cell(results, m, c, results.picked(m.phrase)) for c in CONDITIONS),
            ]
            for m in models
        ],
    )
    lines += ["", "## Recall per voice group at the model's threshold (clean / snr5 / far)", ""]
    group_rows: list[list[object]] = []
    for m in models:
        cells: list[object] = [f"{m.name} ({m.threshold})"]
        for g in VOICE_GROUPS:
            own = results.picked(m.phrase, g)
            shares = (
                [
                    f"{_mean(results.hits(m.name, c, m.threshold, own)) * 100:.0f}"
                    for c in ("clean", "snr5", "far")
                ]
                if own
                else ["n/a"]
            )
            cells.append(f"n={len(own)}: " + " / ".join(shares))
        group_rows.append(cells)
    lines += _table(["Model (threshold)", *VOICE_GROUPS], group_rows)
    lines += [
        "",
        (
            "## Recall per text at the model's threshold (clean / snr5 / far); latency after a "
            "bare phrase ends, s (median / 90th, clean ; far)"
        ),
        "",
    ]
    text_rows: list[list[object]] = []
    for m in models:
        for text in PHRASES.get(m.phrase, ()):
            own = results.picked(m.phrase, text=text)
            shares = [
                f"{_mean(results.hits(m.name, c, m.threshold, own)) * 100:.0f}"
                for c in ("clean", "snr5", "far")
            ]
            text_rows.append([m.name, text, " / ".join(shares)])
        text_rows.append(
            [m.name, "latency", f"{latency(results, m, 'clean')} ; {latency(results, m, 'far')}"]
        )
    lines += _table(["Model", "Text", "Recall"], text_rows)
    return lines


def _pair_tables(results: Results, models: Sequence[Model]) -> list[str]:
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1 :] if a.phrase == b.phrase]
    if not pairs:
        return []
    rows: list[list[object]] = []
    for a, b in pairs:
        own = results.picked(a.phrase)
        voices_of = [results.clips[i].voice for i in own]
        for c in CONDITIONS:
            ha = results.hits(a.name, c, a.threshold, own)
            hb = results.hits(b.name, c, b.threshold, own)
            diff, low, high = clustered_difference(ha, hb, voices_of)
            only_a, only_b = int((ha & ~hb).sum()), int((hb & ~ha).sum())
            rows.append(
                [
                    f"{b.name} - {a.name}",
                    c,
                    f"{diff * 100:+.1f} ({low * 100:+.1f} to {high * 100:+.1f})",
                    f"{only_a} / {only_b}",
                    f"{mcnemar_p(only_a, only_b):.3f}",
                ]
            )
    header = [
        "Pair",
        "Condition",
        "Recall difference, pp (voices resampled)",
        "Only first / only second",
        "McNemar p",
    ]
    return [
        "",
        "## Paired comparison on the same clips, each at its threshold",
        "",
        *_table(header, rows),
    ]


def _phrase_lists(results: Results, models: Sequence[Model]) -> list[str]:
    lines = [
        "",
        "## Similar phrases that wake each model at its threshold (share of voices, clean / snr5)",
        "",
    ]
    texts = dict.fromkeys(c.text for c in results.clips if c.kind == SIMILAR_KIND)
    for m in models:
        woke: list[str] = []
        for text in texts:
            picked = results.picked(SIMILAR_KIND, text=text)
            clean = _mean(results.hits(m.name, "clean", m.threshold, picked))
            noisy = _mean(results.hits(m.name, "snr5", m.threshold, picked))
            if clean > 0 or noisy > 0:
                mark = " (homophone)" if text in HOMOPHONES else ""
                woke.append(f"{text.rstrip('.')}{mark} {clean * 100:.0f}/{noisy * 100:.0f} %")
        rest = [i for i in results.picked(SIMILAR_KIND) if results.clips[i].text not in HOMOPHONES]
        share = pct(int(results.hits(m.name, "clean", m.threshold, rest).sum()), len(rest))
        lines.append(
            f"- {m.name} ({m.threshold}), all but homophones {share}: "
            + (", ".join(woke) or "none")
        )
    replies = results.picked(REPLY_KIND)
    if replies:
        lines += [
            "",
            f"## Jarvis's own replies that reach a model's threshold ({len(replies)}, clean)",
            "",
        ]
        for m in models:
            woke = [
                results.clips[i].text
                for i in replies
                if results.clip_scores[m.name, "clean"][i] >= m.threshold
            ]
            lines.append(
                f"- {m.name} ({m.threshold}): {len(woke)}"
                + (f": {'; '.join(woke)}" if woke else "")
            )
    lines += ["", "## Which phrases wake each model at its threshold (clean)", ""]
    lines += _table(
        ["Model", *PHRASES],
        [
            [
                m.name,
                *(f"{_share(results, m, 'clean', results.picked(p)) * 100:.0f}" for p in PHRASES),
            ]
            for m in models
        ],
    )
    return lines


def replay_section(rep: Mapping[str, Any]) -> list[str]:
    """Render one replay: wakes an hour per half and media kind, and the strongest wakes."""
    models = cast("dict[str, float]", rep["models"])
    hours = cast("dict[str, float]", rep["hours"])
    wakes = cast("list[dict[str, Any]]", rep["wakes"])
    label = ", ".join(f"{n} {t}" for n, t in models.items())
    kinds = sorted({k.split("|")[0] for k in hours})
    rows: list[list[object]] = []
    for half in ("A", "B", None):
        for kind in [*kinds, None]:
            h = sum(
                v
                for k, v in hours.items()
                if kind in {None, k.split("|")[0]} and half in {None, k.split("|")[1]}
            )
            n = sum(1 for w in wakes if kind in {None, w["kind"]} and half in {None, w["half"]})
            if h > 0:
                rows.append([half or "all", kind or "all", f"{h:.2f}", rate(n, h)])
    top = sorted(wakes, key=lambda w: -max(cast("dict[str, float]", w["passed"]).values()))[
        :TOP_WAKES
    ]
    return [
        "",
        f"## Replay as Jarvis listens: {label} (reset on a wake, then {rep['deaf_s']} s deaf)",
        "",
        *_table(["Half", "Media", "Hours", "Wakes/h"], rows),
        "",
        "Strongest wakes (listen to these to find a real mention of the name):",
        "",
        *(
            f"- {w['id']} at {w['t']:.1f} s ({w['kind']}): "
            + ", ".join(f"{n} {s:.2f}" for n, s in w["passed"].items())
            for w in top
        ),
    ]


def report(
    results: Results, models: Sequence[Model], replays: Sequence[Mapping[str, Any]] = ()
) -> str:
    """Render every table as Markdown.

    Raises:
        BenchError: a model is not in the clip scores.
    """
    for m in models:
        if m.name not in results.clip_models:
            msg = f"{m.name}: not in the clip scores ({', '.join(results.clip_models)})"
            raise BenchError(msg)
    voices_n = len({c.voice for c in results.clips})
    lines = [
        (
            f"Clips: {len(results.clips)} from {voices_n} voices. Rates: per hour [events] "
            "(95 % interval); recall: % (95 % Wilson interval)."
        ),
        "",
    ]
    if results.media:
        lines += _media_tables(results, models)
    lines += (
        _clip_tables(results, models)
        + _pair_tables(results, models)
        + _phrase_lists(results, models)
    )
    for rep in replays:
        lines += replay_section(rep)
    return "\n".join(lines) + "\n"


def parse_model(spec: str) -> Model:
    """Read `name=phrase:threshold`."""
    name, rest = spec.split("=", 1)
    phrase, threshold = rest.rsplit(":", 1)
    return Model(name, phrase, float(threshold))


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synth", help="make the clips with every local voice")
    s.add_argument("out", type=Path)
    s.add_argument("--data-dir", type=Path, help="Jarvis's data folder (default: from the config)")
    s.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    c = sub.add_parser("score-clips", help="score every clip in every condition")
    c.add_argument("clips", type=Path)
    c.add_argument("media", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("models", nargs="+", metavar="name=model.onnx")
    c.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    m = sub.add_parser("score-media", help="score every 80 ms of the media set")
    m.add_argument("media", type=Path)
    m.add_argument("out", type=Path)
    m.add_argument("models", nargs="+", metavar="name=model.onnx")
    m.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p = sub.add_parser("replay", help="stream the media through models together, as Jarvis listens")
    p.add_argument("media", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("models", nargs="+", metavar="name=model.onnx@threshold")
    p.add_argument("--deaf", type=float, default=DEAF_S, help="seconds not heard after a wake")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    r = sub.add_parser("report", help="print the tables")
    r.add_argument("out", type=Path)
    r.add_argument("clips", type=Path)
    r.add_argument("models", nargs="+", metavar="name=phrase:threshold", type=parse_model)
    r.add_argument("--replay", type=Path, action="append", default=[], help="a replay folder")
    r.add_argument("--no-media", action="store_true", help="report the clips alone")
    return parser


def run(args: argparse.Namespace) -> None:
    """Run the step `args` names."""
    if args.command == "synth":
        from jarvis import config  # ruff: ignore[import-outside-top-level] - only synth runs in Jarvis's environment

        cfg = config.load()
        count = synth(args.data_dir or cfg.data_dir, cfg.audio.voice, args.out, args.workers)
        print(f"clips {count}")
    elif args.command == "score-clips":
        score_clips(args.clips, args.media, args.out, args.models, args.workers)
    elif args.command == "score-media":
        score_media(args.media, args.out, args.models, args.workers)
    elif args.command == "replay":
        replay(args.media, args.out, args.models, args.deaf, args.workers)
    else:
        results = load_results(args.out, args.clips, media=not args.no_media)
        replays = [json.loads((p / "replay.json").read_text(encoding="utf-8")) for p in args.replay]
        print(report(results, args.models, replays), end="")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step; return 0, or 2 when the inputs cannot give honest numbers."""
    try:
        run(build_parser().parse_args(argv))
    except BenchError as exc:
        print(f"wake_bench: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
