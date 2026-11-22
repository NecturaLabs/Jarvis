#!/usr/bin/env python3
"""Benchmark LiveKit wake word models on voices they never heard and on hours of real media.

The trainer's own test set (`livekit-wakeword eval`) speaks with the same speech service the models
were trained on, and holds about an hour of other audio, so a single false wake there already reads
as one an hour. This adds two checks that do not share those limits:

- `synth` makes each wake phrase and a list of similar-sounding phrases with every local Kokoro
  voice (English and accented) and the Piper voice, at three speeds, through Jarvis's own
  synthesizer. Runs in Jarvis's environment; plays nothing.
- `score-clips` gives each model's highest score on each clip, clean, mixed with TV and music from
  the media set at 10, 5 and 0 dB SNR, and with a synthetic room echo.
- `score-media` streams the media set built by `lkwb media` (about 24 h of TV, music and speeches)
  through every model in 80 ms frames and keeps every frame's score.
- `report` turns both into Markdown tables: false wakes an hour per threshold, recall at a matched
  false-wake rate, recall per voice group, the similar phrases that wake each model, and the false
  wakes of the models together, each at its own threshold, as Jarvis runs them.

`score-clips` and `score-media` need `pyarrow` and `soundfile` beside `livekit-wakeword`, as the
trainer's environment (`livekit-wakeword[train]`) has them. Models are given as `name=path.onnx`;
`report` takes each model's own phrase and threshold as `name=phrase:threshold`.

    uv run tools/wake_bench.py synth clips/
    python tools/wake_bench.py score-clips clips/ media/ out/ hey=hey_jarvis.onnx jarvis=jarvis.onnx
    python tools/wake_bench.py score-media media/ out/ hey=hey_jarvis.onnx jarvis=jarvis.onnx
    uv run tools/wake_bench.py report out/ clips/ hey=hey_jarvis:0.5 jarvis=jarvis:0.85
"""

# The trainer's environment runs Python 3.13, which evaluates annotations when it defines them.
from __future__ import annotations

import argparse
import importlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

RATE = 16000
FRAME = 1280  # 80 ms, the model's step and what Jarvis feeds it
DEAF_FRAMES = 25  # 2 s after a wake in which another does not count: Jarvis is listening then
CONDITIONS = ("clean", "snr10", "snr5", "snr0", "reverb")
NOISE_KINDS = ("tv_youtube", "music")
NOISE_S = 1800  # background audio kept for mixing
CLIP_BATCH = 300
DEFAULT_WORKERS = 6
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
TARGETS = (1.0, 0.5, 0.2, 0.1)
SPEEDS = (0.8, 1.0, 1.2)
PHRASES: Mapping[str, Sequence[str]] = {
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
SIMILAR = (
    "Hey Travis.",
    "Hey Marvin.",
    "Hey Charlie.",
    "Hey Harvey.",
    "Jervis.",
    "Jarvik.",
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
    "Marvis.",
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
SIMILAR_KIND = "similar"
PIPER_VOICE = "en_GB-alan-medium"
ENGLISH_PREFIXES = {"af": "us", "am": "us", "bf": "gb", "bm": "gb"}
VOICE_GROUPS = ("us", "gb", "accented", "piper")

type Audio = NDArray[np.float32]
type Scores = NDArray[np.float32]


@dataclass(frozen=True)
class Clip:
    """One synthesized clip: what it says and who says it."""

    kind: str  # a phrase key of PHRASES, or SIMILAR_KIND
    text: str
    speed: float
    voice: str


@dataclass(frozen=True)
class Model:
    """A model under test in `report`: its own phrase and the threshold it would run at."""

    name: str
    phrase: str
    threshold: float


def voice_group(voice: str) -> str:
    """Name a voice's group: American or British Kokoro English, accented Kokoro, or Piper."""
    if not voice.startswith("kokoro:"):
        return "piper"
    return ENGLISH_PREFIXES.get(voice.removeprefix("kokoro:")[:2], "accented")


def wake_events(over: NDArray[np.bool_]) -> int:
    """Count wakes in a run of frames over threshold: each wake leaves the detector deaf 2 s."""
    count = 0
    ready = 0
    for frame in np.flatnonzero(over):
        if frame >= ready:
            count += 1
            ready = int(frame) + DEAF_FRAMES
    return count


def mix(audio: Audio, condition: str, seed: int, noise: Audio | None) -> Audio:
    """Pad a clip with 1 s before and 0.5 s after, then add noise or echo for `condition`.

    `snrN` adds background audio from `noise` at N dB below the speech; `reverb` convolves a
    decaying noise tail (RT60 0.3 to 0.7 s). The result never clips.

    Raises:
        ValueError: a noise condition without background audio.
    """
    rng = np.random.default_rng(seed)
    out = np.concatenate(
        [np.zeros(RATE, np.float32), audio, np.zeros(RATE // 2, np.float32)]
    ).astype(np.float32)
    if condition == "reverb":
        out = np.convolve(out, _room(rng, rng.uniform(0.3, 0.7)))[: out.size].astype(np.float32)
    elif condition != "clean":
        if noise is None:
            msg = f"{condition} needs background audio"
            raise ValueError(msg)
        snr_db = float(condition.removeprefix("snr"))
        start = int(rng.integers(0, noise.size - out.size))
        part = noise[start : start + out.size]
        gain = np.sqrt(np.mean(audio**2) / ((np.mean(part**2) + 1e-12) * 10 ** (snr_db / 10)))
        out = (out + part * gain).astype(np.float32)
    peak = float(np.abs(out).max(initial=0.0))
    return (out / peak * 0.5).astype(np.float32) if peak > 0.99 else out  # ruff: ignore[magic-value-comparison] - full scale


def _room(rng: np.random.Generator, rt60: float) -> Audio:
    n = int(rt60 * RATE)
    tail = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / RATE / rt60)
    tail[0] = 1.0
    return (tail / np.sqrt(np.sum(tail**2))).astype(np.float32)


def lowest_threshold(grid: Sequence[float], rates: Sequence[float], target: float) -> float | None:
    """Return the lowest threshold whose false wakes an hour are at or under `target`."""
    for threshold, rate in zip(grid, rates, strict=True):
        if rate <= target:
            return threshold
    return None


# --- synth --------------------------------------------------------------------------------------


def clip_plan() -> list[tuple[str, str, float]]:
    """Every (kind, text, speed) one voice says: each phrase at three speeds, similar ones once."""
    plan = [
        (kind, text, speed) for kind, texts in PHRASES.items() for text in texts for speed in SPEEDS
    ]
    return plan + [(SIMILAR_KIND, text, 1.0) for text in SIMILAR]


def _voice_clips(job: tuple[Path, str]) -> list[tuple[Clip, NDArray[np.int16]]]:
    from livekit.wakeword._resample import Resampler  # ruff: ignore[import-private-name] - the library's own resampler

    from jarvis.audio import make_synthesizer  # ruff: ignore[import-outside-top-level] - loads ONNX voices in the worker

    data_dir, voice = job
    synths = {speed: make_synthesizer(data_dir, voice, speed) for speed in SPEEDS}
    out: list[tuple[Clip, NDArray[np.int16]]] = []
    for kind, text, speed in clip_plan():
        pcm, rate = synths[speed](text)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if rate != RATE:
            audio = Resampler(rate)(audio)
        out.append(
            (Clip(kind, text, speed, voice), np.clip(audio * 32768, -32768, 32767).astype(np.int16))
        )
    return out


def voices(data_dir: Path) -> list[str]:
    """Every Kokoro voice in the voice pack, and the Piper voice if its file is there."""
    with np.load(data_dir / "voices" / "kokoro" / "voices-v1.0.bin") as pack:
        names = [f"kokoro:{name}" for name in sorted(pack.files)]
    if (data_dir / "voices" / f"{PIPER_VOICE}.onnx").is_file():
        names.append(PIPER_VOICE)
    return names


def synth(data_dir: Path, out: Path, workers: int) -> int:
    """Write `clips.npz` and `clips.json` to `out`; return the number of clips."""
    out.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, object]] = []
    audio: list[NDArray[np.int16]] = []
    jobs = [(data_dir, voice) for voice in voices(data_dir)]
    with ProcessPoolExecutor(workers) as pool:
        for n, rows in enumerate(pool.map(_voice_clips, jobs), 1):
            for clip, pcm in rows:
                clips.append(clip.__dict__)
                audio.append(pcm)
            print(f"voice {n}/{len(jobs)} done", flush=True)
    np.savez(out / "clips.npz", *audio)
    (out / "clips.json").write_text(json.dumps(clips), encoding="utf-8")
    return len(clips)


def load_clips(folder: Path) -> list[Clip]:
    """Read what `synth` wrote about each clip."""
    rows = cast(
        "list[dict[str, Any]]", json.loads((folder / "clips.json").read_text(encoding="utf-8"))
    )
    return [Clip(row["kind"], row["text"], float(row["speed"]), row["voice"]) for row in rows]


# --- score --------------------------------------------------------------------------------------


def _module(name: str) -> Any:  # ruff: ignore[any-type] - a package Jarvis does not depend on
    return importlib.import_module(name)


def media_clips(media: Path) -> Iterator[tuple[str, str, Audio]]:
    """Yield (id, kind, audio) for every clip of an `lkwb media` set, in a fixed order.

    Raises:
        ValueError: a clip is not at 16 kHz.
    """
    parquet = _module("pyarrow.parquet")
    soundfile = _module("soundfile")
    for shard in sorted((media / "data").glob("media-*.parquet")):
        for row in parquet.read_table(shard, columns=["id", "noise_type", "audio"]).to_pylist():
            audio, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if rate != RATE:
                msg = f"{row['id']}: {rate} Hz, expected {RATE}"
                raise ValueError(msg)
            yield str(row["id"]), str(row["noise_type"]), np.asarray(audio, np.float32)


def background(media: Path) -> Audio:
    """Return about half an hour of TV and music from the media set, for mixing under speech."""
    parts: list[Audio] = []
    total = 0
    for _id, kind, audio in media_clips(media):
        if kind in NOISE_KINDS and float(np.sqrt(np.mean(audio**2))) > 1e-3:  # ruff: ignore[magic-value-comparison] - silent
            parts.append(audio)
            total += audio.size
            if total > NOISE_S * RATE:
                break
    return np.concatenate(parts).astype(np.float32)


def _detectors(specs: Sequence[str]) -> dict[str, Any]:
    """Load each model with one thread: ONNX Runtime otherwise takes every core per session."""
    from livekit.wakeword import WakeWordModel

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    out: dict[str, Any] = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        model = WakeWordModel()
        model.load_model(path, model_name="m", sess_options=options)
        out[name] = model
    return out


def _score_clips(job: tuple[Path, Path, str, tuple[str, ...], int, int]) -> list[list[object]]:
    clips, media, condition, specs, start, end = job
    noise = background(media) if condition.startswith("snr") else None
    detectors = _detectors(specs)
    rows: list[list[object]] = []
    with np.load(clips / "clips.npz") as data:
        for i in range(start, end):
            audio = mix(
                data[f"arr_{i}"].astype(np.float32) / 32768,
                condition,
                i * 7 + CONDITIONS.index(condition),
                noise,
            )
            rows.append(
                [i, condition, *(float(d.predict_clip(audio)["m"]) for d in detectors.values())]
            )
    return rows


def score_clips(clips: Path, media: Path, out: Path, specs: Sequence[str], workers: int) -> None:
    """Write `clip_scores.json`: each model's highest score on each clip in each condition."""
    n = len(load_clips(clips))
    jobs = [
        (clips, media, condition, tuple(specs), start, min(start + CLIP_BATCH, n))
        for condition in CONDITIONS
        for start in range(0, n, CLIP_BATCH)
    ]
    rows: list[list[object]] = []
    with ProcessPoolExecutor(workers) as pool:
        for k, part in enumerate(pool.map(_score_clips, jobs), 1):
            rows += part
            print(f"clip batch {k}/{len(jobs)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    names = [spec.split("=", 1)[0] for spec in specs]
    (out / "clip_scores.json").write_text(
        json.dumps({"models": names, "rows": rows}), encoding="utf-8"
    )


def _score_media(job: tuple[Path, tuple[str, ...], int, int]) -> dict[str, tuple[str, Scores]]:
    media, specs, part, parts = job
    detectors = list(_detectors(specs).values())
    out: dict[str, tuple[str, Scores]] = {}
    for k, (clip_id, kind, audio) in enumerate(media_clips(media)):
        if k % parts != part:
            continue
        for detector in detectors:
            detector.reset()
        frames = audio.size // FRAME
        scores = np.zeros((len(detectors), frames), np.float32)
        for j in range(frames):
            frame = audio[j * FRAME : (j + 1) * FRAME]
            for m, detector in enumerate(detectors):
                scores[m, j] = detector.predict(frame)["m"]
        out[clip_id] = (kind, scores)
    return out


def score_media(media: Path, out: Path, specs: Sequence[str], workers: int) -> None:
    """Write `media_frames.npz` (scores per model and frame) and `media_meta.json`."""
    results: dict[str, tuple[str, Scores]] = {}
    with ProcessPoolExecutor(workers) as pool:
        for k, part in enumerate(
            pool.map(_score_media, [(media, tuple(specs), p, workers) for p in range(workers)]), 1
        ):
            results.update(part)
            print(f"media part {k}/{workers}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    frames: dict[str, Any] = {cid: scores for cid, (_kind, scores) in results.items()}
    np.savez_compressed(out / "media_frames.npz", **frames)
    meta = {
        "models": [s.split("=", 1)[0] for s in specs],
        "clips": {cid: [kind] for cid, (kind, _s) in results.items()},
    }
    (out / "media_meta.json").write_text(json.dumps(meta), encoding="utf-8")


# --- report -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Results:
    """Both score sets, loaded."""

    clips: list[Clip]
    clip_models: list[str]
    clip_scores: dict[tuple[str, str], NDArray[np.float64]]  # (model, condition) -> score per clip
    media_models: list[str]
    media: dict[str, tuple[str, Scores]]  # clip id -> (kind, scores per model and frame)

    def hours(self, kind: str | None = None) -> float:
        """Hours of media, of one kind or all."""
        frames = sum(s.shape[1] for k, s in self.media.values() if kind in {None, k})
        return frames * FRAME / RATE / 3600

    def false_wakes_per_hour(
        self, thresholds: Mapping[str, float], kind: str | None = None
    ) -> float:
        """False wakes an hour of the models together, each at its threshold, like Jarvis."""
        count = 0
        for k, scores in self.media.values():
            if kind not in {None, k}:
                continue
            over = np.zeros(scores.shape[1], np.bool_)
            for name, threshold in thresholds.items():
                over |= scores[self.media_models.index(name)] >= threshold
            count += wake_events(over)
        return count / self.hours(kind)

    def recall(
        self, name: str, condition: str, kind: str, threshold: float, group: str | None = None
    ) -> float:
        """Share of the clips of `kind` (and voice group) where `name` reaches `threshold`."""
        picked = [
            i
            for i, c in enumerate(self.clips)
            if c.kind == kind and group in {None, voice_group(c.voice)}
        ]
        if not picked:
            return float("nan")  # printed as nan: no clip says it
        return float(np.mean(self.clip_scores[name, condition][picked] >= threshold))


def load_results(out: Path, clips: Path) -> Results:
    """Read what `score-clips` and `score-media` wrote."""
    meta = load_clips(clips)
    scored = cast(
        "dict[str, Any]", json.loads((out / "clip_scores.json").read_text(encoding="utf-8"))
    )
    names = cast("list[str]", scored["models"])
    table: dict[tuple[str, str], NDArray[np.float64]] = {}
    for index, condition, *values in cast("list[list[Any]]", scored["rows"]):
        for name, value in zip(names, values, strict=True):
            table.setdefault((name, condition), np.zeros(len(meta)))[int(index)] = float(value)
    media_meta = cast(
        "dict[str, Any]", json.loads((out / "media_meta.json").read_text(encoding="utf-8"))
    )
    with np.load(out / "media_frames.npz") as frames:
        media = {cid: (str(kinds[0]), frames[cid]) for cid, kinds in media_meta["clips"].items()}
    return Results(meta, names, table, cast("list[str]", media_meta["models"]), media)


def _mean_recall(results: Results, model: Model, threshold: float) -> float:
    return float(
        np.mean([results.recall(model.name, c, model.phrase, threshold) for c in CONDITIONS])
    )


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows)]


def report(results: Results, models: Sequence[Model]) -> str:
    """Render every table as Markdown."""
    kinds = sorted({k for k, _s in results.media.values()})
    lines = [
        f"Media: {results.hours():.1f} h ("
        + ", ".join(f"{k} {results.hours(k):.1f} h" for k in kinds)
        + f"); clips: {len(results.clips)} from {len({c.voice for c in results.clips})} voices",
        "",
        "## False wakes an hour on the media, per model and threshold",
        "",
        *_table(
            ["Model", *map(str, THRESHOLDS)],
            [
                [m.name, *(f"{results.false_wakes_per_hour({m.name: t}):.2f}" for t in THRESHOLDS)]
                for m in models
            ],
        ),
        "",
        "## Recall on the model's own phrase, mean of the five conditions",
        "",
        *_table(
            ["Model", *map(str, THRESHOLDS)],
            [
                [
                    m.name,
                    *(f"{_mean_recall(results, m, t) * 100:.1f}" for t in THRESHOLDS),
                ]
                for m in models
            ],
        ),
    ]
    grid = [round(0.2 + 0.01 * i, 2) for i in range(80)]
    curves = {m.name: [results.false_wakes_per_hour({m.name: t}) for t in grid] for m in models}
    for target in TARGETS:
        rows: list[list[object]] = []
        for m in models:
            threshold = lowest_threshold(grid, curves[m.name], target)
            if threshold is None:
                rows.append([m.name, "none", "", *([""] * (len(CONDITIONS) + 1))])
                continue
            rate = curves[m.name][grid.index(threshold)]
            recalls = [
                f"{results.recall(m.name, c, m.phrase, threshold) * 100:.1f}" for c in CONDITIONS
            ]
            similar = results.recall(m.name, "clean", SIMILAR_KIND, threshold) * 100
            rows.append([m.name, f"{threshold:.2f}", f"{rate:.2f}", *recalls, f"{similar:.1f}"])
        lines += [
            "",
            f"## Matched rate: recall at the lowest threshold with at most {target} false wakes/h",
            "",
            *_table(["Model", "Threshold", "False wakes/h", *CONDITIONS, "Similar woke"], rows),
        ]
    lines += [
        "",
        "## Recall per voice group at the model's threshold (clean / snr5 / snr0)",
        "",
        *_table(
            ["Model (threshold)", *VOICE_GROUPS],
            [
                [
                    f"{m.name} ({m.threshold})",
                    *(
                        " / ".join(
                            f"{results.recall(m.name, c, m.phrase, m.threshold, g) * 100:.0f}"
                            for c in ("clean", "snr5", "snr0")
                        )
                        for g in VOICE_GROUPS
                    ),
                ]
                for m in models
            ],
        ),
        "",
        "## Similar phrases that wake each model at its threshold (clean, share of voices)",
        "",
    ]
    said = dict.fromkeys(c.text for c in results.clips if c.kind == SIMILAR_KIND)
    for m in models:
        woke: list[str] = []
        for text in said:
            picked = [
                i for i, c in enumerate(results.clips) if c.kind == SIMILAR_KIND and c.text == text
            ]
            share = float(np.mean(results.clip_scores[m.name, "clean"][picked] >= m.threshold))
            if share > 0:
                woke.append(f"{text.rstrip('.')} {share * 100:.0f} %")
        lines.append(f"- {m.name} ({m.threshold}): " + (", ".join(woke) or "none"))
    phrases = list(PHRASES)
    lines += [
        "",
        "## Which phrases wake each model at its threshold (clean)",
        "",
        *_table(
            ["Model", *phrases],
            [
                [
                    m.name,
                    *(
                        f"{results.recall(m.name, 'clean', p, m.threshold) * 100:.0f}"
                        for p in phrases
                    ),
                ]
                for m in models
            ],
        ),
        "",
        "## All the models together, each at its threshold: false wakes an hour",
        "",
        *_table(
            ["Media", "False wakes/h"],
            [
                [
                    k or "all",
                    f"{results.false_wakes_per_hour({m.name: m.threshold for m in models}, k):.2f}",
                ]
                for k in [None, *kinds]
            ],
        ),
    ]
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
    r = sub.add_parser("report", help="print the tables")
    r.add_argument("out", type=Path)
    r.add_argument("clips", type=Path)
    r.add_argument("models", nargs="+", metavar="name=phrase:threshold", type=parse_model)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step; return 0."""
    args = build_parser().parse_args(argv)
    if args.command == "synth":
        from jarvis import config  # ruff: ignore[import-outside-top-level] - only synth runs in Jarvis's environment

        print(f"clips {synth(args.data_dir or config.load().data_dir, args.out, args.workers)}")
    elif args.command == "score-clips":
        score_clips(args.clips, args.media, args.out, args.models, args.workers)
    elif args.command == "score-media":
        score_media(args.media, args.out, args.models, args.workers)
    else:
        print(report(load_results(args.out, args.clips), args.models), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
