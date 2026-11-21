"""What the PC is doing: CPU, memory, graphics card and temperatures, read from /proc and /sys.

Reading these files starts no program and costs a few hundred microseconds, so the chat's toolbar
can sample them every few seconds. Every figure is optional: a file that is missing or unreadable
(another kernel, a card without the driver's counters) leaves its figure out instead of failing.
"""

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3
KIB = 1024
MILLI = 1000.0
_CPU_SENSORS = ("k10temp", "coretemp", "zenpower")
_CPU_LABELS = ("Tctl", "Tdie", "Package id 0")
_GPU_LABELS = ("junction", "edge")


@dataclass(frozen=True, slots=True)
class Gpu:
    """One graphics card: how busy it is and its memory, in bytes."""

    busy_percent: int | None
    vram_used: int | None
    vram_total: int | None
    temperature_c: float | None


@dataclass(frozen=True, slots=True)
class Stats:
    """One sample of the PC's load. `cpu_percent` is None on the first sample (it needs two)."""

    cpu_percent: float | None
    cpu_temperature_c: float | None
    ram_used: int | None
    ram_total: int | None
    gpu: Gpu | None


@dataclass(frozen=True, slots=True)
class Specs:
    """What the PC is: processor, its threads, memory and the main graphics card's memory."""

    cpu_model: str
    cpu_threads: int
    ram_total: int | None
    gpu_vram_total: int | None


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError, UnicodeDecodeError:
        return None


def _number(path: Path) -> int | None:
    text = _read(path)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _temperature(folder: Path, labels: tuple[str, ...]) -> float | None:
    """Return the first temperature in a hwmon folder whose label is one of `labels`, in °C."""
    found: dict[str, float] = {}
    with contextlib.suppress(OSError):
        for label_file in folder.glob("temp*_label"):
            label = _read(label_file)
            value = _number(label_file.with_name(label_file.name.replace("_label", "_input")))
            if label is not None and value is not None:
                found[label] = value / MILLI
    return next((found[label] for label in labels if label in found), None)


class Sampler:
    """Reads the PC's load; keep one and call `sample` repeatedly for the CPU's busy share."""

    def __init__(self, proc: Path = Path("/proc"), sys: Path = Path("/sys")) -> None:
        """Read from `proc` and `sys` (tests point them at a fake tree)."""
        super().__init__()
        self._proc = proc
        self._sys = sys
        self._last_cpu: tuple[int, int] | None = None

    def sample(self) -> Stats:
        """Return the PC's load now."""
        used, total = self._memory()
        return Stats(self._cpu(), self._cpu_temperature(), used, total, self.gpu())

    def specs(self) -> Specs:
        """Return what the PC is made of, as far as the files say."""
        model = ""
        info = _read(self._proc / "cpuinfo") or ""
        for line in info.splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name":
                model = value.strip()
                break
        gpu = self.gpu()
        return Specs(model, os.cpu_count() or 0, self._memory()[1], gpu.vram_total if gpu else None)

    def _cpu(self) -> float | None:
        first = (_read(self._proc / "stat") or "").splitlines()[:1]
        fields = first[0].split()[1:] if first and first[0].startswith("cpu ") else []
        try:
            ticks = [int(value) for value in fields]
        except ValueError:
            return None
        if len(ticks) < 5:  # ruff: ignore[magic-value-comparison] - user..iowait at least
            return None
        idle, total = ticks[3] + ticks[4], sum(ticks[:8])
        last, self._last_cpu = self._last_cpu, (idle, total)
        if last is None or total <= last[1]:
            return None
        busy = 1.0 - (idle - last[0]) / (total - last[1])
        return round(max(0.0, min(1.0, busy)) * 100.0, 1)

    def _memory(self) -> tuple[int | None, int | None]:
        values: dict[str, int] = {}
        for line in (_read(self._proc / "meminfo") or "").splitlines():
            key, _, rest = line.partition(":")
            with contextlib.suppress(ValueError, IndexError):
                values[key] = int(rest.split()[0]) * KIB
        total, available = values.get("MemTotal"), values.get("MemAvailable")
        if total is None or available is None:
            return None, total
        return total - available, total

    def _cpu_temperature(self) -> float | None:
        with contextlib.suppress(OSError):
            for folder in sorted((self._sys / "class/hwmon").iterdir()):
                if _read(folder / "name") in _CPU_SENSORS:
                    return _temperature(folder, _CPU_LABELS)
        return None

    def gpu(self) -> Gpu | None:
        """Return the graphics card with the most memory (the one models and games run on)."""
        best: tuple[int, Path] | None = None
        with contextlib.suppress(OSError):
            for card in (self._sys / "class/drm").glob("card*/device"):
                total = _number(card / "mem_info_vram_total")
                if total is not None and (best is None or total > best[0]):
                    best = (total, card)
        if best is None:
            return None
        total, card = best
        temperature = None
        with contextlib.suppress(OSError):
            for folder in sorted((card / "hwmon").iterdir()):
                temperature = _temperature(folder, _GPU_LABELS)
                if temperature is not None:
                    break
        return Gpu(
            _number(card / "gpu_busy_percent"),
            _number(card / "mem_info_vram_used"),
            total,
            temperature,
        )
