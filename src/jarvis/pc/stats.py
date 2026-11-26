"""`pc_stats`: what this PC is and how busy it is (processor, memory, graphics card, disks).

Every figure is read from /proc, /sys and the file systems' own counters (`jarvis.sysinfo`), so
answering starts no program. The words are Jarvis's own, built from numbers and the processor's
name, so the result needs no `file` or `web` origin.
"""

import contextlib
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.operations import Operation, Result
from jarvis.shortcuts import Shortcut

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.pc.processes import Usage
    from jarvis.sysinfo import Sampler, Specs, Stats

GIB = 1024**3
CPU_WINDOW_S = 0.25  # between the two samples a CPU share needs
TOP = 3  # programs named as using the most memory and CPU
BUSY_PERCENT = 1.0  # a program below this much CPU is not named as busy
MIB = 1024**2
DAY_S = 86400
HOUR_S = 3600
MINUTE_S = 60

_STATS = (
    Shortcut(r"(?:how is|hows) (?:my |the )?(?:pc|computer|system) (?:doing|running)"),
    Shortcut(r"(?:my |the )?(?:pc|computer|system) (?:stats|status|load|usage|specs)"),
    Shortcut(
        r"(?:(?:whats|what is) (?:my |the )?)?(?:cpu|gpu|vram|ram|memory) "
        r"(?:usage|load|temperature|temp)"
    ),
)


@dataclass(frozen=True, slots=True)
class Disk:
    """One file system: the name it is shown by, and its size and use in bytes."""

    name: str
    total: int
    used: int


def home_disks() -> list[Disk]:
    """Return the file systems of the home folder and the root, once each.

    Two btrfs subvolumes of one file system have their own device numbers but the same size and
    use, so either match counts as the same file system.
    """
    seen: set[object] = set()
    disks: list[Disk] = []
    for name, path in (("~", Path.home()), ("/", Path("/"))):
        try:
            device = path.stat().st_dev
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        if device not in seen and (usage.total, usage.used) not in seen:
            seen.update((device, (usage.total, usage.used)))
            disks.append(Disk(name, usage.total, usage.used))
    return disks


def _size(value: int) -> str:
    return f"{_gib(value)} GiB" if value >= GIB else f"{value // MIB} MiB"


def _gib(value: int) -> str:
    return f"{value / GIB:.1f}"


def _share(used: int, total: int) -> str:
    return f"{round(100 * used / total)}%" if total else "?"


def _uptime(seconds: float) -> str:
    days, rest = divmod(int(seconds), DAY_S)
    hours, rest = divmod(rest, HOUR_S)
    parts = [f"{days} day{'' if days == 1 else 's'}"] if days else []
    if hours or days:
        parts.append(f"{hours} hour{'' if hours == 1 else 's'}")
    if not days:
        minutes = rest // MINUTE_S
        parts.append(f"{minutes} minute{'' if minutes == 1 else 's'}")
    return " ".join(parts)


class StatsOps:
    """The `pc_stats` operation over one `Sampler`."""

    def __init__(
        self,
        sampler: Sampler,
        *,
        proc: Path = Path("/proc"),
        wait: Callable[[float], None] = time.sleep,
        disks: Callable[[], list[Disk]] = home_disks,
        usage: Callable[[], list[Usage]] | None = None,
    ) -> None:
        """Read through `sampler` and `proc`; `usage` names the programs that use the most.

        `wait` and `disks` are for tests.
        """
        super().__init__()
        self._sampler = sampler
        self._proc = proc
        self._wait = wait
        self._disks = disks
        self._usage = usage

    def _read(self, name: str) -> list[str]:
        try:
            return (self._proc / name).read_text(encoding="utf-8").split()
        except OSError, UnicodeDecodeError:
            return []

    def stats(self, _args: dict[str, Any]) -> Result:
        """Report the PC's specs and its load now."""
        sample = self._sampler.sample()
        if sample.cpu_percent is None:
            self._wait(CPU_WINDOW_S)
            sample = self._sampler.sample()
        specs = self._sampler.specs()
        lines = [*self._cpu_lines(specs, sample), *self._memory_lines(sample), self._swap_line()]
        lines.extend(self._top_lines())
        lines.extend(
            f"Disk {d.name}: {round(d.used / GIB)} of {round(d.total / GIB)} GiB used "
            f"({_share(d.used, d.total)}), {round((d.total - d.used) / GIB)} GiB free."
            for d in self._disks()
        )
        up = self._read("uptime")[:1]
        with contextlib.suppress(ValueError):
            lines.append(f"Up {_uptime(float(up[0]))}." if up else "")
        return Result("\n".join(line for line in lines if line), spoken=_spoken(sample))

    def _cpu_lines(self, specs: Specs, sample: Stats) -> list[str]:
        about = [specs.cpu_model or "Processor"]
        if specs.cpu_threads:
            about.append(f"({specs.cpu_threads} threads)")
        head = "PC: " + " ".join(about)
        if specs.ram_total:
            head += f", {_gib(specs.ram_total)} GiB RAM"
        if specs.gpu_vram_total:
            head += f", graphics card with {_gib(specs.gpu_vram_total)} GiB VRAM"
        cpu: list[str] = []
        if sample.cpu_percent is not None:
            cpu.append(f"{round(sample.cpu_percent)}% busy")
        if sample.cpu_temperature_c is not None:
            cpu.append(f"{round(sample.cpu_temperature_c)} °C")
        load = self._read("loadavg")[:3]
        if len(load) == 3:  # ruff: ignore[magic-value-comparison] - the 1, 5 and 15 minute loads
            cpu.append(f"load {' / '.join(load)} (1, 5, 15 min) on {os.cpu_count() or '?'} threads")
        return [head + ".", f"CPU: {', '.join(cpu)}." if cpu else ""]

    def _swap_line(self) -> str:
        values: dict[str, int] = {}
        try:
            text = (self._proc / "meminfo").read_text(encoding="utf-8")
        except OSError, UnicodeDecodeError:
            return ""
        for line in text.splitlines():
            key, _, rest = line.partition(":")
            with contextlib.suppress(ValueError, IndexError):
                values[key] = int(rest.split()[0]) * 1024  # kB
        total, free = values.get("SwapTotal"), values.get("SwapFree")
        if not total or free is None:
            return ""
        used = total - free
        return f"Swap: {_gib(used)} of {_gib(total)} GiB used ({_share(used, total)})."

    def _top_lines(self) -> list[str]:
        """Name the programs using the most memory and CPU, when a process reader is given."""
        if self._usage is None:
            return []
        try:
            usage = self._usage()
        except OSError:
            return []
        lines: list[str] = []
        by_memory = sorted(usage, key=lambda u: -u.rss)[:TOP]
        if by_memory:
            lines.append(
                "Most memory: " + ", ".join(f"{u.name} {_size(u.rss)}" for u in by_memory) + "."
            )
        busy = [
            u for u in sorted(usage, key=lambda u: -u.cpu_percent) if u.cpu_percent >= BUSY_PERCENT
        ]
        if busy:
            lines.append(
                "Most CPU: "
                + ", ".join(f"{u.name} {round(u.cpu_percent)}%" for u in busy[:TOP])
                + "."
            )
        return lines

    @staticmethod
    def _memory_lines(sample: Stats) -> list[str]:
        lines: list[str] = []
        if sample.ram_used is not None and sample.ram_total:
            free = sample.ram_total - sample.ram_used
            lines.append(
                f"Memory: {_gib(sample.ram_used)} of {_gib(sample.ram_total)} GiB used "
                f"({_share(sample.ram_used, sample.ram_total)}), {_gib(free)} GiB available."
            )
        gpu = sample.gpu
        if gpu is None:
            return lines
        parts: list[str] = []
        if gpu.busy_percent is not None:
            parts.append(f"{gpu.busy_percent}% busy")
        if gpu.vram_used is not None and gpu.vram_total:
            parts.append(
                f"{_gib(gpu.vram_used)} of {_gib(gpu.vram_total)} GiB VRAM used "
                f"({_share(gpu.vram_used, gpu.vram_total)})"
            )
        if gpu.temperature_c is not None:
            parts.append(f"{round(gpu.temperature_c)} °C")
        if parts:
            lines.append(f"Graphics card: {', '.join(parts)}.")
        return lines

    def operations(self) -> list[Operation]:
        """Return `pc_stats`."""
        return [
            Operation(
                "pc_stats",
                "This PC's specs and how busy it is now: CPU, memory and swap, graphics card "
                "and VRAM, temperatures, disks, uptime, and the programs using the most memory "
                "and CPU.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                self.stats,
                side_effect=False,
                spoken_summary=lambda _a: "check the PC",
                shortcuts=_STATS,
            )
        ]


def _spoken(sample: Stats) -> str:
    """Return the load as one spoken sentence, or "" when nothing is known."""
    parts: list[str] = []
    if sample.cpu_percent is not None:
        cpu = f"the CPU is at {round(sample.cpu_percent)} percent"
        if sample.cpu_temperature_c is not None:
            cpu += f" and {round(sample.cpu_temperature_c)} degrees"
        parts.append(cpu)
    if sample.ram_used is not None and sample.ram_total:
        parts.append(
            f"memory is {round(sample.ram_used / GIB)} of {round(sample.ram_total / GIB)} gigabytes"
        )
    gpu = sample.gpu
    if gpu is not None and gpu.busy_percent is not None:
        card = f"the graphics card is at {gpu.busy_percent} percent"
        if gpu.vram_used is not None and gpu.vram_total:
            card += (
                f" with {round(gpu.vram_used / GIB)} of {round(gpu.vram_total / GIB)} gigabytes "
                "of video memory used"
            )
        parts.append(card)
    if not parts:
        return ""
    sentence = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
    return sentence[0].upper() + sentence[1:] + "."
