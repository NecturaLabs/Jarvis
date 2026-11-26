"""`pc_stats`: what this PC is and how busy it is (processor, memory, graphics card, disks).

Every figure is read from /proc, /sys and the file systems' own counters (`jarvis.sysinfo`), so
answering starts no program. The words are Jarvis's own, built from numbers and the processor's
name, so the result needs no `file` or `web` origin.
"""

import contextlib
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.operations import Operation, Result
from jarvis.policy import Origin
from jarvis.shortcuts import Shortcut

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.pc.processes import Usage
    from jarvis.sysinfo import Sampler, Specs, Stats

GIB = 1024**3  # shown as "GB", the unit people say, as file sizes are (`jarvis.pc.files`)
CPU_WINDOW_S = 0.25  # between the two samples a CPU share needs
TOP = 3  # programs named as using the most memory and CPU
BUSY_PERCENT = 1.0  # a program below this much CPU is not named as busy
MIB = 1024**2
MOUNT_POINT = 4  # the mount point's field in a /proc/self/mountinfo line
HOT_C = 90  # a processor or graphics card at this temperature is named as hot
NEARLY_FULL = 0.9  # memory, video memory or a disk used this far is named as nearly full
ALL_WELL = "Nothing is near its limit."
DAY_S = 86400
HOUR_S = 3600
MINUTE_S = 60


def _plain(_groups: dict[str, str]) -> dict[str, Any]:
    """A plain question gets the short report, spoken or laid out for a window: no programs."""
    return {"programs": False}


_STATS = (
    Shortcut(r"(?:how is|hows) (?:my |the )?(?:pc|computer|system) (?:doing|running)", _plain),
    Shortcut(r"(?:my |the )?(?:pc|computer|system) (?:stats|status|load|usage|specs)", _plain),
    Shortcut(
        r"(?:(?:whats|what is) (?:my |the )?)?(?:cpu|gpu|vram|ram|memory) "
        r"(?:usage|load|temperature|temp)",
        _plain,
    ),
)


@dataclass(frozen=True, slots=True)
class Disk:
    """One file system: the name it is shown by, and its size and use in bytes."""

    name: str
    total: int
    used: int


def _unescape(field: str) -> str:
    r"""Return a mountinfo field with its octal escapes (`\040` for a space) decoded."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def _mount_source(path: Path, mountinfo: str) -> tuple[str, str] | None:
    """Return the file system type and source of the mount `path` is on, from mountinfo's text.

    Two btrfs subvolumes of one file system have their own device numbers but one source.
    """
    target = os.path.realpath(path)
    best, found = "", None
    for line in mountinfo.splitlines():
        head, sep, tail = line.partition(" - ")
        fields, rest = head.split(), tail.split()
        if not sep or len(fields) < MOUNT_POINT + 1 or len(rest) < 2:  # ruff: ignore[magic-value-comparison] - fs type and source
            continue
        point = _unescape(fields[MOUNT_POINT])
        inside = target == point or target.startswith(point.rstrip("/") + "/")
        if inside and len(point) >= len(best):
            best, found = point, (rest[0], _unescape(rest[1]))
    return found


def home_disks(mountinfo: Path = Path("/proc/self/mountinfo")) -> list[Disk]:
    """Return the file systems of the home folder and the root, once each.

    One file system is known by its mount's type and source; without mountinfo, by its device
    number or, for btrfs subvolumes (own device numbers), by an equal size and use.
    """
    try:
        mounts = mountinfo.read_text(encoding="utf-8")
    except OSError, UnicodeDecodeError:
        mounts = ""
    seen: set[object] = set()
    disks: list[Disk] = []
    for name, path in (("~", Path.home()), ("/", Path("/"))):
        try:
            usage = shutil.disk_usage(path)
            keys: tuple[object, ...] = (
                (source,)
                if (source := _mount_source(path, mounts)) is not None
                else (path.stat().st_dev, (usage.total, usage.used))
            )
        except OSError:
            continue
        if not seen.intersection(keys):
            seen.update(keys)
            disks.append(Disk(name, usage.total, usage.used))
    return disks


def _size(value: int) -> str:
    return f"{_gib(value)} GB" if value >= GIB else f"{value // MIB} MB"


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

    def stats(self, args: dict[str, Any]) -> Result:
        """Report what is near its limit, the PC's specs and load, and the busiest programs.

        The first line names what is hot, nearly full or swamped, or says that nothing is, so
        the model words a verdict that was measured. `programs=false` leaves the programs out.

        Program names are other programs' words, so a result that names them carries the
        `process` origin, as `find_processes` does.
        """
        sample = self._sampler.sample()
        if sample.cpu_percent is None:
            self._wait(CPU_WINDOW_S)
            sample = self._sampler.sample()
        specs = self._sampler.specs()
        swamped = self._swamped()
        lines = [
            *self._cpu_lines(specs, sample, swamped=swamped),
            *self._memory_lines(sample),
            self._swap_line(),
        ]
        top = self._top_lines() if args.get("programs", True) else []
        lines.extend(top)
        disks = self._disks()
        lines.extend(
            f"Disk {d.name}: {round(d.used / GIB)} of {round(d.total / GIB)} GB used "
            f"({_share(d.used, d.total)}), {round((d.total - d.used) / GIB)} GB free."
            for d in disks
        )
        up = self._read("uptime")[:1]
        with contextlib.suppress(ValueError):
            lines.append(f"Up {_uptime(float(up[0]))}." if up else "")
        strained = _strained(sample, disks, swamped=swamped)
        verdict = ""
        if strained:
            noted = "; ".join(f"{what} ({how})" if how else what for what, how in strained)
            verdict = f"Worth a look: {noted}."
        elif any(lines[1:]):  # with nothing measured, nothing is known to be well
            verdict = ALL_WELL
        figures = [line for line in lines if line]
        text = "\n".join([verdict, *figures] if verdict else figures)
        spoken = _spoken(sample, [what for what, _how in strained])
        if top:
            return Result(text, Origin.PROCESS, spoken=spoken)
        # Laid out only without program names: those are other programs' words.
        return Result(text, spoken=spoken, written=_written(verdict, figures) if spoken else "")

    def _swamped(self) -> bool:
        """Whether more work has waited over the last minute than the CPU's threads can run."""
        load = self._read("loadavg")[:1]
        threads = os.cpu_count()
        with contextlib.suppress(ValueError):
            return bool(load and threads) and float(load[0]) > (threads or 0)
        return False

    @staticmethod
    def _cpu_lines(specs: Specs, sample: Stats, *, swamped: bool) -> list[str]:
        about = [specs.cpu_model or "Processor"]
        if specs.cpu_threads:
            about.append(f"({specs.cpu_threads} threads)")
        head = "PC: " + " ".join(about)
        if specs.ram_total:
            head += f", {_gib(specs.ram_total)} GB RAM"
        if specs.gpu_vram_total:
            head += f", graphics card with {_gib(specs.gpu_vram_total)} GB VRAM"
        cpu: list[str] = []
        if sample.cpu_percent is not None:
            cpu.append(f"{round(sample.cpu_percent)}% busy")
        if sample.cpu_temperature_c is not None:
            cpu.append(f"{round(sample.cpu_temperature_c)} °C")
        if swamped:
            cpu.append(f"more work waiting than its {os.cpu_count()} threads can run at once")
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
        return f"Swap: {_gib(used)} of {_gib(total)} GB used ({_share(used, total)})."

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
                "Using the most memory: "
                + ", ".join(f"{u.name} {_size(u.rss)}" for u in by_memory)
                + "."
            )
        busy = [
            u for u in sorted(usage, key=lambda u: -u.cpu_percent) if u.cpu_percent >= BUSY_PERCENT
        ]
        if busy:
            lines.append(
                "Using the most CPU: "
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
                f"Memory: {_gib(sample.ram_used)} of {_gib(sample.ram_total)} GB used "
                f"({_share(sample.ram_used, sample.ram_total)}), {_gib(free)} GB free."
            )
        gpu = sample.gpu
        if gpu is None:
            return lines
        parts: list[str] = []
        if gpu.busy_percent is not None:
            parts.append(f"{gpu.busy_percent}% busy")
        if gpu.vram_used is not None and gpu.vram_total:
            parts.append(
                f"{_gib(gpu.vram_used)} of {_gib(gpu.vram_total)} GB VRAM used "
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
                "and VRAM, temperatures, disks, uptime, the programs using the most memory "
                "and CPU.",
                {
                    "type": "object",
                    "properties": {
                        "programs": {
                            "type": "boolean",
                            "default": True,
                            "description": "false: leave out the busiest programs",
                        }
                    },
                    "additionalProperties": False,
                },
                self.stats,
                side_effect=False,
                spoken_summary=lambda _a: "check the PC",
                shortcuts=_STATS,
            )
        ]


# The disks by name in a verdict, which is also spoken: a voice would read "~" and "/" as signs.
_DISK_WORDS = {"~": "the home disk", "/": "the root disk"}


def _strained(sample: Stats, disks: list[Disk], *, swamped: bool) -> list[tuple[str, str]]:
    """Return what is near its limit, each as words and the figure that says so ("" for none)."""
    found: list[tuple[str, str]] = []
    if sample.cpu_temperature_c is not None and sample.cpu_temperature_c >= HOT_C:
        found.append(("the CPU is hot", f"{round(sample.cpu_temperature_c)} °C"))
    if swamped:
        found.append(("the CPU has more work than it can run at once", ""))
    ram_used, ram_total = sample.ram_used, sample.ram_total
    if ram_used is not None and ram_total and ram_used >= NEARLY_FULL * ram_total:
        found.append(("memory is nearly full", _share(ram_used, ram_total)))
    gpu = sample.gpu
    if gpu is not None and gpu.temperature_c is not None and gpu.temperature_c >= HOT_C:
        found.append(("the graphics card is hot", f"{round(gpu.temperature_c)} °C"))
    vram_used, vram_total = (gpu.vram_used, gpu.vram_total) if gpu is not None else (None, 0)
    if vram_used is not None and vram_total and vram_used >= NEARLY_FULL * vram_total:
        found.append(("video memory is nearly full", _share(vram_used, vram_total)))
    found.extend(
        (f"{_DISK_WORDS.get(d.name, f'disk {d.name}')} is nearly full", _share(d.used, d.total))
        for d in disks
        if d.total and d.used >= NEARLY_FULL * d.total
    )
    return found


def _written(verdict: str, figures: list[str]) -> str:
    """Return the report for a window: the verdict, each figure as a labelled bullet, the rest."""
    rows: list[str] = []
    rest: list[str] = []
    for line in figures:
        label, found, value = line.partition(": ")
        if found:
            rows.append(f"- **{label}**: {value.removesuffix('.')}")
        else:
            rest.append(line)
    return "\n\n".join(part for part in (verdict, "\n".join(rows), " ".join(rest)) if part)


def _spoken(sample: Stats, strained: list[str]) -> str:
    """Return the load as spoken sentences, what is near its limit first, or "" when unknown."""
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
    return f"{_sentence(strained) if strained else ALL_WELL} {_sentence(parts)}"


def _sentence(parts: list[str]) -> str:
    sentence = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
    return sentence[0].upper() + sentence[1:] + "."
