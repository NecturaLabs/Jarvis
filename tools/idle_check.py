#!/usr/bin/env python3
"""Measure whether Jarvis (or any process) is idle: no CPU or GPU load, no runaway memory.

Jarvis keeps its model loaded and runs beside games. Holding RAM and VRAM is fine, but while idle it
must use no GPU and next to no CPU. This reads /proc twice, `--seconds` apart, for each root process
and every descendant of it, and reports per root:

- CPU: utime + stime from /proc/<pid>/stat, as a percentage of one core;
- GPU: the `drm-engine-<name>: <n> ns` counters in /proc/<pid>/fdinfo/* (amdgpu and i915 publish
  them), as a busy percentage per engine, counting each DRM client once (`drm-client-id`);
- RSS: VmRSS in /proc/<pid>/status summed over the tree at the start and at the end (pages shared
  between processes count once per process, so a growing total is what matters, not its size).

It exits 1 when any root is above `--max-cpu-percent`, grew by more than `--max-rss-growth-mib` or
has a GPU engine above `--max-gpu-percent`, or when a root is gone at the end, and 2 on a usage
error. It is read-only: it never starts, stops or signals anything, and it only runs
`systemctl --user show` to resolve a unit's main process.

    tools/idle_check.py --unit jarvis.service --unit jarvis-strata.service --seconds 60
    tools/idle_check.py --pid 1234
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

PROC = Path("/proc")
_KIB_PER_MIB = 1024
DEFAULT_MAX_CPU_PERCENT = 5.0
DEFAULT_MAX_RSS_GROWTH_MIB = 50.0
DEFAULT_MAX_GPU_PERCENT = 1.0
_UTIME = 11  # utime and stime are fields 14 and 15 of stat; the text after ")" starts at field 3
_STIME = 12
_RSS = re.compile(r"^VmRSS:\s+(\d+)\s+kB", re.MULTILINE)
_ENGINE = re.compile(r"^drm-engine-(?P<name>[^:\s]+):\s+(?P<ns>\d+)\s+ns\s*$")
_CLIENT = re.compile(r"^drm-client-id:\s+(?P<id>\d+)\s*$")
_PDEV = re.compile(r"^drm-pdev:\s+(?P<pdev>\S+)\s*$")

type GpuKey = tuple[str, int, str]  # (PCI device, DRM client id, engine name)
type Run = Callable[..., subprocess.CompletedProcess[str]]


class IdleCheckError(Exception):
    """The measurement cannot be made (a unit is not running, systemctl failed)."""


@dataclass(frozen=True, slots=True)
class Root:
    """A process to measure, with the name to report it under."""

    label: str
    pid: int


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One reading of a process tree."""

    when: float
    ticks: dict[int, int]
    rss_kib: int
    gpu_ns: dict[GpuKey, int]


@dataclass(frozen=True, slots=True)
class Host:
    """What the measurement reads and waits on, replaceable in tests.

    `proc` is the /proc to read, `clock` and `sleep` stand in for real time, and `clk_tck` is the
    kernel's clock ticks per second (None: this system's).
    """

    proc: Path = PROC
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    clk_tck: int | None = None


@dataclass(frozen=True, slots=True)
class Limits:
    """What counts as idle."""

    max_cpu_percent: float = DEFAULT_MAX_CPU_PERCENT
    max_rss_growth_mib: float = DEFAULT_MAX_RSS_GROWTH_MIB
    max_gpu_percent: float = DEFAULT_MAX_GPU_PERCENT


@dataclass(slots=True)
class Result:
    """The measurement of one root."""

    root: Root
    processes: int = 0
    seconds: float = 0.0
    cpu_percent: float = 0.0
    gpu_percent: dict[str, float] = field(default_factory=dict[str, float])
    rss_start_mib: float = 0.0
    rss_end_mib: float = 0.0
    problems: list[str] = field(default_factory=list[str])


def descendants(proc: Path, pid: int) -> list[int]:
    """Return `pid` followed by every descendant, read from /proc/<pid>/task/*/children."""
    seen = {pid}
    order = [pid]
    queue = [pid]
    while queue:
        current = queue.pop()
        try:
            task_dirs = list((proc / str(current) / "task").iterdir())
        except OSError:
            continue
        for task_dir in task_dirs:
            try:
                children = (task_dir / "children").read_text(encoding="ascii").split()
            except OSError, ValueError:
                continue
            for word in children:
                if word.isdecimal() and int(word) not in seen:
                    seen.add(int(word))
                    order.append(int(word))
                    queue.append(int(word))
    return order


def cpu_ticks(proc: Path, pid: int) -> int | None:
    """Return utime + stime of `pid` in clock ticks, or None when it is gone or unreadable."""
    try:
        text = (proc / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # The command name sits in parentheses and may hold spaces and parentheses: split after the
    # last one.
    fields = text.rpartition(")")[2].split()
    if len(fields) <= _STIME:
        return None
    try:
        return int(fields[_UTIME]) + int(fields[_STIME])
    except ValueError:
        return None


def rss_kib(proc: Path, pid: int) -> int:
    """Return VmRSS of `pid` in KiB (0 for a kernel thread or a process that is gone)."""
    try:
        text = (proc / str(pid) / "status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    found = _RSS.search(text)
    return int(found.group(1)) if found else 0


def _fdinfo_engines(text: str) -> tuple[str, int, dict[str, int]] | None:
    """Return (pdev, client id, {engine: ns}) from one fdinfo file, or None for a non-DRM file."""
    client: int | None = None
    pdev = ""
    engines: dict[str, int] = {}
    for line in text.splitlines():
        if found := _CLIENT.match(line):
            client = int(found["id"])
        elif found := _PDEV.match(line):
            pdev = found["pdev"]
        elif found := _ENGINE.match(line):  # `drm-engine-capacity-*` has no "ns": not matched
            engines[found["name"]] = int(found["ns"])
    if client is None:
        return None
    return pdev, client, engines


def gpu_ns(proc: Path, pids: Sequence[int]) -> dict[GpuKey, int]:
    """Return busy nanoseconds per (device, DRM client, engine) over `pids`, each client once."""
    totals: dict[GpuKey, int] = {}
    for pid in pids:
        try:
            fdinfos = sorted((proc / str(pid) / "fdinfo").iterdir())
        except OSError:
            continue
        for fdinfo in fdinfos:
            try:
                text = fdinfo.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parsed = _fdinfo_engines(text)
            if parsed is None:
                continue
            pdev, client, engines = parsed
            # Keyed by client, so a client reachable through several descriptors (or processes) is
            # counted once: its counters are the same on each.
            for name, ns in engines.items():
                totals[pdev, client, name] = ns
    return totals


def snapshot(proc: Path, root: int, clock: Callable[[], float]) -> Snapshot | None:
    """Read the tree under `root`, or return None when `root` is not running."""
    when = clock()
    if cpu_ticks(proc, root) is None:
        return None
    pids = descendants(proc, root)
    ticks = {pid: t for pid in pids if (t := cpu_ticks(proc, pid)) is not None}
    return Snapshot(
        when=when,
        ticks=ticks,
        rss_kib=sum(rss_kib(proc, pid) for pid in pids),
        gpu_ns=gpu_ns(proc, pids),
    )


def _cpu_delta(start: Snapshot, end: Snapshot) -> int:
    """Ticks used between the snapshots; a process that appeared in between counts in full."""
    return sum(max(0, ticks - start.ticks.get(pid, 0)) for pid, ticks in end.ticks.items())


def _gpu_delta(start: Snapshot, end: Snapshot) -> dict[str, int]:
    """Nanoseconds per engine between the snapshots (a client that appeared counts in full)."""
    busy: dict[str, int] = {}
    for key, ns in end.gpu_ns.items():
        engine = key[2]
        busy[engine] = busy.get(engine, 0) + max(0, ns - start.gpu_ns.get(key, 0))
    return busy


def _judge(result: Result, limits: Limits) -> None:
    if result.cpu_percent > limits.max_cpu_percent:
        result.problems.append(
            f"CPU {result.cpu_percent:.1f} % of one core is above {limits.max_cpu_percent:g} %"
        )
    growth = result.rss_end_mib - result.rss_start_mib
    if growth > limits.max_rss_growth_mib:
        result.problems.append(
            f"RSS grew {growth:.1f} MiB, above {limits.max_rss_growth_mib:g} MiB"
        )
    for engine, percent in sorted(result.gpu_percent.items()):
        if percent > limits.max_gpu_percent:
            result.problems.append(
                f"GPU engine {engine} {percent:.1f} % busy is above {limits.max_gpu_percent:g} %"
            )


_REAL_HOST = Host()


def measure(
    roots: Sequence[Root],
    seconds: float,
    limits: Limits,
    host: Host = _REAL_HOST,
) -> list[Result]:
    """Sample every root, wait `seconds`, sample again and judge each root against `limits`."""
    proc, clock = host.proc, host.clock
    ticks_per_second = host.clk_tck if host.clk_tck is not None else os.sysconf("SC_CLK_TCK")
    starts = {root: snapshot(proc, root.pid, clock) for root in roots}
    host.sleep(seconds)
    results: list[Result] = []
    for root in roots:
        result = Result(root=root)
        results.append(result)
        start = starts[root]
        end = snapshot(proc, root.pid, clock) if start is not None else None
        if start is None:
            result.problems.append("not running at the start")
            continue
        if end is None:
            result.problems.append("exited during the measurement")
            continue
        elapsed = end.when - start.when
        if elapsed <= 0:
            result.problems.append("no time passed between the two readings")
            continue
        result.processes = len(end.ticks)
        result.seconds = elapsed
        result.cpu_percent = 100 * _cpu_delta(start, end) / ticks_per_second / elapsed
        result.gpu_percent = {
            engine: 100 * ns / 1e9 / elapsed
            for engine, ns in sorted(_gpu_delta(start, end).items())
        }
        result.rss_start_mib = start.rss_kib / _KIB_PER_MIB
        result.rss_end_mib = end.rss_kib / _KIB_PER_MIB
        _judge(result, limits)
    return results


def main_pid(unit: str, run: Run = subprocess.run) -> int:
    """Return the main PID of the systemd user unit `unit` (read-only: `systemctl --user show`).

    Raises:
        IdleCheckError: systemctl is missing or fails, or the unit has no running main process.
    """
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        msg = "systemctl is not installed"
        raise IdleCheckError(msg)
    command = [systemctl, "--user", "show", "-p", "MainPID", "--value", unit]
    try:
        done = run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        msg = f"systemctl failed for {unit}: {exc}"
        raise IdleCheckError(msg) from exc
    text = done.stdout.strip()
    if done.returncode != 0 or not text.isdecimal():
        why = done.stderr.strip() or text or "no output"
        msg = f"systemctl could not show MainPID of {unit}: {why}"
        raise IdleCheckError(msg)
    if int(text) == 0:
        msg = f"{unit} has no running main process"
        raise IdleCheckError(msg)
    return int(text)


def format_result(result: Result, limits: Limits) -> str:
    """Return the report for one root as text."""
    root = result.root
    lines = [f"{root.label} (pid {root.pid})"]
    if result.seconds > 0:
        gpu = (
            ", ".join(f"{engine} {percent:.1f} %" for engine, percent in result.gpu_percent.items())
            or "no DRM client"
        )
        growth = result.rss_end_mib - result.rss_start_mib
        lines += [
            f"  processes: {result.processes} (with descendants), over {result.seconds:.1f} s",
            f"  cpu: {result.cpu_percent:.1f} % of one core (limit {limits.max_cpu_percent:g} %)",
            f"  gpu: {gpu} (limit {limits.max_gpu_percent:g} % per engine)",
            (
                f"  rss: {result.rss_start_mib:.1f} MiB -> {result.rss_end_mib:.1f} MiB"
                f" ({growth:+.1f} MiB, limit +{limits.max_rss_growth_mib:g} MiB)"
            ),
        ]
    lines += [f"  FAIL: {problem}" for problem in result.problems] or ["  ok: idle"]
    return "\n".join(lines)


def _positive(text: str) -> float:
    value = float(text)
    if not value > 0:
        msg = f"{text} is not above zero"
        raise argparse.ArgumentTypeError(msg)
    return value


def _unit(text: str) -> str:
    if not text or text.startswith("-"):
        msg = f"{text!r} is not a unit name"
        raise argparse.ArgumentTypeError(msg)
    return text


def _pid(text: str) -> int:
    if not text.isdecimal() or int(text) < 1:
        msg = f"{text!r} is not a process id"
        raise argparse.ArgumentTypeError(msg)
    return int(text)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Read-only idle check: CPU, GPU engine time and RSS of process trees.",
    )
    parser.add_argument(
        "--unit",
        action="append",
        default=[],
        type=_unit,
        metavar="NAME",
        help="a systemd user unit; its MainPID is measured (repeatable)",
    )
    parser.add_argument(
        "--pid",
        action="append",
        default=[],
        type=_pid,
        metavar="N",
        help="a process id to measure (repeatable)",
    )
    parser.add_argument(
        "--seconds", type=_positive, default=60.0, help="measurement window (default 60)"
    )
    parser.add_argument(
        "--max-cpu-percent",
        type=_positive,
        default=DEFAULT_MAX_CPU_PERCENT,
        help="CPU limit, percent of one core (default 5)",
    )
    parser.add_argument(
        "--max-rss-growth-mib",
        type=_positive,
        default=DEFAULT_MAX_RSS_GROWTH_MIB,
        help="RSS growth limit in MiB (default 50)",
    )
    parser.add_argument(
        "--max-gpu-percent",
        type=_positive,
        default=DEFAULT_MAX_GPU_PERCENT,
        help="busy limit per GPU engine, percent (default 1)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    proc: Path = PROC,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    run: Run = subprocess.run,
) -> int:
    """Run the check and return the exit status: 0 idle, 1 not idle, 2 cannot measure."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.unit and not args.pid:
        parser.error("give at least one --unit or --pid")
    roots: list[Root] = []
    try:
        roots += [Root(unit, main_pid(unit, run)) for unit in args.unit]
    except IdleCheckError as exc:
        print(f"idle_check: {exc}", file=sys.stderr)
        return 2
    roots += [Root(f"pid {pid}", pid) for pid in args.pid]
    limits = Limits(args.max_cpu_percent, args.max_rss_growth_mib, args.max_gpu_percent)
    results = measure(roots, args.seconds, limits, Host(proc, clock, sleep))
    for result in results:
        print(format_result(result, limits))
    return 1 if any(result.problems for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
