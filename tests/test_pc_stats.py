"""`pc_stats`: what the PC is and how busy it is, read from fake /proc and /sys trees."""

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from jarvis import shortcuts
from jarvis.operations import Registry, validate_args
from jarvis.pc import stats
from jarvis.pc.processes import Usage
from jarvis.pc.stats import Disk, StatsOps
from jarvis.policy import OpClass, Origin
from jarvis.sysinfo import Sampler

if TYPE_CHECKING:
    from pathlib import Path

GIB = 1024**3


def _tree(root: Path) -> tuple[Path, Path]:
    proc, sys = root / "proc", root / "sys"
    proc.mkdir()
    (proc / "meminfo").write_text(
        f"MemTotal: {64 * 1024 * 1024} kB\nMemAvailable: {48 * 1024 * 1024} kB\n", encoding="utf-8"
    )
    (proc / "cpuinfo").write_text("model name\t: Test CPU 9000\n", encoding="utf-8")
    (proc / "uptime").write_text("183600.5 9000.0\n", encoding="utf-8")
    (proc / "loadavg").write_text("1.25 0.90 0.50 2/900 1234\n", encoding="utf-8")
    (proc / "stat").write_text("cpu  100 0 0 900 0 0 0 0 0 0\n", encoding="utf-8")
    cpu = sys / "class/hwmon/hwmon7"
    cpu.mkdir(parents=True)
    (cpu / "name").write_text("k10temp\n", encoding="utf-8")
    (cpu / "temp1_label").write_text("Tctl\n", encoding="utf-8")
    (cpu / "temp1_input").write_text("54000\n", encoding="utf-8")
    card = sys / "class/drm/card1/device"
    card.mkdir(parents=True)
    (card / "mem_info_vram_total").write_text(f"{20 * GIB}\n", encoding="utf-8")
    (card / "mem_info_vram_used").write_text(f"{14 * GIB}\n", encoding="utf-8")
    (card / "gpu_busy_percent").write_text("3\n", encoding="utf-8")
    return proc, sys


class _Ticks:
    """Moves /proc/stat on between the two samples, as time would."""

    def __init__(self, proc: Path) -> None:
        super().__init__()
        self.proc = proc
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)
        (self.proc / "stat").write_text("cpu  112 0 0 988 0 0 0 0 0 0\n", encoding="utf-8")


def _stats(tmp_path: Path) -> tuple[StatsOps, _Ticks]:
    proc, sys = _tree(tmp_path)
    ticks = _Ticks(proc)
    disks = [Disk("~", 2000 * GIB, 500 * GIB)]
    return StatsOps(Sampler(proc, sys), proc=proc, wait=ticks, disks=lambda: disks), ticks


def test_pc_stats_is_a_safe_read(tmp_path: Path) -> None:
    (op,) = _stats(tmp_path)[0].operations()

    assert op.name == "pc_stats"
    assert op.default_class is OpClass.SAFE
    assert not op.side_effect
    assert not op.outbound


def test_pc_stats_reports_the_load_with_a_real_cpu_share(tmp_path: Path) -> None:
    ops, ticks = _stats(tmp_path)
    (op,) = ops.operations()

    result = op.run(validate_args(op, {}))

    assert ticks.waits  # the CPU share needs two samples
    assert result.origin is Origin.TOOL_STATUS
    for figure in ("Test CPU 9000", "12%", "54 °C", "16.0 of 64.0 GiB", "14.0 of 20.0 GiB", "3%"):
        assert figure in result.text
    assert "1.25" in result.text
    assert "500 of 2000 GiB" in result.text
    assert "2 days 3 hours" in result.text
    assert result.spoken.startswith("The CPU is at 12 percent")


def test_missing_figures_are_left_out(tmp_path: Path) -> None:
    ops = StatsOps(
        Sampler(tmp_path / "proc", tmp_path / "sys"),
        proc=tmp_path / "proc",
        wait=lambda _s: None,
        disks=list,
    )
    (op,) = ops.operations()

    result = op.run({})

    assert "None" not in result.text
    assert result.text


@pytest.mark.parametrize(
    "said",
    ["how's my PC doing", "system stats", "what's my GPU usage", "CPU temperature", "PC status"],
)
def test_plain_questions_about_the_pc_need_no_model(tmp_path: Path, said: str) -> None:
    registry = Registry(_stats(tmp_path)[0].operations())

    hit = shortcuts.match(said, registry.shortcuts())

    assert hit is not None
    assert hit.operation == "pc_stats"


def test_pc_stats_says_what_is_free_the_swap_and_who_uses_the_most(tmp_path: Path) -> None:
    proc, sys = _tree(tmp_path)
    (proc / "meminfo").write_text(
        f"MemTotal: {64 * 1024 * 1024} kB\nMemAvailable: {12 * 1024 * 1024} kB\n"
        f"SwapTotal: {8 * 1024 * 1024} kB\nSwapFree: {6 * 1024 * 1024} kB\n",
        encoding="utf-8",
    )
    usage = [
        Usage("model-server", 1, 36 * GIB, 0.0),
        Usage("browser", 12, 2 * GIB, 74.0),
        Usage("editor", 3, 512 * 1024**2, 9.0),
        Usage("idle-thing", 1, 100 * 1024**2, 0.0),
    ]
    ops = StatsOps(
        Sampler(proc, sys), proc=proc, wait=_Ticks(proc), disks=list, usage=lambda: usage
    )
    (op,) = ops.operations()

    text = op.run(validate_args(op, {})).text

    assert "12.0 GiB available" in text
    assert "Swap: 2.0 of 8.0 GiB used" in text
    assert "Most memory: model-server 36.0 GiB, browser 2.0 GiB, editor 512 MiB." in text
    assert "Most CPU: browser 74%, editor 9%." in text  # an idle program is left out


def test_pc_stats_without_a_process_reader_still_answers(tmp_path: Path) -> None:
    ops, _ = _stats(tmp_path)
    (op,) = ops.operations()

    text = op.run(validate_args(op, {})).text

    assert "Most memory" not in text
    assert "Swap" not in text  # no swap counters in this /proc


def test_two_subvolumes_of_one_file_system_are_one_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(stats.Path, "home", staticmethod(lambda: home))
    usage = SimpleNamespace(total=900 * GIB, used=300 * GIB, free=600 * GIB)

    def disk_usage(_path: object) -> SimpleNamespace:
        return usage

    monkeypatch.setattr(stats.shutil, "disk_usage", disk_usage)

    assert [d.name for d in stats.home_disks()] == ["~"]
