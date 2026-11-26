"""`pc_stats`: what the PC is and how busy it is, read from fake /proc and /sys trees."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis import shortcuts
from jarvis.operations import Registry, validate_args
from jarvis.pc import stats
from jarvis.pc.processes import Usage
from jarvis.pc.stats import Disk, StatsOps
from jarvis.policy import OpClass, Origin
from jarvis.sysinfo import Sampler

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
    for figure in ("Test CPU 9000", "12%", "54 °C", "16.0 of 64.0 GB", "14.0 of 20.0 GB", "3%"):
        assert figure in result.text
    assert "500 of 2000 GB" in result.text
    assert "2 days 3 hours" in result.text
    assert "The CPU is at 12 percent" in result.spoken


def test_sizes_are_in_the_units_people_say(tmp_path: Path) -> None:
    (op,) = _stats(tmp_path)[0].operations()

    text = op.run(validate_args(op, {})).text

    assert "GiB" not in text
    assert "MiB" not in text


def test_a_load_the_cpu_keeps_up_with_is_not_reported(tmp_path: Path) -> None:
    (op,) = _stats(tmp_path)[0].operations()

    text = op.run(validate_args(op, {})).text

    assert "1.25" not in text
    assert "load" not in text.lower()


def test_more_work_than_the_cpu_can_run_is_said_in_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ops, _ = _stats(tmp_path)
    (tmp_path / "proc" / "loadavg").write_text("40.50 12.00 6.00 9/900 1234\n", encoding="utf-8")
    monkeypatch.setattr(stats.os, "cpu_count", lambda: 16)
    (op,) = ops.operations()

    result = op.run(validate_args(op, {}))

    assert "40.50" not in result.text
    assert "more work waiting than its 16 threads can run at once" in result.text
    assert "Worth a look: the CPU has more work than it can run at once." in result.text


def test_a_pc_with_room_everywhere_says_so_first_when_spoken(tmp_path: Path) -> None:
    (op,) = _stats(tmp_path)[0].operations()

    result = op.run(validate_args(op, {}))

    assert result.text.startswith("Nothing is near its limit.\nPC: Test CPU 9000")
    assert result.spoken.startswith("Nothing is near its limit. The CPU is at 12 percent")


def test_a_window_gets_the_figures_as_a_list_under_the_verdict(tmp_path: Path) -> None:
    (op,) = _stats(tmp_path)[0].operations()

    written = op.run(validate_args(op, {"programs": False})).written

    verdict, figures, rest = written.split("\n\n")
    specs, *rows = figures.splitlines()
    assert verdict == "Nothing is near its limit."
    assert specs.startswith("- **PC**: Test CPU 9000")  # then this PC's own thread count
    assert specs.endswith("64.0 GB RAM, graphics card with 20.0 GB VRAM")
    assert rows == [
        "- **CPU**: 12% busy, 54 °C",
        "- **Memory**: 16.0 of 64.0 GB used (25%), 48.0 GB free",
        "- **Graphics card**: 3% busy, 14.0 of 20.0 GB VRAM used (70%)",
        "- **Home disk**: 500 of 2000 GB used (25%), 1500 GB free",
    ]
    assert rest == "Up 2 days 3 hours."


def test_the_disks_are_named_in_words_in_the_figures_and_the_verdict(tmp_path: Path) -> None:
    proc, sys = _tree(tmp_path)
    disks = [
        Disk("~", 1000 * GIB, 960 * GIB),
        Disk("/", 500 * GIB, 475 * GIB),
        Disk("/mnt/Games", 100 * GIB, 10 * GIB),
    ]
    ops = StatsOps(
        Sampler(proc, sys), proc=proc, wait=_Ticks(proc), disks=lambda: disks, usage=list
    )
    (op,) = ops.operations()

    result = op.run(validate_args(op, {}))

    assert "the home disk is nearly full (96%); the root disk is nearly full (95%)" in result.text
    lines = result.text.splitlines()
    assert "Home disk: 960 of 1000 GB used (96%), 40 GB free." in lines
    assert "Root disk: 475 of 500 GB used (95%), 25 GB free." in lines
    assert "Disk /mnt/Games: 10 of 100 GB used (10%), 90 GB free." in lines
    assert not any(line.startswith(("Disk ~", "Disk /:")) for line in lines)
    assert "- **Home disk**: 960 of 1000 GB used (96%), 40 GB free" in result.written
    assert "- **Root disk**: 475 of 500 GB used (95%), 25 GB free" in result.written


def test_a_report_that_names_programs_is_never_laid_out_for_a_window(tmp_path: Path) -> None:
    proc, sys = _tree(tmp_path)
    usage = [Usage("some-program", 1, 2 * GIB, 5.0)]
    ops = StatsOps(
        Sampler(proc, sys), proc=proc, wait=_Ticks(proc), disks=list, usage=lambda: usage
    )
    (op,) = ops.operations()

    assert not op.run(validate_args(op, {})).written


def test_what_is_near_its_limit_is_named(tmp_path: Path) -> None:
    proc, sys = _tree(tmp_path)
    (proc / "meminfo").write_text(
        f"MemTotal: {64 * 1024 * 1024} kB\nMemAvailable: {3 * 1024 * 1024} kB\n", encoding="utf-8"
    )
    (sys / "class/hwmon/hwmon7/temp1_input").write_text("93000\n", encoding="utf-8")
    (sys / "class/drm/card1/device/mem_info_vram_used").write_text(
        f"{19 * GIB}\n", encoding="utf-8"
    )
    disks = [Disk("~", 1000 * GIB, 960 * GIB)]
    ops = StatsOps(Sampler(proc, sys), proc=proc, wait=_Ticks(proc), disks=lambda: disks)
    (op,) = ops.operations()

    result = op.run(validate_args(op, {}))

    assert result.text.startswith(
        "Worth a look: the CPU is hot (93 °C); memory is nearly full (95%); "
        "video memory is nearly full (95%); the home disk is nearly full (96%).\n"
    )
    assert "Nothing is near its limit" not in result.text
    assert result.spoken.startswith(
        "The CPU is hot, memory is nearly full, video memory is nearly full and the home "
        "disk is nearly full. The CPU is at"
    )


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
    assert "limit" not in result.text  # nothing was measured, so nothing is called well
    assert not result.spoken


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

    assert "12.0 GB free" in text
    assert "Swap: 2.0 of 8.0 GB used" in text
    assert "Using the most memory: model-server 36.0 GB, browser 2.0 GB, editor 512 MB." in text
    assert "Using the most CPU: browser 74%, editor 9%." in text  # an idle program is left out


def test_pc_stats_without_a_process_reader_still_answers(tmp_path: Path) -> None:
    ops, _ = _stats(tmp_path)
    (op,) = ops.operations()

    text = op.run(validate_args(op, {})).text

    assert "the most memory" not in text
    assert "Swap" not in text  # no swap counters in this /proc


MOUNTS = (
    "26 1 0:33 /@ / rw,relatime shared:1 - btrfs /dev/mapper/root rw,subvol=/@\n"
    "27 26 0:35 /@home /home rw,relatime shared:2 - btrfs /dev/mapper/root rw,subvol=/@home\n"
)


def _disks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mounts: str) -> list[str]:
    monkeypatch.setattr(stats.Path, "home", staticmethod(lambda: Path("/home/me")))
    usage = SimpleNamespace(total=900 * GIB, used=300 * GIB, free=600 * GIB)

    def disk_usage(_path: object) -> SimpleNamespace:
        return usage

    monkeypatch.setattr(stats.shutil, "disk_usage", disk_usage)
    info = tmp_path / "mountinfo"
    info.write_text(mounts, encoding="utf-8")
    return [d.name for d in stats.home_disks(info)]


def test_two_subvolumes_of_one_file_system_are_one_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _disks(monkeypatch, tmp_path, MOUNTS) == ["~"]


def test_two_file_systems_of_equal_size_are_two_disks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mounts = MOUNTS.replace(
        "/@home /home rw,relatime shared:2 - btrfs /dev/mapper/root", "/ /home rw - ext4 /dev/sdb1"
    )

    assert _disks(monkeypatch, tmp_path, mounts) == ["~", "/"]


def test_a_result_that_names_programs_is_process_data(tmp_path: Path) -> None:
    proc, sys = _tree(tmp_path)
    usage = [Usage("some-program", 1, 2 * GIB, 5.0)]
    ops = StatsOps(
        Sampler(proc, sys), proc=proc, wait=_Ticks(proc), disks=list, usage=lambda: usage
    )
    (op,) = ops.operations()

    named = op.run(validate_args(op, {}))
    plain = op.run(validate_args(op, {"programs": False}))

    assert named.origin is Origin.PROCESS
    assert "some-program" not in plain.text
    assert plain.origin is Origin.TOOL_STATUS


@pytest.mark.parametrize("said", ["how's my PC doing", "system stats", "what's my GPU usage"])
def test_a_plain_question_does_not_look_up_the_programs(tmp_path: Path, said: str) -> None:
    proc, sys = _tree(tmp_path)
    calls: list[int] = []

    def usage() -> list[Usage]:
        calls.append(1)
        return []

    ops = StatsOps(Sampler(proc, sys), proc=proc, wait=_Ticks(proc), disks=list, usage=usage)
    registry = Registry(ops.operations())
    hit = shortcuts.match(said, registry.shortcuts())
    assert hit is not None
    (op,) = ops.operations()

    op.run(validate_args(op, hit.arguments))

    assert calls == []


@pytest.mark.parametrize(
    ("path", "source"),
    [
        ("/", ("btrfs", "/dev/mapper/root")),
        ("/home/me", ("btrfs", "/dev/mapper/root")),
        ("/homework", ("btrfs", "/dev/mapper/root")),  # not inside /home
        ("/mnt/my disk/x", ("ext4", "/dev/sdc1")),
        ("/mnt/my diskette", ("btrfs", "/dev/mapper/root")),
    ],
)
def test_the_mount_a_path_is_on_is_found(path: str, source: tuple[str, str]) -> None:
    mounts = MOUNTS + "40 26 8:33 / /mnt/my\\040disk rw - ext4 /dev/sdc1 rw\n"

    assert stats._mount_source(Path(path), mounts) == source  # pyright: ignore[reportPrivateUsage]


def test_an_overmount_wins() -> None:
    mounts = MOUNTS + "50 27 0:50 / /home rw - tmpfs tmpfs rw\n"

    assert stats._mount_source(Path("/home/me"), mounts) == ("tmpfs", "tmpfs")  # pyright: ignore[reportPrivateUsage]


def test_without_mountinfo_equal_subvolumes_still_count_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(stats.Path, "home", staticmethod(lambda: tmp_path))
    usage = SimpleNamespace(total=900 * GIB, used=300 * GIB, free=600 * GIB)

    def disk_usage(_path: object) -> SimpleNamespace:
        return usage

    monkeypatch.setattr(stats.shutil, "disk_usage", disk_usage)

    assert [d.name for d in stats.home_disks(tmp_path / "missing")] == ["~"]
