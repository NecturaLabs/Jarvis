"""The chat window's parts alone: command suggestions, kept input, PC load and the toolbar."""

import io
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from jarvis.activity import Shown
from jarvis.agent import ContextUse
from jarvis.chat import commands
from jarvis.chat.history import InputHistory
from jarvis.chat.widgets import entry_renderable, meter, tokens_short, toolbar
from jarvis.sysinfo import Gpu, Sampler, Stats

if TYPE_CHECKING:
    from pathlib import Path

GIB = 1024**3


def _plain(renderable: object) -> str:
    console = Console(width=300, record=True, color_system=None, file=io.StringIO())
    console.print(renderable)
    return console.export_text().strip()


# Commands


def test_a_prefix_comes_before_letters_in_order() -> None:
    found = [s.value.strip() for s in commands.suggest("/c")]

    assert found[:4] == ["/clear", "/compact", "/context", "/config"]
    assert "/copy" in found


def test_letters_in_order_find_a_command() -> None:
    assert [s.value for s in commands.suggest("/cmpt")] == ["/compact"]


def test_a_command_with_arguments_offers_them_after_a_space() -> None:
    assert [s.value for s in commands.suggest("/verbosity ")] == [
        "/verbosity replies",
        "/verbosity steps",
        "/verbosity everything",
    ]
    assert [s.value for s in commands.suggest("/verbosity ev")] == ["/verbosity everything"]


@pytest.mark.parametrize("value", ["hello", "", "/clear now", "/nope ", "/help\nmore"])
def test_nothing_is_suggested_outside_a_command(value: str) -> None:
    assert commands.suggest(value) == []


def test_aliases_find_their_command() -> None:
    quit_command = commands.find("/exit")
    clear_command = commands.find("/new")

    assert quit_command is not None
    assert quit_command.name == "/quit"
    assert clear_command is not None
    assert clear_command.name == "/clear"
    assert commands.find("/nonsense") is None


# Kept input


def test_the_history_keeps_its_size_and_skips_a_repeat(tmp_path: Path) -> None:
    kept = tmp_path / "history"
    history = InputHistory(kept, 3)
    for line in ("a", "b", "b", "c", "d"):
        history.add(line)

    assert history.entries == ["b", "c", "d"]
    assert InputHistory(kept, 3).entries == ["b", "c", "d"]


def test_a_damaged_history_line_is_skipped(tmp_path: Path) -> None:
    kept = tmp_path / "history"
    kept.write_text('"one"\nnot json\n42\n"two"\n', encoding="utf-8")

    assert InputHistory(kept, 10).entries == ["one", "two"]


def test_a_size_of_0_keeps_nothing(tmp_path: Path) -> None:
    kept = tmp_path / "history"
    history = InputHistory(kept, 0)
    history.add("secret")

    assert not kept.exists()


# PC load


def _tree(root: Path) -> tuple[Path, Path]:
    proc, sys = root / "proc", root / "sys"
    proc.mkdir()
    (proc / "meminfo").write_text("MemTotal: 1000 kB\nMemAvailable: 250 kB\n", encoding="utf-8")
    (proc / "cpuinfo").write_text("model name\t: Test CPU 9000\n", encoding="utf-8")
    cpu = sys / "class/hwmon/hwmon7"
    cpu.mkdir(parents=True)
    (cpu / "name").write_text("k10temp\n", encoding="utf-8")
    (cpu / "temp1_label").write_text("Tctl\n", encoding="utf-8")
    (cpu / "temp1_input").write_text("51500\n", encoding="utf-8")
    for name, vram in (("card0", 2 * GIB), ("card1", 20 * GIB)):
        card = sys / f"class/drm/{name}/device"
        card.mkdir(parents=True)
        (card / "mem_info_vram_total").write_text(f"{vram}\n", encoding="utf-8")
        (card / "mem_info_vram_used").write_text(f"{vram // 2}\n", encoding="utf-8")
        (card / "gpu_busy_percent").write_text("41\n" if name == "card1" else "3\n")
        hwmon = card / "hwmon/hwmon3"
        hwmon.mkdir(parents=True)
        (hwmon / "temp2_label").write_text("junction\n", encoding="utf-8")
        (hwmon / "temp2_input").write_text("67000\n", encoding="utf-8")
    return proc, sys


def _stat(proc: Path, busy: int, idle: int) -> None:
    (proc / "stat").write_text(f"cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\n", encoding="utf-8")


def test_the_sampler_reads_load_memory_and_the_biggest_card(tmp_path: Path) -> None:
    proc, sys = _tree(tmp_path)
    sampler = Sampler(proc, sys)
    _stat(proc, 100, 900)
    first = sampler.sample()
    _stat(proc, 130, 970)

    stats = sampler.sample()

    assert first.cpu_percent is None
    assert stats.cpu_percent == pytest.approx(30.0)
    assert stats.cpu_temperature_c == pytest.approx(51.5)
    assert (stats.ram_used, stats.ram_total) == (750 * 1024, 1000 * 1024)
    assert stats.gpu == Gpu(41, 10 * GIB, 20 * GIB, 67.0)
    assert sampler.specs().cpu_model == "Test CPU 9000"


def test_missing_files_leave_their_figures_out(tmp_path: Path) -> None:
    stats = Sampler(tmp_path / "proc", tmp_path / "sys").sample()

    assert stats == Stats(None, None, None, None, None)


# The toolbar


def test_the_toolbar_shows_each_figure_it_has() -> None:
    stats = Stats(7.5, 53.0, 33 * GIB, 60 * GIB, Gpu(41, 10 * GIB, 20 * GIB, 67.0))
    use = ContextUse(9100, 24576, 5100, 2400, 12000, 1, summarized=False)

    line = _plain(toolbar("", stats, use))

    for part in ("CPU 8% 53°", "GPU 41% 67°", "VRAM 10.0/20.0G", "RAM 33.0/60.0G", "9.1k/24.6k"):
        assert part in line
    assert "compacts soon" not in line


def test_the_toolbar_fits_a_narrow_window_on_one_line() -> None:
    stats = Stats(7.5, 53.0, 33 * GIB, 60 * GIB, Gpu(41, 10 * GIB, 20 * GIB, 67.0))
    use = ContextUse(9100, 24576, 5100, 2400, 12000, 1, summarized=False)

    wide = toolbar("", stats, use, width=200)
    narrow = toolbar("", stats, use, width=74)
    narrower = toolbar("", stats, use, width=66)

    assert "°" in _plain(wide)
    assert narrow.cell_len <= 74
    for part in ("CPU 8%", "GPU 41%", "VRAM 10.0/20.0G", "RAM 33.0/60.0G", "9.1k/24.6k"):
        assert part in _plain(narrow)
    assert narrower.cell_len <= 66
    for part in ("CPU 8%", "GPU 41%", "VRAM 10/20G", "RAM 33/60G", "9.1k/24.6k"):
        assert part in _plain(narrower)


def test_the_toolbar_warns_before_the_conversation_is_compacted() -> None:
    use = ContextUse(20000, 24576, 5100, 11000, 12000, 9, summarized=False)

    assert "compacts soon" in _plain(toolbar("", None, use))


def test_the_toolbar_leaves_out_what_it_does_not_know() -> None:
    assert not _plain(toolbar("", Stats(None, None, None, None, None), None))


@pytest.mark.parametrize(("count", "shown"), [(950, "950"), (7600, "7.6k"), (24576, "24.6k")])
def test_token_counts_read_short(count: int, shown: str) -> None:
    assert tokens_short(count) == shown


def test_a_meter_fills_to_its_share_and_never_past_it() -> None:
    assert _plain(meter(0.5, 10)).count("━") == 10
    assert len(meter(1.7, 10)) == 10
    assert meter(0.3, 10).spans[0].style == "green"
    assert meter(0.95, 10).spans[0].style == "red"


# Entries


def test_a_steps_details_are_quiet_beside_its_bold_name() -> None:
    console = Console(width=120, file=io.StringIO(), color_system="standard")
    shown = Shown("", "step", "Web search", note='"red pandas"')

    segments = list(console.render(entry_renderable(shown)))

    name = next(s for s in segments if "Web search" in s.text)
    detail = next(s for s in segments if "red pandas" in s.text)
    assert name.style is not None
    assert name.style.bold
    assert detail.style is None or not detail.style.bold
