"""tools/idle_check.py on a fake /proc tree with an injected clock (no real time passes)."""

import subprocess
from typing import TYPE_CHECKING

import idle_check
import pytest
from idle_check import Host, Limits, Root

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

TICKS = 100  # clock ticks per second on the fake system
LIMITS = Limits()


class FakeProc:
    """A /proc tree in a temporary folder, with the files idle_check reads."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root

    def process(
        self,
        pid: int,
        *,
        comm: str = "worker",
        cpu_ticks: int = 0,
        rss_kib: int = 1024,
        children: Sequence[int] = (),
    ) -> None:
        folder = self.root / str(pid)
        (folder / "task" / str(pid)).mkdir(parents=True, exist_ok=True)
        self.set_cpu(pid, cpu_ticks, comm=comm)
        self.set_rss(pid, rss_kib)
        (folder / "task" / str(pid) / "children").write_text(
            " ".join(str(child) for child in children), encoding="ascii"
        )
        (folder / "fdinfo").mkdir(exist_ok=True)

    def set_cpu(
        self, pid: int, ticks: int, *, comm: str = "worker", reaped: int = 0, ppid: int = 1
    ) -> None:
        # Fields 3 to 17: state, ppid, pgrp, session, tty_nr, tpgid, flags, minflt, cminflt, majflt,
        # cmajflt, utime, stime, cutime, cstime. Half the ticks go to user time, half to system
        # time; `reaped` is the time of children this process has waited for (cutime + cstime).
        user, system = ticks - ticks // 2, ticks // 2
        c_user, c_system = reaped - reaped // 2, reaped // 2
        after_comm = (
            f"S {ppid} 1 1 0 -1 0 0 0 0 0 {user} {system} {c_user} {c_system} 20 0 1 0 12345"
        )
        (self.root / str(pid) / "stat").write_text(
            f"{pid} ({comm}) {after_comm}\n", encoding="utf-8"
        )

    def set_rss(self, pid: int, kib: int) -> None:
        (self.root / str(pid) / "status").write_text(
            f"Name:\tworker\nVmSize:\t999999 kB\nVmRSS:\t{kib} kB\nThreads:\t1\n", encoding="utf-8"
        )

    def set_children(self, pid: int, children: Sequence[int], *, thread: int | None = None) -> None:
        task = self.root / str(pid) / "task" / str(thread if thread is not None else pid)
        task.mkdir(parents=True, exist_ok=True)
        (task / "children").write_text(" ".join(str(c) for c in children), encoding="ascii")

    def drm_fd(
        self,
        pid: int,
        fd: int,
        *,
        client: int,
        engines: dict[str, int],
        pdev: str = "0000:03:00.0",
        capacity: dict[str, int] | None = None,
    ) -> None:
        lines = ["pos:\t0", "flags:\t02000002", "drm-driver:\tamdgpu", f"drm-pdev:\t{pdev}"]
        lines.append(f"drm-client-id:\t{client}")
        lines += [f"drm-engine-{name}:\t{ns} ns" for name, ns in engines.items()]
        lines += [f"drm-engine-capacity-{name}:\t{n}" for name, n in (capacity or {}).items()]
        lines.append("drm-memory-vram:\t1024 KiB")
        (self.root / str(pid) / "fdinfo" / str(fd)).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def plain_fd(self, pid: int, fd: int) -> None:
        (self.root / str(pid) / "fdinfo" / str(fd)).write_text(
            "pos:\t0\nflags:\t02\n", encoding="utf-8"
        )

    def remove(self, pid: int) -> None:
        for path in sorted((self.root / str(pid)).rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        (self.root / str(pid)).rmdir()


class FakeClock:
    """A clock that only moves when `sleep` is called, running `during` once in the middle."""

    def __init__(self, during: Callable[[], None] | None = None) -> None:
        super().__init__()
        self.now = 1000.0
        self.slept: list[float] = []
        self._during = during

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        if self._during is not None:
            self._during()
        self.now += seconds


@pytest.fixture
def proc(tmp_path: Path) -> FakeProc:
    root = tmp_path / "proc"
    root.mkdir()
    return FakeProc(root)


def _measure(
    proc: FakeProc,
    roots: Sequence[Root],
    during: Callable[[], None] | None = None,
    seconds: float = 30.0,
    limits: Limits = LIMITS,
) -> tuple[list[idle_check.Result], FakeClock]:
    clock = FakeClock(during)
    drm = proc.root / "drm"  # no graphics card: the wait is one sleep
    host = Host(proc=proc.root, clock=clock, sleep=clock.sleep, clk_tck=TICKS, drm=drm)
    return idle_check.measure(roots, seconds, limits, host), clock


def test_an_idle_process_passes_and_the_window_is_the_injected_clock(proc: FakeProc) -> None:
    proc.process(100, cpu_ticks=500, rss_kib=2048 * 1024)

    def tick() -> None:
        proc.set_cpu(100, 500 + 30)  # 0.3 s of CPU over 30 s: 1 % of a core

    (result,), clock = _measure(proc, [Root("svc", 100)], tick)

    assert clock.slept == [30.0]
    assert result.problems == []
    assert result.seconds == pytest.approx(30.0)
    assert result.cpu_percent == pytest.approx(1.0)
    assert result.rss_start_mib == pytest.approx(2048.0)
    assert result.gpu_percent == {}


def test_cpu_above_the_limit_fails(proc: FakeProc) -> None:
    proc.process(100, cpu_ticks=0)

    (result,), _ = _measure(proc, [Root("svc", 100)], lambda: proc.set_cpu(100, 300))  # 10 %

    assert result.cpu_percent == pytest.approx(10.0)
    assert [p for p in result.problems if p.startswith("CPU 10.0 %")]


def test_the_cpu_limit_is_a_parameter(proc: FakeProc) -> None:
    proc.process(100)

    (result,), _ = _measure(
        proc, [Root("svc", 100)], lambda: proc.set_cpu(100, 300), limits=Limits(max_cpu_percent=20)
    )

    assert result.problems == []


def test_a_command_name_with_spaces_and_parentheses_is_parsed(proc: FakeProc) -> None:
    proc.process(100, comm="odd ) (name", cpu_ticks=0)

    (result,), _ = _measure(
        proc, [Root("svc", 100)], lambda: proc.set_cpu(100, 300, comm="odd ) (name")
    )

    assert result.cpu_percent == pytest.approx(10.0)


def test_descendants_count_towards_the_root(proc: FakeProc) -> None:
    proc.process(100, cpu_ticks=0, children=[200])
    proc.process(200, cpu_ticks=0, children=[300])
    proc.process(300, cpu_ticks=0)
    proc.process(999, cpu_ticks=0)  # not in the tree

    def work() -> None:
        proc.set_cpu(200, 150)
        proc.set_cpu(300, 150)
        proc.set_cpu(999, 100_000)

    (result,), _ = _measure(proc, [Root("svc", 100)], work)

    assert result.processes == 3
    assert result.cpu_percent == pytest.approx(10.0)


def test_children_of_any_thread_are_found(proc: FakeProc) -> None:
    proc.process(100, cpu_ticks=0)
    proc.set_children(100, [200], thread=101)
    proc.process(200, cpu_ticks=0)

    (result,), _ = _measure(proc, [Root("svc", 100)], lambda: proc.set_cpu(200, 300))

    assert result.processes == 2
    assert result.cpu_percent == pytest.approx(10.0)


def test_a_child_started_during_the_window_counts_in_full(proc: FakeProc) -> None:
    proc.process(100, cpu_ticks=0)

    def spawn() -> None:
        proc.set_children(100, [200])
        proc.process(200, cpu_ticks=300)

    (result,), _ = _measure(proc, [Root("svc", 100)], spawn)

    assert result.cpu_percent == pytest.approx(10.0)


def test_a_child_that_exits_during_the_window_counts_through_its_parent(proc: FakeProc) -> None:
    proc.process(100, cpu_ticks=100, children=[200])
    proc.process(200)
    proc.set_cpu(200, 5000, ppid=100)

    def finish() -> None:  # the child works 300 more ticks, exits and is waited for
        proc.set_children(100, [])
        proc.remove(200)
        proc.set_cpu(100, 130, reaped=5300)

    (result,), _ = _measure(proc, [Root("svc", 100)], finish)

    assert result.cpu_percent == pytest.approx(11.0)


def test_short_lived_children_never_seen_still_count(proc: FakeProc) -> None:
    """A loop that starts a busy program over and over shows only in the parent's reaped time."""
    proc.process(100, cpu_ticks=100)

    (result,), _ = _measure(proc, [Root("svc", 100)], lambda: proc.set_cpu(100, 100, reaped=600))

    assert result.cpu_percent == pytest.approx(20.0)
    assert result.problems


def test_a_child_that_exits_does_not_count_negatively(proc: FakeProc) -> None:
    proc.process(100, cpu_ticks=100, children=[200])
    proc.process(200, cpu_ticks=5000)

    def finish() -> None:
        proc.set_children(100, [])
        proc.remove(200)
        proc.set_cpu(100, 130)

    (result,), _ = _measure(proc, [Root("svc", 100)], finish)

    assert result.cpu_percent == pytest.approx(1.0)


def test_rss_growth_above_the_limit_fails_and_shrinking_passes(proc: FakeProc) -> None:
    proc.process(100, rss_kib=100 * 1024, children=[200])
    proc.process(200, rss_kib=100 * 1024)

    def grow() -> None:
        proc.set_rss(100, 100 * 1024 + 40 * 1024)
        proc.set_rss(200, 100 * 1024 + 20 * 1024)

    (grown,), _ = _measure(proc, [Root("svc", 100)], grow)  # +60 MiB over the tree

    assert grown.rss_start_mib == pytest.approx(200.0)
    assert grown.rss_end_mib == pytest.approx(260.0)
    assert [p for p in grown.problems if p.startswith("RSS grew 60.0 MiB")]

    proc.set_rss(100, 100 * 1024)
    proc.set_rss(200, 100 * 1024)
    (shrunk,), _ = _measure(proc, [Root("svc", 100)], lambda: proc.set_rss(200, 1024))
    assert shrunk.problems == []


def test_gpu_busy_time_is_reported_per_engine(proc: FakeProc) -> None:
    proc.process(100)
    proc.drm_fd(100, 7, client=11, engines={"gfx": 1_000_000_000, "compute": 0})

    def render() -> None:
        # gfx busy for 0.15 s of 30 s: 0.5 %; compute busy 6 s: 20 %.
        proc.drm_fd(100, 7, client=11, engines={"gfx": 1_150_000_000, "compute": 6_000_000_000})

    (result,), _ = _measure(proc, [Root("svc", 100)], render)

    assert result.gpu_percent == {"compute": pytest.approx(20.0), "gfx": pytest.approx(0.5)}
    assert [p for p in result.problems if p.startswith("GPU engine compute 20.0 %")]
    assert not [p for p in result.problems if "gfx" in p]


def test_a_gpu_engine_at_the_limit_passes_and_above_it_fails(proc: FakeProc) -> None:
    proc.process(100)
    proc.drm_fd(100, 7, client=11, engines={"gfx": 0})

    (quiet,), _ = _measure(
        proc,
        [Root("svc", 100)],
        lambda: proc.drm_fd(100, 7, client=11, engines={"gfx": 300_000_000}),  # exactly 1 %
    )
    (busy,), _ = _measure(
        proc,
        [Root("svc", 100)],
        lambda: proc.drm_fd(
            100, 7, client=11, engines={"gfx": 900_000_000}
        ),  # 2 % more than before
    )

    assert quiet.problems == []
    assert [p for p in busy.problems if p.startswith("GPU engine gfx")]


def test_one_drm_client_on_several_descriptors_and_processes_counts_once(proc: FakeProc) -> None:
    proc.process(100, children=[200])
    proc.process(200)
    for pid, fd in ((100, 7), (100, 8), (200, 9)):
        proc.drm_fd(pid, fd, client=11, engines={"gfx": 0})

    def render() -> None:
        for pid, fd in ((100, 7), (100, 8), (200, 9)):
            proc.drm_fd(pid, fd, client=11, engines={"gfx": 600_000_000})  # 2 % once, not 6 %

    (result,), _ = _measure(proc, [Root("svc", 100)], render)

    assert result.gpu_percent["gfx"] == pytest.approx(2.0)


def test_separate_drm_clients_and_devices_add_up(proc: FakeProc) -> None:
    proc.process(100)
    proc.drm_fd(100, 7, client=1, engines={"gfx": 0})
    proc.drm_fd(100, 8, client=2, engines={"gfx": 0})
    proc.drm_fd(100, 9, client=1, engines={"gfx": 0}, pdev="0000:0a:00.0")  # same id, other card

    def render() -> None:
        proc.drm_fd(100, 7, client=1, engines={"gfx": 300_000_000})
        proc.drm_fd(100, 8, client=2, engines={"gfx": 300_000_000})
        proc.drm_fd(100, 9, client=1, engines={"gfx": 300_000_000}, pdev="0000:0a:00.0")

    (result,), _ = _measure(proc, [Root("svc", 100)], render)

    assert result.gpu_percent["gfx"] == pytest.approx(3.0)


def test_engine_capacity_lines_and_non_drm_descriptors_are_ignored(proc: FakeProc) -> None:
    proc.process(100)
    proc.plain_fd(100, 3)
    proc.drm_fd(100, 7, client=11, engines={"gfx": 0}, capacity={"gfx": 2})

    (result,), _ = _measure(proc, [Root("svc", 100)])

    assert result.gpu_percent == {"gfx": 0.0}
    assert result.problems == []


def test_a_client_opened_during_the_window_counts_in_full(proc: FakeProc) -> None:
    proc.process(100)

    (result,), _ = _measure(
        proc,
        [Root("svc", 100)],
        lambda: proc.drm_fd(100, 7, client=5, engines={"gfx": 600_000_000}),
    )

    assert result.gpu_percent["gfx"] == pytest.approx(2.0)
    assert result.problems


def test_a_root_that_vanishes_fails(proc: FakeProc) -> None:
    proc.process(100)

    (result,), _ = _measure(proc, [Root("svc", 100)], lambda: proc.remove(100))

    assert result.problems == ["exited during the measurement"]


def test_a_root_that_is_not_running_fails(proc: FakeProc) -> None:
    (result,), _ = _measure(proc, [Root("svc", 100)])

    assert result.problems == ["not running at the start"]


def test_every_root_is_judged_on_its_own(proc: FakeProc) -> None:
    proc.process(100)
    proc.process(200)

    results, _ = _measure(
        proc, [Root("calm", 100), Root("busy", 200)], lambda: proc.set_cpu(200, 600)
    )

    assert next(r.problems for r in results) == []
    assert results[1].cpu_percent == pytest.approx(20.0)
    assert results[1].problems


# --- command line -------------------------------------------------------------------------------


def _main(
    proc: FakeProc,
    *argv: str,
    during: Callable[[], None] | None = None,
    run: idle_check.Run | None = None,
    drm: Path | None = None,
) -> tuple[int, FakeClock]:
    clock = FakeClock(during)
    status = idle_check.main(
        list(argv),
        proc=proc.root,
        clock=clock,
        sleep=clock.sleep,
        run=run or subprocess.run,
        drm=drm or proc.root / "drm",
    )
    return status, clock


def test_main_reports_an_idle_pid_and_exits_zero(
    proc: FakeProc, capsys: pytest.CaptureFixture[str]
) -> None:
    proc.process(100, rss_kib=512 * 1024)

    status, clock = _main(proc, "--pid", "100", "--seconds", "30")

    out = capsys.readouterr().out
    assert status == 0
    assert clock.slept == [30.0]
    assert "pid 100 (pid 100)" in out
    assert "cpu: 0.0 % of one core (limit 5 %)" in out
    assert "gpu: no GPU engine time published" in out
    assert "rss: 512.0 MiB -> 512.0 MiB" in out
    assert "ok: idle" in out


def test_main_exits_one_and_says_why_when_busy(
    proc: FakeProc, capsys: pytest.CaptureFixture[str]
) -> None:
    proc.process(100)

    status, _ = _main(proc, "--pid", "100", during=lambda: proc.set_cpu(100, 3000))

    assert status == 1
    assert "FAIL: CPU 50.0 % of one core is above 5 %" in capsys.readouterr().out


def test_main_defaults_to_sixty_seconds(proc: FakeProc) -> None:
    proc.process(100)

    _, clock = _main(proc, "--pid", "100")

    assert clock.slept == [60.0]


def test_main_takes_the_limits_from_the_command_line(proc: FakeProc) -> None:
    proc.process(100)
    proc.drm_fd(100, 7, client=1, engines={"gfx": 0})

    def work() -> None:
        proc.set_cpu(100, 300)
        proc.set_rss(100, 1024 + 5 * 1024)
        proc.drm_fd(100, 7, client=1, engines={"gfx": 600_000_000})

    strict, _ = _main(
        proc, "--pid", "100", "--seconds", "30", "--max-rss-growth-mib", "1", during=work
    )
    proc.set_cpu(100, 0)
    proc.set_rss(100, 1024)
    proc.drm_fd(100, 7, client=1, engines={"gfx": 0})
    lenient, _ = _main(
        proc,
        "--pid", "100",
        "--seconds", "30",
        "--max-cpu-percent", "20",
        "--max-rss-growth-mib", "10",
        "--max-gpu-percent", "5",
        during=work,
    )  # fmt: skip

    assert strict == 1
    assert lenient == 0


def test_main_measures_several_pids_and_a_unit(
    proc: FakeProc, capsys: pytest.CaptureFixture[str]
) -> None:
    proc.process(100)
    proc.process(200)
    proc.process(300)
    calls: list[list[str]] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="300\n", stderr="")

    status, _ = _main(proc, "--unit", "jarvis.service", "--pid", "100", "--pid", "200", run=run)

    out = capsys.readouterr().out
    assert status == 0
    assert calls[0][1:] == ["--user", "show", "-p", "MainPID", "--value", "jarvis.service"]
    assert (
        out.index("jarvis.service (pid 300)")
        < out.index("pid 100 (pid 100)")
        < out.index("pid 200 (pid 200)")
    )


def test_main_needs_a_unit_or_a_pid(proc: FakeProc, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        _main(proc)

    assert stopped.value.code == 2
    assert "at least one --unit or --pid" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--pid", "0"],
        ["--pid", "abc"],
        ["--pid", "1", "--seconds", "0"],
        ["--pid", "1", "--seconds", "-5"],
        ["--unit", "--now"],
        ["--pid", "1", "--max-cpu-percent", "0"],
    ],
)
def test_main_rejects_bad_arguments(proc: FakeProc, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        _main(proc, *argv)

    assert stopped.value.code == 2


def test_main_exits_two_when_a_unit_is_not_running(
    proc: FakeProc, capsys: pytest.CaptureFixture[str]
) -> None:
    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="0\n", stderr="")

    status, clock = _main(proc, "--unit", "jarvis.service", run=run)

    assert status == 2
    assert clock.slept == []
    assert "jarvis.service has no running main process" in capsys.readouterr().err


def test_main_pid_reports_a_systemctl_failure() -> None:
    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="Failed to connect to bus")

    with pytest.raises(idle_check.IdleCheckError, match="Failed to connect to bus"):
        idle_check.main_pid("jarvis.service", run)


def test_main_pid_reads_the_pid() -> None:
    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="4242\n", stderr="")

    assert idle_check.main_pid("jarvis.service", run) == 4242


def test_a_drm_client_without_engine_counters_is_not_reported_as_no_client(
    proc: FakeProc, capsys: pytest.CaptureFixture[str]
) -> None:
    """ROCm compute goes through /dev/kfd; amdgpu then lists the client with no drm-engine lines."""
    proc.process(100)
    proc.drm_fd(100, 8, client=13783, engines={})

    _main(proc, "--pid", "100", "--seconds", "30")

    out = capsys.readouterr().out
    assert (
        "gpu: no GPU engine time published (no DRM client, or one without drm-engine counters)"
        in out
    )
    # Its own GPU use was not seen, so the verdict must not read as a clean bill for the GPU.
    assert "ok: idle on CPU and memory; its GPU use was not measured" in out
    assert "  ok: idle\n" not in out


def test_a_process_whose_gpu_time_was_read_gets_a_plain_ok(
    proc: FakeProc, capsys: pytest.CaptureFixture[str]
) -> None:
    proc.process(100)
    proc.drm_fd(100, 8, client=1, engines={"gfx": 0})

    _main(proc, "--pid", "100", "--seconds", "30")

    assert "  ok: idle\n" in capsys.readouterr().out + "\n"


def _card(drm: Path, name: str, busy: int) -> Path:
    device = drm / name / "device"
    device.mkdir(parents=True)
    (device / "gpu_busy_percent").write_text(f"{busy}\n", encoding="utf-8")
    return device / "gpu_busy_percent"


def test_the_whole_cards_busy_share_is_sampled_over_the_window(
    proc: FakeProc, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """KFD compute publishes no per-process time; the card's own busy share is the next best."""
    drm = tmp_path / "drm"
    busy = _card(drm, "card1", 0)
    _card(drm, "card1-DP-1", 99)  # a connector, not a card
    readings = iter([10, 30, 20])
    proc.process(100)

    def tick() -> None:
        busy.write_text(f"{next(readings, 20)}\n", encoding="utf-8")

    status, clock = _main(proc, "--pid", "100", "--seconds", "3", during=tick, drm=drm)

    out = capsys.readouterr().out
    assert status == 0  # the whole card is every program's, so it never fails the check
    assert clock.slept == [1.0, 1.0, 1.0]
    assert "card1 20.0 % busy on average" in out
    assert "DP-1" not in out


def test_a_process_that_leaves_the_tree_does_not_hide_the_trees_work(proc: FakeProc) -> None:
    """A grandchild re-parented to systemd took its earlier time with it; nothing reaped it here."""
    proc.process(100, children=[200])
    proc.process(200, cpu_ticks=10, children=[300])
    proc.set_cpu(200, 10, ppid=100)
    proc.process(300, cpu_ticks=3000)
    proc.set_cpu(300, 3000, ppid=200)

    def finish() -> None:  # 200 exits and is reaped by 100; 300 moves to systemd; 100 reaps 2100
        proc.set_children(100, [])
        proc.remove(200)
        proc.remove(300)
        proc.set_cpu(100, 0, reaped=10 + 2100)

    (result,), _ = _measure(proc, [Root("svc", 100)], finish)

    assert result.cpu_percent == pytest.approx(70.0)
