"""Process watches: a PID find_processes listed, read from a fake /proc, said once per change."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from test_pc_processes import NOW, OWNER, SELF, TICKS, FakeProc
from test_pc_screenshot import Grim
from test_pc_watch import FakeJudge, Time
from test_pc_windows import FakeDesktop

from jarvis.audit import AuditLog
from jarvis.operations import OperationError, Result, request_scope, validate_args
from jarvis.pc.processes import ProcessOps
from jarvis.pc.screenshot import ScreenshotOps
from jarvis.pc.watch import MAX_PROCESS_WATCHES, MAX_WATCHES, STUCK_S, ScreenWatch
from jarvis.pc.windows import WindowOps

if TYPE_CHECKING:
    from pathlib import Path


GAME = 4242


def _ops(proc: FakeProc) -> ProcessOps:
    return ProcessOps(
        proc=proc.root, uid=OWNER, self_pid=SELF, windows=list, clock=lambda: NOW, ticks_per_s=TICKS
    )


def _find(ops: ProcessOps, name: str) -> str:
    op = next(o for o in ops.operations() if o.name == "find_processes")
    return op.run(validate_args(op, {"name": name})).text


class Rig:
    def __init__(self, tmp_path: Path, *, sight: bool = True) -> None:
        super().__init__()
        self.proc = FakeProc(tmp_path / "proc")
        self.proc.add(GAME, "somegame", argv=["/opt/somegame/somegame"])
        self.processes: ProcessOps = _ops(self.proc)
        desktop = FakeDesktop(open=[], showing="1")
        windows = WindowOps(desktop)
        shots = ScreenshotOps(
            windows,
            tmp_path,
            capture=Grim(desktop),
            copy=lambda _d: True,
            now=lambda: datetime.now(UTC),
        )
        self.judge = FakeJudge(sight=sight)
        self.clock = Time()
        self.audit = AuditLog(None)
        self.watch = ScreenWatch(windows, shots, self.processes, clock=self.clock, audit=self.audit)
        self.watch.attach(self.judge)

    def start(self, **args: Any) -> Result:  # ruff: ignore[any-type] - the arguments a test sets
        op = next(o for o in self.watch.operations() if o.name == "pc_watch")
        return self.watch.start(validate_args(op, {"of": "process", **args}))

    def listed_start(self, pid: int = GAME, **args: Any) -> Result:  # ruff: ignore[any-type] - the arguments a test sets
        with request_scope(foreign=False):
            _find(self.processes, "somegame")
            return self.start(pid=pid, **args)

    def tick(self, seconds: float = 20) -> list[str]:
        self.clock.now += seconds
        return self.watch.check_due()

    def set_state(self, state: str) -> None:
        self.proc.add(GAME, "somegame", argv=["/opt/somegame/somegame"], state=state)


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def test_a_listed_process_is_watched_without_pictures_or_the_model(tmp_path: Path) -> None:
    rig = Rig(tmp_path, sight=False)  # no picture reader: a process watch needs none

    result = rig.listed_start()

    assert "somegame (PID 4242)" in result.text
    assert rig.tick() == []
    assert rig.judge.shown == []


def test_the_pid_must_come_from_find_processes_in_this_request(rig: Rig) -> None:
    with request_scope(foreign=False), pytest.raises(OperationError, match="not listed"):
        rig.start(pid=GAME)


def test_never_from_pc_audio_or_a_job(rig: Rig) -> None:
    with request_scope(foreign=True), pytest.raises(OperationError, match="own request"):
        rig.start(pid=GAME)


def test_an_exit_is_said_once_and_ends_the_watch(rig: Rig) -> None:
    rig.listed_start()
    rig.proc.remove(GAME)

    (line,) = rig.tick()

    assert line == "Heads up: somegame (PID 4242) has exited: it closed, crashed or was killed."
    assert rig.tick() == []
    assert rig.watch.ready() is False


def test_a_new_process_on_the_same_pid_counts_as_an_exit(rig: Rig) -> None:
    rig.listed_start()
    rig.proc.add(GAME, "other", start=900_000)

    assert "has exited" in rig.tick()[0]


def test_a_zombie_counts_as_exited(rig: Rig) -> None:
    rig.listed_start()
    rig.set_state("Z")

    assert "has exited" in rig.tick()[0]


def test_stopped_and_running_again_are_each_said_once(rig: Rig) -> None:
    rig.listed_start()
    rig.set_state("T")

    assert rig.tick() == [
        "Heads up: somegame (PID 4242) is stopped, frozen until something resumes it."
    ]
    assert rig.tick() == []
    rig.set_state("S")
    assert rig.tick() == ["somegame (PID 4242) is running again."]
    assert rig.tick() == []


def test_waiting_on_a_disk_counts_as_stuck_only_after_a_while(rig: Rig) -> None:
    rig.listed_start(every_s=10)
    rig.set_state("D")

    assert rig.tick(10) == []
    assert rig.tick(STUCK_S - 10) == []
    assert "stuck for 30 seconds" in rig.tick(10)[0]
    rig.set_state("R")
    assert rig.tick(10) == ["somegame (PID 4242) is running again."]


def test_a_short_wait_on_a_disk_says_nothing(rig: Rig) -> None:
    rig.listed_start(every_s=10)
    rig.set_state("D")
    assert rig.tick(10) == []
    rig.set_state("S")
    assert rig.tick(10) == []
    rig.set_state("D")
    assert rig.tick(STUCK_S - 10) == []  # the clock starts again with each wait


def test_the_same_process_is_not_watched_twice(rig: Rig) -> None:
    rig.listed_start()
    with pytest.raises(OperationError, match="already watched"):
        rig.listed_start()


def test_process_watches_have_their_own_cap(rig: Rig) -> None:
    for offset in range(1, MAX_PROCESS_WATCHES):
        rig.proc.add(GAME + offset, "somegame", argv=["/opt/somegame/somegame"])
    with request_scope(foreign=False):
        _find(rig.processes, "somegame")
        for offset in range(MAX_PROCESS_WATCHES):
            rig.start(pid=GAME + offset)
    rig.proc.add(GAME + 99, "somegame", argv=["/opt/somegame/somegame"])
    with pytest.raises(OperationError, match="process watches already run"):
        rig.listed_start(pid=GAME + 99)
    assert MAX_PROCESS_WATCHES > MAX_WATCHES  # a process watch uses no GPU


def test_it_ends_after_for_min(rig: Rig) -> None:
    rig.listed_start(for_min=1)

    assert rig.tick(61) == ["My watch on somegame (PID 4242) is over."]
