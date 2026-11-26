"""Process watches: a PID find_processes listed, read from a fake /proc, said once per change."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from test_pc_processes import NOW, OWNER, SELF, TICKS, FakeProc
from test_pc_screenshot import Grim
from test_pc_watch import FakeJudge, Time
from test_pc_windows import FakeDesktop

from jarvis.audit import AuditLog
from jarvis.operations import OperationError, Result, current_request, request_scope, validate_args
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


def _read(ops: ProcessOps, name: str) -> str:
    """List the processes, then go on as the model's next message would, having read the list."""
    text = _find(ops, name)
    request = current_request()
    assert request is not None
    request.next_message()
    return text


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
            _read(self.processes, "somegame")
            return self.start(pid=pid, **args)

    def tick(self, seconds: float = 20) -> list[str]:
        self.clock.now += seconds
        return self.watch.check_due()

    def set_state(self, state: str, threads: int = 1) -> None:
        self.proc.add(
            GAME, "somegame", argv=["/opt/somegame/somegame"], state=state, threads=threads
        )


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
        _read(rig.processes, "somegame")
        for offset in range(MAX_PROCESS_WATCHES):
            rig.start(pid=GAME + offset)
    rig.proc.add(GAME + 99, "somegame", argv=["/opt/somegame/somegame"])
    with pytest.raises(OperationError, match="process watches already run"):
        rig.listed_start(pid=GAME + 99)


def test_it_ends_after_for_min(rig: Rig) -> None:
    rig.listed_start(for_min=1)

    assert rig.tick(61) == ["My watch on somegame (PID 4242) is over."]


def test_a_pid_written_beside_its_listing_is_refused(rig: Rig) -> None:
    with request_scope(foreign=False):
        _find(rig.processes, "somegame")  # the model has not read the list yet
        with pytest.raises(OperationError, match="not read yet"):
            rig.start(pid=GAME)


def test_a_scheduled_job_sets_no_process_watch(rig: Rig) -> None:
    with request_scope(foreign=False, scheduled=True), pytest.raises(OperationError, match="job"):
        rig.start(pid=GAME)


def test_outside_text_refuses_a_process_watch_in_words_about_watching(rig: Rig) -> None:
    with request_scope(foreign=False) as request:
        request.saw("web")
        _read(rig.processes, "somegame")
        with pytest.raises(OperationError, match="outside") as refused:
            rig.start(pid=GAME)
    assert "watch" in str(refused.value)
    assert "stop" not in str(refused.value)


def test_a_listed_process_that_ended_before_the_watch_is_refused(rig: Rig) -> None:
    with request_scope(foreign=False):
        _read(rig.processes, "somegame")
        rig.proc.remove(GAME)
        with pytest.raises(OperationError, match="no longer"):
            rig.start(pid=GAME)


def test_two_screen_watches_and_the_process_watches_run_together(rig: Rig) -> None:
    for offset in range(1, MAX_PROCESS_WATCHES):
        rig.proc.add(GAME + offset, "somegame", argv=["/opt/somegame/somegame"])
    with request_scope(foreign=False):
        for _ in range(MAX_WATCHES):
            rig.watch.start(
                {
                    "of": "screen",
                    "condition": "a dialog",
                    "every_s": 20,
                    "for_min": 60,
                    "repeat": True,
                }
            )
        _read(rig.processes, "somegame")
        for offset in range(MAX_PROCESS_WATCHES):
            rig.start(pid=GAME + offset)

    listed = rig.watch.list_watches({}).text.splitlines()
    assert len(listed) == MAX_WATCHES + MAX_PROCESS_WATCHES


def test_a_stop_breaks_the_wait_on_a_disk(rig: Rig) -> None:
    rig.listed_start(every_s=10)
    rig.set_state("D")
    assert rig.tick(10) == []
    rig.set_state("T")
    assert "is stopped" in rig.tick(10)[0]
    rig.set_state("D")
    assert rig.tick(10) == ["somegame (PID 4242) is running again."]  # resumed, if waiting
    assert rig.tick(10) == []  # the wait began again at the last check, not the first
    assert rig.tick(10) == []
    assert "stuck for 30 seconds" in rig.tick(10)[0]


def test_stuck_needs_three_checks_in_a_row_whatever_the_pace(rig: Rig) -> None:
    rig.listed_start(every_s=300)
    rig.set_state("D")
    assert rig.tick(300) == []
    assert rig.tick(300) == []  # two looks 300 s apart prove no wait of 300 s
    assert "stuck for 600 seconds" in rig.tick(300)[0]


def test_a_main_thread_that_ended_while_others_run_is_not_an_exit(rig: Rig) -> None:
    rig.listed_start()
    rig.set_state("Z", threads=4)

    assert rig.tick() == []
    rig.set_state("Z", threads=1)
    assert "has exited" in rig.tick()[0]
