"""The screen watch tools: capture, clock and model faked, the compositor never asked to act."""

import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import pytest
from test_pc_processes import OWNER, TICKS, FakeProc
from test_pc_screenshot import Grim
from test_pc_windows import FakeDesktop, make_window, seen

from jarvis.agent import PICTURES_OFF
from jarvis.audit import AuditLog
from jarvis.operations import ArgumentError, OperationError, Result, request_scope, validate_args
from jarvis.pc.processes import ProcessOps
from jarvis.pc.screenshot import ScreenshotOps
from jarvis.pc.watch import MAX_SAY, MAX_WATCHES, NEEDS_PICTURES, ScreenWatch
from jarvis.pc.windows import WindowOps

if TYPE_CHECKING:
    from jarvis.pictures import Picture


@dataclass
class FakeJudge:
    """Answers each glance from a script (then no), and says what it was shown."""

    answers: list[bool] = field(default_factory=list[bool])
    sight: bool = True
    shown: list[tuple[int, str]] = field(default_factory=list[tuple[int, str]])
    broken: bool = False

    def sees(self) -> bool:
        return self.sight

    def glance(self, picture: Picture, condition: str) -> bool:
        self.shown.append((max(picture.width, picture.height), condition))
        if self.broken:
            msg = "model server answered 500"
            raise OSError(msg)
        return self.answers.pop(0) if self.answers else False


class Time:
    def __init__(self) -> None:
        super().__init__()
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Frames(Grim):
    """Vary raw PNG bytes independently of the decoded picture."""

    frame: int | None = None

    @override
    def __call__(self, args: list[str]) -> bytes:
        data = super().__call__(args)
        frame = len(self.calls) if self.frame is None else self.frame
        return data + frame.to_bytes(4)


@dataclass
class Rig:
    desktop: FakeDesktop
    grim: Frames
    proc: FakeProc
    windows: WindowOps
    judge: FakeJudge
    clock: Time
    awake: Time
    audit: AuditLog
    watch: ScreenWatch

    def start(self, **args: Any) -> Result:  # ruff: ignore[any-type] - the arguments a test sets
        op = next(o for o in self.watch.operations() if o.name == "pc_watch")
        return self.watch.start(validate_args(op, args))

    def stop(self, **args: Any) -> Result:  # ruff: ignore[any-type] - the arguments a test sets
        return self.watch.stop(args)

    def prepare(self, **args: Any) -> dict[str, Any]:  # ruff: ignore[any-type] - the arguments a test sets
        op = next(o for o in self.watch.operations() if o.name == "pc_watch")
        assert op.prepare is not None
        return op.prepare(validate_args(op, args))

    def tick(self, seconds: float = 20, *, frame: int | None = None) -> list[str]:
        self.clock.now += seconds
        self.awake.now += seconds
        if frame is not None:
            self.grim.frame = frame
        return self.watch.check_due()

    def events(self) -> list[str]:
        return [str(e["event"]) for e in self.audit.events]


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    desktop = FakeDesktop(
        open=[
            make_window("0xa1", "kitty", "2", width=1280, height=1440),
            make_window("0xa2", "firefox", "2", x=1280, width=1280, height=1440, focused=True),
        ],
        showing="1",
    )
    grim = Frames(desktop)
    proc = FakeProc(tmp_path / "proc")
    processes = ProcessOps(proc=proc.root, uid=OWNER, ticks_per_s=TICKS)
    windows = WindowOps(desktop)
    shots = ScreenshotOps(
        windows, tmp_path, capture=grim, copy=lambda _d: True, now=lambda: datetime.now(UTC)
    )
    judge, clock, awake, audit = FakeJudge(), Time(), Time(), AuditLog(None)
    watch = ScreenWatch(windows, shots, processes, clock=clock, freeze_clock=awake, audit=audit)
    watch.attach(judge)
    return Rig(desktop, grim, proc, windows, judge, clock, awake, audit, watch)


def _listed_start(rig: Rig, **args: Any) -> Result:  # ruff: ignore[any-type] - the arguments a test sets
    with request_scope(foreign=False):
        seen(rig.windows)
        return rig.start(of="window", window=1, condition="the build finished", **args)


def test_a_watch_on_a_listed_window_starts_and_says_what_it_costs(rig: Rig) -> None:
    result = _listed_start(rig)

    assert "the kitty window" in result.text
    assert "the build finished" in result.text
    assert "20 s" in result.text
    assert "model call with a picture" in result.text
    assert rig.events() == ["watch_started"]
    assert rig.desktop.ran == []


def test_the_window_must_come_from_a_listing_the_model_read(rig: Rig) -> None:
    with request_scope(foreign=False), pytest.raises(OperationError, match="not listed"):
        rig.start(of="window", window=1, condition="x")
    with request_scope(foreign=False):
        rig.windows.list_windows({})  # written beside the call: not read yet
        with pytest.raises(OperationError, match="not read yet"):
            rig.start(of="window", window=1, condition="x")
    assert rig.watch.ready() is False


@pytest.mark.parametrize(
    ("args", "refusal"),
    [
        ({"of": "process"}, "Say which process"),
        ({"of": "window", "condition": "x"}, "Say which window"),
        ({"of": "window", "window": 1, "condition": "x"}, "not listed"),
        ({"of": "workspace", "condition": "x"}, "Say which workspace"),
        ({"of": "screen", "condition": "..."}, "Say what to look for"),
    ],
)
def test_a_watch_that_would_be_refused_is_refused_before_the_question(
    rig: Rig, args: dict[str, Any], refusal: str
) -> None:
    """Regression: a doomed watch reached the owner's question, then failed after the yes."""
    with request_scope(foreign=False), pytest.raises(OperationError, match=refusal):
        rig.prepare(**args)


def test_a_screen_watch_is_refused_before_the_question_when_the_model_cannot_see(
    rig: Rig,
) -> None:
    rig.judge.sight = False
    with request_scope(foreign=False), pytest.raises(OperationError, match="picture"):
        rig.prepare(of="screen", condition="a dialog")


def test_a_watch_that_can_start_passes_its_arguments_through_unchanged(rig: Rig) -> None:
    with request_scope(foreign=False):
        seen(rig.windows)
        bound = rig.prepare(of="window", window=1, condition="the build finished")
    assert bound["window"] == 1
    assert bound["condition"] == "the build finished"


def test_a_listing_of_another_request_does_not_bind(rig: Rig) -> None:
    with request_scope(foreign=False):
        seen(rig.windows)
    with request_scope(foreign=False), pytest.raises(OperationError, match="not listed"):
        rig.start(of="window", window=1, condition="x")


def test_the_screen_and_a_workspace_need_no_listing(rig: Rig) -> None:
    with request_scope(foreign=False):
        assert "the screen" in rig.start(of="screen", condition="a dialog").text
        assert "workspace 2" in rig.start(of="workspace", workspace="2", condition="x").text
        rig.stop()
        with pytest.raises(OperationError, match="which workspace"):
            rig.start(of="workspace", condition="x")


def test_at_most_two_watches_run_at_once(rig: Rig) -> None:
    with request_scope(foreign=False):
        for _ in range(MAX_WATCHES):
            rig.start(of="screen", condition="x")
        with pytest.raises(OperationError, match="Two watches"):
            rig.start(of="screen", condition="y")


def test_a_request_heard_in_pc_audio_sets_no_watch(rig: Rig) -> None:
    with request_scope(foreign=True), pytest.raises(OperationError, match="never from PC audio"):
        rig.start(of="screen", condition="x")
    assert rig.watch.ready() is False


def test_a_model_that_cannot_see_pictures_gets_no_watch(rig: Rig) -> None:
    rig.judge.sight = False

    with request_scope(foreign=False), pytest.raises(OperationError) as refused:
        rig.start(of="screen", condition="x")

    assert str(refused.value) == NEEDS_PICTURES
    assert PICTURES_OFF in NEEDS_PICTURES


def test_the_limits_are_in_the_schema() -> None:
    op = next(o for o in ScreenWatch.operations(_bare()) if o.name == "pc_watch")
    props = op.parameters["properties"]

    assert (props["every_s"]["minimum"], props["every_s"]["maximum"]) == (10, 300)
    assert props["every_s"]["default"] == 20
    assert (props["for_min"]["maximum"], props["for_min"]["default"]) == (720, 120)
    assert props["repeat"]["default"] is True
    assert op.default_class.value == "safe"
    with pytest.raises(ArgumentError, match="every_s"):
        validate_args(op, {"of": "screen", "condition": "x", "every_s": 5})


def _bare() -> ScreenWatch:
    windows = WindowOps(FakeDesktop())
    return ScreenWatch(windows, ScreenshotOps(windows, Path("/nonexistent")))


def test_a_check_captures_the_window_by_its_id_and_never_acts_on_the_desktop(rig: Rig) -> None:
    _listed_start(rig)
    rig.judge.answers = [False]

    assert rig.watch.check_due() == []

    assert rig.grim.calls == [["-T", "a1"]]
    assert rig.desktop.ran == []
    assert rig.judge.shown == [(1024, "the build finished")]  # shrunk to the shown size


def test_a_yes_is_said_in_a_fixed_line_built_from_the_condition(rig: Rig) -> None:
    _listed_start(rig)
    rig.judge.answers = [True]

    assert rig.watch.check_due() == ["Heads up: the build finished."]
    assert rig.events() == ["watch_started", "watch_announced"]


def test_checks_come_at_the_set_pace(rig: Rig) -> None:
    _listed_start(rig, every_s=30)
    assert rig.watch.ready() is True
    rig.watch.check_due()

    assert rig.watch.ready() is False
    rig.clock.now += 29
    assert rig.watch.ready() is False
    rig.clock.now += 1
    assert rig.watch.ready() is True


def test_after_a_yes_it_stays_quiet_for_the_cool_down_then_says_it_again(rig: Rig) -> None:
    _listed_start(rig, every_s=20)  # the cool-down is the longer of 60 s and three checks
    rig.judge.answers = [True, False, True, True]  # it went away and came back after the cool-down

    said = [rig.watch.check_due()]
    for _ in range(3):
        rig.clock.now += 20
        said.append(rig.watch.check_due())
    rig.clock.now += 20  # 80 s after the first
    said.append(rig.watch.check_due())

    line = ["Heads up: the build finished."]
    assert said == [line, [], [], [], line]  # said at 0 s, quiet until 60 s, gone, back at 80 s
    assert len(rig.judge.shown) == 3  # no picture and no model call while it is quiet
    assert len(rig.grim.calls) == 3


def test_the_cool_down_is_three_checks_when_they_are_far_apart(rig: Rig) -> None:
    _listed_start(rig, every_s=100)
    rig.judge.answers = [True, False, True]  # gone at the second look, back at the third

    first = rig.watch.check_due()
    rig.clock.now += 100
    second = rig.watch.check_due()
    rig.clock.now += 200  # 300 s after the first: three checks of 100 s
    third = rig.watch.check_due()

    assert (len(first), len(second), len(third)) == (1, 0, 0)
    assert len(rig.judge.shown) == 2  # the second check fell in the cool-down: no look at all


def test_a_watch_that_does_not_repeat_ends_after_its_first_yes(rig: Rig) -> None:
    _listed_start(rig, repeat=False)
    rig.judge.answers = [False, True]

    assert rig.watch.check_due() == []
    rig.clock.now += 20
    assert rig.watch.check_due() == ["Heads up: the build finished."]

    assert rig.watch.ready() is False
    assert rig.events()[-1] == "watch_stopped"
    assert "No watches" in rig.watch.list_watches({}).text


def test_a_closed_window_ends_its_watch_and_it_is_said_once(rig: Rig) -> None:
    _listed_start(rig)
    rig.desktop.open = [w for w in rig.desktop.open if w.address != "0xa1"]

    said = rig.watch.check_due()
    rig.clock.now += 20
    again = rig.watch.check_due()

    assert len(said) == 1
    assert "gone" in said[0]
    assert "build finished" in said[0]
    assert again == []
    assert rig.judge.shown == []  # no picture was made of what is not there
    assert rig.grim.calls == []
    assert rig.events() == ["watch_started", "watch_stopped"]


def test_a_window_whose_address_now_holds_another_program_counts_as_gone(rig: Rig) -> None:
    _listed_start(rig)
    rig.desktop.open = [make_window("0xa1", "another", "2")]

    assert "gone" in rig.watch.check_due()[0]


def test_a_watch_ends_when_its_time_is_up(rig: Rig) -> None:
    _listed_start(rig, for_min=2)
    rig.watch.check_due()
    rig.clock.now += 121

    assert rig.watch.ready() is True
    said = rig.watch.check_due()

    assert said == ["My watch for the build finished is over."]
    assert rig.watch.ready() is False
    assert len(rig.judge.shown) == 1  # none after the time was up


def test_checks_that_keep_failing_end_the_watch_with_one_line(rig: Rig) -> None:
    _listed_start(rig)
    rig.judge.broken = True

    said: list[list[str]] = []
    for _ in range(4):
        said.append(rig.watch.check_due())
        rig.clock.now += 20

    assert [len(lines) for lines in said] == [0, 0, 1, 0]
    assert "can't check" in said[2][0]
    assert rig.events().count("watch_check_failed") == 3


def test_a_failed_capture_is_a_skipped_check_and_a_good_one_resets_the_count(rig: Rig) -> None:
    _listed_start(rig)
    rig.grim.fails = True
    rig.watch.check_due()
    rig.clock.now += 20
    rig.watch.check_due()
    rig.grim.fails = False
    rig.clock.now += 20
    rig.watch.check_due()  # a good check: two failures forgotten
    rig.grim.fails = True
    for _ in range(2):
        rig.clock.now += 20
        assert rig.watch.check_due() == []

    assert "the build finished" in rig.watch.list_watches({}).text


def test_a_check_is_not_made_by_a_model_that_lost_its_picture_reader(rig: Rig) -> None:
    _listed_start(rig)
    rig.judge.sight = False

    assert rig.watch.check_due() == []

    assert rig.judge.shown == []
    assert rig.grim.calls == []


def test_the_screen_is_captured_by_monitor_and_a_workspace_from_its_windows(rig: Rig) -> None:
    with request_scope(foreign=False):
        rig.start(of="screen", condition="a")
        rig.start(of="workspace", workspace="2", condition="b")

    rig.watch.check_due()

    assert rig.grim.calls == [[], ["-T", "a1"], ["-T", "a2"]]
    assert rig.desktop.ran == []


def test_the_watches_are_listed_and_stopped_by_number_or_all(rig: Rig) -> None:
    with request_scope(foreign=False):
        rig.start(of="screen", condition="a dialog", every_s=60)
        rig.start(of="screen", condition="a login", for_min=30, repeat=False)

    listing = rig.watch.list_watches({}).text
    assert "1. the screen for: a dialog (every 60 s" in listing
    assert "2. the screen for: a login" in listing
    assert rig.stop(watch=1).text == "Stopped watch 1."
    assert rig.stop(watch=1).ok is False
    assert rig.stop().text == "Stopped watch 2."
    assert rig.stop().ok is False
    assert rig.watch.list_watches({}).text == "No watches are running."
    assert rig.events().count("watch_stopped") == 2


def test_a_stopped_watch_says_nothing_even_if_its_check_was_under_way(rig: Rig) -> None:
    with request_scope(foreign=False):
        rig.start(of="screen", condition="a dialog")
    rig.judge.answers = [True]
    watch = next(iter(rig.watch._shared.watches.values()))  # pyright: ignore[reportPrivateUsage]

    original = rig.judge.glance

    def glance_then_stop(picture: Picture, condition: str) -> bool:
        answer = original(picture, condition)
        rig.stop()
        return answer

    rig.judge.glance = glance_then_stop

    assert rig.watch.check_due() == []
    assert watch.active is False


def test_the_operations_are_safe_and_the_listing_is_a_pure_read(rig: Rig) -> None:
    ops = {op.name: op for op in rig.watch.operations()}

    assert {name: op.default_class.value for name, op in ops.items()} == {
        "pc_watch": "safe",
        "pc_watches": "safe",
        "pc_watch_stop": "safe",
    }
    assert ops["pc_watches"].side_effect is False
    assert all(not op.shortcuts for op in ops.values())
    assert "model call with a picture" in ops["pc_watch"].description


def test_a_cool_down_still_ends_a_watch_whose_window_closed_or_time_ran_out(rig: Rig) -> None:
    _listed_start(rig, for_min=3)
    rig.judge.answers = [True]
    rig.watch.check_due()  # said: quiet for 60 s
    rig.clock.now += 20
    rig.desktop.open = [w for w in rig.desktop.open if w.address != "0xa1"]

    assert "gone" in rig.watch.check_due()[0]
    assert len(rig.judge.shown) == 1


def test_a_check_the_voice_service_stops_makes_no_picture_and_is_due_again(rig: Rig) -> None:
    _listed_start(rig)

    assert rig.watch.check_due(lambda: True) == []

    assert (rig.grim.calls, rig.judge.shown) == ([], [])
    assert rig.watch.ready() is True  # still due: not pushed back, not a failure
    assert "watch_check_failed" not in rig.events()


@pytest.mark.parametrize("stops_after", [1, 2, 3])
def test_a_check_is_dropped_whenever_the_service_stops_being_idle(
    rig: Rig, stops_after: int
) -> None:
    # Asked before the picture, before the model and after its answer: a request began meanwhile.
    _listed_start(rig)
    rig.judge.answers = [True]
    asked: list[None] = []

    def stop() -> bool:
        asked.append(None)
        return len(asked) > stops_after

    # check_due asks once first, then _look asks before capture, before the model and after it.
    assert rig.watch.check_due(stop) == []

    assert rig.watch.ready() is True
    assert "watch_announced" not in rig.events()
    assert "watch_check_failed" not in rig.events()
    assert (len(rig.grim.calls), len(rig.judge.shown)) == (
        min(stops_after - 1, 1),
        min(max(stops_after - 2, 0), 1),
    )


def test_a_watch_stopped_while_another_is_checked_is_not_checked_or_audited_twice(
    rig: Rig,
) -> None:
    with request_scope(foreign=False):
        rig.start(of="screen", condition="a")
        rig.start(of="workspace", workspace="2", condition="b")
    original = rig.judge.glance

    def glance_then_stop_all(picture: Picture, condition: str) -> bool:
        answer = original(picture, condition)
        rig.stop()
        return answer

    rig.judge.glance = glance_then_stop_all
    rig.judge.answers = [True, True]

    assert rig.watch.check_due() == []

    assert rig.grim.calls == [[]]  # the second watch was never looked at
    assert rig.events().count("watch_stopped") == 2  # once for each, said by the owner's stop
    assert rig.events().count("watch_announced") == 0


def test_a_failed_read_of_the_windows_is_a_failed_check_and_three_end_the_watch(
    rig: Rig,
) -> None:
    _listed_start(rig)
    rig.desktop.readable = False

    said: list[list[str]] = []
    for _ in range(3):
        said.append(rig.watch.check_due())
        rig.clock.now += 20

    assert [len(lines) for lines in said] == [0, 0, 1]
    assert rig.events().count("watch_check_failed") == 3
    assert rig.watch.ready() is False


def test_the_next_look_is_one_pace_after_a_failed_check(rig: Rig) -> None:
    _listed_start(rig)
    rig.grim.fails = True
    rig.watch.check_due()

    assert rig.watch.ready() is False
    rig.clock.now += 20
    assert rig.watch.ready() is True


def test_a_scheduled_request_sets_no_watch(rig: Rig) -> None:
    with (
        request_scope(foreign=False, scheduled=True),
        pytest.raises(OperationError, match="own request"),
    ):
        rig.start(of="screen", condition="x")


def test_the_watches_time_is_the_wall_clock_that_runs_through_a_suspend() -> None:
    assert (ScreenWatch.__init__.__kwdefaults__ or {})["clock"] is time.time


def test_a_condition_that_stays_on_screen_is_said_once_and_again_after_it_goes(rig: Rig) -> None:
    _listed_start(rig)
    rig.judge.answers = [True, True, True, True, False, True]
    said: list[str] = []
    for _ in range(6):
        rig.clock.now += 120  # past every cool-down: only the change from no to yes speaks
        said += rig.watch.check_due()

    assert said == ["Heads up: the build finished.", "Heads up: the build finished."]


def test_a_watch_with_its_own_line_says_that_line_and_still_judges_the_condition(rig: Rig) -> None:
    result = _listed_start(rig, say="the build is done.")
    rig.judge.answers = [True]

    assert '"Heads up: the build is done."' in result.text
    assert rig.watch.check_due() == ["Heads up: the build is done."]
    assert rig.judge.shown == [(1024, "the build finished")]  # the model is asked the condition
    assert rig.audit.events[0]["say"] == "the build is done"


def test_the_spoken_line_is_short_by_schema(rig: Rig) -> None:
    with pytest.raises(ArgumentError):
        _listed_start(rig, say="x" * (MAX_SAY + 1))


def test_the_question_names_the_line_a_watch_will_say(rig: Rig) -> None:
    op = next(o for o in rig.watch.operations() if o.name == "pc_watch")
    asked = {"of": "screen", "condition": "the build finished", "say": "go check the build"}

    assert op.spoken_summary is not None
    assert "go check the build" in op.spoken_summary(asked)


def test_a_screen_watch_without_a_condition_is_refused(rig: Rig) -> None:
    with request_scope(foreign=False), pytest.raises(OperationError, match="what to look for"):
        rig.start(of="screen")


def test_the_listing_shows_the_line_a_watch_will_say(rig: Rig) -> None:
    _listed_start(rig, say="go check the build")

    assert "go check the build" in rig.watch.list_watches({}).text


WINDOW_PID = 4242


def _window_process(rig: Rig, *, ticks: int = 0, state: str = "S") -> None:
    rig.proc.add(
        WINDOW_PID,
        "worker",
        argv=["/opt/worker"],
        state=state,
        utime=ticks // 2,
        stime=ticks - ticks // 2,
    )


def _freeze_target(rig: Rig) -> None:
    original = rig.desktop.open[0]
    rig.desktop.open = [
        replace(original, pid=WINDOW_PID, klass="window-class", title="outside title")
    ]
    rig.desktop.showing = original.workspace
    _window_process(rig)


def _freeze_start(rig: Rig, **args: Any) -> Result:  # ruff: ignore[any-type] - the arguments a test sets
    _freeze_target(rig)
    rig.grim.frame = 0
    with request_scope(foreign=False):
        seen(rig.windows)
        return rig.start(of="window", window=1, frozen=True, **args)


def _motion(rig: Rig) -> None:
    assert rig.watch.check_due() == []
    assert rig.tick(10, frame=1) == []
    assert rig.tick(10, frame=2) == []


def _frozen_for(seconds: int) -> list[str]:
    return [f"Heads up: worker seems frozen: its picture has not changed for {seconds} seconds."]


@pytest.mark.parametrize("changes", [0, 1])
def test_a_freeze_watch_needs_two_changes_after_its_first_frame(rig: Rig, changes: int) -> None:
    _freeze_start(rig, every_s=10)
    assert rig.watch.check_due() == []
    if changes:
        assert rig.tick(10, frame=1) == []

    for _ in range(5):
        assert rig.tick(60) == []


@pytest.mark.parametrize("elapsed", [59, 60])
def test_a_freeze_watch_needs_sixty_seconds_even_after_three_identical_checks(
    rig: Rig, elapsed: int
) -> None:
    _freeze_start(rig, every_s=10)
    _motion(rig)

    assert rig.tick(20) == []
    assert rig.tick(20) == []
    assert rig.tick(elapsed - 40) == (_frozen_for(60) if elapsed == 60 else [])


def test_a_freeze_watch_needs_three_identical_checks_even_after_sixty_seconds(rig: Rig) -> None:
    _freeze_start(rig, every_s=10)
    _motion(rig)

    assert rig.tick(30) == []
    assert rig.tick(30) == []  # 60 seconds, only two identical checks
    assert rig.tick(10) == _frozen_for(70)


@pytest.mark.parametrize(("ticks", "alarmed"), [(114, True), (120, False), (126, False)])
def test_a_freeze_at_sixty_seconds_requires_under_two_percent_cpu(
    rig: Rig, ticks: int, alarmed: bool
) -> None:
    _freeze_start(rig, every_s=10)
    _motion(rig)
    assert rig.tick() == []
    assert rig.tick() == []

    _window_process(rig, ticks=ticks)  # 1.9%, 2.0%, or 2.1% of one core over 60 s
    assert rig.tick() == (_frozen_for(60) if alarmed else [])


@pytest.mark.parametrize("elapsed", [179, 180])
def test_a_freeze_watch_can_alarm_at_180_seconds_even_with_high_cpu(rig: Rig, elapsed: int) -> None:
    _freeze_start(rig, every_s=10)
    _motion(rig)
    _window_process(rig, ticks=20_000)
    assert rig.tick(90) == []
    assert rig.tick(elapsed - 90) == (_frozen_for(180) if elapsed == 180 else [])


def test_a_hidden_workspace_pauses_freeze_time_and_cpu_without_resetting_the_streak(
    rig: Rig,
) -> None:
    _freeze_start(rig, every_s=10)
    _motion(rig)
    assert rig.tick() == []
    assert rig.tick() == []
    captured = len(rig.grim.calls)
    rig.desktop.showing = "1"
    assert rig.tick(150) == []
    assert rig.tick(150) == []
    assert len(rig.grim.calls) == captured
    _window_process(rig, ticks=20_000)  # CPU spent while hidden is not part of the shown streak
    rig.desktop.showing = "2"

    assert rig.tick() == []
    assert rig.tick() == _frozen_for(60)


def test_a_freeze_is_said_once_and_motion_rearms_only_after_two_changes(rig: Rig) -> None:
    _freeze_start(rig, every_s=10)
    _motion(rig)
    assert rig.tick() == []
    assert rig.tick() == []
    assert rig.tick() == _frozen_for(60)
    assert rig.tick(180) == []
    assert rig.tick(10, frame=3) == ["worker is moving again."]
    assert rig.tick(180) == []
    assert rig.tick(10, frame=4) == []
    assert rig.tick() == []
    assert rig.tick() == []
    assert rig.tick() == _frozen_for(60)
    assert rig.tick() == []
    assert [e["frozen"] for e in rig.audit.events] == [True] * 4
    assert rig.events().count("watch_announced") == 3
    assert rig.judge.shown == []
    assert rig.desktop.ran == []


@pytest.mark.parametrize(
    ("state", "fact"), [("T", "is stopped"), ("D", "has been stuck"), ("Z", "has exited")]
)
def test_process_facts_take_precedence_over_a_freeze_on_the_same_streak(
    rig: Rig, state: str, fact: str
) -> None:
    _freeze_start(rig, every_s=10)
    _motion(rig)
    if state == "D":
        _window_process(rig, state=state)
    assert rig.tick() == []
    assert rig.tick() == []
    _window_process(rig, state=state)
    if state == "Z":
        rig.desktop.open = []  # process exit takes precedence even if its window is gone

    (line,) = rig.tick()
    assert line.startswith("Heads up: worker (PID 4242)")
    assert fact in line
    assert rig.tick() == []
    if state != "Z":
        _window_process(rig)
        assert rig.tick() == ["worker (PID 4242) is running again."]
        assert rig.tick(180) == []  # first sample after recovery
        assert rig.tick(180) == []  # the same streak was already explained
        assert rig.tick(10, frame=3) == []


def test_unchanged_condition_frames_keep_both_verdicts_and_skip_the_model(rig: Rig) -> None:
    _listed_start(rig)
    rig.grim.frame = 0
    rig.judge.answers = [False, True, True, False, True]

    assert rig.watch.check_due() == []
    assert rig.tick() == []
    assert len(rig.judge.shown) == 1
    assert rig.tick(frame=1) == ["Heads up: the build finished."]
    assert rig.tick(60) == []  # still YES after the cool-down
    assert len(rig.judge.shown) == 2
    assert rig.tick(frame=2) == []  # changed picture, still YES
    assert rig.tick(frame=3) == []  # NO
    assert rig.tick() == []
    assert len(rig.judge.shown) == 4
    assert rig.tick(frame=4) == ["Heads up: the build finished."]


def test_a_combined_watch_captures_once_and_checks_the_condition_only_on_changes(rig: Rig) -> None:
    _freeze_start(rig, condition="a dialog", every_s=10)
    _motion(rig)
    assert rig.tick() == []
    assert rig.tick() == []
    assert rig.tick() == _frozen_for(60)
    assert len(rig.grim.calls) == 6
    assert len(rig.judge.shown) == 3


def test_a_freeze_only_watch_needs_no_picture_reader_and_explains_its_cost(rig: Rig) -> None:
    rig.judge.sight = False
    result = _freeze_start(rig)
    assert "freeze" in result.text
    assert "capture and a hash" in result.text
    assert "no model call" in result.text
    assert "freeze" in rig.watch.list_watches({}).text
    assert rig.watch.check_due() == []
    assert len(rig.grim.calls) == 1
    assert rig.judge.shown == []


def test_a_freeze_watch_binds_the_read_window_before_confirming(rig: Rig) -> None:
    _freeze_target(rig)
    op = next(o for o in rig.watch.operations() if o.name == "pc_watch")
    assert op.prepare is not None
    assert op.spoken_summary is not None
    args = validate_args(op, {"of": "window", "window": 1, "frozen": True})
    with request_scope(foreign=False):
        rig.windows.list_windows({})
        with pytest.raises(OperationError, match="not read yet"):
            op.prepare(args)
        seen(rig.windows)
        bound = op.prepare(args)
        question = op.spoken_summary(bound)
        assert "freeze" in question
        assert "outside title" not in question
        another = make_window("0xa2", "a different class", "2", focused=True)
        rig.desktop.open.append(another)
        seen(rig.windows)
        op.run(bound)

    assert rig.watch.check_due() == []
    assert rig.grim.calls == [["-T", "a1"]]


@pytest.mark.parametrize("kind", ["screen", "workspace", "process"])
def test_frozen_is_only_available_for_a_window(rig: Rig, kind: str) -> None:
    with request_scope(foreign=False), pytest.raises(OperationError, match="window"):
        rig.start(of=kind, frozen=True)


def test_a_combined_watch_keeps_a_new_condition_edge_through_cooldown(rig: Rig) -> None:
    _freeze_start(rig, condition="a dialog", every_s=10)
    rig.judge.answers = [True, False, True]
    line = ["Heads up: a dialog."]

    assert rig.watch.check_due() == line
    assert rig.tick(10, frame=1) == []
    assert rig.tick(10, frame=2) == []
    assert len(rig.judge.shown) == 3  # the last changed frame has already been judged
    assert rig.tick(40) == line
    assert rig.tick(10) == []
    assert rig.tick(20) == _frozen_for(70)
    assert len(rig.judge.shown) == 3


def test_a_freeze_watch_expires_even_if_its_window_is_hidden(rig: Rig) -> None:
    _freeze_start(rig, for_min=1)
    rig.desktop.showing = "1"
    assert rig.watch.check_due() == []
    assert rig.tick(61) == ["My watch for a freeze is over."]
    assert rig.events() == ["watch_started", "watch_stopped"]
    assert rig.grim.calls == []


@pytest.mark.parametrize("foreign", [True, False])
def test_a_freeze_watch_only_accepts_the_owners_immediate_request(rig: Rig, foreign: bool) -> None:
    op = next(o for o in rig.watch.operations() if o.name == "pc_watch")
    assert op.prepare is not None
    with (
        request_scope(foreign=foreign, scheduled=not foreign),
        pytest.raises(OperationError, match="own request"),
    ):
        op.prepare(validate_args(op, {"of": "window", "frozen": True, "window": 1}))
    assert rig.grim.calls == []


def test_freeze_watches_share_the_screen_watch_cap(rig: Rig) -> None:
    _freeze_start(rig)
    with request_scope(foreign=False):
        rig.start(of="screen", condition="a dialog")
        with pytest.raises(OperationError, match="Two watches"):
            rig.start(of="screen", condition="a login")


def test_a_combined_watch_keeps_checking_the_condition_when_hidden_during_cooldown(
    rig: Rig,
) -> None:
    _freeze_start(rig, condition="a dialog", every_s=10)
    rig.judge.answers = [True, False, True]
    line = ["Heads up: a dialog."]

    assert rig.watch.check_due() == line
    rig.desktop.showing = "1"
    assert rig.tick(10, frame=1) == []
    assert rig.tick(10, frame=0) == []
    assert rig.tick(40) == line
    assert len(rig.judge.shown) == 3


def test_a_suspend_does_not_count_as_time_with_a_shown_frozen_frame(rig: Rig) -> None:
    _freeze_start(rig, every_s=10)
    _motion(rig)

    rig.clock.now += 180  # expiry and scheduling use wall time, even while asleep
    assert rig.watch.check_due() == []
    assert rig.tick() == []
    assert rig.tick() == []
    assert rig.tick() == _frozen_for(60)


def test_an_unreadable_proc_sample_does_not_claim_the_process_exited(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _freeze_start(rig, every_s=10)
    original = Path.read_text
    unreadable = rig.proc.root / str(WINDOW_PID) / "stat"

    def read(path: Path, *args: Any, **kwargs: Any) -> str:  # ruff: ignore[any-type] - Path.read_text's arguments
        if path == unreadable:
            msg = "fake proc read error"
            raise OSError(msg)
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", read)
        assert rig.watch.check_due() == []
    assert rig.events() == ["watch_started", "watch_check_failed"]
    assert rig.tick() == []
    assert len(rig.grim.calls) == 1


@pytest.mark.parametrize(
    ("condition", "cost"),
    [("", "capture and a hash"), ("a dialog", "a model call with a picture only when")],
)
def test_a_freeze_watch_discloses_its_cost_in_the_question_and_list(
    rig: Rig, condition: str, cost: str
) -> None:
    _freeze_target(rig)
    op = next(o for o in rig.watch.operations() if o.name == "pc_watch")
    assert op.spoken_summary is not None
    args = {"condition": condition} if condition else {}
    with request_scope(foreign=False):
        seen(rig.windows)
        bound = rig.prepare(of="window", window=1, frozen=True, repeat=False, **args)
        assert cost in op.spoken_summary(bound)
        op.run(bound)
    listing = rig.watch.list_watches({}).text
    assert cost in listing
    assert ("once" in listing) is bool(condition)


def test_a_process_recovery_does_not_hide_a_combined_watch_condition_edge(rig: Rig) -> None:
    _freeze_start(rig, condition="a dialog", every_s=10)
    rig.judge.answers = [True, False, True]

    assert rig.watch.check_due() == ["Heads up: a dialog."]
    _window_process(rig, state="T")
    assert "is stopped" in rig.tick(10)[0]
    _window_process(rig)
    assert rig.tick(10, frame=1) == ["worker (PID 4242) is running again."]
    assert rig.tick(10, frame=0) == []
    assert rig.tick(30) == ["Heads up: a dialog."]
    assert len(rig.judge.shown) == 3


def test_a_failed_combined_capture_does_not_drop_a_process_fact(rig: Rig) -> None:
    _freeze_start(rig, condition="a dialog", every_s=10)
    _window_process(rig, state="T")
    rig.grim.fails = True

    (line,) = rig.watch.check_due()

    assert "is stopped" in line
    assert "watch_check_failed" in rig.events()


def test_a_missed_proc_read_breaks_an_unconfirmed_disk_wait(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _freeze_start(rig, every_s=10)
    _window_process(rig, state="D")
    assert rig.watch.check_due() == []

    def unavailable(*_args: object) -> tuple[str, int]:
        msg = "fake proc read error"
        raise OSError(msg)

    with monkeypatch.context() as patch:
        patch.setattr(rig.watch.processes, "sample", unavailable)
        assert rig.tick() == []
    assert rig.tick() == []  # first confirmed wait after the missed read
    assert rig.tick() == []  # 20 seconds since that first read
    (line,) = rig.tick()
    assert "stuck for 40 seconds" in line
