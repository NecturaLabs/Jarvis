"""The screen watch tools: capture, clock and model faked, the compositor never asked to act."""

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from test_pc_screenshot import Grim
from test_pc_windows import FakeDesktop, make_window, seen

from jarvis.agent import PICTURES_OFF
from jarvis.audit import AuditLog
from jarvis.operations import ArgumentError, OperationError, Result, request_scope, validate_args
from jarvis.pc.screenshot import ScreenshotOps
from jarvis.pc.watch import MAX_WATCHES, NEEDS_PICTURES, ScreenWatch
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


@dataclass
class Rig:
    desktop: FakeDesktop
    grim: Grim
    windows: WindowOps
    judge: FakeJudge
    clock: Time
    audit: AuditLog
    watch: ScreenWatch

    def start(self, **args: Any) -> Result:  # ruff: ignore[any-type] - the arguments a test sets
        op = next(o for o in self.watch.operations() if o.name == "pc_watch")
        return self.watch.start(validate_args(op, args))

    def stop(self, **args: Any) -> Result:  # ruff: ignore[any-type] - the arguments a test sets
        return self.watch.stop(args)

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
    grim = Grim(desktop)
    windows = WindowOps(desktop)
    shots = ScreenshotOps(
        windows, tmp_path, capture=grim, copy=lambda _d: True, now=lambda: datetime.now(UTC)
    )
    judge, clock, audit = FakeJudge(), Time(), AuditLog(None)
    watch = ScreenWatch(windows, shots, clock=clock, audit=audit)
    watch.attach(judge)
    return Rig(desktop, grim, windows, judge, clock, audit, watch)


def _listed_start(rig: Rig, **args: Any) -> Result:  # ruff: ignore[any-type] - the arguments a test sets
    with request_scope(foreign=False):
        seen(rig.windows)
        return rig.start(of="window", window=1, condition="the build finished", **args)


def test_a_watch_on_a_listed_window_starts_and_says_what_it_costs(rig: Rig) -> None:
    result = _listed_start(rig)

    assert "the kitty window" in result.text
    assert "the build finished" in result.text
    assert "20 s" in result.text
    assert "GPU" in result.text
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
    rig.judge.answers = [True, True, True, True]

    said = [rig.watch.check_due()]
    for _ in range(3):
        rig.clock.now += 20
        said.append(rig.watch.check_due())
    rig.clock.now += 20  # 80 s after the first
    said.append(rig.watch.check_due())

    line = ["Heads up: the build finished."]
    assert said == [line, [], [], line, []]  # at 0 s, quiet until 60 s, again, quiet again
    assert len(rig.judge.shown) == 2  # no picture and no model call while it is quiet
    assert len(rig.grim.calls) == 2


def test_the_cool_down_is_three_checks_when_they_are_far_apart(rig: Rig) -> None:
    _listed_start(rig, every_s=100)
    rig.judge.answers = [True, True, True]

    first = rig.watch.check_due()
    rig.clock.now += 100
    second = rig.watch.check_due()
    rig.clock.now += 200  # 300 s after the first: three checks of 100 s
    third = rig.watch.check_due()

    assert (len(first), len(second), len(third)) == (1, 0, 1)


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
    assert "GPU" in ops["pc_watch"].description


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
