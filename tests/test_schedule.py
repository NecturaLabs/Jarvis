"""Reminders and scheduled requests: when they fall due, the jobs file, and the operations."""

import json
import stat
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pytest

from jarvis.operations import ArgumentError, OperationError, Result, validate_args
from jarvis.schedule import (
    LATE_REMINDER_S,
    LATE_S,
    MAX_JOBS,
    POLL_S,
    Due,
    Job,
    JobStore,
    ScheduleOps,
    following,
    plan,
    reminder_words,
    spoken_when,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

LONDON = ZoneInfo("Europe/London")
pytestmark = pytest.mark.usefixtures("london")


@pytest.fixture
def london(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run in one time zone with a summer time change, whatever this PC's is."""
    monkeypatch.setenv("TZ", "Europe/London")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _at(*parts: int) -> float:
    return datetime(*parts, tzinfo=LONDON).timestamp()  # type: ignore[misc]


FRIDAY_10 = _at(2026, 11, 20, 10, 0)  # a Friday


def _local(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, LONDON)


def _plan(now: float = FRIDAY_10, **args: object) -> Job:
    return plan({"text": "stretch", **args}, now, 1)


# --- when a job falls due ------------------------------------------------------------------------


def test_a_reminder_in_some_minutes_is_due_then() -> None:
    assert _plan(in_minutes=20).next == FRIDAY_10 + 20 * 60


def test_a_reminder_in_some_seconds_is_due_then() -> None:
    """A timer shorter than a minute ("in 30 seconds") could not be set in whole minutes."""
    assert _plan(in_seconds=30).next == FRIDAY_10 + 30


@pytest.mark.parametrize(
    ("at", "day", "expected"),
    [
        ("14:30", "", (2026, 11, 20, 14, 30)),  # later today
        ("09:00", "", (2026, 11, 21, 9, 0)),  # passed today: tomorrow
        ("9:05", "tomorrow", (2026, 11, 21, 9, 5)),
        ("10:30", "today", (2026, 11, 20, 10, 30)),
        ("08:00", "monday", (2026, 11, 23, 8, 0)),
        ("11:00", "friday", (2026, 11, 20, 11, 0)),  # this Friday, still to come
        ("09:00", "friday", (2026, 11, 27, 9, 0)),  # this Friday's has passed: next week's
    ],
)
def test_a_reminder_at_a_time_of_day_is_due_at_its_next_one(
    at: str, day: str, expected: tuple[int, ...]
) -> None:
    args: dict[str, Any] = {"at": at, "day": day} if day else {"at": at}
    job = plan({"text": "stretch", **args}, FRIDAY_10, 1)

    assert _local(job.next).timetuple()[:5] == expected


@pytest.mark.parametrize(
    "args",
    [
        {"at": "09:00", "day": "today"},  # already passed
        {"in_minutes": 5, "at": "10:30"},  # said twice
        {},  # not said at all
        {"at": "25:00"},
        {"at": "noon"},
        {"repeat": "daily"},  # a daily job needs its time of day
        {"repeat": "every"},  # ... and an every job its minutes
        {"in_minutes": 5, "every_minutes": 30},  # minutes between, without repeat every
        {"in_seconds": 30, "in_minutes": 1},  # said twice
        {"in_seconds": 30, "at": "10:30"},  # said twice
        {"in_seconds": 30, "at": "07:30", "repeat": "daily"},  # seconds are for a one-off job
        {"in_seconds": 30, "repeat": "every", "every_minutes": 30},
    ],
)
def test_a_time_that_cannot_be_a_time_to_come_is_refused(args: dict[str, Any]) -> None:
    with pytest.raises(ArgumentError):
        _plan(**args)


def test_a_daily_job_comes_back_at_the_same_time_each_day() -> None:
    job = _plan(at="07:30", repeat="daily")

    assert _local(job.next).timetuple()[:5] == (2026, 11, 21, 7, 30)
    again = following(job, job.next)
    assert again is not None
    assert _local(again).timetuple()[:5] == (2026, 11, 22, 7, 30)


def test_a_daily_job_keeps_its_local_time_over_a_summer_time_change() -> None:
    friday = _at(2026, 3, 27, 12, 0)  # summer time starts at 1:00 on Sunday 29 March
    job = _plan(friday, at="07:30", repeat="daily")

    again = following(job, job.next)

    assert again is not None
    assert _local(job.next).timetuple()[:5] == (2026, 3, 28, 7, 30)
    assert _local(again).timetuple()[:5] == (2026, 3, 29, 7, 30)
    assert again - job.next == 23 * 3600  # the night the clocks went forward was an hour short


def test_a_weekday_job_set_on_friday_evening_comes_on_monday() -> None:
    job = _plan(_at(2026, 11, 20, 20, 0), at="08:00", repeat="weekdays")

    assert _local(job.next).strftime("%A %H:%M") == "Monday 08:00"


def test_a_weekend_and_a_weekly_job_come_on_their_days() -> None:
    weekend = _plan(at="09:00", repeat="weekends")
    weekly = _plan(at="18:00", repeat="weekly", day="wednesday")
    this_day = _plan(at="18:00", repeat="weekly")

    assert _local(weekend.next).strftime("%A") == "Saturday"
    assert _local(weekly.next).strftime("%A %d") == "Wednesday 25"
    assert _local(this_day.next).strftime("%A %d") == "Friday 20"


def test_an_every_job_skips_the_times_it_missed() -> None:
    job = _plan(repeat="every", every_minutes=30)
    assert job.next == FRIDAY_10 + 1800

    later = following(job, job.next + 2 * 3600 + 60)  # two hours late

    assert later == job.next + 5 * 1800


@pytest.mark.parametrize(
    ("args", "said"),
    [
        ({"in_minutes": 20}, "today at 10:20 AM"),
        ({"in_seconds": 30}, "in 30 seconds"),
        ({"in_seconds": 1}, "in 1 second"),
        ({"in_seconds": 90}, "in 90 seconds"),
        ({"at": "09:00"}, "tomorrow at 9:00 AM"),
        ({"at": "08:00", "day": "wednesday"}, "on Wednesday at 8:00 AM"),
        ({"at": "07:30", "repeat": "daily"}, "every day at 7:30 AM"),
        ({"at": "07:30", "repeat": "weekdays"}, "on weekdays at 7:30 AM"),
        ({"at": "18:00", "repeat": "weekly", "day": "sunday"}, "every Sunday at 6:00 PM"),
        ({"repeat": "every", "every_minutes": 60}, "every hour, next at 11:00 AM"),
        ({"repeat": "every", "every_minutes": 120}, "every 2 hours, next at 12:00 PM"),
        ({"repeat": "every", "every_minutes": 45}, "every 45 minutes, next at 10:45 AM"),
        (
            {"repeat": "every", "every_minutes": 30, "at": "14:00"},
            "every 30 minutes, next at 2:00 PM",
        ),
        (
            {"repeat": "every", "every_minutes": 30, "at": "09:00"},
            "every 30 minutes, next tomorrow at 9:00 AM",
        ),
    ],
)
def test_when_a_job_is_due_is_said_plainly(args: dict[str, Any], said: str) -> None:
    assert spoken_when(_plan(**args), FRIDAY_10, twelve_hour=True) == said


def test_a_late_one_off_reminder_is_still_said_but_nothing_else_is() -> None:
    reminder = _plan(in_minutes=1)
    request = _plan(in_minutes=1, action="do")
    daily = _plan(at="07:30", repeat="daily")

    assert not Due(reminder, LATE_REMINDER_S - 1).missed
    assert Due(reminder, LATE_REMINDER_S + 1).missed
    assert not Due(request, LATE_S - 1).missed
    assert Due(request, LATE_S + 1).missed
    assert Due(daily, LATE_S + 1).missed


# --- the jobs file -------------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        super().__init__()
        self.now = FRIDAY_10

    def __call__(self) -> float:
        return self.now


def _store(tmp_path: Path) -> tuple[JobStore, _Clock]:
    clock = _Clock()
    return JobStore(tmp_path / "state" / "schedule.json", clock), clock


def test_jobs_are_kept_in_a_private_file_and_numbered(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)

    first = store.add({"text": "stretch", "in_minutes": 30})
    second = store.add({"text": "tea", "in_minutes": 5})

    path = tmp_path / "state" / "schedule.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert (first.id, second.id) == (1, 2)
    assert [job.text for job in JobStore(path).jobs()] == ["tea", "stretch"]  # soonest first


def test_a_cancelled_number_is_not_given_again(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    store.add({"text": "a", "in_minutes": 5})
    store.add({"text": "b", "in_minutes": 5})

    assert [job.text for job in store.cancel(2)] == ["b"]
    assert store.add({"text": "c", "in_minutes": 5}).id == 3
    assert [job.text for job in store.cancel(None)] == ["a", "c"]
    assert store.jobs() == []


def test_no_more_than_the_most_jobs_are_kept(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    for number in range(MAX_JOBS):
        store.add({"text": f"job {number}", "in_minutes": 5})

    with pytest.raises(OperationError, match=str(MAX_JOBS)):
        store.add({"text": "one more", "in_minutes": 5})


def test_due_jobs_are_taken_once_and_repeating_ones_move_on(tmp_path: Path) -> None:
    store, clock = _store(tmp_path)
    store.add({"text": "tea", "in_minutes": 5})
    store.add({"text": "stretch", "at": "10:05", "repeat": "daily"})
    store.add({"text": "later", "in_minutes": 60})

    clock.now += 6 * 60
    due = store.take_due()
    clock.now += POLL_S
    again = store.take_due()

    assert [(item.job.text, round(item.late_s)) for item in due] == [("tea", 60), ("stretch", 60)]
    assert again == []
    left = {job.text: job for job in store.jobs()}
    assert sorted(left) == ["later", "stretch"]
    assert _local(left["stretch"].next).timetuple()[:5] == (2026, 11, 21, 10, 5)


def test_nothing_is_due_before_its_time_and_the_file_is_not_read_each_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _store(tmp_path)
    store.add({"text": "tea", "in_minutes": 5})
    reads: list[int] = []
    read = store._read  # pyright: ignore[reportPrivateUsage] - counted, to show the file is left alone
    monkeypatch.setattr(store, "_read", lambda: reads.append(1) or read())

    for _ in range(50):
        clock.now += POLL_S
        assert store.take_due() == []

    assert len(reads) == 1  # once to learn when the soonest job is due


def test_between_two_frames_the_file_is_not_even_looked_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _store(tmp_path)
    store.add({"text": "tea", "in_minutes": 1})
    clock.now += 2 * 60
    looks: list[int] = []
    stamp = store._stamp  # pyright: ignore[reportPrivateUsage] - counted, to show nothing is read
    monkeypatch.setattr(store, "_stamp", lambda: looks.append(1) or stamp())

    first = store.take_due()
    clock.now += POLL_S / 2
    store.add({"text": "milk", "in_minutes": 1})
    looks.clear()
    soon = store.take_due()

    assert [item.job.text for item in first] == ["tea"]
    assert soon == []
    assert looks == []


def test_a_job_another_process_adds_is_seen(tmp_path: Path) -> None:
    store, clock = _store(tmp_path)
    assert store.take_due() == []
    other = JobStore(tmp_path / "state" / "schedule.json", clock)
    other.add({"text": "tea", "in_minutes": 1})

    clock.now += 2 * 60
    due = store.take_due()

    assert [item.job.text for item in due] == ["tea"]


def test_a_damaged_jobs_file_holds_no_jobs(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    store.add({"text": "tea", "in_minutes": 5})
    path = tmp_path / "state" / "schedule.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["jobs"].append({"id": 9, "action": "explode", "repeat": "none"})
    data["jobs"].append({"id": 10, "action": "remind", "repeat": "every", "every": 1, "text": "x"})
    path.write_text(json.dumps(data), encoding="utf-8")
    path.chmod(0o600)  # as the store wrote it

    assert [job.text for job in store.jobs()] == ["tea"]
    path.write_text("not json", encoding="utf-8")
    assert store.jobs() == []


def test_a_jobs_file_from_a_newer_jarvis_is_never_written_over(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)
    path = tmp_path / "state" / "schedule.json"
    path.parent.mkdir(mode=0o700)
    newer = json.dumps({"version": 2, "next_id": 7, "jobs": [{"id": 6, "shape": "new"}]})
    path.write_text(newer, encoding="utf-8")
    path.chmod(0o600)

    assert store.jobs() == []
    with pytest.raises(OperationError, match="newer"):
        store.add({"text": "tea", "in_minutes": 5})
    assert store.cancel(None) == []
    assert store.take_due() == []
    assert path.read_text(encoding="utf-8") == newer


# --- the operations ------------------------------------------------------------------------------


def _ops(tmp_path: Path, *, running: bool = True) -> tuple[ScheduleOps, JobStore]:
    store, clock = _store(tmp_path)
    return ScheduleOps(store, clock=clock, voice_running=lambda: running), store


def _run(ops: ScheduleOps, name: str, **args: object) -> Result:
    op = next(op for op in ops.operations() if op.name == name)
    return op.run(validate_args(op, args))


def test_adding_a_reminder_says_when_it_will_be_said(tmp_path: Path) -> None:
    ops, store = _ops(tmp_path)

    result = _run(ops, "schedule_add", text="Take the bread out", in_minutes=20)

    assert result.ok
    assert result.spoken == "Okay, I'll remind you today at 10:20 AM."
    assert [job.text for job in store.jobs()] == ["Take the bread out"]


def test_adding_one_says_so_when_no_voice_service_will_say_it(tmp_path: Path) -> None:
    ops, _ = _ops(tmp_path, running=False)

    result = _run(ops, "schedule_add", text="tea", in_minutes=20)

    assert "isn't running" in result.spoken


def test_the_list_numbers_each_job_and_cancel_takes_a_number_or_all(tmp_path: Path) -> None:
    ops, _ = _ops(tmp_path)
    _run(ops, "schedule_add", text="tea", in_minutes=20)
    _run(ops, "schedule_add", text="pause the music", at="23:00", action="do", repeat="daily")

    listed = _run(ops, "schedule_list").text.splitlines()
    one = _run(ops, "schedule_cancel", id="1")
    unknown = _run(ops, "schedule_cancel", id="7")

    assert listed == [
        "1. reminder 'tea', today at 10:20 AM",
        "2. request 'pause the music', every day at 11:00 PM",
    ]
    assert one.spoken == "Cancelled: tea."
    assert not unknown.ok
    with pytest.raises(OperationError, match="number"):
        _run(ops, "schedule_cancel", id="tea")
    assert _run(ops, "schedule_cancel", id="all").spoken == "Cancelled: pause the music."
    _run(ops, "schedule_add", text="tea", in_minutes=20)
    _run(ops, "schedule_add", text="milk", in_minutes=30)
    assert _run(ops, "schedule_cancel", id="all").spoken == "Cancelled all 2."
    assert _run(ops, "schedule_list").spoken == "You have no reminders or scheduled requests."


def test_the_arguments_are_checked_before_anything_is_kept(tmp_path: Path) -> None:
    ops, store = _ops(tmp_path)
    op = next(op for op in ops.operations() if op.name == "schedule_add")

    with pytest.raises(ArgumentError):
        validate_args(op, {"text": "tea", "in_minutes": 0})
    with pytest.raises(ArgumentError):
        validate_args(op, {"text": "tea", "repeat": "hourly"})
    with pytest.raises(ArgumentError):
        validate_args(op, {"text": "tea", "in_minutes": 5, "sound": "loud"})
    assert store.jobs() == []


def test_only_listing_changes_nothing() -> None:
    ops = ScheduleOps(JobStore.__new__(JobStore))
    effects = {op.name: op.side_effect for op in ops.operations()}

    assert effects == {"schedule_add": True, "schedule_list": False, "schedule_cancel": True}


def test_a_day_that_is_not_one_is_refused() -> None:
    with pytest.raises(ArgumentError, match="day"):
        _plan(at="09:00", day="someday")
    assert _local(_plan(at="09:00", day=" Monday ").next).strftime("%A") == "Monday"


def _reminder(late_s: float) -> str:
    return reminder_words(Due(Job(1, "remind", "stretch", "none", "", (), 0, 0.0), late_s))


@pytest.mark.parametrize(
    ("late_s", "said"),
    [
        (0.0, "Reminder: stretch"),
        (120.0, "Reminder: stretch"),
        (180.0, "Reminder from 3 minutes ago: stretch"),
        (80 * 60.0, "Reminder from 80 minutes ago: stretch"),
        (95 * 60.0, "Reminder from 2 hours ago: stretch"),
        (5.4 * 3600, "Reminder from 5 hours ago: stretch"),
    ],
)
def test_a_late_reminder_says_how_late_it_is(late_s: float, said: str) -> None:
    assert _reminder(late_s) == said


def test_the_question_about_a_scheduled_request_holds_all_of_its_words() -> None:
    # The owner approves exactly what will run later, never only its start.
    text = "pause the music " + "and then " * 20 + "turn the volume to ten"

    assert ScheduleOps.add_summary({"text": text, "action": "do"}).endswith(text)


@pytest.mark.parametrize(
    ("args", "said"),
    [
        ({"text": "tea", "at": "09:00", "day": "someday"}, "day must be"),
        ({"text": "tea", "at": "08:00", "day": "today"}, "already passed"),
        ({"text": "tea", "in_minutes": 5, "at": "09:00"}, "not both"),
        ({"text": "tea"}, "say when"),
    ],
)
def test_a_time_the_model_got_wrong_is_explained_to_it(
    tmp_path: Path, args: dict[str, Any], said: str
) -> None:
    # Only an OperationError's words reach the model; anything else is "an internal error", so
    # the model could not see what to fix (seen with a local model, 2026-11-23).
    ops, store = _ops(tmp_path)

    with pytest.raises(OperationError, match=said):
        _run(ops, "schedule_add", **args)
    assert store.jobs() == []


@pytest.mark.parametrize("repeat", ["daily", "weekdays", "weekends"])
def test_a_day_beside_a_repeat_that_sets_its_own_days_is_ignored(
    tmp_path: Path, repeat: str
) -> None:
    ops, store = _ops(tmp_path)

    _run(ops, "schedule_add", text="stretch", at="07:30", day="weekday", repeat=repeat)

    assert [job.repeat for job in store.jobs()] == [repeat]
