"""Reminders and scheduled requests: set, listed and cancelled by the owner, said when due.

A job is a reminder (words Jarvis says) or a request (words Jarvis acts on, as if the owner had
said them then), once or again and again: every day, on weekdays, at weekends, every week, or
every so many minutes. The jobs live in one private file in the state folder (mode 0600), so the
chat window and `jarvis ask` can set them; only the voice service (`jarvis run`) says them, between
requests, since Jarvis replies by voice only. A scheduled request goes through the agent like any
other: policy, confirmation (asked aloud, a no when nobody answers), the cap on actions and the
audit.

A job whose time passed while the voice service was not running, the PC slept or Jarvis was
paused is late: a one-off reminder is still said up to `LATE_REMINDER_S` late, and says when it
was due; anything else later than `LATE_S` is not done, but audited as missed, and a repeating job
moves on to its next time. A request is never run hours late: "pause the music at midnight" is
wrong at breakfast.
"""

import contextlib
import fcntl
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, cast

from jarvis.files import write_atomic
from jarvis.jsonutil import obj, objs
from jarvis.operations import ArgumentError, Operation, OperationError, Result, current_request
from jarvis.policy import Origin
from jarvis.secretfile import read_private
from jarvis.web.ops import spoken_time

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable
    from pathlib import Path

log = logging.getLogger(__name__)

FILE_VERSION = 1
MAX_JOBS = 50
MAX_TEXT = 200
MAX_IN_MINUTES = 7 * 24 * 60  # a week ahead
MAX_IN_SECONDS = 3600  # past an hour, minutes say it
_IN_SECONDS_UNDER = 120  # a one-off job due sooner than this is said in seconds
MIN_EVERY_MINUTES = 5
MAX_EVERY_MINUTES = 24 * 60
LATE_S = 10 * 60.0  # later than this, a request or a repeating reminder is not done
LATE_REMINDER_S = 6 * 3600.0  # a one-off reminder is still said this late
POLL_S = 1.0  # how often the voice service looks at the jobs file

type Action = Literal["remind", "do"]
type Repeat = Literal["none", "daily", "weekdays", "weekends", "weekly", "every"]
ACTIONS: tuple[Action, ...] = ("remind", "do")
REPEATS: tuple[Repeat, ...] = ("none", "daily", "weekdays", "weekends", "weekly", "every")
DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_DAYS_OF: dict[str, tuple[int, ...]] = {"weekdays": (0, 1, 2, 3, 4), "weekends": (5, 6)}
_CLOCK = re.compile(r"([01]?\d|2[0-3]):([0-5]\d)")
_WEEK = 7
_HOUR = 60


@dataclass(frozen=True, slots=True)
class Job:
    """One reminder or scheduled request; `next` is when it is due, in seconds since the epoch.

    `at` is the time of day ("07:30") it repeats at, `days` the weekdays (0 is Monday) for
    `weekdays`, `weekends` and `weekly`, and `every` the minutes between two of an `every` job.
    `foreign` says the request that added it was heard in PC audio, and `outside` holds the
    origins of the outside text that request had seen (`operations.Request`): a scheduled
    request runs with both, so what could steer it then still counts when it is due.
    """

    id: int
    action: Action
    text: str
    repeat: Repeat
    at: str
    days: tuple[int, ...]
    every: int
    next: float
    foreign: bool = False
    outside: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Due:
    """A job whose time came: what to do with it, and how late it is."""

    job: Job
    late_s: float

    @property
    def missed(self) -> bool:
        """Whether it came too late to be done."""
        limit = (
            LATE_REMINDER_S if self.job.action == "remind" and self.job.repeat == "none" else LATE_S
        )
        return self.late_s > limit


_SAYS_LATE_S = 120.0  # a reminder later than this says how late it is
_ONE_HOUR_S = 3600.0
_IN_MINUTES_UNDER = 90  # a delay shorter than this many minutes is said in minutes


def reminder_words(due: Due) -> str:
    """Return what Jarvis says for a due reminder: its words, and how late it is when it is late."""
    if due.late_s <= _SAYS_LATE_S:
        return f"Reminder: {due.job.text}"
    minutes = round(due.late_s / 60)
    if minutes < _IN_MINUTES_UNDER:
        ago = f"{minutes} minutes"
    else:
        hours = round(due.late_s / _ONE_HOUR_S)
        ago = "an hour" if hours == 1 else f"{hours} hours"
    return f"Reminder from {ago} ago: {due.job.text}"


def scheduled_request(job: Job) -> str:
    """Return the words a scheduled request is handed to the agent with."""
    return f"(A request you scheduled earlier, due now.) {job.text}"


def _local(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch).astimezone()


def _at(day: datetime, clock: str) -> datetime:
    """Return `clock` ("07:30") on `day`'s date, in local time with that date's own offset."""
    hour, minute = (int(part) for part in clock.split(":"))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=None).astimezone()


def _clock_of(text: str) -> str:
    """Return a time of day as "HH:MM".

    Raises:
        ArgumentError: it is not one.
    """
    found = _CLOCK.fullmatch(text.strip())
    if found is None:
        msg = "at must be a time of day on a 24-hour clock, such as 07:30 or 18:05"
        raise ArgumentError(msg)
    return f"{int(found.group(1)):02d}:{found.group(2)}"


_EVERY_DAY = tuple(range(_WEEK))
_OWN_DAYS = frozenset({"daily", "weekdays", "weekends", "every"})  # repeats that set their days
_REPEAT_OF: dict[str, Repeat] = {repeat: repeat for repeat in REPEATS}


def _next_on(days: tuple[int, ...], clock: str, after: float) -> float:
    """Return the first `clock` on one of `days` (never empty) later than `after`."""
    start = _local(after)
    moments = (_at(start + timedelta(days=ahead), clock) for ahead in range(_WEEK + 1))
    return next(m.timestamp() for m in moments if m.weekday() in days and m.timestamp() > after)


def following(job: Job, after: float) -> float | None:
    """Return when a job is due next after `after`, or None for a one-off job."""
    if job.repeat == "none":
        return None
    if job.repeat == "every":
        step = job.every * 60.0
        missed = max(0, int((after - job.next) // step) + 1)
        return job.next + missed * step
    return _next_on(job.days, job.at, after)


@dataclass(frozen=True, slots=True)
class _When:
    """The time arguments of `schedule_add`, checked."""

    minutes: int | None
    clock: str
    day: str
    every: int
    seconds: int | None = None


def _when_of(args: dict[str, Any]) -> _When:
    """Return the time arguments of `schedule_add`.

    Raises:
        ArgumentError: `at` is not a time of day, `day` not a day, or more than one of
            `in_seconds`, `in_minutes` and `at` is given.
    """
    minutes = args.get("in_minutes")
    seconds = args.get("in_seconds")
    clock = _clock_of(str(args["at"])) if args.get("at") else ""
    if sum(given is not None for given in (minutes, seconds, clock or None)) > 1:
        msg = "give only one of in_seconds, in_minutes and at"
        raise ArgumentError(msg)
    every = int(args.get("every_minutes") or 0)
    day = str(args.get("day", "")).strip().casefold()
    if args.get("repeat") in _OWN_DAYS:
        day = ""  # the repeat sets the days: a `day` beside it ("weekday") says nothing more
    if day and day not in {"today", "tomorrow", *DAY_NAMES}:
        msg = "day must be today, tomorrow or a weekday's name"
        raise ArgumentError(msg)
    return _When(
        None if minutes is None else int(minutes),
        clock,
        day,
        every,
        None if seconds is None else int(seconds),
    )


MAX_ORIGIN = 32  # characters of one outside origin a job keeps ("web", "file", "memory")


def plan(args: dict[str, Any], now: float, ident: int) -> Job:
    """Return the job the arguments of `schedule_add` describe, due first after `now`.

    Raises:
        ArgumentError: the time is missing, said twice, in the past or does not fit the repeat.
    """
    action: Action = "do" if args.get("action") == "do" else "remind"
    repeat = _REPEAT_OF.get(str(args.get("repeat", "none")), "none")
    text = " ".join(str(args["text"]).split())
    when = _when_of(args)
    if (repeat == "every") != bool(when.every):
        msg = "repeat every goes with every_minutes, and every_minutes only with it"
        raise ArgumentError(msg)
    if when.seconds is not None:
        if repeat != "none":
            msg = "in_seconds is for a one-off job; a repeating one takes in_minutes or at"
            raise ArgumentError(msg)
        return Job(ident, action, text, repeat, "", (), 0, now + float(when.seconds))
    if repeat == "every":
        first = now + 60.0 * (when.minutes if when.minutes is not None else when.every)
        if when.clock:
            first = _next_on(_EVERY_DAY, when.clock, now)
        return Job(ident, action, text, repeat, "", (), when.every, first)
    if repeat != "none":
        if not when.clock:
            msg = f"repeat {repeat} needs at, the time of day"
            raise ArgumentError(msg)
        days = _DAYS_OF.get(repeat, _EVERY_DAY)
        if repeat == "weekly":
            days = (DAY_NAMES.index(when.day) if when.day in DAY_NAMES else _local(now).weekday(),)
        first = _next_on(days, when.clock, now)
        return Job(ident, action, text, repeat, when.clock, days, 0, first)
    if when.minutes is not None:
        return Job(ident, action, text, repeat, "", (), 0, now + 60.0 * when.minutes)
    return Job(ident, action, text, repeat, "", (), 0, _once_at(when, now))


def _once_at(when: _When, now: float) -> float:
    """Return the next `when.clock` on `when.day` (today, tomorrow, a weekday), or the soonest.

    Raises:
        ArgumentError: no time was given, or it has passed today.
    """
    if not when.clock:
        msg = "say when: in_seconds, in_minutes, or at a time of day"
        raise ArgumentError(msg)
    today = _local(now)
    if when.day == "today":
        moment = _at(today, when.clock).timestamp()
        if moment <= now:
            msg = f"{when.clock} has already passed today"
            raise ArgumentError(msg)
        return moment
    if when.day == "tomorrow":
        return _at(today + timedelta(days=1), when.clock).timestamp()
    days = (DAY_NAMES.index(when.day),) if when.day in DAY_NAMES else _EVERY_DAY
    return _next_on(days, when.clock, now)


def _on_day(moment: datetime, now: float, clock: str) -> str:
    """Return the day and time of `moment` as said from `now`: "tomorrow at 9:00 AM"."""
    days = (moment.date() - _local(now).date()).days
    if days in {0, 1}:
        return f"{'today' if days == 0 else 'tomorrow'} at {clock}"
    if days < _WEEK:
        return f"on {moment:%A} at {clock}"
    return f"on {moment:%A} {moment.day} {moment:%B} at {clock}"


def _repeating(job: Job, now: float, clock: str) -> str | None:
    """Return how a repeating job's times are said, or None for a one-off job."""
    if job.repeat == "every":
        hours, minutes = divmod(job.every, _HOUR)
        step = f"{hours} hours" if hours and not minutes else f"{job.every} minutes"
        moment = _local(job.next)
        next_one = (
            f"at {clock}" if moment.date() == _local(now).date() else _on_day(moment, now, clock)
        )
        return f"every {'hour' if job.every == _HOUR else step}, next {next_one}"
    if job.repeat == "daily":
        return f"every day at {clock}"
    if job.repeat in {"weekdays", "weekends"}:
        return f"on {job.repeat} at {clock}"
    if job.repeat == "weekly":
        return f"every {DAY_NAMES[job.days[0]].capitalize()} at {clock}"
    return None


def spoken_when(job: Job, now: float, *, twelve_hour: bool) -> str:
    """Return when a job is due as it is said: "today at 7:30 AM", "every day at 7:30 AM"."""
    moment = _local(job.next)
    clock = spoken_time(moment, twelve_hour=twelve_hour)
    repeating = _repeating(job, now, clock)
    if repeating is not None:
        return repeating
    seconds = round(job.next - now)
    if 0 < seconds < _IN_SECONDS_UNDER:
        return f"in {seconds} second{'' if seconds == 1 else 's'}"
    return _on_day(moment, now, clock)


class JobStore:
    """The jobs file: read and changed under a lock, so every Jarvis process sees one list."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        """Keep the jobs in `path` (with a lock file beside it); `clock` is wall time (tests)."""
        super().__init__()
        self._path = path
        self._clock = clock
        self._seen: tuple[int, int, int] | None = None
        self._soonest: float | None = None
        self._looked = float("-inf")
        self._foreign = False  # the file was written by another version of Jarvis: never overwrite

    @contextlib.contextmanager
    def _locked(self) -> Generator[tuple[int, list[Job]]]:
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = self._path.with_name(self._path.name + ".lock")
        fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield self._read()
        finally:
            os.close(fd)

    def _read(self) -> tuple[int, list[Job]]:
        """Return the next free id and the jobs; a missing or damaged file holds none.

        A file of another version holds none either, and is kept as it is (see `_write`).
        """
        self._foreign = False
        try:
            text = read_private(self._path)
            data = obj(json.loads(text or "{}"))
        except (OSError, ValueError) as exc:
            log.warning("the jobs file could not be read: %s", exc)
            return 1, []
        if data.get("version") not in {None, FILE_VERSION}:
            self._foreign = True
            return 1, []
        jobs: list[Job] = []
        for item in objs(data.get("jobs")):
            with contextlib.suppress(KeyError, TypeError, ValueError):
                jobs.append(_job(item))
        free = data.get("next_id")
        start = free if isinstance(free, int) else 1
        return max([start, *(job.id + 1 for job in jobs)]), jobs

    def _write(self, free: int, jobs: list[Job]) -> None:
        """Save the jobs, unless the file is another version's, which would lose its jobs.

        Raises:
            OperationError: the file was written by a newer (or unknown) version of Jarvis.
        """
        if self._foreign:
            msg = (
                "the reminders file was written by a newer version of Jarvis, so this one "
                "leaves it alone; update Jarvis to change reminders"
            )
            raise OperationError(msg)
        data = {"version": FILE_VERSION, "next_id": free, "jobs": [asdict(job) for job in jobs]}
        write_atomic(self._path, json.dumps(data, ensure_ascii=False, indent=1), mode=0o600)
        self._seen = None  # look again at once

    def jobs(self) -> list[Job]:
        """Return every job, the soonest first."""
        with self._locked() as (_free, jobs):
            return sorted(jobs, key=lambda job: job.next)

    def add(
        self, args: dict[str, Any], *, foreign: bool = False, outside: Iterable[str] = ()
    ) -> Job:
        """Add the job `schedule_add`'s arguments describe and return it.

        `foreign` and `outside` are where the request that adds it came from (`Job`).
        `plan` raises `ArgumentError` when they do not describe a time to come.

        Raises:
            OperationError: there are `MAX_JOBS` already.
        """
        with self._locked() as (free, jobs):
            if len(jobs) >= MAX_JOBS:
                msg = f"there are {MAX_JOBS} reminders and scheduled requests already"
                raise OperationError(msg)
            job = replace(
                plan(args, self._clock(), free),
                foreign=foreign,
                outside=_origins(list(outside)),
            )
            self._write(free + 1, [*jobs, job])
            return job

    def cancel(self, ident: int | None) -> list[Job]:
        """Remove the job with `ident`, or every job for None; return what was removed."""
        with self._locked() as (free, jobs):
            gone = [job for job in jobs if ident is None or job.id == ident]
            if gone:
                self._write(free, [job for job in jobs if job not in gone])
            return gone

    def take_due(self) -> list[Due]:
        """Return the jobs whose time came, moving each on to its next time or removing it.

        It reads the file at most every `POLL_S` and only when it changed, so it is cheap to call
        between every two frames of sound.
        """
        now = self._clock()
        if now - self._looked < POLL_S:
            return []
        self._looked = now
        if not self._changed() and (self._soonest is None or now < self._soonest):
            return []
        with self._locked() as (free, jobs):
            due = [Due(job, now - job.next) for job in jobs if job.next <= now]
            kept: list[Job] = []
            for job in jobs:
                if job.next > now:
                    kept.append(job)
                elif (later := following(job, now)) is not None:
                    kept.append(replace(job, next=later))
            if due:
                self._write(free, kept)
            self._soonest = min((job.next for job in kept), default=None)
            self._seen = self._stamp()
        return sorted(due, key=lambda item: item.job.next)

    def _stamp(self) -> tuple[int, int, int] | None:
        try:
            info = self._path.stat()
        except OSError:
            return None
        return info.st_ino, info.st_mtime_ns, info.st_size

    def _changed(self) -> bool:
        stamp = self._stamp()
        if stamp == self._seen:
            return False
        self._seen = None
        return True


def _job(item: dict[str, Any]) -> Job:
    """Return a job read back from the file.

    An entry that is not one raises `KeyError`, `TypeError` or, as here, `ValueError`.

    Raises:
        ValueError: an unknown action or repeat, or a repeat without its days or minutes.
    """
    action, repeat = str(item["action"]), str(item["repeat"])
    if action not in ACTIONS or repeat not in REPEATS:
        raise ValueError(action)
    days = tuple(
        int(day) for day in cast("list[int]", item.get("days") or []) if day in range(_WEEK)
    )
    every = int(item.get("every") or 0)
    if (repeat == "every" and every < MIN_EVERY_MINUTES) or (
        repeat not in {"none", "every"} and not days
    ):
        raise ValueError(repeat)
    at = _clock_of(str(item["at"])) if repeat not in {"none", "every"} else ""
    return Job(
        int(item["id"]),
        action,
        str(item["text"])[:MAX_TEXT],
        repeat,
        at,
        days,
        every,
        float(item["next"]),
        # A file from before jobs kept these reads as the owner's own request, as it ran then.
        foreign=item.get("foreign") is True,
        outside=_origins(item.get("outside")),
    )


def _origins(value: object) -> tuple[str, ...]:
    """Return the outside origins a job's entry holds: short texts, each once, sorted."""
    if not isinstance(value, list):
        return ()
    texts = cast("list[object]", value)
    return tuple(sorted({o for o in texts if isinstance(o, str) and 0 < len(o) <= MAX_ORIGIN}))


def _words_of(jobs: Iterable[Job]) -> Origin:
    """Return the origin of the jobs' words, untrusted when outside text could have steered them.

    A job written beside outside text or for PC audio has the model's words, so they reach a later
    request as data, not instructions.
    """
    return Origin.MODEL if any(job.foreign or job.outside for job in jobs) else Origin.TOOL_STATUS


_TEXT = {"type": "string", "minLength": 1, "maxLength": MAX_TEXT}


class ScheduleOps:
    """`schedule_add`, `schedule_list` and `schedule_cancel` over one jobs file."""

    def __init__(
        self,
        store: JobStore,
        *,
        twelve_hour: bool = True,
        clock: Callable[[], float] = time.time,
        voice_running: Callable[[], bool] = lambda: True,
    ) -> None:
        """Keep jobs in `store`; `voice_running` says whether a voice service will say them."""
        super().__init__()
        self._store = store
        self._twelve_hour = twelve_hour
        self._clock = clock
        self._voice_running = voice_running

    def _when(self, job: Job) -> str:
        return spoken_when(job, self._clock(), twelve_hour=self._twelve_hour)

    def add(self, args: dict[str, Any]) -> Result:
        """Add a reminder or a scheduled request and say when it is due.

        Raises:
            OperationError: the arguments describe no time to come (the model is told why, to
                fix them), or there are `MAX_JOBS` already.
        """
        request = current_request()
        try:
            job = (
                self._store.add(args)
                if request is None
                else self._store.add(args, foreign=request.foreign, outside=request.outside)
            )
        except ArgumentError as exc:
            raise OperationError(str(exc)) from exc
        what = "remind you" if job.action == "remind" else "do that"
        sentence = f"Okay, I'll {what} {self._when(job)}."
        if job.action == "do" and (job.foreign or job.outside):
            # It runs as it was asked for: its actions will need a yes (ADR 0011).
            sentence += " I'll check with you before it acts."
        if not self._voice_running():
            sentence += " Jarvis's voice service isn't running now; start it, or it won't be said."
        text = f"Added job {job.id} ({job.action}): {job.text!r}, {self._when(job)}."
        return Result(text, Origin.TOOL_STATUS, spoken=sentence)

    def list_jobs(self, _args: dict[str, Any]) -> Result:
        """List the reminders and scheduled requests, the soonest first."""
        jobs = self._store.jobs()
        if not jobs:
            sentence = "You have no reminders or scheduled requests."
            return Result(sentence, Origin.TOOL_STATUS, spoken=sentence)
        lines = [
            f"{job.id}. {'reminder' if job.action == 'remind' else 'request'} "
            f"{job.text!r}, {self._when(job)}"
            for job in jobs
        ]
        return Result("\n".join(lines), _words_of(jobs))

    def cancel(self, args: dict[str, Any]) -> Result:
        """Cancel one job by its number, or every job.

        Raises:
            OperationError: `id` is neither a number nor `all` (the model is told so).
        """
        which = str(args["id"]).strip().lower()
        if which != "all" and not which.isdigit():
            msg = "id must be a job's number from schedule_list, or all"
            raise OperationError(msg)
        gone = self._store.cancel(None if which == "all" else int(which))
        if not gone:
            sentence = (
                "There was nothing to cancel." if which == "all" else f"There is no job {which}."
            )
            return Result(sentence, Origin.TOOL_STATUS, ok=False, spoken=sentence)
        if len(gone) == 1:
            sentence = f"Cancelled: {gone[0].text}."
            origin = _words_of(gone)
        else:
            sentence = f"Cancelled all {len(gone)}."
            origin = Origin.TOOL_STATUS
        return Result(sentence, origin, spoken=sentence)

    @staticmethod
    def add_summary(args: dict[str, Any]) -> str:
        """The words of a question about adding a job."""
        what = "remind you" if args.get("action", "remind") == "remind" else "later do"
        return f"{what}: {args.get('text', '')}"

    def operations(self) -> list[Operation]:
        """Return the schedule operations."""
        return [
            Operation(
                "schedule_add",
                "Set a reminder (remind: said then) or a request (do: done then). When:"
                " in_seconds or in_minutes, or at HH:MM with day; repeat, every_minutes for every.",
                {
                    "type": "object",
                    "properties": {
                        "text": _TEXT,
                        "action": {"type": "string", "enum": list(ACTIONS)},
                        "in_seconds": {"type": "integer", "minimum": 1, "maximum": MAX_IN_SECONDS},
                        "in_minutes": {"type": "integer", "minimum": 1, "maximum": MAX_IN_MINUTES},
                        "at": {"type": "string", "maxLength": 5},
                        "day": {"type": "string", "maxLength": 9},
                        "repeat": {"type": "string", "enum": list(REPEATS)},
                        "every_minutes": {
                            "type": "integer",
                            "minimum": MIN_EVERY_MINUTES,
                            "maximum": MAX_EVERY_MINUTES,
                        },
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
                self.add,
                spoken_summary=self.add_summary,
            ),
            Operation(
                "schedule_list",
                "List reminders and scheduled requests by number.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                self.list_jobs,
                side_effect=False,
            ),
            Operation(
                "schedule_cancel",
                "Cancel a reminder or request: its number, or all.",
                {
                    "type": "object",
                    "properties": {"id": {"type": "string", "minLength": 1, "maxLength": 6}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
                self.cancel,
                spoken_summary=lambda a: (
                    f"cancel {'every job' if a.get('id') == 'all' else 'that job'}"
                ),
            ),
        ]
