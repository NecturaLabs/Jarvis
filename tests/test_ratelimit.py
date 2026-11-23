"""The cap on side-effecting actions: so many in a window, shared by every Jarvis process."""

import errno
import fcntl
import json
import logging
import os
import stat
import threading
import time
import types
from typing import TYPE_CHECKING

import pytest

from jarvis import ratelimit as ratelimit_module
from jarvis.ratelimit import ActionRate

if TYPE_CHECKING:
    from pathlib import Path


class Clock:
    def __init__(self) -> None:
        super().__init__()
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


def test_the_limit_holds_within_the_window_and_frees_up_as_it_slides() -> None:
    clock = Clock()
    rate = ActionRate(None, limit=3, window_s=600.0, clock=clock)

    assert [rate.take() for _ in range(4)] == [True, True, True, False]
    clock.now += 599
    assert not rate.take()
    clock.now += 2
    assert rate.take()


def test_the_voice_service_and_the_chat_share_one_count(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "state" / "actions.json"
    voice = ActionRate(path, limit=3, window_s=600.0, clock=clock)
    chat = ActionRate(path, limit=3, window_s=600.0, clock=clock)

    assert voice.take()
    assert chat.take()
    assert voice.take()
    assert not chat.take()
    assert not voice.take()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert len(json.loads(path.read_text(encoding="ascii"))["taken"]) == 3


def test_a_clock_set_back_does_not_lift_or_jam_the_limit(tmp_path: Path) -> None:
    clock = Clock()
    rate = ActionRate(tmp_path / "actions.json", limit=2, window_s=600.0, clock=clock)
    assert rate.take()
    assert rate.take()

    clock.now -= 3600  # what was taken now lies in the future: it still counts, for one window
    assert not rate.take()
    clock.now += 601
    assert rate.take()


def test_a_damaged_file_starts_the_count_again(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "actions.json"
    path.write_text("{not json", encoding="ascii")
    path.chmod(0o600)
    rate = ActionRate(path, limit=1, window_s=600.0, clock=Clock())

    with caplog.at_level(logging.WARNING, logger="jarvis.ratelimit"):
        assert rate.take()
    assert not rate.take()
    assert "actions.json" in caplog.text


@pytest.mark.parametrize("contents", ["", "[]", "{not json"])
def test_emptying_the_shared_file_does_not_lift_this_process_own_count(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "actions.json"
    rate = ActionRate(path, limit=3, window_s=600.0, clock=Clock())
    assert [rate.take() for _ in range(3)] == [True, True, True]

    path.write_text(contents, encoding="ascii")

    assert rate.full()
    assert not rate.take()
    other = ActionRate(path, limit=3, window_s=600.0, clock=Clock())  # nothing was taken in it
    assert [other.take() for _ in range(4)] == [True, True, True, False]


def test_a_file_that_cannot_be_used_still_leaves_this_process_its_own_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocked = tmp_path / "actions.json"
    blocked.mkdir()
    rate = ActionRate(blocked, limit=2, window_s=600.0, clock=Clock())

    with caplog.at_level(logging.WARNING, logger="jarvis.ratelimit"):
        assert [rate.take() for _ in range(3)] == [True, True, False]


def test_a_link_in_place_of_the_file_is_not_followed(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("keep me", encoding="ascii")
    (tmp_path / "actions.json").symlink_to(elsewhere)
    rate = ActionRate(tmp_path / "actions.json", limit=1, window_s=600.0, clock=Clock())

    assert rate.take()
    assert not rate.take()
    assert elsewhere.read_text(encoding="ascii") == "keep me"


# --- a damaged file, a failed write and a held lock (the review of #30) -----------------------


DEEP = '{"taken": ' + "[" * 100_000 + "]" * 100_000 + "}"


def _too_deep(_raw: object) -> object:
    raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")


@pytest.mark.parametrize(
    ("damaged", "deep"),
    [
        ('{"taken": [' + "9" * 400 + "]}", False),  # too long for a float: OverflowError
        # How deep JSON may nest before it raises depends on the stack the machine gives Python,
        # so what `json.loads` raises then is made to happen here.
        (DEEP, True),
    ],
    ids=["huge-number", "deep-nesting"],
)
def test_a_damaged_file_counts_as_empty_and_is_written_again(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    damaged: str,
    deep: bool,
) -> None:
    if deep:
        monkeypatch.setattr(
            ratelimit_module, "json", types.SimpleNamespace(loads=_too_deep, dumps=json.dumps)
        )
    path = tmp_path / "actions.json"
    path.write_text(damaged, encoding="ascii")
    path.chmod(0o600)
    clock = Clock()
    rate = ActionRate(path, limit=2, window_s=600.0, clock=clock)

    with caplog.at_level(logging.WARNING, logger="jarvis.ratelimit"):
        assert rate.take()
    assert json.loads(path.read_text(encoding="ascii")) == {"taken": [clock.now]}
    assert "damaged" in caplog.text

    path.write_text(damaged, encoding="ascii")  # damaged again: a look rewrites it too
    assert not rate.full()
    assert json.loads(path.read_text(encoding="ascii")) == {"taken": []}


def test_full_says_whether_the_window_has_room_and_counts_nothing(tmp_path: Path) -> None:
    path = tmp_path / "actions.json"
    rate = ActionRate(path, limit=1, window_s=600.0, clock=Clock())
    alone = ActionRate(None, limit=1, window_s=600.0, clock=Clock())

    assert [rate.full(), rate.full(), alone.full(), alone.full()] == [False] * 4
    assert rate.take()
    assert alone.take()
    assert rate.full()
    assert alone.full()


def test_a_failed_write_leaves_the_shared_count_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    path = tmp_path / "actions.json"
    voice = ActionRate(path, limit=3, window_s=600.0, clock=clock)
    chat = ActionRate(path, limit=3, window_s=600.0, clock=clock)
    assert voice.take()
    assert voice.take()
    before = path.read_bytes()

    def no_room(_fd: int, _data: bytes) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as patched:
        patched.setattr(ratelimit_module.os, "write", no_room)
        assert chat.take()  # this process still counts its own

    assert path.read_bytes() == before
    assert voice.take()
    assert not voice.take()


def test_a_write_the_system_takes_in_pieces_is_written_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    path = tmp_path / "actions.json"
    real = os.write

    def three_bytes(fd: int, data: bytes) -> int:
        return real(fd, bytes(data[:3]))

    monkeypatch.setattr(ratelimit_module.os, "write", three_bytes)
    rate = ActionRate(path, limit=3, window_s=600.0, clock=clock)

    assert rate.take()
    assert rate.take()
    assert json.loads(path.read_text(encoding="ascii")) == {"taken": [clock.now, clock.now]}


def test_a_lock_held_too_long_falls_back_to_this_process_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    clock = Clock()
    path = tmp_path / "actions.json"
    ActionRate(path, limit=5, window_s=600.0, clock=clock).take()
    before = path.read_bytes()
    rate = ActionRate(path, limit=2, window_s=600.0, clock=clock, lock_wait_s=0.05)
    holder = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX)
        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="jarvis.ratelimit"):
            assert [rate.take() for _ in range(3)] == [True, True, False]
            assert rate.full()  # this process's own count, with the file out of reach
        assert time.monotonic() - started < 5.0
    finally:
        os.close(holder)

    assert "actions.json" in caplog.text
    assert "lock" in caplog.text
    assert path.read_bytes() == before


def test_processes_taking_places_at_once_never_take_more_than_the_limit(tmp_path: Path) -> None:
    path = tmp_path / "actions.json"
    limit, workers, tries = 25, 8, 10
    clock = Clock()
    taken: list[bool] = []
    start = threading.Barrier(workers)

    def work() -> None:
        rate = ActionRate(path, limit=limit, window_s=600.0, clock=clock)
        start.wait()
        taken.extend(rate.take() for _ in range(tries))

    threads = [threading.Thread(target=work) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert taken.count(True) == limit
    assert len(json.loads(path.read_text(encoding="ascii"))["taken"]) == limit
