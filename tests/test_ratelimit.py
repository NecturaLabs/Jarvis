"""The cap on side-effecting actions: so many in a window, shared by every Jarvis process."""

import json
import logging
import stat
from typing import TYPE_CHECKING

from jarvis.ratelimit import ActionRate

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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
