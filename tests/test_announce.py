"""The announcements hub: a ring of what Jarvis said on its own, and subscribers to the next."""

import threading
import time
from typing import override

import pytest

from jarvis import announce
from jarvis.announce import Announcement, Hub


def test_a_published_line_is_numbered_timed_and_kept() -> None:
    hub = Hub(clock=lambda: 1000.0)

    first = hub.publish("  Reminder: tea  ", announce.REMINDER)
    second = hub.publish("Mail is waiting", announce.WATCH)

    assert first == Announcement(1, 1000.0, "reminder", "Reminder: tea")
    assert second is not None
    assert second.id == 2
    assert hub.recent() == [first, second]
    assert hub.latest() == 2


def test_nothing_is_published_for_no_words() -> None:
    hub = Hub()

    assert hub.publish("   ", announce.REMINDER) is None
    assert hub.recent() == []
    assert hub.latest() == 0


def test_a_long_line_is_cut_to_the_limit() -> None:
    hub = Hub()

    item = hub.publish("x" * (announce.MAX_TEXT + 500), announce.WATCH)

    assert item is not None
    assert len(item.text) == announce.MAX_TEXT


def test_the_ring_keeps_only_the_last_few_and_recent_continues_after_a_number() -> None:
    hub = Hub(keep=3)
    for number in range(1, 6):
        hub.publish(f"line {number}", announce.PROCESS)

    assert [a.id for a in hub.recent()] == [3, 4, 5]
    assert [a.text for a in hub.recent(after=3)] == ["line 4", "line 5"]
    assert hub.recent(after=5) == []


def test_a_subscriber_gets_what_is_published_from_then_on() -> None:
    hub = Hub()
    hub.publish("before", announce.REMINDER)

    with hub.subscribe() as feed:
        hub.publish("after", announce.REMINDER)

        got = feed.get(1.0)
        assert got is not None
        assert got.text == "after"
        assert feed.get(0.01) is None


def test_a_subscriber_that_asks_to_continue_gets_what_it_missed_first() -> None:
    hub = Hub()
    for text in ("a", "b", "c"):
        hub.publish(text, announce.REMINDER)

    with hub.subscribe(after=1) as feed:
        hub.publish("d", announce.REMINDER)

        assert [getattr(feed.get(1.0), "text", None) for _ in range(3)] == ["b", "c", "d"]


def test_a_full_queue_drops_its_oldest_line_and_counts_it_and_the_hub_never_waits() -> None:
    hub = Hub(queue=2)

    with hub.subscribe() as slow:
        started = time.monotonic()
        for number in range(1, 6):
            hub.publish(f"line {number}", announce.WATCH)
        assert time.monotonic() - started < 1.0

        assert [getattr(slow.get(0.1), "text", None) for _ in range(2)] == ["line 4", "line 5"]
        assert slow.take_dropped() == 3
        assert slow.take_dropped() == 0
    assert [a.id for a in hub.recent()] == [1, 2, 3, 4, 5]  # the ring is not the queue


def test_one_slow_subscriber_does_not_hold_another_back() -> None:
    hub = Hub(queue=1)

    with hub.subscribe() as slow, hub.subscribe() as quick:
        hub.publish("one", announce.WATCH)
        assert getattr(quick.get(1.0), "text", None) == "one"
        hub.publish("two", announce.WATCH)

        assert getattr(quick.get(1.0), "text", None) == "two"
        assert getattr(slow.get(1.0), "text", None) == "two"


def test_closing_a_subscription_stops_it_and_wakes_a_waiting_reader() -> None:
    hub = Hub()
    feed = hub.subscribe()
    got: list[Announcement | None] = []
    reader = threading.Thread(target=lambda: got.append(feed.get(10.0)))
    reader.start()

    feed.close()
    reader.join(5.0)

    assert not reader.is_alive()
    assert got == [None]
    assert hub.subscribers() == 0
    hub.publish("late", announce.WATCH)
    assert feed.get(0.01) is None


def test_the_module_level_publish_goes_to_this_process_hub_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = Hub()
    monkeypatch.setattr(announce, "HUB", hub)

    announce.publish("Reminder: tea", announce.REMINDER)
    assert [a.text for a in hub.recent()] == ["Reminder: tea"]

    class Broken(Hub):
        @override
        def publish(self, text: str, kind: str) -> Announcement | None:
            msg = text + kind
            raise RuntimeError(msg)

    monkeypatch.setattr(announce, "HUB", Broken())
    announce.publish("unseen", announce.WATCH)  # the voice loop is not stopped by it


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"id": "1", "at": 1.0, "kind": "x", "text": "t"},
        {"id": True, "at": 1.0, "kind": "x", "text": "t"},
        {"id": 1, "at": "now", "kind": "x", "text": "t"},
        {"id": 1, "at": 1.0, "kind": 3, "text": "t"},
    ],
)
def test_a_line_that_is_not_an_announcement_is_not_read_as_one(value: object) -> None:
    assert Announcement.from_json(value) is None


def test_an_announcement_survives_the_trip_through_json() -> None:
    item = Announcement(7, 12.5, "process", "firefox stopped")

    assert Announcement.from_json(item.to_json()) == item
