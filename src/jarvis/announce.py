"""Announcements: the lines Jarvis says on its own (a reminder, a watch alert), kept to pass on.

The voice service says an announcement aloud. It also `publish`es it here, so a text front end
that follows the service (`jarvis acp`, through the control socket's `subscribe`, see `control`)
can show it too. Nothing here speaks, notifies the desktop or reaches the network.

To announce something, call `publish(text, kind)` from the voice service's process:

    announce.publish("Your tea is ready.", "reminder")

`kind` names where the line came from: `REMINDER`, `WATCH` or `PROCESS`. `publish` never blocks and
never raises: a line goes into a short ring in memory (the last `KEEP`, each with an id and a
time) and into each subscriber's own bounded queue, whose oldest line is dropped when it is full.
"""

import collections
import contextlib
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from jarvis.jsonutil import obj

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

REMINDER = "reminder"
WATCH = "watch"
PROCESS = "process"
KINDS = frozenset({REMINDER, WATCH, PROCESS})
KEEP = 50  # announcements the ring remembers
QUEUE = 100  # lines one subscriber may have waiting before its oldest is dropped
MAX_TEXT = 2000  # characters of one announcement


@dataclass(frozen=True, slots=True)
class Announcement:
    """One line: its number (from 1, in the order published), the time and where it came from."""

    id: int
    at: float  # seconds since the epoch
    kind: str
    text: str

    def to_json(self) -> dict[str, Any]:
        """Return the announcement as it travels on the control socket."""
        return {"id": self.id, "at": self.at, "kind": self.kind, "text": self.text}

    @classmethod
    def from_json(cls, value: object) -> Announcement | None:
        """Return the announcement `value` holds, or None when it is not one."""
        found = obj(value)
        number, at, kind, text = (found.get(k) for k in ("id", "at", "kind", "text"))
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or not isinstance(at, int | float)
            or not isinstance(kind, str)
            or not isinstance(text, str)
        ):
            return None
        return cls(number, float(at), kind, text)


class Subscription:
    """What one subscriber has been sent and not yet taken: a bounded queue of announcements."""

    def __init__(self, hub: Hub, size: int) -> None:
        """Wait for announcements from `hub`, at most `size` at a time."""
        super().__init__()
        self._hub = hub
        self._queue = collections.deque[Announcement](maxlen=size)
        self._ready = threading.Condition()
        self._closed = False
        self.dropped = 0  # announcements lost to a full queue, not yet reported

    def push(self, item: Announcement) -> None:
        """Queue `item` (the hub's call); a full queue loses its oldest first."""
        with self._ready:
            if len(self._queue) == self._queue.maxlen:
                self.dropped += 1  # the oldest goes
            self._queue.append(item)
            self._ready.notify()

    def get(self, timeout: float) -> Announcement | None:
        """Return the next announcement, or None after `timeout` seconds or once closed."""
        with self._ready:
            self._ready.wait_for(lambda: bool(self._queue) or self._closed, timeout)
            return self._queue.popleft() if self._queue else None

    def take_dropped(self) -> int:
        """Return how many announcements were dropped since the last call."""
        with self._ready:
            lost, self.dropped = self.dropped, 0
            return lost

    def close(self) -> None:
        """Stop receiving; a waiting `get` returns."""
        self._hub.unsubscribe(self)
        with self._ready:
            self._closed = True
            self._ready.notify_all()

    def __enter__(self) -> Self:
        """Use the subscription in a `with` that closes it."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the subscription."""
        self.close()


class Hub:
    """The ring of recent announcements and the subscribers waiting for the next."""

    def __init__(
        self, keep: int = KEEP, clock: Callable[[], float] = time.time, queue: int = QUEUE
    ) -> None:
        """Remember the last `keep` announcements; each subscriber may queue `queue`."""
        super().__init__()
        self._ring = collections.deque[Announcement](maxlen=keep)
        self._clock = clock
        self._queue = queue
        self._lock = threading.Lock()
        self._subscribers: list[Subscription] = []
        self._last = 0

    def publish(self, text: str, kind: str) -> Announcement | None:
        """Remember `text` and hand it to every subscriber; return it, or None for no words.

        `text` is cut to `MAX_TEXT` characters. An unknown `kind` is kept as it is.
        """
        words = text.strip()[:MAX_TEXT]
        if not words:
            return None
        with self._lock:
            self._last += 1
            item = Announcement(self._last, self._clock(), kind, words)
            self._ring.append(item)
            for subscriber in self._subscribers:
                subscriber.push(item)
        return item

    def recent(self, after: int = 0) -> list[Announcement]:
        """Return the remembered announcements numbered above `after`, oldest first."""
        with self._lock:
            return [item for item in self._ring if item.id > after]

    def latest(self) -> int:
        """Return the number of the last announcement (0 before the first)."""
        with self._lock:
            return self._last

    def subscribe(self, after: int | None = None) -> Subscription:
        """Return a subscription to every announcement from now on.

        With `after`, those remembered above that number come first, so nothing between is lost.
        """
        subscription = Subscription(self, self._queue)
        with self._lock:
            if after is not None:
                for item in self._ring:
                    if item.id > after:
                        subscription.push(item)
            self._subscribers.append(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        """Stop handing announcements to `subscription`."""
        with self._lock, contextlib.suppress(ValueError):
            self._subscribers.remove(subscription)

    def subscribers(self) -> int:
        """Return how many subscriptions are open."""
        with self._lock:
            return len(self._subscribers)


HUB = Hub()  # this process's own; the voice service's is the one the control socket serves


def publish(text: str, kind: str) -> None:
    """Announce `text` (a line the voice service says on its own) to every text front end.

    Never blocks and never raises: an announcement that cannot be passed on is only not shown.
    """
    with contextlib.suppress(Exception):
        HUB.publish(text, kind)
