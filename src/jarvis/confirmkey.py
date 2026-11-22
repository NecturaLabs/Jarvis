"""The confirm key: a `strong` action runs only after the owner presses it (`jarvis confirm`).

Jarvis asks about one exact action, and only once the question has been heard does it wait for
the key (`arm`): the wait is bound to that action's hash, and one press allows that action
alone, once. The owner binds `jarvis confirm` to a key; it reaches the voice loop through the
control socket, which only this user's processes not started by Jarvis can use (`control`).

A press that cannot be the owner answering allows nothing and is audited: one with nothing
waiting, or one sooner after the wait began than a person can react. Either holds off the next
wait for `STRAY_HOLD_S`, so a program pressing the key over and over cannot catch a question.
Silence, a timeout or a cancelled wait is no.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

MIN_PRESS_S = 0.3  # a press sooner after the wait began than this is no answer to it
STRAY_HOLD_S = 30.0  # how long a stray press holds off the next wait
NOTHING_WAITING = "nothing is waiting for the confirm key"


class _Audit(Protocol):
    def record(self, event: str, **fields: object) -> None: ...


class _NoAudit:
    def record(self, event: str, **fields: object) -> None:
        pass


@dataclass(slots=True)
class Waiting:
    """One action waiting for the key."""

    action: str  # its hash (`policy.action_hash`)
    summary: str  # what the owner was asked, shown by `jarvis confirm`
    armed_at: float
    pressed: threading.Event = field(default_factory=threading.Event)
    spoiled: bool = False  # a press came too soon: nothing allows this wait any more


class ConfirmKey:
    """Holds at most one action waiting for the key, shared by the voice loop and the socket."""

    def __init__(
        self, *, clock: Callable[[], float] = time.monotonic, audit: _Audit | None = None
    ) -> None:
        """Time presses with `clock`; record each press and each refused wait in `audit`."""
        super().__init__()
        self._clock = clock
        self._audit = audit or _NoAudit()
        self._lock = threading.Lock()
        self._waiting: Waiting | None = None
        self._stray_at: float | None = None

    def arm(self, action: str, summary: str) -> Waiting | None:
        """Start waiting for the key for `action`; None while held off or another one waits."""
        with self._lock:
            now = self._clock()
            if self._waiting is not None:
                self._audit.record("confirm_key", outcome="busy", action=action)
                return None
            if self._stray_at is not None and now - self._stray_at < STRAY_HOLD_S:
                self._audit.record("confirm_key", outcome="held off", action=action)
                return None
            self._waiting = Waiting(action, summary, now)
            return self._waiting

    def press(self) -> dict[str, str | None]:
        """Take one press of the key (`jarvis confirm`); return what it confirmed, if anything."""
        with self._lock:
            now = self._clock()
            waiting = self._waiting
            if waiting is None:
                self._stray_at = now
                self._audit.record("confirm_key", outcome="nothing waiting")
                return {"confirmed": None, "reason": NOTHING_WAITING}
            if waiting.spoiled:
                self._audit.record("confirm_key", outcome="spoiled", action=waiting.action)
                return {"confirmed": None, "reason": "that question was already answered too soon"}
            if now - waiting.armed_at < MIN_PRESS_S:
                waiting.spoiled = True
                self._stray_at = now
                self._audit.record("confirm_key", outcome="too soon", action=waiting.action)
                return {"confirmed": None, "reason": "pressed before the question was heard"}
            waiting.pressed.set()
            self._audit.record("confirm_key", outcome="pressed", action=waiting.action)
            return {"confirmed": waiting.summary}

    def disarm(self, waiting: Waiting) -> bool:
        """Stop waiting; return True if the key was pressed for `waiting` in time."""
        with self._lock:
            if self._waiting is waiting:
                self._waiting = None
            return waiting.pressed.is_set()  # a spoiled wait is never pressed
