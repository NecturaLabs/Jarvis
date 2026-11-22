"""The confirm key: a `strong` action runs only after the owner presses it (`jarvis confirm`).

Jarvis asks about one exact action, and only once the question has been heard does it wait for
the key (`arm`): the wait is bound to that action's hash and lasts `window_s`, timed on the boot
clock so a suspend does not stretch it. One press inside that time allows that action alone, once.
The owner binds `jarvis confirm` to a key; it reaches the voice loop through the control socket,
which only this user's processes can use (`control`).

A press that cannot be the owner answering allows nothing and is audited: one with nothing
waiting (none asked, or the wait timed out or was cancelled), or one sooner after the wait began
than a person can react. Either holds off the next wait for `STRAY_HOLD_S`, so a program pressing
the key over and over cannot catch a question. Silence, a timeout or a cancel is no.

The key guards against the model and what it reads, not against other programs running as the
owner: any of them could run `jarvis confirm`. The socket refuses a press from a program Jarvis
started as a best effort (`control.started_by_jarvis`).
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


def boot_clock() -> float:
    """Seconds since boot, suspend included: a wait cannot outlast its time by a sleep."""
    return time.clock_gettime(time.CLOCK_BOOTTIME)


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
    expires_at: float
    pressed: threading.Event = field(default_factory=threading.Event)


class ConfirmKey:
    """Holds at most one action waiting for the key, shared by the voice loop and the socket."""

    def __init__(
        self, *, clock: Callable[[], float] = boot_clock, audit: _Audit | None = None
    ) -> None:
        """Time waits and presses with `clock`; record presses and refused waits in `audit`."""
        super().__init__()
        self._clock = clock
        self._audit = audit or _NoAudit()
        self._lock = threading.Lock()
        self._waiting: Waiting | None = None
        self._stray_at: float | None = None

    def arm(self, action: str, summary: str, window_s: float) -> Waiting | None:
        """Wait `window_s` for the key for `action`; None while held off or another one waits."""
        with self._lock:
            now = self._clock()
            if self._waiting is not None:
                self._audit.record("confirm_key", outcome="busy", action=action)
                return None
            if self._stray_at is not None and now - self._stray_at < STRAY_HOLD_S:
                self._audit.record("confirm_key", outcome="held off", action=action)
                return None
            self._waiting = Waiting(action, summary, now, now + window_s)
            return self._waiting

    def press(self) -> dict[str, str | None]:
        """Take one press of the key (`jarvis confirm`); return what it confirmed, if anything."""
        with self._lock:
            now = self._clock()
            waiting = self._waiting
            if waiting is None or waiting.pressed.is_set() or now >= waiting.expires_at:
                self._stray_at = now
                self._audit.record(
                    "confirm_key",
                    outcome="nothing waiting",
                    **({} if waiting is None else {"action": waiting.action}),
                )
                return {"confirmed": None, "reason": NOTHING_WAITING}
            if now - waiting.armed_at < MIN_PRESS_S:
                self._waiting = None  # spoiled: nothing allows this wait any more
                self._stray_at = now
                self._audit.record("confirm_key", outcome="too soon", action=waiting.action)
                return {"confirmed": None, "reason": "pressed before the question was heard"}
            waiting.pressed.set()
            self._waiting = None  # answered: a second press finds nothing waiting
            self._audit.record("confirm_key", outcome="pressed", action=waiting.action)
            return {"confirmed": waiting.summary}

    def open(self, waiting: Waiting) -> bool:
        """Whether `waiting` still waits: not pressed, timed out, spoiled or cancelled."""
        with self._lock:
            return self._waiting is waiting and self._clock() < waiting.expires_at

    def cancel(self) -> None:
        """End whatever waits, unpressed (`jarvis hush` or `pause`)."""
        with self._lock:
            waiting, self._waiting = self._waiting, None
        if waiting is not None:
            self._audit.record("confirm_key", outcome="cancelled", action=waiting.action)

    def disarm(self, waiting: Waiting, why: str) -> bool:
        """Stop waiting; return True if the key was pressed for `waiting` in time.

        `why` is what ended the wait, for the audit log when it was not a press.
        """
        with self._lock:
            if self._waiting is waiting:
                self._waiting = None
            pressed = waiting.pressed.is_set()
        if not pressed:
            self._audit.record("confirm_key", outcome="ended", action=waiting.action, why=why)
        return pressed
