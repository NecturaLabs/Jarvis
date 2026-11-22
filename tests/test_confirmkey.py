"""The confirm key: one press allows one exact `strong` action, and only while it waits."""

from jarvis.audit import AuditLog
from jarvis.confirmkey import MIN_PRESS_S, STRAY_HOLD_S, ConfirmKey


class Clock:
    def __init__(self) -> None:
        super().__init__()
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _key() -> tuple[ConfirmKey, Clock, AuditLog]:
    clock = Clock()
    audit = AuditLog(None)
    return ConfirmKey(clock=clock, audit=audit), clock, audit


def _key_events(audit: AuditLog) -> list[dict[str, object]]:
    return [e for e in audit.events if e["event"] == "confirm_key"]


def test_a_press_while_an_action_waits_allows_that_action() -> None:
    key, clock, audit = _key()
    waiting = key.arm("hash-a", "stop process 42")
    assert waiting is not None
    clock.now += 2

    reply = key.press()

    assert reply == {"confirmed": "stop process 42"}
    assert waiting.pressed.is_set()
    assert key.disarm(waiting) is True
    assert _key_events(audit)[-1] == {
        "event": "confirm_key",
        "ts": audit.events[-1]["ts"],
        "outcome": "pressed",
        "action": "hash-a",
    }


def test_an_action_nobody_pressed_for_is_not_allowed() -> None:
    key, clock, _ = _key()
    waiting = key.arm("hash-a", "stop process 42")
    assert waiting is not None
    clock.now += 30

    assert key.disarm(waiting) is False


def test_a_press_with_nothing_waiting_allows_nothing_and_holds_off_the_next_action() -> None:
    key, clock, audit = _key()

    assert key.press() == {"confirmed": None, "reason": "nothing is waiting for the confirm key"}
    clock.now += STRAY_HOLD_S - 1
    assert key.arm("hash-a", "stop process 42") is None
    assert [e["outcome"] for e in _key_events(audit)] == ["nothing waiting", "held off"]


def test_the_hold_ends_after_its_time() -> None:
    key, clock, _ = _key()
    key.press()
    clock.now += STRAY_HOLD_S

    assert key.arm("hash-a", "stop process 42") is not None


def test_a_press_before_the_owner_could_have_heard_the_question_spoils_it() -> None:
    key, clock, audit = _key()
    waiting = key.arm("hash-a", "stop process 42")
    assert waiting is not None
    clock.now += MIN_PRESS_S / 2

    reply = key.press()
    clock.now += 2
    key.press()  # a second, later press cannot rescue it

    assert reply["confirmed"] is None
    assert key.disarm(waiting) is False
    assert _key_events(audit)[0]["outcome"] == "too soon"


def test_a_press_after_the_action_stopped_waiting_allows_nothing() -> None:
    key, clock, _ = _key()
    waiting = key.arm("hash-a", "stop process 42")
    assert waiting is not None
    key.disarm(waiting)
    clock.now += 2

    assert key.press()["confirmed"] is None
    second = key.arm("hash-b", "stop process 43")
    assert second is None  # that late press holds off the next one too


def test_only_one_action_waits_at_a_time() -> None:
    key, _, _ = _key()
    first = key.arm("hash-a", "stop process 42")

    assert first is not None
    assert key.arm("hash-b", "stop process 43") is None


def test_a_press_allows_only_the_action_it_was_made_for() -> None:
    key, clock, _ = _key()
    first = key.arm("hash-a", "stop process 42")
    assert first is not None
    clock.now += 2
    key.press()
    assert key.disarm(first) is True
    clock.now += 1

    second = key.arm("hash-b", "stop process 43")

    assert second is not None
    assert not second.pressed.is_set()  # the earlier press does not carry over
    clock.now += 30
    assert key.disarm(second) is False


def test_disarming_an_old_wait_leaves_a_newer_one_alone() -> None:
    key, clock, _ = _key()
    first = key.arm("hash-a", "stop process 42")
    assert first is not None
    key.disarm(first)
    clock.now += STRAY_HOLD_S
    second = key.arm("hash-b", "stop process 43")
    assert second is not None

    key.disarm(first)
    clock.now += 2

    assert key.press() == {"confirmed": "stop process 43"}
