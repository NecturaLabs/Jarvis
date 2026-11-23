"""What the chat window shows of the audit log: new events as they are written, one line each."""

import json
from typing import TYPE_CHECKING

import pytest

from jarvis.activity import NO_REASON_WORDS, AuditTail, describe, working_label
from jarvis.agent import NoReason
from jarvis.audit import AuditLog

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


@pytest.fixture
def audit_tail() -> Iterator[Callable[[Path], AuditTail]]:
    """Make tails that are closed when the test ends, so none leaves its file open."""
    made: list[AuditTail] = []

    def make(path: Path) -> AuditTail:
        tail = AuditTail(path)
        made.append(tail)
        return tail

    yield make
    for tail in made:
        tail.close()


def _write(path: Path, *entries: dict[str, object]) -> None:
    with path.open("a", encoding="ascii") as file:
        file.writelines(json.dumps(entry) + "\n" for entry in entries)


def test_the_tail_starts_at_the_end_and_reads_only_new_events(
    tmp_path: Path, audit_tail: Callable[[Path], AuditTail]
) -> None:
    path = tmp_path / "audit.jsonl"
    _write(path, {"event": "old"})
    tail = audit_tail(path)

    assert tail.read() == []
    _write(path, {"event": "new"})
    assert tail.read() == [{"event": "new"}]
    assert tail.read() == []


def test_the_tail_waits_for_a_file_that_does_not_exist_yet(
    tmp_path: Path, audit_tail: Callable[[Path], AuditTail]
) -> None:
    path = tmp_path / "audit.jsonl"
    tail = audit_tail(path)

    assert tail.read() == []
    _write(path, {"event": "first"})
    assert tail.read() == [{"event": "first"}]


def test_a_half_written_line_waits_for_its_end(
    tmp_path: Path, audit_tail: Callable[[Path], AuditTail]
) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("")
    tail = audit_tail(path)
    with path.open("a") as file:
        file.write('{"event": "re')

    assert tail.read() == []
    with path.open("a") as file:
        file.write('ply"}\n')
    assert tail.read() == [{"event": "reply"}]


def test_lines_that_are_not_event_objects_are_skipped(
    tmp_path: Path, audit_tail: Callable[[Path], AuditTail]
) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("")
    tail = audit_tail(path)
    with path.open("ab") as file:
        file.write(b'not json\n[1, 2]\n\xff\xfe\n{"event": "kept"}\n')

    assert tail.read() == [{"event": "kept"}]


def test_nothing_is_lost_when_the_log_is_rotated_between_reads(
    tmp_path: Path, audit_tail: Callable[[Path], AuditTail]
) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, max_bytes=300)
    log.record("start")
    tail = audit_tail(path)
    rotated = path.with_name(path.name + ".1")
    written: list[int] = []
    while not rotated.exists():
        written.append(len(written))
        log.record("step", n=written[-1])
    written += [len(written), len(written) + 1]
    for number in written[-2:]:
        log.record("step", n=number)

    seen = [entry["n"] for entry in tail.read()]

    assert seen == written


def test_a_truncated_log_is_read_again_from_its_start(
    tmp_path: Path, audit_tail: Callable[[Path], AuditTail]
) -> None:
    path = tmp_path / "audit.jsonl"
    _write(path, {"event": "a"}, {"event": "b"})
    tail = audit_tail(path)
    path.write_text("")
    _write(path, {"event": "c"})

    assert tail.read() == [{"event": "c"}]


def _line(entry: dict[str, object]) -> str:
    shown = describe({"ts": "2026-11-21T10:00:00+00:00", **entry})
    assert shown is not None
    return shown.text


@pytest.mark.parametrize(
    ("entry", "kind", "words"),
    [
        ({"event": "request", "text": "pause", "origin": "owner_speech"}, "you", "pause"),
        ({"event": "request", "text": "pause", "origin": "pc_audio"}, "pc", "PC audio"),
        ({"event": "request", "text": "pause", "via": "chat"}, "you", "typed"),
        ({"event": "reply", "text": "Paused it.", "seconds": 0.3}, "jarvis", "Paused it."),
        ({"event": "tool_call", "op": "spotify_pause", "verdict": "allow"}, "step", "Pause"),
        ({"event": "tool_call", "op": "spotify_queue", "verdict": "confirm"}, "ask", "yes"),
        ({"event": "tool_call", "op": "x", "verdict": "deny", "reason": "off"}, "warn", "off"),
        ({"event": "confirmation", "approved": True}, "result", "yes"),
        ({"event": "confirmation", "approved": False}, "warn", "no"),
        ({"event": "confirmation", "approved": False, "reason": "slept"}, "warn", "PC slept"),
        ({"event": "confirmation", "approved": False, "reason": "no_answer"}, "warn", "in time"),
        ({"event": "confirmation", "approved": False, "reason": "closed"}, "warn", "closed"),
        ({"event": "tool_call", "op": "x", "verdict": "strong"}, "ask", "confirm key"),
        ({"event": "tool_call", "op": "x", "verdict": "strong", "refused": "no"}, "warn", "key"),
        ({"event": "confirmation", "approved": True, "method": "key"}, "result", "pressed"),
        ({"event": "confirmation", "approved": False, "method": "key"}, "warn", "no key press"),
        ({"event": "confirm_key", "outcome": "pressed"}, "detail", "pressed"),
        ({"event": "confirm_key", "outcome": "nothing waiting"}, "warn", "nothing was waiting"),
        ({"event": "confirm_key", "outcome": "too soon"}, "warn", "too soon"),
        ({"event": "confirm_key", "outcome": "held off"}, "warn", "held off"),
        ({"event": "tool_failed", "op": "spotify_play", "error": "no device"}, "warn", "device"),
        ({"event": "tool_uncertain", "op": "spotify_queue"}, "warn", "not sure"),
        ({"event": "model_error", "error": "refused"}, "error", "refused"),
        ({"event": "model_call", "seconds": 3.2}, "detail", "3.2"),
        ({"event": "shortcut", "op": "spotify_pause"}, "step", "without the model"),
    ],
)
def test_each_event_reads_as_one_entry(entry: dict[str, object], kind: str, words: str) -> None:
    shown = describe({"ts": "2026-11-21T10:00:00+00:00", **entry})

    assert shown is not None
    assert shown.kind == kind
    assert words in f"{shown.text} {shown.note}"


@pytest.mark.parametrize(
    ("event", "nested"),
    [
        ("request", False),
        ("tool_call", False),
        ("reply", False),
        ("shortcut", False),
        ("model_error", False),
        ("tool_result", True),
        ("confirmation", True),
        ("tool_failed", True),
        ("model_call", True),
    ],
)
def test_results_belong_to_the_step_above(event: str, nested: bool) -> None:
    shown = describe({"ts": "2026-11-21T10:00:00+00:00", "event": event, "op": "spotify_pause"})

    assert shown is not None
    assert shown.nested is nested


def test_only_a_request_opens_a_turn_and_only_details_hide() -> None:
    entries: dict[str, dict[str, object]] = {
        "request": {"event": "request", "text": "pause"},
        "pc": {"event": "request", "text": "pause", "origin": "pc_audio"},
        "reply": {"event": "reply", "text": "ok"},
        "model_call": {"event": "model_call", "seconds": 1},
    }
    shown = {name: describe(entry) for name, entry in entries.items()}

    assert {name for name, s in shown.items() if s and s.starts_turn} == {"request", "pc"}
    assert {name for name, s in shown.items() if s and s.detail} == {"model_call"}


def test_an_unknown_event_still_shows_its_name() -> None:
    assert "brand_new" in _line({"event": "brand_new"})


def test_outside_text_cannot_draw_on_the_terminal() -> None:
    text = _line({"event": "reply", "text": "Glass\x1b[2J Harbor\x07\r\nNight [b]Circuit[/b]"})

    assert "\x1b" not in text
    assert "\x07" not in text
    assert "\r" not in text
    assert "Glass [2J Harbor\nNight [b]Circuit[/b]" in text


def test_a_step_stays_one_line() -> None:
    text = _line({"event": "tool_failed", "op": "web_read", "error": "bad\npage"})

    assert "\n" not in text


def test_a_reply_is_kept_whole_but_a_step_is_cut() -> None:
    reply = "la " * 2000

    assert _line({"event": "reply", "text": reply}).startswith(reply.strip())
    assert len(_line({"event": "tool_failed", "op": "x", "error": "la " * 2000})) < 700


def test_a_call_shows_the_gist_of_its_arguments_quietly() -> None:
    shown = describe(
        {"event": "tool_call", "op": "web_search", "args": {"query": "moonveil"}, "verdict": "run"}
    )

    assert shown is not None
    assert (shown.text, shown.note) == ("Web search", '"moonveil"')


@pytest.mark.parametrize(
    ("entry", "level"),
    [
        ({"event": "request", "text": "hi"}, "replies"),
        ({"event": "reply", "text": "hi"}, "replies"),
        ({"event": "model_error", "error": "down"}, "replies"),
        ({"event": "tool_denied", "op": "x", "reason": "no"}, "replies"),
        ({"event": "memory_summarized", "turns": 3}, "replies"),
        ({"event": "tool_call", "op": "web_search", "verdict": "run"}, "steps"),
        ({"event": "tool_result", "op": "web_search", "ok": True, "text": "r"}, "steps"),
        ({"event": "tool_failed", "op": "web_read", "error": "x"}, "steps"),
        ({"event": "model_call", "seconds": 1}, "everything"),
    ],
)
def test_each_event_shows_from_its_verbosity_level(entry: dict[str, object], level: str) -> None:
    shown = describe(entry)

    assert shown is not None
    assert shown.level == level


@pytest.mark.parametrize(
    ("entry", "label"),
    [
        ({"event": "tool_call", "op": "web_search"}, "Searching the web"),
        ({"event": "tool_call", "op": "spotify_play"}, "Talking to Spotify"),
        ({"event": "tool_call", "op": "brand_new_op"}, "Running Brand new op"),
        ({"event": "tool_call", "op": "time_now"}, "Checking the time"),
        ({"event": "model_call"}, "Thinking"),
        ({"event": "request", "text": "hi"}, None),
    ],
)
def test_the_working_line_follows_what_jarvis_does(
    entry: dict[str, object], label: str | None
) -> None:
    assert working_label(entry) == label


def test_the_time_is_shown_in_local_time() -> None:
    shown = describe({"ts": "not a time", "event": "reply", "text": "hi"})

    assert shown is not None
    assert not shown.time


def test_an_entry_without_an_event_name_is_skipped() -> None:
    assert describe({"ts": "2026-11-21T10:00:00+00:00"}) is None


@pytest.mark.parametrize(
    ("command", "text"),
    [
        ("pause", "Paused: the microphone is let go (jarvis pause)"),
        ("resume", "Listening again (jarvis resume)"),
        ("hush", "Hushed (jarvis hush)"),
    ],
)
def test_a_control_command_shows_as_a_note(command: str, text: str) -> None:
    shown = describe({"ts": "2026-11-21T10:00:00+00:00", "event": "control", "command": command})

    assert shown is not None
    assert (shown.kind, shown.text) == ("note", text)


def test_a_confirm_key_press_from_a_program_jarvis_started_is_a_warning() -> None:
    shown = describe(
        {
            "ts": "2026-11-22T10:00:00+00:00",
            "event": "control_refused",
            "command": "confirm",
            "reason": "started by Jarvis",
        }
    )

    assert shown is not None
    assert (shown.kind, shown.text) == (
        "warn",
        "refused the confirm key from a program Jarvis started",
    )


def test_a_refused_control_connection_is_a_warning() -> None:
    shown = describe({"ts": "2026-11-21T10:00:00+00:00", "event": "control_refused"})

    assert shown is not None
    assert shown.kind == "warn"


def test_a_wake_word_ignored_during_dictation_shows_as_a_note() -> None:
    shown = describe(
        {"ts": "2026-11-22T10:00:00+00:00", "event": "wake_ignored", "reason": "dictating"}
    )

    assert shown is not None
    assert (shown.kind, shown.text) == ("note", "Ignored the wake word: you were dictating")


def test_a_request_dropped_when_dictation_started_shows_as_a_note() -> None:
    shown = describe(
        {
            "ts": "2026-11-22T10:00:00+00:00",
            "event": "wake_ignored",
            "reason": "dictation started mid-request",
        }
    )

    assert shown is not None
    assert (shown.kind, shown.text) == ("note", "Dropped a request: you started dictating")


@pytest.mark.parametrize(
    ("entry", "text"),
    [
        ({"event": "consent_changed", "bypass_consent": True}, "Questions turned off"),
        ({"event": "consent_changed", "bypass_consent": False}, "Questions turned on"),
        ({"event": "service_started", "bypass_consent": True}, "acting without asking"),
    ],
)
def test_consent_changes_and_the_services_start_show_as_notes(
    entry: dict[str, object], text: str
) -> None:
    shown = describe({"ts": "2026-11-22T10:00:00+00:00", **entry})

    assert shown is not None
    assert shown.kind == "note"
    assert text in shown.text


def test_every_no_reason_has_words_in_the_window() -> None:
    asked = {NoReason.EARLIER_NO, NoReason.NO_KEY}  # shown as "not asked", or by the key lines
    assert set(NO_REASON_WORDS) == {reason.value for reason in NoReason} - asked
