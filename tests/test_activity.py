"""What the chat window shows of the audit log: new events as they are written, one line each."""

import json
from typing import TYPE_CHECKING

import pytest

from jarvis.activity import AuditTail, describe
from jarvis.audit import AuditLog

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, *entries: dict[str, object]) -> None:
    with path.open("a", encoding="ascii") as file:
        file.writelines(json.dumps(entry) + "\n" for entry in entries)


def test_the_tail_starts_at_the_end_and_reads_only_new_events(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _write(path, {"event": "old"})
    tail = AuditTail(path)

    assert tail.read() == []
    _write(path, {"event": "new"})
    assert tail.read() == [{"event": "new"}]
    assert tail.read() == []


def test_the_tail_waits_for_a_file_that_does_not_exist_yet(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    tail = AuditTail(path)

    assert tail.read() == []
    _write(path, {"event": "first"})
    assert tail.read() == [{"event": "first"}]


def test_a_half_written_line_waits_for_its_end(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("")
    tail = AuditTail(path)
    with path.open("a") as file:
        file.write('{"event": "re')

    assert tail.read() == []
    with path.open("a") as file:
        file.write('ply"}\n')
    assert tail.read() == [{"event": "reply"}]


def test_lines_that_are_not_event_objects_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("")
    tail = AuditTail(path)
    with path.open("ab") as file:
        file.write(b'not json\n[1, 2]\n\xff\xfe\n{"event": "kept"}\n')

    assert tail.read() == [{"event": "kept"}]


def test_nothing_is_lost_when_the_log_is_rotated_between_reads(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, max_bytes=300)
    log.record("start")
    tail = AuditTail(path)
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


def test_a_truncated_log_is_read_again_from_its_start(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _write(path, {"event": "a"}, {"event": "b"})
    tail = AuditTail(path)
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
        ({"event": "tool_call", "op": "spotify_pause", "verdict": "allow"}, "step", "pause"),
        ({"event": "tool_call", "op": "spotify_queue", "verdict": "confirm"}, "step", "yes"),
        ({"event": "tool_call", "op": "x", "verdict": "deny", "reason": "off"}, "warn", "off"),
        ({"event": "confirmation", "approved": True}, "step", "yes"),
        ({"event": "confirmation", "approved": False}, "warn", "no"),
        ({"event": "tool_failed", "op": "spotify_play", "error": "no device"}, "warn", "device"),
        ({"event": "tool_uncertain", "op": "spotify_queue"}, "warn", "not sure"),
        ({"event": "model_error", "error": "refused"}, "error", "refused"),
        ({"event": "model_call", "seconds": 3.2}, "detail", "3.2"),
        ({"event": "shortcut", "op": "spotify_pause"}, "step", "without the model"),
    ],
)
def test_each_event_reads_as_one_line(entry: dict[str, object], kind: str, words: str) -> None:
    shown = describe({"ts": "2026-11-21T10:00:00+00:00", **entry})

    assert shown is not None
    assert shown.kind == kind
    assert words in shown.text


def test_an_unknown_event_still_shows_its_name() -> None:
    assert "brand_new" in _line({"event": "brand_new"})


def test_outside_text_cannot_draw_on_the_terminal() -> None:
    text = _line({"event": "reply", "text": "Glass\x1b[2J Harbor\x07\nNight [b]Circuit[/b]"})

    assert "\x1b" not in text
    assert "\x07" not in text
    assert "\n" not in text
    assert "[b]Circuit[/b]" in text


def test_a_long_text_is_cut() -> None:
    assert len(_line({"event": "reply", "text": "la " * 2000})) < 700


def test_the_time_is_shown_in_local_time() -> None:
    shown = describe({"ts": "not a time", "event": "reply", "text": "hi"})

    assert shown is not None
    assert shown.time == ""


def test_an_entry_without_an_event_name_is_skipped() -> None:
    assert describe({"ts": "2026-11-21T10:00:00+00:00"}) is None
