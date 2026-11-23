"""The audit log: private, one ASCII JSON line per event, bounded, never why a turn fails."""

import json
import logging
import pathlib
import stat
import threading
from typing import TYPE_CHECKING

from jarvis.audit import AuditLog

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


def test_each_event_is_one_json_line_with_a_timestamp(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "state" / "audit.jsonl")

    log.record("request", text="pause", origin="owner_speech")
    log.record("reply", text="Paused it.")

    first, second = _lines(tmp_path / "state" / "audit.jsonl")
    assert (first["event"], first["text"], first["origin"]) == ("request", "pause", "owner_speech")
    assert second["event"] == "reply"
    assert isinstance(first["ts"], str)


def test_the_log_and_its_folder_are_private(tmp_path: Path) -> None:
    path = tmp_path / "state" / "audit.jsonl"

    AuditLog(path).record("request", text="pause")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_an_existing_readable_log_is_made_private(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("", encoding="ascii")
    path.chmod(0o644)

    AuditLog(path).record("request", text="pause")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_text_that_cannot_be_encoded_is_still_recorded(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"

    AuditLog(path).record("tool_result", text="half an emoji \ud800", args={"query": "\udfff"})

    (entry,) = _lines(path)
    assert entry["text"] == "half an emoji \ud800"


def test_line_separators_in_outside_text_do_not_split_a_record(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"

    AuditLog(path).record("tool_result", text="one\u2028two\u2029three\u0085four")

    assert len(path.read_text(encoding="ascii").splitlines()) == 1


def test_a_log_that_cannot_be_written_does_not_fail_the_caller(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocked = tmp_path / "audit.jsonl"
    blocked.mkdir()

    with caplog.at_level(logging.ERROR, logger="jarvis.audit"):
        AuditLog(blocked).record("tool_call", op="spotify_pause")

    assert "spotify_pause" in caplog.text


def test_a_value_json_cannot_hold_is_recorded_as_text(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"

    AuditLog(path).record("tool_call", args={1: {2, 3}})

    (entry,) = _lines(path)
    assert entry["event"] == "tool_call"


def test_a_full_log_is_rotated_once_and_keeps_writing(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, max_bytes=400, generations=1)

    for number in range(20):
        log.record("request", text=f"request number {number}")

    older = path.with_name("audit.jsonl.1")
    assert older.exists()
    assert stat.S_IMODE(older.stat().st_mode) == 0o600
    assert path.stat().st_size <= 400
    assert _lines(path)[-1]["text"] == "request number 19"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audit.jsonl", "audit.jsonl.1"]


def test_without_a_path_events_stay_in_memory() -> None:
    log = AuditLog(None)

    log.record("request", text="pause")

    assert [e["event"] for e in log.events] == ["request"]


def test_tags_go_on_every_event_and_cannot_be_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, tags={"via": "chat"})

    log.record("request", text="pause")
    log.record("reply", text="Paused.", via="voice")

    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert [entry["via"] for entry in entries] == ["chat", "chat"]


def test_a_full_log_keeps_five_older_generations(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, max_bytes=200)

    for number in range(60):
        log.record("request", text=f"request number {number:02}")

    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["audit.jsonl", *(f"audit.jsonl.{n}" for n in range(1, 6))]
    numbers = [
        int(str(entry["text"]).rsplit(" ", 1)[1])
        for generation in (5, 4, 3, 2, 1)
        for entry in _lines(path.with_name(f"audit.jsonl.{generation}"))
    ] + [int(str(entry["text"]).rsplit(" ", 1)[1]) for entry in _lines(path)]
    assert numbers == list(range(numbers[0], 60))  # oldest first, nothing lost between them
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in tmp_path.iterdir())


def test_an_event_written_by_another_process_during_rotation_does_not_lose_this_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "audit.jsonl"
    rotating = AuditLog(path, max_bytes=300, generations=3)
    other = AuditLog(path, max_bytes=300, generations=3)  # another Jarvis process, own lock
    rotating.record("request", text="x" * 150)
    replace = pathlib.Path.replace

    def replace_then_other_writes(source: Path, target: Path) -> Path:
        moved = replace(source, target)
        if source == path:  # the full log moved aside, before the new one is opened
            other.record("reply", text="from the other process")
        return moved

    monkeypatch.setattr(pathlib.Path, "replace", replace_then_other_writes)

    rotating.record("tool_call", text="the call that rotated")

    assert [entry["event"] for entry in _lines(path.with_name("audit.jsonl.1"))] == ["request"]
    assert sorted(str(entry["event"]) for entry in _lines(path)) == ["reply", "tool_call"]


def test_writers_in_several_processes_lose_nothing_and_rotate_each_full_log_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.jsonl"
    writers = [AuditLog(path, max_bytes=2000, generations=40) for _ in range(4)]

    def write(number: int) -> None:
        for count in range(150):
            writers[number].record("request", text=f"{number}-{count:03}")

    threads = [threading.Thread(target=write, args=(number,)) for number in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    files = [path, *(path.with_name(f"audit.jsonl.{n}") for n in range(1, 41))]
    texts = [str(entry["text"]) for file in files if file.exists() for entry in _lines(file)]
    assert sorted(texts) == sorted(f"{n}-{c:03}" for n in range(4) for c in range(150))
    assert all(file.stat().st_size <= 2000 for file in files if file.exists())
