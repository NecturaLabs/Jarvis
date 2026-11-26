"""The chat's archive: each conversation kept, listed, resumed with its provenance, capped."""

import json
import stat
from typing import TYPE_CHECKING

from jarvis.chat.archive import ChatArchive
from jarvis.memory import Conversation, Turn

if TYPE_CHECKING:
    from pathlib import Path


def _talk(archive: ChatArchive, request: str, *, outside: str = "") -> Conversation:
    memory = Conversation(budget=1000)
    archive.record({"event": "request", "text": request, "ts": "2026-11-25T21:44:03+00:00"})
    archive.record({"event": "tool_call", "op": "web_search", "args": {"query": request}})
    archive.record({"event": "reply", "text": "An answer.", "seconds": 1.2})
    memory.add(Turn(request, "An answer.", "web_search -> ok", outside, 1.0))
    return memory


def test_a_conversation_is_kept_whole_and_private(tmp_path: Path) -> None:
    folder = tmp_path / "chat-archive"
    archive = ChatArchive(folder, keep=10)
    archive.save(_talk(archive, "who wrote this song"))

    (kept,) = archive.listed()
    assert (kept.name, kept.requests, kept.named) == ("who wrote this song", 1, False)
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    for path in folder.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    lines = (folder / f"{kept.id}.log.jsonl").read_text().splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["request", "tool_call", "reply"]


def test_nothing_is_kept_for_an_empty_conversation(tmp_path: Path) -> None:
    archive = ChatArchive(tmp_path / "a", keep=10)
    archive.save(Conversation(budget=1000))
    assert archive.listed() == []


def test_a_resumed_conversation_keeps_its_provenance_and_name(tmp_path: Path) -> None:
    archive = ChatArchive(tmp_path / "a", keep=10)
    archive.save(_talk(archive, "read that page", outside="web"))
    archive.rename("Research")
    archive.save(_talk(archive, "and the next one", outside="web"))
    first = archive.current
    archive.start()
    archive.save(_talk(archive, "something else"))

    loaded = Conversation(budget=1000)
    assert archive.resume(first, loaded.load)

    assert loaded.holds_outside_text  # outside text from then is still data
    assert archive.current == first
    assert archive.name == "Research"
    archive.save(loaded)
    names = [a.name for a in archive.listed()]
    assert names.count("Research") == 1  # still one conversation, not a copy


def test_only_the_newest_conversations_are_kept(tmp_path: Path) -> None:
    archive = ChatArchive(tmp_path / "a", keep=2)
    for number in range(4):
        archive.save(_talk(archive, f"request {number}"))
        archive.start()

    assert [a.name for a in archive.listed()] == ["request 3", "request 2"]
    assert len(list((tmp_path / "a").iterdir())) == 6  # three files each


def test_forget_deletes_every_kept_conversation(tmp_path: Path) -> None:
    archive = ChatArchive(tmp_path / "a", keep=10)
    for number in range(3):
        archive.save(_talk(archive, f"request {number}"))
        archive.start()

    assert archive.forget() == 3
    assert archive.listed() == []
    assert not list((tmp_path / "a").iterdir())


def test_a_conversation_that_cannot_be_read_is_not_resumed(tmp_path: Path) -> None:
    archive = ChatArchive(tmp_path / "a", keep=10)
    archive.save(_talk(archive, "hello"))
    kept = archive.listed()[0].id
    archive.start()
    (tmp_path / "a" / f"{kept}.memory.json").unlink()

    assert not archive.resume(kept, Conversation(budget=1000).load)
    assert archive.current != kept


def test_a_conversation_opened_and_closed_with_nothing_new_is_not_copied(tmp_path: Path) -> None:
    folder = tmp_path / "a"
    archive = ChatArchive(folder, keep=10)
    memory = _talk(archive, "who wrote this song")
    archive.save(memory)

    for _ in range(3):  # the chat opens with the conversation it remembers, and closes again
        reopened = ChatArchive(folder, keep=10)
        reopened.save(memory)
        reopened.save(memory)

    assert [a.name for a in ChatArchive(folder, keep=10).listed()] == ["who wrote this song"]


def test_a_resumed_conversation_is_written_again_only_when_it_changes(tmp_path: Path) -> None:
    archive = ChatArchive(tmp_path / "a", keep=10)
    archive.save(_talk(archive, "hello"))
    first = archive.current
    archive.start()
    loaded = Conversation(budget=1000)
    assert archive.resume(first, loaded.load)
    (before,) = archive.listed()

    archive.save(loaded)

    assert archive.listed() == [before]
    archive.rename("Greetings")
    archive.save(loaded)
    assert [a.name for a in archive.listed()] == ["Greetings"]


def test_the_list_and_the_cap_read_no_log(tmp_path: Path) -> None:
    folder = tmp_path / "a"
    archive = ChatArchive(folder, keep=10)
    memory = _talk(archive, "read the whole site")
    for _ in range(200):
        archive.record({"event": "tool_result", "text": "x" * 2000})
    archive.save(memory)
    kept = archive.current

    meta = folder / f"{kept}.json"
    assert meta.stat().st_size < 1000  # the log is not in the file the list reads
    (folder / f"{kept}.log.jsonl").write_text("[" * 100_000)  # a log that cannot be read
    (listed,) = archive.listed()
    assert (listed.name, listed.requests) == ("read the whole site", 1)


def test_the_log_is_added_to_not_rewritten(tmp_path: Path) -> None:
    folder = tmp_path / "a"
    archive = ChatArchive(folder, keep=10)
    memory = _talk(archive, "first")
    archive.save(memory)
    archive.record({"event": "request", "text": "second"})
    archive.save(memory)

    lines = (folder / f"{archive.current}.log.jsonl").read_text().splitlines()
    assert [json.loads(line)["event"] for line in lines] == [
        "request",
        "tool_call",
        "reply",
        "request",
    ]
    assert stat.S_IMODE((folder / f"{archive.current}.log.jsonl").stat().st_mode) == 0o600
    assert archive.listed()[0].requests == 2


def test_a_conversation_kept_in_the_first_format_is_listed_resumed_and_carried_on(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "a"
    folder.mkdir(mode=0o700)
    old = "20261125-214403-abcdef"
    entries = [{"event": "request", "text": "hello"}, {"event": "reply", "text": "Hi."}]
    meta = {
        "version": 1,
        "id": old,
        "name": "hello",
        "named": False,
        "created": "2026-11-25T21:44:03+00:00",
        "updated": "2026-11-25T21:44:09+00:00",
        "log": entries,
    }
    for path, text in (
        (folder / f"{old}.json", json.dumps(meta)),
        (folder / f"{old}.memory.json", ""),
    ):
        path.write_text(text)
        path.chmod(0o600)
    memory = Conversation(budget=1000)
    memory.add(Turn("hello", "Hi.", "", "", 1.0))
    memory.save(folder / f"{old}.memory.json")
    archive = ChatArchive(folder, keep=10)

    (listed,) = archive.listed()
    assert (listed.id, listed.name, listed.requests) == (old, "hello", 1)
    assert archive.resume(old, Conversation(budget=1000).load)
    archive.record({"event": "request", "text": "again"})
    archive.save(memory)

    lines = (folder / f"{old}.log.jsonl").read_text().splitlines()
    assert [json.loads(line)["text"] for line in lines] == ["hello", "Hi.", "again"]
    assert archive.listed()[0].requests == 2


def test_a_file_nested_too_deep_to_read_is_skipped(tmp_path: Path) -> None:
    folder = tmp_path / "a"
    archive = ChatArchive(folder, keep=10)
    archive.save(_talk(archive, "hello"))
    kept = archive.current
    archive.start()
    deep = folder / "20261125-000000-000000.json"
    deep.write_text("[" * 100_000)
    deep.chmod(0o600)
    (folder / f"{kept}.memory.json").write_text("[" * 100_000)

    assert [a.id for a in archive.listed()] == [kept]
    assert not archive.resume(kept, Conversation(budget=1000).load)
