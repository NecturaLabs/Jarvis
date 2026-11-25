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
    log = json.loads((folder / f"{kept.id}.json").read_text())["log"]
    assert [e["event"] for e in log] == ["request", "tool_call", "reply"]


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
    assert len(list((tmp_path / "a").iterdir())) == 4  # two files each


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
