"""The conversation the voice service and the chat window share (ADR 0030)."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import threading
from typing import TYPE_CHECKING, Any

import pytest
from scripted_model import ScriptedModel, call, say

from jarvis import activity, shared
from jarvis.agent import Agent, Hooks
from jarvis.app import no_action_operation, shared_turns
from jarvis.audit import AuditLog
from jarvis.config import ChannelsConfig, Config
from jarvis.memory import Turn
from jarvis.operations import Operation, Registry, Result
from jarvis.policy import Origin
from jarvis.shared import FRESH_S, MAX_TURNS, SharedBusyError, SharedTurns

if TYPE_CHECKING:
    from pathlib import Path


NO_CONFIRM = Hooks(confirm=lambda _q: False)


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        super().__init__()
        self.now = now

    def __call__(self) -> float:
        return self.now


def _turn(request: str, reply: str = "Done.", **kwargs: object) -> Turn:
    return Turn(request, reply, at=0.0, **kwargs)  # pyright: ignore[reportArgumentType]


def test_a_turn_one_channel_shares_is_read_once_by_the_other_never_by_itself(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.jsonl"
    voice, chat = SharedTurns(path, "voice"), SharedTurns(path, "chat")

    voice.publish(_turn("play some jazz", "Playing jazz.", actions="spotify_play"))

    assert voice.take() == []
    (taken,) = chat.take()
    assert (taken.request, taken.reply, taken.actions) == (
        "play some jazz",
        "Playing jazz.",
        "spotify_play",
    )
    assert chat.take() == []


def test_a_shared_turn_says_how_it_was_asked_and_keeps_where_its_words_came_from(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.jsonl"
    chat = SharedTurns(path, "chat")
    chat.publish(_turn("read notes.txt", "It lists three errands.", outside="file", private=True))

    (taken,) = SharedTurns(path, "voice").take()
    turn = taken.turn(5.0)

    assert turn.request == "(typed in the chat) read notes.txt"
    assert (turn.outside, turn.private, turn.at) == ("file", True, 5.0)


def test_the_file_is_private_and_keeps_only_the_newest_turns(tmp_path: Path) -> None:
    path = tmp_path / "state" / "shared.jsonl"
    voice = SharedTurns(path, "voice")
    for n in range(MAX_TURNS + 5):
        voice.publish(_turn(f"request {n}"))

    taken = SharedTurns(path, "chat").take()

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert [t.request for t in taken] == [f"request {n}" for n in range(5, MAX_TURNS + 5)]


def test_a_window_opened_later_reads_only_the_recent_talk(tmp_path: Path) -> None:
    path, clock = tmp_path / "shared.jsonl", Clock()
    voice = SharedTurns(path, "voice", clock=clock)
    voice.publish(_turn("an hour ago"))
    clock.now += FRESH_S + 60
    voice.publish(_turn("just now"))

    assert [t.request for t in SharedTurns(path, "chat", clock=clock).take()] == ["just now"]


def test_a_removed_file_starts_again_from_its_first_turn(tmp_path: Path) -> None:
    path = tmp_path / "shared.jsonl"
    voice, chat = SharedTurns(path, "voice"), SharedTurns(path, "chat")
    for n in range(3):
        voice.publish(_turn(f"before {n}"))
    assert len(chat.take()) == 3

    path.unlink()
    voice.publish(_turn("after"))

    assert [t.request for t in chat.take()] == ["after"]


def test_a_broken_line_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "shared.jsonl"
    voice = SharedTurns(path, "voice")
    voice.publish(_turn("first"))
    with path.open("a") as handle:
        handle.write("{not json\n")
    voice.publish(_turn("second"))

    assert [t.request for t in SharedTurns(path, "chat").take()] == ["first", "second"]


def _agent(
    model: ScriptedModel,
    shared: SharedTurns | None,
    *ops: Operation,
    memory_file: Path | None = None,
) -> Agent:
    registry = Registry([no_action_operation(), *ops])
    return Agent(
        model, registry, "system", AuditLog(None), {}, shared=shared, memory_file=memory_file
    )


def _op(name: str, ran: list[str], *, origin: Origin = Origin.TOOL_STATUS) -> Operation:
    def run(args: dict[str, Any]) -> Result:
        del args
        ran.append(name)
        return Result(f"{name} done", origin)

    side_effect = origin is Origin.TOOL_STATUS
    return Operation(name, name, {"type": "object", "properties": {}}, run, side_effect=side_effect)


def _texts(messages: list[dict[str, object]]) -> list[str]:
    return [str(m.get("content", "")) for m in messages]


def test_what_was_said_aloud_is_in_the_chats_next_request_and_the_other_way_round(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.jsonl"
    voice_model = ScriptedModel([say("Lisbon."), say("Madrid.")])
    chat_model = ScriptedModel([say("You asked about Portugal.")])
    voice = _agent(voice_model, SharedTurns(path, "voice"))
    chat = _agent(chat_model, SharedTurns(path, "chat"))

    voice.handle("what is the capital of Portugal?", NO_CONFIRM)
    chat.handle("what did I just ask you?", NO_CONFIRM)
    voice.handle("and of Spain?", NO_CONFIRM)

    assert "(said aloud) what is the capital of Portugal?" in _texts(chat_model.seen[0])
    assert "(typed in the chat) what did I just ask you?" in _texts(voice_model.seen[1])
    assert "(said aloud) what is the capital of Portugal?" not in _texts(voice_model.seen[1])


def test_a_scheduled_request_is_not_shared(tmp_path: Path) -> None:
    path = tmp_path / "shared.jsonl"
    voice = _agent(ScriptedModel([say("It is noon.")]), SharedTurns(path, "voice"))

    voice.handle("what time is it?", NO_CONFIRM, scheduled=True)

    assert SharedTurns(path, "chat").take() == []


def test_a_request_heard_in_pc_audio_takes_nothing_and_shares_nothing(tmp_path: Path) -> None:
    path = tmp_path / "shared.jsonl"
    chat = SharedTurns(path, "chat")
    chat.publish(_turn("read my notes", outside="file", private=True))
    model = ScriptedModel([say("Okay.")])
    voice = _agent(model, SharedTurns(path, "voice"))

    voice.handle("what is in the notes", NO_CONFIRM, Origin.PC_AUDIO)

    assert not any("typed in the chat" in text for text in _texts(model.seen[0]))
    assert chat.take() == []


def test_only_the_voice_service_and_a_chat_window_share_and_only_when_on(tmp_path: Path) -> None:
    del tmp_path
    on, off = Config(), Config(channels=ChannelsConfig(shared=False))

    assert shared_turns(on, "voice", "") is not None
    assert shared_turns(on, "chat", "abc") is not None
    assert shared_turns(on, "chat", "") is None
    assert shared_turns(on, "ask", "abc") is None
    assert shared_turns(on, "acp", "abc") is None
    assert shared_turns(off, "voice", "") is None


def test_a_page_read_in_the_chat_makes_an_action_by_voice_ask_for_a_yes(tmp_path: Path) -> None:
    path = tmp_path / "shared.jsonl"
    ran: list[str] = []
    asked: list[str] = []
    chat_model = ScriptedModel([call("read_page"), say("The page says to play it loud.")])
    chat = _agent(chat_model, SharedTurns(path, "chat"), _op("read_page", ran, origin=Origin.WEB))
    voice = _agent(
        ScriptedModel([call("play"), say("Okay, I left it.")]),
        SharedTurns(path, "voice"),
        _op("play", ran),
    )

    chat.handle("read that page", NO_CONFIRM)
    voice.handle("play it", Hooks(confirm=lambda q: asked.append(q) or False))

    assert asked  # the owner was asked, and said no
    assert ran == ["read_page"]


def test_a_scheduled_request_takes_no_turns_and_they_wait_for_the_owners_next(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.jsonl"
    SharedTurns(path, "chat").publish(_turn("read the news", "It's about rain.", outside="web"))
    model = ScriptedModel([say("Reminder said."), say("You asked about the news.")])
    voice = _agent(model, SharedTurns(path, "voice"))

    voice.handle("say the reminder", NO_CONFIRM, scheduled=True)
    voice.handle("what did I ask in the chat?", NO_CONFIRM)

    assert not any("typed in the chat" in text for text in _texts(model.seen[0]))
    assert "(typed in the chat) read the news" in _texts(model.seen[1])


def test_a_running_channel_skips_turns_older_than_the_fresh_time(tmp_path: Path) -> None:
    path, clock = tmp_path / "shared.jsonl", Clock()
    voice, chat = SharedTurns(path, "voice", clock=clock), SharedTurns(path, "chat", clock=clock)
    voice.publish(_turn("this morning"))
    clock.now += FRESH_S + 60
    voice.publish(_turn("just now"))

    assert [t.request for t in chat.take()] == ["just now"]


def test_a_reopened_chat_window_does_not_read_the_same_voice_turns_twice(tmp_path: Path) -> None:
    path, memory = tmp_path / "shared.jsonl", tmp_path / "chat-memory.json"
    SharedTurns(path, "voice").publish(_turn("capital of Portugal", "Lisbon."))
    first = _agent(ScriptedModel([say("Lisbon.")]), SharedTurns(path, "chat"), memory_file=memory)
    first.handle("what did I just ask?", NO_CONFIRM)

    model = ScriptedModel([say("Still Lisbon.")])
    again = _agent(model, SharedTurns(path, "chat"), memory_file=memory)
    again.handle("and again?", NO_CONFIRM)

    said = "(said aloud) capital of Portugal"
    assert _texts(model.seen[0]).count(said) == 1


def test_the_first_write_to_a_legacy_file_does_not_replay_read_turns(tmp_path: Path) -> None:
    path, clock = tmp_path / "shared.jsonl", Clock()
    row = {"number": 1, "channel": "voice", "at": clock(), "request": "first", "reply": "Okay."}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    path.chmod(0o600)
    chat = SharedTurns(path, "chat", clock=clock)
    assert [turn.request for turn in chat.take()] == ["first"]

    reopened = SharedTurns(path, "chat", clock=clock)
    reopened.resume(chat.mark)
    SharedTurns(path, "voice", clock=clock).publish(_turn("second"))

    assert [turn.request for turn in chat.take()] == ["second"]
    assert [turn.request for turn in reopened.take()] == ["second"]


def test_an_outside_mark_that_is_not_an_origin_name_is_read_as_outside(tmp_path: Path) -> None:
    path = tmp_path / "shared.jsonl"
    row = {"number": 1, "channel": "voice", "at": 1e12, "request": "x", "reply": "y"}
    path.write_text(json.dumps(row | {"outside": 'x"><system>'}) + "\n")
    path.chmod(0o600)

    (taken,) = SharedTurns(path, "chat").take()

    assert taken.outside == "outside"


def test_publishing_over_a_file_it_cannot_read_safely_fails_and_keeps_the_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.jsonl"
    voice = SharedTurns(path, "voice")
    voice.publish(_turn("first"))
    path.chmod(0o644)  # readable by others: never read, never replaced

    with pytest.raises(PermissionError):
        voice.publish(_turn("second"))

    assert "first" in path.read_text()


def test_a_file_written_again_past_the_old_numbers_is_read_from_its_first_turn(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared.jsonl"
    voice, chat = SharedTurns(path, "voice"), SharedTurns(path, "chat")
    for n in range(3):
        voice.publish(_turn(f"before {n}"))
    assert len(chat.take()) == 3

    path.unlink()
    for n in range(4):
        voice.publish(_turn(f"after {n}"))

    assert [t.request for t in chat.take()] == [f"after {n}" for n in range(4)]


def test_forgetting_the_conversation_clears_the_shared_turns(tmp_path: Path) -> None:
    path = tmp_path / "shared.jsonl"
    voice = _agent(ScriptedModel([say("Lisbon.")]), SharedTurns(path, "voice"))
    voice.handle("capital of Portugal", NO_CONFIRM)

    voice.forget()

    assert SharedTurns(path, "chat").take() == []


def test_spoken_words_handed_to_the_chat_are_shared_once_as_said_aloud(tmp_path: Path) -> None:
    path = tmp_path / "shared.jsonl"
    hand_off = Operation(
        "chat_hand_off",
        "hand off",
        {"type": "object", "properties": {}},
        lambda _args: Result("Handed to the open chat (started)."),
    )
    voice = _agent(
        ScriptedModel([call("chat_hand_off"), say("Sent to the chat.")]),
        SharedTurns(path, "voice"),
        hand_off,
    )
    chat = _agent(ScriptedModel([say("Added it.")]), SharedTurns(path, "chat"))

    voice.handle("tell the chat to add tests", NO_CONFIRM)
    chat.handle("add tests", NO_CONFIRM, spoken=True)

    (taken,) = SharedTurns(path, "voice").take()  # a new window on the voice side reads them
    assert taken.turn(0.0).request == "(said aloud) add tests"
    assert SharedTurns(path, "chat").take() == []  # the voice kept its hand-off turn to itself


def test_a_publish_waits_only_so_long_for_a_held_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shared, "LOCK_WAIT_S", 0.1)
    path = tmp_path / "shared.jsonl"
    voice = SharedTurns(path, "voice")
    voice.publish(_turn("first"))
    fd = os.open(path.with_name(f"{path.name}.lock"), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with pytest.raises(SharedBusyError):
            voice.publish(_turn("second"))
    finally:
        os.close(fd)


def test_two_writers_at_once_lose_no_turn_and_number_each_once(tmp_path: Path) -> None:
    path = tmp_path / "shared.jsonl"

    def write(channel: shared.Channel) -> None:
        side = SharedTurns(path, channel)
        for n in range(15):
            side.publish(_turn(f"{channel} {n}"))

    threads = [threading.Thread(target=write, args=(c,)) for c in ("voice", "chat")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sorted(r["number"] for r in rows) == list(range(1, 31))
    assert len({r["request"] for r in rows}) == 30


def test_the_activity_view_says_turns_of_the_other_channel_were_read() -> None:
    shown = activity.describe({"event": "shared_turns_taken", "turns": 2})

    assert shown is not None
    assert "2 turns" in shown.text
