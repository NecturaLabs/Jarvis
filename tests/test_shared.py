"""The conversation the voice service and the chat window share (ADR 0030)."""

from __future__ import annotations

import stat
from typing import TYPE_CHECKING

from scripted_model import ScriptedModel, say

from jarvis.agent import Agent, Hooks
from jarvis.app import no_action_operation, shared_turns
from jarvis.audit import AuditLog
from jarvis.config import ChannelsConfig, Config
from jarvis.memory import Turn
from jarvis.operations import Registry
from jarvis.policy import Origin
from jarvis.shared import FRESH_S, MAX_TURNS, SharedTurns

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


def _agent(model: ScriptedModel, shared: SharedTurns | None) -> Agent:
    registry = Registry([no_action_operation()])
    return Agent(model, registry, "system", AuditLog(None), {}, shared=shared)


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
