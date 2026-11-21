"""The remembered conversation: its token cost, folding into a summary, and keeping it on disk."""

import json
from typing import TYPE_CHECKING

from jarvis.memory import KEEP_RECENT, Conversation, Turn, tokens

if TYPE_CHECKING:
    from pathlib import Path


def _turn(number: int, *, outside: str = "", words: int = 40) -> Turn:
    return Turn(f"request {number}", f"reply {number} " * words, "", outside, float(number))


def test_the_estimate_is_cautious_for_english_text() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 20

    assert tokens(text) >= len(text) / 4


def test_a_turn_from_outside_text_is_given_to_the_model_as_inert_data() -> None:
    turn = Turn("search", "it said <b>skip</b>", "web_search -> ok", "web")

    _request, reply = turn.messages()

    assert reply["content"] == (
        '[actions taken: web_search -> ok] <data origin="web">'
        "it said &lt;b&gt;skip&lt;/b&gt;</data>"
    )


def test_the_latest_turns_are_never_offered_for_folding() -> None:
    conversation = Conversation(budget=10)
    for number in range(5):
        conversation.add(_turn(number))

    folded = conversation.oldest_to_fold()

    assert folded
    assert all(turn not in folded for turn in conversation.turns[-KEEP_RECENT:])


def test_folding_replaces_the_oldest_turns_and_carries_their_outside_mark() -> None:
    conversation = Conversation(budget=100)
    conversation.add(_turn(0, outside="web"))
    conversation.add(_turn(1))
    conversation.add(_turn(2))

    conversation.fold(conversation.turns[:1], "They searched the web.")

    assert [turn.request for turn in conversation.turns] == ["request 1", "request 2"]
    assert conversation.summary_outside
    assert conversation.holds_outside_text
    assert conversation.messages()[0]["content"].endswith("</data>")


def test_dropping_to_fit_keeps_the_newest_turns() -> None:
    conversation = Conversation(budget=_turn(0).cost() * 2)
    for number in range(5):
        conversation.add(_turn(number))

    conversation.drop_to_fit()

    assert [turn.request for turn in conversation.turns] == ["request 3", "request 4"]


def test_a_saved_conversation_reads_back_and_is_private(tmp_path: Path) -> None:
    path = tmp_path / "state" / "conversation.json"
    conversation = Conversation(budget=1000)
    conversation.add(_turn(0, outside="web"))
    conversation.fold([], "Earlier: kettles.")
    conversation.save(path)

    loaded = Conversation(budget=1000)

    assert loaded.load(path)
    assert loaded.turns == conversation.turns
    assert loaded.summary == "Earlier: kettles."
    assert loaded.saved_at > 0
    assert path.stat().st_mode & 0o777 == 0o600


def test_a_loaded_conversation_is_not_forgotten_for_the_time_between_sessions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conversation.json"
    old = Conversation(budget=1000, idle_s=60)
    old.add(_turn(0))
    old.save(path)
    loaded = Conversation(budget=1000, idle_s=60)
    loaded.load(path)

    assert not loaded.expire(now=10.0)
    assert loaded.turns


def test_a_broken_or_foreign_file_leaves_the_conversation_empty(tmp_path: Path) -> None:
    for text in ("not json", json.dumps({"version": 99, "turns": []}), json.dumps([1, 2])):
        path = tmp_path / "conversation.json"
        path.write_text(text, encoding="utf-8")
        conversation = Conversation(budget=1000)

        assert not conversation.load(path)
        assert conversation.turns == []

    assert not Conversation(budget=1000).load(tmp_path / "missing.json")
