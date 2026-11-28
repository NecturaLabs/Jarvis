"""`set_persona`: the owner's passing style for Jarvis, kept privately until they reset it."""

import stat
from typing import TYPE_CHECKING

import pytest
from scripted_model import ScriptedModel, say

from jarvis.agent import Agent, Hooks
from jarvis.audit import AuditLog
from jarvis.operations import ArgumentError, OperationError, Registry, request_scope, validate_args
from jarvis.persona import MAX_STYLE, PersonaStore, style_note

if TYPE_CHECKING:
    from pathlib import Path


def test_a_style_is_kept_privately_until_it_is_reset(tmp_path: Path) -> None:
    store = PersonaStore(tmp_path / "state" / "persona.json")
    op = store.operation()

    with request_scope(foreign=False, words="talk like a pirate"):
        result = op.run(validate_args(op, {"style": "a pirate: arr, matey, nautical slang"}))

    assert result.ok
    assert store.style() == "a pirate: arr, matey, nautical slang"
    assert stat.S_IMODE((tmp_path / "state" / "persona.json").stat().st_mode) == 0o600
    assert PersonaStore(tmp_path / "state" / "persona.json").style() == store.style()  # a restart

    with request_scope(foreign=False, words="be yourself again"):
        op.run(validate_args(op, {"style": ""}))

    assert not store.style()


@pytest.mark.parametrize("source", ["pc_audio", "scheduled", "outside"])
def test_only_the_owner_present_can_change_the_style(tmp_path: Path, source: str) -> None:
    store = PersonaStore(tmp_path / "persona.json")
    op = store.operation()
    with request_scope(foreign=source == "pc_audio", scheduled=source == "scheduled") as request:
        if source == "outside":
            request.saw("page")
        with pytest.raises(OperationError, match="owner"):
            op.run(validate_args(op, {"style": "rude"}))
    assert not store.style()


def test_a_style_longer_than_the_limit_is_refused(tmp_path: Path) -> None:
    op = PersonaStore(tmp_path / "persona.json").operation()

    with pytest.raises(ArgumentError):
        validate_args(op, {"style": "x" * (MAX_STYLE + 1)})


def test_an_unreadable_or_damaged_file_means_no_style(tmp_path: Path) -> None:
    path = tmp_path / "persona.json"
    path.write_text("{not json", encoding="utf-8")

    assert not PersonaStore(path).style()
    assert not PersonaStore(tmp_path / "missing.json").style()


def test_the_note_keeps_the_style_to_wording_and_out_of_the_boundaries() -> None:
    note = style_note("a hyped streamer")

    assert "a hyped streamer" in note
    assert "never" in note  # it changes no tool, rule or confirmation
    assert not style_note("")


def test_the_agent_speaks_in_the_style_set_from_the_next_request_on(tmp_path: Path) -> None:
    store = PersonaStore(tmp_path / "persona.json")
    model = ScriptedModel([say("Hi."), say("Ahoy.")])
    agent = Agent(
        model,
        Registry([]),
        "You are Jarvis.",
        AuditLog(tmp_path / "audit.jsonl"),
        {},
        style=store.style,
    )

    agent.handle("hello", Hooks(confirm=lambda _q: False))
    store.save("a pirate")
    agent.handle("hello again", Hooks(confirm=lambda _q: False))

    first, last = (str(seen[0]["content"]) for seen in (model.seen[0], model.seen[-1]))
    assert "pirate" not in first
    assert last.startswith("You are Jarvis.")
    assert "a pirate" in last
