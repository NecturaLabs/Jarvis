"""Spotify through the agent: what a confirmation is bound to, and requests the model never sees."""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from test_spotify_ops import FakeSpotify, build_ops, playing_state

from jarvis.agent import Agent, Hooks
from jarvis.audit import AuditLog
from jarvis.operations import Registry
from jarvis.spotify.ops import SpotifyOps

if TYPE_CHECKING:
    from pathlib import Path


def _call(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }
    return {"role": "assistant", "content": "", "tool_calls": [call]}


@dataclass
class ScriptedModel:
    """A model that answers from a script."""

    replies: list[dict[str, Any]]
    calls: int = 0
    seen: list[list[dict[str, Any]]] = field(default_factory=list[list[dict[str, Any]]])

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],  # noqa: ARG002
        *,
        require_tool: bool = False,  # noqa: ARG002
    ) -> dict[str, Any]:
        self.calls += 1
        self.seen.append(messages)
        return self.replies.pop(0)

    def is_loaded(self) -> bool | None:
        return True


def _agent(ops: SpotifyOps, model: ScriptedModel, overrides: dict[str, str] | None = None) -> Agent:
    registry = Registry(ops.operations())
    return Agent(model, registry, "system", AuditLog(None), overrides or {}, names=("Jarvis",))


def test_a_confirmed_like_saves_the_song_that_was_asked_about(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state())
    asked: list[str] = []

    def confirm(question: str) -> bool:
        asked.append(question)
        fake.state = {
            **playing_state(),
            "item": {**playing_state()["item"], "uri": "spotify:track:next"},
        }
        return True

    model = ScriptedModel([_call("spotify_like"), {"role": "assistant", "content": "Saved."}])
    agent = _agent(build_ops(fake, tmp_path), model, {"spotify_like": "confirm"})

    reply = agent.handle("put this in my favourites", Hooks(confirm=confirm))

    assert reply == "Saved."
    assert asked == ["Should I save Low Tide by Glass Harbor to your library?"]
    (request,) = fake.calls("PUT", "/me/library")
    assert request.url.params["uris"] == "spotify:track:lowtide"


def test_a_confirmed_volume_change_sets_the_level_that_was_asked_about(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(volume=40))
    asked: list[str] = []

    def confirm(question: str) -> bool:
        asked.append(question)
        fake.state = playing_state(volume=80)
        return True

    model = ScriptedModel(
        [_call("spotify_change_volume", {"delta": 10}), {"role": "assistant", "content": "Done."}]
    )
    agent = _agent(build_ops(fake, tmp_path), model, {"spotify_change_volume": "confirm"})

    agent.handle("a bit louder please, would you", Hooks(confirm=confirm))

    assert asked == ["Should I change the volume from 40 to 50 percent?"]
    assert fake.calls("PUT", "/me/player/volume")[0].url.params["volume_percent"] == "50"


def test_a_declined_like_saves_nothing(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state())
    model = ScriptedModel([_call("spotify_like"), {"role": "assistant", "content": "Okay."}])
    agent = _agent(build_ops(fake, tmp_path), model, {"spotify_like": "confirm"})

    agent.handle("put this in my favourites", Hooks(confirm=lambda _q: False))

    assert not fake.calls("PUT", "/me/library")


def test_like_with_nothing_playing_fails_before_any_question(tmp_path: Path) -> None:
    fake = FakeSpotify()
    asked: list[str] = []

    def confirm(question: str) -> bool:
        asked.append(question)
        return True

    model = ScriptedModel(
        [_call("spotify_like"), {"role": "assistant", "content": "Nothing is on."}]
    )
    agent = _agent(build_ops(fake, tmp_path), model, {"spotify_like": "confirm"})

    agent.handle("put this in my favourites", Hooks(confirm=confirm))

    assert asked == []
    assert "Nothing is playing." in json.dumps(model.seen[-1])
