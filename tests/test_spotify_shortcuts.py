"""Spotify through the agent: what a confirmation is bound to, and requests the model never sees."""

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from test_spotify_ops import (
    EPISODE,
    FakeSpotify,
    build_ops,
    find_op,
    no_device,
    playing_state,
    spotify_error,
)

from jarvis import shortcuts
from jarvis.agent import Agent, Hooks
from jarvis.audit import AuditLog
from jarvis.operations import ArgumentError, Registry, Result, validate_args
from jarvis.policy import Origin

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.model import Thinking
    from jarvis.spotify.ops import SpotifyOps


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
    loaded: bool | None = True
    seen: list[list[dict[str, Any]]] = field(default_factory=list[list[dict[str, Any]]])

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],  # ruff: ignore[unused-method-argument] - the model's own signature
        *,
        require_tool: bool = False,  # ruff: ignore[unused-method-argument] - the model's own signature
        timeout_s: float | None = None,  # ruff: ignore[unused-method-argument] - the model's own signature
        on_delta: object = None,  # ruff: ignore[unused-method-argument] - the model's own signature
        max_tokens: int | None = None,  # ruff: ignore[unused-method-argument] - the model's own signature
        load_s: float = 0.0,  # ruff: ignore[unused-method-argument] - the model's own signature
        thinking: Thinking | None = None,  # ruff: ignore[unused-method-argument] - the model's own signature
    ) -> dict[str, Any]:
        self.calls += 1
        self.seen.append(messages)
        return self.replies.pop(0)

    def is_loaded(self) -> bool | None:
        return self.loaded


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


# --- The sentence each operation says for itself (7a) ---


def _run(
    fake: FakeSpotify, tmp_path: Path, name: str, args: dict[str, Any] | None = None
) -> Result:
    ops = build_ops(fake, tmp_path)
    operation = find_op(ops, name)
    bound = operation.prepare(args or {}) if operation.prepare else args or {}
    return operation.run(bound)


@pytest.mark.parametrize(
    ("name", "args", "spoken"),
    [
        ("spotify_pause", {}, "Paused."),
        ("spotify_resume", {}, "Playing again."),
        ("spotify_next", {}, "Skipped."),
        ("spotify_previous", {}, "Back one."),
        ("spotify_set_volume", {"percent": 40}, "Volume's at 40."),
        ("spotify_change_volume", {"delta": 10}, "Volume's at 50."),
        ("spotify_change_volume", {"delta": -50}, "Volume's at 0."),
        ("spotify_shuffle", {"on": True}, "Shuffle's on."),
        ("spotify_shuffle", {"on": False}, "Shuffle's off."),
        ("spotify_repeat", {"mode": "context"}, "Repeat's on."),
        ("spotify_repeat", {"mode": "track"}, "Repeating this song."),
        ("spotify_repeat", {"mode": "off"}, "Repeat's off."),
    ],
)
def test_a_done_command_has_a_short_sentence_of_its_own(
    tmp_path: Path, name: str, args: dict[str, Any], spoken: str
) -> None:
    result = _run(FakeSpotify(state=playing_state(volume=40)), tmp_path, name, args)

    assert result.ok
    assert result.spoken == spoken
    assert result.origin is Origin.TOOL_STATUS


def test_like_says_what_it_saved_and_marks_the_words_as_spotifys(tmp_path: Path) -> None:
    result = _run(FakeSpotify(state=playing_state()), tmp_path, "spotify_like")

    assert result.spoken == "Saved Low Tide by Glass Harbor to your library."
    assert result.origin is Origin.SPOTIFY


def test_now_playing_says_the_song_and_marks_the_words_as_spotifys(tmp_path: Path) -> None:
    result = _run(FakeSpotify(state=playing_state()), tmp_path, "spotify_now_playing")

    assert result.spoken == "This is Low Tide by Glass Harbor."
    assert result.origin is Origin.SPOTIFY


@pytest.mark.parametrize(
    ("state", "spoken"),
    [
        (playing_state(playing=False), "This is Low Tide by Glass Harbor, paused."),
        (None, "Nothing is playing."),
        ({**playing_state(), "item": None, "currently_playing_type": "ad"}, "An ad is playing."),
        (
            {**playing_state(), "item": EPISODE, "currently_playing_type": "episode"},
            "This is the episode Quiet Hours, from Evening Notes.",
        ),
        (
            {**playing_state(), "item": None, "currently_playing_type": "unknown"},
            "I can't tell what this is.",
        ),
    ],
)
def test_now_playing_has_a_sentence_for_every_state(
    tmp_path: Path, state: dict[str, Any] | None, spoken: str
) -> None:
    result = _run(FakeSpotify(state=state), tmp_path, "spotify_now_playing")

    assert result.ok
    assert result.spoken == spoken


def test_already_paused_and_already_playing_are_spoken(tmp_path: Path) -> None:
    refusal = spotify_error(403, "UNKNOWN", "Restriction violated")
    paused = FakeSpotify(state=playing_state(playing=False))
    paused.script("PUT", "/me/player/pause", refusal)
    playing = FakeSpotify(state=playing_state(playing=True))
    playing.script("PUT", "/me/player/play", refusal)

    assert _run(paused, tmp_path, "spotify_pause").spoken == "It's already paused."
    assert _run(playing, tmp_path, "spotify_resume").spoken == "It's already playing."


@pytest.mark.parametrize(
    ("name", "method", "path"),
    [
        ("spotify_pause", "PUT", "/me/player/pause"),
        ("spotify_next", "POST", "/me/player/next"),
        ("spotify_previous", "POST", "/me/player/previous"),
    ],
)
def test_with_nothing_active_the_answer_is_spoken(
    tmp_path: Path, name: str, method: str, path: str
) -> None:
    fake = FakeSpotify(devices=[])
    fake.script(method, path, no_device())

    result = _run(fake, tmp_path, name)

    assert not result.ok  # nothing was done, yet the answer needs no model
    assert result.spoken == "Nothing is playing."


def test_no_sentence_names_a_device(tmp_path: Path) -> None:
    for name, args in [
        ("spotify_now_playing", {}),
        ("spotify_pause", {}),
        ("spotify_change_volume", {"delta": 5}),
    ]:
        result = _run(FakeSpotify(state=playing_state()), tmp_path, name, args)
        assert "pc" not in result.spoken.casefold().split()


def test_a_failure_sets_no_sentence(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(supports_volume=False))
    fake.script("PUT", "/me/player/volume", spotify_error(403, "VOLUME_CONTROL_DISALLOW", "no"))

    refused = _run(fake, tmp_path, "spotify_set_volume", {"percent": 30})

    assert not refused.ok
    assert not refused.spoken


def test_a_model_run_result_still_has_its_text_for_the_model(tmp_path: Path) -> None:
    result = _run(FakeSpotify(state=playing_state()), tmp_path, "spotify_pause")

    assert result.text == "Paused."


# --- Plain requests that skip the model (7b) ---

POSITIVE: list[tuple[str, str, dict[str, Any]]] = [
    *[
        (p, "spotify_pause", {})
        for p in [
            "pause",
            "pause it",
            "pause the music",
            "pause the song",
            "pause the playback",
            "stop",
            "stop the music",
            "stop playing",
        ]
    ],
    *[
        (p, "spotify_resume", {})
        for p in [
            "resume",
            "resume the music",
            "resume the playback",
            "continue",
            "continue playing",
            "unpause",
            "keep playing",
            "play",
        ]
    ],
    *[
        (p, "spotify_next", {})
        for p in [
            "next",
            "skip",
            "next song",
            "next track",
            "next one",
            "skip it",
            "skip this",
            "skip this song",
            "skip this track",
            "skip this one",
            "play the next song",
            "play the next track",
            "play the next one",
        ]
    ],
    *[
        (p, "spotify_previous", {})
        for p in [
            "previous",
            "previous song",
            "previous track",
            "previous one",
            "go back a song",
            "go back one song",
            "play the previous song",
            "play the previous track",
            "play the previous one",
        ]
    ],
    *[
        (p, "spotify_set_volume", {"percent": 40})
        for p in [
            "volume 40",
            "volume to 40",
            "set the volume to 40",
            "set volume to 40 percent",
            "volume 40%",
            "turn the volume to 40",
        ]
    ],
    *[
        (p, "spotify_change_volume", {"delta": 10})
        for p in ["volume up", "turn it up", "turn the volume up", "louder", "a bit louder"]
    ],
    *[
        (p, "spotify_change_volume", {"delta": -10})
        for p in ["volume down", "turn it down", "turn the volume down", "quieter", "a bit quieter"]
    ],
    *[
        (p, "spotify_shuffle", {"on": True})
        for p in ["shuffle on", "turn on shuffle", "turn shuffle on", "enable shuffle"]
    ],
    *[
        (p, "spotify_shuffle", {"on": False})
        for p in [
            "shuffle off",
            "turn off shuffle",
            "turn shuffle off",
            "disable shuffle",
            "stop shuffling",
        ]
    ],
    *[
        (p, "spotify_now_playing", {})
        for p in [
            "whats playing",
            "what is playing",
            "whats this song",
            "what song is this",
            "what is this song",
            "what am i listening to",
            "whats playing right now",
        ]
    ],
    *[
        (p, "spotify_like", {})
        for p in [
            "like this",
            "like this song",
            "like this track",
            "save this song",
            "save this track",
            "i like this song",
            "add this song to my library",
            "add this song to my liked songs",
        ]
    ],
]


def _said(phrase: str, style: int) -> str:
    """Return `phrase` the way a person might say it: plain, prefixed, polite or shouted."""
    return [
        phrase,
        f"Jarvis, {phrase}",
        f"{phrase} please",
        f"Hey Jarvis, could you {phrase}, please?",
        f"{phrase.upper()}!",
        f"{phrase.capitalize()}.",
    ][style]


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(build_ops(FakeSpotify(), tmp_path).operations())


@pytest.mark.parametrize(("phrase", "operation", "arguments"), POSITIVE)
@pytest.mark.parametrize("style", range(6))
def test_every_listed_phrasing_maps_to_its_operation(
    registry: Registry, phrase: str, operation: str, arguments: dict[str, Any], style: int
) -> None:
    said = _said(phrase, style)

    hit = shortcuts.match(said, registry.shortcuts(), names=["Jarvis"])

    assert hit is not None, said
    assert hit.operation == operation
    op = registry.get(operation)
    assert op is not None
    assert validate_args(op, hit.arguments) == arguments


@pytest.mark.parametrize(
    "said",
    [
        "play some jazz",
        "pause in five minutes",
        "next friday",
        "skip to the chorus",
        "volume",
        "shuffle",
        "shuffle my liked songs",
        "stop being rude",
        "turn it up to eleven and play metal",
        "resume my podcast",
        "play Low Tide",
        "what's playing on the radio",
        "like this and that",
        "volume 40 and play jazz",
        "set the volume",
        "pause the music in the kitchen",
    ],
)
def test_anything_else_is_left_to_the_model(registry: Registry, said: str) -> None:
    assert shortcuts.match(said, registry.shortcuts(), names=["Jarvis"]) is None


def test_no_two_operations_share_a_phrasing(registry: Registry) -> None:
    for phrase, operation, _arguments in POSITIVE:
        words = shortcuts.normalise(phrase)
        hits = [name for name, sc in registry.shortcuts() if re.fullmatch(sc.pattern, words)]
        assert set(hits) == {operation}, phrase


def test_a_volume_above_a_hundred_matches_but_is_refused_by_validation(registry: Registry) -> None:
    hit = shortcuts.match("volume 101", registry.shortcuts(), names=["Jarvis"])

    assert hit is not None
    op = registry.get(hit.operation)
    assert op is not None
    with pytest.raises(ArgumentError):
        validate_args(op, hit.arguments)


# --- The whole request through the agent, without the model ---


@dataclass
class ForbiddenModel:
    """A model that fails the test the moment it is used."""

    used: list[str] = field(default_factory=list[str])

    def chat(
        self,
        messages: list[dict[str, Any]],  # ruff: ignore[unused-method-argument] - the model's own signature
        tools: list[dict[str, Any]],  # ruff: ignore[unused-method-argument] - the model's own signature
        *,
        require_tool: bool = False,  # ruff: ignore[unused-method-argument] - the model's own signature
        timeout_s: float | None = None,  # ruff: ignore[unused-method-argument] - the model's own signature
        on_delta: object = None,  # ruff: ignore[unused-method-argument] - the model's own signature
        max_tokens: int | None = None,  # ruff: ignore[unused-method-argument] - the model's own signature
        load_s: float = 0.0,  # ruff: ignore[unused-method-argument] - the model's own signature
        thinking: Thinking | None = None,  # ruff: ignore[unused-method-argument] - the model's own signature
    ) -> dict[str, Any]:
        self.used.append("chat")
        pytest.fail("the model was called for a plain command")

    def is_loaded(self) -> bool | None:
        self.used.append("is_loaded")
        pytest.fail("the model was asked whether it is loaded")


NO_YES = Hooks(confirm=lambda _q: False)


def test_pause_please_is_answered_without_the_model(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state())
    agent = Agent(
        ForbiddenModel(),
        Registry(build_ops(fake, tmp_path).operations()),
        "system",
        AuditLog(None),
        {},
        names=("Jarvis",),
    )

    reply = agent.handle("Jarvis, pause please", NO_YES)

    assert reply == "Paused."
    assert len(fake.calls("PUT", "/me/player/pause")) == 1


@pytest.mark.parametrize(
    ("said", "spoken", "method", "path"),
    [
        ("Skip", "Skipped.", "POST", "/me/player/next"),
        ("Jarvis, previous song", "Back one.", "POST", "/me/player/previous"),
        ("Volume 40%", "Volume's at 40.", "PUT", "/me/player/volume"),
        ("turn it up", "Volume's at 50.", "PUT", "/me/player/volume"),
        ("Shuffle off, please.", "Shuffle's off.", "PUT", "/me/player/shuffle"),
        ("play", "Playing again.", "PUT", "/me/player/play"),
        ("What's playing?", "This is Low Tide by Glass Harbor.", "GET", "/me/player"),
        ("like this", "Saved Low Tide by Glass Harbor to your library.", "PUT", "/me/library"),
    ],
)
def test_plain_spotify_commands_never_reach_the_model(
    tmp_path: Path, said: str, spoken: str, method: str, path: str
) -> None:
    fake = FakeSpotify(state=playing_state(volume=40))
    agent = Agent(
        ForbiddenModel(),
        Registry(build_ops(fake, tmp_path).operations()),
        "system",
        AuditLog(None),
        {},
        names=("Jarvis",),
    )

    assert agent.handle(said, NO_YES) == spoken
    assert fake.calls(method, path)


def test_a_shortcut_that_fails_goes_on_to_the_model(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state())
    fake.script("PUT", "/me/player/pause", spotify_error(500, "", "boom"))
    model = ScriptedModel([{"role": "assistant", "content": "Spotify is having trouble."}])
    agent = _agent(build_ops(fake, tmp_path), model)

    reply = agent.handle("pause", NO_YES)

    assert model.calls == 1
    assert "Spotify answered 500" in json.dumps(model.seen[0])
    assert reply != "Paused."


def test_a_volume_that_validation_refuses_goes_on_to_the_model(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state())
    model = ScriptedModel([{"role": "assistant", "content": "That's too loud."}])
    agent = _agent(build_ops(fake, tmp_path), model)

    agent.handle("volume 101", NO_YES)

    assert model.calls == 1
    assert not fake.calls("PUT", "/me/player/volume")


def test_a_shortcut_like_is_still_confirmed_when_the_owner_wants_it(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state())
    asked: list[str] = []

    def confirm(question: str) -> bool:
        asked.append(question)
        return True

    agent = Agent(
        ForbiddenModel(),
        Registry(build_ops(fake, tmp_path).operations()),
        "system",
        AuditLog(None),
        {"spotify_like": "confirm"},
        names=("Jarvis",),
    )

    reply = agent.handle("like this", Hooks(confirm=confirm))

    assert asked == ["Should I save Low Tide by Glass Harbor to your library?"]
    assert reply == "Saved Low Tide by Glass Harbor to your library."
