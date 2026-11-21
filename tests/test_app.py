import json
from typing import TYPE_CHECKING, Any

import pytest

from jarvis.agent import HISTORY_TURNS, MAX_CALLS_PER_TURN, MAX_TOOL_TEXT, Agent, Hooks
from jarvis.app import (
    MAX_HINT_CHARS,
    echoes_hint,
    no_action_operation,
    speech_hint,
    system_prompt,
    web_operations,
)
from jarvis.audit import AuditLog
from jarvis.config import AudioConfig, Config, PersonaConfig
from jarvis.operations import Operation, Registry, Result
from jarvis.spotify.api import SpotifyApi
from jarvis.spotify.auth import TokenStore
from jarvis.spotify.ops import SpotifyOps

if TYPE_CHECKING:
    from pathlib import Path


def _op(name: str) -> Operation:
    return Operation(name, name, {"type": "object", "properties": {}}, lambda _a: Result("ok"))


def test_speech_hint_comes_from_config_tools_and_library() -> None:
    cfg = Config(persona=PersonaConfig(name="Friday"), audio=AudioConfig(hint_names=2))
    registry = Registry([_op("spotify_play"), _op("spotify_set_volume"), no_action_operation()])

    hint = speech_hint(cfg, registry, ["Artist A", "Artist B", "Artist C"])

    assert hint.startswith("Friday.")
    assert "play" in hint
    assert "set volume" in hint
    assert "no action" not in hint
    assert "Artist B" in hint
    assert "Artist C" not in hint


def test_speech_hint_without_library_still_names_the_tools() -> None:
    hint = speech_hint(Config(), Registry([_op("spotify_pause")]), [])

    assert "pause" in hint


def test_speech_hint_keeps_only_printable_bounded_names() -> None:
    cfg = Config(audio=AudioConfig(hint_names=50))
    names = ["Glass\x00Harbor", "Night\nCircuit", "\ud800Low", "A" * 101, "Second Sun", ""]

    hint = speech_hint(cfg, Registry([_op("spotify_pause")]), names)

    assert "Second Sun" in hint
    for refused in ("Glass", "Night", "Low", "A" * 101):
        assert refused not in hint
    hint.encode("utf-8")  # never an unencodable argument for the transcriber


def test_speech_hint_names_stop_at_the_length_cap() -> None:
    cfg = Config(audio=AudioConfig(hint_names=200))
    names = [f"Band Number {i:03d}" for i in range(200)]

    hint = speech_hint(cfg, Registry([_op("spotify_pause")]), names)

    assert len(hint) <= MAX_HINT_CHARS
    assert "Band Number 000" in hint


@pytest.mark.parametrize(
    ("transcript", "echo"),
    [
        ("Glass Harbor, Night Circuit, Second Sun, Low Tide.", True),
        ("pause, resume, next, previous", True),
        ("play Glass Harbor please", False),
        ("Night Circuit", False),  # too short to tell from a real request
        ("", False),
    ],
)
def test_a_transcript_that_only_repeats_the_hint_is_an_echo(transcript: str, echo: bool) -> None:
    hint = (
        "Jarvis. pause, resume, next, previous. Glass Harbor, Night Circuit, Second Sun, Low Tide"
    )

    assert echoes_hint(transcript, hint) is echo


@pytest.mark.parametrize("names", [[], ["Glass Harbor", "Night Circuit"]])
def test_speech_hint_ends_as_a_sentence(names: list[str]) -> None:
    """Regression: a hint ending without a full stop made Whisper retry short requests (1.8 s)."""
    assert speech_hint(Config(), Registry([_op("spotify_pause")]), names).endswith(".")


def test_the_default_context_holds_a_full_research_request(tmp_path: Path) -> None:
    # The prompt and every tool schema, the remembered turns, a whole request's tool results at
    # their cap, and the reply, counted at a conservative 3 characters per token.
    config = Config()
    spotify = SpotifyOps(SpotifyApi(TokenStore("id", tmp_path / "token.json")), config.spotify)
    registry = Registry([*spotify.operations(), *web_operations(config), no_action_operation()])
    fixed = len(system_prompt(config)) + len(json.dumps(registry.tool_schemas()))
    results = MAX_CALLS_PER_TURN * MAX_TOOL_TEXT
    history = HISTORY_TURNS * 1200
    tokens = (fixed + results + history) / 3 + config.model.max_tokens

    assert tokens <= config.strata.context
    assert config.research.page_chars + 200 <= MAX_TOOL_TEXT


def test_a_read_page_reaches_the_model_at_its_configured_length() -> None:
    config = Config()
    text = "word " * 2000
    model = _Model([_tool("web_read", {"url": "https://kettles.example/"}), _reply("Done.")])
    page = Operation(
        "web_read",
        "Read.",
        {"type": "object", "properties": {"url": {"type": "string"}}},
        lambda _a: Result(f"Page (kettles.example):\n{text[: config.research.page_chars]}"),
        side_effect=False,
    )
    agent = Agent(model, Registry([page]), "system", AuditLog(None), {})

    agent.handle("read it", Hooks(confirm=lambda _q: False))

    seen = model.seen[1][-1]["content"]
    assert len(seen) >= config.research.page_chars


class _Model:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        super().__init__()
        self.replies = replies
        self.seen: list[list[dict[str, Any]]] = []
        self.loaded = True

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **_: object
    ) -> dict[str, Any]:
        self.seen.append([dict(m) for m in messages])
        del tools
        return self.replies.pop(0)

    def is_loaded(self) -> bool:
        return self.loaded


def _tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def _reply(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}
