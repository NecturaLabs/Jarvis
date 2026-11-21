import json
from typing import TYPE_CHECKING, Any

import pytest

from jarvis import paths
from jarvis.agent import Agent, Hooks
from jarvis.app import (
    MAX_HINT_CHARS,
    PAGE_HEADROOM,
    Channel,
    channel_limits,
    channel_registry,
    echoes_hint,
    no_action_operation,
    pc_operations,
    speech_hint,
    system_prompt,
)
from jarvis.audit import AuditLog
from jarvis.config import AudioConfig, Config, FilesConfig, PersonaConfig
from jarvis.operations import Operation, OperationError, Registry, Result
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


@pytest.mark.parametrize("channel", ["voice", "chat"])
def test_the_default_context_holds_the_whole_remembered_conversation(
    tmp_path: Path, channel: Channel
) -> None:
    # The prompt, every tool schema, one reply and two tool results at their cap leave room for
    # the configured memory, so the defaults never squeeze it.
    config = Config()
    spotify = SpotifyOps(SpotifyApi(TokenStore("id", tmp_path / "token.json")), config.spotify)
    limits = channel_limits(config, channel)
    registry = channel_registry(config, spotify.operations(), limits, channel)
    agent = Agent(
        _Model([]), registry, system_prompt(config, channel), AuditLog(None), {}, limits=limits
    )

    assert agent.memory.budget == limits.memory_tokens
    assert config.research.page_chars + PAGE_HEADROOM <= limits.tool_text_chars


def test_the_chat_reads_a_page_to_its_own_room(tmp_path: Path) -> None:
    config = Config()
    spotify = SpotifyOps(SpotifyApi(TokenStore("id", tmp_path / "token.json")), config.spotify)
    limits = channel_limits(config, "chat")

    registry = channel_registry(config, spotify.operations(), limits, "chat")

    assert registry.get("web_read") is not None
    assert registry.get("read_skill") is not None
    reply = registry.get("no_action")
    assert reply is not None
    assert reply.parameters["properties"]["reply"]["maxLength"] >= limits.reply_tokens * 4


def test_the_chat_prompt_asks_for_markdown_and_the_voice_prompt_for_speech() -> None:
    chat = system_prompt(Config(), "chat")
    voice = system_prompt(Config(), "voice")

    assert "Markdown" in chat
    assert "spoken" in voice
    assert "read_skill" in chat
    assert "web-research" in voice


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


@pytest.mark.parametrize("channel", ["voice", "chat"])
def test_both_channels_offer_the_pc_and_file_operations(tmp_path: Path, channel: Channel) -> None:
    config = Config()
    spotify = SpotifyOps(SpotifyApi(TokenStore("id", tmp_path / "token.json")), config.spotify)
    limits = channel_limits(config, channel)

    registry = channel_registry(config, spotify.operations(), limits, channel)

    for name in ("pc_stats", "list_dir", "read_file", "write_file"):
        assert registry.get(name) is not None


def test_jarvis_own_state_is_never_readable_through_the_file_operations(tmp_path: Path) -> None:
    state = paths.state_dir()
    state.mkdir(parents=True, exist_ok=True)
    (state / "spotify-token.json").write_text("{}", encoding="utf-8")
    config = Config(files=FilesConfig(read_roots=[str(state.parent)]))
    ops = {op.name: op for op in pc_operations(config, channel_limits(config, "chat"), tmp_path)}

    with pytest.raises(OperationError, match="private"):
        ops["read_file"].run({"path": str(state / "spotify-token.json")})
