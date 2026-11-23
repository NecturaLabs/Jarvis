import json
import os
from typing import TYPE_CHECKING, Any

import pytest

from jarvis import osskills, paths
from jarvis.agent import Agent, Hooks
from jarvis.app import (
    MAX_HINT_CHARS,
    PAGE_HEADROOM,
    Channel,
    SetupError,
    action_rate,
    build,
    channel_limits,
    channel_registry,
    chat_pictures,
    echoes_hint,
    no_action_operation,
    overrides_below_floor,
    pc_operations,
    speech_hint,
    strata_key_file,
    system_prompt,
    unknown_overrides,
)
from jarvis.audit import AuditLog
from jarvis.config import (
    AudioConfig,
    BoundariesConfig,
    Config,
    FilesConfig,
    OpClassName,
    PersonaConfig,
    ShellConfig,
    SpotifyConfig,
)
from jarvis.operations import Operation, OperationError, Registry, Result
from jarvis.secretfile import ensure_api_key
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


def _full_catalog(root: Path) -> None:
    """Fill `root` with Omarchy skills up to the catalog's size, long names and all."""
    words = " ".join(["word"] * 80)
    for n in range(40):
        folder = root / f"a-long-omarchy-skill-name-number-{n:02}"
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(
            f"---\ndescription: {words}. Second.\n---\n# Skill\n", encoding="utf-8"
        )
        for m in range(6):
            (folder / f"a-long-guide-file-name-number-{m:02}.md").write_text("# Guide\n")


@pytest.mark.parametrize("channel", ["voice", "chat"])
def test_the_default_context_holds_the_whole_remembered_conversation(
    tmp_path: Path, channel: Channel, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The prompt, every tool schema, one reply and two tool results at their cap leave room for
    # the configured memory, so the defaults never squeeze it, even with a full Omarchy catalog.
    _full_catalog(tmp_path / "skills")
    monkeypatch.setattr("jarvis.osskills.ROOT", tmp_path / "skills")
    monkeypatch.setattr("jarvis.osskills.OWNER", os.getuid())
    full = osskills.catalog()
    assert sum(skill.catalog_chars() for skill in full) > osskills.MAX_CATALOG_CHARS * 0.8
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

    for name in (
        "pc_stats",
        "list_dir",
        "read_file",
        "write_file",
        "find_processes",
        "stop_process",
    ):
        assert registry.get(name) is not None


@pytest.mark.parametrize("channel", ["voice", "chat"])
def test_both_channels_offer_the_calculator(tmp_path: Path, channel: Channel) -> None:
    config = Config()
    spotify = SpotifyOps(SpotifyApi(TokenStore("id", tmp_path / "token.json")), config.spotify)

    registry = channel_registry(
        config, spotify.operations(), channel_limits(config, channel), channel
    )

    assert registry.get("calculate") is not None


def test_jarvis_own_state_is_never_readable_through_the_file_operations(tmp_path: Path) -> None:
    state = paths.state_dir()
    state.mkdir(parents=True, exist_ok=True)
    (state / "spotify-token.json").write_text("{}", encoding="utf-8")
    config = Config(files=FilesConfig(read_roots=[str(state.parent)]))
    ops = {op.name: op for op in pc_operations(config, channel_limits(config, "chat"), tmp_path)}

    with pytest.raises(OperationError, match="private"):
        ops["read_file"].run({"path": str(state / "spotify-token.json")})


def test_only_the_chat_offers_pictures_and_keeps_them_in_its_own_folder(tmp_path: Path) -> None:
    config = Config()
    spotify = SpotifyOps(SpotifyApi(TokenStore("id", tmp_path / "token.json")), config.spotify)

    chat = channel_registry(config, spotify.operations(), channel_limits(config, "chat"), "chat")
    voice = channel_registry(config, spotify.operations(), channel_limits(config, "voice"), "voice")

    assert chat.get("image_search") is not None
    assert chat.get("show_image") is not None
    assert voice.get("image_search") is None
    assert voice.get("show_image") is None
    assert chat_pictures().folder == paths.state_dir() / "chat-pictures"


def test_every_operation_named_in_boundaries_classes_must_exist(tmp_path: Path) -> None:
    # A typo in an override (`spotify_nxt = "deny"`) would otherwise leave the operation at its
    # default class without a word (audit finding A4).
    spotify = SpotifyOps(SpotifyApi(TokenStore("id", tmp_path / "token.json")), Config().spotify)
    config = Config(
        boundaries=BoundariesConfig(classes={"spotify_nxt": "deny", "write_file": "deny"})
    )

    assert unknown_overrides(config, spotify.operations()) == ["spotify_nxt"]


@pytest.mark.parametrize(
    ("override", "refused"), [("safe", True), ("confirm", True), ("strong", False), ("deny", False)]
)
def test_stop_process_may_only_be_set_to_strong_or_deny(
    tmp_path: Path, override: OpClassName, *, refused: bool
) -> None:
    # A program is stopped only by the owner's own command and the confirm key.
    spotify = SpotifyOps(SpotifyApi(TokenStore("id", tmp_path / "token.json")), Config().spotify)
    config = Config(boundaries=BoundariesConfig(classes={"stop_process": override}))

    expected = ["stop_process"] if refused else []
    assert list(overrides_below_floor(config, spotify.operations())) == expected


def test_build_refuses_an_override_that_takes_stop_process_below_strong() -> None:
    ensure_api_key(strata_key_file())
    config = Config(
        spotify=SpotifyConfig(client_id="id"),
        boundaries=BoundariesConfig(classes={"stop_process": "confirm"}),
    )

    with pytest.raises(SetupError, match="stop_process"):
        build(config)


def test_build_names_what_the_overridden_operation_itself_allows() -> None:
    ensure_api_key(strata_key_file())
    config = Config(
        spotify=SpotifyConfig(client_id="id"),
        shell=ShellConfig(enabled=True),
        boundaries=BoundariesConfig(classes={"pc_shell": "confirm"}),
    )

    with pytest.raises(SetupError, match="pc_shell") as refused:
        build(config)

    assert "stop_process" not in str(refused.value)  # the floor named is pc_shell's own
    assert 'pc_shell takes only "strong" or "deny"' in str(refused.value)


def test_the_system_prompt_says_never_to_stop_a_program_the_owner_did_not_name() -> None:
    prompt = system_prompt(Config())

    assert "find_processes" in prompt
    assert "never stop, close or end any program" in prompt.casefold()
    assert "ask which" in prompt.casefold()


def test_an_override_for_a_chat_only_operation_is_known(tmp_path: Path) -> None:
    spotify = SpotifyOps(SpotifyApi(TokenStore("id", tmp_path / "token.json")), Config().spotify)
    config = Config(boundaries=BoundariesConfig(classes={"show_image": "deny"}))

    assert unknown_overrides(config, spotify.operations()) == []


def test_the_cap_on_actions_comes_from_the_config_and_is_shared_through_the_state_folder() -> None:
    config = Config(boundaries=BoundariesConfig(max_actions=5, actions_window_s=90.0))

    rate = action_rate(config)

    assert (rate.limit, rate.window_s) == (5, 90.0)
    assert [rate.take() for _ in range(6)] == [True] * 5 + [False]
    assert (paths.state_dir() / "actions.json").exists()


@pytest.mark.parametrize(("separator", "said"), [(",", "1,488"), (" ", "1 488"), ("", "1488")])
def test_the_prompt_tells_the_model_how_to_write_numbers(separator: str, said: str) -> None:
    config = Config.model_validate({"format": {"thousands_separator": separator}})

    assert said in system_prompt(config, "chat")


def test_both_channels_offer_omarchy_skills_when_they_are_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "skills"
    (root / "omarchy").mkdir(parents=True)
    (root / "omarchy" / "SKILL.md").write_text(
        "---\nname: omarchy\ndescription: Change the desktop.\n---\n# Omarchy\n", encoding="utf-8"
    )
    monkeypatch.setattr("jarvis.osskills.ROOT", root)
    monkeypatch.setattr("jarvis.osskills.OWNER", os.getuid())
    config = Config()

    for channel in ("voice", "chat"):
        limits = channel_limits(config, channel)
        op = channel_registry(config, [], limits, channel).get("os_skill")
        assert op is not None
        result = op.run({"file": "omarchy/SKILL.md", "part": 1})
        assert "# Omarchy" in result.text
        assert len(result.text) <= limits.tool_text_chars
        assert "  - omarchy: Change the desktop.\n" in system_prompt(config, channel)


def test_no_omarchy_skill_operation_without_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("jarvis.osskills.ROOT", tmp_path / "absent")
    config = Config()

    registry = channel_registry(config, [], channel_limits(config, "voice"), "voice")

    assert registry.get("os_skill") is None
    assert "os_skill" not in system_prompt(config, "voice")


def test_an_os_skill_override_is_known_even_without_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PC without Omarchy's guides must still start with `os_skill` set in the config."""
    monkeypatch.setattr("jarvis.osskills.ROOT", tmp_path / "absent")
    config = Config(boundaries=BoundariesConfig(classes={"os_skill": "deny"}))

    assert unknown_overrides(config, []) == []
