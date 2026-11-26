"""Coding mode in the agent, the assembled app and the `jarvis model` command."""

import contextlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pytest
from strata_fixture import engine_run_config, install_engine, touch

from jarvis import cli, config, modes, paths, services
from jarvis.agent import Agent, Hooks, Limits, Setup
from jarvis.app import (
    AgentModes,
    channel_limits,
    channel_registry,
    mode_model,
    no_action_operation,
    switch_operation,
    system_prompt,
)
from jarvis.audit import AuditLog
from jarvis.cli import main
from jarvis.config import Config
from jarvis.operations import Operation, Registry, Result
from jarvis.policy import Origin
from jarvis.secretfile import write_private

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from jarvis.model import Delta, Thinking

HOOKS = Hooks(confirm=lambda _question: True)
_coding_checks = cli._coding_checks  # pyright: ignore[reportPrivateUsage] - the doctor's checks for coding mode


def _call(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    function = {"name": name, "arguments": json.dumps(args or {})}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": function}],
    }


def _reply(text: str) -> dict[str, Any]:
    return _call("no_action", {"reply": text})


@dataclass
class _Model:
    replies: list[dict[str, Any]]
    systems: list[str] = field(default_factory=list[str])
    max_tokens: list[int | None] = field(default_factory=list[int | None])
    tools: list[list[str]] = field(default_factory=list[list[str]])

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        require_tool: bool | str = False,
        timeout_s: float | None = None,
        on_delta: Callable[[Delta], None] | None = None,
        max_tokens: int | None = None,
        load_s: float = 0.0,
        thinking: Thinking | None = None,
    ) -> dict[str, Any]:
        del require_tool, timeout_s, on_delta, load_s, thinking
        self.systems.append(str(messages[0]["content"]))
        self.max_tokens.append(max_tokens)
        self.tools.append([tool["function"]["name"] for tool in tools])
        return self.replies.pop(0)

    @staticmethod
    def is_loaded() -> bool | None:
        return True


def _switch(to: str) -> Operation:
    return Operation(
        "switch_model",
        "Switch.",
        {
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": [to]}},
            "required": ["mode"],
            "additionalProperties": False,
        },
        lambda args: Result("Switching.", mode=str(args["mode"])),
        spoken_summary=lambda _args: "switch",
    )


def _write(ran: list[str]) -> Operation:
    def run(_args: dict[str, Any]) -> Result:
        ran.append("write")
        return Result("Written.")

    return Operation(
        "write_note",
        "Write a note.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        run,
        spoken_summary=lambda _args: "write a note",
    )


@dataclass
class _Modes:
    """The modes of a test: where Jarvis is, what each mode's agent works with, the switches."""

    setups: dict[str, Setup]
    mode: str = "normal"
    refuse: str = ""
    switched: list[str] = field(default_factory=list[str])
    said: list[str] = field(default_factory=list[str])
    uses: int = 0
    answered: list[str] = field(default_factory=list[str])
    during: list[list[str]] = field(default_factory=list[list[str]])

    def current(self) -> str:
        return self.mode

    def setup(self, mode: str) -> Setup:
        return self.setups[mode]

    def switch(self, to: str, say: Callable[[str], None]) -> str:
        self.during.append(list(self.answered))
        self.switched.append(to)
        if self.refuse:
            return self.refuse
        say(f"Switching to {to}.")
        self.mode = to
        return ""

    def used(self) -> None:
        self.uses += 1

    @contextlib.contextmanager
    def answering(self) -> Generator[None]:
        self.answered.append("begins")
        try:
            yield
        finally:
            self.answered.append("ends")


def _agent(
    model: _Model,
    extra: list[Operation] | None = None,
    *,
    mode: str = "normal",
    refuse: str = "",
    bypass_consent: bool = True,
) -> tuple[Agent, _Modes]:
    normal = Setup(
        "normal prompt",
        Registry([_switch("coding"), *(extra or []), no_action_operation()]),
        Limits(reply_tokens=400),
    )
    coding = Setup(
        "coding prompt",
        Registry([_switch("normal"), *(extra or []), no_action_operation()]),
        Limits(reply_tokens=8192),
    )
    kept = _Modes({"normal": normal, "coding": coding}, mode, refuse)
    agent = Agent(
        model,
        normal.registry,
        normal.system_prompt,
        AuditLog(None),
        {},
        limits=normal.limits,
        modes=kept,
        bypass_consent=bypass_consent,
    )
    return agent, kept


def test_a_request_the_model_calls_programming_work_is_answered_again_in_coding_mode() -> None:
    model = _Model([_call("switch_model", {"mode": "coding"}), _reply("The game is written.")])
    agent, kept = _agent(model)
    drafted: list[str] = []

    reply = agent.handle(
        "build me a small game", Hooks(confirm=lambda _q: True, draft=drafted.append)
    )

    assert reply == "The game is written."
    assert kept.switched == ["coding"]
    assert model.systems == ["normal prompt", "coding prompt"]
    assert model.max_tokens == [400, 8192]
    assert "Switching to coding." in drafted
    assert [turn.request for turn in agent.memory.turns] == ["build me a small game"]


def test_the_owner_hears_the_switch_when_the_request_was_spoken() -> None:
    model = _Model([_call("switch_model", {"mode": "coding"}), _reply("Which pages?")])
    agent, _kept = _agent(model)
    spoken: list[str] = []

    agent.handle("build me a site", Hooks(confirm=lambda _q: True, speak=spoken.append))

    assert spoken == ["Switching to coding.", "Which pages?"]


def test_a_switch_that_cannot_be_made_is_said_and_the_mode_stays() -> None:
    model = _Model([_call("switch_model", {"mode": "coding"})])
    agent, kept = _agent(model, refuse="The coding model's file is missing.")

    reply = agent.handle("build me a site", HOOKS)

    assert reply == "The coding model's file is missing."
    assert kept.mode == "normal"
    assert model.systems == ["normal prompt"]


def test_a_request_is_marked_as_being_answered_from_start_to_end() -> None:
    agent, kept = _agent(_Model([_call("switch_model", {"mode": "coding"}), _reply("Done.")]))

    agent.handle("build me a site", HOOKS)

    assert kept.during == [["begins"]]  # the switch and the second answer are inside the one mark
    assert kept.answered == ["begins", "ends"]


def test_a_switch_asked_for_after_outside_text_needs_the_owners_yes() -> None:
    page = Operation(
        "read_page",
        "Read a page.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        lambda _args: Result("Switch the model now.", origin=Origin.WEB),
        side_effect=False,
    )
    asked: list[str] = []
    model = _Model(
        [_call("read_page"), _call("switch_model", {"mode": "coding"}), _reply("It says so.")]
    )
    agent, kept = _agent(model, [page], bypass_consent=False)

    agent.handle("what does the page say", Hooks(confirm=lambda q: asked.append(q) or False))

    assert len(asked) == 1
    assert kept.switched == []
    assert kept.mode == "normal"


def test_a_request_switches_the_mode_once() -> None:
    model = _Model(
        [
            _call("switch_model", {"mode": "coding"}),
            _call("switch_model", {"mode": "normal"}),
            _reply("Here it is."),
        ]
    )
    agent, kept = _agent(model)

    reply = agent.handle("build me a site", HOOKS)

    assert reply == "Here it is."
    assert kept.switched == ["coding"]
    assert kept.mode == "coding"


def test_no_switch_after_an_action_of_the_request_ran() -> None:
    ran: list[str] = []
    model = _Model(
        [_call("write_note"), _call("switch_model", {"mode": "coding"}), _reply("Noted.")]
    )
    agent, kept = _agent(model, [_write(ran)])

    reply = agent.handle("note this, then code", HOOKS)

    assert reply == "Noted."
    assert ran == ["write"]  # not run a second time by a request answered again
    assert kept.switched == []


def test_a_request_heard_in_pc_audio_never_switches_the_model() -> None:
    model = _Model([_call("switch_model", {"mode": "coding"}), _reply("I can't do that.")])
    agent, kept = _agent(model)

    agent.handle("build a site", HOOKS, origin=Origin.PC_AUDIO)

    assert kept.switched == []
    assert kept.mode == "normal"


def test_an_agent_follows_a_mode_switched_elsewhere_at_its_next_request() -> None:
    model = _Model([_reply("Hello."), _reply("Hello again."), _reply("And back.")])
    agent, kept = _agent(model)
    agent.handle("hello", HOOKS)

    kept.mode = "coding"  # `jarvis model coding`, or another window
    agent.handle("hello", HOOKS)
    kept.mode = "normal"
    agent.handle("hello", HOOKS)

    assert model.systems == ["normal prompt", "coding prompt", "normal prompt"]
    assert model.max_tokens == [400, 8192, 400]
    assert model.tools[1] == ["switch_model", "no_action"]
    assert kept.uses == 3


def test_an_agent_starts_in_the_mode_jarvis_is_in() -> None:
    model = _Model([_reply("Hello.")])
    agent, _kept = _agent(model, mode="coding")

    agent.handle("hello", HOOKS)

    assert model.systems == ["coding prompt"]


def test_the_model_may_switch_itself_in_the_chat_and_not_by_voice_by_default() -> None:
    cfg = Config()

    chat = channel_registry(cfg, [], channel_limits(cfg, "chat"), "chat")
    voice = channel_registry(cfg, [], channel_limits(cfg, "voice"), "voice")

    assert "switch_model" in chat.names()
    assert "switch_model" not in voice.names()


def test_the_switch_tool_offers_only_the_other_mode() -> None:
    cfg = Config()

    into = switch_operation(cfg, coding=False).tool_schema()["function"]
    out = switch_operation(cfg, coding=True).tool_schema()["function"]

    assert into["parameters"]["properties"]["mode"]["enum"] == ["coding"]
    assert out["parameters"]["properties"]["mode"]["enum"] == ["normal"]
    assert switch_operation(cfg, coding=False).side_effect  # outside text needs a yes for it


def test_the_prompt_gains_its_coding_part_and_the_owners_own_in_coding_mode() -> None:
    cfg = Config.model_validate({"coding": {"prompt": "Use tabs."}})

    usual = system_prompt(cfg, "chat")
    coding = system_prompt(cfg, "chat", coding=True)

    assert "coding mode" not in usual
    assert "You are in coding mode" in coding
    assert coding.rstrip().endswith("Use tabs.")
    assert coding.startswith(usual[:200])


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """A config with a model of coding mode's own, set up as `jarvis setup` leaves it."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    install_engine(tmp_path / "components")
    usual = engine_run_config(tmp_path)
    data = json.loads(usual.read_text(encoding="utf-8"))
    coder = touch(tmp_path / "models" / "coder.gguf")
    data["args"] = [str(coder) if a.endswith("q4.gguf") else a for a in data["args"]]
    data["model_name"] = "flash-coder"
    coding = tmp_path / "install" / "strata-coder.json"
    coding.write_text(json.dumps(data), encoding="utf-8")
    cfg = Config.model_validate(
        {
            "strata": {
                "home": str(tmp_path / "strata-home"),
                "components_parent": str(tmp_path / "components"),
                "run_config": str(usual),
                "coding_run_config": str(coding),
                "vram_reserve_mib": 12000,
            }
        }
    )
    config.save(cfg)
    write_private(paths.strata_key_file(), "STRATA_API_KEY=k\n")
    services.write_strata_profile(cfg)
    services.write_strata_unit(cfg, paths.strata_key_file())
    return cfg


def _systemd(monkeypatch: pytest.MonkeyPatch, state: str) -> list[tuple[str, ...]]:
    """Put the model server's unit in `state`; return the systemctl calls, recorded, never run."""
    ran: list[tuple[str, ...]] = []

    def systemctl(*args: str) -> int:
        ran.append(args)
        return 0

    def unit_state(_unit: str) -> str:
        return state

    monkeypatch.setattr(cli, "_systemctl", systemctl)
    monkeypatch.setattr(cli, "_unit_state", unit_state)
    monkeypatch.setattr(cli, "voice_busy", lambda: False)
    return ran


@pytest.fixture
def stopped_server(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """A model server that is not running: systemctl is recorded, never run."""
    return _systemd(monkeypatch, "inactive")


@pytest.mark.usefixtures("installed", "stopped_server")
def test_jarvis_model_shows_the_mode_and_each_modes_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["model"]) == 0

    out = capsys.readouterr().out
    assert "mode: normal (model server not running)" in out
    assert "* normal: flash-q4, 24,576 tokens of context" in out
    assert "  coding: flash-coder, 65,536 tokens of context" in out


@pytest.mark.usefixtures("installed")
def test_jarvis_model_switches_and_audits_it(
    stopped_server: list[tuple[str, ...]], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["model", "coding"]) == 0

    assert "Now in coding mode: flash-coder." in capsys.readouterr().out
    assert modes.current() == "coding"
    assert stopped_server == []  # a stopped server is not started by a switch
    events = [json.loads(line) for line in paths.audit_file().read_text().splitlines()]
    assert events[-1]["event"] == "mode_switched"
    assert (events[-1]["to"], events[-1]["by"], events[-1]["via"]) == ("coding", "owner", "cli")

    assert main(["model", "coding"]) == 0
    assert "Already in coding mode" in capsys.readouterr().out


@pytest.mark.usefixtures("installed")
def test_a_running_server_is_restarted_by_a_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    ran = _systemd(monkeypatch, "active")

    class Server:
        @staticmethod
        def status() -> str:
            return "loaded"

        @staticmethod
        def load() -> bool:
            return True

    def strata(_cfg: Config) -> Server:
        return Server()

    monkeypatch.setattr(cli, "_strata", strata)

    assert main(["model", "coding"]) == 0

    assert ("daemon-reload",) in ran
    assert ran[-1] == ("restart", services.STRATA_UNIT)


@pytest.mark.usefixtures("stopped_server")
def test_a_restart_in_coding_mode_keeps_the_coding_models_profile(installed: Config) -> None:
    assert main(["model", "coding"]) == 0
    services.strata_profile_file().unlink()

    cli._refresh_strata()  # pyright: ignore[reportPrivateUsage] - what `jarvis restart` runs first

    profile = json.loads(services.strata_profile_file().read_text(encoding="utf-8"))
    assert (
        str(installed.strata.coding_run_config.parent.parent / "models" / "coder.gguf")
        in (profile["args"])
    )


@pytest.mark.usefixtures("stopped_server")
def test_the_doctor_checks_the_coding_models_files(installed: Config, tmp_path: Path) -> None:
    assert all(ok for ok, _what, _fix in _coding_checks(installed))

    (tmp_path / "models" / "coder.gguf").unlink()

    failed = [what for ok, what, _fix in _coding_checks(installed) if not ok]
    assert any(what.startswith("coding mode: model ") for what in failed)


def test_the_doctor_has_nothing_to_check_without_a_coding_run_config() -> None:
    assert _coding_checks(Config()) == []


def test_the_chats_config_sets_the_coding_settings_in_their_own_sections() -> None:
    config.save(Config.model_validate({"coding": {"context": 32768, "chat": {"max_steps": 9}}}))

    cli._set_shared("coding.voice.auto", True)  # pyright: ignore[reportPrivateUsage] - the chat's hook
    cli._set_shared("coding.chat.auto", False)  # pyright: ignore[reportPrivateUsage] - the chat's hook
    cli._set_shared("coding.back_after_s", 600.0)  # pyright: ignore[reportPrivateUsage] - the chat's hook

    coding = config.load().coding
    assert (coding.voice.auto, coding.chat.auto, coding.back_after_s) == (True, False, 600.0)
    assert (coding.context, coding.chat.max_steps) == (32768, 9)  # the rest of each section stays
    events = [json.loads(line) for line in paths.audit_file().read_text().splitlines()]
    assert [(e["setting"], e["was"], e["to"], e["source"]) for e in events] == [
        ("coding.voice.auto", False, True, "chat"),
        ("coding.chat.auto", True, False, "chat"),
        ("coding.back_after_s", 1800.0, 600.0, "chat"),
    ]
    shown = {s.key: s for s in cli._shared_settings()}  # pyright: ignore[reportPrivateUsage] - the chat's hook
    assert shown["coding.voice.auto"].value is True
    assert shown["coding.voice.auto"].risky  # asked about before it is turned on
    assert {shown[key].section for key in shown if key.startswith("coding.")} == {"Mode"}


def test_a_coding_setting_takes_only_the_values_the_chat_offers() -> None:
    config.save(Config())

    with pytest.raises(config.ConfigError, match="cannot set"):
        cli._set_shared("coding.back_after_s", 5.0)  # pyright: ignore[reportPrivateUsage] - the chat's hook
    with pytest.raises(config.ConfigError, match="cannot set"):
        cli._set_shared("coding.context", 1024)  # pyright: ignore[reportPrivateUsage] - the chat's hook


def _followed(made: list[str]) -> AgentModes:
    def make(mode: modes.Mode, chat: config.ChatConfig | None) -> Setup:
        made.append(mode)
        cfg = modes.effective(Config() if chat is None else Config(chat=chat), mode)
        return Setup(
            system_prompt(cfg, "chat", coding=mode == "coding"),
            Registry([no_action_operation()]),
            channel_limits(cfg, "chat"),
        )

    return AgentModes(cast("modes.Switcher", None), make)  # no switch is made here


def test_a_mode_change_keeps_what_the_chat_set_since_it_started() -> None:
    made: list[str] = []
    followed = _followed(made)
    before = followed.setup("normal")
    assert followed.setup("normal") is before  # built once
    chat = config.ChatConfig(thinking="high", thinking_max="high", response_format="verbose")

    followed.restyle(chat)

    after = followed.setup("normal")
    assert made == ["normal", "normal"]
    assert after.limits == channel_limits(Config(chat=chat), "chat")
    assert after.limits != before.limits
    assert after.system_prompt == system_prompt(Config(chat=chat), "chat")
    assert after.system_prompt != before.system_prompt
    coding = modes.effective(Config(chat=chat), "coding")
    assert followed.setup("coding").limits == channel_limits(coding, "chat")  # its own on top


def test_saving_the_chats_settings_tells_the_running_chats_modes() -> None:
    config.save(Config())
    made: list[str] = []
    followed = _followed(made)
    followed.setup("normal")
    chat = config.ChatConfig(thinking="high", thinking_max="high")

    cli._save_chat_settings(chat, followed)  # pyright: ignore[reportPrivateUsage] - the chat's hook

    assert followed.setup("normal").limits == channel_limits(Config(chat=chat), "chat")
    assert config.load().chat.thinking == "high"


@pytest.mark.usefixtures("stopped_server")
def test_the_chat_renders_its_prompt_for_the_mode_jarvis_is_in(installed: Config) -> None:
    chat = installed.chat.model_copy(update={"response_format": "verbose"})
    usual = cli._chat_prompt(chat)  # pyright: ignore[reportPrivateUsage] - the chat's hook
    assert main(["model", "coding"]) == 0

    coding = cli._chat_prompt(chat)  # pyright: ignore[reportPrivateUsage] - the chat's hook

    assert "You are in coding mode" not in usual
    assert "You are in coding mode" in coding
    served = modes.effective(installed.model_copy(update={"chat": chat}), "coding")
    assert coding == system_prompt(served, "chat", coding=True)


@dataclass
class _Unit:
    restarts: int = 0

    @staticmethod
    def active() -> bool:
        return True

    def restart(self) -> bool:
        self.restarts += 1
        return True


class _Up:
    @staticmethod
    def status() -> str | None:
        return "loaded"

    @staticmethod
    def load() -> bool:
        return True


def _process(cfg: Config, unit: _Unit) -> AgentModes:
    """The modes of one Jarvis process: its own switcher and mark, the shared state and unit."""
    switcher = modes.Switcher(cfg, unit, lambda _cfg: _Up(), AuditLog(None), sleep=lambda _s: None)
    return AgentModes(switcher, lambda _mode, _chat: cast("Setup", None))


def test_a_switch_never_cuts_a_request_another_jarvis_process_is_answering(
    installed: Config,
) -> None:
    unit = _Unit()
    chat, voice = _process(installed, unit), _process(installed, unit)

    with chat.answering():
        refused = voice.switch("coding", lambda _line: None)
        mine = chat.switch("coding", lambda _line: None)  # the request's own switch is made
        back = voice.switch("normal", lambda _line: None)

    assert refused == modes.BUSY
    assert not mine
    assert back == modes.BUSY
    assert (unit.restarts, modes.current()) == (1, "coding")


def test_a_missing_model_file_is_the_one_of_the_mode_jarvis_is_in(
    installed: Config, tmp_path: Path
) -> None:
    model = mode_model(installed, "k")
    _process(installed, _Unit()).switch("coding", lambda _line: None)
    assert not model.file_missing()

    (tmp_path / "models" / "coder.gguf").unlink()  # the coding model's drive went away

    assert model.file_missing()
    assert model.served_file() == tmp_path / "models" / "coder.gguf"
    _process(installed, _Unit()).switch("normal", lambda _line: None)
    assert not model.file_missing()  # the usual model's file is there
