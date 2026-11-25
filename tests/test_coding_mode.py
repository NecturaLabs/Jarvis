"""Coding mode in the agent, the assembled app and the `jarvis model` command."""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from strata_fixture import engine_run_config, install_engine, touch

from jarvis import cli, config, modes, paths, services
from jarvis.agent import Agent, Hooks, Limits, Setup
from jarvis.app import (
    channel_limits,
    channel_registry,
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
    from collections.abc import Callable
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

    def current(self) -> str:
        return self.mode

    def setup(self, mode: str) -> Setup:
        return self.setups[mode]

    def switch(self, to: str, say: Callable[[str], None]) -> str:
        self.switched.append(to)
        if self.refuse:
            return self.refuse
        say(f"Switching to {to}.")
        self.mode = to
        return ""

    def used(self) -> None:
        self.uses += 1


def _agent(
    model: _Model, extra: list[Operation] | None = None, *, mode: str = "normal", refuse: str = ""
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
        bypass_consent=True,
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
