"""The tool router: groups, the toolbox, `tools = "routed"` and `use_tools`."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from test_pc_windows import FakeDesktop

from jarvis.agent import Agent, Hooks, Limits
from jarvis.app import (
    ALWAYS_OFFERED,
    channel_limits,
    channel_registry,
    no_action_operation,
    offered_registry,
    system_prompt,
    unknown_overrides,
)
from jarvis.audit import AuditLog
from jarvis.config import Config
from jarvis.memory import tokens
from jarvis.operations import Operation, Registry, Result, validate_args
from jarvis.osskills import OsSkill
from jarvis.pc.screenshot import ScreenshotOps
from jarvis.pc.watch import ScreenWatch
from jarvis.pc.windows import WindowOps
from jarvis.policy import Origin
from jarvis.router import toolbox_section, use_tools_operation
from jarvis.shortcuts import Shortcut
from jarvis.spotify.api import SpotifyApi
from jarvis.spotify.auth import TokenStore
from jarvis.spotify.ops import SpotifyOps
from jarvis.toolgroups import CORE, GROUPS, USE_TOOLS, group_name

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from jarvis.agent import _Turn  # pyright: ignore[reportPrivateUsage] - typing a spy only
    from jarvis.app import Channel
    from jarvis.config import ToolMode

NO_CONFIRM = Hooks(confirm=lambda _q: False)


def _full_config() -> Config:
    """Return a config whose registry offers every operation there is."""
    config = Config()
    return config.model_copy(
        update={
            "audio": config.audio.model_copy(update={"listen_to_pc": True}),
            "shell": config.shell.model_copy(update={"enabled": True}),
            "coding": config.coding.model_copy(
                update={
                    "voice": config.coding.voice.model_copy(update={"auto": True}),
                    "chat": config.coding.chat.model_copy(update={"auto": True}),
                }
            ),
        }
    )


def _registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, channel: Channel, *, routed: bool = False
) -> Registry:
    def which(_name: str) -> str:
        return "/usr/bin/agent-browser"

    monkeypatch.setattr("jarvis.app.shutil.which", which)
    # Omarchy's skills are offered only where they are installed: one, whatever this machine has.
    guide = OsSkill("example", "An example skill.", ("SKILL.md",))
    monkeypatch.setattr("jarvis.app.osskills.catalog", lambda: (guide,))
    config = _full_config()
    mode: ToolMode = "routed" if routed else "all"
    watch = None
    if channel == "voice":
        windows = WindowOps(FakeDesktop())
        watch = ScreenWatch(windows, ScreenshotOps(windows, tmp_path))
        config = config.model_copy(
            update={"agent": config.agent.model_copy(update={"tools": mode})}
        )
    else:
        config = config.model_copy(update={"chat": config.chat.model_copy(update={"tools": mode})})
    spotify = SpotifyOps(SpotifyApi(TokenStore("id", tmp_path / "token.json")), config.spotify)
    return channel_registry(
        config, spotify.operations(), channel_limits(config, channel), channel, watching=watch
    )


def _no_program(_name: str) -> None:
    return None


def _names(schemas: list[dict[str, Any]]) -> list[str]:
    return [s["function"]["name"] for s in schemas]


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("channel", ["voice", "chat"])
def test_every_operation_has_a_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, channel: Channel, *, routed: bool
) -> None:
    registry = _registry(monkeypatch, tmp_path, channel, routed=routed)

    assert registry.ungrouped() == []
    assert registry.names()  # the registry is not empty


def test_every_group_member_is_an_operation_somewhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A stale name in a group would hide that the operation was renamed or removed.
    known = {
        name
        for channel in ("voice", "chat")
        for name in _registry(monkeypatch, tmp_path, channel, routed=True).names()  # type: ignore[arg-type]
    }
    named = {name for group in (CORE, *GROUPS) for name in group.tools}

    assert named == known
    assert len(named) == sum(len(group.tools) for group in (CORE, *GROUPS))  # in one group only


def test_toolbox_lists_every_group_the_registry_has_with_its_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = _registry(monkeypatch, tmp_path, "chat", routed=True)

    section = toolbox_section(registry)

    for group in registry.groups():
        assert f"  - {group.name}: {group.summary} Tools: {', '.join(group.tools)}." in section
    assert {g.name for g in registry.groups()} == {g.name for g in GROUPS}
    assert "use_tools" in section


def test_a_group_with_no_tool_in_this_registry_is_not_in_the_toolbox() -> None:
    registry = Registry([_op("weather"), _op("calculate"), no_action_operation()])

    section = toolbox_section(registry)

    assert "everyday" in section
    assert "music" not in section
    assert [g.tools for g in registry.groups()] == [("weather", "calculate")]


# The shipped prompt with no skills listed, version "0" and the default persona, as it was before
# the router (`{toolbox}` empty). Changing the prompt's own text changes these on purpose: replace
# them with the new digests; they exist so that routing never changes what "all" mode sends.
_PROMPT_SHA256 = {
    "voice": "156248c293d7095abe5271aabff2e677e2b482f85f7b079d1818de8a8fdce752",
    "chat": "c0563b18fe5eb9064c610f53efae25c21a39715e2830d331355fabbb49ef07c7",
}


def _bare_prompt(
    monkeypatch: pytest.MonkeyPatch, channel: Channel, registry: Registry | None
) -> str:
    monkeypatch.setattr("jarvis.app.skills.available", lambda: ())
    monkeypatch.setattr("jarvis.app.osskills.catalog", lambda: ())
    monkeypatch.setattr("jarvis.app.__version__", "0")
    return system_prompt(Config(), channel, registry=registry)


@pytest.mark.parametrize("channel", ["voice", "chat"])
def test_all_mode_prompt_bytes_are_the_ones_pinned_before_the_router(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, channel: Channel
) -> None:
    everything = _registry(monkeypatch, tmp_path, channel)

    for registry in (None, everything):
        text = _bare_prompt(monkeypatch, channel, registry)
        assert hashlib.sha256(text.encode()).hexdigest() == _PROMPT_SHA256[channel]


@pytest.mark.parametrize("channel", ["voice", "chat"])
def test_the_prompt_has_the_toolbox_only_when_routed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, channel: Channel
) -> None:
    routed = _registry(monkeypatch, tmp_path, channel, routed=True)

    box = system_prompt(Config(), channel, registry=routed)

    assert box.replace(toolbox_section(routed), "", 1) == system_prompt(Config(), channel)
    assert toolbox_section(routed) in box
    assert box.index("Your toolbox") < box.index("Skills:")


def test_use_tools_is_a_safe_read_only_operation() -> None:
    op = use_tools_operation(["music", "web"])

    assert op.name == USE_TOOLS == "use_tools"
    assert not op.side_effect
    assert op.default_class.value == "safe"
    assert USE_TOOLS in ALWAYS_OFFERED
    assert validate_args(op, {"groups": ["web"]}) == {"groups": ["web"]}
    wrong: list[dict[str, object]] = [{"groups": []}, {"groups": ["nope"]}, {"groups": "web"}, {}]
    for bad in wrong:
        with pytest.raises(ValueError, match=r"."):
            validate_args(op, bad)


# --- the agent -------------------------------------------------------------------------------


@dataclass
class _Model:
    replies: list[dict[str, Any]]
    offered: list[list[str]] = field(default_factory=list[list[str]])
    schemas: list[str] = field(default_factory=list[str])
    messages: list[list[dict[str, Any]]] = field(default_factory=list[list[dict[str, Any]]])
    required: list[bool | str] = field(default_factory=list[bool | str])
    loaded: bool = True

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        require_tool: bool | str = False,
        **_: object,
    ) -> dict[str, Any]:
        self.required.append(require_tool)
        self.messages.append([dict(m) for m in messages])
        self.offered.append(_names(tools))
        self.schemas.append(json.dumps(tools))
        return self.replies.pop(0)

    def is_loaded(self) -> bool:
        return self.loaded


def _call(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"c{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(calls)
        ],
    }


def _say(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _op(name: str, ran: list[str] | None = None, *, side_effect: bool = False) -> Operation:
    def run(_args: dict[str, Any]) -> Result:
        if ran is not None:
            ran.append(name)
        return Result(f"{name} done", spoken=f"{name} done")

    return Operation(
        name,
        name,
        {"type": "object", "properties": {}},
        run,
        side_effect=side_effect,
    )


def _routed_agent(
    model: _Model,
    ran: list[str],
    *,
    mode: str = "routed",
    overrides: dict[str, str] | None = None,
    extra: Sequence[Operation] = (),
    limits: Limits | None = None,
) -> tuple[Agent, AuditLog, Registry]:
    ops = [
        *extra,
        _op("spotify_play", ran, side_effect=True),
        _op("web_search", ran),
        _op("weather", ran),
        _op("stop_process", ran, side_effect=True),
        _op("stay_quiet", ran),
        _op("think_harder", ran),
    ]
    if mode == "routed":
        ops.append(use_tools_operation(["music", "web", "everyday", "apps", "files"]))
    registry = Registry([*ops, no_action_operation()])
    audit = AuditLog(None)
    limits = limits or Limits(tools="routed" if mode == "routed" else "all")
    return Agent(model, registry, "system", audit, overrides or {}, limits=limits), audit, registry


def test_the_first_call_offers_only_the_core_tools() -> None:
    model = _Model([_say("Hello.")])
    agent, _, _ = _routed_agent(model, [])

    agent.handle("hi", NO_CONFIRM)

    assert model.offered == [["stay_quiet", "think_harder", "use_tools", "no_action"]]


def test_use_tools_adds_the_groups_in_the_registry_order_whatever_the_order_asked() -> None:
    orders: list[list[str]] = []
    for asked in (["web", "music"], ["music", "web"]):
        model = _Model([_call(("use_tools", {"groups": asked})), _say("It is ready.")])
        agent, _, _ = _routed_agent(model, [])

        agent.handle("play something and look it up", NO_CONFIRM)

        orders.append(model.offered[1])
        assert model.schemas[1].startswith(model.schemas[0][:-1])  # the core tools stay the start
    core = ["stay_quiet", "think_harder", "use_tools", "no_action"]
    assert orders[0] == orders[1] == [*core, "spotify_play", "web_search"]


def test_the_groups_stay_offered_to_the_end_of_the_request_and_not_the_next() -> None:
    ran: list[str] = []
    model = _Model(
        [
            _call(("use_tools", {"groups": ["music"]})),
            _call(("spotify_play", {})),
            _say("Playing."),
            _say("Hello."),
        ]
    )
    agent, _, _ = _routed_agent(model, ran)

    agent.handle("play it", NO_CONFIRM)
    agent.handle("hi", NO_CONFIRM)

    assert ran == ["spotify_play"]
    assert model.offered[0] == model.offered[3]  # the next request starts with the core again
    assert model.offered[1] == model.offered[2]  # one request: the same tools to its end
    assert "spotify_play" in model.offered[2]


def test_a_tool_of_a_group_not_asked_for_is_not_run_and_the_model_is_told_how_to_get_it() -> None:
    ran: list[str] = []
    model = _Model(
        [
            _call(("weather", {})),
            _call(("use_tools", {"groups": ["everyday"]})),
            _call(("weather", {})),
            _say("Sunny."),
        ]
    )
    agent, audit, _ = _routed_agent(model, ran)

    reply = agent.handle("weather?", NO_CONFIRM)

    assert reply == "Sunny."
    assert ran == ["weather"]  # once: the first call was refused, the third ran
    refusals = [e for e in audit.events if e["event"] == "tool_not_offered"]
    assert [(e["op"], e["group"]) for e in refusals] == [("weather", "everyday")]
    told = [m["content"] for m in model.messages[-1] if m.get("role") == "tool"]
    assert 'use_tools with groups ["everyday"]' in told[0]
    assert told[0].startswith("Not run:")


def test_a_call_in_the_same_reply_as_use_tools_is_not_run_either() -> None:
    ran: list[str] = []
    model = _Model(
        [
            _call(("use_tools", {"groups": ["everyday"]}), ("weather", {})),
            _say("Done."),
        ]
    )
    agent, audit, _ = _routed_agent(model, ran)

    agent.handle("weather?", NO_CONFIRM)

    assert ran == []  # it was written before its tools were offered
    assert [e["op"] for e in audit.events if e["event"] == "tool_not_offered"] == ["weather"]


def test_use_tools_is_decided_and_audited_like_any_operation() -> None:
    model = _Model([_call(("use_tools", {"groups": ["web", "files"]})), _say("It is ready.")])
    agent, audit, _ = _routed_agent(model, [])

    agent.handle("look it up", NO_CONFIRM)

    events = {e["event"]: e for e in audit.events if e["event"] in {"tool_call", "tools_loaded"}}
    assert events["tool_call"]["op"] == "use_tools"
    assert events["tool_call"]["verdict"] == "run"
    assert events["tools_loaded"]["groups"] == ["web"]
    assert [e["event"] for e in audit.events].count("tool_call") == 1


def test_the_model_is_told_which_tools_it_got_and_which_groups_do_not_exist() -> None:
    model = _Model([_call(("use_tools", {"groups": ["web"]})), _say("It is ready.")])
    agent, _, _ = _routed_agent(model, [])

    agent.handle("look it up", NO_CONFIRM)

    told = [m["content"] for m in model.messages[-1] if m.get("role") == "tool"]
    assert told == ["Offered from your next reply, to the end of this request: web: web_search."]


def test_asking_for_a_group_twice_is_answered_and_is_not_a_stuck_loop() -> None:
    ran: list[str] = []
    model = _Model(
        [
            _call(("use_tools", {"groups": ["music"]})),
            _call(("use_tools", {"groups": ["music"]})),
            _call(("spotify_play", {})),
            _say("Playing."),
        ]
    )
    agent, _, _ = _routed_agent(model, ran)

    assert agent.handle("play it", NO_CONFIRM) == "Playing."
    assert ran == ["spotify_play"]


def test_all_mode_offers_every_tool_every_call_with_the_same_bytes_as_before() -> None:
    model = _Model([_call(("weather", {})), _say("Sunny.")])
    agent, _, registry = _routed_agent(model, [], mode="all")

    agent.handle("weather?", NO_CONFIRM)

    assert model.schemas == [json.dumps(registry.tool_schemas())] * 2
    assert "use_tools" not in model.offered[0]
    assert model.offered[0] == registry.names()


def test_routed_without_use_tools_offers_everything() -> None:
    # Where use_tools is not there (a registry that has none), nothing could be asked for.
    model = _Model([_say("Hi.")])
    registry = Registry([_op("weather"), no_action_operation()])
    agent = Agent(model, registry, "system", AuditLog(None), {}, limits=Limits(tools="routed"))

    agent.handle("hi", NO_CONFIRM)

    assert model.offered == [["weather", "no_action"]]


def test_a_heard_request_can_only_do_what_it_could_before() -> None:
    # Loading a group changes what is offered, never what the policy lets run: an action in a
    # request heard from the PC's speakers is still declined, in both modes.
    outcomes: list[list[str]] = []
    for mode in ("all", "routed"):
        ran: list[str] = []
        calls = [_call(("stop_process", {}))]
        if mode == "routed":
            calls = [_call(("use_tools", {"groups": ["apps"]})), *calls]
        model = _Model([*calls, _say("I can't.")])
        agent, audit, _ = _routed_agent(model, ran, mode=mode)

        agent.handle("stop it", NO_CONFIRM, Origin.PC_AUDIO)

        assert ran == []
        outcomes.append(
            [
                str(e["verdict"])
                for e in audit.events
                if e["event"] == "tool_call" and e["op"] == "stop_process"
            ]
        )
    assert outcomes[0] == outcomes[1] != []


def test_the_owners_class_for_use_tools_is_kept() -> None:
    # Raising it to "confirm" asks the owner before tools are offered, like any operation.
    asked: list[str] = []
    model = _Model([_call(("use_tools", {"groups": ["web"]})), _say("No.")])
    agent, audit, _ = _routed_agent(model, [], overrides={"use_tools": "confirm"})

    agent.handle("look it up", Hooks(confirm=lambda q: asked.append(q) is None and False))

    assert len(asked) == 1
    assert [
        e["verdict"] for e in audit.events if e["event"] == "tool_call" and e["op"] == "use_tools"
    ] == ["confirm"]
    assert model.offered[-1] == model.offered[0]  # declined: nothing was added


def test_prime_reads_the_same_start_a_routed_request_begins_with() -> None:
    model = _Model([_say("x")])
    agent, _, _ = _routed_agent(model, [])

    agent.prime()

    assert model.offered == [["stay_quiet", "think_harder", "use_tools", "no_action"]]


def test_the_toolbox_group_names_are_the_use_tools_choices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = _registry(monkeypatch, tmp_path, "chat", routed=True)

    op = registry.get(USE_TOOLS)

    assert op is not None
    enum = op.parameters["properties"]["groups"]["items"]["enum"]
    assert enum == [group.name for group in registry.groups()]
    assert group_name("use_tools") == "core"
    assert registry.core_names() == [
        "think_harder",
        "switch_model",
        "use_tools",
        "no_action",
    ]


# --- review fixes ------------------------------------------------------------------------------


def test_a_plain_command_runs_in_routed_mode_as_in_all_mode_without_the_model() -> None:
    # Regression: a shortcut (ADR 0004) was refused as a tool of a group not asked for.
    for mode in ("all", "routed"):
        ran: list[str] = []
        pause = Operation(
            "spotify_pause",
            "Pause.",
            {"type": "object", "properties": {}},
            lambda _args, ran=ran: (ran.append("pause"), Result("Paused.", spoken="Paused."))[1],
            shortcuts=(Shortcut("pause", lambda _g: {}),),
        )
        model = _Model([])
        agent, audit, _ = _routed_agent(model, ran, mode=mode, extra=[pause])

        assert agent.handle("pause", NO_CONFIRM) == "Paused."

        assert ran == ["pause"]
        assert model.offered == []  # no model call
        assert not [e for e in audit.events if e["event"] == "tool_not_offered"]


def test_a_refused_unoffered_call_is_not_an_action_tried() -> None:
    ran: list[str] = []
    model = _Model(
        [
            _call(("stop_process", {})),
            _call(("use_tools", {"groups": ["apps"]})),
            _call(("stop_process", {})),
            _say("Stopped it."),
        ]
    )
    agent, _, _ = _routed_agent(model, ran)

    reply = agent.handle("stop it", Hooks(confirm=lambda _q: True))

    assert reply == "Stopped it."  # the refused call left no "failed attempt" behind
    assert ran == ["stop_process"]
    assert not agent.memory.turns[-1].actions.startswith("stop_process -> error")
    assert "unknown" not in agent.memory.turns[-1].actions
    assert agent.memory.turns[-1].actions.count("stop_process") == 1


def test_a_declined_use_tools_repeated_is_a_repeat_not_already_offered() -> None:
    model = _Model(
        [
            _call(("use_tools", {"groups": ["web"]})),
            _call(("use_tools", {"groups": ["web"]})),
            _say("No."),
            _say("No."),
        ]
    )
    agent, audit, _ = _routed_agent(model, [], overrides={"use_tools": "confirm"})

    agent.handle("look it up", NO_CONFIRM)

    told = [m["content"] for m in model.messages[-1] if m.get("role") == "tool"]
    assert not any("already offered" in str(t) for t in told)
    assert [e["event"] for e in audit.events].count("tools_loaded") == 0
    assert "web_search" not in model.offered[-1]


def test_a_forced_answer_keeps_the_groups_loaded_so_far() -> None:
    # The last step may only answer, with the tool list a normal call would have.
    model = _Model(
        [
            _call(("use_tools", {"groups": ["web"]})),
            _call(("web_search", {})),
            _call(("no_action", {"reply": "Found it."})),
        ]
    )
    agent, _, _ = _routed_agent(model, [], limits=Limits(tools="routed", max_steps=3))

    assert agent.handle("look it up", NO_CONFIRM) == "Found it."

    assert model.required[2] == "no_action"
    assert model.offered[2] == model.offered[1]
    assert "web_search" in model.offered[2]


def test_calls_to_tools_not_offered_still_end_in_a_forced_answer() -> None:
    # A refused call is no action tried, but the last step must still answer, not give up.
    model = _Model(
        [
            _call(("stop_process", {})),
            _call(("stop_process", {})),
            _call(("no_action", {"reply": "I could not do that."})),
        ]
    )
    agent, _, _ = _routed_agent(model, [], limits=Limits(tools="routed", max_steps=3))

    assert agent.handle("stop it", NO_CONFIRM) == "I could not do that."

    assert model.required[2] == "no_action"


def test_a_stuck_request_is_forced_to_answer_with_the_same_routed_tools() -> None:
    model = _Model(
        [
            _call(("use_tools", {"groups": ["web"]})),
            _call(("web_search", {})),
            _call(("web_search", {})),  # the same call again: stuck
            _call(("no_action", {"reply": "That is all I found."})),
        ]
    )
    agent, _, _ = _routed_agent(model, [])

    assert agent.handle("look it up", NO_CONFIRM) == "That is all I found."

    assert model.required[3] == "no_action"
    assert model.offered[3] == model.offered[2]


def test_words_the_owner_steers_in_keep_the_loaded_groups() -> None:
    sent: list[list[str]] = [[], ["and the weather"]]
    model = _Model([_call(("use_tools", {"groups": ["web"]})), _say("It is ready.")])
    agent, _, _ = _routed_agent(model, [])

    agent.handle("look it up", Hooks(confirm=lambda _q: False, steer=lambda: sent.pop(0)))

    assert "and the weather" in str(model.messages[1])  # the steer reached the second call
    assert "web_search" in model.offered[1]  # ... which still had the group


def test_a_group_switched_off_is_not_in_the_toolbox_or_the_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jarvis.app.shutil.which", _no_program)
    config = Config()
    config = config.model_copy(
        update={
            "chat": config.chat.model_copy(update={"tools": "routed"}),
            "boundaries": config.boundaries.model_copy(
                update={
                    "classes": {
                        "schedule_add": "deny",
                        "schedule_list": "deny",
                        "schedule_cancel": "deny",
                    }
                }
            ),
        }
    )
    registry = offered_registry(config, [], "chat", coding=False, via="chat", can_switch=True)

    op = registry.get(USE_TOOLS)

    assert op is not None
    assert "schedule" not in op.parameters["properties"]["groups"]["items"]["enum"]
    assert "schedule" not in [g.name for g in registry.groups()]
    assert "schedule_add" not in toolbox_section(registry)
    assert "web" in op.parameters["properties"]["groups"]["items"]["enum"]


def test_a_tool_coding_mode_switches_off_is_not_in_its_toolbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jarvis.app.shutil.which", _no_program)
    config = Config()
    config = config.model_copy(
        update={
            "chat": config.chat.model_copy(update={"tools": "routed"}),
            "coding": config.coding.model_copy(update={"tools_off": ["web_search", "web_read"]}),
        }
    )

    registry = offered_registry(config, [], "chat", coding=True, via="chat", can_switch=True)

    assert "web_search" not in toolbox_section(registry)
    assert "web_search" not in registry.names()


def test_routed_budget_counts_every_group_but_a_call_counts_what_it_is_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The remembered conversation's room is sized for the request that loads every group, since
    # the window must hold it; making room inside a request counts what that call is offered.
    counted: list[int] = []
    made_room = Agent._make_room  # pyright: ignore[reportPrivateUsage]

    def record(self: Agent, turn: _Turn, schemas: int) -> None:
        counted.append(schemas)
        made_room(self, turn, schemas)

    monkeypatch.setattr(Agent, "_make_room", record)
    model = _Model([_call(("use_tools", {"groups": ["web"]})), _say("It is ready.")])
    agent, _, registry = _routed_agent(model, [])

    assert agent.context_use().fixed == tokens("system") + tokens(
        json.dumps(registry.tool_schemas())
    )
    agent.handle("look it up", NO_CONFIRM)

    assert counted == [
        tokens(json.dumps(registry.tool_schemas(()))),
        tokens(json.dumps(registry.tool_schemas(("web",)))),
    ]
    assert counted[0] < counted[1] < agent.context_use().fixed


def test_a_class_for_use_tools_is_not_an_unknown_name_in_all_mode() -> None:
    config = Config().model_copy(
        update={
            "boundaries": Config().boundaries.model_copy(update={"classes": {"use_tools": "safe"}})
        }
    )

    assert unknown_overrides(config, []) == []
