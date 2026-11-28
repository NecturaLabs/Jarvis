"""tools/router_bench.py on a fake model: its request set, scoring, recording and tables.

Nothing here reaches a server: a scripted model answers in place of one, and the agent, the
prompt and the tool schemas under test are Jarvis's own.
"""

import json
import os
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import router_bench
from router_bench import (
    CASES_FILE,
    Call,
    Case,
    CaseError,
    CaseResult,
    ModelCall,
    Probe,
    Turn,
    UnavailableError,
    Variant,
    Want,
    arg_matches,
    judge,
    load_cases,
    parse_case,
)
from test_agent import Clock, ScriptedModel

from jarvis import config
from jarvis.model import ModelStatus
from jarvis.secretfile import write_private
from jarvis.toolgroups import CORE, GROUPS

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from router_bench import Outcome

    from jarvis.model import ChatModel

NAMES = {name for group in (CORE, *GROUPS) for name in group.tools}


def _case(
    expect: list[tuple[Want, ...]] | None = None,
    *,
    allow: tuple[str, ...] = (),
    forbid: tuple[str, ...] = (),
    channels: tuple[str, ...] = ("voice", "chat"),
) -> Case:
    return Case(
        "c",
        "music",
        "typed",
        ("hello",),
        tuple(expect or ()),
        frozenset(allow),
        frozenset(forbid),
        frozenset(channels),
    )


def _call(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    function = {"name": name, "arguments": json.dumps(args or {})}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": f"id-{name}", "type": "function", "function": function}],
    }


def _say(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _with_usage(message: dict[str, Any], prompt: int) -> dict[str, Any]:
    return {**message, "usage": {"prompt_tokens": prompt, "completion_tokens": 7}}


# --- the request set ----------------------------------------------------------------------


def test_the_shipped_request_set_loads() -> None:
    cases = load_cases(CASES_FILE)

    assert len(cases) == len({c.id for c in cases})
    assert {c.kind for c in cases} == {"typed", "spoken"}


def test_every_tool_group_has_a_typed_case() -> None:
    cases = load_cases(CASES_FILE)

    covered = {c.group for c in cases if c.kind == "typed"}

    assert {g.name for g in GROUPS} <= covered
    assert {
        "core",
        "multi",
    } <= covered  # a request that needs no tool, and ones that need two groups


def test_a_case_names_only_tools_that_exist() -> None:
    for case in load_cases(CASES_FILE):
        named = {w.tool for alternatives in case.expect for w in alternatives}

        assert named | case.allow | case.forbid <= NAMES, case.id


def test_each_expected_tool_is_offered_on_every_channel_of_its_case(tmp_path: Path) -> None:
    cases = load_cases(CASES_FILE)
    for channel in ("voice", "chat"):
        with router_bench.isolated(tmp_path / channel), router_bench.offering_everything():
            session = router_bench.make_session(
                config.Config(),
                ScriptedModel([]),
                Variant(channel, "all"),
                tmp_path / channel,  # type: ignore[arg-type]
            )
        offered = set(session.registry.names())
        for case in cases:
            if channel in case.channels:
                wanted = {w.tool for alternatives in case.expect for w in alternatives}
                assert wanted <= offered, (channel, case.id, wanted - offered)
            else:
                # A case left out of a channel is one its tools are not offered on.
                wanted = {w.tool for alternatives in case.expect for w in alternatives}
                assert not wanted <= offered, (channel, case.id)


def test_every_tool_has_a_stub_or_runs_as_shipped() -> None:
    assert set(router_bench.STUBS) | router_bench.REAL >= NAMES - {"switch_model", "chat_hand_off"}
    assert set(router_bench.STUBS) <= NAMES
    assert not set(router_bench.STUBS) & router_bench.REAL


MALFORMED: list[object] = [
    "text",
    {"id": "x", "group": "music", "set": "typed", "turns": ["a"]},  # no `expect`
    {"id": "x", "group": "nowhere", "set": "typed", "turns": ["a"], "expect": []},
    {"id": "x", "group": "music", "set": "odd", "turns": ["a"], "expect": []},
    {"id": "x", "group": "music", "set": "typed", "turns": [], "expect": []},
    {"id": "x", "group": "music", "set": "typed", "turns": ["a"], "expect": [], "what": 1},
    {
        "id": "x",
        "group": "music",
        "set": "typed",
        "turns": ["a"],
        "expect": [],
        "channels": ["tv"],
    },
    {"id": "x", "group": "music", "set": "typed", "turns": ["a"], "expect": [{"args": {}}]},
    {
        "id": "x",
        "group": "music",
        "set": "typed",
        "turns": ["a"],
        "expect": [{"any_of": [{"tool": "a"}]}],
    },
    {
        "id": "x",
        "group": "music",
        "set": "typed",
        "turns": ["a"],
        "expect": [{"tool": "spotify_next", "args": {"count": {"close_to": 2}}}],
    },
    {
        "id": "x",
        "group": "music",
        "set": "typed",
        "turns": ["a"],
        "expect": [{"tool": "spotify_next", "args": {"count": {"lt": "two"}}}],
    },
    {
        "id": "x",
        "group": "music",
        "set": "typed",
        "turns": ["a"],
        "expect": [{"tool": "spotify_next", "args": {"count": [2]}}],
    },
]


@pytest.mark.parametrize("raw", MALFORMED)
def test_a_malformed_case_is_refused(raw: object) -> None:
    with pytest.raises(CaseError):
        parse_case(raw)


def test_a_request_set_with_a_repeated_id_or_no_cases_is_refused(tmp_path: Path) -> None:
    one = {"id": "x", "group": "core", "set": "typed", "turns": ["a"], "expect": []}
    twice = tmp_path / "twice.json"
    twice.write_text(json.dumps({"cases": [one, one]}), encoding="utf-8")
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"cases": []}), encoding="utf-8")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    for path in (twice, empty, broken, tmp_path / "missing.json"):
        with pytest.raises(CaseError):
            load_cases(path)


def test_cases_are_chosen_by_channel_id_group_and_set() -> None:
    cases = load_cases(CASES_FILE)

    voice = router_bench.select(cases, channel="voice")
    chat = router_bench.select(cases, channel="chat")

    assert "sing_a_line" in {c.id for c in voice} - {c.id for c in chat}
    assert "find_picture" in {c.id for c in chat} - {c.id for c in voice}
    assert {c.group for c in router_bench.select(cases, channel="voice", only=["music"])} == {
        "music"
    }
    assert [c.id for c in router_bench.select(cases, channel="chat", only=["skip_two"])] == [
        "skip_two"
    ]
    assert {c.kind for c in router_bench.select(cases, channel="chat", kind="spoken")} == {"spoken"}


# --- matching and scoring -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "value", "expected"),
    [
        ("Tokyo", "tokyo", True),
        ("Tokyo", " TOKYO ", True),
        ("Tokyo", "Kyoto", False),
        (2, 2, True),
        (2, 2.0, True),
        (2, 3, False),
        (2, "2", False),
        (True, True, True),
        (True, 1, False),
        (1, True, False),
        (False, False, True),
        ({"contains": "daft"}, "Daft Punk", True),
        ({"contains": "daft"}, "Radiohead", False),
        ({"contains": "daft"}, 3, False),
        ({"one_of": ["07:30", "7:30"]}, "7:30", True),
        ({"one_of": ["07:30", "7:30"]}, "8:30", False),
        ({"lt": 0}, -5, True),
        ({"lt": 0}, 0, False),
        ({"lt": 0}, "-5", False),
        ({"gt": 0}, 5, True),
        ({"gt": 0}, True, False),
    ],
)
def test_an_argument_is_matched_by_value_or_by_matcher(
    spec: object, value: object, expected: bool
) -> None:
    assert arg_matches(spec, value) is expected


PLAY = Want("spotify_play", {"kind": "artist", "query": {"contains": "radio"}})
PAUSE = Want("spotify_pause")
LATER = Want("schedule_add", {"at": "23:00"})


@pytest.mark.parametrize(
    ("case", "calls", "failed", "outcome"),
    [
        (
            _case([(PLAY,)]),
            [Call("spotify_play", {"kind": "artist", "query": "Radiohead"})],
            False,
            "right",
        ),
        (
            _case([(PLAY,)]),
            [Call("spotify_play", {"kind": "track", "query": "Radiohead"})],
            False,
            "wrong_args",
        ),
        (_case([(PLAY,)]), [Call("spotify_play", {"kind": "artist"})], False, "wrong_args"),
        (_case([(PLAY,)]), [Call("spotify_pause", {})], False, "wrong_tool"),
        (_case([(PLAY,)]), [], False, "no_tool"),
        (_case([(PLAY,)]), [Call("use_tools", {"groups": ["music"]})], False, "no_tool"),
        (_case([(PLAY,)]), [Call("use_tools", {}), Call("no_action", {})], False, "no_tool"),
        (
            _case([(PLAY,)]),
            [Call("spotify_play", {"kind": "artist", "query": "radio"})],
            True,
            "error",
        ),
        # a call no case entry matches is extra, unless the case allows it or it is a core tool
        (
            _case([(PLAY,)]),
            [Call("spotify_play", {"kind": "artist", "query": "radio"}), Call("pc_stats", {})],
            False,
            "extra",
        ),
        (
            _case([(PLAY,)], allow=("pc_stats",)),
            [Call("spotify_play", {"kind": "artist", "query": "radio"}), Call("pc_stats", {})],
            False,
            "right",
        ),
        (
            _case([(PLAY,)]),
            [
                Call("use_tools", {"groups": ["music"]}),
                Call("spotify_play", {"kind": "artist", "query": "radio"}),
                Call("no_action", {"reply": "ok"}),
            ],
            False,
            "right",
        ),
        # a forbidden call is wrong even when it is also an allowed one
        (
            _case([(LATER,)], forbid=("spotify_pause",)),
            [Call("schedule_add", {"at": "23:00"}), Call("spotify_pause", {})],
            False,
            "extra",
        ),
        # each entry needs a call of its own, in any order
        (
            _case([(PAUSE,), (LATER,)]),
            [Call("schedule_add", {"at": "23:00"}), Call("spotify_pause", {})],
            False,
            "right",
        ),
        # a second call is missing: a tool answered once is not enough for two
        (_case([(PAUSE,), (PAUSE,)]), [Call("spotify_pause", {})], False, "no_tool"),
        # alternatives: either call satisfies the entry
        (_case([(PLAY, PAUSE)]), [Call("spotify_pause", {})], False, "right"),
        (
            _case([(PAUSE, LATER), (PAUSE,)]),
            [Call("spotify_pause", {}), Call("schedule_add", {"at": "23:00"})],
            False,
            "right",
        ),
        # a request that needs no tool
        (_case(), [], False, "right"),
        (_case(), [Call("no_action", {"reply": "8"})], False, "right"),
        (_case(), [Call("pc_stats", {})], False, "extra"),
    ],
)
def test_a_request_is_judged_by_the_calls_that_ran(
    case: Case, calls: list[Call], *, failed: bool, outcome: str
) -> None:
    assert judge(case, calls, failed=failed) == outcome


# --- recording ----------------------------------------------------------------------------


def test_the_probe_notes_each_call_and_passes_the_answer_on() -> None:
    reply = _with_usage(_call("use_tools", {"groups": ["music", "web"]}), 1234)
    reply["timings"] = {"cache_n": 1000}
    clock = Clock()
    model = ScriptedModel([reply, _say("done")])
    probe = Probe(model, clock)  # type: ignore[arg-type]

    first = probe.chat([{"role": "user", "content": "hi"}], [{"function": {"name": "a"}}] * 3)
    second = probe.chat([], [{"function": {"name": "a"}}] * 5)

    assert first is reply
    assert second["content"] == "done"
    assert [c.offered for c in probe.calls] == [3, 5]
    assert probe.calls[0].prompt_tokens == 1234
    assert probe.calls[0].completion_tokens == 7
    assert probe.calls[0].reused_tokens == 1000
    assert probe.calls[1].prompt_tokens is None  # the server sent no usage
    assert probe.calls[0].asked == ["use_tools"]
    assert probe.asked == [Call("use_tools", {"groups": ["music", "web"]})]
    assert probe.is_loaded() is True


def test_a_tool_call_with_unreadable_arguments_is_noted_with_none() -> None:
    reply = _call("spotify_pause")
    reply["tool_calls"][0]["function"]["arguments"] = "{not json"
    probe = Probe(ScriptedModel([reply]), Clock())  # type: ignore[arg-type]

    probe.chat([], [])

    assert probe.asked == [Call("spotify_pause", {})]


# --- a case through Jarvis's own agent ----------------------------------------------------


def _run(
    case_id: str, replies: list[dict[str, Any]], variant: Variant, tmp_path: Path
) -> tuple[CaseResult, ScriptedModel]:
    case = next(c for c in load_cases(CASES_FILE) if c.id == case_id)
    model = ScriptedModel(replies)
    clock = Clock()
    with router_bench.isolated(tmp_path), router_bench.offering_everything():
        session = router_bench.make_session(
            config.Config(),
            model,  # type: ignore[arg-type]
            variant,
            tmp_path,
            clock,
        )
        return router_bench.run_case(case, session), model


def test_an_all_request_is_one_call_and_runs_the_stubbed_tool(tmp_path: Path) -> None:
    play = {"kind": "artist", "query": "Radiohead"}
    replies = [_with_usage(_call("spotify_play", play), 8000), _say("Playing Radiohead.")]

    result, model = _run("play_artist", replies, Variant("voice", "all"), tmp_path)

    assert result.outcome == "right"
    assert result.calls == [Call("spotify_play", play)]
    assert result.loaded == []
    assert result.refused == []
    assert [len(t.model_calls) for t in result.turns] == [2]
    assert result.turns[0].model_calls[0].prompt_tokens == 8000
    assert result.turns[0].first_s is not None
    assert result.turns[0].reply == "Playing Radiohead."
    assert model.tools_offered[0] > 40  # every tool, on the first call


def test_a_routed_request_pays_a_use_tools_call_and_is_offered_more_after_it(
    tmp_path: Path,
) -> None:
    play = {"kind": "artist", "query": "Radiohead"}
    replies = [
        _call("use_tools", {"groups": ["music"]}),
        _call("spotify_play", play),
        _say("Playing Radiohead."),
    ]

    result, model = _run("play_artist", replies, Variant("voice", "routed"), tmp_path)

    assert result.outcome == "right"
    assert result.loaded == ["music"]
    assert [len(t.model_calls) for t in result.turns] == [3]
    first, second = model.tools_offered[:2]
    assert first < 10 < second  # the core tools, then the core and the music tools
    assert [c.tool for c in result.calls] == ["use_tools", "spotify_play"]


def test_the_file_skill_reaches_the_model_and_counts_toward_a_file_request(tmp_path: Path) -> None:
    skill = _call("read_skill", {"name": "files"})
    listing = _call("list_dir", {"path": "Documents"})
    result, model = _run(
        "list_folder",
        [skill, listing, _say("Here are the files.")],
        Variant("chat", "all"),
        tmp_path,
    )

    assert result.outcome == "right"
    assert any(
        "Working with the owner's files" in message["content"]
        for message in model.seen[1]
        if message["role"] == "tool"
    )
    assert (
        judge(
            next(case for case in load_cases(CASES_FILE) if case.id == "list_folder"),
            [Call("list_dir", {"path": "Documents"})],
        )
        == "no_tool"
    )


def test_the_shell_stub_preserves_refusal_after_file_text(tmp_path: Path) -> None:
    replies = [
        _call("list_dir", {"path": "Documents"}),
        _call("pc_shell", {"command": "uname -r"}),
        _say("I did not run it."),
    ]
    result, _ = _run("kernel_version", replies, Variant("chat", "all"), tmp_path)

    assert [call.tool for call in result.calls] == ["list_dir"]
    assert result.refused == ["pc_shell"]


def test_a_routed_call_to_a_tool_that_was_not_asked_for_is_refused_and_counted(
    tmp_path: Path,
) -> None:
    play = {"kind": "artist", "query": "Radiohead"}
    replies = [
        _call("spotify_play", play),  # no use_tools first: not offered, so it does not run
        _say("Sorry, I could not."),
    ]

    result, _ = _run("play_artist", replies, Variant("voice", "routed"), tmp_path)

    assert result.outcome == "no_tool"
    assert result.calls == []
    assert result.refused == ["spotify_play"]


def test_answering_without_the_needed_tool_is_counted(tmp_path: Path) -> None:
    result, _ = _run("play_artist", [_say("Sure!")], Variant("chat", "all"), tmp_path)

    assert result.outcome == "no_tool"
    assert result.calls == []
    assert result.turns[0].first_s is not None  # the chat's first draft of the reply


def test_a_failed_model_is_an_error_not_a_wrong_answer(tmp_path: Path) -> None:
    case = next(c for c in load_cases(CASES_FILE) if c.id == "play_artist")
    model = ScriptedModel([], error=RuntimeError("boom"))
    with router_bench.isolated(tmp_path), router_bench.offering_everything():
        session = router_bench.make_session(
            config.Config(),
            model,  # type: ignore[arg-type]
            Variant("voice", "all"),
            tmp_path,
            Clock(),
        )
        result = router_bench.run_case(case, session)

    assert result.outcome == "error"
    assert result.reason == "the model did not answer"


def test_a_conversation_keeps_its_turns_and_judges_them_together(tmp_path: Path) -> None:
    replies = [
        _call("schedule_add", {"text": "water the plants", "at": "21:00"}),
        _say("I will remind you at 9 pm."),
        _call("schedule_list"),
        _say("Nothing is left."),
    ]
    case = next(c for c in load_cases(CASES_FILE) if c.id == "remind_then_cancel")
    model = ScriptedModel(replies)
    with router_bench.isolated(tmp_path), router_bench.offering_everything():
        session = router_bench.make_session(
            config.Config(),
            model,  # type: ignore[arg-type]
            Variant("voice", "all"),
            tmp_path,
            Clock(),
        )
        result = router_bench.run_case(case, session)

    assert [t.text for t in result.turns] == list(case.turns)
    assert result.outcome == "no_tool"  # it never cancelled: schedule_cancel was missing
    assert [c.tool for c in result.calls] == ["schedule_add", "schedule_list"]


def test_a_variant_sends_a_warm_up_first_and_leaves_it_out() -> None:
    cases = [c for c in load_cases(CASES_FILE) if c.id in {"plain_knowledge", "say_joke"}]
    model = ScriptedModel([_say("366."), _say("Eight."), _say("Why did the chicken...")])
    lines: list[str] = []

    results = router_bench.run_variant(
        cases,
        config.Config(),
        model,  # type: ignore[arg-type]
        Variant("chat", "all"),
        clock=Clock(),
        progress=lines.append,
    )

    assert [r.id for r in results] == ["plain_knowledge", "say_joke"]
    assert [r.outcome for r in results] == ["right", "right"]
    assert [r.run for r in results] == [1, 1]
    assert len(lines) == 2
    assert lines[0].startswith("chat/all plain_knowledge: right")
    assert not model.replies


def test_a_case_leaves_nothing_of_the_environment_behind(tmp_path: Path) -> None:
    before = dict(os.environ)

    with router_bench.isolated(tmp_path / "one"):
        assert os.environ["HOME"].startswith(str(tmp_path))
        assert os.environ["XDG_STATE_HOME"].startswith(str(tmp_path))
        assert os.environ["PULSE_SERVER"] == "unix:/nonexistent"

    assert dict(os.environ) == before


def test_the_settings_of_a_variant_set_its_channel_only() -> None:
    base = config.Config()

    voice = router_bench.settings_for(base, Variant("voice", "routed", "low", 49152))
    chat = router_bench.settings_for(base, Variant("chat", "all"))

    assert (voice.agent.tools, voice.chat.tools) == ("routed", base.chat.tools)
    assert (chat.chat.tools, chat.agent.tools) == ("all", base.agent.tools)
    assert voice.agent.thinking == "low"
    assert voice.strata.context == 49152
    assert chat.strata.context == base.strata.context
    assert voice.shell.enabled
    assert voice.audio.listen_to_pc


# --- tables -------------------------------------------------------------------------------


def _result(
    case_id: str,
    outcome: Outcome,
    *,
    group: str = "music",
    kind: str = "typed",
    calls: tuple[ModelCall, ...] = (),
    first: float | None = 1.0,
    total: float = 2.0,
) -> CaseResult:
    turn = Turn("hi", "ok", first, total, list(calls))
    return CaseResult(case_id, group, kind, 1, outcome, "", [], [], [], [turn])


def test_a_summary_counts_outcomes_calls_times_and_prompt_tokens() -> None:
    results = [
        _result(
            "a", "right", calls=(ModelCall(50, 1.0, 8000), ModelCall(50, 1.0, 8100)), total=2.0
        ),
        _result(
            "b", "wrong_tool", group="web", calls=(ModelCall(50, 1.0, None),), first=None, total=4.0
        ),
        _result(
            "c", "no_tool", kind="spoken", calls=(ModelCall(50, 1.0, 7900),), first=3.0, total=6.0
        ),
    ]

    summary = router_bench.summarize("voice", "all", results)

    assert summary.cases == 3
    assert summary.outcomes == {
        "right": 1, "wrong_tool": 1, "wrong_args": 0, "no_tool": 1, "extra": 0, "error": 0
    }  # fmt: skip
    assert summary.requests == 3
    assert summary.calls_per_request == pytest.approx(4 / 3)
    assert (summary.first_median_s, summary.total_median_s) == (
        pytest.approx(2.0),
        pytest.approx(4.0),
    )
    assert summary.prompt_tokens_per_call == pytest.approx(8000)
    assert summary.prompt_tokens_known == 3
    assert summary.by_group == {"music": (1, 2), "web": (0, 1)}


def test_a_summary_of_requests_that_said_nothing_has_no_first_time() -> None:
    summary = router_bench.summarize("chat", "routed", [_result("a", "right", first=None)])

    assert summary.first_median_s is None
    assert summary.prompt_tokens_per_call is None


def _both() -> dict[tuple[str, str], list[CaseResult]]:
    return {
        ("voice", "all"): [
            _result("a", "right", calls=(ModelCall(50, 1, 8000),)),
            _result("b", "right", group="web", kind="spoken", calls=(ModelCall(50, 1, 8000),)),
        ],
        ("voice", "routed"): [
            _result("a", "right", calls=(ModelCall(6, 1, 1000), ModelCall(20, 1, 3000))),
            _result("b", "no_tool", group="web", kind="spoken", calls=(ModelCall(6, 1, 1000),)),
        ],
    }


def test_the_markdown_has_one_row_per_variant_one_column_per_measure_and_the_differences() -> None:
    both = _both()
    summaries = [router_bench.summarize(c, m, r) for (c, m), r in both.items()]
    report = router_bench.Report("Some-Model", "2026-11-26", 2, "none", 24576)

    text = router_bench.markdown(report, summaries, both)
    lines = text.splitlines()

    assert lines[0] == (
        "Router benchmark of Some-Model, 2026-11-26: 2 requests (1 typed, 1 spoken-style), 2 runs"
        " each, thinking `none`, context 24576. A case is right when it called what it needed and"
        " nothing else."
    )
    header, rule = lines[2], lines[3]
    assert header.count("|") == rule.count("|") == 13
    assert (
        "| voice | all | 2/2 (100%) | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.0 / 1.0 | 2.0 / 2.0 | 8,000 |"
        in lines
    )
    assert (
        "| voice | routed | 1/2 (50%) | 0 | 0 | 1 | 0 | 0 | 1.50 | 1.0 / 1.0 | 2.0 / 2.0 | 1,667 |"
        in lines
    )
    assert "| Group | voice all | voice routed |" in lines
    assert "| web | 1/1 (100%) | 0/1 (0%) |" in lines
    assert "- voice `b`: all 1/1, routed 0/1 (1 no_tool)" in lines
    assert not any("`a`:" in line for line in lines if line.startswith("- "))


def test_the_markdown_says_when_no_case_differs() -> None:
    only = {("chat", "all"): [_result("a", "right")], ("chat", "routed"): [_result("a", "right")]}
    summaries = [router_bench.summarize(c, m, r) for (c, m), r in only.items()]

    text = router_bench.markdown(router_bench.Report("m", "d", 1), summaries, only)

    assert text.splitlines()[-1] == "- none"
    assert "1 run each, thinking `as configured`." in text


def test_a_variant_is_written_as_json_the_report_can_be_read_back_from(tmp_path: Path) -> None:
    results = [_result("a", "right", calls=(ModelCall(50, 1, 8000),))]
    summary = router_bench.summarize("voice", "all", results)

    path = router_bench.write_results(
        tmp_path / "out", router_bench.Report("a/b model", "2026-11-26", 1), summary, results
    )

    assert path.name == "a-b-model-voice-all.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["report"]["label"] == "a/b model"
    assert data["summary"]["outcomes"]["right"] == 1
    assert data["cases"][0]["turns"][0]["model_calls"][0]["prompt_tokens"] == 8000


# --- the command --------------------------------------------------------------------------


def _answers(
    model: ChatModel, status: ModelStatus | None = None
) -> Callable[[str, str, str, float], tuple[ChatModel, ModelStatus]]:
    def connect(
        _url: str, _key: str, _model: str, _timeout: float
    ) -> tuple[ChatModel, ModelStatus]:
        return model, status or ModelStatus(
            reachable=True, latency_ms=1, state="loaded", name="Fake"
        )

    return connect


def test_main_runs_the_chosen_cases_writes_json_and_prints_the_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    keys = tmp_path / "key"
    write_private(keys, "STRATA_API_KEY=secret\n")
    model = ScriptedModel([_say("366."), _say("Eight."), _say("366."), _say("Eight.")])
    argv = [
        "--base-url",
        "http://127.0.0.1:1/v1",
        "--key-file",
        str(keys),
        "--out",
        str(tmp_path / "out"),
        "--only",
        "plain_knowledge",
        "--channel",
        "chat",
        "--runs",
        "1",
    ]

    status = router_bench.main(
        argv,
        connector=_answers(model),
        clock=Clock(),
        today=lambda: "2026-11-26",  # type: ignore[arg-type]
    )

    seen = capsys.readouterr()
    assert status == 0
    assert "Router benchmark of Fake, 2026-11-26: 1 requests (1 typed, 0 spoken-style)" in seen.out
    assert "| chat | all | 1/1 (100%) |" in seen.out
    assert "| chat | routed | 1/1 (100%) |" in seen.out
    assert "secret" not in seen.out + seen.err
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "Fake-chat-all.json",
        "Fake-chat-routed.json",
    ]


def test_main_exits_3_when_a_request_failed(tmp_path: Path) -> None:
    model = ScriptedModel([], error=RuntimeError("boom"))
    argv = [
        "--base-url",
        "http://127.0.0.1:1/v1",
        "--key",
        "k",
        "--out",
        str(tmp_path),
        "--only",
        "plain_knowledge",
        "--channel",
        "chat",
        "--tools",
        "all",
    ]

    assert (
        router_bench.main(argv, connector=_answers(model), clock=Clock())
        == router_bench.EXIT_ERRORS
    )  # type: ignore[arg-type]


def test_main_refuses_without_running_anything_when_the_server_is_not_there(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def down(_url: str, _key: str, _model: str, _timeout: float) -> tuple[ChatModel, ModelStatus]:
        msg = "no answer from the model server (unreachable)"
        raise UnavailableError(msg)

    argv = ["--base-url", "http://127.0.0.1:1/v1", "--key", "k", "--out", str(tmp_path / "out")]

    assert router_bench.main(argv, connector=down) == router_bench.EXIT_REFUSED
    assert "no answer from the model server" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_main_refuses_a_choice_that_matches_no_case(tmp_path: Path) -> None:
    argv = [
        "--base-url",
        "http://127.0.0.1:1/v1",
        "--key",
        "k",
        "--out",
        str(tmp_path / "out"),
        "--only",
        "no-such-case",
    ]

    status = router_bench.main(argv, connector=_answers(ScriptedModel([])))  # type: ignore[arg-type]

    assert status == router_bench.EXIT_REFUSED


def test_main_refuses_a_key_file_without_a_key(tmp_path: Path) -> None:
    keys = tmp_path / "key"
    write_private(keys, "OTHER=1\n")
    argv = ["--base-url", "http://127.0.0.1:1/v1", "--key-file", str(keys), "--out", "x"]

    assert (
        router_bench.main(argv, connector=_answers(ScriptedModel([]))) == router_bench.EXIT_REFUSED
    )


def test_main_refuses_an_unreadable_key_file(tmp_path: Path) -> None:
    argv = [
        "--base-url",
        "http://127.0.0.1:1/v1",
        "--key-file",
        str(tmp_path / "none"),
        "--out",
        "x",
    ]

    assert (
        router_bench.main(argv, connector=_answers(ScriptedModel([]))) == router_bench.EXIT_REFUSED
    )  # type: ignore[arg-type]


def _server(listing: object, status: int = 200) -> httpx.MockTransport:
    return httpx.MockTransport(lambda _request: httpx.Response(status, json=listing))


def test_connect_reads_the_model_list_and_nothing_else() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "jarvis", "status": {"value": "loaded"}}]})

    _, status = router_bench.connect(
        "http://127.0.0.1:1/v1/", "k", "jarvis", 5.0, httpx.MockTransport(handler)
    )

    assert status.reachable
    assert [(r.method, r.url.path) for r in seen] == [("GET", "/v1/models")]
    assert seen[0].headers["authorization"] == "Bearer k"


@pytest.mark.parametrize(
    ("transport", "message"),
    [
        (_server({}, 500), "no answer"),
        (_server({"data": [{"id": "jarvis", "status": {"value": "unloaded"}}]}), "not loaded"),
        (_server({"data": []}), "not loaded"),
    ],
)
def test_connect_refuses_a_server_that_is_down_or_has_no_model_loaded(
    transport: httpx.MockTransport, message: str
) -> None:
    with pytest.raises(UnavailableError, match=message):
        router_bench.connect("http://127.0.0.1:1/v1", "k", "jarvis", 5.0, transport)


def test_the_parser_wants_a_key_a_server_and_an_output_folder() -> None:
    parser = router_bench.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--base-url", "http://127.0.0.1:1/v1", "--out", "x"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--base-url", "u", "--key", "k", "--key-file", "f", "--out", "x"])
    assert parser.parse_args(["--base-url", "u", "--key", "k", "--out", "x"]).runs == 1


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _stream(*events: dict[str, Any]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def test_a_request_over_http_keeps_the_prompt_tokens_the_server_streams_back(
    tmp_path: Path,
) -> None:
    play = json.dumps({"kind": "artist", "query": "Radiohead"})
    bodies: list[dict[str, Any]] = []
    call = {"index": 0, "id": "c1", "function": {"name": "spotify_play", "arguments": play}}
    answers = [
        _stream(
            _chunk({"tool_calls": [call]}),
            _chunk({}, "tool_calls"),
            {"choices": [], "usage": {"prompt_tokens": 8000, "completion_tokens": 20}},
        ),
        _stream(
            _chunk({"content": "Playing Radiohead."}),
            _chunk({}, "stop"),
            {"choices": [], "usage": {"prompt_tokens": 8100, "completion_tokens": 6}},
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "m", "status": {"value": "loaded"}}]})
        bodies.append(json.loads(request.content))
        return answers.pop(0)

    model, _ = router_bench.connect(
        "http://127.0.0.1:1/v1", "k", "m", 5.0, httpx.MockTransport(handler)
    )
    case = next(c for c in load_cases(CASES_FILE) if c.id == "play_artist")
    with router_bench.isolated(tmp_path), router_bench.offering_everything():
        session = router_bench.make_session(
            config.Config(), model, Variant("voice", "all"), tmp_path, Clock()
        )
        result = router_bench.run_case(case, session)

    assert result.outcome == "right"
    assert [c.prompt_tokens for c in result.turns[0].model_calls] == [8000, 8100]
    assert bodies[0]["stream_options"] == {"include_usage": True}
    assert len(bodies[0]["tools"]) > 40
