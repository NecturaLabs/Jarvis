"""tools/first_audio.py on a fake model and a fake clock: what it measures, and what it refuses."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import first_audio
import httpx
import pytest
from first_audio import Sample, UnavailableError
from test_agent import Clock, ScriptedModel
from test_first_audio_timing import (
    FINISHED_S,
    REPLY,
    SENTENCES,
    START,
    TimedModel,
    owl_reply,
)

from jarvis import agent as agent_module
from jarvis import config
from jarvis.agent import Agent
from jarvis.app import no_action_operation
from jarvis.audit import AuditLog
from jarvis.model import ModelError
from jarvis.operations import Registry
from jarvis.paths import strata_key_file
from jarvis.secretfile import write_private

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.model import Delta

TOOL = Path(first_audio.__file__)


def _agent(model: TimedModel, clock: Clock, audit: AuditLog) -> Agent:
    registry = Registry([no_action_operation()])
    return Agent(model, registry, "system", audit, {}, clock=clock)


def test_a_request_is_timed_to_its_first_sentence_and_to_the_end_of_the_reply() -> None:
    clock = Clock()
    clock.now = START
    model = TimedModel(clock, [owl_reply()])

    audit = AuditLog(None)

    sample = first_audio.time_request(_agent(model, clock, audit), "owls", clock, audit)

    assert sample.prompt == "owls"
    assert sample.first_s == pytest.approx(0.4)
    assert sample.done_s == pytest.approx(FINISHED_S)
    assert sample.sentences == len(SENTENCES)
    assert sample.failed is None


def test_measure_sends_every_prompt_each_round_with_no_tool_but_no_action() -> None:
    clock = Clock()
    clock.now = START
    model = TimedModel(clock, [owl_reply() for _ in range(4)])

    samples = first_audio.measure(model, config.Config(), ["one", "two"], 2, clock)

    assert [s.prompt for s in samples] == ["one", "two", "one", "two"]
    assert model.offered == [["no_action"]] * 4


@pytest.mark.parametrize(("fraction", "expected"), [(0.5, 3.0), (0.9, 5.0), (1.0, 5.0), (0.0, 1.0)])
def test_percentile_takes_the_nearest_rank(fraction: float, expected: float) -> None:
    assert first_audio.percentile([5.0, 1.0, 3.0, 2.0, 4.0], fraction) == expected


def test_the_report_shows_each_request_and_the_time_saved() -> None:
    samples = [Sample("a", 0.4, 3.0, 4), Sample("b", 0.6, 2.0, 2), Sample("c", None, 1.0, 0)]

    text = first_audio.report(samples)

    assert "0.40     3.00     2.60          4  a" in text
    assert "first sentence  median   0.50 s" in text
    assert "saved           median   2.00 s" in text
    assert "1 request(s) spoke nothing" in text


def test_main_prints_the_measurement(capsys: pytest.CaptureFixture[str]) -> None:
    clock = Clock()
    clock.now = START
    model = TimedModel(clock, [owl_reply(), owl_reply()])

    code = first_audio.main(
        ["--runs", "2", "--prompt", "tell me about owls"],
        connector=lambda _cfg: model,
        clock=clock,
    )

    out = capsys.readouterr().out
    assert code == 0
    assert out.count("tell me about owls") == 2
    assert "saved           median   2.60 s" in out
    assert REPLY not in out  # the words are timed, not echoed


def test_main_refuses_when_the_model_cannot_be_reached(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(_cfg: config.Config) -> TimedModel:
        msg = "no answer from the model server"
        raise UnavailableError(msg)

    assert first_audio.main([], connector=refuse) == 2
    assert "no answer from the model server" in capsys.readouterr().err


def test_main_rejects_zero_runs() -> None:
    with pytest.raises(SystemExit, match="2"):
        first_audio.main(["--runs", "0"])


def _server(state: str | None, seen: list[httpx.Request]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if state is None:
            msg = "refused"
            raise httpx.ConnectError(msg)
        return httpx.Response(200, json={"data": [{"status": {"value": state}}]})

    return httpx.MockTransport(handle)


def _key() -> None:
    write_private(strata_key_file(), "STRATA_API_KEY=test-key\n")


def test_connect_without_a_key_refuses() -> None:
    with pytest.raises(UnavailableError, match="no Strata API key"):
        first_audio.connect(config.Config(), _server("loaded", []))


def test_connect_to_a_server_that_is_down_refuses() -> None:
    _key()
    seen: list[httpx.Request] = []

    with pytest.raises(UnavailableError, match="no answer"):
        first_audio.connect(config.Config(), _server(None, seen))


def test_connect_to_an_unloaded_model_refuses_and_loads_nothing() -> None:
    _key()
    seen: list[httpx.Request] = []

    with pytest.raises(UnavailableError, match="unloaded"):
        first_audio.connect(config.Config(), _server("unloaded", seen))

    assert [(r.method, r.url.path) for r in seen] == [("GET", "/v1/models")]


def test_connect_to_a_loaded_model_only_reads_its_status() -> None:
    _key()
    seen: list[httpx.Request] = []

    model = first_audio.connect(config.Config(), _server("loaded", seen))

    assert model.is_loaded() is True
    assert {r.method for r in seen} == {"GET"}


def test_importing_the_tool_loads_no_audio_voice_or_subprocess_module() -> None:
    """A fresh interpreter that imports only the tool: what ends up in `sys.modules`."""
    banned = [
        "subprocess",
        "jarvis.audio",
        "jarvis.voice",
        "jarvis.kokoro",
        "jarvis.models",
        "jarvis.phonemes",
        "jarvis.activity",
        "jarvis.control",
        "jarvis.azure_wake_helper",
        "jarvis.cli",
        "jarvis.chat",
        "sounddevice",
        "livekit",
        "piper",
        "onnxruntime",
    ]
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); import first_audio; "
        "banned = json.loads(sys.argv[2]); "
        "print(json.dumps(sorted(m for m in sys.modules "
        "if any(m == b or m.startswith(b + '.') for b in banned))))"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code, str(TOOL.parent), json.dumps(banned)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env={**os.environ, "PIPEWIRE_REMOTE": "/nonexistent", "PULSE_SERVER": "unix:/nonexistent"},
    )

    assert json.loads(done.stdout) == []


def _unknown_call(number: int) -> dict[str, Any]:
    """Return a model message that calls a tool the tool's agent does not have, each time anew."""
    function = {"name": "search", "arguments": json.dumps({"q": str(number)})}
    call = {"id": f"c{number}", "type": "function", "function": function}
    return {"role": "assistant", "content": "", "tool_calls": [call]}


@dataclass
class GoodThenBad:
    """A model that writes one good owl reply, then behaves as `bad` says."""

    clock: Clock
    bad: ScriptedModel
    good: TimedModel = field(init=False)

    def __post_init__(self) -> None:
        self.good = TimedModel(self.clock, [owl_reply()])

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        require_tool: bool = False,
        timeout_s: float | None = None,
        on_delta: Callable[[Delta], None] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        model = self.good if self.good.calls else self.bad
        return model.chat(
            messages,
            tools,
            require_tool=require_tool,
            timeout_s=timeout_s,
            on_delta=on_delta,
            max_tokens=max_tokens,
        )

    def is_loaded(self) -> bool | None:  # ruff: ignore[no-self-use] - the ChatModel protocol
        return True


def _run_main(model: GoodThenBad, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = first_audio.main(
        ["--runs", "1", "--prompt", "tell me about owls", "--prompt", "and tides"],
        connector=lambda _cfg: model,
        clock=model.clock,
    )
    return code, capsys.readouterr().out


def _failing_cases() -> dict[str, GoodThenBad]:
    clock = Clock()
    clock.now = START

    def slow() -> None:
        clock.now += agent_module.MAX_SECONDS + 1

    return {
        "model error": GoodThenBad(clock, ScriptedModel([], error=ModelError("down"))),
        "step cap": GoodThenBad(
            clock, ScriptedModel([_unknown_call(i) for i in range(agent_module.MAX_STEPS + 1)])
        ),
        "time cap": GoodThenBad(clock, ScriptedModel([_unknown_call(0)] * 2, during=slow)),
        "reply cut": GoodThenBad(
            clock,
            ScriptedModel(
                [{"role": "assistant", "content": REPLY, "finish_reason": "length"}],
            ),
        ),
    }


@pytest.mark.parametrize("case", ["model error", "step cap", "time cap", "reply cut"])
def test_a_request_that_failed_is_not_counted_and_the_run_exits_3(
    case: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _run_main(_failing_cases()[case], capsys)

    assert code == first_audio.EXIT_FAILED == 3
    (bad,) = [line for line in out.splitlines() if "and tides" in line]
    (good,) = [line for line in out.splitlines() if "tell me about owls" in line]
    assert "failed" in bad
    assert "failed" not in good
    # Only the good request is in the medians: its first sentence at 0.4 s, not the apology's.
    assert "first sentence  median   0.40 s" in out
    assert "1 request(s) failed" in out


def test_a_sample_says_why_it_failed() -> None:
    reasons: dict[str, str | None] = {}
    for case, model in _failing_cases().items():
        agent = Agent(
            model,
            Registry([no_action_operation()]),
            "system",
            audit := AuditLog(None),
            {},
            clock=model.clock,
        )
        model.clock.now = START
        model.good.calls[:] = []
        reasons[case] = first_audio.time_request(agent, "x", model.clock, audit).failed

    assert all(reasons.values())
    assert len(set(reasons.values())) == len(reasons)


def test_a_run_where_every_request_failed_prints_no_medians(
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = Clock()
    clock.now = START
    model = GoodThenBad(clock, ScriptedModel([], error=ModelError("down")))
    model.good.calls.clear()

    code, out = _run_main(model, capsys)

    assert code == 3
    assert "median" not in out
    assert "2 request(s) failed" in out


def test_a_run_with_no_failure_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    clock = Clock()
    clock.now = START
    model = TimedModel(clock, [owl_reply()])

    code = first_audio.main(
        ["--runs", "1", "--prompt", "owls"], connector=lambda _c: model, clock=clock
    )

    assert code == 0
    assert "failed" not in capsys.readouterr().out


def test_the_report_marks_a_failed_row_and_keeps_it_out_of_the_medians() -> None:
    samples = [
        Sample("a", 0.4, 3.0, 4),
        Sample("b", 0.1, 0.1, 1, failed="the model did not answer"),
    ]

    text = first_audio.report(samples)

    assert "failed (the model did not answer)" in text
    assert "first sentence  median   0.40 s" in text
    assert "1 request(s) failed" in text


def test_the_report_says_the_absolute_times_are_lower_than_in_real_use() -> None:
    text = first_audio.report([Sample("a", 0.4, 3.0, 4)])

    assert "only no_action" in text
    assert "2,900" in text
    assert "only `saved` carries over" in text


class LoadedThenUnloaded(TimedModel):
    """A model that is loaded for `loaded_checks` status reads, then is not."""

    def __init__(self, clock: Clock, calls: int, loaded_checks: int, state: bool | None) -> None:
        super().__init__(clock, [owl_reply() for _ in range(calls)])
        self.checks = 0
        self.loaded_checks = loaded_checks
        self.state = state

    @override
    def is_loaded(self) -> bool | None:
        self.checks += 1
        return True if self.checks <= self.loaded_checks else self.state


@pytest.mark.parametrize("state", [False, None])
def test_a_model_unloaded_between_samples_stops_the_run_before_the_next_request(
    state: bool | None,
) -> None:
    clock = Clock()
    clock.now = START
    model = LoadedThenUnloaded(clock, calls=4, loaded_checks=1, state=state)

    with pytest.raises(UnavailableError, match="no longer loaded"):
        first_audio.measure(model, config.Config(), ["one", "two"], 2, clock)

    assert len(model.ended) == 1  # one request was sent; none after the model went away


def test_main_refuses_with_exit_2_when_the_model_is_unloaded_mid_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = Clock()
    clock.now = START
    model = LoadedThenUnloaded(clock, calls=4, loaded_checks=0, state=False)

    code = first_audio.main(["--runs", "2"], connector=lambda _cfg: model, clock=clock)

    assert code == first_audio.EXIT_REFUSED == 2
    assert model.ended == []
    assert "no longer loaded" in capsys.readouterr().err


def test_a_key_file_that_is_not_private_refuses_without_showing_the_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _key()
    strata_key_file().chmod(0o644)
    seen: list[httpx.Request] = []

    with pytest.raises(UnavailableError) as raised:
        first_audio.connect(config.Config(), _server("loaded", seen))

    assert "chmod 600" in str(raised.value)
    assert "test-key" not in str(raised.value)
    assert seen == []
    code = first_audio.main(
        [], connector=lambda cfg: first_audio.connect(cfg, _server("loaded", seen))
    )
    assert code == 2
    assert "test-key" not in capsys.readouterr().err
