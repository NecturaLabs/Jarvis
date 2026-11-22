"""tools/first_audio.py on a fake model and a fake clock: what it measures, and what it refuses."""

import ast
from pathlib import Path

import first_audio
import httpx
import pytest
from first_audio import Sample, UnavailableError
from test_agent import Clock
from test_first_audio_timing import (
    FINISHED_S,
    REPLY,
    SENTENCES,
    START,
    TimedModel,
    owl_reply,
)

from jarvis import config
from jarvis.agent import Agent
from jarvis.app import no_action_operation
from jarvis.audit import AuditLog
from jarvis.operations import Registry
from jarvis.paths import strata_key_file
from jarvis.secretfile import write_private

TOOL = Path(first_audio.__file__)


def _agent(model: TimedModel, clock: Clock) -> Agent:
    registry = Registry([no_action_operation()])
    return Agent(model, registry, "system", AuditLog(None), {}, clock=clock)


def test_a_request_is_timed_to_its_first_sentence_and_to_the_end_of_the_reply() -> None:
    clock = Clock()
    clock.now = START
    model = TimedModel(clock, [owl_reply()])

    sample = first_audio.time_request(_agent(model, clock), "owls", clock)

    assert sample.prompt == "owls"
    assert sample.first_s == pytest.approx(0.4)
    assert sample.done_s == pytest.approx(FINISHED_S)
    assert sample.sentences == len(SENTENCES)


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


def test_the_tool_imports_nothing_that_plays_sound_or_starts_a_service() -> None:
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    imported = {
        alias.name if isinstance(node, ast.Import) else f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    banned = ("audio", "kokoro", "voice", "services", "subprocess", "control", "activity")

    assert not [name for name in imported if any(part in banned for part in name.split("."))]
