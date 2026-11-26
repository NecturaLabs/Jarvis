"""`jarvis --agent`: every command answers an agent with one line of JSON; people get text."""

import json
from typing import Any

import pytest

from jarvis import cli, cliout, config
from jarvis.cli import main


@pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse] - every test starts as a person would
def _human_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(cliout.AGENT_ENV, raising=False)
    monkeypatch.setattr(cli, "_restart_if_running", lambda: None)
    monkeypatch.setattr(cli, "_user_memlock_limit", lambda: "")
    config.save(config.Config())


def _answer(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    out = capsys.readouterr().out
    assert out.count("\n") == 1, out  # one line, nothing else around it
    return json.loads(out)


@pytest.mark.parametrize(
    "argv",
    [["--agent", "speed"], ["speed", "--agent"], ["speed", "--agent-format"]],
    ids=["before", "after", "long-name"],
)
def test_the_agent_flag_works_before_or_after_the_command(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 0

    answer = _answer(capsys)
    assert answer == {"command": "speed", "ok": True, "exit": 0, "speech_speed": 1.0}


def test_the_environment_asks_for_the_agent_format_once_for_every_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(cliout.AGENT_ENV, "1")

    assert main(["consent"]) == 0

    answer = _answer(capsys)
    assert answer["command"] == "consent"
    assert answer["consent"] in {"bypassed", "asks"}
    assert answer["said"]


def test_people_get_text_without_the_flag(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["speed"]) == 0

    assert capsys.readouterr().out == "speech speed: 1\n"


def test_tools_give_an_agent_every_tool_with_its_whole_description(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["tools", "--agent"]) == 0

    tools = {tool["name"]: tool for tool in _answer(capsys)["tools"]}
    calculate = tools["calculate"]
    assert set(calculate) == {"name", "setting", "state", "shipped", "changed", "description"}
    assert calculate["changed"] is False
    assert not any(tool["description"].endswith("…") for tool in tools.values())


def test_a_command_without_its_own_fields_still_answers_in_one_json_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["pc-audio", "--agent"]) == 0

    answer = _answer(capsys)
    assert answer["command"] == "pc-audio"
    assert answer["ok"] is True
    assert answer["output"][0].startswith("pc audio: ")
    assert answer["errors"] == []


def test_a_failed_command_says_so_and_carries_its_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["tools", "no_such_tool", "on", "--agent"]) == 2

    answer = _answer(capsys)
    assert answer["ok"] is False
    assert answer["exit"] == 2
    assert any("no_such_tool" in line for line in answer["errors"])


@pytest.mark.parametrize("command", ["chat", "logs", "run", "spotify-login"])
def test_an_interactive_command_refuses_the_agent_format(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command, "--agent"]) == 2

    answer = _answer(capsys)
    assert answer["ok"] is False
    assert "interactive" in answer["errors"][0]


@pytest.mark.parametrize(
    ("argv", "code"),
    [(["tools", "--agent", "--no-such-flag"], 2), (["--agent", "--help"], 0)],
    ids=["refused", "help"],
)
def test_even_a_command_line_argparse_answers_itself_comes_back_as_one_json_line(
    argv: list[str], code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == code

    answer = _answer(capsys)
    assert answer["ok"] is (code == 0)
    assert answer["exit"] == code
    assert answer["output"] if code == 0 else any("--no-such-flag" in e for e in answer["errors"])


def test_a_command_that_crashes_still_answers_with_its_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def crash(_args: object) -> int:
        message = "the config could not be read"
        raise RuntimeError(message)

    monkeypatch.setitem(cli.COMMANDS, "pc-audio", crash)

    assert main(["pc-audio", "--agent"]) == 1

    answer = _answer(capsys)
    assert answer["ok"] is False
    assert any("RuntimeError: the config could not be read" in line for line in answer["errors"])
