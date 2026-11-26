"""`jarvis.cliout`: one look for command lines, colour only on a terminal, JSON for agents."""

import io
import json
import os
from typing import override

import pytest

from jarvis import cliout


class Terminal(io.StringIO):
    """A stream that says it is a terminal."""

    @override
    def isatty(self) -> bool:
        return True


def test_a_check_line_puts_its_status_in_one_column_and_its_fix_after_an_arrow() -> None:
    plain = io.StringIO()

    assert cliout.check("config", "run jarvis setup", ok=True, stream=plain) == "ok   config"
    assert cliout.check("config", "run jarvis setup", ok=False, stream=plain) == (
        "FAIL config  -> run jarvis setup"
    )


@pytest.mark.parametrize("text", ["note Spotify is off", "note: Spotify is off", "Spotify is off"])
def test_a_note_is_written_one_way_whatever_it_started_as(text: str) -> None:
    assert cliout.note(text, io.StringIO()) == "note Spotify is off"


def test_colour_is_used_only_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    shown = cliout.check("config", "run jarvis setup", ok=False, stream=Terminal())

    assert "\x1b[" in shown
    assert cliout.strip(shown) == "FAIL config  -> run jarvis setup"
    assert "\x1b[" not in cliout.check("config", "run jarvis setup", ok=False, stream=io.StringIO())


@pytest.mark.parametrize(("name", "value"), [("NO_COLOR", "1"), ("TERM", "dumb")])
def test_no_color_and_a_dumb_terminal_turn_colour_off(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv(name, value)

    assert "\x1b[" not in cliout.check("config", ok=True, stream=Terminal())


def test_json_is_one_line_with_its_words_as_written() -> None:
    out = io.StringIO()

    cliout.emit({"what": "Spotify\N{RIGHT SINGLE QUOTATION MARK}s login", "ok": True}, out)

    text = out.getvalue()
    assert text.count("\n") == 1
    assert json.loads(text) == {"what": "Spotify\N{RIGHT SINGLE QUOTATION MARK}s login", "ok": True}
    assert "\N{RIGHT SINGLE QUOTATION MARK}" in text  # not escaped, so a person can read it too


def test_a_summary_names_what_failed() -> None:
    checks = [(True, "config", ""), (False, "Strata server (unreachable)", "jarvis start")]

    assert cliout.summary(checks, extra=3) == "5 checks, 1 failed: Strata server (unreachable)"
    assert cliout.summary(checks[:1]) == "1 check, none failed"


def test_captured_holds_what_a_child_program_writes_to_the_file_descriptors() -> None:
    out, err = io.StringIO(), io.StringIO()

    with cliout.captured(out, err):
        print("from python")
        os.write(1, b"from a child\n")
        os.write(2, b"a child's error\n")

    assert out.getvalue().splitlines() == ["from python", "from a child"]
    assert err.getvalue() == "a child's error\n"
