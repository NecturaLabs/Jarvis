"""tools/bar_status.py: what the Omarchy bar shows for each `jarvis status --json` reading (#14).

The source is a fake that stages each state and each way the reading can go wrong; one test
runs the script itself against a stand-in `jarvis` on the PATH.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import bar_status
import pytest
from bar_status import FACES, HINT, StatusError, jarvis_status, render, show

from jarvis import control

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "bar_status.py"
DOC = ROOT / "docs" / "configuration.md"


def reading(state: str, **fields: object) -> str:
    """Return a `jarvis status --json` line the way `cmd_status` prints it."""
    return json.dumps({"state": state, "paused": state == "paused", "model": "loaded", **fields})


def fake(raw: str) -> Callable[[], str]:
    """Return a source that answers `raw`."""
    return lambda: raw


def failing(error: Exception) -> Callable[[], str]:
    """Return a source that fails with `error`."""

    def source() -> str:
        raise error

    return source


def test_every_public_state_has_a_face() -> None:
    assert set(FACES) == set(control.STATES)


@pytest.mark.parametrize(
    ("state", "active"),
    [
        ("idle", False),
        ("listening", True),
        ("thinking", True),
        ("speaking", True),
        ("paused", False),
        ("model loading", True),
        ("unavailable", False),
    ],
)
def test_each_state_shows_its_own_glyph_class_and_tooltip(state: str, active: bool) -> None:
    shown = show(fake(reading(state, running=True)))
    assert shown.text == FACES[state].glyph
    assert shown.classes[0] == state.replace(" ", "-")
    assert ("active" in shown.classes) is active
    assert shown.tooltip.startswith(f"Jarvis: {FACES[state].says}")
    assert "Model: loaded" in shown.tooltip
    assert shown.tooltip.endswith(HINT)


def test_the_states_look_different_from_one_another() -> None:
    glyphs = {face.glyph for face in FACES.values()}
    assert len(glyphs) == len(FACES)
    assert len(
        {show(fake(reading(state, running=True))).tooltip for state in control.STATES}
    ) == len(control.STATES)


def test_a_jarvis_that_is_not_running_is_told_apart_from_an_unavailable_one() -> None:
    stopped = show(fake(reading("unavailable", running=False, model="unloaded")))
    broken = show(fake(reading("unavailable", running=True, model="unreachable")))
    assert stopped.classes == ("not-running",)
    assert "not running" in stopped.tooltip
    assert "Model: unloaded" in stopped.tooltip
    assert broken.classes == ("unavailable",)
    assert "cannot work" in broken.tooltip
    assert stopped.text != broken.text


def test_a_reading_without_running_counts_as_running() -> None:
    assert show(fake(reading("idle"))).classes == ("idle",)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        '{"state": "idle"',
        "[]",
        "null",
        '"idle"',
        "{}",
        '{"paused": false}',
        '{"state": "dancing"}',
        '{"state": 3}',
        '{"state": null}',
        '{"state": ["idle"]}',
        '{"state": "idle"}' + " " * bar_status.MAX_READING_CHARS,
    ],
)
def test_a_reading_that_is_not_a_status_shows_malformed(raw: str) -> None:
    shown = show(fake(raw))
    assert shown.classes == ("malformed",)
    assert "unreadable" in shown.tooltip
    assert shown.text


def test_a_source_that_times_out_shows_timeout() -> None:
    shown = show(failing(TimeoutError("took too long")))
    assert shown.classes == ("timeout",)
    assert "did not answer" in shown.tooltip


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("jarvis"), PermissionError("jarvis"), OSError("boom"), StatusError("2")],
)
def test_a_missing_or_failing_command_shows_failed(error: Exception) -> None:
    shown = show(failing(error))
    assert shown.classes == ("failed",)
    assert "failed" in shown.tooltip


@pytest.mark.parametrize("error", [TimeoutError, FileNotFoundError, StatusError])
def test_the_error_text_never_reaches_the_bar(error: type[Exception]) -> None:
    assert "SECRET-DETAIL" not in show(failing(error("SECRET-DETAIL"))).tooltip


@pytest.mark.parametrize(
    "model", ["<img src=https://example.invalid/t>", "loaded\nAPI key: x", "x" * 80, "", 7, None]
)
def test_a_model_state_that_is_not_a_plain_word_is_not_shown(model: object) -> None:
    shown = show(fake(reading("idle", running=True, model=model)))
    assert "Model:" not in shown.tooltip
    assert "<" not in shown.tooltip


@pytest.mark.parametrize(
    "model", ["loaded", "unloaded", "loading", "failed", "unreachable", "missing"]
)
def test_the_model_states_jarvis_reports_are_shown(model: str) -> None:
    assert f"Model: {model}" in show(fake(reading("idle", running=True, model=model))).tooltip


def test_extra_fields_in_the_reading_are_ignored() -> None:
    shown = show(fake(reading("idle", running=True, error="<b>x</b>", surprise={"a": 1})))
    assert "<b>" not in shown.tooltip
    assert shown.classes == ("idle",)


def test_the_output_is_one_json_line_with_text_tooltip_and_class() -> None:
    line = show(fake(reading("thinking", running=True))).as_json()
    assert "\n" not in line
    assert json.loads(line) == {
        "text": FACES["thinking"].glyph,
        "tooltip": render(reading("thinking", running=True)).tooltip,
        "class": ["thinking", "active"],
    }


def test_main_prints_one_line_and_exits_zero_whatever_happens(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for source in (fake(reading("idle")), failing(TimeoutError()), fake("garbage")):
        monkeypatch.setattr(bar_status, "jarvis_status", source)
        assert bar_status.main() == 0
    lines = capsys.readouterr().out.splitlines()
    assert [json.loads(line)["class"][0] for line in lines] == ["idle", "timeout", "malformed"]


def found(_name: str) -> str:
    """Stand in for `shutil.which` when `jarvis` is installed."""
    return "/opt/bin/jarvis"


def missing(_name: str) -> None:
    """Stand in for `shutil.which` when `jarvis` is not installed."""


class FakeRun:
    """A stand-in for `subprocess.run` that finishes with `code` and `out` and keeps its calls."""

    def __init__(self, code: int, out: str = "") -> None:
        super().__init__()
        self.code = code
        self.out = out
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # ruff: ignore[any-type]
        self.calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, self.code, stdout=self.out, stderr="")


def test_jarvis_status_runs_only_the_status_command_with_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bar_status.shutil, "which", found)
    run = FakeRun(0, reading("idle"))
    assert jarvis_status(run, timeout=3.0) == reading("idle")
    ((command, kwargs),) = run.calls
    assert command == ["/opt/bin/jarvis", "status", "--json"]
    assert kwargs["timeout"] == 3
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert "shell" not in kwargs


def test_jarvis_status_raises_when_the_command_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bar_status.shutil, "which", missing)
    with pytest.raises(FileNotFoundError):
        jarvis_status(FakeRun(0))


def test_jarvis_status_raises_when_the_command_exits_with_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bar_status.shutil, "which", found)
    with pytest.raises(StatusError):
        jarvis_status(FakeRun(2, "jarvis: bad config"))


def test_jarvis_status_turns_a_subprocess_timeout_into_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bar_status.shutil, "which", found)

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # ruff: ignore[any-type]
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    with pytest.raises(TimeoutError):
        jarvis_status(run, timeout=1.0)


def stand_in(folder: Path, body: str) -> dict[str, str]:
    """Write an executable `jarvis` running `body` and return an environment that finds only it."""
    program = folder / "jarvis"
    program.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    program.chmod(0o755)
    return {**os.environ, "PATH": f"{folder}{os.pathsep}{os.environ['PATH']}"}


def run_script(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the script by path, as the bar does."""
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
        check=False,
    )


def test_the_script_runs_end_to_end_against_a_stand_in_jarvis(tmp_path: Path) -> None:
    env = stand_in(tmp_path, f"echo '{reading('speaking', running=True)}'")
    done = run_script(env)
    assert done.returncode == 0
    assert json.loads(done.stdout)["class"] == ["speaking", "active"]


def test_the_script_shows_a_failed_state_when_jarvis_exits_with_an_error(tmp_path: Path) -> None:
    done = run_script(stand_in(tmp_path, "echo 'jarvis: bad config' >&2; exit 2"))
    assert done.returncode == 0
    assert json.loads(done.stdout)["class"] == ["failed"]


def test_a_stuck_jarvis_is_stopped_at_the_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pidfile = tmp_path / "pid"
    monkeypatch.setenv("PATH", stand_in(tmp_path, f"echo $$ > {pidfile}; exec sleep 30")["PATH"])
    with pytest.raises(TimeoutError):
        jarvis_status(timeout=0.5)
    pid = int(pidfile.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # signal 0 only checks that the process is gone


def test_the_script_is_executable_and_stdlib_only() -> None:
    assert os.access(SCRIPT, os.X_OK)
    imports = set(
        re.findall(r"^(?:from|import) (\w+)", SCRIPT.read_text(encoding="utf-8"), re.MULTILINE)
    )
    assert imports <= set(sys.stdlib_module_names)


def documented_entry() -> dict[str, Any]:
    """Return the bar entry the docs tell the owner to add."""
    text = DOC.read_text(encoding="utf-8")
    section = text[text.index("## The Omarchy bar module") :]
    match = re.search(r"^ *```json\n(.*?)\n *```", section, re.DOTALL | re.MULTILINE)
    assert match, "the bar section has no JSON entry"
    entry: dict[str, Any] = json.loads(match.group(1))
    return entry


def test_the_documented_entry_offers_only_the_fixed_pause_and_resume() -> None:
    entry = documented_entry()
    clicks = {key: value for key, value in entry.items() if key.startswith("on")}
    assert clicks == {"onClick": "jarvis pause", "onRightClick": "jarvis resume"}


def test_the_documented_entry_runs_this_script_on_a_short_interval() -> None:
    entry = documented_entry()
    assert entry["type"] == "command"
    assert entry["exec"].endswith("jarvis-status")  # the link the docs have the owner make
    assert 3 <= entry["interval"] <= 10
    assert set(entry) <= {"id", "type", "exec", "interval", "tooltip", "onClick", "onRightClick"}


def test_the_documented_timeout_is_the_scripts() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert f"did not answer in {bar_status.TIMEOUT_S:.0f} seconds" in text
