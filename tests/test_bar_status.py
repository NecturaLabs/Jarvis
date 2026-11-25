"""tools/bar_status.py: what the Omarchy bar shows for each reading of Jarvis's status (#14).

`render` and `show` get a fake source that stages each state and each way a reading can go wrong;
`control_status` talks to a real `jarvis.control.ControlServer` and to raw sockets that misbehave;
the script itself runs by path, as the bar runs it. The script repeats the socket's path and request
instead of importing `jarvis` (that would cost 0.2 s of CPU a poll), so the tests here compare both
with `jarvis.control`'s own.
"""

import contextlib
import json
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import bar_status
import pytest
from bar_status import (
    ACTIVE,
    FACES,
    HINT,
    NOT_RUNNING,
    REQUEST,
    control_status,
    render,
    show,
    socket_path,
)

from jarvis import app, cli, control

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "bar_status.py"
DOC = ROOT / "docs" / "configuration.md"


def reading(state: str, **fields: object) -> str:
    """Return a status reading the way the control socket sends it."""
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
    stopped = show(fake(NOT_RUNNING))
    broken = show(fake(reading("unavailable", running=True, model="unreachable")))
    assert stopped.classes == ("not-running",)
    assert "not running" in stopped.tooltip
    assert "Model:" not in stopped.tooltip  # the model server is not asked while Jarvis is down
    assert "Left click" not in stopped.tooltip  # nothing running to pause or resume
    assert HINT not in stopped.tooltip
    assert HINT in broken.tooltip
    assert broken.classes == ("unavailable",)
    assert "cannot work" in broken.tooltip
    assert stopped.text != broken.text


def test_a_reading_without_running_counts_as_running() -> None:
    assert show(fake(reading("idle"))).classes == ("idle",)


@pytest.mark.parametrize("reply", ['{"ok": false, "error": "boom <b>"}', '{"ok": false}'])
def test_a_jarvis_that_refuses_or_fails_the_request_shows_unavailable(reply: str) -> None:
    shown = show(fake(reply))
    assert shown.classes == ("unavailable",)
    assert "cannot work" in shown.tooltip
    assert "boom" not in shown.tooltip


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
    "error", [ConnectionResetError("reset"), BrokenPipeError("pipe"), OSError("boom")]
)
def test_a_connection_that_breaks_shows_failed(error: Exception) -> None:
    shown = show(failing(error))
    assert shown.classes == ("failed",)
    assert "failed" in shown.tooltip


@pytest.mark.parametrize("error", [TimeoutError, ConnectionResetError, OSError])
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
    sources = (fake(reading("idle")), failing(TimeoutError()), failing(OSError()), fake("junk"))
    for source in sources:
        monkeypatch.setattr(bar_status, "control_status", source)
        assert bar_status.main() == 0
    lines = capsys.readouterr().out.splitlines()
    classes = [json.loads(line)["class"][0] for line in lines]
    assert classes == ["idle", "timeout", "failed", "malformed"]


@contextlib.contextmanager
def raw_server(behave: Callable[[socket.socket], None]) -> Generator[Path]:
    """Serve the control socket's path, giving each connection to `behave`, until the block ends."""
    path = control.socket_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(4)

    def accept() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return  # closed
            # The client hanging up ends the connection: a client that timed out closes while
            # `behave` may still be writing, and that is the case under test, not a server fault.
            with conn, contextlib.suppress(BrokenPipeError, ConnectionResetError):
                behave(conn)

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    try:
        yield path
    finally:
        with contextlib.suppress(OSError):
            server.shutdown(socket.SHUT_RDWR)  # wakes the accept that close alone would not
        server.close()
        thread.join(5)


def read_line(conn: socket.socket) -> bytes:
    """Return what the client sent up to its newline."""
    data = b""
    while b"\n" not in data:
        chunk = conn.recv(1024)
        if not chunk:
            break
        data += chunk
    return data


def answering(reply: bytes) -> contextlib.AbstractContextManager[Path]:
    """Serve `reply` to each request."""

    def behave(conn: socket.socket) -> None:
        read_line(conn)
        conn.sendall(reply)

    return raw_server(behave)


@dataclass
class _Loop:
    state: str

    def hush(self) -> None:
        pass


@dataclass
class _Mic:
    paused: bool

    def pause(self) -> bool:
        changed, self.paused = not self.paused, True
        return changed

    def resume(self) -> bool:
        changed, self.paused = self.paused, False
        return changed


@dataclass
class _Model:
    state: str | None
    unloads: int = 0
    warms: int = 0
    missing: bool = False

    def status(self) -> str | None:
        return self.state

    def file_missing(self) -> bool:
        return self.missing

    def unload(self) -> bool:
        self.unloads += 1
        return True

    def warm(self, *, keep_going: Callable[[], bool] = lambda: True) -> bool:  # ruff: ignore[unused-method-argument] - the fake never retries
        self.warms += 1
        return True

    def is_loaded(self) -> bool | None:
        return None if self.state is None else self.state == "loaded"

    def load(self) -> bool:
        self.warms += 1
        return True


LOOP_STATES = ("idle", "capture", "follow-up", "confirm", "transcribe", "think", "speak", "no-mic")


@pytest.mark.parametrize("loop", LOOP_STATES)
@pytest.mark.parametrize("model", ["loaded", "loading", "failed", None])
@pytest.mark.parametrize("paused", [False, True])
def test_the_bar_shows_the_state_the_real_control_server_reports(
    loop: str, model: str | None, *, paused: bool
) -> None:
    controls = control.Controls(_Loop(loop), _Mic(paused=paused), _Model(model))
    server = control.ControlServer(controls)
    server.start()
    try:
        shown = show(lambda: control_status(str(server.path)))
    finally:
        server.close()
    expected = control.public_state(loop, paused=paused, model=model)
    assert shown.classes[0] == expected.replace(" ", "-")
    assert shown.text == FACES[expected].glyph
    if model is not None:
        assert f"Model: {model}" in shown.tooltip


# What the script repeats from jarvis/control.py, so the script need not import `jarvis`.


@pytest.mark.parametrize("runtime", [None, "", "relative/dir", "/run/user/4242", "/srv/run/"])
def test_the_scripts_socket_path_is_the_control_sockets(
    monkeypatch: pytest.MonkeyPatch, runtime: str | None
) -> None:
    if runtime is None:
        monkeypatch.delenv("XDG_RUNTIME_DIR")
    else:
        monkeypatch.setenv("XDG_RUNTIME_DIR", runtime)
    assert Path(socket_path()) == control.socket_path()


@pytest.mark.parametrize("uid", [0, 1000, 4242])
def test_without_a_runtime_folder_the_socket_is_under_run_user_for_the_uid(
    monkeypatch: pytest.MonkeyPatch, uid: int
) -> None:
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    monkeypatch.setattr(os, "getuid", lambda: uid)
    assert socket_path() == f"/run/user/{uid}/jarvis/control.sock"
    assert Path(socket_path()) == control.socket_path()


def test_the_scripts_request_is_the_one_control_send_writes() -> None:
    received: list[bytes] = []

    def behave(conn: socket.socket) -> None:
        received.append(read_line(conn))
        conn.sendall(b'{"ok": true, "state": "idle"}\n')

    with raw_server(behave) as path:
        assert control.send("status", path) == {"ok": True, "state": "idle"}
        assert control_status(str(path)) == '{"ok": true, "state": "idle"}'
    assert received == [REQUEST, REQUEST]
    assert json.loads(REQUEST) == {"command": "status"}
    assert "status" in control.COMMANDS


def test_the_not_running_reading_is_a_public_state_that_is_not_running() -> None:
    reading_ = json.loads(NOT_RUNNING)
    assert reading_["state"] in control.STATES
    assert reading_["running"] is False


def test_the_not_running_reading_is_what_jarvis_status_json_prints_without_a_jarvis(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Drift in `cmd_status`'s not-running JSON would otherwise show the bar the wrong state."""

    def unreachable_model(_model: object, _key: str, **_known: object) -> SimpleNamespace:
        return SimpleNamespace(status=lambda: None, file_missing=lambda: False)

    monkeypatch.setattr(app, "StrataModel", unreachable_model)
    assert cli.main(["status", "--json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed.pop("model") == "unreachable"  # the one field the script leaves out
    assert printed == json.loads(NOT_RUNNING)


def test_control_status_returns_what_control_send_would_have_parsed() -> None:
    server = control.ControlServer(lambda _command: {"state": "idle", "paused": False})
    server.start()
    try:
        assert json.loads(control_status(str(server.path))) == control.send("status", server.path)
    finally:
        server.close()


def test_control_status_reads_the_default_path_when_none_is_given() -> None:
    server = control.ControlServer(lambda _command: {"state": "speaking"})
    server.start()
    try:
        assert show(control_status).classes == ("speaking", "active")
    finally:
        server.close()


def test_nobody_serving_the_socket_is_not_running() -> None:
    assert control_status() == NOT_RUNNING  # no folder, no file
    path = control.socket_path()
    path.parent.mkdir(mode=0o700, parents=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as dead:
        dead.bind(str(path))  # bound, never listening: what a killed run leaves behind
        assert control_status() == NOT_RUNNING
        assert show(control_status).classes == ("not-running",)


def test_a_socket_that_cannot_be_opened_is_not_running() -> None:
    path = control.socket_path()
    path.parent.mkdir(mode=0o700, parents=True)
    path.write_text("a file, not a socket", encoding="utf-8")
    assert control_status() == NOT_RUNNING


def test_a_jarvis_that_never_answers_times_out() -> None:
    release = threading.Event()

    def behave(conn: socket.socket) -> None:
        read_line(conn)
        release.wait(5)

    with raw_server(behave) as path:
        try:
            with pytest.raises(TimeoutError):
                control_status(str(path), timeout=0.3)
            assert show(lambda: control_status(str(path), timeout=0.3)).classes == ("timeout",)
        finally:
            release.set()


def test_a_jarvis_that_answers_in_pieces_within_the_time_is_read() -> None:
    def behave(conn: socket.socket) -> None:
        read_line(conn)
        for piece in (b'{"ok": true, ', b'"state": "think', b'ing"}\n'):
            conn.sendall(piece)

    with raw_server(behave) as path:
        assert show(lambda: control_status(str(path))).classes == ("thinking", "active")


def test_a_jarvis_that_dribbles_its_answer_past_the_time_times_out() -> None:
    release = threading.Event()

    def behave(conn: socket.socket) -> None:
        read_line(conn)
        while not release.is_set():
            conn.sendall(b" ")
            release.wait(0.05)

    with raw_server(behave) as path:
        try:
            with pytest.raises(TimeoutError):
                control_status(str(path), timeout=0.5)
        finally:
            release.set()


def test_a_byte_sent_just_before_the_deadline_does_not_extend_it() -> None:
    """Each read gets only the time left; a full timeout each would hold the poll twice as long."""
    timeout = 0.5
    release = threading.Event()

    def behave(conn: socket.socket) -> None:
        read_line(conn)
        release.wait(timeout - 0.1)  # just before the deadline
        conn.sendall(b" ")
        release.wait(5)  # then nothing more

    with raw_server(behave) as path:
        began = time.monotonic()
        try:
            with pytest.raises(TimeoutError):
                control_status(str(path), timeout=timeout)
            took = time.monotonic() - began
        finally:
            release.set()
    assert took < timeout + 0.2  # without `settimeout(left)` this is about 2 * timeout - 0.1


def test_a_connection_that_resets_shows_failed() -> None:
    def behave(conn: socket.socket) -> None:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        # closed with the request unread: the client's read fails with a reset

    with raw_server(behave) as path:
        assert show(lambda: control_status(str(path))).classes == ("failed",)


@pytest.mark.parametrize(
    ("reply", "css"),
    [
        (b"", "malformed"),
        (b"\n", "malformed"),
        (b"not json\n", "malformed"),
        (b'{"ok": true, "state": "dancing"}\n', "malformed"),
        (b"\xff\xfe\n", "malformed"),
        (b'{"ok": true, "state": "idle"}', "idle"),  # closes without the newline
        (b"x" * (bar_status.MAX_READING_CHARS * 3), "malformed"),
        (b'{"ok": true, "state": "idle", "pad": "' + b"x" * 5000 + b'"}\n', "malformed"),
        (b'{"ok": false, "error": "pause failed; see the log"}\n', "unavailable"),
    ],
)
def test_what_the_socket_says_is_shown_or_malformed(reply: bytes, css: str) -> None:
    with answering(reply) as path:
        assert show(lambda: control_status(str(path))).classes[0] == css


def test_an_endless_reply_is_cut_short() -> None:
    def behave(conn: socket.socket) -> None:
        read_line(conn)
        with contextlib.suppress(OSError):  # the client stops reading
            for _ in range(1000):
                conn.sendall(b"x" * 1024)

    with raw_server(behave) as path:
        raw = control_status(str(path))
    assert bar_status.MAX_READING_CHARS < len(raw) <= bar_status.MAX_READING_CHARS + 1
    assert show(lambda: raw).classes == ("malformed",)


def stdlib_modules_of(script_output: str) -> set[str]:
    """Return the modules a `-X importtime` run imported (the last column of its lines)."""
    return {line.rsplit("|", 1)[1].strip() for line in script_output.splitlines() if "|" in line}


def run_script(*, flags: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
    """Run the script by its path, as the bar does (`bash -lc <path>`: the shebang picks Python)."""
    command = [sys.executable, *flags, str(SCRIPT)] if flags else [str(SCRIPT)]
    return subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)


def test_the_script_runs_end_to_end_against_a_real_control_server() -> None:
    server = control.ControlServer(lambda _command: {"state": "speaking", "model": "loaded"})
    server.start()
    try:
        done = run_script()
    finally:
        server.close()
    assert done.returncode == 0
    assert not done.stderr
    line = json.loads(done.stdout)
    assert line["class"] == ["speaking", "active"]
    assert line["text"] == FACES["speaking"].glyph
    assert "Model: loaded" in line["tooltip"]


def test_the_script_shows_not_running_when_nobody_serves_the_socket() -> None:
    done = run_script()
    assert done.returncode == 0
    assert json.loads(done.stdout)["class"] == ["not-running"]


def test_the_script_exits_zero_with_a_state_when_the_runtime_folder_is_odd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A relative folder falls back to the user's own, where a real Jarvis may be running, so the
    # state shown depends on this machine: any state the script can show, never a crash.
    monkeypatch.setenv("XDG_RUNTIME_DIR", "relative")
    assert socket_path() == f"/run/user/{os.getuid()}/jarvis/control.sock"
    done = run_script()
    assert done.returncode == 0
    # A busy Jarvis (listening, thinking, speaking) adds the one highlighted class after its state.
    shown, *highlight = json.loads(done.stdout)["class"]
    faces = {name.replace(" ", "-"): face for name, face in FACES.items()}
    assert shown in {*faces, "not-running", "timeout", "failed", "malformed"}
    busy = shown in faces and faces[shown].active
    assert highlight == ([ACTIVE] if busy else [])


def test_the_script_never_starts_jarvis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "ran"
    program = tmp_path / "jarvis"
    program.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
    program.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    done = run_script()
    assert done.returncode == 0
    assert not marker.exists()


def test_the_script_is_executable_and_stdlib_only() -> None:
    assert os.access(SCRIPT, os.X_OK)
    imports = set(
        re.findall(r"^(?:from|import) (\w+)", SCRIPT.read_text(encoding="utf-8"), re.MULTILINE)
    )
    assert imports <= set(sys.stdlib_module_names)


def test_a_poll_imports_no_site_and_none_of_the_heavy_modules() -> None:
    """Each import is milliseconds of CPU a poll; `jarvis` alone is 0.2 s (measured in the docs)."""
    first = SCRIPT.read_text(encoding="utf-8").splitlines()[0]
    assert first == "#!/usr/bin/env -S python3 -I -S"  # no site import, no user site, no cwd
    done = run_script(flags=("-I", "-S", "-X", "importtime"))
    assert done.returncode == 0
    imported = stdlib_modules_of(done.stderr)
    assert "json" in imported  # the check reads the right output
    heavy = {"jarvis", "site", "typing", "dataclasses", "subprocess", "shutil", "inspect", "httpx"}
    assert not heavy & imported


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
