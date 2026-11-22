"""The control socket and commands: `jarvis status`, `pause`, `resume` and `hush` (#22)."""

import json
import os
import socket
import stat
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, override

import numpy as np
import pytest

from jarvis import cli, control, voice
from jarvis.audio import FRAME, AudioError, MicPausedError, PausableMic
from jarvis.audit import AuditLog

if TYPE_CHECKING:
    from numpy.typing import NDArray

WAIT_S = 5.0
FRAME_DATA = np.zeros(FRAME, dtype=np.int16)


@pytest.mark.parametrize(
    ("loop", "paused", "model", "shown"),
    [
        ("idle", False, "loaded", "idle"),
        ("idle", False, "unloaded", "idle"),
        ("capture", False, "loaded", "listening"),
        ("follow-up", False, "loaded", "listening"),
        ("confirm", False, "loaded", "listening"),
        ("transcribe", False, "loaded", "thinking"),
        ("think", False, "loading", "thinking"),
        ("speak", False, "loaded", "speaking"),
        ("idle", True, "loaded", "paused"),
        ("speak", True, None, "paused"),
        ("idle", False, None, "unavailable"),
        ("idle", False, "failed", "unavailable"),
        ("idle", False, "loading", "model loading"),
        ("no-mic", False, "loaded", "unavailable"),
        ("no-mic", True, "loaded", "paused"),
    ],
)
def test_the_public_state_is_one_of_the_bars_states(
    loop: str, *, paused: bool, model: str | None, shown: str
) -> None:
    state = control.public_state(loop, paused=paused, model=model)

    assert state == shown
    assert state in control.STATES


def test_the_socket_lives_in_the_users_runtime_folder() -> None:
    assert control.socket_path() == Path(os.environ["XDG_RUNTIME_DIR"]) / "jarvis" / "control.sock"


@pytest.fixture
def served() -> Iterator[tuple[control.ControlServer, list[str]]]:
    seen: list[str] = []

    def handler(command: str) -> dict[str, object]:
        seen.append(command)
        return {"state": "idle"}

    server = control.ControlServer(handler)
    server.start()
    try:
        yield server, seen
    finally:
        server.close()


def _raw(path: Path, data: bytes) -> dict[str, object]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(WAIT_S)
        conn.connect(str(path))
        conn.sendall(data)
        reply = b""
        while not reply.endswith(b"\n"):
            chunk = conn.recv(4096)
            if not chunk:
                break
            reply += chunk
    return json.loads(reply)


def test_a_command_gets_the_handlers_answer(
    served: tuple[control.ControlServer, list[str]],
) -> None:
    server, seen = served

    reply = control.send("status", server.path)

    assert reply == {"ok": True, "state": "idle"}
    assert seen == ["status"]


def test_only_the_owner_can_reach_the_socket(
    served: tuple[control.ControlServer, list[str]],
) -> None:
    server, _ = served

    assert stat.S_IMODE(server.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(server.path.parent.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ("data", "error"),
    [
        (b'{"command": "shutdown"}\n', "unknown command 'shutdown'"),
        (b"not json\n", "send one JSON line with a command"),
        (b'{"command": 3}\n', "send one JSON line with a command"),
        (b'{"command": "status", "pad": "' + b"x" * 2000 + b'"}\n', "send one JSON line"),
    ],
)
def test_anything_but_a_known_command_is_refused_without_running(
    served: tuple[control.ControlServer, list[str]], data: bytes, error: str
) -> None:
    server, seen = served

    reply = _raw(server.path, data)

    assert reply["ok"] is False
    assert str(reply["error"]).startswith(error)
    assert seen == []


def test_a_failing_command_answers_with_an_error_and_keeps_serving() -> None:
    def handler(command: str) -> dict[str, object]:
        if command == "pause":
            msg = "boom"
            raise RuntimeError(msg)
        return {}

    server = control.ControlServer(handler)
    server.start()
    try:
        assert control.send("pause", server.path) == {
            "ok": False,
            "error": "pause failed; see the log",
        }
        assert control.send("status", server.path) == {"ok": True}
    finally:
        server.close()


def test_a_second_jarvis_does_not_take_over_the_socket(
    served: tuple[control.ControlServer, list[str]],
) -> None:
    server, _ = served
    second = control.ControlServer(lambda _c: {})

    with pytest.raises(OSError, match="already serving"):
        second.start()
    assert control.send("status", server.path) == {"ok": True, "state": "idle"}


def test_a_stale_socket_left_by_a_crash_is_replaced() -> None:
    path = control.socket_path()
    path.parent.mkdir(mode=0o700, parents=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as dead:
        dead.bind(str(path))  # bound, never listening: what a killed run leaves behind
    server = control.ControlServer(lambda _c: {"state": "idle"})

    server.start()
    try:
        assert control.send("status") == {"ok": True, "state": "idle"}
    finally:
        server.close()
    assert not path.exists()


def test_nothing_running_answers_none() -> None:
    assert control.send("status") is None


@dataclass
class Loop:
    state: str = "idle"
    hushes: int = 0

    def hush(self) -> None:
        self.hushes += 1


@dataclass
class Mic:
    paused: bool = False

    def pause(self) -> bool:
        changed, self.paused = not self.paused, True
        return changed

    def resume(self) -> bool:
        changed, self.paused = self.paused, False
        return changed


@dataclass
class Model:
    state: str | None = "loaded"
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

    def warm(self) -> bool:
        self.warms += 1
        return True


@dataclass
class Later:
    """Holds background work until the test runs it."""

    work: list[Callable[[], object]] = field(default_factory=list[Callable[[], object]])

    def __call__(self, work: Callable[[], object]) -> None:
        self.work.append(work)

    def run(self) -> None:
        work, self.work = self.work, []
        for item in work:
            item()


def test_status_reports_the_loop_the_pause_and_the_model() -> None:
    controls = control.Controls(Loop("speak"), Mic(), Model("loaded"))

    assert controls("status") == {"state": "speaking", "paused": False, "model": "loaded"}


def test_an_unreachable_model_server_shows_unavailable() -> None:
    controls = control.Controls(Loop(), Mic(), Model(None))

    assert controls("status") == {"state": "unavailable", "paused": False, "model": "unreachable"}


def test_a_model_server_down_for_a_missing_model_file_says_so() -> None:
    # `/mnt/models` not mounted: systemd skips Strata, whose model file is not there.
    controls = control.Controls(Loop(), Mic(), Model(None, missing=True))

    assert controls("status") == {"state": "unavailable", "paused": False, "model": "missing"}


def test_a_missing_model_file_does_not_matter_while_the_server_answers() -> None:
    controls = control.Controls(Loop(), Mic(), Model("loaded", missing=True))

    assert controls("status")["model"] == "loaded"


def test_after_a_suspend_the_model_is_checked_and_loaded_again() -> None:
    model, later = Model(), Later()
    controls = control.Controls(Loop(), Mic(), model, background=later)

    controls.after_sleep()

    assert model.warms == 0  # in the background, not on the voice loop's thread
    later.run()
    assert model.warms == 1


def test_after_a_suspend_a_paused_jarvis_leaves_the_model_unloaded() -> None:
    model, later = Model(), Later()
    controls = control.Controls(Loop(), Mic(paused=True), model, background=later)

    controls.after_sleep()
    later.run()

    assert model.warms == 0


def test_a_pause_while_waiting_for_the_microphone_frees_the_model_at_once() -> None:
    def sleep(_seconds: float) -> None:
        pytest.fail("no request is in flight while the microphone is gone")

    model, later = Model(), Later()
    controls = control.Controls(Loop("no-mic"), Mic(), model, background=later, sleep=sleep)
    controls("pause")
    later.run()

    assert model.unloads == 1


def test_pause_hushes_lets_go_of_the_mic_and_then_frees_the_model() -> None:
    loop, mic, model, later = Loop(), Mic(), Model(), Later()
    controls = control.Controls(loop, mic, model, background=later)

    assert controls("pause") == {"paused": True, "changed": True}
    assert loop.hushes == 1
    assert mic.paused
    assert model.unloads == 0  # in the background, not on the socket's thread
    later.run()
    assert model.unloads == 1
    assert controls("status")["state"] == "paused"


def test_a_second_pause_changes_nothing() -> None:
    model, later = Model(), Later()
    controls = control.Controls(Loop(), Mic(), model, background=later)
    controls("pause")
    later.run()

    assert controls("pause") == {"paused": True, "changed": False}
    later.run()
    assert model.unloads == 1


def test_the_model_is_freed_only_once_the_request_in_flight_ends() -> None:
    loop, model, later = Loop("think"), Model(), Later()
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds
        if now[0] >= 1.0:
            loop.state = "idle"

    controls = control.Controls(
        loop, Mic(), model, background=later, clock=lambda: now[0], sleep=sleep
    )
    controls("pause")
    later.run()

    assert model.unloads == 1
    assert now[0] >= 1.0


def test_resume_before_the_unload_keeps_the_model_and_warms_it() -> None:
    mic, model, later = Mic(), Model(), Later()
    controls = control.Controls(Loop(), mic, model, background=later)
    controls("pause")

    assert controls("resume") == {"paused": False, "changed": True}
    later.run()
    assert model.unloads == 0
    assert model.warms == 1
    assert not mic.paused


def test_resume_when_not_paused_changes_nothing() -> None:
    model, later = Model(), Later()
    controls = control.Controls(Loop(), Mic(), model, background=later)

    assert controls("resume") == {"paused": False, "changed": False}
    later.run()
    assert model.warms == 0


def test_hush_only_hushes() -> None:
    loop, mic = Loop("speak"), Mic()
    controls = control.Controls(loop, mic, Model())

    assert controls("hush") == {}
    assert loop.hushes == 1
    assert not mic.paused


class Capture:
    """A capture process stand-in: frames flow from start to stop, like `Microphone`."""

    def __init__(self) -> None:
        super().__init__()
        self.starts = 0
        self.stops = 0
        self._ended = threading.Event()
        self._started = threading.Event()

    @property
    def pid(self) -> int | None:
        return self.starts if not self._ended.is_set() else None

    def start(self) -> None:
        self.starts += 1
        self._ended = threading.Event()
        self._started.set()

    def stop(self) -> None:
        self.stops += 1
        self._ended.set()

    def end(self) -> None:
        """End on its own, the way `pw-record` does when PipeWire restarts."""
        self._ended.set()

    def drain(self) -> None:
        pass

    def frames(self) -> Iterator[NDArray[np.int16]]:
        # Like `Microphone`, a reader follows the capture running now, so one opened before a
        # quick pause and resume would read on from the new capture by itself.
        pause = threading.Event()
        while not self._ended.is_set():
            pause.wait(0.002)
            yield FRAME_DATA
        msg = "the microphone stream ended"
        raise AudioError(msg)


def test_a_reader_open_at_a_pause_ends_with_mic_paused() -> None:
    capture = Capture()
    mic = PausableMic(capture)
    mic.start()
    reader = mic.frames()
    next(reader)

    assert mic.pause()
    with pytest.raises(MicPausedError):
        next(reader)
    assert capture.stops == 1
    assert mic.paused


def test_a_reader_open_across_a_quick_pause_and_resume_still_ends() -> None:
    mic = PausableMic(Capture())
    mic.start()
    reader = mic.frames()
    next(reader)
    mic.pause()
    mic.resume()

    with pytest.raises(MicPausedError):
        next(reader)
    assert next(mic.frames()) is FRAME_DATA


def test_a_reader_opened_while_paused_ends_at_once() -> None:
    # A request that would start listening after the pause must not wait and then take
    # whatever is said after resume as a request.
    mic = PausableMic(Capture())
    mic.start()
    mic.pause()

    with pytest.raises(MicPausedError):
        next(mic.frames())


def test_the_loop_waits_for_resume_then_reads_the_reopened_microphone() -> None:
    capture = Capture()
    mic = PausableMic(capture)
    mic.start()
    mic.pause()
    got = threading.Event()

    def wait_then_read() -> None:
        mic.wait_resumed()
        next(mic.frames())
        got.set()

    threading.Thread(target=wait_then_read, daemon=True).start()

    assert not got.wait(0.1)
    assert mic.resume()
    assert got.wait(WAIT_S)
    assert capture.starts == 2
    assert not mic.paused


def test_stop_ends_a_wait_for_resume_and_then_every_reader() -> None:
    mic = PausableMic(Capture())
    mic.start()
    mic.pause()
    done = threading.Event()

    def wait() -> None:
        mic.wait_resumed()
        done.set()

    threading.Thread(target=wait, daemon=True).start()
    mic.stop()

    assert done.wait(WAIT_S)
    with pytest.raises(AudioError) as error:
        next(mic.frames())
    assert not isinstance(error.value, MicPausedError)
    assert not mic.resume()


class Broken(Capture):
    """A capture whose tool cannot be started (PipeWire's `pw-record` missing, say)."""

    @override
    def start(self) -> None:
        self.starts += 1
        self.end()  # like `Microphone`: a capture that never started has ended
        msg = "could not start pw-record"
        raise AudioError(msg)


@dataclass
class Waits:
    """Records each wait a recovery asks for, without waiting; `stop_at` answers 'stopped'."""

    seconds: list[float] = field(default_factory=list[float])
    stop_at: int | None = None

    def __call__(self, _event: threading.Event, seconds: float) -> bool:
        self.seconds.append(seconds)
        return self.stop_at is not None and len(self.seconds) >= self.stop_at


def test_a_capture_that_ended_on_its_own_is_started_again() -> None:
    capture = Capture()
    mic = PausableMic(capture, wait=Waits())
    mic.start()
    capture.end()
    with pytest.raises(AudioError) as error:
        next(mic.frames())
    assert not isinstance(error.value, MicPausedError)

    assert mic.recover()
    assert capture.starts == 2
    assert next(mic.frames()) is FRAME_DATA


def test_each_failed_reopen_waits_longer_up_to_a_cap_and_a_frame_starts_over() -> None:
    waits = Waits()
    capture = Capture()
    mic = PausableMic(capture, wait=waits)
    mic.start()
    for _ in range(8):
        capture.end()
        assert mic.recover()

    assert waits.seconds == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]
    next(mic.frames())  # the microphone works again
    capture.end()
    assert mic.recover()
    assert waits.seconds[-1:] == [1.0]


def _until_interrupted(event: threading.Event, _seconds: float) -> bool:
    """A reopen wait that only a pause or stop ends (twice as long as a test waits)."""
    return event.wait(2 * WAIT_S)


def test_stop_ends_a_reopen_wait_without_starting_a_capture() -> None:
    capture = Capture()
    mic = PausableMic(capture, wait=_until_interrupted)
    mic.start()
    capture.end()
    done = threading.Event()
    answer: list[bool] = []

    def recover() -> None:
        answer.append(mic.recover())
        done.set()

    threading.Thread(target=recover, daemon=True).start()
    assert not done.wait(0.1)  # waiting before the first reopen
    mic.stop()

    assert done.wait(WAIT_S)
    assert answer == [False]
    assert capture.starts == 1


def test_a_stopped_microphone_is_not_reopened() -> None:
    capture = Capture()
    mic = PausableMic(capture, wait=Waits())
    mic.start()
    mic.stop()

    assert not mic.recover()
    assert capture.starts == 1


def test_a_pause_during_a_reopen_wait_leaves_the_microphone_closed_until_resume() -> None:
    capture = Capture()
    mic = PausableMic(capture, wait=_until_interrupted)
    mic.start()
    capture.end()
    done = threading.Event()
    threading.Thread(target=lambda: (mic.recover(), done.set()), daemon=True).start()
    assert not done.wait(0.1)

    assert mic.pause()
    assert done.wait(WAIT_S)  # the pause ends the wait
    assert capture.starts == 1
    with pytest.raises(MicPausedError):
        next(mic.frames())
    assert mic.resume()
    assert capture.starts == 2


def test_a_capture_tool_that_cannot_start_is_tried_again_later() -> None:
    capture = Broken()
    mic = PausableMic(capture, wait=Waits())

    assert mic.recover()  # the failure is logged, not raised: the loop listens on and retries
    assert capture.starts == 1
    with pytest.raises(AudioError):
        next(mic.frames())


def _serve_fake(reply: dict[str, object]) -> control.ControlServer:
    server = control.ControlServer(lambda _c: reply)
    server.start()
    return server


def _model_server(state: str | None) -> Callable[[object], Model]:
    return lambda _cfg: Model(state)


def test_status_json_when_jarvis_is_not_running(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_strata", _model_server("loaded"))

    assert cli.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "state": "unavailable",
        "paused": False,
        "model": "loaded",
        "running": False,
    }


def test_status_without_jarvis_names_a_missing_model_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing(_cfg: object) -> Model:
        return Model(None, missing=True)

    monkeypatch.setattr(cli, "_strata", missing)

    assert cli.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["model"] == "missing"
    assert cli.main(["status"]) == 1
    assert capsys.readouterr().out == (
        "jarvis: not running, model: file missing (is its drive mounted?)\n"
    )


def test_status_without_jarvis_or_model_server_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_strata", _model_server(None))

    assert cli.main(["status"]) == 1
    assert capsys.readouterr().out == "jarvis: not running, model: server unreachable\n"


def test_status_json_from_the_running_jarvis(capsys: pytest.CaptureFixture[str]) -> None:
    server = _serve_fake({"state": "speaking", "paused": False, "model": "loaded"})
    try:
        assert cli.main(["status", "--json"]) == 0
    finally:
        server.close()

    assert json.loads(capsys.readouterr().out) == {
        "state": "speaking",
        "paused": False,
        "model": "loaded",
        "running": True,
    }


def test_status_text_from_the_running_jarvis(capsys: pytest.CaptureFixture[str]) -> None:
    server = _serve_fake({"state": "idle", "paused": False, "model": "loaded"})
    try:
        assert cli.main(["status"]) == 0
    finally:
        server.close()

    assert capsys.readouterr().out == "jarvis: idle, model: loaded\n"


@pytest.mark.parametrize(
    ("command", "reply", "said"),
    [
        ("pause", {"paused": True, "changed": True}, "jarvis: paused\n"),
        ("pause", {"paused": True, "changed": False}, "jarvis: already paused\n"),
        ("resume", {"paused": False, "changed": True}, "jarvis: listening again\n"),
        ("resume", {"paused": False, "changed": False}, "jarvis: was not paused\n"),
        ("hush", {}, "jarvis: hushed\n"),
    ],
)
def test_control_commands_say_what_happened(
    capsys: pytest.CaptureFixture[str], command: str, reply: dict[str, object], said: str
) -> None:
    server = _serve_fake(reply)
    try:
        assert cli.main([command]) == 0
    finally:
        server.close()

    assert capsys.readouterr().out == said


@pytest.mark.parametrize("command", ["pause", "resume", "hush"])
def test_control_commands_without_a_running_jarvis_fail(
    capsys: pytest.CaptureFixture[str], command: str
) -> None:
    assert cli.main([command]) == 1
    assert "not running" in capsys.readouterr().out


def test_commands_that_change_something_are_audited() -> None:
    audit = AuditLog(None)
    controls = control.Controls(Loop(), Mic(), Model(), audit, background=Later())

    controls("status")
    controls("pause")
    controls("resume")
    controls("hush")

    assert [(e["command"], e.get("changed")) for e in audit.events] == [
        ("pause", True),
        ("resume", True),
        ("hush", None),
    ]


def test_a_slow_status_does_not_hold_up_a_hush() -> None:
    release = threading.Event()
    hushed = threading.Event()

    def handler(command: str) -> dict[str, object]:
        if command == "status":
            release.wait(WAIT_S)  # a model server slow to answer
        else:
            hushed.set()
        return {}

    server = control.ControlServer(handler)
    server.start()
    try:
        slow = threading.Thread(target=control.send, args=("status", server.path), daemon=True)
        slow.start()
        assert control.send("hush", server.path, timeout=WAIT_S) == {"ok": True}
        assert hushed.is_set()
    finally:
        release.set()
        server.close()


def test_a_jarvis_that_does_not_answer_in_time_is_not_reported_as_not_running() -> None:
    release = threading.Event()

    def handler(_command: str) -> dict[str, object]:
        release.wait(WAIT_S)
        return {}

    server = control.ControlServer(handler)
    server.start()
    try:
        reply = control.send("status", server.path, timeout=0.2)
    finally:
        release.set()
        server.close()

    assert reply == {"ok": False, "error": "Jarvis did not answer in time"}


def test_status_json_exits_zero_even_with_nothing_reachable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_strata", _model_server(None))

    assert cli.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "unavailable"


def test_an_unload_and_the_load_a_quick_resume_starts_take_turns() -> None:
    order: list[str] = []
    unloading = threading.Event()
    finish_unload = threading.Event()

    def unload() -> bool:
        unloading.set()
        finish_unload.wait(WAIT_S)
        order.append("unloaded")
        return True

    def warm() -> bool:
        order.append("warmed")
        return True

    model = Model()
    model.unload = unload  # this test's slow server
    model.warm = warm
    mic = Mic()
    workers: list[threading.Thread] = []

    def background(work: Callable[[], object]) -> None:
        worker = threading.Thread(target=work, daemon=True)
        workers.append(worker)
        worker.start()

    controls = control.Controls(Loop(), mic, model, background=background)
    controls("pause")
    assert unloading.wait(WAIT_S)
    controls("resume")
    finish_unload.set()
    for worker in workers:
        worker.join(WAIT_S)

    assert order == ["unloaded", "warmed"]


def test_the_lost_microphone_state_is_the_voice_loops() -> None:
    assert control.NO_MIC == voice.State.NO_MIC
