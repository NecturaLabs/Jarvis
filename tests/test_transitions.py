"""Session transitions, one fake environment per cell of `docs/playbook/session-transitions.md`.

The matrix there crosses each command with each transition (a lost microphone, a suspend, a lock, a
missing model drive, the desktop session starting and stopping). The cells already covered live in
`test_voice.py`, `test_control.py`, `test_cli.py`, `test_model.py` and `test_services.py`; this file
holds the rest that need no hardware. Nothing here touches a real microphone, drive, `systemctl`
or Strata.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from strata_fixture import engine_run_config

from jarvis import cli, control, services, voice
from jarvis.agent import MODEL_DOWN, MODEL_MISSING, Agent, Hooks
from jarvis.app import no_action_operation, strata_model
from jarvis.audit import AuditLog
from jarvis.config import Config, ModelConfig, StrataConfig
from jarvis.model import ModelMissingError, StrataModel
from jarvis.operations import Registry

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.audio import VoxtypeWatch
    from jarvis.voice import VoiceLoop

HI: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
NO_TOOLS: list[dict[str, Any]] = []


# --- fakes -----------------------------------------------------------------------------------


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
    """What the control handler sees of the model server; `drive_missing` is its file not there."""

    state: str | None = "loaded"
    drive_missing: bool = False
    accepts: bool = True
    unloads: int = 0
    warms: int = 0

    def status(self) -> str | None:
        return self.state

    def file_missing(self) -> bool:
        return self.drive_missing

    def unload(self) -> bool:
        self.unloads += 1
        return self.accepts

    def warm(self, *, keep_going: Callable[[], bool] = lambda: True) -> bool:  # ruff: ignore[unused-method-argument] - the fake never retries
        self.warms += 1
        return self.accepts

    def is_loaded(self) -> bool | None:
        return None if self.state is None else self.state == "loaded"

    def load(self) -> bool:
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


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def _ticking() -> Callable[[], float]:
    now = [0.0]

    def clock() -> float:
        now[0] += 0.5
        return now[0]

    return clock


def _strata(
    model_file: Path | None,
    handler: Callable[[httpx.Request], httpx.Response] = _refused,
    naps: list[float] | None = None,
) -> StrataModel:
    return StrataModel(
        ModelConfig(base_url=Config().model_base_url()),
        "key",
        transport=httpx.MockTransport(handler),
        sleep=naps.append if naps is not None else lambda _s: None,
        clock=_ticking(),
        model_file=model_file,
    )


def _agent(model: StrataModel) -> Agent:
    return Agent(model, Registry([no_action_operation()]), "system", AuditLog(None), {})


def _speaking(spoken: list[str]) -> Hooks:
    return Hooks(confirm=lambda _question: False, speak=spoken.append)


# --- the model's drive is missing ------------------------------------------------------------


def test_a_spoken_request_with_the_model_drive_missing_is_answered_at_once_with_why(
    tmp_path: Path,
) -> None:
    # systemd skipped Strata (`ConditionPathExists=`), so nothing answers and nothing will.
    naps: list[float] = []
    spoken: list[str] = []
    agent = _agent(_strata(tmp_path / "gone.gguf", naps=naps))

    reply = agent.handle("hi", _speaking(spoken))

    assert reply == MODEL_MISSING
    assert spoken == [MODEL_MISSING]  # the owner hears it, not only the log
    assert naps == []  # no waiting for a server that is not coming


def test_a_streamed_chat_with_the_model_drive_missing_fails_at_once(tmp_path: Path) -> None:
    # The voice path streams; the existing test of this case sends a plain request.
    naps: list[float] = []
    model = _strata(tmp_path / "gone.gguf", naps=naps)

    with pytest.raises(ModelMissingError, match=r"gone\.gguf"):
        model.chat(HI, NO_TOOLS, on_delta=lambda _delta: None)
    assert naps == []


def test_the_model_drive_coming_back_changes_the_answer_from_missing_to_waiting_to_a_reply(
    tmp_path: Path,
) -> None:
    gguf = tmp_path / "model.gguf"
    server_up = [False]

    def server(request: httpx.Request) -> httpx.Response:
        if not server_up[0]:
            return _refused(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"status": {"value": "loaded"}}]})
        message = {"role": "assistant", "content": "Back."}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    naps: list[float] = []
    agent = _agent(_strata(gguf, server, naps))
    hooks = Hooks(confirm=lambda _question: False)

    assert agent.handle("hi", hooks) == MODEL_MISSING  # the drive is not mounted
    assert naps == []

    gguf.write_bytes(b"")  # mounted, but `jarvis start` has not started Strata yet
    assert agent.handle("hi", hooks) == MODEL_DOWN
    assert naps  # now it waited for the server to come back

    server_up[0] = True  # Strata started
    assert agent.handle("hi", hooks) == "Back."


def test_a_drive_that_goes_away_after_the_start_is_noticed_at_the_next_request(
    tmp_path: Path,
) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    model = _strata(gguf)

    assert not model.file_missing()
    gguf.unlink()  # the drive unmounted
    assert model.file_missing()
    gguf.write_bytes(b"")  # and mounted again
    assert not model.file_missing()


def test_the_model_client_knows_the_model_file_of_the_run_config(tmp_path: Path) -> None:
    gguf = tmp_path / "models" / "q4.gguf"
    cfg = Config(strata=StrataConfig(run_config=engine_run_config(tmp_path, model=gguf)))

    model = strata_model(cfg, "key")

    gguf.unlink()  # the drive unmounted
    assert model.file_missing()
    gguf.write_bytes(b"")
    assert not model.file_missing()


def test_a_run_config_that_cannot_be_read_never_reports_the_drive_missing(tmp_path: Path) -> None:
    cfg = Config(strata=StrataConfig(run_config=tmp_path / "gone.json"))

    assert not strata_model(cfg, "key").file_missing()


def test_a_config_without_a_run_config_never_reports_the_drive_missing() -> None:
    assert not strata_model(Config(), "key").file_missing()


def test_a_server_that_is_not_there_is_not_loaded_or_unloaded_and_says_so(tmp_path: Path) -> None:
    # What `jarvis resume` and the wake word's prewarm meet with the drive missing: no error.
    model = _strata(tmp_path / "gone.gguf")

    assert model.status() is None
    assert model.is_loaded() is None
    assert model.load() is False
    assert model.unload() is False


def test_pause_with_the_model_drive_missing_still_lets_go_of_the_microphone() -> None:
    model, later, mic = Model(None, drive_missing=True, accepts=False), Later(), Mic()
    controls = control.Controls(Loop(), mic, model, background=later)

    assert controls("pause") == {"paused": True, "changed": True}
    later.run()

    assert mic.paused
    assert model.unloads == 1  # asked, and nobody answered
    assert controls("status") == {"state": "paused", "paused": True, "model": "missing"}


def test_resume_with_the_model_drive_missing_listens_again_and_asks_for_the_model() -> None:
    model, later, mic = Model(None, drive_missing=True, accepts=False), Later(), Mic(paused=True)
    controls = control.Controls(Loop(), mic, model, background=later)

    assert controls("resume") == {"paused": False, "changed": True}
    later.run()

    assert not mic.paused  # Jarvis listens whatever became of the load
    assert model.warms == 1
    assert controls("status") == {"state": "unavailable", "paused": False, "model": "missing"}


def test_after_a_suspend_a_missing_model_drive_is_logged_and_nothing_is_loaded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An external drive can be gone when the PC wakes.
    model, later = Model(None, drive_missing=True), Later()
    controls = control.Controls(Loop(), Mic(), model, background=later)

    with caplog.at_level(logging.INFO, logger="jarvis.control"):
        controls.after_sleep()
        later.run()

    assert "back from sleep: the model is missing" in caplog.messages
    assert model.warms == 0


@pytest.mark.parametrize("state", ["loading", None])
def test_after_a_suspend_a_model_still_loading_or_unreachable_is_left_alone(
    state: str | None,
) -> None:
    model, later = Model(state), Later()
    controls = control.Controls(Loop(), Mic(), model, background=later)

    controls.after_sleep()
    later.run()

    assert model.warms == 0


def test_the_voice_service_wants_the_model_server_but_does_not_need_it() -> None:
    # `Requires=` or `BindsTo=` would take Jarvis down with a Strata that systemd skipped for want
    # of its drive; `Wants=` leaves Jarvis running to say why it cannot answer.
    unit = services.jarvis_unit().splitlines()

    assert f"Wants={services.STRATA_UNIT}" in unit
    assert not [
        line
        for line in unit
        if line.startswith(("Requires=", "Requisite=", "BindsTo=")) and services.STRATA_UNIT in line
    ]


def _unload_run(
    monkeypatch: pytest.MonkeyPatch, model: Model, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str]:
    def strata(_cfg: object) -> Model:
        return model

    monkeypatch.setattr(cli, "_strata", strata)
    code = cli.main(["unload"])
    return code, capsys.readouterr().out


def test_unload_frees_the_model_when_the_server_answers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, said = _unload_run(monkeypatch, Model("loaded"), capsys)

    assert (code, said) == (0, "model unloaded\n")


def test_unload_with_the_model_drive_missing_says_it_could_not(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model = Model(None, drive_missing=True, accepts=False)

    code, said = _unload_run(monkeypatch, model, capsys)

    assert (code, said) == (1, "could not unload (busy or unreachable)\n")


# --- the desktop session stops and starts ----------------------------------------------------


def _systemctl_calls(monkeypatch: pytest.MonkeyPatch, code: int = 0) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def systemctl(*args: str) -> int:
        calls.append(args)
        return code

    monkeypatch.setattr(cli, "_systemctl", systemctl)

    def inactive(_unit: str) -> str:
        return "inactive"

    monkeypatch.setattr(cli, "_unit_state", inactive)
    return calls


def test_stop_stops_the_voice_service_and_the_model_server_and_says_where_they_are(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _systemctl_calls(monkeypatch)

    assert cli.main(["stop"]) == 0

    assert calls == [("stop", services.JARVIS_UNIT, services.STRATA_UNIT)]
    assert capsys.readouterr().out == "jarvis: inactive, model server: inactive\n"


def test_a_stop_that_systemd_refuses_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _systemctl_calls(monkeypatch, code=1)

    assert cli.main(["stop"]) == 1


@dataclass
class _Voxtype:
    started: bool = False
    stopped: bool = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


@dataclass
class _EndingLoop:
    """A voice loop that answers on the control socket while it runs, then ends or fails."""

    fails: Exception | None = None
    answered: dict[str, Any] | None = None

    def run(self, _should_stop: Callable[[], bool]) -> None:
        self.answered = control.send("status")
        if self.fails is not None:
            raise self.fails


def _run_ending(loop: _EndingLoop) -> tuple[int, _Voxtype]:
    handler = control.Controls(Loop(), Mic(), Model(), background=Later())
    voxtype = _Voxtype()
    controls = cli._serve_controls(handler, AuditLog(None))  # pyright: ignore[reportPrivateUsage] - the run's own helper, tested through its use
    code = cli._run_until_stopped(  # pyright: ignore[reportPrivateUsage] - what `jarvis run` ends with
        cast("VoiceLoop", loop),
        threading.Event(),
        controls,
        cast("VoxtypeWatch", voxtype),
        [],
    )
    return code, voxtype


def test_a_run_ended_with_the_session_leaves_no_control_socket_for_the_next_one() -> None:
    loop = _EndingLoop()

    code, voxtype = _run_ending(loop)

    assert code == 0
    assert loop.answered is not None
    assert loop.answered["ok"] is True  # `jarvis status` reached it while it ran
    assert voxtype.started
    assert voxtype.stopped
    assert control.send("status") is None  # nothing answers once the session is over
    next_run = control.ControlServer(lambda _command: {})
    next_run.start()  # and the next session's run can serve it
    next_run.close()


def test_a_run_that_fails_leaves_no_control_socket_either() -> None:
    loop = _EndingLoop(fails=RuntimeError("the loop broke"))

    with pytest.raises(RuntimeError, match="the loop broke"):
        _run_ending(loop)

    assert control.send("status") is None


def test_a_run_that_cannot_serve_the_control_socket_still_listens(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A run that starts while an older one still holds the socket (a restart that overlaps).
    older = control.ControlServer(lambda _command: {"state": "idle"})
    older.start()
    try:
        handler = control.Controls(Loop(), Mic(), Model(), background=Later())

        with caplog.at_level(logging.WARNING, logger="jarvis.cli"):
            newer = cli._serve_controls(handler, AuditLog(None))  # pyright: ignore[reportPrivateUsage] - the run's own helper

        assert [m for m in caplog.messages if m.startswith("no control socket")]
        assert control.send("status") == {
            "ok": True,
            "state": "idle",
        }  # the older one is undisturbed
        newer.close()
        assert control.send("status") == {"ok": True, "state": "idle"}  # and closing the loser
    finally:
        older.close()


# --- the microphone is gone ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "answer"),
    [
        ("pause", {"paused": True, "changed": True}),
        ("resume", {"paused": False, "changed": False}),
        ("hush", {}),
        ("confirm", {"confirmed": None, "reason": control.NO_KEY}),
    ],
)
def test_every_command_still_answers_while_the_microphone_is_gone(
    command: str, answer: dict[str, object]
) -> None:
    # The loop waits to reopen the microphone; the socket is served by another thread.
    controls = control.Controls(Loop("no-mic"), Mic(), Model(), background=Later())

    assert controls(command) == answer


def test_a_hush_while_the_microphone_is_gone_reaches_the_loop() -> None:
    loop = Loop("no-mic")
    controls = control.Controls(loop, Mic(), Model(), background=Later())

    controls("hush")

    assert loop.hushes == 1


# --- suspend and resume ----------------------------------------------------------------------


def test_the_time_asleep_is_how_far_the_boot_clock_has_run_ahead_of_the_monotonic_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clocks: list[int] = []

    def boot_clock(clock_id: int) -> float:
        clocks.append(clock_id)
        return 1000.0

    with monkeypatch.context() as patched:  # only for the call: other threads read the clocks too
        patched.setattr(voice.time, "clock_gettime", boot_clock)
        patched.setattr(voice.time, "monotonic", lambda: 400.0)
        asleep = voice.time_asleep()

    assert asleep == pytest.approx(600.0)
    assert clocks == [voice.time.CLOCK_BOOTTIME]  # the clock that counts the sleep


def test_awake_the_time_asleep_does_not_move_between_two_reads() -> None:
    first = voice.time_asleep()
    second = voice.time_asleep()

    assert abs(second - first) < voice.SLEPT_S  # an awake PC never looks suspended
