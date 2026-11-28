"""The voice service and the chat as two channels that cooperate (ADR 0028).

A spoken request is the voice agent's own and never shows in a chat window; `chat_hand_off` passes
the owner's spoken words to the open window as their own input; neither channel cuts the other's
work except where the owner's words call for it (a pause, a mode switch).
"""

import asyncio
import contextlib
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, override

import pytest
import test_chat
import test_cli
from test_chat import FakeSession, FakeTail
from test_control import Loop, Mic, Model

from jarvis import cli, control, handoff, modes, paths
from jarvis.activity import VOICE_VIA
from jarvis.agent import Agent, Hooks
from jarvis.app import build, no_action_operation
from jarvis.audit import AuditLog
from jarvis.chat.widgets import Reply
from jarvis.cli import main
from jarvis.config import ChatConfig, Config, SpotifyConfig
from jarvis.handoff import ChatListener, HandoffError, Sent, State
from jarvis.operations import (
    Operation,
    OperationError,
    Registry,
    Result,
    UncertainOutcomeError,
    request_scope,
)
from jarvis.paths import audit_file, strata_key_file
from jarvis.secretfile import ensure_api_key

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator

    from textual.pilot import Pilot

# The chat tests' own fixtures for a window, a transcript and a typed line.
_app = test_chat._app  # pyright: ignore[reportPrivateUsage]
_cmd_run = test_cli._cmd_run  # pyright: ignore[reportPrivateUsage]
_opened_chat = test_cli._opened_chat  # pyright: ignore[reportPrivateUsage]
_has = test_chat._has  # pyright: ignore[reportPrivateUsage]
_run = test_chat._run  # pyright: ignore[reportPrivateUsage]
_type = test_chat._type  # pyright: ignore[reportPrivateUsage]
_until = test_chat._until  # pyright: ignore[reportPrivateUsage]

SPOKEN = "add a test for the parser to what you are doing"


def _restart_recorder(done: list[bool]) -> Callable[..., bool]:
    def restart(**_kwargs: bool) -> bool:
        done.append(True)
        return True

    return restart


def _restart_refused(**_kwargs: bool) -> bool:
    return False


def _any_peer(_pid: int) -> bool:
    """The test is the sender and the window at once, so any peer counts as the voice service."""
    return True


def _no_peer(_pid: int) -> bool:
    return False


@dataclass
class Chat:
    """A chat window's socket: what it was handed, and what it answers."""

    answer: State | HandoffError = State.STARTED
    words: list[str] = field(default_factory=list[str])

    def take(self, words: str) -> State:
        self.words.append(words)
        if isinstance(self.answer, HandoffError):
            raise self.answer
        return self.answer


@pytest.fixture
def window() -> Iterator[Chat]:
    chat = Chat()
    listener = ChatListener("w1", chat.take, allowed=_any_peer)
    listener.start()
    yield chat
    listener.close()


# The socket


def test_an_open_chat_takes_spoken_words_and_says_what_it_did(window: Chat) -> None:
    window.answer = State.STEERED

    assert handoff.chat_open()
    assert handoff.send(SPOKEN).state is State.STEERED
    assert window.words == [SPOKEN]


def test_the_socket_and_its_folder_are_the_owners_alone(window: Chat) -> None:
    del window
    path = handoff.socket_path("w1")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_without_a_chat_window_nothing_is_open_and_a_send_is_refused() -> None:
    assert not handoff.chat_open()
    with pytest.raises(HandoffError, match="no chat window"):
        handoff.send(SPOKEN)


def test_a_socket_nobody_answers_on_is_not_an_open_chat_and_is_left_alone() -> None:
    path = handoff.socket_path("gone")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(str(path))  # a window between bind and listen looks the same: never removed

    assert not handoff.chat_open()
    assert path.exists()


def test_a_window_cleans_up_its_own_stale_socket_when_it_starts() -> None:
    path = handoff.socket_path("w1")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(str(path))
    listener = ChatListener("w1", Chat().take, allowed=_any_peer)
    listener.start()
    try:
        assert handoff.chat_open()
    finally:
        listener.close()


def test_the_runtime_folder_above_the_sockets_is_the_owners_alone_too() -> None:
    listener = ChatListener("w1", Chat().take, allowed=_any_peer)
    listener.start()
    try:
        assert stat.S_IMODE(handoff.chats_dir().parent.stat().st_mode) == 0o700
    finally:
        listener.close()


def test_the_probe_for_an_open_chat_is_remembered_for_a_few_seconds() -> None:
    now = [100.0]
    handoff.forget_chat_probe()
    assert not handoff.chat_open(clock=lambda: now[0])
    listener = ChatListener("w1", Chat().take, allowed=_any_peer)
    path = handoff.socket_path("w1")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(str(path))
        sock.listen(1)
        assert not handoff.chat_open(clock=lambda: now[0])  # remembered: not asked again
        handoff.forget_chat_probe()
        assert handoff.chat_open(clock=lambda: now[0])
        assert handoff.chat_open(clock=lambda: now[0] + handoff.CACHE_S / 2)
    del listener


def test_non_ascii_speech_is_not_refused_as_too_long() -> None:
    chat = Chat()
    listener = ChatListener("w1", chat.take, allowed=_any_peer)
    listener.start()
    words = "ü" * handoff.MAX_CHARS
    try:
        handoff.send(words)
    finally:
        listener.close()

    assert chat.words == [words]


def test_a_window_that_accepts_and_never_answers_leaves_the_outcome_uncertain() -> None:
    path = handoff.socket_path("silent")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    held: list[socket.socket] = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(str(path))
        sock.listen(4)

        def accept() -> None:
            while True:
                try:
                    held.append(sock.accept()[0])  # taken and never answered
                except OSError:
                    return

        threading.Thread(target=accept, daemon=True).start()
        with pytest.raises(UncertainOutcomeError):
            handoff.send(SPOKEN, timeout=0.3)
        for conn in held:
            conn.close()


def test_a_refusal_by_the_window_reaches_the_sender(window: Chat) -> None:
    window.answer = HandoffError("the chat cannot take requests: no key")

    with pytest.raises(HandoffError, match="no key"):
        handoff.send(SPOKEN)


def test_with_two_windows_the_newest_gets_the_words(window: Chat) -> None:
    second = Chat()
    listener = ChatListener("w2", second.take, allowed=_any_peer)
    listener.start()
    os.utime(handoff.socket_path("w1"), (1, 1))
    try:
        handoff.send(SPOKEN)
    finally:
        listener.close()

    assert second.words == [SPOKEN]
    assert window.words == []


# The tool


def test_the_tool_is_offered_only_while_a_chat_window_is_open() -> None:
    registry = Registry([handoff.operation()])
    assert registry.tool_schemas() == []

    listener = ChatListener("w1", Chat().take, allowed=_any_peer)
    listener.start()
    try:
        names = [t["function"]["name"] for t in registry.tool_schemas()]
    finally:
        listener.close()

    assert names == [handoff.NAME]
    assert [t["function"]["name"] for t in registry.tool_schemas(every=True)] == [handoff.NAME]


def test_the_tool_takes_no_words_of_the_models() -> None:
    op = handoff.operation()

    assert op.name == handoff.NAME

    assert op.parameters["properties"] == {}
    assert op.parameters["additionalProperties"] is False
    assert op.default_class.value == "safe"


def _run_tool(
    sent: list[str], *, foreign: bool = False, scheduled: bool = False, words: str = SPOKEN
) -> Result:
    op = handoff.operation(lambda said: sent.append(said) or Sent(State.STARTED))
    with request_scope(foreign=foreign, words=words, scheduled=scheduled):
        return op.run({})


def test_the_tool_passes_on_the_owners_own_words() -> None:
    sent: list[str] = []

    result = _run_tool(sent)

    assert sent == [SPOKEN]
    assert result.ends
    assert result.spoken == "Started in the chat."


def _refused(words: str = SPOKEN, *, foreign: bool = False, scheduled: bool = False) -> list[str]:
    sent: list[str] = []
    with request_scope(foreign=foreign, words=words, scheduled=scheduled):
        run = handoff.operation(lambda said: sent.append(said) or Sent(State.STARTED)).run
        with pytest.raises(OperationError):
            run({})
    return sent


def test_the_tool_is_refused_for_a_request_heard_in_pc_audio() -> None:
    assert _refused(foreign=True) == []


def test_the_tool_is_refused_for_a_scheduled_request() -> None:
    assert _refused(scheduled=True) == []


def test_the_tool_is_refused_without_the_owners_words() -> None:
    assert _refused(words="") == []


def test_the_tool_is_refused_once_the_request_read_outside_text() -> None:
    sent: list[str] = []
    op = handoff.operation(lambda said: sent.append(said) or Sent(State.STARTED))
    with request_scope(foreign=False, words=SPOKEN) as request:
        request.saw("web_page")
        with pytest.raises(OperationError):
            op.run({})

    assert sent == []


def test_the_tool_is_refused_outside_a_request() -> None:
    with pytest.raises(OperationError):
        handoff.operation().run({})


def test_remembered_outside_text_does_not_refuse_the_owners_own_request() -> None:
    sent: list[str] = []
    op = handoff.operation(lambda said: sent.append(said) or Sent(State.QUEUED))

    with request_scope(foreign=False, words=SPOKEN) as request:
        request.saw("memory")
        result = op.run({})

    assert sent == [SPOKEN]
    assert result.spoken == "Queued in the chat."


def test_a_window_that_cannot_be_reached_is_told_to_the_model() -> None:
    def deliver(_words: str) -> Sent:
        raise HandoffError("no chat window is open")

    op = handoff.operation(deliver)
    with request_scope(foreign=False, words=SPOKEN), pytest.raises(OperationError, match="no chat"):
        op.run({})


@dataclass
class _Model:
    replies: list[dict[str, Any]]
    calls: int = 0
    offered: list[list[str]] = field(default_factory=list[list[str]])
    loaded: bool = True

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **_: object
    ) -> dict[str, Any]:
        del messages
        self.offered.append([t["function"]["name"] for t in tools])
        self.calls += 1
        return self.replies.pop(0)

    def is_loaded(self) -> bool:
        return self.loaded


def _call(name: str) -> dict[str, Any]:
    call = {"id": "c1", "type": "function", "function": {"name": name, "arguments": "{}"}}
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def test_the_voice_request_ends_with_one_short_line_after_a_hand_off() -> None:
    sent: list[str] = []
    model = _Model([_call(handoff.NAME)])
    op = handoff.operation(lambda said: sent.append(said) or Sent(State.STEERED))
    agent = Agent(model, Registry([op, no_action_operation()]), "system", AuditLog(None), {})
    spoken: list[str] = []

    reply = agent.handle(SPOKEN, Hooks(confirm=lambda _q: False, speak=spoken.append))

    assert reply == "Added to what the chat is working on."
    assert model.calls == 1  # no second model call: the request ended
    assert sent == [SPOKEN]


def test_a_failed_hand_off_does_not_end_the_request() -> None:
    def deliver(_words: str) -> Sent:
        raise HandoffError("no chat window is open")

    model = _Model([_call(handoff.NAME), {"role": "assistant", "content": "I did it myself."}])
    agent = Agent(
        model,
        Registry([handoff.operation(deliver), no_action_operation()]),
        "system",
        AuditLog(None),
        {},
    )

    reply = agent.handle(SPOKEN, Hooks(confirm=lambda _q: False))

    assert model.calls == 2
    assert reply == "I did it myself."


# Where it is offered


def _names(via: str | None) -> list[str]:
    ensure_api_key(strata_key_file())
    config = Config(spotify=SpotifyConfig(client_id="id"))
    return build(config, via=via).agent._registry.names()  # pyright: ignore[reportPrivateUsage] - what the model is given


def test_only_the_voice_channel_has_the_tool() -> None:
    assert handoff.NAME in _names("voice")
    for via in ("chat", "ask", "speech-test"):
        assert handoff.NAME not in _names(via)


def test_the_voice_service_marks_its_audit_events_as_its_own() -> None:
    ensure_api_key(strata_key_file())
    jarvis = build(Config(), via="voice")

    jarvis.audit.record("probe")

    event = json.loads(audit_file().read_text(encoding="utf-8").splitlines()[-1])
    assert event["via"] == "voice"


# The chat window


async def _handed(app: test_chat.ChatApp, words: str = SPOKEN) -> State:
    return await asyncio.to_thread(app.take_handoff, words)


def test_spoken_words_start_a_request_in_an_idle_chat_marked_as_spoken() -> None:
    session = FakeSession()
    app = _app(session=session)

    async def test(pilot: Pilot[None]) -> None:
        assert await _handed(app) is State.STARTED
        await _until(pilot, lambda: session.asked == [SPOKEN])
        assert _has(app, "spoken")
        assert not _has(app, "typed")

    _run(test, app)


def test_spoken_words_steer_the_request_the_chat_is_running() -> None:
    steered: list[str] = []

    def answer(text: str, hooks: Hooks) -> str:
        if text == "pause":
            for _ in range(500):
                if taken := hooks.steer():
                    steered.extend(taken)
                    break
                threading.Event().wait(0.01)
        return "Done."

    session = FakeSession(answer)
    app = _app(session=session)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: session.asked == ["pause"])
        assert await _handed(app) is State.STEERED
        await _until(pilot, lambda: bool(steered))
        assert _has(app, "spoken")

    _run(test, app)
    assert steered == [SPOKEN]
    assert session.asked == ["pause"]  # the running request was neither cut nor restarted


def test_with_queueing_set_spoken_words_wait_for_the_running_request() -> None:
    release = threading.Event()

    def answer(text: str, _hooks: Hooks) -> str:
        if text == "pause":
            release.wait(5)
        return "Done."

    session = FakeSession(answer)
    app = _app(session=session, settings=ChatConfig(while_working="queue"))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: session.asked == ["pause"])
        assert await _handed(app) is State.QUEUED
        assert session.asked == ["pause"]
        release.set()
        await _until(pilot, lambda: session.asked == ["pause", SPOKEN])

    _run(test, app)


def test_a_chat_that_cannot_take_requests_refuses_the_words() -> None:
    app = _app(session=None, unavailable="no Strata API key")

    async def test(pilot: Pilot[None]) -> None:
        del pilot
        with pytest.raises(HandoffError, match="no Strata API key"):
            await _handed(app)

    _run(test, app)


def test_with_sharing_off_a_voice_request_stays_out_of_the_chat_but_typed_and_ask_show() -> None:
    # `[channels] shared = false` keeps ADR 0028's split; the window is built with show_voice off.
    tail = FakeTail()
    voice = {"via": "voice"}
    tail.entries = [
        {"event": "request", "text": "play some jazz", **voice},
        {"event": "tool_call", "op": "spotify_play", "args": {}, **voice},
        {"event": "reply", "text": "Playing jazz.", "seconds": 0.4, **voice},
        {"event": "request", "text": "what is in the terminal", "via": "ask"},
    ]
    app = _app(tail, FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "what is in the terminal"))
        await pilot.pause(0.05)
        assert not _has(app, "play some jazz")
        assert not _has(app, "Playing jazz.")
        assert not _has(app, "Spotify")

    _run(test, app)


def test_with_sharing_on_a_voice_request_and_its_spoken_reply_show_in_the_open_chat() -> None:
    # ADR 0030 (owner, 2026-11-28): voice and chat are one conversation, so what was said aloud
    # shows in the window, marked as spoken.
    tail = FakeTail()
    voice = {"via": "voice"}
    tail.entries = [
        {"event": "request", "text": "play some jazz", **voice},
        {"event": "reply", "text": "Playing jazz.", "seconds": 0.4, **voice},
    ]
    app = _app(tail, FakeSession(), services=test_chat.Services(show_voice=True))

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "Playing jazz."))
        assert _has(app, "play some jazz")
        assert _has(app, "by voice")
        (reply,) = [w for w in app.query(Reply) if w.text == "Playing jazz."]
        assert reply._note.endswith("spoken")  # pyright: ignore[reportPrivateUsage] - the line under it

    _run(test, app)


# Neither channel cuts the other


@contextlib.contextmanager
def _busy_then_free(looks: list[bool]) -> Generator[bool]:
    yield looks.pop(0) if looks else True


def test_a_pause_frees_the_model_only_once_a_chat_request_has_ended() -> None:
    looks = [False, False, False]  # a chat window's request, for three looks
    sleeps = 0
    model = Model()
    work: list[Callable[[], object]] = []

    def sleep(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        assert model.unloads == 0

    controls = control.Controls(
        Loop(),
        Mic(),
        model,
        background=work.append,
        nobody_answering=lambda: _busy_then_free(looks),
        sleep=sleep,
    )
    controls("pause")
    work.pop()()

    assert sleeps == 3
    assert model.unloads == 1


def test_the_unload_is_made_while_no_request_can_begin() -> None:
    held: list[bool] = []

    @contextlib.contextmanager
    def nobody() -> Generator[bool]:
        held.append(True)
        try:
            yield True
        finally:
            held.append(False)

    class Unloading(Model):
        @override
        def unload(self) -> bool:
            held.append(held[-1])  # still inside the lock when the server is asked
            return super().unload()

    work: list[Callable[[], object]] = []
    controls = control.Controls(
        Loop(), Mic(), Unloading(), background=work.append, nobody_answering=nobody
    )
    controls("pause")
    work.pop()()

    assert held == [True, True, False]


def test_a_request_that_holds_the_modes_lock_holds_the_unload_back() -> None:
    model = Model()
    work: list[Callable[[], object]] = []
    seen: list[int] = []
    answering = modes.Answering()

    def sleep(_seconds: float) -> None:
        seen.append(model.unloads)
        if len(seen) == 2:
            answering.__exit__()  # the chat's request ends

    controls = control.Controls(Loop(), Mic(), model, background=work.append, sleep=sleep)
    with answering:
        controls("pause")
        work.pop()()

    assert seen == [0, 0]
    assert model.unloads == 1


def test_a_resume_while_a_chat_request_runs_drops_the_waiting_unload() -> None:
    model, mic = Model(), Mic()
    work: list[Callable[[], object]] = []

    def sleep(_seconds: float) -> None:
        controls("resume")

    controls = control.Controls(
        Loop(),
        mic,
        model,
        background=work.append,
        nobody_answering=lambda: _busy_then_free([False] * 5),
        sleep=sleep,
    )
    controls("pause")
    work.pop(0)()

    assert model.unloads == 0


def test_the_voice_service_is_the_process_serving_the_control_socket() -> None:
    sock = control.socket_path()
    assert control.serving_pid(sock) is None
    server = control.ControlServer(lambda _c: {})
    server.start()
    try:
        assert control.serving_pid(sock) == os.getpid()
    finally:
        server.close()


def test_a_program_the_voice_service_started_cannot_hand_the_chat_words() -> None:
    refused: list[str] = []
    chat = Chat()
    server = control.ControlServer(lambda _c: {})
    server.start()  # this process plays the voice service
    listener = ChatListener("w1", chat.take, refused=refused.append)
    listener.start()
    code = (
        "from jarvis import handoff\n"
        "try:\n"
        "    handoff.send('do this')\n"
        "except handoff.HandoffError as e:\n"
        "    print(e)\n"
    )
    try:
        assert handoff.send(SPOKEN).state is State.STARTED  # the service itself
        child = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env={**os.environ},
        )
    finally:
        listener.close()
        server.close()

    assert "only the voice service" in child.stdout
    assert chat.words == [SPOKEN]
    assert len(refused) == 1


def test_without_a_voice_service_the_chat_takes_no_hand_off() -> None:
    chat = Chat()
    listener = ChatListener("w1", chat.take)
    listener.start()
    try:
        with pytest.raises(HandoffError, match="only the voice service"):
            handoff.send(SPOKEN)
    finally:
        listener.close()

    assert chat.words == []


def test_a_setting_changed_in_the_chat_leaves_a_note_while_a_spoken_request_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restarts: list[bool] = []
    monkeypatch.setattr(cli, "_restart_if_running", lambda: restarts.append(True) or True)

    with modes.voice_request() as admitted:
        assert admitted
        said = cli._apply_from_chat()  # pyright: ignore[reportPrivateUsage] - the chat's hook

    assert said == cli.APPLIES_WHEN_IDLE
    assert paths.restart_pending_file().exists()
    assert restarts == []


def test_the_voice_service_restarts_for_the_note_once_it_is_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restarts: list[bool] = []
    monkeypatch.setattr(cli, "_restart_if_running", _restart_recorder(restarts))
    monkeypatch.setattr(cli, "PENDING_CHECK_S", 0.01)
    paths.restart_pending_file().parent.mkdir(parents=True, exist_ok=True)
    paths.restart_pending_file().write_text("x", encoding="utf-8")
    stop = threading.Event()
    busy = SimpleNamespace(state="think")
    watcher = threading.Thread(
        target=cli._restart_when_pending,  # pyright: ignore[reportPrivateUsage] - the voice service's watcher
        args=(busy, stop),
    )
    watcher.start()
    try:
        threading.Event().wait(0.1)
        assert restarts == []  # a request is under way
        busy.state = "idle"
        for _ in range(300):
            if restarts:
                break
            threading.Event().wait(0.01)
    finally:
        stop.set()
        watcher.join(5)

    assert restarts == [True]
    assert not paths.restart_pending_file().exists()


def test_a_request_beginning_at_the_moment_of_the_restart_is_not_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted_during: list[bool] = []

    def restart() -> bool:
        with modes.voice_request() as admitted:
            admitted_during.append(admitted)
        return True

    monkeypatch.setattr(cli, "_restart_if_running", restart)

    assert not cli._apply_from_chat()  # pyright: ignore[reportPrivateUsage] - the chat's hook
    assert admitted_during == [False]


def test_a_setting_changed_in_the_chat_restarts_at_once_when_nothing_is_spoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restarts: list[bool] = []
    paths.restart_pending_file().parent.mkdir(parents=True, exist_ok=True)
    paths.restart_pending_file().write_text("x", encoding="utf-8")
    monkeypatch.setattr(cli, "_restart_if_running", lambda: restarts.append(True) or True)

    assert not cli._apply_from_chat()  # pyright: ignore[reportPrivateUsage] - the chat's hook
    assert restarts == [True]
    assert not paths.restart_pending_file().exists()


def test_a_hand_off_beside_other_calls_does_not_end_the_request() -> None:
    sent: list[str] = []
    other = Operation(
        "pc_volume_probe",
        "Probe.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        lambda _a: Result("Volume is 40."),
        side_effect=False,
    )
    calls = [
        {"id": "a", "type": "function", "function": {"name": handoff.NAME, "arguments": "{}"}},
        {"id": "b", "type": "function", "function": {"name": "pc_volume_probe", "arguments": "{}"}},
    ]
    model = _Model(
        [
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "assistant", "content": "The volume is 40."},
        ]
    )
    op = handoff.operation(lambda said: sent.append(said) or Sent(State.STARTED))
    agent = Agent(model, Registry([op, other, no_action_operation()]), "system", AuditLog(None), {})

    reply = agent.handle(SPOKEN, Hooks(confirm=lambda _q: False))

    assert model.calls == 2
    assert "40" in reply
    assert sent == []  # not sent: the owner's words would carry the other work into the chat


# The wiring, as the services use it


def test_the_voice_service_is_built_with_the_voice_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    run, code = _cmd_run(
        monkeypatch, lambda r, _stop: r.handlers[signal.SIGTERM](signal.SIGTERM, None)
    )

    assert code == 0
    assert run.vias == [VOICE_VIA]


def _hand_off_model(model: _Model) -> Agent:
    return Agent(
        model, Registry([handoff.operation(), no_action_operation()]), "system", AuditLog(None), {}
    )


def test_the_model_is_offered_the_tool_only_while_a_chat_answers() -> None:
    model = _Model([_reply_of("No."), _reply_of("Still no."), _reply_of("Now.")])
    agent = _hand_off_model(model)
    handoff.forget_chat_probe()

    agent.handle("hello", Hooks(confirm=lambda _q: False))
    listener = ChatListener("w1", Chat().take, allowed=_any_peer)
    listener.start()  # forgets what the probe saw
    try:
        agent.handle("hello again", Hooks(confirm=lambda _q: False))
    finally:
        listener.close()

    assert [handoff.NAME in names for names in model.offered] == [False, True]


def _reply_of(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def test_a_hand_off_the_window_may_have_taken_is_neither_resent_nor_done_by_the_model() -> None:
    audit = AuditLog(None)

    def deliver(_words: str) -> Sent:
        raise UncertainOutcomeError("the window did not answer")

    model = _Model([_call(handoff.NAME), _reply_of("I can't tell if it arrived.")])
    agent = Agent(
        model,
        Registry([handoff.operation(deliver), no_action_operation()]),
        "system",
        audit,
        {},
    )

    reply = agent.handle(SPOKEN, Hooks(confirm=lambda _q: False))

    assert reply == "I can't tell if it arrived."
    assert any(e["event"] == "tool_uncertain" for e in audit.events)


def test_a_request_that_already_tried_an_action_cannot_be_handed_over() -> None:
    sent: list[str] = []
    play = Operation(
        "play_probe",
        "Play.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        lambda _a: Result("Playing."),
    )
    model = _Model([_call("play_probe"), _call(handoff.NAME), _reply_of("The music is on.")])
    op = handoff.operation(lambda said: sent.append(said) or Sent(State.STARTED))
    agent = Agent(model, Registry([play, op, no_action_operation()]), "system", AuditLog(None), {})

    agent.handle("play jazz and add tests in the chat", Hooks(confirm=lambda _q: False))

    assert sent == []


def test_the_tool_counts_as_the_requests_action() -> None:
    assert handoff.operation().side_effect


def test_the_spoken_line_names_the_newest_window_when_two_are_open() -> None:
    op = handoff.operation(lambda _words: Sent(State.STEERED, windows=2))
    with request_scope(foreign=False, words=SPOKEN):
        result = op.run({})

    assert result.spoken == "Added to what the chat is working on, in the newest chat window."


def test_a_handed_over_request_is_audited_with_its_source() -> None:
    audit = AuditLog(None)
    agent = Agent(_Model([_reply_of("Done.")]), Registry([no_action_operation()]), "s", audit, {})

    agent.handle(SPOKEN, Hooks(confirm=lambda _q: False), spoken=True)
    agent.handle("typed", Hooks(confirm=lambda _q: False))

    requests = [e for e in audit.events if e["event"] == "request"]
    assert requests[0]["source"] == "spoken"
    assert "source" not in requests[1]


def _audited() -> list[dict[str, Any]]:
    lines = audit_file().read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_the_chat_command_serves_hand_offs_and_audits_them_before_they_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = _opened_chat(monkeypatch)
    del opened
    taken: list[str] = []
    answers: list[Sent] = []

    def take(_self: object, words: str) -> State:
        taken.append(words)
        assert [e["event"] for e in _audited()][-1] == "chat_handoff_received"  # before it starts
        return State.STARTED

    def run(*_a: object, **_k: object) -> None:
        answers.append(handoff.send(SPOKEN))
        assert handoff.chat_open()

    monkeypatch.setattr("jarvis.chat.app.ChatApp.take_handoff", take)
    monkeypatch.setattr("jarvis.chat.app.ChatApp.run", run)
    monkeypatch.setattr(handoff, "_from_the_voice_service", _any_peer)
    handoff.forget_chat_probe()

    assert main(["chat"]) == 0

    assert taken == [SPOKEN]
    assert answers[0].state is State.STARTED
    events = [e for e in _audited() if e["event"].startswith("chat_handoff")]
    assert [e["event"] for e in events] == ["chat_handoff_received", "chat_handoff_placed"]
    assert all(e["via"] == "chat" and e["session"] for e in events)
    assert events[0]["text"] == SPOKEN
    assert not handoff.chat_open()  # the window's socket goes with it


def test_the_chat_command_refuses_and_audits_a_hand_off_that_is_not_the_voice_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusals: list[str] = []

    def run(*_a: object, **_k: object) -> None:
        try:
            handoff.send(SPOKEN)
        except HandoffError as exc:
            refusals.append(str(exc))

    monkeypatch.setattr("jarvis.chat.app.ChatApp.run", run)
    monkeypatch.setattr(handoff, "_from_the_voice_service", _no_peer)
    handoff.forget_chat_probe()

    assert main(["chat"]) == 0

    assert refusals
    assert [e["event"] for e in _audited() if e["event"].startswith("chat_handoff")] == [
        "chat_handoff_refused"
    ]


def test_the_chat_says_when_a_pause_keeps_the_model_loaded_for_its_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def active(_unit: str) -> str:
        return "active"

    def strata(_cfg: Config) -> SimpleNamespace:
        return SimpleNamespace(status=lambda: "loaded")

    def paused(*_a: object, **_k: object) -> dict[str, object]:
        return {"ok": True, "paused": True}

    monkeypatch.setattr(cli, "_unit_state", active)
    monkeypatch.setattr(cli, "_strata", strata)
    monkeypatch.setattr(control, "send", paused)

    line = cli._chat_status(Config())  # pyright: ignore[reportPrivateUsage] - the toolbar's line

    assert "paused: it unloads when no request is running" in line


# The second review


def test_a_pause_during_a_warm_up_neither_deadlocks_nor_holds_requests_off() -> None:
    """The unload and the warm-up take the model's turn before the modes' lock, both ways."""
    inside = threading.Event()
    release = threading.Event()
    model = Model()

    def prime() -> None:  # as `Agent.prime` is: a model call under the shared Answering mark
        inside.set()
        release.wait(5)
        with modes.Answering():
            pass

    controls = control.Controls(Loop(), Mic(), model, prime=prime)
    controls.warm_at_start()
    assert inside.wait(5)  # the warm-up holds the model's turn and is about to read its prompt
    controls("pause")  # its unload waits for the turn, not holding the lock a request needs
    threading.Event().wait(0.3)
    entered = threading.Event()

    def request() -> None:
        with modes.Answering():
            entered.set()

    threading.Thread(target=request, daemon=True).start()
    assert entered.wait(2)  # no request is held off while the model loads

    release.set()
    for _ in range(500):
        if model.unloads:
            break
        threading.Event().wait(0.01)

    assert model.unloads == 1  # and the warm-up finished, so the pause got its unload


def test_the_voice_service_starts_its_watcher_for_the_restart_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[object] = []
    ready = threading.Event()

    def watcher(loop: object, stop: threading.Event) -> None:
        started.append(loop)
        assert not stop.is_set()
        ready.set()

    monkeypatch.setattr(cli, "_restart_when_pending", watcher)

    def body(run: test_cli._Run, _stop: object) -> None:  # pyright: ignore[reportPrivateUsage]
        assert ready.wait(5)
        run.handlers[signal.SIGTERM](signal.SIGTERM, None)

    _run, code = _cmd_run(monkeypatch, body)

    assert code == 0
    assert len(started) == 1


def test_a_note_left_before_the_start_is_dropped_before_the_config_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note = paths.restart_pending_file()
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("x", encoding="utf-8")
    seen: list[bool] = []

    def load() -> Config:
        seen.append(note.exists())
        raise SystemExit(0)  # enough: the order is what is under test

    monkeypatch.setattr(cli, "_load_client", load)

    with pytest.raises(SystemExit):
        cli.cmd_run(SimpleNamespace())  # pyright: ignore[reportArgumentType] - the command takes no arguments

    assert seen == [False]


def test_a_restart_for_the_note_is_not_blocking_and_the_note_goes_once_it_was_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, bool]] = []
    notes: list[bool] = []

    def restart(**kwargs: bool) -> bool:
        calls.append(kwargs)
        notes.append(paths.restart_pending_file().exists())
        return True

    monkeypatch.setattr(cli, "_restart_if_running", restart)
    monkeypatch.setattr(cli, "PENDING_CHECK_S", 0.01)
    paths.restart_pending_file().parent.mkdir(parents=True, exist_ok=True)
    paths.restart_pending_file().write_text("x", encoding="utf-8")
    stop = threading.Event()
    watcher = threading.Thread(
        target=cli._restart_when_pending,  # pyright: ignore[reportPrivateUsage] - the voice service's watcher
        args=(SimpleNamespace(state="idle"), stop),
    )
    watcher.start()
    for _ in range(300):
        if calls:
            break
        threading.Event().wait(0.01)
    stop.set()
    watcher.join(5)

    assert calls == [{"no_block": True}]  # the unit's own SIGTERM must not read as a failure
    assert notes == [True]  # the note is still there when the restart is asked for
    assert not paths.restart_pending_file().exists()


def test_a_restart_that_was_refused_keeps_the_note(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_restart_if_running", _restart_refused)
    monkeypatch.setattr(cli, "PENDING_CHECK_S", 0.01)
    paths.restart_pending_file().parent.mkdir(parents=True, exist_ok=True)
    paths.restart_pending_file().write_text("x", encoding="utf-8")
    stop = threading.Event()
    watcher = threading.Thread(
        target=cli._restart_when_pending,  # pyright: ignore[reportPrivateUsage] - the voice service's watcher
        args=(SimpleNamespace(state="idle"), stop),
    )
    watcher.start()
    threading.Event().wait(0.1)
    stop.set()
    watcher.join(5)

    assert paths.restart_pending_file().exists()


def test_a_failure_in_the_watcher_is_logged_and_it_goes_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    attempts: list[bool] = []

    def restart(**_k: bool) -> bool:
        attempts.append(True)
        raise OSError("systemctl is gone")

    monkeypatch.setattr(cli, "_restart_if_running", restart)
    monkeypatch.setattr(cli, "PENDING_CHECK_S", 0.01)
    paths.restart_pending_file().parent.mkdir(parents=True, exist_ok=True)
    paths.restart_pending_file().write_text("x", encoding="utf-8")
    stop = threading.Event()
    watcher = threading.Thread(
        target=cli._restart_when_pending,  # pyright: ignore[reportPrivateUsage] - the voice service's watcher
        args=(SimpleNamespace(state="idle"), stop),
    )
    watcher.start()
    for _ in range(300):
        if len(attempts) >= 2:
            break
        threading.Event().wait(0.01)
    stop.set()
    watcher.join(5)

    assert len(attempts) >= 2  # it tried again: the thread did not die
    assert "systemctl is gone" in caplog.text


def test_words_handed_over_and_queued_run_later_still_marked_as_spoken() -> None:
    release = threading.Event()

    def answer(text: str, _hooks: Hooks) -> str:
        if text == "pause":
            release.wait(5)
        return "Done."

    session = FakeSession(answer)
    app = _app(session=session, settings=ChatConfig(while_working="queue"))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: session.asked == ["pause"])
        assert await _handed(app) is State.QUEUED
        release.set()
        await _until(pilot, lambda: session.asked == ["pause", SPOKEN])

    _run(test, app)
    assert session.spoken == [False, True]


def test_words_handed_over_to_a_request_that_ended_before_taking_them_stay_spoken() -> None:
    release = threading.Event()

    def answer(text: str, _hooks: Hooks) -> str:
        if text == "pause":
            release.wait(5)  # ends without taking the steering
        return "Done."

    session = FakeSession(answer)
    app = _app(session=session)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: session.asked == ["pause"])
        assert await _handed(app) is State.STEERED
        release.set()
        await _until(pilot, lambda: session.asked == ["pause", SPOKEN])

    _run(test, app)
    assert session.spoken == [False, True]


def test_a_window_that_answers_with_a_state_nobody_knows_has_the_words_all_the_same() -> None:
    path = handoff.socket_path("odd")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(str(path))
        sock.listen(1)

        def answer() -> None:
            while True:  # the first connection is the probe for an open window
                conn, _ = sock.accept()
                with conn:
                    if conn.recv(65536):
                        conn.sendall(b'{"ok": true, "state": "sideways"}\n')
                        return

        threading.Thread(target=answer, daemon=True).start()
        with pytest.raises(UncertainOutcomeError):
            handoff.send(SPOKEN)


def test_control_characters_in_the_transcript_cannot_blow_up_the_line() -> None:
    chat = Chat()
    listener = ChatListener("w1", chat.take, allowed=_any_peer)
    listener.start()
    try:
        handoff.send("do this" + "\x01" * handoff.MAX_CHARS)
    finally:
        listener.close()

    assert chat.words == ["do this"]  # the controls became spaces and the line stayed short


def test_a_line_that_is_too_long_is_audited_after_the_peer_is_known() -> None:
    refused: list[str] = []
    listener = ChatListener("w1", Chat().take, allowed=_any_peer, refused=refused.append)
    listener.start()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.connect(str(handoff.socket_path("w1")))
            conn.sendall(b"x" * (handoff.MAX_LINE + 10))
            reply = conn.recv(4096)
    finally:
        listener.close()

    assert b"too long" in reply
    assert refused == ["too long"]


def test_a_probe_is_not_a_refusal() -> None:
    refused: list[str] = []
    listener = ChatListener("w1", Chat().take, allowed=_no_peer, refused=refused.append)
    listener.start()
    try:
        assert handoff.chat_open()
    finally:
        listener.close()

    assert refused == []
