"""The voice service and the chat as two channels that cooperate (ADR 0028).

A spoken request is the voice agent's own and never shows in a chat window; `chat_hand_off` passes
the owner's spoken words to the open window as their own input; neither channel cuts the other's
work except where the owner's words call for it (a pause, a mode switch).
"""

import asyncio
import json
import os
import socket
import stat
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
import test_chat
from test_chat import FakeSession, FakeTail
from test_control import Loop, Mic, Model

from jarvis import cli, control, handoff, modes
from jarvis.agent import Agent, Hooks
from jarvis.app import build, no_action_operation
from jarvis.audit import AuditLog
from jarvis.config import ChatConfig, Config, SpotifyConfig
from jarvis.handoff import ChatListener, HandoffError, State
from jarvis.operations import Operation, OperationError, Registry, Result, request_scope
from jarvis.paths import audit_file, strata_key_file
from jarvis.secretfile import ensure_api_key

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from textual.pilot import Pilot

    from jarvis.chat.app import ChatApp

# The chat tests' own fixtures for a window, a transcript and a typed line.
_app = test_chat._app  # pyright: ignore[reportPrivateUsage]
_has = test_chat._has  # pyright: ignore[reportPrivateUsage]
_run = test_chat._run  # pyright: ignore[reportPrivateUsage]
_type = test_chat._type  # pyright: ignore[reportPrivateUsage]
_until = test_chat._until  # pyright: ignore[reportPrivateUsage]

SPOKEN = "add a test for the parser to what you are doing"


def _not_ours(_pid: int) -> bool:
    """The test is the sender and the window at once, so no peer is a program the window started."""
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
    listener = ChatListener("w1", chat.take, started_by_jarvis=_not_ours)
    listener.start()
    yield chat
    listener.close()


# The socket


def test_an_open_chat_takes_spoken_words_and_says_what_it_did(window: Chat) -> None:
    window.answer = State.STEERED

    assert handoff.chat_open()
    assert handoff.send(SPOKEN) is State.STEERED
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


def test_a_socket_left_by_a_window_that_was_killed_is_not_an_open_chat() -> None:
    path = handoff.socket_path("gone")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(str(path))  # the file stays when the process dies; nobody listens on it

    assert path.exists()
    assert not handoff.chat_open()
    assert not path.exists()


def test_a_refusal_by_the_window_reaches_the_sender(window: Chat) -> None:
    window.answer = HandoffError("the chat cannot take requests: no key")

    with pytest.raises(HandoffError, match="no key"):
        handoff.send(SPOKEN)


def test_with_two_windows_the_newest_gets_the_words(window: Chat) -> None:
    second = Chat()
    listener = ChatListener("w2", second.take, started_by_jarvis=_not_ours)
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

    listener = ChatListener("w1", Chat().take, started_by_jarvis=_not_ours)
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
    op = handoff.operation(lambda said: sent.append(said) or State.STARTED)
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
        run = handoff.operation(lambda said: sent.append(said) or State.STARTED).run
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
    op = handoff.operation(lambda said: sent.append(said) or State.STARTED)
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
    op = handoff.operation(lambda said: sent.append(said) or State.QUEUED)

    with request_scope(foreign=False, words=SPOKEN) as request:
        request.saw("memory")
        result = op.run({})

    assert sent == [SPOKEN]
    assert result.spoken == "Queued in the chat."


def test_a_window_that_cannot_be_reached_is_told_to_the_model() -> None:
    def deliver(_words: str) -> State:
        raise HandoffError("no chat window is open")

    op = handoff.operation(deliver)
    with request_scope(foreign=False, words=SPOKEN), pytest.raises(OperationError, match="no chat"):
        op.run({})


@dataclass
class _Model:
    replies: list[dict[str, Any]]
    calls: int = 0
    loaded: bool = True

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **_: object
    ) -> dict[str, Any]:
        del messages, tools
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
    op = handoff.operation(lambda said: sent.append(said) or State.STEERED)
    agent = Agent(model, Registry([op, no_action_operation()]), "system", AuditLog(None), {})
    spoken: list[str] = []

    reply = agent.handle(SPOKEN, Hooks(confirm=lambda _q: False, speak=spoken.append))

    assert reply == "Added to what the chat is working on."
    assert model.calls == 1  # no second model call: the request ended
    assert sent == [SPOKEN]


def test_a_failed_hand_off_does_not_end_the_request() -> None:
    def deliver(_words: str) -> State:
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


async def _handed(app: ChatApp, words: str = SPOKEN) -> State:
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


def test_a_voice_request_never_shows_in_the_open_chat_but_typed_and_ask_still_do() -> None:
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


# Neither channel cuts the other


def test_a_pause_frees_the_model_only_once_a_chat_request_has_ended() -> None:
    answering = [True, True, True]  # a chat window's request, for three looks
    sleeps = 0
    model = Model()
    work: list[Callable[[], object]] = []

    def asked() -> bool:
        return answering.pop(0) if answering else False

    def sleep(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        assert model.unloads == 0

    controls = control.Controls(
        Loop(), Mic(), model, background=work.append, answering=asked, sleep=sleep
    )
    controls("pause")
    work.pop()()

    assert sleeps == 3
    assert model.unloads == 1


def test_a_resume_while_a_chat_request_runs_drops_the_waiting_unload() -> None:
    model, mic = Model(), Mic()
    work: list[Callable[[], object]] = []

    def sleep(_seconds: float) -> None:
        controls("resume")

    controls = control.Controls(
        Loop(), mic, model, background=work.append, answering=lambda: True, sleep=sleep
    )
    controls("pause")
    work.pop(0)()

    assert model.unloads == 0


def test_any_process_answering_is_seen_through_the_modes_lock() -> None:
    assert control._anyone_answering() is False  # pyright: ignore[reportPrivateUsage] - the default look
    with modes.Answering():
        assert control._anyone_answering() is True  # pyright: ignore[reportPrivateUsage] - the default look


def test_a_setting_changed_in_the_chat_waits_for_a_spoken_request_before_restarting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restarted = threading.Event()
    looks = iter([True, True, False])  # the first look is the chat's own, the rest the waiting

    def busy() -> bool:
        return next(looks, False)

    monkeypatch.setattr(cli, "voice_busy", busy)
    monkeypatch.setattr(cli, "_restart_if_running", lambda: restarted.set() or True)

    said = cli._apply_from_chat(lambda _s: None)  # pyright: ignore[reportPrivateUsage] - the chat's hook

    assert said == cli.APPLIES_WHEN_IDLE
    assert restarted.wait(5)


def test_a_setting_changed_in_the_chat_restarts_at_once_when_nothing_is_spoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restarts: list[bool] = []
    monkeypatch.setattr(cli, "voice_busy", lambda: False)
    monkeypatch.setattr(cli, "_restart_if_running", lambda: restarts.append(True) or True)

    assert not cli._apply_from_chat()  # pyright: ignore[reportPrivateUsage] - the chat's hook
    assert restarts == [True]


def test_a_queued_restart_is_dropped_behind_a_request_that_never_ends() -> None:
    restarts: list[bool] = []
    slept: list[float] = []

    cli._restart_when_idle(  # pyright: ignore[reportPrivateUsage] - the queued restart
        lambda: True, lambda: restarts.append(True) or True, slept.append
    )

    assert restarts == []
    assert sum(slept) >= cli.IDLE_GIVE_UP_S


def test_a_program_the_chat_started_cannot_hand_it_words() -> None:
    chat = Chat()
    listener = ChatListener("w9", chat.take, started_by_jarvis=lambda _pid: True)
    listener.start()
    try:
        with pytest.raises(HandoffError, match="not from a program"):
            handoff.send(SPOKEN)
    finally:
        listener.close()

    assert chat.words == []


def test_only_a_refused_or_missing_socket_counts_as_a_dead_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect = handoff._connect  # pyright: ignore[reportPrivateUsage] - the probe's verdict
    folder = handoff.chats_dir()
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)

    assert connect(folder / "none.sock", 0.1) == (None, True)

    class Slow:
        timeout = TimeoutError

        def settimeout(self, _s: float) -> None:
            pass

        def connect(self, _path: str) -> None:
            raise self.timeout

        def close(self) -> None:
            pass

    def slow(*_args: object) -> Slow:
        return Slow()

    monkeypatch.setattr(handoff.socket, "socket", slow)

    assert connect(folder / "slow.sock", 0.1) == (None, False)


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
    op = handoff.operation(lambda said: sent.append(said) or State.STARTED)
    agent = Agent(model, Registry([op, other, no_action_operation()]), "system", AuditLog(None), {})

    reply = agent.handle(SPOKEN, Hooks(confirm=lambda _q: False))

    assert model.calls == 2
    assert "40" in reply
