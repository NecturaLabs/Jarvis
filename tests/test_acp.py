"""`jarvis acp`: the Agent Client Protocol front end, driven by a fake client over pipes."""

import base64
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Self

import pytest
from PIL import Image
from test_pictures import claimed_png

from jarvis import acp
from jarvis.acp import AcpServer, StartError
from jarvis.agent import Declined, Hooks
from jarvis.jsonutil import obj

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from jarvis.pictures import Picture

WAIT_S = 10.0


class FakeAgent:
    """An agent that runs a script for each request and keeps what it was asked."""

    def __init__(self, script: Callable[[str, Hooks, Sequence[Picture]], str]) -> None:
        super().__init__()
        self.script = script
        self.calls: list[tuple[str, list[Picture]]] = []

    def handle(self, text: str, hooks: Hooks, *, pictures: Sequence[Picture] = ()) -> str:
        self.calls.append((text, list(pictures)))
        return self.script(text, hooks, pictures)


class Client:
    """The other end of the pipes: sends lines, collects every line the server writes."""

    def __init__(
        self,
        agent: FakeAgent | None = None,
        *,
        version: str = "",
        question_s: float = acp.QUESTION_S,
    ) -> None:
        super().__init__()
        self.agent = agent or FakeAgent(lambda _t, _h, _p: "ok")
        self.made: list[str] = []
        read_in, write_in = os.pipe()
        read_out, write_out = os.pipe()
        self._to_server = os.fdopen(write_in, "w", encoding="utf-8")
        self._from_server = os.fdopen(read_out, "r", encoding="utf-8")
        self.raw: list[str] = []
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self.seen: list[dict[str, Any]] = []
        self._out = os.fdopen(write_out, "w", encoding="utf-8")
        self._source = os.fdopen(read_in, "r", encoding="utf-8")
        self.server = AcpServer(self._make, self._out, version=version, question_s=question_s)
        self._serving = threading.Thread(
            target=self.server.serve, args=(self._source,), daemon=True
        )
        self._reading = threading.Thread(target=self._read, daemon=True)
        self._serving.start()
        self._reading.start()
        self._next = 0

    def _make(self, session_id: str) -> FakeAgent:
        self.made.append(session_id)
        return self.agent

    def _read(self) -> None:
        for line in self._from_server:
            self.raw.append(line)
            self.messages.put(json.loads(line))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._to_server.close()
        self._serving.join(WAIT_S)
        assert not self._serving.is_alive()
        self._out.close()  # the client's reader sees the end
        self._reading.join(WAIT_S)
        self._source.close()
        self._from_server.close()

    def line(self, text: str) -> None:
        self._to_server.write(text + "\n")
        self._to_server.flush()

    def send(self, message: dict[str, Any]) -> None:
        self.line(json.dumps({"jsonrpc": "2.0", **message}))

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.send({"method": method, "params": params})

    def call(self, method: str, params: dict[str, Any]) -> int:
        self._next += 1
        self.send({"id": self._next, "method": method, "params": params})
        return self._next

    def next(self, timeout: float = WAIT_S) -> dict[str, Any]:
        message = self.messages.get(timeout=timeout)
        self.seen.append(message)
        return message

    def response(self, request_id: int) -> dict[str, Any]:
        """Return the answer to `request_id`; what came before it stays in `seen`."""
        while True:
            message = self.next()
            if message.get("id") == request_id:
                return message

    def ask(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return self.response(self.call(method, params))

    def start(self) -> str:
        self.ask("initialize", {"protocolVersion": 1})
        return str(
            obj(self.ask("session/new", {"cwd": "/", "mcpServers": []})["result"])["sessionId"]
        )

    def prompt(self, session: str, *blocks: dict[str, Any]) -> int:
        return self.call("session/prompt", {"sessionId": session, "prompt": list(blocks)})

    def said(self) -> str:
        """Return all the text of the reply chunks seen so far."""
        return "".join(
            obj(obj(m["params"])["update"])["content"]["text"]
            for m in self.seen
            if m.get("method") == "session/update"
        )


def text(words: str) -> dict[str, Any]:
    return {"type": "text", "text": words}


def png_block(size: int = 8) -> dict[str, Any]:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (200, 10, 10)).save(buffer, "PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image", "data": data, "mimeType": "image/png"}


def until(check: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + WAIT_S
    while not check():
        assert time.monotonic() < deadline, f"never: {what}"
        time.sleep(0.01)


# The handshake


@pytest.mark.parametrize(("asked", "answered"), [(1, 1), (2, 1), (0, 1)])
def test_initialize_answers_the_version_it_supports_or_else_its_latest(
    asked: int, answered: int
) -> None:
    with Client(version="9.9") as client:
        answer = client.ask("initialize", {"protocolVersion": asked})["result"]

    assert answer["protocolVersion"] == answered
    assert answer["authMethods"] == []
    assert answer["agentInfo"]["version"] == "9.9"
    capabilities = answer["agentCapabilities"]
    assert capabilities["loadSession"] is False
    assert capabilities["promptCapabilities"] == {
        "image": True,
        "audio": False,
        "embeddedContext": False,
    }


def test_initialize_without_a_version_is_invalid_params() -> None:
    with Client() as client:
        answer = client.ask("initialize", {})

    assert answer["error"]["code"] == acp.INVALID_PARAMS


def test_a_new_session_gets_an_id_and_an_agent_of_its_own() -> None:
    with Client() as client:
        first = client.start()
        second = obj(
            client.ask("session/new", {"cwd": "/work", "mcpServers": [{"x": 1}]})["result"]
        )

        assert first != second["sessionId"]
        assert client.made == [first, second["sessionId"]]


def test_a_session_that_cannot_start_says_why() -> None:
    def refuse(_session: str) -> FakeAgent:
        msg = "no Strata API key; run `jarvis setup`"
        raise StartError(msg)

    with Client() as client:
        client.server._make_agent = refuse  # pyright: ignore[reportPrivateUsage] - the agent's factory
        answer = client.ask("session/new", {"cwd": "/", "mcpServers": []})

    assert answer["error"]["code"] == acp.INTERNAL_ERROR
    assert "jarvis setup" in answer["error"]["message"]


# A prompt


def test_a_reply_streams_as_chunks_and_then_the_prompt_ends() -> None:
    def script(_text: str, hooks: Hooks, _pictures: Sequence[Picture]) -> str:
        assert hooks.draft is not None
        hooks.draft("It is noon.")
        hooks.draft("It is noon. Sunny.")
        return "It is noon. Sunny. Take a coat."

    with Client(FakeAgent(script)) as client:
        session = client.start()
        done = client.response(client.prompt(session, text("what time is it")))

        updates = [m for m in client.seen if m.get("method") == "session/update"]
        assert done["result"] == {"stopReason": "end_turn"}
        assert client.seen.index(done) > client.seen.index(updates[-1])  # chunks first
        assert [obj(obj(u["params"])["update"])["content"]["text"] for u in updates] == [
            "It is noon.",
            " Sunny.",
            " Take a coat.",
        ]
        assert {obj(u["params"])["sessionId"] for u in updates} == {session}
        assert obj(obj(updates[0]["params"])["update"])["sessionUpdate"] == "agent_message_chunk"
        assert client.agent.calls == [("what time is it", [])]


def test_a_reply_that_was_not_streamed_is_written_whole() -> None:
    with Client(FakeAgent(lambda _t, _h, _p: "Done.")) as client:
        session = client.start()
        client.response(client.prompt(session, text("go")))

        assert client.said() == "Done."


def test_a_reply_that_differs_from_its_draft_follows_it_set_apart() -> None:
    def script(_text: str, hooks: Hooks, _pictures: Sequence[Picture]) -> str:
        assert hooks.draft is not None
        hooks.draft("Checking.")
        return "It is raining."

    with Client(FakeAgent(script)) as client:
        session = client.start()
        client.response(client.prompt(session, text("weather")))

        assert client.said() == "Checking.\n\nIt is raining."


def test_an_image_block_reaches_the_agent_as_a_checked_picture() -> None:
    with Client() as client:
        session = client.start()
        client.response(client.prompt(session, text("what is this"), png_block(16)))

        (words, pictures) = client.agent.calls[0]
        assert words == "what is this"
        assert len(pictures) == 1
        assert pictures[0].mime == "image/png"
        assert (pictures[0].width, pictures[0].height) == (16, 16)


def test_a_picture_without_words_is_still_a_request() -> None:
    with Client() as client:
        session = client.start()
        done = client.response(client.prompt(session, png_block()))

        assert done["result"] == {"stopReason": "end_turn"}
        assert len(client.agent.calls[0][1]) == 1


def refused_blocks() -> list[tuple[str, list[dict[str, Any]], str]]:
    big = {"type": "image", "mimeType": "image/png", "data": "A" * (acp.MAX_BASE64 + 1)}
    huge = base64.b64encode(claimed_png(20_000, 20_000)).decode("ascii")
    return [
        ("too big", [big], "too big"),
        ("not a picture", [{**png_block(), "data": base64.b64encode(b"hello").decode()}], "isn't"),
        ("not base64", [{**png_block(), "data": "!!not base64!!"}], "not valid"),
        ("wrong type", [{**png_block(), "mimeType": "image/svg+xml"}], "PNG, JPEG"),
        ("no data", [{"type": "image", "mimeType": "image/png"}], "no data"),
        (
            "too many pixels",
            [{"type": "image", "mimeType": "image/png", "data": huge}],
            "too large",
        ),
        ("too many pictures", [png_block() for _ in range(acp.MAX_IMAGES + 1)], "at most"),
        ("audio", [{"type": "audio", "data": "", "mimeType": "audio/wav"}], "audio"),
        ("a link", [{"type": "resource_link", "uri": "file:///x", "name": "x"}], "resource_link"),
        ("empty", [], "nothing"),
    ]


@pytest.mark.parametrize(
    ("blocks", "said"),
    [(b, s) for _n, b, s in refused_blocks()],
    ids=[n for n, _b, _s in refused_blocks()],
)
def test_a_prompt_that_cannot_be_read_is_refused_and_never_reaches_the_agent(
    blocks: list[dict[str, Any]], said: str
) -> None:
    with Client() as client:
        session = client.start()
        done = client.response(client.prompt(session, *blocks))

        assert done["result"] == {"stopReason": "refusal"}
        assert said in client.said()
        assert client.agent.calls == []
        # The session is not left busy by a refusal.
        assert client.response(client.prompt(session, text("hi")))["result"]["stopReason"] == (
            "end_turn"
        )


def test_cancel_stops_the_request_and_the_prompt_ends_cancelled() -> None:
    started = threading.Event()

    def script(_text: str, hooks: Hooks, _pictures: Sequence[Picture]) -> str:
        started.set()
        deadline = time.monotonic() + WAIT_S
        while not hooks.cancelled() and time.monotonic() < deadline:
            time.sleep(0.005)
        return ""

    with Client(FakeAgent(script)) as client:
        session = client.start()
        request = client.prompt(session, text("long job"))
        assert started.wait(WAIT_S)
        client.notify("session/cancel", {"sessionId": session})

        assert client.response(request)["result"] == {"stopReason": "cancelled"}


def test_a_prompt_while_another_runs_is_refused_as_busy() -> None:
    release = threading.Event()

    def script(_text: str, _hooks: Hooks, _pictures: Sequence[Picture]) -> str:
        release.wait(WAIT_S)
        return "late"

    with Client(FakeAgent(script)) as client:
        session = client.start()
        first = client.prompt(session, text("one"))
        until(lambda: bool(client.agent.calls), "the first request began")
        second = client.response(client.prompt(session, text("two")))
        release.set()

        assert second["error"]["code"] == acp.BUSY
        assert client.response(first)["result"] == {"stopReason": "end_turn"}


def test_a_failing_request_is_an_error_and_the_session_goes_on() -> None:
    calls: list[str] = []

    def script(_text: str, _hooks: Hooks, _pictures: Sequence[Picture]) -> str:
        calls.append("call")
        if len(calls) == 1:
            msg = "private detail"
            raise RuntimeError(msg)
        return "fine"

    with Client(FakeAgent(script)) as client:
        session = client.start()
        failed = client.response(client.prompt(session, text("a")))
        after = client.response(client.prompt(session, text("b")))

        assert failed["error"]["code"] == acp.INTERNAL_ERROR
        assert "RuntimeError" in failed["error"]["message"]
        assert "private detail" not in failed["error"]["message"]
        assert after["result"] == {"stopReason": "end_turn"}


# Consent stays Jarvis's own


def closing_script(ran: list[str]) -> Callable[[str, Hooks, Sequence[Picture]], str]:
    def script(_text: str, hooks: Hooks, _pictures: Sequence[Picture]) -> str:
        answer = hooks.confirm("Close the window named Notes?")
        if answer is True:
            ran.append("closed")
            return "Closed it."
        return "Left it open."

    return script


def test_a_question_is_asked_in_the_reply_and_the_next_prompts_yes_runs_the_action() -> None:
    ran: list[str] = []
    with Client(FakeAgent(closing_script(ran))) as client:
        session = client.start()
        asked = client.response(client.prompt(session, text("close notes")))

        assert asked["result"] == {"stopReason": "end_turn"}
        assert "Close the window named Notes?" in client.said()
        assert "yes" in client.said().lower()
        assert ran == []

        answered = client.response(client.prompt(session, text("yes")))

        assert answered["result"] == {"stopReason": "end_turn"}
        assert ran == ["closed"]
        assert client.said().endswith("Closed it.")
        assert len(client.agent.calls) == 1  # the yes was an answer, not a request


@pytest.mark.parametrize("answer", ["no", "nope, leave it", "yes, but don't", "later", ""])
def test_anything_but_a_clear_yes_cancels_the_action(answer: str) -> None:
    ran: list[str] = []
    with Client(FakeAgent(closing_script(ran))) as client:
        session = client.start()
        client.response(client.prompt(session, text("close notes")))
        client.response(client.prompt(session, text(answer)))

        assert ran == []
        assert client.said().endswith("Left it open.")
        assert len(client.agent.calls) == 1


def test_a_permission_answer_from_the_client_never_stands_for_the_owner() -> None:
    ran: list[str] = []
    with Client(FakeAgent(closing_script(ran))) as client:
        session = client.start()
        client.response(client.prompt(session, text("close notes")))
        for request_id in (7, 8):  # what T3 Code sends back by itself in its full-access mode
            client.send(
                {
                    "id": request_id,
                    "result": {"outcome": {"outcome": "selected", "optionId": "allow_once"}},
                }
            )
        time.sleep(0.2)

        assert ran == []
        assert all(m.get("method") != "session/request_permission" for m in client.seen)
        client.response(client.prompt(session, text("yes")))
        assert ran == ["closed"]


def test_the_server_never_asks_the_client_for_permission() -> None:
    ran: list[str] = []
    with Client(FakeAgent(closing_script(ran))) as client:
        session = client.start()
        client.response(client.prompt(session, text("close notes")))
        client.response(client.prompt(session, text("yes")))

    assert all("request_permission" not in line for line in client.raw)


def test_cancel_while_a_question_waits_answers_it_no_and_frees_the_session() -> None:
    ran: list[str] = []
    with Client(FakeAgent(closing_script(ran))) as client:
        session = client.start()
        client.response(client.prompt(session, text("close notes")))
        client.notify("session/cancel", {"sessionId": session})
        again = client.response(client.prompt(session, text("yes")))

        assert ran == []  # the yes after a cancel is a new request, not the answer
        assert again["result"]["stopReason"] == "end_turn"
        assert [call[0] for call in client.agent.calls] == ["close notes", "yes"]


def test_a_question_nobody_answers_is_a_no_and_the_next_prompt_is_a_request() -> None:
    ran: list[str] = []
    with Client(FakeAgent(closing_script(ran)), question_s=0.05) as client:
        session = client.start()
        client.response(client.prompt(session, text("close notes")))
        until(lambda: client.server._sessions[session].run is None, "the question timed out")  # pyright: ignore[reportPrivateUsage] - the session's state
        client.response(client.prompt(session, text("yes")))

        assert ran == []
        assert [call[0] for call in client.agent.calls] == ["close notes", "yes"]


def test_a_declined_answer_keeps_its_reason() -> None:
    seen: list[Any] = []

    def script(_text: str, hooks: Hooks, _pictures: Sequence[Picture]) -> str:
        seen.append(hooks.confirm("Do it?"))
        return "ok"

    with Client(FakeAgent(script)) as client:
        session = client.start()
        client.response(client.prompt(session, text("go")))
        client.notify("session/cancel", {"sessionId": session})
        until(lambda: bool(seen), "the question was answered")

    assert isinstance(seen[0], Declined)


# The wire


def test_an_unknown_method_is_method_not_found_and_an_unknown_notification_is_ignored() -> None:
    with Client() as client:
        client.notify("session/set_mode", {"sessionId": "x"})
        answer = client.ask("session/load", {"sessionId": "x"})
        client.ask("initialize", {"protocolVersion": 1})

    assert answer["error"]["code"] == acp.METHOD_NOT_FOUND
    assert not any("set_mode" in line for line in client.raw)


def test_a_prompt_for_an_unknown_session_is_invalid_params() -> None:
    with Client() as client:
        answer = client.ask("session/prompt", {"sessionId": "nope", "prompt": []})

    assert answer["error"]["code"] == acp.INVALID_PARAMS


def test_a_prompt_that_is_not_a_list_is_invalid_params() -> None:
    with Client() as client:
        session = client.start()
        answer = client.ask("session/prompt", {"sessionId": session, "prompt": "hello"})

    assert answer["error"]["code"] == acp.INVALID_PARAMS


def test_what_is_not_json_is_a_parse_error_and_the_connection_goes_on() -> None:
    with Client() as client:
        client.line("this is not json")
        client.line("[1, 2]")
        client.line("")
        parse = client.next()
        batch = client.next()
        later = client.ask("initialize", {"protocolVersion": 1})

    assert parse["error"]["code"] == acp.PARSE_ERROR
    assert parse["id"] is None
    assert batch["error"]["code"] == acp.INVALID_REQUEST
    assert later["result"]["protocolVersion"] == 1


def test_a_line_over_the_limit_is_refused_whole_and_the_next_line_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acp, "MAX_LINE", 200)
    with Client() as client:
        client.line(json.dumps({"id": 1, "method": "x", "params": {"pad": "z" * 1000}}))
        refused = client.next()
        later = client.ask("initialize", {"protocolVersion": 1})

    assert refused["error"]["code"] == acp.INVALID_REQUEST
    assert later["result"]["protocolVersion"] == 1


def test_everything_the_server_writes_is_one_json_rpc_message_per_line() -> None:
    def script(_text: str, hooks: Hooks, _pictures: Sequence[Picture]) -> str:
        assert hooks.draft is not None
        hooks.draft('line one\nline two with "quotes" and ünïcode')
        return 'line one\nline two with "quotes" and ünïcode, done'

    with Client(FakeAgent(script)) as client:
        session = client.start()
        client.response(client.prompt(session, text("go")))
        client.line("garbage")
        client.next()

    assert client.raw
    for line in client.raw:
        assert line.endswith("\n")
        assert "\n" not in line[:-1]
        assert json.loads(line)["jsonrpc"] == "2.0"


def test_closing_the_input_stops_a_request_in_flight() -> None:
    started = threading.Event()
    stopped = threading.Event()

    def script(_text: str, hooks: Hooks, _pictures: Sequence[Picture]) -> str:
        started.set()
        while not hooks.cancelled():
            time.sleep(0.005)
        stopped.set()
        return ""

    client = Client(FakeAgent(script))
    session = client.start()
    client.prompt(session, text("long job"))
    assert started.wait(WAIT_S)
    client.close()

    assert stopped.is_set()


# The command


def test_the_command_serves_the_protocol_on_its_own_output_only(tmp_path: Path) -> None:
    requests: list[dict[str, Any]] = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1}},
        {"jsonrpc": "2.0", "id": 2, "method": "nope", "params": {}},
    ]
    code = "import sys; from jarvis.cli import main; sys.exit(main(['acp']))"

    done = subprocess.run(
        [sys.executable, "-I", "-c", code],
        input="".join(json.dumps(r) + "\n" for r in requests),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=tmp_path,
        env=os.environ,
    )

    assert done.returncode == 0, done.stderr
    answers = [json.loads(line) for line in done.stdout.splitlines()]
    assert [a["id"] for a in answers] == [1, 2]
    assert answers[0]["result"]["agentInfo"]["name"] == "jarvis"
    assert answers[1]["error"]["code"] == acp.METHOD_NOT_FOUND
