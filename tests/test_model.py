"""The Strata client: what it sends, and that a broken answer never reaches the agent."""

import itertools
import json
import threading
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast, override

import httpx
import pytest

from jarvis.config import Config, ModelConfig
from jarvis.model import (
    DEFAULT_MAX_TOKENS,
    LOAD_TIMEOUT_S,
    WARM_RETRY_S,
    Delta,
    ModelError,
    ModelMissingError,
    StrataModel,
    Thinking,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

TOOLS: list[dict[str, Any]] = [{"type": "function", "function": {"name": "x"}}]
MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]


def _model(handler: httpx.MockTransport) -> StrataModel:
    local = ModelConfig(base_url=Config().model_base_url())
    return StrataModel(local, "key", transport=handler)


def _answering(message: object, seen: list[httpx.Request] | None = None) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": message}]})

    return httpx.MockTransport(handle)


@pytest.mark.parametrize(
    ("require", "choice"),
    [
        (True, "required"),
        (False, "auto"),
        ("no_action", {"type": "function", "function": {"name": "no_action"}}),
    ],
)
def test_chat_sends_the_key_the_tools_and_the_tool_choice(
    require: bool | str, choice: object
) -> None:
    seen: list[httpx.Request] = []
    model = _model(_answering({"role": "assistant", "content": "hello"}, seen))

    reply = model.chat(MESSAGES, TOOLS, require_tool=require)

    body = json.loads(seen[0].content)
    assert reply == {"role": "assistant", "content": "hello"}
    assert seen[0].headers["Authorization"] == "Bearer key"
    assert body["tools"] == TOOLS
    assert body["tool_choice"] == choice
    assert body["messages"] == MESSAGES


def test_chat_without_tools_sends_no_tool_choice() -> None:
    seen: list[httpx.Request] = []
    model = _model(_answering({"role": "assistant", "content": "hello"}, seen))

    model.chat(MESSAGES, [])

    body = json.loads(seen[0].content)
    assert "tools" not in body
    assert "tool_choice" not in body


@pytest.mark.parametrize("message", ["just text", None, ["a"], 7])
def test_an_answer_whose_message_is_not_an_object_is_a_model_error(message: object) -> None:
    model = _model(_answering(message))

    with pytest.raises(ModelError):
        model.chat(MESSAGES, TOOLS)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json=["no", "choices"]),
    ],
)
def test_a_refusal_or_malformed_answer_is_a_model_error(response: httpx.Response) -> None:
    model = _model(httpx.MockTransport(lambda _request: response))

    with pytest.raises(ModelError):
        model.chat(MESSAGES, TOOLS)


def test_a_dropped_connection_is_retried_once_when_the_server_is_back() -> None:
    posts: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"status": {"value": "loaded"}}]})
        posts.append(1)
        if len(posts) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "back"}}]})

    reply = _model(httpx.MockTransport(handle)).chat(MESSAGES, TOOLS)

    assert reply == {"content": "back"}
    assert len(posts) == 2


def test_status_reads_the_load_state_without_generating() -> None:
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(200, json={"data": [{"status": {"value": "unloaded"}}]})

    model = _model(httpx.MockTransport(handle))

    assert model.status() == "unloaded"
    assert model.is_loaded() is False
    assert seen == ["GET /v1/models", "GET /v1/models"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"data": ["text"]}),
        httpx.Response(200, json={"data": [{"status": "loaded"}]}),
        httpx.Response(503, text="starting"),
    ],
)
def test_an_unreadable_status_is_unknown(response: httpx.Response) -> None:
    model = _model(httpx.MockTransport(lambda _request: response))

    assert model.status() is None
    assert model.is_loaded() is None


def test_the_api_key_never_appears_in_the_repr() -> None:
    model = StrataModel(ModelConfig(), "sk-very-secret")

    assert "sk-very-secret" not in repr(model)
    assert "sk-very-secret" not in str(model)


def test_a_retry_waits_for_the_server_with_the_injected_sleep() -> None:
    gets: list[int] = []
    posts: list[int] = []
    naps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            gets.append(1)
            if len(gets) < 3:
                return httpx.Response(503, text="starting")
            return httpx.Response(200, json={"data": [{"status": {"value": "loaded"}}]})
        posts.append(1)
        if len(posts) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "back"}}]})

    local = ModelConfig(base_url=Config().model_base_url())
    model = StrataModel(local, "key", transport=httpx.MockTransport(handle), sleep=naps.append)

    assert model.chat(MESSAGES, TOOLS) == {"content": "back"}
    assert naps == [1.0, 1.0]
    assert len(gets) == 3


def _read_timeout(request: httpx.Request) -> float:
    timeout = request.extensions["timeout"]
    assert isinstance(timeout, dict)
    return float(cast("dict[str, float]", timeout)["read"])


@pytest.mark.parametrize(("asked", "sent"), [(None, 120.0), (3.5, 3.5), (999.0, 120.0)])
def test_a_call_waits_no_longer_than_the_turn_has_left(asked: float | None, sent: float) -> None:
    seen: list[httpx.Request] = []
    model = replace(_model(_answering({"content": "hi"}, seen)), clock=lambda: 0.0)

    model.chat(MESSAGES, TOOLS, timeout_s=asked)

    assert _read_timeout(seen[0]) == sent


@pytest.mark.parametrize(("asked", "sent"), [(270.0, 270.0), (999.0, 300.0), (60.0, 60.0)])
def test_a_call_that_must_load_the_model_may_wait_that_much_longer(
    asked: float, sent: float
) -> None:
    seen: list[httpx.Request] = []
    model = replace(_model(_answering({"content": "hi"}, seen)), clock=lambda: 0.0)

    model.chat(MESSAGES, TOOLS, timeout_s=asked, load_s=180.0)

    assert _read_timeout(seen[0]) == sent


def test_warming_waits_for_the_server_and_retries_a_busy_answer() -> None:
    """Regression: the load sent at service start met a server still starting (409) and was lost."""
    answers = iter([httpx.Response(409), httpx.Response(409), httpx.Response(200, json={})])
    loads: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"status": {"value": "unloaded"}}]})
        loads.append(_read_timeout(request))
        return next(answers)

    waits: list[float] = []
    model = StrataModel(
        ModelConfig(base_url=Config().model_base_url()),
        "key",
        transport=httpx.MockTransport(handle),
        sleep=waits.append,
    )

    assert model.warm() is True
    assert len(loads) == 3
    assert min(loads) >= LOAD_TIMEOUT_S  # a cold load can take minutes
    assert waits


def test_warming_gives_up_on_a_server_that_never_comes_up() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    waits: list[float] = []
    model = StrataModel(
        ModelConfig(base_url=Config().model_base_url()),
        "key",
        transport=httpx.MockTransport(handle),
        sleep=waits.append,
    )

    assert model.warm(attempts=5) is False
    assert len(waits) == 5


SYSTEM: dict[str, Any] = {"role": "system", "content": "be brief"}


def _sent(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    seen: list[httpx.Request] = []
    _model(_answering({"role": "assistant", "content": "hello"}, seen)).chat(messages, tools)
    return cast("dict[str, Any]", json.loads(seen[0].content))


def test_the_system_prompt_and_tools_are_marked_as_the_shared_prefix() -> None:
    body = _sent([SYSTEM, *MESSAGES], TOOLS)

    assert body["strata_prefix"] == {"messages": 1}


@pytest.mark.parametrize(
    ("messages", "tools"),
    [([SYSTEM, *MESSAGES], []), (MESSAGES, TOOLS), ([SYSTEM], TOOLS)],
    ids=["no tools", "no system prompt", "nothing after the system prompt"],
)
def test_no_shared_prefix_is_marked_when_it_is_not_the_usual_one(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> None:
    assert "strata_prefix" not in _sent(messages, tools)


def test_a_retry_after_waiting_for_the_server_gets_only_the_time_left() -> None:
    now = [0.0]
    posts: list[httpx.Request] = []
    gets: list[int] = []

    def nap(seconds: float) -> None:
        now[0] += seconds

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            gets.append(1)
            if len(gets) < 3:
                return httpx.Response(503, text="starting")
            return httpx.Response(200, json={"data": [{"status": {"value": "loaded"}}]})
        posts.append(request)
        if len(posts) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "back"}}]})

    local = ModelConfig(base_url=Config().model_base_url())
    model = StrataModel(
        local, "key", transport=httpx.MockTransport(handle), sleep=nap, clock=lambda: now[0]
    )

    assert model.chat(MESSAGES, TOOLS, timeout_s=10.0) == {"content": "back"}
    assert _read_timeout(posts[0]) == pytest.approx(10.0)
    assert _read_timeout(posts[1]) == pytest.approx(8.0)


def _sse(*events: object, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    return ("".join(lines) + ("data: [DONE]\n\n" if done else "")).encode()


def _chunk(delta: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta}]}


class _Breaking(httpx.SyncByteStream):
    """A response body that sends `first`, then drops the connection."""

    def __init__(self, first: bytes) -> None:
        super().__init__()
        self._first = first

    @override
    def __iter__(self) -> Iterator[bytes]:
        yield self._first
        msg = "connection reset"
        raise httpx.ReadError(msg)


FINISH: dict[str, Any] = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
CALL_EVENTS: tuple[dict[str, Any], ...] = (
    _chunk({"role": "assistant", "content": ""}),
    _chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "no_action", "arguments": ""},
                }
            ]
        }
    ),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"reply":"Hi'}}]}),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' there."}'}}]}),
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
)


def test_a_streamed_answer_is_passed_on_piece_by_piece_and_returned_whole() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=_sse(*CALL_EVENTS))

    pieces: list[Delta] = []
    reply = _model(httpx.MockTransport(handle)).chat(MESSAGES, TOOLS, on_delta=pieces.append)

    assert json.loads(seen[0].content)["stream"] is True
    assert pieces == [
        Delta(call=0, name="no_action"),
        Delta(call=0, arguments='{"reply":"Hi'),
        Delta(call=0, arguments=' there."}'),
    ]
    assert reply == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "no_action", "arguments": '{"reply":"Hi there."}'},
            }
        ],
        "finish_reason": "tool_calls",
    }


def test_streamed_text_arrives_as_text_pieces() -> None:
    events = (_chunk({"content": "Paused "}), _chunk({"content": "it."}), FINISH)
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, content=_sse(*events)))
    pieces: list[Delta] = []

    reply = _model(transport).chat(MESSAGES, TOOLS, on_delta=pieces.append)

    assert pieces == [Delta(text="Paused "), Delta(text="it.")]
    assert reply == {"role": "assistant", "content": "Paused it.", "finish_reason": "stop"}


def test_a_stream_dropped_after_a_piece_was_passed_on_is_not_retried() -> None:
    posts: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"status": {"value": "loaded"}}]})
        posts.append(1)
        return httpx.Response(200, stream=_Breaking(_sse(_chunk({"content": "Half"}), done=False)))

    pieces: list[Delta] = []
    with pytest.raises(ModelError):
        _model(httpx.MockTransport(handle)).chat(MESSAGES, TOOLS, on_delta=pieces.append)

    assert posts == [1]
    assert pieces == [Delta(text="Half")]


def test_a_stream_that_fails_before_any_piece_is_retried_once() -> None:
    posts: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"status": {"value": "loaded"}}]})
        posts.append(1)
        if len(posts) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, content=_sse(_chunk({"content": "back"}), FINISH))

    pieces: list[Delta] = []
    reply = _model(httpx.MockTransport(handle)).chat(MESSAGES, TOOLS, on_delta=pieces.append)

    assert reply == {"role": "assistant", "content": "back", "finish_reason": "stop"}
    assert len(posts) == 2


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, content=b"data: not json\n\n"),
        httpx.Response(200, content=_sse(_chunk({"tool_calls": [{"function": {"name": "x"}}]}))),
        httpx.Response(200, content=_sse(_chunk({"content": "cut off"}))),  # no finish_reason
    ],
)
def test_a_refused_or_unreadable_stream_is_a_model_error(response: httpx.Response) -> None:
    model = _model(httpx.MockTransport(lambda _request: response))

    with pytest.raises(ModelError):
        model.chat(MESSAGES, TOOLS, on_delta=lambda _piece: None)


def test_a_stream_that_runs_past_its_time_is_a_model_error() -> None:
    now = [0.0]
    events = (
        _chunk({"content": "one "}),
        _chunk({"content": "two "}),
        _chunk({"content": "x"}),
        FINISH,
    )
    local = ModelConfig(base_url=Config().model_base_url())
    model = StrataModel(
        local,
        "key",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=_sse(*events))),
        clock=lambda: now[0],
    )

    def slow(_piece: Delta) -> None:
        now[0] += 3.0

    with pytest.raises(ModelError, match="too long"):
        model.chat(MESSAGES, TOOLS, timeout_s=5.0, on_delta=slow)


class _Stalling(httpx.SyncByteStream):
    """A response body that sends `first`, then sends nothing until `release` is set."""

    def __init__(self, first: bytes, release: threading.Event) -> None:
        super().__init__()
        self._first = first
        self._release = release

    @override
    def __iter__(self) -> Iterator[bytes]:
        yield self._first
        self._release.wait(30)


def test_a_stream_that_stalls_ends_at_its_time_not_when_the_next_piece_comes() -> None:
    release = threading.Event()
    body = _Stalling(_sse(_chunk({"content": "Half"}), done=False), release)
    model = _model(httpx.MockTransport(lambda _r: httpx.Response(200, stream=body)))
    raised: list[BaseException] = []

    def call() -> None:
        try:
            model.chat(MESSAGES, TOOLS, timeout_s=0.3, on_delta=lambda _piece: None)
        except ModelError as exc:
            raised.append(exc)

    caller = threading.Thread(target=call, daemon=True)
    caller.start()
    try:
        caller.join(5)
        assert not caller.is_alive()
    finally:
        release.set()
    assert len(raised) == 1
    assert "too long" in str(raised[0])


def test_an_error_raised_by_the_listener_ends_the_stream_and_is_raised() -> None:
    class EnoughError(Exception):
        pass

    def stop(_piece: Delta) -> None:
        raise EnoughError

    transport = httpx.MockTransport(
        lambda _r: httpx.Response(200, content=_sse(_chunk({"content": "a"})))
    )
    with pytest.raises(EnoughError):
        _model(transport).chat(MESSAGES, TOOLS, on_delta=stop)


def test_each_call_carries_its_own_token_cap() -> None:
    seen: list[httpx.Request] = []
    model = _model(_answering({"role": "assistant", "content": "hello"}, seen))

    model.chat(MESSAGES, [], max_tokens=2048)
    model.chat(MESSAGES, [])

    assert json.loads(seen[0].content)["max_tokens"] == 2048
    assert json.loads(seen[1].content)["max_tokens"] == DEFAULT_MAX_TOKENS


def test_a_whole_answer_keeps_its_finish_reason_and_token_counts() -> None:
    usage = {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230}

    def handle(_request: httpx.Request) -> httpx.Response:
        choice = {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "length"}
        return httpx.Response(200, json={"choices": [choice], "usage": usage})

    reply = _model(httpx.MockTransport(handle)).chat(MESSAGES, [])

    assert reply["finish_reason"] == "length"
    assert reply["usage"] == usage


def test_a_streamed_answer_asks_for_and_keeps_the_token_counts() -> None:
    seen: list[httpx.Request] = []
    usage = {"prompt_tokens": 800, "completion_tokens": 12}
    tail: dict[str, Any] = {"choices": [], "usage": usage}

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=_sse(_chunk({"content": "ok"}), FINISH, tail))

    reply = _model(httpx.MockTransport(handle)).chat(MESSAGES, [], on_delta=lambda _d: None)

    assert json.loads(seen[0].content)["stream_options"] == {"include_usage": True}
    assert reply["usage"] == usage
    assert reply["finish_reason"] == "stop"


@pytest.mark.parametrize(
    ("header", "wait"),
    [
        ("7", 7.0),
        ("0", 1.0),
        ("999", 30.0),
        (None, WARM_RETRY_S),
        ("soon", WARM_RETRY_S),
        ("nan", WARM_RETRY_S),
    ],
)
def test_warming_waits_as_long_as_a_busy_server_asks(header: str | None, wait: float) -> None:
    """Regression: a 503 with `Retry-After` was retried after a fixed two seconds."""
    answers = iter(
        [
            httpx.Response(503, headers={"Retry-After": header} if header else {}),
            httpx.Response(200, json={}),
        ]
    )

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"status": {"value": "unloaded"}}]})
        return next(answers)

    waits: list[float] = []
    model = StrataModel(
        ModelConfig(base_url=Config().model_base_url()),
        "key",
        transport=httpx.MockTransport(handle),
        sleep=waits.append,
    )

    assert model.warm() is True
    assert waits == [wait]


def test_a_long_retry_after_does_not_stretch_the_number_of_attempts() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"status": {"value": "unloaded"}}]})
        return httpx.Response(503, headers={"Retry-After": "20"})

    waits: list[float] = []
    model = StrataModel(
        ModelConfig(base_url=Config().model_base_url()),
        "key",
        transport=httpx.MockTransport(handle),
        sleep=waits.append,
    )

    assert model.warm(attempts=3) is False
    assert waits == [20.0, 20.0, 20.0]


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def test_a_server_down_with_its_model_file_missing_fails_at_once_and_says_why(
    tmp_path: Path,
) -> None:
    # The model's drive not mounted: systemd skipped Strata, so there is nothing to wait for.
    naps: list[float] = []
    local = ModelConfig(base_url=Config().model_base_url())
    model = StrataModel(
        local,
        "key",
        transport=httpx.MockTransport(_refused),
        sleep=naps.append,
        model_file=tmp_path / "gone.gguf",
    )

    with pytest.raises(ModelMissingError, match=r"gone\.gguf"):
        model.chat(MESSAGES, TOOLS)
    assert naps == []
    assert model.file_missing()


def test_a_server_down_with_its_model_file_present_is_waited_for(tmp_path: Path) -> None:
    present = tmp_path / "model.gguf"
    present.write_bytes(b"")
    naps: list[float] = []
    local = ModelConfig(base_url=Config().model_base_url())
    model = StrataModel(
        local,
        "key",
        transport=httpx.MockTransport(_refused),
        sleep=naps.append,
        clock=_ticking(),
        model_file=present,
    )

    with pytest.raises(ModelError) as error:
        model.chat(MESSAGES, TOOLS, timeout_s=3.0)
    assert not isinstance(error.value, ModelMissingError)
    assert naps  # it waited for the server to come back
    assert not model.file_missing()


def test_without_a_model_file_none_is_missing() -> None:
    assert not _model(httpx.MockTransport(_refused)).file_missing()


def _ticking() -> Callable[[], float]:
    now = [0.0]

    def clock() -> float:
        now[0] += 0.5
        return now[0]

    return clock


def _loading_server(
    listed: dict[str, Any], *, release: threading.Event, started: threading.Event
) -> httpx.MockTransport:
    """Answer `/models` with `listed` and hold `/load` until `release` is set, as Strata does."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=listed)
        started.set()
        release.wait(5)
        return httpx.Response(200, json={"status": "loaded"})

    return httpx.MockTransport(handle)


UNLOADED: dict[str, Any] = {"data": [{"status": {"value": "unloaded"}}]}


@pytest.mark.parametrize(
    ("how", "listed", "after"),
    [
        ("warm", UNLOADED, "unloaded"),
        ("load", UNLOADED, "unloaded"),
        ("load", {"data": []}, "unloaded"),  # no idle unload: no model listed while it loads
    ],
)
def test_a_load_jarvis_asked_for_shows_as_loading_until_it_ends(
    how: str, listed: dict[str, Any], after: str | None
) -> None:
    """Regression: Strata reports only loaded or unloaded, so a 2-minute load showed `unloaded`."""
    release, started = threading.Event(), threading.Event()
    model = _model(_loading_server(listed, release=release, started=started))
    assert model.status() == after

    asking = threading.Thread(target=getattr(model, how))
    asking.start()
    assert started.wait(5)
    try:
        assert model.status() == "loading"
        assert model.is_loaded() is False
    finally:
        release.set()
        asking.join(5)

    assert model.status() == after


def test_a_load_waits_as_long_as_strata_lets_an_engine_start() -> None:
    """Strata tries an engine start up to 3 times, 900 s each; the load request outwaits them."""
    loads: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=UNLOADED)
        loads.append(_read_timeout(request))
        return httpx.Response(200, json={"status": "loaded"})

    model = _model(httpx.MockTransport(handle))

    assert model.load() is True
    assert model.warm() is True
    assert LOAD_TIMEOUT_S > 3 * 900
    assert loads == [LOAD_TIMEOUT_S, LOAD_TIMEOUT_S]


def test_a_load_whose_answer_never_came_is_not_shown_as_loading_after() -> None:
    """Past the wait Jarvis no longer knows; it shows what the server says, never a guess."""

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=UNLOADED)
        raise httpx.ReadTimeout("no answer", request=request)

    model = _model(httpx.MockTransport(handle))

    assert model.warm() is True
    assert model.status() == "unloaded"


def test_warming_loads_a_server_that_lists_no_model_yet() -> None:
    """Regression: a reachable Strata listing no model (no idle unload) was never sent the load."""
    release, started = threading.Event(), threading.Event()
    model = StrataModel(
        ModelConfig(base_url=Config().model_base_url()),
        "key",
        transport=_loading_server({"data": []}, release=release, started=started),
        sleep=lambda _s: None,
    )

    warming = threading.Thread(target=model.warm, kwargs={"attempts": 3})
    warming.start()
    try:
        assert started.wait(5)
        assert model.status() == "loading"
    finally:
        release.set()
        warming.join(5)


def test_a_load_still_going_keeps_the_state_loading_when_another_ends() -> None:
    gates = [threading.Event(), threading.Event()]
    asked = itertools.count()
    both = threading.Barrier(3)
    ended = threading.Semaphore(0)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=UNLOADED)
        gate = gates[next(asked)]
        both.wait(5)
        gate.wait(5)
        return httpx.Response(200, json={})

    model = _model(httpx.MockTransport(handle))

    def load() -> None:  # the start-up load and a resume's, both under way
        model.warm()
        ended.release()

    threads = [threading.Thread(target=load) for _ in gates]
    for thread in threads:
        thread.start()
    both.wait(5)
    gates[0].set()
    assert ended.acquire(timeout=5)
    try:
        assert model.status() == "loading"
    finally:
        gates[1].set()
        for thread in threads:
            thread.join(5)
    assert model.status() == "unloaded"


def test_an_unreachable_server_is_unknown_even_while_a_load_was_asked_for() -> None:
    release, started = threading.Event(), threading.Event()
    reachable = [True]

    def handle(request: httpx.Request) -> httpx.Response:
        if not reachable[0]:
            raise httpx.ConnectError("refused")
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"status": {"value": "unloaded"}}]})
        started.set()
        release.wait(5)
        return httpx.Response(200, json={})

    model = _model(httpx.MockTransport(handle))
    asking = threading.Thread(target=model.load)
    asking.start()
    assert started.wait(5)
    reachable[0] = False
    try:
        assert model.status() is None
    finally:
        release.set()
        asking.join(5)


def test_a_status_read_begun_during_a_load_that_ends_meanwhile_still_says_loading() -> None:
    """The server's answer was sampled while the load ran; it cannot say the load is over."""
    finish_load, sampled, release_read = threading.Event(), threading.Event(), threading.Event()
    loading = threading.Event()

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            answer = httpx.Response(200, json=UNLOADED)
            if loading.is_set():
                sampled.set()
                release_read.wait(5)
            return answer
        loading.set()
        finish_load.wait(5)
        return httpx.Response(200, json={"status": "loaded"})

    model = _model(httpx.MockTransport(handle))
    seen: list[str | None] = []
    loader = threading.Thread(target=model.load)
    loader.start()
    assert loading.wait(5)
    reader = threading.Thread(target=lambda: seen.append(model.status()))
    reader.start()
    assert sampled.wait(5)
    finish_load.set()
    loader.join(5)
    release_read.set()
    reader.join(5)

    assert seen == ["loading"]


def test_warming_stops_retrying_once_it_is_no_longer_wanted() -> None:
    """A resume's load must not go on asking after a pause."""
    loads: list[str] = []
    wanted = iter([True, True, False])  # each attempt asks before its check and before its load

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=UNLOADED)
        loads.append(request.url.path)
        return httpx.Response(503, headers={"Retry-After": "1"})

    model = StrataModel(
        ModelConfig(base_url=Config().model_base_url()),
        "key",
        transport=httpx.MockTransport(handle),
        sleep=lambda _s: None,
    )

    assert model.warm(keep_going=lambda: next(wanted, False)) is False
    assert len(loads) == 1


def test_warming_sends_no_load_when_a_pause_came_during_its_check() -> None:
    loads: list[str] = []
    wanted = iter([True, False])

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=UNLOADED)
        loads.append(request.url.path)
        return httpx.Response(200, json={})

    model = _model(httpx.MockTransport(handle))

    assert model.warm(keep_going=lambda: next(wanted, False)) is False
    assert loads == []


def test_a_load_asked_for_while_one_is_under_way_joins_it() -> None:
    """A wake word during the start-up load must not queue a second load behind the first."""
    release, started = threading.Event(), threading.Event()
    posts: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=UNLOADED)
        posts.append(request.url.path)
        started.set()
        release.wait(5)
        return httpx.Response(200, json={})

    model = _model(httpx.MockTransport(handle))
    first = threading.Thread(target=model.load)
    first.start()
    assert started.wait(5)
    try:
        assert model.load() is True
        assert model.status() == "loading"
    finally:
        release.set()
        first.join(5)
    assert posts == ["/v1/load"]


def test_a_server_that_lists_no_model_has_it_unloaded() -> None:
    """Regression: Strata lists no model while its engine is down or restarting, and Jarvis
    called that server unreachable."""
    model = _model(httpx.MockTransport(lambda _r: httpx.Response(200, json={"data": []})))

    assert model.status() == "unloaded"
    assert model.is_loaded() is False


def test_without_thinking_the_model_is_asked_to_answer_at_once() -> None:
    seen: list[httpx.Request] = []

    _model(_answering({"role": "assistant", "content": "hi"}, seen)).chat(
        MESSAGES, TOOLS, max_tokens=400
    )

    body = json.loads(seen[0].content)
    assert body["reasoning_effort"] == "none"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == 400
    assert "reasoning_budget_tokens" not in body


def test_thinking_asks_for_its_effort_and_cap_and_keeps_the_replys_room() -> None:
    # The server's max_tokens counts the thinking too: the reply's own cap must stay whole.
    seen: list[httpx.Request] = []

    _model(_answering({"role": "assistant", "content": "hi"}, seen)).chat(
        MESSAGES, TOOLS, max_tokens=400, thinking=Thinking("low", 192)
    )

    body = json.loads(seen[0].content)
    assert body["reasoning_effort"] == "low"
    assert body["reasoning_budget_tokens"] == 192
    assert body["max_tokens"] == 400 + 192
    assert "chat_template_kwargs" not in body


def test_streamed_thinking_is_neither_passed_on_nor_part_of_the_answer() -> None:
    events = (
        _chunk({"reasoning_content": "The owner wants the music paused."}),
        _chunk({"content": "Paused it."}),
        FINISH,
    )
    transport = httpx.MockTransport(lambda _r: httpx.Response(200, content=_sse(*events)))
    pieces: list[Delta] = []

    reply = _model(transport).chat(
        MESSAGES, TOOLS, on_delta=pieces.append, thinking=Thinking("low", 192)
    )

    assert pieces == [Delta(text="Paused it.")]
    assert reply["content"] == "Paused it."
    assert "reasoning_content" not in reply
