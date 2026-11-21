"""The Strata client: what it sends, and that a broken answer never reaches the agent."""

import json
from typing import Any, cast

import httpx
import pytest

from jarvis.config import Config, ModelConfig
from jarvis.model import LOAD_TIMEOUT_S, ModelError, StrataModel

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


@pytest.mark.parametrize(("require", "choice"), [(True, "required"), (False, "auto")])
def test_chat_sends_the_key_the_tools_and_the_tool_choice(require: bool, choice: str) -> None:
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
        httpx.Response(200, json={"data": []}),
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
    _model(_answering({"content": "hi"}, seen)).chat(MESSAGES, TOOLS, timeout_s=asked)

    assert _read_timeout(seen[0]) == pytest.approx(sent)


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
