"""The Strata client: what it sends, and that a broken answer never reaches the agent."""

import json
from typing import Any

import httpx
import pytest

from jarvis.config import Config, ModelConfig
from jarvis.model import ModelError, StrataModel

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
