"""OpenAI-compatible chat client for the Strata server, with non-generating health checks."""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import httpx

from jarvis.jsonutil import obj, objs

if TYPE_CHECKING:
    from jarvis.config import ModelConfig


class ModelError(Exception):
    """The model server could not answer."""


class ChatModel(Protocol):
    """What the agent needs from a model."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        require_tool: bool = False,
    ) -> dict[str, Any]:
        """Return the assistant message; with `require_tool`, the model must call a tool."""
        ...

    def is_loaded(self) -> bool | None:
        """Return whether the model is in memory (None when unknown); never generates."""
        ...


def _message(response: httpx.Response) -> dict[str, Any]:
    """Return the assistant message of a chat completion answer.

    Raises:
        ModelError: the server refused or the answer holds no message object.
    """
    if response.status_code >= 400:  # ruff: ignore[magic-value-comparison]
        msg = f"model server answered {response.status_code}: {response.text[:200]}"
        raise ModelError(msg)
    try:
        choices = objs(obj(response.json()).get("choices"))
    except ValueError as exc:
        msg = f"model server sent an unexpected answer: {exc}"
        raise ModelError(msg) from exc
    message = choices[0].get("message") if choices else None
    if not isinstance(message, dict):
        msg = "model server sent an answer without a message"
        raise ModelError(msg)
    return cast("dict[str, Any]", message)


@dataclass(frozen=True, slots=True)
class StrataModel:
    """Talks to `strata serve` over its OpenAI API."""

    config: ModelConfig
    api_key: str
    transport: httpx.BaseTransport | None = None  # tests answer in place of the server

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(
            base_url=self.config.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
            trust_env=False,
            transport=self.transport,
        )

    def status(self) -> str | None:
        """Return the server's model state (`loaded`, `unloaded`...) without generating."""
        try:
            with self._client(5.0) as client:
                response = client.get("/models")
                response.raise_for_status()
                data = objs(obj(response.json()).get("data"))
        except httpx.HTTPError, ValueError:
            return None
        state = obj(data[0].get("status")).get("value") if data else None
        return str(state) if state is not None else None

    def is_loaded(self) -> bool | None:
        """Return whether the model is in memory (None when the server is unreachable)."""
        state = self.status()
        return None if state is None else state == "loaded"

    def load(self) -> bool:
        """Ask the server to load the model now (ahead of a request); return whether it accepted."""
        try:
            with self._client(5.0) as client:
                response = client.post("/load", json={})
        except httpx.HTTPError:
            return False
        return response.status_code < 300  # ruff: ignore[magic-value-comparison]

    def unload(self) -> bool:
        """Ask the server to free the model now; return whether it did."""
        try:
            with self._client(30.0) as client:
                response = client.post("/unload", json={})
        except httpx.HTTPError:
            return False
        return response.status_code < 300  # ruff: ignore[magic-value-comparison]

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        require_tool: bool = False,
    ) -> dict[str, Any]:
        """Send one chat completion with thinking off; retry once after a dropped connection.

        Raises:
            ModelError: the server refused, failed or stayed unreachable.
        """
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "required" if require_tool else "auto"
        for attempt in (1, 2):
            try:
                with self._client(self.config.request_timeout_s) as client:
                    response = client.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                if attempt == 2 or isinstance(exc, httpx.ReadTimeout):  # ruff: ignore[magic-value-comparison]
                    msg = f"model server unreachable: {exc}"
                    raise ModelError(msg) from exc
                self._wait_ready(30.0)
                continue
            return _message(response)
        msg = "unreachable"
        raise ModelError(msg)

    def _wait_ready(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.status() is not None:
                return
            time.sleep(1.0)
