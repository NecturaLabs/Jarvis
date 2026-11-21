"""OpenAI-compatible chat client for the Strata server, with non-generating health checks."""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

import httpx

from jarvis.jsonutil import obj, objs

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.config import ModelConfig


LOAD_TIMEOUT_S = 300.0  # a cold load from disk took about 110 s while a game ran
WARM_ATTEMPTS = 60
WARM_RETRY_S = 2.0


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
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Return the assistant message; with `require_tool`, the model must call a tool.

        `timeout_s` caps how long this one call may wait (the time the request has left).
        """
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
    api_key: str = field(repr=False)
    transport: httpx.BaseTransport | None = None  # tests answer in place of the server
    sleep: Callable[[float], None] = time.sleep  # tests wait without waiting
    clock: Callable[[], float] = time.monotonic  # tests move time with their sleep

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

    def warm(self, attempts: int = WARM_ATTEMPTS) -> bool:
        """Load the model once the server is up (at service start); return whether it loaded.

        The server may still be starting (unreachable, or answering 409 while busy), so this
        waits and tries again, up to `attempts` times. A load reads the whole model and can take
        minutes when it is not in the page cache, so the request waits for it.
        """
        for _ in range(attempts):
            if self.status() is not None:
                try:
                    with self._client(LOAD_TIMEOUT_S) as client:
                        response = client.post("/load", json={})
                except httpx.ReadTimeout:
                    return True  # still loading on the server's side
                except httpx.HTTPError:
                    pass
                else:
                    if response.status_code < 300:  # ruff: ignore[magic-value-comparison]
                        return True
            self.sleep(WARM_RETRY_S)
        return False

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
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Send one chat completion with thinking off; retry once after a dropped connection.

        The call waits at most `timeout_s` (when given) and never longer than the configured
        request timeout.

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
            if len(messages) > 1 and messages[0].get("role") == "system":
                # Strata keeps the system prompt and tools read once, pinned, so a request reads
                # only its own words (an engine without pin support ignores the mark).
                body["strata_prefix"] = {"messages": 1}
        limit = self.config.request_timeout_s
        if timeout_s is not None:
            limit = min(limit, timeout_s)
        deadline = self.clock() + limit
        for attempt in (1, 2):
            try:
                with self._client(max(1.0, deadline - self.clock())) as client:
                    response = client.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                if attempt == 2 or isinstance(exc, httpx.ReadTimeout):  # ruff: ignore[magic-value-comparison]
                    msg = f"model server unreachable: {exc}"
                    raise ModelError(msg) from exc
                self._wait_ready(min(30.0, max(0.0, deadline - self.clock())))
                continue
            return _message(response)
        msg = "unreachable"
        raise ModelError(msg)

    def _wait_ready(self, seconds: float) -> None:
        deadline = self.clock() + seconds
        while self.clock() < deadline:
            if self.status() is not None:
                return
            self.sleep(1.0)
