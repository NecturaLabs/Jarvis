"""A model that answers from a script, and a clock the test moves: for tests of the agent loop."""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.model import Delta, Thinking


def call(name: str, args: dict[str, Any] | None = None, call_id: str = "c1") -> dict[str, Any]:
    """Return a model message that calls one tool."""
    function = {"name": name, "arguments": json.dumps(args or {})}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function", "function": function}],
    }


def say(text: str) -> dict[str, Any]:
    """Return a model message that only talks."""
    return {"role": "assistant", "content": text}


@dataclass
class ScriptedModel:
    """Replies in order and keeps the messages each call was given."""

    replies: list[dict[str, Any]]
    seen: list[list[dict[str, Any]]] = field(default_factory=list[list[dict[str, Any]]])

    def chat(  # ruff: ignore[too-many-arguments] - the model's own call, keyword-only
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        require_tool: bool | str = False,
        timeout_s: float | None = None,
        on_delta: Callable[[Delta], None] | None = None,
        max_tokens: int | None = None,
        load_s: float = 0.0,
        thinking: Thinking | None = None,
    ) -> dict[str, Any]:
        """Return the next scripted message."""
        del tools, require_tool, timeout_s, on_delta, max_tokens, load_s, thinking
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)

    @staticmethod
    def is_loaded() -> bool:
        """The model is always in."""
        return True


class Clock:
    """A clock that only moves when the test says so."""

    def __init__(self, now: float = 1000.0) -> None:
        """Start at `now`."""
        super().__init__()
        self.now = now

    def __call__(self) -> float:
        """Return the time now."""
        return self.now
