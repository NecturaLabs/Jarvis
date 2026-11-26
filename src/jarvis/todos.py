"""The chat's todo list: `todo_write` keeps it, the window shows it above the input.

The list belongs to the conversation: `/compact` leaves it, `/clear` and a resumed conversation
start it empty (the registry's resets). The operation changes nothing outside the chat, so it is
`safe`; it is offered only to the chat window (`jarvis chat`), never by voice.
"""

import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from jarvis.jsonutil import objs
from jarvis.operations import Operation, OperationError, Result, inert, one_line
from jarvis.policy import OpClass

if TYPE_CHECKING:
    from collections.abc import Callable

NAME = "todo_write"
MAX_ITEMS = 20


class Status(StrEnum):
    """Where a task stands."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class Todo:
    """One task of the list."""

    content: str
    status: Status


MARKS = {Status.PENDING: "[ ]", Status.IN_PROGRESS: "[>]", Status.COMPLETED: "[x]"}


class TodoList:
    """The conversation's tasks. A window subscribes to hear of each change (from any thread)."""

    def __init__(self) -> None:
        """Start empty."""
        super().__init__()
        self._lock = threading.Lock()
        self._items: tuple[Todo, ...] = ()
        self._listeners: list[Callable[[tuple[Todo, ...]], None]] = []

    @property
    def items(self) -> tuple[Todo, ...]:
        """The tasks now, in order."""
        with self._lock:
            return self._items

    def subscribe(self, listener: Callable[[tuple[Todo, ...]], None]) -> None:
        """Call `listener` with the whole list after every change."""
        with self._lock:
            self._listeners.append(listener)

    def replace(self, items: tuple[Todo, ...]) -> None:
        """Make `items` the list, and tell the listeners."""
        with self._lock:
            self._items = items
            listeners = list(self._listeners)
        for listener in listeners:
            listener(items)

    def clear(self) -> None:
        """Empty the list (a new conversation)."""
        if self._items:
            self.replace(())


def short_form(items: tuple[Todo, ...]) -> str:
    """Return the list as one line the model reads back: "2 of 4 done: [x] a; [>] b; [ ] c"."""
    if not items:
        return "The todo list is empty."
    done = sum(1 for item in items if item.status is Status.COMPLETED)
    body = "; ".join(f"{MARKS[item.status]} {inert(item.content)}" for item in items)
    return f"{done} of {len(items)} done: {body}"


ABOUT = (
    "Keep the chat's todo list, shown above the input. Use it for multi-step work, not a "
    "one-step request. Send the whole list each call; mark one task in_progress before you "
    "start it and completed right after you finish it. An empty list clears it."
)


def operation(todos: TodoList) -> Operation:
    """Return `todo_write`: the model's way to keep the conversation's todo list.

    The call replaces the list; at most `MAX_ITEMS` tasks and at most one in progress. Its text is
    the model's own, shown as data in the window.
    """

    def run(args: dict[str, object]) -> Result:
        items = tuple(
            Todo(one_line(str(row["content"])), Status(str(row["status"])))
            for row in objs(args["todos"])
        )
        if any(not item.content for item in items):
            msg = "every task needs words of its own"
            raise OperationError(msg)
        if sum(1 for item in items if item.status is Status.IN_PROGRESS) > 1:
            msg = "only one task can be in_progress at a time; the list was not changed"
            raise OperationError(msg)
        todos.replace(items)
        return Result(f"Todo list updated. {short_form(items)}")

    return Operation(
        NAME,
        ABOUT,
        {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "maxItems": MAX_ITEMS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": [s.value for s in Status]},
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
            "additionalProperties": False,
        },
        run,
        default_class=OpClass.SAFE,
        side_effect=False,
        spoken_summary=lambda _args: "update the todo list",
    )
