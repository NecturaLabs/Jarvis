"""The chat's todo list: what `todo_write` accepts, what it keeps and where it is offered."""

from typing import TYPE_CHECKING, Any

import pytest
from scripted_model import ScriptedModel, call, say

from jarvis.agent import Agent, Hooks
from jarvis.app import (
    channel_limits,
    channel_registry,
    offered_registry,
)
from jarvis.audit import AuditLog
from jarvis.config import Config
from jarvis.operations import ArgumentError, OperationError, Registry, validate_args
from jarvis.policy import OpClass
from jarvis.todos import MAX_ITEMS, Status, Todo, TodoList, operation, short_form

if TYPE_CHECKING:
    from jarvis.operations import Result


def _rows(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"content": content, "status": status} for content, status in pairs]


def _call(todos: TodoList, rows: object) -> Result:
    op = operation(todos)
    return op.run(validate_args(op, {"todos": rows}))


def test_a_list_replaces_the_last_one_and_the_listeners_hear_of_it() -> None:
    todos = TodoList()
    heard: list[tuple[Todo, ...]] = []
    todos.subscribe(heard.append)

    _call(todos, _rows(("Read the file", "completed"), ("Write the fix", "in_progress")))
    result = _call(todos, _rows(("Run the tests", "pending")))

    assert todos.items == (Todo("Run the tests", Status.PENDING),)
    assert [len(items) for items in heard] == [2, 1]
    assert result.text == "Todo list updated. 0 of 1 done: [ ] Run the tests"


def test_the_model_reads_the_list_back_in_a_short_form() -> None:
    items = (
        Todo("Read it", Status.COMPLETED),
        Todo("Write <it>", Status.IN_PROGRESS),
        Todo("Test it", Status.PENDING),
    )

    assert short_form(items) == "1 of 3 done: [x] Read it; [>] Write &lt;it&gt;; [ ] Test it"
    assert short_form(()) == "The todo list is empty."


def test_an_empty_list_clears_it() -> None:
    todos = TodoList()
    _call(todos, _rows(("One", "pending")))

    _call(todos, [])

    assert todos.items == ()


def test_the_most_it_takes_is_twenty_tasks() -> None:
    todos = TodoList()
    _call(todos, _rows(*[(f"Task {n}", "pending") for n in range(MAX_ITEMS)]))
    assert len(todos.items) == MAX_ITEMS

    with pytest.raises(ArgumentError):
        _call(todos, _rows(*[(f"Task {n}", "pending") for n in range(MAX_ITEMS + 1)]))
    assert len(todos.items) == MAX_ITEMS


def test_only_one_task_may_be_in_progress_and_the_list_stays_as_it_was() -> None:
    todos = TodoList()
    _call(todos, _rows(("Keep me", "pending")))

    with pytest.raises(OperationError, match="one task"):
        _call(todos, _rows(("A", "in_progress"), ("B", "in_progress")))

    assert todos.items == (Todo("Keep me", Status.PENDING),)


@pytest.mark.parametrize(
    "arguments",
    [
        {},  # no list
        {"todos": "do it"},  # not a list
        {"todos": [{"content": "A", "status": "done"}]},  # a status it does not have
        {"todos": [{"content": "A", "status": "pending", "id": 1}]},  # a key it does not have
        {"todos": [{"content": "A", "status": "pending"}], "extra": 1},  # an argument it lacks
    ],
)
def test_a_malformed_call_is_refused_before_it_runs(arguments: dict[str, Any]) -> None:
    with pytest.raises(ArgumentError):
        validate_args(operation(TodoList()), arguments)


@pytest.mark.parametrize(
    "rows", [[{"content": "A"}], [{"status": "pending"}]], ids=["no status", "no words"]
)
def test_a_task_missing_its_words_or_status_is_refused_and_the_list_kept(
    rows: list[dict[str, str]],
) -> None:
    todos = TodoList()
    _call(todos, _rows(("Keep me", "pending")))

    with pytest.raises(OperationError):
        _call(todos, rows)

    assert [item.content for item in todos.items] == ["Keep me"]


def test_a_task_needs_words_and_one_line_of_them() -> None:
    todos = TodoList()
    with pytest.raises(OperationError, match="words"):
        _call(todos, _rows(("   ", "pending")))

    _call(todos, _rows(("Read\n\x1b[31mthe\tfile", "pending")))

    assert todos.items[0].content == "Read [31mthe file"


def test_a_new_conversation_empties_the_list() -> None:
    todos = TodoList()
    heard: list[tuple[Todo, ...]] = []
    todos.subscribe(heard.append)
    registry = Registry([operation(todos)], resets=(todos.clear,))
    _call(todos, _rows(("One", "pending")))

    registry.reset()
    registry.reset()  # an empty list is not announced again

    assert todos.items == ()
    assert [len(items) for items in heard] == [1, 0]


def test_it_is_a_safe_operation_that_changes_nothing_outside_the_chat() -> None:
    op = operation(TodoList())

    assert op.default_class is OpClass.SAFE
    assert not op.side_effect
    assert op.shortcuts == ()


def test_only_the_chat_window_is_offered_it() -> None:
    config = Config()
    voice = channel_registry(config, [], channel_limits(config, "voice"), "voice")
    chat = channel_registry(config, [], channel_limits(config, "chat"), "chat")

    assert voice.get("todo_write") is None
    assert chat.get("todo_write") is not None
    for via in (None, "ask", "acp", "speech-test"):
        other = offered_registry(
            config, [], "chat" if via == "acp" else "voice", coding=False, via=via, can_switch=False
        )
        assert other.get("todo_write") is None, via
    window = offered_registry(config, [], "chat", coding=False, via="chat", can_switch=False)
    assert window.get("todo_write") is not None


def test_a_call_is_audited_runs_without_a_question_and_the_list_survives_compacting() -> None:
    config = Config()
    todos = TodoList()
    registry = channel_registry(config, [], channel_limits(config, "chat"), "chat", todos=todos)
    audit = AuditLog(None)
    rows = _rows(("Read it", "completed"), ("Write it", "in_progress"))
    model = ScriptedModel(
        [
            call("todo_write", {"todos": rows}),
            say("Working on it."),
            say("The request so far: reading and writing."),
        ]
    )
    agent = Agent(model, registry, "system", audit, {}, limits=channel_limits(config, "chat"))
    asked: list[str] = []

    reply = agent.handle("fix it", Hooks(confirm=lambda question: asked.append(question) or False))

    assert reply == "Working on it."
    assert asked == []  # a safe operation: no question
    made = next(e for e in audit.events if e["event"] == "tool_call")
    assert made["op"] == "todo_write"
    assert made["verdict"] == "run"
    assert any(e["event"] == "tool_result" and e["op"] == "todo_write" for e in audit.events)
    assert len(todos.items) == 2

    agent.compact(everything=True)
    assert len(todos.items) == 2  # compacting keeps the list

    agent.forget()
    assert todos.items == ()  # a new conversation does not


def test_it_does_not_count_as_an_action_or_a_look_up() -> None:
    config = Config()
    todos = TodoList()
    registry = channel_registry(config, [], channel_limits(config, "chat"), "chat", todos=todos)
    model = ScriptedModel([call("todo_write", {"todos": _rows(("One", "pending"))}), say("Done.")])
    agent = Agent(
        model, registry, "system", AuditLog(None), {}, limits=channel_limits(config, "chat")
    )

    agent.handle("do it", Hooks(confirm=lambda _q: False))

    turn = agent.memory.turns[-1]
    assert not turn.looked
