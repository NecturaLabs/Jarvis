"""Read-only look-ups of one model message run side by side; everything else runs in turn."""

import json
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest
from test_agent import Clock, ScriptedModel

from jarvis.agent import MAX_SIDE_BY_SIDE, Agent, Hooks, Limits
from jarvis.app import no_action_operation
from jarvis.audit import AuditLog
from jarvis.operations import Operation, OperationError, Registry, Result, current_request
from jarvis.policy import OpClass, Origin

NO_CONFIRM = Hooks(confirm=lambda _q: False)
WAIT_S = 5.0


def _agent(
    model: ScriptedModel,
    ops: list[Operation],
    *,
    overrides: dict[str, str] | None = None,
    limits: Limits | None = None,
) -> tuple[Agent, AuditLog]:
    audit = AuditLog(None)
    registry = Registry([*ops, no_action_operation()])
    agent = Agent(model, registry, "system", audit, overrides or {}, clock=Clock(), limits=limits)
    return agent, audit


def _say(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _events(audit: AuditLog) -> list[object]:
    return [e["event"] for e in audit.events]


def _message(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """One assistant message holding every call, each with its own id."""
    tool_calls = [
        {
            "id": f"c{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        for i, (name, args) in enumerate(calls)
    ]
    return {"role": "assistant", "content": "", "tool_calls": tool_calls}


@dataclass
class Board:
    """Records what runs, and how many look-ups are running at once."""

    log: list[str] = field(default_factory=list[str])
    running: int = 0
    most: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def enter(self, name: str) -> None:
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
            self.log.append(f"start {name}")

    def leave(self, name: str) -> None:
        with self.lock:
            self.running -= 1
            self.log.append(f"end {name}")


def _lookup(
    board: Board,
    name: str,
    *,
    concurrent: bool = True,
    meet: threading.Barrier | None = None,
    wait_for: threading.Event | None = None,
    then: threading.Event | None = None,
    error: str = "",
    crash: bool = False,
    cls: OpClass = OpClass.SAFE,
) -> Operation:
    """A read-only look-up; `meet` makes it wait until the others reach it too."""

    def run(args: dict[str, Any]) -> Result:
        board.enter(name)
        try:
            if meet is not None:
                meet.wait(WAIT_S)
            if wait_for is not None:
                assert wait_for.wait(WAIT_S)
            if error:
                raise OperationError(error)
            if crash:
                raise RuntimeError("a bug in the look-up")
            return Result(f"{name} found {args.get('q', '')}", Origin.WEB)
        finally:
            board.leave(name)
            if then is not None:
                then.set()

    return Operation(
        name,
        name,
        {"type": "object", "properties": {"q": {"type": "string"}}},
        run,
        default_class=cls,
        side_effect=False,
        concurrent=concurrent,
    )


def _action(board: Board, name: str) -> Operation:
    def run(_args: dict[str, Any]) -> Result:
        board.enter(name)
        board.leave(name)
        return Result(f"{name} done", spoken=f"{name} done.")

    return Operation(name, name, {"type": "object", "properties": {}}, run)


def _tool_messages(model: ScriptedModel) -> list[dict[str, Any]]:
    return [m for m in model.seen[1] if m.get("role") == "tool"]


def test_two_look_ups_of_one_message_run_side_by_side() -> None:
    # Each waits for the other: run one after the other, the first would wait in vain.
    board = Board()
    meet = threading.Barrier(2)
    model = ScriptedModel([_message(("a", {"q": "x"}), ("b", {"q": "y"})), _say("Both.")])
    agent, _ = _agent(model, [_lookup(board, "a", meet=meet), _lookup(board, "b", meet=meet)])

    assert agent.handle("look both up", NO_CONFIRM) == "Both."

    assert not meet.broken
    assert board.most == 2
    tools = _tool_messages(model)
    assert "a found x" in tools[0]["content"]
    assert "b found y" in tools[1]["content"]


def test_results_keep_the_order_of_the_calls_whichever_finishes_first() -> None:
    board = Board()
    second_done = threading.Event()
    model = ScriptedModel([_message(("slow", {"q": "1"}), ("quick", {"q": "2"})), _say("Done.")])
    ops = [
        _lookup(board, "slow", wait_for=second_done),
        _lookup(board, "quick", then=second_done),
    ]
    agent, audit = _agent(model, ops)

    agent.handle("look both up", NO_CONFIRM)

    assert board.log.index("end quick") < board.log.index("end slow")
    tools = _tool_messages(model)
    assert [m["tool_call_id"] for m in tools] == ["c0", "c1"]
    assert "slow found 1" in tools[0]["content"]
    assert "quick found 2" in tools[1]["content"]
    # The audit log reads in the order of the calls too, not the order they finished in.
    results = [e["op"] for e in audit.events if e["event"] == "tool_result"]
    assert results == ["slow", "quick"]


def test_every_call_is_decided_and_audited_before_any_look_up_runs() -> None:
    board = Board()
    model = ScriptedModel([_message(("a", {"q": "1"}), ("b", {"q": "2"})), _say("Done.")])
    agent, audit = _agent(model, [_lookup(board, "a"), _lookup(board, "b")])

    agent.handle("look both up", NO_CONFIRM)

    events = [e["event"] for e in audit.events if e["event"] in {"tool_call", "tool_result"}]
    assert events == ["tool_call", "tool_call", "tool_result", "tool_result"]


def test_an_action_never_runs_beside_a_look_up() -> None:
    board = Board()
    message = _message(("a", {"q": "1"}), ("b", {"q": "2"}), ("act", {}), ("c", {"q": "3"}))
    model = ScriptedModel([message, _say("Done.")])
    ops = [_lookup(board, "a"), _lookup(board, "b"), _action(board, "act"), _lookup(board, "c")]
    agent, _ = _agent(model, ops)

    agent.handle("look up, act, look up", NO_CONFIRM)

    start, end = board.log.index("start act"), board.log.index("end act")
    assert end == start + 1  # nothing started or ended while it ran
    assert board.log.index("end a") < start
    assert board.log.index("end b") < start
    assert board.log.index("start c") > end


def test_a_look_up_not_marked_concurrent_runs_on_its_own() -> None:
    board = Board()
    model = ScriptedModel([_message(("a", {"q": "1"}), ("b", {"q": "2"})), _say("Done.")])
    ops = [_lookup(board, "a", concurrent=False), _lookup(board, "b", concurrent=False)]
    agent, _ = _agent(model, ops)

    agent.handle("look both up", NO_CONFIRM)

    assert board.most == 1


def test_a_refused_look_up_runs_nothing_and_the_others_still_run() -> None:
    board = Board()
    model = ScriptedModel([_message(("a", {"q": "1"}), ("b", {"q": "2"})), _say("Done.")])
    agent, audit = _agent(
        model, [_lookup(board, "a"), _lookup(board, "b")], overrides={"a": "deny"}
    )

    agent.handle("look both up", NO_CONFIRM)

    assert "start a" not in board.log
    assert "start b" in board.log
    tools = _tool_messages(model)
    assert tools[0]["content"].startswith("Refused")
    assert "b found 2" in tools[1]["content"]
    assert [e["verdict"] for e in audit.events if e["event"] == "tool_call"] == ["deny", "run"]


def test_a_look_up_that_needs_a_yes_is_asked_before_any_runs() -> None:
    board = Board()
    asked: list[str] = []

    def confirm(question: str) -> bool:
        asked.append(question)
        asked.append(f"running then: {len(board.log)}")
        return True

    model = ScriptedModel([_message(("a", {"q": "1"}), ("b", {"q": "2"})), _say("Done.")])
    ops = [_lookup(board, "a"), _lookup(board, "b", cls=OpClass.CONFIRM)]
    agent, _ = _agent(model, ops)

    agent.handle("look both up", Hooks(confirm=confirm))

    assert asked[1] == "running then: 0"
    assert {"start a", "start b"} <= set(board.log)


def test_a_failing_look_up_is_reported_for_itself_only() -> None:
    board = Board()
    model = ScriptedModel([_message(("a", {"q": "1"}), ("b", {"q": "2"})), _say("Done.")])
    agent, audit = _agent(
        model, [_lookup(board, "a", error="the site is down"), _lookup(board, "b")]
    )

    agent.handle("look both up", NO_CONFIRM)

    tools = _tool_messages(model)
    assert tools[0]["content"] == "Failed: the site is down"
    assert "b found 2" in tools[1]["content"]
    assert "tool_failed" in _events(audit)


def test_a_look_up_sees_the_request_it_runs_for() -> None:
    seen: list[object] = []

    def run(_args: dict[str, Any]) -> Result:
        seen.append(current_request())
        return Result("ok")

    def op(name: str) -> Operation:
        return Operation(
            name,
            name,
            {"type": "object", "properties": {}},
            run,
            side_effect=False,
            concurrent=True,
        )

    model = ScriptedModel([_message(("a", {}), ("b", {})), _say("Done.")])
    agent, _ = _agent(model, [op("a"), op("b")])

    agent.handle("look both up", NO_CONFIRM)

    assert len(seen) == 2
    first, second = seen
    assert first is not None
    assert first is second  # the very request they run for, in each look-up's thread
    assert getattr(first, "words", None) == "look both up"


def test_an_operation_that_acts_cannot_claim_to_run_beside_others() -> None:
    with pytest.raises(ValueError, match="read-only"):
        Operation(
            "act",
            "act",
            {"type": "object", "properties": {}},
            lambda _a: Result(""),
            concurrent=True,
        )


def test_no_more_look_ups_run_at_once_than_the_cap() -> None:
    board = Board()
    go = threading.Event()
    names = [f"l{n}" for n in range(MAX_SIDE_BY_SIDE + 2)]
    model = ScriptedModel([_message(*((n, {"q": n}) for n in names)), _say("Done.")])
    agent, _ = _agent(model, [_lookup(board, n, wait_for=go) for n in names])

    def release() -> None:
        # Let them go only once the cap is reached and any look-up over it had time to start.
        while board.running < MAX_SIDE_BY_SIDE:
            threading.Event().wait(0.01)
        threading.Event().wait(0.2)
        go.set()

    helper = threading.Thread(target=release, daemon=True)
    helper.start()
    agent.handle("look them all up", NO_CONFIRM)
    helper.join(WAIT_S)

    assert board.most == MAX_SIDE_BY_SIDE
    assert len([line for line in board.log if line.startswith("end")]) == len(names)


def test_a_look_up_that_crashes_is_reported_as_an_internal_error() -> None:
    board = Board()
    model = ScriptedModel([_message(("a", {"q": "1"}), ("b", {"q": "2"})), _say("Done.")])
    agent, _ = _agent(model, [_lookup(board, "a", crash=True), _lookup(board, "b")])

    agent.handle("look both up", NO_CONFIRM)

    tools = _tool_messages(model)
    assert tools[0]["content"] == "Failed: an internal error."
    assert "b found 2" in tools[1]["content"]


def test_a_group_over_the_cap_on_tool_calls_runs_only_what_fits() -> None:
    board = Board()
    model = ScriptedModel([_message(("a", {"q": "1"}), ("b", {"q": "2"})), _say("Done.")])
    agent, _ = _agent(
        model, [_lookup(board, "a"), _lookup(board, "b")], limits=Limits(max_tool_calls=1)
    )

    agent.handle("look both up", NO_CONFIRM)

    assert "start a" in board.log
    assert "start b" not in board.log
    assert "too many actions" in _tool_messages(model)[1]["content"]


def test_a_no_in_a_group_declines_the_later_questions_unasked() -> None:
    board = Board()
    asked: list[str] = []

    def confirm(question: str) -> bool:
        asked.append(question)
        return False

    model = ScriptedModel([_message(("a", {"q": "1"}), ("b", {"q": "2"})), _say("Done.")])
    ops = [_lookup(board, "a", cls=OpClass.CONFIRM), _lookup(board, "b", cls=OpClass.CONFIRM)]
    agent, _ = _agent(model, ops)

    agent.handle("look both up", Hooks(confirm=confirm))

    assert len(asked) == 1
    assert board.log == []


def test_a_look_up_repeated_from_an_earlier_message_is_not_run_again() -> None:
    board = Board()
    model = ScriptedModel(
        [
            _message(("a", {"q": "1"})),
            _message(("a", {"q": "1"}), ("b", {"q": "2"})),
            _say("Done."),
        ]
    )
    agent, _ = _agent(model, [_lookup(board, "a"), _lookup(board, "b")])

    agent.handle("look it up", NO_CONFIRM)

    assert board.log.count("start a") == 1
    assert "start b" in board.log


def test_look_ups_approved_before_jarvis_stopped_do_not_run() -> None:
    board = Board()
    stopping = threading.Event()

    def confirm(_question: str) -> bool:
        stopping.set()  # Jarvis is told to stop while the second call is asked about
        return True

    model = ScriptedModel([_message(("a", {"q": "1"}), ("b", {"q": "2"})), _say("Done.")])
    ops = [_lookup(board, "a"), _lookup(board, "b", cls=OpClass.CONFIRM)]
    agent, audit = _agent(model, ops)

    agent.handle("look both up", Hooks(confirm=confirm, cancelled=stopping.is_set))

    assert board.log == []
    results = [e for e in audit.events if e["event"] == "tool_result"]
    assert [e["ok"] for e in results] == [False, False]
