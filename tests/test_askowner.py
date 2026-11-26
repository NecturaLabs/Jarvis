"""`ask_owner`: the question, its answer, its timer and why an answer is never consent."""

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

import pytest
from scripted_model import Clock, ScriptedModel, call, say

from jarvis.agent import Agent, Declined, Hooks, Limits
from jarvis.app import channel_limits, channel_registry, offered_registry
from jarvis.askowner import (
    Choice,
    Outcome,
    Question,
    Questions,
    operation,
)
from jarvis.audit import AuditLog
from jarvis.config import ChatConfig, Config
from jarvis.memory import NOTED
from jarvis.operations import (
    ArgumentError,
    Notice,
    Operation,
    OperationError,
    Registry,
    Result,
    current_request,
    request_scope,
    validate_args,
)
from jarvis.policy import OpClass, Origin

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class FakeView:
    """The window: records what it was told, and may act when a question opens."""

    opened: list[Question] = field(default_factory=list[Question])
    closed: list[tuple[Question, str]] = field(default_factory=list[tuple[Question, str]])
    notices: list[Notice] = field(default_factory=list[Notice])
    on_open: Callable[[Question], object] | None = None

    def open(self, question: Question) -> None:
        self.opened.append(question)
        if self.on_open is not None:
            self.on_open(question)

    def close(self, question: Question, note: str) -> None:
        self.closed.append((question, note))

    def deliver(self, notice: Notice) -> None:
        self.notices.append(notice)


def _setup(timeout_s: float = 300.0) -> tuple[Questions, FakeView, Clock]:
    clock = Clock()
    questions = Questions(timeout_s, clock)
    view = FakeView()
    questions.attach(view)
    return questions, view, clock


OPTIONS = [
    {"label": "Alpha", "description": "the first"},
    {"label": "Bravo", "description": "the second"},
    {"label": "Charlie", "description": "the third"},
]


def _args(**more: object) -> dict[str, object]:
    return {"question": "Which one?", "header": "Pick", "options": OPTIONS, "recommended": 1} | more


def _ask(questions: Questions, **more: object) -> Question:
    """Open a question the way the operation does."""
    with request_scope(foreign=False):
        op = operation(questions)
        result = op.run(validate_args(op, _args(**more)))
    assert result.ok
    question = questions.current
    assert question is not None
    return question


# The question and its arguments


def test_the_recommended_option_is_listed_first() -> None:
    questions, view, _ = _setup()

    question = _ask(questions)

    assert view.opened == [question]
    assert question.order == (1, 0, 2)
    assert question.options[question.order[0]].label == "Bravo"


@pytest.mark.parametrize(
    "arguments",
    [
        _args() | {"recommended": None},
        {k: v for k, v in _args().items() if k != "recommended"},  # it is required
        {k: v for k, v in _args().items() if k != "header"},
        _args(options=OPTIONS[:1]),  # fewer than two options
        _args(options=[*OPTIONS, *OPTIONS[:2]]),  # more than four
        _args(header="a header over twelve"),
        _args(options=[{"label": "A"}, {"label": "B", "description": ""}]),  # no description
        _args(options=[{"label": "A", "description": "", "extra": 1}, OPTIONS[0]]),
        _args(recommended=-1),
        _args(wait="yes"),
        _args(extra=1),
    ],
)
def test_a_malformed_question_is_refused_before_it_runs(arguments: dict[str, Any]) -> None:
    with pytest.raises(ArgumentError):
        validate_args(operation(Questions()), arguments)


def test_the_arguments_default_to_a_single_choice_that_does_not_wait() -> None:
    cleaned = validate_args(operation(Questions()), _args())

    assert cleaned.get("wait") is None
    assert cleaned.get("multi_select") is None


@pytest.mark.parametrize(
    ("more", "why"),
    [
        ({"recommended": 3}, "recommended must be 0 to 2"),
        ({"options": [OPTIONS[0], {"label": "alpha", "description": "x"}]}, "must differ"),
        ({"header": "   "}, "need words"),
        ({"question": "   "}, "need words"),
        ({"options": [{"label": " ", "description": ""}, OPTIONS[0]]}, "need words"),
    ],
)
def test_a_question_that_makes_no_sense_is_refused_and_nothing_is_shown(
    more: dict[str, Any], why: str
) -> None:
    questions, view, _ = _setup()
    op = operation(questions)

    with request_scope(foreign=False), pytest.raises(OperationError, match=why):
        op.run(validate_args(op, _args(**more)))

    assert view.opened == []


def test_the_model_s_words_are_one_plain_line_each() -> None:
    questions, _, _ = _setup()

    question = _ask(
        questions,
        question="Use [bold]this[/bold]?\n\x1b[31mred",
        header="A\tb",
        options=[
            {"label": "[red]x[/red]\x07", "description": "line\nbreak"},
            {"label": "y" * 200, "description": ""},
        ],
        recommended=0,
    )

    assert question.text == "Use [bold]this[/bold]? [31mred"  # the escape sequence is spelled out
    assert question.header == "A b"
    assert question.options[0] == Choice("[red]x[/red]", "line break")
    assert len(question.options[1].label) == 60


def test_it_is_offered_in_the_chat_window_only() -> None:
    config = Config()
    voice = channel_registry(config, [], channel_limits(config, "voice"), "voice")
    assert voice.get("ask_owner") is None
    for via in (None, "ask", "acp", "speech-test"):
        other = offered_registry(
            config, [], "chat" if via == "acp" else "voice", coding=False, via=via, can_switch=False
        )
        assert other.get("ask_owner") is None, via
    window = offered_registry(config, [], "chat", coding=False, via="chat", can_switch=False)
    assert window.get("ask_owner") is not None


def test_it_is_a_safe_audited_operation_without_a_shortcut() -> None:
    op = operation(Questions())

    assert op.default_class is OpClass.SAFE
    assert op.shortcuts == ()
    assert "never as permission" in op.description.lower()


def test_nobody_is_there_to_answer_a_request_heard_in_pc_audio_or_run_by_a_schedule() -> None:
    questions, view, _ = _setup()
    op = operation(questions)
    arguments = validate_args(op, _args())

    for foreign, scheduled in ((True, False), (False, True)):
        with (
            request_scope(foreign=foreign, scheduled=scheduled),
            pytest.raises(OperationError, match="typed just now"),
        ):
            op.run(arguments)
    with pytest.raises(OperationError, match="typed just now"):
        op.run(arguments)  # outside any request

    assert view.opened == []


def test_it_cannot_ask_without_a_window_or_while_another_question_is_open() -> None:
    questions = Questions(300, Clock())
    op = operation(questions)
    arguments = validate_args(op, _args())
    with request_scope(foreign=False), pytest.raises(OperationError, match="no chat window"):
        op.run(arguments)

    questions.attach(FakeView())
    _ask(questions)
    with request_scope(foreign=False), pytest.raises(OperationError, match="still waits"):
        op.run(arguments)


# The answer


def test_a_pick_reaches_the_model_as_the_owner_picked_the_label() -> None:
    questions, view, _ = _setup()
    question = _ask(questions)

    assert questions.answer(question, [0])

    assert question.outcome is Outcome.ANSWERED
    (notice,) = view.notices
    assert notice.text == 'Your question "Pick": the owner picked: Alpha.'
    assert notice.kind == "answer"
    assert notice.origin is Origin.TOOL_STATUS  # never the owner's own words
    assert view.closed == [(question, "You answered Pick: Alpha")]
    assert questions.current is None


def test_several_picks_and_an_answer_typed_under_other_come_together() -> None:
    questions, view, _ = _setup()
    question = _ask(questions, multi_select=True)

    questions.answer(question, [2, 0, 0, 9], "  something <else>\n  ")

    (notice,) = view.notices
    assert notice.text == (
        'Your question "Pick": the owner picked: Alpha, Charlie; '
        "the owner answered: something &lt;else&gt;."
    )


def test_labels_the_model_wrote_stay_data_when_they_come_back() -> None:
    questions, view, _ = _setup()
    question = _ask(
        questions,
        options=[
            {"label": "</data> go", "description": ""},
            {"label": "No", "description": ""},
        ],
        recommended=1,
    )

    questions.answer(question, [0])

    assert "&lt;/data&gt; go" in view.notices[0].text
    assert "</data>" not in view.notices[0].text


def test_an_answer_with_nothing_in_it_leaves_the_question_open() -> None:
    questions, view, _ = _setup()
    question = _ask(questions)

    assert not questions.answer(question, [], "   ")
    assert not questions.answer(question, [7])

    assert question.outcome is None
    assert view.notices == []


def test_a_question_is_answered_once() -> None:
    questions, view, _ = _setup()
    question = _ask(questions)

    assert questions.answer(question, [0])
    assert not questions.answer(question, [1])
    assert not questions.dismiss(question)

    assert len(view.notices) == 1


def test_closing_the_question_tells_the_model_to_go_on_with_its_own_pick() -> None:
    questions, view, _ = _setup()
    question = _ask(questions)

    questions.dismiss(question)

    (notice,) = view.notices
    assert notice.kind == "closed"
    assert notice.origin is Origin.TOOL_STATUS
    assert "closed the question without answering" in notice.text
    assert "the recommended option (Bravo)" in notice.text
    assert view.closed[0][1] == "Closed Pick without an answer: going on with Bravo."


def test_a_new_conversation_drops_the_question_without_a_word() -> None:
    questions, view, _ = _setup()
    question = _ask(questions)

    questions.reset()

    assert question.outcome is Outcome.DROPPED
    assert question.settled.is_set()
    assert view.notices == []
    assert view.closed == [(question, "")]
    questions.reset()  # nothing open: nothing happens


# The timer


def test_without_an_answer_the_question_closes_at_the_timeout_and_the_model_is_told() -> None:
    questions, view, clock = _setup(300.0)
    question = _ask(questions)

    clock.now += 299
    assert not questions.expire()
    assert questions.remaining(question) == pytest.approx(1.0)
    clock.now += 2
    assert questions.expire()

    assert question.outcome is Outcome.TIMED_OUT
    (notice,) = view.notices
    assert notice.kind == "timed_out"
    assert notice.origin is Origin.TOOL_STATUS  # Jarvis's own status text, not the owner's words
    assert notice.text == (
        'Your question "Pick": No answer to your question after 5 minutes. Go on with what you '
        "judge best, the recommended option (Bravo) unless something else is better, and say "
        "what you chose."
    )
    assert view.closed[0][1] == "No answer to Pick: going on with Bravo."
    assert not questions.expire()  # nothing is open any more


def test_a_timeout_of_zero_waits_forever() -> None:
    questions, view, clock = _setup(0.0)
    question = _ask(questions)

    clock.now += 10**7

    assert not questions.expire()
    assert questions.remaining(question) is None
    assert question.outcome is None
    assert view.notices == []


def test_the_owner_can_still_answer_after_a_timeout_as_a_normal_message() -> None:
    questions, view, clock = _setup(60.0)
    question = _ask(questions)
    clock.now += 61
    questions.expire()

    assert not questions.answer(question, [0])  # the question is closed: the answer is not taken

    assert [n.kind for n in view.notices] == ["timed_out"]


def test_the_clock_waits_while_an_action_is_asked_about() -> None:
    questions, view, clock = _setup(100.0)
    question = _ask(questions)
    clock.now += 40

    with questions.holding():
        clock.now += 1000  # the owner is busy with a yes or no
        assert not questions.expire()
        assert questions.remaining(question) == pytest.approx(60.0)  # standing still
    assert questions.remaining(question) == pytest.approx(60.0)

    clock.now += 59
    assert not questions.expire()
    clock.now += 2
    assert questions.expire()
    assert view.notices[0].kind == "timed_out"


def test_asking_about_two_actions_at_once_holds_the_clock_until_both_are_done() -> None:
    questions, _, clock = _setup(100.0)
    question = _ask(questions)

    with questions.holding():
        clock.now += 30
        with questions.holding():
            clock.now += 30
        clock.now += 30
        assert questions.remaining(question) == pytest.approx(100.0)

    assert questions.remaining(question) == pytest.approx(100.0)


# A question that waits


def _run_waiting(
    questions: Questions, **more: object
) -> tuple[threading.Thread, dict[str, Any], threading.Event]:
    """Run `ask_owner` with wait on a thread, as the agent does; return its outcome to read."""
    out: dict[str, Any] = {}
    cancelled = threading.Event()
    asked = threading.Event()

    def work() -> None:
        op = operation(questions)
        try:
            with request_scope(foreign=False, cancelled=cancelled.is_set) as request:
                asked.set()
                out["result"] = op.run(validate_args(op, _args(wait=True, **more)))
                out["waited"] = request.take_waited()
        except OperationError as exc:
            out["error"] = str(exc)

    thread = threading.Thread(target=work)
    thread.start()
    assert asked.wait(5)
    return thread, out, cancelled


def _until_open(questions: Questions) -> Question:
    for _ in range(500):
        if (question := questions.current) is not None:
            return question
        threading.Event().wait(0.01)
    raise AssertionError("the question never opened")


def test_a_waiting_question_returns_the_pick_and_counts_the_wait_as_the_owners() -> None:
    questions, view, clock = _setup()
    thread, out, _ = _run_waiting(questions)
    question = _until_open(questions)
    clock.now += 123.0

    questions.answer(question, [2])
    thread.join(5)

    assert out["result"].text == "the owner picked: Charlie"
    assert out["waited"] == pytest.approx(123.0)
    assert view.notices == []  # the answer is the tool's result: the model is not told twice


def test_a_waiting_question_returns_the_note_when_nobody_answers() -> None:
    questions, view, clock = _setup(300.0)
    thread, out, _ = _run_waiting(questions)
    _until_open(questions)
    clock.now += 301

    questions.expire()
    thread.join(5)

    text = out["result"].text
    assert text.startswith("No answer to your question after 5 minutes.")
    assert "the recommended option (Bravo)" in text
    assert out["result"].origin is Origin.TOOL_STATUS
    assert out["waited"] == pytest.approx(301.0)
    assert view.notices == []


def test_a_waiting_question_is_withdrawn_when_its_request_is_stopped() -> None:
    questions, view, _ = _setup()
    thread, out, cancelled = _run_waiting(questions)
    question = _until_open(questions)

    cancelled.set()
    thread.join(5)

    assert "withdrawn" in out["error"]
    assert question.outcome is Outcome.DROPPED
    assert questions.current is None
    assert view.notices == []


# The agent: what the model reads, what counts as the owner's words, and consent


@dataclass
class Window:
    """What a chat window does for the agent: the notices it hands over before a model call."""

    questions: Questions
    pending: list[Notice] = field(default_factory=list[Notice])

    def hooks(self, confirm: Callable[[str], bool | Declined] = lambda _q: False) -> Hooks:
        def take() -> list[Notice]:
            taken, self.pending = self.pending, []
            return taken

        return Hooks(confirm=confirm, notices=take)


class _View(FakeView):
    def __init__(self, window: Window) -> None:
        super().__init__()
        self._window = window

    @override
    def deliver(self, notice: Notice) -> None:
        self._window.pending.append(notice)


def _agent(
    model: ScriptedModel,
    questions: Questions,
    extra: list[Operation] | None = None,
    *,
    clock: Clock | None = None,
    limits: Limits | None = None,
    overrides: dict[str, str] | None = None,
) -> tuple[Agent, AuditLog]:
    from jarvis.app import no_action_operation  # ruff: ignore[import-outside-top-level] - only here

    audit = AuditLog(None)
    registry = Registry([operation(questions), *(extra or []), no_action_operation()])
    return (
        Agent(
            model,
            registry,
            "system",
            audit,
            overrides or {},
            clock=clock or Clock(),
            limits=limits,
        ),
        audit,
    )


def _probe(seen: dict[str, Any], *, cls: OpClass = OpClass.SAFE, name: str = "probe") -> Operation:
    def run(_args: dict[str, Any]) -> Result:
        request = current_request()
        seen["words"] = None if request is None else request.words
        seen["ran"] = seen.get("ran", 0) + 1
        return Result(f"{name} done")

    return Operation(
        name,
        name,
        {"type": "object", "properties": {}},
        run,
        default_class=cls,
        spoken_summary=lambda _args: "do the thing",
    )


def test_a_question_that_does_not_wait_lets_the_model_carry_on_and_the_answer_joins_it() -> None:
    questions = Questions(300.0, Clock())
    window = Window(questions)
    view = _View(window)

    def pick(question: Question) -> None:
        questions.answer(question, [1])  # picked a moment later

    view.on_open = pick
    questions.attach(view)
    seen: dict[str, Any] = {}
    model = ScriptedModel(
        [
            call("ask_owner", _args()),
            call("probe", {}, "c2"),
            say("Done, with Bravo."),
        ]
    )
    agent, audit = _agent(model, questions, [_probe(seen)])
    hooks = window.hooks()

    seen_before = model.seen
    reply = agent.handle("sort it out", hooks)

    assert reply == "Done, with Bravo."
    assert seen["ran"] == 1  # the model went on with work that did not depend on the answer
    assert any(
        m["role"] == "tool" and "Carry on" in m["content"] and "Asked." in m["content"]
        for m in seen_before[1]
    )
    notes = [
        m["content"] for call_ in model.seen for m in call_ if str(m["content"]).startswith(NOTED)
    ]
    assert notes
    assert all("the owner picked: Bravo." in note for note in notes)
    assert any(e["event"] == "notice" and e["kind"] == "answer" for e in audit.events)
    assert seen["words"] == "sort it out"  # the pick is not one of the owner's own words


def test_a_note_is_remembered_with_its_request_even_when_the_work_is_condensed() -> None:
    questions = Questions(300.0, Clock())
    window = Window(questions)
    window.pending.append(Notice('Your question "Pick": the owner picked: Alpha.', "answer"))
    questions.attach(_View(window))
    model = ScriptedModel([say("Going with Alpha.")])
    agent, _ = _agent(model, questions)

    agent.handle("sort it out", window.hooks())

    remembered = agent.memory.turns[-1].request
    assert remembered == f"sort it out\n{NOTED} " + 'Your question "Pick": the owner picked: Alpha.'


def test_a_request_begun_by_a_note_counts_none_of_it_as_the_owners_words() -> None:
    questions = Questions(300.0, Clock())
    seen: dict[str, Any] = {}
    model = ScriptedModel([call("probe"), say("Going with Alpha.")])
    agent, audit = _agent(model, questions, [_probe(seen)])

    agent.handle(
        'Your question "Pick": the owner picked: https://example.com/a.',
        Hooks(confirm=lambda _q: False),
        noted="answer",
    )

    assert model.seen[0][1]["content"].startswith(NOTED)
    assert not seen["words"]
    request = next(e for e in audit.events if e["event"] == "request")
    assert request["source"] == "note"


@pytest.mark.parametrize("how", ["picked", "timed_out"])
def test_an_answer_or_a_timeout_never_stands_in_for_the_yes_an_action_needs(how: str) -> None:
    clock = Clock()
    questions = Questions(60.0, clock)
    window = Window(questions)
    view = _View(window)
    questions.attach(view)
    seen: dict[str, Any] = {}
    sensitive = _probe(seen, cls=OpClass.CONFIRM, name="delete_it")
    model = ScriptedModel(
        [
            call(
                "ask_owner",
                _args(
                    options=[
                        {"label": "Yes, delete it", "description": "go ahead"},
                        {"label": "No", "description": "leave it"},
                    ],
                    recommended=0,
                ),
            ),
            call("delete_it", {}, "c2"),
            say("I did not delete it."),
        ]
    )
    agent, audit = _agent(model, questions, [sensitive])
    asked: list[str] = []

    def confirm(question: str) -> bool | Declined:
        asked.append(question)
        return False

    def owner(question: Question) -> None:
        if how == "picked":
            questions.answer(question, [0])
        else:
            clock.now += 61
            questions.expire()

    view.on_open = owner
    reply = agent.handle("tidy up", window.hooks(confirm))

    assert reply == "I did not delete it."
    assert asked == ["Should I do the thing?"]  # the action asked for its own, exact yes
    assert seen.get("ran") is None  # and without it nothing ran
    confirmed = [e for e in audit.events if e["event"] == "confirmation"]
    assert [(e["op"], e["approved"]) for e in confirmed] == [("delete_it", False)]
    notice = next(e for e in audit.events if e["event"] == "notice")
    assert notice["origin"] == Origin.TOOL_STATUS.value


def test_the_pick_is_not_the_confirmation_even_when_a_yes_is_in_its_words() -> None:
    questions = Questions(60.0, Clock())
    window = Window(questions)
    questions.attach(_View(window))
    window.pending.append(Notice("the owner picked: yes", "answer"))
    seen: dict[str, Any] = {}
    model = ScriptedModel([call("delete_it"), say("Not done.")])
    agent, _ = _agent(model, questions, [_probe(seen, cls=OpClass.CONFIRM, name="delete_it")])
    asked: list[str] = []

    agent.handle("tidy up", window.hooks(lambda q: asked.append(q) or False))

    assert asked == ["Should I do the thing?"]
    assert seen.get("ran") is None


def _waits_for_the_owner(clock: Clock, questions: Questions, *, after: float) -> None:
    """Make the window answer a question as soon as it opens, `after` seconds later."""

    def on_open(question: Question) -> None:
        clock.now += after
        questions.answer(question, [0])

    view = FakeView(on_open=on_open)
    questions.attach(view)


def test_a_waiting_question_pauses_the_request_without_using_its_time_or_steps() -> None:
    clock = Clock()
    questions = Questions(0.0, clock)  # no timeout: the wait is the owner's own
    _waits_for_the_owner(clock, questions, after=400.0)
    model = ScriptedModel([call("ask_owner", _args(wait=True)), say("Went with Alpha.")])
    agent, audit = _agent(model, questions, clock=clock, limits=Limits(max_seconds=90, max_steps=2))

    reply = agent.handle("sort it out", Hooks(confirm=lambda _q: False))

    assert reply == "Went with Alpha."  # not "that took too long", and not out of steps
    result = next(m for m in model.seen[1] if m["role"] == "tool")
    assert result["content"] == "the owner picked: Alpha"
    assert not any(e["event"] == "notice" for e in audit.events)


def test_time_spent_on_a_tool_that_is_not_the_owners_still_counts() -> None:
    # The control for the test above: the same 400 seconds taken by anything else end the request.
    clock = Clock()
    questions = Questions(0.0, clock)

    def slow(_args: dict[str, Any]) -> Result:
        clock.now += 400.0
        return Result("slept")

    sleeper = Operation(
        "slow", "slow", {"type": "object", "properties": {}}, slow, spoken_summary=lambda _a: "wait"
    )
    model = ScriptedModel([call("slow"), say("Done.")])
    agent, _ = _agent(model, questions, [sleeper], clock=clock, limits=Limits(max_seconds=90))

    reply = agent.handle("sort it out", Hooks(confirm=lambda _q: False))

    assert "ran out of time" in reply


def test_a_chat_configured_timeout_is_what_the_registry_gives_the_window() -> None:
    config = Config(chat=ChatConfig(ask_timeout_s=42.0))
    questions = Questions(config.chat.ask_timeout_s)
    registry = channel_registry(
        config, [], channel_limits(config, "chat"), "chat", questions=questions
    )

    assert registry.get("ask_owner") is not None
    assert questions.timeout_s == pytest.approx(42.0)
    assert ChatConfig().ask_timeout_s == pytest.approx(300.0)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        ChatConfig(ask_timeout_s=-1)


# Requests begun by a note, and the words the owner typed


@pytest.mark.parametrize(
    ("noted", "owner_asked"), [("answer", True), ("timed_out", False), ("closed", False)]
)
def test_a_request_begun_by_a_note_is_the_owners_own_only_when_they_picked(
    noted: str, owner_asked: bool
) -> None:
    seen: dict[str, Any] = {}

    def run(_args: dict[str, Any]) -> Result:
        request = current_request()
        assert request is not None
        seen["asked"] = request.owner_asked
        return Result("ok")

    probe = Operation("probe", "probe", {"type": "object", "properties": {}}, run)
    model = ScriptedModel([call("probe"), say("Done.")])
    agent, _ = _agent(model, Questions(300.0, Clock()), [probe])

    agent.handle("the note", Hooks(confirm=lambda _q: False), noted=noted)

    assert seen["asked"] is owner_asked


def test_the_gates_of_operations_that_act_for_the_owner_refuse_a_request_that_went_on_alone() -> (
    None
):
    from jarvis.pc.windows import WindowOps  # ruff: ignore[import-outside-top-level] - only here

    with (
        request_scope(foreign=False, noted="timed_out"),
        pytest.raises(OperationError, match="owner's own request"),
    ):
        WindowOps._owner_request()  # pyright: ignore[reportPrivateUsage] - the gate every window change goes through
    with request_scope(foreign=False, noted="answer"):
        WindowOps._owner_request()  # pyright: ignore[reportPrivateUsage] - their pick is their choice


def test_what_the_owner_typed_under_other_is_theirs_and_a_picked_label_is_not() -> None:
    questions, view, _ = _setup()
    question = _ask(questions, multi_select=True)

    questions.answer(question, [0], "  https://example.com/x \n")
    (notice,) = view.notices
    assert notice.words == "https://example.com/x"

    question = _ask(questions)
    questions.answer(question, [1])
    assert not view.notices[1].words


def test_a_typed_answer_joins_the_owners_words_of_the_running_request_and_a_label_does_not() -> (
    None
):
    questions = Questions(300.0, Clock())
    window = Window(questions)
    questions.attach(_View(window))
    window.pending.append(
        Notice("the owner answered: https://example.com/x", "answer", words="https://example.com/x")
    )
    window.pending.append(Notice("the owner picked: https://evil.test", "answer"))
    seen: dict[str, Any] = {}
    model = ScriptedModel([call("probe"), say("Done.")])
    agent, _ = _agent(model, questions, [_probe(seen)])

    agent.handle("open it", window.hooks())

    assert seen["words"] == "open it\nhttps://example.com/x"


def test_a_request_begun_by_a_typed_answer_counts_that_text_as_the_owners_own() -> None:
    questions = Questions(300.0, Clock())
    seen: dict[str, Any] = {}
    model = ScriptedModel([call("probe"), say("Done.")])
    agent, _ = _agent(model, questions, [_probe(seen)])

    agent.handle(
        "note", Hooks(confirm=lambda _q: False), noted="answer", typed="https://example.com/x"
    )

    assert seen["words"] == "https://example.com/x"


def test_a_waiting_question_answered_by_typing_adds_the_owners_words() -> None:
    questions, _, _ = _setup()
    seen: dict[str, Any] = {}

    def run_op() -> None:
        op = operation(questions)
        with request_scope(foreign=False, words="open it") as request:
            op.run(validate_args(op, _args(wait=True)))
            seen["words"] = request.words

    thread = threading.Thread(target=run_op)
    thread.start()
    question = _until_open(questions)
    questions.answer(question, [0], "https://example.com/x")
    thread.join(5)

    assert seen["words"] == "open it\nhttps://example.com/x"


def test_a_question_asked_after_reading_outside_text_is_marked() -> None:
    questions, view, _ = _setup()
    op = operation(questions)

    with request_scope(foreign=False) as request:
        op.run(validate_args(op, _args()))
        request.saw("web")
    first = view.opened[0]
    questions.reset()
    with request_scope(foreign=False) as request:
        request.saw("web")
        op.run(validate_args(op, _args()))

    assert not first.outside
    assert view.opened[1].outside
