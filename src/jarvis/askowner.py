"""`ask_owner`: a question with options the chat shows as a picker, and what happens to its answer.

The model asks when it needs a decision it cannot judge itself and always names the option it
would pick. The chat window shows the picker (it adds "Other (type an answer)"), so the question
is offered only there, never by voice. With `wait` false the call returns at once and the model
carries on; the answer, or a note that none came, joins the request that runs or starts one.
With `wait` true the request pauses on the question, and the pause counts against neither its
time limit nor its steps. With no answer in `chat.ask_timeout_s` the question closes with a
visible note and the model is told, as Jarvis's own status text, to go on with what it judges best.

An answer is never consent (ADR 0026). It reaches the model as a note or a tool result and goes
nowhere else: the confirmation flow of an action is a separate question of its own, asked of the
owner by the window's yes-or-no choice, and the timer waits while one is open.
"""

import contextlib
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from jarvis.jsonutil import objs
from jarvis.operations import (
    Notice,
    Operation,
    OperationError,
    Result,
    inert,
    one_line,
)
from jarvis.operations import current_request as request_now
from jarvis.policy import OpClass

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

NAME = "ask_owner"
MAX_HEADER = 12
MIN_OPTIONS = 2
MAX_OPTIONS = 4
MAX_LABEL = 60
MAX_TYPED = 400
DEFAULT_TIMEOUT_S = 300.0
MINUTE = 60
POLL_S = 0.25  # how often a waiting call looks whether its request was stopped


class Outcome(StrEnum):
    """How a question ended."""

    ANSWERED = "answered"  # the owner picked options or typed an answer
    TIMED_OUT = "timed_out"  # no answer within `chat.ask_timeout_s`
    CLOSED = "closed"  # the owner closed the picker without answering
    DROPPED = "dropped"  # the request was stopped, or the conversation or window ended


@dataclass(frozen=True, slots=True)
class Choice:
    """One option: a short label and what choosing it means."""

    label: str
    description: str


@dataclass(eq=False)
class Question:
    """A question open to the owner. `recommended` is the option's position in `options`."""

    header: str
    text: str
    options: tuple[Choice, ...]
    multi: bool
    recommended: int
    wait: bool
    opened: float
    deadline: float | None
    limit_s: float
    outside: bool = False  # written after the request read outside text: the picker says so
    settled: threading.Event = field(default_factory=threading.Event)
    outcome: Outcome | None = None
    picks: tuple[int, ...] = ()
    typed: str = ""
    closed: float = 0.0

    @property
    def order(self) -> tuple[int, ...]:
        """The options' positions as they are listed: the recommended one first."""
        return (self.recommended, *(n for n in range(len(self.options)) if n != self.recommended))

    @property
    def waited(self) -> float:
        """Seconds from the question opening to its end (0 while open)."""
        return max(0.0, self.closed - self.opened) if self.outcome is not None else 0.0


class View(Protocol):
    """What shows a question to the owner (the chat window); any thread may call it."""

    def open(self, question: Question) -> None:
        """Show the picker."""
        ...

    def close(self, question: Question, note: str) -> None:
        """Take the picker down and show `note` (empty: nothing) in its place."""
        ...

    def deliver(self, notice: Notice) -> None:
        """Hand the model a notice about a question that did not wait (a pick or a timeout)."""
        ...


def span(seconds: float) -> str:
    """Return a stretch of time as it is said: "5 minutes", "45 seconds"."""
    if seconds >= MINUTE:
        count = round(seconds / MINUTE)
        return f"{count} minute{'' if count == 1 else 's'}"
    count = round(seconds)
    return f"{count} second{'' if count == 1 else 's'}"


def _data(text: str) -> str:
    """Return written words as the model reads them: one line, angle brackets spelled out."""
    return inert(one_line(text))


class Questions:
    """The one question open at a time, its timer and its answer.

    `timeout_s` is how long the owner has (0 waits forever); `clock` counts seconds. The window
    answers (`answer`, `dismiss`), ticks the timer (`expire`) and tells it while a confirmation
    of an action waits (`holding`): the countdown stops then, so a question cannot run out
    while the owner is busy with a yes or no.
    """

    def __init__(
        self, timeout_s: float = DEFAULT_TIMEOUT_S, clock: Callable[[], float] = time.monotonic
    ) -> None:
        """Start with no window and no question."""
        super().__init__()
        self.timeout_s = timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._view: View | None = None
        self._current: Question | None = None
        self._holds = 0
        self._held_at = 0.0

    def attach(self, view: View | None) -> None:
        """Show questions in `view` (None: no window; asking then fails)."""
        with self._lock:
            self._view = view

    @property
    def current(self) -> Question | None:
        """The open question, if any."""
        with self._lock:
            return self._current

    def ask(  # ruff: ignore[too-many-arguments] - a question's parts, keyword-only
        self,
        *,
        header: str,
        text: str,
        options: Sequence[Choice],
        multi: bool,
        recommended: int,
        wait: bool,
        outside: bool = False,
    ) -> Question:
        """Open a question in the window and return it.

        Raises:
            OperationError: no window shows questions, or another question is still open.
        """
        with self._lock:
            view = self._view
            if view is None:
                msg = "there is no chat window to ask in"
                raise OperationError(msg)
            if self._current is not None:
                msg = (
                    "another question still waits for the owner; do not ask a second one until "
                    "it is answered or closes"
                )
                raise OperationError(msg)
            now = self._clock()
            question = Question(
                header,
                text,
                tuple(options),
                multi,
                recommended,
                wait,
                now,
                now + self.timeout_s if self.timeout_s > 0 else None,
                self.timeout_s,
                outside,
            )
            self._current = question
        try:
            view.open(question)
        except Exception:
            with self._lock:
                self._current = None
            raise
        return question

    def remaining(self, question: Question) -> float | None:
        """Return the seconds left to answer (None: no limit), still while an action is asked."""
        if question.deadline is None:
            return None
        with self._lock:
            now = self._held_at if self._holds else self._clock()
        return max(0.0, question.deadline - now)

    @contextlib.contextmanager
    def holding(self) -> Generator[None]:
        """Stop the countdown while the owner is asked about an action; it resumes after."""
        with self._lock:
            if not self._holds:
                self._held_at = self._clock()
            self._holds += 1
        try:
            yield
        finally:
            with self._lock:
                self._holds -= 1
                question = self._current
                if not self._holds and question is not None and question.deadline is not None:
                    question.deadline += self._clock() - self._held_at

    def answer(self, question: Question, picks: Sequence[int], typed: str = "") -> bool:
        """Settle `question` with the owner's picks (positions in `options`) and typed words.

        Returns False when it already ended or the answer holds nothing.
        """
        typed = one_line(typed)[:MAX_TYPED]
        chosen = tuple(sorted({n for n in picks if 0 <= n < len(question.options)}))
        if not chosen and not typed:
            return False
        return self._settle(question, Outcome.ANSWERED, chosen, typed)

    def dismiss(self, question: Question) -> bool:
        """Settle `question` as closed by the owner without an answer."""
        return self._settle(question, Outcome.CLOSED)

    def withdraw(self, question: Question) -> bool:
        """Settle `question` as dropped (its request was stopped)."""
        return self._settle(question, Outcome.DROPPED)

    def expire(self) -> bool:
        """Close the open question when its time is up, unless an action is being asked about.

        Returns whether it closed one. The window calls it a few times a second.
        """
        with self._lock:
            question = self._current
            late = (
                question is not None
                and question.deadline is not None
                and not self._holds
                and self._clock() >= question.deadline
            )
        return late and question is not None and self._settle(question, Outcome.TIMED_OUT)

    def reset(self) -> None:
        """Drop the open question, telling the model nothing (new conversation, closing window)."""
        with self._lock:
            question = self._current
        if question is not None:
            self._settle(question, Outcome.DROPPED, silent=True)

    def _settle(
        self,
        question: Question,
        outcome: Outcome,
        picks: tuple[int, ...] = (),
        typed: str = "",
        *,
        silent: bool = False,
    ) -> bool:
        with self._lock:
            if question.outcome is not None or self._current is not question:
                return False
            question.outcome, question.picks, question.typed = outcome, picks, typed
            question.closed = self._clock()
            self._current = None
            view = self._view
        note = visible_note(question)
        question.settled.set()
        if view is None:
            return True
        view.close(question, "" if silent else note)
        if not question.wait and not silent and outcome is not Outcome.DROPPED:
            view.deliver(notice_of(question))
        return True

    def wait(self, question: Question, cancelled: Callable[[], bool]) -> None:
        """Block until `question` ends; a request that is stopped meanwhile withdraws it."""
        while not question.settled.wait(POLL_S):
            if cancelled():
                self.withdraw(question)
                return


def recommended_label(question: Question) -> str:
    """Return the label of the option the model marked as its pick."""
    return _data(question.options[question.recommended].label)


def answer_text(question: Question) -> str:
    """Return what the owner chose, as the model reads it: "the owner picked: A, B"."""
    parts: list[str] = []
    if question.picks:
        labels = ", ".join(_data(question.options[n].label) for n in question.picks)
        parts.append(f"the owner picked: {labels}")
    if question.typed:
        parts.append(f"the owner answered: {_data(question.typed)}")
    return "; ".join(parts)


def _go_on(question: Question) -> str:
    return (
        "Go on with what you judge best, the recommended option "
        f"({recommended_label(question)}) unless something else is better, and say what you chose."
    )


def result_text(question: Question) -> str:
    """Return what a waiting call hands back to the model once its question ended."""
    match question.outcome:
        case Outcome.ANSWERED:
            return answer_text(question)
        case Outcome.TIMED_OUT:
            return f"No answer to your question after {span(question.limit_s)}. {_go_on(question)}"
        case Outcome.CLOSED:
            return f"The owner closed the question without answering. {_go_on(question)}"
        case _:
            return "The question was withdrawn before the owner answered."


def notice_of(question: Question) -> Notice:
    """Return the notice a question that did not wait gives the model when it ends."""
    heading = f'Your question "{_data(question.header)}": '
    text = result_text(question)
    if question.outcome is Outcome.ANSWERED:
        return Notice(f"{heading}{text}.", "answer", words=question.typed)
    kind = question.outcome.value if question.outcome is not None else "dropped"
    return Notice(f"{heading}{text}", kind)


def visible_note(question: Question) -> str:
    """Return the line the window shows once a question ended."""
    match question.outcome:
        case Outcome.ANSWERED:
            given = [question.options[n].label for n in question.picks]
            given += [question.typed] if question.typed else []
            return f"You answered {question.header}: {', '.join(given)}"
        case Outcome.TIMED_OUT:
            return (
                f"No answer to {question.header}: going on with "
                f"{question.options[question.recommended].label}."
            )
        case Outcome.CLOSED:
            return (
                f"Closed {question.header} without an answer: going on with "
                f"{question.options[question.recommended].label}."
            )
        case _:
            return f"Withdrew {question.header}."


ABOUT = (
    "Ask the owner to pick among 2 to 4 options, for a decision you cannot judge "
    "yourself. Never as permission for an action: an answer is never a yes. `recommended` is "
    "the 0-based option you would pick. wait true pauses the request for the answer; otherwise "
    "carry on and it reaches you later."
)


def operation(questions: Questions) -> Operation:
    """Return `ask_owner`: the model's way to put a choice to the owner in the chat.

    It is refused for a request heard in PC audio or run by a schedule (nobody is there to
    answer), when the options repeat or `recommended` points at none, and while another question
    is open. Everything the model wrote is shown as plain text, and a question written after the
    request read outside text says so, since a page may have steered its words.
    """

    def run(args: dict[str, object]) -> Result:
        request = request_now()
        if request is None or request.foreign or request.scheduled:
            msg = "only a request the owner typed just now can ask them a question"
            raise OperationError(msg)
        options = tuple(
            Choice(one_line(str(row["label"]))[:MAX_LABEL], one_line(str(row["description"])))
            for row in objs(args["options"])
        )
        recommended = int(str(args["recommended"]))
        header = one_line(str(args["header"]))[:MAX_HEADER]
        text = one_line(str(args["question"]))
        if not header or not text or any(not option.label for option in options):
            msg = "the question, its header and every option's label need words"
            raise OperationError(msg)
        if len({option.label.casefold() for option in options}) < len(options):
            msg = "the options must differ"
            raise OperationError(msg)
        if not 0 <= recommended < len(options):
            msg = f"recommended must be 0 to {len(options) - 1}, the position of one option"
            raise OperationError(msg)
        wait = args.get("wait") is True
        question = questions.ask(
            header=header,
            text=text,
            options=options,
            multi=args.get("multi_select") is True,
            recommended=recommended,
            wait=wait,
            outside=bool(request.outside),
        )
        if not wait:
            limit = (
                f"or a note after {span(questions.timeout_s)} without one"
                if questions.timeout_s > 0
                else "and nothing closes it on its own"
            )
            return Result(
                "Asked. The answer reaches you as a message when the owner gives it, "
                f"{limit}. Carry on with what does not depend on it; if nothing is left, say "
                "you are waiting for their answer."
            )
        questions.wait(question, request.cancelled)
        request.waited(question.waited)
        if question.outcome is Outcome.ANSWERED and question.typed:
            request.added(question.typed)  # typed by the owner: theirs, unlike a picked label
        if question.outcome is Outcome.DROPPED:
            msg = "the question was withdrawn before the owner answered"
            raise OperationError(msg)
        return Result(result_text(question))

    return Operation(
        "ask_owner",  # the scan in test_product_rules reads a literal
        ABOUT,
        {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "header": {"type": "string", "maxLength": MAX_HEADER},
                "options": {
                    "type": "array",
                    "minItems": MIN_OPTIONS,
                    "maxItems": MAX_OPTIONS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["label", "description"],
                    },
                },
                "multi_select": {"type": "boolean"},
                "recommended": {"type": "integer", "minimum": 0},
                "wait": {"type": "boolean"},
            },
            "required": ["question", "header", "options", "recommended"],
            "additionalProperties": False,
        },
        run,
        default_class=OpClass.SAFE,
        side_effect=False,
        spoken_summary=lambda _args: "ask you a question",
    )
