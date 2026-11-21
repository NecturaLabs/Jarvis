"""The chat window: typed requests, the owner's yes or no, and what Jarvis does, shown live."""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from textual.widgets import Input, RichLog, Static

from jarvis.chat import ChatApp

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.pilot import Pilot

    from jarvis.agent import Hooks


@dataclass
class FakeTail:
    entries: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    closed: bool = False

    def read(self) -> list[dict[str, object]]:
        out, self.entries = self.entries, []
        return out

    def close(self) -> None:
        self.closed = True


def _shown(app: ChatApp) -> list[str]:
    return [line.text for line in app.query_one("#activity", RichLog).lines]


async def _until(pilot: Pilot[None], ready: Callable[[], bool]) -> None:
    for _ in range(500):
        if ready():
            return
        await pilot.pause(0.01)
    raise AssertionError("the window never got there")


async def _type(pilot: Pilot[None], text: str) -> None:
    pilot.app.query_one(Input).value = text
    await pilot.press("enter")


def _run(test: Callable[[Pilot[None]], object], app: ChatApp) -> None:
    async def body() -> None:
        async with app.run_test() as pilot:
            await test(pilot)  # pyright: ignore[reportGeneralTypeIssues]

    asyncio.run(body())


def test_a_typed_request_runs_and_what_happens_is_shown() -> None:
    tail = FakeTail()
    asked: list[str] = []

    def ask(text: str, _hooks: Hooks) -> str:
        asked.append(text)
        tail.entries += [
            {"event": "request", "text": text, "via": "chat"},
            {"event": "reply", "text": "Paused it for you.", "seconds": 0.2},
        ]
        return "Paused it for you."

    app = ChatApp(tail, ask, poll_s=0.01)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: any("Paused it" in line for line in _shown(app)))
        assert asked == ["pause"]
        assert any("You (typed): pause" in line for line in _shown(app))

    _run(test, app)


def test_voice_activity_shows_without_typing() -> None:
    tail = FakeTail([{"event": "request", "text": "skip", "origin": "owner_speech"}])
    app = ChatApp(tail, None, poll_s=0.01)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: any("You: skip" in line for line in _shown(app)))

    _run(test, app)


def _confirming(answers: list[bool], done: threading.Event) -> Callable[[str, Hooks], str]:
    def ask(_text: str, hooks: Hooks) -> str:
        answers.append(hooks.confirm("Should I queue Low Tide?"))
        done.set()
        return ""

    return ask


def test_the_owner_answers_a_question_by_typing() -> None:
    answers: list[bool] = []
    done = threading.Event()
    app = ChatApp(FakeTail(), _confirming(answers, done), poll_s=0.01)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: any("Should I queue Low Tide?" in s for s in _shown(app)))
        await _type(pilot, "yes")
        await _until(pilot, done.is_set)

    _run(test, app)
    assert answers == [True]


def test_anything_but_a_clear_yes_is_a_no() -> None:
    answers: list[bool] = []
    done = threading.Event()
    app = ChatApp(FakeTail(), _confirming(answers, done), poll_s=0.01)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: any("Should I" in line for line in _shown(app)))
        await _type(pilot, "yes, don't")
        await _until(pilot, done.is_set)

    _run(test, app)
    assert answers == [False]


def test_leaving_while_a_question_waits_answers_no() -> None:
    answers: list[bool] = []
    done = threading.Event()
    app = ChatApp(FakeTail(), _confirming(answers, done), poll_s=0.01)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: any("Should I" in line for line in _shown(app)))

    _run(test, app)
    assert done.wait(5)
    assert answers == [False]


def test_one_request_at_a_time() -> None:
    release = threading.Event()
    asked: list[str] = []

    def ask(text: str, _hooks: Hooks) -> str:
        asked.append(text)
        release.wait(5)
        return ""

    app = ChatApp(FakeTail(), ask, poll_s=0.01)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: asked == ["pause"])
        await _type(pilot, "skip")
        await _until(pilot, lambda: any("still working" in line for line in _shown(app)))
        release.set()

    _run(test, app)
    assert asked == ["pause"]


def test_a_failed_request_is_shown_not_raised() -> None:
    def ask(_text: str, _hooks: Hooks) -> str:
        raise RuntimeError("boom")

    app = ChatApp(FakeTail(), ask, poll_s=0.01)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: any("went wrong" in line for line in _shown(app)))

    _run(test, app)


def test_without_an_agent_typing_says_why() -> None:
    app = ChatApp(FakeTail(), None, unavailable="no Spotify client ID", poll_s=0.01)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: any("no Spotify client ID" in s for s in _shown(app)))

    _run(test, app)


def test_outside_text_is_shown_as_it_is() -> None:
    tail = FakeTail([{"event": "reply", "text": "Night [b]Circuit[/b]", "seconds": 1}])
    app = ChatApp(tail, None, poll_s=0.01)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: any("[b]Circuit[/b]" in line for line in _shown(app)))

    _run(test, app)


def test_commands_clear_help_and_quit() -> None:
    tail = FakeTail([{"event": "request", "text": "skip"}])
    app = ChatApp(tail, None, poll_s=0.01)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: any("skip" in line for line in _shown(app)))
        await _type(pilot, "/clear")
        await _until(pilot, lambda: not any("skip" in line for line in _shown(app)))
        await _type(pilot, "/help")
        await _until(pilot, lambda: any("/quit" in line for line in _shown(app)))
        await _type(pilot, "/quit")
        await _until(pilot, lambda: not app.is_running)

    _run(test, app)
    assert tail.closed


def test_the_status_line_shows_what_is_running() -> None:
    app = ChatApp(FakeTail(), None, status=lambda: "voice: on · model: loaded", poll_s=0.01)

    async def test(pilot: Pilot[None]) -> None:
        status = app.query_one("#status", Static)
        await _until(pilot, lambda: "model: loaded" in str(status.content))

    _run(test, app)
