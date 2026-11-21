"""The chat window: typed requests, the owner's yes or no, and what Jarvis does, shown live."""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from textual.widgets import OptionList, RichLog, Static

from jarvis.chat import ChatApp, PromptInput

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


@dataclass
class Clock:
    """A clock the test moves on; starts well past any arming delay."""

    now: float = 1000.0

    def __call__(self) -> float:
        return self.now


def _shown(app: ChatApp) -> list[str]:
    return [line.text for line in app.query_one("#activity", RichLog).lines]


def _has(app: ChatApp, words: str) -> bool:
    return any(words in line for line in _shown(app))


async def _until(pilot: Pilot[None], ready: Callable[[], bool]) -> None:
    for _ in range(500):
        if ready():
            return
        await pilot.pause(0.01)
    raise AssertionError("the window never got there")


async def _type(pilot: Pilot[None], text: str) -> None:
    pilot.app.query_one(PromptInput).value = text
    await pilot.press("enter")


def _run(test: Callable[[Pilot[None]], object], app: ChatApp) -> None:
    async def body() -> None:
        async with app.run_test() as pilot:
            await test(pilot)  # pyright: ignore[reportGeneralTypeIssues]

    asyncio.run(body())


def _app(
    tail: FakeTail | None = None, ask: Callable[[str, Hooks], str] | None = None, **kw: object
) -> ChatApp:
    return ChatApp(tail or FakeTail(), ask, poll_s=0.01, **kw)  # pyright: ignore[reportArgumentType]


def test_a_typed_request_runs_and_what_happens_is_shown() -> None:
    tail = FakeTail()
    asked: list[str] = []

    def ask(text: str, _hooks: Hooks) -> str:
        asked.append(text)
        heard: dict[str, object] = {"event": "request", "text": text, "via": "chat"}
        answered: dict[str, object] = {"event": "reply", "text": "Paused it.", "seconds": 0.2}
        tail.entries += [heard, answered]
        return "Paused it."

    app = _app(tail, ask)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: _has(app, "Paused it."))
        assert asked == ["pause"]
        assert sum("pause" in line and "typed" in line for line in _shown(app)) == 1

    _run(test, app)


def test_a_typed_request_shows_at_once() -> None:
    release = threading.Event()

    def ask(_text: str, _hooks: Hooks) -> str:
        release.wait(5)
        return ""

    app = _app(ask=ask)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "skip")
        await _until(pilot, lambda: _has(app, "skip"))
        release.set()

    _run(test, app)


def test_voice_activity_shows_without_typing() -> None:
    tail = FakeTail([{"event": "request", "text": "skip", "origin": "owner_speech"}])
    app = _app(tail)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "skip") and _has(app, "by voice"))

    _run(test, app)


def test_pc_audio_says_who_spoke() -> None:
    tail = FakeTail([{"event": "request", "text": "skip", "origin": "pc_audio"}])
    app = _app(tail)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "PC audio: skip"))

    _run(test, app)


def test_a_result_sits_under_its_step_and_requests_are_kept_apart() -> None:
    tail = FakeTail(
        [
            {"event": "request", "text": "pause", "origin": "owner_speech"},
            {"event": "tool_call", "op": "spotify_pause", "verdict": "allow"},
            {"event": "tool_result", "op": "spotify_pause", "ok": True, "text": "Paused."},
            {"event": "request", "text": "skip", "origin": "owner_speech"},
        ]
    )
    app = _app(tail)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "skip"))
        lines = [line.rstrip() for line in _shown(app)]
        step = next(i for i, line in enumerate(lines) if line.endswith("● pause"))
        assert lines[step + 1].lstrip().startswith("⎿")
        assert "Paused." in lines[step + 1]
        assert not lines[step + 2].strip()  # a blank line before the next request

    _run(test, app)


def _confirming(answers: list[bool], done: threading.Event) -> Callable[[str, Hooks], str]:
    def ask(_text: str, hooks: Hooks) -> str:
        answers.append(hooks.confirm("Should I queue Low Tide?"))
        done.set()
        return ""

    return ask


def _question_open(app: ChatApp) -> bool:
    return app.query_one("#question").display and "Low Tide" in str(
        app.query_one("#question-text", Static).content
    )


def _asking(key: str, clock: Clock | None = None) -> list[bool]:
    answers: list[bool] = []
    done = threading.Event()
    clock = clock or Clock()
    app = _app(ask=_confirming(answers, done), clock=clock)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: _question_open(app))
        clock.now += 1
        await pilot.press(key)
        await _until(pilot, done.is_set)
        assert not app.query_one("#question").display
        assert app.query_one("#prompt").display

    _run(test, app)
    return answers


def test_enter_on_yes_answers_yes() -> None:
    assert _asking("enter") == [True]


def test_1_or_y_answers_yes() -> None:
    assert _asking("1") == [True]
    assert _asking("y") == [True]


def test_2_n_and_esc_answer_no() -> None:
    assert _asking("2") == [False]
    assert _asking("n") == [False]
    assert _asking("escape") == [False]


def test_down_then_enter_answers_no() -> None:
    answers: list[bool] = []
    done = threading.Event()
    clock = Clock()
    app = _app(ask=_confirming(answers, done), clock=clock)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: _question_open(app))
        clock.now += 1
        await pilot.press("down", "enter")
        await _until(pilot, done.is_set)

    _run(test, app)
    assert answers == [False]


def test_a_yes_pressed_as_the_question_opens_does_not_count() -> None:
    answers: list[bool] = []
    done = threading.Event()
    clock = Clock()
    app = _app(ask=_confirming(answers, done), clock=clock)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: _question_open(app))
        await pilot.press("y")  # no time has passed on the clock
        await pilot.pause(0.05)
        assert not done.is_set()
        assert app.query_one("#question").display
        clock.now += 1
        await pilot.press("n")
        await _until(pilot, done.is_set)

    _run(test, app)
    assert answers == [False]


def test_leaving_while_a_question_waits_answers_no() -> None:
    answers: list[bool] = []
    done = threading.Event()
    app = _app(ask=_confirming(answers, done))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: _question_open(app))

    _run(test, app)
    assert done.wait(5)
    assert answers == [False]


def test_esc_stops_the_request() -> None:
    seen: list[bool] = []
    started = threading.Event()

    def ask(_text: str, hooks: Hooks) -> str:
        started.set()
        for _ in range(500):
            if hooks.cancelled():
                seen.append(True)
                return ""
            threading.Event().wait(0.01)
        return ""

    app = _app(ask=ask)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "play everything")
        await _until(pilot, started.is_set)
        await pilot.press("escape")
        await _until(pilot, lambda: _has(app, "stopped"))

    _run(test, app)
    assert seen == [True]


def test_the_working_line_shows_while_a_request_runs() -> None:
    release = threading.Event()
    waking = threading.Event()

    def ask(_text: str, hooks: Hooks) -> str:
        hooks.waking()
        waking.set()
        release.wait(5)
        return ""

    app = _app(ask=ask)

    async def test(pilot: Pilot[None]) -> None:
        working = app.query_one("#working", Static)
        assert not working.display
        await _type(pilot, "pause")
        await _until(pilot, waking.is_set)
        await _until(pilot, lambda: "Waking the model" in str(working.content))
        assert "esc to stop" in str(working.content)
        release.set()
        await _until(pilot, lambda: not working.display)

    _run(test, app)


def test_one_request_at_a_time() -> None:
    release = threading.Event()
    asked: list[str] = []

    def ask(text: str, _hooks: Hooks) -> str:
        asked.append(text)
        release.wait(5)
        return ""

    app = _app(ask=ask)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: asked == ["pause"])
        await _type(pilot, "skip")
        await _until(pilot, lambda: _has(app, "still working"))
        release.set()

    _run(test, app)
    assert asked == ["pause"]


def test_a_failed_request_is_shown_not_raised() -> None:
    def ask(_text: str, _hooks: Hooks) -> str:
        raise RuntimeError("boom")

    app = _app(ask=ask)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: _has(app, "went wrong"))

    _run(test, app)


def test_without_an_agent_the_window_and_typing_say_why() -> None:
    app = _app(unavailable="no Spotify client ID")

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "no Spotify client ID"))
        before = sum("no Spotify client ID" in line for line in _shown(app))
        await _type(pilot, "pause")
        await _until(pilot, lambda: sum("no Spotify" in s for s in _shown(app)) > before)

    _run(test, app)


def test_outside_text_is_shown_as_it_is() -> None:
    tail = FakeTail([{"event": "reply", "text": "Night [b]Circuit[/b]", "seconds": 1}])
    app = _app(tail)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "[b]Circuit[/b]"))

    _run(test, app)


def test_details_are_hidden_until_asked_for() -> None:
    tail = FakeTail(
        [
            {"event": "request", "text": "pause", "origin": "owner_speech"},
            {"event": "model_call", "seconds": 4.6},
        ]
    )
    app = _app(tail)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "pause"))
        await pilot.pause(0.05)
        assert not _has(app, "4.6 s")
        await pilot.press("ctrl+o")
        await _until(pilot, lambda: _has(app, "model answered in 4.6 s"))
        await _type(pilot, "/details")
        await _until(pilot, lambda: not _has(app, "4.6 s"))
        assert _has(app, "pause")

    _run(test, app)


def test_up_and_down_bring_back_earlier_requests() -> None:
    app = _app(ask=lambda _text, _hooks: "")

    async def test(pilot: Pilot[None]) -> None:
        prompt = app.query_one(PromptInput)
        for text in ("pause", "skip"):
            await _until(pilot, lambda: app._work is None)  # pyright: ignore[reportPrivateUsage]
            await _type(pilot, text)
        await _until(pilot, lambda: app._work is None)  # pyright: ignore[reportPrivateUsage]
        prompt.value = "draft"
        await pilot.press("up")
        assert prompt.value == "skip"
        await pilot.press("up")
        assert prompt.value == "pause"
        await pilot.press("up")
        assert prompt.value == "pause"
        await pilot.press("down", "down")
        assert prompt.value == "draft"

    _run(test, app)


def test_a_slash_lists_the_commands_and_tab_completes() -> None:
    app = _app()

    async def test(pilot: Pilot[None]) -> None:
        prompt = app.query_one(PromptInput)
        menu = app.query_one("#menu", OptionList)
        prompt.value = "/"
        await _until(pilot, lambda: menu.display and menu.option_count == 5)
        prompt.value = "/de"
        await _until(pilot, lambda: menu.option_count == 1)
        await pilot.press("tab")
        assert prompt.value == "/details"
        assert not menu.display
        await pilot.press("escape")  # nothing open: nothing to close, nothing running
        prompt.value = "play /de"
        await _until(pilot, lambda: not menu.display)

    _run(test, app)


def test_enter_runs_the_highlighted_command() -> None:
    tail = FakeTail([{"event": "request", "text": "skip"}])
    app = _app(tail)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "skip"))
        prompt = app.query_one(PromptInput)
        prompt.value = "/cl"
        await _until(pilot, lambda: app.query_one("#menu").display)
        await pilot.press("enter")
        await _until(pilot, lambda: not _has(app, "skip"))

    _run(test, app)


def test_commands_clear_help_and_quit() -> None:
    tail = FakeTail([{"event": "request", "text": "skip"}])
    app = _app(tail)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "skip"))
        await _type(pilot, "/clear")
        await _until(pilot, lambda: not _has(app, "skip"))
        await _type(pilot, "/help")
        await _until(pilot, lambda: _has(app, "/quit"))
        await _type(pilot, "/nonsense")
        await _until(pilot, lambda: _has(app, "No command /nonsense"))
        await _type(pilot, "/quit")
        await _until(pilot, lambda: not app.is_running)

    _run(test, app)
    assert tail.closed


def test_a_question_mark_shows_the_shortcuts() -> None:
    app = _app()

    async def test(pilot: Pilot[None]) -> None:
        shortcuts = app.query_one("#shortcuts", Static)
        await pilot.press("?")
        await _until(pilot, lambda: shortcuts.display and "ctrl+o" in str(shortcuts.content))
        assert not app.query_one(PromptInput).value
        await pilot.press("escape")
        await _until(pilot, lambda: not shortcuts.display)

    _run(test, app)


def test_the_status_line_shows_what_is_running() -> None:
    app = _app(status=lambda: "voice active · model loaded")

    async def test(pilot: Pilot[None]) -> None:
        status = app.query_one("#status", Static)
        await _until(pilot, lambda: "model loaded" in str(status.content))

    _run(test, app)


def test_a_failing_status_says_so() -> None:
    def status() -> str:
        raise OSError

    app = _app(status=status)

    async def test(pilot: Pilot[None]) -> None:
        line = app.query_one("#status", Static)
        await _until(pilot, lambda: "status unknown: OSError" in str(line.content))

    _run(test, app)


def test_the_welcome_card_stays_through_a_redraw_and_goes_with_clear() -> None:
    app = _app()

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "everything Jarvis does"))
        await pilot.press("ctrl+o")
        await _until(pilot, lambda: _has(app, "everything Jarvis does"))
        await pilot.press("ctrl+l")
        await _until(pilot, lambda: not _has(app, "everything Jarvis does"))
        await pilot.press("ctrl+o")
        await pilot.pause(0.05)
        assert not _has(app, "everything Jarvis does")

    _run(test, app)


def test_the_window_uses_the_assistants_name() -> None:
    answers: list[bool] = []
    done = threading.Event()
    app = _app(ask=_confirming(answers, done), name="Friday")

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "everything Friday does"))
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: _question_open(app))
        assert app.query_one("#question").border_title == "Friday asks"
        await pilot.press("escape")
        await _until(pilot, done.is_set)

    _run(test, app)
