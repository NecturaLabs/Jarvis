"""The chat window: typed requests, replies in Markdown, the owner's yes or no, commands and the
toolbar, and what Jarvis does, shown live at the chosen verbosity."""

import asyncio
import io
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

from PIL import Image
from rich.console import Console
from rich.text import Text
from test_pictures import claimed_png
from textual import events
from textual.containers import VerticalScroll
from textual.widgets import Markdown, OptionList, Static

from jarvis.agent import Agent, ContextUse, Declined, Hooks, NoReason
from jarvis.announce import Announcement
from jarvis.app import no_action_operation
from jarvis.audit import AuditLog
from jarvis.chat import app as chat_app
from jarvis.chat import commands
from jarvis.chat.app import NO_REPLY, SETTINGS, ChatApp, Services
from jarvis.chat.archive import ChatArchive
from jarvis.chat.clipboard import Clip, ClipboardError
from jarvis.chat.history import InputHistory
from jarvis.chat.settings import SettingsPanel
from jarvis.chat.widgets import (
    ACCENT,
    CHEVRON,
    Chain,
    Entry,
    PictureView,
    PointerList,
    PromptArea,
    Reply,
    Request,
)
from jarvis.config import ChatConfig, FilesConfig
from jarvis.memory import Conversation, Turn
from jarvis.operations import Registry
from jarvis.pc.files import Boundary, FileOps
from jarvis.pictures import Picture, PictureCache
from jarvis.sysinfo import Gpu, Stats

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator, Sequence
    from pathlib import Path
    from typing import Any

    import pytest
    from textual.app import App
    from textual.pilot import Pilot
    from textual.widget import Widget

SESSION = "s1"


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


@dataclass
class FakeSession:
    """An agent stand-in: `answer` handles each request."""

    answer: Callable[[str, Hooks], str] = lambda _text, _hooks: "Done."
    memory: Conversation = field(default_factory=lambda: Conversation(budget=1000))
    asked: list[str] = field(default_factory=list[str])
    spoken: list[bool] = field(default_factory=list[bool])
    pictures: list[list[str]] = field(default_factory=list[list[str]])
    pasted: list[list[str]] = field(default_factory=list[list[str]])
    hooks: list[Hooks] = field(default_factory=list[Hooks])
    forgotten: int = 0
    compacted: list[bool] = field(default_factory=list[bool])
    use: ContextUse = field(
        default_factory=lambda: ContextUse(9100, 24576, 5100, 2400, 12000, 1, summarized=False)
    )

    def handle(
        self,
        text: str,
        hooks: Hooks,
        *,
        pictures: Sequence[Picture] = (),
        pasted: Sequence[str] = (),
        spoken: bool = False,
    ) -> str:
        self.asked.append(text)
        self.spoken.append(spoken)
        self.pictures.append([p.name for p in pictures])
        self.pasted.append(list(pasted))
        self.hooks.append(hooks)
        return self.answer(text, hooks)

    rate: float | None = 50.4

    def last_rate(self) -> float | None:
        return self.rate

    def forget(self) -> None:
        self.forgotten += 1
        self.memory.clear()

    def compact(self, *, everything: bool = False) -> bool:
        self.compacted.append(everything)
        return True

    def context_use(self) -> ContextUse:
        return self.use

    thought: list[str] = field(default_factory=list[str])

    def set_thinking(self, section: ChatConfig) -> None:
        self.thought.append(section.thinking)

    systems: list[str] = field(default_factory=list[str])

    def set_system(self, prompt: str) -> None:
        self.systems.append(prompt)

    def resume(self, path: Path) -> bool:
        return self.memory.load(path)

    bypass: list[bool] = field(default_factory=list[bool])

    def set_bypass_consent(self, *, on: bool) -> None:
        self.bypass.append(on)


def _stats() -> Stats:
    gib = 1024**3
    return Stats(12.0, 53.0, 33 * gib, 60 * gib, Gpu(41, 10 * gib, 20 * gib, 67.0))


def _plain(renderable: object) -> str:
    console = Console(width=200, record=True, color_system=None, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


def _words(widget: Widget) -> str:
    if isinstance(widget, Reply):
        return widget.text
    if isinstance(widget, Chain):  # its lines, as far as the verbosity shows them
        return "".join(_words(line) for line in widget.children if line.display)
    if isinstance(widget, Static):
        return _plain(widget.content)
    return ""


def _transcript(app: ChatApp) -> list[Widget]:
    return list(app.query_one("#transcript").children)


def _shown(app: ChatApp) -> list[str]:
    return [_words(widget) for widget in _transcript(app) if widget.display]


def _has(app: ChatApp, words: str) -> bool:
    return any(words in line for line in _shown(app))


async def _until(pilot: Pilot[None], ready: Callable[[], bool]) -> None:
    for _ in range(500):
        if ready():
            return
        await pilot.pause(0.01)
    raise AssertionError("the window never got there")


def _input(app: App[None]) -> PromptArea:
    return app.query_one("#input", PromptArea)


async def _type(pilot: Pilot[None], text: str) -> None:
    _input(pilot.app).set_text(text)
    await pilot.pause(0.02)
    await pilot.press("enter")


def _run(test: Callable[[Pilot[None]], Awaitable[object]], app: ChatApp) -> None:
    async def body() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await test(pilot)

    asyncio.run(body())


def _app(
    tail: FakeTail | None = None,
    session: FakeSession | None = None,
    *,
    unavailable: str = "",
    name: str = "Jarvis",
    settings: ChatConfig | None = None,
    services: Services | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ChatApp:
    return ChatApp(
        tail or FakeTail(),
        session,
        unavailable=unavailable,
        name=name,
        settings=settings,
        services=services,
        session_id=SESSION,
        poll_s=0.01,
        clock=clock,
    )


# A request and its reply


def test_a_typed_request_runs_and_its_reply_is_shown_once() -> None:
    tail = FakeTail()

    def answer(text: str, _hooks: Hooks) -> str:
        own: dict[str, object] = {"session": SESSION, "via": "chat"}
        tail.entries.append({"event": "request", "text": text, **own})
        tail.entries.append({"event": "reply", "text": "Paused it.", "seconds": 0.2, **own})
        return "Paused it."

    session = FakeSession(answer)
    app = _app(tail, session)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: _has(app, "Paused it."))
        await pilot.pause(0.05)
        assert session.asked == ["pause"]
        assert sum(isinstance(w, Request) and w.text == "pause" for w in _transcript(app)) == 1
        assert sum(isinstance(w, Reply) for w in _transcript(app)) == 1

    _run(test, app)


def test_a_typed_request_shows_at_once() -> None:
    release = threading.Event()

    def answer(_text: str, _hooks: Hooks) -> str:
        release.wait(5)
        return ""

    app = _app(session=FakeSession(answer))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "skip")
        await _until(pilot, lambda: _has(app, "skip"))
        release.set()

    _run(test, app)


def test_a_long_reply_is_shown_whole_as_markdown() -> None:
    reply = "# Moonveil\n\n" + " ".join(["A katana with a magic skill."] * 40) + "\n\n- one\n- two"
    app = _app(session=FakeSession(lambda _t, _h: reply))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "tell me about the moonveil")
        await _until(pilot, lambda: any(isinstance(w, Reply) for w in _transcript(app)))
        shown = next(w for w in _transcript(app) if isinstance(w, Reply))
        assert shown.text == reply
        assert shown.query_one(Markdown).source == reply

    _run(test, app)


CLAUDE_CODE_SAMPLE = """# Katanas

Some text.

## Where

- one
- two

### Steps

1. Go.
2. Look.

| Name | Kind |
|---|---|
| Moonveil | katana |

```python
print("hi")
```

> A quote.

Done."""


def test_a_reply_is_set_as_claude_code_sets_it() -> None:
    app = _app(session=FakeSession(lambda _t, _h: CLAUDE_CODE_SAMPLE))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "compare them")
        await _until(pilot, lambda: bool(app.query(".reply-note")))
        await pilot.pause(0.1)
        title, part, small = (
            app.query_one(kind) for kind in ("MarkdownH1", "MarkdownH2", "MarkdownH3")
        )
        assert {"bold", "italic", "underline"} <= set(str(title.styles.text_style).split())
        for heading in (part, small):
            assert set(str(heading.styles.text_style).split()) == {"bold"}
        paragraph = app.query("MarkdownParagraph").first()
        bullets = app.query_one("MarkdownBulletList")
        # One blank line under each heading and one above, never two.
        assert paragraph.region.y == title.region.bottom + 1
        assert part.region.y == paragraph.region.bottom + 1
        assert bullets.region.y == part.region.bottom + 1
        marks = [_plain(bullet.render()).rstrip() for bullet in app.query("MarkdownBullet")]
        assert marks == ["-", "-", "1.", "2."]  # dashes, and numbers with no space before them
        assert app.query("MarkdownBullet").first().styles.color.ansi == -1  # not dimmed
        fence = app.query_one("MarkdownFence")
        assert fence.styles.border_left[0] in {"", "none"}  # code sits in the text's column
        assert fence.region.x == paragraph.region.x
        quote = app.query_one("MarkdownBlockQuote")
        assert quote.styles.color.ansi == -1  # italic words in the text's colour, a dim bar
        assert "italic" in str(quote.styles.text_style)
        header = app.query_one("MarkdownTableContent > .header")
        assert "bold" not in str(header.styles.text_style)  # a plain header, as Claude Code's
        assert header.styles.color.ansi == -1
        reply = next(w for w in _transcript(app) if isinstance(w, Reply))
        note, mark = reply.query_one(".reply-note"), reply.query_one(".reply-mark")
        assert note.region.x == mark.region.x  # "✻ Worked for …" under the reply's dot

    _run(test, app)


def test_the_reply_grows_while_it_is_written_and_the_whole_one_replaces_it() -> None:
    release = threading.Event()

    def answer(_text: str, hooks: Hooks) -> str:
        assert hooks.draft is not None
        hooks.draft("The first sentence.")
        release.wait(5)
        return "The first sentence. And the rest."

    app = _app(session=FakeSession(answer))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "talk")
        await _until(pilot, lambda: bool(app.query(".reply.draft")))
        assert app.query_one(".reply.draft", Reply).text == "The first sentence."
        release.set()
        await _until(pilot, lambda: _has(app, "And the rest."))
        replies = [w for w in _transcript(app) if isinstance(w, Reply)]
        assert len(replies) == 1
        assert not replies[0].has_class("draft")

    _run(test, app)


def test_hush_drops_a_draft_that_was_not_the_reply() -> None:
    release = threading.Event()

    def answer(_text: str, hooks: Hooks) -> str:
        assert hooks.draft is not None
        hooks.draft("Playing it now.")
        hooks.hush()
        release.wait(5)
        return ""

    app = _app(session=FakeSession(answer))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "play")
        await pilot.pause(0.2)
        assert not app.query(".reply.draft")
        release.set()

    _run(test, app)


def test_voice_activity_shows_without_typing() -> None:
    tail = FakeTail(
        [
            {"event": "request", "text": "skip", "origin": "owner_speech"},
            {"event": "reply", "text": "Skipped.", "seconds": 0.4},
        ]
    )
    app = _app(tail)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "by voice") and _has(app, "Skipped."))

    _run(test, app)


def test_pc_audio_says_who_spoke() -> None:
    tail = FakeTail([{"event": "request", "text": "skip", "origin": "pc_audio"}])
    app = _app(tail)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "PC audio: skip"))

    _run(test, app)


def test_a_request_without_a_time_says_only_how_it_came() -> None:
    tail = FakeTail([{"event": "request", "text": "pause", "via": "chat", "session": "other"}])
    app = _app(tail)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "typed"))
        line = next(_words(w) for w in _transcript(app) if isinstance(w, Request)).strip()
        assert line.endswith("typed")

    _run(test, app)


def test_a_request_from_pc_audio_names_it_once() -> None:
    tail = FakeTail([{"event": "request", "text": "skip", "origin": "pc_audio"}])
    app = _app(tail)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "PC audio: skip"))
        line = next(_words(w) for w in _transcript(app) if isinstance(w, Request))
        assert line.count("PC audio") == 1

    _run(test, app)


def test_outside_text_is_shown_as_it_is() -> None:
    tail = FakeTail([{"event": "request", "text": "Night [b]Circuit[/b]", "origin": "x"}])
    app = _app(tail)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "[b]Circuit[/b]"))

    _run(test, app)


def test_a_failed_request_is_shown_not_raised() -> None:
    def answer(_text: str, _hooks: Hooks) -> str:
        raise RuntimeError("boom")

    app = _app(session=FakeSession(answer))

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


def test_without_an_agent_the_input_says_typing_is_off() -> None:
    app = _app(unavailable="no model")

    async def test(pilot: Pilot[None]) -> None:
        await pilot.pause(0.05)
        assert "off" in str(_input(app).placeholder)

    _run(test, app)


# Verbosity and the working line

STEPS: list[dict[str, object]] = [
    {"event": "request", "text": "pause", "origin": "owner_speech"},
    {"event": "tool_call", "op": "spotify_pause", "verdict": "run"},
    {"event": "tool_result", "op": "spotify_pause", "ok": True, "text": "Paused."},
    {"event": "model_call", "seconds": 4.6},
    {"event": "reply", "text": "Paused it.", "seconds": 5.0},
]


def test_by_default_only_requests_and_replies_show() -> None:
    app = _app(FakeTail(list(STEPS)))

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "Paused it."))
        await pilot.pause(0.05)
        assert not _has(app, "Pause\n")
        assert not _has(app, "4.6 s")

    _run(test, app)


class FakeAnnouncements:
    """The voice service's announcements as a test hands them out."""

    def __init__(self) -> None:
        super().__init__()
        self.items: queue.Queue[Announcement | None] = queue.Queue()
        self.opened = 0
        self.closed = threading.Event()

    def open(self) -> FakeAnnouncements:
        self.opened += 1
        return self

    def __iter__(self) -> Iterator[Announcement]:
        while (item := self.items.get()) is not None:
            yield item

    def close(self) -> None:
        self.closed.set()
        self.items.put(None)


def test_an_announcement_of_the_voice_service_is_shown_as_a_line_while_the_window_is_open() -> None:
    feed = FakeAnnouncements()
    app = _app(services=Services(announcements=feed.open))

    async def test(pilot: Pilot[None]) -> None:
        feed.items.put(Announcement(1, 1.0, "reminder", "Reminder: take the bread out"))
        await _until(pilot, lambda: _has(app, "Jarvis: Reminder: take the bread out"))

    _run(test, app)
    assert feed.closed.is_set()  # the window let go of the feed as it closed


def test_the_window_looks_again_for_the_voice_service_when_it_was_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chat_app, "ANNOUNCEMENTS_RETRY_S", 0.01)
    first = FakeAnnouncements()
    feeds: list[FakeAnnouncements | None] = [None, first]
    app = _app(services=Services(announcements=lambda: feeds.pop(0) if feeds else None))

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: not feeds)
        first.items.put(Announcement(1, 1.0, "watch", "A door opened"))
        await _until(pilot, lambda: _has(app, "Jarvis: A door opened"))

    _run(test, app)


def test_ctrl_o_shows_the_steps_then_everything_then_replies_again() -> None:
    saved: list[str] = []
    services = Services(save_settings=lambda chat: saved.append(chat.verbosity))
    app = _app(FakeTail(list(STEPS)), services=services)

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "Paused it."))
        await pilot.press("ctrl+o")
        await _until(pilot, lambda: _has(app, "Pause\n"))
        assert not _has(app, "4.6 s")
        await pilot.press("ctrl+o")
        await _until(pilot, lambda: _has(app, "4.6 s"))
        await pilot.press("ctrl+o")
        await _until(pilot, lambda: not _has(app, "4.6 s"))

    _run(test, app)
    assert saved == ["steps", "everything", "replies"]


def test_the_verbosity_command_sets_a_level() -> None:
    app = _app(FakeTail(list(STEPS)))

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "Paused it."))
        await _type(pilot, "/verbosity steps")
        await _until(pilot, lambda: _has(app, "Pause\n"))

    _run(test, app)


def test_the_working_line_says_what_the_request_is_doing() -> None:
    tail = FakeTail()
    release = threading.Event()

    def answer(_text: str, hooks: Hooks) -> str:
        hooks.waking()
        tail.entries.append({"event": "tool_call", "op": "web_search", "session": SESSION})
        release.wait(5)
        return ""

    app = _app(tail, FakeSession(answer))

    async def test(pilot: Pilot[None]) -> None:
        working = app.query_one("#working", Static)
        assert not working.display
        await _type(pilot, "search")
        await _until(pilot, lambda: "Searching the web" in _plain(working.content))
        assert "esc to interrupt" in _plain(working.content)
        release.set()
        await _until(pilot, lambda: not working.display)

    _run(test, app)


def test_esc_stops_the_request() -> None:
    seen: list[bool] = []
    started = threading.Event()

    def answer(_text: str, hooks: Hooks) -> str:
        started.set()
        for _ in range(500):
            if hooks.cancelled():
                seen.append(True)
                return ""
            threading.Event().wait(0.01)
        return ""

    app = _app(session=FakeSession(answer))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "play everything")
        await _until(pilot, started.is_set)
        await pilot.press("escape")
        await _until(pilot, lambda: _has(app, "Stopped."))

    _run(test, app)
    assert seen == [True]


def test_a_request_that_ends_without_words_says_so() -> None:
    app = _app(session=FakeSession(lambda _text, _hooks: ""))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "hush")
        await _until(pilot, lambda: _has(app, NO_REPLY))
        assert not _has(app, "Stopped.")

    _run(test, app)


def test_words_typed_while_a_request_works_steer_it() -> None:
    steered: list[str] = []

    def answer(text: str, hooks: Hooks) -> str:
        if text == "pause":
            for _ in range(500):  # as the agent does before each model call
                if taken := hooks.steer():
                    steered.extend(taken)
                    break
                threading.Event().wait(0.01)
        return "Done."

    session = FakeSession(answer)
    app = _app(session=session)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: session.asked == ["pause"])
        await _type(pilot, "use high effort")
        await _until(pilot, lambda: bool(steered))
        await _until(pilot, lambda: _has(app, "steering"))

    _run(test, app)
    assert steered == ["use high effort"]
    assert session.asked == ["pause"]  # one request, steered


def test_words_the_request_ended_before_taking_run_next() -> None:
    release = threading.Event()

    def answer(text: str, _hooks: Hooks) -> str:
        if text == "pause":
            release.wait(5)
        return "Done."

    session = FakeSession(answer)
    app = _app(session=session)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: session.asked == ["pause"])
        await _type(pilot, "skip")
        await _until(pilot, lambda: _has(app, "steering"))
        release.set()
        await _until(pilot, lambda: len(session.asked) == 2)

    _run(test, app)
    assert session.asked == ["pause", "skip"]


def test_with_queue_set_words_typed_while_working_run_as_the_next_request() -> None:
    release = threading.Event()
    steered: list[list[str]] = []

    def answer(text: str, hooks: Hooks) -> str:
        if text == "pause":
            release.wait(5)
            steered.append(hooks.steer())
        return "Done."

    session = FakeSession(answer)
    app = _app(session=session, settings=ChatConfig(while_working="queue"))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: session.asked == ["pause"])
        await _type(pilot, "skip")
        await _type(pilot, "louder")
        await _until(pilot, lambda: _has(app, "queued"))
        release.set()
        await _until(pilot, lambda: len(session.asked) == 3)

    _run(test, app)
    assert steered == [[]]
    assert session.asked == ["pause", "skip", "louder"]


# The owner's yes or no


def _confirming(answers: list[bool | Declined], done: threading.Event) -> FakeSession:
    def answer(_text: str, hooks: Hooks) -> str:
        answers.append(hooks.confirm("Should I queue Low Tide?"))
        done.set()
        return ""

    return FakeSession(answer)


def _question_open(app: ChatApp) -> bool:
    return app.query_one("#question").display and "Low Tide" in _plain(
        app.query_one("#question-text", Static).content
    )


def _asking(keys: tuple[str, ...], *, wait: bool = True) -> list[bool | Declined]:
    answers: list[bool | Declined] = []
    done = threading.Event()
    clock = Clock()
    app = _app(session=_confirming(answers, done), clock=clock)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: _question_open(app))
        if wait:
            clock.now += 1
        await pilot.press(*keys)
        if not wait:
            await pilot.pause(0.05)
            assert not done.is_set()
            clock.now += 1
            await pilot.press("n")
        await _until(pilot, done.is_set)
        assert not app.query_one("#question").display
        assert app.query_one("#prompt").display

    _run(test, app)
    return answers


def test_the_working_line_waits_for_the_answer_while_a_question_is_open() -> None:
    answers: list[bool | Declined] = []
    done = threading.Event()
    app = _app(session=_confirming(answers, done))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: _question_open(app))
        await pilot.pause(0.2)
        line = _plain(app.query_one("#working", Static).content)
        assert "Waiting for your answer" in line
        assert "esc to interrupt" not in line

    _run(test, app)


def test_the_highlighted_choice_is_pointed_at() -> None:
    answers: list[bool | Declined] = []
    done = threading.Event()
    clock = Clock()
    app = _app(session=_confirming(answers, done), clock=clock)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: _question_open(app))
        assert _pointed(app, "#choices") == "2. No"  # no is the choice Enter takes at first
        await pilot.press("up")
        assert _pointed(app, "#choices") == "1. Yes"

    _run(test, app)


def test_1_or_up_then_enter_answers_yes() -> None:
    # Requirement changed (audit): y and Enter no longer say yes, as letters typed ahead did.
    for keys in (("1",), ("up", "enter")):
        assert _asking(keys) == [True]


def test_2_n_esc_or_enter_answer_no() -> None:
    for keys in (("2",), ("n",), ("escape",), ("enter",)):
        assert _asking(keys) == [False]


def test_a_yes_pressed_as_the_question_opens_does_not_count() -> None:
    assert _asking(("1",), wait=False) == [False]


def test_words_typed_ahead_never_answer_the_question() -> None:
    answers: list[bool | Declined] = []
    done = threading.Event()
    clock = Clock()
    app = _app(session=_confirming(answers, done), clock=clock)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: _question_open(app))
        clock.now += 1
        await pilot.press("w", "h", "y", "space", "1")
        await pilot.pause(0.05)
        assert not done.is_set()  # typing, then a 1 among the words: still open
        clock.now += 1
        await pilot.press("1")
        await _until(pilot, done.is_set)

    _run(test, app)
    assert answers == [True]


def test_leaving_while_a_question_waits_answers_no() -> None:
    answers: list[bool | Declined] = []
    done = threading.Event()
    app = _app(session=_confirming(answers, done))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: _question_open(app))

    _run(test, app)
    assert done.wait(5)
    assert answers == [Declined(NoReason.CLOSED)]  # a no, audited as the window closing


# Typing


def test_up_and_down_bring_back_earlier_requests_across_sessions(tmp_path: Path) -> None:
    kept = tmp_path / "chat-history"
    first = _app(session=FakeSession(), services=Services(history=InputHistory(kept, 100)))

    async def type_two(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _type(pilot, "skip")

    _run(type_two, first)
    app = _app(session=FakeSession(), services=Services(history=InputHistory(kept, 100)))

    async def test(pilot: Pilot[None]) -> None:
        _input(app).set_text("draft")
        await pilot.press("up")
        assert _input(app).text == "skip"
        await pilot.press("up")
        assert _input(app).text == "pause"
        await pilot.press("down", "down")
        assert _input(app).text == "draft"

    _run(test, app)
    assert kept.stat().st_mode & 0o777 == 0o600


def test_shift_enter_and_a_backslash_start_new_lines() -> None:
    session = FakeSession()
    app = _app(session=session)

    async def test(pilot: Pilot[None]) -> None:
        _input(app).set_text("one")
        await pilot.press("shift+enter")
        await pilot.press("t", "w", "o", "backslash", "enter")
        await pilot.press("t", "h", "r", "e", "e")
        assert _input(app).text == "one\ntwo\nthree"
        await pilot.press("enter")
        await _until(pilot, lambda: session.asked == ["one\ntwo\nthree"])

    _run(test, app)


def test_a_slash_lists_matching_commands_and_tab_completes() -> None:
    app = _app(session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        menu = app.query_one("#menu", OptionList)
        _input(app).set_text("/cle")
        await _until(pilot, lambda: menu.display)
        assert menu.option_count == 1
        await pilot.press("tab")
        assert _input(app).text == "/clear"
        _input(app).set_text("/verb")
        await _until(pilot, lambda: menu.display)
        await pilot.press("tab")
        await _until(pilot, lambda: menu.option_count == 3)
        assert _input(app).text == "/verbosity "

    _run(test, app)


def _pointed(app: ChatApp, selector: str) -> str:
    """Return the option the list points at with the chevron, and check it is the only one."""
    listing = app.query_one(selector, OptionList)
    prompts = [_plain(listing.get_option_at_index(i).prompt) for i in range(listing.option_count)]
    pointed = [
        prompt
        for prompt in prompts
        if prompt.lstrip().startswith("\N{HEAVY RIGHT-POINTING ANGLE QUOTATION MARK ORNAMENT}")
    ]
    assert len(pointed) == 1
    return (
        pointed[0].strip().lstrip("\N{HEAVY RIGHT-POINTING ANGLE QUOTATION MARK ORNAMENT}").strip()
    )


def test_the_menu_points_at_the_highlighted_command_and_says_how_to_use_it() -> None:
    app = _app(session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        _input(app).set_text("/c")
        await _until(pilot, lambda: app.query_one("#menu").display)
        assert _pointed(app, "#menu").startswith("/clear")
        await pilot.press("down")
        assert _pointed(app, "#menu").startswith("/compact")
        assert "tab" in _plain(app.query_one("#hints", Static).content)
        await pilot.press("escape")
        assert "? for shortcuts" in _plain(app.query_one("#hints", Static).content)

    _run(test, app)


def test_enter_runs_the_highlighted_command() -> None:
    app = _app(session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        _input(app).set_text("/hel")
        await _until(pilot, lambda: app.query_one("#menu").display)
        await pilot.press("enter")
        await _until(pilot, lambda: _has(app, "/compact"))

    _run(test, app)


def test_ctrl_c_clears_the_input_and_twice_leaves() -> None:
    app = _app(session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        _input(app).set_text("half a thought")
        await pilot.press("ctrl+c")
        assert not _input(app).text
        await pilot.press("ctrl+c")
        assert app.is_running
        await pilot.press("ctrl+c")
        await pilot.pause(0.05)

    _run(test, app)
    assert app.return_code is not None


def test_a_question_mark_shows_the_shortcuts() -> None:
    app = _app(session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        shortcuts = app.query_one("#shortcuts", Static)
        await pilot.press("question_mark")
        await _until(pilot, lambda: shortcuts.display)
        assert "ctrl+o" in _plain(shortcuts.content)
        assert not _input(app).text

    _run(test, app)


# Commands


def test_clear_starts_a_new_conversation_and_keeps_the_welcome() -> None:
    session = FakeSession()
    app = _app(session=session)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: _has(app, "Done."))
        await _type(pilot, "/clear")
        await _until(pilot, lambda: session.forgotten == 1)
        await _until(pilot, lambda: not _has(app, "Done."))
        assert _has(app, "Welcome to")
        assert _has(app, "new conversation")

    _run(test, app)


def test_compact_summarizes_everything_but_the_latest() -> None:
    session = FakeSession()
    app = _app(session=session)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/compact")
        await _until(pilot, lambda: _has(app, "Compacted"))

    _run(test, app)
    assert session.compacted == [True]


def test_context_shows_how_full_the_window_is() -> None:
    app = _app(session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/context")
        await _until(pilot, lambda: _has(app, "9.1k/24.6k"))
        assert _has(app, "conversation")
        assert _has(app, "compacts past 12.0k")

    _run(test, app)


def test_status_shows_the_services_and_details() -> None:
    services = Services(status=lambda: "voice active", details=lambda: [("config", "/c.toml")])
    app = _app(session=FakeSession(), services=services)

    async def test(pilot: Pilot[None]) -> None:
        status = app.query_one("#status", Static)
        await _until(pilot, lambda: "voice active" in _plain(status.content))
        await _type(pilot, "/status")
        await _until(pilot, lambda: _has(app, "/c.toml") and _has(app, "voice active"))

    _run(test, app)


def test_a_failing_status_says_so() -> None:
    def broken() -> str:
        raise OSError

    app = _app(services=Services(status=broken))

    async def test(pilot: Pilot[None]) -> None:
        status = app.query_one("#status", Static)
        await _until(pilot, lambda: "status unknown" in _plain(status.content))

    _run(test, app)


def test_copy_puts_the_last_reply_on_the_clipboard() -> None:
    app = _app(session=FakeSession(lambda _t, _h: "**Glass Harbor**"))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "what's playing")
        await _until(pilot, lambda: _has(app, "Glass Harbor"))
        await _type(pilot, "/copy")
        await _until(pilot, lambda: app.clipboard == "**Glass Harbor**")

    _run(test, app)


def test_export_saves_the_conversation_as_markdown(tmp_path: Path) -> None:
    session = FakeSession(lambda _t, _h: "Paused it.")
    app = _app(session=session, services=Services(export_dir=tmp_path))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: _has(app, "Paused it."))
        await _type(pilot, "/export")
        await _until(pilot, lambda: any(tmp_path.iterdir()))

    _run(test, app)
    (saved,) = tmp_path.iterdir()
    text = saved.read_text(encoding="utf-8")
    assert "**You:** pause" in text
    assert "Paused it." in text


def test_config_changes_a_setting_and_saves_it() -> None:
    saved: list[ChatConfig] = []
    app = _app(session=FakeSession(), services=Services(save_settings=saved.append))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await pilot.press("right")  # on Verbosity: the next value
        await _until(pilot, lambda: bool(saved))
        await pilot.press("down", "enter")  # Toolbar is on or off: Enter turns it over
        await _until(pilot, lambda: len(saved) == 2)
        await pilot.press("escape")
        await _until(pilot, lambda: app.query_one("#prompt").display)

    _run(test, app)
    assert saved[0].verbosity == "steps"
    assert saved[1].toolbar is False


def test_config_lists_the_tools_and_moves_one_on() -> None:
    from jarvis.policy import OpClass  # ruff: ignore[import-outside-top-level] - only this test
    from jarvis.toolswitch import ToolSetting  # ruff: ignore[import-outside-top-level] - only this test

    tools = {
        "reddit_search": ToolSetting(
            "reddit_search", "Find Reddit threads", OpClass.SAFE, OpClass.SAFE, tuple(OpClass)
        ),
        "web_search": ToolSetting(
            "web_search", "Search the web", OpClass.SAFE, OpClass.SAFE, tuple(OpClass)
        ),
    }
    moved: list[str] = []

    def move(name: str, _by: int = 1) -> ToolSetting:
        moved.append(name)
        old = tools[name]
        tools[name] = ToolSetting(name, old.summary, old.following(), old.shipped, old.choices)
        return tools[name]

    services = Services(tools=lambda: list(tools.values()), next_tool_setting=move)
    app = _app(session=FakeSession(), services=services)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await _open_tab(pilot, "Tools")
        await pilot.press("enter")  # reddit_search, first
        await _until(pilot, lambda: bool(moved))
        await _until(pilot, lambda: "asks first" in _settings_text(app))
        await pilot.press("escape")
        await _until(pilot, lambda: app.query_one("#prompt").display)

    _run(test, app)
    assert moved == ["reddit_search"]
    assert tools["reddit_search"].state is OpClass.CONFIRM


def test_config_changes_jarvis_settings_and_restarts_the_voice_service_once_on_close() -> None:
    from jarvis.chat.app import SharedSetting  # ruff: ignore[import-outside-top-level] - only this test

    values: dict[str, object] = {"audio.speech_speed": 1}
    applied: list[bool] = []

    def shared() -> list[SharedSetting]:
        speeds = (1, 2, 3)
        return [
            SharedSetting(
                "audio.speech_speed", "Speaking speed", speeds, values["audio.speech_speed"]
            )
        ]

    def change(key: str, value: object) -> None:
        values[key] = value

    services = Services(shared=shared, set_shared=change, apply=lambda: applied.append(True))
    app = _app(session=FakeSession(), services=services)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await _open_tab(pilot, "Voice")
        assert "restarts when you close settings" not in _settings_text(app)  # nothing changed yet
        await pilot.press("right")  # Speaking speed
        await _until(pilot, lambda: values["audio.speech_speed"] == 2)
        await pilot.press("right")
        await _until(pilot, lambda: values["audio.speech_speed"] == 3)
        assert "restarts when you close settings" in _settings_text(app)
        assert applied == []  # nothing restarts while the panel is open
        await pilot.press("escape")
        await _until(pilot, lambda: applied == [True])

    _run(test, app)


BACK_TEN_MINUTES, BACK_AN_HOUR = 600, 3600


def _sectioned_services(values: dict[str, object]) -> Services:
    from jarvis.chat.app import SharedSetting  # ruff: ignore[import-outside-top-level] - only these tests

    def shared() -> list[SharedSetting]:
        return [
            SharedSetting(
                "audio.speech_speed", "Speaking speed", (1, 2, 3), values["speed"], "Voice"
            ),
            SharedSetting(
                "coding.back_after_s",
                "Leave coding mode after",
                (0, BACK_TEN_MINUTES, BACK_AN_HOUR),
                values["back"],
                "Mode",
                unit="seconds",
                zero="never",
            ),
        ]

    return Services(
        shared=shared,
        set_shared=lambda key, value: values.update({key.split(".")[-1][:4]: value}),
        apply=lambda: None,
        save_settings=lambda _c: None,
    )


def test_config_has_a_tab_for_each_section_and_a_search_across_them() -> None:
    values: dict[str, object] = {"speed": 1, "back": BACK_TEN_MINUTES}
    app = _app(session=FakeSession(), services=_sectioned_services(values))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        panel = app.query_one("#settings", SettingsPanel)
        assert [section.title for section in panel.sections] == ["Chat", "Mode", "Voice"]
        await pilot.press("tab")  # the next tab; Shift+Tab goes back
        assert panel.tab is not None
        assert panel.tab.title == "Mode"
        assert "10 min" in _settings_text(app)  # seconds are said as minutes
        await pilot.press("shift+tab", "shift+tab")  # round to the last
        assert panel.tab is not None
        assert panel.tab.title == "Voice"
        for letter in "speak":  # typing anywhere searches every section
            await pilot.press(letter)
        assert [row.name for row in panel.rows] == ["Speaking speed"]
        assert "1 found" in _settings_text(app)
        await pilot.press("escape")  # the first Esc clears the search, not the screen
        assert not panel.searched
        assert app.query_one("#settings").display
        await pilot.press("escape")
        await _until(pilot, lambda: app.query_one("#prompt").display)

    _run(test, app)


def test_config_keeps_its_section_tabs_on_the_keyboard() -> None:
    values: dict[str, object] = {"speed": 1, "back": BACK_TEN_MINUTES}
    app = _app(session=FakeSession(), services=_sectioned_services(values))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        panel = app.query_one("#settings", SettingsPanel)
        await pilot.press("up")  # from the first setting to the tab row
        await pilot.press("right")  # ← → switch tabs there, and change values in the list
        assert panel.tab is not None
        assert panel.tab.title == "Mode"
        assert values["back"] == BACK_TEN_MINUTES
        await pilot.press("down", "right")
        await _until(pilot, lambda: values["back"] == BACK_AN_HOUR)

    _run(test, app)


def test_enter_on_a_setting_with_many_values_opens_a_list_that_esc_leaves_alone() -> None:
    saved: list[ChatConfig] = []
    app = _app(session=FakeSession(), services=Services(save_settings=saved.append))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await pilot.press("down", "down", "down", "enter")  # Replies: concise, normal, verbose
        listed = app.query_one("#settings-list", OptionList)
        await _until(pilot, lambda: listed.option_count == 3)
        text = _settings_text(app)
        assert "concise" in text
        assert "normal" in text
        assert "(now)" in text  # the value in force is marked
        await pilot.press("down", "escape")
        await _until(pilot, lambda: listed.option_count > 3)
        assert saved == []  # Esc left it as it was, and kept the screen open
        await pilot.press("enter", "down", "down", "enter")
        await _until(pilot, lambda: bool(saved))
        assert listed.option_count > 3  # back in the list

    _run(test, app)
    assert saved[0].response_format == "verbose"


def test_every_setting_says_what_it_does_in_two_lines() -> None:
    from jarvis.cli import SHARED_SETTINGS  # ruff: ignore[import-outside-top-level] - only this test

    about = [s.about for s in SETTINGS] + [s.about for s in SHARED_SETTINGS]
    assert all(about)
    assert max(len(text) for text in about) <= 140  # two lines of the footer at 80 columns


def test_the_lines_under_the_input_line_up_with_its_box() -> None:
    services = Services(status=lambda: "voice inactive · model server not running", stats=_stats)

    async def check(size: tuple[int, int]) -> None:
        app = _app(session=FakeSession(), services=services)
        async with app.run_test(size=size) as pilot:
            status = app.query_one("#status", Static)
            toolbar = app.query_one("#toolbar", Static)
            await _until(pilot, lambda: "voice" in _plain(status.content))
            await _until(pilot, lambda: "CPU" in _plain(toolbar.content))
            prompt = app.query_one("#prompt")
            column = prompt.region.x + 2  # the box's border and its padding
            for name in ("#hints", "#toolbar"):
                assert app.query_one(name).region.x == column, (size, name)
            assert status.region.right == prompt.region.right  # the box's right edge
            assert toolbar.region.right == prompt.region.right
            line = _plain(toolbar.content).rstrip("\n")
            assert len(line) <= toolbar.region.width  # never wraps or clips
            assert line.endswith("k")  # the context ends the line, as the status does

    async def body() -> None:
        for size in ((80, 24), (120, 40), (200, 60)):
            await check(size)

    asyncio.run(body())


def test_the_command_menu_has_a_surface_of_its_own_in_line_with_the_input_box() -> None:
    entries: list[dict[str, object]] = [
        {"event": "reply", "text": f"A line of the transcript, number {n}.", "session": SESSION}
        for n in range(40)
    ]
    app = _app(FakeTail(entries), session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        await pilot.pause(0.1)
        _input(app).set_text("/")
        await _until(pilot, lambda: app.query_one("#menu").display)
        await pilot.pause(0.05)
        menu, prompt = app.query_one("#menu"), app.query_one("#prompt")
        assert (menu.region.x, menu.region.right) == (prompt.region.x, prompt.region.right)
        assert menu.styles.border.top[0] == "round"  # a frame, so transcript lines never join it

    _run(test, app)


def test_the_menu_cuts_a_long_description_with_an_ellipsis_at_a_narrow_width() -> None:
    app = _app(session=FakeSession())

    async def body() -> None:
        async with app.run_test(size=(60, 24)) as pilot:
            _input(app).set_text("/")
            await _until(pilot, lambda: app.query_one("#menu").display)
            await pilot.pause(0.05)
            menu = app.query_one("#menu", PointerList)
            rows = [
                "".join(seg.text for seg in menu.render_line(y)) for y in range(menu.size.height)
            ]
            assert any(row.rstrip().endswith("…") for row in rows)
            assert all(len(row.rstrip()) <= menu.size.width for row in rows)

    asyncio.run(body())


def test_effort_shows_how_hard_the_model_thinks() -> None:
    app = _app(session=FakeSession(), settings=ChatConfig(thinking="low"))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/effort")
        await _until(pilot, lambda: _has(app, "Thinking effort: low"))
        assert _has(app, "up to high")  # how far the model may raise it itself

    _run(test, app)


def test_effort_sets_how_hard_the_model_thinks_from_the_next_request() -> None:
    saved: list[ChatConfig] = []
    session = FakeSession()
    app = _app(session=session, services=Services(save_settings=saved.append))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/effort medium")
        await _until(pilot, lambda: bool(saved))
        await _until(pilot, lambda: _has(app, "Thinking effort: medium"))

    _run(test, app)
    assert session.thought == ["medium"]
    assert saved[-1].thinking == "medium"


def test_effort_above_the_ceiling_raises_the_ceiling_too() -> None:
    saved: list[ChatConfig] = []
    settings = ChatConfig(thinking="none", thinking_max="low")
    app = _app(
        session=FakeSession(), settings=settings, services=Services(save_settings=saved.append)
    )

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/effort high")
        await _until(pilot, lambda: bool(saved))

    _run(test, app)
    assert (saved[-1].thinking, saved[-1].thinking_max) == ("high", "high")
    ChatConfig.model_validate(saved[-1].model_dump())  # still a valid section


def test_effort_refuses_a_level_it_does_not_know() -> None:
    session = FakeSession()
    app = _app(session=session)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/effort extreme")
        await _until(pilot, lambda: _has(app, "none, low, medium or high"))

    _run(test, app)
    assert session.thought == []


def test_model_shows_the_mode_and_each_modes_model() -> None:
    services = Services(
        model_report=lambda: ["mode: normal", "* normal: flash", "  coding: flash-coder"],
        switch_model=lambda _to, _say: "",
    )
    app = _app(session=FakeSession(), services=services)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/model")
        await _until(pilot, lambda: _has(app, "coding: flash-coder"))

    _run(test, app)


def test_model_switches_the_mode_and_says_what_is_loading() -> None:
    asked: list[str] = []

    def switch(to: str, say: Callable[[str], None]) -> str:
        asked.append(to)
        say("Switching to the coding model, flash-coder. It takes about 20 seconds.")
        return "Now in coding mode: flash-coder, loaded in 21 s."

    services = Services(model_report=list, switch_model=switch)
    app = _app(session=FakeSession(), services=services)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/model coding")
        await _until(pilot, lambda: _has(app, "Now in coding mode"))
        assert _has(app, "about 20 seconds")

    _run(test, app)
    assert asked == ["coding"]


def test_model_refuses_a_mode_it_does_not_know() -> None:
    asked: list[str] = []
    services = Services(model_report=list, switch_model=lambda to, _say: asked.append(to) or "")
    app = _app(session=FakeSession(), services=services)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/model turbo")
        await _until(pilot, lambda: _has(app, "it is normal or coding"))

    _run(test, app)
    assert asked == []


def test_a_voice_restart_that_fails_after_config_is_shown() -> None:
    from jarvis.chat.app import SharedSetting  # ruff: ignore[import-outside-top-level] - only this test

    def failed() -> None:
        raise RuntimeError(
            "the voice service did not restart, so it still runs on the old settings"
        )

    def shared() -> list[SharedSetting]:
        return [SharedSetting("audio.speech_speed", "Speaking speed", (1, 2), 1)]

    services = Services(shared=shared, set_shared=lambda _k, _v: None, apply=failed)
    app = _app(session=FakeSession(), services=services)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await _open_tab(pilot, "Voice")
        await pilot.press("right", "escape")
        await _until(pilot, lambda: _has(app, "still runs on the old settings"))

    _run(test, app)


def _risky_services(values: dict[str, object]) -> Services:
    from jarvis.chat.app import SharedSetting  # ruff: ignore[import-outside-top-level] - only these tests

    def shared() -> list[SharedSetting]:
        return [
            SharedSetting(
                "boundaries.bypass_consent",
                "Act without asking first",
                (False, True),
                values["boundaries.bypass_consent"],
                "Safety",
                "Acts without a yes.",
                "Jarvis will act without asking you first.",
            )
        ]

    def change(key: str, value: object) -> None:
        values[key] = value

    return Services(shared=shared, set_shared=change, apply=lambda: None)


def _settings_text(app: ChatApp) -> str:
    """Everything the settings screen says: the tabs, the search line, the list and the footer."""
    listed = app.query_one("#settings-list", OptionList)
    parts = [
        _plain(app.query_one(f"#settings-{name}", Static).content)
        for name in ("tabs", "search", "footer")
    ]
    return "\n".join(
        [parts[0], parts[1]]
        + [_plain(listed.get_option_at_index(i).prompt) for i in range(listed.option_count)]
        + [parts[2]]
    )


async def _open_tab(pilot: Pilot[None], title: str) -> None:
    """Press Tab until the settings screen shows the section `title`.

    Raises:
        AssertionError: the screen has no such section.
    """
    panel = pilot.app.query_one("#settings", SettingsPanel)
    for _ in range(len(panel.sections)):
        if panel.tab is not None and panel.tab.title == title:
            return
        await pilot.press("tab")
        await pilot.pause(0.02)
    raise AssertionError(f"no section {title}")


def test_a_risky_setting_is_turned_on_only_after_it_says_what_that_means() -> None:
    values: dict[str, object] = {"boundaries.bypass_consent": False}
    app = _app(session=FakeSession(), services=_risky_services(values))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await _open_tab(pilot, "Safety")
        await pilot.press("enter")
        await _until(pilot, lambda: "Turn on?" in _settings_text(app))
        assert values["boundaries.bypass_consent"] is False
        assert "act without asking you first" in _settings_text(app)
        await pilot.press("enter")
        await _until(pilot, lambda: values["boundaries.bypass_consent"] is True)
        assert "⚠ on" in _settings_text(app)  # marked while it is on
        await pilot.press("enter")  # turning it off asks nothing
        await _until(pilot, lambda: values["boundaries.bypass_consent"] is False)

    _run(test, app)


def test_acting_without_asking_changed_in_config_applies_to_this_chat_at_once() -> None:
    values: dict[str, object] = {"boundaries.bypass_consent": False}
    session = FakeSession()
    app = _app(session=session, services=_risky_services(values))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await _open_tab(pilot, "Safety")
        await pilot.press("enter")
        await _until(pilot, lambda: "Turn on?" in _settings_text(app))
        assert session.bypass == []
        await pilot.press("enter")
        await _until(pilot, lambda: session.bypass == [True])
        await pilot.press("enter")
        await _until(pilot, lambda: session.bypass == [True, False])

    _run(test, app)


def test_esc_or_moving_away_leaves_a_risky_setting_off() -> None:
    values: dict[str, object] = {"boundaries.bypass_consent": False}
    app = _app(session=FakeSession(), services=_risky_services(values))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await _open_tab(pilot, "Safety")
        await pilot.press("right")
        await _until(pilot, lambda: "Turn on?" in _settings_text(app))
        await pilot.press("escape")
        await _until(pilot, lambda: "Turn on?" not in _settings_text(app))
        assert app.query_one("#settings").display  # Esc took back the question, not the screen
        await pilot.press("enter")
        await _until(pilot, lambda: "Turn on?" in _settings_text(app))
        await pilot.press("up")
        await _until(pilot, lambda: "Turn on?" not in _settings_text(app))
        await pilot.press("down", "enter")
        await _until(pilot, lambda: "Turn on?" in _settings_text(app))

    _run(test, app)
    assert values["boundaries.bypass_consent"] is False


def test_config_shows_sections_and_what_each_setting_does_and_left_steps_back() -> None:
    saved: list[ChatConfig] = []
    app = _app(session=FakeSession(), services=Services(save_settings=saved.append))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        text = _settings_text(app)
        assert "Chat" in text  # the first tab
        assert "How much of the work the transcript shows" in text  # said once, in the footer
        assert "Thinking effort" in text
        assert "Earlier requests keep" in text
        assert not app.query_one("#stage").display  # the screen takes the transcript's place
        await pilot.press("left")  # replies, back round to everything
        await _until(pilot, lambda: bool(saved))
        await pilot.press("escape")
        await _until(pilot, lambda: app.query_one("#stage").display)

    _run(test, app)
    assert saved[0].verbosity == "everything"


def test_config_switches_the_browser_tools_off_from_the_next_chat() -> None:
    saved: list[ChatConfig] = []
    app = _app(session=FakeSession(), services=Services(save_settings=saved.append))
    row = [setting.key for setting in SETTINGS].index("browser")

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        assert "Headless browser" in _settings_text(app)
        await pilot.press(*["down"] * row, "enter")  # on or off: Enter turns it over
        await _until(pilot, lambda: bool(saved))
        await pilot.press("escape")
        await _until(pilot, lambda: app.query_one("#prompt").display)

    _run(test, app)
    assert saved[0].browser is False
    assert not next(s for s in SETTINGS if s.key == "browser").live  # its tools are set at start


def test_thinking_effort_set_in_config_reaches_the_agent() -> None:
    session = FakeSession()
    app = _app(session=session, services=Services(save_settings=lambda _c: None))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await pilot.press("down", "down", "down", "down", "right")  # past Replies to Thinking
        await _until(pilot, lambda: bool(session.thought))

    _run(test, app)
    assert session.thought == ["low"]


def _write_call(mode: str, chars: int) -> list[dict[str, object]]:
    own: dict[str, object] = {"session": SESSION, "via": "chat"}
    args = {"content": {"sha256": "ab", "chars": chars}, "mode": mode, "path": "/x/game.js"}
    done = "Added to /x/game.js" if mode == "append" else "Created /x/game.js"
    return [
        {"event": "tool_call", "op": "write_file", "args": args, "verdict": "run", **own},
        {"event": "tool_result", "op": "write_file", "ok": True, "text": done, **own},
    ]


def _rows(app: ChatApp) -> list[str]:
    return [_words(w) for w in app.query(".entry") if w.display and _words(w).strip()]


def test_repeated_calls_of_one_tool_show_as_one_row_that_counts_them() -> None:
    entries = [*_write_call("create", 10), *[e for _ in range(3) for e in _write_call("append", 5)]]
    app = _app(FakeTail(entries), settings=ChatConfig(verbosity="steps"))

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: any("4 calls" in row for row in _rows(app)))
        rows = _rows(app)
        assert sum("Write file" in row for row in rows) == 1
        assert sum("Added to /x/game.js" in row for row in rows) == 1  # the newest result
        assert not any("Created /x/game.js" in row for row in rows)

    _run(test, app)


def test_every_call_of_a_chain_shows_with_every_detail() -> None:
    entries = [*_write_call("create", 10), *_write_call("append", 5)]
    app = _app(FakeTail(entries), settings=ChatConfig(verbosity="everything"))

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: any("2 calls" in row for row in _rows(app)))
        assert any("Created /x/game.js" in row for row in _rows(app))

    _run(test, app)


def test_a_failure_or_a_question_ends_a_chain_and_stays_its_own_row() -> None:
    own: dict[str, object] = {"session": SESSION, "via": "chat"}
    failed = {"event": "tool_result", "op": "write_file", "ok": False, "text": "disk full", **own}
    asked = {**_write_call("append", 5)[0], "verdict": "confirm"}
    entries = [
        *_write_call("create", 10),
        _write_call("append", 5)[0],
        failed,
        *_write_call("append", 5),
        asked,
    ]
    app = _app(FakeTail(entries), settings=ChatConfig(verbosity="steps"))

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: any("needs your yes" in row for row in _rows(app)))
        rows = _rows(app)
        assert any("not done: disk full" in row for row in rows)
        assert sum("Write file" in row for row in rows) == 3  # a chain of 2, one after, one asked

    _run(test, app)


def test_format_sets_how_replies_are_written_from_the_next_request() -> None:
    saved: list[ChatConfig] = []
    session = FakeSession()
    services = Services(
        save_settings=saved.append, system_prompt=lambda chat: f"prompt {chat.response_format}"
    )
    app = _app(session=session, services=services)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/format")
        await _until(pilot, lambda: _has(app, "Replies: concise"))
        await _type(pilot, "/format verbose")
        await _until(pilot, lambda: bool(saved))
        await _type(pilot, "/format rambling")
        await _until(pilot, lambda: _has(app, "concise, normal or verbose"))

    _run(test, app)
    assert session.systems == ["prompt verbose"]
    assert saved[-1].response_format == "verbose"


def test_a_risky_setting_with_several_values_asks_before_leaving_off() -> None:
    from jarvis.chat.app import SharedSetting  # ruff: ignore[import-outside-top-level] - only this test

    values: dict[str, object] = {"audio.recall": 0}

    def shared() -> list[SharedSetting]:
        return [
            SharedSetting(
                "audio.recall",
                "Recall",
                (0, 10, 30),
                values["audio.recall"],
                "Listening",
                "",
                "Jarvis will keep audio.",
            )
        ]

    def change(key: str, value: object) -> None:
        values[key] = value

    services = Services(shared=shared, set_shared=change, apply=lambda: None)
    app = _app(session=FakeSession(), services=services)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await _open_tab(pilot, "Listening")
        await pilot.press("right")
        await _until(pilot, lambda: "Turn on?" in _settings_text(app))
        assert values["audio.recall"] == 0
        await pilot.press("enter")  # Enter takes the value it asked about
        await _until(pilot, lambda: values["audio.recall"] == 10)
        await pilot.press("right")  # from one on value to another: no question
        await _until(pilot, lambda: values["audio.recall"] == 30)

    _run(test, app)


def _drawn(widget: Widget) -> str:
    """What a widget draws, line by line, as the terminal shows it (what is clipped is missing)."""
    from textual.geometry import Region  # ruff: ignore[import-outside-top-level] - only these tests

    lines = widget.render_lines(Region(0, 0, widget.size.width, widget.size.height))
    return "\n".join(strip.text for strip in lines)


def _squeezed(text: str) -> str:
    return "".join(text.split())


def _full_services(
    values: dict[str, object], moves: list[tuple[str, int]] | None = None
) -> Services:
    """Every tab: Jarvis's own settings at their defaults (acting without asking off), two tools."""
    from jarvis.chat.app import SharedSetting  # ruff: ignore[import-outside-top-level] - only these tests
    from jarvis.cli import SHARED_SETTINGS  # ruff: ignore[import-outside-top-level] - only these tests
    from jarvis.config import Config  # ruff: ignore[import-outside-top-level] - only these tests
    from jarvis.policy import OpClass  # ruff: ignore[import-outside-top-level] - only these tests
    from jarvis.toolswitch import (  # ruff: ignore[import-outside-top-level] - only these tests
        ToolSetting,
        tool_settings,
    )

    cfg = Config()
    real = max(tool_settings(cfg, []), key=lambda tool: len(tool.summary))  # the longest summary
    tools = {
        "web_search": ToolSetting(
            "web_search", real.summary, OpClass.SAFE, OpClass.SAFE, tuple(OpClass)
        ),
        "write_file": ToolSetting(
            "write_file", "Write a file", OpClass.CONFIRM, OpClass.CONFIRM, tuple(OpClass)
        ),
    }

    def shared() -> list[SharedSetting]:
        out: list[SharedSetting] = []
        for item in SHARED_SETTINGS:
            key = f"{item.section}.{item.key}"
            found: object = cfg
            for part in item.section.split("."):
                found = getattr(found, part)
            default = False if key == "boundaries.bypass_consent" else getattr(found, item.key)
            out.append(
                SharedSetting(
                    key,
                    item.label,
                    item.values,
                    values.get(key, default),
                    item.heading,
                    item.about,
                    item.risky,
                    item.unit,
                    item.zero,
                )
            )
        return out

    def move(name: str, by: int = 1) -> ToolSetting:
        if moves is not None:
            moves.append((name, by))
        old = tools[name]
        now = old.following() if by > 0 else old.preceding()
        tools[name] = ToolSetting(name, old.summary, now, old.shipped, old.choices)
        return tools[name]

    return Services(
        shared=shared,
        set_shared=lambda key, value: values.update({key: value}),
        apply=lambda: None,
        save_settings=lambda _c: None,
        tools=lambda: list(tools.values()),
        next_tool_setting=move,
    )


def test_a_space_typed_into_the_search_is_part_of_the_search() -> None:
    saved: list[ChatConfig] = []
    app = _app(session=FakeSession(), services=Services(save_settings=saved.append))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        panel = app.query_one("#settings", SettingsPanel)
        await pilot.press(*"headless", "space", *"browser")
        await pilot.pause(0.05)
        assert saved == []  # the space did not turn the highlighted setting over
        assert panel.searched == "headless browser"
        assert [row.name for row in panel.rows] == ["Headless browser"]
        await pilot.press("escape", "down", "space")  # with nothing searched, space still changes
        await _until(pilot, lambda: bool(saved))

    _run(test, app)
    assert saved[0].toolbar is False


async def _in_size(
    size: tuple[int, int],
    services: Services,
    test: Callable[[ChatApp, Pilot[None], tuple[int, int]], Awaitable[None]],
) -> None:
    app = _app(session=FakeSession(), services=services)
    async with app.run_test(size=size) as pilot:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await pilot.pause(0.05)
        await test(app, pilot, size)


def test_a_narrow_window_still_shows_the_active_tab_and_the_whole_danger_prompt() -> None:
    async def body() -> None:
        for size in ((40, 24), (44, 24), (50, 24)):

            async def check(app: ChatApp, pilot: Pilot[None], size: tuple[int, int]) -> None:
                tabs = app.query_one("#settings-tabs", Static)
                for title in ("Safety", "Tools"):
                    await _open_tab(pilot, title)
                    await pilot.pause(0.02)
                    assert title in _drawn(tabs), (size, title)  # the active tab is in view
                await _open_tab(pilot, "Safety")
                await pilot.press("enter")
                await _until(pilot, lambda: "Turn on?" in _settings_text(app))
                await pilot.pause(0.05)
                shown = _squeezed(_drawn(app.query_one("#settings-footer", Static)))
                assert "esc:keepitoff" in shown, size  # the key hint is not cut
                assert "enteror→again:turniton" in shown, size

            await _in_size(size, _full_services({}), check)

    asyncio.run(body())


def _holds(app: ChatApp, about: str, size: tuple[int, int]) -> None:
    """The footer shows the restart note whole and last, then the about text or the start of it."""
    footer = app.query_one("#settings-footer", Static)
    shown = _squeezed(_drawn(footer))
    assert shown.endswith("whenyouclosesettings."), size  # never cut
    assert "Applies" in shown, size
    assert footer.region.height <= 8, size
    if size[0] >= 60:
        assert _squeezed(about) in shown, size  # room for all of it
    else:
        assert _squeezed(about)[:20] in shown, size  # trimmed with an ellipsis, not the note


def test_the_footer_grows_to_hold_the_longest_setting_with_the_restart_note() -> None:
    async def body() -> None:
        for size in ((40, 24), (44, 24), (60, 24), (80, 24)):

            async def check(app: ChatApp, pilot: Pilot[None], size: tuple[int, int]) -> None:
                from jarvis.cli import SHARED_SETTINGS  # ruff: ignore[import-outside-top-level] - only this test

                await _open_tab(pilot, "Voice")
                await pilot.press("right")  # a change that waits for the restart
                await _open_tab(pilot, "Mode")
                longest = max((s.about for s in SHARED_SETTINGS if s.heading == "Mode"), key=len)
                panel = app.query_one("#settings", SettingsPanel)
                for _ in range(len(panel.rows)):
                    if panel.row is not None and panel.row.about == longest:
                        break
                    await pilot.press("down")
                await pilot.pause(0.05)
                _holds(app, longest, size)
                await _open_tab(pilot, "Tools")
                await pilot.pause(0.05)
                assert panel.row is not None
                assert len(panel.row.about) > 100  # the real longest summary and the suffix
                _holds(app, panel.row.about, size)

            await _in_size(size, _full_services({}), check)

    asyncio.run(body())


def test_config_fits_the_smallest_usual_window() -> None:
    async def body() -> None:
        async def check(app: ChatApp, pilot: Pilot[None], _size: tuple[int, int]) -> None:
            panel = app.query_one("#settings", SettingsPanel)
            assert panel.region.bottom <= app.size.height
            tabs = _drawn(app.query_one("#settings-tabs", Static))
            for title in ("Chat", "Mode", "Voice", "Listening", "Safety", "Tools"):
                assert title in tabs
            assert "How much of the work" in _drawn(app.query_one("#settings-footer", Static))
            await pilot.press("end")  # the list scrolls, and the footer stays
            await pilot.pause(0.05)
            assert panel.row is not None
            assert panel.row.name in _drawn(app.query_one("#settings-list", OptionList))
            assert panel.row.about.split(";")[0][:20] in _drawn(
                app.query_one("#settings-footer", Static)
            )
            assert app.query_one("#settings-footer").region.bottom <= panel.region.bottom

        await _in_size((80, 24), _full_services({}), check)

    asyncio.run(body())


def test_a_value_picked_from_the_list_for_a_risky_setting_asks_and_enter_confirms() -> None:
    values: dict[str, object] = {}
    app = _app(session=FakeSession(), services=_full_services(values))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await _open_tab(pilot, "Listening")
        await pilot.press("down", "enter")  # Recall PC audio: the list of values
        await pilot.press("down", "enter")  # pick the second
        await _until(pilot, lambda: "Turn on?" in _settings_text(app))
        assert "audio.pc_audio_recall_s" not in values
        footer = _drawn(app.query_one("#settings-footer", Static))
        assert "enter again" in footer
        assert "→" not in footer  # → would step to another value, not confirm this one
        await pilot.press("enter")
        await _until(pilot, lambda: values.get("audio.pc_audio_recall_s") == 10)
        await pilot.press("enter", "down", "enter")  # from one on value to another: no question
        await _until(pilot, lambda: values.get("audio.pc_audio_recall_s") == 30)
        assert "Turn on?" not in _settings_text(app)

    _run(test, app)


def test_left_moves_a_tool_back_as_right_moves_it_on_and_never_below_how_it_ships() -> None:
    moves: list[tuple[str, int]] = []
    app = _app(session=FakeSession(), services=_full_services({}, moves))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await _open_tab(pilot, "Tools")
        await pilot.press("right", "right")
        await _until(pilot, lambda: len(moves) == 2)
        assert "needs the confirm key" in _settings_text(app)
        await pilot.press("left")
        await _until(pilot, lambda: len(moves) == 3)
        assert "asks first" in _settings_text(app)
        await pilot.press(
            "down", "left"
        )  # write_file ships asking first: back goes round, not below
        await _until(pilot, lambda: len(moves) == 4)
        assert "off" in _settings_text(app).split("write_file")[1]

    _run(test, app)
    assert moves == [("web_search", 1), ("web_search", 1), ("web_search", -1), ("write_file", -1)]


def test_switching_tab_leaves_a_risky_setting_off() -> None:
    values: dict[str, object] = {}
    app = _app(session=FakeSession(), services=_full_services(values))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await _open_tab(pilot, "Safety")
        await pilot.press("enter")
        await _until(pilot, lambda: "Turn on?" in _settings_text(app))
        await pilot.press("tab", "shift+tab")
        await pilot.pause(0.05)
        assert "Turn on?" not in _settings_text(app)
        await pilot.press("escape", "enter")  # nothing was turned on behind the question
        assert values.get("boundaries.bypass_consent") is not True

    _run(test, app)


def test_typing_backspace_and_tab_change_nothing_while_the_list_of_values_is_open() -> None:
    saved: list[ChatConfig] = []
    app = _app(session=FakeSession(), services=Services(save_settings=saved.append))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        panel = app.query_one("#settings", SettingsPanel)
        await pilot.press("down", "down", "down", "enter")
        listed = app.query_one("#settings-list", OptionList)
        await _until(pilot, lambda: listed.option_count == 3)
        await pilot.press("x", "backspace", "tab", "shift+tab")
        await pilot.pause(0.05)
        assert not panel.searched
        assert panel.tab is not None
        assert panel.tab.title == "Chat"
        assert listed.option_count == 3  # still the list of values
        assert saved == []

    _run(test, app)


def test_a_risky_setting_is_turned_on_only_to_the_value_it_asked_about() -> None:
    from jarvis.chat.app import SharedSetting  # ruff: ignore[import-outside-top-level] - only this test

    values: dict[str, object] = {"audio.recall": 0}

    def shared() -> list[SharedSetting]:
        return [
            SharedSetting(
                "audio.recall",
                "Recall",
                (0, 10, 30, 60),
                values["audio.recall"],
                "Listening",
                "",
                "Jarvis will keep audio.",
            )
        ]

    def change(key: str, value: object) -> None:
        values[key] = value

    services = Services(shared=shared, set_shared=change, apply=lambda: None)
    app = _app(session=FakeSession(), services=services)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/config")
        await _until(pilot, lambda: app.query_one("#settings").display)
        await _open_tab(pilot, "Listening")
        await pilot.press("right")
        await _until(pilot, lambda: "set 10" in _settings_text(app))
        await pilot.press("left")  # another value: asked about again, not turned on
        await _until(pilot, lambda: "set 60" in _settings_text(app))
        assert values["audio.recall"] == 0
        assert "Turn on?" in _settings_text(app)
        await pilot.press("left")
        await _until(pilot, lambda: values["audio.recall"] == 60)

    _run(test, app)


def test_an_unknown_command_says_so() -> None:
    app = _app(session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/nonsense")
        await _until(pilot, lambda: _has(app, "No command /nonsense"))

    _run(test, app)


# Opening, the toolbar and the name


def test_an_earlier_conversation_is_shown_when_the_window_opens() -> None:
    memory = Conversation(budget=1000)
    memory.add(Turn("my cat is called Miso", "A fine name.", "", "", 1.0))
    app = _app(session=FakeSession(memory=memory))

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "my cat is called Miso"))
        assert _has(app, "A fine name.")
        assert _has(app, "Continuing the conversation")

    _run(test, app)


def test_the_toolbar_shows_the_pcs_load_and_the_context() -> None:
    gib = 1024**3

    def stats() -> Stats:
        return Stats(12.0, 53.0, 33 * gib, 60 * gib, Gpu(41, 10 * gib, 20 * gib, 67.0))

    app = _app(session=FakeSession(), services=Services(stats=stats))

    async def test(pilot: Pilot[None]) -> None:
        bar = app.query_one("#toolbar", Static)
        await _until(pilot, lambda: "GPU  41%" in _plain(bar.content))
        line = _plain(bar.content)
        assert "CPU  12%" in line
        assert "VRAM 10.0/20.0G" in line
        assert "9.1k/24.6k" in line

    _run(test, app)


def test_the_transcript_and_the_input_share_the_window_width() -> None:
    app = _app(session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        await pilot.pause(0.05)
        transcript = app.query_one("#transcript")
        prompt = app.query_one("#prompt")
        assert transcript.region.width == app.size.width
        assert transcript.scrollable_content_region.right == prompt.region.right

    _run(test, app)


def test_the_footer_hint_fits_a_narrow_window_whole() -> None:
    services = Services(status=lambda: "voice on · model loaded")
    app = _app(session=FakeSession(), services=services)

    async def body() -> None:
        async with app.run_test(size=(80, 30)) as pilot:
            hints = app.query_one("#hints", Static)
            status = app.query_one("#status", Static)
            await _until(pilot, lambda: "voice on" in _plain(status.content))
            await pilot.pause(0.05)
            assert len(_plain(hints.content).strip()) <= hints.size.width
            assert "ctrl+o" in _plain(hints.content)

    asyncio.run(body())


def test_the_window_draws_nothing_on_the_terminals_black() -> None:
    """Omarchy's terminals set colour 0 (black) to the background, so a band on it never shows."""
    assert "ansi_black" not in ChatApp.CSS


def test_the_toolbar_can_be_turned_off() -> None:
    app = _app(session=FakeSession(), settings=ChatConfig(toolbar=False))

    async def test(pilot: Pilot[None]) -> None:
        await pilot.pause(0.05)
        assert not app.query_one("#toolbar").display

    _run(test, app)


def test_the_window_uses_the_assistants_name() -> None:
    answers: list[bool | Declined] = []
    done = threading.Event()
    app = _app(session=_confirming(answers, done), name="Friday")

    async def test(pilot: Pilot[None]) -> None:
        assert _has(app, "Welcome to Friday")
        await _type(pilot, "queue low tide")
        await _until(pilot, lambda: _question_open(app))
        assert app.query_one("#question").border_title == "Friday asks"

    _run(test, app)


def test_timers_that_come_while_the_window_closes_do_nothing() -> None:
    tail = FakeTail()
    app = _app(tail, session=FakeSession(), settings=ChatConfig(toolbar=False))

    async def test(pilot: Pilot[None]) -> None:
        await pilot.pause(0.05)
        tail.entries.append({"event": "reply", "text": "Paused.", "session": "other"})
        await app.screen.remove_children()  # as when the window is taken down
        # The timers' own calls:
        app._tick()  # pyright: ignore[reportPrivateUsage] - the timer callback, run by hand
        app._poll()  # pyright: ignore[reportPrivateUsage] - the poll callback, run by hand
        app._refresh_stats()  # pyright: ignore[reportPrivateUsage] - the stats callback, run by hand

    _run(test, app)


def test_steps_carry_their_level_as_a_class() -> None:
    app = _app(FakeTail(list(STEPS)), settings=ChatConfig(verbosity="everything"))

    async def test(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _has(app, "4.6 s"))
        levels = {w.shown.level for w in _transcript(app) if isinstance(w, Entry | Chain)}
        assert levels == {"steps", "everything"}

    _run(test, app)


# Pictures: Ctrl+V, a pasted or dropped path, and /image


def _png(width: int = 40, height: int = 20) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, "PNG")
    return buffer.getvalue()


def _clipboard(clip: Clip | Exception) -> Callable[[], Clip]:
    def read() -> Clip:
        if isinstance(clip, Exception):
            raise clip
        return clip

    return read


def _picture_app(
    session: FakeSession,
    *,
    paste: Callable[[], Clip] | None = None,
    screenshots: tuple[Path, ...] = (),
) -> ChatApp:
    return _app(
        session=session, services=Services(pictures=True, paste=paste, screenshots=screenshots)
    )


def _attached(app: ChatApp) -> str:
    return _plain(app.query_one("#pictures", Static).content)


def test_ctrl_v_attaches_a_copied_picture_as_a_tag_like_claude_code() -> None:
    session = FakeSession()
    app = _picture_app(session, paste=_clipboard(Clip(image=_png(), mime="image/png")))

    async def test(pilot: Pilot[None]) -> None:
        await pilot.press("ctrl+v")
        await _until(pilot, lambda: "[Image #1]" in _input(app).text)
        assert "pasted picture" in _attached(app)
        assert "40x20" in _attached(app)
        await pilot.press("w", "h", "a", "t", "?")
        await pilot.press("enter")
        await _until(pilot, lambda: session.pictures != [])
        assert session.asked == ["[Image #1] what?"]
        assert session.pictures == [["pasted picture"]]
        await _until(pilot, lambda: not app.query_one("#pictures").display)

    _run(test, app)


def test_a_picture_claiming_a_huge_size_is_refused_with_a_note() -> None:
    huge = Clip(image=claimed_png(20000, 20000), mime="image/png")
    app = _picture_app(FakeSession(), paste=_clipboard(huge))

    async def test(pilot: Pilot[None]) -> None:
        await pilot.press("ctrl+v")
        await _until(pilot, lambda: _has(app, "too large"))
        assert app.is_running

    _run(test, app)


def test_a_picture_that_fails_to_load_unexpectedly_leaves_the_window_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(_data: bytes, _name: str) -> Picture:
        raise RuntimeError("decoder bug")

    monkeypatch.setattr("jarvis.chat.app.prepare", broken)
    app = _picture_app(FakeSession(), paste=_clipboard(Clip(image=_png(), mime="image/png")))

    async def test(pilot: Pilot[None]) -> None:
        await pilot.press("ctrl+v")
        await _until(pilot, lambda: _has(app, "couldn't be read"))
        assert app.is_running

    _run(test, app)


def test_a_request_with_a_picture_sent_while_another_works_is_queued_with_it() -> None:
    release = threading.Event()

    def answer(text: str, _hooks: Hooks) -> str:
        if text == "pause":
            release.wait(5)
        return "Done."

    session = FakeSession(answer)
    app = _picture_app(session, paste=_clipboard(Clip(image=_png(), mime="image/png")))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: session.asked == ["pause"])
        await pilot.press("ctrl+v")
        await _until(pilot, lambda: "[Image #1]" in _input(app).text)
        await pilot.press("w", "h", "a", "t", "?")
        await pilot.press("enter")
        await _until(pilot, lambda: _has(app, "queued"))
        release.set()
        await _until(pilot, lambda: len(session.asked) == 2)

    _run(test, app)
    assert session.asked == ["pause", "[Image #1] what?"]
    assert session.pictures == [[], ["pasted picture"]]


def test_ctrl_v_pastes_copied_text_where_the_cursor_is() -> None:
    session = FakeSession()
    app = _picture_app(session, paste=_clipboard(Clip(text="the blue one")))

    async def test(pilot: Pilot[None]) -> None:
        _input(app).set_text("play ")
        await pilot.press("ctrl+v")
        await _until(pilot, lambda: _input(app).text == "play the blue one")

    _run(test, app)


def test_ctrl_v_of_a_copied_picture_file_attaches_the_file(tmp_path: Path) -> None:
    shot = tmp_path / "Shot.png"
    shot.write_bytes(_png())
    session = FakeSession()
    app = _picture_app(session, paste=_clipboard(Clip(text=f"file://{shot}")))

    async def test(pilot: Pilot[None]) -> None:
        await pilot.press("ctrl+v")
        await _until(pilot, lambda: "[Image #1]" in _input(app).text)
        assert "Shot.png" in _attached(app)

    _run(test, app)


def test_ctrl_v_says_when_nothing_is_copied_or_the_clipboard_fails() -> None:
    reads = iter([Clip(), ClipboardError("Pasting a picture needs wl-paste")])

    def read() -> Clip:
        value = next(reads)
        if isinstance(value, Exception):
            raise value
        return value

    app = _picture_app(FakeSession(), paste=read)

    async def test(pilot: Pilot[None]) -> None:
        await pilot.press("ctrl+v")
        await _until(pilot, lambda: _has(app, "Nothing is copied"))
        await pilot.press("ctrl+v")
        await _until(pilot, lambda: _has(app, "needs wl-paste"))
        assert not _input(app).text

    _run(test, app)


def test_a_picture_is_not_attached_while_the_picture_reader_is_off() -> None:
    paste = _clipboard(Clip(image=_png(), mime="image/png"))
    app = _app(session=FakeSession(), services=Services(paste=paste))

    async def test(pilot: Pilot[None]) -> None:
        await pilot.press("ctrl+v")
        await _until(pilot, lambda: _has(app, "picture reader is off"))
        assert "[Image" not in _input(app).text

    _run(test, app)


def test_a_dropped_picture_file_is_attached_and_other_pastes_stay_text(tmp_path: Path) -> None:
    shot = tmp_path / "drop me.png"
    shot.write_bytes(_png())
    session = FakeSession()
    app = _picture_app(session)

    async def test(pilot: Pilot[None]) -> None:
        _input(app).post_message(events.Paste(f"'{shot}'"))
        await _until(pilot, lambda: "[Image #1]" in _input(app).text)
        assert "drop me.png" in _attached(app)
        _input(app).post_message(events.Paste(" and the notes"))
        await _until(pilot, lambda: _input(app).text.endswith(" and the notes"))
        assert "[Image #2]" not in _input(app).text

    _run(test, app)


def test_deleting_the_tag_leaves_the_picture_out() -> None:
    session = FakeSession()
    app = _picture_app(session, paste=_clipboard(Clip(image=_png(), mime="image/png")))

    async def test(pilot: Pilot[None]) -> None:
        await pilot.press("ctrl+v")
        await _until(pilot, lambda: "[Image #1]" in _input(app).text)
        _input(app).set_text("never mind")
        await pilot.pause(0.02)
        assert not app.query_one("#pictures").display
        await pilot.press("enter")
        await _until(pilot, lambda: session.pictures != [])
        assert session.pictures == [[]]

    _run(test, app)


def test_slash_image_attaches_a_file_or_the_newest_screenshot(tmp_path: Path) -> None:
    named = tmp_path / "Diagram.PNG"
    named.write_bytes(_png(8, 8))
    shots = tmp_path / "Screenshots"
    shots.mkdir()
    (shots / "screenshot-1.png").write_bytes(_png(16, 9))
    session = FakeSession()
    app = _picture_app(session, screenshots=(shots,))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, f"/image {named}")
        await _until(pilot, lambda: "[Image #1]" in _input(app).text)
        assert "Diagram.PNG" in _attached(app)
        await _type(pilot, "/image")
        await _until(pilot, lambda: "[Image #2]" in _input(app).text)
        assert "screenshot-1.png" in _attached(app)
        await _type(pilot, f"/image {tmp_path / 'gone.png'}")
        await _until(pilot, lambda: _has(app, "no file"))

    _run(test, app)


def test_slash_image_without_a_screenshot_says_where_it_looked(tmp_path: Path) -> None:
    app = _picture_app(FakeSession(), screenshots=(tmp_path,))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/image")
        await _until(pilot, lambda: _has(app, "No screenshot"))
        assert _has(app, str(tmp_path))

    _run(test, app)


def _cached(tmp_path: Path, width: int = 400, height: int = 200) -> tuple[PictureCache, Path]:
    cache = PictureCache(tmp_path / "chat-pictures")
    picture = Picture("panda", "image/png", _png(width, height), width, height)
    return cache, cache.keep(picture)


def test_a_picture_jarvis_shows_appears_with_its_caption(tmp_path: Path) -> None:
    cache, path = _cached(tmp_path)
    tail = FakeTail()
    app = _app(tail, FakeSession(), services=Services(shown=cache))
    own = {"session": SESSION, "via": "chat"}

    async def test(pilot: Pilot[None]) -> None:
        tail.entries.append(
            {
                "event": "tool_result",
                "op": "show_image",
                "ok": True,
                "text": "Shown",
                "picture": str(path),
                "caption": "Red panda · zoo.example",
                **own,
            }
        )
        await _until(pilot, lambda: bool(app.query(".picture-caption")))  # mounted after the view
        view = app.query_one(PictureView)
        assert view.path == path
        assert "Red panda · zoo.example" in _plain(
            view.query_one(".picture-caption", Static).content
        )

    _run(test, app)


def test_a_tall_picture_is_kept_to_a_few_rows(tmp_path: Path) -> None:
    _, path = _cached(tmp_path, 200, 2000)

    assert PictureView.cells(path) <= 6
    assert PictureView.cells(_cached(tmp_path, 2000, 200)[1]) == 64


def test_a_picture_from_outside_the_chat_folder_or_another_session_is_not_shown(
    tmp_path: Path,
) -> None:
    cache, path = _cached(tmp_path)
    stray = tmp_path / "elsewhere.png"
    stray.write_bytes(_png())
    tail = FakeTail()
    app = _app(tail, FakeSession(), services=Services(shown=cache))

    async def test(pilot: Pilot[None]) -> None:
        base = {"event": "tool_result", "op": "show_image", "ok": True, "text": "Shown"}
        tail.entries.append({**base, "picture": str(stray), "session": SESSION})
        tail.entries.append({**base, "picture": str(path), "session": "someone-else"})
        tail.entries.append({**base, "picture": str(path), "session": SESSION, "via": "chat"})
        await _until(pilot, lambda: bool(app.query(PictureView)))
        await pilot.pause(0.05)
        assert [view.path for view in app.query(PictureView)] == [path]

    _run(test, app)


def test_under_a_reply_it_says_how_long_it_worked_when_it_was_done_and_how_fast() -> None:
    session = FakeSession(lambda _t, _h: "Paused.")
    app = _app(session=session, clock=Clock())

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: any(isinstance(w, Reply) for w in _transcript(app)))
        reply = next(w for w in _transcript(app) if isinstance(w, Reply))
        await _until(pilot, lambda: bool(reply.query(".reply-note")))
        note = _plain(reply.query_one(".reply-note", Static).content)
        assert "Worked for 1s · done " in note
        assert note.rstrip().endswith("· 50 TPS")

    _run(test, app)


def test_a_spoken_reply_from_the_log_gets_the_same_line() -> None:
    tail = FakeTail()
    app = _app(tail, FakeSession(), services=Services(twelve_hour=False))

    async def test(pilot: Pilot[None]) -> None:
        tail.entries.append(
            {
                "event": "reply",
                "text": "Paused.",
                "seconds": 281.2,
                "ts": "2026-11-21T17:37:00+00:00",
            }
        )
        await _until(pilot, lambda: any(isinstance(w, Reply) for w in _transcript(app)))
        reply = next(w for w in _transcript(app) if isinstance(w, Reply))
        await _until(pilot, lambda: bool(reply.query(".reply-note")))
        note = _plain(reply.query_one(".reply-note", Static).content)
        assert "Worked for 4m 41s · done " in note

    _run(test, app)


def test_the_command_menu_is_drawn_over_the_transcript_right_above_the_input() -> None:
    app = _app(session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        def layout() -> tuple[object, ...]:
            return (
                app.query_one("#transcript").region,
                app.query_one("#prompt").region,
                app.query_one("#footer").display,
                app.query_one("#toolbar").display,
            )

        await pilot.pause(0.05)
        before = layout()
        _input(app).set_text("/")
        await _until(pilot, lambda: app.query_one("#menu").display)
        await pilot.pause(0.05)
        menu = app.query_one("#menu").region
        assert layout() == before  # as in Claude Code, nothing moves
        assert menu.bottom == app.query_one("#prompt").region.y - 1  # one line above the input
        assert app.query_one("#transcript").region.contains_region(menu)
        for typed in ("/c", "/co", "/con", "/e", "/"):
            _input(app).set_text(typed)
            await pilot.pause(0.05)
            assert layout() == before, typed
        await pilot.press("escape")
        await pilot.pause(0.05)
        assert layout() == before

    _run(test, app)


def test_the_menu_marks_the_typed_letters_and_lights_the_whole_chosen_row() -> None:
    app = _app(session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        _input(app).set_text("/co")
        await _until(pilot, lambda: app.query_one("#menu").display)
        await pilot.pause(0.05)
        menu = app.query_one("#menu", PointerList)
        rows = [menu.get_option_at_index(i).prompt for i in range(menu.option_count)]
        assert all(isinstance(row, Text) for row in rows)
        chosen, other = rows[0], rows[1]
        assert isinstance(chosen, Text)
        assert isinstance(other, Text)
        assert chosen.plain.lstrip().startswith(f"{CHEVRON} /compact")

        def bold_at(row: Text, place: int) -> bool:
            return any(
                span.start <= place < span.end and "bold" in str(span.style) for span in row.spans
            )

        for row in (chosen, other):
            start = row.plain.index("/co")
            assert all(bold_at(row, start + offset) for offset in range(3)), row.plain
        assert not bold_at(other, other.plain.index("/co") + 3)  # the letters not typed are not
        end = len(chosen.plain.rstrip())  # the chosen row is lit to its end, description too
        assert any(ACCENT in str(span.style) and span.end >= end for span in chosen.spans)

    _run(test, app)


def test_the_menu_keeps_its_columns_and_one_line_a_command_on_a_narrow_window() -> None:
    app = _app(session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        await pilot.resize_terminal(60, 30)
        _input(app).set_text("/")
        await _until(pilot, lambda: app.query_one("#menu").display)
        await pilot.pause(0.05)
        menu = app.query_one("#menu", PointerList)

        def about_column(name: str) -> int:
            line = next(
                _plain(menu.get_option_at_index(index).prompt)
                for index in range(menu.option_count)
                if name in _plain(menu.get_option_at_index(index).prompt)
            )
            after = line.index(name) + len(name)
            return after + len(line[after:]) - len(line[after:].lstrip())

        everything = about_column("/exit")
        assert menu.option_count == len(commands.COMMANDS)
        assert menu.virtual_size.height == menu.option_count  # no command wraps
        _input(app).set_text("/ex")
        await pilot.pause(0.05)
        assert about_column("/exit") == everything

    _run(test, app)


def test_the_transcript_stays_at_its_newest_line_when_the_menu_opens() -> None:
    app = _app(session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        transcript = app.query_one("#transcript", VerticalScroll)
        for number in range(60):
            transcript.mount(Static(f"line {number}"))
        await pilot.pause(0.05)
        assert transcript.is_vertical_scroll_end
        _input(app).set_text("/")
        await pilot.pause()
        assert app.query_one("#menu").display
        assert transcript.is_vertical_scroll_end

    _run(test, app)


def test_slash_exit_leaves() -> None:
    app = _app(session=FakeSession())

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/exit")
        await pilot.pause(0.1)
        assert not app.is_running

    _run(test, app)


# What was pasted, and what a question shows (issue #30, A6 and P5)

LONG_PASTE = "Copied from a page: " + "the kettle needs descaling every month or so. " * 6


def test_a_paste_still_in_the_request_goes_with_it() -> None:
    session = FakeSession()
    app = _app(session=session)

    async def test(pilot: Pilot[None]) -> None:
        _input(app).post_message(events.Paste(LONG_PASTE))
        await _until(pilot, lambda: _input(app).text == LONG_PASTE)
        _input(app).insert_text(" sum it up")
        await pilot.press("enter")
        await _until(pilot, lambda: session.pasted != [])
        assert session.asked == [f"{LONG_PASTE} sum it up"]
        assert session.pasted == [[LONG_PASTE.strip()]]
        await _type(pilot, "pause")
        await _until(pilot, lambda: len(session.pasted) == 2)
        assert session.pasted[1] == []

    _run(test, app)


def test_ctrl_v_text_counts_as_pasted_too() -> None:
    session = FakeSession()
    app = _app(session=session, services=Services(paste=_clipboard(Clip(text=LONG_PASTE))))

    async def test(pilot: Pilot[None]) -> None:
        await pilot.press("ctrl+v")
        await _until(pilot, lambda: _input(app).text == LONG_PASTE)
        await pilot.press("enter")
        await _until(pilot, lambda: session.pasted != [])
        assert session.pasted == [[LONG_PASTE.strip()]]

    _run(test, app)


def test_a_long_paste_edited_in_the_box_still_counts_as_pasted() -> None:
    session = FakeSession()
    app = _app(session=session)

    async def test(pilot: Pilot[None]) -> None:
        _input(app).post_message(events.Paste(LONG_PASTE))
        await _until(pilot, lambda: _input(app).text == LONG_PASTE)
        edited = LONG_PASTE.replace("kettle", "teapot", 1)
        _input(app).set_text(edited)
        await pilot.pause(0.02)
        await pilot.press("enter")
        await _until(pilot, lambda: session.pasted != [])
        assert session.pasted == [[edited.strip()]]

    _run(test, app)


def test_a_cleared_box_forgets_what_was_pasted() -> None:
    session = FakeSession()
    app = _app(session=session)

    async def test(pilot: Pilot[None]) -> None:
        _input(app).post_message(events.Paste(LONG_PASTE))
        await _until(pilot, lambda: _input(app).text == LONG_PASTE)
        _input(app).set_text("")
        await pilot.pause(0.02)
        typed = "play something calm for the evening, " * 8
        assert len(typed) > 200
        await _type(pilot, typed)
        await _until(pilot, lambda: session.pasted != [])
        assert session.pasted == [[]]

    _run(test, app)


def test_the_window_asks_for_previews_and_shows_them_under_the_question() -> None:
    answers: list[bool | Declined] = []
    done = threading.Event()

    def answer(_text: str, hooks: Hooks) -> str:
        answers.append(hooks.confirm("Should I replace todo.md in Documents?\n# Plan\nbuy a boat"))
        done.set()
        return ""

    session = FakeSession(answer)
    app = _app(session=session)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "write my plan")
        await _until(pilot, lambda: bool(app.query_one("#question").display))
        shown = _plain(app.query_one("#question-text", Static).content)
        assert shown.splitlines()[:3] == [
            "Should I replace todo.md in Documents?",
            "# Plan",
            "buy a boat",
        ]
        assert session.hooks[0].preview is True
        await pilot.press("2")
        await _until(pilot, done.is_set)

    _run(test, app)


# The review of #30: a recalled paste, several short pastes, and a name in a question


@dataclass
class ScriptedModel:
    """A model that answers from a list, for a real agent behind the window."""

    replies: list[dict[str, Any]]

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **_kw: object
    ) -> dict[str, Any]:
        del messages, tools
        return self.replies.pop(0)

    @staticmethod
    def is_loaded() -> bool:
        return True


def _tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    function = {"name": name, "arguments": json.dumps(args)}
    return {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": function}]}


def test_a_long_paste_brought_back_with_up_still_counts_as_pasted(tmp_path: Path) -> None:
    kept = tmp_path / "chat-history"
    session = FakeSession()
    app = _app(session=session, services=Services(history=InputHistory(kept, 100)))

    async def test(pilot: Pilot[None]) -> None:
        _input(app).post_message(events.Paste(LONG_PASTE))
        await _until(pilot, lambda: _input(app).text == LONG_PASTE)
        await pilot.press("enter")
        await _until(pilot, lambda: len(session.pasted) == 1)
        await _type(pilot, "pause")
        await _until(pilot, lambda: len(session.pasted) == 2)
        _input(app).insert_text("draft")
        await pilot.press("up", "up")
        assert _input(app).text == LONG_PASTE.strip()
        await pilot.press("down", "down")
        assert _input(app).text == "draft"
        _input(app).set_text("")
        await pilot.pause(0.02)
        await pilot.press("up", "up", "enter")
        await _until(pilot, lambda: len(session.pasted) == 3)
        assert session.pasted[2] == [LONG_PASTE.strip()]

    _run(test, app)
    later = FakeSession()
    again = _app(session=later, services=Services(history=InputHistory(kept, 100)))

    async def next_session(pilot: Pilot[None]) -> None:
        await _until(pilot, lambda: _input(again).has_focus)
        await pilot.press("up")  # the newest: the paste, sent again above
        await _until(pilot, lambda: _input(again).text == LONG_PASTE.strip())
        await pilot.press("enter")
        await _until(pilot, lambda: later.pasted != [])
        assert later.asked == [LONG_PASTE.strip()]
        assert later.pasted == [[LONG_PASTE.strip()]]

    _run(next_session, again)


def test_a_draft_keeps_its_paste_when_up_and_down_pass_over_it() -> None:
    session = FakeSession()
    app = _app(session=session, services=Services(history=InputHistory(None, 100)))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: len(session.pasted) == 1)
        _input(app).post_message(events.Paste(LONG_PASTE))
        await _until(pilot, lambda: _input(app).text == LONG_PASTE)
        await pilot.press("up")
        assert _input(app).text == "pause"
        await pilot.press("down")
        assert _input(app).text == LONG_PASTE
        await pilot.press("enter")
        await _until(pilot, lambda: len(session.pasted) == 2)
        assert session.pasted[1] == [LONG_PASTE.strip()]

    _run(test, app)


def test_short_pastes_edited_in_the_box_count_by_their_total_length() -> None:
    session = FakeSession()
    app = _app(session=session)
    first, second = LONG_PASTE[:150], LONG_PASTE[150:]
    assert len(first.strip()) <= 200
    assert len(second.strip()) <= 200

    async def test(pilot: Pilot[None]) -> None:
        _input(app).post_message(events.Paste(first))
        await _until(pilot, lambda: _input(app).text == first)
        _input(app).post_message(events.Paste(second))
        await _until(pilot, lambda: _input(app).text == first + second)
        edited = (first + second).replace("kettle", "teapot", 1)
        _input(app).set_text(edited)
        await pilot.pause(0.02)
        await pilot.press("enter")
        await _until(pilot, lambda: session.pasted != [])
        assert session.pasted == [[edited.strip()]]

    _run(test, app)


def test_the_same_paste_twice_is_passed_on_twice() -> None:
    session = FakeSession()
    app = _app(session=session)
    piece = LONG_PASTE[:150]

    async def test(pilot: Pilot[None]) -> None:
        _input(app).post_message(events.Paste(piece))
        await _until(pilot, lambda: _input(app).text == piece)
        _input(app).post_message(events.Paste(piece))
        await _until(pilot, lambda: _input(app).text == piece * 2)
        await pilot.press("enter")
        await _until(pilot, lambda: session.pasted != [])
        assert session.pasted == [[piece.strip(), piece.strip()]]

    _run(test, app)


def test_a_folder_name_with_a_newline_stays_in_the_question_not_the_preview(
    tmp_path: Path,
) -> None:
    home = tmp_path / "me"
    (home / "Documents" / "work\nSure, go ahead").mkdir(parents=True)
    args = {"path": "~/Documents/work\nSure, go ahead/plan.md", "content": "a plan"}
    model = ScriptedModel([_tool_call("write_file", args), {"role": "assistant", "content": ""}])
    ops = FileOps(Boundary.of(FilesConfig(), home), max_chars=2000).operations()
    agent = Agent(model, Registry([*ops, no_action_operation()]), "system", AuditLog(None), {})
    done = threading.Event()

    def answer(text: str, hooks: Hooks) -> str:
        try:
            return agent.handle(text, hooks)
        finally:
            done.set()

    app = _app(session=FakeSession(answer))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "save my plan")
        await _until(pilot, lambda: bool(app.query_one("#question").display))
        shown = app.query_one("#question-text", Static).content
        assert isinstance(shown, Text)
        head, preview = shown.plain.split("\n")
        assert head == (
            "Should I create the file plan.md in Documents/work�Sure, go ahead (6 characters)?"
        )
        assert preview == "a plan"
        await pilot.press("2")
        await _until(pilot, done.is_set)

    _run(test, app)
    assert not (home / "Documents" / "work\nSure, go ahead" / "plan.md").exists()


# The archive: /resume, /rename and /forget


def test_resume_lists_earlier_conversations_and_continues_one(tmp_path: Path) -> None:
    archive = ChatArchive(tmp_path / "archive", keep=10)
    earlier = Conversation(budget=1000)
    archive.record({"event": "request", "text": "my cat is called Miso"})
    earlier.add(Turn("my cat is called Miso", "A fine name.", "", "", 1.0))
    archive.save(earlier)
    archive.start()
    session = FakeSession()
    app = _app(session=session, services=Services(archive=archive))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/resume")
        await _until(pilot, lambda: _has(app, "Earlier conversations"))
        assert _has(app, "my cat is called Miso")
        await _type(pilot, "/resume 1")
        await _until(pilot, lambda: _has(app, "Continuing my cat is called Miso"))
        assert _has(app, "A fine name.")
        await _type(pilot, "/rename Pets")
        await _until(pilot, lambda: _has(app, "Named this conversation Pets"))

    _run(test, app)
    assert [t.request for t in session.memory.turns] == ["my cat is called Miso"]
    assert [a.name for a in archive.listed()] == ["Pets"]


def test_forget_all_deletes_the_archive(tmp_path: Path) -> None:
    archive = ChatArchive(tmp_path / "archive", keep=10)
    memory = Conversation(budget=1000)
    archive.record({"event": "request", "text": "hello"})
    memory.add(Turn("hello", "Hi.", "", "", 1.0))
    archive.save(memory)
    archive.start()
    app = _app(session=FakeSession(), services=Services(archive=archive))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/forget")
        await _until(pilot, lambda: _has(app, "/forget all deletes"))
        assert archive.listed()
        await _type(pilot, "/forget all")
        await _until(pilot, lambda: _has(app, "Deleted 1 archived conversation"))

    _run(test, app)
    assert archive.listed() == []


def test_forget_all_also_forgets_this_conversation_so_none_of_it_is_archived_again(
    tmp_path: Path,
) -> None:
    archive = ChatArchive(tmp_path / "archive", keep=10)
    tail = FakeTail()
    session = FakeSession()

    def answer(text: str, _hooks: Hooks) -> str:
        tail.entries.append({"event": "request", "text": text, "session": SESSION})
        session.memory.add(Turn(text, "Noted.", "", "", 1.0))
        return "Noted."

    session.answer = answer
    app = _app(tail, session, services=Services(archive=archive))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "my locker code is 4711")
        await _until(pilot, lambda: bool(archive.listed()))
        await _type(pilot, "/forget all")
        await _until(pilot, lambda: _has(app, "Deleted 1 archived conversation"))
        assert not _has(app, "my locker code is 4711")
        await _type(pilot, "/clear")
        await _until(pilot, lambda: _has(app, "Started a new conversation"))

    _run(test, app)
    assert session.memory.turns == []
    assert archive.listed() == []


def test_forget_all_waits_for_a_request_to_finish(tmp_path: Path) -> None:
    archive = ChatArchive(tmp_path / "archive", keep=10)
    memory = Conversation(budget=1000)
    archive.record({"event": "request", "text": "hello"})
    memory.add(Turn("hello", "Hi.", "", "", 1.0))
    archive.save(memory)
    archive.start()
    release = threading.Event()

    def answer(_text: str, _hooks: Hooks) -> str:
        release.wait(5)
        return "Done."

    session = FakeSession(answer)
    app = _app(session=session, services=Services(archive=archive))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "work on it")
        await _until(pilot, lambda: bool(session.asked))
        await _type(pilot, "/forget all")
        await _until(pilot, lambda: _has(app, "Wait for the request to finish"))
        release.set()

    _run(test, app)
    assert [a.name for a in archive.listed()] == ["hello"]


class _WatchedArchive(ChatArchive):
    """An archive that notes whether each save came while the agent still worked."""

    def __init__(self, folder: Path, working: threading.Event) -> None:
        super().__init__(folder, keep=10)
        self.working = working
        self.saved_while_working: list[bool] = []

    @override
    def save(self, memory: Conversation) -> None:
        self.saved_while_working.append(self.working.is_set())
        super().save(memory)


def test_a_rename_during_a_request_is_saved_once_it_settles(tmp_path: Path) -> None:
    working, release = threading.Event(), threading.Event()
    archive = _WatchedArchive(tmp_path / "archive", working)
    session = FakeSession()

    def answer(text: str, _hooks: Hooks) -> str:
        working.set()
        release.wait(5)
        session.memory.add(Turn(text, "Done.", "", "", 1.0))
        working.clear()
        return "Done."

    session.answer = answer
    app = _app(session=session, services=Services(archive=archive))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "plan the trip")
        await _until(pilot, working.is_set)
        await _type(pilot, "/rename Trip")
        await _until(pilot, lambda: _has(app, "Named this conversation Trip"))
        release.set()
        await _until(pilot, lambda: bool(archive.listed()))

    _run(test, app)
    assert True not in archive.saved_while_working
    (kept,) = archive.listed()
    assert kept.name == "Trip"


def test_closing_during_a_request_archives_it_once_the_agent_has_finished(tmp_path: Path) -> None:
    working = threading.Event()
    archive = _WatchedArchive(tmp_path / "archive", working)
    session = FakeSession()

    def answer(text: str, hooks: Hooks) -> str:
        working.set()
        for _ in range(500):
            if hooks.cancelled():
                break
            time.sleep(0.01)
        session.memory.add(Turn(text, "(The owner stopped the request here.)", "", "", 1.0))
        working.clear()
        return ""

    session.answer = answer
    app = _app(session=session, services=Services(archive=archive))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "write the report")
        await _until(pilot, working.is_set)

    _run(test, app)  # the window closes while the request runs
    assert True not in archive.saved_while_working
    (kept,) = archive.listed()
    restored = Conversation(budget=1000)
    assert restored.load(tmp_path / "archive" / f"{kept.id}.memory.json")
    assert [t.request for t in restored.turns] == ["write the report"]


def test_a_finished_request_is_archived_with_the_windows_own_log(tmp_path: Path) -> None:
    archive = ChatArchive(tmp_path / "archive", keep=10)
    tail = FakeTail()
    session = FakeSession()

    def answer(text: str, _hooks: Hooks) -> str:
        own: dict[str, object] = {"session": SESSION, "via": "chat"}
        tail.entries.append({"event": "request", "text": text, **own})
        session.memory.add(Turn(text, "Paused it.", "", "", 1.0))
        return "Paused it."

    session.answer = answer
    app = _app(tail, session, services=Services(archive=archive))

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "pause")
        await _until(pilot, lambda: bool(archive.listed()))

    _run(test, app)
    (kept,) = archive.listed()
    assert kept.name == "pause"


def test_forget_all_that_cannot_delete_says_so_and_forgets_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = ChatArchive(tmp_path / "archive", keep=10)
    session = FakeSession()
    session.memory.add(Turn("keep me", "Kept.", "", "", 1.0))
    app = _app(FakeTail(), session, services=Services(archive=archive))

    def refuse() -> int:
        raise PermissionError

    monkeypatch.setattr(archive, "forget", refuse)

    async def test(pilot: Pilot[None]) -> None:
        await _type(pilot, "/forget all")
        await _until(pilot, lambda: _has(app, "Couldn't delete the archive"))

    _run(test, app)
    assert len(session.memory.turns) == 1
