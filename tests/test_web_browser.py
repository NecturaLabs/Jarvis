"""The headless browser driver, with agent-browser faked: no browser is ever started here."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from jarvis.web import browser as browser_module
from jarvis.web.browser import (
    ALLOWED_ACTIONS,
    CHROME_ARGS,
    Browser,
    BrowserError,
    BrowserTimeoutError,
    Element,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


class FakeProxy:
    """Stands in for the egress proxy: a fixed port, and the hosts it refused."""

    def __init__(self, refused: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.port = 40123
        self.started = self.stopped = False
        self._refused = set(refused)

    def start(self) -> int:
        self.started = True
        return self.port

    def stop(self) -> None:
        self.stopped = True

    def was_refused(self, host: str, port: int) -> bool:
        return f"{host}:{port}" in self._refused


class Page:
    """What the fake browser shows."""

    def __init__(
        self,
        url: str = "https://site.example/",
        title: str = "Site",
        text: str = "Hello from the page.",
        refs: dict[str, dict[str, str]] | None = None,
        attrs: dict[tuple[str, str], str] | None = None,
    ) -> None:
        super().__init__()
        self.url, self.title, self.text = url, title, text
        self.refs = refs if refs is not None else {"e1": {"role": "link", "name": "More"}}
        self.attrs = attrs or {}


class FakeAgentBrowser:
    """Answers `batch --json` the way agent-browser does, and records every call."""

    def __init__(self, page: Page | None = None) -> None:
        super().__init__()
        self.page = page or Page()
        self.calls: list[tuple[list[str], list[list[str]], dict[str, str]]] = []
        self.fail: str | None = None  # a command name that fails
        self.fail_after: str | None = None  # ... only once this command has run
        self.timeout = False
        self.on: dict[str, Callable[[list[str]], None]] = {}

    def __call__(
        self, argv: Sequence[str], *, stdin: str, env: Mapping[str, str], cwd: Path, timeout: float
    ) -> str:
        assert timeout > 0
        assert cwd.is_dir()
        commands: list[list[str]] = json.loads(stdin) if stdin else []
        self.calls.append((list(argv), commands, dict(env)))
        if self.timeout and argv[-1] != "close":
            raise BrowserTimeoutError("the browser took too long")
        if argv[-1] == "close":
            return json.dumps({"success": True, "data": {"closed": True}})
        rows: list[dict[str, Any]] = []
        for command in commands:
            ran = [c[0] for c in self.commands()]
            if command[0] == self.fail and (self.fail_after is None or self.fail_after in ran):
                rows.append({"command": command, "success": False, "error": "x", "result": None})
                break
            result = self._result(command)
            if command[0] in self.on:  # after the command: what the page did of its own
                self.on[command[0]](command)
            rows.append({"command": command, "success": True, "error": None, "result": result})
        return json.dumps(rows)

    def _result(self, command: list[str]) -> dict[str, Any]:  # ruff: ignore[too-many-return-statements] - one answer for each command
        page = self.page
        if command[0] == "open":
            page.url = command[1]
            return {"url": page.url, "title": page.title}
        if command[:2] == ["get", "url"]:
            return {"url": page.url}
        if command[:2] == ["get", "title"]:
            return {"title": page.title}
        if command[:2] == ["get", "text"]:
            return {"text": page.text}
        if command[:2] == ["get", "attr"]:
            return {"value": page.attrs.get((command[2].lstrip("@"), command[3]), "")}
        if command[0] == "snapshot":
            return {"refs": page.refs, "snapshot": ""}
        if command[0] == "screenshot":
            Path(command[1]).write_bytes(b"\x89PNG fake")
            return {"path": command[1]}
        return {}

    def commands(self) -> list[list[str]]:
        return [c for _argv, batch, _env in self.calls for c in batch]


class FakeTimer:
    """A timer that only says whether it is running."""

    def __init__(self) -> None:
        super().__init__()
        self.state = [False]

    def start(self) -> None:
        self.state[0] = True

    def cancel(self) -> None:
        self.state[0] = False


class Timers:
    """Fake idle timers: nothing fires until the test says so."""

    def __init__(self) -> None:
        super().__init__()
        self.made: list[tuple[float, Callable[[], None], list[bool]]] = []

    def __call__(self, seconds: float, action: Callable[[], None]) -> FakeTimer:
        timer = FakeTimer()
        self.made.append((seconds, action, timer.state))
        return timer

    def fire(self) -> None:
        seconds, action, state = self.made[-1]
        assert seconds > 0
        assert state[0], "the last timer was cancelled"
        action()


def _browser(
    tmp_path: Path, fake: FakeAgentBrowser, proxy: FakeProxy | None = None
) -> tuple[Browser, Timers, FakeProxy]:
    timers = Timers()
    egress = proxy or FakeProxy()
    made = Browser(
        "/usr/bin/agent-browser",
        tmp_path / "browser",
        idle_close_s=60,
        run=fake,
        proxy=lambda: egress,  # pyright: ignore[reportArgumentType] - a stand-in
        timer=timers,  # pyright: ignore[reportArgumentType] - a stand-in
    )
    return made, timers, egress


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


# --- how it is started ----------------------------------------------------------------------


def test_it_starts_headless_through_the_proxy_with_its_own_profile_and_policy(
    tmp_path: Path,
) -> None:
    fake = FakeAgentBrowser()
    made, _, proxy = _browser(tmp_path, fake)

    made.open("https://site.example/")

    argv, _commands, _env = fake.calls[0]
    assert argv[0] == "/usr/bin/agent-browser"
    assert _option(argv, "--headed") == "false"
    assert _option(argv, "--proxy") == f"http://127.0.0.1:{proxy.port}"
    assert _option(argv, "--proxy-bypass") == "<-loopback>"
    assert "--no-webmcp" in argv
    root = (tmp_path / "browser").resolve()
    for option in ("--config", "--profile", "--download-path", "--action-policy"):
        assert Path(_option(argv, option)).resolve().is_relative_to(root)
    assert json.loads(Path(_option(argv, "--config")).read_text(encoding="utf-8")) == {}
    policy = json.loads(Path(_option(argv, "--action-policy")).read_text(encoding="utf-8"))
    assert policy == {"default": "deny", "allow": list(ALLOWED_ACTIONS)}
    for banned in ("eval", "upload", "download", "cookies_set", "state_load", "tab_new", "back"):
        assert banned not in policy["allow"]
    chrome = _option(argv, "--args").split("\n")
    assert list(CHROME_ARGS) == chrome
    assert "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1" in chrome
    assert "--force-webrtc-ip-handling-policy=disable_non_proxied_udp" in chrome
    assert "--mute-audio" in chrome
    assert proxy.started


def test_its_environment_has_no_display_no_sound_and_no_outside_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "WAYLAND_DISPLAY": "wayland-1",
        "DISPLAY": ":0",
        "AGENT_BROWSER_HEADED": "1",
        "AGENT_BROWSER_PROVIDER": "browserbase",
        "AGENT_BROWSER_EXTENSIONS": "/x",
        "HTTPS_PROXY": "http://elsewhere:1",
        "AI_GATEWAY_API_KEY": "secret",
    }.items():
        monkeypatch.setenv(name, value)
    fake = FakeAgentBrowser()
    made, _, _ = _browser(tmp_path, fake)

    made.open("https://site.example/")

    env = fake.calls[0][2]
    assert set(env) <= {
        "PATH",
        "HOME",
        "XDG_RUNTIME_DIR",
        "LANG",
        "TZ",
        "PIPEWIRE_REMOTE",
        "PULSE_SERVER",
        "AGENT_BROWSER_NO_VERSION_CHECK",
        "AGENT_BROWSER_DEFAULT_TIMEOUT",
    }
    assert env["PIPEWIRE_REMOTE"] == "/nonexistent"
    assert env["PULSE_SERVER"] == "unix:/nonexistent"
    assert env["AGENT_BROWSER_NO_VERSION_CHECK"] == "1"


def test_every_call_of_a_session_passes_the_same_options(tmp_path: Path) -> None:
    fake = FakeAgentBrowser()
    made, _, _ = _browser(tmp_path, fake)

    made.open("https://site.example/")
    made.open("https://site.example/two")

    assert fake.calls[0][0] == fake.calls[1][0]


# --- what may be opened -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "chrome://settings",
        "javascript:alert(1)",
        "data:text/html,hi",
        "http://site.example:8080/",
        "https://user:pw@site.example/",
        "ftp://site.example/",
    ],
)
def test_only_plain_web_addresses_are_opened_and_nothing_runs_otherwise(
    tmp_path: Path, url: str
) -> None:
    fake = FakeAgentBrowser()
    made, _, _ = _browser(tmp_path, fake)

    with pytest.raises(BrowserError):
        made.open(url)

    assert fake.calls == []


def test_opening_returns_what_the_page_shows(tmp_path: Path) -> None:
    page = Page(
        text="Rendered by scripts.",
        refs={
            "e1": {"role": "searchbox", "name": "Search"},
            "e2": {"role": "link", "name": "  About \n us "},
        },
    )
    made, _, _ = _browser(tmp_path, FakeAgentBrowser(page))

    view = made.open("https://site.example/")

    assert view.url == "https://site.example/"
    assert view.title == "Site"
    assert view.text == "Rendered by scripts."
    assert view.elements == (
        Element("e1", "searchbox", "Search"),
        Element("e2", "link", "About us"),
    )
    assert made.view == view


def test_an_error_page_is_said_as_unreachable(tmp_path: Path) -> None:
    fake = FakeAgentBrowser(Page())
    fake.on["open"] = lambda _c: setattr(fake.page, "url", "chrome-error://chromewebdata/")
    made, _, _ = _browser(tmp_path, fake)

    with pytest.raises(BrowserError, match="can't be reached"):
        made.open("https://site.example/")
    assert made.view is None


def test_a_page_the_proxy_refused_is_not_passed_off_as_its_text(tmp_path: Path) -> None:
    fake = FakeAgentBrowser(Page(text="'10.0.0.1' is not a public internet address"))
    made, _, _ = _browser(tmp_path, fake, FakeProxy(refused=("10.0.0.1:80",)))

    with pytest.raises(BrowserError, match="not a public web address"):
        made.open("http://10.0.0.1/")


def test_a_failed_command_is_an_error_without_the_page_words(tmp_path: Path) -> None:
    fake = FakeAgentBrowser()
    fake.fail = "open"
    made, _, _ = _browser(tmp_path, fake)

    with pytest.raises(BrowserError, match="couldn't open that page"):
        made.open("https://site.example/")


def test_an_answer_that_is_not_rows_is_an_error(tmp_path: Path) -> None:
    def garbled(*_args: object, **_kwargs: object) -> str:
        return "not json"

    made, _, _ = _browser(tmp_path, garbled)  # pyright: ignore[reportArgumentType] - a stand-in

    with pytest.raises(BrowserError):
        made.open("https://site.example/")


# --- the session's life -----------------------------------------------------------------------


def test_it_closes_when_idle_and_leaves_nothing_behind(tmp_path: Path) -> None:
    fake = FakeAgentBrowser()
    made, timers, proxy = _browser(tmp_path, fake)
    made.open("https://site.example/")
    folders = list((tmp_path / "browser").iterdir())
    assert len(folders) == 1

    timers.fire()

    assert fake.calls[-1][0][-1] == "close"
    assert proxy.stopped
    assert not made.running
    assert made.view is None
    assert list((tmp_path / "browser").iterdir()) == []


def test_each_use_puts_the_idle_close_off(tmp_path: Path) -> None:
    fake = FakeAgentBrowser()
    made, timers, _ = _browser(tmp_path, fake)
    made.open("https://site.example/")
    first = timers.made[-1][2]

    made.open("https://site.example/two")

    assert not first[0]  # cancelled
    assert timers.made[-1][2][0]


def test_a_timeout_closes_the_session(tmp_path: Path) -> None:
    fake = FakeAgentBrowser()
    fake.timeout = True
    made, _, proxy = _browser(tmp_path, fake)

    with pytest.raises(BrowserTimeoutError):
        made.open("https://site.example/")

    assert proxy.stopped
    assert not made.running


def test_closing_twice_or_before_use_does_nothing(tmp_path: Path) -> None:
    fake = FakeAgentBrowser()
    made, _, _ = _browser(tmp_path, fake)

    made.close()
    made.close()

    assert fake.calls == []


def test_folders_a_gone_jarvis_left_are_swept_and_living_ones_kept(tmp_path: Path) -> None:
    root = tmp_path / "browser"
    root.mkdir()
    gone = root / "session-999999999-abc"
    gone.mkdir()
    alive = root / f"session-{os.getppid()}-abc"
    alive.mkdir()
    made, _, _ = _browser(tmp_path, FakeAgentBrowser())

    made.open("https://site.example/")

    assert not gone.exists()
    assert alive.exists()


# --- links, clicks and typing -------------------------------------------------------------------


def test_a_link_resolves_against_the_page(tmp_path: Path) -> None:
    page = Page(url="https://site.example/a/", attrs={("e1", "href"): "../b?c=1"})
    made, _, _ = _browser(tmp_path, FakeAgentBrowser(page))
    made.open("https://site.example/a/")

    assert made.link("e1") == "https://site.example/b?c=1"


@pytest.mark.parametrize("href", ["javascript:steal()", "file:///etc/passwd", "mailto:a@b.c", ""])
def test_a_link_to_anything_but_a_web_page_is_refused(tmp_path: Path, href: str) -> None:
    page = Page(attrs={("e1", "href"): href})
    made, _, _ = _browser(tmp_path, FakeAgentBrowser(page))
    made.open("https://site.example/")

    with pytest.raises(BrowserError):
        made.link("e1")


def test_only_a_link_is_followed(tmp_path: Path) -> None:
    page = Page(refs={"e1": {"role": "button", "name": "Buy"}})
    fake = FakeAgentBrowser(page)
    made, _, _ = _browser(tmp_path, fake)
    made.open("https://site.example/")

    with pytest.raises(BrowserError):
        made.link("e1")
    with pytest.raises(BrowserError):
        made.link("e9")


def _set(page: Page, name: str, value: object) -> None:
    setattr(page, name, value)


BUTTON = Element("e1", "button", "Search")
BOX = Element("e2", "searchbox", "Search")
PAGE_REFS = {
    "e1": {"role": "button", "name": "Search"},
    "e2": {"role": "searchbox", "name": "Search"},
}


def test_a_click_happens_only_when_the_page_still_shows_that_element(tmp_path: Path) -> None:
    fake = FakeAgentBrowser(Page(refs=dict(PAGE_REFS)))
    made, _, _ = _browser(tmp_path, fake)
    made.open("https://site.example/")

    made.click(BUTTON, "https://site.example/")

    assert ["click", "@e1"] in fake.commands()


@pytest.mark.parametrize(
    "change",
    [
        ("refs", {"e1": {"role": "button", "name": "Buy now"}}),
        ("refs", {}),
        ("url", "https://elsewhere.example/"),
    ],
    ids=["renamed", "gone", "moved"],
)
def test_a_click_on_a_page_that_changed_since_the_question_does_nothing(
    tmp_path: Path, change: tuple[str, object]
) -> None:
    fake = FakeAgentBrowser(Page(refs=dict(PAGE_REFS)))
    made, _, _ = _browser(tmp_path, fake)
    made.open("https://site.example/")
    _set(fake.page, *change)

    with pytest.raises(BrowserError, match="changed"):
        made.click(BUTTON, "https://site.example/")

    assert not any(c[0] == "click" for c in fake.commands())


def test_typing_fills_the_field_with_the_words_as_data_then_presses_enter(tmp_path: Path) -> None:
    fake = FakeAgentBrowser(Page(refs=dict(PAGE_REFS)))
    made, _, _ = _browser(tmp_path, fake)
    made.open("https://site.example/")

    made.fill(BOX, "https://site.example/", "--json cats", submit=True)

    commands = fake.commands()
    assert ["fill", "@e2", "--", "--json cats"] in commands
    assert commands.index(["press", "Enter"]) > commands.index(["fill", "@e2", "--", "--json cats"])


def test_typing_without_submit_presses_nothing(tmp_path: Path) -> None:
    fake = FakeAgentBrowser(Page(refs=dict(PAGE_REFS)))
    made, _, _ = _browser(tmp_path, fake)
    made.open("https://site.example/")

    made.fill(BOX, "https://site.example/", "cats", submit=False)

    assert not any(c[0] == "press" for c in fake.commands())


@pytest.mark.parametrize("kind", ["password", "PASSWORD", "hidden", "file"])
def test_nothing_is_typed_into_a_password_field(tmp_path: Path, kind: str) -> None:
    page = Page(refs=dict(PAGE_REFS), attrs={("e2", "type"): kind})
    fake = FakeAgentBrowser(page)
    made, _, _ = _browser(tmp_path, fake)
    made.open("https://site.example/")

    with pytest.raises(BrowserError, match="password"):
        made.fill(BOX, "https://site.example/", "hunter2", submit=True)

    assert not any(c[0] == "fill" for c in fake.commands())


def test_nothing_is_typed_into_a_button(tmp_path: Path) -> None:
    fake = FakeAgentBrowser(Page(refs=dict(PAGE_REFS)))
    made, _, _ = _browser(tmp_path, fake)
    made.open("https://site.example/")

    with pytest.raises(BrowserError):
        made.fill(BUTTON, "https://site.example/", "cats", submit=True)

    assert not any(c[0] == "fill" for c in fake.commands())


# --- pictures -----------------------------------------------------------------------------------


def test_a_screenshot_of_a_lower_part_scrolls_there_first(tmp_path: Path) -> None:
    fake = FakeAgentBrowser()
    made, _, _ = _browser(tmp_path, fake)
    made.open("https://site.example/")

    data = made.screenshot(3)

    assert data == b"\x89PNG fake"
    commands = fake.commands()
    assert ["scroll", "down", str(2 * browser_module.SCREEN_STEP_PX)] in commands
    shot = next(c for c in commands if c[0] == "screenshot")
    assert not Path(shot[1]).exists()  # read, then deleted


def test_no_screenshot_without_a_page(tmp_path: Path) -> None:
    fake = FakeAgentBrowser()
    made, _, _ = _browser(tmp_path, fake)

    with pytest.raises(BrowserError):
        made.screenshot(1)
    assert fake.calls == []


@pytest.mark.parametrize("after", ["wait", "snapshot"])
def test_a_click_that_happened_but_the_page_then_failed_is_uncertain(
    tmp_path: Path, after: str
) -> None:
    # The click went through: saying it failed would invite doing it again.
    fake = FakeAgentBrowser(Page(refs=dict(PAGE_REFS)))
    made, _, _ = _browser(tmp_path, fake)
    made.open("https://site.example/")
    fake.fail = after
    fake.fail_after = "click"

    with pytest.raises(BrowserTimeoutError):
        made.click(BUTTON, "https://site.example/")

    assert ["click", "@e1"] in fake.commands()


def test_typing_sent_then_an_error_page_is_uncertain(tmp_path: Path) -> None:
    fake = FakeAgentBrowser(Page(refs=dict(PAGE_REFS)))
    made, _, _ = _browser(tmp_path, fake)
    made.open("https://site.example/")
    fake.on["press"] = lambda _c: setattr(fake.page, "url", "chrome-error://chromewebdata/")

    with pytest.raises(BrowserTimeoutError):
        made.fill(BOX, "https://site.example/", "cats", submit=True)


def test_a_click_refused_before_it_happened_is_a_plain_failure(tmp_path: Path) -> None:
    fake = FakeAgentBrowser(Page(refs=dict(PAGE_REFS)))
    made, _, _ = _browser(tmp_path, fake)
    made.open("https://site.example/")
    fake.fail = "click"

    with pytest.raises(BrowserError) as raised:
        made.click(BUTTON, "https://site.example/")

    assert not isinstance(raised.value, BrowserTimeoutError)
