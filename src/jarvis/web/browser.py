"""A headless browser for pages that need one, driven through the `agent-browser` program.

`web_read` reads what a server sends; some pages build their text with JavaScript, or show it only
after a search box or a button. For those, Jarvis drives a real Chrome through agent-browser, with
these fences, each in code:

- Headless, muted and with no display variables: it can open no window and make no sound.
- Its own throwaway profile and home folder in the user's runtime folder, deleted once the
  browser has ended: never the owner's browser, cookies, logins or Downloads. Its download folder
  is read-only, and so is its home's Downloads, so a download a page starts fails instead of
  being saved. Each launch has its own agent-browser session and socket folder.
- Every connection it makes goes through `jarvis.web.proxy`, which reaches only public addresses
  on ports 80 and 443; loopback goes through the proxy too, and Chrome resolves no name itself,
  so page scripts, frames, redirects and WebRTC cannot reach this PC or the home network.
- agent-browser's own config, plugins and settings files are not read (`--config` points at an
  empty file of Jarvis's), and its action policy allows only the commands this module sends: no
  scripts, uploads, downloads, cookies, tabs or clipboard.
- It closes after `idle_close_s` without use (and when the conversation is forgotten), so it uses
  no CPU or memory between requests; agent-browser's own idle timeout ends it too if Jarvis dies.
"""

import atexit
import contextlib
import json
import os
import re
import secrets
import selectors
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlsplit

from jarvis.operations import current_request
from jarvis.web.page import Hidden, PageError, check_url, hidden_parts, without_hidden
from jarvis.web.proxy import EgressProxy

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

PROGRAM = "agent-browser"
# One batch of commands, launching the browser included, at most; less near the request's end.
BATCH_LIMIT_S = 40.0
ACTION_TIMEOUT_MS = 15_000  # agent-browser's own limit for one command
CLOSE_LIMIT_S = 15.0
# Each batch ends this long before the request's own time limit: room to close a browser that
# did not answer, and for the model to say what happened.
RESERVE_S = CLOSE_LIMIT_S + 10.0
MIN_BATCH_S = 3.0  # less time than this left in the request, and the browser is not used
SETTLE_MS = 1200  # after a page loads or a click: scripts that fill the page in afterwards
MAX_OUTPUT = 60_000  # characters of a page's text kept (agent-browser's --max-output cuts no JSON)
MAX_STDOUT = 8_000_000  # bytes of one answer read at all (a page's text and its HTML)
MAX_NAME = 80  # characters of an element's name kept
MAX_SCREENSHOT_BYTES = 8_000_000
SCREEN_STEP_PX = 640  # how far one screenshot part is below the last (the window is 1280x720)
MAX_PARTS = 20
# The roles text may be typed into; anything else (a button, a link, a password) is refused.
TYPABLE = frozenset({"textbox", "searchbox", "combobox"})
# Input types never typed into; a password field's role is a plain `textbox`.
UNTYPABLE = frozenset({"password", "hidden", "file"})
# A link's line in `snapshot -i -u`'s text: its name (a JSON string, so nothing the page wrote can
# end it), then `[…ref=eN, url=…]`. The address is the browser's own (`this.href`), which has no
# whitespace when it is http or https; the `]` after it closes the brackets.
_LINK_LINE = re.compile(
    r'\s*- link(?: "(?:[^"\\]|\\.)*")? \[(?:[a-z]+(?:=[^,\]]*)?, )*ref=(e\d+), url=(\S*)'
)
# agent-browser's action names: only the commands this module sends, nothing else.
ALLOWED_ACTIONS = (
    "navigate",
    "url",
    "title",
    "gettext",
    "innerhtml",
    "getattribute",
    "snapshot",
    "screenshot",
    "click",
    "fill",
    "focus",
    "press",
    "scroll",
    "wait",
    "close",
)
CHROME_ARGS = (
    # Resolve no name in Chrome: every address goes to the proxy as a name, checked there.
    "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1",
    # WebRTC may not send UDP past the proxy, and so gathers no local address either.
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--disable-quic",
    "--mute-audio",
    "--deny-permission-prompts",
    "--disable-notifications",
    "--disable-extensions",
    "--no-pings",
    "--disable-client-side-phishing-detection",
    "--disable-domain-reliability",
    # No update, safe-browsing or other checks of Chrome's own: they would only be refused.
    "--disable-background-networking",
    "--disk-cache-size=33554432",  # the profile is in the runtime folder, which is memory
)
# What the browser's environment keeps of Jarvis's: where programs are, nothing else. Its home is
# the session's own (`_environment`).
_KEPT_ENVIRONMENT = ("PATH", "XDG_RUNTIME_DIR", "LANG", "TZ")
# The session folders in use in this process, so one Browser's sweep never takes another's.
_IN_USE: set[Path] = set()
_IN_USE_LOCK = threading.Lock()
_CHROME_ERROR = "chrome-error:"


class BrowserError(Exception):
    """The browser could not do what was asked; the message says why in plain words."""


class BrowserTimeoutError(BrowserError):
    """The browser did not answer in time: what was sent to it may or may not have happened."""


class BrowserAnswerError(BrowserError):
    """The browser answered, but its answer could not be read: what was sent to it ran."""


class UncertainActionError(BrowserError):
    """A click or typing was sent, and whether it happened is not known: never retry it blindly."""


@dataclass(frozen=True, slots=True)
class Element:
    """Something on a page that can be used: its reference (`e3`), role and visible name."""

    ref: str
    role: str
    name: str


@dataclass(frozen=True, slots=True)
class View:
    """What the page shows now: its address, title, rendered text and the elements it offers."""

    url: str
    title: str
    text: str
    elements: tuple[Element, ...]
    links: tuple[tuple[str, str], ...] = ()  # each link's ref and where the browser has it go

    def goes_to(self, ref: str) -> str:
        """Return the full address a link goes to, as the browser resolved it ("" for none)."""
        return next((url for found, url in self.links if found == ref), "")

    @property
    def host(self) -> str:
        """The page's host name."""
        return urlsplit(self.url).hostname or ""

    def element(self, ref: str) -> Element | None:
        """Return the element with this reference, or None."""
        return next((e for e in self.elements if e.ref == ref), None)


class Runner(Protocol):
    """Runs agent-browser once; a fake in tests."""

    def __call__(
        self, argv: Sequence[str], *, stdin: str, env: Mapping[str, str], cwd: Path, timeout: float
    ) -> str:
        """Return what the program wrote, or raise `BrowserTimeoutError` or `OSError`."""
        ...


def run_program(
    argv: Sequence[str], *, stdin: str, env: Mapping[str, str], cwd: Path, timeout: float
) -> str:
    """Run agent-browser and return its output (its exit status is in the JSON it prints).

    The output is read as it comes, at most `MAX_STDOUT` bytes of it; what the program writes
    to its error stream is dropped. A `BrowserTimeoutError` says it did not finish in `timeout`
    seconds, and a `BrowserAnswerError` that it wrote more than `MAX_STDOUT`; it is stopped then.

    Raises:
        OSError: it could not be started, or its pipes could not be made.
    """
    import subprocess  # ruff: ignore[import-outside-top-level] - only a browser in use starts one, and a typed request may never need it

    deadline = time.monotonic() + timeout
    with subprocess.Popen(
        list(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=dict(env),
        cwd=cwd,
    ) as process:
        try:
            if process.stdin is None or process.stdout is None:  # both are piped
                msg = "the browser's pipes are missing"
                raise OSError(msg)
            return _read_capped(process.stdin, process.stdout, stdin.encode(), deadline)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def _read_capped(sink: IO[bytes], source: IO[bytes], stdin: bytes, deadline: float) -> str:
    """Send `stdin` to `sink`, then read `source` until it closes, within the caps.

    Raises:
        BrowserTimeoutError: past `deadline`.
        BrowserAnswerError: more than `MAX_STDOUT` bytes.
    """
    with contextlib.suppress(BrokenPipeError):
        sink.write(stdin)
    with contextlib.suppress(BrokenPipeError):
        sink.close()
    out = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(source, selectors.EVENT_READ)
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not selector.select(left):
                msg = "the browser took too long"
                raise BrowserTimeoutError(msg)
            chunk = os.read(source.fileno(), 1 << 16)
            if not chunk:
                return out.decode("utf-8", errors="replace")
            out += chunk
            if len(out) > MAX_STDOUT:
                msg = "the page is too big to read"
                raise BrowserAnswerError(msg)


def _one_line(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def _elements(snapshot: Mapping[str, Any]) -> tuple[Element, ...]:
    """Return the elements of a `snapshot -i` result, in page order."""
    refs = snapshot.get("refs")
    if not isinstance(refs, dict):
        return ()
    found: list[Element] = []
    for ref, info in cast("dict[str, object]", refs).items():
        if not isinstance(info, dict):
            continue
        fields = cast("dict[str, object]", info)
        role, name = fields.get("role"), fields.get("name")
        if isinstance(role, str) and role:
            found.append(Element(ref, _one_line(role, 40), _one_line(str(name or ""), MAX_NAME)))
    return tuple(found)


def _shown(snapshot: Mapping[str, Any], hidden: Hidden) -> tuple[Element, ...]:
    """Return the elements a snapshot row lists, less those named by text the page hides.

    The accessibility tree keeps what is only made transparent or of no size. When the page
    hides too many elements to tell (`Hidden.names` is None), none is offered.
    """
    result = snapshot.get("result")
    if hidden.names is None or not isinstance(result, dict):
        return ()
    names = hidden.names
    found = _elements(cast("dict[str, Any]", result))
    return tuple(element for element in found if element.name not in names)


def _field(row: Mapping[str, Any], key: str) -> str:
    result = row.get("result")
    value = cast("dict[str, object]", result).get(key) if isinstance(result, dict) else None
    return value if isinstance(value, str) else ""


class Browser:
    """One headless browser session, started on first use and closed when idle."""

    def __init__(  # ruff: ignore[too-many-arguments] - the program, its folder, the idle time, and fakes for tests
        self,
        program: str,
        root: Path,
        *,
        idle_close_s: float,
        run: Runner = run_program,
        proxy: Callable[[], EgressProxy] = EgressProxy,
        timer: Callable[[float, Callable[[], None]], threading.Timer] | None = None,
        origin: str = "",
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Drive `program` (agent-browser's full path), keeping its throwaway files under `root`.

        With `origin` (`http://127.0.0.1:<port>`, a preview's), only addresses there are opened.
        `on_close` runs once `close` has closed the browser (a preview's server stops then).
        """
        super().__init__()
        self._origin = origin
        self._program = program
        self._root = root
        self._idle_s = idle_close_s
        self._run = run
        self._make_proxy = proxy
        self._make_timer = timer or _daemon_timer
        self._on_close = on_close
        self._session = ""  # each launch's own, set with the folder
        self._chrome = ""
        self._lock = threading.RLock()
        self._dir: Path | None = None
        self._proxy: EgressProxy | None = None
        self._timer: threading.Timer | None = None
        self._uses = 0  # counts the idle timers set, so only the latest one closes anything
        self._view: View | None = None

    @property
    def view(self) -> View | None:
        """What the last command left on the page, or None when nothing is open."""
        with self._lock:
            return self._view

    @property
    def running(self) -> bool:
        """Whether a browser session is open now."""
        with self._lock:
            return self._dir is not None

    # --- the session -------------------------------------------------------------------------

    def _start(self) -> Path:
        """Make the throwaway folder and the proxy for a new session, if none is open.

        Raises:
            BrowserError: the folder could not be made or the proxy could not start (the folder
                is then deleted again).
        """
        if self._dir is not None:
            return self._dir
        try:
            folder = _new_folder(self._root)
        except OSError as exc:
            msg = f"the browser's folder couldn't be made: {exc.strerror or type(exc).__name__}"
            raise BrowserError(msg) from exc
        proxy = self._make_proxy()
        try:
            proxy.start()
        except OSError as exc:
            _release(folder, remove=True)
            msg = f"the browser's proxy couldn't start: {exc.strerror or type(exc).__name__}"
            raise BrowserError(msg) from exc
        self._dir, self._proxy = folder, proxy
        self._session = f"jarvis-{secrets.token_hex(4)}"
        # The real home's: agent-browser no longer finds its own Chrome under the session's home.
        found = chrome_path(Path.home())
        self._chrome = str(found) if found is not None else ""
        atexit.register(self._at_exit)
        return folder

    def _flags(self, folder: Path) -> list[str]:
        """Return agent-browser's options: the same on every call, so it is never relaunched.

        The proxy, with its credential, is in the environment (`_environment`), not here: a
        program's arguments are readable by every user of this PC.
        """
        idle = int(self._idle_s) + 60  # agent-browser's own backstop, after Jarvis's
        chrome = ["--executable-path", self._chrome] if self._chrome else []
        return [
            self._program,
            "--config",
            str(folder / "config.json"),
            "--session",
            self._session,
            *chrome,
            "--json",
            "--headed",
            "false",
            "--engine",
            "chrome",
            "--profile",
            str(folder / "profile"),
            "--download-path",
            str(folder / "downloads"),
            "--action-policy",
            str(folder / "policy.json"),
            "--proxy-bypass",
            "<-loopback>",
            "--args",
            "\n".join(CHROME_ARGS),
            "--no-webmcp",
            "--idle-timeout",
            f"{idle}s",
        ]

    @staticmethod
    def _environment(folder: Path, proxy: EgressProxy) -> dict[str, str]:
        """Return the browser's environment: no display, no sound server, no outside settings.

        Its home and XDG folders are the session's (`folder / "work"`), so nothing Chrome or
        agent-browser keeps or downloads by default lands in the owner's. The proxy comes with
        its credential, which agent-browser answers the proxy's `407` with.
        """
        env = {name: os.environ[name] for name in _KEPT_ENVIRONMENT if name in os.environ}
        home = folder / "work"
        env.update(
            {
                "HOME": str(home),
                "XDG_CONFIG_HOME": str(home / ".config"),
                "XDG_CACHE_HOME": str(home / ".cache"),
                "XDG_DATA_HOME": str(home / ".local" / "share"),
                "XDG_STATE_HOME": str(home / ".local" / "state"),
                "XDG_DOWNLOAD_DIR": str(folder / "downloads"),
                "AGENT_BROWSER_SOCKET_DIR": str(folder / "s"),
                "AGENT_BROWSER_PROXY": proxy.url(proxy.port),
                "PIPEWIRE_REMOTE": "/nonexistent",
                "PULSE_SERVER": "unix:/nonexistent",
                # agent-browser would ask a blog for Chrome's security news, past the proxy.
                "AGENT_BROWSER_NO_VERSION_CHECK": "1",
                "AGENT_BROWSER_DEFAULT_TIMEOUT": str(ACTION_TIMEOUT_MS),
            }
        )
        return env

    def _batch(self, commands: list[list[str]]) -> list[dict[str, Any]]:
        """Run commands in one agent-browser call, stopping at the first that fails.

        It gets `BATCH_LIMIT_S`, or less when the request has less time left (`RESERVE_S` kept).
        Returns one row per command that ran: `success`, `result` and `error`.

        Raises:
            BrowserTimeoutError: it did not finish in time (the browser is then closed).
            BrowserAnswerError: its answer is too big to read, or not what agent-browser writes.
            BrowserError: the request has no time left for it, or it could not start (nothing
                was sent either way).
        """
        limit = _time_for_batch()
        if limit < MIN_BATCH_S:
            msg = "there's no time left in this request to use the browser"
            raise BrowserError(msg)
        folder = self._start()
        proxy = self._proxy
        if proxy is None:  # set with the folder
            msg = "the browser's proxy is not running"
            raise BrowserError(msg)
        argv = [*self._flags(folder), "batch", "--bail"]
        try:
            out = self._run(
                argv,
                stdin=json.dumps(commands),
                env=self._environment(folder, proxy),
                cwd=folder / "work",
                timeout=limit,
            )
        except BrowserTimeoutError, BrowserAnswerError:
            self._shut()
            raise
        except OSError as exc:
            self._shut()
            msg = f"the browser couldn't be started: {exc.strerror or type(exc).__name__}"
            raise BrowserError(msg) from exc
        if len(out) > MAX_STDOUT:  # a runner that read it whole
            self._shut()
            msg = "the page is too big to read"
            raise BrowserAnswerError(msg)
        return _rows(out)

    def _idle(self) -> None:
        """Close the session after `idle_close_s` without another command, if one is open."""
        if self._dir is None:
            return
        if self._timer is not None:
            self._timer.cancel()
        self._uses += 1
        use = self._uses
        self._timer = self._make_timer(self._idle_s, lambda: self._expire(use))
        self._timer.start()

    def _expire(self, use: int) -> None:
        """Close the session for the idle timer set at `use`, unless a later command reset it.

        A timer that fired while a command held the browser waits for it, then finds that the
        command set a new timer: it closes nothing.
        """
        with self._lock:
            if use == self._uses:
                self._shut()

    def close(self) -> None:
        """Close the browser, stop its proxy and delete its folder (its profile and downloads).

        Then `on_close` runs, if given.
        """
        with self._lock:
            self._shut()
        if self._on_close is not None:
            self._on_close()

    def _at_exit(self) -> None:
        """Close as Jarvis exits, waiting at most `CLOSE_LIMIT_S` for a command still running.

        Past that, only the proxy is stopped, so the browser has no way out until agent-browser's
        own idle timeout ends it; its folder is swept at a later start.
        """
        if self._lock.acquire(timeout=CLOSE_LIMIT_S):
            try:
                self._shut()
            finally:
                self._lock.release()
        else:
            proxy = self._proxy
            if proxy is not None:
                proxy.stop()
        if self._on_close is not None:
            self._on_close()

    def _shut(self) -> None:
        """Close the session: the browser, then its proxy, then its folder once the browser ended.

        A browser whose close did not end it (the close failed or timed out) keeps its folder,
        with its socket folder and agent-browser's record of it, until a later sweep finds it
        gone: the next launch is a session of its own either way.
        """
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        folder, proxy = self._dir, self._proxy
        self._dir, self._proxy, self._view = None, None, None
        if folder is None:
            return
        atexit.unregister(self._at_exit)
        if proxy is not None:
            with contextlib.suppress(BrowserError, OSError):
                self._run(
                    [*self._flags(folder), "close"],
                    stdin="",
                    env=self._environment(folder, proxy),
                    cwd=folder / "work",
                    timeout=CLOSE_LIMIT_S,
                )
            proxy.stop()
        _release(folder, remove=not _daemon_running(folder))

    # --- what the page shows -----------------------------------------------------------------

    @staticmethod
    def _done(rows: list[dict[str, Any]], count: int, what: str) -> None:
        """Check that all `count` commands of a batch succeeded.

        Raises:
            BrowserError: one failed; the message names `what` failed, without the page's words.
        """
        if len(rows) == count and all(row.get("success") is True for row in rows):
            return
        msg = f"the browser couldn't {what}"
        raise BrowserError(msg)

    def _read(self, rows: list[dict[str, Any]]) -> View:
        """Make the view from the rows of `_LOOK`: url, title, text, HTML and snapshot.

        The text leaves out what the page hides, by the rules `web_read` uses (`without_hidden`),
        and so do the elements: none named by text inside a hidden element is offered.

        Raises:
            BrowserError: the page is an error page (the site could not be reached).
        """
        url, title, text, html, snapshot = rows[-len(self._LOOK) :]
        address = _field(url, "url")
        if address.startswith(_CHROME_ERROR) or not address:
            msg = "that site can't be reached from here"
            raise BrowserError(msg)
        proxy = self._proxy
        parts = urlsplit(address)
        port = _port(parts.port, parts.scheme)
        if proxy is not None and proxy.was_refused(parts.hostname or "", port):
            msg = "that address is not a public web address, so it isn't opened"
            raise BrowserError(msg)
        hidden = hidden_parts(_field(html, "html"), MAX_NAME)
        page_text = without_hidden(_field(text, "text"), hidden)[:MAX_OUTPUT]
        view = View(
            address,
            _one_line(_field(title, "title"), 160),
            page_text,
            _shown(snapshot, hidden),
            _links(snapshot),
        )
        self._view = view
        return view

    _LOOK = (
        ["get", "url"],
        ["get", "title"],
        ["get", "text", "body"],
        ["get", "html", "body"],
        ["snapshot", "-i", "-u"],
    )

    def _act(self, commands: list[list[str]], what: str, *, acting: bool = False) -> View:
        """Run commands, let the page settle, and return what it shows then.

        Raises:
            UncertainActionError: `acting` (a click, typing), and the commands were sent but the
                browser did not answer, its answer could not be read, or one of them went
                through and a later step or the page failed: what was sent may have happened, so
                it must not be retried as if it had not.
            BrowserTimeoutError: not `acting`, and the browser did not answer.
            BrowserAnswerError: not `acting`, and its answer could not be read.
            BrowserError: the commands failed before any went through, or the page could not be
                read.
        """
        batch = [*commands, ["wait", str(SETTLE_MS)], *(list(c) for c in self._LOOK)]
        try:
            try:
                rows = self._batch(batch)
            except (BrowserTimeoutError, BrowserAnswerError) as exc:
                if acting:
                    msg = f"it was sent, but the browser's answer couldn't be read ({exc})"
                    raise UncertainActionError(msg) from exc
                raise
            acted = any(row.get("success") is True for row in rows[: len(commands)])
            try:
                self._done(rows, len(batch), what)
                return self._read(rows)
            except BrowserError as exc:
                if acting and acted:
                    msg = f"it may have gone through, but the page couldn't be read after ({exc})"
                    raise UncertainActionError(msg) from exc
                raise
        finally:
            self._idle()

    def open(self, url: str) -> View:
        """Open a plain http or https address (a preview's own) and return what the page shows.

        A `BrowserError` says the address is not one this session opens, or the page could not be
        opened.
        """
        address = self._checked(url)
        with self._lock:
            return self._act([["open", address]], "open that page")

    def _checked(self, url: str) -> str:
        """Return `url` when this session may open it: a plain web address, or the preview's.

        Raises:
            BrowserError: it may not be opened here.
        """
        if self._origin:
            parts = urlsplit(url)
            if parts.scheme != "http" or f"http://{parts.netloc}" != self._origin:
                msg = "a preview opens only the project it serves"
                raise BrowserError(msg)
            return url
        try:
            return check_url(url)
        except PageError as exc:
            raise BrowserError(str(exc)) from exc

    def href(self, ref: str) -> str:
        """Return where a link on the current page goes, in full, unchecked ("" for nowhere).

        It is the browser's own address for it (resolved against the page's base address, a
        `<base href>` included, or its frame's), from the last look at the page.

        Raises:
            BrowserError: no page is open, or the element is not a link.
        """
        with self._lock:
            view = self._view
            found = view.element(ref) if view is not None else None
            if view is None or found is None or found.role != "link":
                msg = "that is not a link on the page now; look at the page again"
                raise BrowserError(msg)
            return view.goes_to(ref)

    def input_type(self, ref: str) -> str:
        """Return a field's input type (`password`, `text`, "" for none), lower case.

        Raises:
            BrowserError: no page is open, or the browser could not read it.
        """
        with self._lock:
            self._still_open()
            try:
                rows = self._batch([["get", "attr", f"@{ref}", "type"]])
            except BrowserTimeoutError as exc:
                msg = "the browser didn't answer"
                raise BrowserError(msg) from exc
            finally:
                self._idle()
            self._done(rows, 1, "read that field")
            return _field(rows[0], "value").strip().lower()

    def link(self, ref: str) -> str:
        """Return the full address a link on the current page points to, to open it.

        Raises:
            BrowserError: no page is open, the element is not a link, or it points nowhere Jarvis
                opens (another scheme, a script).
        """
        address = self.href(ref)
        if not address:
            msg = "that link points nowhere"
            raise BrowserError(msg)
        return self._checked(address)

    def _still_open(self) -> None:
        """Check that the page a yes was given for is still open.

        Raises:
            BrowserError: the browser closed meanwhile (idle while the owner was asked, or the
                conversation was forgotten); a new browser would show another page.
        """
        if self._view is None:
            msg = "the browser closed while waiting for the yes; open the page again"
            raise BrowserError(msg)

    def _same(
        self, expect: Element, url: str, also: Sequence[list[str]] = ()
    ) -> tuple[View, list[dict[str, Any]]]:
        """Check that the page is still at `url` and shows `expect` under its reference.

        agent-browser keeps a ref for one element of one document: an element the page replaced
        has a new ref, and the old one is gone. `also` runs in the same call, after the page is
        looked at; the view and their rows are returned. It only reads the page: nothing has
        been done when it fails, a timeout included.

        Raises:
            BrowserError: the page moved on or the element changed, so the yes no longer fits, or
                the browser did not answer (then it is closed).
        """
        look = len(self._LOOK)
        try:
            rows = self._batch([*(list(c) for c in self._LOOK), *also])
        except BrowserTimeoutError as exc:
            msg = "the browser didn't answer before acting, so nothing was done"
            raise BrowserError(msg) from exc
        self._done(rows, look + len(also), "look at the page")
        view = self._read(rows[:look])
        if view.url != url or view.element(expect.ref) != expect:
            msg = "the page changed since; look at it again before acting"
            raise BrowserError(msg)
        return view, rows[look:]

    def click(self, expect: Element, url: str, href: str | None = None) -> View:
        """Click an element the owner agreed to, if the page still shows it, and return the page.

        With `href` (a link's, from `href`), the link must still point there.

        An `UncertainActionError` says the click was sent and may have happened.

        Raises:
            BrowserError: the page changed, the link points elsewhere now, or the click failed.
        """
        with self._lock:
            self._still_open()
            try:
                view, _rows = self._same(expect, url)
            finally:
                self._idle()
            if href is not None and view.goes_to(expect.ref) != href:
                msg = "that link points somewhere else now; look at the page again"
                raise BrowserError(msg)
            return self._act([["click", f"@{expect.ref}"]], "click that", acting=True)

    def fill(self, expect: Element, url: str, text: str, *, submit: bool) -> View:
        """Type `text` into a field the owner agreed to, then press Enter if `submit`.

        An `UncertainActionError` says the text was sent and may have reached the page.

        Raises:
            BrowserError: the page changed, the element takes no text, or typing failed.
        """
        if expect.role not in TYPABLE:
            msg = "that element takes no typing"  # its name is the page's words, not said here
            raise BrowserError(msg)
        with self._lock:
            self._still_open()
            try:
                _view, rows = self._same(expect, url, [["get", "attr", f"@{expect.ref}", "type"]])
            finally:
                self._idle()
            if _field(rows[0], "value").strip().lower() in UNTYPABLE:
                msg = "Jarvis doesn't type into password fields"
                raise BrowserError(msg)
            commands = [["fill", f"@{expect.ref}", "--", text]]
            if submit:
                # Enter goes to whatever has focus, which the page may have moved as it was typed
                # into: the field is focused again first, in the same call.
                commands += [["focus", f"@{expect.ref}"], ["press", "Enter"]]
            return self._act(commands, "type there", acting=True)

    def screenshot(self, part: int) -> bytes:
        """Return a PNG of one screen of the current page: part 1 is its top.

        Raises:
            BrowserError: no page is open, or the picture could not be taken or read.
        """
        with self._lock:
            if self._view is None or self._dir is None:
                msg = "no page is open in the browser"
                raise BrowserError(msg)
            path = self._dir / "work" / "screen.png"
            path.unlink(missing_ok=True)
            commands = [
                ["scroll", "up", "1000000"],
                *([["scroll", "down", str((part - 1) * SCREEN_STEP_PX)]] if part > 1 else []),
                ["screenshot", str(path)],
            ]
            try:
                rows = self._batch(commands)
                self._done(rows, len(commands), "take that picture")
            finally:
                self._idle()
            try:
                if path.stat().st_size > MAX_SCREENSHOT_BYTES:
                    msg = "the picture of the page is too big"
                    raise BrowserError(msg)
                data = path.read_bytes()
            except OSError as exc:
                msg = "the picture of the page couldn't be read"
                raise BrowserError(msg) from exc
            finally:
                path.unlink(missing_ok=True)
            return data


def _links(snapshot: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return each link's ref and address from a `snapshot -i -u` row's text."""
    result = snapshot.get("result")
    tree = cast("dict[str, object]", result).get("snapshot") if isinstance(result, dict) else None
    if not isinstance(tree, str):
        return ()
    found: list[tuple[str, str]] = []
    for line in tree.splitlines():
        match = _LINK_LINE.match(line)
        if match is not None and match.group(2).endswith("]"):
            found.append((match.group(1), match.group(2)[:-1]))
    return tuple(found)


def _time_for_batch() -> float:
    """Return the seconds one batch may take: `BATCH_LIMIT_S`, within the request's time."""
    request = current_request()
    left = request.time_left() if request is not None else BATCH_LIMIT_S + RESERVE_S
    return min(BATCH_LIMIT_S, left - RESERVE_S)


def _port(port: int | None, scheme: str) -> int:
    """Return an address's port, the scheme's own when it names none."""
    if port is not None:
        return port
    return 443 if scheme == "https" else 80


def _rows(out: str) -> list[dict[str, Any]]:
    """Return the rows of a `batch --json` answer.

    Raises:
        BrowserAnswerError: the answer is not a list of rows (agent-browser failed, perhaps
            before the batch, perhaps not).
    """
    try:
        parsed: object = json.loads(out)
    except ValueError as exc:
        msg = "the browser gave an answer Jarvis doesn't read"
        raise BrowserAnswerError(msg) from exc
    if not isinstance(parsed, list):
        msg = "the browser couldn't start"
        raise BrowserAnswerError(msg)
    return [
        cast("dict[str, Any]", row) for row in cast("list[object]", parsed) if isinstance(row, dict)
    ]


def _new_folder(root: Path) -> Path:
    """Make a session's folder: agent-browser's config and action policy, and its folders.

    The work folder is the browser's home and working folder; the download folder, and the
    home's Downloads, no one can write in. A folder that could not be made whole is removed.

    Raises:
        OSError: it could not be made.
    """
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _sweep(root)
    folder = Path(tempfile.mkdtemp(prefix=f"session-{os.getpid()}-", dir=root))
    with _IN_USE_LOCK:
        _IN_USE.add(folder)
    try:
        _fill_folder(folder)
    except OSError:
        _release(folder, remove=True)
        raise
    return folder


def _fill_folder(folder: Path) -> None:
    (folder / "config.json").write_text("{}\n", encoding="utf-8")
    policy = {"default": "deny", "allow": list(ALLOWED_ACTIONS)}
    (folder / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    (folder / "s").mkdir(mode=0o700)
    downloads = folder / "downloads"
    config = folder / "work" / ".config"
    config.mkdir(mode=0o700, parents=True)
    # Chrome's own default Downloads, should it ever pass over the download folder.
    (config / "user-dirs.dirs").write_text(f'XDG_DOWNLOAD_DIR="{downloads}"\n', encoding="utf-8")
    (folder / "work" / "Downloads").mkdir(mode=0o500)
    # agent-browser has Chrome save every download here (`Browser.setDownloadBehavior` "allow",
    # with this path), so a page can start one itself; Chrome cannot write here, and the
    # download fails instead of being saved.
    downloads.mkdir(mode=0o500)


def _release(folder: Path, *, remove: bool) -> None:
    """Note a session folder as no longer in use here, and delete it if `remove`."""
    with _IN_USE_LOCK:
        _IN_USE.discard(folder)
    if remove:
        _remove(folder)


def _remove(folder: Path) -> None:
    """Delete a session folder, its read-only folders included."""
    for name in ("downloads", "work/Downloads"):
        with contextlib.suppress(OSError):
            (folder / name).chmod(0o700)
    shutil.rmtree(folder, ignore_errors=True)


def _daemon_running(folder: Path) -> bool:
    """Return whether the agent-browser of a session folder still runs (its record says so)."""
    for record in (folder / "s").rglob("*.pid"):
        try:
            pid = int(record.read_text(encoding="ascii").strip())
            command = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError, ValueError:
            continue
        if PROGRAM.encode() in command:
            return True
    return False


def _sweep(root: Path) -> None:
    """Delete the session folders no browser uses any more.

    That is a folder of a Jarvis that is gone (it stopped without closing), or of this one but no
    longer in use, once its agent-browser has ended too.
    """
    for folder in root.glob("session-*-*"):
        pid = folder.name.split("-")[1]
        if not pid.isdigit():
            continue
        if int(pid) == os.getpid():
            with _IN_USE_LOCK:
                if folder in _IN_USE:
                    continue
        elif _alive(int(pid)):
            continue
        if not _daemon_running(folder):
            _remove(folder)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # someone else's
    return True


# Where agent-browser (0.44) looks for a Chrome on Linux, under the home folder, after its own.
_CHROME_IN_HOME = (
    ".agent-browser/browsers/chrome-*/chrome",  # `agent-browser install`
    ".agent-browser/browsers/chrome-*/chrome-linux64/chrome",
    ".cache/puppeteer/chrome/*/chrome-linux64/chrome",
    ".cache/ms-playwright/chromium-*/chrome-linux*/chrome",
)
_CHROME_ON_PATH = (
    "google-chrome",
    "google-chrome-stable",
    "chromium-browser",
    "chromium",
    "brave-browser",
    "brave-browser-stable",
)


def chrome_path(home: Path) -> Path | None:
    """Return the Chrome agent-browser installed (or one it falls back to) under `home`, or None.

    One on PATH agent-browser finds itself; this one it would find only under the real home.
    """
    for pattern in _CHROME_IN_HOME:
        for path in sorted(home.glob(pattern), reverse=True):
            if path.is_file():
                return path
    return None


def chrome_found(home: Path) -> bool:
    """Return whether agent-browser has a Chrome to drive: its own, or one it falls back to."""
    if any(shutil.which(name) for name in _CHROME_ON_PATH):
        return True
    return chrome_path(home) is not None


def _daemon_timer(seconds: float, action: Callable[[], None]) -> threading.Timer:
    timer = threading.Timer(seconds, action)
    timer.daemon = True
    return timer
