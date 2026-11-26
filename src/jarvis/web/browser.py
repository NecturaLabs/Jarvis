"""A headless browser for pages that need one, driven through the `agent-browser` program.

`web_read` reads what a server sends; some pages build their text with JavaScript, or show it only
after a search box or a button. For those, Jarvis drives a real Chrome through agent-browser, with
these fences, each in code:

- Headless, muted and with no display variables: it can open no window and make no sound.
- Its own throwaway profile and download folder in the user's runtime folder, deleted when the
  browser closes: never the owner's browser, cookies or logins.
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
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urljoin, urlsplit

from jarvis.operations import current_request
from jarvis.web.page import PageError, check_url
from jarvis.web.proxy import EgressProxy

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

PROGRAM = "agent-browser"
NAMESPACE = "jarvis"
# One batch of commands, launching the browser included, at most; less near the request's end.
BATCH_LIMIT_S = 40.0
ACTION_TIMEOUT_MS = 15_000  # agent-browser's own limit for one command
CLOSE_LIMIT_S = 15.0
# Each batch ends this long before the request's own time limit: room to close a browser that
# did not answer, and for the model to say what happened.
RESERVE_S = CLOSE_LIMIT_S + 10.0
MIN_BATCH_S = 3.0  # less time than this left in the request, and the browser is not used
SETTLE_MS = 1200  # after a page loads or a click: scripts that fill the page in afterwards
MAX_OUTPUT = 60_000  # characters of page text or snapshot agent-browser returns
MAX_STDOUT = 2_000_000
MAX_NAME = 80  # characters of an element's name kept
MAX_SCREENSHOT_BYTES = 8_000_000
SCREEN_STEP_PX = 640  # how far one screenshot part is below the last (the window is 1280x720)
MAX_PARTS = 20
# The roles text may be typed into; anything else (a button, a link, a password) is refused.
TYPABLE = frozenset({"textbox", "searchbox", "combobox"})
# agent-browser's action names: only the commands this module sends, nothing else.
ALLOWED_ACTIONS = (
    "navigate",
    "url",
    "title",
    "gettext",
    "getattribute",
    "snapshot",
    "screenshot",
    "click",
    "fill",
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
    "--disk-cache-size=33554432",  # the profile is in the runtime folder, which is memory
)
# What the browser's environment keeps of Jarvis's: where programs and Chrome are, nothing else.
_KEPT_ENVIRONMENT = ("PATH", "HOME", "XDG_RUNTIME_DIR", "LANG", "TZ")
_CHROME_ERROR = "chrome-error:"


class BrowserError(Exception):
    """The browser could not do what was asked; the message says why in plain words."""


class BrowserTimeoutError(BrowserError):
    """The browser did not answer in time: an action sent to it may or may not have happened."""


@dataclass(frozen=True, slots=True)
class Element:
    """Something on a page that can be used: its reference (`e3`), role and visible name."""

    ref: str
    role: str
    name: str

    def said(self) -> str:
        """Name it as the owner hears it: `the button "Search"`."""
        label = f' "{self.name}"' if self.name else ""
        return f"the {self.role}{label}"


@dataclass(frozen=True, slots=True)
class View:
    """What the page shows now: its address, title, rendered text and the elements it offers."""

    url: str
    title: str
    text: str
    elements: tuple[Element, ...]

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

    An `OSError` says it could not be started.

    Raises:
        BrowserTimeoutError: it did not finish in `timeout` seconds.
    """
    import subprocess  # ruff: ignore[import-outside-top-level] - only a browser in use starts one, and a typed request may never need it

    try:
        done = subprocess.run(
            list(argv),
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(env),
            cwd=cwd,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        msg = "the browser took too long"
        raise BrowserTimeoutError(msg) from exc
    return done.stdout[:MAX_STDOUT]


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
    ) -> None:
        """Drive `program` (agent-browser's full path), keeping its throwaway files under `root`.

        With `origin` (`http://127.0.0.1:<port>`, a preview's), only addresses there are opened.
        """
        super().__init__()
        self._origin = origin
        self._program = program
        self._root = root
        self._idle_s = idle_close_s
        self._run = run
        self._make_proxy = proxy
        self._make_timer = timer or _daemon_timer
        self._session = f"jarvis-{os.getpid()}"
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
            shutil.rmtree(folder, ignore_errors=True)
            msg = f"the browser's proxy couldn't start: {exc.strerror or type(exc).__name__}"
            raise BrowserError(msg) from exc
        self._dir, self._proxy = folder, proxy
        atexit.register(self.close)
        return folder

    def _flags(self, folder: Path, proxy: EgressProxy) -> list[str]:
        """Return agent-browser's options: the same on every call, so it is never relaunched."""
        idle = int(self._idle_s) + 60  # agent-browser's own backstop, after Jarvis's
        return [
            self._program,
            "--config",
            str(folder / "config.json"),
            "--namespace",
            NAMESPACE,
            "--session",
            self._session,
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
            "--proxy",
            f"http://127.0.0.1:{proxy.port}",
            "--proxy-bypass",
            "<-loopback>",
            "--args",
            "\n".join(CHROME_ARGS),
            "--no-webmcp",
            "--max-output",
            str(MAX_OUTPUT),
            "--idle-timeout",
            f"{idle}s",
        ]

    @staticmethod
    def _environment() -> dict[str, str]:
        """Return the browser's environment: no display, no sound server, no outside settings."""
        env = {name: os.environ[name] for name in _KEPT_ENVIRONMENT if name in os.environ}
        env.update(
            {
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
            BrowserError: the request has no time left for it (nothing was sent), it could not
                start, or its answer was not what agent-browser writes.
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
        argv = [*self._flags(folder, proxy), "batch", "--bail"]
        try:
            out = self._run(
                argv,
                stdin=json.dumps(commands),
                env=self._environment(),
                cwd=folder / "work",
                timeout=limit,
            )
        except BrowserTimeoutError:
            self._shut()
            raise
        except OSError as exc:
            self._shut()
            msg = f"the browser couldn't be started: {exc.strerror or type(exc).__name__}"
            raise BrowserError(msg) from exc
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
        """Close the browser, stop its proxy and delete its folder (its profile and downloads)."""
        with self._lock:
            self._shut()

    def _shut(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        folder, proxy = self._dir, self._proxy
        self._dir, self._proxy, self._view = None, None, None
        if folder is None:
            return
        atexit.unregister(self.close)
        if proxy is not None:
            with contextlib.suppress(BrowserTimeoutError, OSError):
                self._run(
                    [*self._flags(folder, proxy), "close"],
                    stdin="",
                    env=self._environment(),
                    cwd=folder / "work",
                    timeout=CLOSE_LIMIT_S,
                )
            proxy.stop()
        shutil.rmtree(folder, ignore_errors=True)

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
        """Make the view from the last four rows: url, title, text and snapshot.

        Raises:
            BrowserError: the page is an error page (the site could not be reached).
        """
        url, title, text, snapshot = rows[-4:]
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
        result = snapshot.get("result")
        elements = _elements(cast("dict[str, Any]", result)) if isinstance(result, dict) else ()
        view = View(address, _one_line(_field(title, "title"), 160), _field(text, "text"), elements)
        self._view = view
        return view

    _LOOK = (["get", "url"], ["get", "title"], ["get", "text", "body"], ["snapshot", "-i"])

    def _act(self, commands: list[list[str]], what: str, *, acting: bool = False) -> View:
        """Run commands, let the page settle, and return what it shows then.

        Raises:
            BrowserTimeoutError: `acting` (a click, typing), the commands went through and the page
                could not be read afterwards: the action happened, so it must not be retried as if
                it had not.
            BrowserError: the commands failed, or the page could not be read.
        """
        batch = [*commands, ["wait", str(SETTLE_MS)], *(list(c) for c in self._LOOK)]
        try:
            rows = self._batch(batch)
            done = rows[: len(commands)]
            acted = len(done) == len(commands) and all(r.get("success") is True for r in done)
            try:
                self._done(rows, len(batch), what)
                return self._read(rows)
            except BrowserError as exc:
                if acting and acted:
                    msg = f"it was done, but the page couldn't be read afterwards ({exc})"
                    raise BrowserTimeoutError(msg) from exc
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

    def link(self, ref: str) -> str:
        """Return the full address a link on the current page points to.

        Raises:
            BrowserError: no page is open, the element is not a link, or it points nowhere Jarvis
                opens (another scheme, a script).
        """
        with self._lock:
            view = self._view
            found = view.element(ref) if view is not None else None
            if view is None or found is None or found.role != "link":
                msg = "that is not a link on the page now; look at the page again"
                raise BrowserError(msg)
            try:
                rows = self._batch([["get", "attr", f"@{ref}", "href"]])
                self._done(rows, 1, "read that link")
            finally:
                self._idle()
            href = _field(rows[0], "value").strip()
            if not href:
                msg = "that link points nowhere"
                raise BrowserError(msg)
            return self._checked(urljoin(view.url, href))

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
    ) -> list[dict[str, Any]]:
        """Check that the page is still at `url` and shows `expect` under its reference.

        `also` runs in the same call, after the page is looked at; their rows are returned.

        Raises:
            BrowserError: the page moved on or the element changed, so the yes no longer fits.
        """
        look = len(self._LOOK)
        rows = self._batch([*(list(c) for c in self._LOOK), *also])
        self._done(rows, look + len(also), "look at the page")
        view = self._read(rows[:look])
        if view.url != url or view.element(expect.ref) != expect:
            msg = "the page changed since; look at it again before acting"
            raise BrowserError(msg)
        return rows[look:]

    def click(self, expect: Element, url: str) -> View:
        """Click an element the owner agreed to, if the page still shows it, and return the page.

        A `BrowserTimeoutError` says the browser did not answer (the click may have happened), a
        `BrowserError` that the page changed or the click failed.
        """
        with self._lock:
            self._still_open()
            try:
                self._same(expect, url)
            finally:
                self._idle()
            return self._act([["click", f"@{expect.ref}"]], "click that", acting=True)

    def fill(self, expect: Element, url: str, text: str, *, submit: bool) -> View:
        """Type `text` into a field the owner agreed to, then press Enter if `submit`.

        A `BrowserTimeoutError` says the browser did not answer (the text may have been sent).

        Raises:
            BrowserError: the page changed, the element takes no text, or typing failed.
        """
        if expect.role not in TYPABLE:
            msg = f"{expect.said()} takes no typing"
            raise BrowserError(msg)
        with self._lock:
            self._still_open()
            try:
                rows = self._same(expect, url, [["get", "attr", f"@{expect.ref}", "type"]])
            finally:
                self._idle()
            if _field(rows[0], "value").strip().lower() in {"password", "hidden", "file"}:
                msg = "Jarvis doesn't type into password fields"
                raise BrowserError(msg)
            commands = [["fill", f"@{expect.ref}", "--", text]]
            if submit:
                commands.append(["press", "Enter"])
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
        BrowserError: the answer is not a list of rows (agent-browser failed before the batch).
    """
    try:
        parsed: object = json.loads(out)
    except ValueError as exc:
        msg = "the browser gave an answer Jarvis doesn't read"
        raise BrowserError(msg) from exc
    if not isinstance(parsed, list):
        msg = "the browser couldn't start"
        raise BrowserError(msg)
    return [
        cast("dict[str, Any]", row) for row in cast("list[object]", parsed) if isinstance(row, dict)
    ]


def _new_folder(root: Path) -> Path:
    """Make a session's folder: agent-browser's empty config, its action policy, a work folder.

    An `OSError` says it could not be made.
    """
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _sweep(root)
    folder = Path(tempfile.mkdtemp(prefix=f"session-{os.getpid()}-", dir=root))
    (folder / "config.json").write_text("{}\n", encoding="utf-8")
    policy = {"default": "deny", "allow": list(ALLOWED_ACTIONS)}
    (folder / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    (folder / "work").mkdir(mode=0o700)
    return folder


def _sweep(root: Path) -> None:
    """Delete the session folders a Jarvis that is gone left behind (it stopped without closing)."""
    for folder in root.glob("session-*-*"):
        pid = folder.name.split("-")[1]
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            continue  # alive, someone else's


def _daemon_timer(seconds: float, action: Callable[[], None]) -> threading.Timer:
    timer = threading.Timer(seconds, action)
    timer.daemon = True
    return timer
