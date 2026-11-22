"""Reads one web page or picture: checked destinations, a size cap, and a page's main text.

Redirects are followed here, one hop at a time, so each new address is checked before it is
reached (the connection itself is checked again in `jarvis.web.guard`).
"""

import copy
import time
import warnings
import zlib
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import urljoin, urlsplit

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

MAX_REDIRECTS = 5
# A whole read, every redirect and chunk. A hop begun just before it may still spend its own DNS,
# connect and read timeouts, so the worst case is about 45 s, inside the agent's 90 s.
READ_LIMIT_S = 15.0
MAX_TITLE = 160
MAX_BYTES = 2_000_000
# Most HTML elements trafilatura is given. Its time grows faster than the size of the page
# (about 0.7 s at 20,000 elements, 3 s at 60,000), and it runs outside the read deadline, so a
# page with more is read from its start only.
MAX_ELEMENTS = 20_000
MAX_PICTURE_BYTES = 8_000_000
_TEXT_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain"})
_PICTURE_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp"})
_ACCEPT_PAGE = "text/html,text/plain"
_ACCEPT_PICTURE = "image/png,image/jpeg,image/webp,image/gif,image/bmp"
_REDIRECTS = frozenset({301, 302, 303, 307, 308})


class PageError(Exception):
    """The page could not be read; the message says why in plain words."""


@dataclass(frozen=True, slots=True)
class Page:
    """A page's final address, its title and its main text (all someone else's words).

    `image` is the page's own picture (`og:image`) at a full address Jarvis may open, or "".
    """

    url: str
    title: str
    text: str
    image: str = ""


@dataclass(frozen=True, slots=True)
class _Body:
    """What one address answered in the end: the bytes (capped), their type and text encoding."""

    url: str
    data: bytes
    kind: str
    encoding: str


def _picture_address(image: str, page: str) -> str:
    """Return a page's picture at its full address, or "" when Jarvis may not open it."""
    if not image.strip():
        return ""
    try:
        return check_url(urljoin(page, image.strip()))
    except PageError:
        return ""


def _inflater(encoding: str) -> Callable[[bytes, int], bytes] | None:
    """Return the inflate step for a body's `Content-Encoding`, or None when it is not compressed.

    The step takes compressed bytes and the most bytes it may return.

    Raises:
        PageError: a compression other than gzip or deflate.
    """
    name = encoding.strip().lower()
    if name in {"", "identity"}:
        return None
    if name in {"gzip", "x-gzip"}:
        return zlib.decompressobj(wbits=zlib.MAX_WBITS | 16).decompress
    if name == "deflate":
        return zlib.decompressobj().decompress
    msg = "the page came in a compression Jarvis doesn't read"
    raise PageError(msg)


def check_url(url: str) -> str:
    """Return `url` when it is a plain http or https address Jarvis may open.

    Raises:
        PageError: another scheme, a user name or password, an unusual port or no host.
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        msg = "that address is not a valid web address"
        raise PageError(msg) from exc
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        msg = "only http and https pages can be opened"
        raise PageError(msg)
    if parts.username is not None or parts.password is not None:
        msg = "an address with a user name or password is not opened"
        raise PageError(msg)
    if port not in {None, 80, 443}:
        msg = "only the standard web ports are opened"
        raise PageError(msg)
    return url


# What a browser does not show: the `hidden` attribute, and inline styles that hide or shrink
# text to nothing. Injected instructions are often put there (page text is someone else's words).
_LOWER = "'ABCDEFGHIJKLMNOPQRSTUVWXYZ \t\n\r'"
_STYLE = f"concat(translate(@style, {_LOWER}, 'abcdefghijklmnopqrstuvwxyz'), ';')"
_ZERO_SIZES = ("", "px", "pt", "em", "rem", "%", "ex", "ch", "vw", "vh")
_HIDING_STYLES = (
    "display:none",
    "visibility:hidden",
    "opacity:0",
    *(f"font-size:0{unit}" for unit in _ZERO_SIZES),
)
_HIDDEN = " | ".join(
    [
        "//*[@hidden]",
        "//*[translate(@aria-hidden, 'TRUE', 'true')='true']",
        *(
            f"//*[contains({_STYLE}, '{style}{end}')]"
            for style in _HIDING_STYLES
            for end in (";", "!important")
        ),
    ]
)


class _Element(Protocol):
    """The part of an lxml element used to trim a page (lxml has no type information)."""

    def xpath(self, path: str) -> list[_Element]: ...
    def iter(self) -> Iterator[_Element]: ...
    def getparent(self) -> _Element | None: ...
    def drop_tree(self) -> None: ...
    def remove(self, element: _Element) -> None: ...


def _without_hidden_and_excess(tree: _Element) -> None:
    """Remove what a browser hides from the page, then every element past `MAX_ELEMENTS`."""
    for element in tree.xpath(_HIDDEN):
        if element.getparent() is not None:
            element.drop_tree()
    elements = list(tree.iter())
    # Last first, so a parent goes only after its children.
    for element in reversed(elements[MAX_ELEMENTS:]):
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)


def _extract(html: str, url: str) -> tuple[str, str, str]:
    """Return the page's title, main text and picture (trafilatura, imported on first use)."""
    with warnings.catch_warnings():
        # urllib3, which trafilatura imports for its own downloads (unused here), warns that
        # optional SOCKS proxy support is not installed.
        warnings.filterwarnings("ignore", message="SOCKS support in urllib3")
        import trafilatura

    tree = trafilatura.load_html(html)
    if tree is not None:
        _without_hidden_and_excess(cast("_Element", tree))
    # The title too comes from what is shown (trafilatura takes it from an `<h1>`); a copy,
    # since either reader may trim the tree it is given.
    meta = trafilatura.extract_metadata(
        copy.deepcopy(tree) if tree is not None else html, default_url=url
    )
    text = trafilatura.extract(
        tree if tree is not None else html,
        url=url,
        include_comments=False,
        include_tables=False,
        include_links=False,
        include_images=False,
        favor_precision=True,
        # Its duplicate check remembers text across calls, so a page read again (or one much
        # like an earlier one) would come back empty.
        deduplicate=False,
    )
    title = meta.title or ""
    image = _picture_address(str(meta.image or ""), url)
    return " ".join(title.split())[:MAX_TITLE], text or "", image


class PageReader:
    """Fetches pages through a client that reaches only public addresses."""

    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        factory: Callable[[], httpx.Client] | None = None,
        max_bytes: int = MAX_BYTES,
        max_picture_bytes: int = MAX_PICTURE_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Use `http` (tests) or a client `factory` makes on first use (no socket while idle)."""
        super().__init__()
        self._http = http
        self._factory = factory
        self._max_bytes = max_bytes
        self._max_picture_bytes = max_picture_bytes
        self._clock = clock
        self._deadline = 0.0

    def read(self, url: str) -> Page:
        """Return the page at `url`, following at most `MAX_REDIRECTS` redirects.

        Raises:
            PageError: the address or a redirect is refused, the page failed, is not text, or
                has no readable text.
        """
        body = self._follow(url, _TEXT_TYPES, self._max_bytes, picture=False)
        text = body.data.decode(body.encoding or "utf-8", errors="replace")
        if body.kind == "text/plain":
            return Page(body.url, "", " ".join(text.split()))
        title, main, image = _extract(text, body.url)
        if not main.strip():
            msg = "the page has no readable text"
            raise PageError(msg)
        return Page(body.url, title, main, image)

    def picture(self, url: str) -> bytes:
        """Return the bytes of the picture at `url`, following redirects as `read` does.

        Raises:
            PageError: the address or a redirect is refused, the answer failed, is not a
                picture, or is larger than `max_picture_bytes`.
        """
        body = self._follow(url, _PICTURE_TYPES, self._max_picture_bytes, picture=True)
        if len(body.data) > self._max_picture_bytes:
            msg = "that picture is too big to show"
            raise PageError(msg)
        return body.data

    def _follow(self, url: str, kinds: frozenset[str], limit: int, *, picture: bool) -> _Body:
        """Return what `url` answers in the end, one checked redirect hop at a time.

        A picture's body is read to one byte past `limit`, so one over it can be told apart.

        Raises:
            PageError: a refused address, a failed or slow answer, a type outside `kinds`, or
                too many redirects.
        """
        if self._http is None:
            if self._factory is None:
                msg = "no web client is set up"
                raise PageError(msg)
            self._http = self._factory()
        current = check_url(url)
        self._deadline = self._clock() + READ_LIMIT_S
        accept = _ACCEPT_PICTURE if picture else _ACCEPT_PAGE
        for _ in range(MAX_REDIRECTS + 1):
            self._in_time()
            body, location, status = self._fetch(
                self._http, current, accept, kinds, limit + 1 if picture else limit
            )
            if status in _REDIRECTS:
                if not location:
                    msg = "the page redirected nowhere"
                    raise PageError(msg)
                current = check_url(urljoin(current, location))
                continue
            if status != HTTPStatus.OK:
                msg = f"the page answered {status}"
                raise PageError(msg)
            return _Body(current, body.data, body.kind, body.encoding)
        msg = "the page redirected too many times"
        raise PageError(msg)

    def _fetch(
        self, http: httpx.Client, url: str, accept: str, kinds: frozenset[str], limit: int
    ) -> tuple[_Body, str, int]:
        """Return the body (capped at `limit`), any redirect target and the status.

        Raises:
            PageError: the site is unreachable, too slow, of a type outside `kinds`, or the read
                failed.
        """
        try:
            headers = {"Accept": accept, "Accept-Encoding": "gzip, deflate"}
            with http.stream("GET", url, headers=headers) as resp:
                return self._body(resp, kinds, limit)
        except httpx.ConnectError as exc:
            msg = "that site can't be reached from here"
            raise PageError(msg) from exc
        except httpx.TimeoutException as exc:
            msg = "the page took too long to answer"
            raise PageError(msg) from exc
        except httpx.HTTPError as exc:
            msg = "the page couldn't be read"
            raise PageError(msg) from exc

    def _in_time(self) -> None:
        """Stop a read that has taken longer than `READ_LIMIT_S` in all.

        Raises:
            PageError: the time is up.
        """
        if self._clock() > self._deadline:
            msg = "the page took too long to answer"
            raise PageError(msg)

    def _body(
        self, resp: httpx.Response, kinds: frozenset[str], limit: int
    ) -> tuple[_Body, str, int]:
        """Read at most `limit` bytes of an answer of a type in `kinds`, or note its redirect.

        Raises:
            PageError: the answer is of another type, arrives too slowly or is badly compressed.
        """
        if resp.status_code in _REDIRECTS:
            empty = _Body("", b"", "", "")
            return empty, resp.headers.get("location", ""), resp.status_code
        kind = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if resp.status_code == HTTPStatus.OK and kind not in kinds:
            msg = (
                "that address is not a picture"
                if kinds is _PICTURE_TYPES
                else ("that address is not a web page")
            )
            raise PageError(msg)
        inflate = _inflater(resp.headers.get("content-encoding", ""))
        chunks: list[bytes] = []
        size = 0
        for raw_chunk in resp.iter_raw():
            # Inflated here, never more than the cap allows: a small compressed body can hold
            # gigabytes of text.
            try:
                chunk = inflate(raw_chunk, limit - size + 1) if inflate else raw_chunk
            except zlib.error as exc:
                msg = "the page's compressed text is broken"
                raise PageError(msg) from exc
            chunks.append(chunk)
            size += len(chunk)
            if size >= limit:
                break
            self._in_time()
        raw = b"".join(chunks)[:limit]
        return _Body(str(resp.url), raw, kind, resp.encoding or ""), "", resp.status_code
