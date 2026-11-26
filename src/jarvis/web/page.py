"""Reads one web page or picture: checked destinations, a size cap, and a page's main text.

Redirects are followed here, one hop at a time, so each new address is checked before it is
reached (the connection itself is checked again in `jarvis.web.guard`).
"""

import codecs
import copy
import re
import threading
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
    from types import ModuleType

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
# Hidden text shorter than this stays in a browser's text of a page: it could as well be part of a
# visible word or number ("1" in "$19.99").
MIN_HIDDEN_CHARS = 3
# At most this many characters searched, for each of the page's own, to take hidden text out of
# it; past that, the rest of the page is left out unchecked.
SEARCH_WORK = 50
# The first import of trafilatura changes the warning filters for a moment; two reads at once must
# not, or the filter one sets can be left in place for good.
_IMPORTING = threading.Lock()
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
        return _Gunzip()
    if name == "deflate":
        return zlib.decompressobj().decompress
    msg = "the page came in a compression Jarvis doesn't read"
    raise PageError(msg)


_GZIP_BITS = zlib.MAX_WBITS | 16
# A page's own word on its text encoding, in its first bytes: `<meta charset=...>` or the
# `http-equiv` form's `content="...; charset=..."` (the HTML standard's prescan reads 1024).
_DECLARED = re.compile(
    rb"<meta[^>]{0,200}?charset\s*=\s*[\"']?\s*([A-Za-z0-9._:-]{1,40})", re.IGNORECASE
)
_PRESCAN = 1024
_BOMS = ((b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16"))


class _Gunzip:
    """Inflates a gzip body member after member: one file may be several joined together."""

    def __init__(self) -> None:
        super().__init__()
        self._member = zlib.decompressobj(wbits=_GZIP_BITS)

    def __call__(self, data: bytes, max_length: int) -> bytes:
        """Return at most `max_length` bytes inflated from `data` and what came before it.

        A body that is not gzip, or is broken, raises zlib's own error.
        """
        out = b""
        while data and len(out) < max_length:
            if self._member.eof:
                if not data.strip(b"\0"):
                    break  # zeros padding the end of the file
                self._member = zlib.decompressobj(wbits=_GZIP_BITS)
            out += self._member.decompress(data, max_length - len(out))
            data = self._member.unused_data if self._member.eof else b""
        return out


def _charset(data: bytes, kind: str, header: str | None) -> str:
    """Return the text encoding of a body.

    The header's, else a byte order mark's, else the page's own declaration, else UTF-8; one
    Python does not know counts as none.
    """
    found = header
    if not found:
        found = next((name for mark, name in _BOMS if data.startswith(mark)), None)
    if not found and kind != "text/plain":
        declared = _DECLARED.search(data[:_PRESCAN])
        found = declared.group(1).decode("ascii") if declared else None
    try:
        return codecs.lookup(found).name if found else "utf-8"
    except LookupError:
        return "utf-8"


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
    def itertext(self) -> Iterator[str]: ...
    def iterancestors(self) -> Iterator[_Element]: ...
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


def _trafilatura() -> ModuleType:
    """Import trafilatura on first use."""
    with _IMPORTING, warnings.catch_warnings():
        # urllib3, which trafilatura imports for its own downloads (unused here), warns that
        # optional SOCKS proxy support is not installed.
        warnings.filterwarnings("ignore", message="SOCKS support in urllib3")
        import trafilatura

    return trafilatura


@dataclass(frozen=True, slots=True)
class Hidden:
    """What a page hides from view, found in its rendered HTML by the rules `web_read` uses.

    `keys` are the words of each hidden element, without whitespace, in page order (one element
    for nested ones, none shorter than `MIN_HIDDEN_CHARS`). `names` are the one-line starts of
    every element inside a hidden one, as an accessibility tree names them; None when the page
    has more such elements than `MAX_ELEMENTS`, too many to tell.
    """

    keys: tuple[str, ...]
    names: frozenset[str] | None


def hidden_parts(html: str, name_chars: int) -> Hidden:
    """Return what the page's rendered `html` hides from view.

    That is what `web_read` drops: the `hidden` attribute, `aria-hidden`, and inline styles that
    hide text or make it transparent or of no size. `name_chars` is how much of an element's
    name is kept (`names` are cut to it).
    """
    tree = _trafilatura().load_html(f"<html><body>{html}</body></html>")
    if tree is None:
        return Hidden((), frozenset())
    found = cast("_Element", tree).xpath(_HIDDEN)
    marked = set(found)
    outer = [e for e in found if not any(parent in marked for parent in e.iterancestors())]
    keys = tuple(
        key
        for element in outer
        if len(key := "".join("".join(element.itertext()).split())) >= MIN_HIDDEN_CHARS
    )
    names: set[str] = set()
    for element in outer:
        for inner in element.iter():
            if len(names) >= MAX_ELEMENTS:
                return Hidden(keys, None)
            if name := _start(inner, name_chars):
                names.add(name)
    return Hidden(keys, frozenset(names))


def _start(element: _Element, chars: int) -> str:
    """Return the start of an element's words on one line, as far as `chars` (read no further)."""
    pieces: list[str] = []
    size = 0
    for piece in element.itertext():
        pieces.append(piece)
        size += len(piece)
        if size > 2 * chars:
            break
    return " ".join("".join(pieces).split())[:chars]


def without_hidden(text: str, hidden: Hidden) -> str:
    """Return a browser's text of a page (`innerText`) without what the page hides from view.

    `innerText` already leaves out what CSS hides (`display: none`, `visibility: hidden`). What
    `web_read` also drops (`hidden`) is taken out of `text`, best effort: each hidden element's
    words in page order, each found after the one before it, whatever whitespace `innerText` put
    between them. A hidden text the browser left out already costs a search of the rest of the
    page; past `SEARCH_WORK` for each character, the rest is left out unchecked.
    """
    squeezed = "".join(text.split())
    budget = SEARCH_WORK * len(squeezed)
    gone: list[tuple[int, int]] = []
    cursor = work = 0
    for key in hidden.keys:
        at = squeezed.find(key, cursor)
        work += (at + len(key) if at >= 0 else len(squeezed)) - cursor
        if work > budget:
            gone.append((cursor, len(squeezed)))
            break
        if at >= 0:  # else CSS hid it already, and `innerText` left it out
            gone.append((at, at + len(key)))
            cursor = at + len(key)
    return _cut_out(text, gone) if gone else text


def _cut_out(text: str, gone: list[tuple[int, int]]) -> str:
    """Return `text` without the spans `gone`, counted in characters that are not whitespace."""
    out: list[str] = []
    counted = 0  # characters other than whitespace before the chunk
    span = 0
    for chunk in re.split(r"(\s+)", text):
        if not chunk or chunk[0].isspace():
            out.append(chunk)
            continue
        start, end = counted, counted + len(chunk)
        counted = end
        at = start
        while span < len(gone) and gone[span][0] < end:
            first, last = gone[span]
            if first > at:
                out.append(chunk[at - start : first - start])
            at = max(at, min(last, end))
            if last > end:
                break  # the span goes on into the next chunk
            span += 1
        out.append(chunk[at - start :])
    return "".join(out)


def _extract(html: str, url: str) -> tuple[str, str, str]:
    """Return the page's title, main text and picture (trafilatura, imported on first use)."""
    trafilatura = _trafilatura()
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
        self._lock = threading.Lock()

    def read(self, url: str) -> Page:
        """Return the page at `url`, following at most `MAX_REDIRECTS` redirects.

        Raises:
            PageError: the address or a redirect is refused, the page failed, is not text, or
                has no readable text.
        """
        body = self._follow(url, _TEXT_TYPES, self._max_bytes, picture=False)
        text = body.data.decode(_charset(body.data, body.kind, body.encoding), errors="replace")
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
        current = check_url(url)
        # Each read keeps its own deadline: two pages of one request may be read side by side.
        deadline = self._clock() + READ_LIMIT_S
        accept = _ACCEPT_PICTURE if picture else _ACCEPT_PAGE
        for _ in range(MAX_REDIRECTS + 1):
            self._in_time(deadline)
            body, location, status = self._fetch(
                current, accept, kinds, limit + 1 if picture else limit, deadline=deadline
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

    def _client(self) -> httpx.Client:
        """Return the web client, made on first use (once, however many reads start at once).

        Raises:
            PageError: no client and no way to make one was given.
        """
        with self._lock:
            if self._http is None:
                if self._factory is None:
                    msg = "no web client is set up"
                    raise PageError(msg)
                self._http = self._factory()
            return self._http

    def _fetch(
        self,
        url: str,
        accept: str,
        kinds: frozenset[str],
        limit: int,
        *,
        deadline: float,
    ) -> tuple[_Body, str, int]:
        """Return the body (capped at `limit`), any redirect target and the status.

        Raises:
            PageError: the site is unreachable, too slow, of a type outside `kinds`, or the read
                failed.
        """
        try:
            headers = {"Accept": accept, "Accept-Encoding": "gzip, deflate"}
            with self._client().stream("GET", url, headers=headers) as resp:
                return self._body(resp, kinds, limit, deadline=deadline)
        except httpx.ConnectError as exc:
            msg = "that site can't be reached from here"
            raise PageError(msg) from exc
        except httpx.TimeoutException as exc:
            msg = "the page took too long to answer"
            raise PageError(msg) from exc
        except httpx.HTTPError as exc:
            msg = "the page couldn't be read"
            raise PageError(msg) from exc

    def _in_time(self, deadline: float) -> None:
        """Stop a read that has taken longer than `READ_LIMIT_S` in all (it ends at `deadline`).

        Raises:
            PageError: the time is up.
        """
        if self._clock() > deadline:
            msg = "the page took too long to answer"
            raise PageError(msg)

    def _body(
        self, resp: httpx.Response, kinds: frozenset[str], limit: int, *, deadline: float
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
            self._in_time(deadline)
        raw = b"".join(chunks)[:limit]
        return _Body(str(resp.url), raw, kind, resp.charset_encoding or ""), "", resp.status_code
