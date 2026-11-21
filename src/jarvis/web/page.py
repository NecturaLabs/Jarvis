"""Reads one web page: checked destinations, a size cap, and the main text pulled out of the HTML.

Redirects are followed here, one hop at a time, so each new address is checked before it is
reached (the connection itself is checked again in `jarvis.web.guard`).
"""

import warnings
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable

MAX_REDIRECTS = 5
MAX_BYTES = 2_000_000
_TEXT_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain"})
_REDIRECTS = frozenset({301, 302, 303, 307, 308})


class PageError(Exception):
    """The page could not be read; the message says why in plain words."""


@dataclass(frozen=True, slots=True)
class Page:
    """A page's final address, its title and its main text (all someone else's words)."""

    url: str
    title: str
    text: str


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


def _extract(html: str, url: str) -> tuple[str, str]:
    """Return the page's title and main text (trafilatura; imported on first use, it is large)."""
    with warnings.catch_warnings():
        # urllib3, which trafilatura imports for its own downloads (unused here), warns that
        # optional SOCKS proxy support is not installed.
        warnings.filterwarnings("ignore", message="SOCKS support in urllib3")
        import trafilatura  # ruff: ignore[import-outside-top-level] - loaded only when a page is read

    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        include_links=False,
        include_images=False,
        favor_precision=True,
        deduplicate=True,
    )
    meta = trafilatura.extract_metadata(html, default_url=url)
    title = meta.title or ""
    return " ".join(title.split()), text or ""


class PageReader:
    """Fetches pages through a client that reaches only public addresses."""

    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        factory: Callable[[], httpx.Client] | None = None,
        max_bytes: int = MAX_BYTES,
    ) -> None:
        """Use `http` (tests) or a client `factory` makes on first use (no socket while idle)."""
        super().__init__()
        self._http = http
        self._factory = factory
        self._max_bytes = max_bytes

    def read(self, url: str) -> Page:
        """Return the page at `url`, following at most `MAX_REDIRECTS` redirects.

        Raises:
            PageError: the address or a redirect is refused, the page failed, is not text, or
                has no readable text.
        """
        if self._http is None:
            if self._factory is None:
                msg = "no web client is set up"
                raise PageError(msg)
            self._http = self._factory()
        current = check_url(url)
        for _ in range(MAX_REDIRECTS + 1):
            body, kind, location, status = self._fetch(self._http, current)
            if status in _REDIRECTS:
                if not location:
                    msg = "the page redirected nowhere"
                    raise PageError(msg)
                current = check_url(urljoin(current, location))
                continue
            if status != HTTPStatus.OK:
                msg = f"the page answered {status}"
                raise PageError(msg)
            if kind == "text/plain":
                return Page(current, "", " ".join(body.split()))
            title, text = _extract(body, current)
            if not text.strip():
                msg = "the page has no readable text"
                raise PageError(msg)
            return Page(current, title, text)
        msg = "the page redirected too many times"
        raise PageError(msg)

    def _fetch(self, http: httpx.Client, url: str) -> tuple[str, str, str, int]:
        """Return the body (capped), its media type, any redirect target and the status.

        Raises:
            PageError: the site is unreachable, too slow, not a page, or the read failed.
        """
        try:
            with http.stream("GET", url, headers={"Accept": "text/html,text/plain"}) as resp:
                return self._body(resp)
        except httpx.ConnectError as exc:
            msg = "that site can't be reached from here"
            raise PageError(msg) from exc
        except httpx.TimeoutException as exc:
            msg = "the page took too long to answer"
            raise PageError(msg) from exc
        except httpx.HTTPError as exc:
            msg = "the page couldn't be read"
            raise PageError(msg) from exc

    def _body(self, resp: httpx.Response) -> tuple[str, str, str, int]:
        """Read at most `max_bytes` of an answer that is a page, or note where it redirects.

        Raises:
            PageError: the answer is not text.
        """
        if resp.status_code in _REDIRECTS:
            return "", "", resp.headers.get("location", ""), resp.status_code
        kind = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if resp.status_code == HTTPStatus.OK and kind not in _TEXT_TYPES:
            msg = "that address is not a web page"
            raise PageError(msg)
        chunks: list[bytes] = []
        size = 0
        for chunk in resp.iter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size >= self._max_bytes:
                break
        raw = b"".join(chunks)[: self._max_bytes]
        text = raw.decode(resp.encoding or "utf-8", errors="replace")
        return text, kind, "", resp.status_code
