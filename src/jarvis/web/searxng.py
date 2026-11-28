"""The client of the local SearXNG instance (`deploy/searxng/`); it never reaches another host."""

import re
import threading
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit

import httpx

from jarvis.jsonutil import obj, objs

if TYPE_CHECKING:
    from collections.abc import Callable

MAX_TITLE = 160
MAX_SNIPPET = 300
MAX_REFUSALS = 6
MAX_REASON = 60
TIMEOUT = httpx.Timeout(15.0, connect=2.0)


class SearchUnavailableError(Exception):
    """The local search service did not answer, or answered with something unusable."""


@dataclass(frozen=True, slots=True)
class Hit:
    """One search result: someone else's title and snippet, and the page's address."""

    title: str
    url: str
    snippet: str


@dataclass(frozen=True, slots=True)
class ImageHit:
    """One picture an image search found: someone else's title, the picture and its page.

    `width` and `height` are what the search engine reported, 0 when it did not say.
    """

    title: str
    image_url: str
    page_url: str
    width: int
    height: int


# "1920x1080", "2464 x 1643", and with a multiplication sign.
_RESOLUTION = re.compile(r"^\s*(\d{1,6})\s*[xX\u00d7]\s*(\d{1,6})\s*$")


def _size(text: object) -> tuple[int, int]:
    match = _RESOLUTION.match(str(text or ""))
    return (int(match[1]), int(match[2])) if match else (0, 0)


def _clean(text: object, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def _refusals(engines: object) -> str:
    """Return SearXNG's `unresponsive_engines` ([name, reason] pairs) as "name: reason, ..."."""
    if not isinstance(engines, list):
        return ""
    named: list[str] = []
    for entry in cast("list[object]", engines)[:MAX_REFUSALS]:
        parts = cast("list[object]", entry) if isinstance(entry, list) else [entry]
        named.append(": ".join(_clean(part, MAX_REASON) for part in parts[:2] if part))
    return ", ".join(name for name in named if name)


def _openable(url: str) -> bool:
    """Return whether a result's URL is a plain web address (some engines return broken ones)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme in {"http", "https"} and bool(parts.hostname) and not parts.username


def _fetchable(url: str) -> bool:
    """Return whether a picture's URL is one the page reader may fetch: a standard port too."""
    try:
        port = urlsplit(url).port
    except ValueError:
        return False
    return _openable(url) and port in {None, 80, 443}


class Searxng:
    """Searches through SearXNG's JSON API at `base_url` (loopback, checked by the config)."""

    def __init__(
        self,
        base_url: str,
        http: httpx.Client | None = None,
        *,
        factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        """Use `http` (tests) or a client made on first use, so an idle Jarvis holds no socket."""
        super().__init__()
        self._base = base_url.rstrip("/")
        self._http = http
        self._lock = threading.Lock()
        self._factory = factory or (lambda: httpx.Client(timeout=TIMEOUT, trust_env=False))

    def _results(self, params: dict[str, str | int]) -> list[dict[str, object]]:
        """Return the raw results SearXNG answers for `params`.

        Raises:
            SearchUnavailableError: SearXNG is not running, timed out or answered an error.
        """
        with self._lock:  # one client, however many look-ups start at once
            if self._http is None:
                self._http = self._factory()
            http = self._http
        try:
            response = http.get(
                f"{self._base}/search", params={**params, "format": "json", "safesearch": 1}
            )
        except httpx.HTTPError as exc:
            msg = "search isn't running on this PC"
            raise SearchUnavailableError(msg) from exc
        if response.status_code != HTTPStatus.OK:
            msg = f"search answered {response.status_code}"
            raise SearchUnavailableError(msg)
        try:
            answer = obj(response.json())
            results = objs(answer.get("results"))
        except ValueError as exc:
            msg = "search answered something that is not JSON"
            raise SearchUnavailableError(msg) from exc
        refused = _refusals(answer.get("unresponsive_engines"))
        if not results and refused:
            # Engines that refuse this PC (a CAPTCHA, a rate limit) leave an empty answer that is
            # not "nothing found"; SearXNG keeps them suspended for a while.
            msg = f"the search engines did not answer ({refused}); try again later"
            raise SearchUnavailableError(msg)
        return results

    def search(self, query: str, *, limit: int, site: str = "") -> list[Hit]:
        """Return up to `limit` results for `query`, optionally only from `site` (a host and path).

        SearXNG not running, timing out or answering an error raises `SearchUnavailableError`.
        """
        results = self._results({"q": f"site:{site} {query}" if site else query})
        hits: list[Hit] = []
        for result in results:
            url = str(result.get("url") or "")
            if not _openable(url) or any(hit.url == url for hit in hits):
                continue
            hits.append(
                Hit(
                    _clean(result.get("title"), MAX_TITLE) or urlsplit(url).hostname or url,
                    url,
                    _clean(result.get("content"), MAX_SNIPPET),
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def images(self, query: str, *, limit: int) -> list[ImageHit]:
        """Return up to `limit` pictures for `query` that Jarvis may fetch, each picture once.

        SearXNG not running, timing out or answering an error raises `SearchUnavailableError`.
        """
        results = self._results({"q": query, "categories": "images"})
        found: list[ImageHit] = []
        for result in results:
            image = str(result.get("img_src") or "")
            if not _fetchable(image) or any(hit.image_url == image for hit in found):
                continue
            page = str(result.get("url") or "")
            page = page if _openable(page) else ""
            title = _clean(result.get("title"), MAX_TITLE)
            width, height = _size(result.get("resolution"))
            host = urlsplit(page or image).hostname or ""
            found.append(ImageHit(title or host, image, page, width, height))
            if len(found) >= limit:
                break
        return found
