"""The client of the local SearXNG instance (`deploy/searxng/`); it never reaches another host."""

from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx

from jarvis.jsonutil import obj, objs

if TYPE_CHECKING:
    from collections.abc import Callable

MAX_TITLE = 160
MAX_SNIPPET = 300
TIMEOUT = httpx.Timeout(15.0, connect=2.0)


class SearchUnavailableError(Exception):
    """The local search service did not answer, or answered with something unusable."""


@dataclass(frozen=True, slots=True)
class Hit:
    """One search result: someone else's title and snippet, and the page's address."""

    title: str
    url: str
    snippet: str


def _clean(text: object, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def _openable(url: str) -> bool:
    """Return whether a result's URL is a plain web address (some engines return broken ones)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme in {"http", "https"} and bool(parts.hostname) and not parts.username


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
        self._factory = factory or (lambda: httpx.Client(timeout=TIMEOUT, trust_env=False))

    def search(self, query: str, *, limit: int, site: str = "") -> list[Hit]:
        """Return up to `limit` results for `query`, optionally only from `site` (a host and path).

        Raises:
            SearchUnavailableError: SearXNG is not running, timed out or answered an error.
        """
        if self._http is None:
            self._http = self._factory()
        text = f"site:{site} {query}" if site else query
        try:
            response = self._http.get(
                f"{self._base}/search", params={"q": text, "format": "json", "safesearch": 1}
            )
        except httpx.HTTPError as exc:
            msg = "search isn't running on this PC"
            raise SearchUnavailableError(msg) from exc
        if response.status_code != HTTPStatus.OK:
            msg = f"search answered {response.status_code}"
            raise SearchUnavailableError(msg)
        try:
            results = objs(obj(response.json()).get("results"))
        except ValueError as exc:
            msg = "search answered something that is not JSON"
            raise SearchUnavailableError(msg) from exc
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
