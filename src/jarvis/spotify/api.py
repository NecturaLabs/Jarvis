"""A small Spotify Web API client: playback, search, library and devices.

A request whose result is unknown (a failure after it may have been sent) raises
`UncertainOutcomeError` and is never replayed. Reads are retried once on a transient failure;
a rate limit is waited out only when it is short. Quota errors are reported, never looped on.
"""

import logging
import math
import re
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from itertools import starmap
from typing import TYPE_CHECKING, Any, cast

import httpx

from jarvis.jsonutil import obj, objs
from jarvis.operations import UncertainOutcomeError
from jarvis.policy import Origin
from jarvis.spotify.errors import NoDeviceError, SpotifyAuthError, SpotifyError, SpotifyLoginExpired

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.spotify.auth import TokenStore

log = logging.getLogger(__name__)

API = "https://api.spotify.com/v1"
SEARCH_LIMIT = 10
PLAYLIST_PAGE = 50
MAX_PLAYLISTS = 500
SAVED_PAGE = 50  # saved tracks Spotify returns in one page at most
RETRY_DELAY_S = 0.3
MAX_RETRY_AFTER_S = 2.0
DEFAULT_RETRY_AFTER_S = 1.0
# Calls pause this long after a 429 that gave no usable wait, or one for a used-up quota.
UNKNOWN_BLOCK_S = 60.0
QUOTA_BLOCK_S = 3600.0
LIBRARY_TTL_S = 6 * 3600.0
_LIBRARY_RETRY_S = 60.0
# (part, path, parameters, keys to descend before `items`)
_LIBRARY_PARTS: tuple[tuple[str, str, dict[str, Any], tuple[str, ...]], ...] = (
    ("top_artists", "/me/top/artists", {"limit": 50}, ()),
    ("followed", "/me/following", {"type": "artist", "limit": 50}, ("artists",)),
    ("top_tracks", "/me/top/tracks", {"limit": 50}, ()),
    ("recent", "/me/player/recently-played", {"limit": 50}, ()),
)
_DETAIL_CHARS = 160
_REASON = re.compile(r"[A-Z_]{1,40}")
_NO_DEVICE = "NO_ACTIVE_DEVICE"
_QUOTA = "QUOTA_EXCEEDED"
_GATEWAY_ERRORS = frozenset(
    {HTTPStatus.BAD_GATEWAY, HTTPStatus.SERVICE_UNAVAILABLE, HTTPStatus.GATEWAY_TIMEOUT}
)
# A gateway gives up on the server behind it with these; that server may still have run the command.
_UPSTREAM_TIMEOUTS = frozenset({HTTPStatus.BAD_GATEWAY, HTTPStatus.GATEWAY_TIMEOUT})
# Failures after which a request may already have reached Spotify.
_MAY_HAVE_BEEN_SENT = (
    httpx.ReadTimeout,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.DecodingError,
)


@dataclass(frozen=True, slots=True)
class Device:
    """A Spotify Connect device."""

    id: str
    name: str
    type: str
    is_active: bool


@dataclass(frozen=True, slots=True)
class SearchPage:
    """One page of search hits: the real items, and how many entries Spotify returned.

    Spotify may return null entries; `returned` counts them, so the next page starts at
    `offset + returned`.
    """

    items: list[dict[str, Any]]
    returned: int


@dataclass(slots=True)
class _LibraryCache:
    parts: dict[str, list[dict[str, Any]]] = field(default_factory=dict[str, list[dict[str, Any]]])
    result: dict[str, list[dict[str, Any]]] | None = None
    loaded_at: float | None = None


class SpotifyApi:
    """Calls the Web API with the owner's token."""

    def __init__(
        self,
        tokens: TokenStore,
        http: httpx.Client | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        library_ttl_s: float = LIBRARY_TTL_S,
    ) -> None:
        """Use `tokens` for auth; `http`, `sleep` and `clock` are injectable for tests.

        `library_ttl_s` is how long `library()` keeps its answer.
        """
        super().__init__()
        self._tokens = tokens
        self._http = http or httpx.Client(
            base_url=API, timeout=httpx.Timeout(10.0, connect=5.0), trust_env=False
        )
        self._sleep = sleep
        self._clock = clock
        self._library_ttl_s = library_ttl_s
        self._library = _LibraryCache()
        # After a 429 nothing is sent until this time (on `clock`); `_blocked_quota` says why.
        self._blocked_until = 0.0
        self._blocked_quota = False

    def ensure_login(self) -> None:
        """Make sure there is a valid access token, refreshing it if it is about to expire.

        It fails with `SpotifyAuthError` when there is no usable login or the refresh fails
        (`SpotifyLoginExpired` when Spotify no longer accepts the stored one).
        """
        self._tokens.access_token()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> object:
        """Send one request, refreshing the token once on 401.

        A GET is retried once on a transport error or a 502, 503 or 504. A 429 is retried once
        when Spotify says to wait two seconds or less (it was not executed); a longer or unknown
        wait or an exhausted quota is reported, and nothing is sent until that wait is over. A
        mutating request that failed after it may have been sent (a transport failure, an
        unreadable answer, a 502 or a 504) raises `UncertainOutcomeError` and is never replayed.

        Raises:
            SpotifyError: Spotify answered with an error (`NoDeviceError` when no device is
                active), a quota limit or could not be reached.
            UncertainOutcomeError: a mutating request failed after it may have been sent.
            SpotifyLoginExpired: Spotify rejected the login even after a token refresh.
        """
        self._refuse_while_blocked()
        read = method == "GET"
        force = refreshed = retried = waited = False
        while True:
            token = self._tokens.access_token(force_refresh=force)
            force = False
            try:
                response = self._http.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    headers={"Authorization": f"Bearer {token}"},
                )
            except httpx.HTTPError as exc:
                if read and not retried:
                    retried = True
                    self._sleep(RETRY_DELAY_S)
                    continue
                raise self._transport_failure(method, path, exc, read=read) from exc
            status = response.status_code
            if status == HTTPStatus.UNAUTHORIZED:
                if refreshed:
                    log.warning("Spotify %s %s: 401 again after a token refresh", method, path)
                    msg = "Spotify rejected the login; log in again: run `jarvis spotify-login`"
                    raise SpotifyLoginExpired(msg, status=status)
                refreshed = force = True
                continue
            if read and not retried and status in _GATEWAY_ERRORS:
                retried = True
                self._sleep(RETRY_DELAY_S)
                continue
            if status == HTTPStatus.TOO_MANY_REQUESTS:
                wait = self._rate_limit_wait(method, path, response)
                if wait is None or waited or wait > MAX_RETRY_AFTER_S:
                    self._block(UNKNOWN_BLOCK_S if wait is None else wait)
                    raise SpotifyError(self._limit_message(wait), status=status)
                waited = True
                self._sleep(wait)
                continue
            if not read and status in _UPSTREAM_TIMEOUTS:
                log.warning("Spotify %s %s: %s after the command was sent", method, path, status)
                msg = (
                    "Spotify did not answer after the command was sent; it may or may not have run"
                )
                raise UncertainOutcomeError(msg)
            return self._parse(method, path, response)

    @staticmethod
    def _transport_failure(
        method: str, path: str, exc: httpx.HTTPError, *, read: bool
    ) -> SpotifyError | UncertainOutcomeError:
        log.warning("Spotify %s %s failed: %s", method, path, type(exc).__name__)
        if not read and isinstance(exc, _MAY_HAVE_BEEN_SENT):
            msg = "Spotify did not answer after the command was sent; it may or may not have run"
            return UncertainOutcomeError(msg)
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            return SpotifyError("could not reach Spotify")
        if isinstance(exc, httpx.TimeoutException):
            return SpotifyError("Spotify did not answer in time")
        return SpotifyError("Spotify request failed")

    @staticmethod
    def _error_info(response: httpx.Response) -> tuple[str, str]:
        """Return Spotify's error `reason` (a bare code) and `message` (untrusted text)."""
        try:
            error = obj(obj(response.json()).get("error"))
        except ValueError:
            return "", ""
        reason, message = error.get("reason"), error.get("message")
        code = reason if isinstance(reason, str) and _REASON.fullmatch(reason) else ""
        return code, message[:_DETAIL_CHARS] if isinstance(message, str) else ""

    @staticmethod
    def _limit_message(wait: float | None) -> str:
        """Say how long to wait after a 429 that is not retried, or that it is unknown."""
        if wait is None:
            return "Spotify's request limit was reached; how long to wait is unknown"
        seconds = math.ceil(wait)
        plural = "" if seconds == 1 else "s"
        return f"Spotify's request limit was reached; try again in {seconds} second{plural}"

    def _block(self, seconds: float, *, quota: bool = False) -> None:
        """Send nothing to Spotify for `seconds` (after a 429)."""
        self._blocked_until = self._clock() + seconds
        self._blocked_quota = quota

    def _refuse_while_blocked(self) -> None:
        """Refuse a call before the wait Spotify asked for is over, without sending anything.

        Raises:
            SpotifyError: the same limit error the 429 itself raised.
        """
        remaining = self._blocked_until - self._clock()
        if remaining <= 0:
            return
        if self._blocked_quota:
            msg = "Spotify's daily developer quota is used up"
            raise SpotifyError(msg, status=HTTPStatus.TOO_MANY_REQUESTS, reason=_QUOTA)
        raise SpotifyError(self._limit_message(remaining), status=HTTPStatus.TOO_MANY_REQUESTS)

    def _rate_limit_wait(self, method: str, path: str, response: httpx.Response) -> float | None:
        """Return how long Spotify wants us to wait after a 429, or None when it said something odd.

        A missing or unreadable header means a short default wait; a number that is negative or
        not finite (`inf`, `nan`) is an unknown wait.

        Raises:
            SpotifyError: the 429 says the developer quota is used up; waiting will not help.
        """
        reason, _message = self._error_info(response)
        log.warning("Spotify %s %s: 429 %s", method, path, reason)
        if reason == _QUOTA:
            self._block(QUOTA_BLOCK_S, quota=True)
            msg = "Spotify's daily developer quota is used up"
            raise SpotifyError(msg, status=response.status_code, reason=reason)
        try:
            wait = float(response.headers["Retry-After"])
        except KeyError, ValueError:
            return DEFAULT_RETRY_AFTER_S
        return wait if math.isfinite(wait) and wait >= 0 else None

    def _parse(self, method: str, path: str, response: httpx.Response) -> object:
        status = response.status_code
        if status >= HTTPStatus.BAD_REQUEST:
            reason, message = self._error_info(response)
            log.warning("Spotify %s %s: %s %s", method, path, status, reason)
            no_device = reason == _NO_DEVICE or (
                status == HTTPStatus.NOT_FOUND
                and (_NO_DEVICE in response.text or "no active device" in message.casefold())
            )
            if no_device:
                msg = "no active Spotify device"
                raise NoDeviceError(msg, status=status, reason=reason)
            if message:
                msg = f"Spotify answered {status}: {message}"
                raise SpotifyError(msg, origin=Origin.SPOTIFY, status=status, reason=reason)
            msg = f"Spotify answered {status}"
            raise SpotifyError(msg, status=status, reason=reason)
        if status == HTTPStatus.NO_CONTENT or not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return None

    def devices(self) -> list[Device]:
        """Return the owner's Spotify Connect devices."""
        data = obj(self.request("GET", "/me/player/devices"))
        return [
            Device(
                str(d.get("id") or ""),
                str(d.get("name", "")),
                str(d.get("type", "")),
                bool(d.get("is_active")),
            )
            for d in objs(data.get("devices"))
            if d.get("id") and not d.get("is_restricted")
        ]

    def transfer(self, device_id: str, *, play: bool) -> None:
        """Move playback to `device_id`, starting it or not."""
        self.request("PUT", "/me/player", json={"device_ids": [device_id], "play": play})

    def player(self) -> dict[str, Any] | None:
        """Return the playback state, or None when nothing is playing anywhere.

        Podcast episodes are included: without `additional_types` Spotify sends a null `item`.
        """
        state = obj(self.request("GET", "/me/player", params={"additional_types": "episode"}))
        return state or None

    def queue(self) -> list[dict[str, Any]]:
        """Return what is playing and queued, the playing item first (Spotify caps the list)."""
        data = obj(self.request("GET", "/me/player/queue"))
        current = obj(data.get("currently_playing"))
        return ([current] if current else []) + objs(data.get("queue"))

    def search_page(self, query: str, kind: str, *, offset: int = 0) -> SearchPage:
        """Return one page of items of one kind (track, album, artist, playlist).

        Spotify allows at most ten per page; callers that need more continue at
        `offset + returned`.
        """
        params = {"q": query, "type": kind, "limit": SEARCH_LIMIT, "offset": offset}
        raw = obj(self.request("GET", "/search", params=params))
        entries = obj(raw.get(f"{kind}s")).get("items")
        return SearchPage(objs(entries), _count(entries))

    def search(self, query: str, kind: str, *, offset: int = 0) -> list[dict[str, Any]]:
        """Return the items of one search page, without null entries."""
        return self.search_page(query, kind, offset=offset).items

    def my_playlists(self) -> list[dict[str, Any]]:
        """Return the owner's playlists, paging until there is no next page (up to 500)."""
        playlists: list[dict[str, Any]] = []
        offset = 0
        while len(playlists) < MAX_PLAYLISTS:
            params = {"limit": PLAYLIST_PAGE, "offset": offset}
            data = obj(self.request("GET", "/me/playlists", params=params))
            entries = data.get("items")
            playlists += objs(entries)
            if not data.get("next") or not entries:
                break
            offset += _count(entries)
        return playlists[:MAX_PLAYLISTS]

    def saved_tracks(self, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Return one page of the owner's saved ("liked") tracks, newest first, and how many.

        A page holds up to `SAVED_PAGE`, starting `offset` songs in.
        """
        params = {"limit": SAVED_PAGE, "offset": offset}
        data = obj(self.request("GET", "/me/tracks", params=params))
        tracks = [obj(item.get("track")) for item in objs(data.get("items"))]
        total = data.get("total")
        return [t for t in tracks if t.get("uri")], total if isinstance(total, int) else 0

    def _library_part(
        self, name: str, path: str, params: dict[str, Any], inside: tuple[str, ...]
    ) -> bool:
        """Fetch one part of the library; on a failure keep the previous good list.

        Raises:
            SpotifyAuthError: the login is gone, which no old list can paper over.
        """
        try:
            data = obj(self.request("GET", path, params=params))
        except SpotifyAuthError:
            raise
        except SpotifyError:
            return False
        for key in inside:
            data = obj(data.get(key))
        self._library.parts[name] = objs(data.get("items"))
        return True

    def library(self, *, max_age_s: float | None = None) -> dict[str, list[dict[str, Any]]]:
        """Return the owner's own artists, tracks and albums (top, followed, recently played).

        The answer is kept for `max_age_s` (default: the client's time to live); when one endpoint
        fails, the previous good list for that part is kept. A lost login is raised.
        """
        ttl = self._library_ttl_s if max_age_s is None else max_age_s
        cache, now = self._library, self._clock()
        if cache.result is not None and cache.loaded_at is not None and now - cache.loaded_at < ttl:
            return cache.result
        return self._load_library()[0]

    def reload_library(self) -> dict[str, list[dict[str, Any]]] | None:
        """Load the library now, whatever its age; None when no part of it could be loaded.

        A part that failed keeps its previous good list, as in `library`. A lost login is raised.
        """
        result, loaded = self._load_library()
        return result if loaded else None

    def _load_library(self) -> tuple[dict[str, list[dict[str, Any]]], bool]:
        """Fetch every part of the library; return it and whether any part loaded."""
        now = self._clock()
        cache = self._library
        loaded = any(list(starmap(self._library_part, _LIBRARY_PARTS)))
        parts = cache.parts
        recent = [obj(i.get("track")) for i in parts.get("recent", [])]
        tracks = [t for t in parts.get("top_tracks", []) + recent if t.get("uri")]
        artists = parts.get("top_artists", []) + parts.get("followed", [])
        artists += [a for t in tracks for a in objs(t.get("artists"))]
        albums = [obj(t.get("album")) for t in tracks]
        cache.result = {
            "artist": [a for a in artists if a.get("uri")],
            "track": tracks,
            "album": [a for a in albums if a.get("uri")],
        }
        cache.loaded_at = now if loaded else now - self._library_ttl_s + _LIBRARY_RETRY_S
        return cache.result, loaded


def _count(value: object) -> int:
    """Return how many entries a JSON list holds, null ones included."""
    return len(cast("list[object]", value)) if isinstance(value, list) else 0


def pick_device(devices: list[Device], preferred: str) -> Device | None:
    """Pick the preferred device by name, else the active one, else a computer, else any."""
    if not devices:
        return None
    if preferred:
        for device in devices:
            if device.name.casefold() == preferred.casefold():
                return device
    for device in devices:
        if device.is_active:
            return device
    for device in devices:
        if device.type.casefold() == "computer":
            return device
    return devices[0]
