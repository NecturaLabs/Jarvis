"""A small Spotify Web API client: playback, search, library and devices.

A request whose result is unknown (a timeout after it was sent) raises `UncertainOutcomeError` and
is never replayed. Quota errors (429) are reported, never retried in a loop.
"""

import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import httpx

from jarvis.jsonutil import obj, objs
from jarvis.operations import UncertainOutcomeError

if TYPE_CHECKING:
    from jarvis.spotify.auth import TokenStore

API = "https://api.spotify.com/v1"
SEARCH_LIMIT = 5


class SpotifyError(Exception):
    """Spotify refused or failed a request."""


class NoDeviceError(SpotifyError):
    """No Spotify player is available to play on."""


@dataclass(frozen=True, slots=True)
class Device:
    """A Spotify Connect device."""

    id: str
    name: str
    type: str
    is_active: bool


class SpotifyApi:
    """Calls the Web API with the owner's token."""

    def __init__(self, tokens: TokenStore, http: httpx.Client | None = None) -> None:
        """Use `tokens` for auth; `http` is injectable for tests."""
        super().__init__()
        self._tokens = tokens
        self._http = http or httpx.Client(
            base_url=API, timeout=httpx.Timeout(10.0, connect=5.0), trust_env=False
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> object:
        """Send one request, refreshing the token once on 401.

        Raises:
            SpotifyError: Spotify answered with an error (`NoDeviceError` when no device is
                active), a quota limit or could not be reached.
            UncertainOutcomeError: a mutating request timed out after it was sent.
        """
        for attempt in (1, 2):
            token = self._tokens.access_token(force_refresh=attempt == 2)  # ruff: ignore[magic-value-comparison]
            try:
                response = self._http.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    headers={"Authorization": f"Bearer {token}"},
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                msg = f"could not reach Spotify: {exc}"
                raise SpotifyError(msg) from exc
            except httpx.TimeoutException as exc:
                if method == "GET":
                    msg = "Spotify did not answer in time"
                    raise SpotifyError(msg) from exc
                msg = "Spotify did not answer in time after the command was sent"
                raise UncertainOutcomeError(msg) from exc
            except httpx.HTTPError as exc:
                msg = f"Spotify request failed: {exc}"
                raise SpotifyError(msg) from exc
            if response.status_code == HTTPStatus.UNAUTHORIZED and attempt == 1:
                continue
            return self._parse(response)
        msg = "Spotify rejected the login; run `jarvis spotify-login` again"
        raise SpotifyError(msg)

    @staticmethod
    def _parse(response: httpx.Response) -> object:
        status = response.status_code
        if status == HTTPStatus.TOO_MANY_REQUESTS:
            msg = "Spotify's request limit was reached; try again later"
            raise SpotifyError(msg)
        if status == HTTPStatus.NOT_FOUND and "NO_ACTIVE_DEVICE" in response.text:
            msg = "no active Spotify device"
            raise NoDeviceError(msg)
        if status >= HTTPStatus.BAD_REQUEST:
            try:
                detail = obj(obj(response.json()).get("error")).get("message", "")
            except ValueError:
                detail = response.text[:120]
            msg = f"Spotify answered {status}: {detail}"
            raise SpotifyError(msg)
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

    def wait_for_device(self, preferred: str, seconds: float) -> Device | None:
        """Poll for a usable device (the preferred one by name if given) for up to `seconds`."""
        deadline = time.monotonic() + seconds
        while True:
            found = pick_device(self.devices(), preferred)
            if found or time.monotonic() >= deadline:
                return found
            time.sleep(1.0)

    def player(self) -> dict[str, Any] | None:
        """Return the playback state, or None when nothing is playing anywhere."""
        state = obj(self.request("GET", "/me/player"))
        return state or None

    def search(self, query: str, kind: str, *, offset: int = 0) -> list[dict[str, Any]]:
        """Return one page of items of one kind (track, album, artist, playlist).

        This app's Spotify access returns at most about five items a page and refuses a limit
        above ten, so callers that need more page with `offset`.
        """
        params = {"q": query, "type": kind, "limit": SEARCH_LIMIT, "offset": offset}
        data = obj(self.request("GET", "/search", params=params))
        return objs(obj(data.get(f"{kind}s")).get("items"))

    def my_playlists(self) -> list[dict[str, Any]]:
        """Return the owner's playlists (first 50)."""
        return objs(obj(self.request("GET", "/me/playlists", params={"limit": 50})).get("items"))

    def _optional(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET something that only helps matching; a refusal (e.g. a missing scope) gives {}."""
        try:
            return obj(self.request("GET", path, params=params))
        except SpotifyError:
            return {}

    def library(self) -> dict[str, list[dict[str, Any]]]:
        """Return the owner's own artists, tracks and albums (top, followed, recently played)."""
        top_artists = objs(self._optional("/me/top/artists", {"limit": 50}).get("items"))
        followed = objs(
            obj(
                self._optional("/me/following", {"type": "artist", "limit": 50}).get("artists")
            ).get("items")
        )
        top_tracks = objs(self._optional("/me/top/tracks", {"limit": 50}).get("items"))
        recent = [
            obj(i.get("track"))
            for i in objs(self._optional("/me/player/recently-played", {"limit": 50}).get("items"))
        ]
        tracks = [t for t in top_tracks + recent if t.get("uri")]
        artists = top_artists + followed + [a for t in tracks for a in objs(t.get("artists"))]
        albums = [obj(t.get("album")) for t in tracks]
        return {
            "artist": [a for a in artists if a.get("uri")],
            "track": tracks,
            "album": [a for a in albums if a.get("uri")],
        }


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
