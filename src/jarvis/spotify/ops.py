"""Spotify operations offered to the model. Names and metadata from Spotify are untrusted data."""

import difflib
import operator
import random
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, Any

from jarvis.jsonutil import obj, objs
from jarvis.operations import Operation, Result, UncertainOutcomeError
from jarvis.policy import OpClass, Origin
from jarvis.spotify.api import (
    NoDeviceError,
    SpotifyApi,
    SpotifyError,
    pick_device,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.config import SpotifyConfig
    from jarvis.spotify.api import Device

DEVICE_WAIT_S = 15.0
LOOSE_MATCH = 0.6
_NAMED_IN_REPLY = 5
_NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
_QUERY = {
    "type": "string",
    "minLength": 1,
    "maxLength": 120,
    "description": "What to search for, as said",
}


def _names(item: dict[str, Any]) -> str:
    artists = ", ".join(str(a["name"]) for a in objs(item.get("artists")) if a.get("name"))
    name = str(item.get("name", "something"))
    return f"{name} by {artists}" if artists else name


def _letters(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def similarity(heard: str, name: str) -> float:
    """Return how alike two names are by spelling (0..1), ignoring case, spaces and punctuation."""
    a, b = _letters(heard), _letters(name)
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def best_match(
    heard: str, candidates: list[dict[str, Any]], threshold: float = LOOSE_MATCH
) -> dict[str, Any] | None:
    """Return the candidate whose name is most like what was heard, if it is alike enough."""
    scored = [(similarity(heard, str(c.get("name", ""))), c) for c in candidates if c.get("uri")]
    if not scored:
        return None
    score, item = max(scored, key=operator.itemgetter(0))
    return item if score >= threshold else None


def _song_key(item: dict[str, Any]) -> str:
    """Return a song's identity ignoring versions: "Breath - Remix" is the same song as "Breath"."""
    return _letters(str(item.get("name", "")).split(" - ", 1)[0])


def launch_spotify() -> None:
    """Start the Spotify desktop app the Omarchy way, without waiting for it.

    Raises:
        SpotifyError: neither the Omarchy launcher nor Spotify is installed.
    """
    launcher = shutil.which("omarchy-launch-spotify") or shutil.which("spotify")
    if launcher is None:
        msg = "Spotify is not installed"
        raise SpotifyError(msg)
    subprocess.Popen(
        [launcher], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )


class SpotifyOps:
    """Builds the operations over one API client."""

    def __init__(
        self,
        api: SpotifyApi,
        settings: SpotifyConfig,
        launcher: Callable[[], None] = launch_spotify,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Use `api` with the owner's `settings` (device by name, match strictness, limits).

        `rng` orders random picks; by default it draws from the OS entropy source, so the order
        is neither fixed nor predictable. Tests pass a seeded one.
        """
        super().__init__()
        self._api = api
        self._settings = settings
        self._preferred = settings.device_name
        self._launch = launcher
        self._rng = rng or random.SystemRandom()
        self._clock = clock
        self._names: list[str] = []
        self._names_at: float | None = None

    def artist_names(self) -> list[str]:
        """Return the owner's artist names (top, followed, recently played), most played first.

        Cached and refreshed every `library_refresh_s`; when Spotify cannot be reached, the last
        list (or none) is used.
        """
        now = self._clock()
        due = self._names_at is None or now - self._names_at >= self._settings.library_refresh_s
        if due:
            try:
                artists = self._api.library().get("artist", [])
            except SpotifyError:
                return self._names
            self._names = list(dict.fromkeys(str(a["name"]) for a in artists if a.get("name")))
            self._names_at = now
        return self._names

    def _device(self) -> Device:
        found = pick_device(self._api.devices(), self._preferred)
        if found is None:
            self._launch()
            found = self._api.wait_for_device(self._preferred, DEVICE_WAIT_S)
        if found is None:
            msg = "no Spotify player came up; open Spotify and try again"
            raise NoDeviceError(msg)
        return found

    def _player_call(self, method: str, path: str, **kwargs: Any) -> None:  # ruff: ignore[any-type]
        try:
            self._api.request(method, path, **kwargs)
        except NoDeviceError:
            device = self._device()
            params = dict(kwargs.pop("params", None) or {})
            params["device_id"] = device.id
            self._api.request(method, path, params=params, **kwargs)

    def _library_match(self, query: str, kind: str) -> tuple[str, dict[str, Any]] | None:
        """Return the owner's own item most like `query`, and its kind.

        Speech-to-text often mishears names and the model may guess the wrong kind ("a Bad Omens
        song" heard as "bad moments", asked as a track), so every kind in the owner's library is
        compared, not only the one asked for. Playlists are matched only when asked for.
        """
        if kind == "playlist":
            found = best_match(query, self._api.my_playlists(), self._settings.loose_match)
            return (kind, found) if found else None
        scored = [
            (similarity(query, str(item.get("name", ""))), k, item)
            for k, items in self._api.library().items()
            for item in items
            if item.get("uri")
        ]
        if not scored:
            return None
        score, best_kind, item = max(scored, key=operator.itemgetter(0))
        return (best_kind, item) if score >= self._settings.loose_match else None

    def play(self, args: dict[str, Any]) -> Result:
        """Find what was asked for and play it; misheard names fall back to the owner's music."""
        query, kind = args["query"], args["kind"]
        item: dict[str, Any] | None = None
        if kind == "playlist":
            item = best_match(query, self._api.my_playlists(), threshold=self._settings.close_match)
        if item is None:
            found = self._api.search(query, kind)
            item = found[0] if found else None
        if (
            item is None
            or similarity(query, str(item.get("name", ""))) < self._settings.close_match
        ):
            mine = self._library_match(query, kind)
            if mine is not None and (
                item is None
                or similarity(query, str(mine[1].get("name", "")))
                > similarity(query, str(item.get("name", "")))
            ):
                kind, item = mine
        if item is None:
            return Result(
                f"Nothing on Spotify matched {query!r} as a {kind}.", Origin.TOOL_STATUS, ok=False
            )
        uri = str(item["uri"])
        body: dict[str, Any] = {"uris": [uri]} if kind == "track" else {"context_uri": uri}
        device = self._device()
        self._api.request("PUT", "/me/player/play", params={"device_id": device.id}, json=body)
        label = _names(item) if kind in {"track", "album"} else str(item.get("name", query))
        return Result(f"Now playing the {kind} {label} on {device.name}.", Origin.SPOTIFY)

    def _track_search_query(self, query: str) -> str:
        """Search only an artist's own songs when the query names an artist."""
        artists = self._api.search(query, "artist")
        named = best_match(query, artists[:1], threshold=self._settings.close_match)
        if named is None:
            mine = self._library_match(query, "artist")
            named = mine[1] if mine is not None and mine[0] == "artist" else None
        return f'artist:"{named["name"]}"' if named else query

    def _distinct_tracks(self, query: str, wanted: int) -> list[dict[str, Any]]:
        """Collect up to `wanted` different songs for `query`, paging through search results."""
        search = self._track_search_query(query) if wanted > 1 else query
        picked: list[dict[str, Any]] = []
        keys: set[str] = set()
        offset = 0
        for _page in range(self._settings.search_pages):
            found = self._api.search(search, "track", offset=offset)
            offset += len(found)
            if not found:
                break
            for item in found:
                key = _song_key(item)
                if item.get("uri") and key not in keys:
                    keys.add(key)
                    picked.append(item)
            if len(picked) >= wanted:
                break
        return picked[:wanted]

    def queue(self, args: dict[str, Any]) -> Result:
        """Add one or more different songs to the queue, optionally in random order.

        Raises:
            SpotifyError: Spotify failed before anything was queued.
            UncertainOutcomeError: the first queue request timed out after it was sent.
        """
        count = int(args.get("count", 1))
        shuffled = bool(args.get("random_order"))
        pool = self._distinct_tracks(
            args["query"], self._settings.queue_max * 2 if shuffled else count
        )
        if not pool:
            return Result(f"No track matched {args['query']!r}.", Origin.TOOL_STATUS, ok=False)
        tracks = self._rng.sample(pool, min(count, len(pool))) if shuffled else pool[:count]
        queued: list[str] = []
        for item in tracks:
            try:
                self._player_call("POST", "/me/player/queue", params={"uri": item["uri"]})
            except (SpotifyError, UncertainOutcomeError) as exc:
                if not queued:
                    raise
                uncertain = isinstance(exc, UncertainOutcomeError)
                return Result(
                    f"Queued {len(queued)} of {len(tracks)} ({', '.join(queued)}), then "
                    + (
                        "the next one may not have gone through; do not retry."
                        if uncertain
                        else f"Spotify failed: {exc}"
                    ),
                    Origin.SPOTIFY,
                    ok=False,
                )
            queued.append(_names(item))
        named = ", ".join(queued[:_NAMED_IN_REPLY])
        more = len(queued) - _NAMED_IN_REPLY
        tail = f" and {more} more" if more > 0 else ""
        fewer = f" (only {len(queued)} found, {count} asked)" if len(queued) < count else ""
        return Result(f"Queued {len(queued)} song(s){fewer}: {named}{tail}.", Origin.SPOTIFY)

    def now_playing(self, _args: dict[str, Any]) -> Result:
        """Describe what is playing."""
        state = self._api.player()
        item = obj((state or {}).get("item"))
        if not state or not item:
            return Result("Nothing is playing.")
        status = "playing" if state.get("is_playing") else "paused"
        device = obj(state.get("device")).get("name", "")
        volume = obj(state.get("device")).get("volume_percent")
        shuffle = "on" if state.get("shuffle_state") else "off"
        return Result(
            f"{status}: {_names(item)}, from {obj(item.get('album')).get('name', '')}; "
            f"device {device}, volume {volume}%, shuffle {shuffle}.",
            Origin.SPOTIFY,
        )

    def like(self, _args: dict[str, Any]) -> Result:
        """Save the current track to the owner's library."""
        item = obj((self._api.player() or {}).get("item"))
        if not item.get("uri"):
            return Result("Nothing is playing to like.", ok=False)
        self._api.request("PUT", "/me/library", params={"uris": item["uri"]})
        return Result(f"Saved {_names(item)} to your library.", Origin.SPOTIFY)

    def _simple(
        self, method: str, path: str, done: str, **params: object
    ) -> Callable[[dict[str, Any]], Result]:
        def run(_args: dict[str, Any]) -> Result:
            self._player_call(method, path, params=dict(params) or None)
            return Result(done)

        return run

    def set_volume(self, args: dict[str, Any]) -> Result:
        """Set the player's volume."""
        self._player_call("PUT", "/me/player/volume", params={"volume_percent": args["percent"]})
        return Result(f"Volume set to {args['percent']} percent.")

    def change_volume(self, args: dict[str, Any]) -> Result:
        """Raise or lower the volume by a step."""
        state = self._api.player() or {}
        current = obj(state.get("device")).get("volume_percent")
        if not isinstance(current, int):
            return Result("I can't read the current volume.", ok=False)
        target = max(0, min(100, current + int(args["delta"])))
        self._player_call("PUT", "/me/player/volume", params={"volume_percent": target})
        return Result(f"Volume changed from {current} to {target} percent.")

    def shuffle(self, args: dict[str, Any]) -> Result:
        """Turn shuffle on or off."""
        on = bool(args["on"])
        self._player_call("PUT", "/me/player/shuffle", params={"state": "true" if on else "false"})
        return Result(f"Shuffle {'on' if on else 'off'}.")

    def repeat(self, args: dict[str, Any]) -> Result:
        """Set repeat to off, track or context."""
        self._player_call("PUT", "/me/player/repeat", params={"state": args["mode"]})
        return Result(f"Repeat set to {args['mode']}.")

    def operations(self) -> list[Operation]:
        """Return every Spotify operation with its schema and default class."""
        return [
            Operation(
                "spotify_play",
                "Play a track, album, artist or playlist on Spotify. "
                "Use kind=playlist for the owner's playlists.",
                {
                    "type": "object",
                    "properties": {
                        "query": _QUERY,
                        "kind": {
                            "type": "string",
                            "enum": ["track", "album", "artist", "playlist"],
                        },
                    },
                    "required": ["query", "kind"],
                    "additionalProperties": False,
                },
                self.play,
                spoken_summary=lambda a: f"play the {a['kind']} {a['query']}",
            ),
            Operation(
                "spotify_pause",
                "Pause playback.",
                _NO_ARGS,
                self._simple("PUT", "/me/player/pause", "Paused."),
                spoken_summary=lambda _a: "pause the music",
            ),
            Operation(
                "spotify_resume",
                "Resume playback.",
                _NO_ARGS,
                self._simple("PUT", "/me/player/play", "Resumed."),
                spoken_summary=lambda _a: "resume the music",
            ),
            Operation(
                "spotify_next",
                "Skip to the next track.",
                _NO_ARGS,
                self._simple("POST", "/me/player/next", "Skipped to the next track."),
                spoken_summary=lambda _a: "skip this track",
            ),
            Operation(
                "spotify_previous",
                "Go back to the previous track.",
                _NO_ARGS,
                self._simple("POST", "/me/player/previous", "Went back a track."),
                spoken_summary=lambda _a: "go back a track",
            ),
            Operation(
                "spotify_set_volume",
                "Set Spotify's volume to a percentage.",
                {
                    "type": "object",
                    "properties": {"percent": {"type": "integer", "minimum": 0, "maximum": 100}},
                    "required": ["percent"],
                    "additionalProperties": False,
                },
                self.set_volume,
                spoken_summary=lambda a: f"set the volume to {a['percent']} percent",
            ),
            Operation(
                "spotify_change_volume",
                "Turn Spotify's volume up (positive) or down (negative) "
                "by a number of percentage points.",
                {
                    "type": "object",
                    "properties": {"delta": {"type": "integer", "minimum": -50, "maximum": 50}},
                    "required": ["delta"],
                    "additionalProperties": False,
                },
                self.change_volume,
                spoken_summary=lambda a: f"change the volume by {a['delta']}",
            ),
            Operation(
                "spotify_shuffle",
                "Turn shuffle on or off.",
                {
                    "type": "object",
                    "properties": {"on": {"type": "boolean"}},
                    "required": ["on"],
                    "additionalProperties": False,
                },
                self.shuffle,
                spoken_summary=lambda a: f"turn shuffle {'on' if a['on'] else 'off'}",
            ),
            Operation(
                "spotify_repeat",
                "Set repeat: off, track (this song) or context (the album or playlist).",
                {
                    "type": "object",
                    "properties": {"mode": {"type": "string", "enum": ["off", "track", "context"]}},
                    "required": ["mode"],
                    "additionalProperties": False,
                },
                self.repeat,
                spoken_summary=lambda a: f"set repeat to {a['mode']}",
            ),
            Operation(
                "spotify_queue",
                "Add songs to the play queue: one song by title, or several different songs "
                "by an artist in one call (set count, and random_order if asked). "
                "Never call it more than once for one request.",
                {
                    "type": "object",
                    "properties": {
                        "query": _QUERY,
                        "count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": self._settings.queue_max,
                            "description": "How many different songs to queue (default 1)",
                        },
                        "random_order": {
                            "type": "boolean",
                            "description": "Queue them in random order",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                self.queue,
                spoken_summary=lambda a: f"queue {a.get('count', 1)} of {a['query']}",
            ),
            Operation(
                "spotify_like",
                "Save the track that is playing now to the owner's library.",
                _NO_ARGS,
                self.like,
                spoken_summary=lambda _a: "save this track to your library",
            ),
            Operation(
                "spotify_now_playing",
                "Say what is playing, on which device, and the volume.",
                _NO_ARGS,
                self.now_playing,
                side_effect=False,
                spoken_summary=lambda _a: "check what's playing",
            ),
        ]


__all__ = ["OpClass", "SpotifyOps"]
