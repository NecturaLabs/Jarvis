"""Spotify operations offered to the model. Names and metadata from Spotify are untrusted data."""

import difflib
import logging
import operator
import random
import re
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from jarvis.jsonutil import obj, objs
from jarvis.operations import Operation, OperationError, Result, UncertainOutcomeError
from jarvis.policy import Origin
from jarvis.shortcuts import Shortcut
from jarvis.spotify.api import SAVED_PAGE, SpotifyApi, pick_device
from jarvis.spotify.errors import NoDeviceError, SpotifyAuthError, SpotifyError, SpotifyLoginExpired

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.config import SpotifyConfig
    from jarvis.spotify.api import Device

log = logging.getLogger(__name__)

CLOSED = (
    "Spotify is not open on this PC. Ask the owner whether to open it, and on which workspace or "
    "in the background (open_app); never open it unasked."
)
PLAYLIST_TTL_S = 300.0
REFRESH_RETRY_S = 60.0
QUEUE_BUDGET_S = 30.0
LIKED_PAGES = 4  # pages of saved songs read to pick a play of liked songs from
RECENT_QUEUE_S = 30 * 60.0
ALREADY_QUEUED = "That's already in the queue."
QUEUED_LATELY = "I queued that a few minutes ago."
SAY_WHAT_TO_PLAY = "say what to play"
NOTHING_PLAYING = "Nothing is playing."
NO_VOLUME_CONTROL = "This device doesn't allow volume control from here."
_VOLUME_DISALLOWED = "VOLUME_CONTROL_DISALLOW"
_NAMED_IN_REPLY = 5
_NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
MAX_SKIP = 10  # tracks one skip request may move by
_SKIP_ARGS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "count": {"type": "integer", "minimum": 1, "maximum": MAX_SKIP},
    },
    "additionalProperties": False,
}
_QUERY = {
    "type": "string",
    "minLength": 1,
    "maxLength": 120,
    "description": "What to search for, as said: only the title when an artist is named",
}
_ARTIST = {
    "type": "string",
    "maxLength": 120,
    "description": 'The artist, when the request names one ("the song X by Y": query X, artist Y)',
}
# Requests that mean one operation and need no model (matched in full, after `normalise`, which
# strips politeness, punctuation and the assistant's name; it also strips a trailing "now", so
# "what's playing right now" arrives as "whats playing right").
_PAUSE = Shortcut(r"(?:pause(?: it| the (?:music|song|playback))?|stop(?: the music| playing)?)")
_RESUME = Shortcut(
    r"(?:resume(?: the (?:music|playback))?|continue(?: playing)?|unpause|keep playing|play)"
)
_NEXT = Shortcut(
    r"(?:next(?: (?:song|track|one))?|skip(?: it| this(?: song| track| one)?)?"
    r"|play the next (?:song|track|one))"
)
_PREVIOUS = Shortcut(
    r"(?:previous(?: (?:song|track|one))?|go back (?:a|one) song"
    r"|play the previous (?:song|track|one))"
)
_SET_VOLUME = Shortcut(
    r"(?:(?:set|turn) (?:the )?)?volume (?:to )?(?P<percent>\d{1,3})(?: percent|%)?",
    lambda g: {"percent": int(g["percent"])},
)
_VOLUME_STEP = 10
_LOUDER = Shortcut(
    r"(?:volume up|turn (?:it|the volume) up|turn up the volume|(?:a bit |a little )?louder)",
    lambda _g: {"delta": _VOLUME_STEP},
)
_QUIETER = Shortcut(
    r"(?:volume down|turn (?:it|the volume) down|turn down the volume"
    r"|(?:a bit |a little )?quieter)",
    lambda _g: {"delta": -_VOLUME_STEP},
)
_SHUFFLE_ON = Shortcut(
    r"(?:shuffle on|turn on shuffle|turn shuffle on|enable shuffle)", lambda _g: {"on": True}
)
_SHUFFLE_OFF = Shortcut(
    r"(?:shuffle off|turn off shuffle|turn shuffle off|disable shuffle|stop shuffling)",
    lambda _g: {"on": False},
)
_NOW_PLAYING = Shortcut(
    r"(?:(?:whats|what is) (?:playing(?: right)?|this (?:song|track))"
    r"|what (?:song|track) is this|what am i listening to)"
)
_LIKE = Shortcut(
    r"(?:like this(?: song| track)?|save this (?:song|track)|i like this song"
    r"|add this song to my (?:library|liked songs))"
)
# A bracketed or trailing part that only says which version of a song this is.
_VERSION_PAREN = re.compile(
    r"\s*[(\[]\s*(?:live|remaster|\d{4}\b|feat\b|ft\b|featuring)[^)\]]*[)\]]", re.IGNORECASE
)
_FEATURING = re.compile(r"\s+(?:feat|ft|featuring)\b.*$", re.IGNORECASE)


def _names(item: dict[str, Any]) -> str:
    artists = ", ".join(str(a["name"]) for a in objs(item.get("artists")) if a.get("name"))
    name = str(item.get("name", "something"))
    return f"{name} by {artists}" if artists else name


MAX_SPOKEN_NAME = 100


def _spoken(name: object) -> str:
    """Return a name from Spotify or the model as one short line for a spoken question.

    Names are other people's text: a long or multi-line title must not turn the question into
    something else.
    """
    text = " ".join("".join(c if c.isprintable() else " " for c in str(name)).split())
    return text if len(text) <= MAX_SPOKEN_NAME else text[: MAX_SPOKEN_NAME - 1].rstrip() + "…"


def _play_question(args: dict[str, Any]) -> str:
    """Return what a play request is about, naming the item `prepare_play` found."""
    if args["kind"] == "liked":
        return "play your liked songs"
    if args.get("found"):
        return f"play the {args['found']} {_spoken(args['name'])}"
    return f"play the {args['kind']} {_spoken(args['query'])}"


def _letters(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def similarity(heard: str, name: str) -> float:
    """Return how alike two names are by spelling (0..1), ignoring case, spaces and punctuation."""
    a, b = _letters(heard), _letters(name)
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def best_match(
    heard: str, candidates: list[dict[str, Any]], threshold: float
) -> dict[str, Any] | None:
    """Return the candidate whose name is most like what was heard, if it is alike enough."""
    scored = [(similarity(heard, str(c.get("name", ""))), c) for c in candidates if c.get("uri")]
    if not scored:
        return None
    score, item = max(scored, key=operator.itemgetter(0))
    return item if score >= threshold else None


def _base_title(name: str) -> str:
    """Return a song's title without its version: no "(Live ...)", "- Remastered", "feat. ..."."""
    bare = _FEATURING.sub("", _VERSION_PAREN.sub("", name))
    return bare.split(" - ", 1)[0].strip()


def _song_key(item: dict[str, Any]) -> str:
    """Return a song's identity ignoring versions: "Breath - Remix" is the same song as "Breath".

    The identity is the base title with the primary artist (their URI, else their name), so the
    same title by another artist is another song.
    """
    artists = objs(item.get("artists"))
    primary: dict[str, Any] = artists[0] if artists else {}
    artist = str(primary.get("uri") or "") or _letters(str(primary.get("name", "")))
    return f"{_letters(_base_title(str(item.get('name', ''))))}|{artist}"


def _identities(item: dict[str, Any]) -> tuple[str, str]:
    """Return what makes a track "the same" for the queue: its URI and its song (any version)."""
    return str(item.get("uri", "")), f"song:{_song_key(item)}"


@dataclass(frozen=True, slots=True)
class _Taken:
    """The identities of what is queued or playing now, and of what Jarvis queued lately."""

    live: frozenset[str]
    recent: frozenset[str]

    def holds(self, identity: str) -> bool:
        """Return whether `identity` is in either set."""
        return identity in self.live or identity in self.recent

    def is_live(self, item: dict[str, Any]) -> bool:
        """Return whether Spotify's own queue (or the player) holds this track."""
        return any(identity in self.live for identity in _identities(item))


def _unquoted(text: str) -> str:
    """Return `text` fit for a search filter value: double quotes removed."""
    return " ".join(text.replace('"', " ").split())


@dataclass(frozen=True, slots=True)
class _Current:
    """What the player is on: nothing, an ad, a song, an episode, or something it did not say."""

    kind: str
    item: dict[str, Any]
    playing: bool


def _current(state: dict[str, Any] | None) -> _Current:
    """Read what is playing from a player state; `item` is null for ads and some content."""
    if not state:
        return _Current("nothing", {}, playing=False)
    item = obj(state.get("item"))
    playing = bool(state.get("is_playing"))
    if state.get("currently_playing_type") == "ad":
        return _Current("ad", {}, playing)
    if not item:
        return _Current("unknown", {}, playing)
    kind = "episode" if item.get("type") == "episode" else "track"
    return _Current(kind, item, playing)


def _say(sentence: str) -> Result:
    """Return a finished result whose text and spoken sentence are Jarvis's own."""
    return Result(sentence, spoken=sentence)


def _nothing_playing() -> Result:
    """Return the answer to a player command with no player: nothing was done."""
    return Result(NOTHING_PLAYING, ok=False, spoken=NOTHING_PLAYING)


@dataclass(frozen=True, slots=True)
class _Hit:
    """The best candidate for a request, with how well it matched."""

    kind: str
    item: dict[str, Any]
    score: float


class SpotifyOps:
    """Builds the operations over one API client."""

    def __init__(
        self,
        api: SpotifyApi,
        settings: SpotifyConfig,
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
        self._rng = rng or random.SystemRandom()
        self._clock = clock
        self._names: list[str] = []
        self._warm = False
        self._login_expired = False
        self._playlist_cache: list[dict[str, Any]] | None = None
        self._playlist_at = 0.0
        self._recent: dict[str, float] = {}

    def artist_names(self) -> list[str]:
        """Return the owner's artist names (top, followed, recently played), most played first.

        Only what `refresh` last loaded: it never touches the network, so a speech hint built from
        it costs the request no time. Empty until the first refresh has finished.
        """
        return self._names

    def refresh(self) -> None:
        """Load the owner's library and artist names once, ahead of any request that needs them.

        A failure is logged and swallowed, and the previous names are kept.
        """
        self._warm = False
        self._login_expired = False
        try:
            self._api.ensure_login()
            artists = self._api.library(max_age_s=0).get("artist", [])
        except SpotifyLoginExpired:
            self._login_expired = True
            log.warning("Spotify login has expired; run `jarvis spotify-login`")
            return
        except SpotifyError as exc:
            log.warning("Spotify library refresh failed: %s", exc)
            return
        except Exception:  # a warm-up must never take Jarvis down
            log.exception("Spotify library refresh failed unexpectedly")
            return
        self._names = list(dict.fromkeys(str(a["name"]) for a in artists if a.get("name")))
        self._warm = True

    def start_refresh(self, stop: threading.Event) -> threading.Thread:
        """Refresh now and then every `library_refresh_s` on a daemon thread, until `stop` is set.

        A refresh that failed (offline at start-up, say) is tried again after a minute; one that
        failed because the login expired is not, as only a new login can fix that.
        """

        def loop() -> None:
            while not stop.is_set():
                self.refresh()
                period = self._settings.library_refresh_s
                retry_soon = not (self._warm or self._login_expired)
                if stop.wait(min(period, REFRESH_RETRY_S) if retry_soon else period):
                    return

        thread = threading.Thread(target=loop, name="jarvis-spotify-refresh", daemon=True)
        thread.start()
        return thread

    def _listed_device(self) -> Device | None:
        """Return the player to use from what Spotify lists now, or None when none will do.

        A configured device that is not listed is not silently replaced by another one, unless
        that other one is the only player there is.
        """
        devices = self._api.devices()
        found = pick_device(devices, self._preferred)
        if (
            found is not None
            and self._preferred
            and len(devices) > 1
            and not self._is_preferred(found)
        ):
            return None
        return found

    def _is_preferred(self, device: Device) -> bool:
        return device.name.casefold() == self._preferred.casefold()

    def _bring_up(self, *, play: bool) -> tuple[Device, bool]:
        """Return a player that accepts commands, and whether playback was started on it.

        A listed player that is not active gets playback transferred to it (started when `play`).
        Spotify is never opened here: with no player, the owner is asked first (`open_app`).

        Raises:
            NoDeviceError: no player is available.
        """
        device = self._listed_device()
        if device is None:
            if self._preferred and self._api.devices():
                msg = f"Spotify has no player called {self._preferred!r} right now. {CLOSED}"
                raise NoDeviceError(msg)
            raise NoDeviceError(CLOSED)
        if device.is_active:
            return device, False
        self._api.transfer(device.id, play=play)
        return device, play

    def _player_call(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> None:
        """Send a player command; with no active player, bring one up and send it again."""
        try:
            self._api.request(method, path, params=params, json=json)
        except NoDeviceError:
            device, _started = self._bring_up(play=False)
            self._api.request(
                method, path, params={**(params or {}), "device_id": device.id}, json=json
            )

    def _start(self, body: dict[str, Any]) -> None:
        """Start playback of `body`: at once, or on the configured player when there is one."""
        if self._preferred:
            device, _started = self._bring_up(play=False)
            params = {"device_id": device.id}
            self._api.request("PUT", "/me/player/play", params=params, json=body)
        else:
            self._player_call("PUT", "/me/player/play", json=body)

    def _library_match(self, query: str) -> _Hit | None:
        """Return the owner's own item most like `query`, and its kind.

        Speech-to-text often mishears names and the model may guess the wrong kind ("a Glass
        Harbor song" heard as "grass harbor", asked as a track), so every kind in the owner's
        library is compared, not only the one asked for.
        """
        scored = [
            _Hit(k, item, similarity(query, str(item.get("name", ""))))
            for k, items in self._api.library(
                max_age_s=2 * self._settings.library_refresh_s
            ).items()
            for item in items
            if item.get("uri")
        ]
        best = max(scored, key=operator.attrgetter("score"), default=None)
        return best if best and best.score >= self._settings.loose_match else None

    def _playlists_cached(self) -> bool:
        """Return whether the owner's playlists are held from an earlier request."""
        return (
            self._playlist_cache is not None and self._clock() - self._playlist_at < PLAYLIST_TTL_S
        )

    def _playlists(self, *, fresh: bool = False) -> list[dict[str, Any]]:
        """Return the owner's playlists, kept for a few minutes."""
        now = self._clock()
        if fresh or self._playlist_cache is None or now - self._playlist_at >= PLAYLIST_TTL_S:
            self._playlist_cache = self._api.my_playlists()
            self._playlist_at = now
        return self._playlist_cache

    def _score(
        self, item: dict[str, Any], query: str, kind: str, artist: str
    ) -> tuple[bool, bool, float]:
        """Rank one hit: right artist, exact title, then title similarity (API order breaks ties).

        The title is compared with the item's own name only, ignoring its version suffix; the
        artist is checked separately.
        """
        name = str(item.get("name", ""))
        if kind in {"track", "album"}:
            score = max(similarity(query, name), similarity(query, _base_title(name)))
            exact = _letters(query) in {_letters(name), _letters(_base_title(name))}
        else:
            score = similarity(query, name)
            exact = _letters(query) == _letters(name)
        by_artist = not artist or any(
            similarity(artist, str(a.get("name", ""))) >= self._settings.loose_match
            for a in objs(item.get("artists"))
        )
        return by_artist, exact, score

    def _best_hit(
        self, hits: list[dict[str, Any]], query: str, kind: str, artist: str
    ) -> _Hit | None:
        """Examine every hit and return the best one (an exact title first, then similarity)."""
        ranked = [
            (self._score(item, query, kind, artist), item) for item in hits if item.get("uri")
        ]
        if not ranked:
            return None
        key, item = max(ranked, key=operator.itemgetter(0))
        return _Hit(kind, item, key[2])

    def _search_hits(self, query: str, kind: str, artist: str) -> list[dict[str, Any]]:
        """Search one page; with an artist, filter by title and artist first, then plain text."""
        if artist and kind in {"track", "album"}:
            filtered = f'{kind}:"{_unquoted(query)}" artist:"{_unquoted(artist)}"'
            return self._api.search(filtered, kind) or self._api.search(f"{query} {artist}", kind)
        return self._api.search(query, kind)

    def _find(self, query: str, kind: str, artist: str) -> _Hit | None:
        """Return what to play: the best hit, or the owner's own music for a misheard name."""
        best = self._best_hit(self._search_hits(query, kind, artist), query, kind, artist)
        if best is not None and best.score >= self._settings.close_match:
            return best
        mine = self._library_match(query)
        if mine is not None and (best is None or mine.score > best.score):
            return mine
        if best is not None and artist and not self._score(best.item, query, kind, artist)[0]:
            return None
        return best

    def _find_playlist(self, query: str, *, mine: bool) -> _Hit | None:
        """Return a playlist to play: the owner's own first; a public one only if not `mine`."""
        close, loose = self._settings.close_match, self._settings.loose_match
        was_cached = self._playlists_cached()
        own = self._best_hit(self._playlists(), query, "playlist", "")
        if was_cached and (own is None or own.score < close):
            fresh = self._best_hit(self._playlists(fresh=True), query, "playlist", "")
            own = fresh if fresh and (own is None or fresh.score > own.score) else own
        if own is not None and own.score >= close:
            return own
        if mine:
            return own if own is not None and own.score >= loose else None
        public = self._best_hit(self._api.search(query, "playlist"), query, "playlist", "")
        if public is not None and public.score >= close:
            return public
        if own is not None and own.score >= loose and (public is None or own.score > public.score):
            return own
        return public

    def _play_liked(self) -> Result:
        """Play the owner's saved tracks in random order, drawn from the whole library.

        One play takes `SAVED_PAGE` songs. A bigger library is read from its newest page and
        `LIKED_PAGES` - 1 more at random places in it, and the songs are picked from all of them.
        """
        tracks, total = self._api.saved_tracks()
        pages = range(SAVED_PAGE, total, SAVED_PAGE)
        for offset in self._rng.sample(pages, min(LIKED_PAGES - 1, len(pages))):
            tracks += self._api.saved_tracks(offset)[0]
        uris = list(dict.fromkeys(str(t["uri"]) for t in tracks))
        if not uris:
            return Result("You have no saved songs to play.", Origin.TOOL_STATUS, ok=False)
        chosen = self._rng.sample(uris, min(SAVED_PAGE, len(uris)))
        self._start({"uris": chosen})
        some = f"{len(chosen)} of " if len(chosen) < max(total, len(uris)) else ""
        return Result(f"Now playing {some}your liked songs, shuffled.", Origin.SPOTIFY)

    @staticmethod
    def _play_query(args: dict[str, Any]) -> str:
        """Return what to search for; every kind but the owner's liked songs needs it.

        Raises:
            OperationError: the request does not say what to play.
        """
        query = str(args.get("query") or "")
        if not query and args["kind"] != "liked":
            raise OperationError(SAY_WHAT_TO_PLAY)
        return query

    def prepare_play(self, args: dict[str, Any]) -> dict[str, Any]:
        """Find what a play request names, before it is decided on or confirmed.

        The item is bound into the arguments (`found` is its kind, or "" for nothing found, with
        its `uri` and `name`), so a yes plays exactly what the owner was asked about. Liked songs
        need no search. A request that names nothing fails with an `OperationError`.
        """
        query, kind = self._play_query(args), args["kind"]
        if kind == "liked":
            return args
        artist = str(args.get("artist") or "").strip()
        if kind == "playlist":
            found = self._find_playlist(query, mine=bool(args.get("mine")))
        else:
            found = self._find(query, kind, artist)
        if found is None:
            return {**args, "found": "", "uri": "", "name": ""}
        item = found.item
        label = _names(item) if found.kind in {"track", "album"} else str(item.get("name", query))
        return {**args, "found": found.kind, "uri": str(item["uri"]), "name": label}

    def play(self, args: dict[str, Any]) -> Result:
        """Play what `prepare_play` found, finding it now when it was not prepared.

        Misheard names fall back to the owner's music. A request that names nothing to play
        (every kind but liked needs a query) fails with an `OperationError` that says so.
        """
        bound = args if "found" in args else self.prepare_play(args)
        query, kind = self._play_query(bound), bound["kind"]
        if kind == "liked":
            return self._play_liked()
        if not bound["found"]:
            if kind == "playlist" and bound.get("mine"):
                return Result(
                    f"You don't have a playlist called {query!r}.", Origin.TOOL_STATUS, ok=False
                )
            return Result(
                f"Nothing on Spotify matched {query!r} as a {kind}.", Origin.TOOL_STATUS, ok=False
            )
        kind, uri = bound["found"], bound["uri"]
        body: dict[str, Any] = {"uris": [uri]} if kind == "track" else {"context_uri": uri}
        self._start(body)
        return Result(f"Now playing the {kind} {bound['name']}.", Origin.SPOTIFY)

    def _artist_search(self, name: str) -> tuple[str, str]:
        """Return the track search for an artist's own songs, and the artist's URI.

        An artist Spotify (or the owner's library, for a misheard name) does not recognise is
        searched as plain text, with no URI to filter by.
        """
        artists = self._api.search(name, "artist")
        named = best_match(name, artists[:1], threshold=self._settings.close_match)
        if named is None:
            mine = self._library_match(name)
            named = mine.item if mine is not None and mine.kind == "artist" else None
        if named is None:
            return name, ""
        return f'artist:"{_unquoted(str(named["name"]))}"', str(named["uri"])

    def _collect(
        self, search: str, wanted: int, taken: _Taken, deadline: float, by_artist: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Collect up to `wanted` different new songs, paging through the search results.

        A song counts once whatever its version; one in `taken` is skipped and returned. With
        `by_artist` (an artist URI) only that artist's tracks are kept. Paging by the number of
        entries Spotify returned, and stops at the deadline.
        """
        picked: list[dict[str, Any]] = []
        seen: set[str] = set()
        skipped: list[dict[str, Any]] = []
        offset = 0
        for _page in range(self._settings.search_pages):
            if self._clock() >= deadline:
                break
            page = self._api.search_page(search, "track", offset=offset)
            offset += page.returned
            if not page.returned:
                break
            for item in page.items:
                uri, key = _identities(item)
                by = objs(item.get("artists"))
                if not item.get("uri") or key in seen:
                    continue
                if by_artist and not any(a.get("uri") == by_artist for a in by):
                    continue
                seen.add(key)
                if taken.holds(uri) or taken.holds(key):
                    skipped.append(item)
                else:
                    picked.append(item)
            if len(picked) >= wanted:
                break
        return picked, skipped

    def _single_song(
        self, query: str, artist: str, taken: _Taken
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return the one song a title (and artist) names: the best of the first page of hits."""
        hits = [h for h in self._search_hits(query, "track", artist) if h.get("uri")]
        best = self._best_hit(hits, query, "track", artist)
        chosen = best.item if best and best.score >= self._settings.close_match else None
        chosen = chosen or next(iter(hits), None)
        if chosen is None:
            return [], []
        uri, key = _identities(chosen)
        return ([], [chosen]) if taken.holds(uri) or taken.holds(key) else ([chosen], [])

    def _queued_now(self) -> _Taken:
        """Return what is queued or playing, from Spotify and from what Jarvis queued lately.

        Spotify's own list is partial and can lag, so Jarvis also remembers its own queueing for
        a while. When Spotify's list cannot be read, the memory alone guards.

        Raises:
            SpotifyAuthError: the login is gone.
        """
        now = self._clock()
        self._recent = {k: t for k, t in self._recent.items() if now - t < RECENT_QUEUE_S}
        live: set[str] = set()
        try:
            for item in self._api.queue():
                live.update(_identities(item))
        except NoDeviceError:
            pass
        except SpotifyAuthError:
            raise
        except SpotifyError:
            log.warning("could not read Spotify's queue; relying on what was queued lately")
        return _Taken(frozenset(live), frozenset(self._recent))

    def _candidates(
        self, args: dict[str, Any], taken: _Taken, deadline: float
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """Return the songs to queue, the candidates skipped, and how many songs were asked."""
        count = int(args.get("count", 1))
        shuffled = bool(args.get("random_order"))
        query, artist = str(args["query"]), str(args.get("artist") or "").strip()
        if count == 1 and not shuffled:
            pool, skipped = self._single_song(query, artist, taken)
        else:
            search, by_artist = self._artist_search(artist or query)
            wanted = self._settings.queue_max * 2 if shuffled else count
            pool, skipped = self._collect(search, wanted, taken, deadline, by_artist)
        chosen = self._rng.sample(pool, min(count, len(pool))) if shuffled else pool[:count]
        return chosen, skipped, count

    def queue(self, args: dict[str, Any]) -> Result:
        """Add one or more different songs to the queue, optionally in random order.

        Songs already queued or playing, and songs queued by Jarvis in the last half hour (a send
        whose outcome is unknown counts), are left out and replaced from the candidates; the whole
        operation stops after 30 seconds and reports what it queued.

        Raises:
            SpotifyError: Spotify failed before anything was queued.
            UncertainOutcomeError: the first queue request timed out after it was sent.
        """
        deadline = self._clock() + QUEUE_BUDGET_S
        taken = self._queued_now()
        tracks, skipped, count = self._candidates(args, taken, deadline)
        if not tracks:
            if skipped:
                live = any(taken.is_live(item) for item in skipped)
                return Result(ALREADY_QUEUED if live else QUEUED_LATELY)
            if self._clock() >= deadline:
                return Result("Searching ran out of time, so nothing was queued.", ok=False)
            return Result(f"No track matched {args['query']!r}.", ok=False)
        queued: list[str] = []
        for item in tracks:
            if queued and self._clock() >= deadline:
                return self._partial(queued, count, "it ran out of time.")
            try:
                self._player_call("POST", "/me/player/queue", params={"uri": item["uri"]})
            except (SpotifyError, UncertainOutcomeError) as exc:
                uncertain = isinstance(exc, UncertainOutcomeError)
                if uncertain:
                    self._remember(item)
                if not queued:
                    raise
                return self._partial(
                    queued,
                    count,
                    "the next one may not have gone through; do not retry."
                    if uncertain
                    else f"Spotify failed: {exc}",
                )
            queued.append(_names(item))
            self._remember(item)
        named = ", ".join(queued[:_NAMED_IN_REPLY])
        more = len(queued) - _NAMED_IN_REPLY
        tail = f" and {more} more" if more > 0 else ""
        new = " new" if skipped else ""
        fewer = f" (only {len(queued)}{new} found, {count} asked)" if len(queued) < count else ""
        return Result(f"Queued {len(queued)} song(s){fewer}: {named}{tail}.", Origin.SPOTIFY)

    def _remember(self, item: dict[str, Any]) -> None:
        """Note that Jarvis sent `item` to the queue, so a quick repeat of the ask is not sent."""
        self._recent.update(dict.fromkeys(_identities(item), self._clock()))

    @staticmethod
    def _partial(queued: list[str], asked: int, why: str) -> Result:
        """Report a queue request that stopped part-way: what went in, and why it stopped."""
        return Result(
            f"Queued {len(queued)} of {asked} ({', '.join(queued)}), then {why}", Origin.SPOTIFY
        )

    def now_playing(self, _args: dict[str, Any]) -> Result:
        """Describe what is playing."""
        state = self._api.player()
        current = _current(state)
        if state is None or current.kind == "nothing":
            return _say(NOTHING_PLAYING)
        if current.kind == "ad":
            return _say("An ad is playing.")
        status = "playing" if current.playing else "paused"
        if current.kind == "unknown":
            return Result(
                f"Spotify says it is {status} but not what.", spoken="I can't tell what this is."
            )
        device = obj(state.get("device"))
        shuffle = "on" if state.get("shuffle_state") else "off"
        if current.kind == "episode":
            title, show = (
                current.item.get("name", ""),
                obj(current.item.get("show")).get("name", ""),
            )
            what = f"podcast episode {title}, from {show}"
            said = f"the episode {title}, from {show}"
        else:
            what = f"{_names(current.item)}, from {obj(current.item.get('album')).get('name', '')}"
            said = _names(current.item)
        return Result(
            f"{status}: {what}; device {device.get('name', '')}, "
            f"volume {device.get('volume_percent')}%, shuffle {shuffle}.",
            Origin.SPOTIFY,
            spoken=f"This is {said}{'' if current.playing else ', paused'}.",
        )

    def _playing_song(self) -> dict[str, Any]:
        """Return the song that is playing now as {"uri", "name"}.

        Raises:
            OperationError: nothing, an ad, an episode or something unreadable is playing.
        """
        current = _current(self._api.player())
        refusals = {
            "nothing": NOTHING_PLAYING,
            "ad": "An ad is playing.",
            "episode": "That's a podcast episode, and I can only save songs.",
        }
        if current.kind in refusals:
            raise OperationError(refusals[current.kind])
        if not current.item.get("uri"):
            msg = "I can't tell what is playing, so I can't save it."
            raise OperationError(msg)
        return {"uri": str(current.item["uri"]), "name": _names(current.item)}

    def like(self, args: dict[str, Any]) -> Result:
        """Save a track to the owner's library: the one `prepare` bound, else the one playing."""
        song = args if args.get("uri") else self._playing_song()
        self._api.request("PUT", "/me/library", params={"uris": song["uri"]})
        saved = f"Saved {song['name']} to your library."
        return Result(saved, Origin.SPOTIFY, spoken=saved)

    def _command(
        self, method: str, path: str, done: Result, *, refusal_check: str = "", **params: object
    ) -> Result:
        """Send a command to the active player; with none, nothing is playing and nothing to do.

        Raises:
            SpotifyError: Spotify refused, and `refusal_check` (pause) did not explain it.
        """
        try:
            self._api.request(method, path, params=dict(params) or None)
        except NoDeviceError:
            return _nothing_playing()
        except SpotifyError as exc:
            if exc.status != HTTPStatus.FORBIDDEN or not refusal_check:
                raise
            return self._already(refusal_check, exc)
        return done

    def _already(self, command: str, refusal: SpotifyError) -> Result:
        """Explain a 403 on pause or resume from the player's state, or pass the refusal on.

        Spotify refuses a pause while paused and a play while playing with a bare 403.
        """
        state = self._api.player()
        playing = bool(state and state.get("is_playing"))
        if command == "pause" and not playing:
            return _say("It's already paused.") if state else _nothing_playing()
        if command == "resume" and playing:
            return _say("It's already playing.")
        raise refusal

    def pause(self, _args: dict[str, Any]) -> Result:
        """Pause playback; it never starts Spotify."""
        return self._command(
            "PUT", "/me/player/pause", Result("Paused.", spoken="Paused."), refusal_check="pause"
        )

    def resume(self, _args: dict[str, Any]) -> Result:
        """Resume playback, moving it to a listed player when none is active; never opens Spotify.

        Raises:
            SpotifyError: Spotify refused for a reason other than "already playing".
        """
        try:
            self._api.request("PUT", "/me/player/play")
        except NoDeviceError:
            device, started = self._bring_up(play=True)
            if not started:
                self._api.request("PUT", "/me/player/play", params={"device_id": device.id})
        except SpotifyError as exc:
            if exc.status != HTTPStatus.FORBIDDEN:
                raise
            return self._already("resume", exc)
        return Result("Resumed.", spoken="Playing again.")

    def next_track(self, args: dict[str, Any]) -> Result:
        """Skip to the next track, or `count` tracks ahead; it never starts Spotify."""
        count = int(args.get("count", 1))
        if count == 1:
            done = Result("Skipped to the next track.", spoken="Skipped.")
        else:
            done = Result(f"Skipped {count} tracks.", spoken=f"Skipped {count}.")
        return self._skip("/me/player/next", count, done, "Skipped")

    def previous_track(self, args: dict[str, Any]) -> Result:
        """Go back a track, or `count` tracks; it never starts Spotify."""
        count = int(args.get("count", 1))
        if count == 1:
            done = Result("Went back a track.", spoken="Back one.")
        else:
            done = Result(f"Went back {count} tracks.", spoken=f"Back {count}.")
        return self._skip("/me/player/previous", count, done, "Went back")

    def _skip(self, path: str, count: int, done: Result, verb: str) -> Result:
        """Send one skip request per track; report how far it got when one fails part-way.

        Raises:
            SpotifyError: Spotify failed before any track was skipped.
            UncertainOutcomeError: the first request timed out after it was sent.
        """
        for skipped in range(count):
            try:
                self._api.request("POST", path)
            except NoDeviceError:
                if not skipped:
                    return _nothing_playing()
                return self._skipped_part(verb, skipped, count, "the player went away.")
            except (SpotifyError, UncertainOutcomeError) as exc:
                if not skipped:
                    raise
                why = (
                    "the next one may not have gone through; do not retry."
                    if isinstance(exc, UncertainOutcomeError)
                    else f"Spotify failed: {exc}"
                )
                return self._skipped_part(verb, skipped, count, why)
        return done

    @staticmethod
    def _skipped_part(verb: str, skipped: int, count: int, why: str) -> Result:
        """Report a skip that stopped part-way: the tracks it moved by did move."""
        return Result(f"{verb} {skipped} of {count} tracks, then {why}", ok=False, acted=True)

    def _set_volume(self, percent: int) -> Result:
        try:
            self._player_call("PUT", "/me/player/volume", params={"volume_percent": percent})
        except SpotifyError as exc:
            refused = exc.reason == _VOLUME_DISALLOWED or "control device volume" in str(exc)
            if exc.status == HTTPStatus.FORBIDDEN and refused:
                return Result(NO_VOLUME_CONTROL, ok=False)
            raise
        return Result(f"Volume set to {percent} percent.", spoken=f"Volume's at {percent}.")

    def set_volume(self, args: dict[str, Any]) -> Result:
        """Set the player's volume."""
        return self._set_volume(int(args["percent"]))

    def _volume_plan(self, delta: int) -> dict[str, Any]:
        """Read the volume now and return the one it would change to, as {"current", "target"}.

        Raises:
            OperationError: nothing is playing, the device cannot be turned up from here, or it
                does not say what its volume is.
        """
        state = self._api.player()
        if not state:
            raise OperationError(NOTHING_PLAYING)
        device = obj(state.get("device"))
        if device.get("supports_volume") is False:
            raise OperationError(NO_VOLUME_CONTROL)
        current = device.get("volume_percent")
        if not isinstance(current, int):
            msg = "I can't read the current volume."
            raise OperationError(msg)
        return {"current": current, "target": max(0, min(100, current + delta))}

    def prepare_volume(self, args: dict[str, Any]) -> dict[str, Any]:
        """Bind a volume change to the exact level it will set."""
        return {**args, **self._volume_plan(int(args["delta"]))}

    def change_volume(self, args: dict[str, Any]) -> Result:
        """Raise or lower the volume by a step: to the level `prepare` bound, else from now."""
        plan = args if "target" in args else self._volume_plan(int(args["delta"]))
        result = self._set_volume(plan["target"])
        if not result.ok:
            return result
        return Result(
            f"Volume changed from {plan['current']} to {plan['target']} percent.",
            spoken=f"Volume's at {plan['target']}.",
        )

    def shuffle(self, args: dict[str, Any]) -> Result:
        """Turn shuffle on or off."""
        on = bool(args["on"])
        self._player_call("PUT", "/me/player/shuffle", params={"state": "true" if on else "false"})
        return Result(
            f"Shuffle {'on' if on else 'off'}.", spoken=f"Shuffle's {'on' if on else 'off'}."
        )

    def repeat(self, args: dict[str, Any]) -> Result:
        """Set repeat to off, track or context."""
        self._player_call("PUT", "/me/player/repeat", params={"state": args["mode"]})
        said = {"off": "Repeat's off.", "context": "Repeat's on.", "track": "Repeating this song."}
        return Result(f"Repeat set to {args['mode']}.", spoken=said[args["mode"]])

    def operations(self) -> list[Operation]:
        """Return every Spotify operation with its schema and default class."""
        return [
            Operation(
                "spotify_play",
                "Play a track, album, artist or playlist on Spotify (query says which), or "
                "kind=liked for the owner's saved songs (shuffled; no query). Use kind=playlist "
                "for the owner's playlists, "
                'with mine=true when they said "my" playlist. For "the song X by Y" set '
                "query=X and artist=Y.",
                {
                    "type": "object",
                    "properties": {
                        "query": _QUERY,
                        "kind": {
                            "type": "string",
                            "enum": ["track", "album", "artist", "playlist", "liked"],
                        },
                        "artist": _ARTIST,
                        "mine": {
                            "type": "boolean",
                            "description": "The owner said it is their own playlist",
                        },
                    },
                    "required": ["kind"],
                    "additionalProperties": False,
                },
                self.play,
                spoken_summary=_play_question,
                prepare=self.prepare_play,
            ),
            Operation(
                "spotify_pause",
                "Pause playback.",
                _NO_ARGS,
                self.pause,
                spoken_summary=lambda _a: "pause the music",
                shortcuts=(_PAUSE,),
            ),
            Operation(
                "spotify_resume",
                "Resume playback.",
                _NO_ARGS,
                self.resume,
                spoken_summary=lambda _a: "resume the music",
                shortcuts=(_RESUME,),
            ),
            Operation(
                "spotify_next",
                "Skip to the next track, or count tracks ahead.",
                _SKIP_ARGS,
                self.next_track,
                spoken_summary=lambda a: (
                    f"skip {a['count']} tracks" if a.get("count", 1) > 1 else "skip this track"
                ),
                shortcuts=(_NEXT,),
            ),
            Operation(
                "spotify_previous",
                "Go back to the previous track, or count tracks back.",
                _SKIP_ARGS,
                self.previous_track,
                spoken_summary=lambda a: (
                    f"go back {a['count']} tracks" if a.get("count", 1) > 1 else "go back a track"
                ),
                shortcuts=(_PREVIOUS,),
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
                shortcuts=(_SET_VOLUME,),
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
                spoken_summary=lambda a: (
                    f"change the volume from {a['current']} to {a['target']} percent"
                    if "target" in a
                    else f"change the volume by {a['delta']}"
                ),
                prepare=self.prepare_volume,
                shortcuts=(_LOUDER, _QUIETER),
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
                shortcuts=(_SHUFFLE_ON, _SHUFFLE_OFF),
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
                        "artist": _ARTIST,
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
                spoken_summary=lambda a: f"queue {a.get('count', 1)} of {_spoken(a['query'])}",
            ),
            Operation(
                "spotify_like",
                "Save the track that is playing now to the owner's library.",
                _NO_ARGS,
                self.like,
                spoken_summary=lambda a: (
                    f"save {_spoken(a.get('name', 'this track'))} to your library"
                ),
                prepare=lambda _a: self._playing_song(),
                shortcuts=(_LIKE,),
            ),
            Operation(
                "spotify_now_playing",
                "Say what is playing, on which device, and the volume.",
                _NO_ARGS,
                self.now_playing,
                side_effect=False,
                spoken_summary=lambda _a: "check what's playing",
                shortcuts=(_NOW_PLAYING,),
            ),
        ]
