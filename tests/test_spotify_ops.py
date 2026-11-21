"""Spotify operations against an in-process fake of the Web API (httpx.MockTransport)."""

import json
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from jarvis.config import SpotifyConfig
from jarvis.policy import Origin
from jarvis.secretfile import write_private_json
from jarvis.spotify.api import API, SpotifyApi
from jarvis.spotify.auth import TokenStore
from jarvis.spotify.ops import SpotifyOps, best_match, similarity

if TYPE_CHECKING:
    from pathlib import Path


Answer = httpx.Response | Exception
PC = {"id": "d1", "name": "pc", "type": "Computer", "is_active": True}


def _track(name: str, artist: str, uri: str) -> dict[str, Any]:
    return {
        "type": "track",
        "name": name,
        "uri": uri,
        "artists": [{"name": artist, "uri": f"spotify:artist:{artist}"}],
        "album": {"name": f"{name} album", "uri": f"spotify:album:{name}"},
    }


@dataclass
class Clock:
    """One fake clock for the client and the operations; sleeping moves it."""

    now: float = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeSpotify:
    """A tiny Spotify: canned catalogue, library and player state, with every request recorded."""

    search: dict[str, list[dict[str, Any]]] = field(default_factory=dict[str, list[dict[str, Any]]])
    by_query: dict[tuple[str, str], list[dict[str, Any]]] = field(
        default_factory=dict[tuple[str, str], list[dict[str, Any]]]
    )
    top_artists: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    top_tracks: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    playlists: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    saved: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    queue_items: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    devices: list[dict[str, Any]] = field(default_factory=lambda: [dict(PC)])
    state: dict[str, Any] | None = None
    page_size: int = 10
    scripts: dict[tuple[str, str], list[Answer]] = field(
        default_factory=dict[tuple[str, str], list[Answer]]
    )
    requests: list[httpx.Request] = field(default_factory=list[httpx.Request])
    launches: list[int] = field(default_factory=list[int])
    clock: Clock = field(default_factory=Clock)

    def script(self, method: str, path: str, *answers: Answer) -> None:
        """Answer the next requests to `method path` with `answers` before the normal route."""
        self.scripts.setdefault((method, path), []).extend(answers)

    def calls(self, method: str, path: str) -> list[httpx.Request]:
        return [
            r for r in self.requests if (r.method, r.url.path.removeprefix("/v1")) == (method, path)
        ]

    def bodies(self, method: str, path: str) -> list[Any]:
        return [json.loads(r.content) if r.content else {} for r in self.calls(method, path)]

    @property
    def played(self) -> list[dict[str, Any]]:
        return self.bodies("PUT", "/me/player/play")

    @property
    def queued(self) -> list[str]:
        return [r.url.params["uri"] for r in self.calls("POST", "/me/player/queue")]

    @property
    def searches(self) -> list[tuple[str, str]]:
        return [(r.url.params["q"], r.url.params["type"]) for r in self.calls("GET", "/search")]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        method, path = request.method, request.url.path.removeprefix("/v1")
        scripted = self.scripts.get((method, path))
        if scripted:
            answer = scripted.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        if path == "/search":
            return self._search(request)
        if method in {"PUT", "POST"}:
            return httpx.Response(204)
        return self._read(path)

    def _search(self, request: httpx.Request) -> httpx.Response:
        kind = request.url.params["type"]
        offset = int(request.url.params.get("offset", "0"))
        items = self.by_query.get((request.url.params["q"], kind), self.search.get(kind, []))
        page = items[offset : offset + self.page_size]
        return httpx.Response(200, json={f"{kind}s": {"items": page}})

    def _read(self, path: str) -> httpx.Response:
        routes: dict[str, object] = {
            "/me/player/devices": {"devices": self.devices},
            "/me/top/artists": {"items": self.top_artists},
            "/me/top/tracks": {"items": self.top_tracks},
            "/me/following": {"artists": {"items": []}},
            "/me/player/recently-played": {"items": []},
            "/me/playlists": {"items": self.playlists},
            "/me/tracks": {"items": [{"track": t} for t in self.saved]},
            "/me/player/queue": {"currently_playing": None, "queue": self.queue_items},
        }
        if path == "/me/player":
            if self.state is None:
                return httpx.Response(204)
            return httpx.Response(200, json=self.state)
        if path in routes:
            return httpx.Response(200, json=routes[path])
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})


def _ops(fake: FakeSpotify, tmp_path: Path, config: SpotifyConfig | None = None) -> SpotifyOps:
    """Build the operations over `fake`: no network, no real launcher, no real sleeping."""
    token_file = tmp_path / "token.json"
    write_private_json(
        token_file,
        {"access_token": "a", "refresh_token": "r", "expires_at": time.time() + 3600},
    )
    http = httpx.Client(base_url=API, transport=httpx.MockTransport(fake.handle))
    api = SpotifyApi(
        TokenStore("client", token_file), http, sleep=fake.clock.sleep, clock=fake.clock
    )
    return SpotifyOps(
        api,
        config or SpotifyConfig(device_name="pc"),
        launcher=lambda: fake.launches.append(1),
        rng=random.Random(7),  # ruff: ignore[suspicious-non-cryptographic-random-usage] - song order
        clock=fake.clock,
    )


GLASS_HARBOR = {"name": "Glass Harbor", "uri": "spotify:artist:glassharbor"}
LOW_TIDE = _track("Low Tide", "Glass Harbor", "spotify:track:lowtide")


@pytest.mark.parametrize("heard", ["grass harbor", "glass harbour", "glass arbor"])
def test_misheard_artist_matches_by_spelling(heard: str) -> None:
    assert (
        best_match(heard, [{"name": "Paper Moons", "uri": "x"}, GLASS_HARBOR], 0.6) is GLASS_HARBOR
    )


def test_similarity_ignores_case_and_punctuation() -> None:
    assert similarity("Glass-Harbor!", "glass harbor") == pytest.approx(1.0)
    assert similarity("", "glass harbor") == pytest.approx(0.0)


def test_exact_search_hit_is_played_with_one_search(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]})

    result = _ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert result.ok
    assert fake.played == [{"uris": ["spotify:track:lowtide"]}]
    assert fake.searches == [("low tide", "track")]
    assert not fake.calls("GET", "/me/top/artists")


def test_a_song_by_an_artist_searches_with_both_filters(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]})

    _ops(fake, tmp_path).play({"query": "Low Tide", "kind": "track", "artist": "Glass Harbor"})

    assert fake.searches == [('track:"Low Tide" artist:"Glass Harbor"', "track")]
    assert fake.played == [{"uris": ["spotify:track:lowtide"]}]


def test_double_quotes_are_removed_from_the_filter_values(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]})

    _ops(fake, tmp_path).play({"query": 'Low "Tide"', "kind": "track", "artist": 'Glass "Harbor'})

    assert fake.searches == [('track:"Low Tide" artist:"Glass Harbor"', "track")]


def test_an_empty_filtered_search_falls_back_to_plain_text(tmp_path: Path) -> None:
    fake = FakeSpotify(
        by_query={
            ('track:"Low Tide" artist:"Glass Harbor"', "track"): [],
            ("Low Tide Glass Harbor", "track"): [LOW_TIDE],
        }
    )

    result = _ops(fake, tmp_path).play(
        {"query": "Low Tide", "kind": "track", "artist": "Glass Harbor"}
    )

    assert result.ok
    assert [q for q, _ in fake.searches] == [
        'track:"Low Tide" artist:"Glass Harbor"',
        "Low Tide Glass Harbor",
    ]
    assert fake.played == [{"uris": ["spotify:track:lowtide"]}]


def test_the_title_is_scored_against_the_name_only(tmp_path: Path) -> None:
    # A title-plus-artist query used to score 0.71 against the right song and trigger the
    # library fallback; now the title alone decides, so no library call is made.
    fake = FakeSpotify(search={"track": [LOW_TIDE]})

    _ops(fake, tmp_path).play({"query": "Low Tide", "kind": "track", "artist": "Glass Harbor"})

    assert not fake.calls("GET", "/me/top/artists")
    assert not fake.calls("GET", "/me/top/tracks")


def test_every_hit_is_examined_not_only_the_first(tmp_path: Path) -> None:
    hits = [
        _track("Low Tide Remixed Megamix", "DJ Nobody", "spotify:track:mix"),
        _track("Slow Tide", "Other", "spotify:track:slow"),
        LOW_TIDE,
    ]
    fake = FakeSpotify(search={"track": hits})

    _ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert fake.played == [{"uris": ["spotify:track:lowtide"]}]


def test_an_exact_title_beats_a_closer_spelling_and_ties_keep_api_order(tmp_path: Path) -> None:
    first = _track("Low Tide", "A", "spotify:track:first")
    second = _track("Low Tide", "B", "spotify:track:second")
    fake = FakeSpotify(
        search={"track": [_track("Low Tides", "C", "spotify:track:near"), first, second]}
    )

    _ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert fake.played == [{"uris": ["spotify:track:first"]}]


def test_the_requested_artist_decides_between_equal_titles(tmp_path: Path) -> None:
    cover = _track("Low Tide", "Cover Band", "spotify:track:cover")
    fake = FakeSpotify(search={"track": [cover, LOW_TIDE]})

    _ops(fake, tmp_path).play({"query": "Low Tide", "kind": "track", "artist": "glass harbour"})

    assert fake.played == [{"uris": ["spotify:track:lowtide"]}]


def test_a_well_matched_title_by_another_artist_is_still_played(tmp_path: Path) -> None:
    # The artist may be misheard beyond recognition; the title matched, so play it.
    fake = FakeSpotify(search={"track": [_track("Low Tide", "Cover Band", "spotify:track:cover")]})

    result = _ops(fake, tmp_path).play(
        {"query": "Low Tide", "kind": "track", "artist": "Night Circuit"}
    )

    assert result.ok
    assert fake.played == [{"uris": ["spotify:track:cover"]}]


def test_a_remastered_version_still_counts_as_the_title(tmp_path: Path) -> None:
    remaster = _track("Low Tide - Remastered 2011", "Glass Harbor", "spotify:track:remaster")
    fake = FakeSpotify(
        search={"track": [remaster]}, top_tracks=[_track("Low Tides", "X", "spotify:track:x")]
    )

    _ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert fake.played == [{"uris": ["spotify:track:remaster"]}]
    assert not fake.calls("GET", "/me/top/tracks")


def test_misheard_artist_asked_as_track_plays_the_owners_artist(tmp_path: Path) -> None:
    # The owner said "play a Glass Harbor song"; speech-to-text heard "grass harbor" and the
    # model asked for a track. Spotify's top track hit is an unrelated song.
    fake = FakeSpotify(
        search={"track": [_track("Harbor Lights", "Paper Moons", "spotify:track:moons")]},
        top_artists=[GLASS_HARBOR],
    )

    _ops(fake, tmp_path).play({"query": "grass harbor", "kind": "track"})

    assert fake.played == [{"context_uri": "spotify:artist:glassharbor"}]


def test_misheard_track_falls_back_to_owners_track(tmp_path: Path) -> None:
    fake = FakeSpotify(
        search={"track": [_track("Something Else", "Other", "spotify:track:other")]},
        top_tracks=[_track("Second Sun", "Night Circuit", "spotify:track:sun")],
    )

    _ops(fake, tmp_path).play({"query": "secund sun", "kind": "track"})

    assert fake.played == [{"uris": ["spotify:track:sun"]}]


def test_owners_playlist_is_preferred(tmp_path: Path) -> None:
    fake = FakeSpotify(
        playlists=[{"name": "2031 NEON TASTE", "uri": "spotify:playlist:mine"}],
        search={"playlist": [{"name": "Neon Classics", "uri": "spotify:playlist:public"}]},
    )

    _ops(fake, tmp_path).play({"query": "2031 neon taste", "kind": "playlist"})

    assert fake.played == [{"context_uri": "spotify:playlist:mine"}]
    assert not fake.calls("GET", "/search")


def test_a_public_playlist_is_played_when_the_owner_has_none_like_it(tmp_path: Path) -> None:
    fake = FakeSpotify(
        playlists=[{"name": "Gym", "uri": "spotify:playlist:mine"}],
        search={"playlist": [{"name": "Rainy Day Jazz", "uri": "spotify:playlist:public"}]},
    )

    _ops(fake, tmp_path).play({"query": "rainy day jazz", "kind": "playlist"})

    assert fake.played == [{"context_uri": "spotify:playlist:public"}]


def test_my_playlist_never_falls_back_to_a_public_one(tmp_path: Path) -> None:
    fake = FakeSpotify(
        playlists=[{"name": "Gym", "uri": "spotify:playlist:mine"}],
        search={"playlist": [{"name": "Rainy Day Jazz", "uri": "spotify:playlist:public"}]},
    )

    result = _ops(fake, tmp_path).play(
        {"query": "rainy day jazz", "kind": "playlist", "mine": True}
    )

    assert not result.ok
    assert result.text == "You don't have a playlist called 'rainy day jazz'."
    assert result.origin is Origin.TOOL_STATUS
    assert fake.played == []
    assert not fake.calls("GET", "/search")


def test_my_playlist_accepts_a_loose_spelling_of_the_owners_own(tmp_path: Path) -> None:
    fake = FakeSpotify(playlists=[{"name": "Morning Run Mix", "uri": "spotify:playlist:mine"}])

    _ops(fake, tmp_path).play({"query": "morning ron", "kind": "playlist", "mine": True})

    assert fake.played == [{"context_uri": "spotify:playlist:mine"}]


def test_liked_songs_play_the_owners_saved_tracks_shuffled(tmp_path: Path) -> None:
    saved = [_track(f"Song {i}", "Night Circuit", f"spotify:track:{i}") for i in range(30)]
    fake = FakeSpotify(saved=saved)

    result = _ops(fake, tmp_path).play({"query": "my liked songs", "kind": "liked"})

    assert result.ok
    (body,) = fake.played
    assert sorted(body["uris"]) == sorted(t["uri"] for t in saved)
    assert body["uris"] != [t["uri"] for t in saved]
    assert fake.calls("GET", "/me/tracks")[0].url.params["limit"] == "50"
    assert not fake.calls("GET", "/search")


def test_liked_songs_with_nothing_saved_say_so(tmp_path: Path) -> None:
    fake = FakeSpotify()

    result = _ops(fake, tmp_path).play({"query": "liked songs", "kind": "liked"})

    assert not result.ok
    assert "saved" in result.text
    assert fake.played == []


def test_nothing_found_reports_failure(tmp_path: Path) -> None:
    fake = FakeSpotify()

    result = _ops(fake, tmp_path).play({"query": "zzzz", "kind": "track"})

    assert not result.ok
    assert fake.played == []


def test_a_found_item_reports_with_spotifys_origin(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]})

    result = _ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert result.origin is Origin.SPOTIFY
    assert "Low Tide" in result.text


def _catalogue(n: int, artist: str = "Night Circuit") -> list[dict[str, Any]]:
    return [_track(f"Song {i}", artist, f"spotify:track:{i}") for i in range(n)]


def test_queue_one_track_by_default(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(3)})

    _ops(fake, tmp_path).queue({"query": "song 0"})

    assert fake.queued == ["spotify:track:0"]


def test_queue_several_distinct_tracks_across_search_pages(tmp_path: Path) -> None:
    # "queue up 14 Night Circuit songs": Spotify returns at most 10 results per page.
    fake = FakeSpotify(search={"track": _catalogue(24)})

    result = _ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 14})

    assert result.ok
    assert fake.queued == [f"spotify:track:{i}" for i in range(14)]


def test_queue_skips_versions_of_the_same_song(tmp_path: Path) -> None:
    tracks = [
        _track("Low Tide", "Night Circuit", "spotify:track:a"),
        _track("Low Tide - Remix", "Night Circuit", "spotify:track:b"),
        _track("Paper Boats - Single Version", "Night Circuit", "spotify:track:c"),
        _track("Paper Boats", "Night Circuit", "spotify:track:d"),
        _track("Second Sun", "Night Circuit", "spotify:track:e"),
    ]
    fake = FakeSpotify(search={"track": tracks})

    _ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 3})

    assert fake.queued == ["spotify:track:a", "spotify:track:c", "spotify:track:e"]


def test_queue_in_random_order_is_a_shuffled_pick(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(30)})

    _ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 10, "random_order": True})

    assert len(set(fake.queued)) == 10
    assert fake.queued != [f"spotify:track:{i}" for i in range(10)]
    assert max(int(uri.rsplit(":", 1)[1]) for uri in fake.queued) >= 20  # drawn from the wider pool


def test_random_order_is_not_fixed_by_default(tmp_path: Path) -> None:
    # Two default-built players must not pick the same ten of forty songs in the same order;
    # with a fixed seed they would. A false failure is about 1 in 10**15.
    orders: list[list[str]] = []
    for run in range(2):
        fake = FakeSpotify(search={"track": _catalogue(40)})
        token_file = tmp_path / f"token{run}.json"
        write_private_json(
            token_file,
            {"access_token": "a", "refresh_token": "r", "expires_at": time.time() + 3600},
        )
        http = httpx.Client(base_url=API, transport=httpx.MockTransport(fake.handle))
        ops = SpotifyOps(
            SpotifyApi(TokenStore("c", token_file), http),
            SpotifyConfig(device_name="pc"),
            lambda: None,  # a launcher that does nothing: the default one starts the real app
        )
        ops.queue({"query": "Night Circuit", "count": 10, "random_order": True})
        orders.append(fake.queued)
    assert orders[0] != orders[1]


def test_queue_reports_fewer_when_spotify_has_fewer(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(4)})

    result = _ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 10})

    assert result.ok
    assert len(fake.queued) == 4


def test_artist_names_are_distinct_and_cached(tmp_path: Path) -> None:
    fake = FakeSpotify(
        top_artists=[GLASS_HARBOR, GLASS_HARBOR, {"name": "Night Circuit", "uri": "u"}]
    )
    ops = _ops(fake, tmp_path)

    names = ops.artist_names()
    fetched = len(fake.requests)
    fake.clock.now += 60
    assert ops.artist_names() == names
    assert len(fake.requests) == fetched
    fake.clock.now += SpotifyConfig().library_refresh_s
    ops.artist_names()
    assert len(fake.requests) > fetched

    assert names == ["Glass Harbor", "Night Circuit"]


def test_queue_pages_by_what_spotify_actually_returned(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(12)}, page_size=3)

    _ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 7})

    assert fake.queued == [f"spotify:track:{i}" for i in range(7)]


def test_a_playlist_made_after_the_cache_is_found_without_waiting(tmp_path: Path) -> None:
    fake = FakeSpotify(playlists=[{"name": "Gym", "uri": "spotify:playlist:gym"}])
    ops = _ops(fake, tmp_path)
    ops.play({"query": "gym", "kind": "playlist", "mine": True})

    fake.playlists.append({"name": "Sunday Chores", "uri": "spotify:playlist:chores"})
    result = ops.play({"query": "sunday chores", "kind": "playlist", "mine": True})

    assert result.ok
    assert fake.played[-1] == {"context_uri": "spotify:playlist:chores"}


def test_playlists_are_not_fetched_again_for_every_request(tmp_path: Path) -> None:
    fake = FakeSpotify(playlists=[{"name": "Gym", "uri": "spotify:playlist:gym"}])
    ops = _ops(fake, tmp_path)

    ops.play({"query": "gym", "kind": "playlist", "mine": True})
    ops.play({"query": "gym", "kind": "playlist", "mine": True})

    assert len(fake.calls("GET", "/me/playlists")) == 1
