"""Spotify operations against an in-process fake of the Web API (httpx.MockTransport)."""

import json
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from jarvis.config import SpotifyConfig
from jarvis.secretfile import write_private_json
from jarvis.spotify.api import API, SpotifyApi
from jarvis.spotify.auth import TokenStore
from jarvis.spotify.ops import SpotifyOps, best_match, similarity

if TYPE_CHECKING:
    from pathlib import Path


def _track(name: str, artist: str, uri: str) -> dict[str, Any]:
    return {
        "name": name,
        "uri": uri,
        "artists": [{"name": artist, "uri": f"spotify:artist:{artist}"}],
        "album": {"name": f"{name} album", "uri": f"spotify:album:{name}"},
    }


@dataclass
class FakeSpotify:
    search: dict[str, list[dict[str, Any]]] = field(default_factory=dict[str, list[dict[str, Any]]])
    top_artists: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    top_tracks: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    playlists: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    played: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    queued: list[str] = field(default_factory=list[str])
    page_size: int = 5

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        routes: dict[str, object] = {
            "/me/player/devices": {
                "devices": [{"id": "d1", "name": "pc", "type": "Computer", "is_active": True}]
            },
            "/me/top/artists": {"items": self.top_artists},
            "/me/top/tracks": {"items": self.top_tracks},
            "/me/following": {"artists": {"items": []}},
            "/me/player/recently-played": {"items": []},
            "/me/playlists": {"items": self.playlists},
        }
        if request.method == "PUT" and path == "/me/player/play":
            self.played.append(json.loads(request.content))
            return httpx.Response(204)
        if request.method == "POST" and path == "/me/player/queue":
            self.queued.append(request.url.params["uri"])
            return httpx.Response(204)
        if path == "/search":
            kind = request.url.params["type"]
            offset = int(request.url.params.get("offset", "0"))
            items = self.search.get(kind, [])[offset : offset + self.page_size]
            return httpx.Response(200, json={f"{kind}s": {"items": items}})
        if path in routes:
            return httpx.Response(200, json=routes[path])
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})


def _ops(fake: FakeSpotify, tmp_path: Path) -> SpotifyOps:
    token_file = tmp_path / "token.json"
    write_private_json(
        token_file,
        {"access_token": "a", "refresh_token": "r", "expires_at": time.time() + 3600},
    )
    http = httpx.Client(base_url=API, transport=httpx.MockTransport(fake.handle))
    return SpotifyOps(
        SpotifyApi(TokenStore("client", token_file), http),
        SpotifyConfig(device_name="pc"),
        launcher=lambda: None,
        rng=random.Random(7),  # ruff: ignore[suspicious-non-cryptographic-random-usage] - song order
    )


BAD_OMENS = {"name": "Bad Omens", "uri": "spotify:artist:badomens"}


@pytest.mark.parametrize("heard", ["bad moments", "Bandoments", "bad omen"])
def test_misheard_artist_matches_by_spelling(heard: str) -> None:
    assert best_match(heard, [{"name": "Moments Like This", "uri": "x"}, BAD_OMENS]) is BAD_OMENS


def test_similarity_ignores_case_and_punctuation() -> None:
    assert similarity("AC/DC", "acdc") == pytest.approx(1.0)
    assert similarity("", "acdc") == pytest.approx(0.0)


def test_exact_search_hit_is_played(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [_track("Like A Villain", "Bad Omens", "spotify:track:v")]})

    result = _ops(fake, tmp_path).play({"query": "like a villain", "kind": "track"})

    assert result.ok
    assert fake.played == [{"uris": ["spotify:track:v"]}]


def test_misheard_artist_asked_as_track_plays_the_owners_artist(tmp_path: Path) -> None:
    # The owner said "play a Bad Omens song"; speech-to-text heard "bad moments" and the model
    # asked for a track. Spotify's top track hit is an unrelated song.
    fake = FakeSpotify(
        search={"track": [_track("Moments Like This", "The Afters", "spotify:track:afters")]},
        top_artists=[BAD_OMENS],
    )

    _ops(fake, tmp_path).play({"query": "bad moments", "kind": "track"})

    assert fake.played == [{"context_uri": "spotify:artist:badomens"}]


def test_misheard_track_falls_back_to_owners_track(tmp_path: Path) -> None:
    fake = FakeSpotify(
        search={"track": [_track("Something Else", "Other", "spotify:track:other")]},
        top_tracks=[_track("Just Pretend", "Bad Omens", "spotify:track:jp")],
    )

    _ops(fake, tmp_path).play({"query": "just pretent", "kind": "track"})

    assert fake.played == [{"uris": ["spotify:track:jp"]}]


def test_owners_playlist_is_preferred(tmp_path: Path) -> None:
    fake = FakeSpotify(
        playlists=[{"name": "2026 ROCK TASTE", "uri": "spotify:playlist:mine"}],
        search={"playlist": [{"name": "Rock Classics", "uri": "spotify:playlist:public"}]},
    )

    _ops(fake, tmp_path).play({"query": "2026 rock taste", "kind": "playlist"})

    assert fake.played == [{"context_uri": "spotify:playlist:mine"}]


def test_nothing_found_reports_failure(tmp_path: Path) -> None:
    fake = FakeSpotify()

    result = _ops(fake, tmp_path).play({"query": "zzzz", "kind": "track"})

    assert not result.ok
    assert fake.played == []


def _catalogue(n: int, artist: str = "Breaking Benjamin") -> list[dict[str, Any]]:
    return [_track(f"Song {i}", artist, f"spotify:track:{i}") for i in range(n)]


def test_queue_one_track_by_default(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(3)})

    _ops(fake, tmp_path).queue({"query": "song 0"})

    assert fake.queued == ["spotify:track:0"]


def test_queue_several_distinct_tracks_across_search_pages(tmp_path: Path) -> None:
    # "queue up 10 Breaking Benjamin songs": Spotify returns 5 results per page here.
    fake = FakeSpotify(search={"track": _catalogue(14)})

    result = _ops(fake, tmp_path).queue({"query": "Breaking Benjamin", "count": 10})

    assert result.ok
    assert fake.queued == [f"spotify:track:{i}" for i in range(10)]


def test_queue_skips_versions_of_the_same_song(tmp_path: Path) -> None:
    tracks = [
        _track("Breath", "Breaking Benjamin", "spotify:track:a"),
        _track("Breath - Remix", "Breaking Benjamin", "spotify:track:b"),
        _track("The Diary of Jane - Single Version", "Breaking Benjamin", "spotify:track:c"),
        _track("The Diary Of Jane", "Breaking Benjamin", "spotify:track:d"),
        _track("Angels Fall", "Breaking Benjamin", "spotify:track:e"),
    ]
    fake = FakeSpotify(search={"track": tracks})

    _ops(fake, tmp_path).queue({"query": "Breaking Benjamin", "count": 3})

    assert fake.queued == ["spotify:track:a", "spotify:track:c", "spotify:track:e"]


def test_queue_in_random_order_is_a_shuffled_pick(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(30)})

    _ops(fake, tmp_path).queue({"query": "Breaking Benjamin", "count": 10, "random_order": True})

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
            lambda: None,
        )
        ops.queue({"query": "Breaking Benjamin", "count": 10, "random_order": True})
        orders.append(fake.queued)
    assert orders[0] != orders[1]


def test_queue_reports_fewer_when_spotify_has_fewer(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(4)})

    result = _ops(fake, tmp_path).queue({"query": "Breaking Benjamin", "count": 10})

    assert result.ok
    assert len(fake.queued) == 4


def test_artist_names_are_distinct_and_cached(tmp_path: Path) -> None:
    fake = FakeSpotify(top_artists=[BAD_OMENS, BAD_OMENS, {"name": "Linkin Park", "uri": "u"}])
    now = [0.0]
    calls: list[str] = []
    handle = fake.handle

    def counting(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return handle(request)

    token_file = tmp_path / "token.json"
    write_private_json(
        token_file, {"access_token": "a", "refresh_token": "r", "expires_at": time.time() + 3600}
    )
    http = httpx.Client(base_url=API, transport=httpx.MockTransport(counting))
    ops = SpotifyOps(
        SpotifyApi(TokenStore("c", token_file), http),
        SpotifyConfig(device_name="pc"),
        clock=lambda: now[0],
    )

    names = ops.artist_names()
    fetched = len(calls)
    now[0] += 60
    assert ops.artist_names() == names
    assert len(calls) == fetched
    now[0] += SpotifyConfig().library_refresh_s
    ops.artist_names()
    assert len(calls) > fetched

    assert names == ["Bad Omens", "Linkin Park"]


def test_queue_pages_by_what_spotify_actually_returned(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(12)}, page_size=3)

    _ops(fake, tmp_path).queue({"query": "Breaking Benjamin", "count": 7})

    assert fake.queued == [f"spotify:track:{i}" for i in range(7)]
