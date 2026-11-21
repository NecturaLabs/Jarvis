"""Spotify operations against an in-process fake of the Web API (httpx.MockTransport)."""

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

import httpx
import pytest

from jarvis.config import SpotifyConfig
from jarvis.operations import Operation, OperationError, UncertainOutcomeError
from jarvis.policy import Origin
from jarvis.secretfile import write_private_json
from jarvis.spotify.api import API, SpotifyApi
from jarvis.spotify.auth import TokenStore
from jarvis.spotify.errors import NoDeviceError, SpotifyAuthError, SpotifyError, SpotifyLoginExpired
from jarvis.spotify.ops import DEVICE_WAIT_S, REFRESH_RETRY_S, SpotifyOps, best_match, similarity

if TYPE_CHECKING:
    from pathlib import Path


Answer = httpx.Response | Exception
PC = {"id": "d1", "name": "pc", "type": "Computer", "is_active": True}


def _artist_uri(name: str) -> str:
    return f"spotify:artist:{name.replace(' ', '').casefold()}"


def _track(name: str, artist: str, uri: str) -> dict[str, Any]:
    return {
        "type": "track",
        "name": name,
        "uri": uri,
        "artists": [{"name": artist, "uri": _artist_uri(artist)}],
        "album": {"name": f"{name} album", "uri": f"spotify:album:{name}"},
    }


@dataclass
class Clock:
    """One fake clock for the client and the operations; sleeping moves it."""

    now: float = 1000.0
    per_request: float = 0.0

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
    playing_item: dict[str, Any] | None = None
    devices: list[dict[str, Any]] = field(default_factory=lambda: [dict(PC)])
    state: dict[str, Any] | None = None
    page_size: int = 10
    scripts: dict[tuple[str, str], list[Answer]] = field(
        default_factory=dict[tuple[str, str], list[Answer]]
    )
    requests: list[httpx.Request] = field(default_factory=list[httpx.Request])
    launches: list[int] = field(default_factory=list[int])
    appears_on_launch: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    clock: Clock = field(default_factory=Clock)

    def script(self, method: str, path: str, *answers: Answer) -> None:
        """Answer the next requests to `method path` with `answers` before the normal route."""
        self.scripts.setdefault((method, path), []).extend(answers)

    def launch(self) -> None:
        """What the injected launcher does: Spotify starts and its player is listed."""
        self.launches.append(1)
        self.devices.extend(self.appears_on_launch)

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
    def transfers(self) -> list[dict[str, Any]]:
        return self.bodies("PUT", "/me/player")

    @property
    def queued(self) -> list[str]:
        return [r.url.params["uri"] for r in self.calls("POST", "/me/player/queue")]

    @property
    def searches(self) -> list[tuple[str, str]]:
        return [(r.url.params["q"], r.url.params["type"]) for r in self.calls("GET", "/search")]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.clock.now += self.clock.per_request
        method, path = request.method, request.url.path.removeprefix("/v1")
        scripted = self.scripts.get((method, path))
        if scripted:
            answer = scripted.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        if path == "/search":
            return self._search(request)
        if (method, path) == ("PUT", "/me/player"):
            wanted = json.loads(request.content)["device_ids"]
            for device in self.devices:
                device["is_active"] = device["id"] in wanted
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
            "/me/player/queue": {"currently_playing": self.playing_item, "queue": self.queue_items},
        }
        if path == "/me/player":
            if self.state is None:
                return httpx.Response(204)
            return httpx.Response(200, json=self.state)
        if path in routes:
            return httpx.Response(200, json=routes[path])
        return httpx.Response(404, json={"error": {"message": f"no route {path}"}})


def build_api[T: SpotifyApi](
    fake: FakeSpotify, tmp_path: Path, settings: SpotifyConfig, api_type: type[T]
) -> T:
    """Build a client over `fake` with a valid stored login, no network and no real sleeping."""
    token_file = tmp_path / f"token-{len(list(tmp_path.glob('token-*.json')))}.json"
    write_private_json(
        token_file,
        {"access_token": "a", "refresh_token": "r", "expires_at": time.time() + 3600},
    )
    http = httpx.Client(base_url=API, transport=httpx.MockTransport(fake.handle))
    return api_type(
        TokenStore("client", token_file),
        http,
        sleep=fake.clock.sleep,
        clock=fake.clock,
        library_ttl_s=settings.library_refresh_s,
    )


def build_ops(
    fake: FakeSpotify,
    tmp_path: Path,
    config: SpotifyConfig | None = None,
    *,
    seeded: bool = True,
    api: SpotifyApi | None = None,
) -> SpotifyOps:
    """Build the operations over `fake`: no network, no real launcher, no real sleeping.

    The song order is seeded unless `seeded` is false (then it is the default entropy source).
    `api` lets a test swap in a client that fails on purpose.
    """
    settings = config or SpotifyConfig(device_name="pc")
    return SpotifyOps(
        api or build_api(fake, tmp_path, settings, SpotifyApi),
        settings,
        launcher=fake.launch,
        # ruff: ignore[suspicious-non-cryptographic-random-usage] - song order
        rng=random.Random(7) if seeded else None,
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

    result = build_ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert result.ok
    assert fake.played == [{"uris": ["spotify:track:lowtide"]}]
    assert fake.searches == [("low tide", "track")]
    assert not fake.calls("GET", "/me/top/artists")


def test_a_song_by_an_artist_searches_with_both_filters(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]})

    build_ops(fake, tmp_path).play({"query": "Low Tide", "kind": "track", "artist": "Glass Harbor"})

    assert fake.searches == [('track:"Low Tide" artist:"Glass Harbor"', "track")]
    assert fake.played == [{"uris": ["spotify:track:lowtide"]}]


def test_double_quotes_are_removed_from_the_filter_values(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]})

    build_ops(fake, tmp_path).play(
        {"query": 'Low "Tide"', "kind": "track", "artist": 'Glass "Harbor'}
    )

    assert fake.searches == [('track:"Low Tide" artist:"Glass Harbor"', "track")]


def test_an_empty_filtered_search_falls_back_to_plain_text(tmp_path: Path) -> None:
    fake = FakeSpotify(
        by_query={
            ('track:"Low Tide" artist:"Glass Harbor"', "track"): [],
            ("Low Tide Glass Harbor", "track"): [LOW_TIDE],
        }
    )

    result = build_ops(fake, tmp_path).play(
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

    build_ops(fake, tmp_path).play({"query": "Low Tide", "kind": "track", "artist": "Glass Harbor"})

    assert not fake.calls("GET", "/me/top/artists")
    assert not fake.calls("GET", "/me/top/tracks")


def test_every_hit_is_examined_not_only_the_first(tmp_path: Path) -> None:
    hits = [
        _track("Low Tide Remixed Megamix", "DJ Nobody", "spotify:track:mix"),
        _track("Slow Tide", "Other", "spotify:track:slow"),
        LOW_TIDE,
    ]
    fake = FakeSpotify(search={"track": hits})

    build_ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert fake.played == [{"uris": ["spotify:track:lowtide"]}]


def test_an_exact_title_beats_a_closer_spelling_and_ties_keep_api_order(tmp_path: Path) -> None:
    first = _track("Low Tide", "A", "spotify:track:first")
    second = _track("Low Tide", "B", "spotify:track:second")
    fake = FakeSpotify(
        search={"track": [_track("Low Tides", "C", "spotify:track:near"), first, second]}
    )

    build_ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert fake.played == [{"uris": ["spotify:track:first"]}]


def test_the_requested_artist_decides_between_equal_titles(tmp_path: Path) -> None:
    cover = _track("Low Tide", "Cover Band", "spotify:track:cover")
    fake = FakeSpotify(search={"track": [cover, LOW_TIDE]})

    build_ops(fake, tmp_path).play(
        {"query": "Low Tide", "kind": "track", "artist": "glass harbour"}
    )

    assert fake.played == [{"uris": ["spotify:track:lowtide"]}]


def test_a_well_matched_title_by_another_artist_is_still_played(tmp_path: Path) -> None:
    # The artist may be misheard beyond recognition; the title matched, so play it.
    fake = FakeSpotify(search={"track": [_track("Low Tide", "Cover Band", "spotify:track:cover")]})

    result = build_ops(fake, tmp_path).play(
        {"query": "Low Tide", "kind": "track", "artist": "Night Circuit"}
    )

    assert result.ok
    assert fake.played == [{"uris": ["spotify:track:cover"]}]


def test_a_remastered_version_still_counts_as_the_title(tmp_path: Path) -> None:
    remaster = _track("Low Tide - Remastered 2011", "Glass Harbor", "spotify:track:remaster")
    fake = FakeSpotify(
        search={"track": [remaster]}, top_tracks=[_track("Low Tides", "X", "spotify:track:x")]
    )

    build_ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert fake.played == [{"uris": ["spotify:track:remaster"]}]
    assert not fake.calls("GET", "/me/top/tracks")


def test_misheard_artist_asked_as_track_plays_the_owners_artist(tmp_path: Path) -> None:
    # The owner said "play a Glass Harbor song"; speech-to-text heard "grass harbor" and the
    # model asked for a track. Spotify's top track hit is an unrelated song.
    fake = FakeSpotify(
        search={"track": [_track("Harbor Lights", "Paper Moons", "spotify:track:moons")]},
        top_artists=[GLASS_HARBOR],
    )

    build_ops(fake, tmp_path).play({"query": "grass harbor", "kind": "track"})

    assert fake.played == [{"context_uri": "spotify:artist:glassharbor"}]


def test_misheard_track_falls_back_to_owners_track(tmp_path: Path) -> None:
    fake = FakeSpotify(
        search={"track": [_track("Something Else", "Other", "spotify:track:other")]},
        top_tracks=[_track("Second Sun", "Night Circuit", "spotify:track:sun")],
    )

    build_ops(fake, tmp_path).play({"query": "secund sun", "kind": "track"})

    assert fake.played == [{"uris": ["spotify:track:sun"]}]


def test_owners_playlist_is_preferred(tmp_path: Path) -> None:
    fake = FakeSpotify(
        playlists=[{"name": "2031 NEON TASTE", "uri": "spotify:playlist:mine"}],
        search={"playlist": [{"name": "Neon Classics", "uri": "spotify:playlist:public"}]},
    )

    build_ops(fake, tmp_path).play({"query": "2031 neon taste", "kind": "playlist"})

    assert fake.played == [{"context_uri": "spotify:playlist:mine"}]
    assert not fake.calls("GET", "/search")


def test_a_public_playlist_is_played_when_the_owner_has_none_like_it(tmp_path: Path) -> None:
    fake = FakeSpotify(
        playlists=[{"name": "Gym", "uri": "spotify:playlist:mine"}],
        search={"playlist": [{"name": "Rainy Day Jazz", "uri": "spotify:playlist:public"}]},
    )

    build_ops(fake, tmp_path).play({"query": "rainy day jazz", "kind": "playlist"})

    assert fake.played == [{"context_uri": "spotify:playlist:public"}]


def test_my_playlist_never_falls_back_to_a_public_one(tmp_path: Path) -> None:
    fake = FakeSpotify(
        playlists=[{"name": "Gym", "uri": "spotify:playlist:mine"}],
        search={"playlist": [{"name": "Rainy Day Jazz", "uri": "spotify:playlist:public"}]},
    )

    result = build_ops(fake, tmp_path).play(
        {"query": "rainy day jazz", "kind": "playlist", "mine": True}
    )

    assert not result.ok
    assert result.text == "You don't have a playlist called 'rainy day jazz'."
    assert result.origin is Origin.TOOL_STATUS
    assert fake.played == []
    assert not fake.calls("GET", "/search")


def test_my_playlist_accepts_a_loose_spelling_of_the_owners_own(tmp_path: Path) -> None:
    fake = FakeSpotify(playlists=[{"name": "Morning Run Mix", "uri": "spotify:playlist:mine"}])

    build_ops(fake, tmp_path).play({"query": "morning ron", "kind": "playlist", "mine": True})

    assert fake.played == [{"context_uri": "spotify:playlist:mine"}]


def test_liked_songs_play_the_owners_saved_tracks_shuffled(tmp_path: Path) -> None:
    saved = [_track(f"Song {i}", "Night Circuit", f"spotify:track:{i}") for i in range(30)]
    fake = FakeSpotify(saved=saved)

    result = build_ops(fake, tmp_path).play({"query": "my liked songs", "kind": "liked"})

    assert result.ok
    (body,) = fake.played
    assert sorted(body["uris"]) == sorted(t["uri"] for t in saved)
    assert body["uris"] != [t["uri"] for t in saved]
    assert fake.calls("GET", "/me/tracks")[0].url.params["limit"] == "50"
    assert not fake.calls("GET", "/search")


def test_liked_songs_with_nothing_saved_say_so(tmp_path: Path) -> None:
    fake = FakeSpotify()

    result = build_ops(fake, tmp_path).play({"query": "liked songs", "kind": "liked"})

    assert not result.ok
    assert "saved" in result.text
    assert fake.played == []


def test_nothing_found_reports_failure(tmp_path: Path) -> None:
    fake = FakeSpotify()

    result = build_ops(fake, tmp_path).play({"query": "zzzz", "kind": "track"})

    assert not result.ok
    assert fake.played == []


def test_a_found_item_reports_with_spotifys_origin(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]})

    result = build_ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert result.origin is Origin.SPOTIFY
    assert "Low Tide" in result.text


def _catalogue(n: int, artist: str = "Night Circuit") -> list[dict[str, Any]]:
    return [_track(f"Song {i}", artist, f"spotify:track:{i}") for i in range(n)]


def test_queue_one_track_by_default(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(3)})

    build_ops(fake, tmp_path).queue({"query": "song 0"})

    assert fake.queued == ["spotify:track:0"]


def test_queue_several_distinct_tracks_across_search_pages(tmp_path: Path) -> None:
    # "queue up 14 Night Circuit songs": Spotify returns at most 10 results per page.
    fake = FakeSpotify(search={"track": _catalogue(24)})

    result = build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 14})

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

    build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 3})

    assert fake.queued == ["spotify:track:a", "spotify:track:c", "spotify:track:e"]


def test_queue_in_random_order_is_a_shuffled_pick(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(30)})

    build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 10, "random_order": True})

    assert len(set(fake.queued)) == 10
    assert fake.queued != [f"spotify:track:{i}" for i in range(10)]
    assert max(int(uri.rsplit(":", 1)[1]) for uri in fake.queued) >= 20  # drawn from the wider pool


def test_random_order_is_not_fixed_by_default(tmp_path: Path) -> None:
    # Two default-built players must not pick the same ten of forty songs in the same order;
    # with a fixed seed they would. A false failure is about 1 in 10**15.
    orders: list[list[str]] = []
    for _run in range(2):
        fake = FakeSpotify(search={"track": _catalogue(40)})
        ops = build_ops(fake, tmp_path, seeded=False)
        ops.queue({"query": "Night Circuit", "count": 10, "random_order": True})
        orders.append(fake.queued)
    assert orders[0] != orders[1]


def test_queue_reports_fewer_when_spotify_has_fewer(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(4)})

    result = build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 10})

    assert result.ok
    assert len(fake.queued) == 4


def test_queue_pages_by_what_spotify_actually_returned(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(12)}, page_size=3)

    build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 7})

    assert fake.queued == [f"spotify:track:{i}" for i in range(7)]


def test_a_playlist_made_after_the_cache_is_found_without_waiting(tmp_path: Path) -> None:
    fake = FakeSpotify(playlists=[{"name": "Gym", "uri": "spotify:playlist:gym"}])
    ops = build_ops(fake, tmp_path)
    ops.play({"query": "gym", "kind": "playlist", "mine": True})

    fake.playlists.append({"name": "Sunday Chores", "uri": "spotify:playlist:chores"})
    result = ops.play({"query": "sunday chores", "kind": "playlist", "mine": True})

    assert result.ok
    assert fake.played[-1] == {"context_uri": "spotify:playlist:chores"}


def test_playlists_are_not_fetched_again_for_every_request(tmp_path: Path) -> None:
    fake = FakeSpotify(playlists=[{"name": "Gym", "uri": "spotify:playlist:gym"}])
    ops = build_ops(fake, tmp_path)

    ops.play({"query": "gym", "kind": "playlist", "mine": True})
    ops.play({"query": "gym", "kind": "playlist", "mine": True})

    assert len(fake.calls("GET", "/me/playlists")) == 1


def spotify_error(
    status: int, reason: str, message: str = "Player command failed"
) -> httpx.Response:
    return httpx.Response(
        status, json={"error": {"status": status, "message": message, "reason": reason}}
    )


def no_device() -> httpx.Response:
    return spotify_error(404, "NO_ACTIVE_DEVICE", "Player command failed: No active device found")


def inactive_pc(name: str = "pc", device_id: str = "d1") -> dict[str, Any]:
    return {"id": device_id, "name": name, "type": "Computer", "is_active": False}


def playing_state(
    *, playing: bool = True, volume: int | None = 40, supports_volume: bool = True
) -> dict[str, Any]:
    return {
        "is_playing": playing,
        "currently_playing_type": "track",
        "item": LOW_TIDE,
        "device": {**PC, "volume_percent": volume, "supports_volume": supports_volume},
        "shuffle_state": False,
    }


# --- Where playback goes: devices, transfer and launching (4e, 4f) ---


def test_play_without_a_configured_device_sends_the_command_at_once(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]})

    result = build_ops(fake, tmp_path, SpotifyConfig()).play({"query": "low tide", "kind": "track"})

    assert result.ok
    assert fake.played == [{"uris": ["spotify:track:lowtide"]}]
    (request,) = fake.calls("PUT", "/me/player/play")
    assert "device_id" not in request.url.params
    assert not fake.calls("GET", "/me/player/devices")
    assert not fake.transfers
    assert result.origin is Origin.SPOTIFY
    assert "pc" not in result.text


def test_a_listed_but_inactive_device_gets_playback_transferred_then_the_command(
    tmp_path: Path,
) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]}, devices=[inactive_pc()])
    fake.script("PUT", "/me/player/play", no_device())

    result = build_ops(fake, tmp_path, SpotifyConfig()).play({"query": "low tide", "kind": "track"})

    assert result.ok
    assert fake.transfers == [{"device_ids": ["d1"], "play": False}]
    first, second = fake.calls("PUT", "/me/player/play")
    assert "device_id" not in first.url.params
    assert second.url.params["device_id"] == "d1"
    assert fake.played[-1] == {"uris": ["spotify:track:lowtide"]}
    assert fake.launches == []


def test_play_launches_spotify_when_no_device_is_listed_then_transfers(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]}, devices=[], appears_on_launch=[inactive_pc()])
    fake.script("PUT", "/me/player/play", no_device())

    result = build_ops(fake, tmp_path, SpotifyConfig()).play({"query": "low tide", "kind": "track"})

    assert result.ok
    assert fake.launches == [1]
    assert fake.transfers == [{"device_ids": ["d1"], "play": False}]
    assert fake.calls("PUT", "/me/player/play")[-1].url.params["device_id"] == "d1"


def test_play_says_so_when_no_player_ever_comes_up(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]}, devices=[])
    fake.script("PUT", "/me/player/play", no_device())

    with pytest.raises(NoDeviceError, match="no Spotify player came up"):
        build_ops(fake, tmp_path, SpotifyConfig()).play({"query": "low tide", "kind": "track"})

    assert fake.launches == [1]
    assert fake.clock.now >= 1000.0 + DEVICE_WAIT_S


def test_a_configured_device_is_looked_up_first_and_targeted(tmp_path: Path) -> None:
    phone = {"id": "p1", "name": "phone", "type": "Smartphone", "is_active": False}
    fake = FakeSpotify(search={"track": [LOW_TIDE]}, devices=[phone, dict(PC)])

    build_ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    (request,) = fake.calls("PUT", "/me/player/play")
    assert request.url.params["device_id"] == "d1"
    assert not fake.transfers


def test_a_configured_device_that_is_inactive_gets_a_transfer_first(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]}, devices=[inactive_pc()])

    build_ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert fake.transfers == [{"device_ids": ["d1"], "play": False}]
    assert fake.calls("PUT", "/me/player/play")[0].url.params["device_id"] == "d1"
    assert fake.launches == []


def test_a_configured_device_that_is_missing_is_launched_for_and_waited_for(
    tmp_path: Path,
) -> None:
    phone = {"id": "p1", "name": "phone", "type": "Smartphone", "is_active": True}
    tablet = {"id": "t1", "name": "tablet", "type": "Tablet", "is_active": False}
    fake = FakeSpotify(
        search={"track": [LOW_TIDE]}, devices=[phone, tablet], appears_on_launch=[inactive_pc()]
    )

    build_ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert fake.launches == [1]
    assert fake.transfers == [{"device_ids": ["d1"], "play": False}]
    assert fake.calls("PUT", "/me/player/play")[0].url.params["device_id"] == "d1"


def test_a_configured_device_that_never_appears_is_reported_not_replaced(tmp_path: Path) -> None:
    phone = {"id": "p1", "name": "phone", "type": "Smartphone", "is_active": True}
    tablet = {"id": "t1", "name": "tablet", "type": "Tablet", "is_active": False}
    fake = FakeSpotify(search={"track": [LOW_TIDE]}, devices=[phone, tablet])

    with pytest.raises(NoDeviceError, match="no player called 'pc'") as caught:
        build_ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert caught.value.origin is Origin.TOOL_STATUS
    assert fake.played == []


def test_a_configured_device_that_is_missing_is_replaced_only_by_the_only_device(
    tmp_path: Path,
) -> None:
    phone = {"id": "p1", "name": "phone", "type": "Smartphone", "is_active": True}
    fake = FakeSpotify(search={"track": [LOW_TIDE]}, devices=[phone])

    result = build_ops(fake, tmp_path).play({"query": "low tide", "kind": "track"})

    assert result.ok
    assert fake.launches == []
    assert fake.calls("PUT", "/me/player/play")[0].url.params["device_id"] == "p1"


def test_liked_songs_start_the_same_way(tmp_path: Path) -> None:
    fake = FakeSpotify(saved=[LOW_TIDE], devices=[inactive_pc()])
    fake.script("PUT", "/me/player/play", no_device())

    result = build_ops(fake, tmp_path, SpotifyConfig()).play({"query": "liked", "kind": "liked"})

    assert result.ok
    assert fake.transfers == [{"device_ids": ["d1"], "play": False}]
    assert fake.played[-1] == {"uris": ["spotify:track:lowtide"]}


@pytest.mark.parametrize(
    ("method", "path", "run"),
    [
        ("PUT", "/me/player/pause", "pause"),
        ("POST", "/me/player/next", "next"),
        ("POST", "/me/player/previous", "previous"),
    ],
)
def test_pause_next_and_previous_never_launch_or_look_up_devices(
    tmp_path: Path, method: str, path: str, run: str
) -> None:
    fake = FakeSpotify(devices=[])
    fake.script(method, path, no_device())
    ops = {op.name: op for op in build_ops(fake, tmp_path, SpotifyConfig()).operations()}

    result = ops[f"spotify_{run}"].run({})

    assert result.ok
    assert result.text == "Nothing is playing."
    assert fake.launches == []
    assert not fake.calls("GET", "/me/player/devices")
    assert not fake.transfers


def test_resume_transfers_to_an_inactive_device_and_starts_it(tmp_path: Path) -> None:
    fake = FakeSpotify(devices=[inactive_pc()])
    fake.script("PUT", "/me/player/play", no_device())
    ops = {op.name: op for op in build_ops(fake, tmp_path, SpotifyConfig()).operations()}

    result = ops["spotify_resume"].run({})

    assert result.ok
    assert fake.transfers == [{"device_ids": ["d1"], "play": True}]
    assert len(fake.calls("PUT", "/me/player/play")) == 1
    assert fake.launches == []


def test_resume_may_launch_spotify(tmp_path: Path) -> None:
    fake = FakeSpotify(devices=[], appears_on_launch=[inactive_pc()])
    fake.script("PUT", "/me/player/play", no_device())
    ops = {op.name: op for op in build_ops(fake, tmp_path, SpotifyConfig()).operations()}

    result = ops["spotify_resume"].run({})

    assert result.ok
    assert fake.launches == [1]
    assert fake.transfers == [{"device_ids": ["d1"], "play": True}]


@pytest.mark.parametrize("name", ["spotify_shuffle", "spotify_repeat", "spotify_set_volume"])
def test_settings_commands_do_not_launch_spotify(tmp_path: Path, name: str) -> None:
    fake = FakeSpotify(devices=[])
    for method, path in [
        ("PUT", "/me/player/shuffle"),
        ("PUT", "/me/player/repeat"),
        ("PUT", "/me/player/volume"),
    ]:
        fake.script(method, path, no_device())
    ops = {op.name: op for op in build_ops(fake, tmp_path, SpotifyConfig()).operations()}
    args = {"spotify_shuffle": {"on": True}, "spotify_repeat": {"mode": "off"}}.get(
        name, {"percent": 30}
    )

    with pytest.raises(NoDeviceError):
        ops[name].run(args)

    assert fake.launches == []


def test_a_setting_on_an_inactive_device_transfers_then_retries(tmp_path: Path) -> None:
    fake = FakeSpotify(devices=[inactive_pc()])
    fake.script("PUT", "/me/player/shuffle", no_device())
    ops = {op.name: op for op in build_ops(fake, tmp_path, SpotifyConfig()).operations()}

    result = ops["spotify_shuffle"].run({"on": True})

    assert result.ok
    assert fake.transfers == [{"device_ids": ["d1"], "play": False}]
    assert fake.calls("PUT", "/me/player/shuffle")[-1].url.params["device_id"] == "d1"
    assert fake.launches == []


# --- Already paused, already playing (4g) ---


def find_op(ops: SpotifyOps, name: str) -> Operation:
    return next(op for op in ops.operations() if op.name == name)


def test_pausing_what_is_already_paused_is_a_success(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(playing=False))
    fake.script("PUT", "/me/player/pause", spotify_error(403, "UNKNOWN", "Restriction violated"))

    result = find_op(build_ops(fake, tmp_path), "spotify_pause").run({})

    assert result.ok
    assert result.text == "It's already paused."
    assert result.origin is Origin.TOOL_STATUS
    assert len(fake.calls("GET", "/me/player")) == 1


def test_resuming_what_is_already_playing_is_a_success(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(playing=True))
    fake.script("PUT", "/me/player/play", spotify_error(403, "UNKNOWN", "Restriction violated"))

    result = find_op(build_ops(fake, tmp_path), "spotify_resume").run({})

    assert result.ok
    assert result.text == "It's already playing."


def test_a_refusal_that_the_state_does_not_explain_keeps_its_error(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(playing=True))
    fake.script(
        "PUT", "/me/player/pause", spotify_error(403, "PREMIUM_REQUIRED", "Premium required")
    )

    with pytest.raises(SpotifyError, match="Premium required"):
        find_op(build_ops(fake, tmp_path), "spotify_pause").run({})


def test_pausing_with_no_player_state_after_a_refusal_says_nothing_is_playing(
    tmp_path: Path,
) -> None:
    fake = FakeSpotify(state=None)
    fake.script("PUT", "/me/player/pause", spotify_error(403, "UNKNOWN", "Restriction violated"))

    result = find_op(build_ops(fake, tmp_path), "spotify_pause").run({})

    assert result.ok
    assert result.text == "Nothing is playing."


def test_resuming_while_paused_with_a_refusal_keeps_the_error(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(playing=False))
    fake.script("PUT", "/me/player/play", spotify_error(403, "UNKNOWN", "Restriction violated"))

    with pytest.raises(SpotifyError, match="Restriction violated"):
        find_op(build_ops(fake, tmp_path), "spotify_resume").run({})


# --- What is playing: ads, podcasts, a missing item (4h) ---

EPISODE = {
    "type": "episode",
    "name": "Quiet Hours",
    "uri": "spotify:episode:quiet",
    "show": {"name": "Evening Notes"},
}


def test_an_ad_is_named_as_an_ad(tmp_path: Path) -> None:
    fake = FakeSpotify(state={**playing_state(), "item": None, "currently_playing_type": "ad"})

    result = find_op(build_ops(fake, tmp_path), "spotify_now_playing").run({})

    assert result.ok
    assert result.text == "An ad is playing."


def test_a_podcast_episode_is_named(tmp_path: Path) -> None:
    fake = FakeSpotify(
        state={**playing_state(), "item": EPISODE, "currently_playing_type": "episode"}
    )

    result = find_op(build_ops(fake, tmp_path), "spotify_now_playing").run({})

    assert result.ok
    assert "Quiet Hours" in result.text
    assert "Evening Notes" in result.text
    assert result.origin is Origin.SPOTIFY


def test_a_state_without_an_item_says_what_is_known(tmp_path: Path) -> None:
    fake = FakeSpotify(state={**playing_state(), "item": None, "currently_playing_type": "unknown"})

    result = find_op(build_ops(fake, tmp_path), "spotify_now_playing").run({})

    assert result.ok
    assert "playing" in result.text
    assert "Nothing is playing" not in result.text


def test_with_no_player_state_nothing_is_playing(tmp_path: Path) -> None:
    result = find_op(build_ops(FakeSpotify(), tmp_path), "spotify_now_playing").run({})

    assert result.ok
    assert result.text == "Nothing is playing."


def test_now_playing_describes_a_track(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state())

    result = find_op(build_ops(fake, tmp_path), "spotify_now_playing").run({})

    assert result.origin is Origin.SPOTIFY
    assert "Low Tide by Glass Harbor" in result.text
    assert "volume 40%" in result.text
    assert fake.calls("GET", "/me/player")[0].url.params["additional_types"] == "episode"


def test_like_saves_the_playing_track(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state())

    result = find_op(build_ops(fake, tmp_path), "spotify_like").run({})

    assert result.ok
    (request,) = fake.calls("PUT", "/me/library")
    assert request.url.params["uris"] == "spotify:track:lowtide"
    assert "Low Tide" in result.text


@pytest.mark.parametrize(
    ("state", "said"),
    [
        (None, "Nothing is playing."),
        ({"item": None, "currently_playing_type": "ad", "is_playing": True}, "An ad is playing."),
        ({"item": EPISODE, "currently_playing_type": "episode", "is_playing": True}, "podcast"),
        ({"item": None, "currently_playing_type": "unknown", "is_playing": True}, "can't tell"),
    ],
)
def test_like_declines_what_is_not_a_song(
    tmp_path: Path, state: dict[str, Any] | None, said: str
) -> None:
    fake = FakeSpotify(state=state)

    with pytest.raises(OperationError, match=said) as caught:
        find_op(build_ops(fake, tmp_path), "spotify_like").run({})

    assert caught.value.origin is Origin.TOOL_STATUS
    assert not fake.calls("PUT", "/me/library")


# --- Volume (4i) ---


def test_change_volume_clamps_to_the_range(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(volume=95))

    result = find_op(build_ops(fake, tmp_path), "spotify_change_volume").run({"delta": 10})

    assert result.ok
    (request,) = fake.calls("PUT", "/me/player/volume")
    assert request.url.params["volume_percent"] == "100"


def test_change_volume_cannot_go_below_zero(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(volume=5))

    find_op(build_ops(fake, tmp_path), "spotify_change_volume").run({"delta": -50})

    assert fake.calls("PUT", "/me/player/volume")[0].url.params["volume_percent"] == "0"


def test_change_volume_says_so_when_the_volume_is_unknown(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(volume=None))

    with pytest.raises(OperationError, match="can't read the current volume"):
        find_op(build_ops(fake, tmp_path), "spotify_change_volume").run({"delta": 10})

    assert not fake.calls("PUT", "/me/player/volume")


def test_a_device_without_volume_control_is_refused_without_sending(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(supports_volume=False))

    with pytest.raises(OperationError, match="doesn't allow volume control"):
        find_op(build_ops(fake, tmp_path), "spotify_change_volume").run({"delta": 10})

    assert not fake.calls("PUT", "/me/player/volume")


def test_a_volume_refusal_from_spotify_gets_the_same_answer(tmp_path: Path) -> None:
    fake = FakeSpotify()
    fake.script(
        "PUT",
        "/me/player/volume",
        spotify_error(
            403, "VOLUME_CONTROL_DISALLOW", "Player command failed: Cannot control device volume"
        ),
    )

    result = find_op(build_ops(fake, tmp_path), "spotify_set_volume").run({"percent": 30})

    assert not result.ok
    assert result.text == "This device doesn't allow volume control from here."
    assert result.origin is Origin.TOOL_STATUS


def test_another_volume_refusal_keeps_its_error(tmp_path: Path) -> None:
    fake = FakeSpotify()
    fake.script(
        "PUT", "/me/player/volume", spotify_error(403, "PREMIUM_REQUIRED", "Premium required")
    )

    with pytest.raises(SpotifyError, match="Premium required"):
        find_op(build_ops(fake, tmp_path), "spotify_set_volume").run({"percent": 30})


# --- The queue: no song twice (5) ---

ALREADY_QUEUED = "That's already in the queue."


def test_asking_for_the_same_song_twice_in_a_row_queues_it_once(tmp_path: Path) -> None:
    # Spotify's queue read can lag or be cut short; Jarvis remembers what it queued itself.
    fake = FakeSpotify(search={"track": [LOW_TIDE]})
    ops = build_ops(fake, tmp_path)

    first = ops.queue({"query": "low tide"})
    second = ops.queue({"query": "low tide"})

    assert first.ok
    assert "Low Tide" in first.text
    assert second.ok
    assert second.text == ALREADY_QUEUED
    assert second.origin is Origin.TOOL_STATUS
    assert fake.queued == ["spotify:track:lowtide"]


def test_a_later_request_inside_half_an_hour_is_still_a_repeat(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]})
    ops = build_ops(fake, tmp_path)
    ops.queue({"query": "low tide"})

    fake.clock.now += 29 * 60
    again = ops.queue({"query": "low tide"})

    assert again.text == ALREADY_QUEUED
    assert fake.queued == ["spotify:track:lowtide"]


def test_after_half_an_hour_the_song_may_be_queued_again(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]})
    ops = build_ops(fake, tmp_path)
    ops.queue({"query": "low tide"})

    fake.clock.now += 31 * 60
    again = ops.queue({"query": "low tide"})

    assert again.ok
    assert fake.queued == ["spotify:track:lowtide"] * 2


def test_a_song_already_in_spotifys_queue_is_not_added_again(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]}, queue_items=[LOW_TIDE])

    result = build_ops(fake, tmp_path).queue({"query": "low tide"})

    assert result.ok
    assert result.text == ALREADY_QUEUED
    assert fake.queued == []
    assert len(fake.calls("GET", "/me/player/queue")) == 1


def test_the_song_playing_now_is_not_queued_again(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]}, playing_item=LOW_TIDE)

    result = build_ops(fake, tmp_path).queue({"query": "low tide"})

    assert result.text == ALREADY_QUEUED
    assert fake.queued == []


@pytest.mark.parametrize(
    "version",
    [
        "Low Tide (Live at Harbor Hall)",
        "Low Tide - Live at Harbor Hall",
        "Low Tide (Remastered 2011)",
        "Low Tide - Remastered 2011",
        "Low Tide (2011 Remaster)",
        "Low Tide (feat. Paper Moons)",
        "Low Tide feat. Paper Moons",
        "Low Tide [Live]",
    ],
)
def test_another_version_of_a_queued_song_counts_as_the_same_song(
    tmp_path: Path, version: str
) -> None:
    other = _track(version, "Glass Harbor", "spotify:track:other")
    fake = FakeSpotify(search={"track": [other]}, queue_items=[LOW_TIDE])

    result = build_ops(fake, tmp_path).queue({"query": "low tide"})

    assert result.text == ALREADY_QUEUED
    assert fake.queued == []


def test_versions_of_one_song_in_one_request_count_once(tmp_path: Path) -> None:
    tracks = [
        _track("Low Tide (Live at Harbor Hall)", "Glass Harbor", "spotify:track:live"),
        _track("Low Tide - Remastered 2011", "Glass Harbor", "spotify:track:remaster"),
        _track("Second Sun (feat. Paper Moons)", "Glass Harbor", "spotify:track:sun"),
        _track("Second Sun", "Glass Harbor", "spotify:track:sun2"),
        _track("Paper Boats", "Glass Harbor", "spotify:track:boats"),
    ]
    fake = FakeSpotify(search={"track": tracks})

    build_ops(fake, tmp_path).queue({"query": "Glass Harbor", "count": 3})

    assert fake.queued == ["spotify:track:live", "spotify:track:sun", "spotify:track:boats"]


def test_replacements_come_from_the_pool_so_the_count_is_still_met(tmp_path: Path) -> None:
    catalogue = _catalogue(8)
    fake = FakeSpotify(search={"track": catalogue}, queue_items=[catalogue[1], catalogue[2]])

    result = build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 4})

    assert result.ok
    assert fake.queued == [
        "spotify:track:0",
        "spotify:track:3",
        "spotify:track:4",
        "spotify:track:5",
    ]


def test_a_second_request_for_more_songs_by_the_artist_adds_different_ones(
    tmp_path: Path,
) -> None:
    fake = FakeSpotify(search={"track": _catalogue(12)})
    ops = build_ops(fake, tmp_path)

    ops.queue({"query": "Night Circuit", "count": 4})
    ops.queue({"query": "Night Circuit", "count": 4})

    assert len(fake.queued) == 8
    assert len(set(fake.queued)) == 8


def test_when_every_candidate_is_queued_nothing_is_sent(tmp_path: Path) -> None:
    catalogue = _catalogue(3)
    fake = FakeSpotify(search={"track": catalogue}, queue_items=catalogue)

    result = build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 3})

    assert result.ok
    assert result.text == ALREADY_QUEUED
    assert fake.queued == []


def test_fewer_new_songs_than_asked_says_so(tmp_path: Path) -> None:
    catalogue = _catalogue(4)
    fake = FakeSpotify(search={"track": catalogue}, queue_items=[catalogue[0]])

    result = build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 6})

    assert result.ok
    assert len(fake.queued) == 3
    assert "only 3 new" in result.text
    assert "6 asked" in result.text


def test_an_unreadable_queue_still_leaves_the_memory_of_what_was_added(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]})
    ops = build_ops(fake, tmp_path)
    fake.script("GET", "/me/player/queue", spotify_error(500, "", "boom"))

    first = ops.queue({"query": "low tide"})
    fake.script("GET", "/me/player/queue", spotify_error(500, "", "boom"))
    second = ops.queue({"query": "low tide"})

    assert first.ok
    assert second.text == ALREADY_QUEUED
    assert fake.queued == ["spotify:track:lowtide"]


def test_a_song_whose_queueing_failed_may_be_asked_for_again(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]})
    ops = build_ops(fake, tmp_path)
    fake.script("POST", "/me/player/queue", spotify_error(500, "", "boom"))

    with pytest.raises(SpotifyError):
        ops.queue({"query": "low tide"})
    result = ops.queue({"query": "low tide"})

    assert result.ok
    assert fake.queued == ["spotify:track:lowtide"] * 2


def test_a_single_title_queues_the_best_matching_hit_not_the_first(tmp_path: Path) -> None:
    hits = [
        _track("Low Tide Remixed Megamix", "DJ Nobody", "spotify:track:mix"),
        _track("Slow Tide", "Other", "spotify:track:slow"),
        LOW_TIDE,
    ]
    fake = FakeSpotify(search={"track": hits})

    build_ops(fake, tmp_path).queue({"query": "low tide"})

    assert fake.queued == ["spotify:track:lowtide"]


def test_a_single_song_by_an_artist_uses_the_filtered_search(tmp_path: Path) -> None:
    cover = _track("Low Tide", "Cover Band", "spotify:track:cover")
    fake = FakeSpotify(
        by_query={('track:"Low Tide" artist:"Glass Harbor"', "track"): [cover, LOW_TIDE]}
    )

    build_ops(fake, tmp_path).queue({"query": "Low Tide", "artist": "Glass Harbor"})

    assert fake.searches == [('track:"Low Tide" artist:"Glass Harbor"', "track")]
    assert fake.queued == ["spotify:track:lowtide"]


def test_a_repeat_of_a_single_title_does_not_queue_the_next_best_song(tmp_path: Path) -> None:
    hits = [LOW_TIDE, _track("Low Tides", "Other", "spotify:track:near")]
    fake = FakeSpotify(search={"track": hits}, queue_items=[LOW_TIDE])

    result = build_ops(fake, tmp_path).queue({"query": "low tide"})

    assert result.text == ALREADY_QUEUED
    assert fake.queued == []


def test_queueing_by_artist_keeps_only_that_artists_tracks(tmp_path: Path) -> None:
    own = [_track(f"Song {i}", "Glass Harbor", f"spotify:track:own{i}") for i in range(2)]
    featuring = _track("Duet", "Paper Moons", "spotify:track:duet")
    featuring["artists"].append({"name": "Glass Harbor", "uri": _artist_uri("Glass Harbor")})
    stranger = _track("Glass Harbor Blues", "Harbor Glass Tribute", "spotify:track:tribute")
    fake = FakeSpotify(
        search={"artist": [GLASS_HARBOR]},
        by_query={('artist:"Glass Harbor"', "track"): [stranger, own[0], featuring, own[1]]},
    )

    build_ops(fake, tmp_path).queue({"query": "Glass Harbor", "count": 3})

    assert fake.queued == ["spotify:track:own0", "spotify:track:duet", "spotify:track:own1"]


def test_the_whole_queue_operation_stops_after_thirty_seconds(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(10)})
    fake.clock.per_request = 4.0

    result = build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 10})

    assert 0 < len(fake.queued) < 10
    assert result.ok
    assert f"Queued {len(fake.queued)} of 10" in result.text
    assert "ran out of time" in result.text
    assert result.origin is Origin.SPOTIFY


def test_running_out_of_time_before_anything_is_queued_says_so(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(10)})
    fake.clock.per_request = 40.0

    result = build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 5})

    assert not result.ok
    assert "ran out of time" in result.text
    assert fake.queued == []


def test_a_failure_part_way_reports_what_was_queued(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(6)})
    fake.script(
        "POST",
        "/me/player/queue",
        httpx.Response(204),
        httpx.Response(204),
        spotify_error(500, "", "boom"),
    )

    result = build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 5})

    assert result.ok
    assert result.text.startswith(
        "Queued 2 of 5 (Song 0 by Night Circuit, Song 1 by Night Circuit)"
    )
    assert "Spotify failed" in result.text
    assert result.origin is Origin.SPOTIFY
    assert fake.queued == ["spotify:track:0", "spotify:track:1", "spotify:track:2"]


def test_a_timeout_part_way_says_not_to_retry(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(6)})
    fake.script(
        "POST",
        "/me/player/queue",
        httpx.Response(204),
        httpx.ReadTimeout("slow"),
    )

    result = build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 3})

    assert result.text.startswith("Queued 1 of 3")
    assert "may not have gone through" in result.text
    assert "do not retry" in result.text


def test_a_failure_before_anything_is_queued_is_raised(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(3)})
    fake.script("POST", "/me/player/queue", spotify_error(500, "", "boom"))

    with pytest.raises(SpotifyError):
        build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 2})


def test_a_timeout_on_the_first_song_is_uncertain(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": _catalogue(3)})
    fake.script("POST", "/me/player/queue", httpx.ReadTimeout("slow"))

    with pytest.raises(UncertainOutcomeError):
        build_ops(fake, tmp_path).queue({"query": "Night Circuit", "count": 2})


def test_queueing_on_an_inactive_player_transfers_then_retries(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]}, devices=[inactive_pc()])
    fake.script("POST", "/me/player/queue", no_device())

    result = build_ops(fake, tmp_path, SpotifyConfig()).queue({"query": "low tide"})

    assert result.ok
    assert fake.transfers == [{"device_ids": ["d1"], "play": False}]
    assert fake.calls("POST", "/me/player/queue")[-1].url.params["device_id"] == "d1"
    assert fake.launches == []


def test_queueing_with_no_player_does_not_launch_spotify(tmp_path: Path) -> None:
    fake = FakeSpotify(search={"track": [LOW_TIDE]}, devices=[])
    fake.script("POST", "/me/player/queue", no_device())

    with pytest.raises(NoDeviceError):
        build_ops(fake, tmp_path, SpotifyConfig()).queue({"query": "low tide"})

    assert fake.launches == []


# --- Binding what a confirmation is about (6) ---


def test_like_binds_the_track_that_is_playing_when_it_is_asked_about(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state())
    like = find_op(build_ops(fake, tmp_path), "spotify_like")
    assert like.prepare is not None

    bound = like.prepare({})

    assert bound == {"uri": "spotify:track:lowtide", "name": "Low Tide by Glass Harbor"}
    assert like.spoken_summary(bound) == "save Low Tide by Glass Harbor to your library"


def test_like_saves_exactly_the_track_it_asked_about(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state())
    like = find_op(build_ops(fake, tmp_path), "spotify_like")
    assert like.prepare is not None
    bound = like.prepare({})
    fake.state = {
        **playing_state(),
        "item": _track("Second Sun", "Night Circuit", "spotify:track:sun"),
    }

    result = like.run(bound)

    assert result.ok
    (request,) = fake.calls("PUT", "/me/library")
    assert request.url.params["uris"] == "spotify:track:lowtide"
    assert "Low Tide" in result.text


def test_like_prepare_refuses_when_nothing_is_playing(tmp_path: Path) -> None:
    like = find_op(build_ops(FakeSpotify(), tmp_path), "spotify_like")
    assert like.prepare is not None

    with pytest.raises(OperationError, match=r"^Nothing is playing\.$"):
        like.prepare({})


def test_change_volume_binds_the_level_it_will_set(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(volume=40))
    volume = find_op(build_ops(fake, tmp_path), "spotify_change_volume")
    assert volume.prepare is not None

    bound = volume.prepare({"delta": 10})

    assert bound == {"delta": 10, "current": 40, "target": 50}
    assert volume.spoken_summary(bound) == "change the volume from 40 to 50 percent"


def test_change_volume_sets_the_bound_level_even_if_the_volume_moved(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(volume=40))
    volume = find_op(build_ops(fake, tmp_path), "spotify_change_volume")
    assert volume.prepare is not None
    bound = volume.prepare({"delta": 10})
    fake.state = playing_state(volume=80)

    result = volume.run(bound)

    assert result.ok
    assert fake.calls("PUT", "/me/player/volume")[0].url.params["volume_percent"] == "50"
    assert "from 40 to 50" in result.text


def test_change_volume_binds_a_clamped_level(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(volume=95))
    volume = find_op(build_ops(fake, tmp_path), "spotify_change_volume")
    assert volume.prepare is not None

    assert volume.prepare({"delta": 50})["target"] == 100
    fake.state = playing_state(volume=10)
    assert volume.prepare({"delta": -50})["target"] == 0


def test_change_volume_prepare_refuses_without_a_player_or_a_level(tmp_path: Path) -> None:
    volume = find_op(build_ops(FakeSpotify(), tmp_path), "spotify_change_volume")
    assert volume.prepare is not None
    with pytest.raises(OperationError, match=r"^Nothing is playing\.$"):
        volume.prepare({"delta": 10})

    fake = FakeSpotify(state=playing_state(volume=None))
    volume = find_op(build_ops(fake, tmp_path), "spotify_change_volume")
    assert volume.prepare is not None
    with pytest.raises(OperationError, match="can't read the current volume"):
        volume.prepare({"delta": 10})


# --- Warm-up: the speech hint never waits on Spotify (8) ---


class _Waits(threading.Event):
    """An event whose waits return at once: the first `rounds` say "go on", then "stop"."""

    def __init__(self, rounds: int) -> None:
        super().__init__()
        self.rounds = rounds
        self.waited: list[float | None] = []

    @override
    def wait(self, timeout: float | None = None) -> bool:
        self.waited.append(timeout)
        return len(self.waited) > self.rounds


class _Failing(SpotifyApi):
    """A client whose login check fails on demand."""

    failure: Exception | None = None

    @override
    def ensure_login(self) -> None:
        if self.failure is not None:
            raise self.failure
        super().ensure_login()


def _failing_ops(fake: FakeSpotify, tmp_path: Path) -> tuple[SpotifyOps, _Failing]:
    api = build_api(fake, tmp_path, SpotifyConfig(device_name="pc"), _Failing)
    return build_ops(fake, tmp_path, api=api), api


def test_artist_names_never_touch_the_network(tmp_path: Path) -> None:
    fake = FakeSpotify(top_artists=[GLASS_HARBOR])
    ops = build_ops(fake, tmp_path)

    assert ops.artist_names() == []
    fake.clock.now += SpotifyConfig().library_refresh_s * 3
    assert ops.artist_names() == []

    assert fake.requests == []


def test_a_refresh_loads_the_names_distinct_and_in_order(tmp_path: Path) -> None:
    fake = FakeSpotify(
        top_artists=[GLASS_HARBOR, GLASS_HARBOR, {"name": "Night Circuit", "uri": "u"}]
    )
    ops = build_ops(fake, tmp_path)

    ops.refresh()
    fetched = len(fake.requests)

    assert ops.artist_names() == ["Glass Harbor", "Night Circuit"]
    assert len(fake.requests) == fetched


def test_every_refresh_reloads_even_inside_the_cache_time(tmp_path: Path) -> None:
    fake = FakeSpotify(top_artists=[GLASS_HARBOR])
    ops = build_ops(fake, tmp_path)

    ops.refresh()
    fake.top_artists.append({"name": "Night Circuit", "uri": "u"})
    fake.clock.now += 60
    ops.refresh()

    assert ops.artist_names() == ["Glass Harbor", "Night Circuit"]
    assert len(fake.calls("GET", "/me/top/artists")) == 2


def test_a_failed_refresh_keeps_the_previous_names_and_raises_nothing(tmp_path: Path) -> None:
    fake = FakeSpotify(top_artists=[GLASS_HARBOR])
    ops, api = _failing_ops(fake, tmp_path)
    ops.refresh()

    api.failure = SpotifyError("Spotify is down")
    ops.refresh()

    assert ops.artist_names() == ["Glass Harbor"]


def test_a_lost_login_is_logged_as_a_warning_that_names_the_login_command(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ops, api = _failing_ops(FakeSpotify(top_artists=[GLASS_HARBOR]), tmp_path)
    ops.refresh()

    api.failure = SpotifyLoginExpired("expired")
    with caplog.at_level(logging.WARNING, logger="jarvis.spotify.ops"):
        ops.refresh()

    assert any(
        r.levelno == logging.WARNING and "jarvis spotify-login" in r.getMessage()
        for r in caplog.records
    )
    assert ops.artist_names() == ["Glass Harbor"]


def test_a_refresh_makes_sure_the_login_is_valid_first(tmp_path: Path) -> None:
    fake = FakeSpotify()
    ops, api = _failing_ops(fake, tmp_path)
    api.failure = SpotifyAuthError("no login; run `jarvis spotify-login`")

    ops.refresh()

    assert fake.requests == []


def test_the_refresh_thread_refreshes_now_and_every_period_until_stopped(tmp_path: Path) -> None:
    fake = FakeSpotify(top_artists=[GLASS_HARBOR])
    ops = build_ops(fake, tmp_path)
    stop = _Waits(rounds=2)

    thread = ops.start_refresh(stop)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert thread.daemon
    assert stop.waited == [SpotifyConfig().library_refresh_s] * 3
    assert len(fake.calls("GET", "/me/top/artists")) == 3
    assert ops.artist_names() == ["Glass Harbor"]


def test_the_refresh_thread_does_nothing_when_already_stopped(tmp_path: Path) -> None:
    fake = FakeSpotify(top_artists=[GLASS_HARBOR])
    stop = threading.Event()
    stop.set()

    thread = build_ops(fake, tmp_path).start_refresh(stop)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert fake.requests == []


def test_the_refresh_thread_survives_a_surprise(tmp_path: Path) -> None:
    fake = FakeSpotify(top_artists=[GLASS_HARBOR])
    ops, api = _failing_ops(fake, tmp_path)
    api.failure = RuntimeError("unexpected")
    stop = _Waits(rounds=1)

    thread = ops.start_refresh(stop)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(stop.waited) == 2


def test_a_fuzzy_match_uses_the_warmed_library_even_past_its_time(tmp_path: Path) -> None:
    fake = FakeSpotify(
        search={"track": [_track("Something Else", "Other", "spotify:track:other")]},
        top_tracks=[_track("Second Sun", "Night Circuit", "spotify:track:sun")],
    )
    ops = build_ops(fake, tmp_path)
    ops.refresh()
    fake.clock.now += SpotifyConfig().library_refresh_s + 10

    ops.play({"query": "secund sun", "kind": "track"})

    assert fake.played == [{"uris": ["spotify:track:sun"]}]
    assert len(fake.calls("GET", "/me/top/tracks")) == 1


def test_a_failed_refresh_is_tried_again_after_a_minute_not_a_period(tmp_path: Path) -> None:
    fake = FakeSpotify(top_artists=[GLASS_HARBOR])
    ops, api = _failing_ops(fake, tmp_path)
    api.failure = SpotifyError("Spotify is down")
    stop = _Waits(rounds=1)

    thread = ops.start_refresh(stop)
    thread.join(timeout=5)

    assert stop.waited == [REFRESH_RETRY_S, REFRESH_RETRY_S]
