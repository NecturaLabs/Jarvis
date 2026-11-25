"""Seeking, moving playback, the queue and devices, and changing the library for what plays."""

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from test_spotify_ops import LOW_TIDE, PC, FakeSpotify, build_ops, find_op, playing_state

from jarvis.operations import Operation, OperationError, validate_args
from jarvis.policy import OpClass, Origin
from jarvis.secretfile import write_private_json
from jarvis.spotify.auth import SCOPES

if TYPE_CHECKING:
    from pathlib import Path

    from jarvis.operations import Result

KITCHEN = {"id": "d2", "name": "Kitchen speaker", "type": "Speaker", "is_active": False}
MINE = {"id": "p1", "name": "Road Trip", "owner": {"id": "me"}, "collaborative": False}
SHARED = {"id": "p2", "name": "Friday Mix", "owner": {"id": "friend"}, "collaborative": True}
THEIRS = {"id": "p3", "name": "Top Hits", "owner": {"id": "spotify"}, "collaborative": False}


def _state(progress_s: int = 60, duration_s: int = 200) -> dict[str, Any]:
    item = {**LOW_TIDE, "duration_ms": duration_s * 1000}
    return {**playing_state(), "item": item, "progress_ms": progress_s * 1000}


def _run(op: Operation, args: dict[str, Any]) -> tuple[Result, dict[str, Any]]:
    checked = validate_args(op, args)
    prepared = op.prepare(checked) if op.prepare else checked
    return op.run(prepared), prepared


def _positions(fake: FakeSpotify) -> list[str]:
    return [r.url.params["position_ms"] for r in fake.calls("PUT", "/me/player/seek")]


# --- seeking ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "position", "summary"),
    [
        ({"to_s": 0}, "0", "restart this track"),
        ({"to_s": 95}, "95000", "jump to 1:35 in this track"),
        ({"by_s": 30}, "90000", "skip ahead 30 seconds"),
        ({"by_s": -90}, "0", "go back 90 seconds"),
        ({"by_s": 600}, "199000", "skip ahead 600 seconds"),
        ({"to_s": 900}, "199000", "jump to 15:00 in this track"),
    ],
)
def test_seeking_is_bound_to_its_position_before_any_question(
    tmp_path: Path, args: dict[str, Any], position: str, summary: str
) -> None:
    fake = FakeSpotify(state=_state())
    seek = find_op(build_ops(fake, tmp_path), "spotify_seek")

    result, prepared = _run(seek, args)

    assert _positions(fake) == [position]
    assert seek.spoken_summary(prepared) == summary
    assert result.ok


def test_a_seek_needs_one_place_to_go(tmp_path: Path) -> None:
    fake = FakeSpotify(state=_state())
    seek = find_op(build_ops(fake, tmp_path), "spotify_seek")
    for args in ({}, {"to_s": 1, "by_s": 1}):
        with pytest.raises(OperationError, match="one of"):
            _run(seek, args)
    assert _positions(fake) == []


def test_seeking_with_nothing_playing_says_so(tmp_path: Path) -> None:
    fake = FakeSpotify(state=None)
    with pytest.raises(OperationError, match="Nothing is playing"):
        _run(find_op(build_ops(fake, tmp_path), "spotify_seek"), {"to_s": 0})


# --- moving playback to a device -------------------------------------------------------------


def test_resume_on_a_named_device_moves_playback_there(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(), devices=[dict(PC), dict(KITCHEN)])
    resume = find_op(build_ops(fake, tmp_path), "spotify_resume")

    result, prepared = _run(resume, {"device": "the kitchen speaker"})

    assert fake.transfers == [{"device_ids": ["d2"], "play": True}]
    assert resume.spoken_summary(prepared) == "move the music to Kitchen speaker"
    assert result.origin is Origin.SPOTIFY
    assert "Kitchen speaker" in result.spoken


def test_a_device_no_one_has_is_refused_with_the_names_there_are(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(), devices=[dict(PC), dict(KITCHEN)])

    with pytest.raises(OperationError, match="Kitchen speaker") as caught:
        _run(find_op(build_ops(fake, tmp_path), "spotify_resume"), {"device": "garage radio"})

    assert caught.value.origin is Origin.SPOTIFY  # device names are Spotify's words
    assert fake.transfers == []


def test_resume_without_a_device_resumes_where_it_plays(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(playing=False))
    resume = find_op(build_ops(fake, tmp_path), "spotify_resume")

    _, prepared = _run(resume, {})

    assert fake.transfers == []
    assert len(fake.calls("PUT", "/me/player/play")) == 1
    assert resume.spoken_summary(prepared) == "resume the music"


# --- what is queued, and the devices ---------------------------------------------------------


def test_now_playing_can_also_say_what_is_queued_and_the_devices(tmp_path: Path) -> None:
    queued = [{**LOW_TIDE, "name": f"Song {n}", "uri": f"spotify:track:{n}"} for n in range(8)]
    fake = FakeSpotify(
        state=playing_state(),
        playing_item=LOW_TIDE,
        queue_items=queued,
        devices=[dict(PC), dict(KITCHEN)],
    )
    now = find_op(build_ops(fake, tmp_path), "spotify_now_playing")

    result, _ = _run(now, {"more": True})

    assert "Up next: Song 0 by Glass Harbor; Song 1" in result.text
    assert "Song 4" in result.text
    assert "Song 5" not in result.text
    assert "Devices: pc (playing now), Kitchen speaker." in result.text
    assert result.origin is Origin.SPOTIFY


def test_now_playing_alone_asks_for_nothing_more(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state())

    _run(find_op(build_ops(fake, tmp_path), "spotify_now_playing"), {})

    assert not fake.calls("GET", "/me/player/queue")
    assert not fake.calls("GET", "/me/player/devices")


# --- changing the library for the song playing -----------------------------------------------


def _with_scopes(fake: FakeSpotify, tmp_path: Path, scopes: str) -> Operation:
    ops = build_ops(fake, tmp_path)
    (token,) = tmp_path.glob("token-*.json")
    stored = json.loads(token.read_text(encoding="utf-8"))
    write_private_json(token, {**stored, "scope": scopes})
    return find_op(ops, "spotify_library")


def test_the_library_changes_need_a_yes(tmp_path: Path) -> None:
    library = find_op(build_ops(FakeSpotify(), tmp_path), "spotify_library")
    assert library.default_class is OpClass.CONFIRM


def test_unlike_removes_the_song_playing_from_liked_songs(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state())
    library = find_op(build_ops(fake, tmp_path), "spotify_library")

    result, prepared = _run(library, {"change": "unlike"})

    (request,) = fake.calls("DELETE", "/me/library")
    assert request.url.params["uris"] == "spotify:track:lowtide"
    assert (
        library.spoken_summary(prepared) == "remove Low Tide by Glass Harbor from your Liked Songs"
    )
    assert "Removed Low Tide" in result.spoken


@pytest.mark.parametrize(("said", "playlist"), [("road trip", "p1"), ("friday mix", "p2")])
def test_adding_to_a_playlist_of_the_owners_binds_it_first(
    tmp_path: Path, said: str, playlist: str
) -> None:
    fake = FakeSpotify(state=playing_state(), playlists=[MINE, SHARED, THEIRS])
    library = _with_scopes(fake, tmp_path, " ".join(SCOPES))

    result, prepared = _run(library, {"change": "add_to_playlist", "playlist": said})

    assert [json.loads(r.content) for r in fake.calls("POST", f"/playlists/{playlist}/items")] == [
        {"uris": ["spotify:track:lowtide"]}
    ]
    assert library.spoken_summary(prepared).startswith(
        "add Low Tide by Glass Harbor to your playlist "
    )
    assert result.origin is Origin.SPOTIFY


def test_a_playlist_someone_else_owns_is_not_offered(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(), playlists=[MINE, THEIRS])
    library = _with_scopes(fake, tmp_path, " ".join(SCOPES))

    with pytest.raises(OperationError, match="Road Trip"):
        _run(library, {"change": "add_to_playlist", "playlist": "top hits"})
    assert not fake.calls("POST", "/playlists/p3/items")


def test_adding_to_a_playlist_needs_its_name(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(), playlists=[MINE])
    with pytest.raises(OperationError, match="which playlist"):
        _run(_with_scopes(fake, tmp_path, " ".join(SCOPES)), {"change": "add_to_playlist"})


def test_a_login_from_before_playlists_could_be_changed_says_to_log_in_again(
    tmp_path: Path,
) -> None:
    fake = FakeSpotify(state=playing_state(), playlists=[MINE])
    old = " ".join(s for s in SCOPES if not s.startswith("playlist-modify"))
    library = _with_scopes(fake, tmp_path, old)

    with pytest.raises(OperationError, match="jarvis spotify-login"):
        _run(library, {"change": "add_to_playlist", "playlist": "road trip"})
    assert not [r for r in fake.requests if r.method == "POST"]


def test_the_login_asks_for_changing_playlists() -> None:
    assert {"playlist-modify-private", "playlist-modify-public"} <= set(SCOPES)


def test_a_refused_change_is_said_plainly(tmp_path: Path) -> None:
    fake = FakeSpotify(state=playing_state(), playlists=[MINE])
    fake.script(
        "POST", "/playlists/p1/items", httpx.Response(403, json={"error": {"message": "nope"}})
    )
    library = _with_scopes(fake, tmp_path, " ".join(SCOPES))

    with pytest.raises(Exception, match="nope"):
        _run(library, {"change": "add_to_playlist", "playlist": "road trip"})
