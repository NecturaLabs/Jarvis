"""Recording and scoring a private set of the owner's requests (`jarvis.speechtest`)."""

import json
import random
import stat
from typing import TYPE_CHECKING

import numpy as np
import pytest

from jarvis import speechtest
from jarvis.speechtest import Card

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

ARTISTS = ["Glass Harbor", "Paper Moons", "Night Circuit", "Phlox Daze"]
SPEECH = (np.sin(np.arange(16000) / 5) * 3000).astype(np.int16)


def _record(
    folder: Path,
    to_say: list[Card],
    replies: list[str],
    heard: list[NDArray[np.int16] | None] | None = None,
    transcript: str = "play glass harbor",
) -> int:
    answers = iter(replies)
    captures = iter(heard if heard is not None else [SPEECH] * len(to_say))
    return speechtest.record(
        folder,
        to_say,
        lambda: next(captures),
        lambda _samples: transcript,
        ask=lambda _prompt: next(answers),
        show=lambda _line: None,
    )


def test_about_half_the_cards_name_one_of_the_owners_artists() -> None:
    many = [f"Band {n}" for n in range(40)]

    made = speechtest.cards(many, 20, random.Random(1))  # ruff: ignore[suspicious-non-cryptographic-random-usage] - card order

    assert len(made) == 20
    assert sum(card.kind == "artist" for card in made) == 10
    assert all(card.name in card.prompt for card in made if card.kind == "artist")


def test_without_a_library_every_card_is_an_open_request() -> None:
    made = speechtest.cards([], 6, random.Random(1))  # ruff: ignore[suspicious-non-cryptographic-random-usage] - card order

    assert len(made) == 6
    assert not any(card.kind == "artist" for card in made)


def test_a_confirmed_request_is_saved_privately_with_what_was_said(tmp_path: Path) -> None:
    folder = tmp_path / "set"
    card = Card("artist", "Ask for Glass Harbor.", "Glass Harbor")

    saved = _record(folder, [card], ["", ""])

    assert saved == 1
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    files = sorted(folder.iterdir())
    assert {stat.S_IMODE(path.stat().st_mode) for path in files} == {0o600}
    (entry,) = speechtest.read_set(folder)
    assert entry["said"] == "play glass harbor"  # Enter kept what was heard
    assert entry["name"] == "Glass Harbor"
    assert entry["levels"]["peak_dbfs"] < 0
    assert np.array_equal(speechtest.read_wav(folder / f"{entry['id']}.wav"), SPEECH)


def test_the_owner_types_what_they_said_when_it_was_misheard(tmp_path: Path) -> None:
    _record(tmp_path, [Card("artist", "Ask.", "Phlox Daze")], ["", "play phlox daze"])

    assert speechtest.read_set(tmp_path)[0]["said"] == "play phlox daze"


def test_a_card_that_lets_the_owner_choose_asks_which_name_they_said(tmp_path: Path) -> None:
    _record(tmp_path, [Card("time", "Ask the time.", asks_name=True)], ["", "", "Lisbon"])

    assert speechtest.read_set(tmp_path)[0]["name"] == "Lisbon"


def test_skipped_unheard_and_stopped_requests_are_not_saved(tmp_path: Path) -> None:
    to_say = [Card("free", "Ask.")] * 4

    # Skipped before speaking, nothing heard, skipped after hearing, then stopped.
    saved = _record(tmp_path, to_say, ["s", "", "", "s", "q"], heard=[None, SPEECH])

    assert saved == 0
    assert speechtest.read_set(tmp_path) == []


def test_a_second_session_numbers_on_from_the_first(tmp_path: Path) -> None:
    _record(tmp_path, [Card("free", "Ask.")] * 2, ["", ""] * 2)
    _record(tmp_path, [Card("free", "Ask.")], ["", ""])

    assert [entry["id"] for entry in speechtest.read_set(tmp_path)] == ["0001", "0002", "0003"]


@pytest.mark.parametrize(
    ("said", "heard", "errors"),
    [
        ("play glass harbor", "play glass harbor", 0),
        ("play glass harbor", "play grass harbor", 1),
        ("play glass harbor", "play harbor", 1),
        ("play glass harbor", "please play glass harbor now", 2),
        ("", "noise", 1),
    ],
)
def test_word_errors_count_substitutions_drops_and_additions(
    said: str, heard: str, errors: int
) -> None:
    assert speechtest.word_errors(speechtest.words(said), speechtest.words(heard)) == errors


def test_words_ignore_case_and_punctuation() -> None:
    assert speechtest.words("Play, Glass Harbor!") == ["play", "glass", "harbor"]


def test_a_score_reports_word_errors_names_and_time(tmp_path: Path) -> None:
    _record(
        tmp_path,
        [Card("artist", "Ask.", "Glass Harbor"), Card("player", "Pause.")],
        ["", "", "", "pause the music"],
    )

    results = speechtest.score(tmp_path, lambda _samples: "play grass harbor", lambda _l: None)

    assert [r["name_heard"] for r in results] == [False, None]
    assert [r["errors"] for r in results] == [1, 3]
    text = speechtest.report(results)
    assert "word error rate 66.7% (4 of 6 words)" in text
    assert "names heard 0 of 1" in text
    assert "  player: 1 requests" in text


def test_an_empty_set_reports_that_nothing_is_recorded(tmp_path: Path) -> None:
    assert speechtest.report(speechtest.score(tmp_path, str, lambda _l: None)) == (
        "No requests recorded yet."
    )


@pytest.mark.parametrize(
    ("wins", "losses", "p"), [(0, 0, 1.0), (6, 0, 0.03125), (5, 1, 0.21875), (3, 3, 1.0)]
)
def test_the_sign_test_is_exact(wins: int, losses: int, p: float) -> None:
    assert speechtest.two_sided_p(wins, losses) == pytest.approx(p)


def _result(id_: str, *, name_heard: bool | None, errors: int) -> dict[str, object]:
    return {
        "id": id_,
        "kind": "artist",
        "music": False,
        "heard": "",
        "errors": errors,
        "words": 3,
        "name_heard": name_heard,
        "seconds": 0.5,
    }


def test_two_scores_are_compared_request_by_request() -> None:
    first = [_result(f"{n:04d}", name_heard=False, errors=1) for n in range(6)]
    second = [_result(f"{n:04d}", name_heard=True, errors=0) for n in range(6)]
    second.append(_result("0099", name_heard=True, errors=0))  # not in the first: left out

    text = speechtest.compare(first, second)

    assert "0 heard only by the first, 6 only by the second (p = 0.031" in text
    assert "from the first on 0 requests, from the second on 6 (p = 0.031" in text


def test_scores_of_different_sets_cannot_be_compared() -> None:
    first = [_result("0001", name_heard=True, errors=0)]
    second = [_result("0002", name_heard=True, errors=0)]

    assert speechtest.compare(first, second) == "The two scores share no requests."


def test_a_saved_set_holds_plain_json_lines(tmp_path: Path) -> None:
    _record(tmp_path, [Card("free", "Ask.")], ["", ""])

    (line,) = (tmp_path / speechtest.MANIFEST).read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["kind"] == "free"


def test_an_existing_folder_that_holds_something_else_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "documents"
    folder.mkdir(mode=0o755)
    (folder / "notes.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(speechtest.FolderError, match="not a speech test set"):
        _record(folder, [Card("free", "Ask.")], ["", ""])

    assert stat.S_IMODE(folder.stat().st_mode) == 0o755
    assert sorted(path.name for path in folder.iterdir()) == ["notes.txt"]


def test_an_existing_empty_folder_is_used_as_it_is(tmp_path: Path) -> None:
    folder = tmp_path / "set"
    folder.mkdir(mode=0o750)

    _record(folder, [Card("free", "Ask.")], ["", ""])

    assert stat.S_IMODE(folder.stat().st_mode) == 0o750
    assert len(speechtest.read_set(folder)) == 1


def test_a_recording_left_without_its_line_is_not_overwritten(tmp_path: Path) -> None:
    """A crash between the WAV and its manifest line left an orphan the next session hit."""
    _record(tmp_path, [Card("free", "Ask.")], ["", ""])
    (tmp_path / "0002.wav").write_bytes(b"orphan")

    _record(tmp_path, [Card("free", "Ask.")], ["", ""])

    assert [entry["id"] for entry in speechtest.read_set(tmp_path)] == ["0001", "0003"]
    assert (tmp_path / "0002.wav").read_bytes() == b"orphan"


def test_a_failed_transcription_still_lets_the_owner_type_what_they_said(tmp_path: Path) -> None:
    answers = iter(["", "pause the music"])

    def fail(_samples: object) -> str:
        msg = "voxtype transcribe failed"
        raise speechtest.AudioError(msg)

    saved = speechtest.record(
        tmp_path,
        [Card("player", "Pause.")],
        lambda: SPEECH,
        fail,
        ask=lambda _prompt: next(answers),
        show=lambda _line: None,
    )

    assert saved == 1
    assert speechtest.read_set(tmp_path)[0]["said"] == "pause the music"


def test_end_of_input_stops_the_session_and_keeps_what_was_saved(tmp_path: Path) -> None:
    answers = iter(["", ""])

    def ask(_prompt: str) -> str:
        answer = next(answers, None)
        return _end() if answer is None else answer

    saved = speechtest.record(
        tmp_path,
        [Card("free", "Ask.")] * 3,
        lambda: SPEECH,
        lambda _samples: "hello",
        ask=ask,
        show=lambda _line: None,
    )

    assert saved == 1


def _end() -> str:
    raise EOFError


def test_each_request_notes_which_transcriber_heard_it_first(tmp_path: Path) -> None:
    answers = iter(["", ""])
    speechtest.record(
        tmp_path,
        [Card("free", "Ask.")],
        lambda: SPEECH,
        lambda _samples: "hello",
        ask=lambda _prompt: next(answers),
        show=lambda _line: None,
        heard_with="whisper small.en",
    )

    assert speechtest.read_set(tmp_path)[0]["heard_with"] == "whisper small.en"


def test_a_card_never_carries_terminal_control_characters() -> None:
    made = speechtest.cards(["Glass\x1b[2J Harbor"], 2, random.Random(1))  # ruff: ignore[suspicious-non-cryptographic-random-usage] - card order

    (card,) = (card for card in made if card.kind == "artist")
    assert card.prompt.isprintable()
    assert card.name == "Glass[2J Harbor"


@pytest.mark.parametrize(
    ("name", "heard", "found"),
    [
        ("Eve", "play something even better", False),
        ("Eve", "play eve", True),
        ("Glass Harbor", "play glass harbor now", True),
        ("Glass Harbor", "play glassharbor", True),
        ("Three Days Grace", "play 3 days grace", False),
        ("Guns N' Roses", "play guns n' roses", True),
        ("Don't Stop", "play dont stop", True),
    ],
)
def test_a_name_counts_as_heard_only_as_whole_words(name: str, heard: str, *, found: bool) -> None:
    assert speechtest.name_heard(name, heard) is found


def test_a_folder_left_by_a_session_that_ended_before_its_first_line_is_still_a_set(
    tmp_path: Path,
) -> None:
    (tmp_path / "0001.wav").write_bytes(b"orphan")

    _record(tmp_path, [Card("free", "Ask.")], ["", ""])

    assert [entry["id"] for entry in speechtest.read_set(tmp_path)] == ["0002"]


@pytest.mark.parametrize("score", [{"id": "0001"}, [{"id": "0001"}], ["row"]])
def test_a_score_of_the_wrong_shape_is_refused_with_a_reason(score: object) -> None:
    with pytest.raises(speechtest.ScoreError):
        speechtest.compare(score, [_result("0001", name_heard=True, errors=0)])
