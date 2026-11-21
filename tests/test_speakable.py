import pytest

from jarvis.voice import MAX_SPOKEN_CHARS, speakable, spoken_question


@pytest.mark.parametrize(
    ("reply", "spoken"),
    [
        ("from the album *Bad Omens*.", "from the album Bad Omens."),
        ("**Paused.**", "Paused."),
        ("Playing `Just Pretend` now.", "Playing Just Pretend now."),
        ("# Now playing\n- Like A Villain", "Now playing. Like A Villain"),
        ("See [the album](https://example.com).", "See the album."),
        ("Playing _The Death of Peace of Mind_.", "Playing The Death of Peace of Mind."),
        ("Volume at 47% 🎵", "Volume at 47%"),
        ("Track snake_case_name stays.", "Track snake_case_name stays."),
    ],
)
def test_markdown_is_not_read_aloud(reply: str, spoken: str) -> None:
    assert speakable(reply) == spoken


def test_a_runaway_reply_is_cut_before_it_is_cleaned() -> None:
    assert len(speakable("word " * 5000)) <= MAX_SPOKEN_CHARS
    assert len(spoken_question("word " * 5000)) <= MAX_SPOKEN_CHARS
