import pytest

from jarvis.voice import MAX_SPOKEN_CHARS, speakable, spoken_question


@pytest.mark.parametrize(
    ("reply", "spoken"),
    [
        ("from the album *Glass Harbor*.", "from the album Glass Harbor."),
        ("**Paused.**", "Paused."),
        ("Playing `Low Tide` now.", "Playing Low Tide now."),
        ("# Now playing\n- Second Sun", "Now playing. Second Sun"),
        ("See [the album](https://example.com).", "See the album."),
        ("Playing _The Quiet Hours_.", "Playing The Quiet Hours."),
        ("Volume at 47% 🎵", "Volume at 47%"),
        ("Track snake_case_name stays.", "Track snake_case_name stays."),
    ],
)
def test_markdown_is_not_read_aloud(reply: str, spoken: str) -> None:
    assert speakable(reply) == spoken


def test_a_runaway_reply_is_cut_before_it_is_cleaned() -> None:
    assert len(speakable("word " * 5000)) <= MAX_SPOKEN_CHARS
    assert len(spoken_question("word " * 5000)) <= MAX_SPOKEN_CHARS


@pytest.mark.parametrize(
    ("reply", "spoken"),
    [
        ("I couldn't go back — the player refused.", "I couldn't go back, the player refused."),
        ("Here's some jazz—tell me if it's wrong.", "Here's some jazz, tell me if it's wrong."),
        ("Usually 5\u201310 minutes.", "Usually 5 to 10 minutes."),
        ("It went up \u2013 slowly.", "It went up, slowly."),
        ("A well-known half-open door.", "A well-known half-open door."),
        ("Queued five songs - starting now.", "Queued five songs, starting now."),
        ("That is 48 - 31, so 17.", "That is 48 - 31, so 17."),
        ("It was fine, \u2014 mostly.", "It was fine, mostly."),
        ("Open 9 a.m.\u20135 p.m.", "Open 9 a.m. to 5 p.m."),
        ("Open 9 a.m. \u2013 5 p.m.", "Open 9 a.m. to 5 p.m."),
        ("Open 9am\u20135pm.", "Open 9am to 5pm."),
        ("Mon\u2013Fri.", "Mon to Fri."),
        ("From 2020\u2013present.", "From 2020 to present."),
        ("It costs $5\u2013$10.", "It costs $5 to $10."),
        ("Lows of \u22125\u2013\u22123 \u00b0C.", "Lows of \u22125 to \u22123 degrees Celsius."),
        ("Lows of \u22125 \u2013 \u22123 \u00b0C.", "Lows of \u22125 to \u22123 degrees Celsius."),
        ("Highs of 5\u00b0 \u2013 10\u00b0.", "Highs of 5 degrees to 10 degrees."),
    ],
)
def test_a_dash_is_a_pause_or_a_range_never_a_run_on(reply: str, spoken: str) -> None:
    assert speakable(reply) == spoken


@pytest.mark.parametrize(
    ("reply", "spoken"),
    [
        ("It's 22 °C and overcast.", "It's 22 degrees Celsius and overcast."),
        ("It's -3°C out.", "It's -3 degrees Celsius out."),
        ("About 70 °F.", "About 70 degrees Fahrenheit."),
        ("Turn it 90° left.", "Turn it 90 degrees left."),
        ("Wind at 12 km/h.", "Wind at 12 kilometres per hour."),
        ("Wind at 1 km/h.", "Wind at 1 kilometre per hour."),
        ("Gusts of 4.5 m/s.", "Gusts of 4.5 metres per second."),
        ("Wind at 8 mph.", "Wind at 8 miles per hour."),
        ("Rain: 0.2 mm.", "Rain: 0.2 millimetres."),
        ("48.4 of 60.4 GB used.", "48.4 of 60.4 gigabytes used."),
        ("It takes 512 MB.", "It takes 512 megabytes."),
        ("Volume at 47%.", "Volume at 47%."),
        ("The MB and km/h signs alone stay.", "The MB and km/h signs alone stay."),
    ],
)
def test_a_unit_sign_after_a_number_is_said_as_its_word(reply: str, spoken: str) -> None:
    assert speakable(reply) == spoken


@pytest.mark.parametrize(
    ("reply", "spoken"),
    [
        ("Read it at https://www.example.com/a/b?c=d.", "Read it at example.com."),
        ("See http://example.org", "See example.org"),
        ("See [the page](https://example.com/x).", "See the page."),
    ],
)
def test_a_web_address_is_said_as_its_site(reply: str, spoken: str) -> None:
    assert speakable(reply) == spoken


def test_a_table_is_read_row_by_row_without_its_rules() -> None:
    table = "| Part | Now |\n|---|:---:|\n| CPU | 13% |\n| Memory | 80% |"

    assert speakable(table) == "Part, Now. CPU, 13%. Memory, 80%"


def test_a_code_fence_is_not_read_aloud() -> None:
    assert speakable("Run this:\n```bash\nls\n```\nThat lists them.") == (
        "Run this: ls. That lists them."
    )


@pytest.mark.parametrize(
    ("reply", "spoken"),
    [
        ("From 20 → 1,000.", "From 20 to 1,000."),
        ("It takes ~5 minutes.", "It takes about 5 minutes."),
    ],
)
def test_a_sign_with_a_plain_word_is_said_as_it(reply: str, spoken: str) -> None:
    assert speakable(reply) == spoken
