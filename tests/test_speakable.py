import pytest

from jarvis.sentences import MAX_SPOKEN_CHARS
from jarvis.voice import speakable, spoken_question


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
        ("From 22 \u00b0C \u2013 25 \u00b0C.", "From 22 degrees Celsius to 25 degrees Celsius."),
        ("Open 9 AM \u2013 5 PM.", "Open 9 AM to 5 PM."),
        ("Paris\u2013London takes two hours.", "Paris to London takes two hours."),
        ("Wait\u2013what?", "Wait, what?"),
        ("Wait \u2013what.", "Wait, what."),
        ("That is the end.\u2013Start again.", "That is the end. Start again."),
        ("He left\u2013slowly.", "He left, slowly."),
        ("Is x - y the same as y - x?", "Is x - y the same as y - x?"),
        ("Subtract: a - b.", "Subtract: a - b."),
        ("So x - 5 is zero.", "So x - 5 is zero."),
        ("It is 3 \u2013 2 = 1.", "It is 3 minus 2 = 1."),
        ("It is 3\u20132 = 1.", "It is 3 minus 2 = 1."),
        ("It is (2 + 3) \u2013 1 = 4.", "It is (2 + 3) minus 1 = 4."),
        ("It is 9 \u2013 3 \u2013 2 = 4.", "It is 9 minus 3 minus 2 = 4."),
        ("x \u2013 y = 1.", "x minus y = 1."),
        ("Quick math \u2013 2 + 2 = 4.", "Quick math, 2 + 2 = 4."),
        ("The equation \u2013 y = 2x + 1 is a line.", "The equation, y = 2x + 1 is a line."),
        ("Here is the check \u2013 x = 5.", "Here is the check, x = 5."),
        ("Your balance \u2013 5 = what is left.", "Your balance, 5 = what is left."),
        ("Vitamin C - a great source.", "Vitamin C, a great source."),
        ("Option B - I think better.", "Option B, I think better."),
        ("Take vitamin C \u2013 it helps.", "Take vitamin C, it helps."),
        ("Then x - y = 2.", "Then x - y = 2."),
        ("Then 5 \u2013 10 people came.", "Then 5 to 10 people came."),
        ("5 km \u2013 10 km.", "5 kilometres to 10 kilometres."),
        ("3 GB \u2013 4 GB.", "3 gigabytes to 4 gigabytes."),
        ("12 km/h \u2013 20 km/h.", "12 kilometres per hour to 20 kilometres per hour."),
        ("Gusts of 5 m/s \u2013 9 m/s.", "Gusts of 5 metres per second to 9 metres per second."),
        ("5km\u201310km.", "5 kilometres to 10 kilometres."),
        ("It is 20 \u00b0C \u2013 25 \u00b0C.", "It is 20 degrees Celsius to 25 degrees Celsius."),
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
        ("Reads at 3.5 GB/s.", "Reads at 3.5 gigabytes per second."),
        ("Writes at 1 GB/s.", "Writes at 1 gigabyte per second."),
        ("Copying at 80 MB/s.", "Copying at 80 megabytes per second."),
        ("Pulling 120 KB/s.", "Pulling 120 kilobytes per second."),
        ("Link at 1 Gbit/s.", "Link at 1 gigabit per second."),
        ("Link at 300 Mbit/s.", "Link at 300 megabits per second."),
        ("Reads at 3 GB/second.", "Reads at 3 GB/second."),
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
        ("It takes about ~5 minutes.", "It takes about 5 minutes."),
        ("It takes About ~5 minutes.", "It takes About 5 minutes."),
    ],
)
def test_a_sign_with_a_plain_word_is_said_as_it(reply: str, spoken: str) -> None:
    assert speakable(reply) == spoken


@pytest.mark.parametrize(
    ("reply", "spoken"),
    [
        ("It is 5-10 km away.", "It is 5 to 10 kilometres away."),
        ("A 10-20% chance of rain.", "A 10 to 20% chance of rain."),
        ("Takes 2-3 GB.", "Takes 2 to 3 gigabytes."),
        ("Gusts of 1.5-2 m/s.", "Gusts of 1.5 to 2 metres per second."),
        ("Highs of 20-25\u00b0C.", "Highs of 20 to 25 degrees Celsius."),
        ("Highs of 20-25 \u00b0C.", "Highs of 20 to 25 degrees Celsius."),
        ("It was 2026-11-26.", "It was 2026-11-26."),
        ("Call 555-1234.", "Call 555-1234."),
        ("Version 1.2-3 is out.", "Version 1.2-3 is out."),
        ("The range 10.0.0.1-10.0.0.9 is free.", "The range 10.0.0.1-10.0.0.9 is free."),
        ("The tag 5-10-15 km stays.", "The tag 5-10-15 kilometres stays."),
        ("The date 2026-11-26 km stays.", "The date 2026-11-26 kilometres stays."),
        ("Opens 9-5.", "Opens 9-5."),
        ("Version 1.2.3-4 MB is out.", "Version 1.2.3-4 megabytes is out."),
        ("Ids 1-4.5.6 MB stay.", "Ids 1-4.5.6 megabytes stay."),
        ("Call 555-123-4567 km away.", "Call 555-123-4567 kilometres away."),
        ("Range 10-20.5 km.", "Range 10 to 20.5 kilometres."),
    ],
)
def test_a_hyphen_between_figures_with_a_unit_is_a_range(reply: str, spoken: str) -> None:
    assert speakable(reply) == spoken
