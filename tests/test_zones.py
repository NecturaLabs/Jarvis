"""A country's clocks, from this PC's time zone database (`jarvis.web.zones`)."""

from datetime import UTC, datetime

import pytest

from jarvis.web import zones

WINTER = datetime(2026, 11, 24, 9, 35, tzinfo=UTC)
SUMMER = datetime(2026, 7, 1, 9, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("said", "code"),
    [
        ("the United States", "US"),
        ("United States", "US"),
        ("USA", "US"),
        ("the U.S.", "US"),
        ("America", "US"),
        ("the UK", "GB"),
        ("Britain", "GB"),
        ("England", "GB"),
        ("South Korea", "KR"),
        ("Russia", "RU"),
        ("france", "FR"),
        ("North Korea", "KP"),
        ("Georgia", "GE"),  # the country, as the geocoder has it too; the state is said by a city
    ],
)
def test_a_country_is_found_by_the_names_people_use(said: str, code: str) -> None:
    found = zones.find_country(said)

    assert found is not None
    assert found.code == code


@pytest.mark.parametrize(
    "said", ["Tokyo", "New York", "Atlantis", "", "Korea", "Virgin Islands", "Texas", "Georgia, US"]
)
def test_a_city_or_a_name_two_countries_share_is_not_a_country(said: str) -> None:
    assert zones.find_country(said) is None


def _cities(country: str, now: datetime) -> list[str]:
    found = zones.find_country(country)
    assert found is not None
    return [zone.city for zone in zones.clocks(found, now)]


def test_each_us_clock_is_said_once_from_east_to_west() -> None:
    assert _cities("United States", WINTER) == [
        "New York",
        "Chicago",
        "Denver",
        "Los Angeles",
        "Anchorage",
        "Honolulu",
    ]


def test_zones_sharing_a_time_are_said_by_the_main_city_in_summer_too() -> None:
    # Phoenix keeps standard time, so in summer it shows Los Angeles's time, and Adak leaves
    # Honolulu's: the zone kept for each time is still the one people know.
    assert _cities("United States", SUMMER) == [
        "New York",
        "Chicago",
        "Denver",
        "Los Angeles",
        "Anchorage",
        "Adak",
        "Honolulu",
    ]


@pytest.mark.parametrize(
    ("country", "city"),
    [
        ("Russia", "Moscow"),
        ("Brazil", "Sao Paulo"),
        ("Mexico", "Mexico City"),
        ("Canada", "Vancouver"),
    ],
)
def test_a_large_country_names_its_main_cities(country: str, city: str) -> None:
    assert city in _cities(country, WINTER)


def test_a_country_with_one_clock_has_one_zone() -> None:
    assert len(_cities("France", WINTER)) == 1
    assert len(_cities("Germany", WINTER)) == 1  # Busingen keeps Berlin's time


def _no_table(_name: str) -> str | None:
    return None


def test_without_the_database_tables_no_country_is_known(monkeypatch: pytest.MonkeyPatch) -> None:
    zones.countries.cache_clear()
    monkeypatch.setattr(zones, "_table", _no_table)
    try:
        assert zones.find_country("France") is None
    finally:
        zones.countries.cache_clear()
