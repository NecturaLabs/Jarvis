"""States, provinces and territories said as places (`jarvis.web.regions`)."""

import zoneinfo

import pytest

from jarvis.web import regions


@pytest.mark.parametrize(
    ("said", "city"),
    [
        ("Texas", "Houston"),
        ("texas", "Houston"),
        ("the state of Texas", "Houston"),
        ("Texas, USA", "Houston"),
        ("New York State", "New York"),
        ("Ontario", "Toronto"),
        ("Ontario, Canada", "Toronto"),
        ("Quebec province", "Montreal"),
        ("Victoria", "Melbourne"),
        ("Victoria, Australia", "Melbourne"),
        ("Georgia state", "Atlanta"),
        ("Georgia, US", "Atlanta"),
        ("the state of Washington", "Seattle"),
        ("Washington State", "Seattle"),
    ],
)
def test_a_region_is_found_by_the_ways_people_say_it(said: str, city: str) -> None:
    found = regions.find_region(said)

    assert found is not None
    assert found.city == city


@pytest.mark.parametrize(
    "said",
    [
        "Georgia",  # the country
        "Washington",  # the capital, left to the geocoder
        "Washington, D.C.",
        "Houston",
        "Paris",
        "",
        "Texas, Canada",  # no such province
        "Ontario, Australia",
    ],
)
def test_anything_but_a_region_is_left_to_the_other_lookups(said: str) -> None:
    assert regions.find_region(said) is None


def test_every_region_has_a_real_zone_and_a_point_in_its_hemisphere() -> None:
    for row in regions._ROWS:  # pyright: ignore[reportPrivateUsage] - the table itself
        region = regions.Region(*row)
        zoneinfo.ZoneInfo(region.zone)  # raises for a zone the database does not know
        north = region.country != "AU"
        assert (region.latitude > 0) is north, region.name
        assert region.longitude < 0 if north else region.longitude > 0, region.name
