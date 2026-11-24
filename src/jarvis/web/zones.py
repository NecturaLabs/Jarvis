"""The time zones of a country, from the time zone database on this PC: nothing is sent anywhere.

A country such as the United States or Russia has several clocks at once, so "the time in X" for
it is one time per clock. The tables are the database's own: `iso3166.tab` names the countries
and `zone.tab` lists each country's zones with a comment on the area each covers. Zones that
show the same time right now are said once, by the zone that covers the country's main area.
"""

import functools
import importlib.resources
import re
import unicodedata
import zoneinfo
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

# A comment that marks the zone covering most of a country (or of its region).
_MAIN = re.compile(r"\bmost\b|\bmainland\b", re.IGNORECASE)
_WORDS = re.compile(r"[^a-z0-9 ]+")
# Zones whose city is the largest on its clock in its country, where the database's own ranking
# (below) names a smaller one: Winnipeg over Toronto once Manitoba keeps Eastern time, Boa Vista
# over Manaus, Saratov over Samara, Barnaul over Novosibirsk.
_PRINCIPAL = frozenset({"America/Toronto", "America/Manaus", "Europe/Samara", "Asia/Novosibirsk"})
_ZONE = 2  # the zone's column in the database's tables; the comment follows it
_NAMED = 2  # a country row in `iso3166.tab`: its code, then its name
# English names the database spells otherwise ("Britain (UK)", "Korea (South)").
_ALSO = {
    "america": "US",
    "usa": "US",
    "us": "US",
    "united states of america": "US",
    "uk": "GB",
    "united kingdom": "GB",
    "great britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "northern ireland": "GB",
    "uae": "AE",
    "czechia": "CZ",
    "holland": "NL",
    "democratic republic of the congo": "CD",
    "dr congo": "CD",
    "republic of the congo": "CG",
    "ivory coast": "CI",
    "turkiye": "TR",
    "burma": "MM",
    "swaziland": "SZ",
}


@dataclass(frozen=True, slots=True)
class Zone:
    """One of a country's zones: its IANA name, and its row's place in `zone.tab`."""

    name: str
    comment: str
    row: int

    @property
    def city(self) -> str:
        """The zone's principal city as it is said: "America/Los_Angeles" is "Los Angeles"."""
        return self.name.rsplit("/", 1)[-1].replace("_", " ")


@dataclass(frozen=True, slots=True)
class Country:
    """A country the database knows: its ISO code, its name there, and its zones."""

    code: str
    name: str
    zones: tuple[Zone, ...]


@functools.cache
def _leading() -> frozenset[str]:
    """Return the zones `zonenow.tab` names, the most populous of each set keeping one time."""
    text = _table("zonenow.tab")
    if text is None:
        return frozenset()
    return frozenset(columns[_ZONE] for columns in _rows(text) if len(columns) > _ZONE)


def _table(name: str) -> str | None:
    """Return one of the database's tables, from the system's copy or the `tzdata` package."""
    for folder in zoneinfo.TZPATH:
        path = Path(folder) / name
        if path.is_file():
            return path.read_text(encoding="utf-8")
    try:
        return (importlib.resources.files("tzdata.zoneinfo") / name).read_text(encoding="utf-8")
    except ModuleNotFoundError, FileNotFoundError:
        return None


def _rows(text: str) -> list[list[str]]:
    return [
        line.split("\t") for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]


def bare(text: str) -> str:
    """Return a name as bare lowercase words: "The United States!" is "united states".

    Accents are dropped, so "Côte d'Ivoire" said with or without them is the same name.
    """
    plain = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    words = " ".join(_WORDS.sub(" ", plain.casefold().replace(".", "")).split())
    return words.removeprefix("the ")


def _names(official: str) -> tuple[set[str], str]:
    """Return the ways a database name is said, and its bare base name.

    "Korea (South)" is said "korea south" or "south korea"; its base, "korea", is a name of
    its own only where no other country shares it (North Korea does).
    """
    match = re.fullmatch(r"(?P<base>[^(]+?)\s*\((?P<note>[^)]+)\)", official)
    if match is None:
        return {bare(official)}, bare(official)
    base, note = bare(match["base"]), bare(match["note"])
    return {bare(official), f"{note} {base}"}, base


@functools.cache
def countries() -> dict[str, Country]:
    """Return every country with zones, by each name it may be called; empty without the tables."""
    names, zones = _table("iso3166.tab"), _table("zone.tab")
    if names is None or zones is None:
        return {}
    by_code: dict[str, list[Zone]] = {}
    for row, columns in enumerate(_rows(zones)):
        if len(columns) > _ZONE:
            comment = columns[_ZONE + 1] if len(columns) > _ZONE + 1 else ""
            by_code.setdefault(columns[0], []).append(Zone(columns[_ZONE], comment, row))
    known = {
        columns[0]: Country(columns[0], columns[1], tuple(by_code[columns[0]]))
        for columns in _rows(names)
        if len(columns) == _NAMED and columns[0] in by_code
    }
    found = {said: known[code] for said, code in _ALSO.items() if code in known}
    bases: dict[str, list[Country]] = {}
    for country in known.values():
        said, base = _names(country.name)
        for name in said:
            found.setdefault(name, country)
        bases.setdefault(base, []).append(country)
    for base, sharing in bases.items():
        if len(sharing) == 1:
            found.setdefault(base, sharing[0])
    return found


def find_country(place: str) -> Country | None:
    """Return the country `place` names, or None for a city or anything else.

    It may say what it is: "Georgia (country)", "the country of Georgia".
    """
    said = bare(place)
    name = said.removeprefix("country of ").removesuffix(" country").removesuffix(" the")
    return countries().get(said) or countries().get(name)


def clocks(country: Country, now: datetime) -> list[Zone]:
    """Return one zone for each time shown in `country` at `now`, from east to west.

    Of the zones showing the same time, the one said is a principal city (`_PRINCIPAL`), then
    the database's choice for that clock (`zonenow.tab`, the most populous), then one whose
    comment marks the main area (the database lists those first), then the one with the shortest
    comment, the broadest area.
    """
    groups: dict[timedelta, list[Zone]] = {}
    for zone in country.zones:
        offset = now.astimezone(zoneinfo.ZoneInfo(zone.name)).utcoffset() or timedelta(0)
        groups.setdefault(offset, []).append(zone)

    leading = _leading()

    def rank(zone: Zone) -> tuple[bool, bool, bool, int, int]:
        main = _MAIN.search(zone.comment) is not None
        return (
            zone.name not in _PRINCIPAL,
            zone.name not in leading,
            not main,
            zone.row if main else len(zone.comment),
            zone.row,
        )

    return [min(groups[offset], key=rank) for offset in sorted(groups, reverse=True)]
