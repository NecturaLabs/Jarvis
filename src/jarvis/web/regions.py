"""States, provinces and territories said as places, each answered at its main city.

The geocoder finds places by name, not regions: its best "Texas" is a village in Mexico, its best
"California" a town in Colombia, its best "Ontario" a city in California. A region the owner names
is answered here instead, at its main city (the centre of its largest metropolitan area) and in
that city's time zone, so its time needs nothing from the network and its weather is the forecast
for that city. A region with several time zones gets its main city's.

Three names are a region only with a word that says so ("Georgia state", "Georgia, US",
"Washington state", "the province of Quebec"): plain "Georgia" is the country, and plain
"Washington" and "Quebec" are left to the geocoder, which finds the capital and Quebec City.
"""

import re
from dataclasses import dataclass

from jarvis.web.zones import bare


@dataclass(frozen=True, slots=True)
class Region:
    """A state, province or territory: its country's ISO code, and its main city's place."""

    name: str
    country: str
    city: str
    latitude: float
    longitude: float
    zone: str


# Name, country, main city, its latitude and longitude, and its IANA time zone.
_ROWS: tuple[tuple[str, str, str, float, float, str], ...] = (
    ("Alabama", "US", "Birmingham", 33.52, -86.80, "America/Chicago"),
    ("Alaska", "US", "Anchorage", 61.22, -149.90, "America/Anchorage"),
    ("Arizona", "US", "Phoenix", 33.45, -112.07, "America/Phoenix"),
    ("Arkansas", "US", "Little Rock", 34.75, -92.29, "America/Chicago"),
    ("California", "US", "Los Angeles", 34.05, -118.24, "America/Los_Angeles"),
    ("Colorado", "US", "Denver", 39.74, -104.99, "America/Denver"),
    ("Connecticut", "US", "Hartford", 41.76, -72.68, "America/New_York"),
    ("Delaware", "US", "Wilmington", 39.74, -75.55, "America/New_York"),
    ("Florida", "US", "Miami", 25.77, -80.19, "America/New_York"),
    ("Georgia", "US", "Atlanta", 33.75, -84.39, "America/New_York"),
    ("Hawaii", "US", "Honolulu", 21.31, -157.86, "Pacific/Honolulu"),
    ("Idaho", "US", "Boise", 43.62, -116.20, "America/Boise"),
    ("Illinois", "US", "Chicago", 41.88, -87.63, "America/Chicago"),
    ("Indiana", "US", "Indianapolis", 39.77, -86.16, "America/Indiana/Indianapolis"),
    ("Iowa", "US", "Des Moines", 41.59, -93.62, "America/Chicago"),
    ("Kansas", "US", "Wichita", 37.69, -97.34, "America/Chicago"),
    ("Kentucky", "US", "Louisville", 38.25, -85.76, "America/Kentucky/Louisville"),
    ("Louisiana", "US", "New Orleans", 29.95, -90.07, "America/Chicago"),
    ("Maine", "US", "Portland", 43.66, -70.26, "America/New_York"),
    ("Maryland", "US", "Baltimore", 39.29, -76.61, "America/New_York"),
    ("Massachusetts", "US", "Boston", 42.36, -71.06, "America/New_York"),
    ("Michigan", "US", "Detroit", 42.33, -83.05, "America/Detroit"),
    ("Minnesota", "US", "Minneapolis", 44.98, -93.27, "America/Chicago"),
    ("Mississippi", "US", "Jackson", 32.30, -90.18, "America/Chicago"),
    ("Missouri", "US", "St. Louis", 38.63, -90.20, "America/Chicago"),
    ("Montana", "US", "Billings", 45.78, -108.50, "America/Denver"),
    ("Nebraska", "US", "Omaha", 41.26, -95.94, "America/Chicago"),
    ("Nevada", "US", "Las Vegas", 36.17, -115.14, "America/Los_Angeles"),
    ("New Hampshire", "US", "Manchester", 42.99, -71.46, "America/New_York"),
    ("New Jersey", "US", "Newark", 40.74, -74.17, "America/New_York"),
    ("New Mexico", "US", "Albuquerque", 35.08, -106.65, "America/Denver"),
    ("New York", "US", "New York", 40.71, -74.01, "America/New_York"),
    ("North Carolina", "US", "Charlotte", 35.23, -80.84, "America/New_York"),
    ("North Dakota", "US", "Fargo", 46.88, -96.79, "America/Chicago"),
    ("Ohio", "US", "Columbus", 39.96, -83.00, "America/New_York"),
    ("Oklahoma", "US", "Oklahoma City", 35.47, -97.52, "America/Chicago"),
    ("Oregon", "US", "Portland", 45.52, -122.68, "America/Los_Angeles"),
    ("Pennsylvania", "US", "Philadelphia", 39.95, -75.17, "America/New_York"),
    ("Rhode Island", "US", "Providence", 41.82, -71.41, "America/New_York"),
    ("South Carolina", "US", "Greenville", 34.85, -82.40, "America/New_York"),
    ("South Dakota", "US", "Sioux Falls", 43.54, -96.73, "America/Chicago"),
    ("Tennessee", "US", "Nashville", 36.16, -86.78, "America/Chicago"),
    ("Texas", "US", "Houston", 29.76, -95.37, "America/Chicago"),
    ("Utah", "US", "Salt Lake City", 40.76, -111.89, "America/Denver"),
    ("Vermont", "US", "Burlington", 44.48, -73.21, "America/New_York"),
    ("Virginia", "US", "Virginia Beach", 36.85, -75.98, "America/New_York"),
    ("Washington", "US", "Seattle", 47.61, -122.33, "America/Los_Angeles"),
    ("West Virginia", "US", "Charleston", 38.35, -81.63, "America/New_York"),
    ("Wisconsin", "US", "Milwaukee", 43.04, -87.91, "America/Chicago"),
    ("Wyoming", "US", "Cheyenne", 41.14, -104.82, "America/Denver"),
    ("Ontario", "CA", "Toronto", 43.65, -79.38, "America/Toronto"),
    ("Quebec", "CA", "Montreal", 45.50, -73.57, "America/Toronto"),
    ("British Columbia", "CA", "Vancouver", 49.28, -123.12, "America/Vancouver"),
    ("Alberta", "CA", "Calgary", 51.05, -114.07, "America/Edmonton"),
    ("Manitoba", "CA", "Winnipeg", 49.90, -97.14, "America/Winnipeg"),
    ("Saskatchewan", "CA", "Saskatoon", 52.13, -106.67, "America/Regina"),
    ("Nova Scotia", "CA", "Halifax", 44.65, -63.58, "America/Halifax"),
    ("New Brunswick", "CA", "Moncton", 46.09, -64.78, "America/Moncton"),
    ("Newfoundland", "CA", "St. John's", 47.56, -52.71, "America/St_Johns"),
    ("Newfoundland and Labrador", "CA", "St. John's", 47.56, -52.71, "America/St_Johns"),
    ("Prince Edward Island", "CA", "Charlottetown", 46.24, -63.13, "America/Halifax"),
    ("Yukon", "CA", "Whitehorse", 60.72, -135.06, "America/Whitehorse"),
    ("Northwest Territories", "CA", "Yellowknife", 62.45, -114.37, "America/Edmonton"),
    ("Nunavut", "CA", "Iqaluit", 63.75, -68.52, "America/Iqaluit"),
    ("New South Wales", "AU", "Sydney", -33.87, 151.21, "Australia/Sydney"),
    ("Victoria", "AU", "Melbourne", -37.81, 144.96, "Australia/Melbourne"),
    ("Queensland", "AU", "Brisbane", -27.47, 153.03, "Australia/Brisbane"),
    ("Western Australia", "AU", "Perth", -31.95, 115.86, "Australia/Perth"),
    ("South Australia", "AU", "Adelaide", -34.93, 138.60, "Australia/Adelaide"),
    ("Tasmania", "AU", "Hobart", -42.88, 147.33, "Australia/Hobart"),
    ("Northern Territory", "AU", "Darwin", -12.46, 130.84, "Australia/Darwin"),
    ("Australian Capital Territory", "AU", "Canberra", -35.28, 149.13, "Australia/Sydney"),
)
_BY_NAME = {bare(row[0]): Region(*row) for row in _ROWS}
_COUNTRY_WORDS = {
    "US": ("us", "usa", "united states", "america"),
    "CA": ("canada",),
    "AU": ("australia",),
}
# Said alone, these name something else first: the country Georgia, the capital Washington and
# Quebec City.
_NEEDS_A_WORD = frozenset({"georgia", "washington", "quebec"})
_OF = re.compile(r"^(?:state|province|territory) of ")
_KIND = re.compile(
    r"^(?:(?:state|province|territory) of )?(?P<name>.+?)(?: (?:state|province|territory))?$"
)


def find_region(place: str) -> Region | None:
    """Return the state, province or territory `place` names, or None for anything else.

    It may carry the word for what it is ("the state of Texas", "Ontario province") or its
    country ("Florida, US", "Victoria Australia").
    """
    said = bare(place)
    kind = _KIND.fullmatch(said)
    stripped = kind["name"] if kind else said
    # The name as said first: "territory" in "Northern Territory" is part of the name.
    for name in dict.fromkeys((said, _OF.sub("", said), stripped)):
        region = _BY_NAME.get(name)
        if region is not None:
            return None if name == said and name in _NEEDS_A_WORD else region
    for code, words in _COUNTRY_WORDS.items():
        for word in words:
            if stripped.endswith(f" {word}"):
                found = _BY_NAME.get(stripped.removesuffix(f" {word}"))
                if found is not None and found.country == code:
                    return found
    return None
