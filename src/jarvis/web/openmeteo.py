"""Open-Meteo's free forecast and geocoding APIs (no account, no key; docs/adr/0007).

Only numbers, codes and checked identifiers are taken from the answers, never free text, so a
weather or time answer carries no outside words (see `jarvis.web.ops`).
"""

import json
import math
import re
import threading
from dataclasses import dataclass
from datetime import date as calendar_date
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from jarvis.jsonutil import obj, objs
from jarvis.web.guard import read_capped
from jarvis.web.zones import bare

if TYPE_CHECKING:
    from collections.abc import Callable

FORECAST = "https://api.open-meteo.com/v1/forecast"
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_DAYS = 7
_COUNTRY_CODE = re.compile(r"[A-Z]{2}")
# "Name (qualifier)" or "Name, qualifier": the geocoder takes the name alone.
_QUALIFIED = re.compile(r"([^,()]+?)\s*(?:\(([^()]+)\)|,\s*(.+))")
_CHOICES = 10  # matches asked for when a qualifier picks among them
_CURRENT = "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,precipitation"
_DAILY = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
_HOURLY = "temperature_2m,precipitation_probability,weather_code"
_HOUR = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:00")
_MINUTE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")

# WMO weather interpretation codes, as Open-Meteo documents them, in plain spoken words.
CONDITIONS: dict[int, str] = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy with frost",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "thunderstorms with heavy hail",
}


class WeatherUnavailableError(Exception):
    """Open-Meteo did not answer, or answered something unusable."""


@dataclass(frozen=True, slots=True)
class Place:
    """Where a forecast is for: coordinates, the IANA time zone and the ISO country code.

    `country` is a two-letter code or empty; `zone` is checked against the local zone database.
    `city` names the city a region is answered at (`regions.py`), empty for a place itself.
    """

    latitude: float
    longitude: float
    zone: str
    country: str = ""
    city: str = ""


@dataclass(frozen=True, slots=True)
class Now:
    """The weather right now."""

    temperature: float
    feels_like: float
    condition: str
    wind: float
    precipitation: float


@dataclass(frozen=True, slots=True)
class Day:
    """One day of the forecast (`date` is ISO, in the place's own time zone)."""

    date: str
    condition: str
    high: float
    low: float
    rain_chance: int


@dataclass(frozen=True, slots=True)
class Hour:
    """One hour of the forecast (`time` is ISO to the hour, in the place's own time zone)."""

    time: str
    condition: str
    temperature: float
    rain_chance: int


@dataclass(frozen=True, slots=True)
class Forecast:
    """Now and the next days, with the units the numbers are in.

    `hours` is filled only when they were asked for; `local_now` is the place's own clock
    ("2026-11-21T15:00"), empty when the service did not give it.
    """

    now: Now
    days: tuple[Day, ...]
    temperature_unit: str
    wind_unit: str
    hours: tuple[Hour, ...] = ()
    local_now: str = ""


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        msg = "the forecast had a missing number"
        raise WeatherUnavailableError(msg)
    return float(value)


def _is_date(text: str) -> bool:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return False
    try:
        calendar_date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _condition(code: object) -> str:
    return CONDITIONS.get(int(_number(code)), "unsettled")


def _hours(hourly: dict[str, Any]) -> tuple[Hour, ...]:
    """Read the hourly columns; an hour with a time that is not one, or no number, is left out.

    Raises:
        WeatherUnavailableError: the answer has no hourly columns.
    """
    columns: list[list[object]] = []
    for key in ("time", "weather_code", "temperature_2m", "precipitation_probability"):
        column: object = hourly.get(key)
        if not isinstance(column, list):
            msg = "the forecast had no hours"
            raise WeatherUnavailableError(msg)
        columns.append(cast("list[object]", column))
    found: list[Hour] = []
    for time, code, temperature, chance in zip(*columns, strict=False):
        if not isinstance(time, str) or not _HOUR.fullmatch(time):
            continue
        try:
            found.append(
                Hour(
                    time,
                    _condition(code),
                    _number(temperature),
                    int(_number(chance)) if chance is not None else 0,
                )
            )
        except WeatherUnavailableError:
            continue
    return tuple(found)


def _qualified(name: str) -> tuple[str, tuple[str, ...]]:
    """Split "Springfield (Illinois)" or "Springfield, Illinois, USA": the name, its qualifiers."""
    match = _QUALIFIED.fullmatch(name.strip())
    if match is None:
        return name, ()
    parts = (match.group(2) or match.group(3)).split(",")
    return match.group(1), tuple(q for part in parts if (q := bare(part)))


def _zone(name: object) -> str:
    text = str(name or "")
    try:
        ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        msg = "the place's time zone is unknown"
        raise WeatherUnavailableError(msg) from exc
    return text


class OpenMeteo:
    """Looks places up and fetches forecasts."""

    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        factory: Callable[[], httpx.Client] | None = None,
    ) -> None:
        """Use `http` (tests) or a client `factory` makes on first use (no socket while idle)."""
        super().__init__()
        self._http = http
        self._lock = threading.Lock()
        self._factory = factory or httpx.Client

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:  # one client, however many look-ups start at once
            if self._http is None:
                self._http = self._factory()
            http = self._http
        try:
            with http.stream("GET", url, params=params) as response:
                status = response.status_code
                body = read_capped(response) if status == HTTPStatus.OK else b""
        except httpx.HTTPError as exc:
            msg = "the weather service didn't answer"
            raise WeatherUnavailableError(msg) from exc
        except ValueError as exc:
            msg = "the weather service's answer was too large"
            raise WeatherUnavailableError(msg) from exc
        if status != HTTPStatus.OK:
            msg = f"the weather service answered {status}"
            raise WeatherUnavailableError(msg)
        try:
            return obj(json.loads(body))
        except ValueError as exc:
            msg = "the weather service answered something that is not JSON"
            raise WeatherUnavailableError(msg) from exc

    def find(self, name: str) -> Place | None:
        """Return the best-known place called `name` (the most populous match), or None.

        A name said with its state or country ("Springfield (Illinois)", "Springfield,
        Illinois") is looked up by the name alone, which is all the geocoder takes, and the
        qualifier picks among the matches; when none fits it, the best-known one is taken.
        A failing service raises `WeatherUnavailableError`.
        """
        said, qualifier = _qualified(name)
        params = {"name": said, "count": _CHOICES if qualifier else 1}
        found = objs(
            self._get(GEOCODE, {**params, "language": "en", "format": "json"}).get("results")
        )
        if not found:
            return None

        def matched(place: dict[str, object]) -> int:
            known = {
                bare(str(place.get(key) or "")) for key in ("country", "country_code", "admin1")
            }
            return sum(q in known for q in qualifier)

        # The place most qualifiers name; with none named, the best known (the geocoder's first).
        first = max(found, key=matched) if qualifier else found[0]
        country = str(first.get("country_code") or "")
        return Place(
            _number(first.get("latitude")),
            _number(first.get("longitude")),
            _zone(first.get("timezone")),
            country if _COUNTRY_CODE.fullmatch(country) else "",
        )

    def zone_at(self, latitude: float, longitude: float) -> str:
        """Return the IANA time zone at a point.

        A failing service raises `WeatherUnavailableError`.
        """
        data = self._get(
            FORECAST,
            {"latitude": latitude, "longitude": longitude, "timezone": "auto", "forecast_days": 1},
        )
        return _zone(data.get("timezone"))

    def forecast(
        self,
        latitude: float,
        longitude: float,
        *,
        temperature_unit: str,
        wind_unit: str,
        hours: bool = False,
    ) -> Forecast:
        """Return the weather now and for the next `FORECAST_DAYS` days, by the hour if `hours`.

        Raises:
            WeatherUnavailableError: the service failed or the answer is incomplete.
        """
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "current": _CURRENT,
            "daily": _DAILY,
            "timezone": "auto",
            "forecast_days": FORECAST_DAYS,
            "temperature_unit": temperature_unit,
            "wind_speed_unit": wind_unit,
        }
        if hours:
            params["hourly"] = _HOURLY
        data = self._get(FORECAST, params)
        current = obj(data.get("current"))
        daily = obj(data.get("daily"))
        now = Now(
            _number(current.get("temperature_2m")),
            _number(current.get("apparent_temperature")),
            _condition(current.get("weather_code")),
            _number(current.get("wind_speed_10m")),
            _number(current.get("precipitation")),
        )
        rows: list[list[object]] = []
        for key in (
            "time",
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
        ):
            column: object = daily.get(key)
            if not isinstance(column, list):
                msg = "the forecast had no days"
                raise WeatherUnavailableError(msg)
            rows.append(cast("list[object]", column))
        days: list[Day] = []
        for date, code, high, low, chance in zip(*rows, strict=False):
            if not isinstance(date, str) or not _is_date(date):
                continue
            days.append(
                Day(
                    date,
                    _condition(code),
                    _number(high),
                    _number(low),
                    int(_number(chance)) if chance is not None else 0,
                )
            )
        local_now = current.get("time")
        return Forecast(
            now,
            tuple(days),
            temperature_unit,
            wind_unit,
            _hours(obj(data.get("hourly"))) if hours else (),
            local_now if isinstance(local_now, str) and _MINUTE.fullmatch(local_now) else "",
        )
