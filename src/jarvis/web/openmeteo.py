"""Open-Meteo's free forecast and geocoding APIs (no account, no key; docs/adr/0007).

Only numbers, codes and checked identifiers are taken from the answers, never free text, so a
weather or time answer carries no outside words (see `jarvis.web.ops`).
"""

import json
import math
import re
from dataclasses import dataclass
from datetime import date as calendar_date
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from jarvis.jsonutil import obj, objs
from jarvis.web.guard import read_capped

if TYPE_CHECKING:
    from collections.abc import Callable

FORECAST = "https://api.open-meteo.com/v1/forecast"
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_DAYS = 7
_COUNTRY_CODE = re.compile(r"[A-Z]{2}")
_CURRENT = "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,precipitation"
_DAILY = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"

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
    """

    latitude: float
    longitude: float
    zone: str
    country: str = ""


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
class Forecast:
    """Now and the next days, with the units the numbers are in."""

    now: Now
    days: tuple[Day, ...]
    temperature_unit: str
    wind_unit: str


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
        self._factory = factory or httpx.Client

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._http is None:
            self._http = self._factory()
        try:
            with self._http.stream("GET", url, params=params) as response:
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

        A failing service raises `WeatherUnavailableError`.
        """
        found = objs(
            self._get(GEOCODE, {"name": name, "count": 1, "language": "en", "format": "json"}).get(
                "results"
            )
        )
        if not found:
            return None
        first = found[0]
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
        self, latitude: float, longitude: float, *, temperature_unit: str, wind_unit: str
    ) -> Forecast:
        """Return the weather now and for the next `FORECAST_DAYS` days.

        Raises:
            WeatherUnavailableError: the service failed or the answer is incomplete.
        """
        data = self._get(
            FORECAST,
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": _CURRENT,
                "daily": _DAILY,
                "timezone": "auto",
                "forecast_days": FORECAST_DAYS,
                "temperature_unit": temperature_unit,
                "wind_speed_unit": wind_unit,
            },
        )
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
        return Forecast(now, tuple(days), temperature_unit, wind_unit)
