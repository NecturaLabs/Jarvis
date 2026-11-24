"""Answers from outside this PC: the time, the weather, web search, web pages and Reddit.

None of these changes anything, so none needs confirmation on its own; every one sends only what
it is asked about (a place, a search) to the service named in docs/adr/0007. Text written by
someone else (search results, pages, Reddit) keeps the `web` origin, so it stays data and never
raises a class. Weather and time answers carry no `web` origin: they are built here from numbers,
codes, checked identifiers and the place the model asked about, so they do not make the next
action need a yes. Every operation that sends words the model wrote (a place, a query) is
`outbound`: while text from the owner's files is in the conversation it needs a yes. The time and
the weather send such words only in a place, so asked about here, they need none.

A page or a Reddit thread can be opened only when a search in this conversation returned it: the
model cannot be talked into sending conversation text to an address a page made up. The same holds
for a picture the chat shows (`show_image`): it must come from an image search or a read page.
When the conversation is forgotten (`reset`), so is what its searches returned.
"""

import re
from collections import deque
from dataclasses import dataclass
from datetime import UTC, date, datetime, tzinfo
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from jarvis.operations import Operation, OperationError, Result, cut_inert
from jarvis.pictures import SHOWN_SIDE, PictureError, prepare
from jarvis.policy import Origin
from jarvis.shortcuts import Shortcut
from jarvis.web.arctic import SUBREDDIT, ArcticShift, Post, RedditUnavailableError
from jarvis.web.openmeteo import Day, Forecast, OpenMeteo, Place, WeatherUnavailableError
from jarvis.web.page import PageError, PageReader
from jarvis.web.searxng import Hit, ImageHit, SearchUnavailableError, Searxng
from jarvis.web.zones import Zone, clocks, find_country

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.config import Config
    from jarvis.pictures import PictureCache

REMEMBERED = 60
NOON = 12
WORTH_SAYING = 2  # degrees between the temperature and how it feels
MAX_PLACE = 80
MAX_QUERY = 200
MAX_URL = 2000
MAX_POST_BODY = 600
MAX_COMMENT = 300
NO_HOME = (
    "I don't have a home set, so tell me a place. You can set one in the config under weather."
)
_REDDIT_POST = re.compile(r"^/r/(?P<sub>[A-Za-z0-9_]{2,21})/comments/(?P<id>[a-z0-9]{1,12})(?:/|$)")
_UNIT_SIGNS = {"celsius": "°C", "fahrenheit": "°F"}
_WIND_WORDS = {"kmh": "km/h", "mph": "mph", "ms": "m/s", "kn": "knots"}

_TIME = Shortcut(
    r"(?:what time is it|whats the time|what is the time|tell me the time|(?:the )?time)",
    lambda _g: {"what": "time"},
)
_DATE = Shortcut(
    r"(?:(?:whats|what is) (?:the date|todays date)(?: today)?|what day is (?:it|today)"
    r"(?: today)?)",
    lambda _g: {"what": "date"},
)
_WEATHER_NOW = Shortcut(
    r"(?:(?:whats|what is) the weather(?: like)?|(?:how is|hows) the weather|weather)",
    lambda _g: {"when": "now"},
)
_WEATHER_TODAY = Shortcut(
    r"(?:(?:whats|what is) the (?:weather(?: like)?|forecast)(?: for)? today|weather today)",
    lambda _g: {"when": "today"},
)
_WEATHER_TOMORROW = Shortcut(
    r"(?:(?:whats|what is) the (?:weather(?: like)?|forecast)(?: for)? tomorrow"
    r"|weather tomorrow)",
    lambda _g: {"when": "tomorrow"},
)
_PLACE = {
    "type": "string",
    "maxLength": MAX_PLACE,
    "description": "A city or place the owner named; leave out for the owner's home",
}
_PLACE_ARG = frozenset({"place"})  # the time and weather send the model's words only in it
_COUNT = {"type": "integer", "minimum": 1, "maximum": 8, "description": "How many results"}


def _round(value: float) -> int:
    return round(value)


def spoken_time(moment: datetime, *, twelve_hour: bool) -> str:
    """Return a clock time as it is said: "15:42", or "3:42 PM" on a twelve-hour clock."""
    if not twelve_hour:
        return f"{moment.hour}:{moment.minute:02d}"
    hour = moment.hour % 12 or 12
    return f"{hour}:{moment.minute:02d} {'AM' if moment.hour < NOON else 'PM'}"


def spoken_date(day: date) -> str:
    """Return a date as it is said: "Saturday, 21 November"."""
    return f"{day:%A}, {day.day} {day:%B}"


def _reddit_thread(url: str) -> tuple[str, str] | None:
    """Return the post id and subreddit of a Reddit thread's address, or None for anything else."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if host != "reddit.com" and not host.endswith(".reddit.com"):
        return None
    match = _REDDIT_POST.match(parts.path)
    return None if match is None else (match["id"], match["sub"])


def _and(parts: list[str], last: str = " and ") -> str:
    """Join spoken parts: "a, b and c"."""
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + last + parts[-1]


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _where(place: str, country: str) -> str:
    return f"{place} ({country})" if country else place


def _day_line(day: Day, label: str, unit: str) -> str:
    return (
        f"{label}: {day.condition}, {_round(day.low)} to {_round(day.high)} {unit}, "
        f"{day.rain_chance}% chance of rain"
    )


def _weather_result(forecast: Forecast, when: str, where: str) -> Result:
    """Word a forecast for the model and, for the owner, as one spoken sentence.

    Raises:
    OperationError: the forecast lacks the day asked about.
    """
    sign = _UNIT_SIGNS.get(forecast.temperature_unit, "degrees")
    if when == "now":
        now = forecast.now
        feels = (
            f", feeling like {_round(now.feels_like)}"
            if abs(now.feels_like - now.temperature) >= WORTH_SAYING
            else ""
        )
        text = (
            f"Weather in {where} now: {_round(now.temperature)} {sign}, {now.condition}, "
            f"feels like {_round(now.feels_like)}, wind {_round(now.wind)} "
            f"{_WIND_WORDS.get(forecast.wind_unit, '')}, precipitation {now.precipitation} mm."
        )
        spoken = f"It's {_round(now.temperature)} degrees and {now.condition}{feels}."
        return Result(text, spoken=spoken)
    index = {"today": 0, "tomorrow": 1}.get(when)
    if index is not None:
        if len(forecast.days) <= index:
            msg = "The forecast didn't include that day."
            raise OperationError(msg)
        day = forecast.days[index]
        label = when.capitalize()
        text = f"Weather in {where}. " + _day_line(day, label, sign) + "."
        spoken = (
            f"{label} looks like {day.condition}, {_round(day.low)} to {_round(day.high)} degrees, "
            f"with a {day.rain_chance} percent chance of rain."
        )
        return Result(text, spoken=spoken)
    lines = [_day_line(day, f"{date.fromisoformat(day.date):%A}", sign) for day in forecast.days]
    return Result(f"Weather in {where} this week. " + "; ".join(lines) + ".")


@dataclass(frozen=True, slots=True)
class Clients:
    """The services the operations reach."""

    search: Searxng
    weather: OpenMeteo
    reddit: ArcticShift
    pages: PageReader


class WebOps:
    """The operations, with the clients they use and what recent searches returned."""

    def __init__(
        self,
        config: Config,
        clients: Clients,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        local_zone: tzinfo | None = None,
        shown: PictureCache | None = None,
    ) -> None:
        """Use the clients given; `clock` and `local_zone` are for tests (default: this PC's).

        With `shown`, where the chat's pictures are kept, the pictures operations are offered too.
        """
        super().__init__()
        self._clock_config = config.clock
        self._weather_config = config.weather
        self._research = config.research
        self._search = clients.search
        self._weather = clients.weather
        self._reddit = clients.reddit
        self._pages = clients.pages
        self._clock = clock
        self._local_zone = local_zone
        self._home: Place | None = None
        self._urls: deque[str] = deque(maxlen=REMEMBERED)
        self._posts: deque[str] = deque(maxlen=REMEMBERED)
        self._shown = shown
        self._images: dict[str, ImageHit] = {}
        self._image_number = 0

    def reset(self) -> None:
        """Forget every address, post and picture earlier searches returned."""
        self._urls.clear()
        self._posts.clear()
        self._images.clear()

    # --- time -------------------------------------------------------------------------------

    def time(self, args: dict[str, Any]) -> Result:
        """Say the time (or the date) here, or in a named place.

        A country is looked up in this PC's time zone database, and one with several clocks
        gets each of them. Any other place goes to the weather service: an unknown one, or a
        failure of that service, raises `OperationError`.
        """
        what = str(args.get("what", "time"))
        place = str(args.get("place", "")).strip()
        now = self._clock()
        country = find_country(place) if place else None
        if country is not None and len(found := clocks(country, now)) > 1:
            return self._country_time(what, place, found, now)
        if country is not None:
            moment = now.astimezone(ZoneInfo(country.zones[0].name))
            where = f" in {_where(place, country.code)}"
            said = f" in {place}"
        elif place:
            found = self._find(place)
            moment = now.astimezone(ZoneInfo(found.zone))
            where = f" in {_where(place, found.country)}"
            said = f" in {place}"
        else:
            moment = now.astimezone(self._local_zone) if self._local_zone else now.astimezone()
            where = said = ""
        clock = spoken_time(moment, twelve_hour=self._clock_config.twelve_hour)
        day = spoken_date(moment.date())
        # Only what was asked: the model reads every word it is given aloud.
        if what == "date":
            return Result(f"Date{where}: {day} {moment.year}.", spoken=f"It's {day}{said}.")
        return Result(f"Time{where}: {clock}.", spoken=f"It's {clock}{said}.")

    def _country_time(self, what: str, place: str, zones: list[Zone], now: datetime) -> Result:
        """Say the time on each of a country's clocks, east to west, or the dates they show.

        A clock on another day than here says which day.
        """
        if what == "date":
            return self._country_date(place, zones, now)
        here = (now.astimezone(self._local_zone) if self._local_zone else now.astimezone()).date()
        parts: list[str] = []
        for zone in zones:
            moment = now.astimezone(ZoneInfo(zone.name))
            clock = spoken_time(moment, twelve_hour=self._clock_config.twelve_hour)
            day = "" if moment.date() == here else f" {moment:%A}"
            parts.append(f"{clock}{day} in {zone.city}")
        listed = _and(parts)
        return Result(
            f"Time in {place}, which has {len(zones)} time zones now: {listed}.",
            spoken=f"{_capital(place)} has {len(zones)} time zones: it's {listed}.",
        )

    @staticmethod
    def _country_date(place: str, zones: list[Zone], now: datetime) -> Result:
        """Say the date in a country, and where its clocks show another one."""
        cities: dict[date, list[str]] = {}
        for zone in zones:
            cities.setdefault(now.astimezone(ZoneInfo(zone.name)).date(), []).append(zone.city)
        if len(cities) == 1:
            (day,) = cities
            return Result(
                f"Date in {place}: {spoken_date(day)} {day.year}.",
                spoken=f"It's {spoken_date(day)} in {place}.",
            )
        parts = [f"{spoken_date(day)} in {_and(names)}" for day, names in cities.items()]
        written = [
            f"{spoken_date(day)} {day.year} in {_and(names)}" for day, names in cities.items()
        ]
        return Result(
            f"Dates in {place}: {'; '.join(written)}.", spoken=f"It's {_and(parts, ', and ')}."
        )

    # --- weather ----------------------------------------------------------------------------

    def _find(self, place: str) -> Place:
        try:
            found = self._weather.find(place)
        except WeatherUnavailableError as exc:
            raise OperationError(str(exc)) from exc
        if found is None:
            msg = f"I couldn't find a place called {place}."
            raise OperationError(msg)
        return found

    def _home_point(self) -> tuple[float, float, str]:
        """Return the home's coordinates and the name to say for it.

        Raises:
            OperationError: no home is set.
        """
        cfg = self._weather_config
        name = cfg.place or "home"
        if cfg.coordinates:
            return cfg.coordinates[0], cfg.coordinates[1], name
        if not cfg.place:
            raise OperationError(NO_HOME)
        if self._home is None:
            self._home = self._find(cfg.place)
        return self._home.latitude, self._home.longitude, name

    def weather(self, args: dict[str, Any]) -> Result:
        """Report the weather now, today, tomorrow or for the week, at home or in a named place.

        Raises:
            OperationError: no place is known, or the weather service failed.
        """
        place = str(args.get("place", "")).strip()
        when = str(args.get("when", "now"))
        if place:
            found = self._find(place)
            latitude, longitude = found.latitude, found.longitude
            where = _where(place, found.country)
        else:
            latitude, longitude, where = self._home_point()
        try:
            forecast = self._weather.forecast(
                latitude,
                longitude,
                temperature_unit=self._weather_config.temperature_unit,
                wind_unit=self._weather_config.wind_unit,
            )
        except WeatherUnavailableError as exc:
            raise OperationError(str(exc)) from exc
        return _weather_result(forecast, when, where)

    # --- web --------------------------------------------------------------------------------

    def _hits(self, query: str, count: int, site: str = "") -> list[Hit]:
        try:
            return self._search.search(query, limit=count, site=site)
        except SearchUnavailableError as exc:
            raise OperationError(str(exc)) from exc

    def web_search(self, args: dict[str, Any]) -> Result:
        """Search the web and list the results; their addresses may then be read.

        A failing local search service raises `OperationError`.
        """
        query = str(args["query"])
        hits = self._hits(query, int(args.get("count", self._research.results)))
        if not hits:
            return Result("The search found nothing.")
        lines: list[str] = []
        for number, hit in enumerate(hits, 1):
            self._remember(self._urls, hit.url)
            lines.append(
                f"{number}. {hit.title} ({urlsplit(hit.url).hostname})\n   {hit.url}\n"
                f"   {hit.snippet}"
            )
        return Result(f"Results for {query!r}:\n" + "\n".join(lines), Origin.WEB)

    def web_read(self, args: dict[str, Any]) -> Result:
        """Read a page a recent search returned and give its main text.

        Raises:
            OperationError: the address came from no search, or the page could not be read.
        """
        url = str(args["url"]).strip()
        if url not in self._urls:
            msg = "Only a page a recent web_search returned can be read; search first."
            raise OperationError(msg)
        thread = _reddit_thread(url)
        if thread is not None:
            # Reddit refuses to serve pages to Jarvis; the archive has the thread.
            self._remember(self._posts, thread[0])
            return self.reddit_read({"post_id": thread[0]})
        try:
            page = self._pages.read(url)
        except PageError as exc:
            msg = f"Couldn't read that page: {exc}."
            raise OperationError(msg) from exc
        host = urlsplit(page.url).hostname or ""
        title = f"{page.title} " if page.title else ""
        head = f"Page {title}({host}):"
        if page.image and self._shown is not None:
            # Before the text, so cutting a long page never takes it.
            found = ImageHit(page.title or host, page.image, page.url, 0, 0)
            head += f"\nPicture: [image_id {self._remember_image(found)}]"
        body = cut_inert(page.text, self._research.page_chars)
        return Result(f"{head}\n{body}", Origin.WEB)

    # --- pictures ---------------------------------------------------------------------------

    def _remember_image(self, found: ImageHit) -> str:
        """Return the id a picture is shown by; the oldest are forgotten past `REMEMBERED`."""
        self._image_number += 1
        image_id = f"i{self._image_number}"
        self._images[image_id] = found
        while len(self._images) > REMEMBERED:
            del self._images[next(iter(self._images))]
        return image_id

    def image_search(self, args: dict[str, Any]) -> Result:
        """Search for pictures and list them by id, for `show_image`.

        Raises:
            OperationError: the local search service failed.
        """
        query = str(args["query"])
        try:
            found = self._search.images(query, limit=int(args.get("count", self._research.results)))
        except SearchUnavailableError as exc:
            raise OperationError(str(exc)) from exc
        if not found:
            return Result("The search found no pictures.")
        lines: list[str] = []
        for number, hit in enumerate(found, 1):
            host = urlsplit(hit.page_url or hit.image_url).hostname
            size = f", {hit.width}x{hit.height}" if hit.width and hit.height else ""
            lines.append(
                f"{number}. {hit.title} ({host}){size} [image_id {self._remember_image(hit)}]"
            )
        return Result(f"Pictures for {query!r}:\n" + "\n".join(lines), Origin.WEB)

    def show_image(self, args: dict[str, Any]) -> Result:
        """Fetch a picture a recent image search or read page returned and show it in the chat.

        Raises:
            OperationError: the id came from no search, or the picture could not be fetched,
                read or kept.
        """
        image_id = str(args["image_id"]).strip().lower()
        found = self._images.get(image_id)
        if found is None or self._shown is None:
            msg = (
                "Only a picture a recent image_search or web_read returned can be shown; "
                "search first."
            )
            raise OperationError(msg)
        try:
            data = self._pages.picture(found.image_url)
        except PageError as exc:
            msg = f"Couldn't get that picture: {exc}."
            raise OperationError(msg) from exc
        try:
            picture = prepare(data, found.title, max_side=SHOWN_SIDE)
        except PictureError as exc:
            msg = f"That picture couldn't be shown: {exc}."
            raise OperationError(msg) from exc
        try:
            path = self._shown.keep(picture)
        except OSError as exc:
            msg = f"That picture couldn't be kept to show: {exc.strerror or type(exc).__name__}."
            raise OperationError(msg) from exc
        host = urlsplit(found.page_url or found.image_url).hostname or ""
        return Result(
            f"Shown to the owner in the chat: {found.title} ({host}), {picture.width}x"
            f"{picture.height}. You can't see it, so say no more about it than its title does.",
            Origin.WEB,
            picture=str(path),
            caption=f"{found.title} · {host}",
        )

    # --- reddit -----------------------------------------------------------------------------

    def reddit_search(self, args: dict[str, Any]) -> Result:
        """Find Reddit threads about something, optionally in one subreddit.

        Raises:
            OperationError: the subreddit name is not valid, or the search failed.
        """
        query = str(args["query"])
        subreddit = str(args.get("subreddit", "")).strip().removeprefix("r/")
        if subreddit and not SUBREDDIT.fullmatch(subreddit):
            msg = "That is not a subreddit name."
            raise OperationError(msg)
        count = int(args.get("count", self._research.results))
        site = f"reddit.com/r/{subreddit}" if subreddit else "reddit.com"
        found: list[tuple[str, str, Hit]] = []
        for hit in self._hits(query, count * 2, site):
            thread = _reddit_thread(hit.url)
            if thread is None:
                continue
            if all(post_id != thread[0] for post_id, _sub, _hit in found):
                found.append((thread[0], thread[1], hit))
            if len(found) >= count:
                break
        if not found:
            return Result("No Reddit threads were found.")
        try:
            archived = self._reddit.posts([post_id for post_id, _sub, _hit in found])
        except RedditUnavailableError:
            archived = {}
        lines: list[str] = []
        for number, (post_id, sub, hit) in enumerate(found, 1):
            self._remember(self._posts, post_id)
            post = archived.get(post_id)
            if post is not None:
                line = (
                    f"{number}. r/{post.subreddit}: {post.title} (score {post.score}, "
                    f"{post.comments} comments) [post_id {post_id}]"
                )
            else:
                line = f"{number}. r/{sub}: {hit.title} [post_id {post_id}]"
            lines.append(f"{line}\n   {hit.snippet}" if hit.snippet else line)
        return Result(f"Reddit threads for {query!r}:\n" + "\n".join(lines), Origin.WEB)

    def reddit_read(self, args: dict[str, Any]) -> Result:
        """Read a thread a recent reddit_search returned: the post and its top comments.

        Raises:
            OperationError: the thread came from no search, or the archive failed or lacks it.
        """
        post_id = str(args["post_id"]).strip().lower()
        if post_id not in self._posts:
            msg = "Only a thread a recent reddit_search returned can be read; search first."
            raise OperationError(msg)
        try:
            post: Post | None = self._reddit.posts([post_id]).get(post_id)
            comments = self._reddit.top_comments(post_id, self._research.reddit_comments)
        except RedditUnavailableError as exc:
            raise OperationError(str(exc)) from exc
        if post is None:
            msg = "The Reddit archive doesn't have that thread yet."
            raise OperationError(msg)
        parts = [f"r/{post.subreddit}: {post.title} (score {post.score}, {post.comments} comments)"]
        if post.body:
            parts.append(f"Post: {post.body[:MAX_POST_BODY]}")
        if comments:
            parts.append("Top comments:")
            parts.extend(f"- (score {c.score}) {c.body[:MAX_COMMENT]}" for c in comments)
        else:
            parts.append("No comments archived.")
        return Result(cut_inert("\n".join(parts), self._research.page_chars), Origin.WEB)

    @staticmethod
    def _remember(seen: deque[str], item: str) -> None:
        if item in seen:
            seen.remove(item)
        seen.append(item)

    # --- the operations ---------------------------------------------------------------------

    def operations(self) -> list[Operation]:
        """Return the time, weather, web and Reddit operations."""
        return [
            Operation(
                "time_now",
                "The current time or date, here or in a city or country (each of its time zones);"
                " never search the web for it. One per call: both only when asked for both.",
                {
                    "type": "object",
                    "properties": {
                        "what": {"type": "string", "enum": ["time", "date"], "default": "time"},
                        "place": {
                            **_PLACE,
                            "description": "A city or country the owner named; leave out for here",
                        },
                    },
                    "additionalProperties": False,
                },
                self.time,
                side_effect=False,
                outbound=True,
                outbound_args=_PLACE_ARG,
                spoken_summary=lambda _a: "tell you the time",
                shortcuts=(_TIME, _DATE),
            ),
            Operation(
                "weather",
                "The weather now, today, tomorrow or this week, at home or in a named place.",
                {
                    "type": "object",
                    "properties": {
                        "when": {
                            "type": "string",
                            "enum": ["now", "today", "tomorrow", "week"],
                            "default": "now",
                        },
                        "place": _PLACE,
                    },
                    "additionalProperties": False,
                },
                self.weather,
                side_effect=False,
                outbound=True,
                outbound_args=_PLACE_ARG,
                spoken_summary=lambda _a: "check the weather",
                shortcuts=(_WEATHER_NOW, _WEATHER_TODAY, _WEATHER_TOMORROW),
            ),
            Operation(
                "web_search",
                "Search the web for current facts, news or anything you don't know. Returns "
                "titles, addresses and snippets; read a page with web_read when the snippets "
                "aren't enough.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": MAX_QUERY},
                        "count": _COUNT,
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                self.web_search,
                side_effect=False,
                outbound=True,
                spoken_summary=lambda a: f"search the web for {str(a.get('query', ''))[:60]}",
            ),
            Operation(
                "web_read",
                "Read the main text of a page that web_search returned (pass its address exactly).",
                {
                    "type": "object",
                    "properties": {"url": {"type": "string", "minLength": 1, "maxLength": MAX_URL}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
                self.web_read,
                side_effect=False,
                spoken_summary=lambda a: f"read {urlsplit(str(a.get('url', ''))).hostname}",
            ),
            Operation(
                "reddit_search",
                "Find Reddit threads about something (what people say, recommendations, "
                "experiences), optionally in one subreddit. Read one with reddit_read.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": MAX_QUERY},
                        "subreddit": {
                            "type": "string",
                            "maxLength": 23,
                            "description": "Only this subreddit, when the owner named one",
                        },
                        "count": _COUNT,
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                self.reddit_search,
                side_effect=False,
                outbound=True,
                spoken_summary=lambda a: f"search Reddit for {str(a.get('query', ''))[:60]}",
            ),
            Operation(
                "reddit_read",
                "Read a Reddit thread that reddit_search returned: the post and its top comments.",
                {
                    "type": "object",
                    "properties": {"post_id": {"type": "string", "minLength": 1, "maxLength": 12}},
                    "required": ["post_id"],
                    "additionalProperties": False,
                },
                self.reddit_read,
                side_effect=False,
                spoken_summary=lambda _a: "read that Reddit thread",
            ),
            *(self._picture_operations() if self._shown is not None else []),
        ]

    def _picture_operations(self) -> list[Operation]:
        """Return `image_search` and `show_image`, for a channel that can show pictures."""
        return [
            Operation(
                "image_search",
                "Search for pictures of something to show the owner. Returns titles and an "
                "image_id for each; show one with show_image.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": MAX_QUERY},
                        "count": _COUNT,
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                self.image_search,
                side_effect=False,
                outbound=True,
                spoken_summary=lambda a: f"search for pictures of {str(a.get('query', ''))[:60]}",
            ),
            Operation(
                "show_image",
                "Show the owner a picture that image_search or web_read returned, by its image_id. "
                "You can't see it yourself.",
                {
                    "type": "object",
                    "properties": {"image_id": {"type": "string", "minLength": 1, "maxLength": 12}},
                    "required": ["image_id"],
                    "additionalProperties": False,
                },
                self.show_image,
                side_effect=False,
                spoken_summary=lambda _a: "show you a picture",
            ),
        ]
