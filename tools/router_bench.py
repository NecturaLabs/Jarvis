#!/usr/bin/env python3
"""Benchmark `tools = "all"` against `tools = "routed"` on a running model server (#56).

The same requests (`tools/router_bench_cases.json`: a typed set that covers every tool group, a
short spoken-style set, and requests that need two groups or none) go through Jarvis's real
agent, system prompt and tool schemas, for the voice channel (`[agent]`) and the chat channel
(`[chat]`), offering every tool and routed. Every tool that would touch the PC, Spotify or the
web is a stub that records the call and returns a canned result; `calculate`, `about_jarvis`,
the shipped skills and the reminders run for real in a throwaway state folder. The voice's
speaker is a stand-in that reads the clock, and every case runs with its own empty home and
XDG folders, so nothing of the owner's is read or written.

The model comes from `--base-url`, an OpenAI-compatible endpoint that is already running (Strata
serves `/v1`). This never starts, stops or loads a server, and touches no service: a server that
does not answer, or whose model is not loaded, is a refusal. A request that failed is counted
as an error, not as a wrong answer.

Per request it keeps the tools called with their arguments, whether they match the case, how
many model calls it took (a routed request pays a `use_tools` call), the time to the first
reply text (voice: the first sentence handed to the speaker, chat: the first draft of the reply)
and to the end, and the prompt tokens of every call when the server reports usage. It writes one
JSON file per variant into `--out` and prints a Markdown summary for `docs/models.md`.

Exit status: 0 measured, 2 refused (nothing could be measured), 3 measured with errors.

    uv run tools/router_bench.py --base-url http://127.0.0.1:8000/v1 --key-file KEYFILE --out DIR
    uv run tools/router_bench.py --base-url URL --key KEY --out DIR --channel voice --only music
"""

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest import mock

from first_audio import percentile

from jarvis import app, config
from jarvis.agent import TOO_LONG, TOO_MANY_STEPS, Agent, Hooks
from jarvis.audit import AuditLog
from jarvis.jsonutil import obj, objs
from jarvis.model import ChatModel, Delta, ModelStatus, StrataModel, Thinking
from jarvis.operations import Operation, Registry, Result
from jarvis.osskills import OsSkill
from jarvis.pc.screenshot import ScreenshotOps
from jarvis.pc.watch import ScreenWatch
from jarvis.pc.windows import WindowOps
from jarvis.policy import Origin
from jarvis.secretfile import read_api_key
from jarvis.spotify.api import SpotifyApi
from jarvis.spotify.auth import TokenStore
from jarvis.spotify.ops import SpotifyOps
from jarvis.toolgroups import CORE, GROUPS
from jarvis.web import browser

if TYPE_CHECKING:
    import httpx

type Channel = Literal["voice", "chat"]
type ToolMode = Literal["all", "routed"]
type Outcome = Literal["right", "wrong_tool", "wrong_args", "no_tool", "extra", "error"]

CASES_FILE = Path(__file__).with_name("router_bench_cases.json")
CHANNELS: tuple[Channel, ...] = ("voice", "chat")
MODES: tuple[ToolMode, ...] = ("all", "routed")
OUTCOMES: tuple[Outcome, ...] = ("right", "wrong_tool", "wrong_args", "no_tool", "extra", "error")
CORE_TOOLS = frozenset(CORE.tools)
GROUP_NAMES = frozenset(group.name for group in GROUPS)
CASE_GROUPS = GROUP_NAMES | {"core", "multi"}  # "multi": a request that needs more than one group
SETS = frozenset({"typed", "spoken"})
P90 = 0.9
MIN_CHOICES = 2  # an `any_of` lists at least two calls
EXIT_REFUSED = 2
EXIT_ERRORS = 3
WARM_UP = "How many days are there in a leap year?"  # sent first, so the server holds the prompt
# Off in every variant: a mode switch needs the mode machinery, a hand-off needs a chat window.
LEFT_OUT = ("switch_model", "chat_hand_off")
# Omarchy's guides as this benchmark offers them, the same on every machine.
GUIDES = (
    OsSkill(
        "omarchy",
        "Customise the desktop: Hyprland, themes, key bindings and the bar.",
        ("SKILL.md", "hyprland.md", "theming.md"),
    ),
)
SILENT = {"PIPEWIRE_REMOTE": "/nonexistent", "PULSE_SERVER": "unix:/nonexistent"}
OWN_FOLDERS = ("HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME")

# --- the request set -----------------------------------------------------------------------

type Spec = object  # a value, or one of the matchers below
MATCHERS = frozenset({"contains", "one_of", "lt", "gt"})


class CaseError(ValueError):
    """The request set is malformed; the message says which case and what is wrong."""


@dataclass(frozen=True, slots=True)
class Want:
    """One call a request needs: the tool and what each named argument must be."""

    tool: str
    args: Mapping[str, Spec] = field(default_factory=dict[str, Spec])


@dataclass(frozen=True, slots=True)
class Case:
    """One request (or a conversation of several) and what it needs done.

    Each entry of `expect` is a set of alternatives, one of which must be matched by a call of
    its own. A call that matches no entry is extra unless it is in `allow` or is a core tool; a
    call in `forbid` is always wrong.
    """

    id: str
    group: str
    kind: str
    turns: tuple[str, ...]
    expect: tuple[tuple[Want, ...], ...]
    allow: frozenset[str]
    forbid: frozenset[str]
    channels: frozenset[str]


def _text(raw: Mapping[str, Any], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{where}: `{key}` must be a non-empty string"
        raise CaseError(msg)
    return value


def _texts(
    raw: Mapping[str, Any], key: str, where: str, *, default: Sequence[str] = ()
) -> list[str]:
    value = raw.get(key, list(default))
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):  # pyright: ignore[reportUnknownVariableType]
        msg = f"{where}: `{key}` must be a list of non-empty strings"
        raise CaseError(msg)
    return cast("list[str]", value)


def _check_spec(spec: object, where: str) -> None:
    if isinstance(spec, dict):
        matcher = cast("dict[str, Any]", spec)
        if len(matcher) != 1 or next(iter(matcher)) not in MATCHERS:
            msg = f"{where}: a matcher is one of {sorted(MATCHERS)}, alone"
            raise CaseError(msg)
        kind, value = next(iter(matcher.items()))
        options = cast("list[object]", value) if isinstance(value, list) else []
        valid = {
            "contains": isinstance(value, str) and bool(value),
            "one_of": bool(options),
            "lt": _is_number(cast("object", value)),
            "gt": _is_number(cast("object", value)),
        }[kind]
        if not valid:
            msg = f"{where}: the value of `{kind}` is not usable"
            raise CaseError(msg)
    elif isinstance(spec, list | type(None)):
        msg = f"{where}: an argument is a string, number, boolean or matcher"
        raise CaseError(msg)


def _want(raw: Mapping[str, Any], where: str) -> Want:
    unknown = set(raw) - {"tool", "args"}
    if unknown:
        msg = f"{where}: unknown keys {sorted(unknown)}"
        raise CaseError(msg)
    args = raw.get("args", {})
    if not isinstance(args, dict):
        msg = f"{where}: `args` must be an object"
        raise CaseError(msg)
    checked = {str(name): spec for name, spec in cast("dict[str, Any]", args).items()}
    for name, spec in checked.items():
        _check_spec(spec, f"{where} argument {name}")
    return Want(_text(raw, "tool", where), checked)


def _entry(raw: object, where: str) -> tuple[Want, ...]:
    if not isinstance(raw, dict):
        msg = f"{where}: an expected call is an object"
        raise CaseError(msg)
    entry = cast("dict[str, Any]", raw)
    if "any_of" in entry:
        options = cast("object", entry["any_of"])
        if (
            len(entry) != 1
            or not isinstance(options, list)
            or len(cast("list[object]", options)) < MIN_CHOICES
        ):
            msg = f"{where}: `any_of` stands alone and lists at least two calls"
            raise CaseError(msg)
        return tuple(
            _want(cast("dict[str, Any]", option), where)
            for option in cast("list[Any]", options)
            if isinstance(option, dict)
        )
    return (_want(entry, where),)


def parse_case(raw: object) -> Case:
    """Return one request set entry, checked.

    Raises:
        CaseError: a key is missing, unknown or of the wrong kind.
    """
    if not isinstance(raw, dict):
        msg = "a case must be an object"
        raise CaseError(msg)
    fields = cast("dict[str, Any]", raw)
    name = fields.get("id", "?")
    where = f"case {name}"
    unknown = set(fields) - {"id", "group", "set", "turns", "expect", "allow", "forbid", "channels"}
    if unknown:
        msg = f"{where}: unknown keys {sorted(unknown)}"
        raise CaseError(msg)
    group, kind = _text(fields, "group", where), _text(fields, "set", where)
    if group not in CASE_GROUPS:
        msg = f"{where}: group {group!r} is not a tool group, `core` or `multi`"
        raise CaseError(msg)
    if kind not in SETS:
        msg = f"{where}: set must be one of {sorted(SETS)}"
        raise CaseError(msg)
    expect = fields.get("expect")
    if not isinstance(expect, list):
        msg = f"{where}: `expect` must be a list (empty when no tool is needed)"
        raise CaseError(msg)
    turns = _texts(fields, "turns", where)
    if not turns:
        msg = f"{where}: `turns` needs at least one request"
        raise CaseError(msg)
    channels = _texts(fields, "channels", where, default=CHANNELS)
    if not set(channels) <= set(CHANNELS):
        msg = f"{where}: channels must be among {list(CHANNELS)}"
        raise CaseError(msg)
    return Case(
        _text(fields, "id", where),
        group,
        kind,
        tuple(turns),
        tuple(_entry(entry, where) for entry in cast("list[object]", expect)),
        frozenset(_texts(fields, "allow", where)),
        frozenset(_texts(fields, "forbid", where)),
        frozenset(channels),
    )


def load_cases(path: Path = CASES_FILE) -> list[Case]:
    """Return the request set in `path`.

    Raises:
        CaseError: the file is not readable JSON, has no cases or repeats an id.
    """
    try:
        data = obj(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        msg = f"{path}: {exc}"
        raise CaseError(msg) from exc
    raw = data.get("cases")
    if not isinstance(raw, list) or not raw:
        msg = f"{path}: no `cases` list"
        raise CaseError(msg)
    cases = [parse_case(item) for item in cast("list[object]", raw)]
    repeated = [name for name, count in Counter(c.id for c in cases).items() if count > 1]
    if repeated:
        msg = f"{path}: repeated ids {repeated}"
        raise CaseError(msg)
    return cases


def select(
    cases: Sequence[Case], *, channel: Channel, only: Sequence[str] = (), kind: str = ""
) -> list[Case]:
    """Return the cases for one channel, narrowed to `only` (ids or groups) and a set."""
    return [
        c
        for c in cases
        if channel in c.channels
        and (not only or c.id in only or c.group in only)
        and (not kind or c.kind == kind)
    ]


# --- scoring ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Call:
    """A tool call that ran (or was asked for), with its arguments."""

    tool: str
    args: dict[str, Any]


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _same(spec: object, value: object) -> bool:
    if isinstance(spec, bool) or isinstance(value, bool):
        return spec is value
    if isinstance(spec, str) and isinstance(value, str):
        return spec.strip().casefold() == value.strip().casefold()
    if _is_number(spec) and _is_number(value):
        return math.isclose(float(cast("float", spec)), float(cast("float", value)))
    return False


def arg_matches(spec: Spec, value: object) -> bool:
    """Return whether an argument `value` satisfies `spec`: a value, or a matcher."""
    if not isinstance(spec, dict):
        return _same(spec, value)
    kind, wanted = next(iter(cast("dict[str, Any]", spec).items()))
    match kind:
        case "contains":
            return isinstance(value, str) and str(wanted).casefold() in value.casefold()
        case "one_of":
            return any(_same(option, value) for option in cast("list[object]", wanted))
        case "lt":
            return _is_number(value) and float(cast("float", value)) < float(wanted)
        case "gt":
            return _is_number(value) and float(cast("float", value)) > float(wanted)
        case _:
            return False


def call_matches(want: Want, call: Call) -> bool:
    """Return whether a call is the tool wanted with every named argument right."""
    return call.tool == want.tool and all(
        name in call.args and arg_matches(spec, call.args[name]) for name, spec in want.args.items()
    )


def _unmatched(case: Case, calls: Sequence[Call]) -> tuple[list[Call], list[tuple[Want, ...]]]:
    matched: dict[int, int] = {}

    def assign(call: int, seen: set[int]) -> bool:
        for entry, alternatives in enumerate(case.expect):
            if entry in seen or not any(call_matches(want, calls[call]) for want in alternatives):
                continue
            seen.add(entry)
            previous = matched.get(entry)
            if previous is None or assign(previous, seen):
                matched[entry] = call
                return True
        return False

    # Prefer matching non-optional calls. Move an earlier match to another entry when
    # that frees the only entry a later call can satisfy.
    optional = case.allow | CORE_TOOLS
    for index in sorted(range(len(calls)), key=lambda i: calls[i].tool in optional):
        assign(index, set())
    used = set(matched.values())
    return (
        [call for index, call in enumerate(calls) if index not in used],
        [entry for index, entry in enumerate(case.expect) if index not in matched],
    )


def judge(case: Case, calls: Sequence[Call], *, failed: bool = False) -> Outcome:
    """Return how a request went, from the calls that ran.

    `right` needs every expected call matched and nothing else (apart from core tools and what
    the case allows). Otherwise: `error` (the model failed), `wrong_args` (a tool the request
    needs was called, but not as it needs), `no_tool` (it answered, or only routed, without
    calling the tool it needed), `wrong_tool` (it called other tools instead) or `extra` (all it
    needed was done, and something else too, or something forbidden).
    """
    if failed:
        return "error"
    unmatched, missing = _unmatched(case, calls)
    if missing:
        wanted = {w.tool for alternatives in missing for w in alternatives}
        if any(c.tool in wanted for c in unmatched):
            return "wrong_args"
        others = [c for c in unmatched if c.tool not in CORE_TOOLS and c.tool not in case.allow]
        return "wrong_tool" if others else "no_tool"
    if any(c.tool in case.forbid for c in calls):
        return "extra"
    if any(c.tool not in case.allow and c.tool not in CORE_TOOLS for c in unmatched):
        return "extra"
    return "right"


# --- recording ----------------------------------------------------------------------------


@dataclass(slots=True)
class ModelCall:
    """One call to the model: what it was offered, what the server counted, how long it took."""

    offered: int
    seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reused_tokens: int | None = None  # of the prompt, as the server reports (llama.cpp's cache_n)
    asked: list[str] = field(default_factory=list[str])


def _count(source: Mapping[str, Any], key: str) -> int | None:
    value = source.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class Probe:
    """A model that notes each call (tools offered, tokens, time, tools asked) and passes it on."""

    def __init__(self, inner: ChatModel, clock: Callable[[], float] = time.perf_counter) -> None:
        """Wrap `inner`; `clock` times the calls."""
        super().__init__()
        self._inner = inner
        self._clock = clock
        self.calls: list[ModelCall] = []
        self.asked: list[Call] = []

    def is_loaded(self) -> bool | None:
        """Pass on whether the model is loaded."""
        return self._inner.is_loaded()

    def chat(  # ruff: ignore[too-many-arguments] - the ChatModel protocol's keyword-only settings
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        require_tool: bool | str = False,
        timeout_s: float | None = None,
        on_delta: Callable[[Delta], None] | None = None,
        max_tokens: int | None = None,
        load_s: float = 0.0,
        thinking: Thinking | None = None,
    ) -> dict[str, Any]:
        """Send the call to the wrapped model and note it."""
        began = self._clock()
        message = self._inner.chat(
            messages,
            tools,
            require_tool=require_tool,
            timeout_s=timeout_s,
            on_delta=on_delta,
            max_tokens=max_tokens,
            load_s=load_s,
            thinking=thinking,
        )
        usage, timings = obj(message.get("usage")), obj(message.get("timings"))
        asked = [
            Call(
                str(obj(c.get("function")).get("name", "")),
                _parsed(obj(c.get("function")).get("arguments")),
            )
            for c in objs(message.get("tool_calls"))
        ]
        self.asked += asked
        self.calls.append(
            ModelCall(
                len(tools),
                self._clock() - began,
                _count(usage, "prompt_tokens"),
                _count(usage, "completion_tokens"),
                _count(timings, "cache_n"),
                [c.tool for c in asked],
            )
        )
        return message


def _parsed(arguments: object) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return cast("dict[str, Any]", arguments)
    try:
        return obj(json.loads(str(arguments or "{}")))
    except ValueError:
        return {}


@dataclass(slots=True)
class Turn:
    """One request of a case: its words, the reply, the model calls it took and its times."""

    text: str
    reply: str
    first_s: float | None  # to the first reply text; None: nothing was said
    total_s: float
    model_calls: list[ModelCall]


@dataclass(slots=True)
class CaseResult:
    """A case run once: how it was judged, the calls that ran and the requests it took."""

    id: str
    group: str
    kind: str
    run: int
    outcome: Outcome
    reason: str
    calls: list[Call]
    refused: list[str]  # tools the model asked for that did not run (not offered, or invalid)
    loaded: list[str]  # groups asked for with `use_tools`
    turns: list[Turn]


def failure(reply: str, events: Sequence[Mapping[str, object]]) -> str:
    """Return why a request failed, or "" when the model answered it.

    The agent speaks a fixed apology when the model fails; the audit events say which failure.
    """
    names = {str(event["event"]) for event in events}
    if "model_error" in names:
        return "the model did not answer"
    if reply == TOO_LONG:
        return "ran out of time"
    if reply == TOO_MANY_STEPS:
        return "ran out of steps"
    if "reply_limit" in names:
        return "reply cut at the token limit"
    return ""


@dataclass(slots=True)
class Session:
    """A case's agent with everything that watches it: the probe, the stubs' calls, the audit."""

    agent: Agent
    registry: Registry
    probe: Probe
    ran: list[Call]
    audit: AuditLog
    channel: Channel
    clock: Callable[[], float] = time.perf_counter


def _hooks(session: Session, first: list[float], began: float) -> Hooks:
    def note(_text: str) -> None:
        if not first:
            first.append(session.clock() - began)

    def yes(*_args: str) -> bool:
        return True

    return Hooks(
        confirm=yes,
        confirm_always=yes,
        confirm_key=yes,
        speak=note if session.channel == "voice" else None,
        draft=note if session.channel == "chat" else None,
    )


def run_case(case: Case, session: Session, run: int = 1) -> CaseResult:
    """Send a case's requests one after the other and judge what ran."""
    turns: list[Turn] = []
    reasons: list[str] = []
    for text in case.turns:
        first: list[float] = []
        began = session.clock()
        seen = len(session.probe.calls)
        mark = len(session.audit.events)
        try:
            reply = session.agent.handle(text, _hooks(session, first, began))
        except Exception as exc:  # ruff: ignore[blind-except] - one request's failure is a result, not the end of the run
            reply = ""
            reasons.append(f"raised {type(exc).__name__}")
        else:
            if why := failure(reply, session.audit.events[mark:]):
                reasons.append(why)
        total = session.clock() - began
        if not first and reply:
            first.append(total)  # a reply that came whole, with no earlier draft of it
        turns.append(
            Turn(text, reply, first[0] if first else None, total, session.probe.calls[seen:])
        )
    ran = list(session.ran)
    asked = Counter(c.tool for c in session.probe.asked)
    asked.subtract(Counter(c.tool for c in ran))
    loaded = [
        str(group)
        for call in session.probe.asked
        if call.tool == "use_tools"
        for group in cast("list[object]", call.args.get("groups") or [])
    ]
    return CaseResult(
        case.id,
        case.group,
        case.kind,
        run,
        judge(case, ran, failed=bool(reasons)),
        "; ".join(reasons),
        ran,
        sorted(tool for tool, extra in asked.items() for _ in range(max(0, extra))),
        loaded,
        turns,
    )


# --- the agent under test -----------------------------------------------------------------

type Stub = Callable[[dict[str, Any]], Result]
T = Origin.TOOL_STATUS


def _say(text: str, origin: Origin = T) -> Stub:
    return lambda _args: Result(text, origin)


def _of(make: Callable[[dict[str, Any]], str], origin: Origin = T) -> Stub:
    return lambda args: Result(make(args), origin)


# What each tool that would act on the PC, Spotify or the web answers. A tool not here and not in
# REAL is a mistake the tests catch.
STUBS: dict[str, Stub] = {
    "spotify_play": _of(
        lambda a: f"Playing {a.get('query') or 'your liked songs'} ({a.get('kind')})."
    ),
    "spotify_pause": _say("Paused."),
    "spotify_resume": _say("Resumed."),
    "spotify_seek": _say("Moved within the song."),
    "spotify_next": _of(lambda a: f"Skipped {a.get('count', 1)} track(s)."),
    "spotify_previous": _say("Went back one track."),
    "spotify_set_volume": _of(lambda a: f"Volume set to {a.get('percent')}%."),
    "spotify_change_volume": _of(
        lambda a: f"Volume changed by {a.get('delta')}, now {60 + int(a.get('delta') or 0)}%."
    ),
    "spotify_shuffle": _say("Shuffle changed."),
    "spotify_repeat": _say("Repeat changed."),
    "spotify_queue": _of(
        lambda a: f"Queued {a.get('count', 1)} song(s) for {a.get('artist') or a.get('query')}."
    ),
    "spotify_like": _say("Liked."),
    "spotify_library": _say("Library changed."),
    "spotify_now_playing": _say(
        "Now playing: One More Time by Daft Punk (Discovery), 1:12 of 5:20, volume 60%.",
        Origin.SPOTIFY,
    ),
    "pc_volume": _of(
        lambda a: f"PC volume is now {a.get('percent', 50)}%{' (muted)' if a.get('mute') else ''}."
    ),
    "pc_media": _say("Done: the media player was told."),
    "pc_audio_listen": _say("Heard a song with a singer and a steady beat.", Origin.PC_AUDIO),
    "sing": _say("Sang it."),
    "voice_record": _of(lambda a: "Recording." if a.get("action") == "start" else "Saved it."),
    "pc_windows": _say(
        "Windows (number, program, workspace): 1 Firefox on 1; 2 Terminal on 1;"
        " 3 Discord on 2; 4 Spotify on background.",
        Origin.PROCESS,
    ),
    "pc_window": _say("Done."),
    "pc_window_close": _say("Closed the window."),
    "pc_screenshot": _say("Took a screenshot and saved it to Pictures/Screenshots."),
    "pc_watch": _say("Watch 1 started."),
    "pc_watches": _say("1 watch running: screen, error dialog."),
    "pc_watch_stop": _say("Watch stopped."),
    "open_app": _of(lambda a: f"Opened {a.get('app')} on workspace {a.get('where')}."),
    "find_processes": _of(
        lambda a: (
            f"1 match for {a.get('name')}: PID 4242 discord (window 'Discord'), started 09:14."
        ),
        Origin.PROCESS,
    ),
    "stop_process": _of(lambda a: f"Stopped PID {a.get('pid')} (discord)."),
    "pc_stats": _say(
        "CPU 12% at 48 °C; RAM 21 of 60 GiB; GPU 31% load at 54 °C, VRAM 9.6 of 20 GiB;"
        " disk / 61% full."
    ),
    "pc_shell": _say("6.12.0-arch1-1", Origin.SHELL),
    "list_dir": _say("Documents: Budget.ods, CV.pdf, notes.txt, notes/, recipe.txt", Origin.FILE),
    "read_file": _say(
        "Buy bread. Call the dentist on Friday. Renew the library card.", Origin.FILE
    ),
    "write_file": _say("Wrote the file."),
    "time_now": _of(
        lambda a: f"In {a.get('place') or 'here'} it is 4:32 AM on Tuesday 24 November."
    ),
    "weather": _of(
        lambda a: (
            f"{a.get('place') or 'Home'}, {a.get('when') or 'now'}: light rain likely in the"
            " afternoon (70%), 9 to 13 °C."
        ),
        Origin.WEB,
    ),
    "web_search": _of(
        lambda a: (
            f"1. https://example.org/a - {a.get('query')}: Lando Norris won the Abu Dhabi Grand"
            " Prix on 22 November.\n"
            "2. https://example.org/b - A background article."
        ),
        Origin.WEB,
    ),
    "web_read": _say(
        "A short article: the main point is that the new schedule starts in spring.", Origin.WEB
    ),
    "reddit_search": _say(
        "1. post abc123 r/SteamDeck 'OLED one year later' (2.1k points)", Origin.WEB
    ),
    "reddit_read": _say(
        "Top comments: 'the screen is a huge upgrade', 'battery life is much better'.", Origin.WEB
    ),
    "image_search": _say("3 pictures found: img1, img2, img3.", Origin.WEB),
    "show_image": _say("Showing the picture."),
    "browser_open": _say("Opened the page: 'Example Domain'.", Origin.WEB),
    "browser_preview": _say("Opened the preview of index.html: 'Built'.", Origin.WEB),
    "browser_act": _say("Done."),
    "browser_screenshot": _say("Showing the page."),
    "os_skill": _say(
        "In the general section of the Hyprland configuration, gaps_in sets the space between "
        "windows and gaps_out sets the space between windows and the edge of the screen.",
        Origin.OS_SKILL,
    ),
    "read_jarvis_file": _say("[agent]\nreply_tokens = 400\n", Origin.FILE),
}
# What runs as shipped, on a throwaway state folder: pure arithmetic, Jarvis's own facts, the
# reminders, the shipped skills and the core tools, which are the agent's own words and routing.
REAL = frozenset(
    {
        "calculate",
        "about_jarvis",
        "read_skill",
        "schedule_add",
        "schedule_list",
        "schedule_cancel",
        *CORE.tools,
    }
)


@contextlib.contextmanager
def isolated(root: Path) -> Generator[None]:
    """Run with home and the XDG folders inside `root` and every sound cut, then put them back."""
    kept = {name: os.environ.get(name) for name in (*OWN_FOLDERS, "XDG_RUNTIME_DIR", *SILENT)}
    for name in (*OWN_FOLDERS, "XDG_RUNTIME_DIR"):
        folder = root / name.lower()
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(folder)
    os.environ.update(SILENT)
    try:
        yield
    finally:
        for name, value in kept.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


_WHICH = shutil.which


def _program(name: str) -> str | None:
    """Pretend agent-browser is installed, so the browser tools are offered on any machine."""
    return "/usr/bin/agent-browser" if name == browser.PROGRAM else _WHICH(name)


@contextlib.contextmanager
def offering_everything() -> Generator[None]:
    """Offer the browser tools and a fixed set of Omarchy guides, whatever this machine has."""
    with (
        mock.patch.object(app.shutil, "which", _program),
        mock.patch.object(app.osskills, "catalog", return_value=GUIDES),
    ):
        yield


@dataclass(frozen=True, slots=True)
class Variant:
    """What one run of the request set varies: the channel, the tool mode, thinking, the window."""

    channel: Channel
    mode: ToolMode
    thinking: str = ""  # "": as configured
    context: int = 0  # tokens; 0: as configured


def settings_for(base: config.Config, variant: Variant) -> config.Config:
    """Return the config a variant runs with: its tool mode, every tool switched on, overrides."""
    section = "agent" if variant.channel == "voice" else "chat"
    update: dict[str, Any] = {"tools": variant.mode}
    if variant.thinking:
        update["thinking"] = variant.thinking
    changed = getattr(base, section).model_copy(update=update)
    return base.model_copy(
        update={
            section: changed,
            "audio": base.audio.model_copy(update={"listen_to_pc": True}),
            "shell": base.shell.model_copy(update={"enabled": True}),
            **(
                {"strata": base.strata.model_copy(update={"context": variant.context})}
                if variant.context
                else {}
            ),
        }
    )


def stubbed(registry: Registry, ran: list[Call]) -> Registry:
    """Return `registry` with each operation recording its call, and a stub where it would act."""
    ops: list[Operation] = []
    for name in registry.names():
        op = registry.get(name)
        if op is None:  # pragma: no cover - names() lists operations that are there
            continue
        run = op.run if name in REAL else STUBS.get(name, _say(f"{name} done."))

        def recorded(args: dict[str, Any], name: str = name, run: Stub = run) -> Result:
            ran.append(Call(name, dict(args)))
            return run(args)

        # Shell preparation checks provenance and binds a directory. Its runner is stubbed.
        prepare = op.prepare if name == "pc_shell" else None
        ops.append(replace(op, run=recorded, prepare=prepare, shortcuts=()))
    return Registry(ops)


def make_session(
    base: config.Config,
    model: ChatModel,
    variant: Variant,
    root: Path,
    clock: Callable[[], float] = time.perf_counter,
) -> Session:
    """Return a fresh agent for one case: Jarvis's own prompt, limits and tools for the channel.

    Call it inside `isolated(root)` and `offering_everything()`: the registry reads where
    Jarvis keeps its state while it is made.
    """
    channel = variant.channel
    cfg = settings_for(base, variant)
    limits = app.channel_limits(cfg, channel)
    spotify = SpotifyOps(SpotifyApi(TokenStore("bench", root / "spotify-token.json")), cfg.spotify)
    windows = WindowOps()
    watch = ScreenWatch(windows, ScreenshotOps(windows, root / "screenshots"))
    built = app.channel_registry(
        cfg,
        spotify.operations(),
        limits,
        channel,
        watching=watch if channel == "voice" else None,
        off=LEFT_OUT,
    )
    ran: list[Call] = []
    registry = stubbed(built, ran)
    probe = Probe(model, clock)
    audit = AuditLog(None)
    agent = Agent(
        probe,
        registry,
        app.system_prompt(cfg, channel, registry=registry),
        audit,
        {},
        clock=clock,
        names=(cfg.persona.name,),
        limits=limits,
        bypass_consent=cfg.boundaries.bypass_consent,
    )
    return Session(agent, registry, probe, ran, audit, channel, clock)


def run_variant(  # ruff: ignore[too-many-arguments] - the cases, what they run on and how it is timed and shown
    cases: Sequence[Case],
    base: config.Config,
    model: ChatModel,
    variant: Variant,
    *,
    runs: int = 1,
    clock: Callable[[], float] = time.perf_counter,
    progress: Callable[[str], None] = lambda _line: None,
) -> list[CaseResult]:
    """Run the cases for one variant, `runs` times each, after one warm-up request."""
    results: list[CaseResult] = []
    with tempfile.TemporaryDirectory(prefix="jarvis-router-bench-") as folder:
        warm = Case(
            "warm-up",
            "core",
            "typed",
            (WARM_UP,),
            (),
            frozenset(),
            frozenset(),
            frozenset(CHANNELS),
        )
        todo = [(warm, 0), *((case, run) for run in range(1, runs + 1) for case in cases)]
        for number, (case, run) in enumerate(todo):
            root = Path(folder) / f"case-{number}"
            with isolated(root), offering_everything():
                session = make_session(base, model, variant, root, clock)
                result = run_case(case, session, run)
            if run == 0:
                continue  # the warm-up only loads the prompt on the server
            results.append(result)
            progress(
                f"{variant.channel}/{variant.mode} {case.id}: {result.outcome}"
                f" {sum(len(t.model_calls) for t in result.turns)} calls"
                f" {sum(t.total_s for t in result.turns):.1f}s {[c.tool for c in result.calls]}"
            )
    return results


# --- the report ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Summary:
    """What one variant measured, over every request of every case it ran."""

    channel: str
    tools: str
    cases: int
    outcomes: dict[str, int]
    requests: int
    calls_per_request: float
    first_median_s: float | None
    first_p90_s: float | None
    total_median_s: float
    total_p90_s: float
    prompt_tokens_per_call: float | None
    prompt_tokens_known: int
    reused_tokens_per_call: float | None
    by_group: dict[str, tuple[int, int]]  # right and total, per case group


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(channel: str, tools: str, results: Sequence[CaseResult]) -> Summary:
    """Return a variant's totals: outcomes, model calls, times and prompt tokens."""
    turns = [turn for result in results for turn in result.turns]
    firsts = [t.first_s for t in turns if t.first_s is not None]
    totals = [t.total_s for t in turns]
    model_calls = [call for turn in turns for call in turn.model_calls]
    prompts = [c.prompt_tokens for c in model_calls if c.prompt_tokens is not None]
    reused = [c.reused_tokens for c in model_calls if c.reused_tokens is not None]
    groups: dict[str, tuple[int, int]] = {}
    for result in results:
        right, total = groups.get(result.group, (0, 0))
        groups[result.group] = (right + (result.outcome == "right"), total + 1)
    counts = Counter(r.outcome for r in results)
    return Summary(
        channel,
        tools,
        len(results),
        {name: counts[name] for name in OUTCOMES},
        len(turns),
        len(model_calls) / len(turns) if turns else math.nan,
        statistics.median(firsts) if firsts else None,
        percentile(firsts, P90) if firsts else None,
        statistics.median(totals) if totals else math.nan,
        percentile(totals, P90) if totals else math.nan,
        _mean(prompts),
        len(prompts),
        _mean(reused),
        groups,
    )


def _seconds(value: float | None) -> str:
    return "-" if value is None or math.isnan(value) else f"{value:.1f}"


def _share(right: int, total: int) -> str:
    return f"{right}/{total} ({100 * right / total:.0f}%)" if total else "-"


def differing(by_variant: Mapping[tuple[str, str], Sequence[CaseResult]]) -> list[str]:
    """Return a line for each case whose success differs between `all` and `routed` in a channel."""
    lines: list[str] = []
    for channel in CHANNELS:
        everything, routed = by_variant.get((channel, "all")), by_variant.get((channel, "routed"))
        if everything is None or routed is None:
            continue
        for case in dict.fromkeys(r.id for r in everything):
            mine = [r for r in everything if r.id == case]
            theirs = [r for r in routed if r.id == case]
            wins = (
                sum(r.outcome == "right" for r in mine),
                sum(r.outcome == "right" for r in theirs),
            )
            if theirs and wins[0] / len(mine) != wins[1] / len(theirs):
                how = Counter(r.outcome for r in theirs if r.outcome != "right")
                lines.append(
                    f"- {channel} `{case}`: all {wins[0]}/{len(mine)},"
                    f" routed {wins[1]}/{len(theirs)}"
                    + (f" ({', '.join(f'{n} {k}' for k, n in how.items())})" if how else "")
                )
    return lines


@dataclass(frozen=True, slots=True)
class Report:
    """What a set of results says about itself: whose model, when, how often, how it thought."""

    label: str
    day: str
    runs: int
    thinking: str = ""
    context: int = 0


def markdown(
    report: Report,
    summaries: Sequence[Summary],
    by_variant: Mapping[tuple[str, str], Sequence[CaseResult]],
) -> str:
    """Return the tables for `docs/models.md`: outcomes and speed, and accuracy per group."""
    cases = sorted({(r.id, r.kind) for results in by_variant.values() for r in results})
    typed = sum(kind == "typed" for _, kind in cases)
    spoken = len(cases) - typed
    times = f"{report.runs} run{'s' if report.runs != 1 else ''}"
    window = f", context {report.context}" if report.context else ""
    intro = (
        f"Router benchmark of {report.label}, {report.day}: {len(cases)} requests"
        f" ({typed} typed, {spoken} spoken-style), {times} each,"
        f" thinking `{report.thinking or 'as configured'}`{window}."
        " A case is right when it called what it needed and nothing else."
    )
    columns = (
        "| Channel | Tools | Right | Wrong tool | Wrong args | Answered without the tool | Extra"
        " | Error | Model calls per request | First reply, median / p90 (s)"
        " | Done, median / p90 (s) | Prompt tokens per call |"
    )
    head = [intro, "", columns, "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = [
        f"| {s.channel} | {s.tools} | {_share(s.outcomes['right'], s.cases)}"
        f" | {s.outcomes['wrong_tool']} | {s.outcomes['wrong_args']} | {s.outcomes['no_tool']}"
        f" | {s.outcomes['extra']} | {s.outcomes['error']}"
        f" | {s.calls_per_request:.2f} | {_seconds(s.first_median_s)} / {_seconds(s.first_p90_s)}"
        f" | {_seconds(s.total_median_s)} / {_seconds(s.total_p90_s)}"
        f" | {'-' if s.prompt_tokens_per_call is None else f'{s.prompt_tokens_per_call:,.0f}'} |"
        for s in summaries
    ]
    names = [f"{s.channel} {s.tools}" for s in summaries]
    groups = sorted({g for s in summaries for g in s.by_group})
    per_group = [
        "",
        "| Group | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
        *(
            f"| {group} | "
            + " | ".join(_share(*s.by_group.get(group, (0, 0))) for s in summaries)
            + " |"
            for group in groups
        ),
    ]
    different = differing(by_variant)
    tail = ["", "Cases that differ between `all` and `routed`:", *(different or ["- none"])]
    return "\n".join([*head, *rows, *per_group, *tail])


def write_results(
    out: Path, report: Report, summary: Summary, results: Sequence[CaseResult]
) -> Path:
    """Write one variant's summary and every case's record as JSON; return the file."""
    out.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", report.label).strip("-") or "model"
    path = out / f"{name}-{summary.channel}-{summary.tools}.json"
    document = {
        "report": asdict(report),
        "summary": asdict(summary),
        "cases": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# --- the command --------------------------------------------------------------------------


class UnavailableError(Exception):
    """The model cannot be measured now: the server does not answer, or its model is not loaded."""


def connect(
    base_url: str,
    key: str,
    model: str,
    timeout_s: float,
    transport: httpx.BaseTransport | None = None,
) -> tuple[StrataModel, ModelStatus]:
    """Return the client for a running server and what it says about itself.

    Only the server's model list is read to find out: nothing is generated, loaded or started.

    Raises:
        UnavailableError: no answer from the server, or its model is listed as not loaded.
    """
    endpoint = config.ModelConfig().model_copy(
        update={"base_url": base_url.rstrip("/"), "model": model, "request_timeout_s": timeout_s}
    )
    client = StrataModel(endpoint, key, transport=transport)
    status = client.probe()
    if not status.reachable:
        msg = f"no answer from the model server ({status.issue or 'unreachable'})"
        raise UnavailableError(msg)
    if status.state not in {None, "loaded"}:
        msg = f"the model is {status.state}, not loaded; this loads nothing"
        raise UnavailableError(msg)
    return client, status


def _positive(text: str) -> int:
    number = int(text)
    if number < 1:
        msg = "must be at least 1"
        raise argparse.ArgumentTypeError(msg)
    return number


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="router_bench.py",
        description='Compare tools = "all" with tools = "routed" on a running model server.',
    )
    parser.add_argument(
        "--base-url", required=True, help="the server's OpenAI endpoint, ending in /v1"
    )
    secret = parser.add_mutually_exclusive_group(required=True)
    secret.add_argument(
        "--key", help="the API key (visible to other users of this PC; prefer --key-file)"
    )
    secret.add_argument(
        "--key-file",
        type=Path,
        help="a private file with a STRATA_API_KEY=... line (Jarvis's own strata.env is one)",
    )
    parser.add_argument("--out", type=Path, required=True, help="the folder for the JSON results")
    parser.add_argument("--model", default="jarvis", help="the model name the server expects")
    parser.add_argument(
        "--label", default="", help="the model's name in the results (default: --model)"
    )
    parser.add_argument("--cases", type=Path, default=CASES_FILE, help="the request set")
    parser.add_argument(
        "--config",
        type=Path,
        help="a config file for Jarvis's own settings (default: its defaults)",
    )
    parser.add_argument(
        "--channel", action="append", choices=CHANNELS, help="only this channel (repeatable)"
    )
    parser.add_argument(
        "--tools", action="append", choices=MODES, help="only this tool mode (repeatable)"
    )
    parser.add_argument(
        "--set", dest="kind", choices=sorted(SETS), default="", help="only typed or spoken requests"
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="ID_OR_GROUP",
        help="only these cases or groups (repeatable)",
    )
    parser.add_argument(
        "--runs", type=_positive, default=1, help="times each case runs per variant (default 1)"
    )
    parser.add_argument(
        "--thinking",
        choices=("none", "low", "medium", "high"),
        default="",
        help="override the channels' thinking",
    )
    parser.add_argument(
        "--context",
        type=_positive,
        default=0,
        help="the server's context window in tokens (default: what it reports)",
    )
    parser.add_argument(
        "--timeout", type=float, default=240.0, help="seconds one model call may take"
    )
    return parser


def _key(args: argparse.Namespace) -> str:
    """Return the API key given, or the one in the private environment file given.

    A key file that is not private to its owner is refused, as everywhere in Jarvis.

    Raises:
        UnavailableError: the file has no `STRATA_API_KEY=` line.
    """
    if args.key is not None:
        return str(args.key).strip()
    key = read_api_key(args.key_file)
    if not key:
        msg = f"no STRATA_API_KEY line in {args.key_file}"
        raise UnavailableError(msg)
    return key


def main(
    argv: Sequence[str] | None = None,
    *,
    connector: Callable[[str, str, str, float], tuple[ChatModel, ModelStatus]] = connect,
    clock: Callable[[], float] = time.perf_counter,
    today: Callable[[], str] = lambda: datetime.now(UTC).strftime("%Y-%m-%d"),
) -> int:
    """Run the benchmark and return the exit status: 0 measured, 2 refused, 3 with errors."""
    args = build_parser().parse_args(argv)
    try:
        key = _key(args)
        cases = load_cases(args.cases)
        base = config.load(args.config) if args.config else config.Config()
        model, status = connector(args.base_url, key, args.model, args.timeout)
    except (OSError, CaseError, config.ConfigError, UnavailableError) as exc:
        print(f"router_bench: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    context = args.context or status.context or 0
    report = Report(
        args.label or status.name or args.model, today(), args.runs, args.thinking, context
    )
    by_variant: dict[tuple[str, str], list[CaseResult]] = {}
    summaries: list[Summary] = []
    for channel in args.channel or CHANNELS:
        chosen = select(cases, channel=channel, only=args.only, kind=args.kind)
        for mode in args.tools or MODES:
            if not chosen:
                continue
            variant = Variant(channel, mode, args.thinking, context)
            results = run_variant(
                chosen,
                base,
                model,
                variant,
                runs=args.runs,
                clock=clock,
                progress=lambda line: print(line, file=sys.stderr, flush=True),
            )
            summary = summarize(channel, mode, results)
            by_variant[channel, mode] = results
            summaries.append(summary)
            write_results(args.out, report, summary, results)
    if not summaries:
        print("router_bench: no case matches the choice", file=sys.stderr)
        return EXIT_REFUSED
    print(markdown(report, summaries, by_variant))
    return EXIT_ERRORS if any(s.outcomes["error"] for s in summaries) else 0


if __name__ == "__main__":
    sys.exit(main())
