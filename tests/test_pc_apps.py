"""`open_app`: a program opens only when the owner asks, exactly where the owner said."""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from jarvis.config import AppConfig, Config
from jarvis.operations import (
    ArgumentError,
    OperationError,
    Result,
    request_scope,
    validate_args,
)
from jarvis.pc import hyprland
from jarvis.pc.apps import OPEN_WAIT_S, SPECIAL, AppOps, HyprlandCompositor, Placed

if TYPE_CHECKING:
    from collections.abc import Callable

SPOTIFY = AppConfig(command=["uwsm-app", "--", "spotify"], window=r"\bspotify\b")


@dataclass
class FakeHyprland:
    """Windows and Lua as Hyprland would take them; `opens_on` is where a started app maps."""

    opens_on: str | None = "3"
    window_class: str = "Spotify"
    moves_work: bool = True
    takes_commands: bool = True
    open: list[Placed] = field(default_factory=list[Placed])
    ran: list[str] = field(default_factory=list[str])
    shown: set[str] = field(default_factory=lambda: {"1"})
    readable: bool = True

    def windows(self) -> list[Placed]:
        if not self.readable:
            raise OperationError("Hyprland did not give its window list, so nothing was opened")
        return list(self.open)

    def showing(self) -> set[str]:
        return set(self.shown)

    def run(self, lua: str) -> bool:
        self.ran.append(lua)
        if not self.takes_commands:
            return False
        if lua.startswith("hl.exec_cmd(") and self.opens_on is not None:
            self.open.append(Placed("0xabc", self.opens_on, self.window_class, "Spotify Premium"))
        elif lua.startswith("hl.dispatch(") and self.moves_work:
            target = lua.split("workspace = ", 1)[1].split(",", 1)[0].strip('"')
            address = lua.split('"address:', 1)[1].split('"', 1)[0]
            self.open = [
                Placed(w.address, target, w.klass, w.title) if w.address == address else w
                for w in self.open
            ]
        return True


@dataclass
class Clock:
    now: float = 1000.0
    naps: list[float] = field(default_factory=list[float])

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.naps.append(seconds)
        self.now += seconds


def _ops(fake: FakeHyprland, clock: Clock | None = None) -> AppOps:
    clock = clock or Clock()
    return AppOps({"spotify": SPOTIFY}, fake, clock=clock, sleep=clock.sleep)


def _opened(ops: AppOps, where: str, *, foreign: bool = False) -> Result:
    (op,) = ops.operations()
    with request_scope(foreign=foreign):
        return op.run(validate_args(op, {"app": "spotify", "where": where}))


def _open(ops: AppOps, where: str, *, foreign: bool = False) -> str:
    return _opened(ops, where, foreign=foreign).text


Listed = list[dict[str, object]] | None


def _answers(
    value: list[dict[str, object]] | None,
) -> Callable[..., list[dict[str, object]] | None]:
    def listing(*_args: object) -> list[dict[str, object]] | None:
        return value

    return listing


def test_it_opens_on_the_workspace_the_owner_named_without_focus() -> None:
    fake = FakeHyprland(opens_on="3")

    text = _open(_ops(fake), "3")

    assert text == "Opened spotify on workspace 3."
    assert fake.ran == [
        f"hl.exec_cmd({hyprland.lua_string('[workspace 3 silent] uwsm-app -- spotify')})"
    ]


def test_background_is_a_special_workspace_out_of_sight() -> None:
    fake = FakeHyprland(opens_on=SPECIAL)

    text = _open(_ops(fake), "background")

    assert text == "Opened spotify in the background."
    assert "[workspace special:jarvis silent]" in fake.ran[0]


def test_a_window_that_lands_elsewhere_is_moved_and_checked_again() -> None:
    fake = FakeHyprland(opens_on="1")

    text = _open(_ops(fake), "4")

    assert text == "Opened spotify on workspace 4."
    moves = [lua for lua in fake.ran if lua.startswith("hl.dispatch(")]
    window = '"address:0xabc"'
    assert moves == [
        f"hl.dispatch(hl.dsp.window.move({{ workspace = 4, follow = false, window = {window} }}))"
    ]


def test_a_window_that_will_not_move_is_reported_not_claimed() -> None:
    fake = FakeHyprland(opens_on="1", moves_work=False)

    result = _opened(_ops(fake), "4")

    assert (
        result.text
        == "Opened spotify, but a window of it stayed on workspace 1, not on workspace 4."
    )
    assert not result.ok
    assert result.acted


def test_no_window_in_time_is_said_plainly() -> None:
    clock = Clock()
    fake = FakeHyprland(opens_on=None)

    result = _opened(_ops(fake, clock), "2")

    assert result.text.startswith("Started spotify, but no window of it showed within 15 seconds")
    assert not result.ok
    assert result.acted  # the program was started
    assert clock.now >= 1000.0 + OPEN_WAIT_S


def test_an_open_program_is_not_opened_again() -> None:
    fake = FakeHyprland(open=[Placed("0x1", "5", "Spotify", "Spotify")])

    text = _open(_ops(fake), "3")

    assert text == "spotify is already open on workspace 5; nothing was opened."
    assert fake.ran == []


def test_a_browser_tab_named_after_the_program_is_not_the_program_and_is_never_moved() -> None:
    tab = Placed("0xweb", "2", "firefox", "Spotify - Web Player")
    fake = FakeHyprland(opens_on="1", open=[tab])

    text = _open(_ops(fake), "4")

    assert text == "Opened spotify on workspace 4."
    assert tab in fake.open  # still on workspace 2
    assert all("0xweb" not in lua for lua in fake.ran)


def test_a_request_heard_in_pc_audio_opens_nothing() -> None:
    fake = FakeHyprland()

    with pytest.raises(OperationError, match="never from PC audio"):
        _open(_ops(fake), "3", foreign=True)

    assert fake.ran == []


def test_nothing_opens_outside_a_request() -> None:
    fake = FakeHyprland()
    (op,) = _ops(fake).operations()

    with pytest.raises(OperationError):
        op.run({"app": "spotify", "where": "3"})

    assert fake.ran == []


def test_hyprland_refusing_the_command_is_an_error() -> None:
    fake = FakeHyprland(takes_commands=False)

    with pytest.raises(OperationError, match="was not opened"):
        _open(_ops(fake), "3")


@pytest.mark.parametrize(
    "args",
    [
        {"app": "spotify"},  # where is required: the model must have asked
        {"app": "spotify", "where": "11"},
        {"app": "spotify", "where": "0"},
        {"app": "firefox", "where": "3"},  # only what [apps] lists
        {"app": "spotify", "where": "3", "class": "x"},
    ],
)
def test_arguments_outside_the_schema_are_refused(args: dict[str, str]) -> None:
    (op,) = _ops(FakeHyprland()).operations()

    with pytest.raises(ArgumentError):
        validate_args(op, args)


def test_no_listed_program_means_no_operation() -> None:
    assert AppOps({}, FakeHyprland()).operations() == []


def test_the_default_list_has_spotify_only() -> None:
    assert list(Config().apps.programs) == ["spotify"]


def _lua_bytes(inner: str) -> bytes:
    """Read a Lua string literal's inside as Lua does: `\\ddd` is one byte, up to three digits."""
    out, index = bytearray(), 0
    while index < len(inner):
        if inner[index] == "\\":
            digits = re.match(r"\d{1,3}", inner[index + 1 :])
            assert digits is not None
            value = int(digits.group())
            assert value <= 255, f"Lua refuses the escape \\{digits.group()}"
            out.append(value)
            index += 1 + len(digits.group())
        else:
            out += inner[index].encode()
            index += 1
    return bytes(out)


@pytest.mark.parametrize(
    "text", ["zero\u200bwidth", "line\u2028separator", "tab\there", "plain é and 中文"]
)
def test_a_lua_string_reads_back_as_the_same_text(text: str) -> None:
    quoted = hyprland.lua_string(text)

    assert _lua_bytes(quoted[1:-1]).decode() == text


def test_a_lone_surrogate_is_escaped_rather_than_refused() -> None:
    quoted = hyprland.lua_string("odd \ud800 text")

    assert _lua_bytes(quoted[1:-1]) == "odd \ud800 text".encode("utf-8", "surrogatepass")


@pytest.mark.parametrize("text", ['a" .. os.exit() .. "', "line\nbreak", "back\\slash", "it's"])
def test_a_lua_string_cannot_be_broken_out_of(text: str) -> None:
    quoted = hyprland.lua_string(text)

    inner = quoted[1:-1]
    assert quoted.startswith('"')
    assert quoted.endswith('"')
    assert '"' not in inner
    assert "\n" not in inner
    assert "'" not in inner
    assert inner.count("\\") == sum(1 for ch in text if ch in "\"'\\\n")


@pytest.mark.parametrize("game_on", ["6", SPECIAL])
def test_nothing_opens_on_a_workspace_with_a_fullscreen_window(game_on: str) -> None:
    # A fullscreen program such as a game can crash when a window opens beside it.
    game = Placed("0xg", game_on, "a_game", "A game", fullscreen=True)
    fake = FakeHyprland(open=[game])

    with (
        request_scope(foreign=False),
        pytest.raises(OperationError, match="fullscreen a_game"),
    ):
        _ops(fake).open_app({"app": "spotify", "where": "6" if game_on == "6" else "background"})

    assert fake.ran == []


def test_nothing_opens_while_a_shown_workspace_has_a_fullscreen_window() -> None:
    # A program may ignore where it was sent and open where the owner is looking.
    game = Placed("0xg", "6", "a_game", "A game", fullscreen=True)
    fake = FakeHyprland(open=[game], shown={"6"})

    with request_scope(foreign=False), pytest.raises(OperationError, match="could open there"):
        _ops(fake).open_app({"app": "spotify", "where": "4"})

    assert fake.ran == []


def test_an_unreadable_window_list_opens_nothing() -> None:
    fake = FakeHyprland(readable=False)

    with request_scope(foreign=False), pytest.raises(OperationError, match="window list"):
        _ops(fake).open_app({"app": "spotify", "where": "4"})

    assert fake.ran == []


def test_a_stray_window_on_a_fullscreen_workspace_is_left_where_it_is() -> None:
    # The new window ignored its rule and opened beside a fullscreen game: moving it out would
    # lay the game's workspace out again, so it stays and the owner is told.
    game = Placed("0xg", "6", "a_game", "A game", fullscreen=True)
    fake = FakeHyprland(open=[game], opens_on="6", shown={"1"})

    result = _opened(_ops(fake), "4")

    assert not result.ok
    assert "stayed on workspace 6" in result.text
    assert not any("window.move" in lua for lua in fake.ran)


def test_a_fullscreen_window_matching_the_program_is_never_taken_for_it_or_moved() -> None:
    clock = Clock()
    full = Placed("0xf", "6", "Spotify", "Spotify", fullscreen=True)
    fake = FakeHyprland(opens_on=None)
    ops = _ops(fake, clock)
    (op,) = ops.operations()

    def appears_fullscreen(lua: str) -> bool:
        fake.ran.append(lua)
        fake.open.append(full)  # a fullscreen window of the same class shows during the wait
        return True

    fake.run = appears_fullscreen  # type: ignore[method-assign]
    with request_scope(foreign=False):
        result = op.run(validate_args(op, {"app": "spotify", "where": "4"}))

    assert not result.ok  # not seen: the fullscreen window is not the new one
    assert not any("window.move" in lua for lua in fake.ran)


def test_a_fullscreen_client_with_odd_fields_is_still_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    odd: dict[str, object] = {
        "address": None,
        "workspace": {"name": "6"},
        "fullscreen": 2,
    }  # no class or title
    monkeypatch.setattr(hyprland, "listing", _answers([odd]))

    (found,) = HyprlandCompositor.windows()

    assert found.fullscreen
    assert found.workspace == "6"


def test_a_stray_is_never_moved_onto_a_workspace_that_turned_fullscreen() -> None:
    fake = FakeHyprland(opens_on="5", shown={"1"})
    ops = _ops(fake)
    mine = ops._mine("spotify")  # pyright: ignore[reportPrivateUsage] - the matcher under test
    fake.open = [
        Placed("0xs", "5", "Spotify", "Spotify"),
        Placed("0xg", "4", "a_game", "A game", fullscreen=True),  # came fullscreen meanwhile
    ]

    strays = ops._gather(mine, "4")  # pyright: ignore[reportPrivateUsage] - the mover under test

    assert strays == [Placed("0xs", "5", "Spotify", "Spotify")]
    assert fake.ran == []


def test_an_unreadable_read_back_is_not_claimed_as_opened() -> None:
    fake = FakeHyprland(opens_on="4")
    ops = _ops(fake)
    reads = iter([True, True, True, True, False])
    original = fake.windows

    def flaky() -> list[Placed]:
        if not next(reads, False):
            raise OperationError("Hyprland did not give its window list")
        return original()

    fake.windows = flaky  # type: ignore[method-assign]
    result = _opened(ops, "4")

    assert not result.ok
    assert result.acted
    assert "did not say where" in result.text
