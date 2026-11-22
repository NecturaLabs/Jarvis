"""`open_app`: a program opens only when the owner asks, exactly where the owner said."""

from dataclasses import dataclass, field

import pytest

from jarvis.config import AppConfig, Config
from jarvis.operations import ArgumentError, OperationError, request_scope, validate_args
from jarvis.pc import hyprland
from jarvis.pc.apps import OPEN_WAIT_S, SPECIAL, AppOps, Placed

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

    def windows(self) -> list[Placed]:
        return list(self.open)

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


def _open(ops: AppOps, where: str, *, foreign: bool = False) -> str:
    (op,) = ops.operations()
    with request_scope(foreign=foreign):
        return op.run(validate_args(op, {"app": "spotify", "where": where})).text


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

    result_text = _open(_ops(fake), "4")

    assert (
        result_text
        == "Opened spotify, but a window of it stayed on workspace 1, not on workspace 4."
    )


def test_no_window_in_time_is_said_plainly() -> None:
    clock = Clock()
    fake = FakeHyprland(opens_on=None)

    text = _open(_ops(fake, clock), "2")

    assert text.startswith("Started spotify, but no window of it showed within 15 seconds")
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
