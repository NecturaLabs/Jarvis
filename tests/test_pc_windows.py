"""`pc_windows`, `pc_window` and `pc_window_close`: the owner's windows, through a fake Hyprland."""

import re
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from jarvis.operations import OperationError, Result, request_scope
from jarvis.pc import windows as windows_module
from jarvis.pc.windows import Monitor, Win, WindowOps
from jarvis.policy import OpClass, Origin


def make_window(address: str, klass: str, workspace: str, **kw: Any) -> Win:  # ruff: ignore[any-type] - the fields a test sets
    stable = address.removeprefix("0x")
    base = Win(address, stable, klass, f"{klass} title", workspace, 0, 0, 0, 800, 600)
    return replace(base, **kw)


@dataclass
class FakeDesktop:
    """Windows and monitors as Hyprland lists them; each Lua call it takes changes them."""

    open: list[Win] = field(default_factory=list[Win])
    showing: str = "1"
    takes: bool = True
    obeys: bool = True
    ran: list[str] = field(default_factory=list[str])

    def windows(self) -> list[Win]:
        return list(self.open)

    def monitors(self) -> list[Monitor]:
        return [Monitor(0, "DP-1", 0, 0, 2560, 1440, True, self.showing, "")]

    def run(self, lua: str) -> bool:
        self.ran.append(lua)
        if not self.takes:
            return False
        if self.obeys:
            address = re.search(r'"address:(0x[0-9a-f]+)"', lua)
            assert address is not None
            self.open = [self._apply(w, lua) for w in self.open if not self._closes(w, lua)]
        return True

    @staticmethod
    def _closes(w: Win, lua: str) -> bool:
        return "window.close" in lua and f"address:{w.address}" in lua

    @staticmethod
    def _apply(w: Win, lua: str) -> Win:
        if f"address:{w.address}" not in lua:
            return replace(w, focused=False) if "dsp.focus" in lua else w
        if "dsp.focus" in lua:
            return replace(w, focused=True)
        if "window.move" in lua:
            target = lua.split("workspace = ", 1)[1].split(",", 1)[0].strip('"')
            return replace(w, workspace=target)
        if "window.fullscreen" in lua:
            return replace(w, fullscreen='action = "set"' in lua)
        if "window.float" in lua:
            return replace(w, floating='action = "on"' in lua)
        return w


def _ops(desktop: FakeDesktop) -> WindowOps:
    clock = iter(range(1000))
    return WindowOps(desktop, clock=lambda: float(next(clock)), sleep=lambda _s: None)


@pytest.fixture
def desktop() -> FakeDesktop:
    return FakeDesktop(
        open=[
            make_window("0xa1", "kitty", "2"),
            make_window("0xa2", "firefox", "2", x=800),
            make_window("0xa3", "discord", "4"),
            make_window("0xb1", "steam_app_1", "3", fullscreen=True, width=2560, height=1440),
        ]
    )


def _act(ops: WindowOps, **args: object) -> Result:
    bound = ops.bind(args)
    return ops.act(bound)


def test_the_listing_numbers_each_window_with_where_it_is(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        result = ops.list_windows({})

    assert result.origin is Origin.PROCESS  # titles are other programs' words
    lines = result.text.splitlines()
    assert lines[1].startswith('1. kitty: "kitty title" (on workspace 2, not on screen')
    assert "3. steam_app_1" in result.text
    assert "fullscreen" in lines[3]


def test_a_listing_can_keep_to_one_workspace(desktop: FakeDesktop) -> None:
    with request_scope(foreign=False):
        result = _ops(desktop).list_windows({"workspace": "4"})

    assert "discord" in result.text
    assert "kitty" not in result.text


def test_a_window_moves_where_the_owner_said_and_is_read_back(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        ops.list_windows({})
        result = _act(ops, window=4, action="move", workspace="5")

    assert result.ok
    assert "Moved the discord window on workspace 5" in result.text
    assert "follow = false" in desktop.ran[-1]
    assert next(w for w in desktop.open if w.address == "0xa3").workspace == "5"


@pytest.mark.parametrize(
    ("action", "state"),
    [("focus", "focused"), ("fullscreen_on", "fullscreen"), ("float_on", "floating")],
)
def test_each_action_is_done_and_seen(desktop: FakeDesktop, action: str, state: str) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        ops.list_windows({})
        result = _act(ops, window=1, action=action)

    assert result.ok
    assert getattr(next(w for w in desktop.open if w.address == "0xa1"), state) is True


def test_hyprland_saying_ok_is_not_taken_for_done(desktop: FakeDesktop) -> None:
    desktop.obeys = False  # Hyprland answers ok to a window that is gone, and nothing changes
    ops = _ops(desktop)
    with request_scope(foreign=False):
        ops.list_windows({})
        result = _act(ops, window=1, action="float_on")

    assert not result.ok
    assert result.acted
    assert "does not show it yet" in result.text


@pytest.mark.parametrize("action", ["move", "fullscreen_off", "float_on"])
def test_a_fullscreen_window_is_never_moved_or_resized(desktop: FakeDesktop, action: str) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        ops.list_windows({})
        with pytest.raises(OperationError, match="fullscreen"):
            ops.bind({"window": 3, "action": action, "workspace": "6"})

    assert desktop.ran == []


def test_focusing_the_fullscreen_window_itself_is_allowed(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        ops.list_windows({})
        assert _act(ops, window=3, action="focus").ok


def test_nothing_is_moved_onto_a_workspace_with_a_fullscreen_window(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        ops.list_windows({})
        with pytest.raises(OperationError, match="fullscreen steam_app_1"):
            ops.bind({"window": 1, "action": "move", "workspace": "3"})

    assert desktop.ran == []


def test_a_window_beside_a_fullscreen_one_is_left_alone(desktop: FakeDesktop) -> None:
    desktop.open.append(make_window("0xb2", "pavucontrol", "3"))
    ops = _ops(desktop)
    with request_scope(foreign=False):
        assert "2. pavucontrol" in ops.list_windows({"workspace": "3"}).text
        for action in ("focus", "move", "float_on"):
            with pytest.raises(OperationError, match="shares its workspace"):
                ops.bind({"window": 2, "action": action, "workspace": "6"})
        with pytest.raises(OperationError, match="shares its workspace"):
            ops.bind_close({"window": 2})

    assert desktop.ran == []


def test_a_number_is_good_only_in_the_request_that_listed_it(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        ops.list_windows({})
    with request_scope(foreign=False), pytest.raises(OperationError, match="not listed"):
        ops.bind({"window": 1, "action": "focus"})


def test_a_window_closed_since_the_listing_is_not_acted_on(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        ops.list_windows({})
        bound = ops.bind({"window": 1, "action": "float_on"})
        desktop.open = [w for w in desktop.open if w.address != "0xa1"]
        with pytest.raises(OperationError, match="no longer open"):
            ops.act(bound)

    assert desktop.ran == []


def test_a_request_heard_in_pc_audio_changes_no_window(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=True):
        ops.list_windows({})
        with pytest.raises(OperationError, match="owner's own request"):
            ops.bind({"window": 1, "action": "focus"})


def test_a_move_needs_a_workspace(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        ops.list_windows({})
        with pytest.raises(OperationError, match="which workspace"):
            ops.bind({"window": 1, "action": "move"})


def test_close_asks_first_and_says_when_the_window_went(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    close = next(op for op in ops.operations() if op.name == "pc_window_close")
    with request_scope(foreign=False):
        ops.list_windows({})
        bound = ops.bind_close({"window": 2})
        result = ops.close(bound)

    assert close.default_class is OpClass.CONFIRM
    assert close.spoken_summary(bound) == 'close the firefox window "firefox title"'
    assert result.ok
    assert all(w.address != "0xa2" for w in desktop.open)


def test_close_says_when_the_window_stays_open(desktop: FakeDesktop) -> None:
    desktop.obeys = False  # the program asks to save, say
    ops = _ops(desktop)
    with request_scope(foreign=False):
        ops.list_windows({})
        result = ops.close(ops.bind_close({"window": 2}))

    assert not result.ok
    assert result.acted
    assert "may be asking to save" in result.text


@pytest.mark.parametrize("origin", ["web", "file", "picture"])
def test_close_refuses_a_request_that_holds_outside_text(desktop: FakeDesktop, origin: str) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False) as request:
        ops.list_windows({})
        request.saw(Origin.PROCESS.value)  # the listing itself does not count
        assert ops.bind_close({"window": 1})["app"] == "kitty"
        request.saw(origin)
        with pytest.raises(OperationError, match="outside text"):
            ops.bind_close({"window": 1})


def test_hyprland_refusing_the_command_is_said(desktop: FakeDesktop) -> None:
    desktop.takes = False
    ops = _ops(desktop)
    with request_scope(foreign=False):
        ops.list_windows({})
        with pytest.raises(OperationError, match="did not take"):
            _act(ops, window=1, action="focus")


def test_hyprlands_listing_is_read_with_its_fullscreen_and_focus() -> None:
    client: dict[str, object] = {
        "address": "0x55d1",
        "stableId": "18001da9",
        "class": "firefox",
        "title": "A page",
        "workspace": {"id": 2, "name": "2"},
        "monitor": 0,
        "at": [9, 743],
        "size": [1266, 688],
        "mapped": True,
        "fullscreen": 2,
        "floating": False,
        "focusHistoryID": 0,
    }
    found = windows_module._win(client)  # pyright: ignore[reportPrivateUsage] - the parser under test

    assert found == Win(
        "0x55d1", "18001da9", "firefox", "A page", "2", 0, 9, 743, 1266, 688,
        focused=True, fullscreen=True, floating=False,
    )  # fmt: skip
    assert windows_module._win({**client, "address": "0x1; os.exit()"}) is None  # pyright: ignore[reportPrivateUsage] - the parser under test
    assert windows_module._win({**client, "mapped": False}) is None  # pyright: ignore[reportPrivateUsage] - the parser under test


def test_a_monitor_is_read_in_layout_units() -> None:
    found = windows_module._monitor(  # pyright: ignore[reportPrivateUsage] - the parser under test
        {
            "id": 1,
            "name": "DP-3",
            "x": 2560,
            "y": 0,
            "width": 3840,
            "height": 2160,
            "scale": 1.5,
            "transform": 0,
            "focused": False,
            "activeWorkspace": {"name": "4"},
            "specialWorkspace": {"name": ""},
        }
    )

    assert found == Monitor(1, "DP-3", 2560, 0, 2560, 1440, False, "4", "")
