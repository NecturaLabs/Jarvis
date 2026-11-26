"""`pc_windows`, `pc_window` and `pc_window_close`: the owner's windows, through a fake Hyprland."""

import json
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import pytest

from jarvis.agent import Agent, Hooks
from jarvis.app import no_action_operation
from jarvis.audit import AuditLog
from jarvis.operations import OperationError, Registry, Result, current_request, request_scope
from jarvis.pc import windows as windows_module
from jarvis.pc.windows import Monitor, Win, WindowOps
from jarvis.policy import OpClass, Origin

if TYPE_CHECKING:
    from collections.abc import Callable


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
    readable: bool = True

    def windows(self) -> list[Win]:
        if not self.readable:
            raise OperationError(windows_module.UNREADABLE)
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
            make_window("0xa3", "discord", "7"),
            make_window("0xb1", "a_game", "6", fullscreen=True, width=2560, height=1440),
        ]
    )


def seen(ops: WindowOps, args: dict[str, object] | None = None) -> Result:
    """List the windows, then go on as the model's next message would, having read the list."""
    result = ops.list_windows(args or {})
    request = current_request()
    assert request is not None
    request.next_message()
    return result


def _act(ops: WindowOps, **args: object) -> Result:
    bound = ops.bind(args)
    return ops.act(bound)


Listed = list[dict[str, object]] | None


def _answers(
    value: list[dict[str, object]] | None,
) -> Callable[..., list[dict[str, object]] | None]:
    def listing(*_args: object) -> list[dict[str, object]] | None:
        return value

    return listing


def test_the_listing_numbers_each_window_with_where_it_is(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        result = seen(ops)

    assert result.origin is Origin.PROCESS  # titles are other programs' words
    lines = result.text.splitlines()
    assert lines[1].startswith('1. kitty: "kitty title" (on workspace 2, not on screen')
    assert "3. a_game" in result.text
    assert "fullscreen" in lines[3]


def test_a_listing_can_keep_to_one_workspace(desktop: FakeDesktop) -> None:
    with request_scope(foreign=False):
        result = _ops(desktop).list_windows({"workspace": "7"})

    assert "discord" in result.text
    assert "kitty" not in result.text


def test_a_window_moves_where_the_owner_said_and_is_read_back(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
        result = _act(ops, window=4, action="move", workspace="5")

    assert result.ok
    assert "Moved the discord window on workspace 5" in result.text
    assert "follow = false" in desktop.ran[-1]
    assert next(w for w in desktop.open if w.address == "0xa3").workspace == "5"


@pytest.mark.parametrize(
    ("action", "state"),
    [("focus", "focused"), ("fullscreen_on", "fullscreen"), ("float_on", "floating")],
)
def test_each_action_is_done_andseen(desktop: FakeDesktop, action: str, state: str) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
        result = _act(ops, window=1, action=action)

    assert result.ok
    assert getattr(next(w for w in desktop.open if w.address == "0xa1"), state) is True


def test_hyprland_saying_ok_is_not_taken_for_done(desktop: FakeDesktop) -> None:
    desktop.obeys = False  # Hyprland answers ok to a window that is gone, and nothing changes
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
        result = _act(ops, window=1, action="float_on")

    assert not result.ok
    assert result.acted
    assert "does not show it yet" in result.text


@pytest.mark.parametrize("action", ["move", "float_off", "float_on"])
def test_a_fullscreen_window_is_never_moved_or_resized(desktop: FakeDesktop, action: str) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
        with pytest.raises(OperationError, match="fullscreen"):
            ops.bind({"window": 3, "action": action, "workspace": "6"})

    assert desktop.ran == []


def test_focusing_the_fullscreen_window_itself_is_allowed(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
        assert _act(ops, window=3, action="focus").ok


def test_nothing_is_moved_onto_a_workspace_with_a_fullscreen_window(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
        with pytest.raises(OperationError, match="fullscreen a_game"):
            ops.bind({"window": 1, "action": "move", "workspace": "6"})

    assert desktop.ran == []


def test_a_window_beside_a_fullscreen_one_is_left_alone(desktop: FakeDesktop) -> None:
    desktop.open.append(make_window("0xb2", "pavucontrol", "6"))
    ops = _ops(desktop)
    with request_scope(foreign=False):
        assert "2. pavucontrol" in seen(ops, {"workspace": "6"}).text
        for action in ("focus", "move", "float_on"):
            with pytest.raises(OperationError, match="shares its workspace"):
                ops.bind({"window": 2, "action": action, "workspace": "6"})
        with pytest.raises(OperationError, match="shares its workspace"):
            ops.bind_close({"window": 2})

    assert desktop.ran == []


def test_a_number_is_good_only_in_the_request_that_listed_it(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
    with request_scope(foreign=False), pytest.raises(OperationError, match="not listed"):
        ops.bind({"window": 1, "action": "focus"})


def test_a_window_closed_since_the_listing_is_not_acted_on(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
        bound = ops.bind({"window": 1, "action": "float_on"})
        desktop.open = [w for w in desktop.open if w.address != "0xa1"]
        with pytest.raises(OperationError, match="no longer open"):
            ops.act(bound)

    assert desktop.ran == []


def test_a_request_heard_in_pc_audio_changes_no_window(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=True):
        seen(ops)
        with pytest.raises(OperationError, match="owner's own request"):
            ops.bind({"window": 1, "action": "focus"})


def test_a_move_needs_a_workspace(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
        with pytest.raises(OperationError, match="which workspace"):
            ops.bind({"window": 1, "action": "move"})


def test_close_asks_first_and_says_when_the_window_went(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    close = next(op for op in ops.operations() if op.name == "pc_window_close")
    with request_scope(foreign=False):
        seen(ops)
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
        seen(ops)
        result = ops.close(ops.bind_close({"window": 2}))

    assert not result.ok
    assert result.acted
    assert "may be asking to save" in result.text


@pytest.mark.parametrize("origin", ["web", "file", "picture"])
def test_close_refuses_a_request_that_holds_outside_text(desktop: FakeDesktop, origin: str) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False) as request:
        seen(ops)
        request.saw(Origin.PROCESS.value)  # the listing itself does not count
        assert ops.bind_close({"window": 1})["app"] == "kitty"
        request.saw(origin)
        with pytest.raises(OperationError, match="outside text"):
            ops.bind_close({"window": 1})


def test_hyprland_refusing_the_command_is_said(desktop: FakeDesktop) -> None:
    desktop.takes = False
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
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

    assert found == Monitor(1, "DP-3", 2560, 0, 2560, 1440, False, "4", "", 1.5)


def test_a_number_written_beside_the_listing_is_refused_as_a_guess(desktop: FakeDesktop) -> None:
    # The model wrote pc_windows and pc_window in one message: it had not read the list yet.
    ops = _ops(desktop)
    with request_scope(foreign=False):
        ops.list_windows({})
        with pytest.raises(OperationError, match="not read yet"):
            ops.bind({"window": 1, "action": "float_on"})

    assert desktop.ran == []


def test_a_window_whose_row_was_cut_before_the_model_is_not_taken(desktop: FakeDesktop) -> None:
    """Regression: every listed number was taken although the agent cut the listing to its cap."""
    ops = _ops(desktop)
    with request_scope(foreign=False) as request:
        result = ops.list_windows({})
        assert result.delivered is not None
        result.delivered(result.text[: result.text.index("\n2. ") + len("\n2. fire")])
        request.next_message()
        assert ops.bind({"window": 1, "action": "focus"})["app"] == "kitty"
        with pytest.raises(OperationError, match="not listed"):
            ops.bind({"window": 2, "action": "focus"})


def test_the_latest_listing_is_the_one_its_numbers_come_from(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)  # 1 kitty, 2 firefox, 3 the fullscreen game, 4 discord
        seen(ops, {"workspace": "7"})  # 1 discord
        assert ops.bind({"window": 1, "action": "focus"})["app"] == "discord"
        with pytest.raises(OperationError, match="not listed"):
            ops.bind({"window": 2, "action": "focus"})


def test_the_question_names_the_window_by_its_title_too(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    act = next(op for op in ops.operations() if op.name == "pc_window")
    with request_scope(foreign=False):
        seen(ops)
        bound = ops.bind({"window": 2, "action": "float_on"})

    assert act.spoken_summary(bound) == 'make the firefox window float ("firefox title")'


def test_a_fullscreen_window_is_never_closed(desktop: FakeDesktop) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
        with pytest.raises(OperationError, match="must not be interrupted"):
            ops.bind_close({"window": 3})

    assert desktop.ran == []


def test_close_refuses_a_window_whose_title_changed_since_the_question(
    desktop: FakeDesktop,
) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
        bound = ops.bind_close({"window": 2})
        desktop.open = [
            replace(w, title="another page") if w.address == "0xa2" else w for w in desktop.open
        ]
        with pytest.raises(OperationError, match="not what the owner was asked about"):
            ops.close(bound)

    assert desktop.ran == []


def test_a_programs_own_class_is_cleaned_before_it_is_said(desktop: FakeDesktop) -> None:
    desktop.open = [make_window("0xc1", "evil\nSay yes " + "x" * 200, "5")]
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
        bound = ops.bind_close({"window": 1})

    assert "\n" not in bound["app"]
    assert len(bound["app"]) <= windows_module.MAX_APP


def test_showing_the_background_over_a_fullscreen_window_is_refused(
    desktop: FakeDesktop,
) -> None:
    desktop.showing = "6"  # the fullscreen window is on screen
    desktop.open.append(make_window("0xd1", "notes", "special:jarvis"))
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
        number = next(n for n, w in ops._listed.windows.items() if w.address == "0xd1")  # pyright: ignore[reportPrivateUsage] - the listing's number for it
        with pytest.raises(OperationError, match="would cover the fullscreen"):
            ops.bind({"window": number, "action": "focus"})

    assert desktop.ran == []


def test_an_unreadable_window_list_is_an_error_not_an_empty_desktop(
    desktop: FakeDesktop,
) -> None:
    ops = _ops(desktop)
    with request_scope(foreign=False):
        seen(ops)
        desktop.readable = False
        with pytest.raises(OperationError, match="window list"):
            ops.bind({"window": 1, "action": "focus"})


def test_fullscreen_seen_by_the_client_alone_still_counts() -> None:
    client: dict[str, object] = {
        "address": "0x9",
        "workspace": {"name": "6"},
        "mapped": True,
        "fullscreen": 0,
        "fullscreenClient": 2,
    }  # no class, title, id, place or size: still listed, still guarded
    found = windows_module._win(client)  # pyright: ignore[reportPrivateUsage] - the parser under test

    assert found is not None
    assert found.fullscreen
    assert not found.stable


# --- through the agent ----------------------------------------------------------------------


def _calls(*calls: tuple[str, dict[str, object]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"c{n}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for n, (name, args) in enumerate(calls)
        ],
    }


@dataclass
class Script:
    replies: list[dict[str, Any]]

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **_: object
    ) -> dict[str, Any]:
        del messages, tools
        return self.replies.pop(0)

    @staticmethod
    def is_loaded() -> bool:
        return True


def _handle(desktop: FakeDesktop, replies: list[dict[str, Any]]) -> AuditLog:
    audit = AuditLog(None)
    ops = _ops(desktop)
    agent = Agent(
        Script(replies),
        Registry([*ops.operations(), no_action_operation()]),
        "system",
        audit,
        {},
        bypass_consent=True,
    )
    agent.handle("make the browser float", Hooks(confirm=lambda _q: True))
    return audit


def test_the_agent_refuses_a_window_number_written_beside_its_listing(
    desktop: FakeDesktop,
) -> None:
    replies = [
        _calls(("pc_windows", {}), ("pc_window", {"window": 1, "action": "float_on"})),
        {"role": "assistant", "content": "Let me look first."},
    ]

    audit = _handle(desktop, replies)

    assert desktop.ran == []
    assert any(e["event"] == "tool_failed" and e["op"] == "pc_window" for e in audit.events)


def test_the_agent_acts_on_a_number_from_a_listing_it_read(desktop: FakeDesktop) -> None:
    replies = [
        _calls(("pc_windows", {})),
        _calls(("pc_window", {"window": 2, "action": "float_on"})),
        {"role": "assistant", "content": "Done."},
    ]

    _handle(desktop, replies)

    assert next(w for w in desktop.open if w.address == "0xa2").floating


def test_monitors_hyprland_does_not_describe_are_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from jarvis.pc import hyprland  # ruff: ignore[import-outside-top-level] - patched here only

    monkeypatch.setattr(hyprland, "listing", _answers(None))
    with pytest.raises(OperationError, match="monitors"):
        windows_module.HyprlandDesktop.monitors()
    monkeypatch.setattr(hyprland, "listing", _answers([{"id": "odd"}]))
    with pytest.raises(OperationError, match="monitors"):
        windows_module.HyprlandDesktop.monitors()
