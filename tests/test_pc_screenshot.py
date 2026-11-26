"""`pc_screenshot`: grim and the clipboard faked, the picture saved where screenshots go."""

import io
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image
from test_pc_windows import FakeDesktop, make_window, seen

from jarvis.operations import OperationError, request_scope
from jarvis.pc.screenshot import CaptureError, ScreenshotOps, save_folder
from jarvis.pc.windows import Monitor, WindowOps
from jarvis.pictures import PictureCache
from jarvis.policy import Origin

COLOURS = {"a1": (255, 0, 0), "a2": (0, 255, 0), "b1": (0, 0, 255)}
NOW = datetime(2026, 11, 26, 15, 30, 5, tzinfo=UTC)


def _png(size: tuple[int, int], colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, "PNG")
    return buffer.getvalue()


class Grim:
    """Returns a plain picture for each capture and keeps the arguments it was given."""

    def __init__(self, desktop: FakeDesktop, *, fails: bool = False) -> None:
        super().__init__()
        self.desktop = desktop
        self.fails = fails
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> bytes:
        self.calls.append(args)
        if self.fails:
            msg = "grim failed: compositor doesn't support the screencopy protocol"
            raise CaptureError(msg)
        if args[:1] == ["-T"]:
            window = next(w for w in self.desktop.open if w.stable == args[1])
            return _png((window.width, window.height), COLOURS.get(args[1], (9, 9, 9)))
        return _png((2560, 1440), (200, 200, 200))


@pytest.fixture
def desktop() -> FakeDesktop:
    return FakeDesktop(
        open=[
            make_window("0xa1", "kitty", "2", width=1280, height=1440),
            make_window("0xa2", "firefox", "2", x=1280, width=1280, height=1440, focused=True),
            make_window("0xb1", "a_game", "6", fullscreen=True, width=2560, height=1440),
        ],
        showing="1",
    )


def _ops(
    desktop: FakeDesktop,
    folder: Path,
    grim: Grim,
    *,
    copies: bool = True,
    shown: PictureCache | None = None,
) -> tuple[ScreenshotOps, WindowOps, list[bytes]]:
    copied: list[bytes] = []

    def copy(data: bytes) -> bool:
        copied.append(data)
        return copies

    windows = WindowOps(desktop)
    shots = ScreenshotOps(
        windows, folder, shown_pictures=shown, capture=grim, copy=copy, now=lambda: NOW
    )
    return shots, windows, copied


def test_the_screen_is_saved_with_omarchys_name_and_copied(
    desktop: FakeDesktop, tmp_path: Path
) -> None:
    grim = Grim(desktop)
    shots, _, copied = _ops(desktop, tmp_path / "Screenshots", grim)

    with request_scope(foreign=False):
        result = shots.screenshot({"of": "screen"})

    saved = tmp_path / "Screenshots" / "screenshot-2026-11-26_15-30-05.png"
    assert grim.calls == [[]]
    assert saved.read_bytes() == copied[0]
    assert saved.stat().st_mode & 0o777 == 0o600
    assert result.origin is Origin.TOOL_STATUS
    assert f"to {saved} and copied it to the clipboard" in result.text
    assert result.text.endswith("You can't see it.")
    assert (
        result.spoken
        == "Took a screenshot of the screen; it's in Screenshots and on the clipboard."
    )
    assert desktop.ran == []  # nothing was focused, moved or resized


def test_a_workspace_not_on_screen_is_made_from_its_windows_without_showing_it(
    desktop: FakeDesktop, tmp_path: Path
) -> None:
    grim = Grim(desktop)
    shots, _, _ = _ops(desktop, tmp_path, grim)

    with request_scope(foreign=False):
        result = shots.screenshot({"of": "workspace", "workspace": "2"})

    assert grim.calls == [["-T", "a1"], ["-T", "a2"]]
    with Image.open(next(tmp_path.glob("screenshot-*.png"))) as saved:
        picture = saved.convert("RGB")
    assert picture.size == (2560, 1440)
    assert picture.getpixel((10, 10)) == COLOURS["a1"]
    assert picture.getpixel((2000, 10)) == COLOURS["a2"]
    assert "the windows on workspace 2" in result.text
    assert desktop.ran == []


def test_a_workspace_with_a_fullscreen_game_is_taken_without_touching_it(
    desktop: FakeDesktop, tmp_path: Path
) -> None:
    grim = Grim(desktop)
    shots, _, _ = _ops(desktop, tmp_path, grim)

    with request_scope(foreign=False):
        assert shots.screenshot({"of": "workspace", "workspace": "6"}).ok

    assert grim.calls == [["-T", "b1"]]
    assert desktop.ran == []


def test_a_workspace_on_screen_is_taken_as_its_monitor_shows_it(
    desktop: FakeDesktop, tmp_path: Path
) -> None:
    desktop.showing = "2"
    grim = Grim(desktop)
    shots, _, _ = _ops(desktop, tmp_path, grim)

    with request_scope(foreign=False):
        shots.screenshot({"of": "workspace", "workspace": "2"})

    assert grim.calls == [["-o", "DP-1"]]


def test_a_listed_window_is_taken_by_its_id_and_named_as_other_programs_words(
    desktop: FakeDesktop, tmp_path: Path
) -> None:
    grim = Grim(desktop)
    shots, windows, _ = _ops(desktop, tmp_path, grim)

    with request_scope(foreign=False):
        seen(windows)
        result = shots.screenshot({"of": "window", "window": 1})

    assert grim.calls == [["-T", "a1"]]
    assert result.origin is Origin.PROCESS
    assert not result.spoken
    assert "the kitty window, on workspace 2" in result.text


def test_a_window_number_must_come_from_this_requests_listing(
    desktop: FakeDesktop, tmp_path: Path
) -> None:
    shots, windows, _ = _ops(desktop, tmp_path, Grim(desktop))

    with request_scope(foreign=False):
        seen(windows)
    with request_scope(foreign=False), pytest.raises(OperationError, match="not listed"):
        shots.screenshot({"of": "window", "window": 1})


def test_the_focused_window_is_taken(desktop: FakeDesktop, tmp_path: Path) -> None:
    grim = Grim(desktop)
    shots, _, _ = _ops(desktop, tmp_path, grim)

    with request_scope(foreign=False):
        shots.screenshot({"of": "active_window"})

    assert grim.calls == [["-T", "a2"]]


@pytest.mark.parametrize(
    ("args", "said"),
    [
        ({"of": "workspace"}, "which workspace"),
        ({"of": "window"}, "which window"),
        ({"of": "workspace", "workspace": "7"}, "no window is open on workspace 7"),
    ],
)
def test_what_to_take_must_be_said(
    desktop: FakeDesktop, tmp_path: Path, args: dict[str, object], said: str
) -> None:
    shots, _, _ = _ops(desktop, tmp_path, Grim(desktop))

    with request_scope(foreign=False), pytest.raises(OperationError, match=said):
        shots.screenshot(args)
    assert not list(tmp_path.glob("*.png"))


def test_no_focused_window_is_said(desktop: FakeDesktop, tmp_path: Path) -> None:
    desktop.open = [w for w in desktop.open if not w.focused]
    shots, _, _ = _ops(desktop, tmp_path, Grim(desktop))

    with request_scope(foreign=False), pytest.raises(OperationError, match="No window is focused"):
        shots.screenshot({"of": "active_window"})


def test_a_failed_capture_saves_nothing(desktop: FakeDesktop, tmp_path: Path) -> None:
    shots, _, copied = _ops(desktop, tmp_path, Grim(desktop, fails=True))

    with request_scope(foreign=False), pytest.raises(OperationError, match="screencopy"):
        shots.screenshot({"of": "screen"})
    assert not list(tmp_path.glob("*.png"))
    assert copied == []


def test_a_clipboard_that_refuses_is_said(desktop: FakeDesktop, tmp_path: Path) -> None:
    shots, _, _ = _ops(desktop, tmp_path, Grim(desktop), copies=False)

    with request_scope(foreign=False):
        result = shots.screenshot({"of": "screen"})

    assert result.ok
    assert "the clipboard did not take it" in result.text
    assert result.spoken.endswith("it's in " + tmp_path.name + ".")


def test_two_in_one_second_keep_both(desktop: FakeDesktop, tmp_path: Path) -> None:
    shots, _, _ = _ops(desktop, tmp_path, Grim(desktop))

    with request_scope(foreign=False):
        shots.screenshot({"of": "screen"})
        shots.screenshot({"of": "screen"})

    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "screenshot-2026-11-26_15-30-05-2.png",
        "screenshot-2026-11-26_15-30-05.png",
    ]


def test_the_chat_shows_the_picture(desktop: FakeDesktop, tmp_path: Path) -> None:
    cache = PictureCache(tmp_path / "shown")
    shots, _, _ = _ops(desktop, tmp_path / "shots", Grim(desktop), shown=cache)

    with request_scope(foreign=False):
        result = shots.screenshot({"of": "screen"})

    assert cache.holds(Path(result.picture))
    assert "It shows in the chat." in result.text
    assert result.caption.startswith("the screen · screenshot-")


def test_screenshots_go_where_omarchy_saves_them(tmp_path: Path) -> None:
    assert save_folder(tmp_path, {}) == tmp_path / "Pictures" / "Screenshots"
    assert save_folder(tmp_path, {"OMARCHY_SCREENSHOT_DIR": "/shots"}) == Path("/shots")
    assert save_folder(tmp_path, {"OMASNAP_SCREENSHOT_DIR": "~/s"}) == Path("~/s").expanduser()


def test_a_request_heard_in_pc_audio_takes_no_screenshot(
    desktop: FakeDesktop, tmp_path: Path
) -> None:
    grim = Grim(desktop)
    shots, _, copied = _ops(desktop, tmp_path, grim)

    with request_scope(foreign=True), pytest.raises(OperationError, match="owner's own request"):
        shots.screenshot({"of": "screen"})

    assert grim.calls == []
    assert copied == []


def test_a_workspace_under_a_shown_special_workspace_is_made_from_its_windows(
    desktop: FakeDesktop, tmp_path: Path
) -> None:
    desktop.showing = "2"
    special = Monitor(0, "DP-1", 0, 0, 2560, 1440, True, "2", "special:jarvis")
    desktop.monitors = lambda: [special]  # type: ignore[method-assign]
    grim = Grim(desktop)
    shots, _, _ = _ops(desktop, tmp_path, grim)

    with request_scope(foreign=False):
        shots.screenshot({"of": "workspace", "workspace": "2"})

    assert grim.calls == [["-T", "a1"], ["-T", "a2"]]  # not the monitor, which shows the overlay


def test_a_scaled_monitors_workspace_keeps_its_pixels(desktop: FakeDesktop, tmp_path: Path) -> None:
    scaled = Monitor(0, "DP-1", 0, 0, 1280, 720, True, "1", "", 2.0)
    desktop.monitors = lambda: [scaled]  # type: ignore[method-assign]
    desktop.open = [make_window("0xa1", "kitty", "2", width=1280, height=720)]
    shots, _, _ = _ops(desktop, tmp_path, Grim(desktop))

    with request_scope(foreign=False):
        shots.screenshot({"of": "workspace", "workspace": "2"})

    with Image.open(next(tmp_path.glob("screenshot-*.png"))) as picture:
        assert picture.size == (2560, 1440)


def test_the_owners_own_folder_is_kept_and_a_relative_one_ignored(tmp_path: Path) -> None:
    pictures = str(tmp_path / "Pictures")
    assert save_folder(tmp_path, {"OMARCHY_SCREENSHOT_DIR": pictures}) == Path(pictures)
    assert save_folder(tmp_path, {"OMARCHY_SCREENSHOT_DIR": "shots"}) == (
        tmp_path / "Pictures" / "Screenshots"
    )
