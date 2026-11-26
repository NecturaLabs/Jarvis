"""`pc_screenshot`: a picture of the screen, a workspace or one window, saved and copied.

`grim` takes the picture: the whole screen as it shows now, or one window by its Hyprland id
(`grim -T`), which works for a window on a workspace that is not on screen, so the owner's
view never changes. A workspace on screen is taken as its monitor shows it; one that is not
(or one a special workspace covers) is made from its windows, each captured on its own at full
resolution and laid where it sits (no wallpaper or bar). Capturing only reads frames: no window
is moved, resized or focused. Only the owner's own request takes one, never one heard in PC
audio.

The picture is saved as a PNG where Omarchy saves screenshots, named as Omarchy names them, and
copied to the clipboard (`wl-copy`); the chat also shows it. The model is told what was taken;
where it can see pictures the agent shows it the picture in the next message, as outside data
(`Result.look`), else the result says it cannot see it.
"""

import contextlib
import io
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, UnidentifiedImageError

from jarvis.operations import Operation, OperationError, Result, current_request
from jarvis.pc.apps import BACKGROUND, SPECIAL, WORKSPACES
from jarvis.pc.windows import MAX_LISTED, STABLE_ID, Monitor, Win, WindowOps, shown, where
from jarvis.pictures import SHOWN_SIDE, PictureError, prepare
from jarvis.policy import Origin

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jarvis.pictures import Picture, PictureCache

GRIM_TIMEOUT_S = 10.0
COPY_TIMEOUT_S = 5.0
MAX_PICTURE_BYTES = 64 * 1024 * 1024  # four 4K monitors as PNG fit well within this
MAX_PIXELS = 60_000_000
KINDS = ("screen", "workspace", "window", "active_window")
_BACKDROP = (20, 20, 24)


class CaptureError(Exception):
    """The picture could not be taken; the message says why in plain words."""


def grim(args: list[str]) -> bytes:
    """Return the PNG `grim` writes with `args` (fixed options only).

    Raises:
        CaptureError: grim is missing, failed or did not finish in time.
    """
    import subprocess  # ruff: ignore[import-outside-top-level] - only a capture needs it; building the app must not load it

    try:
        done = subprocess.run(
            ["grim", "-t", "png", *args, "-"],
            capture_output=True,
            timeout=GRIM_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        msg = "grim is not installed"
        raise CaptureError(msg) from exc
    except subprocess.TimeoutExpired as exc:
        msg = f"grim did not finish in {GRIM_TIMEOUT_S:g} seconds"
        raise CaptureError(msg) from exc
    if done.returncode != 0 or not done.stdout:
        said = done.stderr.decode("utf-8", "replace").strip().splitlines()
        msg = f"grim failed: {said[-1] if said else f'exit {done.returncode}'}"
        raise CaptureError(msg)
    if len(done.stdout) > MAX_PICTURE_BYTES:
        msg = "the picture is too large to keep"
        raise CaptureError(msg)
    return done.stdout


def wl_copy(data: bytes) -> bool:
    """Put the PNG `data` on the clipboard with `wl-copy`; True when it took it."""
    import subprocess  # ruff: ignore[import-outside-top-level] - only a copy needs it; building the app must not load it

    try:
        done = subprocess.run(
            ["wl-copy", "--type", "image/png"],
            input=data,
            stdout=subprocess.DEVNULL,  # wl-copy stays behind to serve the clipboard: no pipes
            stderr=subprocess.DEVNULL,
            timeout=COPY_TIMEOUT_S,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return False
    return done.returncode == 0


def folders(home: Path, env: Mapping[str, str]) -> tuple[Path, ...]:
    """Return where Omarchy saves screenshots (its setting first), then the Pictures folder."""
    chosen = [env.get(name, "") for name in ("OMASNAP_SCREENSHOT_DIR", "OMARCHY_SCREENSHOT_DIR")]
    found = [Path(value).expanduser() for value in chosen if value]
    found = [path for path in found if path.is_absolute()]  # a relative one would follow the cwd
    found += [home / "Pictures" / "Screenshots", home / "Pictures"]
    return tuple(dict.fromkeys(found))


def save_folder(home: Path, env: Mapping[str, str]) -> Path:
    """Return the folder a new screenshot goes in: Omarchy's setting, else Pictures/Screenshots."""
    return folders(home, env)[0]


def _decoded(data: bytes) -> Image.Image:
    """Return grim's PNG, decoded after its size was checked.

    Raises:
        CaptureError: not a PNG, too many pixels, or broken.
    """
    unreadable = "grim's picture could not be read"
    try:
        image = Image.open(io.BytesIO(data), formats=["PNG"])
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise CaptureError(unreadable) from exc
    width, height = image.size
    if width * height > MAX_PIXELS:
        msg = "the picture is too large to keep"
        raise CaptureError(msg)
    try:
        image.load()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise CaptureError(unreadable) from exc
    return image


class ScreenshotOps:
    """Builds `pc_screenshot` over the window tools, grim and the clipboard."""

    def __init__(  # ruff: ignore[too-many-arguments] - the readers and writers, each replaceable in tests
        self,
        windows: WindowOps,
        folder: Path,
        *,
        shown_pictures: PictureCache | None = None,
        capture: Callable[[list[str]], bytes] = grim,
        copy: Callable[[bytes], bool] = wl_copy,
        now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ) -> None:
        """Save into `folder`; show in the chat through `shown_pictures` when there is one."""
        super().__init__()
        self._windows = windows
        self._folder = folder
        self._shown = shown_pictures
        self._capture = capture
        self._copy = copy
        self._now = now

    def _window(self, window: Win) -> bytes:
        if not STABLE_ID.fullmatch(window.stable):
            msg = "Hyprland gave that window no id to capture it by"
            raise CaptureError(msg)
        return self._capture(["-T", window.stable])

    def _workspace(self, workspace: str, monitors: list[Monitor], windows: list[Win]) -> bytes:
        """Return the workspace as its monitor shows it, or made from its windows.

        Raises:
            CaptureError: no window is open there, its monitor is unknown, or grim failed.
        """
        on = [w for w in windows if w.workspace == workspace]
        # As shown: its own special workspace, or a regular one no special workspace covers.
        showing = next(
            (
                m
                for m in monitors
                if workspace == m.special or (workspace == m.workspace and not m.special)
            ),
            None,
        )
        if showing is not None:
            return self._capture(["-o", showing.name])
        if not on:
            msg = f"no window is open {where(workspace)}"
            raise CaptureError(msg)
        screen = next((m for m in monitors if m.id == on[0].monitor), None)
        if screen is None:
            msg = "Hyprland did not say which monitor that workspace is on"
            raise CaptureError(msg)
        scale = screen.scale  # windows are captured in pixels, laid out in layout units
        canvas = Image.new(
            "RGB", (round(screen.width * scale), round(screen.height * scale)), _BACKDROP
        )
        # Tiled windows first, floating ones over them, a fullscreen one over everything.
        for window in sorted(on, key=lambda w: (w.fullscreen, w.floating)):
            picture = _decoded(self._window(window)).convert("RGB")
            size = (max(round(window.width * scale), 1), max(round(window.height * scale), 1))
            at = (round((window.x - screen.x) * scale), round((window.y - screen.y) * scale))
            canvas.paste(picture.resize(size), at)
        buffer = io.BytesIO()
        canvas.save(buffer, "PNG")
        return buffer.getvalue()

    def _take(self, args: dict[str, Any]) -> tuple[bytes, str, bool]:
        """Return the picture, what it is of in words, and whether those words name a program.

        Raises:
            OperationError: no workspace or window named, or the window is not one listed.
        """
        kind = str(args["of"])
        if kind == "screen":
            return self._capture([]), "the screen", False
        desktop = self._windows.desktop
        if kind == "workspace":
            wanted = args.get("workspace")
            if wanted is None:
                msg = "Say which workspace to take."
                raise OperationError(msg)
            workspace = SPECIAL if wanted == BACKGROUND else str(wanted)
            data = self._workspace(workspace, desktop.monitors(), desktop.windows())
            return data, f"the windows {where(workspace)}", False
        if kind == "window":
            if args.get("window") is None:
                msg = "Say which window: a number pc_windows listed in this request."
                raise OperationError(msg)
            window, _ = self._windows.now(self._windows.listed(int(args["window"])))
        else:
            window = next((w for w in desktop.windows() if w.focused), None)
            if window is None:
                msg = "No window is focused now."
                raise OperationError(msg)
        visible = "" if shown(window, desktop.monitors()) else f", {where(window.workspace)}"
        return self._window(window), f"the {window.app} window{visible}", True

    def capture(self, kind: str, *, workspace: str = "", window: Win | None = None) -> bytes:
        """Return a PNG of the screen, a workspace (`workspace`) or one window, saved nowhere.

        The same capture as `pc_screenshot`, for a screen watch: only `grim` reads frames (a
        window by `-T`, a monitor by `-o`), so nothing is focused, moved or resized.

        Raises:
            CaptureError: nothing to capture, or `grim` failed.
        """
        if kind == "window":
            if window is None:
                msg = "no window to capture"
                raise CaptureError(msg)
            data = self._window(window)
        elif kind == "workspace":
            desktop = self._windows.desktop
            data = self._workspace(workspace, desktop.monitors(), desktop.windows())
        else:
            data = self._capture([])
        _decoded(data)
        return data

    def _save(self, data: bytes) -> Path:
        self._folder.mkdir(parents=True, exist_ok=True)
        stamp = self._now().strftime("%Y-%m-%d_%H-%M-%S")
        for n in range(1, 100):
            path = self._folder / f"screenshot-{stamp}{'' if n == 1 else f'-{n}'}.png"
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                continue
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            return path
        msg = "too many screenshots were taken this second"
        raise OperationError(msg)

    def screenshot(self, args: dict[str, Any]) -> Result:
        """Take the picture, save it, copy it and show it in the chat; say where it went.

        Raises:
            OperationError: the picture could not be taken, read or saved.
        """
        request = current_request()
        if request is None or request.foreign:
            msg = "a screenshot is taken only on the owner's own request, never from PC audio"
            raise OperationError(msg)
        try:
            data, what, names_program = self._take(args)
            _decoded(data)
        except CaptureError as exc:
            raise OperationError(f"Couldn't take that screenshot: {exc}.") from exc
        try:
            path = self._save(data)
        except OSError as exc:
            msg = f"Couldn't save the screenshot: {exc.strerror or type(exc).__name__}."
            raise OperationError(msg) from exc
        copied = self._copy(data)
        look: Picture | None = None
        with contextlib.suppress(PictureError):  # saved and copied all the same
            look = prepare(data, path.name)
        picture = ""
        if self._shown is not None:
            with contextlib.suppress(PictureError, OSError):  # saved and copied all the same
                picture = str(self._shown.keep(prepare(data, path.name, max_side=SHOWN_SIDE)))
        also = " and copied it to the clipboard" if copied else " (the clipboard did not take it)"
        text = (
            f"Saved a screenshot of {what} to {path}{also}."
            + (" It shows in the chat." if picture else "")
            + ("" if look is not None else " You can't see it.")
        )
        spoken = (
            ""
            if names_program
            else f"Took a screenshot of {what}; it's in {path.parent.name}"
            + (" and on the clipboard." if copied else ".")
        )
        return Result(
            text,
            Origin.PROCESS if names_program else Origin.TOOL_STATUS,
            spoken=spoken,
            picture=picture,
            caption=f"{what} · {path.name}" if picture else "",
            look=look,
        )

    def operations(self) -> list[Operation]:
        """Return `pc_screenshot`."""
        return [
            Operation(
                "pc_screenshot",
                "Take a screenshot: the screen, a workspace (even off screen), a window "
                "pc_windows listed, or the focused one. Saved and copied; you see it next.",
                {
                    "type": "object",
                    "properties": {
                        "of": {"type": "string", "enum": list(KINDS)},
                        "workspace": {"type": "string", "enum": [*WORKSPACES, BACKGROUND]},
                        "window": {"type": "integer", "minimum": 1, "maximum": MAX_LISTED},
                    },
                    "required": ["of"],
                    "additionalProperties": False,
                },
                self.screenshot,
                spoken_summary=lambda a: f"take a screenshot of the {a.get('of', 'screen')}",
            ),
        ]
