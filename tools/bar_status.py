#!/usr/bin/env -S python3 -I -S
r"""Show what Jarvis is doing as an Omarchy bar command module (issue #14).

The bar runs this script on an interval (`docs/configuration.md`, "Omarchy bar") and shows what it
prints: one JSON line with `text` (a Nerd Font glyph), `tooltip` and `class`. The bar highlights a
module whose class includes `active`; the other class names say which state it is.

    tools/bar_status.py

It sends the one `status` request of Jarvis's control socket (`src/jarvis/control.py`) and never
loads the model or generates tokens. It always prints one line and exits 0, so the bar shows a state
for every outcome:

- one of `jarvis.control.STATES` (idle, listening, thinking, speaking, paused, model loading,
  unavailable), with `not running` (nobody serves the socket) told apart from `unavailable`;
- `timeout`: Jarvis did not answer in time;
- `failed`: the connection broke while it was being read;
- `malformed`: it answered with something that is not a status reading.

Nothing the reading says is shown except through the fixed texts below (the model's state only
when it is one of the words the model server uses), because the bar's tooltip is not plain text.

A poll must cost almost nothing, since the bar starts it every few seconds beside a game. So this
does not start `jarvis` (Python's start-up and `jarvis.cli`'s imports cost 0.2 s of CPU) and imports
as little as it can: the shebang runs Python without its `site` import, and the script imports no
`typing`, `dataclasses`, `subprocess` or `shutil` (each costs milliseconds), so a poll takes about
15 ms of CPU. It repeats the socket's path and request from `control.py` (the one place they
are defined); `tests/test_bar_status.py` fails if they drift apart. With Jarvis not running it
does not ask the model server, which would mean reading Jarvis's config: the bar says `not
running` without a model state.
"""

import json
import os
import re
import socket
import sys
import time
from collections.abc import Callable

TIMEOUT_S = 8.0  # a Jarvis that is running but stuck answers in 10 s (control.send)
MAX_READING_CHARS = 4096

# `control.socket_path()` and the request `control.send("status")` writes; the tests compare them.
REQUEST = b'{"command": "status"}\n'
NOT_RUNNING = '{"state": "unavailable", "paused": false, "running": false}'

ACTIVE = "active"  # the one class the bar styles: its module is highlighted
HINT = "Left click: pause. Right click: resume."

# Glyphs from the Nerd Font Omarchy ships (Material Design set).
_ROBOT = "\U000f06a9"
_ROBOT_OUTLINE = "\U000f167a"
_ROBOT_OFF = "\U000f1642"
_ROBOT_CONFUSED = "\U000f169d"
_THINKING = "\U000f051f"
_SPEAKING = "\U000f057e"
_PAUSED = "\U000f03e4"
_LOADING = "\U000f0996"
_ALERT = "\U000f0026"

# A model server's states, as the running Jarvis reports them; anything else is not shown.
_MODEL_STATES = re.compile(r"[a-z][a-z -]{0,19}")

type Source = Callable[[], str]
"""One reading of Jarvis's status, or `TimeoutError` or `OSError`."""


class Shown:
    """What the bar shows for one reading."""

    __slots__ = ("classes", "text", "tooltip")

    def __init__(self, text: str, tooltip: str, classes: tuple[str, ...]) -> None:
        """Hold the glyph, the tooltip and the CSS classes."""
        super().__init__()
        self.text = text
        self.tooltip = tooltip
        self.classes = classes

    def as_json(self) -> str:
        """Return the one line the bar reads."""
        return json.dumps(
            {"text": self.text, "tooltip": self.tooltip, "class": list(self.classes)},
            ensure_ascii=False,
        )


class _Face:
    """How one state looks: its glyph, what the tooltip says and whether the module is lit."""

    __slots__ = ("active", "glyph", "says")

    def __init__(self, glyph: str, says: str, *, active: bool = False) -> None:
        super().__init__()
        self.glyph = glyph
        self.says = says
        self.active = active

    def shown(self, name: str, model: str | None = None) -> Shown:
        """Return what the bar shows for this face under the class `name`."""
        lines = [f"Jarvis: {self.says}"]
        if model is not None:
            lines.append(f"Model: {model}")
        lines.append(HINT)
        css = name.replace(" ", "-")
        return Shown(self.glyph, "\n".join(lines), (css, ACTIVE) if self.active else (css,))


# One face for each of `jarvis.control.STATES`; tests/test_bar_status.py keeps the two in step.
FACES = {
    "idle": _Face(_ROBOT_OUTLINE, "waiting for the wake word"),
    "listening": _Face(_ROBOT, "listening", active=True),
    "thinking": _Face(_THINKING, "thinking", active=True),
    "speaking": _Face(_SPEAKING, "speaking", active=True),
    "paused": _Face(_PAUSED, "paused: the microphone is let go until you resume"),
    "model loading": _Face(_LOADING, "loading the model", active=True),
    "unavailable": _Face(_ALERT, "running but cannot work (microphone lost or model server down)"),
}
_NOT_RUNNING = _Face(_ROBOT_OFF, "not running (start it with `jarvis start`)")
_TIMEOUT = _Face(_ROBOT_CONFUSED, "did not answer the status check in time")
_FAILED = _Face(_ALERT, "the status check failed (the connection broke)")
_MALFORMED = _Face(_ROBOT_CONFUSED, "answered the status check with something unreadable")


def _model(reading: dict[str, object]) -> str | None:
    value = reading.get("model")
    if isinstance(value, str) and _MODEL_STATES.fullmatch(value):
        return value
    return None


def _parse(raw: str) -> dict[str, object] | None:
    """Return the reading's fields, or None when `raw` is not a JSON object."""
    if len(raw) > MAX_READING_CHARS:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    items: dict[object, object] = value  # pyright: ignore[reportUnknownVariableType] - JSON's own
    return {str(key): item for key, item in items.items()}


def render(raw: str) -> Shown:
    """Return what the bar shows for one reading, `malformed` when it is not a status."""
    reading = _parse(raw)
    if reading is None:
        return _MALFORMED.shown("malformed")
    if reading.get("ok") is False:  # running, but it refused or failed the request
        return FACES["unavailable"].shown("unavailable")
    state = reading.get("state")
    if not isinstance(state, str) or state not in FACES:
        return _MALFORMED.shown("malformed")
    if reading.get("running") is False:
        return _NOT_RUNNING.shown("not-running")
    return FACES[state].shown(state, _model(reading))


def show(source: Source) -> Shown:
    """Read once from `source` and return what the bar shows, whatever goes wrong."""
    try:
        raw = source()
    except TimeoutError:
        return _TIMEOUT.shown("timeout")
    except OSError:
        return _FAILED.shown("failed")
    return render(raw)


def socket_path() -> str:
    """Return the control socket's path, as `jarvis.control.socket_path` does."""
    base = os.environ.get("XDG_RUNTIME_DIR", "")
    if not base.startswith("/"):
        base = f"/run/user/{os.getuid()}"
    return f"{base}/jarvis/control.sock"


def control_status(path: str | None = None, timeout: float = TIMEOUT_S) -> str:
    """Ask the running Jarvis for its status over the control socket and return its reply line.

    With nobody serving the socket (no file, nothing listening, not ours to open) the reading is
    `NOT_RUNNING`. A reply over `MAX_READING_CHARS` is cut there, which `render` shows as malformed.

    Raises:
        TimeoutError: Jarvis took more than `timeout` seconds in all to accept or answer. (Any
            other `OSError` after the connection was made is not caught either.)
    """
    deadline = time.monotonic() + timeout
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout)
        try:
            conn.connect(path or socket_path())
        except TimeoutError:
            raise
        except OSError:
            return NOT_RUNNING
        conn.sendall(REQUEST)
        data = b""
        while b"\n" not in data and len(data) <= MAX_READING_CHARS:
            left = deadline - time.monotonic()
            if left <= 0:
                msg = f"Jarvis took more than {timeout} s"
                raise TimeoutError(msg)
            conn.settimeout(left)
            chunk = conn.recv(MAX_READING_CHARS + 1 - len(data))
            if not chunk:
                break
            data += chunk
    return data.split(b"\n", 1)[0].decode("utf-8", "replace")


def main() -> int:
    """Print the bar's line for the current status; always exit 0 so the bar shows a state."""
    sys.stdout.write(show(control_status).as_json() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
