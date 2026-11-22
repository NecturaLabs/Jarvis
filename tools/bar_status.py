#!/usr/bin/env python3
r"""Turn `jarvis status --json` into what an Omarchy bar command module shows (issue #14).

The bar runs this script on an interval (`docs/configuration.md`, "Omarchy bar") and shows what it
prints: one JSON line with `text` (a Nerd Font glyph), `tooltip` and `class`. The bar highlights a
module whose class includes `active`; the other class names say which state it is.

    tools/bar_status.py

It starts nothing but `jarvis status --json`, which reads the control socket (or, with Jarvis not
running, asks the model server how it is) and never loads the model or generates tokens. It
always prints one line and exits 0, so the bar shows a state for every outcome:

- one of `jarvis.control.STATES` (idle, listening, thinking, speaking, paused, model loading,
  unavailable), with `not running` told apart from `unavailable` by the reading's `running`;
- `timeout`: `jarvis status` did not answer in time;
- `failed`: the command is missing or exited with an error;
- `malformed`: it answered with something that is not a status reading.

Nothing the reading says is shown except through the fixed texts below (the model's state only
when it is one of the words the model server uses), because the bar's tooltip is not plain text.
Standard library only, so it starts in milliseconds beside the 0.2 s `jarvis` takes.
"""

import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, cast

COMMAND: Final = ("jarvis", "status", "--json")
TIMEOUT_S: Final = 8.0  # a Jarvis that is running but stuck answers in 10 s (control.send)
MAX_READING_CHARS: Final = 4096

ACTIVE: Final = "active"  # the one class the bar styles: its module is highlighted
HINT: Final = "Left click: pause. Right click: resume."

# Glyphs from the Nerd Font Omarchy ships (Material Design set).
_ROBOT: Final = "\U000f06a9"
_ROBOT_OUTLINE: Final = "\U000f167a"
_ROBOT_OFF: Final = "\U000f1642"
_ROBOT_CONFUSED: Final = "\U000f169d"
_THINKING: Final = "\U000f051f"
_SPEAKING: Final = "\U000f057e"
_PAUSED: Final = "\U000f03e4"
_LOADING: Final = "\U000f0996"
_ALERT: Final = "\U000f0026"

# A model server's states, as `jarvis status` reports them; anything else is not shown.
_MODEL_STATES: Final = re.compile(r"[a-z][a-z -]{0,19}")

type Source = Callable[[], str]
"""One reading of `jarvis status --json`, or `TimeoutError`, `OSError` or `StatusError`."""


class StatusError(Exception):
    """`jarvis status` ran and exited with an error."""


@dataclass(frozen=True, slots=True)
class Shown:
    """What the bar shows for one reading."""

    text: str
    tooltip: str
    classes: tuple[str, ...]

    def as_json(self) -> str:
        """Return the one line the bar reads."""
        return json.dumps(
            {"text": self.text, "tooltip": self.tooltip, "class": list(self.classes)},
            ensure_ascii=False,
        )


@dataclass(frozen=True, slots=True)
class _Face:
    glyph: str
    says: str
    active: bool = False


# One face for each of `jarvis.control.STATES`; tests/test_bar_status.py keeps the two in step.
FACES: Final = {
    "idle": _Face(_ROBOT_OUTLINE, "waiting for the wake word"),
    "listening": _Face(_ROBOT, "listening", active=True),
    "thinking": _Face(_THINKING, "thinking", active=True),
    "speaking": _Face(_SPEAKING, "speaking", active=True),
    "paused": _Face(_PAUSED, "paused: the microphone is let go until you resume"),
    "model loading": _Face(_LOADING, "loading the model", active=True),
    "unavailable": _Face(_ALERT, "running but cannot work (microphone lost or model server down)"),
}
_NOT_RUNNING: Final = _Face(_ROBOT_OFF, "not running (start it with `jarvis start`)")
_TIMEOUT: Final = _Face(_ROBOT_CONFUSED, "did not answer the status check in time")
_FAILED: Final = _Face(_ALERT, "the status check failed (is `jarvis` installed and on the PATH?)")
_MALFORMED: Final = _Face(_ROBOT_CONFUSED, "answered the status check with something unreadable")


def _shown(face: _Face, name: str, model: str | None = None) -> Shown:
    lines = [f"Jarvis: {face.says}"]
    if model is not None:
        lines.append(f"Model: {model}")
    lines.append(HINT)
    classes = (name.replace(" ", "-"), ACTIVE) if face.active else (name.replace(" ", "-"),)
    return Shown(face.glyph, "\n".join(lines), classes)


def _model(reading: dict[str, object]) -> str | None:
    value = reading.get("model")
    if isinstance(value, str) and _MODEL_STATES.fullmatch(value):
        return value
    return None


def render(raw: str) -> Shown:
    """Return what the bar shows for one reading, `malformed` when it is not a status."""
    unreadable = _shown(_MALFORMED, "malformed")
    if len(raw) > MAX_READING_CHARS:
        return unreadable
    try:
        value: object = json.loads(raw)
    except ValueError:
        return unreadable
    if not isinstance(value, dict):
        return unreadable
    reading = {str(key): item for key, item in cast("dict[object, object]", value).items()}
    state = reading.get("state")
    if not isinstance(state, str) or state not in FACES:
        return unreadable
    if reading.get("running") is False:
        return _shown(_NOT_RUNNING, "not-running", _model(reading))
    return _shown(FACES[state], state, _model(reading))


def show(source: Source) -> Shown:
    """Read once from `source` and return what the bar shows, whatever goes wrong."""
    try:
        raw = source()
    except TimeoutError:
        return _shown(_TIMEOUT, "timeout")
    except OSError, StatusError:
        return _shown(_FAILED, "failed")
    return render(raw)


def jarvis_status(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout: float = TIMEOUT_S,
) -> str:
    """Run `jarvis status --json` and return its output.

    Raises:
        TimeoutError: it did not finish in `timeout` seconds (it is killed).
        FileNotFoundError: `jarvis` is not on the PATH (another `OSError` if it cannot be started).
        StatusError: it exited with an error.
    """
    program = shutil.which(COMMAND[0])
    if program is None:
        msg = f"{COMMAND[0]} is not on the PATH"
        raise FileNotFoundError(msg)
    try:
        done = run(
            [program, *COMMAND[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"{' '.join(COMMAND)} took more than {timeout} s"
        raise TimeoutError(msg) from exc
    if done.returncode != 0:
        msg = f"{' '.join(COMMAND)} exited with {done.returncode}"
        raise StatusError(msg)
    return done.stdout


def main() -> int:
    """Print the bar's line for the current status; always exit 0 so the bar shows a state."""
    sys.stdout.write(show(jarvis_status).as_json() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
