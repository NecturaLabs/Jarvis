"""The idle measurement on the real Jarvis services: needs this PC's units, never runs in CI."""

import subprocess
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "idle_check.py"
_WINDOW_SECONDS = 30
_TIMEOUT_SECONDS = 90


@pytest.mark.local
@pytest.mark.timeout(_TIMEOUT_SECONDS)
def test_the_running_services_are_idle() -> None:
    """Read-only: samples /proc for 30 s; starts, stops and signals nothing."""
    done = subprocess.run(
        [
            str(TOOL),
            "--unit",
            "jarvis.service",
            "--unit",
            "jarvis-strata.service",
            "--seconds",
            str(_WINDOW_SECONDS),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=_TIMEOUT_SECONDS,
    )

    assert done.returncode == 0, done.stdout + done.stderr
