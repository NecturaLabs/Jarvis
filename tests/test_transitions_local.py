"""Session transitions on this PC's real desktop, microphone and Strata: never run in CI.

The automatic parts of each transition have fake-environment tests (`test_voice.py`,
`test_control.py`, `test_audio.py`, `test_model.py`); these check that the real pieces answer the
way those fakes assume. The owner runs the ones that open the microphone, and the steps that
need hands (unplugging the microphone, suspending) are in `docs/playbook/session-transitions.md`.
"""

import json

import pytest

from jarvis import cli, config, control, services
from jarvis.app import strata_model
from jarvis.audio import MIC_STALL_S, AudioError, Microphone, lock_state
from jarvis.paths import strata_key_file
from jarvis.secretfile import read_api_key

pytestmark = pytest.mark.local

FRAMES_FOR_TWO_S = 25  # 80 ms frames


@pytest.mark.timeout(10)
def test_the_omarchy_shell_says_the_screen_is_unlocked() -> None:
    """Read-only: asks the running shell; locks nothing. Run it at an unlocked screen."""
    assert lock_state() is False


@pytest.mark.timeout(30)
def test_the_microphone_sends_frames_without_stalling() -> None:
    """Opens the owner's microphone for two seconds and keeps nothing; run with the owner."""
    mic = Microphone(config.load().audio.source, stall_s=MIC_STALL_S)
    mic.start()
    try:
        frames = mic.frames()
        for _ in range(FRAMES_FOR_TWO_S):
            next(frames)
    except AudioError as exc:
        pytest.fail(f"the microphone did not deliver: {exc}")
    finally:
        mic.stop()


@pytest.mark.timeout(15)
def test_strata_answers_and_its_model_file_is_there() -> None:
    """Read-only: one status GET, which generates nothing."""
    cfg = config.load()
    model = strata_model(cfg, read_api_key(strata_key_file()) or "")

    assert not model.file_missing(), f"{cfg.strata.model_file} is missing: is its drive mounted?"
    assert model.status() is not None, (
        "Strata is not answering: systemctl --user status jarvis-strata"
    )


@pytest.mark.timeout(15)
def test_status_names_a_state_the_bar_knows(capsys: pytest.CaptureFixture[str]) -> None:
    """Read-only: asks the running Jarvis, or Strata alone when Jarvis is not running."""
    assert cli.main(["status", "--json"]) == 0

    shown = json.loads(capsys.readouterr().out)
    assert shown["state"] in control.STATES
    assert shown["model"] in {"loaded", "unloaded", "loading", "failed", "unreachable", "missing"}
    assert isinstance(shown["running"], bool)


@pytest.mark.timeout(15)
def test_hush_reaches_the_running_jarvis() -> None:
    """Stops any speech in progress and nothing else; skipped when Jarvis is not running."""
    reply = control.send("hush")
    if reply is None:
        pytest.skip("Jarvis is not running")

    assert reply.get("ok") is True


@pytest.mark.timeout(60)
def test_pause_and_resume_reach_the_running_jarvis() -> None:
    """Changes the running Jarvis: it lets go of the microphone and unloads the model, then listens
    and loads it again (a load can take minutes). Run with the owner; skipped when Jarvis is not
    running or already paused, so a pause the owner chose is never undone.
    """
    before = control.send("status")
    if before is None or not before.get("ok"):
        pytest.skip("Jarvis is not running")
    if before.get("paused"):
        pytest.skip("Jarvis is already paused")

    try:
        paused = control.send("pause")
        assert paused is not None
        assert paused.get("paused") is True
        during = control.send("status")
        assert during is not None
        assert during.get("state") == "paused"
    finally:
        resumed = control.send("resume")
    assert resumed is not None
    assert resumed.get("paused") is False


def test_the_installed_units_follow_the_session_and_wait_for_the_model_file() -> None:
    """Read-only: the unit files systemd runs, not the text `jarvis setup` would write now."""
    cfg = config.load()
    unit_dir = services.unit_dir()
    strata = (unit_dir / services.STRATA_UNIT).read_text().splitlines()
    voice = (unit_dir / services.JARVIS_UNIT).read_text().splitlines()

    for unit in (strata, voice):
        assert "PartOf=graphical-session.target" in unit
        assert "WantedBy=graphical-session.target" in unit
    assert any(
        line.startswith("ConditionPathExists=") and str(cfg.strata.model_file) in line
        for line in strata
    ), "Strata's unit does not wait for the model file: run `jarvis setup`"
    assert f"Wants={services.STRATA_UNIT}" in voice
