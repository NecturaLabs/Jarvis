"""Session transitions on this PC's real desktop, microphone and Strata: never run in CI.

The automatic parts of each transition have fake-environment tests (`test_voice.py`,
`test_control.py`, `test_audio.py`, `test_model.py`); these check that the real pieces answer the
way those fakes assume. The owner runs the ones that open the microphone, and the steps that
need hands (unplugging the microphone, suspending) are in `docs/playbook/session-transitions.md`.
"""

import pytest

from jarvis import config
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
