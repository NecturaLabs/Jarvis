"""The Azure keyword wake word: a helper process with no network, switchable with LiveKit's."""

import io
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from jarvis import azure_wake_helper, cli, config
from jarvis.audio import FRAME, AudioError, AzureWake, OwnVoiceCheck, azure_wake_command
from jarvis.cli import main

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

QUIET = np.zeros(FRAME, dtype=np.int16)
WAIT_S = 5.0

# A stand-in for the helper: "ready", then a "wake" after the first frame it reads.
FAKE_HELPER = """
import sys
out = sys.stdout
out.write("ready\\n"); out.flush()
first = True
while chunk := sys.stdin.buffer.read(2560):
    if first:
        out.write("wake\\n"); out.flush()
        first = False
"""


def _fake(script: str) -> list[str]:
    return [sys.executable, "-I", "-c", script]


def _until(check: Callable[[], bool], timeout: float = WAIT_S) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.01)
    return False


def test_the_helper_runs_only_in_a_namespace_with_no_network() -> None:
    cmd = azure_wake_command(Path("/venv/bin/python"), Path("/models/hey.table"))

    assert cmd[:4] == ["unshare", "--map-current-user", "--net", "--"]
    assert cmd[4:6] == ["/venv/bin/python", "-I"]
    assert cmd[-2:] == [str(Path(azure_wake_helper.__file__)), "/models/hey.table"]


def test_a_wake_line_from_the_helper_is_heard_once() -> None:
    wake = AzureWake(_fake(FAKE_HELPER))
    try:
        assert _until(lambda: wake.heard(QUIET))
        assert not wake.heard(QUIET)
    finally:
        wake.close()


def test_reset_forgets_a_wake_not_yet_reported() -> None:
    wake = AzureWake(_fake(FAKE_HELPER))
    try:
        wake.heard(QUIET)
        time.sleep(0.3)  # the helper has answered the frame by now
        wake.reset()
        assert not wake.heard(QUIET)
    finally:
        wake.close()


def test_a_helper_that_never_gets_ready_is_an_audio_error_with_its_reason() -> None:
    script = "import sys; sys.stderr.write('no SDK here\\n'); sys.exit(3)"

    with pytest.raises(AudioError, match="no SDK here"):
        AzureWake(_fake(script), ready_s=WAIT_S)


def test_a_helper_that_stopped_is_an_audio_error() -> None:
    wake = AzureWake(_fake("print('ready', flush=True)"))
    try:
        assert _until(lambda: wake._proc.poll() is not None)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(AudioError, match="stopped"):
            wake.heard(QUIET)
    finally:
        wake.close()


def test_a_helper_that_stops_reading_never_blocks_the_wake_loop() -> None:
    script = "import sys, time; print('ready', flush=True); time.sleep(30)"
    wake = AzureWake(_fake(script))
    done = threading.Event()

    def feed() -> None:
        for _ in range(200):  # about 500 kB, far more than a pipe holds
            wake.heard(QUIET)
        done.set()

    try:
        threading.Thread(target=feed, daemon=True).start()
        assert done.wait(WAIT_S)
    finally:
        wake.close()


class _Stream:
    """The SDK's push stream: what the helper wrote to it, and an empty chunk once closed."""

    def __init__(self, fed: list[bytes]) -> None:
        super().__init__()
        self.fed = fed

    def write(self, chunk: bytes) -> None:
        self.fed.append(chunk)

    def close(self) -> None:
        self.fed.append(b"")


class _Recognizer:
    """The SDK's keyword recognizer: each recognition ends with the next reason in the list."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__()
        self.reasons = reasons

    def recognize_once_async(self, _model: object) -> SimpleNamespace:
        reason = self.reasons.pop(0)
        return SimpleNamespace(get=lambda: SimpleNamespace(reason=reason))


def _fake_sdk(reasons: list[str], fed: list[bytes]) -> ModuleType:
    sdk = ModuleType("fake_speech_sdk")
    stream = _Stream(fed)
    recognizer = _Recognizer(reasons)

    def model(path: str) -> tuple[str, str]:
        return ("model", path)

    def stream_format(**settings: int) -> dict[str, int]:
        return settings

    def push_stream(stream_format: object) -> _Stream:
        assert stream_format == {"samples_per_second": 16000, "bits_per_sample": 16, "channels": 1}
        return stream

    def audio_config(stream: _Stream) -> _Stream:
        return stream

    def keyword_recognizer(config: _Stream) -> _Recognizer:
        assert config is stream
        return recognizer

    sdk.__dict__.update(
        KeywordRecognitionModel=model,
        KeywordRecognizer=keyword_recognizer,
        ResultReason=SimpleNamespace(RecognizedKeyword="keyword"),
        audio=SimpleNamespace(
            AudioStreamFormat=stream_format,
            PushAudioInputStream=push_stream,
            AudioConfig=audio_config,
        ),
    )
    return sdk


def test_the_helper_says_ready_then_one_wake_per_keyword_until_its_input_ends() -> None:
    fed: list[bytes] = []
    out = io.StringIO()
    sdk = _fake_sdk(["keyword", "keyword", "canceled"], fed)

    assert azure_wake_helper.listen(sdk, "hey.table", io.BytesIO(b"\x00" * 6000), out) == 0

    assert out.getvalue() == "ready\nwake\nwake\n"
    assert _until(lambda: fed[-1:] == [b""])
    assert b"".join(fed) == b"\x00" * 6000


def test_the_helper_imports_nothing_from_jarvis() -> None:
    source = Path(azure_wake_helper.__file__).read_text(encoding="utf-8")

    assert "import jarvis" not in source
    assert "from jarvis" not in source
    assert "SPDX-License-" + "Identifier: MIT" in source  # split, so REUSE does not read it here


def test_with_an_azure_model_a_sentence_naming_jarvis_cannot_be_spoken_over() -> None:
    class Never:
        def __init__(self) -> None:
            super().__init__()
            self.frames = 0

        def heard(self, frame: NDArray[np.int16]) -> bool:
            self.frames += frame.size
            return False

        def reset(self) -> None:
            self.frames = 0

    check = OwnVoiceCheck(Never(), cli.WAKE_NAME)
    pcm = np.zeros(1600, dtype=np.int16).tobytes()

    assert check(pcm, 16000, "Jarvis here, at your service.")
    assert not check(pcm, 16000, "The weather is fine.")
    assert not OwnVoiceCheck(Never())(pcm, 16000, "Jarvis here.")


class _LiveKit:
    """Stands in for LiveKit's model, which needs its file."""

    def __init__(self, path: Path, threshold: float) -> None:
        super().__init__()
        self.loaded = (path, threshold)


def _engine(monkeypatch: pytest.MonkeyPatch, **audio: object) -> config.Config:
    import jarvis.audio  # ruff: ignore[import-outside-top-level] - patched where cli reads it

    monkeypatch.setattr(jarvis.audio, "WakeWord", _LiveKit)
    cfg = config.Config()
    return cfg.model_copy(update={"audio": cfg.audio.model_copy(update=audio)})


def test_livekit_is_the_default_wake_word(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(cli.wake_word(_engine(monkeypatch)), _LiveKit)


def test_azure_without_a_model_file_falls_back_to_livekit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _engine(
        monkeypatch, wake_engine="azure", azure_keyword_model=str(tmp_path / "missing.table")
    )

    assert isinstance(cli.wake_word(cfg), _LiveKit)


def test_azure_that_cannot_start_falls_back_to_livekit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import jarvis.audio  # ruff: ignore[import-outside-top-level] - patched where cli reads it

    table = tmp_path / "hey.table"
    table.write_bytes(b"model")

    def broken(_command: list[str]) -> None:
        msg = "the Azure wake helper did not start"
        raise AudioError(msg)

    monkeypatch.setattr(jarvis.audio, "AzureWake", broken)
    cfg = _engine(monkeypatch, wake_engine="azure", azure_keyword_model=str(table))

    assert isinstance(cli.wake_word(cfg), _LiveKit)


def test_azure_with_a_model_listens_through_the_no_network_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import jarvis.audio  # ruff: ignore[import-outside-top-level] - patched where cli reads it

    table = tmp_path / "hey.table"
    table.write_bytes(b"model")
    started: list[list[str]] = []

    def azure(command: list[str]) -> str:
        started.append(command)
        return "az"

    monkeypatch.setattr(jarvis.audio, "AzureWake", azure)
    cfg = _engine(monkeypatch, wake_engine="azure", azure_keyword_model=str(table))

    assert cast("object", cli.wake_word(cfg)) == "az"
    assert started[0][:3] == ["unshare", "--map-current-user", "--net"]
    assert started[0][4] == str(cfg.data_dir / cli.AZURE_PYTHON)
    assert started[0][-1] == str(table)


def test_the_wake_command_switches_the_engine_and_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    restarts: list[bool] = []
    monkeypatch.setattr(cli, "_restart_if_running", lambda: restarts.append(True))
    table = tmp_path / "hey.table"
    table.write_bytes(b"model")

    assert main(["wake", "azure", "--model", str(table)]) == 0
    saved = config.load().audio
    assert (saved.wake_engine, saved.azure_keyword_model) == ("azure", str(table))

    assert main(["wake", "livekit"]) == 0
    saved = config.load().audio
    assert (saved.wake_engine, saved.azure_keyword_model) == ("livekit", str(table))
    assert restarts == [True, True]
    assert "LiveKit" in capsys.readouterr().out


def test_switching_to_azure_without_a_model_says_livekit_stays(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_restart_if_running", lambda: None)

    assert main(["wake", "azure"]) == 0

    assert "keeps LiveKit's" in capsys.readouterr().out
