"""The Azure keyword wake word: a helper process with no network, switchable with LiveKit's."""

import io
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, override

import numpy as np
import pytest

from jarvis import azure_wake_helper, cli, config
from jarvis.audio import (
    FRAME,
    RATE,
    AudioError,
    AzureWake,
    FallbackWake,
    OwnVoiceCheck,
    azure_wake_command,
)
from jarvis.cli import main

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

QUIET = np.zeros(FRAME, dtype=np.int16)
WAIT_S = 5.0

# A stand-in for the helper: "ready", then, after its fourth frame, a keyword that ended there.
FAKE_HELPER = """
import sys
out = sys.stdout
out.write("ready\\n"); out.flush()
frames = 0
while chunk := sys.stdin.buffer.read(2560):
    frames += 1
    if frames == 4:
        out.write("wake 5120\\n"); out.flush()
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
        for _ in range(3):
            assert not wake.heard(QUIET)
        # The fourth frame goes straight to the helper: through heard() its report could be
        # taken by that same call, and then there would be nothing left to forget.
        os.write(wake._stdin, QUIET.tobytes())  # pyright: ignore[reportPrivateUsage]
        assert _until(lambda: bool(wake._wakes))  # pyright: ignore[reportPrivateUsage]
        wake.reset()
        assert not wake.heard(QUIET)
        assert wake.after_wake() == []
    finally:
        wake.close()


def test_the_words_right_after_the_keyword_are_handed_to_the_request() -> None:
    wake = AzureWake(_fake(FAKE_HELPER))
    numbered = np.arange(20 * FRAME, dtype=np.int16)
    try:
        sent = 0
        for _ in range(3):  # the fake helper reports the keyword after its fourth frame
            assert not wake.heard(numbered[sent * FRAME : (sent + 1) * FRAME])
            sent += 1
        # The report may come back within the fourth call or after it; wait for it, never on
        # how fast the helper answers (a slow runner lagged it, a fast one beat the call).
        heard = wake.heard(numbered[sent * FRAME : (sent + 1) * FRAME])
        sent += 1
        if not heard:
            assert _until(lambda: bool(wake._wakes))  # pyright: ignore[reportPrivateUsage]
            assert wake.heard(numbered[sent * FRAME : (sent + 1) * FRAME])
            sent += 1
        lead = np.concatenate(wake.after_wake())
        # From a quarter second before the keyword's end (sample 5120) to the last frame sent.
        assert lead[0] == 5120 - RATE // 4
        assert lead[-1] == sent * FRAME - 1
        assert wake.after_wake() == []
    finally:
        wake.close()


def test_a_helper_that_closed_its_input_is_an_audio_error() -> None:
    script = "import os, sys, time; os.close(0); print('ready', flush=True); time.sleep(30)"
    wake = AzureWake(_fake(script))  # waits for "ready", which comes after the input closed
    try:

        def feed() -> None:
            for _ in range(5):
                wake.heard(QUIET)

        with pytest.raises(AudioError, match="stopped"):
            feed()
    finally:
        wake.close()


def test_a_helper_that_dies_after_ready_hands_over_to_livekit() -> None:
    class Backup:
        def __init__(self) -> None:
            super().__init__()
            self.frames = 0

        def heard(self, frame: NDArray[np.int16]) -> bool:
            self.frames += frame.size
            return True

        def reset(self) -> None:
            self.frames = 0

    backup = Backup()
    script = "import sys; print('ready', flush=True); sys.stderr.write('model broke\\n')"
    wake = FallbackWake(AzureWake(_fake(script)), lambda: backup)

    assert _until(lambda: wake.heard(QUIET))
    assert backup.frames > 0
    assert wake.after_wake() == []
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


def _result(reason: str, *, error: str = "", offset_s: float = 0.0) -> SimpleNamespace:
    ended = "error" if error else "end"
    return SimpleNamespace(
        reason=reason,
        offset=round(offset_s * 10_000_000),
        duration=10_000_000,  # one second
        cancellation_details=SimpleNamespace(reason=ended, error_details=error),
    )


@dataclass(frozen=True)
class _Due:
    """A recognition result that arrives only once `wait` is true."""

    wait: Callable[[], bool]
    result: SimpleNamespace


type _Next = SimpleNamespace | _Due


class _Recognizer:
    """The SDK's keyword recognizer: each recognition ends with the next result in the list."""

    def __init__(self, results: Sequence[_Next]) -> None:
        super().__init__()
        self.results = list(results)

    def recognize_once_async(self, _model: object) -> SimpleNamespace:
        item = self.results.pop(0)

        def get() -> SimpleNamespace:
            if isinstance(item, _Due):
                assert _until(item.wait)
                return item.result
            return item

        return SimpleNamespace(get=get)


END = _result("canceled")  # the input ended


def _fake_sdk(results: Sequence[_Next], fed: list[bytes]) -> ModuleType:
    """A stand-in for the Speech SDK; the first result answers the helper's probe."""
    sdk = ModuleType("fake_speech_sdk")
    recognizer = _Recognizer(results)

    def model(path: str) -> tuple[str, str]:
        return ("model", path)

    def stream_format(**settings: int) -> dict[str, int]:
        assert settings == {"samples_per_second": 16000, "bits_per_sample": 16, "channels": 1}
        return settings

    def push_stream(stream_format: object) -> _Stream:
        assert stream_format
        return _Stream(fed)

    def audio_config(stream: _Stream) -> _Stream:
        return stream

    def keyword_recognizer(_config: _Stream) -> _Recognizer:
        return recognizer

    sdk.__dict__.update(
        KeywordRecognitionModel=model,
        KeywordRecognizer=keyword_recognizer,
        ResultReason=SimpleNamespace(RecognizedKeyword="keyword", Canceled="canceled"),
        CancellationReason=SimpleNamespace(Error="error"),
        audio=SimpleNamespace(
            AudioStreamFormat=stream_format,
            PushAudioInputStream=push_stream,
            AudioConfig=audio_config,
        ),
    )
    return sdk


SILENT = b"\0" * azure_wake_helper.SILENCE_BYTES


class _Input(io.BytesIO):
    """The helper's standard input: each chunk is served once its condition holds; then it ends."""

    def __init__(self, *steps: tuple[Callable[[], bool], bytes]) -> None:
        super().__init__()
        self._steps = list(steps)

    @override
    def read(self, size: int | None = -1) -> bytes:
        del size
        if not self._steps:
            return b""
        ready, chunk = self._steps.pop(0)
        assert _until(ready)
        return chunk


def _now() -> bool:
    return True


def test_the_helper_says_ready_then_where_each_keyword_ended_until_its_input_ends() -> None:
    fed: list[bytes] = []
    out, err = io.StringIO(), io.StringIO()
    # The SDK counts the silence the helper adds: 1.5 s first, and again after each keyword.
    results = [END, _result("keyword", offset_s=2.0), _result("keyword", offset_s=6.0), END]
    sdk = _fake_sdk(results, fed)
    source = _Input((_now, b"\x00" * 6000), (lambda: fed.count(SILENT) == 3, b""))

    assert azure_wake_helper.listen(sdk, "hey.table", source, out, err) == 0

    assert (
        out.getvalue() == "ready\nwake 24000\nwake 64000\n"
    )  # (offset + 1 s) at 16 kHz, no silence
    assert not err.getvalue()
    assert _until(lambda: b"".join(fed).count(b"\x00") >= 16000 + 6000)


def test_the_keyword_engine_is_given_silence_first_and_after_each_keyword() -> None:
    fed: list[bytes] = []
    out, err = io.StringIO(), io.StringIO()
    first, second, third = (bytes([n]) * 2560 for n in (1, 2, 3))  # 1280 samples each
    silence, chunk = azure_wake_helper.SILENCE_SAMPLES, 1280
    # Where the SDK says each keyword ended in its stream: after the second chunk, then after the
    # third, which came after a second silence.
    end_1 = silence + 2 * chunk
    end_2 = 2 * silence + 3 * chunk
    results = [
        END,
        _Due(lambda: second in fed, _result("keyword", offset_s=end_1 / RATE - 1)),
        _Due(lambda: third in fed, _result("keyword", offset_s=end_2 / RATE - 1)),
        END,
    ]
    source = _Input(
        (_now, first),
        (_now, second),
        (lambda: fed.count(SILENT) == 2, third),
        (lambda: fed.count(SILENT) == 3, b""),
    )

    assert azure_wake_helper.listen(_fake_sdk(results, fed), "hey.table", source, out, err) == 0

    assert _until(lambda: len(fed) == 9)
    assert fed == [
        b"\0" * azure_wake_helper.PROBE_BYTES,
        b"",
        SILENT,
        first,
        second,
        SILENT,
        third,
        SILENT,
        b"",
    ]
    # Reported in the input's own samples: the silence before each keyword's end is taken off.
    assert out.getvalue() == f"ready\nwake {2 * chunk}\nwake {3 * chunk}\n"
    assert not err.getvalue()


def test_a_position_in_the_stream_is_told_as_a_position_in_the_input() -> None:
    fed: list[bytes] = []
    stream = azure_wake_helper.PushedStream(_Stream(fed))
    silence = azure_wake_helper.SILENCE_SAMPLES
    stream.silence()  # the stream's first 24000 samples
    stream.write(b"\x01" * 2560)  # then 1280 of the input
    stream.silence()  # then 24000 more

    assert stream.input_position(silence) == 0  # the end of the first silence
    assert stream.input_position(silence + 1280) == 1280
    assert stream.input_position(silence + 1280 + 100) == 1280  # inside the second silence
    assert stream.input_position(silence + 1280 + silence + 5) == 1285  # then the input again
    assert stream.input_position(100) == 0  # inside the first
    # An end before a silence is not shortened by it.
    assert stream.input_position(silence + 10) == 10


def test_the_silence_is_at_least_what_the_keyword_engine_asks_for() -> None:
    assert azure_wake_helper.SILENCE_S >= 1.5
    assert azure_wake_helper.SILENCE_BYTES == azure_wake_helper.SILENCE_SAMPLES * 2


def test_no_silence_is_written_to_a_stream_whose_input_has_ended() -> None:
    fed: list[bytes] = []
    out, err = io.StringIO(), io.StringIO()
    results = [END, _Due(lambda: fed[-1:] == [b""], _result("keyword", offset_s=2.0)), END]
    source = _Input((_now, b"\x01" * 2560))  # then it ends, closing the stream

    assert azure_wake_helper.listen(_fake_sdk(results, fed), "hey.table", source, out, err) == 0

    assert fed == [b"\0" * azure_wake_helper.PROBE_BYTES, b"", SILENT, b"\x01" * 2560, b""]
    assert out.getvalue().startswith("ready\nwake ")
    assert not err.getvalue()


def test_a_model_the_sdk_rejects_is_said_before_ready_and_ends_the_helper() -> None:
    out, err = io.StringIO(), io.StringIO()
    sdk = _fake_sdk([_result("canceled", error="0xc (SPXERR_UNSUPPORTED_FORMAT)\n[stack]")], [])

    assert azure_wake_helper.listen(sdk, "bad.table", io.BytesIO(b""), out, err) == 1

    assert not out.getvalue()
    assert "SPXERR_UNSUPPORTED_FORMAT" in err.getvalue()
    assert "[stack]" not in err.getvalue()


def test_a_recognizer_error_after_ready_ends_the_helper_with_its_reason() -> None:
    out, err = io.StringIO(), io.StringIO()
    sdk = _fake_sdk([END, _result("canceled", error="0x5 (SPXERR_RUNTIME_ERROR)")], [])

    assert azure_wake_helper.listen(sdk, "hey.table", io.BytesIO(b""), out, err) == 1

    assert out.getvalue() == "ready\n"
    assert "SPXERR_RUNTIME_ERROR" in err.getvalue()


def test_the_helper_check_only_tries_the_model() -> None:
    out, err = io.StringIO(), io.StringIO()
    assert azure_wake_helper.check(_fake_sdk([END], []), "hey.table", out, err) == 0
    assert out.getvalue() == "ok\n"

    bad = _fake_sdk([_result("canceled", error="0xc (SPXERR_UNSUPPORTED_FORMAT)")], [])
    assert azure_wake_helper.check(bad, "bad.table", io.StringIO(), err) == 1
    assert "SPXERR_UNSUPPORTED_FORMAT" in err.getvalue()


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

    def azure(command: list[str]) -> object:
        started.append(command)
        return object()

    monkeypatch.setattr(jarvis.audio, "AzureWake", azure)
    cfg = _engine(monkeypatch, wake_engine="azure", azure_keyword_model=str(table))

    assert isinstance(cli.wake_word(cfg), FallbackWake)
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

    assert "listens with LiveKit's" in capsys.readouterr().out


def test_switching_with_no_arguments_only_shows_the_setting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_restart_if_running", lambda: pytest.fail("no restart"))

    assert main(["wake"]) == 0

    assert capsys.readouterr().out.startswith("wake word: LiveKit")


def _azure_cfg(tmp_path: Path) -> config.Config:
    table = tmp_path / "hey.table"
    table.write_bytes(b"model")
    python = tmp_path / "python"
    python.write_text("")
    cfg = config.Config()
    audio = cfg.audio.model_copy(
        update={
            "wake_engine": "azure",
            "azure_keyword_model": str(table),
            "azure_python": str(python),
        }
    )
    return cfg.model_copy(update={"audio": audio})


def test_the_doctor_tries_the_azure_model_in_its_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ran: list[list[str]] = []

    def run(cmd: list[str], **_kw: object) -> SimpleNamespace:
        ran.append(cmd)
        return SimpleNamespace(returncode=1, stderr="the keyword model cannot be used: 0xc\n")

    monkeypatch.setattr(cli.subprocess, "run", run)
    checks = cli._azure_checks(_azure_cfg(tmp_path))  # pyright: ignore[reportPrivateUsage]

    assert [ok for ok, _, _ in checks] == [True, True, False]
    assert "0xc" in checks[-1][1]
    assert ran[0][:3] == ["unshare", "--map-current-user", "--net"]
    assert ran[0][-1] == "--check"


def test_the_doctor_checks_nothing_azure_with_livekit() -> None:
    assert cli._azure_checks(config.Config()) == []  # pyright: ignore[reportPrivateUsage]


@pytest.mark.local
def test_the_helper_namespace_has_no_network() -> None:
    import subprocess  # ruff: ignore[import-outside-top-level] - only this local test runs it

    probe = (
        "import socket\n"
        "for host in ('1.1.1.1', '127.0.0.1'):\n"
        "    s = socket.socket()\n"
        "    s.settimeout(2)\n"
        "    try:\n"
        "        s.connect((host, 53))\n"
        "        raise SystemExit(f'reached {host}')\n"
        "    except OSError:\n"
        "        pass\n"
    )
    done = subprocess.run(
        ["unshare", "--map-current-user", "--net", "--", sys.executable, "-I", "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stderr
