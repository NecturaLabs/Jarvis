"""Microphone stream handling and transcript parsing, without real audio devices."""

import contextlib
import os
import queue
import signal
import subprocess
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
import pytest

from jarvis import audio as audio_module
from jarvis.audio import (
    FRAME,
    AppAudio,
    AudioError,
    Microphone,
    Speaker,
    app_record_command,
    app_stream_serials,
    capture_sources,
    dictation_active,
    is_phantom,
    match_app_streams,
    parse_transcript,
    pw_record_command,
    transcribe,
    voxtype_command,
    voxtype_settings,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray


def _ended_microphone(frames: int) -> Microphone:
    """A microphone fed by a process that writes `frames` silent frames, then exits."""
    mic = Microphone(command=["head", "-c", str(FRAME * 2 * frames), "/dev/zero"])
    mic.start()
    mic.wait_ended(timeout=5)
    return mic


def _first_outcome(mic: Microphone, timeout: float) -> str:
    """Return 'frame', 'ended' or 'blocked' for the next item `frames()` gives."""
    outcome = ["blocked"]

    def read() -> None:
        try:
            next(mic.frames())
            outcome[0] = "frame"
        except AudioError:
            outcome[0] = "ended"

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    thread.join(timeout)
    return outcome[0]


def test_frames_end_after_drain_instead_of_blocking() -> None:
    """Regression: drain() used to discard the end-of-stream marker, so a stop hung forever."""
    mic = _ended_microphone(frames=2)
    mic.drain()
    assert _first_outcome(mic, timeout=2) == "ended"


def test_frames_are_delivered_before_the_end() -> None:
    mic = _ended_microphone(frames=2)
    assert _first_outcome(mic, timeout=2) == "frame"


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        (
            (
                'Loading audio file: "/tmp/x.wav"\nAudio format: 16000 Hz, 1 channel(s), Int\n'
                "Processing 44954 samples (2.81s)...\n"
                "\x1b[2m2026-11-21T00:54:05.124440Z\x1b[0m"
                " \x1b[32m INFO\x1b[0m Using local whisper\n"
                "\nPlay some daft punk on Spotify.\n"
            ),
            "Play some daft punk on Spotify.",
        ),
        ("\n\n", ""),
    ],
)
def test_parse_transcript_keeps_only_the_text(stdout: str, expected: str) -> None:
    assert parse_transcript(stdout) == expected


@pytest.mark.parametrize(
    ("text", "phantom"),
    [
        ("Thank you for watching!", True),
        ("you", True),
        ("Thank you, play Glass Harbor", False),
        ("Skip this", False),
    ],
)
def test_whisper_phantoms_are_recognised(text: str, phantom: bool) -> None:
    assert is_phantom(text) is phantom


def test_voxtype_command_passes_overrides_before_the_subcommand() -> None:
    cmd = voxtype_command(
        Path("/x.wav"), model="large-v3-turbo", prompt="Glass Harbor, Night Circuit"
    )
    assert cmd == [
        "voxtype",
        "--model=large-v3-turbo",
        "--initial-prompt=Glass Harbor, Night Circuit",
        "transcribe",
        "/x.wav",
    ]


def test_a_hint_that_starts_with_a_dash_stays_the_hint() -> None:
    cmd = voxtype_command(Path("/x.wav"), engine="-v", model="-c", prompt="--help, Low Tide")

    assert cmd[1:4] == ["--model=-c", "--engine=-v", "--initial-prompt=--help, Low Tide"]


def test_voxtype_command_without_overrides_uses_the_owners_config() -> None:
    assert voxtype_command(Path("/x.wav")) == ["voxtype", "transcribe", "/x.wav"]


def _stream(
    serial: int,
    binary: str,
    app: str,
    media_class: str = "Stream/Output/Audio",
    state: str = "running",
) -> dict[str, object]:
    return {
        "id": serial,
        "type": "PipeWire:Interface:Node",
        "info": {
            "state": state,
            "props": {
                "media.class": media_class,
                "object.serial": serial,
                "application.process.binary": binary,
                "application.name": app,
                "node.name": app,
            },
        },
    }


def test_only_the_named_apps_streams_are_followed() -> None:
    objects: list[dict[str, object]] = [
        _stream(1, "spotify", "Spotify"),
        _stream(2, "Discord", "WEBRTC VoiceEngine"),
        _stream(3, "Discord", "Discord mic", media_class="Stream/Input/Audio"),
        {"id": 4, "info": None},
    ]
    assert match_app_streams(objects, ["discord"]) == ["2"]
    assert match_app_streams(objects, ["DISCORD", "spotify"]) == ["1", "2"]
    assert match_app_streams(objects, []) == []


def test_app_audio_waits_for_the_app_then_records_it() -> None:
    finds: list[int] = []
    targets: list[str] = []

    def find(_apps: list[str]) -> list[str]:
        finds.append(1)
        return [] if len(finds) == 1 else ["42"]

    def command_for(target: str) -> list[str]:
        targets.append(target)
        return ["head", "-c", str(FRAME * 2 * 3), "/dev/zero"]

    audio = AppAudio(["discord"], find=find, poll_s=0.01, command_for=command_for)
    try:
        frames = audio.frames()
        for _ in range(3):
            next(frames)
    finally:
        audio.stop()

    assert targets[0] == "42"
    assert len(finds) >= 2


def test_each_app_audio_session_ends_with_its_stream() -> None:
    """A capture over PC audio must end with the stream, not run on into the next one."""
    audio = AppAudio(
        ["discord"],
        find=lambda _apps: ["42"],
        poll_s=0.01,
        command_for=lambda _target: ["head", "-c", str(FRAME * 2 * 3), "/dev/zero"],
        feeds=lambda _pid: ["42"],
    )
    sessions = audio.sessions()
    try:
        first = next(sessions)
        frames = [next(first, None) for _ in range(4)]
        second = next(sessions)
        assert next(second, None) is not None
    finally:
        sessions.close()
        audio.stop()

    assert [frame is not None for frame in frames] == [True, True, True, False]


def test_app_audio_with_no_apps_records_all_pc_output() -> None:
    targets: list[str] = []

    def command_for(target: str) -> list[str]:
        targets.append(target)
        return ["head", "-c", str(FRAME * 2), "/dev/zero"]

    audio = AppAudio([], find=lambda _a: pytest.fail("no lookup needed"), command_for=command_for)
    try:
        next(audio.frames())
    finally:
        audio.stop()

    assert targets == [""]


def test_pw_record_command_for_pc_output() -> None:
    cmd = pw_record_command(capture_output=True)
    assert cmd[-3:] == ["-P", "{ stream.capture.sink = true }", "-"]
    assert "--target" not in cmd


def test_a_playing_stream_is_preferred_over_an_idle_one() -> None:
    objects: list[dict[str, object]] = [
        _stream(1, "Discord", "Chromium", state="idle"),
        _stream(2, "Discord", "WEBRTC VoiceEngine"),
    ]
    assert match_app_streams(objects, ["discord"]) == ["2", "1"]


def test_an_app_stream_recorder_is_pinned_to_its_stream() -> None:
    """Regression: an unpinned recorder was moved to the microphone when its stream went away,

    so every request the owner spoke was heard a second time as PC audio.
    """
    cmd = app_record_command("42")
    assert cmd[cmd.index("--target") + 1] == "42"
    properties = cmd[cmd.index("-P") + 1]
    for key in ("node.dont-reconnect", "node.dont-fallback", "node.dont-move"):
        assert f"{key} = true" in properties
    assert "stream.capture.sink" not in properties


def test_the_all_output_recorder_follows_the_default_output() -> None:
    assert app_record_command("") == pw_record_command(capture_output=True)


def _graph(recorder_pid: int, source_serial: int) -> list[dict[str, object]]:
    """A PipeWire graph where process `recorder_pid` captures from node `source_serial`."""
    return [
        {
            "id": 30,
            "type": "PipeWire:Interface:Client",
            "info": {"props": {"pipewire.sec.pid": recorder_pid}},
        },
        {
            "id": 31,
            "type": "PipeWire:Interface:Node",
            "info": {"props": {"client.id": 30, "object.serial": 900, "node.name": "pw-record"}},
        },
        {
            "id": 32,
            "type": "PipeWire:Interface:Node",
            "info": {"props": {"client.id": 5, "object.serial": 901, "node.name": "pw-record"}},
        },
        {"id": 59, "type": "PipeWire:Interface:Node", "info": {"props": {"object.serial": 77}}},
        _stream(source_serial, "Discord", "WEBRTC VoiceEngine"),
        {
            "id": 40,
            "type": "PipeWire:Interface:Link",
            "info": {"output-node-id": source_serial, "input-node-id": 31},
        },
        {
            "id": 41,
            "type": "PipeWire:Interface:Link",
            "info": {"output-node-id": 59, "input-node-id": 32},
        },
        {"id": 42, "type": "PipeWire:Interface:Link", "info": None},
    ]


def test_capture_sources_names_what_feeds_one_process() -> None:
    graph = _graph(recorder_pid=1234, source_serial=42)
    assert capture_sources(graph, 1234) == ["42"]
    assert capture_sources(graph, 999) == []


def _tone(byte: int) -> list[str]:
    """A capture command that writes the same byte forever (exec: stopping it stops the writer)."""
    return ["sh", "-c", f"exec tr '\\0' '\\{byte:03o}' </dev/zero"]


def _value(byte: int) -> int:
    return byte * 257


def _frames_until(frames: Iterator[NDArray[np.int16]], value: int, limit_s: float = 5.0) -> bool:
    """Read frames until one holds `value`; False if none came within `limit_s`."""
    found = threading.Event()

    def read() -> None:
        for frame in frames:
            if int(frame[0]) == value:
                found.set()
                return

    threading.Thread(target=read, daemon=True).start()
    return found.wait(limit_s)


def test_app_audio_drops_a_recorder_that_another_source_feeds() -> None:
    """Regression: PipeWire re-linked the recorder to the microphone; it must never be heard."""
    targets: list[str] = []
    feeds = iter([["42"], ["7"]])

    def command_for(target: str) -> list[str]:
        targets.append(target)
        return _tone(len(targets))

    audio = AppAudio(
        ["discord"],
        find=lambda _apps: ["42"],
        poll_s=0.01,
        command_for=command_for,
        feeds=lambda _pid: next(feeds, ["42"]),
    )
    frames = audio.frames()
    assert _frames_until(frames, _value(1))
    switched = _frames_until(frames, _value(2))
    audio.stop()

    assert switched
    assert targets == ["42", "42"]


def test_app_audio_drops_a_recorder_whose_stream_is_gone() -> None:
    targets: list[str] = []
    finds = iter([["42"], ["42"], []])

    def command_for(target: str) -> list[str]:
        targets.append(target)
        return _tone(len(targets))

    audio = AppAudio(
        ["discord"],
        find=lambda _apps: next(finds, ["43"]),
        poll_s=0.01,
        command_for=command_for,
        feeds=lambda _pid: [],
    )
    frames = audio.frames()
    assert _frames_until(frames, _value(1))
    switched = _frames_until(frames, _value(2))
    audio.stop()

    assert switched
    assert targets == ["42", "43"]


def test_app_audio_keeps_waiting_when_the_recorder_cannot_start() -> None:
    commands = iter([["/nonexistent/pw-record"]])

    audio = AppAudio(
        ["discord"],
        find=lambda _apps: ["42"],
        poll_s=0.01,
        command_for=lambda _target: next(commands, _tone(2)),
        feeds=lambda _pid: ["42"],
    )
    heard = _frames_until(audio.frames(), _value(2))
    audio.stop()

    assert heard


def test_a_missing_capture_tool_is_an_audio_error() -> None:
    mic = Microphone(command=["/nonexistent/pw-record"])
    with pytest.raises(AudioError):
        mic.start()


def test_a_missing_graph_tool_means_no_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    assert app_stream_serials(["discord"]) == []


def test_a_full_buffer_survives_a_concurrent_drain() -> None:
    """Regression: a drain between "full" and "drop the oldest" killed the reader thread."""

    class DrainedMeanwhile(queue.Queue["NDArray[np.int16]"]):
        def __init__(self) -> None:
            super().__init__()
            self.refused = False

        @override
        def put_nowait(self, item: NDArray[np.int16]) -> None:
            if not self.refused:
                self.refused = True
                raise queue.Full
            super().put_nowait(item)

    buffer = DrainedMeanwhile()
    audio_module.push_frame(buffer, np.zeros(FRAME, dtype=np.int16))

    assert buffer.qsize() == 1


def test_stop_ends_a_capture_tool_that_ignores_the_polite_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_module, "STOP_WAIT_S", 0.2)
    ignores_term = f"trap '' TERM; head -c {FRAME * 2} /dev/zero; while :; do sleep 0.05; done"
    mic = Microphone(command=["sh", "-c", ignores_term])
    mic.start()
    pid = mic.pid
    assert pid is not None
    try:
        next(mic.frames())  # the frame is written after the trap is set

        mic.stop()

        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        # A stop() that fails must not leave the looping child behind.
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


SILENCE = np.zeros(FRAME, dtype=np.int16)


def _voxtype(
    stdout: bytes = b"", returncode: int = 0, error: Exception | None = None
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    def run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if error is not None:
            raise error
        return subprocess.CompletedProcess(cmd, returncode, stdout, b"model missing")

    return run


def test_transcribe_returns_the_text_voxtype_printed() -> None:
    assert transcribe(SILENCE, run=_voxtype(b"Loading audio file: x\nPause the music.\n")) == (
        "Pause the music."
    )


def test_text_that_is_not_valid_utf8_is_kept_readable() -> None:
    assert "play" in transcribe(SILENCE, run=_voxtype(b"\xff\xfe play\n"))


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("voxtype"),
        PermissionError("voxtype"),
        ValueError("embedded null byte"),
        subprocess.TimeoutExpired("voxtype", 60),
    ],
    ids=["missing", "not allowed", "bad argument", "timed out"],
)
def test_a_transcription_that_cannot_run_is_an_audio_error(failure: Exception) -> None:
    with pytest.raises(AudioError):
        transcribe(SILENCE, run=_voxtype(error=failure))


def test_a_failed_transcription_is_an_audio_error() -> None:
    with pytest.raises(AudioError):
        transcribe(SILENCE, run=_voxtype(returncode=1))


def test_a_hint_no_command_line_can_carry_is_an_audio_error() -> None:
    # A NUL in an argument is refused by the operating system before Voxtype ever starts.
    with pytest.raises(AudioError):
        transcribe(SILENCE, prompt="Glass\x00Harbor")


VOICE = np.full(2400, 1000, dtype=np.int16).tobytes()  # 0.1 s of sound (silence is trimmed)


def _speaker(player: list[str], grace_s: float = 10.0) -> Speaker:
    return Speaker(lambda _text: (VOICE, 24000), player=lambda _rate: player, grace_s=grace_s)


def _finishes(action: threading.Thread, seconds: float) -> bool:
    action.join(seconds)
    return not action.is_alive()


def _endless_player(pid_file: Path) -> list[str]:
    """A player that takes the sound and then never exits; its process ID is left in `pid_file`."""
    return ["sh", "-c", f"echo $$ > {pid_file}; cat > /dev/null; exec sleep 600"]


def _end(pid_file: Path) -> None:
    """Kill the endless player whatever the speaker did, so a failing test leaves nothing behind."""
    with contextlib.suppress(OSError, ValueError):
        os.kill(int(pid_file.read_text(encoding="ascii")), signal.SIGKILL)


def test_speech_goes_to_the_player_and_returns_when_it_is_done(tmp_path: Path) -> None:
    heard = tmp_path / "heard.raw"

    _speaker(["sh", "-c", f"cat > {heard}"]).say("hello")

    assert heard.stat().st_size == 4800


def test_a_missing_player_is_an_audio_error() -> None:
    with pytest.raises(AudioError):
        _speaker(["/nonexistent/pw-play"]).say("hello")


def test_a_player_that_never_finishes_is_ended_and_reported(tmp_path: Path) -> None:
    failures: list[BaseException] = []
    speaker = _speaker(_endless_player(tmp_path / "pid"), grace_s=0.2)

    def say() -> None:
        try:
            speaker.say("hello")
        except AudioError as exc:
            failures.append(exc)

    talking = threading.Thread(target=say, daemon=True)
    talking.start()
    try:
        assert _finishes(talking, 5)
        assert len(failures) == 1
    finally:
        _end(tmp_path / "pid")


def test_stop_cuts_speech_short(tmp_path: Path) -> None:
    speaker = _speaker(_endless_player(tmp_path / "pid"))
    talking = threading.Thread(target=speaker.say, args=("hello",), daemon=True)
    talking.start()
    try:
        stopped = False
        for _ in range(100):
            speaker.stop()
            if _finishes(talking, 0.05):
                stopped = True
                break
        assert stopped
    finally:
        _end(tmp_path / "pid")


def test_a_stop_while_the_player_is_starting_still_ends_it(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    speakers: list[Speaker] = []

    def player(_rate: int) -> list[str]:
        speakers[0].stop()  # the stop lands before the player is known to the speaker
        return _endless_player(pid_file)

    speaker = Speaker(lambda _text: (VOICE, 24000), player=player)
    speakers.append(speaker)
    talking = threading.Thread(target=speaker.say, args=("hello",), daemon=True)
    talking.start()
    try:
        assert _finishes(talking, 5)
    finally:
        _end(pid_file)


def test_a_stopped_speaker_says_nothing_more(tmp_path: Path) -> None:
    heard = tmp_path / "heard.raw"
    speaker = _speaker(["sh", "-c", f"cat > {heard}"])

    speaker.stop()
    speaker.say("hello")

    assert not heard.exists()


def test_stop_with_nothing_playing_is_harmless() -> None:
    _speaker(["true"]).stop()


def test_voxtype_command_reads_jarvis_settings_first() -> None:
    cmd = voxtype_command(Path("/x.wav"), model="small.en", settings=Path("/s/voxtype.toml"))
    assert cmd == [
        "voxtype",
        "-c",
        "/s/voxtype.toml",
        "--model=small.en",
        "transcribe",
        "/x.wav",
    ]


@pytest.mark.parametrize(("threads", "short"), [(8, True), (2, False)])
def test_voxtype_settings_hold_the_speed_options(threads: int, short: bool) -> None:
    settings = tomllib.loads(voxtype_settings(threads, short_window=short))
    assert settings == {"whisper": {"threads": threads, "context_window_optimization": short}}


def test_transcribe_passes_jarvis_settings_to_voxtype() -> None:
    seen: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"Pause.\n", b"")

    transcribe(SILENCE, settings=Path("/s/voxtype.toml"), run=run)

    assert seen[0][1:3] == ["-c", "/s/voxtype.toml"]


class _FakePiperVoice:
    """Stands in for `piper.PiperVoice`: notes the synthesis settings it is given."""

    loaded: ClassVar[list[_FakePiperVoice]] = []

    def __init__(self, length_scale: float) -> None:
        super().__init__()
        self.config = SimpleNamespace(length_scale=length_scale)
        self.settings: list[object] = []

    @classmethod
    def load(cls, _path: Path) -> _FakePiperVoice:
        voice = cls(1.2)
        cls.loaded.append(voice)
        return voice

    def synthesize(self, _text: str, syn_config: object = None) -> list[object]:
        self.settings.append(syn_config)
        return [SimpleNamespace(sample_rate=22050, audio_int16_bytes=b"\x00\x00")]


def test_a_piper_voice_speaks_at_the_configured_speed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import piper  # ruff: ignore[import-outside-top-level] - only this test needs it

    _FakePiperVoice.loaded.clear()
    monkeypatch.setattr(piper, "PiperVoice", _FakePiperVoice)

    speak = audio_module.make_synthesizer(tmp_path, "en_GB-alan-medium", 2.0)
    assert speak("Low Tide") == (b"\x00\x00", 22050)

    (setting,) = _FakePiperVoice.loaded[0].settings
    assert getattr(setting, "length_scale", None) == pytest.approx(0.6)


def _status(
    stdout: bytes, returncode: int = 0
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    """A `voxtype status` that prints `stdout`, recording the command it was given."""

    def run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert cmd == ["voxtype", "status", "--format", "json"]
        return subprocess.CompletedProcess(cmd, returncode, stdout, b"")

    return run


@pytest.mark.parametrize(
    ("state", "dictating"),
    [
        ("idle", False),
        ("stopped", False),
        ("recording", True),
        ("streaming", True),
        ("transcribing", True),
        ("a-state-added-later", True),
    ],
)
def test_dictation_is_read_from_voxtypes_status(state: str, dictating: bool) -> None:
    out = f'{{"text": "x", "alt": "{state}", "class": "{state}", "tooltip": "t"}}'.encode()
    assert dictation_active(run=_status(out)) is dictating


def test_no_daemon_means_no_dictation() -> None:
    assert dictation_active(run=_status(b"", returncode=1)) is False


@pytest.mark.parametrize("stdout", [b"", b"not json", b"[]", b'{"alt": 3}', b"{}"])
def test_a_status_that_cannot_be_read_counts_as_dictating(stdout: bytes) -> None:
    assert dictation_active(run=_status(stdout)) is True


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("voxtype"),
        PermissionError("voxtype"),
        subprocess.TimeoutExpired("voxtype", 2),
    ],
    ids=["missing", "not allowed", "timed out"],
)
def test_a_status_that_cannot_run_counts_as_dictating(failure: Exception) -> None:
    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise failure

    assert dictation_active(run=run) is True


def test_a_microphone_started_again_after_a_stop_delivers_frames() -> None:
    # What `jarvis pause` and `resume` do: the stopped capture's end must not end the new one.
    one_frame_then_wait = f"head -c {FRAME * 2} /dev/zero; exec sleep 30"
    mic = Microphone(command=["sh", "-c", one_frame_then_wait])
    mic.start()
    try:
        next(mic.frames())
        mic.stop()
        mic.start()

        assert _first_outcome(mic, timeout=5) == "frame"
        assert _first_outcome(mic, timeout=0.3) == "blocked"
    finally:
        mic.stop()
