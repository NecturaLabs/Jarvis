"""Microphone stream handling and transcript parsing, without real audio devices."""

import contextlib
import json
import logging
import os
import queue
import signal
import struct
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
import pytest

from jarvis import audio as audio_module
from jarvis.audio import (
    FRAME,
    VAD_CHUNK,
    VAD_CONTEXT,
    VOXTYPE_FOLLOW,
    AppAudio,
    AudioError,
    Microphone,
    Speaker,
    Vad,
    VoxtypeState,
    VoxtypeWatch,
    WakeWord,
    app_record_command,
    app_stream_serials,
    capture_sources,
    gguf_architecture,
    installed_models,
    is_phantom,
    match_app_streams,
    next_retry_wait,
    others_playing,
    parse_transcript,
    parse_voxtype_status,
    playing_app_streams,
    pw_record_command,
    screen_locked,
    source_found,
    to_16k,
    transcribe,
    voxtype_command,
    voxtype_settings,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from numpy.typing import NDArray


@pytest.fixture
def ended_microphone() -> Iterator[Callable[[int], Microphone]]:
    """Make microphones fed by a process that writes `frames` silent frames, then exits.

    Each is stopped when the test ends, which reaps its process and closes its pipe.
    """
    made: list[Microphone] = []

    def make(frames: int) -> Microphone:
        mic = Microphone(command=["head", "-c", str(FRAME * 2 * frames), "/dev/zero"])
        made.append(mic)
        mic.start()
        mic.wait_ended(timeout=5)
        return mic

    yield make
    for mic in made:
        mic.stop()


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


def test_frames_end_after_drain_instead_of_blocking(
    ended_microphone: Callable[[int], Microphone],
) -> None:
    """Regression: drain() used to discard the end-of-stream marker, so a stop hung forever."""
    mic = ended_microphone(2)
    mic.drain()
    assert _first_outcome(mic, timeout=2) == "ended"


def test_frames_are_delivered_before_the_end(ended_microphone: Callable[[int], Microphone]) -> None:
    mic = ended_microphone(2)
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


def _sine(hz: float, rate: int, *, level: float = 8000.0) -> bytes:
    """One second of a sine at `hz` as int16 PCM."""
    t = np.arange(rate) / rate
    return (level * np.sin(2 * np.pi * hz * t)).astype(np.int16).tobytes()


def _rms(samples: NDArray[np.int16]) -> float:
    """Return the RMS of the middle of `samples`, away from the filter's start and end."""
    middle = samples[samples.size // 4 : -samples.size // 4].astype(np.float64)
    return float(np.sqrt(np.mean(middle**2)))


@pytest.mark.parametrize("rate", [24000, 22050])
@pytest.mark.parametrize("hz", [8500, 9000, 10000, 11000])
def test_a_tone_above_the_16k_band_is_cut_instead_of_folded_into_it(rate: int, hz: int) -> None:
    """Regression: plain interpolation folded a 10 kHz tone down to 6 kHz at full strength."""
    heard = to_16k(_sine(hz, rate), rate)

    assert _rms(heard) < 0.01 * _rms(np.frombuffer(_sine(hz, rate), dtype=np.int16))


@pytest.mark.parametrize("rate", [24000, 22050])
@pytest.mark.parametrize("hz", [300, 1000, 3000, 6000])
def test_speech_band_tones_keep_their_level_through_the_resampling(rate: int, hz: int) -> None:
    heard = to_16k(_sine(hz, rate), rate)

    assert _rms(heard) == pytest.approx(8000 / np.sqrt(2), rel=0.03)


def test_the_resampled_tone_keeps_its_frequency() -> None:
    heard = to_16k(_sine(1000, 24000), 24000).astype(np.float64)

    spectrum = np.abs(np.fft.rfft(heard))
    assert np.argmax(spectrum) * 16000 / heard.size == pytest.approx(1000, abs=2)


def test_16k_audio_and_nothing_pass_through_untouched() -> None:
    pcm = _sine(500, 16000)

    assert to_16k(pcm, 16000).tobytes() == pcm
    assert to_16k(b"", 24000).size == 0


def test_voxtype_command_passes_overrides_before_the_subcommand() -> None:
    cmd = voxtype_command(
        Path("/x.wav"), model="large-v3-turbo", prompt="Glass Harbor, Night Circuit"
    )
    assert cmd == [
        "voxtype",
        "-q",
        "--model=large-v3-turbo",
        "--initial-prompt=Glass Harbor, Night Circuit",
        "transcribe",
        "/x.wav",
    ]


def test_a_hint_that_starts_with_a_dash_stays_the_hint() -> None:
    cmd = voxtype_command(Path("/x.wav"), engine="-v", model="-c", prompt="--help, Low Tide")

    assert cmd[2:5] == ["--model=-c", "--engine=-v", "--initial-prompt=--help, Low Tide"]


def test_voxtype_command_without_overrides_uses_the_owners_config() -> None:
    assert voxtype_command(Path("/x.wav")) == ["voxtype", "-q", "transcribe", "/x.wav"]


def test_voxtype_is_asked_for_errors_only_so_its_log_lines_stay_off_the_transcript() -> None:
    """`-q` (Voxtype's documented quiet flag) leaves out the tracing lines on stdout."""
    assert voxtype_command(Path("/x.wav"), model="small.en")[:2] == ["voxtype", "-q"]


def _stream(
    serial: int,
    binary: str,
    app: str,
    media_class: str = "Stream/Output/Audio",
    state: str = "running",
    *,
    pid: int | None = None,
) -> dict[str, object]:
    props: dict[str, object] = {
        "media.class": media_class,
        "object.serial": serial,
        "application.process.binary": binary,
        "application.name": app,
        "node.name": app,
    }
    if pid is not None:
        props["application.process.id"] = str(pid)
    return {
        "id": serial,
        "type": "PipeWire:Interface:Node",
        "info": {"state": state, "props": props},
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


def test_audio_playing_elsewhere_is_any_running_output_stream_but_jarvis_own() -> None:
    def ours(pid: int) -> bool:
        return pid == 7

    jarvis = _stream(1, "pw-play", "pw-play", pid=7)
    spotify = _stream(2, "spotify", "Spotify", pid=50)
    paused = _stream(3, "spotify", "Spotify", state="idle", pid=50)
    mic = _stream(4, "Discord", "mic", media_class="Stream/Input/Audio", pid=60)

    assert others_playing([jarvis, spotify], ours)
    assert not others_playing([jarvis, paused, mic, {"id": 5, "info": None}], ours)
    assert others_playing([_stream(6, "mpv", "mpv")], ours)  # no process id: not Jarvis's


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


def _microphones() -> list[dict[str, object]]:
    """What `pw-dump` lists for two microphones, a monitor of an output and a playback stream."""

    def node(item_id: int, media_class: str, name: str) -> dict[str, object]:
        props = {"media.class": media_class, "node.name": name, "object.serial": 100 + item_id}
        return {"id": item_id, "type": "PipeWire:Interface:Node", "info": {"props": props}}

    return [
        node(10, "Audio/Source", "alsa_input.usb-Example_Mic-00.analog-stereo"),
        node(11, "Audio/Source/Virtual", "effect_input.denoise"),
        node(12, "Audio/Sink", "alsa_output.pci-0000_00_1f.3.analog-stereo"),
        node(13, "Stream/Output/Audio", "Discord"),
        {"id": 14, "type": "PipeWire:Interface:Link", "info": None},
    ]


@pytest.mark.parametrize(
    "source",
    [
        "alsa_input.usb-Example_Mic-00.analog-stereo",  # a node name
        "110",  # an object serial
        "effect_input.denoise",  # a virtual microphone
    ],
)
def test_a_source_that_is_a_microphone_node_is_found(source: str) -> None:
    assert source_found(_microphones(), source) is True


@pytest.mark.parametrize(
    "source",
    [
        "alsa_input.usb-Typo_Mic-00.analog-stereo",  # no such node
        "alsa_output.pci-0000_00_1f.3.analog-stereo",  # a speaker, not a microphone
        "Discord",  # a playback stream
        "112",  # the serial of a speaker
        "alsa_input.usb-Example",  # only part of a name
    ],
)
def test_a_source_that_matches_no_microphone_node_is_not_found(source: str) -> None:
    assert source_found(_microphones(), source) is False


def test_nothing_is_found_in_a_graph_that_could_not_be_read() -> None:
    assert source_found([], "alsa_input.usb-Example_Mic-00.analog-stereo") is False
    assert source_found({"unexpected": "shape"}, "110") is False


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


def test_app_audio_hears_every_stream_the_app_plays_mixed() -> None:
    targets: list[str] = []

    def command_for(target: str) -> list[str]:
        targets.append(target)
        return _tone({"42": 1, "43": 2}[target])

    audio = AppAudio(
        ["discord"],
        find=lambda _apps: ["42", "43"],
        poll_s=0.01,
        command_for=command_for,
        feeds=lambda _pid: [],
    )
    heard = _frames_until(audio.frames(), _value(1) + _value(2))
    audio.stop()

    assert heard
    assert sorted(targets[:2]) == ["42", "43"]


def test_app_audio_takes_in_a_stream_the_app_starts_while_it_records() -> None:
    targets: list[str] = []
    finds = iter([["42"], ["42"]])

    def command_for(target: str) -> list[str]:
        targets.append(target)
        return _tone({"42": 1, "43": 2}[target])

    audio = AppAudio(
        ["discord"],
        find=lambda _apps: next(finds, ["42", "43"]),
        poll_s=0.01,
        command_for=command_for,
        feeds=lambda _pid: [],
    )
    heard = _frames_until(audio.frames(), _value(1) + _value(2))
    audio.stop()

    assert heard
    assert targets[0] == "42"


def test_every_playing_stream_of_the_apps_is_followed_or_else_one_idle_one() -> None:
    playing = [
        _stream(1, "Discord", "Chromium", state="idle"),
        _stream(2, "Discord", "WEBRTC VoiceEngine"),
        _stream(3, "Discord", "Notifications"),
    ]
    idle = [
        _stream(1, "Discord", "Chromium", state="idle"),
        _stream(4, "Discord", "x", state="idle"),
    ]

    assert playing_app_streams(playing, ["discord"]) == ["2", "3"]
    assert playing_app_streams(idle, ["discord"]) == ["1"]
    assert playing_app_streams(playing, ["spotify"]) == []


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
        "-q",
        "-c",
        "/s/voxtype.toml",
        "--model=small.en",
        "transcribe",
        "/x.wav",
    ]


@pytest.mark.parametrize(("threads", "short"), [(8, True), (2, False)])
def test_voxtype_settings_hold_the_speed_options(threads: int, short: bool) -> None:
    settings = tomllib.loads(voxtype_settings(threads, short_window=short))
    assert settings["whisper"] == {"threads": threads, "context_window_optimization": short}


def _gguf(path: Path, architecture: str, *, before: int = 0) -> Path:
    """Write the start of a GGUF file: its header and metadata, `before` other keys first."""

    def text(value: str) -> bytes:
        return struct.pack("<Q", len(value.encode())) + value.encode()

    keys = b"".join(
        text(f"other.{n}") + struct.pack("<I", 9) + struct.pack("<IQ", 8, 2) + text("a") + text("b")
        for n in range(before)
    )
    keys += text("general.architecture") + struct.pack("<I", 8) + text(architecture)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, before + 1) + keys + b"\0" * 64)
    return path


@pytest.mark.parametrize("engine", ["parakeet", "cohere", "gguf"])
def test_a_model_file_for_a_gguf_engine_goes_in_jarvis_settings(
    engine: str, tmp_path: Path
) -> None:
    """Voxtype reads a Parakeet, Cohere or GGUF checkpoint from its config, not from `--model`."""
    model = str(_gguf(tmp_path / 'a "quoted" name.gguf', "qwen3_asr"))
    settings = tomllib.loads(voxtype_settings(8, short_window=True, engine=engine, model=model))

    assert settings[engine]["model"] == model
    cmd = voxtype_command(Path("/x.wav"), engine, model, settings=Path("/s/voxtype.toml"))
    assert not any(part.startswith("--model") for part in cmd)
    assert f"--engine={engine}" in cmd


def test_the_gguf_engine_is_told_the_architecture_its_file_names(tmp_path: Path) -> None:
    """Voxtype refuses a GGUF checkpoint whose architecture it was not told."""
    model = _gguf(tmp_path / "q.gguf", "qwen3_asr", before=2)

    settings = tomllib.loads(
        voxtype_settings(8, short_window=True, engine="gguf", model=str(model))
    )

    assert settings["gguf"] == {"model": str(model), "architecture": "qwen3_asr"}


@pytest.mark.parametrize("content", [b"", b"not a gguf file", b"GGUF\x03\x00"])
def test_a_file_that_is_no_gguf_checkpoint_names_no_architecture(
    content: bytes, tmp_path: Path
) -> None:
    model = tmp_path / "bad.gguf"
    model.write_bytes(content)

    assert not gguf_architecture(model)
    assert not gguf_architecture(tmp_path / "missing.gguf")


def _key(name: str, kind: int, value: bytes) -> bytes:
    return struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", kind) + value


def _checkpoint(path: Path, *keys: bytes) -> Path:
    """Write a GGUF header holding `keys`, then the architecture key."""
    architecture = _key("general.architecture", 8, struct.pack("<Q", 9) + b"qwen3_asr")
    count = len(keys) + 1
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, count) + b"".join(keys) + architecture)
    return path


def test_a_value_of_an_unknown_kind_names_no_architecture(tmp_path: Path) -> None:
    """A kind no GGUF version defines means the file is not one Jarvis can read; no crash."""
    model = _checkpoint(tmp_path / "odd.gguf", _key("other", 99, b"\0" * 8))

    assert not gguf_architecture(model)


def test_deeply_nested_arrays_are_read_without_recursion(tmp_path: Path) -> None:
    depth = 5000
    nested = struct.pack("<IQ", 9, 1) * (depth - 1) + struct.pack("<IQ", 0, 1) + b"\0"
    model = _checkpoint(tmp_path / "deep.gguf", _key("other", 9, nested))

    assert gguf_architecture(model) == "qwen3_asr"


def test_metadata_past_the_read_budget_is_not_looked_for(tmp_path: Path) -> None:
    """Only the start of a file is read, so a huge or hostile one costs little."""
    items = 300_000  # each 9 bytes: past the budget, far short of any real checkpoint's tensors
    strings = struct.pack("<IQ", 8, items) + (struct.pack("<Q", 1) + b"a") * items
    model = _checkpoint(tmp_path / "big.gguf", _key("other", 9, strings))

    assert not gguf_architecture(model)


@pytest.mark.parametrize("count", [1 << 40, (1 << 64) - 1])
def test_an_array_longer_than_the_file_names_no_architecture(count: int, tmp_path: Path) -> None:
    model = _checkpoint(tmp_path / "long.gguf", _key("other", 9, struct.pack("<IQ", 4, count)))

    assert not gguf_architecture(model)


def test_a_model_path_that_is_no_regular_file_is_not_opened_for_reading(tmp_path: Path) -> None:
    """A named pipe would block the read until something wrote to it."""
    fifo = tmp_path / "pipe.gguf"
    os.mkfifo(fifo)
    found: list[str] = []
    reader = threading.Thread(target=lambda: found.append(gguf_architecture(fifo)), daemon=True)

    reader.start()
    reader.join(timeout=10)
    blocked = reader.is_alive()
    if blocked:  # release the blocked open so the thread ends
        with fifo.open("wb"):
            pass
        reader.join()

    assert not blocked
    assert found == [""]
    assert not gguf_architecture(tmp_path)
    assert not gguf_architecture(tmp_path / "a\0b.gguf")  # a config can hold a NUL


def test_an_unreadable_gguf_checkpoint_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    model = tmp_path / "bad.gguf"
    model.write_bytes(b"not a gguf file")

    with caplog.at_level(logging.WARNING, logger="jarvis.audio"):
        settings = tomllib.loads(
            voxtype_settings(8, short_window=True, engine="gguf", model=str(model))
        )

    assert settings["gguf"] == {"model": str(model)}
    assert any(str(model) in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "name", ["🎤 model.gguf", 'a "quoted" \\ name.gguf', "tab\tand\x7fdel\x01.gguf", "日本.gguf"]
)
def test_any_utf8_model_path_reaches_voxtype_unchanged(name: str) -> None:
    model = f"/m/{name}"

    settings = tomllib.loads(voxtype_settings(8, short_window=True, engine="parakeet", model=model))

    assert settings["parakeet"] == {"model": model}


def test_a_model_path_that_is_no_utf8_text_is_refused() -> None:
    """Voxtype's config is UTF-8; a path with bytes that are not cannot be written in it."""
    with pytest.raises(ValueError, match="UTF-8"):
        voxtype_settings(8, short_window=True, engine="parakeet", model="/m/\udcff.gguf")


@pytest.mark.parametrize(
    ("engine", "model"), [("", "small.en"), ("whisper", "/m/ggml-small.en.bin"), ("parakeet", "")]
)
def test_other_models_are_still_named_on_the_command_line(engine: str, model: str) -> None:
    settings = tomllib.loads(voxtype_settings(8, short_window=True, engine=engine, model=model))
    cmd = voxtype_command(Path("/x.wav"), engine, model)

    assert set(settings) == {"audio", "whisper"}
    assert (f"--model={model}" in cmd) is bool(model)


def test_voxtype_keeps_no_audio_of_a_failed_transcription_for_jarvis() -> None:
    """Voxtype keeps the audio of a failed transcription by default; Jarvis has none to keep."""
    settings = tomllib.loads(voxtype_settings(4, short_window=True))

    assert settings["audio"] == {"keep_failed_recordings": False}


def test_transcribe_passes_jarvis_settings_to_voxtype() -> None:
    seen: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"Pause.\n", b"")

    transcribe(SILENCE, settings=Path("/s/voxtype.toml"), run=run)

    assert seen[0][2:4] == ["-c", "/s/voxtype.toml"]


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
    import piper

    _FakePiperVoice.loaded.clear()
    monkeypatch.setattr(piper, "PiperVoice", _FakePiperVoice)

    speak = audio_module.make_synthesizer(tmp_path, "en_GB-alan-medium", 2.0)
    assert speak("Low Tide") == (b"\x00\x00", 22050)

    (setting,) = _FakePiperVoice.loaded[0].settings
    assert getattr(setting, "length_scale", None) == pytest.approx(0.6)


@pytest.mark.parametrize("phoneme_map", [False, True])
def test_a_kokoro_voice_is_given_the_phoneme_map_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, phoneme_map: bool
) -> None:
    from jarvis import kokoro

    made: list[dict[str, object]] = []

    class Kokoro:
        def __init__(self, *_args: object, **options: object) -> None:
            super().__init__()
            made.append(options)

        @staticmethod
        def synthesize(_text: str) -> NDArray[np.int16]:
            return np.zeros(0, dtype=np.int16)

    monkeypatch.setattr(kokoro, "Kokoro", Kokoro)

    audio_module.make_synthesizer(tmp_path, "kokoro:bm_george", 1.0, phoneme_map=phoneme_map)

    assert made == [{"speed": 1.0, "phoneme_map": phoneme_map}]


def _status_line(state: str, meeting: str | None = "idle") -> str:
    """One line of `voxtype status --follow --format json --extended`."""
    fields: dict[str, object] = {"text": "x", "alt": state, "class": state, "tooltip": "t"}
    if meeting is not None:
        fields.update(meeting_active=meeting != "idle", meeting_status=meeting)
    return json.dumps(fields)


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
    parsed = parse_voxtype_status(_status_line(state).encode())

    assert parsed is not None
    assert parsed.dictating is dictating
    assert parsed.meeting is False


@pytest.mark.parametrize(
    ("meeting", "running"),
    [("idle", False), ("recording", True), ("a-status-added-later", True), (None, False)],
)
def test_a_meeting_is_any_status_but_idle(meeting: str | None, running: bool) -> None:
    parsed = parse_voxtype_status(_status_line("idle", meeting).encode())

    assert parsed == VoxtypeState(dictating=False, meeting=running)


def test_a_meeting_is_seen_from_its_active_flag_alone() -> None:
    parsed = parse_voxtype_status(b'{"class": "idle", "meeting_active": true}')

    assert parsed == VoxtypeState(dictating=False, meeting=True)


def test_a_meeting_is_seen_from_its_status_alone() -> None:
    parsed = parse_voxtype_status(b'{"class": "idle", "meeting_status": "recording"}')

    assert parsed == VoxtypeState(dictating=False, meeting=True)


@pytest.mark.parametrize("stdout", [b"", b"not json", b"[]", b'{"alt": 3}', b"{}"])
def test_a_status_that_cannot_be_read_is_unknown(stdout: bytes) -> None:
    assert parse_voxtype_status(stdout) is None


def test_the_status_is_followed_with_one_long_lived_command() -> None:
    assert VOXTYPE_FOLLOW == ("voxtype", "status", "--follow", "--format", "json", "--extended")


def _status_helper(*steps: str) -> list[str]:
    """A stand-in for `voxtype status --follow`: each step is a line to print or `wait:<file>`."""
    body = ["import os, sys, time"]
    for step in steps:
        if step.startswith("wait:"):
            body.append(f"while not os.path.exists({step[5:]!r}): time.sleep(0.005)")
        elif step.startswith("exit"):
            body.append(f"sys.exit({step[4:] or 1})")
        elif step == "sleep":
            body.append("time.sleep(60)")
        else:
            body.append(f"print({step!r}, flush=True)")
    return [sys.executable, "-I", "-c", "\n".join(body)]


@pytest.fixture
def watches() -> Iterator[Callable[..., VoxtypeWatch]]:
    """Make watches that are started now and stopped when the test ends."""
    made: list[VoxtypeWatch] = []

    def make(command: list[str], **options: float) -> VoxtypeWatch:
        watch = VoxtypeWatch(command, **options)
        made.append(watch)
        watch.start()
        return watch

    yield make
    for watch in made:
        watch.stop()


def _eventually(check: Callable[[], bool]) -> bool:
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        if check():
            return True
        threading.Event().wait(0.005)
    return False


def test_what_the_follower_prints_is_what_dictating_and_meeting_answer(
    watches: Callable[..., VoxtypeWatch],
) -> None:
    dictation = watches(_status_helper(_status_line("recording"), "sleep"))
    meeting = watches(_status_helper(_status_line("idle", "recording"), "sleep"))

    assert _eventually(lambda: dictation.dictating() is True)
    assert dictation.meeting() is False
    assert _eventually(meeting.meeting)
    assert meeting.dictating() is False


def test_a_status_that_cannot_be_parsed_does_not_leave_the_command_running(
    monkeypatch: pytest.MonkeyPatch,
    started_processes: list[subprocess.Popen[bytes]],
) -> None:
    def broken(_line: bytes) -> VoxtypeState | None:
        msg = "parse failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(audio_module, "parse_voxtype_status", broken)
    monkeypatch.setattr(audio_module, "VOXTYPE_STOP_S", 0.5)
    failures: list[BaseException | None] = []

    def record(args: threading.ExceptHookArgs) -> None:
        failures.append(args.exc_value)

    monkeypatch.setattr(threading, "excepthook", record)
    watch = VoxtypeWatch(_status_helper(_status_line("recording"), "sleep"))
    watch.start()
    try:
        # The follower's thread ends with the error once the command is gone; before, it waited
        # for a command nobody ended.
        assert _eventually(lambda: len(failures) == 1)
        (process,) = started_processes
        assert process.returncode is not None
        assert watch.pid is None
    finally:
        watch.stop()


def test_a_change_the_follower_prints_replaces_the_state(
    watches: Callable[..., VoxtypeWatch], tmp_path: Path
) -> None:
    go = tmp_path / "go"
    watch = watches(
        _status_helper(_status_line("recording"), f"wait:{go}", _status_line("idle"), "sleep")
    )

    assert _eventually(lambda: watch.dictating() is True)
    go.touch()
    assert _eventually(lambda: watch.dictating() is False)


def test_a_slow_status_never_blocks_the_voice_thread(
    watches: Callable[..., VoxtypeWatch], tmp_path: Path
) -> None:
    # The old check ran `voxtype status` on the voice thread and waited for it; a daemon that is
    # slow to answer held the loop (and the microphone's frames) for that long.
    go = tmp_path / "go"
    started = time.perf_counter()
    watch = watches(_status_helper(f"wait:{go}", _status_line("recording"), "sleep"))

    asked = [watch.dictating(), watch.meeting(), watch.dictating()]

    assert time.perf_counter() - started < 1.0
    assert asked == [None, False, None]  # not known yet: it is said, not waited for
    go.touch()
    assert _eventually(lambda: watch.dictating() is True)


def test_a_line_that_cannot_be_read_makes_the_state_unknown_until_the_next_one(
    watches: Callable[..., VoxtypeWatch], tmp_path: Path
) -> None:
    broke, mended = tmp_path / "broke", tmp_path / "mended"
    watch = watches(
        _status_helper(
            _status_line("idle"),
            f"wait:{broke}",
            "garbage",
            f"wait:{mended}",
            _status_line("idle"),
            "sleep",
        )
    )

    assert _eventually(lambda: watch.dictating() is False)
    broke.touch()
    assert _eventually(lambda: watch.dictating() is None)
    mended.touch()
    assert _eventually(lambda: watch.dictating() is False)


def test_a_status_command_that_cannot_start_leaves_the_state_unknown_and_is_tried_again(
    watches: Callable[..., VoxtypeWatch], monkeypatch: pytest.MonkeyPatch
) -> None:
    tried: list[list[str]] = []

    def refuse(command: list[str], **_options: object) -> object:
        tried.append(command)
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(subprocess, "Popen", refuse)

    watch = watches(["voxtype"], retry_s=0.01)

    assert _eventually(lambda: len(tried) >= 3)
    assert watch.dictating() is None


def test_a_status_command_that_exits_is_started_again(
    watches: Callable[..., VoxtypeWatch], tmp_path: Path
) -> None:
    seen = tmp_path / "seen"
    script = (
        "import os, sys, time\n"
        f"if os.path.exists({str(seen)!r}):\n"
        f"    print({_status_line('recording')!r}, flush=True); time.sleep(60)\n"
        f"open({str(seen)!r}, 'w').close(); sys.exit(1)\n"
    )

    watch = watches([sys.executable, "-I", "-c", script], retry_s=0.01)

    # The first run exited with an error; the second reports dictation.
    assert _eventually(lambda: watch.dictating() is True)


def test_a_status_command_that_exits_with_an_error_leaves_the_state_unknown(
    watches: Callable[..., VoxtypeWatch], tmp_path: Path
) -> None:
    # Requirement changed (review of #29): with no daemon, the command prints `stopped` and
    # keeps running, so one that exits is an older Voxtype or a crash, not "no dictation".
    go = tmp_path / "go"
    watch = watches(
        _status_helper(_status_line("recording", "recording"), f"wait:{go}", "exit"), retry_s=60.0
    )

    assert _eventually(lambda: watch.dictating() is True)
    go.touch()
    assert _eventually(lambda: watch.dictating() is None)
    assert watch.meeting() is False


def test_a_status_command_that_ends_quietly_leaves_the_state_unknown(
    watches: Callable[..., VoxtypeWatch], tmp_path: Path
) -> None:
    go = tmp_path / "go"
    watch = watches(_status_helper(_status_line("recording"), f"wait:{go}", "exit0"), retry_s=60.0)

    assert _eventually(lambda: watch.dictating() is True)
    go.touch()
    assert _eventually(lambda: watch.dictating() is None)


def test_a_status_line_that_cannot_be_read_is_said_in_the_log_once(
    watches: Callable[..., VoxtypeWatch],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    one, two = tmp_path / "one", tmp_path / "two"
    watch = watches(
        _status_helper(
            "garbage", f"wait:{one}", "more garbage", f"wait:{two}", _status_line("idle"), "sleep"
        )
    )

    assert _eventually(lambda: watch.dictating() is None and bool(caplog.records))
    one.touch()
    two.touch()
    assert _eventually(lambda: watch.dictating() is False)  # both lines were read by then
    warned = [r for r in caplog.records if "could not read Voxtype's status" in r.getMessage()]
    assert len(warned) == 1


def test_a_status_command_that_cannot_start_is_not_tried_again_at_once(
    watches: Callable[..., VoxtypeWatch], monkeypatch: pytest.MonkeyPatch
) -> None:
    tried: list[list[str]] = []

    def refuse(command: list[str], **_options: object) -> object:
        tried.append(command)
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(subprocess, "Popen", refuse)

    watches(["voxtype"], retry_s=60.0)

    assert _eventually(lambda: len(tried) == 1)
    threading.Event().wait(0.2)  # a retry with no wait would have tried hundreds of times
    assert len(tried) == 1


@pytest.mark.parametrize(
    ("wait", "ran_s", "next_wait"),
    [(1.0, 60.0, 1.0), (8.0, 30.0, 1.0), (1.0, 0.0, 2.0), (2.0, 0.5, 4.0), (20.0, 0.0, 30.0)],
)
def test_the_wait_before_the_status_command_is_tried_again_doubles_up_to_a_limit(
    wait: float, ran_s: float, next_wait: float
) -> None:
    assert next_retry_wait(wait, ran_s=ran_s, first=1.0, longest=30.0) == next_wait


def test_a_status_command_is_started_again_at_once_after_a_healthy_run_ends(
    watches: Callable[..., VoxtypeWatch], tmp_path: Path
) -> None:
    # Six quick failures raise the wait to its 0.5 s limit; a run that lasted that long was
    # healthy, so the next one starts after the first wait, not the raised one.
    runs, ended, started = tmp_path / "runs", tmp_path / "ended", tmp_path / "started"
    script = (
        "import pathlib, sys, time\n"
        f"runs = pathlib.Path({str(runs)!r})\n"
        "n = len(runs.read_text()) if runs.exists() else 0\n"
        "runs.write_text('x' * (n + 1))\n"
        "if n < 6: sys.exit(1)\n"
        "if n == 6:\n"
        "    time.sleep(0.6)\n"
        f"    pathlib.Path({str(ended)!r}).write_text(repr(time.time())); sys.exit(1)\n"
        f"pathlib.Path({str(started)!r}).write_text(repr(time.time()))\n"
        f"print({_status_line('recording')!r}, flush=True); time.sleep(60)\n"
    )

    watch = watches([sys.executable, "-I", "-c", script], retry_s=0.01, max_retry_s=0.5)

    assert _eventually(lambda: watch.dictating() is True)
    assert float(started.read_text()) - float(ended.read_text()) < 0.3


def test_a_status_command_that_prints_once_and_dies_is_not_restarted_every_second() -> None:
    # Requirement changed (review of #29): printing a line no longer resets the wait.
    wait = 1.0
    for _ in range(6):
        wait = next_retry_wait(wait, ran_s=0.2, first=1.0, longest=30.0)

    assert wait == pytest.approx(30.0)


def test_stopping_the_watch_ends_the_status_command_it_started(
    watches: Callable[..., VoxtypeWatch],
) -> None:
    watch = watches(_status_helper(_status_line("idle"), "sleep"))
    assert _eventually(lambda: watch.dictating() is False)
    pid = watch.pid
    assert pid is not None

    watch.stop()

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert watch.pid is None


def test_stopping_the_watch_closes_the_pipe_it_read(
    watches: Callable[..., VoxtypeWatch], started_processes: list[subprocess.Popen[bytes]]
) -> None:
    watch = watches(_status_helper(_status_line("idle"), "sleep"))
    assert _eventually(lambda: watch.dictating() is False)

    watch.stop()

    (process,) = started_processes
    assert process.stdout is not None
    assert process.stdout.closed


def test_a_status_command_that_ends_has_its_pipe_closed(
    watches: Callable[..., VoxtypeWatch], started_processes: list[subprocess.Popen[bytes]]
) -> None:
    watch = watches(_status_helper(_status_line("idle"), "exit0"), retry_s=30.0)
    assert _eventually(lambda: bool(started_processes) and started_processes[0].returncode == 0)

    watch.stop()  # joins the thread, so what it does after the command ends is done

    (process,) = started_processes
    assert process.stdout is not None
    assert process.stdout.closed


def _lock_answer(
    stdout: bytes, returncode: int = 0
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    """The Omarchy shell's lock query, answering `stdout`."""

    def run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert cmd == ["omarchy-shell", "lock", "isLocked"]
        return subprocess.CompletedProcess(cmd, returncode, stdout, b"")

    return run


@pytest.mark.parametrize(("stdout", "locked"), [(b"true\n", True), (b"false\n", False)])
def test_the_lock_screen_is_read_from_the_omarchy_shell(stdout: bytes, *, locked: bool) -> None:
    assert screen_locked(run=_lock_answer(stdout)) is locked


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [(b"", 1), (b"maybe\n", 0), (b"OMARCHY_PATH is not set\n", 1)],
    ids=["no shell", "unreadable", "no omarchy"],
)
def test_a_lock_state_that_cannot_be_read_counts_as_locked(stdout: bytes, returncode: int) -> None:
    # Asked only when the owner wants Jarvis deaf at a locked screen: a fault keeps it deaf.
    assert screen_locked(run=_lock_answer(stdout, returncode)) is True


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("omarchy-shell"), subprocess.TimeoutExpired("omarchy-shell", 2)],
    ids=["missing", "timed out"],
)
def test_a_lock_query_that_cannot_run_counts_as_locked(failure: Exception) -> None:
    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise failure

    assert screen_locked(run=run) is True


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


@pytest.fixture
def started_processes(monkeypatch: pytest.MonkeyPatch) -> list[subprocess.Popen[bytes]]:
    """Every process started while the test runs, so it can check what is left of each."""
    started: list[subprocess.Popen[bytes]] = []
    real_init: Callable[..., None] = subprocess.Popen.__init__

    def recording_init(self: subprocess.Popen[bytes], *args: object, **kwargs: object) -> None:
        real_init(self, *args, **kwargs)
        started.append(self)

    monkeypatch.setattr(subprocess.Popen, "__init__", recording_init)
    return started


def test_stopping_a_microphone_closes_the_pipe_it_read(
    started_processes: list[subprocess.Popen[bytes]],
) -> None:
    mic = Microphone(command=["sh", "-c", "exec sleep 30"])
    mic.start()

    mic.stop()

    (process,) = started_processes
    assert process.returncode is not None
    assert process.stdout is not None
    assert process.stdout.closed


def test_stopping_a_microphone_that_already_ended_closes_its_pipe(
    started_processes: list[subprocess.Popen[bytes]],
    ended_microphone: Callable[[int], Microphone],
) -> None:
    mic = ended_microphone(2)

    mic.stop()

    (process,) = started_processes
    assert process.stdout is not None
    assert process.stdout.closed


def test_stopping_a_microphone_whose_pipe_a_child_keeps_returns_soon_and_keeps_the_pipe(
    started_processes: list[subprocess.Popen[bytes]], tmp_path: Path
) -> None:
    # A capture command that leaves a grandchild holding the pipe's write end: the reader never
    # sees the end, so stop() must give up waiting for it and must not close the pipe under it.
    # `setsid` puts the command and its grandchild in a group of their own, which the test ends.
    child_started = tmp_path / "child-started"
    mic = Microphone(command=["setsid", "sh", "-c", f"sleep 30 & touch {child_started}; wait"])
    mic.start()
    group = mic.pid
    assert group is not None
    try:
        assert _eventually(child_started.exists)
        started = time.perf_counter()
        mic.stop()
        took = time.perf_counter() - started

        (process,) = started_processes
        assert took < audio_module.STOP_WAIT_S / 2
        assert process.stdout is not None
        assert not process.stdout.closed
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)
        mic.wait_ended(timeout=5)  # the reader sees the end now that the group is gone
        for process in started_processes:
            if process.stdout is not None:
                process.stdout.close()


def test_a_stop_between_the_two_halves_of_a_start_does_not_touch_the_new_capture(
    monkeypatch: pytest.MonkeyPatch,
    started_processes: list[subprocess.Popen[bytes]],
) -> None:
    # `jarvis pause` can stop the microphone while a start is under way. A stop that sees the new
    # process with the old (finished) reader thread would close the new pipe under the new reader.
    one_frame_then_wait = f"head -c {FRAME * 2} /dev/zero; exec sleep 30"
    mic = Microphone(command=["sh", "-c", one_frame_then_wait])
    mic.start()
    mic.stop()
    real_thread = threading.Thread

    class StopsWhileStarting(real_thread):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # pyright: ignore[reportArgumentType] - passes Thread's own arguments through untyped
            target = kwargs.get("target")
            if getattr(target, "__name__", "") == "_read":
                mic.stop()

    monkeypatch.setattr(audio_module.threading, "Thread", StopsWhileStarting)
    mic.start()
    monkeypatch.setattr(audio_module.threading, "Thread", real_thread)
    try:
        assert started_processes[-1].stdout is not None
        assert not started_processes[-1].stdout.closed
        assert _first_outcome(mic, timeout=5) == "frame"
    finally:
        mic.stop()


def test_a_microphone_that_sends_nothing_counts_as_ended_after_its_stall_time() -> None:
    # A capture left unlinked (its source gone, no other to move to) keeps running but sends
    # no frames; the loop must hear of it to reopen the microphone.
    mic = Microphone(command=["sh", "-c", "exec sleep 30"], stall_s=0.3)
    mic.start()
    try:
        assert _first_outcome(mic, timeout=5) == "ended"
    finally:
        mic.stop()


def test_time_spent_on_a_request_is_not_counted_as_a_stall() -> None:
    # The loop reads no frame while it answers; a short gap right after that is no stall.
    frame = f"head -c {FRAME * 2} /dev/zero"
    mic = Microphone(
        command=["sh", "-c", f"{frame}; sleep 1.5; {frame}; exec sleep 30"], stall_s=0.8
    )
    mic.start()
    try:
        frames = mic.frames()
        next(frames)
        threading.Event().wait(1.0)  # answering a request
        next(frames)  # 0.5 s more: no stall
    finally:
        mic.stop()


def test_a_microphone_without_a_stall_time_waits_for_frames() -> None:
    mic = Microphone(command=["sh", "-c", "exec sleep 30"])
    mic.start()
    try:
        assert _first_outcome(mic, timeout=0.5) == "blocked"
    finally:
        mic.stop()


class _SileroSession:
    """Stands in for Silero's `InferenceSession`: records what `Vad` feeds and answers in order."""

    def __init__(self, probabilities: list[float]) -> None:
        super().__init__()
        self.feeds: list[dict[str, NDArray[np.generic]]] = []
        self._probabilities = iter(probabilities)

    def run(
        self, _outputs: None, feed: dict[str, NDArray[np.generic]]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        self.feeds.append({name: value.copy() for name, value in feed.items()})
        state = np.asarray(feed["state"], dtype=np.float32) + 1  # the model's new state
        return np.array([[next(self._probabilities)]], dtype=np.float32), state


def _vad(monkeypatch: pytest.MonkeyPatch, probabilities: list[float]) -> tuple[Vad, _SileroSession]:
    session = _SileroSession(probabilities)

    def load(*_args: object, **_kwargs: object) -> _SileroSession:
        return session

    monkeypatch.setattr(audio_module.ort, "InferenceSession", load)
    return Vad(Path("/nonexistent/silero.onnx")), session


def test_the_vad_feeds_silero_chunks_of_512_samples_with_64_of_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vad, session = _vad(monkeypatch, [0.1, 0.2])
    frame = (np.arange(VAD_CHUNK * 2) % 1000).astype(np.int16)

    vad.speech_probability(frame)

    first, second = session.feeds
    assert first["input"].shape == (1, VAD_CONTEXT + VAD_CHUNK)
    assert first["input"].dtype == np.float32
    assert not first["input"][:, :VAD_CONTEXT].any()  # nothing came before the first chunk
    assert first["input"][0, VAD_CONTEXT:].tolist() == pytest.approx(
        (frame[:VAD_CHUNK] / 32768.0).tolist()
    )
    assert first["state"].shape == (2, 1, 128)
    assert first["sr"].dtype == np.int64
    assert int(first["sr"]) == audio_module.RATE
    # The second chunk starts with the last 64 samples of the first, and the model's state.
    np.testing.assert_array_equal(
        second["input"][0, :VAD_CONTEXT], first["input"][0, -VAD_CONTEXT:]
    )
    assert (second["state"] == 1).all()


def test_the_vad_answers_with_the_best_chunk_and_keeps_a_part_chunk_for_the_next_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vad, session = _vad(monkeypatch, [0.2, 0.9, 0.1, 0.4, 0.3])
    frame = np.zeros(FRAME, dtype=np.int16)  # 1280 samples: two chunks and 256 over

    assert vad.speech_probability(frame) == pytest.approx(0.9)
    assert len(session.feeds) == 2
    assert vad.speech_probability(frame) == pytest.approx(0.4)  # 256 + 1280 left: three chunks
    assert len(session.feeds) == 5
    assert vad.speech_probability(frame[:100]) == pytest.approx(
        0.0
    )  # less than a chunk: nothing to say yet
    assert len(session.feeds) == 5


def test_the_vad_forgets_its_state_when_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    vad, session = _vad(monkeypatch, [0.5, 0.5])
    loud = np.full(VAD_CHUNK + 100, 5000, dtype=np.int16)
    vad.speech_probability(loud)

    vad.reset()
    vad.speech_probability(loud)

    after = session.feeds[1]
    assert not after["state"].any()
    assert not after["input"][:, :VAD_CONTEXT].any()
    assert after["input"][0, VAD_CONTEXT] == pytest.approx(5000 / 32768.0)


class _LiveKitModel:
    """Stands in for `livekit.wakeword.WakeWordModel`: scores come from the test."""

    def __init__(
        self, scores: list[dict[str, float]], bad: Mapping[str, Exception] | None = None
    ) -> None:
        super().__init__()
        self.scores = iter(scores)
        self.frames: list[NDArray[np.int16]] = []
        self.resets = 0
        self.bad = bad or {}
        self.loaded: list[Path] = []

    def load_model(self, path: Path) -> None:
        if path.name in self.bad:
            raise self.bad[path.name]
        self.loaded.append(path)

    def predict(self, frame: NDArray[np.int16]) -> dict[str, float]:
        self.frames.append(frame)
        return next(self.scores)

    def reset(self) -> None:
        self.resets += 1


def _wake_word(
    monkeypatch: pytest.MonkeyPatch, scores: list[dict[str, float]], threshold: float = 0.5
) -> tuple[WakeWord, _LiveKitModel]:
    model = _fake_livekit(monkeypatch, _LiveKitModel(scores))
    wake = WakeWord([Path("/nonexistent/hey_jarvis.onnx")], threshold)
    assert model.loaded == [Path("/nonexistent/hey_jarvis.onnx")]
    return wake, model


def _fake_livekit(monkeypatch: pytest.MonkeyPatch, model: _LiveKitModel) -> _LiveKitModel:
    def make(*, sess_options: object) -> _LiveKitModel:
        assert sess_options is not None
        return model

    monkeypatch.setattr("livekit.wakeword.WakeWordModel", make)
    return model


def test_a_wake_word_model_that_cannot_be_used_is_left_out_and_the_rest_listen(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    bad = {
        "yo_jarvis.onnx": ValueError("not a valid ONNX model"),
        "jarvis.onnx": FileNotFoundError("Wake word model not found"),
    }
    model = _fake_livekit(monkeypatch, _LiveKitModel([{"hey_jarvis": 0.9}], bad))
    paths = [Path(f"/models/{name}.onnx") for name in ("yo_jarvis", "hey_jarvis", "jarvis")]

    with caplog.at_level(logging.WARNING):
        wake = WakeWord(paths, 0.5)

    assert model.loaded == [Path("/models/hey_jarvis.onnx")]
    assert "/models/yo_jarvis.onnx cannot be used (not a valid ONNX model)" in caplog.text
    assert "/models/jarvis.onnx cannot be used" in caplog.text
    assert wake.heard(np.zeros(FRAME, dtype=np.int16))


def test_with_no_usable_wake_word_model_the_last_error_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = {"a.onnx": FileNotFoundError("no a"), "b.onnx": ValueError("bad b")}
    _fake_livekit(monkeypatch, _LiveKitModel([], bad))

    with pytest.raises(ValueError, match="bad b"):
        WakeWord([Path("/models/a.onnx"), Path("/models/b.onnx")], 0.5)


def test_the_wake_word_fires_at_the_threshold_and_forgets_what_it_heard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wake, model = _wake_word(
        monkeypatch, [{"hey_jarvis": 0.49}, {"hey_jarvis": 0.5}, {"hey_jarvis": 0.1}]
    )
    frame = np.zeros(FRAME, dtype=np.int16)

    assert not wake.heard(frame)
    assert model.resets == 0
    assert wake.heard(frame)
    assert model.resets == 1  # so one phrase is not reported twice
    assert not wake.heard(frame)
    assert model.frames[0] is frame  # the int16 frame goes in as it is


def test_the_wake_word_takes_the_best_score_and_none_means_nothing_heard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wake, _model = _wake_word(monkeypatch, [{}, {"a": 0.1, "b": 0.8}], threshold=0.6)
    frame = np.zeros(FRAME, dtype=np.int16)

    assert not wake.heard(frame)
    assert wake.heard(frame)


def test_a_wake_word_with_its_own_threshold_is_held_to_it_and_the_rest_to_the_shared_one(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    scores = [
        {"hey_jarvis": 0.1, "jarvis": 0.8},  # over the shared 0.5, under its own 0.85
        {"hey_jarvis": 0.1, "jarvis": 0.85},
        {"hey_jarvis": 0.5, "jarvis": 0.1},
        {"hey_jarvis": 0.6, "jarvis": 0.8},  # the higher score is short; the other one wakes
    ]
    _fake_livekit(monkeypatch, _LiveKitModel(scores))
    paths = [Path("/models/hey_jarvis.onnx"), Path("/models/jarvis.onnx")]
    wake = WakeWord(paths, 0.5, {"jarvis": 0.85})
    frame = np.zeros(FRAME, dtype=np.int16)

    with caplog.at_level(logging.INFO):
        assert not wake.heard(frame)
        assert wake.heard(frame)
        assert wake.heard(frame)
        assert wake.heard(frame)

    assert "wake word jarvis score 0.85 (threshold 0.85)" in caplog.text
    assert "wake word hey_jarvis score 0.60 (threshold 0.50)" in caplog.text


def test_every_wake_word_over_its_threshold_is_logged_in_one_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_livekit(monkeypatch, _LiveKitModel([{"hey_jarvis": 0.6, "jarvis": 0.91, "yo": 0.1}]))
    paths = [Path("/models/hey_jarvis.onnx"), Path("/models/jarvis.onnx")]
    wake = WakeWord(paths, 0.5, {"jarvis": 0.85})

    with caplog.at_level(logging.INFO):
        assert wake.heard(np.zeros(FRAME, dtype=np.int16))

    lines = [r.getMessage() for r in caplog.records if "wake word" in r.getMessage()]
    assert lines == [
        "wake word jarvis score 0.91 (threshold 0.85), hey_jarvis score 0.60 (threshold 0.50)"
    ]


def test_a_wake_word_s_own_threshold_may_be_below_the_shared_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_livekit(monkeypatch, _LiveKitModel([{"yo_jarvis": 0.42}, {"yo_jarvis": 0.45}]))
    wake = WakeWord([Path("/models/yo_jarvis.onnx")], 0.5, {"yo_jarvis": 0.43})
    frame = np.zeros(FRAME, dtype=np.int16)

    assert not wake.heard(frame)
    assert wake.heard(frame)


def test_resetting_the_wake_word_resets_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    wake, model = _wake_word(monkeypatch, [])

    wake.reset()

    assert model.resets == 1


def _models(
    stdout: bytes, returncode: int = 0
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    """A `voxtype info models` that prints `stdout`, checking the command it was given."""

    def run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert cmd == ["voxtype", "info", "models", "--json", "--engine", "whisper"]
        return subprocess.CompletedProcess(cmd, returncode, stdout, b"")

    return run


WHISPER_MODELS = json.dumps(
    {
        "engines": {
            "whisper": {
                "default": "base.en",
                "models": [
                    {"name": "base.en", "installed": True, "downloadable": True},
                    {"name": "small.en", "installed": False, "downloadable": True},
                    {"name": "tiny", "downloadable": True},
                ],
            }
        }
    }
).encode()


def test_voxtype_says_which_models_are_installed() -> None:
    assert installed_models("whisper", run=_models(WHISPER_MODELS)) == {"base.en"}


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [
        (WHISPER_MODELS, 2),  # an older Voxtype without `info models`
        (b"not json", 0),
        (b"{}", 0),
        (b'{"engines": {"whisper": {"models": 3}}}', 0),
    ],
)
def test_a_model_list_that_cannot_be_read_says_nothing(stdout: bytes, returncode: int) -> None:
    assert installed_models("whisper", run=_models(stdout, returncode)) is None


def test_a_voxtype_that_cannot_run_says_nothing() -> None:
    def missing(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError(cmd[0])

    assert installed_models("whisper", run=missing) is None


def test_the_wake_word_logs_which_phrase_was_heard(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    wake, _model = _wake_word(monkeypatch, [{"hey_jarvis": 0.2, "yo_jarvis": 0.9}])

    with caplog.at_level(logging.INFO):
        assert wake.heard(np.zeros(FRAME, dtype=np.int16))

    assert "wake word yo_jarvis score 0.90" in caplog.text
