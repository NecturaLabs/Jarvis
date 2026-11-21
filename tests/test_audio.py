"""Microphone stream handling and transcript parsing, without real audio devices."""

import os
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, override

import numpy as np
import pytest

from jarvis import audio as audio_module
from jarvis.audio import (
    FRAME,
    AppAudio,
    AudioError,
    Microphone,
    app_record_command,
    app_stream_serials,
    capture_sources,
    is_phantom,
    match_app_streams,
    parse_transcript,
    pw_record_command,
    voxtype_command,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

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
        ("Thank you, play Bad Omens", False),
        ("Skip this", False),
    ],
)
def test_whisper_phantoms_are_recognised(text: str, phantom: bool) -> None:
    assert is_phantom(text) is phantom


def test_voxtype_command_passes_overrides_before_the_subcommand() -> None:
    cmd = voxtype_command(Path("/x.wav"), model="large-v3-turbo", prompt="Bad Omens, Linkin Park")
    assert cmd == [
        "voxtype",
        "--model",
        "large-v3-turbo",
        "--initial-prompt",
        "Bad Omens, Linkin Park",
        "transcribe",
        "/x.wav",
    ]


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
    frames = audio.frames()
    got = [next(frames) for _ in range(3)]
    audio.stop()

    assert len(got) == 3
    assert targets[0] == "42"
    assert len(finds) >= 2


def test_app_audio_with_no_apps_records_all_pc_output() -> None:
    targets: list[str] = []

    def command_for(target: str) -> list[str]:
        targets.append(target)
        return ["head", "-c", str(FRAME * 2), "/dev/zero"]

    audio = AppAudio([], find=lambda _a: pytest.fail("no lookup needed"), command_for=command_for)
    next(audio.frames())
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
    """A capture command that writes the same byte forever."""
    return ["sh", "-c", f"tr '\\0' '\\{byte:03o}' </dev/zero"]


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
    next(mic.frames())  # the frame is written after the trap is set

    mic.stop()

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
