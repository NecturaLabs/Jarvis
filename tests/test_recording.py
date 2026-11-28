"""Record only the PCM Jarvis sends to its player."""

import stat
import sys
import wave
from typing import TYPE_CHECKING

import pytest

from jarvis.app import channel_limits, channel_registry
from jarvis.audio import Speaker
from jarvis.config import Config
from jarvis.operations import OperationError, request_scope, validate_args
from jarvis.recording import OutputRecorder

if TYPE_CHECKING:
    from pathlib import Path


def test_output_recordings_keep_pcm_and_are_private(tmp_path: Path) -> None:
    folder = tmp_path / "recordings"
    recording = OutputRecorder(folder)
    pcm = b"\x01\x00\xfe\xff" * 40
    recording.feed(b"\x7f\x7f", 24000)

    recording.start()
    recording.feed(pcm[:60], 24000)
    recording.feed(pcm[60:], 24000)
    path = recording.finish()

    assert path is not None
    assert list(folder.iterdir()) == [path]
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with wave.open(str(path), "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 24000)
        assert audio.readframes(audio.getnframes()) == pcm


@pytest.mark.parametrize("source", ["pc_audio", "scheduled", "outside"])
def test_only_a_present_owner_can_control_recording(tmp_path: Path, source: str) -> None:
    recording = OutputRecorder(tmp_path / "recordings")
    op = recording.operation()
    start = validate_args(op, {"action": "start"})
    with request_scope(foreign=source == "pc_audio", scheduled=source == "scheduled") as request:
        if source == "outside":
            request.saw("file")
        with pytest.raises(OperationError, match="owner"):
            op.run(start)
    assert recording.finish() is None


def test_recording_converts_a_change_of_sample_rate_without_losing_time(tmp_path: Path) -> None:
    recording = OutputRecorder(tmp_path / "recordings")
    recording.start()
    recording.feed(b"\x01\x00" * 1600, 16000)
    recording.feed(b"\xff\xff" * 2400, 24000)

    path = recording.finish()

    assert path is not None
    with wave.open(str(path), "rb") as audio:
        assert audio.getframerate() == 24000
        assert audio.getnframes() == 4800


def test_the_voice_service_records_the_same_song_samples_it_plays(tmp_path: Path) -> None:
    recording = OutputRecorder(tmp_path / "recordings")
    heard = tmp_path / "player.pcm"
    player = [
        sys.executable,
        "-c",
        "import pathlib, sys; pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())",
        str(heard),
    ]
    speaker = Speaker(
        lambda _text: (b"", 24000), player=lambda _rate: player, record_output=recording.feed
    )
    pcm = b"\x01\x00\xfe\xff" * 3000
    recording.start()
    speaker.queue_sound(pcm, 24000)
    speaker.wait()

    path = recording.finish()

    assert path is not None
    with wave.open(str(path), "rb") as audio:
        assert audio.readframes(audio.getnframes()) == heard.read_bytes() == pcm


def test_recording_is_available_only_with_the_live_voice_recorder(tmp_path: Path) -> None:
    cfg = Config()
    recording = OutputRecorder(tmp_path / "recordings")
    voice = channel_registry(cfg, [], channel_limits(cfg, "voice"), "voice", recording=recording)
    offline = channel_registry(cfg, [], channel_limits(cfg, "voice"), "voice")
    chat = channel_registry(cfg, [], channel_limits(cfg, "chat"), "chat", recording=recording)

    assert voice.get("voice_record") is not None
    assert offline.get("voice_record") is None
    assert chat.get("voice_record") is None


def test_a_link_in_place_of_the_recordings_folder_refuses_the_save(tmp_path: Path) -> None:
    folder = tmp_path / "recordings"
    other = tmp_path / "other"
    other.mkdir()
    folder.symlink_to(other, target_is_directory=True)
    recording = OutputRecorder(folder)
    recording.start()
    recording.feed(b"\x00\x00" * 24, 24000)

    with pytest.raises(NotADirectoryError):
        recording.finish()

    assert list(other.iterdir()) == []
    folder.unlink()
    assert recording.finish() is not None
