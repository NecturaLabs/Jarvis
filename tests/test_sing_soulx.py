# ruff: file-ignore[ambiguous-unicode-character-string] - the tests hold espeak's IPA symbols
"""SoulX-Singer through `audiocpp_cli`: the metadata, the command, and the fall back to speech.

The program is never run here: a stand-in runner reads what Jarvis wrote and writes a song.
"""

import contextlib
import json
import os
import signal
import struct
import threading
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, override

import numpy as np
import pytest
from test_sing_psola import Voice

from jarvis.audio import AudioError, SingerStoppedError
from jarvis.sing import arpabet, soulx
from jarvis.sing.budget import Budget, SongStoppedError
from jarvis.sing.psola import PsolaSinger
from jarvis.sing.score import parse

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.sing.score import Score

# An original line, written for these tests.
LINE = "Ri- C4 1, ver E4 1, ~ G4 1, _ 0.5, flow D4 2"


def test_the_metadata_has_each_note_with_its_whole_word_and_edges_of_silence() -> None:
    (segment,) = soulx.metadata(parse(LINE, 120))

    assert segment["language"] == "English"
    river = "en_" + "-".join(arpabet.word("River"))
    flow = "en_" + "-".join(arpabet.word("flow"))
    assert segment["phoneme"].split() == ["<SP>", river, river, river, "<SP>", flow, "<SP>"]
    assert segment["text"].split() == ["<SP>", "River", "River", "River", "<SP>", "flow", "<SP>"]
    assert segment["note_type"].split() == ["1", "2", "3", "3", "1", "2", "1"]
    assert segment["note_pitch"].split() == ["0", "60", "64", "67", "0", "62", "0"]
    lengths = [float(x) for x in segment["duration"].split()]
    assert lengths == pytest.approx([0.2, 0.5, 0.5, 0.5, 0.25, 1.0, 0.2])
    assert segment["time"] == [0, round(sum(lengths) * 1000)]


@pytest.mark.parametrize(
    ("ipa", "phones"),
    [
        ("hˈuː", ["HH", "UW1"]),
        ("pɹˈɪɾi", ["P", "R", "IH1", "T", "IY0"]),
        ("dʒˈɑːɹvɪs", ["JH", "AA1", "R", "V", "IH0", "S"]),
        ("bˈʌʔn̩", ["B", "AH1", "T", "AH0", "N"]),  # a syllabic n is sung on a weak vowel
        ("fˈaɪɚ", ["F", "AY1", "ER0"]),
        ("ʃʃ", ["SH", "AH0", "SH"]),  # a word must be sung on a vowel
    ],
)
def test_espeak_ipa_becomes_arpabet_with_stress(ipa: str, phones: list[str]) -> None:
    assert arpabet.from_ipa(ipa) == phones


def _files(tmp_path: Path) -> soulx.Files:
    found = soulx.files(tmp_path)
    for path in (found.binary, found.model):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return found


class Runner:
    """Stands in for `audiocpp_cli`: keeps what it was given and writes a song, or fails."""

    def __init__(
        self, *, exit_code: int = 0, timeout: bool = False, writes: bytes = b"", stop: bool = False
    ) -> None:
        super().__init__()
        self.stop = stop  # the request is cancelled while it runs
        self.cancels: list[Callable[[], bool]] = []
        self.writes = writes  # what it writes instead of a song, when set
        self.exit_code = exit_code
        self.timeout = timeout
        self.commands: list[list[str]] = []
        self.timeouts: list[float] = []
        self.reference: list[dict[str, object]] = []
        self.target: list[dict[str, object]] = []
        self.reference_seconds = 0.0

    def __call__(
        self, cmd: list[str], timeout_s: float, cancelled: Callable[[], bool]
    ) -> tuple[int, str]:
        self.commands.append(cmd)
        self.cancels.append(cancelled)
        self.timeouts.append(timeout_s)
        if self.stop:
            msg = "stopped"
            raise SingerStoppedError(msg)
        if self.timeout:
            msg = f"ran past {timeout_s} s"
            raise AudioError(msg)
        options = dict(
            cmd[i + 1].split("=", 1) for i, arg in enumerate(cmd) if arg == "--request-option"
        )
        self.reference = json.loads(
            Path(options["reference_metadata_file"]).read_text(encoding="utf-8")
        )
        self.target = json.loads(Path(options["target_metadata_file"]).read_text(encoding="utf-8"))
        with wave.open(cmd[cmd.index("--voice-ref") + 1], "rb") as ref:
            self.reference_seconds = ref.getnframes() / ref.getframerate()
        if self.writes:
            Path(cmd[cmd.index("--out") + 1]).write_bytes(self.writes)
        elif self.exit_code == 0:
            with wave.open(cmd[cmd.index("--out") + 1], "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(soulx.RATE)
                out.writeframes(np.full(4800, 7, dtype=np.int16).tobytes())
        return self.exit_code, "vulkan: out of memory"


def _singer(tmp_path: Path, runner: Runner) -> soulx.SoulxSinger:
    return soulx.SoulxSinger(_files(tmp_path), PsolaSinger(Voice()), device=1, run=runner)


def test_soulx_sings_in_the_voice_of_the_same_song_sung_by_the_speaking_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    runner = Runner()
    score = parse(LINE, 120)

    pcm, rate = _singer(tmp_path, runner).render(score, Budget())

    assert rate == soulx.RATE
    assert pcm.tolist() == [7] * 4800  # what SoulX-Singer wrote
    (cmd,) = runner.commands
    assert cmd[0] == str(soulx.files(tmp_path).binary)
    assert cmd[cmd.index("--backend") + 1] == "vulkan"
    assert cmd[cmd.index("--device") + 1] == "1"
    assert "control=score" in cmd
    assert runner.reference == runner.target  # the reference is the song itself
    assert runner.reference_seconds == pytest.approx(score.seconds + 2 * soulx.EDGE_S)
    assert not list((tmp_path / "run" / "jarvis").iterdir())  # nothing is left behind


def _float_wav() -> bytes:
    """A WAV of 32-bit float samples (format 3), which is not what Jarvis reads."""
    data = np.zeros(100, dtype=np.float32).tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, soulx.RATE, soulx.RATE * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


@pytest.mark.parametrize(
    "runner",
    [
        Runner(exit_code=1),
        Runner(timeout=True),
        Runner(writes=b"not a sound file"),
        Runner(writes=_float_wav()),
        Runner(writes=_float_wav()[:30]),
    ],
    ids=["exit", "timeout", "garbage", "float", "cut-off"],
)
def test_a_failed_or_stuck_soulx_sings_with_the_speaking_voice_instead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    score = parse(LINE, 120)

    pcm, rate = _singer(tmp_path, runner).render(score, Budget())

    assert rate == soulx.RATE
    assert pcm.size == round(score.seconds * soulx.RATE)  # the speaking voice's song
    assert np.any(pcm)


def test_a_song_stopped_while_soulx_is_launched_is_stopped_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    runner = Runner(stop=True)

    with pytest.raises(SongStoppedError):
        _singer(tmp_path, runner).render(parse(LINE, 120), Budget())

    assert not list((tmp_path / "run" / "jarvis").iterdir())
    assert not caplog.records  # a stop is no failure to log


class _Then(PsolaSinger):
    """The speaking-voice singer, calling `after` once it has sung the reference."""

    def __init__(self, after: Callable[[], object]) -> None:
        super().__init__(Voice())
        self._after = after

    @override
    def sing(self, score: Score, budget: Budget = Budget()) -> tuple[np.ndarray, Score]:  # ruff: ignore[function-call-in-default-argument] - a frozen default, as in PsolaSinger
        out = super().sing(score, budget)
        self._after()
        return out


def test_a_song_stopped_before_soulx_starts_never_launches_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    runner = Runner()
    stopped = [False]
    singer = soulx.SoulxSinger(
        _files(tmp_path), _Then(lambda: stopped.__setitem__(0, True)), run=runner
    )

    with pytest.raises(SongStoppedError):
        singer.render(parse(LINE, 120), Budget(stopped=lambda: stopped[0]))

    assert not runner.commands


def test_a_song_stopped_while_soulx_runs_ends_the_program_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    found = _files(tmp_path)
    pid_file = tmp_path / "pid"
    found.binary.write_text(f"#!/bin/sh\necho $$ > {pid_file}\nexec sleep 60\n", encoding="ascii")
    found.binary.chmod(0o755)
    singer = soulx.SoulxSinger(found, PsolaSinger(Voice()))  # the real runner
    stop = threading.Event()
    threading.Thread(target=lambda: (_wait_for(pid_file.exists), stop.set()), daemon=True).start()
    started = time.monotonic()
    try:
        with pytest.raises(SongStoppedError):
            singer.render(parse(LINE, 120), Budget(stopped=stop.is_set))

        assert time.monotonic() - started < 5.0
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text(encoding="ascii")), 0)
        assert not list((tmp_path / "run" / "jarvis").iterdir())
    finally:
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(pid_file.read_text(encoding="ascii")), signal.SIGKILL)


def _wait_for(condition: Callable[[], bool]) -> None:
    pause = threading.Event()
    for _ in range(1000):
        if condition():
            return
        pause.wait(0.005)


def test_without_time_left_for_soulx_the_speaking_voice_sings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    runner = Runner()
    score = parse(LINE, 120)

    pcm, _ = _singer(tmp_path, runner).render(score, Budget(time_left=lambda: 2.0))

    assert not runner.commands  # SoulX-Singer could not finish in the time left
    assert pcm.size == round(score.seconds * soulx.RATE)


def test_soulx_is_given_no_more_than_the_requests_time_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    runner = Runner()

    _singer(tmp_path, runner).render(parse(LINE, 120), Budget(time_left=lambda: 8.0))

    assert runner.timeouts == [8.0]


def test_the_runner_is_given_the_stop_signal_of_the_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    runner = Runner()
    stopped = [False]

    _singer(tmp_path, runner).render(parse(LINE, 120), Budget(stopped=lambda: stopped[0]))
    stopped[0] = True

    assert runner.cancels[0]()
