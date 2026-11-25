# ruff: file-ignore[ambiguous-unicode-character-string] - the tests hold espeak's IPA symbols
"""SoulX-Singer through `audiocpp_cli`: the metadata, the command, and the fall back to speech.

The program is never run here: a stand-in runner reads what Jarvis wrote and writes a song.
"""

import json
import wave
from pathlib import Path

import numpy as np
import pytest
from test_sing_psola import Voice

from jarvis.audio import AudioError
from jarvis.sing import arpabet, soulx
from jarvis.sing.psola import PsolaSinger
from jarvis.sing.score import parse

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

    def __init__(self, *, exit_code: int = 0, timeout: bool = False) -> None:
        super().__init__()
        self.exit_code = exit_code
        self.timeout = timeout
        self.commands: list[list[str]] = []
        self.reference: list[dict[str, object]] = []
        self.target: list[dict[str, object]] = []
        self.reference_seconds = 0.0

    def __call__(self, cmd: list[str], timeout_s: float) -> tuple[int, str]:
        self.commands.append(cmd)
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
        if self.exit_code == 0:
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

    pcm, rate = _singer(tmp_path, runner).render(score)

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


@pytest.mark.parametrize("runner", [Runner(exit_code=1), Runner(timeout=True)])
def test_a_failed_or_stuck_soulx_sings_with_the_speaking_voice_instead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    score = parse(LINE, 120)

    pcm, rate = _singer(tmp_path, runner).render(score)

    assert rate == soulx.RATE
    assert pcm.size == round(score.seconds * soulx.RATE)  # the speaking voice's song
    assert np.any(pcm)
