"""The voice catalogue: the English Kokoro voices Jarvis can speak with, and `jarvis voice`."""

import json

import pytest

from jarvis import cli, config, voices
from jarvis.cli import main


def test_the_catalogue_lists_each_english_voice_once_with_a_description() -> None:
    ids = [v.id for v in voices.VOICES]

    assert len(ids) == len(set(ids)) == 28
    assert all(v.id[:1] in "ab" and v.id[1:2] in "fm" for v in voices.VOICES)
    assert all(v.about and v.grade for v in voices.VOICES)
    assert voices.find("kokoro:bm_george") == voices.find(" BM_George ")
    assert voices.find("zf_xiaobei") is None  # a Mandarin voice: wrong with English sounds


def test_the_default_voice_is_in_the_catalogue() -> None:
    found = voices.find(config.AudioConfig().voice)

    assert found is not None
    assert found.setting == config.AudioConfig().voice


def test_voice_lists_the_voices_and_marks_the_current_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config.save(config.Config())

    assert main(["voice"]) == 0

    out = capsys.readouterr().out.splitlines()
    assert out[0] == "voice: bm_george"
    marked = [line for line in out if line.startswith("*")]
    assert len(marked) == 1
    assert "bm_george" in marked[0]
    assert any("af_heart" in line and "American" in line for line in out)


def test_voice_switches_and_restarts_the_voice_service(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config.save(config.Config())
    restarts: list[bool] = []
    monkeypatch.setattr(cli, "_restart_if_running", lambda: restarts.append(True))

    assert main(["voice", "af_heart"]) == 0

    assert config.load().audio.voice == "kokoro:af_heart"
    assert capsys.readouterr().out.startswith("voice: af_heart (American female, ")
    assert restarts == [True]


def test_voice_refuses_a_name_it_does_not_list(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config.save(config.Config())
    monkeypatch.setattr(cli, "_restart_if_running", pytest.fail)

    assert main(["voice", "nobody"]) == 2

    assert "`jarvis voice` lists them" in capsys.readouterr().err
    assert config.load().audio.voice == "kokoro:bm_george"


def test_voice_answers_an_agent_with_the_catalogue(capsys: pytest.CaptureFixture[str]) -> None:
    config.save(config.Config())

    assert main(["voice", "--agent"]) == 0

    answer = json.loads(capsys.readouterr().out)
    assert answer["voice"] == "kokoro:bm_george"
    assert len(answer["voices"]) == 28
    assert [v["id"] for v in answer["voices"] if v["current"]] == ["bm_george"]
