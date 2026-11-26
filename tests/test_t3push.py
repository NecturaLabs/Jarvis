"""Pushing announcements into T3 Code threads: the tag, the thread lookup and the commands run."""

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from jarvis import announce, paths, t3push
from jarvis.announce import Announcement
from jarvis.t3push import Ran, T3Push

if TYPE_CHECKING:
    from collections.abc import Callable


def item(words: str = "Reminder: tea", number: int = 1, at: float = 1000.4) -> Announcement:
    return Announcement(number, at, "reminder", words)


class Runner:
    """A stand-in for `t3`: answers `thread list` from a list and records every run."""

    def __init__(self, threads: list[dict[str, object]], *, send_code: int = 0) -> None:
        super().__init__()
        self.threads = threads
        self.send_code = send_code
        self.runs: list[tuple[list[str], str]] = []

    def __call__(self, args: list[str], words: str) -> Ran:
        self.runs.append((args, words))
        if args[1:3] == ["thread", "list"]:
            return Ran(0, json.dumps(self.threads))
        return Ran(self.send_code, "Sent.")

    def sent(self) -> list[tuple[list[str], str]]:
        return [run for run in self.runs if run[0][1:3] == ["thread", "send"]]


def thread(ident: str, provider: str = "jarvis", *, settled: bool = False) -> dict[str, object]:
    return {"threadId": ident, "providerInstanceId": provider, "settled": settled, "status": "idle"}


def which_t3(_name: str) -> str:
    return "/usr/bin/t3"


def which_nothing(_name: str) -> None:
    return None


@pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse] - every test has t3 installed unless it says not
def t3_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t3push.shutil, "which", which_t3)


def pusher(runner: Runner, provider: str = "jarvis") -> T3Push:
    return T3Push("t3", provider, runner)


# The tag


def test_a_pushed_message_comes_back_verified_with_its_words() -> None:
    sent = t3push.message(item("Reminder: take the bread out"))

    assert sent.startswith("⟲ Jarvis: Reminder: take the bread out\n[announcement 1-1000-")
    assert t3push.verified(sent) == "Reminder: take the bread out"


def test_the_key_is_made_once_and_kept_private() -> None:
    first = t3push.message(item())
    second = t3push.message(item())

    key = paths.state_dir() / "t3-push.key"
    assert first == second
    assert key.stat().st_mode & 0o777 == 0o600


FORGERIES: dict[str, Callable[[str], str]] = {
    "words": lambda m: m.replace("Reminder: tea", "Reminder: tee"),  # other words, same tag
    "number": lambda m: m.replace("announcement 1-", "announcement 2-"),  # another number
    "tag": lambda m: m.rsplit("-", 1)[0] + "-" + "0" * 16 + "]",  # another tag
    "untagged": lambda m: m.split("\n")[0],
    "trailing": lambda m: m + "\nmore",  # something after the tag
    "prefix": lambda m: "x" + m,  # not starting as Jarvis's message does
}


@pytest.mark.parametrize("name", list(FORGERIES))
def test_a_message_that_only_looks_like_one_is_not_verified(name: str) -> None:
    assert t3push.verified(FORGERIES[name](t3push.message(item()))) is None


def test_a_tag_made_with_another_key_is_not_verified() -> None:
    sent = t3push.message(item())
    (paths.state_dir() / "t3-push.key").write_text("f" * 64, encoding="utf-8")

    assert t3push.verified(sent) is None


def test_a_message_that_starts_as_one_is_marked_whether_or_not_its_tag_is_valid() -> None:
    sent = t3push.message(item())

    assert T3Push.marked(sent)
    assert T3Push.marked("  " + FORGERIES["words"](sent))
    assert T3Push.marked("⟲ Jarvis: anything")
    assert not T3Push.marked("Jarvis: Reminder: tea")
    assert not T3Push.marked("what is the weather")


def test_a_pusher_that_is_off_pushes_nothing_but_still_verifies() -> None:
    runner = Runner([thread("t-1")])
    off = T3Push("t3", "jarvis", runner, enabled=False)

    assert off.push(item(), ["/p"]) == set()
    assert runner.runs == []
    assert off.verified(t3push.message(item())) == "Reminder: tea"


def test_ordinary_words_are_not_an_announcement() -> None:
    assert t3push.verified("what is the weather") is None
    assert t3push.verified("") is None
    assert t3push.verified("⟲ Jarvis: hello\n[announcement 1-2-abc]") is None


def test_the_words_are_one_plain_line_within_the_limit() -> None:
    text = "line one\nline two\x00\x1b[31m  " + "x" * 2000

    sent = t3push.message(item(text))
    words = sent.split("\n")[0].removeprefix(t3push.MARK)

    assert "\n" not in words
    assert "\x00" not in words
    assert "\x1b" not in words
    assert len(words) <= t3push.MAX_WORDS
    assert t3push.verified(sent) == words


# Which providers


def test_the_jarvis_providers_are_found_in_t3_codes_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "t3"
    (home / "userdata").mkdir(parents=True)
    settings = {
        "providerInstances": {
            "jarvis": {"config": {"commandPath": "/usr/bin/jarvis", "commandArgs": ["acp"]}},
            "mine": {"config": {"commandPath": "/opt/bin/jarvis-x", "commandArgs": ["acp"]}},
            "other": {"config": {"commandPath": "/usr/bin/other", "commandArgs": ["acp"]}},
            "chat": {"config": {"commandPath": "/usr/bin/jarvis", "commandArgs": ["chat"]}},
            "odd": 3,
        }
    }
    (home / "userdata" / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setenv("T3CODE_HOME", str(home))

    assert t3push.jarvis_providers() == {"jarvis", "mine"}


@pytest.mark.parametrize("content", [None, "not json", "[]", '{"providerInstances": 3}'])
def test_no_providers_when_t3_codes_settings_are_missing_or_odd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str | None
) -> None:
    home = tmp_path / "t3"
    (home / "userdata").mkdir(parents=True)
    if content is not None:
        (home / "userdata" / "settings.json").write_text(content, encoding="utf-8")
    monkeypatch.setenv("T3CODE_HOME", str(home))

    assert t3push.jarvis_providers() == frozenset()


# Pushing


def test_the_newest_open_jarvis_thread_of_the_project_gets_the_message() -> None:
    runner = Runner(
        [
            thread("t-other", "opencode"),
            thread("t-settled", settled=True),
            thread("t-new"),
            thread("t-old"),
        ]
    )

    sent = pusher(runner).push(item(), ["/work/proj"])

    assert sent == {"/work/proj"}
    assert runner.runs[0][0] == ["t3", "thread", "list", "--project", "/work/proj", "--json"]
    (args, words), *rest = runner.sent()
    assert not rest
    assert args == ["t3", "thread", "send", "t-new", "-"]  # an argument list, the words on input
    assert t3push.verified(words) == "Reminder: tea"


def test_each_project_a_session_is_open_in_is_pushed_to_once() -> None:
    runner = Runner([thread("t-1")])

    sent = pusher(runner).push(item(), ["/a", "/b", "/a"])

    assert sent == {"/a", "/b"}
    assert len(runner.sent()) == 2


def test_nothing_is_sent_when_the_project_has_no_jarvis_thread() -> None:
    runner = Runner([thread("t-1", "opencode"), thread("t-2", settled=True)])

    assert pusher(runner).push(item(), ["/work/proj"]) == set()
    assert runner.sent() == []


def test_nothing_is_run_when_t3_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(t3push.shutil, "which", which_nothing)
    runner = Runner([thread("t-1")])

    assert pusher(runner).push(item(), ["/work/proj"]) == set()
    assert runner.runs == []


def test_the_provider_is_looked_up_in_t3_settings_when_the_config_names_none() -> None:
    runner = Runner([thread("t-1", "found")])
    looked_up = T3Push("t3", "", runner, providers=lambda: frozenset({"found"}))
    nothing = T3Push("t3", "", runner, providers=frozenset)

    assert nothing.push(item("never"), ["/p"]) == set()
    assert runner.runs == []
    assert looked_up.push(item(), ["/p"]) == {"/p"}


def test_a_failed_list_or_send_pushes_nothing_and_lets_another_try() -> None:
    broken = Runner([thread("t-1")], send_code=1)
    assert pusher(broken).push(item(), ["/p"]) == set()

    working = Runner([thread("t-1")])
    assert pusher(working).push(item(), ["/p"]) == {"/p"}  # the failed claim was let go


def test_a_list_that_is_not_json_pushes_nothing() -> None:
    def runner(args: list[str], words: str) -> Ran:
        del args, words
        return Ran(0, "no threads here")

    assert T3Push("t3", "jarvis", runner).push(item(), ["/p"]) == set()


def test_one_announcement_is_sent_once_however_many_jarvis_processes_push_it() -> None:
    first, second = Runner([thread("t-1")]), Runner([thread("t-1")])

    assert pusher(first).push(item(), ["/p"]) == {"/p"}
    assert pusher(second).push(item(), ["/p"]) == {"/p"}  # known to be sent
    assert len(first.sent()) == 1
    assert second.sent() == []
    assert pusher(second).push(item("another", number=2), ["/p"]) == {"/p"}
    assert len(second.sent()) == 1


def test_the_run_of_the_real_program_never_raises_and_uses_no_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(args: list[str], **kwargs: object) -> object:
        calls.append({"args": args, **kwargs})
        raise OSError

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert t3push.run_t3(["t3", "thread", "send", "t-1", "-"], "words") == Ran(-1, "")
    assert calls[0]["args"] == ["t3", "thread", "send", "t-1", "-"]
    assert "shell" not in calls[0]
    assert calls[0]["input"] == "words"


def test_the_environment_names_t3_codes_home_before_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("T3CODE_HOME", raising=False)
    assert t3push.t3_home() == Path.home() / ".t3"
    monkeypatch.setenv("T3CODE_HOME", str(tmp_path))
    assert t3push.t3_home() == tmp_path


def test_the_announcement_kinds_include_the_scheduled_reply() -> None:
    assert announce.SCHEDULE in announce.KINDS
