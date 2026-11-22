"""tools/check_private.py: nothing about one person or machine goes into a commit."""

import io
import subprocess
from typing import TYPE_CHECKING

import check_private as cp
import pytest

if TYPE_CHECKING:
    from pathlib import Path

PRIVATE = "# mine\nKettlefield\nre:\\bmy-drive-\\d\n\n"


def problems(text: str, rules: list[cp.Rule] | None = None) -> list[str]:
    lines = enumerate(text.splitlines(), start=1)
    return [str(f) for f in cp.find("f.md", lines, rules or [])]


def test_example_and_project_addresses_pass() -> None:
    text = (
        "a@example.invalid b@mail.example.org 1+x@users.noreply.github.com support@necturalabs.com"
    )
    assert problems(text + " user@1000.service support@github.com") == []


def test_a_real_address_is_refused() -> None:
    assert problems("mail jane.doe@" + "mailbox.net") == [
        "f.md:1: an e-mail address outside the example domains"
    ]


def test_a_real_home_folder_is_refused_and_a_placeholder_passes() -> None:
    assert problems("/home/me/x /home/someone /Users/you/y") == []
    assert problems("cd /home/" + "jdoe/projects") == [
        "f.md:1: a path in a real home folder; use /home/me"
    ]


def test_a_private_term_matches_whole_words_and_never_names_the_term() -> None:
    rules = cp.private_rules(PRIVATE)

    found = problems("the KETTLEFIELD game\nkettlefields\nmy-drive-2", rules)

    assert found == [
        "f.md:1: a term from the private list (its line 2)",
        "f.md:3: a term from the private list (its line 3)",
    ]
    assert all("ettlefield" not in line for line in found)


def test_added_lines_carry_the_new_file_and_line() -> None:
    diff = (
        "diff --git a/x.md b/x.md\n--- a/x.md\n+++ b/x.md\n@@ -1,0 +4,2 @@\n+one\n+two\n"
        "diff --git a/gone.md b/gone.md\n--- a/gone.md\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n"
    )

    assert list(cp.added_lines(diff)) == [("x.md", 4, "one"), ("x.md", 5, "two")]


def test_a_commit_message_skips_git_comments() -> None:
    rules = cp.private_rules(PRIVATE)
    text = "docs: tidy\n\nKettlefield is gone\n# Kettlefield in a comment\n"

    found = [str(f) for f in cp.in_message("commit message", text, rules)]

    assert found == ["commit message:3: a term from the private list (its line 2)"]


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "dev")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "T")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "a.md").write_text("generic\n", encoding="utf-8")
    _git(root, "add", "a.md")
    _git(root, "commit", "-q", "-m", "docs: start")
    terms = tmp_path / "terms"
    terms.write_text(PRIVATE, encoding="utf-8")
    monkeypatch.setattr(cp, "ROOT", root)
    monkeypatch.setenv("JARVIS_PRIVATE_TERMS", str(terms))
    return root


def test_the_tracked_tree_is_checked(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cp.main([]) == 0
    (repo / "a.md").write_text("generic\nKettlefield\n", encoding="utf-8")

    assert cp.main([]) == 1
    assert "a.md:2: a term from the private list (its line 2)" in capsys.readouterr().out


def test_staged_lines_are_checked(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (repo / "b.md").write_text("x\nsee my-drive-7\n", encoding="utf-8")
    _git(repo, "add", "b.md")

    assert cp.main(["--staged"]) == 1
    assert "b.md:2: a term from the private list (its line 3)" in capsys.readouterr().out


def test_a_push_checks_each_new_commit_and_its_message(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "c.md").write_text("fine\n", encoding="utf-8")
    _git(repo, "add", "c.md")
    _git(repo, "commit", "-q", "-m", "docs: name Kettlefield")
    tip = _git(repo, "rev-parse", "HEAD")
    refs = f"refs/heads/dev {tip} refs/heads/dev {base}\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(refs))

    assert cp.main(["--push"]) == 1
    out = capsys.readouterr().out
    assert f"{tip[:9]} message:1: a term from the private list (its line 2)" in out
    assert "c.md" not in out


def test_a_deleted_ref_sends_nothing(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    refs = f"(delete) {'0' * 40} refs/heads/old {_git(repo, 'rev-parse', 'HEAD')}\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(refs))

    assert cp.main(["--push"]) == 0


def test_this_machine_is_never_named(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cp.socket, "gethostname", lambda: "desk-4417.lan")
    (repo / "a.md").write_text("built on desk-4417\n", encoding="utf-8")

    assert cp.main([]) == 1


def test_without_the_list_the_general_rules_still_run(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("JARVIS_PRIVATE_TERMS", str(tmp_path / "missing"))
    (repo / "a.md").write_text("Kettlefield, jane@" + "mailbox.net\n", encoding="utf-8")

    assert cp.main([]) == 1
    captured = capsys.readouterr()
    assert "only the general rules ran" in captured.err
    assert captured.out.count("\n") == 1


def test_the_repository_itself_is_clean(capsys: pytest.CaptureFixture[str]) -> None:
    assert cp.main([]) == 0, capsys.readouterr().out
