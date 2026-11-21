"""tools/check_pr_commits.py: a pull request's commit subjects and sign-offs, on throwaway repos."""

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import check_pr_commits as cpc
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "check_pr_commits.py"
AUTHOR = ("Test Author", "author@example.invalid")
SHA_LINE = re.compile(r"[0-9a-f]{40} ")


@dataclass
class Repo:
    path: Path

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.path, check=True, capture_output=True, encoding="utf-8"
        ).stdout

    def commit(self, message: str, *args: str, signed: bool = False) -> str:
        sign = ["-s"] if signed else []
        self.git("commit", "-q", "--allow-empty", "--no-verify", "-m", message, *sign, *args)
        return self.git("rev-parse", "HEAD").strip()

    def found(self, base: str, head: str, *, require_signoff: bool) -> list[tuple[str, str]]:
        return [
            (v.sha, v.kind)
            for v in cpc.violations(self.path, base, head, require_signoff=require_signoff)
        ]

    def cli(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), *args],
            cwd=cwd or self.path,
            check=False,
            capture_output=True,
            encoding="utf-8",
        )


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    """A repository with one commit on `main` and the branch `topic` checked out from it."""
    r = Repo(tmp_path / "repo")
    r.path.mkdir()
    r.git("init", "-q", "--template=", "-b", "main")
    r.git("config", "user.name", AUTHOR[0])
    r.git("config", "user.email", AUTHOR[1])
    r.git("config", "commit.gpgsign", "false")
    r.commit("chore: start")
    r.git("switch", "-qc", "topic")
    return r


def test_a_conventional_signed_off_commit_passes(repo: Repo) -> None:
    repo.commit("feat(spotify)!: queue several songs in one request", signed=True)
    assert repo.found("main", "HEAD", require_signoff=True) == []
    done = repo.cli("main", "HEAD", "--signoff", "required")
    assert (done.returncode, done.stderr) == (0, "")


@pytest.mark.parametrize(
    "subject",
    [
        "Add a change without a type",
        "feature: a type CONTRIBUTING.md does not list",
        "Feat: a type in capitals",
        "fix:no space after the colon",
        "fix: " + "x" * 68,  # 73 characters
        "fixup! feat: squash me before merging",
        "",
    ],
)
def test_a_subject_outside_the_listed_form_is_flagged(repo: Repo, subject: str) -> None:
    sha = repo.commit(subject or "placeholder", signed=True)
    if not subject:
        repo.git("commit", "-q", "--amend", "--allow-empty", "--allow-empty-message", "-m", "")
        sha = repo.git("rev-parse", "HEAD").strip()
    assert repo.found("main", "HEAD", require_signoff=False) == [(sha, "subject")]
    done = repo.cli("main", "HEAD")
    assert done.returncode == 1
    assert [line.split(" ", 1)[0] for line in done.stderr.splitlines() if SHA_LINE.match(line)] == [
        sha
    ]


def test_a_72_character_subject_passes(repo: Repo) -> None:
    repo.commit("fix: " + "x" * 67, signed=True)
    assert repo.found("main", "HEAD", require_signoff=True) == []


def test_every_listed_type_passes(repo: Repo) -> None:
    for kind in cpc.TYPES:
        repo.commit(f"{kind}(scope): do the thing", signed=True)
    assert repo.found("main", "HEAD", require_signoff=True) == []


def test_a_missing_sign_off_is_flagged_only_when_required(repo: Repo) -> None:
    sha = repo.commit("docs: add an example")
    assert repo.found("main", "HEAD", require_signoff=True) == [(sha, "signoff")]
    assert repo.found("main", "HEAD", require_signoff=False) == []
    assert repo.cli("main", "HEAD").returncode == 0
    done = repo.cli("main", "HEAD", "--signoff", "required")
    assert done.returncode == 1
    assert f"Signed-off-by: {AUTHOR[0]} <{AUTHOR[1]}>" in done.stderr


@pytest.mark.parametrize(
    "trailer",
    [
        "Signed-off-by: Another Author <other@example.invalid>",
        f"Signed-off-by: {AUTHOR[0]} <other@example.invalid>",
        f"Signed-off-by: Another Author <{AUTHOR[1]}>",
        f"Signed-off-by: {AUTHOR[0].lower()} <{AUTHOR[1]}>",
    ],
)
def test_a_sign_off_for_another_person_is_flagged(repo: Repo, trailer: str) -> None:
    sha = repo.commit("fix: correct the example", "-m", trailer)
    assert repo.found("main", "HEAD", require_signoff=True) == [(sha, "signoff")]


def test_a_sign_off_in_the_body_rather_than_the_trailers_is_flagged(repo: Repo) -> None:
    sha = repo.commit(
        "test: cover the example",
        "-m",
        f"Signed-off-by: {AUTHOR[0]} <{AUTHOR[1]}>\n\nThis is the last paragraph.",
    )
    assert repo.found("main", "HEAD", require_signoff=True) == [(sha, "signoff")]


def test_an_email_differing_only_in_letter_case_passes(repo: Repo) -> None:
    repo.commit(
        "fix: sign with another case", "-m", f"Signed-off-by: {AUTHOR[0]} <AUTHOR@Example.Invalid>"
    )
    assert repo.found("main", "HEAD", require_signoff=True) == []


def test_a_non_ascii_author_who_signs_off_passes(repo: Repo) -> None:
    author = "Zoë Müller <zoe@example.invalid>"
    repo.commit("feat: a non-ASCII author", "--author", author, "-m", f"Signed-off-by: {author}")
    assert repo.found("main", "HEAD", require_signoff=True) == []


@pytest.mark.parametrize(
    "subject",
    ["::error::file=x::not a command", "Bad subject\r::error::boom\x1b[31m", "feat: ok\x07bell"],
)
def test_a_hostile_subject_is_printed_safely(repo: Repo, subject: str) -> None:
    sha = repo.commit(subject)
    done = repo.cli("main", "HEAD", "--signoff", "required")
    assert done.returncode == 1
    lines = done.stderr.splitlines()
    assert [line for line in lines if not SHA_LINE.match(line) and "::" in line] == []
    assert [line for line in lines if line.startswith("::")] == []
    assert [c for c in done.stderr if c != "\n" and not c.isprintable()] == []
    assert any(line.startswith(f"{sha} ") for line in lines)


def test_merge_commits_are_skipped(repo: Repo) -> None:
    repo.git("switch", "-q", "main")
    base = repo.commit("chore: move main on", signed=True)
    repo.git("switch", "-q", "topic")
    work = repo.commit("feat: the work", signed=True)
    repo.git(
        "merge", "-q", "--no-ff", "--no-verify", "-m", "Merge branch 'main' into topic", "main"
    )
    assert repo.git("rev-parse", "HEAD").strip() != work
    assert repo.found(base, "HEAD", require_signoff=True) == []


def test_only_commits_after_the_base_are_read(repo: Repo) -> None:
    base = repo.commit("Old style subject on the base")
    repo.commit("feat: new work", signed=True)
    assert repo.found(base, "HEAD", require_signoff=True) == []


def test_unknown_refs_and_bad_options_are_usage_errors(repo: Repo) -> None:
    done = repo.cli("no-such-ref", "HEAD")
    assert (done.returncode, done.stdout) == (2, "")
    assert repo.cli("main", "HEAD", "--signoff", "sometimes").returncode == 2


def test_a_shallow_checkout_is_refused(repo: Repo, tmp_path: Path) -> None:
    repo.commit("feat: the work", signed=True)
    shallow = tmp_path / "shallow"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--depth",
            "1",
            "--branch",
            "topic",
            repo.path.as_uri(),
            str(shallow),
        ],
        check=True,
        capture_output=True,
    )
    done = repo.cli("HEAD", "HEAD", cwd=shallow)
    assert done.returncode == 2
    assert "shallow" in done.stderr


def types_in(text: str) -> tuple[str, ...]:
    match = re.search(r"types='([a-z|]+)'", text)
    assert match is not None
    return tuple(match.group(1).split("|"))


def test_the_title_check_the_hook_and_contributing_list_the_same_types() -> None:
    title = (ROOT / ".github" / "workflows" / "pull-request-title.yml").read_text(encoding="utf-8")
    hook = (ROOT / "tools" / "hooks" / "commit-msg").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    listed = re.search(r"^- Types: (.+)\.$", contributing, re.MULTILINE)
    assert listed is not None
    assert types_in(title) == cpc.TYPES
    assert types_in(hook) == cpc.TYPES
    assert tuple(re.findall(r"`([a-z]+)`", listed.group(1))) == cpc.TYPES
