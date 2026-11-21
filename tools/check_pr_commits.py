#!/usr/bin/env python3
"""Checks the commit subjects and, when required, the DCO sign-offs of a pull request.

Usage: python3 -B tools/check_pr_commits.py BASE HEAD [--signoff exempt|required]

Reads every non-merge commit in BASE..HEAD. Each subject must be a Conventional Commit with one of
the types in CONTRIBUTING.md, at most 72 characters. With `--signoff required`, each commit also
needs a `Signed-off-by:` trailer naming its author: the name exactly, the email ignoring letter
case. Exits 0 when every commit passes, 1 when one does not, 2 on a usage error or a history it
cannot read (a shallow checkout). Run in CI by `.github/workflows/pull-request-commits.yml`.
Standard library only.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

# The types CONTRIBUTING.md lists; the pull request title check and the commit-msg hook carry the
# same list, and tests/test_check_pr_commits.py holds all four to it.
TYPES: Final = (
    "feat",
    "fix",
    "perf",
    "refactor",
    "docs",
    "test",
    "build",
    "ci",
    "chore",
    "style",
    "revert",
)
SUBJECT: Final = re.compile(rf"(?:{'|'.join(TYPES)})(?:\([^()\s]+\))?!?: \S.*")
MAX_SUBJECT: Final = 72
SIGNER: Final = re.compile(r"(?P<name>.*) <(?P<email>[^<>]*)>")

SUBJECT_HINT: Final = (
    f"Write each subject as 'type(scope): summary', at most {MAX_SUBJECT} characters, with type",
    f"one of: {', '.join(TYPES)}. Change the last commit with git commit --amend and an earlier",
    "one with git rebase -i BASE (BASE is the branch the pull request targets).",
)
SIGNOFF_HINT: Final = (
    "A Signed-off-by trailer must name the commit's author: the name exactly, the email ignoring",
    "letter case (CONTRIBUTING.md, 'Licence of contributions'). git adds it with the name and",
    "email you commit with now, which is the author's only when they are the same person.",
    "- Your own commits, made under the name and email git is set to now: git commit --amend",
    "  --no-edit --signoff for the last one, or git rebase --signoff BASE for several.",
    "- Your own commits under another name or email: set user.name and user.email, then",
    "  git rebase --exec 'git commit --amend --no-edit --signoff --reset-author' BASE.",
    "  --reset-author makes you the author, so use it only on commits you wrote.",
    "- Commits another person wrote: they add their own sign-off.",
)


class Violation(NamedTuple):
    """One problem with one commit."""

    sha: str
    subject: str
    kind: str  # "subject" or "signoff"
    author: str


def git(repo: Path, *args: str) -> str:
    """Runs git in `repo` with no inherited `GIT_*` setting, which could point it at another one."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="surrogateescape",
    ).stdout


def subject_ok(subject: str) -> bool:
    """Whether a subject is a Conventional Commit with a listed type, short enough."""
    return len(subject) <= MAX_SUBJECT and SUBJECT.fullmatch(subject) is not None


def signed_off(name: str, email: str, trailers: str) -> bool:
    """Whether a trailer names the author: the name exactly, the email ignoring letter case."""
    for line in trailers.split("\n"):
        signer = SIGNER.fullmatch(line)
        if signer and signer["name"] == name and signer["email"].lower() == email.lower():
            return True
    return False


def violations(repo: Path, base: str, head: str, *, require_signoff: bool) -> list[Violation]:
    """The problems of every non-merge commit in base..head, oldest first."""
    commits = git(
        repo, "rev-list", "--no-merges", "--reverse", "--end-of-options", f"{base}..{head}", "--"
    )
    found: list[Violation] = []
    for sha in commits.splitlines():
        metadata = git(
            repo,
            "show",
            "--no-patch",
            "--no-show-signature",
            "--encoding=UTF-8",
            "--format=%s%x00%an%x00%ae%x00%(trailers:key=Signed-off-by,valueonly,unfold)",
            sha,
            "--",
        )
        subject, name, email, trailers = metadata.split("\0", 3)
        author = f"{name} <{email}>"
        if not subject_ok(subject):
            found.append(Violation(sha, subject, "subject", author))
        if require_signoff and not signed_off(name, email, trailers):
            found.append(Violation(sha, subject, "signoff", author))
    return found


def plain(text: str) -> str:
    """Text from a commit, safe to print: a control character becomes U+FFFD."""
    return "".join(char if char.isprintable() else "�" for char in text)


def report(found: Sequence[Violation]) -> list[str]:
    """One line per problem, starting with the commit's id, then the hints.

    The subject and author come from the pull request's author. They are printed after the id,
    never at the start of a line, so none can start a workflow command such as `::error::`.
    """
    lines: list[str] = []
    for v in found:
        if v.kind == "subject":
            lines.append(f"{v.sha} {plain(v.subject)} - the subject is not 'type(scope): summary'.")
        else:
            lines.append(
                f"{v.sha} {plain(v.subject)} - no 'Signed-off-by: {plain(v.author)}' trailer."
            )
    for kind, hint in (("subject", SUBJECT_HINT), ("signoff", SIGNOFF_HINT)):
        if any(v.kind == kind for v in found):
            lines.extend(("", *hint))
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Checks BASE..HEAD from the command line; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("base", help="the commit the pull request starts from")
    parser.add_argument("head", help="the pull request's last commit")
    parser.add_argument(
        "--signoff",
        choices=("exempt", "required"),
        default="exempt",
        help="require a Signed-off-by trailer naming each commit's author (default: exempt)",
    )
    args = parser.parse_args(argv)
    repo = Path.cwd()
    try:
        if git(repo, "rev-parse", "--is-shallow-repository").strip() == "true":
            parser.exit(
                2, "This checkout is shallow: fetch the full history (git fetch --unshallow).\n"
            )
        found = violations(repo, args.base, args.head, require_signoff=args.signoff == "required")
    except subprocess.CalledProcessError:
        parser.exit(
            2, "Could not read the commits: check that both refs and their history are here.\n"
        )
    for line in report(found):
        print(line, file=sys.stderr)
    if found:
        return 1
    print(
        "Every commit subject passes"
        + (" and every commit is signed off." if args.signoff == "required" else ".")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
