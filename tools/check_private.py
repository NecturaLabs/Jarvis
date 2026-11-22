#!/usr/bin/env python3
"""Checks that nothing about one person or one machine goes into the public repository.

Usage:
  uv run python tools/check_private.py             every tracked file, as it is on disk
  uv run python tools/check_private.py --staged    the lines a commit adds (pre-commit hook)
  uv run python tools/check_private.py --message F a commit message file (commit-msg hook)
  uv run python tools/check_private.py --push      what a push sends, refs on stdin (pre-push hook)

Prints one `place:line: problem` per finding and exits 1 when there is any (0 when clean, 2 on a
usage error). Every commit is published, so the check refuses:

  - an e-mail address outside the example domains and the project's own addresses;
  - a path inside a real home folder (`/home/<name>`, `/Users/<name>`) other than a placeholder;
  - this machine's host name, the running user's home folder, and the name and e-mail of the
    global git identity, all read when the check runs, so none of them is written down here;
  - any term in the maintainer's private list: one term per line (`#` starts a comment), matched
    without regard to case as a whole word, or a regular expression after `re:`. The list lives in
    `$JARVIS_PRIVATE_TERMS`, else in `info/private-terms` in the repository's git folder, shared by
    its worktrees and never committed. Put there what is personal but looks ordinary: names of
    your own games, apps and drives, places, accounts, and the example phrases you said.

A finding from the private list names the list's line, not the term, so the term does not reach
a log. Without the list only the other rules run (CI has none). Standard library only.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, override

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

ROOT: Final = Path(__file__).resolve().parent.parent

# Managed by agentic-setup, its own project with its own rules (AGENTS.md, Boundaries).
EXCLUDED: Final = (".lint/",)

EMAIL: Final = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ALLOWED_EMAIL: Final = re.compile(
    r"""(?ix)
    .*@(?:[a-z0-9-]+\.)*(?:example|invalid|test|localhost)(?:\.[a-z]+)?$  # RFC 2606 names
    | .*@(?:[a-z0-9-]+\.)*example\.(?:com|net|org)$
    | .*@users\.noreply\.github\.com$
    | (?:support|noreply)@github\.com$  # GitHub's bots, Dependabot's sign-off among them
    | .*@necturalabs\.com$
    | .*@\d+\.service$  # a systemd unit such as user@1000.service
    """
)
HOME_PATH: Final = re.compile(r"/(?:home|Users)/([A-Za-z0-9._-]+)")
PLACEHOLDER_USERS: Final = frozenset(
    {"a", "alice", "b", "bob", "example", "me", "name", "owner", "someone", "user", "you"}
)
ZERO_SHA: Final = re.compile(r"^0+$")
SHORTEST_TERM: Final = 3  # a shorter host name or git identity would match ordinary words
ROOT_DEPTH: Final = 2  # a home folder at least two levels down, never `/` itself


@dataclass(frozen=True)
class Rule:
    """One pattern that must not appear, and what to say when it does."""

    pattern: re.Pattern[str]
    problem: str


@dataclass(frozen=True)
class Finding:
    """One line that names something private."""

    place: str
    line: int
    problem: str

    @override
    def __str__(self) -> str:
        return f"{self.place}:{self.line}: {self.problem}"


def _git(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, input=stdin, capture_output=True, text=True, check=True
    ).stdout


def _word(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE)


def terms_file() -> Path:
    """Where the private list is read from."""
    named = os.environ.get("JARVIS_PRIVATE_TERMS")
    if named:
        return Path(named)
    common = _git("rev-parse", "--git-common-dir").strip()
    return (ROOT / common).resolve() / "info" / "private-terms"


def private_rules(text: str) -> list[Rule]:
    """The private list's rules; each names its line, never the term."""
    rules: list[Rule] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        term = raw.strip()
        if not term or term.startswith("#"):
            continue
        problem = f"a term from the private list (its line {number})"
        if term.startswith("re:"):
            rules.append(Rule(re.compile(term[3:], re.IGNORECASE), problem))
        else:
            rules.append(Rule(_word(term), problem))
    return rules


def _machine_rules() -> list[Rule]:
    rules: list[Rule] = []
    host = socket.gethostname().split(".")[0]
    if len(host) >= SHORTEST_TERM and host.casefold() not in {"localhost", "runner"}:
        rules.append(Rule(_word(host), "this machine's host name"))
    home = Path.home().as_posix()
    if home.count("/") >= ROOT_DEPTH:
        rules.append(Rule(re.compile(re.escape(home) + r"(?![A-Za-z0-9._-])"), "your home folder"))
    for key, what in (("user.email", "e-mail"), ("user.name", "name")):
        try:
            value = _git("config", "--global", "--get", key).strip()
        except subprocess.CalledProcessError:
            continue
        if len(value) >= SHORTEST_TERM and not ALLOWED_EMAIL.fullmatch(value):
            rules.append(Rule(_word(value), f"the {what} of your global git identity"))
    return rules


def find(place: str, lines: Iterable[tuple[int, str]], rules: Sequence[Rule]) -> Iterator[Finding]:
    """Findings in numbered lines of one place."""
    for number, line in lines:
        for address in EMAIL.findall(line):
            if not ALLOWED_EMAIL.fullmatch(address):
                yield Finding(place, number, "an e-mail address outside the example domains")
        for user in HOME_PATH.findall(line):
            if user.casefold() not in PLACEHOLDER_USERS:
                yield Finding(place, number, "a path in a real home folder; use /home/me")
        for rule in rules:
            if rule.pattern.search(line):
                yield Finding(place, number, rule.problem)


def _excluded(path: str) -> bool:
    return path.startswith(EXCLUDED)


def tracked(rules: Sequence[Rule]) -> Iterator[Finding]:
    """Every tracked file as it is on disk."""
    for path in _git("ls-files", "-z").split("\0"):
        if not path or _excluded(path):
            continue
        try:
            text = (ROOT / path).read_text(encoding="utf-8")
        except UnicodeDecodeError, FileNotFoundError, IsADirectoryError:
            continue
        yield from find(path, enumerate(text.splitlines(), start=1), rules)


def added_lines(diff: str) -> Iterator[tuple[str, int, str]]:
    """(path, line in the new file, text) for each line a unified diff adds."""
    path = ""
    number = 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            path = target[2:] if target.startswith("b/") else ""
        elif line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            number = int(match.group(1)) if match else 0
        elif line.startswith("+") and path:
            yield path, number, line[1:]
            number += 1
        elif line.startswith(" "):
            number += 1


def in_diff(diff: str, rules: Sequence[Rule], prefix: str = "") -> Iterator[Finding]:
    """Findings in the lines a unified diff adds."""
    for path, number, text in added_lines(diff):
        if not _excluded(path):
            yield from find(prefix + path, [(number, text)], rules)


def in_message(place: str, text: str, rules: Sequence[Rule]) -> Iterator[Finding]:
    """Findings in a commit message, leaving out git's `#` comment lines."""
    lines = [(n, line) for n, line in enumerate(text.splitlines(), start=1)]
    yield from find(place, [(n, line) for n, line in lines if not line.startswith("#")], rules)


def pushed(refs: str, rules: Sequence[Rule]) -> Iterator[Finding]:
    """Each commit a push sends: its message and the lines it adds against its first parent."""
    for line in refs.splitlines():
        match line.split():
            case [_, local, _, remote] if not ZERO_SHA.match(local):
                pass
            case _:
                continue  # a deleted ref sends nothing
        if ZERO_SHA.match(remote):
            commits = _git("rev-list", local, "--not", "--remotes").split()
        else:
            commits = _git("rev-list", f"{remote}..{local}").split()
        for commit in commits:
            short = commit[:9]
            yield from in_message(
                f"{short} message", _git("log", "-1", "--format=%B", commit), rules
            )
            diff = _git(
                "show", "--format=", "-p", "-U0", "--no-color", "--no-ext-diff",
                "--diff-merges=first-parent", commit,
            )  # fmt: skip
            yield from in_diff(diff, rules, prefix=f"{short} ")


def main(argv: Sequence[str]) -> int:
    """Run the check `argv` names; the exit status the module docstring gives."""
    path = terms_file()
    rules = _machine_rules()
    if path.is_file():
        rules += private_rules(path.read_text(encoding="utf-8"))
    else:
        print(f"note: no private list at {path}; only the general rules ran", file=sys.stderr)
    match list(argv):
        case []:
            findings = list(tracked(rules))
        case ["--staged"]:
            diff = _git("diff", "--cached", "-U0", "--no-color", "--no-ext-diff")
            findings = list(in_diff(diff, rules))
        case ["--message", message]:
            text = Path(message).read_text(encoding="utf-8")
            findings = list(in_message("commit message", text, rules))
        case ["--push"]:
            findings = list(pushed(sys.stdin.read(), rules))
        case _:
            print(__doc__, file=sys.stderr)
            return 2
    for finding in findings:
        print(finding)
    if findings:
        print(
            f"{len(findings)} private detail(s): keep the repository generic (AGENTS.md).",
            file=sys.stderr,
        )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
