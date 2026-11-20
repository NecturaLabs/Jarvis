#!/usr/bin/env python3
"""Fail any GitHub workflow job whose runs-on does not require a self-hosted runner.

Usage: ci_runner_guard.py [WORKFLOW_DIR_OR_FILE ...]   (default: .github/workflows, from the cwd)

A job passes only when every label form it can resolve to contains "self-hosted": a bare label, a
flow list, a block list (also under ``labels:``), or a fromJSON() expression of either. Anything
conditional or indirect (a matrix value, ``cond && 'self-hosted' || 'ubuntu-latest'``) fails, since
one branch of it could reach a hosted runner. Exit 0: every job passes. 1: a job fails, each printed
as ``path:line``. 2: a named path does not exist.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DEFAULT_DIR = ".github/workflows"
RUNS_ON = re.compile(r"^(\s*)runs-on:\s*(.*)$")
EXPRESSION = re.compile(r"\$\{\{")  # no literal brace pair: lint.py reads one as a placeholder
FROM_JSON = re.compile(r"\$\{\{\s*fromJSON\(\s*(['\"])(.*?)\1\s*\)\s*\}\}")
LIST_MARKER = re.compile(r"^(-\s+|labels:\s*)")


def _unquote(text: str) -> str:
    text = text.strip()
    if len(text) > 1 and text[0] in {"'", '"'} and text[-1] == text[0]:
        return text[1:-1]
    return text


def requires_self_hosted(parts: list[str]) -> bool:
    """Say whether a runs-on value (its inline text, then each nested line) is self-hosted only."""
    parts = [part.strip() for part in parts if part.strip()]
    if not parts:
        return False
    text = " ".join(parts)
    if EXPRESSION.match(text):
        match = FROM_JSON.fullmatch(text)
        if not match:
            return False
        try:
            labels = json.loads(match.group(2))
        except json.JSONDecodeError:
            return False
        return labels == "self-hosted" or (isinstance(labels, list) and "self-hosted" in labels)
    items: list[str] = []
    for part in parts:
        items.extend(LIST_MARKER.sub("", part).strip("[]").split(","))
    return any(_unquote(item) == "self-hosted" for item in items)


def violations(path: Path) -> list[int]:
    """Return the line numbers of the runs-on keys in a workflow that do not require self-hosted."""
    # shortcut: a '#' always starts a comment, so a label containing one is cut; no label does.
    found: list[int] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        match = RUNS_ON.match(line)
        if not match or match.group(2).startswith("#"):
            continue
        indent = len(match.group(1))
        parts = [match.group(2).split("#", 1)[0]]
        for following in lines[number:]:
            if not following.strip():
                continue
            if len(following) - len(following.lstrip()) <= indent:
                break
            parts.append(following.split("#", 1)[0])
        if not requires_self_hosted(parts):
            found.append(number)
    return found


def main(argv: list[str]) -> int:
    """Check the named workflow files and folders, or the default folder."""
    targets = [Path(arg) for arg in argv] or [Path(DEFAULT_DIR)]
    files: list[Path] = []
    for target in targets:
        if target.is_dir():
            files.extend(sorted(target.glob("*.y*ml")))
        elif target.is_file():
            files.append(target)
        elif argv:
            print(f"{target}: no such file or folder", file=sys.stderr)
            return 2
    failed = False
    for path in files:
        for number in violations(path):
            failed = True
            print(f"{path}:{number}: runs-on must require self-hosted on every branch")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
