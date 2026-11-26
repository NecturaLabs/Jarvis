#!/usr/bin/env python3
# SPDX-FileCopyrightText: NecturaLabs and the Jarvis contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Print affected test files from changed paths or the Git index, from the repository root."""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

TESTS = Path("tests")
PACKAGE = Path("src/jarvis")
CI_ONLY = {TESTS / "conftest.py", Path("pyproject.toml"), Path("uv.lock")}


def _staged_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"],
        cwd=root,
        stdout=subprocess.PIPE,
        check=True,
    )
    return os.fsdecode(result.stdout).split("\0")


def _existing_paths(root: Path, names: Iterable[str]) -> set[Path]:
    paths: set[Path] = set()
    for name in names:
        candidate = root / name
        if candidate.is_relative_to(root):
            path = candidate.relative_to(root)
            if ".." not in path.parts and candidate.is_file():
                paths.add(path)
    return paths


def _is_test(path: Path) -> bool:
    return path.parent == TESTS and path.name.startswith("test_") and path.suffix == ".py"


def _module(path: Path) -> str:
    parts = path.parent.parts if path.name == "__init__.py" else path.with_suffix("").parts
    return ".".join(parts)


def _imports(path: Path, source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level <= len(path.parent.parts):
            package = path.parts[: -node.level] if node.level else ()
            module = ".".join((*package, node.module)) if node.module else ".".join(package)
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return names


def _helper_dependents(imports: dict[Path, set[str]], helpers: set[Path]) -> set[Path]:
    seen = set(helpers)
    while helpers:
        names = {
            name
            for path in helpers
            for name in (_module(path), _module(path.relative_to(TESTS)))
            if name
        }
        helpers = {
            path
            for path, modules in imports.items()
            if path not in seen and _mentions_module(modules, names)
        }
        seen.update(helpers)
    return seen


def _mentions_module(imports: set[str], modules: set[str]) -> bool:
    return any(
        name == module or name.startswith(f"{module}.") for name in imports for module in modules
    )


def _select_tests(root: Path, changed: set[Path]) -> list[Path]:
    source_paths = {path.relative_to("src") for path in changed if path.is_relative_to(PACKAGE)}
    modules = {_module(path) for path in source_paths if path.suffix == ".py"}
    assets = {path for path in source_paths if path.suffix != ".py"}
    helpers = {path for path in changed if path.is_relative_to(TESTS) and path.suffix == ".py"}
    if not (modules or assets or helpers):
        return []

    packages = {".".join(path.parent.parts) for path in assets}
    modules.update(packages)
    data_mentions = [re.compile(rf"(?<![\w.]){re.escape(name)}(?!\w)") for name in sorted(packages)]
    data_mentions.extend(
        re.compile(rf"(?<![\w.-]){re.escape(name)}(?![\w-]|\.\w)")
        for name in sorted({path.name for path in assets})
    )
    imports: dict[Path, set[str]] = {}
    selected: set[Path] = set()
    test_dir = root / TESTS
    files = test_dir.rglob("*.py") if helpers else test_dir.glob("test_*.py")
    for file in sorted(files):
        path = file.relative_to(root)
        if path in CI_ONLY or not file.is_file() or file.is_symlink():
            continue
        source = file.read_text(encoding="utf-8")
        names = _imports(path, source)
        imports[path] = names
        if _is_test(path) and (
            _mentions_module(names, modules)
            or any(pattern.search(source) for pattern in data_mentions)
        ):
            selected.add(path)

    selected.update(path for path in _helper_dependents(imports, helpers) if _is_test(path))
    return sorted(selected, key=Path.as_posix)


def main(argv: Sequence[str] | None = None) -> None:
    """Print the selected files and describe CI-only changes on stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", metavar="path", help="changed files")
    parser.add_argument(
        "--staged", action="store_true", help="read changed paths from the Git index"
    )
    parser.add_argument(
        "--null", action="store_true", help="end each result with NUL for shell use"
    )
    args = parser.parse_args(argv)
    if args.staged and args.paths:
        parser.error("pass changed paths or --staged")

    root = Path.cwd()
    changed = _existing_paths(root, _staged_paths(root) if args.staged else args.paths)
    if ci_only := changed & CI_ONLY:
        files = ", ".join(sorted(path.as_posix() for path in ci_only))
        print(
            f"affected tests: CI covers {files}. These paths add no pre-commit tests.",
            file=sys.stderr,
        )

    for path in _select_tests(root, changed - CI_ONLY):
        print(path.as_posix(), end="\0" if args.null else "\n")


if __name__ == "__main__":
    main()
