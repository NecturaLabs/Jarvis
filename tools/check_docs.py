#!/usr/bin/env python3
"""Checks that the maintained documentation still matches the repository.

Usage: uv run python tools/check_docs.py

Prints one `path:line: problem` per finding and exits 1 when there is any (0 when clean, 2 on a
usage error). It reads only the files git tracks, never untracked ones. The rules:

  1. every `docs/**/*.md` outside `docs/adr/` is linked from `docs/README.md`
  2. every `docs/adr/NNNN-*.md` is linked from `docs/adr/README.md`
  3. every relative Markdown link in a tracked `.md` file names a tracked file or directory
  4. every backticked repository path in a tracked `.md` file exists: one that starts with
     `src/`, `tests/`, `tools/`, `docs/`, `.github/`, `.claude/`, `.codex/` or `.agents/`, or a
     bare top-level document name such as `AGENTS.md`. Text with `<`, `*`, `{` or a space is a
     pattern or a sentence, not a path, and is skipped
  5. every module `src/jarvis/**/*.py` (not `__init__.py`) is named in `docs/architecture.md`
  6. every file under `tools/` is named in `tools/README.md` (files under `tools/hooks/` may be
     covered by naming the directory)
  7. every `.claude/skills/<name>/SKILL.md` has a byte-identical `.agents/skills/<name>/SKILL.md`
     and the other way round
  8. every `.claude/agents/<name>.md` has a `.codex/agents/<name>.toml` with the same name,
     description and instructions, and the other way round

A tracked symbolic link is reported and never followed. A rule whose index file does not exist yet
is reported as a finding, never skipped. Code fences and inline code are not read for links.
Standard library only.
"""

from __future__ import annotations

import posixpath
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import unquote

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

ROOT: Final = Path(__file__).resolve().parent.parent

# Where a finding is not wanted, with the reason. A key is a path (or fnmatch pattern) of a tracked
# file whose links and paths rules 3 and 4 skip entirely, or `path:pattern` to skip only the link or
# backticked text that matches the pattern in that file.
EXCLUDED: Final = {
    "CHANGELOG.md": "history: it names files and links that have since moved or gone",
    "docs/playbook/practices-audit.md:docs/INTEGRATIONS.md": (
        "Voxtype's file, named as the place its integration contract is documented"
    ),
    "docs/playbook/practices-audit.md:docs/TERMINAL.md": (
        "Strata's file, named as the place its terminal notes are documented"
    ),
    "docs/playbook/practices-audit.md:CONFIGURATION.md": "Voxtype's file, named as a source",
    "docs/playbook/practices-audit.md:DETAILS.md": "Strata's file, named as a source",
    "docs/playbook/practices-audit.md:AMD_HIP.md": "Strata's file, named as a source",
}

# The index files the other rules read.
DOCS_INDEX: Final = "docs/README.md"
ADR_INDEX: Final = "docs/adr/README.md"
ARCHITECTURE: Final = "docs/architecture.md"
TOOLS_INDEX: Final = "tools/README.md"
ADR_DIR: Final = "docs/adr/"
MODULE_ROOT: Final = "src/jarvis/"

# A backticked text that starts with one of these is a path from the repository root.
ROOT_PREFIXES: Final = (
    "src/",
    "tests/",
    "tools/",
    "docs/",
    ".github/",
    ".claude/",
    ".codex/",
    ".agents/",
)
# A bare name like these, with no slash, is a top-level document (a `config.toml` is not: it is
# the user's file).
TOP_LEVEL_DOCUMENT: Final = re.compile(r"[A-Z][A-Z0-9_]*\.md")

_ADR_FILE: Final = re.compile(r"docs/adr/\d{4}-[^/]+\.md")
_SCHEME: Final = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
_INLINE_LINK: Final = re.compile(r"\]\(\s*(?:<([^>]*)>|([^)\s]*))")
_LINK_DEFINITION: Final = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(?:<([^>]*)>|(\S+))")
_CODE_SPAN: Final = re.compile(r"(`+)(.+?)(?<!`)\1(?!`)")
_FENCE: Final = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_LINE_SUFFIX: Final = re.compile(r":\d+(?::\d+)?$")
_SKIPPED_CHARS: Final = re.compile(r"[<*{\s]")


@dataclass(frozen=True)
class Tree:
    """The tracked files of a repository, and what is needed to read and look them up."""

    root: Path
    files: frozenset[str]

    @cached_property
    def directories(self) -> frozenset[str]:
        """Every directory that holds a tracked file, as a root-relative path."""
        found: set[str] = set()
        for name in self.files:
            parent = posixpath.dirname(name)
            while parent and parent not in found:
                found.add(parent)
                parent = posixpath.dirname(parent)
        return frozenset(found)

    def exists(self, path: str) -> bool:
        """Whether `path` is a tracked file or a directory holding one."""
        return path in self.files or path in self.directories

    def text(self, path: str) -> str:
        """The content of a tracked file."""
        return (self.root / path).read_text(encoding="utf-8")


def tracked_files(root: Path) -> list[str]:
    """The files git tracks in `root`, without ones deleted from the working tree."""
    out = subprocess.run(
        ["git", "-c", "core.quotePath=false", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(
        f for f in out.split("\0") if f and ((root / f).is_symlink() or (root / f).is_file())
    )


def excluded_file(path: str) -> bool:
    """Whether rules 3 and 4 skip this whole file."""
    return any(fnmatchcase(path, key) for key in EXCLUDED if ":" not in key)


def excluded_text(path: str, text: str) -> bool:
    """Whether `text` (a link target or backticked path) is excluded in this file."""
    for key in EXCLUDED:
        file_key, _, pattern = key.partition(":")
        if pattern and fnmatchcase(path, file_key) and fnmatchcase(text, pattern):
            return True
    return False


def prose_lines(text: str) -> Iterator[tuple[int, str]]:
    """`(line number, line)` for the lines outside fenced code blocks."""
    fence = ""
    for number, line in enumerate(text.splitlines(), 1):
        match = _FENCE.match(line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence):
                fence = ""
        elif match:
            fence = match[1]
        else:
            yield number, line


def code_spans(line: str) -> list[str]:
    """The text of the inline code spans of a line."""
    return [m[2].strip() for m in _CODE_SPAN.finditer(line)]


def link_targets(line: str) -> list[str]:
    """Targets of the inline links and link definitions on a line, outside inline code."""
    plain = _CODE_SPAN.sub("", line)
    targets = [m[1] if m[1] is not None else m[2] for m in _INLINE_LINK.finditer(plain)]
    definition = _LINK_DEFINITION.match(plain)
    if definition:
        targets.append(definition[1] if definition[1] is not None else definition[2])
    return [t for t in targets if t]


def resolve_link(source: str, target: str) -> str | None:
    """The repository path a relative link names, or None for one that is not a repository path.

    A link with a scheme (http, https, mailto, ...), a pure `#anchor` and an empty target are not.
    A path above the repository root resolves to `..`, which names nothing.
    """
    if _SCHEME.match(target) or target.startswith(("#", "//")):
        return None
    path = unquote(re.split(r"[#?]", target, maxsplit=1)[0])
    if not path:
        return None
    base = "" if path.startswith("/") else posixpath.dirname(source)
    return posixpath.normpath(posixpath.join(base, path.lstrip("/")))


def linked_from(tree: Tree, index: str) -> set[str]:
    """Every repository path the Markdown file `index` links to."""
    linked: set[str] = set()
    for _, line in prose_lines(tree.text(index)):
        for target in link_targets(line):
            path = resolve_link(index, target)
            if path is not None:
                linked.add(path)
    return linked


def names(text: str, name: str) -> bool:
    """Whether `name` appears in `text` as a whole name, not as the tail of a longer path."""
    return re.search(r"(?<![\w./-])" + re.escape(name) + r"(?![\w-])", text) is not None


def suffixes(relative: str) -> list[str]:
    """`a/b/c.py` as `a/b/c.py`, `b/c.py` and `c.py`."""
    parts = relative.split("/")
    return ["/".join(parts[i:]) for i in range(len(parts))]


def missing_index(index: str, why: str) -> str:
    """The finding for an index file that is not tracked."""
    return f"{index}:1: missing; {why}"


def check_indexed(tree: Tree) -> list[str]:
    """Rules 1 and 2: docs and ADRs linked from their index."""
    problems: list[str] = []
    for index, why, wanted in (
        (DOCS_INDEX, "every document under docs/ must be linked from it", _is_doc),
        (ADR_INDEX, "every ADR must be linked from it", _is_adr),
    ):
        if index not in tree.files:
            problems.append(missing_index(index, why))
            continue
        linked = linked_from(tree, index)
        problems.extend(
            f"{name}:1: not linked from {index}"
            for name in sorted(tree.files)
            if wanted(name) and name != index and name not in linked
        )
    return problems


def _is_doc(name: str) -> bool:
    return name.startswith("docs/") and name.endswith(".md") and not name.startswith(ADR_DIR)


def _is_adr(name: str) -> bool:
    return _ADR_FILE.fullmatch(name) is not None


def check_references(tree: Tree) -> list[str]:
    """Rules 3 and 4: links and backticked paths in tracked Markdown files."""
    problems: list[str] = []
    for name in sorted(tree.files):
        if not name.endswith(".md") or excluded_file(name):
            continue
        for number, line in prose_lines(tree.text(name)):
            for target in link_targets(line):
                path = resolve_link(name, target)
                if path is None or excluded_text(name, target):
                    continue
                if not tree.exists(path):
                    problems.append(f"{name}:{number}: link to {target} names nothing tracked")
            for span in code_spans(line):
                written = _LINE_SUFFIX.sub("", span)
                if excluded_text(name, written) or _SKIPPED_CHARS.search(written):
                    continue
                path = written.split("#", 1)[0]
                if (
                    path.startswith(ROOT_PREFIXES) or TOP_LEVEL_DOCUMENT.fullmatch(path)
                ) and not tree.exists(posixpath.normpath(path)):
                    problems.append(f"{name}:{number}: `{span}` names nothing tracked")
    return problems


def check_architecture(tree: Tree) -> list[str]:
    """Rule 5: every source module is named in docs/architecture.md."""
    if ARCHITECTURE not in tree.files:
        return [missing_index(ARCHITECTURE, "every module under src/jarvis/ must be named in it")]
    text = tree.text(ARCHITECTURE)
    problems: list[str] = []
    for name in sorted(tree.files):
        if not (name.startswith(MODULE_ROOT) and name.endswith(".py")):
            continue
        relative = name[len(MODULE_ROOT) :]
        if posixpath.basename(relative) == "__init__.py":
            continue
        # `spotify/auth.py` may be written as `auth.py`; a top-level `audit.py` only as itself.
        candidates = [*suffixes(relative), f"jarvis/{relative}", f"{MODULE_ROOT}{relative}"]
        if not any(names(text, c) for c in candidates):
            problems.append(f"{name}:1: not named in {ARCHITECTURE}")
    return problems


def check_tools(tree: Tree) -> list[str]:
    """Rule 6: every file under tools/ is named in tools/README.md."""
    if TOOLS_INDEX not in tree.files:
        return [missing_index(TOOLS_INDEX, "every file under tools/ must be named in it")]
    text = tree.text(TOOLS_INDEX)
    problems: list[str] = []
    for name in sorted(tree.files):
        if not name.startswith("tools/") or name == TOOLS_INDEX:
            continue
        relative = name[len("tools/") :]
        candidates = [*suffixes(relative), f"tools/{relative}"]
        if relative.startswith("hooks/"):
            candidates += ["hooks/", "tools/hooks/"]
        if not any(names(text, c) for c in candidates):
            problems.append(f"{name}:1: not named in {TOOLS_INDEX}")
    return problems


def check_skill_mirror(tree: Tree) -> list[str]:
    """Rule 7: `.claude/skills/<name>/SKILL.md` and `.agents/skills/<name>/SKILL.md` match."""
    pattern = re.compile(r"\.(claude|agents)/skills/([^/]+)/SKILL\.md")
    skills: dict[str, set[str]] = {"claude": set(), "agents": set()}
    for name in tree.files:
        match = pattern.fullmatch(name)
        if match:
            skills[match[1]].add(match[2])
    problems: list[str] = []
    for skill in sorted(skills["claude"] | skills["agents"]):
        claude = f".claude/skills/{skill}/SKILL.md"
        agents = f".agents/skills/{skill}/SKILL.md"
        if skill not in skills["agents"]:
            problems.append(f"{claude}:1: no mirror {agents}")
        elif skill not in skills["claude"]:
            problems.append(f"{agents}:1: no mirror {claude}")
        elif (tree.root / claude).read_bytes() != (tree.root / agents).read_bytes():
            problems.append(f"{agents}:1: differs from {claude}")
    return problems


def _front_matter(text: str) -> tuple[dict[str, str], str] | None:
    """The `key: value` lines between the opening `---` lines, and the body after them."""
    match = re.fullmatch(r"---\n(.*?)\n---\n\n(.*)", text, re.DOTALL)
    if match is None:
        return None
    fields = dict(line.split(": ", 1) for line in match[1].splitlines() if ": " in line)
    return fields, match[2]


def _codex_mismatch(markdown: str, toml_text: str) -> str | None:
    """Why a Codex agent file does not mirror its Claude agent file, or None when it does."""
    parsed = _front_matter(markdown)
    if parsed is None:
        return "its Claude agent has no front matter to compare"
    fields, body = parsed
    try:
        codex = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError as exc:
        return f"is not valid TOML ({exc})"
    for key, want in (("name", fields.get("name")), ("description", fields.get("description"))):
        if codex.get(key) != want:
            return f"{key} differs from the Claude agent"
    if codex.get("developer_instructions") != body:
        return "developer_instructions differ from the Claude agent's instructions"
    return None


def check_agent_mirror(tree: Tree) -> list[str]:
    """Rule 8: `.claude/agents/<name>.md` and `.codex/agents/<name>.toml` say the same."""
    pattern = re.compile(r"\.(claude|codex)/agents/([^/]+)\.(md|toml)")
    agents: dict[str, set[str]] = {"claude": set(), "codex": set()}
    for name in tree.files:
        match = pattern.fullmatch(name)
        if match and (match[1], match[3]) in {("claude", "md"), ("codex", "toml")}:
            agents[match[1]].add(match[2])
    problems: list[str] = []
    for agent in sorted(agents["claude"] | agents["codex"]):
        claude = f".claude/agents/{agent}.md"
        codex = f".codex/agents/{agent}.toml"
        if agent not in agents["codex"]:
            problems.append(f"{claude}:1: no mirror {codex}")
        elif agent not in agents["claude"]:
            problems.append(f"{codex}:1: no mirror {claude}")
        elif (why := _codex_mismatch(tree.text(claude), tree.text(codex))) is not None:
            problems.append(f"{codex}:1: {why}")
    return problems


def check(root: Path, files: Iterable[str]) -> list[str]:
    """Every finding for the tracked `files` of the repository at `root`.

    A tracked symbolic link is reported and left out: following one could read a file outside the
    repository, such as a private one.
    """
    names = set(files)
    links = sorted(n for n in names if (root / n).is_symlink())
    tree = Tree(root, frozenset(names - set(links)))
    return [
        *(f"{link}:1: a symbolic link; the docs check does not follow it" for link in links),
        *check_indexed(tree),
        *check_references(tree),
        *check_architecture(tree),
        *check_tools(tree),
        *check_skill_mirror(tree),
        *check_agent_mirror(tree),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks; 0 when clean, 1 with findings, 2 on a usage error."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["-h"], ["--help"]):
        print((__doc__ or "").strip())
        return 0
    if args:
        print("usage: check_docs.py (takes no arguments)", file=sys.stderr)
        return 2
    files = tracked_files(ROOT)
    problems = check(ROOT, files)
    for problem in problems:
        print(problem)
    print(f"check_docs: {len(files)} tracked files, {len(problems)} problem(s)", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
