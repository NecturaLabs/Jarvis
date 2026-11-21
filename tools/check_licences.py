#!/usr/bin/env python3
"""Checks that every dependency can be distributed with Jarvis under GPL-3.0-or-later.

Usage: uv run python tools/check_licences.py

Prints one `path:line: problem` per finding and exits 1 when there is any (0 when clean, 2 on a
usage error):

  - a distribution installed in the running environment (except Jarvis itself) whose licence is
    missing, unreadable, or has a term outside ALLOWED. The licence is read from
    `License-Expression`, else from `License` when that is a short SPDX-like expression, else from
    the `License ::` classifiers. A distribution whose metadata yields none of these needs an
    entry in OVERRIDES.
  - a package in uv.lock that does not come from the PyPI registry or from the project itself.

Expressions: `OR` passes when any branch passes, `AND` needs every branch, `X WITH exception` is
judged by X (an exception only adds permissions), and AND binds tighter than OR. Ids compare
case-insensitively, and a legacy `GPL-2.0+` or `LGPL-2.1+` means `-or-later`. Standard library
only.
"""

from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

ROOT: Final = Path(__file__).resolve().parent.parent

# Licences compatible with distributing Jarvis under GPL-3.0-or-later (the FSF's GPL-compatibility
# list). Ones marked "unsure" could not be confirmed against that list from memory; they are
# permissive licences that every reading of it admits.
ALLOWED: Final = frozenset(
    {
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "Apache-2.0",  # compatible with GPL-3.0 (not with GPL-2.0-only)
        "ISC",
        "PSF-2.0",  # unsure: the FSF lists the Python 2.0.1 and newer licence, SPDX splits it
        "Python-2.0",  # unsure: as PSF-2.0; the incompatible ones are Python 1.6 to 2.1
        "MPL-2.0",  # compatible through its secondary-license clause (no "Incompatible" notice)
        "GPL-3.0-only",
        "GPL-3.0-or-later",
        "GPL-2.0-or-later",
        "0BSD",
        "Zlib",
        "CC0-1.0",
        "Unlicense",
        "MIT-CMU",  # unsure: a permissive HPND variant, not named on the FSF list
        "HPND",  # unsure: as MIT-CMU
    }
)
_ALLOWED_FOLDED: Final = frozenset(i.casefold() for i in ALLOWED)
# LGPL 2.0, 2.1 and 3.0 all let their code be combined into a GPL-3.0 work.
_LGPL: Final = re.compile(r"lgpl-\d(?:\.\d)?(?:-only|-or-later)?")

# One distribution may carry a licence the allow-list does not, with the reason. Like a cargo-deny
# `[[licenses.exceptions]]`: it widens only that distribution, and only by the ids named.
EXCEPTIONS: Final[dict[str, tuple[frozenset[str], str]]] = {
    "reuse": (
        frozenset({"CC-BY-SA-4.0", "LicenseRef-Proprietary-Classifier"}),
        (
            "REUSE's METADATA says its documentation (not code) is CC-BY-SA-4.0, which is one-way "
            "compatible with GPL-3.0 (unsure: from the CC and FSF statements, not read here); it "
            "also carries `License :: Other/Proprietary License` beside the free licences of its "
            "License field, unexplained. It is a development tool, never distributed with Jarvis"
        ),
    ),
    "regex": (
        frozenset({"CNRI-Python"}),
        (
            "the owner accepted it on 2026-11-21 so Jarvis can read web pages with trafilatura "
            "(trafilatura -> htmldate -> dateparser -> regex). LICENSE.txt in its dist-info says "
            "the code derives from CPython 2.6/3.1's `re`, under CNRI's Python 1.6 licence; SPDX "
            "lists a separate CNRI-Python-GPL-Compatible id, so this one is treated as "
            "GPL-incompatible. Users install it from PyPI; Jarvis ships no copy"
        ),
    ),
}


@dataclass(frozen=True)
class Override:
    """A licence for a distribution whose metadata says nothing usable, and where it was checked."""

    licence: str
    reason: str


# Keyed by the normalised distribution name (lower case, runs of - _ . as one -). Used only when
# the metadata gives no usable licence; one the metadata has outgrown is reported for removal.
OVERRIDES: Final[dict[str, Override]] = {
    "justext": Override(
        "BSD-2-Clause",
        (
            "only the `BSD License` classifier and `License: The BSD 2-Clause License`; "
            "LICENSE.rst in justext-3.0.2.dist-info is the two-clause BSD text (source and binary "
            "clauses, no endorsement or advertising clause)"
        ),
    ),
    "jinja2": Override(
        "BSD-3-Clause",
        (
            "only the `BSD License` classifier; LICENSE.txt in jinja2-3.1.6.dist-info/licenses is "
            "the three-clause BSD text ('Neither the name of the copyright holder ...')"
        ),
    ),
    "nodeenv": Override(
        "BSD-3-Clause",
        (
            "only the `BSD License` classifier; LICENSE in nodeenv-1.12.0.dist-info/licenses is a "
            "three-clause BSD text (source, binary and no-endorsement clauses, no advertising one)"
        ),
    ),
    "protobuf": Override(
        "BSD-3-Clause",
        (
            "metadata is `License: 3-Clause BSD License` with no classifier; the LICENSE file in "
            "protobuf-7.36.2.dist-info is the three-clause BSD text (it has the clause 'Neither "
            "the name of Google Inc. ...')"
        ),
    ),
}

# Classifiers that name no single licence the check can judge: "BSD License" covers the 4-clause
# licence too, which is not GPL-compatible, so a distribution with only that needs an OVERRIDES
# entry read from its LICENSE file. "Other/Proprietary" is a finding wherever it appears.
AMBIGUOUS_CLASSIFIERS: Final = frozenset({"License :: OSI Approved :: BSD License"})
PROPRIETARY_CLASSIFIER: Final = "License :: Other/Proprietary License"
# An EXCEPTIONS id that lets one distribution carry the proprietary classifier.
PROPRIETARY_ID: Final = "LicenseRef-Proprietary-Classifier"

# `License ::` classifiers that name a licence, as the id they stand for. A classifier that is only
# "OSI Approved" or "DFSG approved" names none and is ignored; an unlisted one is a finding.
CLASSIFIERS: Final = {
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)": (
        "GPL-3.0-or-later"
    ),
    "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)": (
        "GPL-2.0-or-later"
    ),
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0-only",
    "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)": (
        "LGPL-3.0-or-later"
    ),
    "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)": (
        "LGPL-2.0-or-later"
    ),
    "License :: OSI Approved :: zlib/libpng License": "Zlib",
    "License :: OSI Approved :: The Unlicense (Unlicense)": "Unlicense",
    "License :: OSI Approved :: Historical Permission Notice and Disclaimer (HPND)": "HPND",
    "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication": "CC0-1.0",
}
NEUTRAL_CLASSIFIERS: Final = frozenset({"License :: OSI Approved", "License :: DFSG approved"})

PYPI: Final = "https://pypi.org/simple"
PYPI_FILES: Final = "https://files.pythonhosted.org/"
# `source` tables of the project itself in uv.lock.
PROJECT_SOURCES: Final = ({"editable": "."}, {"virtual": "."})

_ID_CHARS: Final = re.compile(r"[A-Za-z0-9.+-]+")
_OPERATORS: Final = frozenset({"and", "or", "with"})


@dataclass(frozen=True)
class Term:
    """One licence id."""

    id: str


@dataclass(frozen=True)
class Op:
    """`AND` (every child must pass) or `OR` (any child may)."""

    kind: str
    children: tuple[Node, ...]


type Node = Term | Op


class InvalidExpressionError(Exception):
    """The text is not an SPDX licence expression."""


@dataclass
class _Parser:
    """Recursive descent: OR of ANDs of ids, each id optionally followed by WITH an exception."""

    tokens: list[str]
    position: int = 0

    def peek(self) -> str:
        """The next token in lower case, or an empty string at the end."""
        return self.tokens[self.position].casefold() if self.position < len(self.tokens) else ""

    def take(self) -> str:
        if self.position >= len(self.tokens):
            raise InvalidExpressionError
        self.position += 1
        return self.tokens[self.position - 1]

    def licence_id(self) -> Term:
        token = self.take()
        if token.casefold() in _OPERATORS or not _ID_CHARS.fullmatch(token):
            raise InvalidExpressionError
        return Term(token)

    def primary(self) -> Node:
        if self.peek() == "(":
            self.take()
            inner = self.or_expression()
            if self.take() != ")":
                raise InvalidExpressionError
            return inner
        term = self.licence_id()
        if self.peek() == "with":
            self.take()
            self.licence_id()  # the exception only adds permissions: the licence alone decides
        return term

    def and_expression(self) -> Node:
        children = [self.primary()]
        while self.peek() == "and":
            self.take()
            children.append(self.primary())
        return children[0] if len(children) == 1 else Op("AND", tuple(children))

    def or_expression(self) -> Node:
        children = [self.and_expression()]
        while self.peek() == "or":
            self.take()
            children.append(self.and_expression())
        return children[0] if len(children) == 1 else Op("OR", tuple(children))


def parse_expression(text: str) -> Node | None:
    """The expression as a tree, or None when `text` is not an SPDX expression."""
    parser = _Parser(re.findall(r"[()]|[^\s()]+", text))
    try:
        tree = parser.or_expression()
    except InvalidExpressionError:
        return None
    return tree if parser.position == len(parser.tokens) else None


def terms(node: Node) -> list[str]:
    """Every licence id in the tree, in order."""
    if isinstance(node, Term):
        return [node.id]
    return [t for child in node.children for t in terms(child)]


def passes(node: Node, allowed: Callable[[str], bool]) -> bool:
    """Whether the expression is satisfiable using only ids `allowed` accepts."""
    if isinstance(node, Term):
        return allowed(node.id)
    check = any if node.kind == "OR" else all
    return check(passes(child, allowed) for child in node.children)


def _fold(licence_id: str) -> str:
    folded = licence_id.casefold()
    if folded.endswith("+") and folded.startswith(("gpl-", "lgpl-")):
        folded = folded[:-1] + "-or-later"
    return folded


def is_allowed(licence_id: str, extra: frozenset[str] = frozenset()) -> bool:
    """Whether `licence_id` is on the allow-list (plus `extra`, one distribution's exceptions)."""
    folded = _fold(licence_id)
    return (
        folded in _ALLOWED_FOLDED
        or _LGPL.fullmatch(folded) is not None
        or folded in {_fold(e) for e in extra}
    )


def _spdx_like(licence_id: str) -> bool:
    """An id the `License` field may be trusted to mean: allowed, or shaped like one with a version.

    Free text such as `BSD`, `MIT License` or `PSFL` is not, and falls through to the classifiers.
    """
    return is_allowed(licence_id) or re.search(r"-\d", licence_id) is not None


@dataclass(frozen=True)
class Dist:
    """The metadata of one installed distribution that matters here."""

    name: str
    version: str
    license_expression: str | None
    license: str | None
    classifiers: tuple[str, ...]
    metadata_path: str


@dataclass(frozen=True)
class Resolved:
    """A distribution's licence: the expression, where it came from, or why there is none."""

    node: Node | None
    text: str
    how: str
    problem: str | None = None


def normalise(name: str) -> str:
    """PEP 503 name normalisation."""
    return re.sub(r"[-_.]+", "-", name).casefold()


def resolve(dist: Dist) -> Resolved:
    """The licence the metadata states: `License-Expression`, then `License`, then classifiers."""
    if dist.license_expression and dist.license_expression.strip():
        text = dist.license_expression.strip()
        node = parse_expression(text)
        if node is None:
            return Resolved(
                None, text, "License-Expression", f"License-Expression `{text}` is not valid SPDX"
            )
        return Resolved(node, text, "License-Expression")
    text = " ".join((dist.license or "").split())
    if text:
        node = parse_expression(text)
        if node is not None and all(_spdx_like(t) for t in terms(node)):
            return Resolved(node, text, "License")
    return _from_classifiers(dist)


def _from_classifiers(dist: Dist) -> Resolved:
    """The licence the `License ::` classifiers name, or the problem with them."""
    named = [
        c for c in dist.classifiers if c.startswith("License ::") and c not in NEUTRAL_CLASSIFIERS
    ]
    ambiguous = [c for c in named if c in AMBIGUOUS_CLASSIFIERS]
    if ambiguous:
        return Resolved(
            None, "", "classifiers", f"licence classifier {ambiguous[0]!r} is ambiguous"
        )
    unknown = [c for c in named if c not in CLASSIFIERS]
    if unknown:
        return Resolved(None, "", "classifiers", f"unmapped licence classifier {unknown[0]!r}")
    if named:
        # Several classifiers might be alternatives or all apply; require all, the cautious reading.
        ids = tuple(Term(CLASSIFIERS[c]) for c in named)
        node = ids[0] if len(ids) == 1 else Op("AND", ids)
        return Resolved(node, " AND ".join(t.id for t in ids), "classifiers")
    return Resolved(
        None, "", "metadata", "no licence in License-Expression, License or the classifiers"
    )


def check_distributions(dists: Iterable[Dist], project: str) -> list[str]:
    """Findings for installed distributions other than the project itself."""
    problems: list[str] = []
    seen: set[str] = set()
    for dist in dists:
        key = normalise(dist.name)
        if key == normalise(project) or key in seen:
            continue
        seen.add(key)
        where = f"{dist.metadata_path}:1: {dist.name} {dist.version}"
        resolved = resolve(dist)
        override = OVERRIDES.get(key)
        if resolved.problem is None and override is not None:
            problems.append(
                f"{where}: the override in tools/check_licences.py is no longer needed; remove it"
            )
        if resolved.problem is not None:
            node = parse_expression(override.licence) if override else None
            if node is None:
                problems.append(
                    f"{where}: {resolved.problem}; add an OVERRIDES entry (read its LICENSE)"
                )
                continue
            resolved = Resolved(node, override.licence if override else "", "override")
        node = resolved.node
        if node is None:
            continue
        extra = EXCEPTIONS.get(key, (frozenset[str](), ""))[0]
        if PROPRIETARY_CLASSIFIER in dist.classifiers and PROPRIETARY_ID not in extra:
            problems.append(f"{where}: has the classifier {PROPRIETARY_CLASSIFIER!r}")
            continue
        if not passes(node, lambda i, extra=extra: is_allowed(i, extra)):
            bad = sorted({t for t in terms(node) if not is_allowed(t, extra)})
            problems.append(
                f"{where}: licence `{resolved.text}` (from {resolved.how}) is not allowed "
                f"for distribution with GPL-3.0-or-later: {', '.join(bad)}"
            )
    return problems


def _foreign_downloads(package: dict[str, Any]) -> list[str]:
    """The sdist and wheel urls of a lock entry that are not PyPI's file host.

    uv downloads from the url written beside each file, whatever the entry's `source` says.
    """
    files: list[object] = [package.get("sdist")]
    wheels: object = package.get("wheels")
    if isinstance(wheels, list):
        files.extend(cast("list[object]", wheels))
    foreign: list[str] = []
    for entry in files:
        if isinstance(entry, dict):
            url = cast("dict[str, object]", entry).get("url")
            if not (isinstance(url, str) and url.startswith(PYPI_FILES)):
                foreign.append(str(url))
    return foreign


def check_lock(text: str, project: str) -> list[str]:
    """Findings for uv.lock packages that are not from PyPI or the project itself."""
    lines = text.splitlines()
    line_of: dict[str, int] = {}
    for number, line in enumerate(lines[:-1], 1):
        if line == "[[package]]":
            match = re.fullmatch(r'name = "([^"]+)"', lines[number])
            if match:
                line_of[normalise(match[1])] = number + 1
    problems: list[str] = []
    for package in tomllib.loads(text).get("package", []):
        name = str(package.get("name", ""))
        source = package.get("source")
        line = line_of.get(normalise(name), 1)
        problems.extend(
            f"uv.lock:{line}: package {name} downloads from {url}; only {PYPI_FILES} is allowed"
            for url in _foreign_downloads(package)
        )
        if source == {"registry": PYPI}:
            continue
        if source in PROJECT_SOURCES and normalise(name) == normalise(project):
            continue
        shown = "no source" if source is None else f"source {source}"
        problems.append(
            f"uv.lock:{line}: package {name} has {shown}; only {PYPI} and the project are allowed"
        )
    return problems


def installed() -> list[Dist]:
    """The distributions of the running environment."""
    found: list[Dist] = []
    for distribution in metadata.distributions():
        meta = distribution.metadata
        path = next((f for f in distribution.files or () if f.name == "METADATA"), None)
        found.append(
            Dist(
                name=meta.get("Name") or "(unnamed)",
                version=meta.get("Version") or "?",
                license_expression=meta.get("License-Expression"),
                license=meta.get("License"),
                classifiers=tuple(meta.get_all("Classifier") or ()),
                metadata_path=str(distribution.locate_file(path)) if path else "(unknown)",
            )
        )
    return sorted(found, key=lambda d: normalise(d.name))


def project_name(root: Path) -> str:
    """The `[project] name` of the repository at `root`."""
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["name"])


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks; 0 when clean, 1 with findings, 2 on a usage error."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["-h"], ["--help"]):
        print((__doc__ or "").strip())
        return 0
    if args:
        print("usage: check_licences.py (takes no arguments)", file=sys.stderr)
        return 2
    project = project_name(ROOT)
    dists = installed()
    problems = check_distributions(dists, project)
    problems += check_lock((ROOT / "uv.lock").read_text(encoding="utf-8"), project)
    for problem in problems:
        print(problem)
    print(
        f"check_licences: {len(dists)} installed distributions, {len(problems)} problem(s)",
        file=sys.stderr,
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
