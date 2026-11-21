"""Invariants of .github/workflows/*.yml, read as text.

No YAML parser is installed and adding one would be a dependency, so this file carries a small
parser for exactly the YAML these workflows use. Its limits, so nobody trusts it with more:

- Block style only: nested mappings and sequences by indentation (spaces; a tab is an error), a
  sequence may sit at its key's indentation, and `- key: value` starts a mapping inside a sequence.
- Scalars are kept as raw text. Flow collections (`{}`, `[a, b]`) are one scalar, not parsed. A
  scalar may continue on deeper lines (joined by a space); `|` and `>` block scalars keep their
  lines as written, comments included (Actions expands `${{ }}` in a shell comment too).
- A trailing `# comment` is split off a scalar unless the `#` is inside quotes or not preceded by a
  space. An apostrophe in a plain scalar can hide a comment after it.
- Not supported, and an error rather than a guess: anchors and aliases, tags, multiple documents,
  quoted scalars that span lines, keys that are not plain or quoted words, tabs.
- Only `run:` of a step is checked for expressions. `with: script:` of github-script, composite
  actions under `.github/actions` and expressions built from `github['event']` are not read.

Each invariant is a function from a workflow's text to a list of problems, so it is run on the real
workflows and, in its own test, shown to fail on a synthetic bad one.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / ".github" / "workflows"
WORKFLOW_FILES = sorted([*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")])


class ParseError(Exception):
    """The text uses YAML this parser does not read."""


@dataclass(frozen=True)
class Scalar:
    """A scalar's raw text, its trailing comment, and where it is.

    `line` is the line of its key; `body_line` is the line its text starts on (the line after a
    `|` or `>` indicator, else the same).
    """

    text: str
    comment: str
    line: int
    body_line: int


@dataclass(frozen=True)
class Mapping:
    items: dict[str, Node]
    key_lines: dict[str, int]
    line: int


@dataclass(frozen=True)
class Sequence:
    items: list[Node]
    line: int


type Node = Scalar | Mapping | Sequence

_KEY = re.compile(r"""("[^"]*"|'[^']*'|[A-Za-z0-9_][A-Za-z0-9_./-]*):(?:[ ]+(.*))?""")
_BLOCK_INDICATOR = re.compile(r"[|>](?:[+-]?\d?|\d[+-])")
_UNSUPPORTED = re.compile(r"^(?:&|\*|!|---|\.\.\.)")


def split_comment(text: str) -> tuple[str, str]:
    """`value # note` as (`value`, `note`); a `#` in quotes or with no space before it stays."""
    quote = ""
    for i, char in enumerate(text):
        if quote:
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
        elif char == "#" and (i == 0 or text[i - 1] in " \t"):
            return text[:i].rstrip(), text[i + 1 :].strip()
    return text.rstrip(), ""


def unquote(text: str) -> str:
    if len(text) > 1 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    return text


@dataclass
class _Parser:
    lines: list[str]
    pos: int = 0

    def indent_of(self, i: int) -> int:
        line = self.lines[i]
        return len(line) - len(line.lstrip(" "))

    def significant(self) -> int | None:
        """The index of the next line that is neither blank nor a comment, without consuming it."""
        for i in range(self.pos, len(self.lines)):
            stripped = self.lines[i].strip()
            if stripped and not stripped.startswith("#"):
                return i
        return None

    def is_item(self, i: int) -> bool:
        content = self.lines[i].strip()
        return content == "-" or content.startswith("- ")

    def block(self) -> Node:
        i = self.significant()
        if i is None:
            raise ParseError("a block was expected at the end of the file")
        if self.is_item(i):
            return self.sequence(self.indent_of(i))
        return self.mapping(self.indent_of(i))

    def mapping(self, indent: int) -> Mapping:
        items: dict[str, Node] = {}
        key_lines: dict[str, int] = {}
        first = (self.significant() or 0) + 1
        while (i := self.significant()) is not None and self.indent_of(i) >= indent:
            if self.indent_of(i) > indent:
                raise ParseError(f"line {i + 1}: unexpected indentation")
            if self.is_item(i):
                break
            content = self.lines[i].strip()
            match = _KEY.fullmatch(content)
            if match is None:
                raise ParseError(f"line {i + 1}: not a `key: value` line: {content!r}")
            key = unquote(match[1])
            self.pos = i + 1
            items[key] = self.value(indent, match[2] or "", i + 1)
            key_lines[key] = i + 1
        return Mapping(items, key_lines, first)

    def sequence(self, indent: int) -> Sequence:
        items: list[Node] = []
        first = (self.significant() or 0) + 1
        while (
            (i := self.significant()) is not None
            and self.indent_of(i) == indent
            and self.is_item(i)
        ):
            content = self.lines[i].strip()
            rest = content[1:].lstrip(" ")
            column = indent + len(content) - len(rest)
            if not rest or rest.startswith("#"):
                self.pos = i + 1
                j = self.significant()
                if j is not None and self.indent_of(j) > indent:
                    items.append(self.block())
                else:
                    items.append(Scalar("", "", i + 1, i + 1))
            elif _KEY.fullmatch(rest):
                self.lines[i] = " " * column + rest  # the item's mapping starts on this line
                self.pos = i
                items.append(self.mapping(column))
            else:
                self.pos = i + 1
                items.append(self.value(indent, rest, i + 1))
        return Sequence(items, first)

    def value(self, indent: int, rest: str, number: int) -> Node:
        text, comment = split_comment(rest)
        if _UNSUPPORTED.match(text):
            raise ParseError(f"line {number}: anchors, aliases, tags and documents are not read")
        if _BLOCK_INDICATOR.fullmatch(text):
            return self.block_scalar(indent, number, comment)
        if not text:
            i = self.significant()
            if i is not None and (
                self.indent_of(i) > indent or (self.indent_of(i) == indent and self.is_item(i))
            ):
                return self.block()
            return Scalar("", comment, number, number)
        parts = [text]
        while (i := self.significant()) is not None and self.indent_of(i) > indent:
            parts.append(split_comment(self.lines[i].strip())[0])
            self.pos = i + 1
        return Scalar(" ".join(parts), comment, number, number)

    def block_scalar(self, indent: int, number: int, comment: str) -> Scalar:
        body: list[str] = []
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if line.strip() and self.indent_of(self.pos) <= indent:
                break
            body.append(line)
            self.pos += 1
        filled = [line for line in body if line.strip()]
        strip = min((len(line) - len(line.lstrip(" ")) for line in filled), default=0)
        while body and not body[-1].strip():
            body.pop()
        return Scalar("\n".join(line[strip:] for line in body), comment, number, number + 1)


def parse(text: str) -> Mapping:
    """The workflow as a tree of Mapping, Sequence and Scalar.

    Raises:
        ParseError: the text uses YAML this parser does not read.
    """
    lines = text.split("\n")
    for number, line in enumerate(lines, 1):
        if line[: len(line) - len(line.lstrip())].count("\t"):
            raise ParseError(f"line {number}: tab indentation")
    parser = _Parser(lines)
    root = parser.block()
    if parser.significant() is not None or not isinstance(root, Mapping):
        raise ParseError("the file is not a single mapping")
    return root


def walk(node: Node) -> Iterator[tuple[str, Node, int]]:
    """Every mapping entry below `node`, as (key, value, line of the key)."""
    if isinstance(node, Mapping):
        for key, value in node.items.items():
            yield key, value, node.key_lines[key]
            yield from walk(value)
    elif isinstance(node, Sequence):
        for item in node.items:
            yield from walk(item)


def jobs(workflow: Mapping) -> dict[str, Mapping]:
    node = workflow.items.get("jobs")
    if not isinstance(node, Mapping):
        raise ParseError("no `jobs:` mapping")
    found: dict[str, Mapping] = {}
    for job_id, job in node.items.items():
        if not isinstance(job, Mapping):
            raise ParseError(f"job {job_id} is not a mapping")
        found[job_id] = job
    return found


def steps(job: Mapping) -> list[Mapping]:
    node = job.items.get("steps")
    if not isinstance(node, Sequence):
        return []
    return [step for step in node.items if isinstance(step, Mapping)]


# The invariants: each takes a workflow's file name and text, and returns its problems.

HOSTED_RUNNER = re.compile(r"ubuntu-[0-9a-z][0-9a-z.-]*")
PINNED_ACTION = re.compile(r"[\w.-]+/[\w.-]+(?:/[\w./-]+)?@[0-9a-f]{40}")
VERSION_COMMENT = re.compile(r"v\d+(?:\.\d+)*(?:\s|$)")
EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)
UNTRUSTED = re.compile(r"(?<![\w.])(?:github\.event(?!\w)|github\.head_ref(?!\w)|inputs\.)")


def permission_problems(name: str, text: str) -> list[str]:
    node = parse(text).items.get("permissions")
    if isinstance(node, Scalar) and node.text == "{}":
        return []
    return [f"{name}: the top level has no `permissions: {{}}`"]


def runner_problems(name: str, text: str) -> list[str]:
    problems: list[str] = []
    for job_id, job in jobs(parse(text)).items():
        node = job.items.get("runs-on")
        if isinstance(node, Scalar) and HOSTED_RUNNER.fullmatch(unquote(node.text)):
            continue
        line = job.key_lines.get("runs-on", job.line)
        problems.append(f"{name}:{line}: job {job_id}: runs-on is not one literal ubuntu-* label")
    return problems


def pin_problems(name: str, text: str) -> list[str]:
    problems: list[str] = []
    for key, node, line in walk(parse(text)):
        if key != "uses" or not isinstance(node, Scalar):
            continue
        action = unquote(node.text)
        if action.startswith("./") and "@" not in action:
            continue  # a local action in this repository cannot carry a SHA
        if not PINNED_ACTION.fullmatch(action):
            problems.append(f"{name}:{line}: `uses: {action}` is not pinned to a 40-hex commit SHA")
        elif not VERSION_COMMENT.match(node.comment):
            problems.append(f"{name}:{line}: `uses: {action}` has no `# vX` version comment")
    return problems


def checkout_problems(name: str, text: str) -> list[str]:
    problems: list[str] = []
    for job in jobs(parse(text)).values():
        for step in steps(job):
            uses = step.items.get("uses")
            if not (
                isinstance(uses, Scalar) and unquote(uses.text).startswith("actions/checkout@")
            ):
                continue
            options = step.items.get("with")
            persist = (
                options.items.get("persist-credentials") if isinstance(options, Mapping) else None
            )
            if not (isinstance(persist, Scalar) and unquote(persist.text) == "false"):
                problems.append(
                    f"{name}:{uses.line}: actions/checkout without `persist-credentials: false`"
                )
    return problems


def injection_problems(name: str, text: str) -> list[str]:
    problems: list[str] = []
    for job in jobs(parse(text)).values():
        for step in steps(job):
            run = step.items.get("run")
            if not isinstance(run, Scalar):
                continue
            for match in EXPRESSION.finditer(run.text):
                found = UNTRUSTED.search(match[1])
                if found:
                    line = run.body_line + run.text.count("\n", 0, match.start())
                    message = f"`{found[0]}` is used inside a run: block; pass it through env:"
                    problems.append(f"{name}:{line}: {message}")
    return problems


# A workflow with every invariant met, and a way to break one at a time.

SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"


@dataclass(frozen=True)
class Variant:
    permissions: str = "permissions: {}"
    runs_on: str = "runs-on: ubuntu-latest"
    uses: str = f"uses: actions/checkout@{SHA} # v7.0.1"
    persist: str = "persist-credentials: false"
    run: list[str] = field(default_factory=lambda: ['echo "hello $NAME"'])


def workflow(variant: Variant | None = None) -> str:
    v = variant or Variant()
    lines = [
        "name: Synthetic",
        "",
        "on:",
        "  push:",
        "",
        v.permissions,
        "",
        "jobs:",
        "  check:",
        f"    {v.runs_on}",
        "    steps:",
        f"      - {v.uses}",
        "        with:",
        f"          {v.persist}",
        "      - name: Say it",
        "        env:",
        "          NAME: ${{ github.event.pull_request.title }}",
        "        run: |",
        *(f"          {line}" for line in v.run),
    ]
    return "\n".join(lines) + "\n"


def test_the_synthetic_good_workflow_meets_every_invariant() -> None:
    text = workflow()
    for check in (
        permission_problems,
        runner_problems,
        pin_problems,
        checkout_problems,
        injection_problems,
    ):
        assert check("good.yml", text) == [], check.__name__


# The parser.


def test_the_parser_reads_the_shapes_the_workflows_use() -> None:
    text = "\n".join(
        [
            "# a header comment",
            "name: Demo # trailing",
            "on:",
            "  push:",
            '    branches: [main, "feature/**"]',
            "  schedule:",
            '    - cron: "17 6 * * 1"',
            "  workflow_dispatch:",
            "permissions: {}",
            "jobs:",
            "  a:",
            "    if: >-",
            "      x == 1 &&",
            "      y == 2",
            "    steps:",
            "      # a comment between steps",
            "      - uses: o/r@" + SHA + " # v1.2.3",
            "        with:",
            "          k: v",
            "      - name: Run",
            "        run: |",
            "          # shell comment",
            "          echo ${{ a }}",
            "",
            "          echo done",
            "      - run: echo inline",
            "    list:",
            "    - one",
            "    - two",
        ]
    )
    root = parse(text)
    assert root.items["name"] == Scalar("Demo", "trailing", 2, 2)
    on = root.items["on"]
    assert isinstance(on, Mapping)
    assert on.items["workflow_dispatch"] == Scalar("", "", 8, 8)
    push = on.items["push"]
    assert isinstance(push, Mapping)
    assert push.items["branches"] == Scalar('[main, "feature/**"]', "", 5, 5)
    assert root.items["permissions"] == Scalar("{}", "", 9, 9)
    job = jobs(root)["a"]
    assert job.items["if"] == Scalar("x == 1 &&\ny == 2", "", 12, 13)
    first, second, third = steps(job)
    assert first.items["uses"] == Scalar("o/r@" + SHA, "v1.2.3", 17, 17)
    assert second.items["run"] == Scalar("# shell comment\necho ${{ a }}\n\necho done", "", 21, 22)
    assert third.items["run"] == Scalar("echo inline", "", 26, 26)
    assert job.items["list"] == Sequence([Scalar("one", "", 28, 28), Scalar("two", "", 29, 29)], 28)


def test_a_plain_scalar_may_continue_on_deeper_lines() -> None:
    root = parse("a: first part # note\n  second part\nb: next\n")
    assert root.items["a"] == Scalar("first part second part", "note", 1, 1)
    assert root.items["b"] == Scalar("next", "", 3, 3)


def test_a_hash_inside_quotes_or_without_a_space_is_not_a_comment() -> None:
    assert split_comment('"a # b" # note') == ('"a # b"', "note")
    assert split_comment("https://example.invalid/x#frag") == ("https://example.invalid/x#frag", "")
    assert split_comment("value") == ("value", "")


@pytest.mark.parametrize(
    "text",
    [
        "a:\n\tb: 1\n",
        "a: &anchor 1\n",
        "a: *alias\n",
        "a: !tag x\n",
        "just text\n",
        "a: 1\n  b: 2\nc\n",
    ],
)
def test_yaml_it_does_not_read_is_an_error_not_a_guess(text: str) -> None:
    with pytest.raises(ParseError):
        parse(text)


def test_there_are_workflows_and_the_parser_finds_their_jobs_and_steps() -> None:
    assert len(WORKFLOW_FILES) >= 1
    for path in WORKFLOW_FILES:
        found = jobs(parse(path.read_text(encoding="utf-8")))
        assert found, path.name
        assert all(steps(job) for job in found.values()), path.name


# Invariant 1: every workflow starts with no permissions.


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_every_workflow_has_empty_top_level_permissions(path: Path) -> None:
    assert permission_problems(path.name, path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "permissions",
    ["permissions: write-all", "permissions: read-all", "permissions:\n  contents: read", "# none"],
)
def test_a_workflow_without_empty_permissions_is_caught(permissions: str) -> None:
    problems = permission_problems("bad.yml", workflow(Variant(permissions=permissions)))
    assert problems == ["bad.yml: the top level has no `permissions: {}`"]


def test_a_job_level_permissions_block_does_not_stand_in_for_the_top_level() -> None:
    text = workflow(Variant(permissions="# none")).replace(
        "    steps:", "    permissions: {}\n    steps:"
    )
    assert len(permission_problems("bad.yml", text)) == 1


# Invariant 2: every job runs on a GitHub-hosted Ubuntu label.


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_every_job_runs_on_a_github_hosted_label(path: Path) -> None:
    assert runner_problems(path.name, path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "runs_on",
    [
        "runs-on: self-hosted",
        "runs-on: [self-hosted, linux]",
        "runs-on: windows-latest",
        "runs-on: ${{ matrix.os }}",
        "runs-on:\n      group: my-runners",
        "name: no runs-on at all",
    ],
)
def test_a_job_off_a_hosted_ubuntu_label_is_caught(runs_on: str) -> None:
    problems = runner_problems("bad.yml", workflow(Variant(runs_on=runs_on)))
    assert len(problems) == 1
    assert "job check: runs-on" in problems[0]


@pytest.mark.parametrize(
    "label", ["ubuntu-latest", "ubuntu-24.04", "ubuntu-24.04-arm", '"ubuntu-22.04"']
)
def test_hosted_ubuntu_labels_are_accepted(label: str) -> None:
    assert runner_problems("good.yml", workflow(Variant(runs_on=f"runs-on: {label}"))) == []


# Invariant 3: every action is pinned to a commit SHA, with a version comment.


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_sha_with_a_version_comment(path: Path) -> None:
    assert pin_problems(path.name, path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "uses",
    [
        "uses: actions/checkout@v4 # v4",
        "uses: actions/checkout@main",
        f"uses: actions/checkout@{SHA[:39]} # v7.0.1",
        f"uses: actions/checkout@{SHA.upper()} # v7.0.1",
        "uses: docker://alpine:3.20",
        "uses: ./.github/actions/setup@main",
        f"uses: actions/checkout@{SHA}",
        f"uses: actions/checkout@{SHA} # latest",
        f"uses: actions/checkout@{SHA} # the v7 line",
    ],
)
def test_an_unpinned_action_or_one_without_a_version_comment_is_caught(uses: str) -> None:
    problems = pin_problems("bad.yml", workflow(Variant(uses=uses)))
    assert len(problems) == 1
    assert problems[0].startswith("bad.yml:12: `uses: ")


def test_a_local_action_is_not_a_pin_problem() -> None:
    assert pin_problems("good.yml", workflow(Variant(uses="uses: ./.github/actions/setup"))) == []


def test_a_job_level_reusable_workflow_must_be_pinned_too() -> None:
    text = workflow().replace(
        "    steps:", "    uses: o/r/.github/workflows/w.yml@main\n    steps:"
    )
    assert len(pin_problems("bad.yml", text)) == 1


# Invariant 4: checkout does not leave the token in the repository's git config.


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_every_checkout_sets_persist_credentials_false(path: Path) -> None:
    assert checkout_problems(path.name, path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "persist",
    ["persist-credentials: true", "fetch-depth: 0", "persist-credentials: ${{ inputs.keep }}"],
)
def test_a_checkout_that_keeps_credentials_is_caught(persist: str) -> None:
    problems = checkout_problems("bad.yml", workflow(Variant(persist=persist)))
    assert problems == ["bad.yml:12: actions/checkout without `persist-credentials: false`"]


def test_a_checkout_with_no_with_block_is_caught() -> None:
    text = workflow().replace("        with:\n          persist-credentials: false\n", "")
    assert len(checkout_problems("bad.yml", text)) == 1


def test_only_checkout_steps_are_held_to_the_checkout_rule() -> None:
    uses = f"uses: actions/upload-artifact@{SHA} # v7.0.2"
    assert checkout_problems("good.yml", workflow(Variant(uses=uses, persist="path: x"))) == []


# Invariant 5: no untrusted expression inside a run: block; it goes through env:.


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_no_run_block_expands_an_untrusted_expression(path: Path) -> None:
    assert injection_problems(path.name, path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    ("expression", "found"),
    [
        ("github.event.pull_request.title", "github.event"),
        ("github.event.head_commit.message", "github.event"),
        ("toJSON(github.event)", "github.event"),
        ("github.head_ref", "github.head_ref"),
        ("inputs.name", "inputs."),
        ("github.event_name == 'push' && inputs.name", "inputs."),
        ("format('{0}', github.head_ref)", "github.head_ref"),
    ],
)
def test_an_untrusted_expression_in_a_run_block_is_caught(expression: str, found: str) -> None:
    run = ["echo start", f"echo ${{{{ {expression} }}}}"]
    problems = injection_problems("bad.yml", workflow(Variant(run=run)))
    assert problems == [
        f"bad.yml:20: `{found}` is used inside a run: block; pass it through env:",
    ]


def test_an_expression_in_a_shell_comment_or_without_spaces_is_still_caught() -> None:
    for run in (["# ${{ github.head_ref }}"], ["echo ${{github.event.number}}"]):
        assert len(injection_problems("bad.yml", workflow(Variant(run=run)))) == 1


def test_an_inline_run_is_read_too() -> None:
    text = workflow().replace('        run: |\n          echo "hello $NAME"\n', "")
    text += "        run: echo ${{ github.head_ref }}\n"
    assert injection_problems("bad.yml", text) == [
        "bad.yml:18: `github.head_ref` is used inside a run: block; pass it through env:"
    ]


@pytest.mark.parametrize(
    "run",
    [
        'echo "$NAME"',
        "echo ${{ github.event_name }}",
        "echo ${{ github.repository }} ${{ github.run_id }} ${{ runner.temp }}",
        "echo ${{ secrets.TOKEN }}",
        "echo inputs.name github.head_ref",
    ],
)
def test_trusted_expressions_and_plain_words_are_not_flagged(run: str) -> None:
    assert injection_problems("good.yml", workflow(Variant(run=[run]))) == []


def test_untrusted_expressions_are_fine_in_env_and_if() -> None:
    text = workflow().replace(
        "      - name: Say it",
        "      - name: Say it\n        if: github.event.pull_request.draft == false",
    )
    assert "github.event.pull_request.title" in text
    assert injection_problems("good.yml", text) == []
