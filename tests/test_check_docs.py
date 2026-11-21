"""tools/check_docs.py: the docs still match the repository, on synthetic trees and the real one."""

import subprocess
from pathlib import Path

import check_docs as cd
import pytest

GOOD: dict[str, str] = {
    "AGENTS.md": "# Agents\n",
    "docs/README.md": "[Plan](plan.md) [Architecture](architecture.md) [ADRs](adr/README.md)\n",
    "docs/plan.md": "# Plan\n",
    "docs/adr/README.md": "# ADRs\n\n- [0001](0001-first.md)\n",
    "docs/adr/0001-first.md": "# First\n",
    "docs/architecture.md": "# Architecture\n\n`audit.py` and `spotify/auth.py`.\n",
    "src/jarvis/__init__.py": "",
    "src/jarvis/audit.py": "",
    "src/jarvis/spotify/__init__.py": "",
    "src/jarvis/spotify/auth.py": "",
    "tools/README.md": "# Tools\n\n`check_docs.py` and the scripts in `hooks/`.\n",
    "tools/check_docs.py": "",
    "tools/hooks/pre-commit": "",
    ".claude/skills/s/SKILL.md": "# Skill\n",
    ".agents/skills/s/SKILL.md": "# Skill\n",
}


def write(root: Path, files: dict[str, str]) -> list[str]:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode())
    return sorted(files)


def findings(tmp_path: Path, changes: dict[str, str | None] | None = None) -> list[str]:
    """The findings for GOOD with `changes` applied; a value of None removes the file."""
    files: dict[str, str] = dict(GOOD)
    for name, content in (changes or {}).items():
        if content is None:
            del files[name]
        else:
            files[name] = content
    return cd.check(tmp_path, write(tmp_path, files))


def only(problems: list[str]) -> str:
    assert len(problems) == 1, problems
    return problems[0]


def test_a_consistent_tree_has_no_findings(tmp_path: Path) -> None:
    assert findings(tmp_path) == []


# Rule 1: docs/ linked from docs/README.md.


def test_a_doc_missing_from_the_docs_index_is_reported(tmp_path: Path) -> None:
    problems = findings(tmp_path, {"docs/extra.md": "# Extra\n"})
    assert only(problems) == "docs/extra.md:1: not linked from docs/README.md"


def test_a_nested_doc_must_be_linked_too(tmp_path: Path) -> None:
    problems = findings(tmp_path, {"docs/playbook/README.md": "# Playbook\n"})
    assert only(problems) == "docs/playbook/README.md:1: not linked from docs/README.md"


def test_a_link_with_an_anchor_counts(tmp_path: Path) -> None:
    index = "[Plan](plan.md#top) [A](architecture.md) [ADRs](adr/README.md)\n"
    assert findings(tmp_path, {"docs/README.md": index}) == []


def test_a_mention_in_code_or_a_fence_does_not_count_as_a_link(tmp_path: Path) -> None:
    index = "[A](architecture.md) [ADRs](adr/README.md)\n`plan.md`\n```\n[Plan](plan.md)\n```\n"
    problems = findings(tmp_path, {"docs/README.md": index})
    assert only(problems) == "docs/plan.md:1: not linked from docs/README.md"


def test_a_missing_docs_index_is_a_finding_not_a_skip(tmp_path: Path) -> None:
    problems = findings(tmp_path, {"docs/README.md": None})
    assert (
        "docs/README.md:1: missing; every document under docs/ must be linked from it" in problems
    )


# Rule 2: ADRs linked from docs/adr/README.md.


def test_an_adr_missing_from_the_adr_index_is_reported(tmp_path: Path) -> None:
    problems = findings(tmp_path, {"docs/adr/0002-second.md": "# Second\n"})
    assert only(problems) == "docs/adr/0002-second.md:1: not linked from docs/adr/README.md"


def test_only_numbered_adrs_need_a_link(tmp_path: Path) -> None:
    assert findings(tmp_path, {"docs/adr/template.md": "# Template\n"}) == []


def test_a_missing_adr_index_is_a_finding_not_a_skip(tmp_path: Path) -> None:
    problems = findings(tmp_path, {"docs/adr/README.md": None})
    assert "docs/adr/README.md:1: missing; every ADR must be linked from it" in problems


# Rule 3: relative links resolve.


def test_a_broken_link_is_reported_with_its_line(tmp_path: Path) -> None:
    problems = findings(tmp_path, {"AGENTS.md": "# Agents\n\nSee [the plan](docs/nope.md).\n"})
    assert only(problems) == "AGENTS.md:3: link to docs/nope.md names nothing tracked"


def test_links_are_resolved_from_the_file_that_holds_them(tmp_path: Path) -> None:
    assert findings(tmp_path, {"docs/plan.md": "[Agents](../AGENTS.md#top) [S](adr/)\n"}) == []
    problems = findings(tmp_path, {"docs/plan.md": "[Agents](AGENTS.md)\n"})
    assert only(problems) == "docs/plan.md:1: link to AGENTS.md names nothing tracked"


def test_a_link_that_leaves_the_repository_names_nothing(tmp_path: Path) -> None:
    problems = findings(tmp_path, {"AGENTS.md": "[x](../outside.md)\n"})
    assert only(problems) == "AGENTS.md:1: link to ../outside.md names nothing tracked"


@pytest.mark.parametrize(
    "link",
    [
        "https://example.invalid/x.md",
        "http://example.invalid/x.md",
        "mailto:a@example.invalid",
        "#section",
        "docs/plan.md#top",
        "docs/plan.md?plain=1",
        "<docs/plan.md>",
        "docs/plan.md 'title'",
        "docs",
        "/docs/plan.md",
    ],
)
def test_links_that_are_not_broken_are_left_alone(tmp_path: Path, link: str) -> None:
    assert findings(tmp_path, {"AGENTS.md": f"[x]({link})\n"}) == []


@pytest.mark.parametrize(
    "text",
    [
        "![badge](docs/nope.md)",
        "[![badge](docs/plan.md)](docs/nope.md)",
        "[ref]: docs/nope.md",
        "[x](<docs/nope.md>)",
        "[x](docs/nope.md#top)",
    ],
)
def test_every_kind_of_link_is_checked(tmp_path: Path, text: str) -> None:
    problems = findings(tmp_path, {"AGENTS.md": text + "\n"})
    assert only(problems).startswith("AGENTS.md:1: link to docs/nope.md")


def test_links_in_code_are_not_read(tmp_path: Path) -> None:
    text = "```\n[x](docs/nope.md)\n```\nand `[y](docs/nope.md)` and ``[z](docs/nope.md)``\n"
    assert findings(tmp_path, {"AGENTS.md": text}) == []


def test_a_longer_fence_is_closed_only_by_a_fence_as_long(tmp_path: Path) -> None:
    text = "````\n```\n[x](docs/nope.md)\n````\n[y](docs/also-nope.md)\n"
    problems = findings(tmp_path, {"AGENTS.md": text})
    assert only(problems) == "AGENTS.md:5: link to docs/also-nope.md names nothing tracked"


def test_an_excluded_file_is_not_read_for_links_or_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "[x](docs/nope.md) `src/nope.py`\n"
    assert "CHANGELOG.md" in cd.EXCLUDED
    assert findings(tmp_path, {"CHANGELOG.md": text}) == []
    monkeypatch.setattr(cd, "EXCLUDED", {})
    assert len(findings(tmp_path, {"CHANGELOG.md": text})) == 2


def test_an_excluded_pattern_skips_only_that_text_in_that_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cd, "EXCLUDED", {"AGENTS.md:docs/EXTERNAL.md": "someone else's file"})
    text = "`docs/EXTERNAL.md` [x](docs/EXTERNAL.md) `docs/OTHER.md`\n"
    problems = findings(tmp_path, {"AGENTS.md": text})
    assert only(problems) == "AGENTS.md:1: `docs/OTHER.md` names nothing tracked"
    problems = findings(tmp_path, {"docs/plan.md": "`docs/EXTERNAL.md`\n"})
    assert only(problems) == "docs/plan.md:1: `docs/EXTERNAL.md` names nothing tracked"


def test_the_external_file_names_the_playbook_mentions_are_excluded() -> None:
    for name in ("docs/INTEGRATIONS.md", "docs/TERMINAL.md"):
        assert cd.excluded_text("docs/playbook/practices-audit.md", name)
        assert not cd.excluded_text("docs/plan.md", name)


# Rule 4: backticked repository paths exist.


@pytest.mark.parametrize(
    "path", ["src/jarvis/audit.py", "src/jarvis/audit.py:12", "src/jarvis/audit.py:3:9"]
)
def test_a_backticked_path_that_exists_passes(tmp_path: Path, path: str) -> None:
    assert findings(tmp_path, {"AGENTS.md": f"`{path}`\n"}) == []


@pytest.mark.parametrize(
    "path",
    [
        "src/jarvis/nope.py",
        "tests/test_nope.py",
        "tools/nope.sh",
        "docs/nope.md",
        ".github/nope.yml",
        ".claude/nope",
        ".codex/nope.toml",
        ".agents/nope/",
        "MISSING.md",
    ],
)
def test_a_backticked_path_that_names_nothing_is_reported(tmp_path: Path, path: str) -> None:
    problems = findings(tmp_path, {"AGENTS.md": f"text `{path}` text\n"})
    assert only(problems) == f"AGENTS.md:1: `{path}` names nothing tracked"


@pytest.mark.parametrize(
    "text",
    [
        "src/jarvis/*.py",
        "tests/test_<name>.py",
        "docs/{a,b}.md",
        "docs/adr/NNNN title.md",
        "src/ is where the code is",
        "config.toml",
        "~/.config/jarvis/config.toml",
        "https://example.invalid/src/x.py",
        "uv run pytest",
        "SKILL.md.bak",
        "docs/plan.md#heading",
        "src/jarvis/",
        "AGENTS.md",
    ],
)
def test_text_that_is_not_a_repository_path_is_skipped(tmp_path: Path, text: str) -> None:
    assert findings(tmp_path, {"AGENTS.md": f"`{text}`\n"}) == []


def test_a_top_level_document_name_must_exist(tmp_path: Path) -> None:
    problems = findings(tmp_path, {"AGENTS.md": "`CONTRIBUTING.md`\n"})
    assert only(problems) == "AGENTS.md:1: `CONTRIBUTING.md` names nothing tracked"


def test_paths_in_code_fences_are_not_read(tmp_path: Path) -> None:
    assert findings(tmp_path, {"AGENTS.md": "```\n`src/nope.py`\n```\n"}) == []


# Rule 5: modules named in docs/architecture.md.


def test_a_module_missing_from_the_architecture_doc_is_reported(tmp_path: Path) -> None:
    problems = findings(tmp_path, {"src/jarvis/voice.py": ""})
    assert only(problems) == "src/jarvis/voice.py:1: not named in docs/architecture.md"


def test_a_nested_module_may_be_named_by_its_file_or_its_path(tmp_path: Path) -> None:
    assert findings(tmp_path, {"docs/architecture.md": "`audit.py` `auth.py`\n"}) == []
    assert (
        findings(tmp_path, {"docs/architecture.md": "`audit.py` `jarvis/spotify/auth.py`\n"}) == []
    )
    problems = findings(tmp_path, {"docs/architecture.md": "`audit.py`\n"})
    assert only(problems) == "src/jarvis/spotify/auth.py:1: not named in docs/architecture.md"


@pytest.mark.parametrize(
    "text", ["`spotify/audit.py` `spotify/auth.py`", "`unaudit.py` `spotify/auth.py`"]
)
def test_a_top_level_module_is_not_named_by_a_nested_one_or_a_longer_name(
    tmp_path: Path, text: str
) -> None:
    files = {**GOOD, "src/jarvis/spotify/audit.py": "", "docs/architecture.md": text}
    problems = cd.check(tmp_path, write(tmp_path, files))
    assert "src/jarvis/audit.py:1: not named in docs/architecture.md" in problems


def test_a_missing_architecture_doc_is_a_finding_not_a_skip(tmp_path: Path) -> None:
    problems = findings(tmp_path, {"docs/architecture.md": None})
    expected = "docs/architecture.md:1: missing; every module under src/jarvis/ must be named in it"
    assert expected in problems


# Rule 6: tools named in tools/README.md.


def test_a_tool_missing_from_the_tools_readme_is_reported(tmp_path: Path) -> None:
    problems = findings(tmp_path, {"tools/other.py": ""})
    assert only(problems) == "tools/other.py:1: not named in tools/README.md"


def test_naming_the_hooks_directory_covers_the_files_in_it(tmp_path: Path) -> None:
    assert findings(tmp_path, {"tools/hooks/commit-msg": ""}) == []
    problems = findings(tmp_path, {"tools/README.md": "`check_docs.py`\n"})
    assert only(problems) == "tools/hooks/pre-commit:1: not named in tools/README.md"


def test_a_file_in_another_subdirectory_must_be_named(tmp_path: Path) -> None:
    problems = findings(tmp_path, {"tools/data/table.csv": ""})
    assert only(problems) == "tools/data/table.csv:1: not named in tools/README.md"
    readme = GOOD["tools/README.md"] + "`data/table.csv`\n"
    assert findings(tmp_path, {"tools/data/table.csv": "", "tools/README.md": readme}) == []


def test_a_missing_tools_readme_is_a_finding_not_a_skip(tmp_path: Path) -> None:
    problems = findings(tmp_path, {"tools/README.md": None})
    assert "tools/README.md:1: missing; every file under tools/ must be named in it" in problems


# Rule 7: the skill mirror.


def test_a_skill_without_a_mirror_is_reported_both_ways(tmp_path: Path) -> None:
    problems = findings(tmp_path, {".agents/skills/s/SKILL.md": None})
    assert only(problems) == ".claude/skills/s/SKILL.md:1: no mirror .agents/skills/s/SKILL.md"
    problems = findings(tmp_path, {".claude/skills/s/SKILL.md": None})
    assert only(problems) == ".agents/skills/s/SKILL.md:1: no mirror .claude/skills/s/SKILL.md"


def test_skills_that_differ_by_a_byte_are_reported(tmp_path: Path) -> None:
    problems = findings(tmp_path, {".agents/skills/s/SKILL.md": "# Skill"})
    assert only(problems) == ".agents/skills/s/SKILL.md:1: differs from .claude/skills/s/SKILL.md"


# Rule 8: the agent mirror for Codex.

AGENT_MD = "---\nname: r\ndescription: Reviews.\n---\n\nYou review.\n"
AGENT_TOML = (
    'name = "r"\ndescription = "Reviews."\ndeveloper_instructions = """\nYou review.\n"""\n'
)


def test_an_agent_and_its_codex_mirror_that_match_are_clean(tmp_path: Path) -> None:
    changes: dict[str, str | None] = {
        ".claude/agents/r.md": AGENT_MD,
        ".codex/agents/r.toml": AGENT_TOML,
    }
    assert findings(tmp_path, changes) == []


def test_an_agent_without_a_codex_mirror_is_reported_both_ways(tmp_path: Path) -> None:
    problems = findings(tmp_path, {".claude/agents/r.md": AGENT_MD})
    assert only(problems) == ".claude/agents/r.md:1: no mirror .codex/agents/r.toml"
    problems = findings(tmp_path, {".codex/agents/r.toml": AGENT_TOML})
    assert only(problems) == ".codex/agents/r.toml:1: no mirror .claude/agents/r.md"


@pytest.mark.parametrize(
    "toml",
    [
        AGENT_TOML.replace("You review.", "You approve."),
        AGENT_TOML.replace('"Reviews."', '"Approves."'),
        AGENT_TOML.replace('name = "r"', 'name = "x"'),
        "not toml = = =\n",
    ],
)
def test_a_codex_mirror_that_differs_is_reported(tmp_path: Path, toml: str) -> None:
    problems = findings(tmp_path, {".claude/agents/r.md": AGENT_MD, ".codex/agents/r.toml": toml})
    assert only(problems).startswith(".codex/agents/r.toml:1: ")


# Symbolic links are never followed: a tracked link could point at a private file.


def test_a_tracked_symlink_is_reported_and_not_read(tmp_path: Path) -> None:
    secret = tmp_path / "outside.md"
    secret.write_text("`src/private/secret-path.py`\n", encoding="utf-8")
    files = write(tmp_path, dict(GOOD))
    (tmp_path / "docs" / "link.md").symlink_to(secret)

    problems = cd.check(tmp_path, [*files, "docs/link.md"])

    assert problems == ["docs/link.md:1: a symbolic link; the docs check does not follow it"]


# Reading only tracked files, and the command line.


def git(root: Path, *args: str) -> None:
    command = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args]
    subprocess.run(command, cwd=root, check=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    return root


def test_tracked_files_lists_only_what_git_tracks(repo: Path) -> None:
    write(repo, {"a.md": "", "docs/b c.md": "", "docs/é.md": "", "gone.md": "", "private.md": ""})
    git(repo, "add", "a.md", "docs/b c.md", "docs/é.md", "gone.md")
    (repo / "gone.md").unlink()
    assert cd.tracked_files(repo) == ["a.md", "docs/b c.md", "docs/é.md"]


def test_an_untracked_file_is_never_read(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(repo, {**GOOD, "private.md": "[x](nope.md) `src/nope.py`\n"})
    git(repo, "add", *GOOD)
    monkeypatch.setattr(cd, "ROOT", repo)
    assert cd.main([]) == 0


def test_main_prints_findings_and_exits_one(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write(repo, {**GOOD, "AGENTS.md": "[x](nope.md)\n"})
    git(repo, "add", *GOOD)
    monkeypatch.setattr(cd, "ROOT", repo)
    assert cd.main([]) == 1
    assert capsys.readouterr().out == "AGENTS.md:1: link to nope.md names nothing tracked\n"


def test_main_rejects_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert cd.main(["--bogus"]) == 2
    assert "usage" in capsys.readouterr().err


def test_the_real_tree_has_no_findings() -> None:
    root = Path(__file__).resolve().parent.parent

    assert cd.check(root, cd.tracked_files(root)) == []
