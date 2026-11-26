"""Jarvis's shipped skills: listed in the prompt, read whole on request, nothing else readable."""

from typing import TYPE_CHECKING

import pytest

from jarvis import skills
from jarvis.operations import ArgumentError, validate_args

if TYPE_CHECKING:
    from pathlib import Path


def test_every_shipped_skill_has_a_name_a_description_and_a_body() -> None:
    shipped = skills.shipped()

    assert {skill.name for skill in shipped} >= {"web-research", "game-lookup"}
    for skill in shipped:
        assert skill.description
        assert skill.body.startswith("# ")


def test_the_prompt_lists_each_skill_with_its_description() -> None:
    section = skills.prompt_section(skills.shipped())

    for skill in skills.shipped():
        assert f"- {skill.name}: {skill.description}" in section
    assert not skills.prompt_section(())


def test_read_skill_returns_the_whole_skill() -> None:
    shipped = skills.shipped()
    op = skills.skill_operation(shipped)

    result = op.run(validate_args(op, {"name": "web-research"}))

    assert result.text == next(s.body for s in shipped if s.name == "web-research")
    assert op.side_effect is False


@pytest.mark.parametrize("name", ["../core", "core", "/etc/passwd", "web-research.md"])
def test_read_skill_reads_nothing_but_a_shipped_skill(name: str) -> None:
    op = skills.skill_operation(skills.shipped())

    with pytest.raises(ArgumentError):
        validate_args(op, {"name": name})


def _write(folder: Path, name: str, text: str, mode: int = 0o600) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


SKILL = (
    "---\ndescription: Watch a window and say when something shows up.\n---\n# Watch\n\nDo it.\n"
)


@pytest.fixture
def mine(tmp_path: Path) -> Path:
    folder = tmp_path / "skills"
    folder.mkdir(mode=0o700)
    return folder


def test_an_owner_skill_is_listed_marked_and_readable(mine: Path) -> None:
    _write(mine, "watch-sales.md", SKILL)

    found = skills.available(mine)
    op = skills.skill_operation(found)

    assert [s.name for s in found][-1] == "watch-sales"
    assert "- watch-sales (yours): Watch a window" in skills.prompt_section(found)
    assert op.run(validate_args(op, {"name": "watch-sales"})).text == "# Watch\n\nDo it.\n"


def test_no_owner_folder_means_only_the_shipped_skills(tmp_path: Path) -> None:
    assert skills.available(tmp_path / "missing") == skills.shipped()
    assert skills.doctor_notes(tmp_path / "missing") == []


@pytest.mark.parametrize(
    ("name", "text", "mode", "why"),
    [
        ("web-research.md", SKILL, 0o600, "a shipped skill has that name"),
        ("Bad Name.md", SKILL, 0o600, "lowercase letters"),
        ("open.md", SKILL, 0o622, "not yours alone"),
        ("plain.md", "# no front matter\n", 0o600, "description: line"),
        ("huge.md", SKILL + "x" * skills.MAX_BODY_BYTES, 0o600, "bigger than"),
        (
            "wordy.md",
            f"---\ndescription: {'w' * (skills.MAX_DESCRIPTION_CHARS + 1)}\n---\n# W\n",
            0o600,
            "description is longer",
        ),
    ],
)
def test_an_owner_file_that_breaks_a_rule_is_left_out_and_said(
    mine: Path, name: str, text: str, mode: int, why: str
) -> None:
    _write(mine, name, text, mode)

    found, skipped = skills.owner_skills(mine)

    assert found == ()
    assert len(skipped) == 1
    assert skipped[0].startswith(f"{name}: ")
    assert why in skipped[0]
    assert any(why in note for note in skills.doctor_notes(mine))


def test_a_linked_owner_skill_is_not_followed(mine: Path, tmp_path: Path) -> None:
    target = _write(tmp_path, "elsewhere.md", SKILL)
    (mine / "linked.md").symlink_to(target)

    found, skipped = skills.owner_skills(mine)

    assert found == ()
    assert skipped == ["linked.md: a link, which is not followed"]


def test_a_folder_others_can_write_to_is_not_read(mine: Path) -> None:
    _write(mine, "watch-sales.md", SKILL)
    mine.chmod(0o777)

    found, skipped = skills.owner_skills(mine)

    assert found == ()
    assert "not yours alone" in skipped[0]


def test_a_linked_owner_folder_is_not_followed(mine: Path, tmp_path: Path) -> None:
    _write(mine, "watch-sales.md", SKILL)
    linked = tmp_path / "linked"
    linked.symlink_to(mine)

    found, skipped = skills.owner_skills(linked)

    assert found == ()
    assert "a link is not followed" in skipped[0]


def test_owner_skills_are_capped_in_number_and_in_prompt_size(mine: Path) -> None:
    line = "d" * skills.MAX_DESCRIPTION_CHARS
    for index in range(skills.MAX_OWNER_SKILLS + 2):
        _write(mine, f"s{index:02d}.md", f"---\ndescription: {line}\n---\n# S\n")

    found, skipped = skills.owner_skills(mine)

    assert sum(len(s.description) for s in found) <= skills.MAX_DESCRIPTIONS_CHARS
    assert len(found) + len(skipped) == skills.MAX_OWNER_SKILLS + 2
    assert len(found) <= skills.MAX_OWNER_SKILLS


def test_the_prompt_and_read_skill_include_the_owners_skills_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    folder = tmp_path / "jarvis" / "skills"
    folder.mkdir(parents=True, mode=0o700)
    _write(folder, "watch-sales.md", SKILL)

    assert skills.owner_folder() == folder
    assert "watch-sales" in {s.name for s in skills.available()}


def test_the_system_prompt_lists_the_owners_skill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jarvis.app import system_prompt  # ruff: ignore[import-outside-top-level] - the app wiring, not the module
    from jarvis.config import Config  # ruff: ignore[import-outside-top-level]

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    folder = tmp_path / "jarvis" / "skills"
    folder.mkdir(parents=True, mode=0o700)
    _write(folder, "watch-sales.md", SKILL)

    assert "- watch-sales (yours): Watch a window" in system_prompt(Config())
