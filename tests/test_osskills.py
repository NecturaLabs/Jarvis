"""Omarchy's OS skills: listed at start, read one listed file at a time, in parts."""

import os
from pathlib import Path

import pytest

from jarvis import osskills
from jarvis.operations import ArgumentError, OperationError, validate_args
from jarvis.policy import TRUSTED_ORIGINS, OpClass, Origin

SKILL = """---
name: {name}
description: >
  {first}
  {second}
---

# {name}

Run `omarchy-{name}-do` to do it.
"""


ME = os.getuid()


def _installed(root: Path) -> tuple[osskills.OsSkill, ...]:
    return osskills.installed(root, ME)


def _skill(root: Path, name: str, *extra: str) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        SKILL.format(name=name, first=f"Use for {name} work.", second="Triggers: a, b."),
        encoding="utf-8",
    )
    for file in extra:
        (folder / file).write_text(f"# {file}\n\nMore about {name}.\n", encoding="utf-8")
    return folder


@pytest.fixture
def root(tmp_path: Path) -> Path:
    skills = tmp_path / "skills"
    _skill(skills, "omarchy", "theming.md", "hooks.md")
    _skill(skills, "diagnose-crash", "reporting.md")
    return skills


def test_lists_each_skill_with_its_folded_description_and_its_files(root: Path) -> None:
    found = _installed(root)

    assert [skill.name for skill in found] == ["diagnose-crash", "omarchy"]
    omarchy = found[1]
    assert omarchy.description == "Use for omarchy work. Triggers: a, b."
    assert omarchy.files == ("SKILL.md", "hooks.md", "theming.md")


def test_lists_nothing_when_the_folder_is_missing(tmp_path: Path) -> None:
    assert _installed(tmp_path / "absent") == ()


def test_lists_no_link_no_odd_name_and_no_folder_without_a_skill_file(
    root: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    _skill(outside, "elsewhere", "secret.md")
    (root / "linked").symlink_to(outside / "elsewhere")
    (root / "omarchy" / "secret.md").symlink_to(outside / "elsewhere" / "secret.md")
    (root / "omarchy" / "notes.txt").write_text("not markdown", encoding="utf-8")
    (root / "omarchy" / ".hidden.md").write_text("hidden", encoding="utf-8")
    (root / "omarchy" / "sub").mkdir()
    (root / "empty").mkdir()
    _skill(root, "Bad Name")

    found = _installed(root)

    assert [skill.name for skill in found] == ["diagnose-crash", "omarchy"]
    assert found[1].files == ("SKILL.md", "hooks.md", "theming.md")


def test_a_long_description_is_cut_at_a_word_and_listed_shorter(tmp_path: Path) -> None:
    folder = tmp_path / "skills" / "long"
    folder.mkdir(parents=True)
    words = " ".join(["word"] * 400)
    (folder / "SKILL.md").write_text(
        f"---\nname: long\ndescription: {words}\n---\n# Long\n", encoding="utf-8"
    )

    (skill,) = _installed(tmp_path / "skills")

    assert len(skill.description) <= osskills.MAX_DESCRIPTION_CHARS
    assert skill.description.endswith("word…")
    assert len(skill.summary()) <= osskills.MAX_SUMMARY_CHARS
    assert skill.summary().endswith("word…")


def test_the_prompt_lists_each_skill_by_its_first_sentence_and_says_that_its_commands_are_not_run(
    root: Path,
) -> None:
    section = osskills.prompt_section(_installed(root))

    assert "os_skill" in section
    assert "  - omarchy: Use for omarchy work.\n" in section
    assert "Triggers" not in section
    assert "never run a command" in section
    assert not osskills.prompt_section(())


def test_reads_one_listed_file_whole_as_os_skill_text(root: Path) -> None:
    op = osskills.os_skill_operation(_installed(root), max_chars=3000, root=root, owner=ME)

    result = op.run(validate_args(op, {"file": "omarchy/theming.md"}))

    assert result.text == "omarchy/theming.md (part 1 of 1):\n# theming.md\n\nMore about omarchy.\n"
    assert result.origin is Origin.OS_SKILL
    assert op.side_effect is False
    assert op.default_class is OpClass.SAFE


def test_a_long_file_is_read_in_parts_that_together_are_the_whole_file(root: Path) -> None:
    lines = [f"Line {n} of the guide, about theming and fonts." for n in range(400)]
    text = "\n".join(lines) + "\n"
    (root / "omarchy" / "theming.md").write_text(text, encoding="utf-8")
    op = osskills.os_skill_operation(_installed(root), max_chars=1000, root=root, owner=ME)

    first = op.run(validate_args(op, {"file": "omarchy/theming.md"}))
    head, _, _ = first.text.partition("\n")
    count = int(head.rsplit(" ", 1)[1].rstrip("):"))
    bodies: list[str] = []
    for part in range(1, count + 1):
        result = op.run(validate_args(op, {"file": "omarchy/theming.md", "part": part}))
        assert len(result.text) <= 1000
        title, _, body = result.text.partition("\n")
        assert title == f"omarchy/theming.md (part {part} of {count}):"
        more = f"\n(Continued in part {part + 1}.)"
        if part < count:
            assert body.endswith(more)
            body = body.removesuffix(more)
        bodies.append(body)

    assert count > 1
    assert "".join(bodies) == text
    with pytest.raises(OperationError, match=f"only {count} parts"):
        op.run(validate_args(op, {"file": "omarchy/theming.md", "part": count + 1}))


@pytest.mark.parametrize(
    "args",
    [
        {"file": "omarchy/../../etc/passwd"},
        {"file": "/etc/passwd"},
        {"file": "../skills/omarchy/SKILL.md"},
        {"file": "omarchy/secret.md"},
        {"file": "omarchy"},
        {"file": "omarchy/theming.md", "part": 0},
        {"file": "omarchy/theming.md", "extra": 1},
        {},
    ],
)
def test_refuses_any_file_not_listed_and_bad_arguments(root: Path, args: dict[str, object]) -> None:
    op = osskills.os_skill_operation(_installed(root), max_chars=3000, root=root, owner=ME)

    with pytest.raises(ArgumentError):
        validate_args(op, args)


def test_a_file_swapped_for_a_link_after_listing_is_not_followed(
    root: Path, tmp_path: Path
) -> None:
    op = osskills.os_skill_operation(_installed(root), max_chars=3000, root=root, owner=ME)
    secret = tmp_path / "secret.txt"
    secret.write_text("private", encoding="utf-8")
    (root / "omarchy" / "theming.md").unlink()
    (root / "omarchy" / "theming.md").symlink_to(secret)

    with pytest.raises(OperationError) as refused:
        op.run(validate_args(op, {"file": "omarchy/theming.md"}))

    assert "private" not in str(refused.value)


def test_a_skill_folder_swapped_for_a_link_after_listing_is_not_followed(
    root: Path, tmp_path: Path
) -> None:
    op = osskills.os_skill_operation(_installed(root), max_chars=3000, root=root, owner=ME)
    outside = tmp_path / "outside"
    _skill(outside, "omarchy", "theming.md")
    (outside / "omarchy" / "theming.md").write_text("private", encoding="utf-8")
    Path(root / "omarchy").rename(tmp_path / "moved")
    (root / "omarchy").symlink_to(outside / "omarchy")

    with pytest.raises(OperationError):
        op.run(validate_args(op, {"file": "omarchy/theming.md"}))


def test_a_file_grown_past_the_cap_is_not_read(root: Path) -> None:
    op = osskills.os_skill_operation(_installed(root), max_chars=3000, root=root, owner=ME)
    (root / "omarchy" / "theming.md").write_text(
        "x" * (osskills.MAX_FILE_BYTES + 1), encoding="utf-8"
    )

    with pytest.raises(OperationError, match="too large"):
        op.run(validate_args(op, {"file": "omarchy/theming.md"}))


def test_os_skill_text_is_outside_text() -> None:
    assert Origin.OS_SKILL not in TRUSTED_ORIGINS


@pytest.mark.local
def test_the_installed_omarchy_skills_are_the_three_expected() -> None:
    found = osskills.catalog()

    assert {skill.name for skill in found} == {"omarchy", "omarchy-app", "diagnose-crash"}
    for skill in found:
        assert "SKILL.md" in skill.files
        assert skill.description


def test_lists_nothing_that_is_not_the_os_owners(root: Path) -> None:
    assert osskills.installed(root, ME + 1) == ()


def test_lists_no_skill_or_file_that_others_can_write(root: Path) -> None:
    (root / "omarchy" / "hooks.md").chmod(0o666)
    (root / "diagnose-crash").chmod(0o777)

    found = _installed(root)

    assert [(skill.name, skill.files) for skill in found] == [
        ("omarchy", ("SKILL.md", "theming.md"))
    ]


def test_a_folder_above_the_skills_that_is_a_link_is_not_followed(
    root: Path, tmp_path: Path
) -> None:
    (tmp_path / "via").symlink_to(root.parent)

    assert _installed(tmp_path / "via" / "skills") == ()
    op = osskills.os_skill_operation(
        _installed(root), max_chars=3000, root=tmp_path / "via" / "skills", owner=ME
    )
    with pytest.raises(OperationError):
        op.run(validate_args(op, {"file": "omarchy/theming.md"}))


def test_a_file_made_writable_by_others_after_listing_is_not_read(root: Path) -> None:
    op = osskills.os_skill_operation(_installed(root), max_chars=3000, root=root, owner=ME)
    (root / "omarchy" / "theming.md").chmod(0o646)

    with pytest.raises(OperationError):
        op.run(validate_args(op, {"file": "omarchy/theming.md"}))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "x" * 300,  # exactly one part's room at the lowest cap
        "x" * 301,
        "é😀\n" * 400,
        ("y" * 150 + "\n") * (osskills.MAX_FILE_BYTES // 151),  # the largest file, in short lines
        "z" * osskills.MAX_FILE_BYTES,  # the largest file, with no line end
    ],
)
def test_parts_at_the_lowest_cap_rejoin_into_the_file_and_stay_within_the_schema(
    root: Path, text: str
) -> None:
    (root / "omarchy" / "theming.md").write_text(text, encoding="utf-8")
    low = osskills.MIN_TOOL_TEXT_CHARS
    op = osskills.os_skill_operation(_installed(root), max_chars=low, root=root, owner=ME)
    count = int(
        op.run({"file": "omarchy/theming.md"}).text.split("\n", 1)[0].rsplit(" ", 1)[1][:-2]
    )

    assert count <= op.parameters["properties"]["part"]["maximum"]
    bodies: list[str] = []
    for part in range(1, count + 1):
        result = op.run(validate_args(op, {"file": "omarchy/theming.md", "part": part}))
        assert len(result.text) <= low
        body = result.text.split("\n", 1)[1]
        bodies.append(body.removesuffix(f"\n(Continued in part {part + 1}.)"))
    assert "".join(bodies) == text


def test_the_catalog_lists_every_guide_before_any_extra_file_and_keeps_to_its_size(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "skills"
    for n in range(12):
        _skill(root, f"skill-{n:02}", *(f"file-{m:02}.md" for m in range(4)))

    found = _installed(root)

    assert [skill.name for skill in found] == [f"skill-{n:02}" for n in range(12)]
    assert all(skill.files[0] == "SKILL.md" for skill in found)
    assert sum(skill.catalog_chars() for skill in found) <= osskills.MAX_CATALOG_CHARS
    assert sum(len(skill.files) for skill in found) < 12 * 5
    assert "left out" in caplog.text


def test_a_guide_too_long_for_the_catalog_is_left_out_and_said(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "skills"
    for n in range(40):
        _skill(root, f"a-guide-with-a-long-name-number-{n:02}")

    found = _installed(root)

    assert 0 < len(found) < 40
    assert sum(skill.catalog_chars() for skill in found) <= osskills.MAX_CATALOG_CHARS
    assert "does not fit in the catalog" in caplog.text
