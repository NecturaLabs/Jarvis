"""Jarvis's shipped skills: listed in the prompt, read whole on request, nothing else readable."""

import pytest

from jarvis import skills
from jarvis.operations import ArgumentError, validate_args


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
