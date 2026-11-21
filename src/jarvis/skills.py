"""Jarvis's own skills: shipped instructions for a kind of task, read when the model needs them.

Each skill is a Markdown file in `prompts/skills/` that opens with a `description:` line. The
system prompt lists the names and descriptions; the model reads a whole skill with `read_skill`
before such a task, so the prompt stays short. Only the files shipped with Jarvis are skills:
nothing else on the PC can be read this way.
"""

import re
from dataclasses import dataclass
from functools import cache
from importlib.resources import files

from jarvis.operations import Operation, OperationError, Result

_FRONT = re.compile(r"\A---\n(?P<head>.*?)\n---\n(?P<body>.*)\Z", re.DOTALL)
_DESCRIPTION = re.compile(r"^description:\s*(?P<text>.+)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Skill:
    """One shipped skill."""

    name: str
    description: str
    body: str


@cache
def shipped() -> tuple[Skill, ...]:
    """Return the skills shipped with Jarvis, by name."""
    folder = files("jarvis").joinpath("prompts/skills")
    found: list[Skill] = []
    for entry in sorted(folder.iterdir(), key=lambda item: item.name):
        if not entry.name.endswith(".md"):
            continue
        text = entry.read_text(encoding="utf-8")
        parts = _FRONT.match(text)
        described = _DESCRIPTION.search(parts["head"]) if parts else None
        if parts is None or described is None:
            continue
        found.append(
            Skill(entry.name.removesuffix(".md"), described["text"].strip(), parts["body"])
        )
    return tuple(found)


def prompt_section(skills: tuple[Skill, ...]) -> str:
    """Return the system prompt's list of skills, or "" when there are none."""
    if not skills:
        return ""
    lines = "\n".join(f"- {skill.name}: {skill.description}" for skill in skills)
    return (
        "- Skills: before a task one of these covers, call read_skill with its name and follow "
        f"it (once per conversation is enough):\n{lines}\n"
    )


def skill_operation(skills: tuple[Skill, ...]) -> Operation:
    """Return `read_skill`, which gives the model one shipped skill's instructions."""
    by_name = {skill.name: skill for skill in skills}

    def run(args: dict[str, object]) -> Result:
        skill = by_name.get(str(args["name"]))
        if skill is None:
            msg = "There is no skill with that name."
            raise OperationError(msg)
        return Result(skill.body)

    return Operation(
        "read_skill",
        "Read the instructions for a kind of task (see Skills in your instructions) before "
        "doing it.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": sorted(by_name), "description": "The skill"}
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        run,
        side_effect=False,
        spoken_summary=lambda args: f"read the {args.get('name', '')} instructions",
    )
