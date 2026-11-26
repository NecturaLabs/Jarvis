"""The tool router: the toolbox in the prompt, and `use_tools`, which offers a group's tools.

There is no separate router. The same model, in the same loop, reads the toolbox (each group of
tools and what it is for, generated from the registry) and decides which groups a request needs.
With `tools = "routed"` it is first offered only the core tools; `use_tools` names the groups it
wants and the agent offers them for the rest of that request (`Agent._load_groups`). Nothing
here matches the owner's words: the model decides, and the system only enforces what is offered.
"""

from typing import TYPE_CHECKING

from jarvis.operations import Operation, Registry, Result

if TYPE_CHECKING:
    from collections.abc import Sequence


def use_tools_operation(groups: Sequence[str]) -> Operation:
    """Return `use_tools`, with which of the toolbox's `groups` it may ask for.

    It changes nothing by itself: the agent offers the groups' tools from the next model call on
    and tells the model which tools it got.
    """
    return Operation(
        "use_tools",  # the agent's USE_TOOLS
        "Ask for the tools of the groups in your toolbox that the request needs, before calling "
        "them. Name every group the request needs in one call. They are offered from your next "
        "reply to the end of this request. Never needed for what no_action answers.",
        {
            "type": "object",
            "properties": {
                "groups": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(groups)},
                    "minItems": 1,
                    "maxItems": max(1, len(groups)),
                    "description": "The groups to get the tools of",
                }
            },
            "required": ["groups"],
            "additionalProperties": False,
        },
        lambda args: Result("Getting the tools.", groups=tuple(map(str, args["groups"]))),
        side_effect=False,
        spoken_summary=lambda _args: "get the tools it needs",
    )


def toolbox_section(registry: Registry) -> str:
    """Return the system prompt's toolbox: each tool group, what it is for and its tools.

    Routed, the model sees only the core tools until it asks, so this is where it learns what it
    has: each group's summary and tool names, and how to ask. "" when there are no groups.
    Where every tool is offered on every call the toolbox is left out: the tools are all in
    front of the model, and its words would take room the conversation needs.
    """
    groups = registry.groups()
    if not groups:
        return ""
    lines = "\n".join(
        f"  - {group.name}: {group.summary} Tools: {', '.join(group.tools)}." for group in groups
    )
    core = ", ".join(registry.core_names())
    return (
        f"- Your toolbox: your tools come in groups. At first you are offered only {core}. "
        "For anything else call use_tools once with every group the request needs, then call "
        "its tools in your next reply; they stay offered for the rest of the request. A tool "
        "of a group you have not asked for does not run. The skills and guides below are "
        f"read with the guides group's tools.\n{lines}\n"
    )
