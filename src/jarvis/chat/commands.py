"""The chat's slash commands, and what the menu suggests while one is being typed.

Matching works as in Claude Code: the menu narrows with every letter, a command whose name starts
with what was typed comes first, then one that holds the letters in order ("/cmp" finds
`/compact`). After a command and a space, the menu offers its arguments.
"""

from dataclasses import dataclass

VERBOSITY = ("replies", "steps", "everything")
EFFORT_LEVELS = ("none", "low", "medium", "high")  # as `chat.thinking` takes them


@dataclass(frozen=True, slots=True)
class Command:
    """A slash command: its name, what it takes, and what it does."""

    name: str
    about: str
    hint: str = ""  # what follows the name, as the menu shows it: "[level]"
    choices: tuple[str, ...] = ()  # the arguments it takes, offered after the name


COMMANDS = (
    Command("/help", "commands and keyboard shortcuts"),
    Command("/clear", "start a new conversation and empty the window"),
    Command("/compact", "summarize the conversation now to free context"),
    Command("/context", "how full the model's context is"),
    Command(
        "/verbosity",
        "how much of the work to show: replies, steps or everything (also ctrl+o)",
        "[level]",
        VERBOSITY,
    ),
    Command(
        "/effort",
        "how hard the model thinks before it answers: none, low, medium or high",
        "[level]",
        EFFORT_LEVELS,
    ),
    Command("/config", "the chat's settings"),
    Command("/status", "the voice service, the model and where things are kept"),
    Command("/copy", "copy the last reply"),
    Command("/export", "save the conversation as a Markdown file"),
    Command("/image", "attach a picture: a file, or the newest screenshot (also ctrl+v)", "[path]"),
    Command("/exit", "leave (also ctrl+q, or ctrl+c twice)"),
)
ALIASES = {"/quit": "/exit", "/new": "/clear", "/reset": "/clear", "/?": "/help"}


@dataclass(frozen=True, slots=True)
class Suggestion:
    """One line of the menu: what Tab or Enter puts in the input, and how it is shown."""

    value: str
    label: str
    about: str
    marked: tuple[int, ...] = ()  # where the typed letters are in `label`, shown in bold


def _in_order(letters: str, name: str) -> bool:
    """Return whether `letters` appear in `name` in that order."""
    rest = iter(name)
    return all(letter in rest for letter in letters)


def _marked(typed: str, name: str) -> tuple[int, ...]:
    """Return where the letters of `typed` fall in `name`: the start, else the first in order."""
    if name.startswith(typed):
        return tuple(range(len(typed)))
    places: list[int] = []
    for letter in typed:
        place = name.find(letter, places[-1] + 1 if places else 0)
        if place < 0:
            return ()
        places.append(place)
    return tuple(places)


def find(name: str) -> Command | None:
    """Return the command a typed name means, aliases included."""
    name = ALIASES.get(name.lower(), name.lower())
    return next((command for command in COMMANDS if command.name == name), None)


def suggest(value: str) -> list[Suggestion]:
    """Return what the menu offers for the input `value` (nothing when it is not a command)."""
    if not value.startswith("/") or "\n" in value:
        return []
    head, space, rest = value.partition(" ")
    if space:
        command = find(head)
        if command is None or not command.choices or " " in rest.strip():
            return []
        typed = rest.strip().lower()
        return [
            Suggestion(f"{command.name} {choice}", choice, "", tuple(range(len(typed))))
            for choice in command.choices
            if choice.startswith(typed)
        ]
    typed = head.lower()
    first = [c for c in COMMANDS if c.name.startswith(typed)]
    later = [c for c in COMMANDS if c not in first and _in_order(typed[1:], c.name[1:])]
    return [
        Suggestion(
            c.name + (" " if c.choices else ""),
            f"{c.name} {c.hint}".strip(),
            c.about,
            _marked(typed, c.name),
        )
        for c in first + later
    ]
