"""Switching Jarvis's tools on and off, and how much each asks first (`boundaries.classes`).

Each tool's setting is its class: on (`safe`), asks first (`confirm`), needs the confirm key
(`strong`) or off (`deny`), which is not even offered to the model. Only the classes a tool allows
are offered (`stop_process` takes only the key or off), and its shipped class is written as no
entry at all, so a later release's default still reaches it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jarvis.app import ALWAYS_OFFERED, channel_limits, channel_registry
from jarvis.config import ConfigError
from jarvis.policy import OpClass, effective_class

if TYPE_CHECKING:
    from jarvis.config import Config
    from jarvis.operations import Operation

_RANK = {OpClass.SAFE: 0, OpClass.CONFIRM: 1, OpClass.STRONG: 2, OpClass.DENY: 3}
LABELS = {
    OpClass.SAFE: "on",
    OpClass.CONFIRM: "asks first",
    OpClass.STRONG: "needs the confirm key",
    OpClass.DENY: "off",
}
# The words `jarvis tools <name> <word>` takes; `default` goes back to the shipped class.
WORDS = {"on": OpClass.SAFE, "ask": OpClass.CONFIRM, "key": OpClass.STRONG, "off": OpClass.DENY}
_SUMMARY = 70  # characters of a tool's description shown beside it


@dataclass(frozen=True, slots=True)
class ToolSetting:
    """One tool: its name, what it does, its class now and shipped, and the classes it allows."""

    name: str
    summary: str
    state: OpClass
    shipped: OpClass
    choices: tuple[OpClass, ...]
    description: str = ""  # what the model is told it does, whole (`summary` is its start)

    @property
    def label(self) -> str:
        """The setting as it is shown: "on", "asks first", "needs the confirm key" or "off"."""
        return LABELS[self.state]

    @property
    def shipped_label(self) -> str:
        """How the tool ships, as `label` says it."""
        return LABELS[self.shipped]

    def following(self) -> OpClass:
        """Return the next class for a menu that cycles: from as shipped, stricter, then back.

        A menu never takes a tool below its shipped class (`write_file` from asking first to
        on): that takes `jarvis tools NAME on`, said on purpose.
        """
        ring = [c for c in self.choices if _RANK[c] >= _RANK[self.shipped]]
        at = ring.index(self.state) if self.state in ring else -1
        return ring[(at + 1) % len(ring)]


def _summary(op: Operation) -> str:
    first = op.description.split(". ", 1)[0].rstrip(".")
    return first if len(first) <= _SUMMARY else f"{first[: _SUMMARY - 1].rstrip()}…"


def tool_settings(config: Config, spotify: list[Operation]) -> list[ToolSetting]:
    """Return every tool of the voice service and the chat with its setting, by name."""
    ops: dict[str, Operation] = {}
    for channel in ("voice", "chat"):
        registry = channel_registry(config, spotify, channel_limits(config, channel), channel)
        for name in registry.names():
            op = registry.get(name)
            if op is not None and name not in ALWAYS_OFFERED:
                ops.setdefault(name, op)
    return [
        ToolSetting(
            name,
            _summary(op),
            effective_class(op, config.boundaries.classes),
            effective_class(op, {}),
            tuple(c for c in OpClass if effective_class(op, {name: c.value}) is c),
            op.description,
        )
        for name, op in sorted(ops.items())
    ]


def with_class(config: Config, setting: ToolSetting, cls: OpClass) -> Config:
    """Return the config with the tool set to `cls`: no entry for its shipped class.

    Raises:
        ConfigError: the tool does not allow that class.
    """
    if cls not in setting.choices:
        allowed = ", ".join(LABELS[c] for c in setting.choices)
        msg = f"{setting.name} can only be {allowed}"
        raise ConfigError(msg)
    classes = dict(config.boundaries.classes)
    if cls is setting.shipped:
        classes.pop(setting.name, None)
    else:
        classes[setting.name] = cls.value
    boundaries = config.boundaries.model_copy(update={"classes": classes})
    return config.model_copy(update={"boundaries": boundaries})
