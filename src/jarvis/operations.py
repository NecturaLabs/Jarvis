"""Typed operations: a name, a JSON schema for its arguments, a class and an executor."""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from jarvis.policy import OpClass, Origin

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from jarvis.shortcuts import Shortcut

MAX_STRING = 200
MAX_INTEGER = 2**31 - 1


class UncertainOutcomeError(Exception):
    """An operation may or may not have taken effect (a timeout after the request was sent)."""


class ArgumentError(ValueError):
    """A tool call's arguments do not match the operation's schema."""


class OperationError(Exception):
    """An operation could not do what was asked; the message is what the model is told.

    `origin` says where the message's words come from: Jarvis's own status text unless the
    operation passes on text an outside service wrote.
    """

    def __init__(self, message: str, *, origin: Origin = Origin.TOOL_STATUS) -> None:
        """Keep the message and the origin of its words."""
        super().__init__(message)
        self.origin = origin


@dataclass(frozen=True, slots=True)
class Result:
    """What an operation returns to the model, with the origin of its text.

    `spoken` is a finished sentence for the owner. An operation sets it when its result needs no
    wording by the model, which lets a shortcut request be answered without the model.
    """

    text: str
    origin: Origin = Origin.TOOL_STATUS
    ok: bool = True
    spoken: str = ""


@dataclass(frozen=True, slots=True)
class Operation:
    """One thing Jarvis can do. `parameters` is a JSON schema object with only flat properties.

    `prepare` runs on the validated arguments once the policy has not refused the call, before
    any confirmation question. It returns the arguments the call will run with, so an operation
    that acts on "whatever is current" can bind the exact target the owner is asked about.
    `shortcuts` are the phrasings that mean this operation without the model's help.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any]], Result]
    default_class: OpClass = OpClass.SAFE
    side_effect: bool = True
    spoken_summary: Callable[[dict[str, Any]], str] = field(default=lambda _args: "do that")
    prepare: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    shortcuts: tuple[Shortcut, ...] = ()

    def tool_schema(self) -> dict[str, Any]:
        """Return the OpenAI tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _check_string(name: str, spec: dict[str, Any], value: object) -> str:
    if not isinstance(value, str):
        raise ArgumentError(f"{name} must be a string")
    text = value.strip()
    if not text and spec.get("minLength", 0) > 0:
        raise ArgumentError(f"{name} must not be empty")
    if len(text) > spec.get("maxLength", MAX_STRING):
        raise ArgumentError(f"{name} is too long")
    if "enum" in spec and text not in spec["enum"]:
        raise ArgumentError(f"{name} must be one of {spec['enum']}")
    return text


def _check_integer(name: str, spec: dict[str, Any], value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArgumentError(f"{name} must be an integer")
    if isinstance(value, float) and not (math.isfinite(value) and value.is_integer()):
        raise ArgumentError(f"{name} must be an integer")
    number = int(value)
    low, high = spec.get("minimum", -MAX_INTEGER), spec.get("maximum", MAX_INTEGER)
    if not low <= number <= high:
        raise ArgumentError(f"{name} must be between {low} and {high}")
    return number


def _check_value(name: str, spec: dict[str, Any], value: object) -> object:
    kind = spec.get("type")
    if kind == "string":
        return _check_string(name, spec, value)
    if kind == "integer":
        return _check_integer(name, spec, value)
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ArgumentError(f"{name} must be true or false")
        return value
    raise ArgumentError(f"{name} has an unsupported type")


def validate_args(op: Operation, args: object) -> dict[str, Any]:
    """Return the arguments cleaned against the operation's schema, with its defaults filled in.

    Raises:
        ArgumentError: the arguments are not an object, miss a required key, have an unknown key or
            a value of the wrong type or range.
    """
    if not isinstance(args, dict):
        raise ArgumentError("arguments must be a JSON object")
    props: dict[str, dict[str, Any]] = op.parameters.get("properties", {})
    required: list[str] = op.parameters.get("required", [])
    cleaned: dict[str, Any] = {}
    for key, value in cast("dict[str, object]", args).items():
        if key not in props:
            raise ArgumentError(f"unknown argument {key!r}")
        cleaned[key] = _check_value(key, props[key], value)
    missing = [key for key in required if key not in cleaned]
    if missing:
        raise ArgumentError(f"missing argument(s): {', '.join(missing)}")
    for key, spec in props.items():
        if key not in cleaned and "default" in spec:
            cleaned[key] = spec["default"]
    return cleaned


class Registry:
    """The operations offered to the model; anything else is refused."""

    def __init__(self, ops: Iterable[Operation]) -> None:
        """Index the operations by name."""
        super().__init__()
        self._ops = {op.name: op for op in ops}

    def get(self, name: str) -> Operation | None:
        """Return the operation or None."""
        return self._ops.get(name)

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return every operation as an OpenAI tool."""
        return [op.tool_schema() for op in self._ops.values()]

    def names(self) -> list[str]:
        """Return the operation names."""
        return list(self._ops)

    def shortcuts(self) -> list[tuple[str, Shortcut]]:
        """Return every shortcut phrasing with the name of the operation it means."""
        return [(op.name, shortcut) for op in self._ops.values() for shortcut in op.shortcuts]
