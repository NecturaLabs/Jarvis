from typing import Any

import pytest

from jarvis.operations import ArgumentError, Operation, Registry, Result, validate_args


def _op(properties: dict[str, Any], required: list[str]) -> Operation:
    return Operation(
        "op",
        "test",
        {"type": "object", "properties": properties, "required": required},
        lambda _args: Result("ok"),
    )


PLAY = _op(
    {
        "query": {"type": "string", "minLength": 1, "maxLength": 10},
        "kind": {"type": "string", "enum": ["track", "album"]},
        "percent": {"type": "integer", "minimum": 0, "maximum": 100},
        "on": {"type": "boolean"},
    },
    ["query"],
)


def test_valid_arguments_are_cleaned() -> None:
    args = {"query": "  bad omens ", "kind": "track", "percent": 40.0, "on": True}
    assert validate_args(PLAY, args) == {
        "query": "bad omens",
        "kind": "track",
        "percent": 40,
        "on": True,
    }


@pytest.mark.parametrize(
    "args",
    [
        [],
        "query",
        {},
        {"query": "a", "extra": 1},
        {"query": 3},
        {"query": "   "},
        {"query": "x" * 11},
        {"query": "a", "kind": "song"},
        {"query": "a", "percent": 101},
        {"query": "a", "percent": -1},
        {"query": "a", "percent": 4.5},
        {"query": "a", "percent": True},
        {"query": "a", "percent": "40"},
        {"query": "a", "on": "yes"},
    ],
)
def test_invalid_arguments_are_rejected(args: object) -> None:
    with pytest.raises(ArgumentError):
        validate_args(PLAY, args)


def test_unsupported_schema_type_is_rejected() -> None:
    op = _op({"items": {"type": "array"}}, [])
    with pytest.raises(ArgumentError):
        validate_args(op, {"items": []})


def test_registry_offers_only_its_operations() -> None:
    registry = Registry([PLAY])
    assert registry.get("op") is PLAY
    assert registry.get("other") is None
    assert [t["function"]["name"] for t in registry.tool_schemas()] == ["op"]
