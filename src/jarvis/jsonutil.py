"""Typed access to JSON from outside (Spotify, the model server)."""

from typing import Any, cast


def obj(value: object) -> dict[str, Any]:
    """Return `value` as a JSON object, or an empty one when it is anything else."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def objs(value: object) -> list[dict[str, Any]]:
    """Return the JSON objects in a list, dropping anything else."""
    if not isinstance(value, list):
        return []
    return [
        cast("dict[str, Any]", item)
        for item in cast("list[object]", value)
        if isinstance(item, dict)
    ]
