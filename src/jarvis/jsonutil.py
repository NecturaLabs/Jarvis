"""Typed access to JSON from outside (Spotify, the model server), and text JSON can carry."""

import re
from typing import Any, cast

_LONE_SURROGATE = re.compile(r"[\ud800-\udfff]")


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


def unbroken(text: str) -> str:
    """Return `text` with each lone surrogate replaced by U+FFFD, so it can be written as UTF-8.

    A file name whose bytes are not UTF-8 reaches Python as one (`caf` and U+DCE9); sent to the
    model, saved or logged as UTF-8, it raises.
    """
    return _LONE_SURROGATE.sub("\ufffd", text)


def unbroken_json(value: object) -> object:
    """Return a JSON value with every string in it `unbroken`, keys included."""
    if isinstance(value, str):
        return unbroken(value)
    if isinstance(value, dict):
        items = cast("dict[object, object]", value).items()
        return {unbroken_json(k): unbroken_json(v) for k, v in items}
    if isinstance(value, list | tuple):
        return [unbroken_json(item) for item in cast("list[object]", value)]
    return value
