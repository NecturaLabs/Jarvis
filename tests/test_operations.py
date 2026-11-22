"""Argument validation and the operation registry: what a model's tool call must pass first."""

from typing import Any

import pytest

from jarvis.operations import (
    ArgumentError,
    Operation,
    OperationError,
    Registry,
    Result,
    current_request,
    request_scope,
    validate_args,
)
from jarvis.policy import Origin
from jarvis.shortcuts import Shortcut


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
    args = {"query": "  low tide ", "kind": "track", "percent": 40.0, "on": True}
    assert validate_args(PLAY, args) == {
        "query": "low tide",
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


@pytest.mark.parametrize("percent", [0, 100, 0.0, 100.0])
def test_the_ends_of_a_range_are_accepted(percent: float) -> None:
    assert validate_args(PLAY, {"query": "a", "percent": percent})["percent"] == int(percent)


@pytest.mark.parametrize(
    "percent",
    [float("nan"), float("inf"), float("-inf"), 1e300, 10**5000, -(10**5000)],
    ids=["nan", "inf", "-inf", "1e300", "5000 digits", "-5000 digits"],
)
def test_numbers_no_integer_can_hold_are_rejected(percent: float) -> None:
    with pytest.raises(ArgumentError):
        validate_args(PLAY, {"query": "a", "percent": percent})


@pytest.mark.parametrize(
    "number", [float("inf"), 1e300, 10**5000, 2**31], ids=["inf", "1e300", "5000 digits", "2**31"]
)
def test_an_integer_without_a_stated_range_still_has_one(number: float) -> None:
    op = _op({"position": {"type": "integer"}}, [])

    with pytest.raises(ArgumentError):
        validate_args(op, {"position": number})

    assert validate_args(op, {"position": 2**31 - 1}) == {"position": 2**31 - 1}


def test_a_string_without_a_stated_length_is_capped() -> None:
    op = _op({"note": {"type": "string"}}, [])

    assert validate_args(op, {"note": "x" * 200}) == {"note": "x" * 200}
    with pytest.raises(ArgumentError):
        validate_args(op, {"note": "x" * 201})


def test_an_omitted_argument_takes_the_schema_default() -> None:
    op = _op(
        {
            "query": {"type": "string", "minLength": 1},
            "count": {"type": "integer", "minimum": 1, "maximum": 20, "default": 1},
            "random_order": {"type": "boolean", "default": False},
        },
        ["query"],
    )

    assert validate_args(op, {"query": "a"}) == validate_args(
        op, {"query": "a", "count": 1, "random_order": False}
    )
    assert validate_args(op, {"query": "a", "count": 3})["count"] == 3


def test_unsupported_schema_type_is_rejected() -> None:
    op = _op({"items": {"type": "array"}}, [])
    with pytest.raises(ArgumentError):
        validate_args(op, {"items": []})


def test_registry_offers_only_its_operations() -> None:
    registry = Registry([PLAY])
    assert registry.get("op") is PLAY
    assert registry.get("other") is None
    assert [t["function"]["name"] for t in registry.tool_schemas()] == ["op"]


def test_an_operation_error_is_trusted_status_unless_it_says_otherwise() -> None:
    assert OperationError("no device").origin is Origin.TOOL_STATUS
    assert OperationError("Spotify said: x", origin=Origin.SPOTIFY).origin is Origin.SPOTIFY
    assert str(OperationError("no device")) == "no device"


def test_a_result_has_no_spoken_sentence_unless_given_one() -> None:
    assert not Result("status: paused").spoken
    assert Result("status: paused", spoken="Paused.").spoken == "Paused."


def test_registry_lists_each_operation_shortcut_with_its_name() -> None:
    pause = Shortcut("pause")
    louder = Shortcut("volume up", lambda _groups: {"delta": 10})
    registry = Registry(
        [
            Operation(
                "pause", "p", {"type": "object"}, lambda _a: Result("ok"), shortcuts=(pause,)
            ),
            Operation(
                "volume", "v", {"type": "object"}, lambda _a: Result("ok"), shortcuts=(louder,)
            ),
            PLAY,
        ]
    )

    assert registry.shortcuts() == [("pause", pause), ("volume", louder)]


def test_an_operation_has_no_prepare_step_unless_given_one() -> None:
    assert PLAY.prepare is None


def test_a_string_shorter_than_its_min_length_is_refused() -> None:
    op = _op({"code": {"type": "string", "minLength": 4}}, ["code"])

    with pytest.raises(ArgumentError, match="at least 4"):
        validate_args(op, {"code": " abc "})
    assert validate_args(op, {"code": "abcd"}) == {"code": "abcd"}


def test_there_is_no_current_request_outside_one() -> None:
    assert current_request() is None


def test_each_request_is_its_own_and_ends_with_its_scope() -> None:
    with request_scope(foreign=False) as first:
        assert current_request() is first
        assert not first.foreign
    with request_scope(foreign=True) as second:
        assert current_request() is second
        assert second.foreign
    assert first is not second
    assert first != second
    assert current_request() is None


def test_a_request_scope_ends_even_when_the_request_fails() -> None:
    with pytest.raises(RuntimeError), request_scope(foreign=False):
        raise RuntimeError
    assert current_request() is None
