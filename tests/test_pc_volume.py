from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from jarvis.operations import (
    ArgumentError,
    Operation,
    OperationError,
    Registry,
    Result,
    validate_args,
)
from jarvis.pc.volume import SINK, VolumeOps

if TYPE_CHECKING:
    from collections.abc import Sequence


class FakeWpctl:
    """Answers `get-volume` with the volume it holds and records every other call."""

    def __init__(self, volume: str = "Volume: 0.25\n", *, error: str = "") -> None:
        super().__init__()
        self.volume = volume
        self.error = error
        self.calls: list[list[str]] = []

    def __call__(self, args: Sequence[str]) -> str:
        self.calls.append(list(args))
        if self.error:
            raise OperationError(self.error)
        return self.volume if args[0] == "get-volume" else ""


def _op(wpctl: FakeWpctl) -> Operation:
    (op,) = VolumeOps(wpctl).operations()
    return op


def _run(wpctl: FakeWpctl, args: dict[str, Any]) -> tuple[Result, dict[str, Any]]:
    op = _op(wpctl)
    checked = validate_args(op, args)
    prepared = op.prepare(checked) if op.prepare else checked
    return op.run(prepared), prepared


def test_no_argument_reads_the_volume_and_changes_nothing() -> None:
    wpctl = FakeWpctl()

    result, _ = _run(wpctl, {})

    assert result.spoken == "The PC's volume is 25 percent."
    assert wpctl.calls == [["get-volume", SINK]]


def test_a_muted_output_says_so() -> None:
    result, _ = _run(FakeWpctl("Volume: 0.40 [MUTED]\n"), {})
    assert result.spoken == "The PC's volume is 40 percent, and it is muted."


def test_a_percentage_sets_the_volume() -> None:
    wpctl = FakeWpctl()

    result, _ = _run(wpctl, {"percent": 40})

    assert ["set-volume", SINK, "0.40"] in wpctl.calls
    assert result.spoken == "The PC's volume is now 40 percent."
    assert result.acted


@pytest.mark.parametrize(
    ("now", "delta", "target"), [(25, 10, 35), (25, -10, 15), (80, 50, 100), (25, -50, 0)]
)
def test_a_change_is_bound_to_its_target_before_any_question(
    now: int, delta: int, target: int
) -> None:
    wpctl = FakeWpctl(f"Volume: {now / 100:.2f}\n")

    _, prepared = _run(wpctl, {"delta": delta})

    assert (prepared["current"], prepared["target"]) == (now, target)
    assert ["set-volume", SINK, f"{target / 100:.2f}"] in wpctl.calls
    assert _op(wpctl).spoken_summary(prepared) == (
        f"change the PC's volume from {now} to {target} percent"
    )


def test_the_confirmed_change_sets_the_bound_target_even_if_the_volume_moved() -> None:
    wpctl = FakeWpctl()
    op = _op(wpctl)
    assert op.prepare is not None
    prepared = op.prepare(validate_args(op, {"delta": 10}))
    wpctl.volume = "Volume: 0.80\n"  # someone turned it up while the question was asked

    op.run(prepared)

    assert wpctl.calls[-1] == ["set-volume", SINK, "0.35"]


@pytest.mark.parametrize(("mute", "flag"), [(True, "1"), (False, "0")])
def test_mute_and_unmute(mute: bool, flag: str) -> None:
    wpctl = FakeWpctl()

    result, _ = _run(wpctl, {"mute": mute})

    assert wpctl.calls == [["set-mute", SINK, flag]]
    assert result.spoken == ("Muted." if mute else "Unmuted.")


def test_more_than_one_way_to_change_it_is_refused() -> None:
    wpctl = FakeWpctl()
    with pytest.raises(OperationError, match="one of"):
        _run(wpctl, {"percent": 40, "mute": True})
    assert not [call for call in wpctl.calls if call[0] != "get-volume"]


@pytest.mark.parametrize(
    "args", [{"percent": 101}, {"percent": -1}, {"delta": 51}, {"volume": 3}, {"mute": "yes"}]
)
def test_arguments_out_of_bounds_are_refused_before_anything_runs(args: dict[str, Any]) -> None:
    wpctl = FakeWpctl()
    with pytest.raises(ArgumentError):
        validate_args(_op(wpctl), args)
    assert wpctl.calls == []


def test_a_sound_server_that_cannot_be_reached_is_said() -> None:
    with pytest.raises(OperationError, match="sound server"):
        _run(FakeWpctl(error="the sound server did not answer"), {"percent": 40})


def test_an_answer_wpctl_does_not_give_normally_is_refused() -> None:
    with pytest.raises(OperationError, match="read the PC's volume"):
        _run(FakeWpctl("nothing here\n"), {})


def test_the_operation_fits_the_registry() -> None:
    registry = Registry(VolumeOps(FakeWpctl()).operations())
    assert [schema["function"]["name"] for schema in registry.tool_schemas()] == ["pc_volume"]


def test_turning_up_a_volume_above_the_usual_top_keeps_it_rather_than_lowering_it() -> None:
    _, prepared = _run(FakeWpctl("Volume: 1.50\n"), {"delta": 10})
    assert prepared["target"] == 150


def test_turning_down_a_volume_above_the_usual_top_lowers_it_by_the_step() -> None:
    _, prepared = _run(FakeWpctl("Volume: 1.50\n"), {"delta": -10})
    assert prepared["target"] == 140


def test_the_description_names_spotifys_own_tools_for_turning_spotify_up_or_down() -> None:
    # With Spotify playing, "louder please" turned up the whole PC (#65): the description named
    # only spotify_set_volume, while turning up needs spotify_change_volume.
    (op,) = VolumeOps(FakeWpctl()).operations()

    assert "spotify_change_volume" in op.description
    assert "spotify_set_volume" in op.description
