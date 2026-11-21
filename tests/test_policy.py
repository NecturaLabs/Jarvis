from dataclasses import dataclass

import pytest

from jarvis.policy import OpClass, Verdict, action_hash, decide, is_yes


@dataclass(frozen=True)
class _Op:
    name: str
    default_class: OpClass
    side_effect: bool = True


@pytest.mark.parametrize(
    ("default", "side_effect", "untrusted", "expected"),
    [
        (OpClass.SAFE, True, False, Verdict.RUN),
        (OpClass.SAFE, True, True, Verdict.CONFIRM),
        (OpClass.SAFE, False, True, Verdict.RUN),
        (OpClass.CONFIRM, True, False, Verdict.CONFIRM),
        (OpClass.CONFIRM, True, True, Verdict.CONFIRM),
        (OpClass.STRONG, True, True, Verdict.STRONG),
        (OpClass.DENY, True, False, Verdict.DENY),
        (OpClass.DENY, False, True, Verdict.DENY),
    ],
)
def test_decide_table(
    default: OpClass, side_effect: bool, untrusted: bool, expected: Verdict
) -> None:
    op = _Op("x", default, side_effect)
    assert decide(op, "x", {}, untrusted_in_context=untrusted).verdict is expected


def test_unknown_operation_is_denied() -> None:
    decision = decide(None, "rm_rf", {}, untrusted_in_context=False)
    assert decision.verdict is Verdict.DENY
    assert decision.op_class is None


def test_owner_override_wins_over_default() -> None:
    op = _Op("x", OpClass.SAFE)
    assert decide(op, "x", {"x": "deny"}, untrusted_in_context=False).verdict is Verdict.DENY
    assert decide(op, "x", {"x": "confirm"}, untrusted_in_context=False).verdict is Verdict.CONFIRM


def test_invalid_override_falls_back_to_default() -> None:
    op = _Op("x", OpClass.CONFIRM)
    assert decide(op, "x", {"x": "whatever"}, untrusted_in_context=False).verdict is Verdict.CONFIRM


def test_untrusted_text_never_lowers_a_class() -> None:
    op = _Op("x", OpClass.SAFE)
    decision = decide(op, "x", {"x": "strong"}, untrusted_in_context=True)
    assert decision.verdict is Verdict.STRONG


def test_action_hash_binds_operation_and_exact_arguments() -> None:
    base = action_hash("spotify_play", {"query": "a", "kind": "track"})
    assert base == action_hash("spotify_play", {"kind": "track", "query": "a"})
    assert base != action_hash("spotify_play", {"query": "b", "kind": "track"})
    assert base != action_hash("spotify_queue", {"query": "a", "kind": "track"})


@pytest.mark.parametrize("said", ["yes", "Yes.", "yeah sure", "go ahead", "yep do it"])
def test_clear_yes(said: str) -> None:
    assert is_yes(said)


@pytest.mark.parametrize(
    "said", ["", "no", "yes no", "no yes", "maybe", "yes but play something else instead"]
)
def test_anything_else_is_no(said: str) -> None:
    assert not is_yes(said)
