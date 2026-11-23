"""The decision table, the action identity and what counts as a spoken yes."""

from dataclasses import dataclass

import pytest

from jarvis.policy import (
    TRUSTED_ORIGINS,
    OpClass,
    Origin,
    Verdict,
    action_hash,
    decide,
    is_yes,
)


@dataclass(frozen=True)
class _Op:
    name: str
    default_class: OpClass
    side_effect: bool = True
    outbound: bool = False
    lowest_class: OpClass = OpClass.SAFE


@pytest.mark.parametrize(
    ("default", "side_effect", "untrusted", "expected"),
    [
        (OpClass.SAFE, True, False, Verdict.RUN),
        (OpClass.SAFE, True, True, Verdict.CONFIRM),
        (OpClass.SAFE, False, True, Verdict.RUN),
        (OpClass.CONFIRM, True, False, Verdict.CONFIRM),
        (OpClass.CONFIRM, True, True, Verdict.CONFIRM),
        (OpClass.CONFIRM, False, True, Verdict.CONFIRM),
        (OpClass.STRONG, True, False, Verdict.STRONG),
        (OpClass.STRONG, True, True, Verdict.STRONG),
        (OpClass.STRONG, False, True, Verdict.STRONG),
        (OpClass.DENY, True, False, Verdict.DENY),
        (OpClass.DENY, True, True, Verdict.DENY),
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


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("safe", Verdict.STRONG),
        ("confirm", Verdict.STRONG),
        ("strong", Verdict.STRONG),
        ("deny", Verdict.DENY),
    ],
)
def test_an_override_never_takes_an_operation_below_its_lowest_class(
    override: str, expected: Verdict
) -> None:
    op = _Op("stop", OpClass.STRONG, lowest_class=OpClass.STRONG)
    assert decide(op, "stop", {"stop": override}, untrusted_in_context=False).verdict is expected


def test_invalid_override_falls_back_to_default() -> None:
    op = _Op("x", OpClass.CONFIRM)
    assert decide(op, "x", {"x": "whatever"}, untrusted_in_context=False).verdict is Verdict.CONFIRM


def test_untrusted_text_never_lowers_a_class() -> None:
    op = _Op("x", OpClass.SAFE)
    decision = decide(op, "x", {"x": "strong"}, untrusted_in_context=True)
    assert decision.verdict is Verdict.STRONG


@pytest.mark.parametrize(
    ("default", "outbound", "private", "expected"),
    [
        (OpClass.SAFE, True, True, Verdict.CONFIRM),
        (OpClass.SAFE, True, False, Verdict.RUN),
        (OpClass.SAFE, False, True, Verdict.RUN),
        (OpClass.CONFIRM, True, True, Verdict.CONFIRM),
        (OpClass.STRONG, True, True, Verdict.STRONG),
        (OpClass.DENY, True, True, Verdict.DENY),
    ],
)
def test_sending_text_off_the_pc_while_file_text_is_in_context_needs_a_yes(
    default: OpClass, outbound: bool, private: bool, expected: Verdict
) -> None:
    # A read-only search still carries words the model wrote to a service outside this PC.
    op = _Op("x", default, side_effect=False, outbound=outbound)
    decision = decide(op, "x", {}, untrusted_in_context=private, private_in_context=private)
    assert decision.verdict is expected


def test_an_owner_class_still_wins_over_the_private_text_rule() -> None:
    op = _Op("x", OpClass.SAFE, side_effect=False, outbound=True)
    decision = decide(op, "x", {"x": "deny"}, untrusted_in_context=True, private_in_context=True)
    assert decision.verdict is Verdict.DENY


def test_action_hash_binds_operation_and_exact_arguments() -> None:
    base = action_hash("spotify_play", {"query": "a", "kind": "track"})
    assert base == action_hash("spotify_play", {"kind": "track", "query": "a"})
    assert base != action_hash("spotify_play", {"query": "b", "kind": "track"})
    assert base != action_hash("spotify_queue", {"query": "a", "kind": "track"})


@pytest.mark.parametrize(
    ("origin", "trusted"),
    [
        (Origin.SYSTEM, True),
        (Origin.OWNER_SPEECH, True),
        (Origin.TOOL_STATUS, True),
        (Origin.PC_AUDIO, False),
        (Origin.MODEL, False),
        (Origin.SPOTIFY, False),
        (Origin.WEB, False),
        (Origin.FILE, False),
        (Origin.PICTURE, False),
        (Origin.PASTE, False),
        (Origin.PROCESS, False),
        (Origin.SHELL, False),
        (Origin.OS_SKILL, False),
    ],
)
def test_which_origins_are_trusted(origin: Origin, trusted: bool) -> None:
    assert (origin in TRUSTED_ORIGINS) is trusted


def test_every_origin_has_a_stated_trust() -> None:
    # A new origin must be added to the table above, where it starts as untrusted.
    assert len(Origin) == 13


@pytest.mark.parametrize(
    "said",
    [
        "yes",
        "Yes.",
        "Yeah, sure.",
        "go ahead",
        "yep do it",
        "Yes, please.",
        "okay",
        "Sure, go ahead.",
        "yes do it",
        "Of course.",
        "Yes, thank you.",
        "do it",
    ],
)
def test_clear_yes(said: str) -> None:
    assert is_yes(said)


def test_the_assistants_name_does_not_spoil_a_yes() -> None:
    assert is_yes("Yes, Jarvis.", names=("Jarvis",))
    assert is_yes("Jarvis, go ahead", names=("Jarvis",))
    assert not is_yes("Yes, Jarvis.")


@pytest.mark.parametrize(
    "said",
    [
        "",
        "   ",
        "no",
        "yes no",
        "no yes",
        "maybe",
        "yes but play something else instead",
        "yeah nah",
        "yes dont",
        "Yes, don't.",
        "yes not now",
        "yeah nope",
        "sure not really",
        "yes cancel that",
        "yes stop",
        "yep wait",
        "yes never",
        "yes yes yes yes yes yes yes",
        "please",
        "do that",
        "thank you",
        "of course not",
        "yes and no",
        "Jarvis",
    ],
)
def test_anything_else_is_no(said: str) -> None:
    assert not is_yes(said, names=("Jarvis",))
