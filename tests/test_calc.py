"""The calculator: exact arithmetic on an expression the model writes, and nothing else."""

import pytest

from jarvis import calc
from jarvis.calc import CalcError, evaluate, operations
from jarvis.config import FormatConfig
from jarvis.operations import ArgumentError, validate_args
from jarvis.policy import OpClass


@pytest.mark.parametrize(
    ("expression", "value"),
    [
        ("48 * 31", "1488"),  # "how much is 48 times 31"
        ("30 / 120 * 100", "25"),  # "what percentage is 30 out of 120"
        ("15% * 80", "12"),  # "what is 15 percent of 80"
        ("200 * (1 + 7.5%)", "215"),  # "200 plus 7.5 percent"
        ("0.1 + 0.2", "0.3"),  # exact, not 0.30000000000000004
        ("7 / 2", "3.5"),
        ("1 / 3", "0.3333333333"),
        ("2 ** 10", "1024"),
        ("2 ^ 10", "1024"),  # said "to the power of", often written ^
        ("-3 + +4", "1"),
        ("17 // 5", "3"),
        ("17 mod 5", "2"),
        ("17 % 5", "2"),
        ("(3 + 4) * 2", "14"),
        ("3 + 4 * 2", "11"),
        ("max(5,100)", "100"),  # a comma between arguments is never a thousands separator
        ("log(1000,10)", "3"),
        ("log(10 ** 400, 10)", "400"),
        ("7 % -3", "-2"),  # % with space around it, before a signed number, is the remainder
        ("80 * 15% + 80", "92"),  # a percentage touching its number, then a plus, is a percentage
        ("100 - 20% - 5", "94.8"),
        ("15% + 5%", "0.2"),
        ("50% + 10", "10.5"),
        ("200 * 10% - 5", "15"),
        ("17 % + 5", "2"),  # spaced on both sides: the remainder of 17 by +5
        ("17%5", "2"),  # touching both numbers: the remainder
        ("17%(2 + 3)", "2"),
        ("(7 // 2)!", "6"),  # a floor division is a whole number like any other
        ("factorial(7 // 2)", "6"),
        ("round(2.567, 4 // 2)", "2.57"),
        ("10**18 // 7", "142857142857142857"),
        ("(10**18 + 1) // 1", "1000000000000000001"),
        ("7.5 // 2", "3"),
        ("10 ** 5000 / 10 ** 4999", "10"),
        ("1e3 + 1", "1001"),
        ("sqrt(144)", "12"),
        ("sqrt(2)", "1.414213562"),
        ("abs(-5)", "5"),
        ("round(2.567, 2)", "2.57"),
        ("floor(2.7) + ceil(2.1)", "5"),
        ("log10(1000)", "3"),
        ("log(8, 2)", "3"),
        ("ln(e)", "1"),
        ("pi * 2", "6.283185307"),
        ("max(3, 9, 4) - min(3, 9, 4)", "6"),
        ("10!", "3628800"),
        ("factorial(5)", "120"),
        ("2 \u00d7 3 \u00f7 4 \u2212 1", "0.5"),  # the signs as typed: times, divide, minus
        ("1 / 3 * 3", "1"),
        ("123456789 * 987654321", "121932631112635269"),
        ("round(2.5)", "3"),  # a half rounds away from zero, as calculators do
        ("round(-2.5)", "-3"),
        ("round(0.125, 2)", "0.13"),
        ("round(cos(0) * 0.5)", "1"),
        ("cos(pi) ** 2", "1"),  # a negative number to a whole power is real
        ("(sin(4) * 2) ** 2", "2.291000068"),
        ("cos(pi) ** 3", "-1"),
    ],
)
def test_an_expression_is_worked_out_exactly(expression: str, value: str) -> None:
    assert evaluate(expression) == value


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('true')",
        "open('/etc/passwd')",
        "x + 1",
        "(1).__class__",
        "[1, 2]",
        "'a' * 3",
        "lambda: 1",
        "1 if 1 else 2",
        "1 < 2",
        "sqrt",
        "sqrt(x=4)",
        "eval('1')",
        "1; 2",
        "",
        "   ",
    ],
)
def test_anything_but_arithmetic_is_refused(expression: str) -> None:
    with pytest.raises(CalcError):
        evaluate(expression)


@pytest.mark.parametrize(
    ("expression", "says"),
    [
        ("1 / 0", "divide by zero"),
        ("5 mod 0", "divide by zero"),
        ("sqrt(-1)", "not a real number"),
        ("log(0)", "not a real number"),
        ("(-8) ** 0.5", "not a real number"),
        ("2.5!", "whole number"),
        ("(-1)!", "whole number"),
    ],
)
def test_an_impossible_sum_says_why(expression: str, says: str) -> None:
    with pytest.raises(CalcError, match=says):
        evaluate(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "9 ** 9 ** 9",  # would take the process down
        "10 ** 100000",
        "2 ** 2 ** 2 ** 2 ** 2",
        "100000!",
        "factorial(10 ** 9)",
        "exp(1000)",
        "(" * 60 + "1" + ")" * 60,
        "1 + " * 300 + "1",
    ],
)
def test_a_sum_too_big_to_work_out_is_refused_at_once(expression: str) -> None:
    with pytest.raises(CalcError, match="too"):
        evaluate(expression)


@pytest.mark.parametrize("expression", ["1,234 + 1", "1,000"])
def test_a_number_written_with_thousands_commas_is_refused(expression: str) -> None:
    # Requirement changed (review): a comma always separates arguments, so the model writes 1234.
    with pytest.raises(CalcError):
        evaluate(expression)


@pytest.mark.parametrize(
    ("expression", "value"),
    [("3 * 10 ** 5000", "3e+5000"), ("10 ** 400 / 3", "3.333333333e+399")],
)
def test_an_answer_past_python_s_own_limits_is_still_given(expression: str, value: str) -> None:
    assert evaluate(expression) == value


@pytest.mark.parametrize("expression", ["1e400", "sqrt(2 * 10 ** 700)"])
def test_a_float_too_big_says_so(expression: str) -> None:
    with pytest.raises(CalcError, match="too big"):
        evaluate(expression)


def test_a_very_large_or_small_answer_is_given_in_powers_of_ten() -> None:
    assert evaluate("10 ** 30") == "1e+30"
    assert evaluate("2 ** 100") == "1.2676506e+30"
    assert evaluate("1 / 10 ** 12") == "1e-12"


def test_the_operation_is_a_safe_pure_read() -> None:
    (op,) = operations()

    assert op.name == "calculate"
    assert op.default_class is OpClass.SAFE
    assert op.side_effect is False
    assert op.outbound is False


def test_the_operation_answers_with_the_sum_and_its_value_in_groups_of_three() -> None:
    (op,) = operations()

    result = op.run({"expression": "48 * 31"})

    assert result.ok
    assert result.text == "48 * 31 = 1,488"


@pytest.mark.parametrize(
    ("separator", "expression", "shown"),
    [
        (",", "1234567.25 * 2", "2,469,134.5"),
        (",", "-1000000", "-1,000,000"),
        (",", "999", "999"),
        (",", "10 ** 30", "1e+30"),
        (" ", "1234567", "1 234 567"),
        ("'", "1234567", "1'234'567"),
        ("", "1234567", "1234567"),
    ],
)
def test_the_thousands_separator_is_the_one_configured(
    separator: str, expression: str, shown: str
) -> None:
    (op,) = operations(FormatConfig.model_validate({"thousands_separator": separator}))

    assert op.run({"expression": expression}).text == f"{expression} = {shown}"


def test_the_separator_is_a_comma_unless_set() -> None:
    assert FormatConfig().thousands_separator == ","


def test_the_operation_reports_an_impossible_sum_as_a_failure() -> None:
    (op,) = operations()

    result = op.run({"expression": "1 / 0"})

    assert not result.ok
    assert "divide by zero" in result.text


@pytest.mark.parametrize(
    "args",
    [{}, {"expression": "1 + 1", "extra": 1}, {"expression": "1" * (calc.MAX_CHARS + 1)}],
)
def test_arguments_are_checked_before_anything_runs(args: dict[str, object]) -> None:
    (op,) = operations()

    with pytest.raises(ArgumentError):
        validate_args(op, args)
