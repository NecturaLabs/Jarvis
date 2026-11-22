"""The calculator: works out an arithmetic expression exactly, so a sum is never guessed.

The model turns what the owner said ("48 times 31", "what percentage is 30 out of 120") into an
expression; this works it out. Only arithmetic is read: numbers, + - * / // % **, brackets, a
few named functions and constants. The text is parsed into Python's syntax tree and walked here,
never run: a name, attribute, call or value outside that list is refused before anything is
worked out. Sums are kept as exact fractions (0.1 + 0.2 is 0.3); a square root, logarithm or
other function that has no exact value gives a float. Every step is bounded, so no expression can
hang Jarvis or fill its memory.
"""

import ast
import math
import re
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from jarvis.config import CalcConfig
from jarvis.operations import Operation, Result

if TYPE_CHECKING:
    from collections.abc import Callable

MAX_CHARS = 300  # a spoken sum is far shorter
MAX_NODES = 200  # numbers, signs and calls in one expression
MAX_DEPTH = 30  # brackets inside brackets
MAX_BITS = 100_000  # the size of an exact number (about 30,000 digits)
MAX_EXPONENT = 10_000
MAX_FACTORIAL = 1000
SIGNIFICANT = 10  # digits in an answer that is not a whole number
EXACT_DIGITS = 20  # a whole number up to this long is given in full

type Number = Fraction | float


class CalcError(ValueError):
    """The expression is not arithmetic, or cannot be worked out; the message says why."""


def _real(value: float) -> float:
    if isinstance(value, complex) or math.isnan(value):
        msg = "the answer is not a real number"
        raise CalcError(msg)
    if math.isinf(value):
        msg = "the answer is too big to work out"
        raise CalcError(msg)
    return value


def _checked(value: Number) -> Number:
    if isinstance(value, Fraction):
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
            msg = "the answer is too big to work out"
            raise CalcError(msg)
        return value
    return _real(value)


def _whole(value: Number, what: str) -> int:
    if isinstance(value, Fraction) and value.denominator == 1:
        return value.numerator
    if isinstance(value, float) and value.is_integer():
        return int(value)
    msg = f"{what} needs a whole number"
    raise CalcError(msg)


def _float_call(function: Callable[..., float], *args: Number) -> float:
    try:
        return _real(function(*(float(arg) for arg in args)))
    except OverflowError as exc:
        msg = "the answer is too big to work out"
        raise CalcError(msg) from exc
    except ValueError as exc:  # math domain error
        msg = "the answer is not a real number"
        raise CalcError(msg) from exc


def _sqrt(x: Number) -> Number:
    if x < 0:
        msg = "the square root of a negative number is not a real number"
        raise CalcError(msg)
    if isinstance(x, Fraction):
        top, bottom = math.isqrt(x.numerator), math.isqrt(x.denominator)
        if top * top == x.numerator and bottom * bottom == x.denominator:
            return Fraction(top, bottom)
    return _float_call(math.sqrt, x)


def _log(x: Number, base: Number | None = None) -> Number:
    if x <= 0 or (base is not None and (base <= 0 or base == 1)):
        msg = "that logarithm is not a real number"
        raise CalcError(msg)
    if base is not None and isinstance(x, Fraction) and isinstance(base, Fraction):
        exact = _exact_log(x, base)
        if exact is not None:
            return exact
    if isinstance(x, Fraction) and (base is None or isinstance(base, Fraction)):
        return _real(_ln(x) / (1.0 if base is None else _ln(base)))
    if base is None:
        return _float_call(math.log, x)
    return _float_call(math.log, x, base)


def _ln(x: Fraction) -> float:
    """The natural logarithm of a positive fraction of any size (math.log takes big integers)."""
    return math.log(x.numerator) - math.log(x.denominator)


def _exact_log(x: Fraction, base: Fraction) -> Fraction | None:
    """Return the logarithm when it is a whole number (log10 of 1000 is exactly 3)."""
    guess = round(_ln(x) / _ln(base))
    if abs(guess) <= MAX_EXPONENT and base**guess == x:
        return Fraction(guess)
    return None


def _factorial(n: Number) -> Number:
    whole = _whole(n, "a factorial")
    if whole < 0:
        msg = "a factorial needs a whole number of 0 or more"
        raise CalcError(msg)
    if whole > MAX_FACTORIAL:
        msg = f"the factorial of more than {MAX_FACTORIAL} is too big to work out"
        raise CalcError(msg)
    return Fraction(math.factorial(whole))


def _round(x: Number, places: Number | None = None) -> Number:
    digits = 0 if places is None else _whole(places, "rounding")
    if not 0 <= digits <= SIGNIFICANT:
        msg = f"rounding takes 0 to {SIGNIFICANT} places"
        raise CalcError(msg)
    if isinstance(x, Fraction):
        return Fraction(round(x, digits))
    return round(x, digits)


def _floor(x: Number) -> Number:
    return Fraction(math.floor(x))


def _ceil(x: Number) -> Number:
    return Fraction(math.ceil(x))


def _abs(x: Number) -> Number:
    return abs(x)


def _min(*values: Number) -> Number:
    return min(values)


def _max(*values: Number) -> Number:
    return max(values)


def _float_function(function: Callable[[float], float]) -> Callable[[Number], Number]:
    def call(x: Number) -> Number:
        return _float_call(function, x)

    return call


def _log10(x: Number) -> Number:
    return _log(x, Fraction(10))


def _log2(x: Number) -> Number:
    return _log(x, Fraction(2))


_FUNCTIONS: dict[str, tuple[Callable[..., Number], int, int]] = {
    # name: (function, fewest arguments, most arguments)
    "sqrt": (_sqrt, 1, 1),
    "abs": (_abs, 1, 1),
    "round": (_round, 1, 2),
    "floor": (_floor, 1, 1),
    "ceil": (_ceil, 1, 1),
    "log": (_log, 1, 2),
    "ln": (_log, 1, 1),
    "log10": (_log10, 1, 1),
    "log2": (_log2, 1, 1),
    "exp": (_float_function(math.exp), 1, 1),
    "sin": (_float_function(math.sin), 1, 1),
    "cos": (_float_function(math.cos), 1, 1),
    "tan": (_float_function(math.tan), 1, 1),
    "radians": (_float_function(math.radians), 1, 1),
    "degrees": (_float_function(math.degrees), 1, 1),
    "factorial": (_factorial, 1, 1),
    "min": (_min, 1, 20),
    "max": (_max, 1, 20),
}
_CONSTANTS: dict[str, Number] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _power(base: Number, exponent: Number) -> Number:
    if isinstance(base, Fraction) and isinstance(exponent, Fraction) and exponent.denominator == 1:
        power = exponent.numerator
        size = max(base.numerator.bit_length(), base.denominator.bit_length())
        if abs(power) > MAX_EXPONENT or size * abs(power) > MAX_BITS:
            msg = "the answer is too big to work out"
            raise CalcError(msg)
        if base == 0 and power < 0:
            msg = "you can't divide by zero"
            raise CalcError(msg)
        return base**power
    if base < 0:
        msg = "a negative number to a fractional power is not a real number"
        raise CalcError(msg)
    return _float_call(math.pow, base, exponent)


def _divide(op: ast.operator, left: Number, right: Number) -> Number:
    if right == 0:
        msg = "you can't divide by zero"
        raise CalcError(msg)
    if isinstance(op, ast.Div):
        return left / right
    if isinstance(op, ast.FloorDiv):
        return left // right
    return left % right


def _binary(op: ast.operator, left: Number, right: Number) -> Number:
    if isinstance(op, ast.Pow):
        return _power(left, right)
    if isinstance(op, ast.Div | ast.FloorDiv | ast.Mod):
        return _divide(op, left, right)
    if isinstance(left, float) or isinstance(right, float):
        left, right = float(left), float(right)
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    msg = "only + - * / // % and powers are worked out"
    raise CalcError(msg)


def _call(node: ast.Call) -> Number:
    if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
        msg = f"there is no function {ast.unparse(node.func)!r} here"
        raise CalcError(msg)
    if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
        msg = "a function here takes plain numbers only"
        raise CalcError(msg)
    function, fewest, most = _FUNCTIONS[node.func.id]
    if not fewest <= len(node.args) <= most:
        msg = f"{node.func.id} takes {fewest} to {most} numbers"
        raise CalcError(msg)
    return function(*(_value(arg) for arg in node.args))


def _constant(value: object) -> Number:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = "only numbers are worked out"
        raise CalcError(msg)
    if isinstance(value, int):
        return _checked(Fraction(value))
    if math.isinf(value):
        msg = "the number is too big to work out"
        raise CalcError(msg)
    return _checked(Fraction(repr(value)))  # 0.1 is one tenth, as written


def _value(node: ast.expr) -> Number:
    if isinstance(node, ast.Constant):
        return _constant(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        operand = _value(node.operand)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        return _checked(_binary(node.op, _value(node.left), _value(node.right)))
    if isinstance(node, ast.Call):
        return _checked(_call(node))
    msg = f"{ast.unparse(node)!r} is not arithmetic"
    raise CalcError(msg)


def _postfix_factorials(text: str) -> str:
    """Rewrite `n!` and `(…)!` as `factorial(…)`.

    Raises:
        CalcError: a `!` follows no number or bracket.
    """
    while (mark := text.find("!")) != -1:
        start = mark
        if start > 0 and text[start - 1] == ")":
            depth = 0
            for start in range(mark - 1, -1, -1):  # start is used after the loop
                depth += {")": 1, "(": -1}.get(text[start], 0)
                if depth == 0:
                    break
        else:
            while start > 0 and (text[start - 1].isdigit() or text[start - 1] == "."):
                start -= 1
        if start == mark:
            msg = "a factorial needs a number before the !"
            raise CalcError(msg)
        text = f"{text[:start]}factorial({text[start:mark]}){text[mark + 1 :]}"
    return text


_NUMBER = r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_PERCENT = re.compile(rf"({_NUMBER})\s*%(?!\s*[-+\d(.])")  # 15% is a fifteenth part; 17 % 5 is mod
_SPOKEN = {  # signs as people type or paste them
    "\u00d7": "*",  # multiplication sign
    "\u00b7": "*",  # middle dot
    "\u00f7": "/",  # division sign
    "\u2212": "-",  # minus sign
    "\u2013": "-",  # en dash
    "^": "**",
}


def _normalised(expression: str) -> str:
    text = expression.strip()
    for said, written in _SPOKEN.items():
        text = text.replace(said, written)
    text = re.sub(r"\bmod\b", "%", text)
    text = _PERCENT.sub(r"(\1/100)", text)
    return _postfix_factorials(text)


def _nesting(text: str) -> int:
    depth = deepest = 0
    for char in text:
        depth += {"(": 1, ")": -1}.get(char, 0)
        deepest = max(deepest, depth)
    return deepest


def _parse(expression: str) -> ast.expr:
    if len(expression) > MAX_CHARS:
        msg = f"the expression is too long (more than {MAX_CHARS} characters)"
        raise CalcError(msg)
    text = _normalised(expression)
    if not text:
        msg = "there is nothing to work out"
        raise CalcError(msg)
    if _nesting(text) > MAX_DEPTH:
        msg = "the expression has too many brackets inside brackets"
        raise CalcError(msg)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        msg = f"{expression!r} is not an expression I can read"
        raise CalcError(msg) from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        msg = "the expression is too long"
        raise CalcError(msg)
    return tree.body


def _shown(value: Number) -> str:
    """Return the answer as people write it: whole, or to ten significant digits."""
    if isinstance(value, Fraction):
        if value.denominator == 1 and abs(value.numerator) < 10**EXACT_DIGITS:
            return str(value.numerator)
        with localcontext() as exact:  # Decimal, not str or float: no size limit of either
            exact.prec = SIGNIFICANT + 5
            quotient = Decimal(value.numerator) / Decimal(value.denominator)
        text = format(quotient, f".{SIGNIFICANT}g")
    else:
        number = float(value)
        if number.is_integer() and abs(number) < 10**EXACT_DIGITS:
            return str(int(number))
        text = format(Decimal(repr(number)), f".{SIGNIFICANT}g")
    mantissa, _, power = text.partition("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{power}" if power else mantissa


def _grouped(text: str, separator: str) -> str:
    """Put `separator` between groups of three digits in the whole part of `text`."""
    if not separator or "e" in text:
        return text
    sign, digits = ("-", text[1:]) if text.startswith("-") else ("", text)
    whole, point, fraction = digits.partition(".")
    return f"{sign}{int(whole):,}".replace(",", separator) + point + fraction


def evaluate(expression: str) -> str:
    """Work out `expression` and return the answer as text.

    Raises:
        CalcError: it is not arithmetic, is too long or too big, or has no real answer (a
            division by zero, the square root of a negative number).
    """
    try:
        return _shown(_value(_parse(expression)))
    except CalcError:
        raise
    except RecursionError as exc:
        msg = "the expression is too long"
        raise CalcError(msg) from exc
    except (OverflowError, ValueError) as exc:  # a float past its range, or a Python size limit
        msg = "the answer is too big to work out"
        raise CalcError(msg) from exc


def operations(config: CalcConfig | None = None) -> list[Operation]:
    """Return `calculate`, writing answers as `config` says (groups of three by default)."""
    separator = (config or CalcConfig()).thousands_separator

    def calculate(args: dict[str, Any]) -> Result:
        expression = str(args["expression"]).strip()
        try:
            value = evaluate(expression)
        except CalcError as exc:
            return Result(f"cannot work out {expression!r}: {exc}", ok=False)
        return Result(f"{expression} = {_grouped(value, separator)}")

    return [
        Operation(
            "calculate",
            "Work out arithmetic exactly, instead of your own sums. Write the owner's words as an"
            " expression, numbers without commas: 48 * 31; 30 / 120 * 100 for a percentage;"
            " 15% * 80; 100 * (1 - 20%) for 20 percent off. Has // % ** ! sqrt"
            " round(x, places) log(x, base) ln sin cos tan min max pi e.",
            {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "minLength": 1, "maxLength": MAX_CHARS}
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            calculate,
            side_effect=False,
            spoken_summary=lambda args: f"work out {args.get('expression', 'that')}",
        )
    ]
