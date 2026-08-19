"""Temporary sign/definedness adapter over the unified domain model."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache

import sympy as sp

from ..domain import ConcentrationDomain, affine_form
from ..model import parse_rational


POSITIVE = "positive"
NONNEGATIVE = "nonnegative"
ZERO = "zero"
NONPOSITIVE = "nonpositive"
NEGATIVE = "negative"
UNKNOWN = "unknown"
_FACTOR_LIMIT = 80
_POLYNOMIAL_LIMIT = 30
_SIGNS = (
    ("is_zero", ZERO),
    ("is_positive", POSITIVE),
    ("is_negative", NEGATIVE),
    ("is_nonnegative", NONNEGATIVE),
    ("is_nonpositive", NONPOSITIVE),
)
_ACCEPTS = {
    POSITIVE: {POSITIVE},
    NONNEGATIVE: {POSITIVE, NONNEGATIVE, ZERO},
    "nonzero": {POSITIVE, NEGATIVE},
}
_REJECTS = {
    POSITIVE: {ZERO, NEGATIVE, NONPOSITIVE},
    NONNEGATIVE: {NEGATIVE},
    "nonzero": {ZERO},
}

Point = Mapping[sp.Symbol, sp.Expr]


def number(value: object) -> sp.Rational:
    return parse_rational(value)


@cache
def _exact(expression: sp.Expr) -> sp.Expr:
    expression = sp.sympify(expression)
    floats = {value: number(value) for value in expression.atoms(sp.Float)}
    return expression.xreplace(floats)


@cache
def _domain_requirements(expression: sp.Expr) -> tuple[tuple[sp.Expr, str], ...]:
    requirements = []
    for item in sp.preorder_traversal(expression):
        requirement = Proof._requirement(item)
        if requirement is not None:
            requirements.append(requirement)
    return tuple(dict.fromkeys(requirements))


def _assumptions(domain: ConcentrationDomain) -> dict[sp.Symbol, sp.Symbol]:
    replacements = {}
    for symbol, interval in domain.all_intervals.items():
        if interval.lower > 0:
            kind = "positive"
        elif interval.upper < 0:
            kind = "negative"
        elif interval.lower == 0:
            kind = "nonnegative" if interval.lower_closed else "positive"
        elif interval.upper == 0:
            kind = "nonpositive" if interval.upper_closed else "negative"
        else:
            kind = "real"
        replacements[symbol] = sp.Dummy(symbol.name, **{kind: True})
    return replacements


def proof_for_domain(domain: ConcentrationDomain) -> "Proof":
    point = domain.exact_witness()
    if point is None:
        raise ValueError(f"{domain.kind.value.title()} domain is empty.")
    return Proof(domain, _assumptions(domain), point)


@dataclass(frozen=True)
class Proof:
    """Legacy proof surface backed by exact specialized affine bounds."""

    domain: ConcentrationDomain
    assumptions: Mapping[sp.Symbol, sp.Symbol]
    point: Point
    _sign_cache: dict[sp.Expr, str] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )
    _defined_cache: dict[sp.Expr, tuple[bool | None, Point | None]] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    @staticmethod
    def _known_sign(expression: sp.Expr) -> str:
        for property_name, result in _SIGNS:
            if getattr(expression, property_name) is True:
                return result
        return UNKNOWN

    def _assumed_sign(self, expression: sp.Expr) -> str:
        result = self._known_sign(expression)
        if result != UNKNOWN:
            return result
        operations = sp.count_ops(expression)
        if operations <= _FACTOR_LIMIT:
            result = self._known_sign(sp.factor_terms(expression))
        if (
            result == UNKNOWN
            and operations <= _POLYNOMIAL_LIMIT
            and expression.is_polynomial(*expression.free_symbols)
        ):
            result = self._known_sign(sp.factor(expression))
        return result

    def sign(self, expression: sp.Expr) -> str:
        expression = sp.sympify(expression)
        if expression in self._sign_cache:
            return self._sign_cache[expression]
        result = self._assumed_sign(expression.xreplace(self.assumptions))
        bounds = None if result != UNKNOWN else self.domain.affine_bounds(_exact(expression))
        if bounds is not None:
            if bounds.lower == bounds.upper == 0:
                result = ZERO
            elif bounds.lower > 0:
                result = POSITIVE
            elif bounds.upper < 0:
                result = NEGATIVE
            elif bounds.lower >= 0:
                result = NONNEGATIVE
            elif bounds.upper <= 0:
                result = NONPOSITIVE
        self._sign_cache[expression] = result
        return result

    @staticmethod
    def _meets(result: str, requirement: str) -> bool | None:
        if result in _ACCEPTS[requirement]:
            return True
        if result in _REJECTS[requirement]:
            return False
        return None

    @staticmethod
    def _requirement(expression: sp.Expr) -> tuple[sp.Expr, str] | None:
        if isinstance(expression, sp.Pow):
            base, exponent = expression.args
            if exponent.is_integer is False:
                return base, POSITIVE if exponent.is_negative else NONNEGATIVE
            if exponent.is_negative:
                return base, "nonzero"
        if expression.func is sp.log:
            return expression.args[0], POSITIVE
        return None

    def violating_point(
        self,
        expression: sp.Expr,
        requirement: str,
    ) -> Point | None:
        bounds = self.domain.affine_bounds(_exact(expression))
        if bounds is None:
            return None
        if requirement == POSITIVE:
            return bounds.lower_witness if bounds.lower <= 0 else None
        if requirement == NONNEGATIVE:
            return bounds.lower_witness if bounds.lower < 0 else None

        for point in (bounds.lower_witness, bounds.upper_witness):
            if point is not None and _exact(expression).subs(point) == 0:
                return point
        if (
            bounds.lower < 0 < bounds.upper
            and bounds.lower_witness is not None
            and bounds.upper_witness is not None
        ):
            weight = -bounds.lower / (bounds.upper - bounds.lower)
            return {
                symbol: bounds.lower_witness[symbol]
                + weight
                * (bounds.upper_witness[symbol] - bounds.lower_witness[symbol])
                for symbol in self.domain.all_intervals
            }
        return None

    def defined(self, expression: sp.Expr) -> tuple[bool | None, Point | None]:
        expression = sp.sympify(expression)
        if expression in self._defined_cache:
            return self._defined_cache[expression]
        unresolved = False
        for argument, needed in _domain_requirements(expression):
            conclusion = self._meets(self.sign(argument), needed)
            if conclusion is False:
                result = False, self.violating_point(argument, needed) or self.point
                self._defined_cache[expression] = result
                return result
            if conclusion is None:
                witness = self.violating_point(argument, needed)
                if witness is not None:
                    result = False, witness
                    self._defined_cache[expression] = result
                    return result
                unresolved = True

        assumed = expression.xreplace(self.assumptions)
        if assumed.is_real is False or assumed.is_finite is False:
            result = False, self.point
        else:
            unresolved |= assumed.is_real is not True or assumed.is_finite is not True
            result = (None if unresolved else True), None
        self._defined_cache[expression] = result
        return result

    def value_sign(self, expression: sp.Expr) -> str:
        return self._assumed_sign(sp.sympify(expression).subs(self.point))

    def proves(self, expression: sp.Expr, requirement: str) -> bool | None:
        conclusion = self._meets(self.sign(expression), requirement)
        if conclusion is not None:
            return conclusion
        witness_sign = self.value_sign(expression)
        if requirement == POSITIVE:
            return False if witness_sign in (ZERO, NEGATIVE, NONPOSITIVE) else None
        return False if witness_sign == NEGATIVE else None


__all__ = (
    "NEGATIVE",
    "NONNEGATIVE",
    "NONPOSITIVE",
    "POSITIVE",
    "Point",
    "Proof",
    "UNKNOWN",
    "ZERO",
    "number",
    "proof_for_domain",
)
