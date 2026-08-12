"""Exact sign and definedness proofs on a bounded linear domain."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import sympy as sp
from sympy.core.relational import Relational
from sympy.polys.polyerrors import PolynomialError
from sympy.solvers.simplex import InfeasibleLPError, lpmax

POSITIVE = "positive"
NONNEGATIVE = "nonnegative"
ZERO = "zero"
NONPOSITIVE = "nonpositive"
NEGATIVE = "negative"
UNKNOWN = "unknown"
_FACTOR_LIMIT = 80
_POLYNOMIAL_LIMIT = 30
_SIGN_PROPERTIES = (
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
Feasibility = tuple[bool, Point | None]


def number(value: object) -> sp.Rational:
    return sp.Rational(str(value))


def _exact(expression: sp.Expr) -> sp.Expr:
    expression = sp.sympify(expression)
    floats = {value: number(value) for value in expression.atoms(sp.Float)}
    return expression.xreplace(floats)


@dataclass
class LinearDomain:
    """Bounded linear constraints, including strict inequalities."""

    variables: tuple[sp.Symbol, ...]
    weak: tuple[Relational, ...]
    strict: tuple[sp.Expr, ...]
    cache: dict[tuple[object, ...], Feasibility] = field(default_factory=dict)

    @property
    def constraints(self) -> tuple[Relational, ...]:
        strict = tuple(sp.Gt(item, 0, evaluate=False) for item in self.strict)
        return self.weak + strict

    def feasible(
        self,
        weak: Sequence[Relational] = (),
        strict: Sequence[sp.Expr] = (),
    ) -> Feasibility:
        key = (*weak, None, *strict)
        if key in self.cache:
            return self.cache[key]

        margin = sp.Dummy("strict_margin", nonnegative=True)
        constraints = [*self.weak, *weak, sp.Ge(margin, 0), sp.Le(margin, 1)]
        constraints += [
            sp.Ge(item, margin, evaluate=False)
            for item in (*self.strict, *strict)
        ]
        usable = []
        for constraint in constraints:
            if constraint.free_symbols:
                usable.append(constraint)
            elif constraint.doit() is sp.false:
                self.cache[key] = (False, None)
                return self.cache[key]

        try:
            optimum, solution = lpmax(margin, usable)
        except InfeasibleLPError:
            result: Feasibility = (False, None)
        else:
            point = MappingProxyType(
                {symbol: solution[symbol] for symbol in self.variables}
            )
            result = (True, point) if optimum > 0 else (False, None)
        self.cache[key] = result
        return result

    def equality(self, expression: sp.Expr) -> "LinearDomain":
        equation = sp.Eq(expression, 0, evaluate=False)
        return LinearDomain(self.variables, self.weak + (equation,), self.strict)


@dataclass(frozen=True)
class Proof:
    """Sign and real-domain prover tied to one region and exact point."""

    domain: LinearDomain
    assumptions: Mapping[sp.Symbol, sp.Symbol]
    point: Point

    def _affine(self, expression: sp.Expr) -> bool:
        if sp.count_ops(expression) > _FACTOR_LIMIT:
            return False
        try:
            return sp.Poly(expression, *self.domain.variables).total_degree() <= 1
        except PolynomialError:
            return False

    @staticmethod
    def _known_sign(expression: sp.Expr) -> str:
        for property_name, result in _SIGN_PROPERTIES:
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
            if result != UNKNOWN:
                return result
        if (
            operations <= _POLYNOMIAL_LIMIT
            and expression.is_polynomial(*expression.free_symbols)
        ):
            return self._known_sign(sp.factor(expression))
        return UNKNOWN

    def sign(self, expression: sp.Expr) -> str:
        expression = _exact(expression)
        result = self._assumed_sign(expression.xreplace(self.assumptions))
        if result in (POSITIVE, ZERO, NEGATIVE) or not self._affine(expression):
            return result
        if not self.domain.feasible(
            weak=(sp.Le(expression, 0, evaluate=False),)
        )[0]:
            return POSITIVE
        if not self.domain.feasible(
            weak=(sp.Ge(expression, 0, evaluate=False),)
        )[0]:
            return NEGATIVE
        if not self.domain.feasible(strict=(-expression,))[0]:
            return NONNEGATIVE
        if not self.domain.feasible(strict=(expression,))[0]:
            return NONPOSITIVE
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

    def _violating_point(self, expression: sp.Expr, requirement: str) -> Point | None:
        if not self._affine(expression):
            return None
        if requirement == POSITIVE:
            feasible, point = self.domain.feasible(
                weak=(sp.Le(expression, 0, evaluate=False),)
            )
        elif requirement == NONNEGATIVE:
            feasible, point = self.domain.feasible(strict=(-expression,))
        else:
            feasible, point = self.domain.equality(expression).feasible()
        return point if feasible else None

    def defined(self, expression: sp.Expr) -> tuple[bool | None, Point | None]:
        """Prove real, finite evaluation and return an exact bad point."""

        expression = _exact(expression)
        unresolved = False
        for item in sp.preorder_traversal(expression):
            requirement = self._requirement(item)
            if requirement is None:
                continue
            argument, needed = requirement
            conclusion = self._meets(self.sign(argument), needed)
            if conclusion is False:
                witness = self._violating_point(argument, needed)
                return False, witness if witness is not None else self.point
            if conclusion is None:
                witness = self._violating_point(argument, needed)
                if witness is not None:
                    return False, witness
                unresolved = True

        assumed = expression.xreplace(self.assumptions)
        if assumed.is_real is False or assumed.is_finite is False:
            return False, self.point
        unresolved |= assumed.is_real is not True or assumed.is_finite is not True
        return (None if unresolved else True), None

    def value_sign(self, expression: sp.Expr) -> str:
        return self._assumed_sign(_exact(expression).subs(self.point))

    def proves(self, expression: sp.Expr, requirement: str) -> bool | None:
        conclusion = self._meets(self.sign(expression), requirement)
        if conclusion is not None:
            return conclusion
        witness_sign = self.value_sign(expression)
        if requirement == POSITIVE:
            return False if witness_sign in (ZERO, NEGATIVE, NONPOSITIVE) else None
        return False if witness_sign == NEGATIVE else None
