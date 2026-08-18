"""Exact sign and definedness proofs on a bounded linear domain."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType

import sympy as sp
from sympy.core.relational import Relational
from sympy.polys.polyerrors import PolynomialError
from sympy.solvers.simplex import InfeasibleLPError, linprog, lpmax

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


@cache
def _exact(expression: sp.Expr) -> sp.Expr:
    expression = sp.sympify(expression)
    floats = {value: number(value) for value in expression.atoms(sp.Float)}
    return expression.xreplace(floats)


@cache
def _domain_requirements(
    expression: sp.Expr,
) -> tuple[tuple[sp.Expr, str], ...]:
    """Extract real-domain requirements once for a shared expression DAG."""

    requirements = []
    for item in sp.preorder_traversal(expression):
        requirement = Proof._requirement(item)
        if requirement is not None:
            requirements.append(requirement)
    return tuple(dict.fromkeys(requirements))


def _linear_expression(constraint: Relational) -> sp.Expr:
    """Return a linear expression which is constrained to be non-negative."""

    if isinstance(constraint, sp.Equality):
        return constraint.lhs - constraint.rhs
    if isinstance(constraint, (sp.GreaterThan, sp.StrictGreaterThan)):
        return constraint.lhs - constraint.rhs
    if isinstance(constraint, (sp.LessThan, sp.StrictLessThan)):
        return constraint.rhs - constraint.lhs
    raise TypeError(f"Unsupported linear constraint: {constraint}.")


@dataclass
class LinearDomain:
    """Bounded linear constraints, including strict inequalities."""

    variables: tuple[sp.Symbol, ...]
    weak: tuple[Relational, ...]
    strict: tuple[sp.Expr, ...]
    bounds: tuple[tuple[sp.Expr, sp.Expr], ...] | None = None
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

        margin = sp.Dummy("strict_margin")
        weak_expressions: list[sp.Expr] = []
        equality_expressions: list[sp.Expr] = []
        for constraint in (*self.weak, *weak):
            if constraint.free_symbols:
                expression = _linear_expression(constraint)
                if isinstance(constraint, sp.Equality):
                    equality_expressions.append(expression)
                else:
                    weak_expressions.append(expression)
            elif constraint.doit() is sp.false:
                self.cache[key] = (False, None)
                return self.cache[key]

        strict_expressions = [
            sp.sympify(item) - margin for item in (*self.strict, *strict)
        ]
        variables = (*self.variables, margin)

        # ``lpmax`` first converts relationals through expensive univariate-set
        # machinery.  These constraints are already known to be affine, so
        # construct the simplex matrices directly instead.
        inequalities = weak_expressions + strict_expressions + [1 - margin]
        if inequalities:
            matrix, vector = sp.linear_eq_to_matrix(inequalities, variables)
            matrix, vector = -matrix, -vector
        else:
            matrix = vector = None
        if equality_expressions:
            equality_matrix, equality_vector = sp.linear_eq_to_matrix(
                equality_expressions,
                variables,
            )
        else:
            equality_matrix = equality_vector = None

        variable_count = len(self.variables)
        if self.bounds is not None:
            # A free-variable split is a very fast feasible-point probe. The
            # SymPy simplex may miss feasible optima with dependent +/-
            # columns, so negative probe results are ignored; a validated
            # positive point is nevertheless an exact certificate.
            def split_probe(
                candidate: sp.MatrixBase | None,
            ) -> sp.MatrixBase | None:
                if candidate is None:
                    return None
                state = candidate[:, :variable_count]
                return state.row_join(-state).row_join(
                    candidate[:, variable_count:]
                )

            try:
                probe_optimum, probe_solution = linprog(
                    [*[sp.S.Zero] * (2 * variable_count), -sp.S.One],
                    split_probe(matrix),
                    vector,
                    split_probe(equality_matrix),
                    equality_vector,
                )
            except InfeasibleLPError:
                pass
            else:
                probe_point = MappingProxyType(
                    {
                        symbol: probe_solution[index]
                        - probe_solution[variable_count + index]
                        for index, symbol in enumerate(self.variables)
                    }
                )
                if (
                    -probe_optimum > 0
                    and all(
                        bool(constraint.subs(probe_point))
                        for constraint in (*self.weak, *weak)
                    )
                    and all(
                        sp.sympify(item).subs(probe_point) > 0
                        for item in (*self.strict, *strict)
                    )
                ):
                    result = True, probe_point
                    self.cache[key] = result
                    return result

        if self.bounds is None:
            # Preserve the general unbounded-domain API by splitting free
            # variables. Recovery domains supply their finite box directly,
            # avoiding these dependent columns in the common path.
            def split_free(matrix: sp.MatrixBase | None) -> sp.MatrixBase | None:
                if matrix is None:
                    return None
                state = matrix[:, :variable_count]
                return state.row_join(-state).row_join(
                    matrix[:, variable_count:]
                )

            matrix = split_free(matrix)
            equality_matrix = split_free(equality_matrix)
            objective = [*[sp.S.Zero] * (2 * variable_count), -sp.S.One]
            simplex_bounds = None
        else:
            lower = sp.Matrix([item[0] for item in self.bounds])
            upper_width = sp.Matrix(
                [item[1] - item[0] for item in self.bounds]
            )
            if matrix is not None:
                state_matrix = matrix[:, :variable_count]
                vector = vector - state_matrix * lower
                upper_matrix = sp.eye(variable_count).row_join(
                    sp.zeros(variable_count, 1)
                )
                matrix = matrix.col_join(upper_matrix)
                vector = vector.col_join(upper_width)
            if equality_matrix is not None:
                equality_vector = (
                    equality_vector
                    - equality_matrix[:, :variable_count] * lower
                )
            objective = [*[sp.S.Zero] * variable_count, -sp.S.One]
            simplex_bounds = None

        def fallback() -> Feasibility:
            fallback_margin = sp.Dummy(
                "strict_margin_fallback",
                nonnegative=True,
            )
            constraints = [
                *self.weak,
                *weak,
                sp.Ge(fallback_margin, 0),
                sp.Le(fallback_margin, 1),
                *(
                    sp.Ge(item, fallback_margin, evaluate=False)
                    for item in (*self.strict, *strict)
                ),
            ]
            usable = [
                constraint for constraint in constraints if constraint.free_symbols
            ]
            try:
                fallback_optimum, fallback_solution = lpmax(
                    fallback_margin,
                    usable,
                )
            except InfeasibleLPError:
                return False, None
            fallback_point = MappingProxyType(
                {
                    symbol: fallback_solution[symbol]
                    for symbol in self.variables
                }
            )
            return (
                (True, fallback_point)
                if fallback_optimum > 0
                else (False, None)
            )

        try:
            optimum, solution = linprog(
                objective,
                matrix,
                vector,
                equality_matrix,
                equality_vector,
                bounds=simplex_bounds,
            )
        except InfeasibleLPError:
            result = (False, None) if self.bounds is not None else fallback()
        else:
            if self.bounds is None:
                point = MappingProxyType(
                    {
                        symbol: solution[index]
                        - solution[variable_count + index]
                        for index, symbol in enumerate(self.variables)
                    }
                )
            else:
                point = MappingProxyType(
                    {
                        symbol: solution[index] + self.bounds[index][0]
                        for index, symbol in enumerate(self.variables)
                    }
                )
            candidate_is_valid = (
                -optimum > 0
                and all(bool(constraint.subs(point)) for constraint in (*self.weak, *weak))
                and all(
                    sp.sympify(item).subs(point) > 0
                    for item in (*self.strict, *strict)
                )
            )
            if candidate_is_valid:
                result = True, point
            elif self.bounds is not None and -optimum <= 0:
                result = False, None
            else:
                # SymPy 1.14's matrix simplex can occasionally return an
                # invalid point or a false zero optimum when free variables
                # were split into dependent positive/negative columns.
                # Preserve exact semantics with the slower relational frontend
                # only for that exceptional case.
                result = fallback()
        self.cache[key] = result
        return result

    def equality(self, expression: sp.Expr) -> "LinearDomain":
        equation = sp.Eq(expression, 0, evaluate=False)
        return LinearDomain(
            self.variables,
            self.weak + (equation,),
            self.strict,
            self.bounds,
        )


@dataclass(frozen=True)
class Proof:
    """Sign and real-domain prover tied to one region and exact point."""

    domain: LinearDomain
    assumptions: Mapping[sp.Symbol, sp.Symbol]
    point: Point
    _sign_cache: dict[sp.Expr, str] = field(
        default_factory=dict,
        init=False,
        compare=False,
        repr=False,
    )
    _defined_cache: dict[sp.Expr, tuple[bool | None, Point | None]] = field(
        default_factory=dict,
        init=False,
        compare=False,
        repr=False,
    )

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
        expression = sp.sympify(expression)
        if expression in self._sign_cache:
            return self._sign_cache[expression]
        result = self._assumed_sign(expression.xreplace(self.assumptions))
        if result in (POSITIVE, ZERO, NEGATIVE) or not self._affine(expression):
            self._sign_cache[expression] = result
            return result
        expression = _exact(expression)
        if not self.domain.feasible(
            weak=(sp.Le(expression, 0, evaluate=False),)
        )[0]:
            result = POSITIVE
        elif not self.domain.feasible(
            weak=(sp.Ge(expression, 0, evaluate=False),)
        )[0]:
            result = NEGATIVE
        elif not self.domain.feasible(strict=(-expression,))[0]:
            result = NONNEGATIVE
        elif not self.domain.feasible(strict=(expression,))[0]:
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

    def _violating_point(self, expression: sp.Expr, requirement: str) -> Point | None:
        if not self._affine(expression):
            return None
        expression = _exact(expression)
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

        expression = sp.sympify(expression)
        if expression in self._defined_cache:
            return self._defined_cache[expression]
        unresolved = False
        for argument, needed in _domain_requirements(expression):
            conclusion = self._meets(self.sign(argument), needed)
            if conclusion is False:
                witness = self._violating_point(argument, needed)
                result = False, witness if witness is not None else self.point
                self._defined_cache[expression] = result
                return result
            if conclusion is None:
                witness = self._violating_point(argument, needed)
                if witness is not None:
                    result = False, witness
                    self._defined_cache[expression] = result
                    return result
                unresolved = True

        assumed = expression.xreplace(self.assumptions)
        if assumed.is_real is False or assumed.is_finite is False:
            result = False, self.point
            self._defined_cache[expression] = result
            return result
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
