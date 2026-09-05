"""Exact affine and compositional interval analysis for SymPy expressions."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import sympy as sp

from ..domain import ConcentrationDomain
from ..model import exact_expr

if TYPE_CHECKING:
    from .lipschitz import LipschitzResult


Point = Mapping[sp.Symbol, sp.Expr]
_FACTOR_TERMS_LIMIT = 64
_WITNESS_LIMIT = 24
_SUPPORTED_FUNCTIONS = frozenset(
    (sp.Abs, sp.Min, sp.Max, sp.exp, sp.log, sp.sin, sp.cos, sp.sinh, sp.cosh, sp.tanh, sp.atan)
)
_DISCONTINUOUS_FUNCTIONS = frozenset((sp.Piecewise, sp.floor, sp.ceiling, sp.sign))
_TRIGONOMETRIC_GUARDS = {sp.tan: sp.cos, sp.sec: sp.cos, sp.cot: sp.sin, sp.csc: sp.sin}


class ProofVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class Sign(StrEnum):
    POSITIVE = "positive"
    NONNEGATIVE = "nonnegative"
    ZERO = "zero"
    NONPOSITIVE = "nonpositive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class SignRequirement(StrEnum):
    POSITIVE = "positive"
    NONNEGATIVE = "nonnegative"
    NONZERO = "nonzero"


@dataclass(frozen=True)
class BoundResult:
    lower: sp.Expr | None
    upper: sp.Expr | None
    exact: bool = False
    lower_witness: Point | None = None
    upper_witness: Point | None = None
    decisive_subexpression: sp.Expr | None = None
    reason: str | None = None

    @property
    def known(self) -> bool:
        return self.lower is not None and self.upper is not None

    @property
    def absolute_upper(self) -> sp.Expr | None:
        if not self.known:
            return None
        return sp.Max(abs(self.lower), abs(self.upper))


@dataclass(frozen=True)
class SignResult:
    sign: Sign
    interval: BoundResult
    decisive_subexpression: sp.Expr | None = None
    witness: Point | None = None
    witness_value: sp.Expr | None = None


@dataclass(frozen=True)
class SignProof:
    verdict: ProofVerdict
    requirement: SignRequirement
    result: SignResult
    witness: Point | None = None
    witness_value: sp.Expr | None = None
    reason: str | None = None


@dataclass(frozen=True)
class DefinednessResult:
    verdict: ProofVerdict
    decisive_subexpression: sp.Expr | None = None
    requirement: SignRequirement | None = None
    witness: Point | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ZeroProof:
    verdict: ProofVerdict
    expression: sp.Expr
    witness: Point | None = None
    witness_value: sp.Expr | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ContributionBound:
    coefficient: sp.Expr
    lower: sp.Expr | None
    upper: sp.Expr | None
    source_lower: sp.Expr | None
    source_upper: sp.Expr | None


@dataclass(frozen=True)
class SumProof:
    verdict: ProofVerdict
    lower: sp.Expr | None
    upper: sp.Expr | None
    contributions: tuple[ContributionBound, ...]
    witness: Point | None = None
    witness_value: sp.Expr | None = None
    reason: str | None = None


def _minimum(values) -> sp.Expr:
    return sp.Min(*tuple(values))


def _maximum(values) -> sp.Expr:
    return sp.Max(*tuple(values))


def _true(relation) -> bool:
    return relation is True or relation is sp.true


def _sign_from_bounds(bounds) -> Sign:
    if not bounds.known:
        return Sign.UNKNOWN
    lower, upper = bounds.lower, bounds.upper
    if lower == upper == 0:
        return Sign.ZERO
    if _true(lower > 0):
        return Sign.POSITIVE
    if _true(upper < 0):
        return Sign.NEGATIVE
    if _true(lower >= 0):
        return Sign.NONNEGATIVE
    if _true(upper <= 0):
        return Sign.NONPOSITIVE
    return Sign.UNKNOWN


def _known_sign(expression) -> Sign:
    for attribute, sign in (
        ("is_zero", Sign.ZERO),
        ("is_positive", Sign.POSITIVE),
        ("is_negative", Sign.NEGATIVE),
        ("is_nonnegative", Sign.NONNEGATIVE),
        ("is_nonpositive", Sign.NONPOSITIVE),
    ):
        if getattr(expression, attribute) is True:
            return sign
    return Sign.UNKNOWN


def _meets(sign, requirement) -> bool | None:
    accepted = {
        SignRequirement.POSITIVE: {Sign.POSITIVE},
        SignRequirement.NONNEGATIVE: {Sign.POSITIVE, Sign.NONNEGATIVE, Sign.ZERO},
        SignRequirement.NONZERO: {Sign.POSITIVE, Sign.NEGATIVE},
    }
    rejected = {
        SignRequirement.POSITIVE: {Sign.ZERO, Sign.NEGATIVE, Sign.NONPOSITIVE},
        SignRequirement.NONNEGATIVE: {Sign.NEGATIVE},
        SignRequirement.NONZERO: {Sign.ZERO},
    }
    if sign in accepted[requirement]:
        return True
    if sign in rejected[requirement]:
        return False
    return None


class ExpressionAnalyzer:
    """One bounded, memoized analysis service for a complete run."""

    def __init__(self) -> None:
        self._bounds: dict[tuple, BoundResult] = {}
        self._signs: dict[tuple, SignResult] = {}
        self._proofs: dict[tuple, SignProof] = {}
        self._definedness: dict[tuple, DefinednessResult] = {}
        self._lipschitz_results: dict[tuple, LipschitzResult] = {}

    def _key(
        self, expression, domain, active_variables
    ) -> tuple[sp.Expr, ConcentrationDomain, tuple[sp.Symbol, ...]]:
        expression = exact_expr(expression)
        active = tuple(
            sorted(
                active_variables if active_variables is not None else domain.intervals,
                key=lambda symbol: symbol.name,
            )
        )
        return expression, domain, active

    def bounds(
        self,
        expression: object,
        domain: ConcentrationDomain,
        active_variables: Iterable[sp.Symbol] | None = None,
    ) -> BoundResult:
        key = self._key(expression, domain, active_variables)
        if key not in self._bounds:
            self._bounds[key] = self._compute_bounds(key[0], domain, key[2])
        return self._bounds[key]

    def _unknown_bound(self, expression, reason) -> BoundResult:
        return BoundResult(None, None, decisive_subexpression=expression, reason=reason)

    def _compute_bounds(self, expression, domain, active) -> BoundResult:
        unknown = expression.free_symbols - set(domain.all_intervals)
        if unknown:
            symbol = min(unknown, key=lambda item: item.name)
            return self._unknown_bound(symbol, f"No domain interval for {symbol}.")

        if not expression.free_symbols:
            if expression.is_real is True and expression.is_finite is True:
                return BoundResult(expression, expression, True)
            return self._unknown_bound(expression, "Constant is not finite and real.")
        affine = domain.affine_bounds(expression)
        if affine is not None:
            return BoundResult(
                affine.lower, affine.upper, True, affine.lower_witness, affine.upper_witness
            )

        child_bounds = tuple(self.bounds(item, domain, active) for item in expression.args)
        missing = next((item for item in child_bounds if not item.known), None)
        if missing is not None:
            return missing

        if isinstance(expression, sp.Add):
            return BoundResult(
                sum(item.lower for item in child_bounds), sum(item.upper for item in child_bounds)
            )
        if isinstance(expression, sp.Mul):
            lower = upper = sp.S.One
            for item in child_bounds:
                products = (
                    lower * item.lower,
                    lower * item.upper,
                    upper * item.lower,
                    upper * item.upper,
                )
                lower, upper = _minimum(products), _maximum(products)
            return BoundResult(lower, upper)
        if isinstance(expression, sp.Pow):
            return self._power_bounds(expression, child_bounds[0])
        if expression.func is sp.Abs:
            item = child_bounds[0]
            if _true(item.lower >= 0):
                return BoundResult(item.lower, item.upper)
            if _true(item.upper <= 0):
                return BoundResult(-item.upper, -item.lower)
            return BoundResult(sp.S.Zero, sp.Max(-item.lower, item.upper))
        if expression.func in {sp.Min, sp.Max}:
            endpoints = (
                (item.lower for item in child_bounds),
                (item.upper for item in child_bounds),
            )
            combine = _minimum if expression.func is sp.Min else _maximum
            return BoundResult(combine(endpoints[0]), combine(endpoints[1]))

        if expression.func is sp.log:
            item = child_bounds[0]
            if _true(item.lower > 0):
                return BoundResult(sp.log(item.lower), sp.log(item.upper))
            return self._unknown_bound(expression.args[0], "Logarithm needs a positive argument.")
        if expression.func in {sp.sin, sp.cos}:
            return BoundResult(-sp.S.One, sp.S.One)
        if expression.func in {sp.exp, sp.sinh, sp.tanh, sp.atan}:
            item = child_bounds[0]
            return BoundResult(expression.func(item.lower), expression.func(item.upper))
        if expression.func is sp.cosh:
            item = child_bounds[0]
            endpoints = (sp.cosh(item.lower), sp.cosh(item.upper))
            return BoundResult(sp.S.One, _maximum(endpoints))
        return self._unknown_bound(expression, f"Unsupported function {expression.func.__name__}.")

    def _power_bounds(self, expression, base) -> BoundResult:
        exponent = expression.exp
        if exponent.is_number is not True or exponent.is_real is not True:
            return self._unknown_bound(expression, "Power exponent is not a real constant.")
        lower, upper = base.lower, base.upper
        if exponent.is_integer:
            integer = int(exponent)
            if integer == 0:
                return BoundResult(sp.S.One, sp.S.One, True)
            if integer < 0:
                positive = self._integer_power_bounds(lower, upper, -integer)
                if not (_true(positive.lower > 0) or _true(positive.upper < 0)):
                    return self._unknown_bound(expression.base, "Negative power base can be zero.")
                reciprocals = (1 / positive.lower, 1 / positive.upper)
                return BoundResult(_minimum(reciprocals), _maximum(reciprocals))
            return self._integer_power_bounds(lower, upper, integer)
        if not _true(lower >= 0):
            return self._unknown_bound(expression.base, "Noninteger power base can be negative.")
        if exponent.is_negative and not _true(lower > 0):
            return self._unknown_bound(expression.base, "Negative power base can be zero.")
        values = (lower**exponent, upper**exponent)
        return BoundResult(_minimum(values), _maximum(values))

    @staticmethod
    def _integer_power_bounds(lower, upper, exponent) -> BoundResult:
        endpoints = (lower**exponent, upper**exponent)
        if exponent % 2 == 0:
            if _true(lower > 0) or _true(upper < 0):
                return BoundResult(_minimum(endpoints), _maximum(endpoints))
            return BoundResult(sp.S.Zero, _maximum(endpoints))
        return BoundResult(_minimum(endpoints), _maximum(endpoints))

    def sign(
        self,
        expression: object,
        domain: ConcentrationDomain,
        active_variables: Iterable[sp.Symbol] | None = None,
    ) -> SignResult:
        key = self._key(expression, domain, active_variables)
        if key not in self._signs:
            self._signs[key] = self._compute_sign(key[0], domain, key[2])
        return self._signs[key]

    def _compute_sign(self, expression, domain, active) -> SignResult:
        interval = self.bounds(expression, domain, active)
        interval_sign = _sign_from_bounds(interval)
        if interval_sign in {Sign.POSITIVE, Sign.NEGATIVE, Sign.ZERO}:
            return SignResult(interval_sign, interval)

        structural = self._structural_sign(expression, domain, active)
        if structural is not Sign.UNKNOWN:
            return SignResult(structural, interval)

        assumed = expression.xreplace(self._assumptions(domain))
        direct = _known_sign(assumed)
        if direct is not Sign.UNKNOWN:
            return SignResult(direct, interval)

        if sp.count_ops(expression) <= _FACTOR_TERMS_LIMIT:
            factored = sp.factor_terms(expression)
            if factored != expression:
                factored_sign = self.sign(factored, domain, active).sign
                if factored_sign is not Sign.UNKNOWN:
                    return SignResult(factored_sign, interval)

        if interval_sign is not Sign.UNKNOWN:
            return SignResult(interval_sign, interval)

        decisive = interval.decisive_subexpression
        if decisive is None and isinstance(expression, (sp.Add, sp.Mul)):
            for argument in expression.args:
                child = self.sign(argument, domain, active)
                if child.sign is Sign.UNKNOWN:
                    decisive = child.decisive_subexpression or argument
                    break
        point, value = self._point_value(expression, domain)
        return SignResult(Sign.UNKNOWN, interval, decisive or expression, point, value)

    def _structural_sign(self, expression, domain, active) -> Sign:
        if isinstance(expression, sp.Add):
            signs = tuple(self.sign(item, domain, active).sign for item in expression.args)
            if all(item in {Sign.POSITIVE, Sign.NONNEGATIVE, Sign.ZERO} for item in signs):
                return Sign.POSITIVE if Sign.POSITIVE in signs else Sign.NONNEGATIVE
            if all(item in {Sign.NEGATIVE, Sign.NONPOSITIVE, Sign.ZERO} for item in signs):
                return Sign.NEGATIVE if Sign.NEGATIVE in signs else Sign.NONPOSITIVE
        if isinstance(expression, sp.Mul):
            signs = tuple(self.sign(item, domain, active).sign for item in expression.args)
            if Sign.ZERO in signs:
                return Sign.ZERO
            if Sign.UNKNOWN not in signs:
                lower = upper = 1
                ranges = {
                    Sign.POSITIVE: (1, 1),
                    Sign.NONNEGATIVE: (0, 1),
                    Sign.NONPOSITIVE: (-1, 0),
                    Sign.NEGATIVE: (-1, -1),
                }
                for item in signs:
                    item_lower, item_upper = ranges[item]
                    products = (
                        lower * item_lower,
                        lower * item_upper,
                        upper * item_lower,
                        upper * item_upper,
                    )
                    lower, upper = min(products), max(products)
                return {
                    (1, 1): Sign.POSITIVE,
                    (0, 1): Sign.NONNEGATIVE,
                    (0, 0): Sign.ZERO,
                    (-1, 0): Sign.NONPOSITIVE,
                    (-1, -1): Sign.NEGATIVE,
                }.get((lower, upper), Sign.UNKNOWN)
        return Sign.UNKNOWN

    @staticmethod
    def _assumptions(domain) -> dict[sp.Symbol, sp.Symbol]:
        replacements = {}
        for symbol, interval in domain.all_intervals.items():
            if interval.lower > 0 or (interval.lower == 0 and not interval.lower_closed):
                assumption = "positive"
            elif interval.upper < 0 or (interval.upper == 0 and not interval.upper_closed):
                assumption = "negative"
            elif interval.lower == 0:
                assumption = "nonnegative"
            elif interval.upper == 0:
                assumption = "nonpositive"
            else:
                assumption = "real"
            replacements[symbol] = sp.Dummy(symbol.name, **{assumption: True})
        return replacements

    def prove_sign(
        self,
        expression: object,
        domain: ConcentrationDomain,
        requirement: SignRequirement,
        active_variables: Iterable[sp.Symbol] | None = None,
    ) -> SignProof:
        requirement = SignRequirement(requirement)
        key = (*self._key(expression, domain, active_variables), requirement)
        if key in self._proofs:
            return self._proofs[key]
        result = self.sign(key[0], domain, key[2])
        conclusion = _meets(result.sign, requirement)
        if conclusion is True:
            proof = SignProof(ProofVerdict.PASS, requirement, result)
        else:
            witness, value = self._counterexample(key[0], domain, requirement)
            if witness is not None:
                proof = SignProof(
                    ProofVerdict.FAIL,
                    requirement,
                    result,
                    witness,
                    value,
                    f"Exact feasible point violates {requirement.value}.",
                )
            else:
                proof = SignProof(
                    ProofVerdict.UNKNOWN,
                    requirement,
                    result,
                    reason="The available enclosure is inconclusive.",
                )
        self._proofs[key] = proof
        return proof

    def _counterexample(
        self, expression, domain, requirement
    ) -> tuple[Point | None, sp.Expr | None]:
        affine = domain.affine_bounds(expression)
        candidates: list[Point] = []
        if affine is not None:
            if affine.lower_witness is not None:
                candidates.append(affine.lower_witness)
            if affine.upper_witness is not None:
                candidates.append(affine.upper_witness)
            if (
                requirement is SignRequirement.NONZERO
                and affine.lower < 0 < affine.upper
                and affine.lower_witness is not None
                and affine.upper_witness is not None
            ):
                weight = -affine.lower / (affine.upper - affine.lower)
                candidates.append(
                    {
                        symbol: affine.lower_witness[symbol]
                        + weight * (affine.upper_witness[symbol] - affine.lower_witness[symbol])
                        for symbol in domain.all_intervals
                    }
                )
        candidates.extend(self._candidate_points(expression, domain))

        for candidate in candidates[:_WITNESS_LIMIT]:
            value = exact_expr(expression.subs(candidate))
            if self._violates(value, requirement):
                return candidate, value
        return None, None

    def prove_zero(self, expression: object, domain: ConcentrationDomain) -> ZeroProof:
        """Prove an identity is zero, or find an exact nonzero value."""
        expression = exact_expr(expression)
        if expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
            return ZeroProof(ProofVerdict.FAIL, expression, reason="Expression is undefined.")
        numerator, denominator = expression.as_numer_denom()
        zero = expression.is_zero
        if zero is None and denominator.is_zero is not True:
            zero = numerator.is_zero
        if zero is None and sp.count_ops(expression) <= _FACTOR_TERMS_LIMIT:
            zero = sp.factor_terms(expression).is_zero
        if zero is True:
            return ZeroProof(ProofVerdict.PASS, expression)
        for point in self._candidate_points(expression, domain):
            value = exact_expr(expression.subs(point, simultaneous=True))
            if value.is_real is True and value.is_finite is True and value.is_zero is False:
                return ZeroProof(ProofVerdict.FAIL, expression, point, value)
        if zero is False:
            return ZeroProof(ProofVerdict.FAIL, expression)
        return ZeroProof(ProofVerdict.UNKNOWN, expression, reason="Zero identity is inconclusive.")

    @staticmethod
    def _candidate_points(expression, domain) -> tuple[Point, ...]:
        points = [domain.exact_witness()]
        for symbol in sorted(expression.free_symbols & set(domain.all_intervals), key=str):
            interval = domain.interval(symbol)
            for value in (interval.upper, (interval.lower + interval.upper) / 2):
                points.append(domain.exact_witness({symbol: value}))
        return tuple(point for point in points[:_WITNESS_LIMIT] if point is not None)

    def prove_sum(
        self,
        terms: Sequence[tuple[sp.Expr, sp.Expr]],
        domain: ConcentrationDomain,
        requirement: SignRequirement,
    ) -> SumProof:
        """Prove the sign of a sparse weighted sum without eagerly expanding it."""
        requirement = SignRequirement(requirement)
        contributions, known = [], True
        for coefficient, expression in terms:
            bound = self.bounds(expression, domain)
            if bound.known:
                values = (coefficient * bound.lower, coefficient * bound.upper)
                lower, upper = _minimum(values), _maximum(values)
            else:
                lower = upper = None
                known = False
            contributions.append(
                ContributionBound(coefficient, bound.lower, bound.upper, lower, upper)
            )
        lower = sum((item.source_lower for item in contributions), sp.S.Zero) if known else None
        upper = sum((item.source_upper for item in contributions), sp.S.Zero) if known else None
        conclusion = _meets(_sign_from_bounds(BoundResult(lower, upper)), requirement)
        if conclusion is True:
            return SumProof(ProofVerdict.PASS, lower, upper, tuple(contributions))
        expression = sum((coefficient * value for coefficient, value in terms), sp.S.Zero)
        if sp.count_ops(expression) <= 96:
            proof = self.prove_sign(expression, domain, requirement)
            if proof.verdict is not ProofVerdict.UNKNOWN:
                return SumProof(
                    proof.verdict,
                    lower,
                    upper,
                    tuple(contributions),
                    proof.witness,
                    proof.witness_value,
                    proof.reason,
                )
        if conclusion is False:
            return SumProof(
                ProofVerdict.FAIL,
                lower,
                upper,
                tuple(contributions),
                reason="The source upper bound violates the sign requirement.",
            )
        return SumProof(
            ProofVerdict.UNKNOWN,
            lower,
            upper,
            tuple(contributions),
            reason="Sparse interval and bounded symbolic analysis are inconclusive.",
        )

    @staticmethod
    def _violates(value, requirement) -> bool:
        if value.is_real is not True or value.is_finite is not True:
            return True
        if requirement is SignRequirement.POSITIVE:
            return value.is_nonpositive is True
        if requirement is SignRequirement.NONNEGATIVE:
            return value.is_negative is True
        return value.is_zero is True

    @staticmethod
    def _point_value(expression, domain) -> tuple[Point | None, sp.Expr | None]:
        point = domain.exact_witness()
        return (None, None) if point is None else (point, exact_expr(expression.subs(point)))

    def defined(
        self,
        expression: object,
        domain: ConcentrationDomain,
        active_variables: Iterable[sp.Symbol] | None = None,
    ) -> DefinednessResult:
        key = self._key(expression, domain, active_variables)
        if key not in self._definedness:
            self._definedness[key] = self._compute_defined(key[0], domain, key[2])
        return self._definedness[key]

    def _compute_defined(self, expression, domain, active) -> DefinednessResult:
        if expression in {sp.nan, sp.zoo, sp.oo, -sp.oo}:
            return DefinednessResult(ProofVerdict.FAIL, expression, reason="Non-finite constant.")
        if expression.is_Atom:
            if expression.is_real is False or expression.is_finite is False:
                return DefinednessResult(
                    ProofVerdict.FAIL, expression, reason="Atom is not finite and real."
                )
            if expression.is_number:
                verdict = (
                    ProofVerdict.PASS
                    if expression.is_real is True and expression.is_finite is True
                    else ProofVerdict.FAIL
                )
                decisive = None if verdict is ProofVerdict.PASS else expression
                return DefinednessResult(verdict, decisive)
            if expression in domain.all_intervals:
                return DefinednessResult(ProofVerdict.PASS)
            return DefinednessResult(
                ProofVerdict.UNKNOWN, expression, reason=f"No domain interval for {expression}."
            )

        for argument in expression.args:
            result = self.defined(argument, domain, active)
            if result.verdict is not ProofVerdict.PASS:
                return result

        requirement = None
        guarded = None
        if isinstance(expression, sp.Pow):
            exponent = expression.exp
            if exponent.is_number is not True or exponent.is_real is not True:
                return DefinednessResult(
                    ProofVerdict.UNKNOWN, exponent, reason="Power exponent is not a real constant."
                )
            if exponent.is_integer and exponent.is_negative:
                guarded, requirement = expression.base, SignRequirement.NONZERO
            elif exponent.is_integer is not True:
                guarded = expression.base
                requirement = (
                    SignRequirement.POSITIVE
                    if exponent.is_negative
                    else SignRequirement.NONNEGATIVE
                )
        elif expression.func is sp.log:
            guarded, requirement = expression.args[0], SignRequirement.POSITIVE
        elif expression.func in _TRIGONOMETRIC_GUARDS:
            guarded = _TRIGONOMETRIC_GUARDS[expression.func](expression.args[0])
            requirement = SignRequirement.NONZERO

        if guarded is not None:
            proof = self.prove_sign(guarded, domain, requirement, active)
            if proof.verdict is not ProofVerdict.PASS:
                decisive = guarded
                if proof.verdict is ProofVerdict.UNKNOWN:
                    decisive = proof.result.decisive_subexpression or guarded
                return DefinednessResult(
                    proof.verdict, decisive, requirement, proof.witness, proof.reason
                )

        if (
            isinstance(expression, (sp.Add, sp.Mul, sp.Pow))
            or expression.func in _SUPPORTED_FUNCTIONS
            or expression.func in _TRIGONOMETRIC_GUARDS
        ):
            return DefinednessResult(ProofVerdict.PASS)
        if expression.func in _DISCONTINUOUS_FUNCTIONS:
            reason = f"Discontinuous function {expression.func.__name__} is unsupported."
        else:
            reason = f"Function {expression.func.__name__} is unsupported."
        return DefinednessResult(ProofVerdict.UNKNOWN, expression, reason=reason)

    def lipschitz(
        self,
        expression: object,
        domain: ConcentrationDomain,
        active_variables: Iterable[sp.Symbol] | None = None,
    ) -> "LipschitzResult":
        """Certify a uniform concentration-space Lipschitz bound."""

        key = self._key(expression, domain, active_variables)
        if not set(key[2]) <= set(domain.intervals):
            raise ValueError("Lipschitz variables must be concentration coordinates.")
        return self.gradient_envelope(key[0], domain, key[2])

    def gradient_envelope(
        self,
        expression: object,
        domain: ConcentrationDomain,
        active_variables: Iterable[sp.Symbol] | None = None,
    ) -> "LipschitzResult":
        """Bound each requested partial derivative in one compositional walk."""

        active = domain.all_intervals if active_variables is None else active_variables
        key = self._key(expression, domain, active)
        if not set(key[2]) <= set(domain.all_intervals):
            raise ValueError("Gradient variables must be coordinates of the domain.")
        if key not in self._lipschitz_results:
            from .lipschitz import compute_lipschitz

            self._lipschitz_results[key] = compute_lipschitz(self, key[0], domain, key[2])
        return self._lipschitz_results[key]
