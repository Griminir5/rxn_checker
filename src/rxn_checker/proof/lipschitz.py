"""Checkable Lipschitz estimates assembled from elementary inequalities.

The distance is measured with the concentration-space infinity norm.  Therefore
the scalar derivative fallback uses the dual one-norm: the sum of the absolute
partial-derivative bounds.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp

from ..domain import ConcentrationDomain, DomainKind
from .analysis import ExpressionAnalyzer, Point, ProofVerdict, SignRequirement


_DERIVATIVE_SIZE_LIMIT = 80
_DISCONTINUOUS = frozenset((sp.Piecewise, sp.floor, sp.ceiling, sp.sign))


@dataclass(frozen=True)
class GuardMargin:
    """A strict margin that makes a domain-sensitive operation safe nearby."""

    expression: sp.Expr
    requirement: SignRequirement
    margin: sp.Expr


@dataclass(frozen=True)
class LipschitzCertificate:
    """A uniform scalar Lipschitz estimate on one selected domain."""

    domain: DomainKind
    norm: str
    constant_bound: sp.Expr
    active_variables: tuple[sp.Symbol, ...]
    uniform_parameters: tuple[sp.Symbol, ...]
    guard_margins: tuple[GuardMargin, ...]


@dataclass(frozen=True)
class LipschitzResult:
    verdict: ProofVerdict
    certificate: LipschitzCertificate | None = None
    decisive_subexpression: sp.Expr | None = None
    witness: Point | None = None
    reason: str | None = None


@dataclass(frozen=True)
class NetworkLipschitzCertificate:
    """A bound for the vector field F(c) = S r(c)."""

    domain: DomainKind
    norm: str
    constant_bound: sp.Expr
    component_bounds: tuple[tuple[str, sp.Expr], ...]
    active_variables: tuple[sp.Symbol, ...]
    uniform_parameters: tuple[sp.Symbol, ...]


def _true(statement: object) -> bool:
    return statement is True or statement is sp.true


def _ordered(symbols: Sequence[sp.Symbol] | set[sp.Symbol]) -> tuple[sp.Symbol, ...]:
    return tuple(sorted(symbols, key=lambda symbol: symbol.name))


def _merge_guards(results: Sequence[LipschitzResult]) -> tuple[GuardMargin, ...]:
    unique: dict[tuple[sp.Expr, SignRequirement], GuardMargin] = {}
    for result in results:
        if result.certificate is None:
            continue
        for guard in result.certificate.guard_margins:
            unique[(guard.expression, guard.requirement)] = guard
    return tuple(unique.values())


def _pass(
    expression: sp.Expr,
    domain: ConcentrationDomain,
    active: tuple[sp.Symbol, ...],
    constant: sp.Expr,
    guards: tuple[GuardMargin, ...] = (),
) -> LipschitzResult:
    used_variables = _ordered(set(expression.free_symbols) & set(active))
    parameters = _ordered(
        set(expression.free_symbols) & set(domain.parameter_intervals)
    )
    return LipschitzResult(
        ProofVerdict.PASS,
        LipschitzCertificate(
            domain.kind,
            "L_infinity",
            sp.sympify(constant),
            used_variables,
            parameters,
            guards,
        ),
    )


def _first_unproved(results: Sequence[LipschitzResult]) -> LipschitzResult | None:
    return next(
        (result for result in results if result.verdict is not ProofVerdict.PASS),
        None,
    )


def _strict_guard(
    analyzer: ExpressionAnalyzer,
    expression: sp.Expr,
    domain: ConcentrationDomain,
    active: tuple[sp.Symbol, ...],
    requirement: SignRequirement,
) -> GuardMargin | LipschitzResult:
    """Prove and quantify the strict distance from a singular boundary."""

    proof = analyzer.prove_sign(expression, domain, requirement, active)
    if proof.verdict is ProofVerdict.FAIL:
        return LipschitzResult(
            ProofVerdict.FAIL,
            decisive_subexpression=expression,
            witness=proof.witness,
            reason=proof.reason,
        )
    if proof.verdict is ProofVerdict.UNKNOWN:
        return LipschitzResult(
            ProofVerdict.UNKNOWN,
            decisive_subexpression=proof.result.decisive_subexpression or expression,
            reason=proof.reason,
        )

    bounds = analyzer.bounds(expression, domain, active)
    margin = None
    if bounds.known and _true(bounds.lower > 0):
        margin = bounds.lower
    elif (
        requirement is SignRequirement.NONZERO
        and bounds.known
        and _true(bounds.upper < 0)
    ):
        margin = -bounds.upper
    if margin is None:
        return LipschitzResult(
            ProofVerdict.UNKNOWN,
            decisive_subexpression=expression,
            reason=(
                "A strict guard was proved, but its positive margin was not bounded."
            ),
        )
    return GuardMargin(expression, requirement, margin)


def _absolute_bound(
    analyzer: ExpressionAnalyzer,
    expression: sp.Expr,
    domain: ConcentrationDomain,
    active: tuple[sp.Symbol, ...],
) -> sp.Expr | LipschitzResult:
    bounds = analyzer.bounds(expression, domain, active)
    if bounds.absolute_upper is None:
        return LipschitzResult(
            ProofVerdict.UNKNOWN,
            decisive_subexpression=bounds.decisive_subexpression or expression,
            reason=bounds.reason or "No finite absolute bound was proved.",
        )
    return bounds.absolute_upper


def _power_lipschitz(
    analyzer: ExpressionAnalyzer,
    expression: sp.Pow,
    domain: ConcentrationDomain,
    active: tuple[sp.Symbol, ...],
) -> LipschitzResult:
    base, exponent = expression.args
    if exponent.is_number is not True or exponent.is_real is not True:
        return LipschitzResult(
            ProofVerdict.UNKNOWN,
            decisive_subexpression=exponent,
            reason="The power exponent is not a real constant.",
        )

    base_result = analyzer.lipschitz(base, domain, active)
    if base_result.verdict is not ProofVerdict.PASS:
        return base_result
    assert base_result.certificate is not None
    guards = base_result.certificate.guard_margins

    requirement = None
    if exponent.is_integer and exponent.is_negative:
        requirement = SignRequirement.NONZERO
    elif exponent.is_integer is not True and set(base.free_symbols) & set(active):
        requirement = SignRequirement.POSITIVE
    elif exponent.is_integer is not True and exponent.is_negative:
        requirement = SignRequirement.POSITIVE

    if requirement is not None:
        guard = _strict_guard(analyzer, base, domain, active, requirement)
        if isinstance(guard, LipschitzResult):
            return guard
        guards = (*guards, guard)

    base_constant = base_result.certificate.constant_bound
    if base_constant == 0 or exponent == 0:
        return _pass(expression, domain, active, sp.S.Zero, guards)

    # L(f**a) <= sup_D |a f**(a - 1)| L(f).
    derivative_factor = exponent * base ** (exponent - 1)
    factor_bound = _absolute_bound(analyzer, derivative_factor, domain, active)
    if isinstance(factor_bound, LipschitzResult):
        return factor_bound
    return _pass(expression, domain, active, factor_bound * base_constant, guards)


def _derivative_fallback(
    analyzer: ExpressionAnalyzer,
    expression: sp.Expr,
    domain: ConcentrationDomain,
    active: tuple[sp.Symbol, ...],
) -> LipschitzResult:
    """Use sup ||gradient r||_1 when no explicit compositional rule exists."""

    if expression.func in _DISCONTINUOUS:
        return LipschitzResult(
            ProofVerdict.UNKNOWN,
            decisive_subexpression=expression,
            reason=f"Discontinuous function {expression.func.__name__} is unsupported.",
        )
    if sp.count_ops(expression) > _DERIVATIVE_SIZE_LIMIT:
        return LipschitzResult(
            ProofVerdict.UNKNOWN,
            decisive_subexpression=expression,
            reason="Expression exceeds the derivative-fallback size budget.",
        )
    if expression.is_real is not True:
        return LipschitzResult(
            ProofVerdict.UNKNOWN,
            decisive_subexpression=expression,
            reason="The unfamiliar function is not known to be real-valued.",
        )

    variables = _ordered(set(expression.free_symbols) & set(active))
    derivative_bounds: list[sp.Expr] = []
    for variable in variables:
        derivative = sp.diff(expression, variable)
        if derivative.has(sp.Derivative):
            return LipschitzResult(
                ProofVerdict.UNKNOWN,
                decisive_subexpression=expression,
                reason=f"Derivative with respect to {variable} is unavailable.",
            )
        defined = analyzer.defined(derivative, domain, active)
        if defined.verdict is not ProofVerdict.PASS:
            return LipschitzResult(
                defined.verdict,
                decisive_subexpression=defined.decisive_subexpression or derivative,
                witness=defined.witness,
                reason=defined.reason,
            )
        bound = _absolute_bound(analyzer, derivative, domain, active)
        if isinstance(bound, LipschitzResult):
            return bound
        derivative_bounds.append(bound)
    return _pass(expression, domain, active, sum(derivative_bounds, sp.S.Zero))


def compute_lipschitz(
    analyzer: ExpressionAnalyzer,
    expression: sp.Expr,
    domain: ConcentrationDomain,
    active: tuple[sp.Symbol, ...],
) -> LipschitzResult:
    """Apply the elementary Lipschitz rule matching ``expression``."""

    defined = analyzer.defined(expression, domain, active)
    if defined.verdict is ProofVerdict.FAIL or (
        defined.verdict is ProofVerdict.UNKNOWN and expression.is_Atom
    ):
        return LipschitzResult(
            defined.verdict,
            decisive_subexpression=defined.decisive_subexpression,
            witness=defined.witness,
            reason=defined.reason,
        )

    if expression.is_Atom:
        constant = sp.S.One if expression in active else sp.S.Zero
        return _pass(expression, domain, active, constant)

    if isinstance(expression, sp.Pow):
        return _power_lipschitz(analyzer, expression, domain, active)

    children = tuple(
        analyzer.lipschitz(argument, domain, active) for argument in expression.args
    )
    unproved = _first_unproved(children)

    if isinstance(expression, sp.Add):
        if unproved is not None:
            return unproved
        # L(sum_i f_i) <= sum_i L(f_i).
        constant = sum(
            result.certificate.constant_bound for result in children
        )
        return _pass(expression, domain, active, constant, _merge_guards(children))

    if isinstance(expression, sp.Mul):
        if unproved is not None:
            return unproved
        # L(product_i f_i) <= sum_i L(f_i) product_(j != i) sup_D |f_j|.
        constants = tuple(result.certificate.constant_bound for result in children)
        varying_indices = tuple(i for i, value in enumerate(constants) if value != 0)
        if not varying_indices:
            return _pass(expression, domain, active, sp.S.Zero, _merge_guards(children))

        needed_bounds = {
            other
            for index in varying_indices
            for other in range(len(children))
            if other != index
        }
        factor_bounds: dict[int, sp.Expr] = {}
        for index in needed_bounds:
            argument = expression.args[index]
            bound = _absolute_bound(analyzer, argument, domain, active)
            if isinstance(bound, LipschitzResult):
                return bound
            factor_bounds[index] = bound
        constant = sum(
            constants[index]
            * sp.prod(
                factor_bounds[other]
                for other in range(len(children))
                if other != index
            )
            for index in varying_indices
        )
        return _pass(expression, domain, active, constant, _merge_guards(children))

    if expression.func is sp.Abs:
        if unproved is not None:
            return unproved
        return _pass(
            expression,
            domain,
            active,
            children[0].certificate.constant_bound,
            _merge_guards(children),
        )

    if expression.func in {sp.Min, sp.Max}:
        if unproved is not None:
            return unproved
        constants = tuple(result.certificate.constant_bound for result in children)
        return _pass(
            expression, domain, active, sp.Max(*constants), _merge_guards(children)
        )

    if expression.func in {sp.sin, sp.cos, sp.tanh, sp.atan}:
        if unproved is not None:
            return unproved
        return _pass(
            expression,
            domain,
            active,
            children[0].certificate.constant_bound,
            _merge_guards(children),
        )

    if expression.func is sp.exp:
        if unproved is not None:
            return unproved
        bounds = analyzer.bounds(expression.args[0], domain, active)
        if not bounds.known:
            return LipschitzResult(
                ProofVerdict.UNKNOWN,
                decisive_subexpression=(
                    bounds.decisive_subexpression or expression.args[0]
                ),
                reason=bounds.reason,
            )
        # L(exp(f)) <= exp(sup_D f) L(f).
        constant = sp.exp(bounds.upper) * children[0].certificate.constant_bound
        return _pass(expression, domain, active, constant, _merge_guards(children))

    if expression.func is sp.log:
        if unproved is not None:
            return unproved
        guard = _strict_guard(
            analyzer,
            expression.args[0],
            domain,
            active,
            SignRequirement.POSITIVE,
        )
        if isinstance(guard, LipschitzResult):
            return guard
        # L(log(f)) <= L(f) / inf_D f.
        constant = children[0].certificate.constant_bound / guard.margin
        return _pass(
            expression, domain, active, constant, (*_merge_guards(children), guard)
        )

    if expression.func in {sp.sinh, sp.cosh}:
        if unproved is not None:
            return unproved
        derivative = (
            sp.cosh(expression.args[0])
            if expression.func is sp.sinh
            else sp.sinh(expression.args[0])
        )
        derivative_bound = _absolute_bound(analyzer, derivative, domain, active)
        if isinstance(derivative_bound, LipschitzResult):
            return derivative_bound
        constant = derivative_bound * children[0].certificate.constant_bound
        return _pass(expression, domain, active, constant, _merge_guards(children))

    return _derivative_fallback(analyzer, expression, domain, active)


def derive_network_lipschitz(
    domain: ConcentrationDomain,
    species_ids: Sequence[str],
    stoichiometry: sp.MatrixBase,
    rate_certificates: Sequence[LipschitzCertificate],
) -> NetworkLipschitzCertificate:
    """Derive ``F = S r`` bounds algebraically, without differentiating F."""

    constants = tuple(item.constant_bound for item in rate_certificates)
    if stoichiometry.cols != len(constants) or stoichiometry.rows != len(species_ids):
        raise ValueError("Stoichiometry dimensions do not match the certificates.")
    # L(F_i) <= sum_j |S_ij| L(r_j); L(F) <= max_i L(F_i).
    component_bounds = tuple(
        (
            species_id,
            sum(
                abs(stoichiometry[row, column]) * constants[column]
                for column in range(stoichiometry.cols)
            ),
        )
        for row, species_id in enumerate(species_ids)
    )
    active = _ordered(
        {
            symbol
            for certificate in rate_certificates
            for symbol in certificate.active_variables
        }
    )
    parameters = _ordered(
        {
            symbol
            for certificate in rate_certificates
            for symbol in certificate.uniform_parameters
        }
    )
    return NetworkLipschitzCertificate(
        domain.kind,
        "L_infinity",
        sp.Max(*(bound for _, bound in component_bounds)),
        component_bounds,
        active,
        parameters,
    )


__all__ = (
    "GuardMargin",
    "LipschitzCertificate",
    "LipschitzResult",
    "NetworkLipschitzCertificate",
    "derive_network_lipschitz",
)
