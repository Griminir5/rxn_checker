"""Composable concentration-space Lipschitz bounds."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import sympy as sp

from ..domain import ConcentrationDomain, DomainKind
from .analysis import ExpressionAnalyzer, Point, ProofVerdict, SignRequirement

_DERIVATIVE_SIZE_LIMIT = 80
_DISCONTINUOUS = frozenset((sp.Piecewise, sp.floor, sp.ceiling, sp.sign))


@dataclass(frozen=True)
class GuardMargin:
    expression: sp.Expr
    requirement: SignRequirement
    margin: sp.Expr


@dataclass(frozen=True)
class GradientEnvelope:
    """Per-variable absolute derivative bounds from one compositional walk."""

    components: tuple[tuple[sp.Symbol, sp.Expr], ...]
    guards: tuple[GuardMargin, ...] = ()

    @property
    def linfinity_lipschitz(self) -> sp.Expr:
        return sum((bound for _, bound in self.components), sp.S.Zero)


@dataclass(frozen=True)
class LipschitzCertificate:
    domain: DomainKind
    norm: str
    constant_bound: sp.Expr
    active_variables: tuple[sp.Symbol, ...]
    uniform_parameters: tuple[sp.Symbol, ...]
    guard_margins: tuple[GuardMargin, ...]
    gradient_envelope: GradientEnvelope | None = None


@dataclass(frozen=True)
class LipschitzResult:
    verdict: ProofVerdict
    certificate: LipschitzCertificate | None = None
    decisive_subexpression: sp.Expr | None = None
    witness: Point | None = None
    reason: str | None = None


@dataclass(frozen=True)
class NetworkLipschitzCertificate:
    domain: DomainKind
    norm: str
    constant_bound: sp.Expr
    component_bounds: tuple[tuple[str, sp.Expr], ...]
    active_variables: tuple[sp.Symbol, ...]
    uniform_parameters: tuple[sp.Symbol, ...]


@dataclass(frozen=True)
class _LipBound:
    constant: sp.Expr
    guards: tuple[GuardMargin, ...] = ()
    components: tuple[tuple[sp.Symbol, sp.Expr], ...] = ()

    def scaled(self, factor, guards=()):
        return _LipBound(
            factor * self.constant,
            self.guards + tuple(guards),
            tuple((variable, factor * bound) for variable, bound in self.components),
        )


def _ordered(symbols):
    return tuple(sorted(set(symbols), key=lambda symbol: symbol.name))


def _combine(children, maximum=False):
    children = tuple(children)
    combine = (lambda values: sp.Max(*values)) if maximum else sum
    gradients = [dict(child.components) for child in children]
    variables = _ordered(variable for gradient in gradients for variable in gradient)
    guards = {
        (guard.expression, guard.requirement): guard for child in children for guard in child.guards
    }
    return _LipBound(
        combine(child.constant for child in children),
        tuple(guards.values()),
        tuple(
            (variable, combine(gradient.get(variable, sp.S.Zero) for gradient in gradients))
            for variable in variables
        ),
    )


def _failure(verdict, expression=None, witness=None, reason=None):
    return LipschitzResult(
        verdict, decisive_subexpression=expression, witness=witness, reason=reason
    )


def compute_lipschitz(
    analyzer: ExpressionAnalyzer,
    expression: sp.Expr,
    domain: ConcentrationDomain,
    active: tuple[sp.Symbol, ...],
) -> LipschitzResult:
    """Propagate absolute derivative bounds through the expression tree."""

    def absolute(value):
        bounds = analyzer.bounds(value, domain)
        if bounds.absolute_upper is not None:
            return bounds.absolute_upper
        return _failure(
            ProofVerdict.UNKNOWN,
            bounds.decisive_subexpression or value,
            reason=bounds.reason or "No finite absolute bound was proved.",
        )

    def guard(value, requirement):
        proof = analyzer.prove_sign(value, domain, requirement)
        if proof.verdict is not ProofVerdict.PASS:
            decisive = (
                value
                if proof.verdict is ProofVerdict.FAIL
                else (proof.result.decisive_subexpression or value)
            )
            return _failure(proof.verdict, decisive, proof.witness, proof.reason)
        bounds = analyzer.bounds(value, domain)
        margin = None
        if bounds.known:
            if bounds.lower > 0:
                margin = bounds.lower
            elif requirement is SignRequirement.NONZERO and bounds.upper < 0:
                margin = -bounds.upper
        if margin is None:
            return _failure(
                ProofVerdict.UNKNOWN,
                value,
                reason="A strict guard was proved, but its positive margin was not bounded.",
            )
        return GuardMargin(value, requirement, margin)

    def derivative_fallback(value):
        if value.func in _DISCONTINUOUS:
            reason = f"Discontinuous function {value.func.__name__} is unsupported."
        elif sp.count_ops(value) > _DERIVATIVE_SIZE_LIMIT:
            reason = "Expression exceeds the derivative-fallback size budget."
        elif value.is_real is not True:
            reason = "The unfamiliar function is not known to be real-valued."
        else:
            components = []
            for variable in _ordered(value.free_symbols & set(active)):
                derivative = sp.diff(value, variable)
                if derivative.has(sp.Derivative):
                    return _failure(
                        ProofVerdict.UNKNOWN,
                        value,
                        reason=f"Derivative with respect to {variable} is unavailable.",
                    )
                defined = analyzer.defined(derivative, domain)
                if defined.verdict is not ProofVerdict.PASS:
                    return _failure(
                        defined.verdict,
                        defined.decisive_subexpression or derivative,
                        defined.witness,
                        defined.reason,
                    )
                bound = absolute(derivative)
                if isinstance(bound, LipschitzResult):
                    return bound
                components.append((variable, bound))
            return _LipBound(
                sum((bound for _, bound in components), sp.S.Zero), components=tuple(components)
            )
        return _failure(ProofVerdict.UNKNOWN, value, reason=reason)

    @cache
    def visit(value):
        defined = analyzer.defined(value, domain)
        if defined.verdict is ProofVerdict.FAIL or (
            defined.verdict is ProofVerdict.UNKNOWN and value.is_Atom
        ):
            return _failure(
                defined.verdict, defined.decisive_subexpression, defined.witness, defined.reason
            )
        if value.is_Atom:
            return (
                _LipBound(sp.S.One, components=((value, sp.S.One),))
                if value in active
                else _LipBound(sp.S.Zero)
            )
        if isinstance(value, sp.Pow):
            base, exponent = value.args
            child = visit(base)
            if isinstance(child, LipschitzResult):
                return child
            if exponent.is_number is not True or exponent.is_real is not True:
                return _failure(
                    ProofVerdict.UNKNOWN,
                    exponent,
                    reason="The power exponent is not a real constant.",
                )
            requirement = None
            if exponent.is_integer and exponent.is_negative:
                requirement = SignRequirement.NONZERO
            elif exponent.is_integer is not True and (
                exponent.is_negative or base.free_symbols & set(active)
            ):
                requirement = SignRequirement.POSITIVE
            guarded = guard(base, requirement) if requirement else None
            if isinstance(guarded, LipschitzResult):
                return guarded
            factor = (
                sp.S.Zero
                if child.constant == 0 or exponent == 0
                else absolute(exponent * base ** (exponent - 1))
            )
            return (
                factor
                if isinstance(factor, LipschitzResult)
                else child.scaled(factor, (guarded,) if guarded else ())
            )

        children = tuple(visit(argument) for argument in value.args)
        failure = next((child for child in children if isinstance(child, LipschitzResult)), None)
        if failure is not None:
            return failure
        if isinstance(value, sp.Add):
            return _combine(children)
        if isinstance(value, sp.Mul):
            terms = []
            for index, child in enumerate(children):
                if child.constant == 0:
                    terms.append(child)
                    continue
                factors = tuple(
                    absolute(argument) for i, argument in enumerate(value.args) if i != index
                )
                failure = next(
                    (factor for factor in factors if isinstance(factor, LipschitzResult)), None
                )
                if failure is not None:
                    return failure
                terms.append(child.scaled(sp.prod(factors)))
            return _combine(terms)
        if value.func in {sp.Min, sp.Max}:
            return _combine(children, maximum=True)

        if not children:
            return derivative_fallback(value)
        child = children[0]
        guards = ()
        if value.func in {sp.Abs, sp.sin, sp.cos, sp.tanh, sp.atan}:
            return child
        if value.func is sp.exp:
            bounds = analyzer.bounds(value.args[0], domain)
            if not bounds.known:
                return _failure(
                    ProofVerdict.UNKNOWN,
                    bounds.decisive_subexpression or value.args[0],
                    reason=bounds.reason,
                )
            factor = sp.exp(bounds.upper)
        elif value.func is sp.log:
            guarded = guard(value.args[0], SignRequirement.POSITIVE)
            if isinstance(guarded, LipschitzResult):
                return guarded
            factor, guards = 1 / guarded.margin, (guarded,)
        elif value.func in {sp.sinh, sp.cosh}:
            factor = absolute((sp.cosh if value.func is sp.sinh else sp.sinh)(value.args[0]))
        else:
            return derivative_fallback(value)
        return factor if isinstance(factor, LipschitzResult) else child.scaled(factor, guards)

    bound = visit(expression)
    if isinstance(bound, LipschitzResult):
        return bound
    used = expression.free_symbols
    certificate = LipschitzCertificate(
        domain.kind,
        "L_infinity",
        sp.sympify(bound.constant),
        _ordered(used & set(active)),
        _ordered(used & (set(domain.parameter_intervals) - set(active))),
        bound.guards,
        GradientEnvelope(bound.components, bound.guards),
    )
    return LipschitzResult(ProofVerdict.PASS, certificate)


def derive_network_lipschitz(
    domain: ConcentrationDomain,
    species_ids: Sequence[str],
    stoichiometry: sp.MatrixBase,
    rate_certificates: Sequence[LipschitzCertificate],
) -> NetworkLipschitzCertificate:
    constants = tuple(item.constant_bound for item in rate_certificates)
    if stoichiometry.cols != len(constants) or stoichiometry.rows != len(species_ids):
        raise ValueError("Stoichiometry dimensions do not match the certificates.")
    components = tuple(
        (
            species_id,
            sum(
                abs(stoichiometry[row, column]) * constants[column]
                for column in range(stoichiometry.cols)
            ),
        )
        for row, species_id in enumerate(species_ids)
    )
    active = _ordered({symbol for item in rate_certificates for symbol in item.active_variables})
    parameters = _ordered(
        {symbol for item in rate_certificates for symbol in item.uniform_parameters}
    )
    return NetworkLipschitzCertificate(
        domain.kind,
        "L_infinity",
        sp.Max(*(bound for _, bound in components)),
        components,
        active,
        parameters,
    )
