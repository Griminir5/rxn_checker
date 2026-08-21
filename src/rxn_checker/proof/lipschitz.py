"""Composable concentration-space Lipschitz bounds."""

from collections.abc import Sequence
from dataclasses import dataclass

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

    def component(self, variable: sp.Symbol) -> sp.Expr:
        return dict(self.components).get(variable, sp.S.Zero)


def _ordered(symbols) -> tuple[sp.Symbol, ...]:
    return tuple(sorted(symbols, key=lambda symbol: symbol.name))


def _failure(verdict, expression=None, witness=None, reason=None):
    return LipschitzResult(verdict, decisive_subexpression=expression,
                           witness=witness, reason=reason)


def compute_lipschitz(analyzer: ExpressionAnalyzer, expression: sp.Expr,
                      domain: ConcentrationDomain,
                      active: tuple[sp.Symbol, ...]) -> LipschitzResult:
    """Compute small recursive bounds and wrap metadata once at the root."""
    cache: dict[sp.Expr, _LipBound | LipschitzResult] = {}

    def scaled_components(child, factor):
        return tuple((variable, factor * bound) for variable, bound in child.components)

    def combined_components(children, combine=sum):
        variables = _ordered(
            variable for child in children for variable, _ in child.components
        )
        return tuple(
            (variable, combine(child.component(variable) for child in children))
            for variable in variables
        )

    def absolute(value):
        bounds = analyzer.bounds(value, domain)
        return bounds.absolute_upper if bounds.absolute_upper is not None else _failure(
            ProofVerdict.UNKNOWN, bounds.decisive_subexpression or value,
            reason=bounds.reason or "No finite absolute bound was proved.")

    def guard(value, requirement):
        proof = analyzer.prove_sign(value, domain, requirement)
        if proof.verdict is not ProofVerdict.PASS:
            decisive = value if proof.verdict is ProofVerdict.FAIL else (
                proof.result.decisive_subexpression or value)
            return _failure(proof.verdict, decisive, proof.witness, proof.reason)
        bounds = analyzer.bounds(value, domain)
        margin = bounds.lower if bounds.known and bounds.lower > 0 else (
            -bounds.upper if requirement is SignRequirement.NONZERO
            and bounds.known and bounds.upper < 0 else None)
        if margin is None:
            return _failure(ProofVerdict.UNKNOWN, value, reason=
                "A strict guard was proved, but its positive margin was not bounded.")
        return GuardMargin(value, requirement, margin)

    def merged(children):
        unique = {}
        for child in children:
            for item in child.guards:
                unique[item.expression, item.requirement] = item
        return tuple(unique.values())

    def derivative_fallback(value):
        if value.func in _DISCONTINUOUS:
            return _failure(ProofVerdict.UNKNOWN, value, reason=
                            f"Discontinuous function {value.func.__name__} is unsupported.")
        if sp.count_ops(value) > _DERIVATIVE_SIZE_LIMIT:
            return _failure(ProofVerdict.UNKNOWN, value, reason=
                            "Expression exceeds the derivative-fallback size budget.")
        if value.is_real is not True:
            return _failure(ProofVerdict.UNKNOWN, value, reason=
                            "The unfamiliar function is not known to be real-valued.")
        components = []
        for variable in _ordered(value.free_symbols & set(active)):
            derivative = sp.diff(value, variable)
            if derivative.has(sp.Derivative):
                return _failure(ProofVerdict.UNKNOWN, value, reason=
                                f"Derivative with respect to {variable} is unavailable.")
            defined = analyzer.defined(derivative, domain)
            if defined.verdict is not ProofVerdict.PASS:
                return _failure(defined.verdict,
                                defined.decisive_subexpression or derivative,
                                defined.witness, defined.reason)
            bound = absolute(derivative)
            if isinstance(bound, LipschitzResult):
                return bound
            components.append((variable, bound))
        return _LipBound(
            sum((bound for _, bound in components), sp.S.Zero),
            components=tuple(components),
        )

    def visit(value):
        if value in cache:
            return cache[value]
        defined = analyzer.defined(value, domain)
        if defined.verdict is ProofVerdict.FAIL or (
            defined.verdict is ProofVerdict.UNKNOWN and value.is_Atom
        ):
            result = _failure(defined.verdict, defined.decisive_subexpression,
                              defined.witness, defined.reason)
        elif value.is_Atom:
            active_atom = value in active
            result = _LipBound(
                sp.S.One if active_atom else sp.S.Zero,
                components=((value, sp.S.One),) if active_atom else (),
            )
        elif isinstance(value, sp.Pow):
            base, exponent = value.args
            child = visit(base)
            if isinstance(child, LipschitzResult):
                result = child
            elif exponent.is_number is not True or exponent.is_real is not True:
                result = _failure(ProofVerdict.UNKNOWN, exponent,
                                  reason="The power exponent is not a real constant.")
            else:
                requirement = (SignRequirement.NONZERO
                    if exponent.is_integer and exponent.is_negative else
                    SignRequirement.POSITIVE
                    if exponent.is_integer is not True and
                    (exponent.is_negative or base.free_symbols & set(active)) else None)
                guarded = guard(base, requirement) if requirement else None
                if isinstance(guarded, LipschitzResult):
                    result = guarded
                elif child.constant == 0 or exponent == 0:
                    result = _LipBound(
                        sp.S.Zero,
                        child.guards + ((guarded,) if guarded else ()),
                    )
                else:
                    factor = absolute(exponent * base ** (exponent - 1))
                    result = factor if isinstance(factor, LipschitzResult) else _LipBound(
                        factor * child.constant,
                        child.guards + ((guarded,) if guarded else ()),
                        scaled_components(child, factor),
                    )
        else:
            children = tuple(visit(argument) for argument in value.args)
            failed = next((item for item in children if isinstance(item, LipschitzResult)), None)
            if failed is not None:
                result = failed
            else:
                guards = merged(children)
                constants = tuple(item.constant for item in children)
                if isinstance(value, sp.Add):
                    result = _LipBound(
                        sum(constants), guards, combined_components(children)
                    )
                elif isinstance(value, sp.Mul):
                    if not any(constants):
                        result = _LipBound(sp.S.Zero, guards)
                    else:
                        terms = []
                        component_terms = {variable: [] for variable in active}
                        for i, constant in enumerate(constants):
                            if constant == 0:
                                continue
                            bounds = tuple(absolute(argument) for j, argument in
                                           enumerate(value.args) if j != i)
                            failed = next((item for item in bounds
                                           if isinstance(item, LipschitzResult)), None)
                            if failed:
                                break
                            terms.append(constant * sp.prod(bounds))
                            factor = sp.prod(bounds)
                            for variable, bound in children[i].components:
                                component_terms[variable].append(bound * factor)
                        result = failed or _LipBound(
                            sum(terms),
                            guards,
                            tuple(
                                (variable, sum(values))
                                for variable, values in component_terms.items()
                                if values
                            ),
                        )
                elif value.func is sp.Abs:
                    result = _LipBound(constants[0], guards, children[0].components)
                elif value.func in {sp.Min, sp.Max}:
                    result = _LipBound(
                        sp.Max(*constants),
                        guards,
                        combined_components(children, lambda values: sp.Max(*tuple(values))),
                    )
                elif value.func in {sp.sin, sp.cos, sp.tanh, sp.atan}:
                    result = _LipBound(constants[0], guards, children[0].components)
                elif value.func is sp.exp:
                    bounds = analyzer.bounds(value.args[0], domain)
                    factor = sp.exp(bounds.upper) if bounds.known else None
                    result = (_LipBound(
                                  factor * constants[0], guards,
                                  scaled_components(children[0], factor))
                              if bounds.known else _failure(ProofVerdict.UNKNOWN,
                              bounds.decisive_subexpression or value.args[0], reason=bounds.reason))
                elif value.func is sp.log:
                    guarded = guard(value.args[0], SignRequirement.POSITIVE)
                    result = guarded if isinstance(guarded, LipschitzResult) else _LipBound(
                        constants[0] / guarded.margin,
                        guards + (guarded,),
                        scaled_components(children[0], 1 / guarded.margin),
                    )
                elif value.func in {sp.sinh, sp.cosh}:
                    derivative = (sp.cosh if value.func is sp.sinh else sp.sinh)(value.args[0])
                    bound = absolute(derivative)
                    result = bound if isinstance(bound, LipschitzResult) else _LipBound(
                        bound * constants[0], guards,
                        scaled_components(children[0], bound),
                    )
                else:
                    result = derivative_fallback(value)
        cache[value] = result
        return result

    bound = visit(expression)
    if isinstance(bound, LipschitzResult):
        return bound
    used = expression.free_symbols
    certificate = LipschitzCertificate(
        domain.kind, "L_infinity", sp.sympify(bound.constant),
        _ordered(used & set(active)),
        _ordered(used & (set(domain.parameter_intervals) - set(active))),
        bound.guards,
        GradientEnvelope(bound.components, bound.guards),
    )
    return LipschitzResult(ProofVerdict.PASS, certificate)


def derive_network_lipschitz(domain: ConcentrationDomain, species_ids: Sequence[str],
                             stoichiometry: sp.MatrixBase,
                             rate_certificates: Sequence[LipschitzCertificate]
                             ) -> NetworkLipschitzCertificate:
    constants = tuple(item.constant_bound for item in rate_certificates)
    if stoichiometry.cols != len(constants) or stoichiometry.rows != len(species_ids):
        raise ValueError("Stoichiometry dimensions do not match the certificates.")
    components = tuple((species_id, sum(abs(stoichiometry[row, column]) * constants[column]
        for column in range(stoichiometry.cols))) for row, species_id in enumerate(species_ids))
    active = _ordered({symbol for item in rate_certificates for symbol in item.active_variables})
    parameters = _ordered({symbol for item in rate_certificates for symbol in item.uniform_parameters})
    return NetworkLipschitzCertificate(domain.kind, "L_infinity",
        sp.Max(*(bound for _, bound in components)), components, active, parameters)


__all__ = ("GradientEnvelope", "GuardMargin", "LipschitzCertificate", "LipschitzResult",
           "NetworkLipschitzCertificate", "derive_network_lipschitz")
