"""Prove componentwise non-repulsion on the augmented state domain."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import sympy as sp
from sympy.core.relational import Relational

from ..case import Case
from .models import (
    CheckContext,
    CheckDefinition,
    CheckOutcome,
    CheckScope,
    CheckStatus,
    CheckValue,
)
from .network import NetworkExpressions, network_expressions
from .symbolic_domain import (
    NEGATIVE,
    NONNEGATIVE,
    NONPOSITIVE,
    POSITIVE,
    ZERO,
    LinearDomain,
    Point,
    Proof,
    number,
)

MAX_SOURCE_OPERATIONS = 5000


@dataclass(frozen=True)
class NegativeSideSpeciesResult:
    """The two nested source-sign implications for one concentration."""

    species_id: str
    source: sp.Expr
    nonrepelling: bool | None
    attracting: bool | None
    nonpositive_feasible: bool
    negative_feasible: bool
    nonrepulsion_counterexample: Point | None = None
    attraction_counterexample: Point | None = None
    diagnostic: str | None = None


@dataclass(frozen=True)
class NegativeSideRecoveryResult:
    """Componentwise recovery conclusions on the complete augmented domain."""

    source_terms: Mapping[str, sp.Expr]
    species: tuple[NegativeSideSpeciesResult, ...]

    @staticmethod
    def _aggregate(conclusions: tuple[bool | None, ...]) -> bool | None:
        if any(conclusion is False for conclusion in conclusions):
            return False
        if any(conclusion is None for conclusion in conclusions):
            return None
        return True

    @property
    def nonrepelling(self) -> bool | None:
        """Aggregate the required non-repulsion conclusions."""

        return self._aggregate(
            tuple(result.nonrepelling for result in self.species)
        )

    @property
    def attracting(self) -> bool | None:
        """Aggregate the optional strict-attraction conclusions."""

        return self._aggregate(
            tuple(result.attracting for result in self.species)
        )


def _augmented_domain(case: Case) -> LinearDomain:
    weak: list[Relational] = []
    strict: list[sp.Expr] = []
    box: list[tuple[sp.Expr, sp.Expr]] = []

    for symbol, state_bound in case.state_bounds.items():
        raw_lower, raw_upper = state_bound.interval(include_excursion=True)
        lower, upper = number(raw_lower), number(raw_upper)
        weak.extend(
            (
                sp.Ge(symbol - lower, 0, evaluate=False),
                sp.Ge(upper - symbol, 0, evaluate=False),
            )
        )
        box.append((lower, upper))

    if case.gas_closure is not None:
        minimum = case.gas_closure.derived_minimum_total(case.state_bounds)
        if minimum is None:
            strict.extend(case.gas_closure.augmented_strict_constraints)
        else:
            weak.append(
                sp.Ge(
                    case.gas_closure.total_concentration - minimum,
                    0,
                    evaluate=False,
                )
            )

    domain = LinearDomain(
        tuple(case.state_bounds),
        tuple(weak),
        tuple(strict),
        tuple(box),
    )
    if not domain.feasible()[0]:
        raise ValueError("Augmented state domain is empty.")
    return domain


def _restricted_domain(
    domain: LinearDomain,
    symbol: sp.Symbol,
    *,
    strict: bool,
) -> LinearDomain:
    if strict:
        return LinearDomain(
            domain.variables,
            domain.weak,
            domain.strict + (-symbol,),
            domain.bounds,
        )
    return LinearDomain(
        domain.variables,
        domain.weak + (sp.Le(symbol, 0, evaluate=False),),
        domain.strict,
        domain.bounds,
    )


def _assumptions(
    case: Case,
    subject: sp.Symbol,
    *,
    strict: bool,
) -> Mapping[sp.Symbol, sp.Symbol]:
    replacements: dict[sp.Symbol, sp.Symbol] = {}
    for symbol, state_bound in case.state_bounds.items():
        if symbol == subject:
            kind = "negative" if strict else "nonpositive"
        else:
            raw_lower, raw_upper = state_bound.interval(include_excursion=True)
            lower, upper = number(raw_lower), number(raw_upper)
            if lower > 0:
                kind = "positive"
            elif upper < 0:
                kind = "negative"
            elif lower == 0:
                kind = "nonnegative"
            elif upper == 0:
                kind = "nonpositive"
            else:
                kind = "real"
        replacements[symbol] = sp.Dummy(symbol.name, **{kind: True})
    return MappingProxyType(replacements)


def _violates_at_point(
    proof: Proof,
    expression: sp.Expr,
    requirement: str,
) -> bool:
    sign = proof.value_sign(expression)
    if requirement == NONNEGATIVE:
        return sign == NEGATIVE
    return sign in (NEGATIVE, ZERO, NONPOSITIVE)


def _prove_universal_sign(
    proof: Proof,
    expression: sp.Expr,
    requirement: str,
) -> tuple[bool | None, Point | None]:
    # An exact violating feasible point disproves the universal implication
    # without requiring a potentially expensive global sign reduction.
    if _violates_at_point(proof, expression, requirement):
        return False, proof.point

    conclusion = proof.proves(expression, requirement)
    if conclusion is True:
        return True, None

    # For affine expressions this is an exact search for a point violating
    # the universal inequality, even when the first feasible point happened
    # to satisfy it.
    counterexample = proof.violating_point(expression, requirement)
    if counterexample is not None:
        return False, counterexample
    return None, None


def _analyse_species(
    case: Case,
    domain: LinearDomain,
    species_id: str,
    source: sp.Expr,
) -> NegativeSideSpeciesResult:
    symbol = case.states.concentration(species_id)
    nonpositive_domain = _restricted_domain(domain, symbol, strict=False)
    nonpositive_feasible, nonpositive_point = nonpositive_domain.feasible()
    negative_domain = _restricted_domain(domain, symbol, strict=True)
    negative_feasible, negative_point = negative_domain.feasible()

    operations = sp.count_ops(source)
    if operations > MAX_SOURCE_OPERATIONS:
        diagnostic = (
            "Source term exceeds the symbolic operation limit "
            f"({operations} > {MAX_SOURCE_OPERATIONS})."
        )
        return NegativeSideSpeciesResult(
            species_id,
            source,
            None if nonpositive_feasible else True,
            None if negative_feasible else True,
            nonpositive_feasible,
            negative_feasible,
            diagnostic=diagnostic,
        )

    if not nonpositive_feasible or nonpositive_point is None:
        nonrepelling, nonrepulsion_counterexample = True, None
    else:
        nonpositive_proof = Proof(
            nonpositive_domain,
            _assumptions(case, symbol, strict=False),
            nonpositive_point,
        )
        nonrepelling, nonrepulsion_counterexample = _prove_universal_sign(
            nonpositive_proof,
            source,
            NONNEGATIVE,
        )

    if not negative_feasible or negative_point is None:
        attracting, attraction_counterexample = True, None
    else:
        negative_proof = Proof(
            negative_domain,
            _assumptions(case, symbol, strict=True),
            negative_point,
        )
        attracting, attraction_counterexample = _prove_universal_sign(
            negative_proof,
            source,
            POSITIVE,
        )

    return NegativeSideSpeciesResult(
        species_id,
        source,
        nonrepelling,
        attracting,
        nonpositive_feasible,
        negative_feasible,
        nonrepulsion_counterexample,
        attraction_counterexample,
    )


def check_negative_side_recovery(
    case: Case,
    *,
    network: NetworkExpressions | None = None,
) -> NegativeSideRecoveryResult:
    """Prove ``x_i <= 0 => f_i(x) >= 0`` on the augmented domain.

    Every other state remains free over its augmented bounds, including its
    own possible negative excursion.  Strict attraction,
    ``x_i < 0 => f_i(x) > 0``, is attempted as an additional conclusion.
    Conservation rays and stoichiometric compatibility classes are
    intentionally not assumed.
    """

    network = network or network_expressions(case)
    domain = _augmented_domain(case)
    eligible = tuple(
        species_id
        for species_id, symbol in case.states.concentrations.items()
        if (
            lower := case.state_bounds[symbol].interval(
                include_excursion=True
            )[0]
        )
        < 0
    )
    results = tuple(
        _analyse_species(
            case,
            domain,
            species_id,
            network.source_terms[species_id],
        )
        for species_id in eligible
    )
    return NegativeSideRecoveryResult(network.source_terms, results)


def _proof_label(conclusion: bool | None) -> str:
    if conclusion is True:
        return "proved"
    if conclusion is False:
        return "disproved"
    return "unresolved"


def _point_detail(point: Point) -> str:
    values = ", ".join(f"{symbol}={value}" for symbol, value in point.items())
    return f"Exact augmented-domain counterexample: {values}."


def _short(expression: sp.Expr, limit: int = 300) -> str:
    rendered = sp.sstr(expression)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _species_details(result: NegativeSideSpeciesResult) -> tuple[str, ...]:
    if not result.nonpositive_feasible:
        nonrepulsion = "vacuously proved (no augmented state has x_i <= 0)"
    else:
        nonrepulsion = _proof_label(result.nonrepelling)
    if not result.negative_feasible:
        attraction = "vacuously proved (no augmented state has x_i < 0)"
    else:
        attraction = _proof_label(result.attracting)

    details = [
        f"{result.species_id}: negative-side non-repulsion {nonrepulsion}; "
        f"strict attraction {attraction}."
    ]
    if result.diagnostic is not None:
        details.append(f"  {result.diagnostic}")
    if result.nonrepulsion_counterexample is not None:
        details.append(
            f"  Non-repulsion fails for source {_short(result.source)}."
        )
        details.append("  " + _point_detail(result.nonrepulsion_counterexample))
    elif result.attraction_counterexample is not None:
        details.append(
            f"  Strict attraction fails for source {_short(result.source)}."
        )
        details.append("  " + _point_detail(result.attraction_counterexample))
    return tuple(details)


def run(case: Case, context: CheckContext) -> CheckOutcome:
    network = context.cached(
        case,
        "network",
        lambda: network_expressions(case),
    )
    result = check_negative_side_recovery(case, network=network)
    if not result.species:
        return CheckOutcome(
            status=CheckStatus.UNAVAILABLE,
            details=("No concentration has a negative excursion.",),
        )

    if result.nonrepelling is False:
        status = CheckStatus.FAIL
    elif result.nonrepelling is None:
        status = CheckStatus.INDETERMINATE
    else:
        status = CheckStatus.PASS

    details = [
        "Required condition: x_i <= 0 implies f_i(x) >= 0 throughout "
        "the augmented domain.",
        "Strict attraction (x_i < 0 implies f_i(x) > 0) is an optional "
        "stronger conclusion.",
        "Other states retain their augmented bounds; no conservation "
        "constraints are assumed.",
    ]
    for species_result in result.species:
        details.extend(_species_details(species_result))
    return CheckOutcome(
        status=status,
        details=tuple(details),
        values=(CheckValue("Species checked", len(result.species)),),
    )


CHECK = CheckDefinition(
    id="negative_side_recovery",
    name="Negative-side recovery",
    group="Physical checks",
    scope=CheckScope.CASE,
    run=run,
)
