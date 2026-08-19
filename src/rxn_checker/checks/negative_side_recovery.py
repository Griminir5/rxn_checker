"""Prove componentwise non-repulsion on the augmented state domain."""

from collections.abc import Mapping
from dataclasses import dataclass

import sympy as sp

from ..case import Case
from ..context import AnalysisContext
from ..domain import ConcentrationDomain
from ..network import ReactionNetwork, build_network
from ..results import Evidence, Finding, Verdict
from .symbolic_domain import (
    NEGATIVE,
    NONNEGATIVE,
    NONPOSITIVE,
    POSITIVE,
    ZERO,
    Point,
    Proof,
    proof_for_domain,
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
    domain: ConcentrationDomain,
    species_id: str,
    source: sp.Expr,
) -> NegativeSideSpeciesResult:
    symbol = case.symbols.concentration(species_id)
    nonpositive_domain = domain.restrict(symbol, upper=0)
    nonpositive_feasible = nonpositive_domain.is_feasible()
    nonpositive_point = nonpositive_domain.exact_witness()
    negative_domain = domain.restrict(symbol, upper=0, strict_upper=True)
    negative_feasible = negative_domain.is_feasible()
    negative_point = negative_domain.exact_witness()

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
        nonpositive_proof = proof_for_domain(nonpositive_domain)
        nonrepelling, nonrepulsion_counterexample = _prove_universal_sign(
            nonpositive_proof,
            source,
            NONNEGATIVE,
        )

    if not negative_feasible or negative_point is None:
        attracting, attraction_counterexample = True, None
    else:
        negative_proof = proof_for_domain(negative_domain)
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
    domain: ConcentrationDomain,
    *,
    network: ReactionNetwork | None = None,
) -> NegativeSideRecoveryResult:
    """Prove ``x_i <= 0 => f_i(x) >= 0`` on the augmented domain.

    Every other state remains free over its augmented bounds, including its
    own possible negative excursion.  Strict attraction,
    ``x_i < 0 => f_i(x) > 0``, is attempted as an additional conclusion.
    Conservation rays and stoichiometric compatibility classes are
    intentionally not assumed.
    """

    network = network or build_network(case)
    eligible = tuple(
        species_id
        for species_id, symbol in case.symbols.concentrations.items()
        if domain.interval(symbol).lower < 0
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


def run(context: AnalysisContext, _dependencies: Mapping) -> Finding:
    domain = context.augmented_domain
    result = check_negative_side_recovery(
        context.case, domain, network=context.network
    )
    if not result.species:
        return Finding(
            context.case.name,
            Verdict.SKIPPED,
            "No concentration has a negative excursion.",
        )

    if result.nonrepelling is False:
        verdict = Verdict.FAIL
    elif result.nonrepelling is None:
        verdict = Verdict.UNKNOWN
    else:
        verdict = Verdict.PASS

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
    counterexamples = {
        item.species_id: {
            str(symbol): str(value)
            for symbol, value in item.nonrepulsion_counterexample.items()
        }
        for item in result.species
        if item.nonrepulsion_counterexample is not None
    }
    return Finding(
        context.case.name,
        verdict,
        " ".join(details),
        Evidence(
            "negative_side",
            {
                "species_checked": len(result.species),
                "counterexamples": counterexamples,
            },
        ),
    )
