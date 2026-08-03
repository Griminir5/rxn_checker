"""Symbolic positivity check for the complete reaction network."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import sympy as sp

from ..case import Case
from .models import (
    CheckContext,
    CheckDefinition,
    CheckOutcome,
    CheckScope,
    CheckStatus,
)


@dataclass(frozen=True)
class NetworkPositivityResult:
    """Source terms and conclusions on every concentration boundary face."""

    passed: bool | None
    source_terms: Mapping[str, sp.Expr]
    boundary_sources: Mapping[str, sp.Expr]
    conclusions: Mapping[str, bool | None]


def _source_terms(case: Case) -> dict[str, sp.Expr]:
    """Construct F = S r in the ordering of the case species."""

    return {
        species_id: sp.simplify(
            sp.Add(
                *(
                    sp.sympify(reaction.net_stoichiometry.get(species_id, 0))
                    * reaction.rate
                    for reaction in case.reactions
                )
            )
        )
        for species_id in case.states.species_ids
    }


def _with_sign_assumptions(
    expression: sp.Expr,
    case: Case,
    *,
    interior: bool,
) -> sp.Expr:
    """Replace state symbols with equivalents carrying physical sign facts."""

    concentration_symbols = frozenset(case.states.concentrations.values())
    replacements: dict[sp.Symbol, sp.Symbol] = {}
    for symbol in expression.free_symbols:
        if symbol in concentration_symbols:
            assumptions = {"positive": True} if interior else {"nonnegative": True}
        else:
            lower = case.state_bounds[symbol].physical_lower
            if lower > 0 or (interior and lower == 0):
                assumptions = {"positive": True}
            elif lower == 0:
                assumptions = {"nonnegative": True}
            else:
                assumptions = {"real": True}
        replacements[symbol] = sp.Dummy(symbol.name, **assumptions)
    return expression.xreplace(replacements)


def _sign_candidates(expression: sp.Expr) -> tuple[sp.Expr, ...]:
    return expression, sp.factor_terms(expression), sp.factor(expression)


def _boundary_conclusion(expression: sp.Expr, case: Case) -> bool | None:
    physical_expression = _with_sign_assumptions(
        expression,
        case,
        interior=False,
    )
    if any(
        candidate.is_nonnegative is True
        for candidate in _sign_candidates(physical_expression)
    ):
        return True

    interior_expression = _with_sign_assumptions(
        expression,
        case,
        interior=True,
    )
    if any(
        candidate.is_negative is True
        for candidate in _sign_candidates(interior_expression)
    ):
        return False
    return None


def check_network_positivity(case: Case) -> NetworkPositivityResult:
    """Check that ``F_i >= 0`` whenever the concentration ``c_i`` is zero.

    The source vector is assembled from all reactions before each boundary is
    examined. Other concentrations are assumed non-negative, while temperature
    and pressure use the signs implied by their physical lower bounds.
    """

    source_terms = _source_terms(case)
    boundary_sources = {
        species_id: sp.simplify(
            source_terms[species_id].subs(
                case.states.concentration(species_id),
                sp.S.Zero,
            )
        )
        for species_id in case.states.species_ids
    }
    conclusions = {
        species_id: _boundary_conclusion(expression, case)
        for species_id, expression in boundary_sources.items()
    }

    if any(conclusion is False for conclusion in conclusions.values()):
        passed: bool | None = False
    elif any(conclusion is None for conclusion in conclusions.values()):
        passed = None
    else:
        passed = True

    return NetworkPositivityResult(
        passed=passed,
        source_terms=MappingProxyType(source_terms),
        boundary_sources=MappingProxyType(boundary_sources),
        conclusions=MappingProxyType(conclusions),
    )


def _outcome(result: NetworkPositivityResult) -> CheckOutcome:
    failed = tuple(
        (species_id, result.boundary_sources[species_id])
        for species_id, conclusion in result.conclusions.items()
        if conclusion is False
    )
    indeterminate = tuple(
        (species_id, result.boundary_sources[species_id])
        for species_id, conclusion in result.conclusions.items()
        if conclusion is None
    )

    if failed:
        details = tuple(
            f"At {species_id}=0, its source term {source} is symbolically "
            "negative in the physical interior."
            for species_id, source in failed
        )
        details += tuple(
            f"Could not prove that the source term at {species_id}=0 is "
            f"non-negative: {source}."
            for species_id, source in indeterminate
        )
        return CheckOutcome(status=CheckStatus.FAIL, details=details)

    if indeterminate:
        return CheckOutcome(
            status=CheckStatus.INDETERMINATE,
            details=tuple(
                f"Could not prove that the source term at {species_id}=0 is "
                f"non-negative: {source}."
                for species_id, source in indeterminate
            ),
        )

    return CheckOutcome(
        status=CheckStatus.PASS,
        details=(
            "Every species source term is non-negative on its zero-"
            "concentration boundary.",
        ),
    )


def run(case: Case, context: CheckContext) -> CheckOutcome:
    """Run the positivity check once for the complete case."""

    return _outcome(check_network_positivity(case))


CHECK = CheckDefinition(
    id="network_positivity",
    name="Network positivity",
    group="Physical checks",
    scope=CheckScope.CASE,
    run=run,
)
