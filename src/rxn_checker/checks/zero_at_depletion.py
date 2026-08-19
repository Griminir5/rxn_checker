"""Symbolic zero-rate check at reactant and catalyst depletion boundaries."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import sympy as sp

from ..context import AnalysisContext
from ..domain import ConcentrationDomain
from ..model import CaseSymbols, Reaction
from ..proof.analysis import Point
from ..results import Evidence, Finding, Verdict
from .prerequisites import reaction_skip


@dataclass(frozen=True)
class ZeroAtDepletionResult:
    """Rates and symbolic conclusion at every required-species boundary."""

    reaction_id: str
    passed: bool | None
    rates_at_depletion: Mapping[str, sp.Expr]
    conclusions: Mapping[str, bool | None]
    counterexamples: Mapping[str, Point]


def _required_species(reaction: Reaction) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*reaction.reactants, *reaction.catalysts)))


def _rate_at_depletion(
    rate: sp.Expr,
    species_id: str,
    symbols: CaseSymbols,
) -> sp.Expr:
    """Set the case-owned concentration coordinate to exactly zero."""

    symbol = symbols.concentration(species_id)
    return rate.subs({symbol: sp.S.Zero}, simultaneous=True)


def _is_exactly_zero(expression: sp.Expr) -> bool | None:
    if expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        return False
    if expression.is_zero is not None:
        return expression.is_zero
    numerator, denominator = expression.as_numer_denom()
    if denominator.is_zero is True:
        return False
    if numerator.is_zero is True and denominator.is_zero is False:
        return True
    if sp.count_ops(expression) <= 64:
        factored = sp.factor_terms(expression)
        if factored != expression and factored.is_zero is not None:
            return factored.is_zero
    return None


def _nonzero_witness(
    expression: sp.Expr,
    depleted_symbol: sp.Symbol,
    domain: ConcentrationDomain,
) -> Point | None:
    """Find a small exact physical point disproving a zero identity."""

    face = domain.restrict(depleted_symbol, lower=0, upper=0)
    candidates = [face.exact_witness()]
    for symbol in sorted(expression.free_symbols, key=lambda item: item.name):
        if symbol in face.all_intervals:
            candidates.append(
                face.exact_witness({symbol: face.interval(symbol).upper})
            )
    for point in candidates:
        if point is None:
            continue
        value = expression.subs(point, simultaneous=True)
        if (
            value.is_real is True
            and value.is_finite is True
            and value.is_zero is False
        ):
            return point
    return None


def check_zero_at_depletion(
    reaction: Reaction,
    symbols: CaseSymbols,
    domain: ConcentrationDomain | None = None,
) -> ZeroAtDepletionResult:
    """Check that each reactant or catalyst independently stops the reaction.

    Each required species concentration is set to exactly zero while the other
    state variables remain symbolic. Products that are not also reactants are
    deliberately excluded.
    """

    rates_at_depletion = {
        species_id: _rate_at_depletion(reaction.rate, species_id, symbols)
        for species_id in _required_species(reaction)
    }
    conclusions = {
        species_id: _is_exactly_zero(rate)
        for species_id, rate in rates_at_depletion.items()
    }
    counterexamples = {}
    if domain is not None:
        for species_id, conclusion in tuple(conclusions.items()):
            if conclusion is not None:
                continue
            point = _nonzero_witness(
                rates_at_depletion[species_id],
                symbols.concentration(species_id),
                domain,
            )
            if point is not None:
                conclusions[species_id] = False
                counterexamples[species_id] = point

    if any(conclusion is False for conclusion in conclusions.values()):
        passed: bool | None = False
    elif any(conclusion is None for conclusion in conclusions.values()):
        passed = None
    else:
        passed = True

    return ZeroAtDepletionResult(
        reaction_id=reaction.id,
        passed=passed,
        rates_at_depletion=MappingProxyType(rates_at_depletion),
        conclusions=MappingProxyType(conclusions),
        counterexamples=MappingProxyType(counterexamples),
    )


def _finding(result: ZeroAtDepletionResult) -> Finding:
    failed = tuple(
        (species_id, rate)
        for species_id, rate in result.rates_at_depletion.items()
        if result.conclusions[species_id] is False
    )
    indeterminate = tuple(
        (species_id, rate)
        for species_id, rate in result.rates_at_depletion.items()
        if result.conclusions[species_id] is None
    )

    if failed:
        details = tuple(
            f"Rate at {species_id}=0 is {rate}, not zero."
            for species_id, rate in failed
        )
        details += tuple(
            f"Could not prove that the rate at {species_id}=0 is zero: {rate}."
            for species_id, rate in indeterminate
        )
        return Finding(
            result.reaction_id,
            Verdict.FAIL,
            " ".join(details),
            Evidence(
                "depletion_rates",
                {
                    species_id: {
                        "rate": str(rate),
                        "point": {
                            str(symbol): str(value)
                            for symbol, value in result.counterexamples.get(
                                species_id, {}
                            ).items()
                        },
                    }
                    for species_id, rate in failed
                },
            ),
        )

    if indeterminate:
        return Finding(
            result.reaction_id,
            Verdict.UNKNOWN,
            " ".join(
                f"Could not prove that the rate at {species_id}=0 is zero: {rate}."
                for species_id, rate in indeterminate
            ),
        )

    if not result.rates_at_depletion:
        summary = "Reaction has no reactants or catalysts."
    else:
        summary = (
            "Rate is exactly zero at every required depletion boundary."
        )
    return Finding(result.reaction_id, Verdict.PASS, summary)


def run(context: AnalysisContext, dependencies: Mapping) -> tuple[Finding, ...]:
    """Run the zero-at-depletion check for every reaction in a case."""

    return tuple(
        reaction_skip(dependencies, "physical_rate_definedness", reaction.id)
        or _finding(
            check_zero_at_depletion(
                reaction,
                context.case.symbols,
                context.physical_domain,
            )
        )
        for reaction in context.case.reactions
    )
