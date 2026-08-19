"""Symbolic rate non-negativity check."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import sympy as sp

from ..context import AnalysisContext
from ..domain import ConcentrationDomain
from ..model import Reaction
from ..results import Finding, Verdict


@dataclass(frozen=True)
class RateNonnegativityResult:
    """Symbolic conclusion for one reaction rate."""

    reaction_id: str
    passed: bool | None
    rate: sp.Expr


def _rate_with_bound_assumptions(
    rate: sp.Expr,
    domain: ConcentrationDomain,
    *,
    interior: bool = False,
) -> sp.Expr:
    """Apply sign assumptions implied by each state's physical lower bound."""

    replacements = {}
    for symbol in rate.free_symbols:
        interval = domain.interval(symbol)
        lower = interval.lower
        if lower > 0 or not interval.lower_closed or (interior and lower == 0):
            assumptions = {"positive": True}
        elif lower == 0:
            assumptions = {"nonnegative": True}
        else:
            assumptions = {"real": True}
        replacements[symbol] = sp.Dummy(symbol.name, **assumptions)
    return rate.xreplace(replacements)


def _sign_candidates(rate: sp.Expr) -> Iterator[sp.Expr]:
    """Yield increasingly expensive rewrites only when the caller needs them."""

    yield rate
    yield sp.factor_terms(rate)
    yield sp.factor(rate)


def check_rate_nonnegativity(
    reaction: Reaction,
    domain: ConcentrationDomain,
) -> RateNonnegativityResult:
    """Symbolically check a rate using signs implied by its physical bounds.

    A negative conclusion is made only when SymPy proves the rate is negative
    in the physical interior. Upper bounds are not yet used, so other unresolved
    expressions remain indeterminate until bounded interval analysis is added.
    """

    physical_rate = _rate_with_bound_assumptions(reaction.rate, domain)
    if any(
        candidate.is_nonnegative is True
        for candidate in _sign_candidates(physical_rate)
    ):
        return RateNonnegativityResult(
            reaction_id=reaction.id,
            passed=True,
            rate=reaction.rate,
        )

    interior_rate = _rate_with_bound_assumptions(
        reaction.rate,
        domain,
        interior=True,
    )
    if any(
        candidate.is_negative is True for candidate in _sign_candidates(interior_rate)
    ):
        return RateNonnegativityResult(
            reaction_id=reaction.id,
            passed=False,
            rate=reaction.rate,
        )

    return RateNonnegativityResult(
        reaction_id=reaction.id,
        passed=None,
        rate=reaction.rate,
    )


def _finding(result: RateNonnegativityResult) -> Finding:
    if result.passed is False:
        return Finding(
            result.reaction_id,
            Verdict.FAIL,
            "Rate is symbolically negative in the physical interior.",
        )
    if result.passed:
        return Finding(
            result.reaction_id,
            Verdict.PASS,
            "Rate is symbolically non-negative under physical bounds.",
        )
    return Finding(
        result.reaction_id,
        Verdict.UNKNOWN,
        "Symbolic sign analysis was inconclusive.",
    )


def run(context: AnalysisContext, _dependencies: Mapping) -> tuple[Finding, ...]:
    """Run rate non-negativity for every reaction in a case."""

    findings: list[Finding] = []
    domain = context.physical_domain
    for reaction in context.case.reactions:
        try:
            result = check_rate_nonnegativity(reaction, domain)
        except ValueError as error:
            findings.append(
                Finding(reaction.id, Verdict.SKIPPED, str(error.args[0]))
            )
            continue
        findings.append(_finding(result))
    return tuple(findings)
