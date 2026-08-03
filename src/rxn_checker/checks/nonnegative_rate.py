"""Symbolic rate non-negativity check."""

from dataclasses import dataclass

import sympy as sp

from ..case import Case
from ..reaction import Reaction
from .models import (
    CheckContext,
    CheckDefinition,
    CheckOutcome,
    CheckScope,
    CheckStatus,
)


@dataclass(frozen=True)
class RateNonnegativityResult:
    """Symbolic conclusion for one reaction rate."""

    reaction_id: str
    passed: bool | None
    rate: sp.Expr


def _rate_with_assumptions(rate: sp.Expr, **assumptions: bool) -> sp.Expr:
    """Return an equivalent expression with assumptions on its free symbols."""

    replacements = {
        symbol: sp.Dummy(symbol.name, **assumptions)
        for symbol in rate.free_symbols
    }
    return rate.xreplace(replacements)


def _sign_candidates(rate: sp.Expr) -> tuple[sp.Expr, ...]:
    return rate, sp.factor_terms(rate), sp.factor(rate)


def check_rate_nonnegativity(
    reaction: Reaction,
) -> RateNonnegativityResult:
    """Symbolically check a rate over non-negative state variables.

    A negative conclusion is made only when SymPy proves the rate is negative
    with every state variable in the positive interior. All other unresolved
    expressions remain indeterminate until bounded interval analysis is added.
    """

    physical_rate = _rate_with_assumptions(reaction.rate, nonnegative=True)
    if any(
        candidate.is_nonnegative is True
        for candidate in _sign_candidates(physical_rate)
    ):
        return RateNonnegativityResult(
            reaction_id=reaction.id,
            passed=True,
            rate=reaction.rate,
        )

    interior_rate = _rate_with_assumptions(reaction.rate, positive=True)
    if any(
        candidate.is_negative is True
        for candidate in _sign_candidates(interior_rate)
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


def _outcome(result: RateNonnegativityResult) -> CheckOutcome:
    if result.passed is False:
        return CheckOutcome(
            status=CheckStatus.FAIL,
            subject=result.reaction_id,
            details=(
                "Rate is symbolically negative in the positive physical interior.",
            ),
        )
    if result.passed:
        return CheckOutcome(
            status=CheckStatus.PASS,
            subject=result.reaction_id,
            details=(
                "Rate is symbolically non-negative for non-negative state variables.",
            ),
        )
    return CheckOutcome(
        status=CheckStatus.INDETERMINATE,
        subject=result.reaction_id,
        details=(
            "Symbolic sign analysis was inconclusive; bounded interval analysis "
            "is not yet available.",
        ),
    )


def run(case: Case, context: CheckContext) -> tuple[CheckOutcome, ...]:
    """Run rate non-negativity for every reaction in a case."""

    outcomes: list[CheckOutcome] = []
    for reaction in case.reactions:
        try:
            result = check_rate_nonnegativity(reaction)
        except ValueError as error:
            outcomes.append(
                CheckOutcome(
                    status=CheckStatus.UNAVAILABLE,
                    subject=reaction.id,
                    details=(str(error.args[0]),),
                )
            )
            continue
        outcomes.append(_outcome(result))
    return tuple(outcomes)


CHECK = CheckDefinition(
    id="rate_nonnegativity",
    name="Rate non-negativity",
    group="Physical checks",
    scope=CheckScope.REACTION,
    run=run,
)
