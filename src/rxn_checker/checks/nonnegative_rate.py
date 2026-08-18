"""Symbolic rate non-negativity check."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import sympy as sp

from ..case import Case
from ..reaction import Reaction
from ..state import VariableBounds
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


def _rate_with_bound_assumptions(
    rate: sp.Expr,
    state_bounds: Mapping[sp.Symbol, VariableBounds],
    *,
    interior: bool = False,
) -> sp.Expr:
    """Apply sign assumptions implied by each state's physical lower bound."""

    replacements = {}
    for symbol in rate.free_symbols:
        bounds = state_bounds[symbol]
        lower = bounds.physical_lower
        if lower > 0 or bounds.strict_lower or (interior and lower == 0):
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
    state_bounds: Mapping[sp.Symbol, VariableBounds],
) -> RateNonnegativityResult:
    """Symbolically check a rate using signs implied by its physical bounds.

    A negative conclusion is made only when SymPy proves the rate is negative
    in the physical interior. Upper bounds are not yet used, so other unresolved
    expressions remain indeterminate until bounded interval analysis is added.
    """

    physical_rate = _rate_with_bound_assumptions(reaction.rate, state_bounds)
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
        state_bounds,
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


def _outcome(result: RateNonnegativityResult) -> CheckOutcome:
    if result.passed is False:
        return CheckOutcome(
            status=CheckStatus.FAIL,
            subject=result.reaction_id,
            details=("Rate is symbolically negative in the physical interior.",),
        )
    if result.passed:
        return CheckOutcome(
            status=CheckStatus.PASS,
            subject=result.reaction_id,
            details=(
                "Rate is symbolically non-negative under its physical "
                "lower-bound assumptions.",
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
            result = check_rate_nonnegativity(reaction, case.state_bounds)
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
