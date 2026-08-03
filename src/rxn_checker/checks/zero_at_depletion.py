"""Symbolic zero-rate check at reactant and catalyst depletion boundaries."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

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
class ZeroAtDepletionResult:
    """Rates and symbolic conclusion at every required-species boundary."""

    reaction_id: str
    passed: bool | None
    rates_at_depletion: Mapping[str, sp.Expr]


def _required_species(reaction: Reaction) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*reaction.reactants, *reaction.catalysts)))


def _rate_at_depletion(rate: sp.Expr, species_id: str) -> sp.Expr:
    """Substitute zero for concentration symbols belonging to one species."""

    replacements = {
        symbol: sp.S.Zero
        for symbol in rate.free_symbols
        if symbol.name == species_id
    }
    return sp.simplify(rate.xreplace(replacements))


def _is_exactly_zero(expression: sp.Expr) -> bool | None:
    if expression.is_zero is not None:
        return expression.is_zero
    return expression.equals(sp.S.Zero)


def check_zero_at_depletion(reaction: Reaction) -> ZeroAtDepletionResult:
    """Check that each reactant or catalyst independently stops the reaction.

    Each required species concentration is set to exactly zero while the other
    state variables remain symbolic. Products that are not also reactants are
    deliberately excluded.
    """

    rates_at_depletion = {
        species_id: _rate_at_depletion(reaction.rate, species_id)
        for species_id in _required_species(reaction)
    }
    conclusions = tuple(
        _is_exactly_zero(rate) for rate in rates_at_depletion.values()
    )
    if any(conclusion is False for conclusion in conclusions):
        passed: bool | None = False
    elif any(conclusion is None for conclusion in conclusions):
        passed = None
    else:
        passed = True

    return ZeroAtDepletionResult(
        reaction_id=reaction.id,
        passed=passed,
        rates_at_depletion=MappingProxyType(rates_at_depletion),
    )


def _outcome(result: ZeroAtDepletionResult) -> CheckOutcome:
    failed = tuple(
        (species_id, rate)
        for species_id, rate in result.rates_at_depletion.items()
        if _is_exactly_zero(rate) is False
    )
    indeterminate = tuple(
        (species_id, rate)
        for species_id, rate in result.rates_at_depletion.items()
        if _is_exactly_zero(rate) is None
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
        return CheckOutcome(
            status=CheckStatus.FAIL,
            subject=result.reaction_id,
            details=details,
        )

    if indeterminate:
        return CheckOutcome(
            status=CheckStatus.INDETERMINATE,
            subject=result.reaction_id,
            details=tuple(
                f"Could not prove that the rate at {species_id}=0 is zero: {rate}."
                for species_id, rate in indeterminate
            ),
        )

    if not result.rates_at_depletion:
        details = ("Reaction has no reactants or catalysts.",)
    else:
        details = (
            "Rate is exactly zero at every reactant and catalyst depletion "
            "boundary.",
        )
    return CheckOutcome(
        status=CheckStatus.PASS,
        subject=result.reaction_id,
        details=details,
    )


def run(case: Case, context: CheckContext) -> tuple[CheckOutcome, ...]:
    """Run the zero-at-depletion check for every reaction in a case."""

    return tuple(
        _outcome(check_zero_at_depletion(reaction)) for reaction in case.reactions
    )


CHECK = CheckDefinition(
    id="zero_at_depletion",
    name="Zero rate at depletion",
    group="Physical checks",
    scope=CheckScope.REACTION,
    run=run,
)
