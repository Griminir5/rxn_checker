"""Atom conservation check."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from ..case import Case
from ..reaction import Reaction
from ..species import PROPERTY_REGISTRY, PropertyRegistry
from .models import (
    CheckContext,
    CheckDefinition,
    CheckOutcome,
    CheckScope,
    CheckStatus,
    CheckValue,
)

ATOM_RELATIVE_TOLERANCE = 1.0e-9
ATOM_ABSOLUTE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class AtomConservationResult:
    """Element totals and product-minus-reactant residuals for one reaction."""

    reaction_id: str
    passed: bool
    reactant_totals: Mapping[str, float]
    product_totals: Mapping[str, float]
    imbalances: Mapping[str, float]

    def __bool__(self) -> bool:
        return self.passed


def _validate_tolerances(rel_tol: float, abs_tol: float) -> None:
    for name, tolerance in (("rel_tol", rel_tol), ("abs_tol", abs_tol)):
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"{name} must be finite and non-negative.")


def _element_totals(
    side: Mapping[str, float],
    property_registry: PropertyRegistry,
) -> dict[str, float]:
    contributions: dict[str, list[float]] = {}
    for species_id, coefficient in side.items():
        record = property_registry.get_record(species_id)
        for element, atom_count in record.atoms.items():
            contributions.setdefault(element, []).append(coefficient * atom_count)
    return {
        element: math.fsum(element_contributions)
        for element, element_contributions in contributions.items()
    }


def check_atom_conservation(
    reaction: Reaction,
    property_registry: PropertyRegistry = PROPERTY_REGISTRY,
    *,
    rel_tol: float = ATOM_RELATIVE_TOLERANCE,
    abs_tol: float = ATOM_ABSOLUTE_TOLERANCE,
) -> AtomConservationResult:
    """Check that each element has equal reactant and product atom totals."""

    _validate_tolerances(rel_tol, abs_tol)
    reactant_totals = _element_totals(reaction.reactants, property_registry)
    product_totals = _element_totals(reaction.products, property_registry)
    elements = tuple(dict.fromkeys((*reactant_totals, *product_totals)))
    imbalances = {
        element: product_totals.get(element, 0.0) - reactant_totals.get(element, 0.0)
        for element in elements
    }
    passed = all(
        math.isclose(
            reactant_totals.get(element, 0.0),
            product_totals.get(element, 0.0),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        for element in elements
    )
    return AtomConservationResult(
        reaction_id=reaction.id,
        passed=passed,
        reactant_totals=MappingProxyType(reactant_totals),
        product_totals=MappingProxyType(product_totals),
        imbalances=MappingProxyType(imbalances),
    )


def run(case: Case, context: CheckContext) -> tuple[CheckOutcome, ...]:
    """Run atom conservation for every reaction in a case."""

    if not case.reactions:
        return (
            CheckOutcome(
                status=CheckStatus.UNAVAILABLE,
                details=("Case has no reactions.",),
            ),
        )

    outcomes: list[CheckOutcome] = []
    for reaction in case.reactions:
        try:
            result = check_atom_conservation(reaction, context.property_registry)
        except (KeyError, ValueError) as error:
            message = (
                error.args[0]
                if len(error.args) == 1 and isinstance(error.args[0], str)
                else str(error)
            )
            outcomes.append(
                CheckOutcome(
                    status=CheckStatus.UNAVAILABLE,
                    subject=reaction.id,
                    details=(message,),
                )
            )
            continue

        values = tuple(
            CheckValue(f"{element} imbalance", imbalance)
            for element, imbalance in result.imbalances.items()
            if imbalance != 0
        )
        outcomes.append(
            CheckOutcome(
                status=(CheckStatus.PASS if result.passed else CheckStatus.FAIL),
                subject=reaction.id,
                details=("Products minus reactants.",) if values else (),
                values=values,
            )
        )
    return tuple(outcomes)


CHECK = CheckDefinition(
    id="atom_conservation",
    name="Atom conservation",
    group="Basic checks",
    scope=CheckScope.REACTION,
    run=run,
)


__all__ = (
    "AtomConservationResult",
    "CHECK",
    "check_atom_conservation",
    "run",
)
