"""Atom conservation check."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from ..context import AnalysisContext
from ..model import Reaction
from ..results import Evidence, Finding, Verdict
from ..species import PROPERTY_REGISTRY, PropertyRegistry

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


def run(context: AnalysisContext, _dependencies: Mapping) -> tuple[Finding, ...]:
    """Run atom conservation for every reaction in a case."""

    findings: list[Finding] = []
    for reaction in context.case.reactions:
        try:
            result = check_atom_conservation(reaction, context.property_registry)
        except (KeyError, ValueError) as error:
            findings.append(
                Finding(reaction.id, Verdict.SKIPPED, str(error.args[0]))
            )
            continue

        imbalances = {
            element: imbalance
            for element, imbalance in result.imbalances.items()
            if imbalance != 0
        }
        findings.append(
            Finding(
                reaction.id,
                Verdict.PASS if result.passed else Verdict.FAIL,
                "All element totals balance."
                if result.passed
                else "Products and reactants have different element totals.",
                Evidence("atom_imbalance", imbalances) if imbalances else None,
            )
        )
    return tuple(findings)
