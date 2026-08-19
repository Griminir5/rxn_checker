"""Atom conservation check."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import sympy as sp

from ..context import AnalysisContext
from ..model import Reaction
from ..results import Evidence, Finding, Verdict
from ..species import PROPERTY_REGISTRY, PropertyRegistry


@dataclass(frozen=True)
class AtomConservationResult:
    """Element totals and product-minus-reactant residuals for one reaction."""

    reaction_id: str
    passed: bool
    reactant_totals: Mapping[str, sp.Expr]
    product_totals: Mapping[str, sp.Expr]
    imbalances: Mapping[str, sp.Expr]


def _element_totals(
    side: Mapping[str, sp.Expr],
    property_registry: PropertyRegistry,
) -> dict[str, sp.Expr]:
    totals: dict[str, sp.Expr] = {}
    for species_id, coefficient in side.items():
        record = property_registry.get_record(species_id)
        for element, atom_count in record.atoms.items():
            totals[element] = totals.get(element, sp.S.Zero) + coefficient * atom_count
    return totals


def check_atom_conservation(
    reaction: Reaction,
    property_registry: PropertyRegistry = PROPERTY_REGISTRY,
) -> AtomConservationResult:
    """Check exact equality of reactant and product element totals."""

    reactant_totals = _element_totals(reaction.reactants, property_registry)
    product_totals = _element_totals(reaction.products, property_registry)
    elements = tuple(dict.fromkeys((*reactant_totals, *product_totals)))
    imbalances = {
        element: product_totals.get(element, sp.S.Zero)
        - reactant_totals.get(element, sp.S.Zero)
        for element in elements
    }
    passed = all(imbalance == 0 for imbalance in imbalances.values())
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
                Finding(reaction.id, Verdict.FAIL, str(error.args[0]))
            )
            continue

        imbalances = {key: value for key, value in result.imbalances.items() if value}
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
