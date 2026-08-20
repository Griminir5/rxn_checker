"""Exact atom conservation."""

from collections.abc import Mapping
from dataclasses import dataclass

import sympy as sp

from ..context import AnalysisContext
from ..model import Reaction, Species
from ..results import Evidence, Finding, Verdict


@dataclass(frozen=True)
class AtomConservationResult:
    reaction_id: str
    passed: bool
    reactant_totals: Mapping[str, sp.Expr]
    product_totals: Mapping[str, sp.Expr]
    imbalances: Mapping[str, sp.Expr]


def _totals(side, species_by_id):
    totals = {}
    for species_id, coefficient in side.items():
        for element, count in species_by_id[species_id].atoms.items():
            totals[element] = totals.get(element, sp.S.Zero) + coefficient * count
    return totals


def check_atom_conservation(reaction: Reaction,
                            species_by_id: Mapping[str, Species]) -> AtomConservationResult:
    reactants = _totals(reaction.reactants, species_by_id)
    products = _totals(reaction.products, species_by_id)
    imbalances = {element: products.get(element, 0) - reactants.get(element, 0)
                  for element in dict.fromkeys((*reactants, *products))}
    return AtomConservationResult(reaction.id, not any(imbalances.values()),
                                  reactants, products, imbalances)


def run(context: AnalysisContext, _dependencies: Mapping) -> tuple[Finding, ...]:
    findings = []
    for reaction in context.case.reactions:
        result = check_atom_conservation(reaction, context.species_by_id)
        imbalance = {key: value for key, value in result.imbalances.items() if value}
        findings.append(Finding(reaction.id,
            Verdict.PASS if result.passed else Verdict.FAIL,
            "All element totals balance." if result.passed else
            "Products and reactants have different element totals.",
            Evidence("atom_imbalance", imbalance) if imbalance else None))
    return tuple(findings)
