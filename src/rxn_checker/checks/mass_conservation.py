"""Exact-tolerance mass conservation."""

from collections.abc import Mapping
from dataclasses import dataclass

import sympy as sp

from ..context import AnalysisContext
from ..model import RationalInput, Reaction, Species, parse_rational
from ..results import Evidence, Finding, Verdict

MASS_RELATIVE_TOLERANCE = sp.Rational(1, 10**9)
MASS_ABSOLUTE_TOLERANCE = sp.Rational(1, 10**12)


@dataclass(frozen=True)
class MassConservationResult:
    reaction_id: str
    passed: bool
    reactant_mass: sp.Expr
    product_mass: sp.Expr
    imbalance: sp.Expr


def check_mass_conservation(reaction: Reaction, species_by_id: Mapping[str, Species], *,
                            rel_tol: RationalInput = MASS_RELATIVE_TOLERANCE,
                            abs_tol: RationalInput = MASS_ABSOLUTE_TOLERANCE
                            ) -> MassConservationResult:
    relative, absolute = parse_rational(rel_tol), parse_rational(abs_tol)
    if relative < 0 or absolute < 0:
        raise ValueError("Mass tolerances must be non-negative.")
    ids = dict.fromkeys((*reaction.reactants, *reaction.products))
    masses = {item: species_by_id[item].molar_mass for item in ids}
    missing = [item for item, mass in masses.items() if mass is None]
    if missing:
        raise ValueError(f"Cannot check mass conservation for reaction '{reaction.id}'; "
                         "molar mass is missing for species: " + ", ".join(missing) + ".")
    total = lambda side: sum((coefficient * masses[item]
                              for item, coefficient in side.items()), sp.S.Zero)
    reactants, products = total(reaction.reactants), total(reaction.products)
    imbalance = products - reactants
    allowed = max(absolute, relative * max(abs(reactants), abs(products)))
    return MassConservationResult(reaction.id, bool(abs(imbalance) <= allowed),
                                  reactants, products, imbalance)


def run(context: AnalysisContext, _dependencies: Mapping) -> tuple[Finding, ...]:
    findings = []
    for reaction in context.case.reactions:
        try:
            result = check_mass_conservation(reaction, context.species_by_id)
        except ValueError as error:
            findings.append(Finding(reaction.id, Verdict.FAIL, str(error)))
            continue
        details = None if result.passed else Evidence("mass_imbalance", {
            "reactant_mass_kg_per_mol": result.reactant_mass,
            "product_mass_kg_per_mol": result.product_mass,
            "imbalance_kg_per_mol": result.imbalance})
        findings.append(Finding(reaction.id,
            Verdict.PASS if result.passed else Verdict.FAIL,
            "Stoichiometric masses balance." if result.passed else
            "Products and reactants have different masses.", details))
    return tuple(findings)
