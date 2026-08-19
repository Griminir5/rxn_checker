"""Mass conservation check."""

from collections.abc import Mapping
from dataclasses import dataclass

import sympy as sp

from ..context import AnalysisContext
from ..model import RationalInput, Reaction, parse_rational
from ..results import Evidence, Finding, Verdict
from ..species import PROPERTY_REGISTRY, PropertyRegistry

MASS_RELATIVE_TOLERANCE = sp.Rational(1, 10**9)
MASS_ABSOLUTE_TOLERANCE = sp.Rational(1, 10**12)


@dataclass(frozen=True)
class MassConservationResult:
    """Mass totals in kg/mol and their product-minus-reactant residual."""

    reaction_id: str
    passed: bool
    reactant_mass: sp.Expr
    product_mass: sp.Expr
    imbalance: sp.Expr


def _tolerance(value: RationalInput, label: str) -> sp.Rational:
    tolerance = parse_rational(value, label=label)
    if tolerance < 0:
        raise ValueError(f"{label} must be non-negative.")
    return tolerance


def _molecular_weights(
    reaction_id: str,
    species_ids: tuple[str, ...],
    property_registry: PropertyRegistry,
) -> dict[str, sp.Expr]:
    molecular_weights: dict[str, sp.Expr] = {}
    missing: list[str] = []
    for species_id in species_ids:
        molecular_weight = property_registry.get_record(species_id).mw
        if molecular_weight is None:
            missing.append(species_id)
        else:
            molecular_weights[species_id] = molecular_weight
    if missing:
        raise ValueError(
            f"Cannot check mass conservation for reaction '{reaction_id}'; "
            "molecular weight is missing for species: " + ", ".join(missing) + "."
        )
    return molecular_weights


def _side_mass(
    side: Mapping[str, sp.Expr],
    molecular_weights: Mapping[str, sp.Expr],
) -> sp.Expr:
    return sum(
        (
            coefficient * molecular_weights[species_id]
            for species_id, coefficient in side.items()
        ),
        sp.S.Zero,
    )


def check_mass_conservation(
    reaction: Reaction,
    property_registry: PropertyRegistry = PROPERTY_REGISTRY,
    *,
    rel_tol: RationalInput = MASS_RELATIVE_TOLERANCE,
    abs_tol: RationalInput = MASS_ABSOLUTE_TOLERANCE,
) -> MassConservationResult:
    """Check equal stoichiometric mass totals on both reaction sides."""

    relative_tolerance = _tolerance(rel_tol, "rel_tol")
    absolute_tolerance = _tolerance(abs_tol, "abs_tol")
    species_ids = tuple(dict.fromkeys((*reaction.reactants, *reaction.products)))
    molecular_weights = _molecular_weights(reaction.id, species_ids, property_registry)
    reactant_mass = _side_mass(reaction.reactants, molecular_weights)
    product_mass = _side_mass(reaction.products, molecular_weights)
    imbalance = product_mass - reactant_mass
    allowed = max(
        absolute_tolerance,
        relative_tolerance * max(abs(reactant_mass), abs(product_mass)),
    )
    return MassConservationResult(
        reaction_id=reaction.id,
        passed=bool(abs(imbalance) <= allowed),
        reactant_mass=reactant_mass,
        product_mass=product_mass,
        imbalance=imbalance,
    )


def run(context: AnalysisContext, _dependencies: Mapping) -> tuple[Finding, ...]:
    """Run mass conservation for every reaction in a case."""

    findings: list[Finding] = []
    for reaction in context.case.reactions:
        try:
            result = check_mass_conservation(reaction, context.property_registry)
        except (KeyError, ValueError) as error:
            findings.append(
                Finding(reaction.id, Verdict.FAIL, str(error.args[0]))
            )
            continue

        evidence = None
        if not result.passed:
            evidence = Evidence(
                "mass_imbalance",
                {
                    "reactant_mass_kg_per_mol": result.reactant_mass,
                    "product_mass_kg_per_mol": result.product_mass,
                    "imbalance_kg_per_mol": result.imbalance,
                },
            )
        findings.append(
            Finding(
                reaction.id,
                Verdict.PASS if result.passed else Verdict.FAIL,
                "Stoichiometric masses balance."
                if result.passed
                else "Products and reactants have different masses.",
                evidence,
            )
        )
    return tuple(findings)
