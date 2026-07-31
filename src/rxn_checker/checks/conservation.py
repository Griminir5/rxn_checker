"""Atom and mass conservation checks for individual reactions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from ..reaction import Reaction
from ..species import PROPERTY_REGISTRY, PropertyRegistry

ATOM_RELATIVE_TOLERANCE = 1.0e-9
ATOM_ABSOLUTE_TOLERANCE = 1.0e-12
MASS_RELATIVE_TOLERANCE = 1.0e-4
MASS_ABSOLUTE_TOLERANCE = 1.0e-9


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


@dataclass(frozen=True)
class MassConservationResult:
    """Mass totals in kg/mol and their product-minus-reactant residual."""

    reaction_id: str
    passed: bool
    reactant_mass: float
    product_mass: float
    imbalance: float

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
    """Check that each element has equal reactant and product atom totals.

    Catalysts are excluded because a :class:`Reaction` defines them as
    non-consumed species. Residuals are signed as products minus reactants.
    """

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


def _molecular_weights(
    reaction_id: str,
    species_ids: tuple[str, ...],
    property_registry: PropertyRegistry,
) -> dict[str, float]:
    molecular_weights: dict[str, float] = {}
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
    side: Mapping[str, float],
    molecular_weights: Mapping[str, float],
) -> float:
    return math.fsum(
        coefficient * molecular_weights[species_id]
        for species_id, coefficient in side.items()
    )


def check_mass_conservation(
    reaction: Reaction,
    property_registry: PropertyRegistry = PROPERTY_REGISTRY,
    *,
    rel_tol: float = MASS_RELATIVE_TOLERANCE,
    abs_tol: float = MASS_ABSOLUTE_TOLERANCE,
) -> MassConservationResult:
    """Check equal stoichiometric mass totals on both sides of a reaction.

    Molecular weights are interpreted in the registry's documented kg/mol
    units. The default relative tolerance accommodates their tabulated
    rounding. A missing molecular weight makes the check unavailable and is
    reported as a :class:`ValueError` rather than as a failed balance.
    """

    _validate_tolerances(rel_tol, abs_tol)
    species_ids = tuple(dict.fromkeys((*reaction.reactants, *reaction.products)))
    molecular_weights = _molecular_weights(reaction.id, species_ids, property_registry)
    reactant_mass = _side_mass(reaction.reactants, molecular_weights)
    product_mass = _side_mass(reaction.products, molecular_weights)
    return MassConservationResult(
        reaction_id=reaction.id,
        passed=math.isclose(
            reactant_mass,
            product_mass,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ),
        reactant_mass=reactant_mass,
        product_mass=product_mass,
        imbalance=product_mass - reactant_mass,
    )


__all__ = (
    "AtomConservationResult",
    "MassConservationResult",
    "check_atom_conservation",
    "check_mass_conservation",
)
