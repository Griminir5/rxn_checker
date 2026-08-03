"""Mass conservation check."""

from collections.abc import Mapping
from dataclasses import dataclass
import math

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

MASS_RELATIVE_TOLERANCE = 1.0e-9
MASS_ABSOLUTE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class MassConservationResult:
    """Mass totals in kg/mol and their product-minus-reactant residual."""

    reaction_id: str
    passed: bool
    reactant_mass: float
    product_mass: float
    imbalance: float


def _validate_tolerances(rel_tol: float, abs_tol: float) -> None:
    for name, tolerance in (("rel_tol", rel_tol), ("abs_tol", abs_tol)):
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"{name} must be finite and non-negative.")


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
    """Check equal stoichiometric mass totals on both reaction sides."""

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


def run(case: Case, context: CheckContext) -> tuple[CheckOutcome, ...]:
    """Run mass conservation for every reaction in a case."""

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
            result = check_mass_conservation(reaction, context.property_registry)
        except (KeyError, ValueError) as error:
            outcomes.append(
                CheckOutcome(
                    status=CheckStatus.UNAVAILABLE,
                    subject=reaction.id,
                    details=(str(error.args[0]),),
                )
            )
            continue

        values = ()
        if not result.passed:
            values = (
                CheckValue("Reactant mass", result.reactant_mass, "kg/mol"),
                CheckValue("Product mass", result.product_mass, "kg/mol"),
                CheckValue("Mass imbalance", result.imbalance, "kg/mol"),
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
    id="mass_conservation",
    name="Mass conservation",
    group="Basic checks",
    scope=CheckScope.REACTION,
    run=run,
)
