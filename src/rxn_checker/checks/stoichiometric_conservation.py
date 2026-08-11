"""Exact conserved quantities for a complete reaction network."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import math
from types import MappingProxyType

import sympy as sp

from ..case import Case
from .models import CheckContext, CheckDefinition, CheckOutcome, CheckScope, CheckValue


@dataclass(frozen=True)
class ConservedQuantity:
    """One primitive-integer linear combination of species concentrations."""

    coefficients: Mapping[str, int]


@dataclass(frozen=True)
class ConservationComponent:
    """Conservation results for one connected part of the reaction network."""

    species_ids: tuple[str, ...]
    reaction_ids: tuple[str, ...]
    basis: tuple[ConservedQuantity, ...]
    extreme_rays: tuple[ConservedQuantity, ...]
    signed_basis: tuple[ConservedQuantity, ...]


@dataclass(frozen=True)
class StoichiometricConservationResult:
    """The exact left-nullspace analysis of a case's stoichiometric matrix."""

    species_ids: tuple[str, ...]
    reaction_ids: tuple[str, ...]
    stoichiometric_matrix: sp.ImmutableMatrix
    rank: int
    dimension: int
    unchanged_species: tuple[str, ...]
    components: tuple[ConservationComponent, ...]


def _rational(value: object) -> sp.Rational:
    """Interpret a coefficient's decimal spelling exactly."""

    return sp.Rational(str(value))


def _stoichiometric_matrix(case: Case) -> sp.ImmutableMatrix:
    """Build S with species as rows and reactions as columns."""

    return sp.ImmutableMatrix(
        [
            [
                _rational(reaction.products.get(species_id, 0))
                - _rational(reaction.reactants.get(species_id, 0))
                for reaction in case.reactions
            ]
            for species_id in case.states.species_ids
        ]
    )


def _primitive_integers(vector: Sequence[sp.Expr]) -> tuple[int, ...]:
    """Clear denominators, common factors, and arbitrary overall sign."""

    rationals = tuple(sp.Rational(value) for value in vector)
    common_denominator = math.lcm(*(int(value.q) for value in rationals))
    integers = tuple(int(value * common_denominator) for value in rationals)
    common_factor = math.gcd(*(abs(value) for value in integers))
    integers = tuple(value // common_factor for value in integers)
    first_nonzero = next(value for value in integers if value)
    if first_nonzero < 0:
        integers = tuple(-value for value in integers)
    return integers


def _quantity(
    species_ids: Sequence[str],
    vector: Sequence[sp.Expr],
) -> ConservedQuantity:
    coefficients = {
        species_id: coefficient
        for species_id, coefficient in zip(
            species_ids,
            _primitive_integers(vector),
            strict=True,
        )
        if coefficient
    }
    return ConservedQuantity(MappingProxyType(coefficients))


def _connected_components(
    matrix: sp.ImmutableMatrix,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return species and reaction indices for each nontrivial component."""

    reaction_species = tuple(
        tuple(row for row in range(matrix.rows) if matrix[row, column] != 0)
        for column in range(matrix.cols)
    )
    species_reactions = [set() for _ in range(matrix.rows)]
    for reaction, species_rows in enumerate(reaction_species):
        for row in species_rows:
            species_reactions[row].add(reaction)

    remaining = {row for row, reactions in enumerate(species_reactions) if reactions}
    components = []
    while remaining:
        pending = [min(remaining)]
        component_species: set[int] = set()
        component_reactions: set[int] = set()
        while pending:
            row = pending.pop()
            if row in component_species:
                continue
            component_species.add(row)
            for reaction in species_reactions[row]:
                if reaction not in component_reactions:
                    component_reactions.add(reaction)
                    pending.extend(reaction_species[reaction])
        remaining -= component_species
        components.append(
            (
                tuple(sorted(component_species)),
                tuple(sorted(component_reactions)),
            )
        )
    return tuple(components)


def _extreme_ray_vectors(constraints: sp.MatrixBase) -> tuple[tuple[int, ...], ...]:
    """Enumerate the support-minimal nonnegative vectors in ``ker(constraints)``."""

    dimension = constraints.cols
    maximum_support = min(dimension, int(constraints.rank()) + 1)
    rays: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    # An extreme ray has a strictly positive vector on a support whose
    # restricted nullspace is one-dimensional. Such a support contains no
    # more than rank(constraints) + 1 species.
    for support_size in range(1, maximum_support + 1):
        for support in combinations(range(dimension), support_size):
            nullspace = constraints[:, support].nullspace()
            if len(nullspace) != 1:
                continue

            supported_vector = tuple(nullspace[0])
            if all(value < 0 for value in supported_vector):
                supported_vector = tuple(-value for value in supported_vector)
            if not all(value > 0 for value in supported_vector):
                continue

            vector = [sp.S.Zero] * dimension
            for index, value in zip(support, supported_vector, strict=True):
                vector[index] = value
            ray = _primitive_integers(vector)
            if ray not in seen:
                seen.add(ray)
                rays.append(ray)
    return tuple(rays)


def _span_rank(vectors: Sequence[Sequence[int]]) -> int:
    if not vectors:
        return 0
    return int(sp.Matrix.hstack(*(sp.Matrix(vector) for vector in vectors)).rank())


def _signed_supplement(
    basis: Sequence[tuple[int, ...]],
    extreme_rays: Sequence[tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    """Complete the extreme-ray span with independent signed relations."""

    spanning_vectors = list(extreme_rays)
    rank = _span_rank(spanning_vectors)
    supplement = []
    for vector in basis:
        candidate_rank = _span_rank((*spanning_vectors, vector))
        if candidate_rank > rank:
            supplement.append(vector)
            spanning_vectors.append(vector)
            rank = candidate_rank
    return tuple(supplement)


def _analyse_component(
    case: Case,
    matrix: sp.ImmutableMatrix,
    species_rows: tuple[int, ...],
    reaction_columns: tuple[int, ...],
) -> ConservationComponent:
    species_ids = tuple(case.states.species_ids[row] for row in species_rows)
    reaction_ids = tuple(case.reactions[column].id for column in reaction_columns)
    component_matrix = matrix.extract(species_rows, reaction_columns)
    constraints = component_matrix.T

    basis_vectors = tuple(
        _primitive_integers(vector) for vector in constraints.nullspace()
    )
    extreme_ray_vectors = _extreme_ray_vectors(constraints)
    signed_vectors = _signed_supplement(basis_vectors, extreme_ray_vectors)

    return ConservationComponent(
        species_ids=species_ids,
        reaction_ids=reaction_ids,
        basis=tuple(_quantity(species_ids, vector) for vector in basis_vectors),
        extreme_rays=tuple(
            _quantity(species_ids, vector) for vector in extreme_ray_vectors
        ),
        signed_basis=tuple(_quantity(species_ids, vector) for vector in signed_vectors),
    )


def find_conserved_quantities(case: Case) -> StoichiometricConservationResult:
    """Find exact linear concentration invariants of the selected reactions."""

    matrix = _stoichiometric_matrix(case)
    components = _connected_components(matrix)
    unchanged_species = tuple(
        species_id
        for row, species_id in enumerate(case.states.species_ids)
        if all(matrix[row, column] == 0 for column in range(matrix.cols))
    )
    rank = int(matrix.rank())

    return StoichiometricConservationResult(
        species_ids=case.states.species_ids,
        reaction_ids=tuple(reaction.id for reaction in case.reactions),
        stoichiometric_matrix=matrix,
        rank=rank,
        dimension=matrix.rows - rank,
        unchanged_species=unchanged_species,
        components=tuple(
            _analyse_component(case, matrix, species_rows, reaction_columns)
            for species_rows, reaction_columns in components
        ),
    )


def _format_quantity(quantity: ConservedQuantity) -> str:
    parts = []
    for species_id, coefficient in quantity.coefficients.items():
        magnitude = abs(coefficient)
        term = f"[{species_id}]" if magnitude == 1 else f"{magnitude} [{species_id}]"
        if not parts:
            parts.append(term if coefficient > 0 else f"-{term}")
        else:
            operator = "+" if coefficient > 0 else "-"
            parts.append(f"{operator} {term}")
    return " ".join(parts)


def _details(result: StoichiometricConservationResult) -> tuple[str, ...]:
    details = [
        "Quantities are conserved by the selected reaction source terms; flows, "
        "dilution, and changing volume are not included.",
        "Only expressions are shown because the case has no initial concentrations.",
    ]
    if result.dimension == 0:
        details.append("No non-zero linear concentration invariant exists.")
        return tuple(details)

    if result.unchanged_species:
        details.append(
            "Individually unchanged species: "
            + ", ".join(result.unchanged_species)
            + "."
        )

    quantity_number = 1
    conserved_components = tuple(
        component for component in result.components if component.basis
    )
    for component_number, component in enumerate(conserved_components, start=1):
        details.append(
            f"Component {component_number} ({', '.join(component.species_ids)}):"
        )
        if component.extreme_rays:
            details.append("  Non-negative extreme rays:")
            for quantity in component.extreme_rays:
                details.append(f"    Q{quantity_number} = {_format_quantity(quantity)}")
                quantity_number += 1
        if component.signed_basis:
            label = (
                "  Additional signed basis relations:"
                if component.extreme_rays
                else "  Signed basis relations:"
            )
            details.append(label)
            for quantity in component.signed_basis:
                details.append(f"    Q{quantity_number} = {_format_quantity(quantity)}")
                quantity_number += 1
    return tuple(details)


def run(case: Case, _context: CheckContext) -> CheckOutcome:
    """Report conserved quantities once for the complete case."""

    result = find_conserved_quantities(case)
    return CheckOutcome(
        details=_details(result),
        values=(
            CheckValue("Stoichiometric rank", result.rank),
            CheckValue("Conserved-quantity dimension", result.dimension),
        ),
    )


CHECK = CheckDefinition(
    id="stoichiometric_conservation",
    name="Conserved stoichiometric quantities",
    group="Network analysis",
    scope=CheckScope.CASE,
    run=run,
)
