"""Exact conserved quantities for a complete reaction network."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

import sympy as sp

from ..case import Case
from .models import CheckContext, CheckDefinition, CheckOutcome, CheckScope, CheckValue
from .network import NetworkExpressions, network_expressions


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
    """Find exact rays of ``ker(constraints)`` intersected with the orthant.

    This is an incremental double-description calculation. Unlike enumerating
    every possible species support up to ``rank + 1``, its combinatorics follow
    the rays actually present while coordinate inequalities are introduced.
    """

    nullspace = constraints.nullspace()
    if not nullspace:
        return ()
    basis = sp.Matrix.hstack(*nullspace)
    cone_dimension = basis.cols

    # Independent coordinate inequalities form an initial simplicial cone in
    # nullspace coordinates: B0*y >= 0. Its rays are B0**-1 columns.
    initial_rows = tuple(int(index) for index in basis.T.rref()[1])
    initial = basis.extract(initial_rows, range(cone_dimension))
    inverse = initial.inv()
    rays = [inverse[:, column] for column in range(cone_dimension)]
    processed_rows = list(initial_rows)

    def direction(vector: sp.MatrixBase) -> tuple[int, ...]:
        rationals = tuple(sp.Rational(value) for value in vector)
        denominator = math.lcm(*(int(value.q) for value in rationals))
        integers = tuple(int(value * denominator) for value in rationals)
        divisor = math.gcd(*(abs(value) for value in integers))
        return tuple(value // divisor for value in integers)

    def retain_extreme(
        candidates: Sequence[sp.MatrixBase],
        rows: Sequence[int],
    ) -> list[sp.Matrix]:
        kept: dict[tuple[int, ...], sp.Matrix] = {}
        inequalities = basis.extract(rows, range(cone_dimension))
        for candidate in candidates:
            key = direction(candidate)
            if key in kept:
                continue
            active = tuple(
                row
                for row in range(inequalities.rows)
                if (inequalities[row, :] * candidate)[0] == 0
            )
            active_matrix = inequalities.extract(active, range(cone_dimension))
            if int(active_matrix.rank()) >= cone_dimension - 1:
                kept[key] = sp.Matrix(candidate)
        return list(kept.values())

    for row in range(basis.rows):
        if row in initial_rows:
            continue
        inequality = basis[row, :]
        positive: list[tuple[sp.Matrix, sp.Expr]] = []
        zero: list[sp.Matrix] = []
        negative: list[tuple[sp.Matrix, sp.Expr]] = []
        for ray in rays:
            value = (inequality * ray)[0]
            if value > 0:
                positive.append((ray, value))
            elif value < 0:
                negative.append((ray, value))
            else:
                zero.append(ray)

        candidates: list[sp.MatrixBase] = [*zero, *(ray for ray, _ in positive)]
        candidates.extend(
            positive_value * negative_ray - negative_value * positive_ray
            for positive_ray, positive_value in positive
            for negative_ray, negative_value in negative
        )
        processed_rows.append(row)
        rays = retain_extreme(candidates, processed_rows)
        if not rays:
            return ()

    result: dict[tuple[int, ...], None] = {}
    for ray in rays:
        vector = basis * ray
        if any(value < 0 for value in vector):
            continue
        result.setdefault(_primitive_integers(vector), None)
    return tuple(
        sorted(
            result,
            key=lambda vector: (
                sum(value != 0 for value in vector),
                tuple(index for index, value in enumerate(vector) if value),
                vector,
            ),
        )
    )


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


def find_conserved_quantities(
    case: Case,
    network: NetworkExpressions | None = None,
) -> StoichiometricConservationResult:
    """Find exact linear concentration invariants of the selected reactions."""

    matrix = (
        network.stoichiometric_matrix
        if network is not None
        else _stoichiometric_matrix(case)
    )
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


def run(case: Case, context: CheckContext) -> CheckOutcome:
    """Report conserved quantities once for the complete case."""

    network = context.cached(
        case,
        "network",
        lambda: network_expressions(case),
    )
    result = context.cached(
        case,
        "conservation",
        lambda: find_conserved_quantities(case, network),
    )
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
