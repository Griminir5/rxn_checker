"""Compact exact analyses derived from reaction-network structure.

These analyses deliberately avoid solving rate equations.  Conservation and
independent equations use exact linear algebra on the stoichiometric matrix;
faces use only reactant, catalyst, and product supports.
"""

from collections.abc import Mapping

import sympy as sp
from sympy.utilities.iterables import connected_components

from ..context import AnalysisContext
from ..network import primitive_integer_vector
from ..results import CheckResult, Evidence, Finding, Verdict

_FACE_LIMIT = 32
_SEARCH_LIMIT = 8192


def _linear_text(terms) -> str:
    """Render a short exact linear combination without formatting expressions."""

    pieces = []
    for coefficient, label in terms:
        magnitude = abs(coefficient)
        body = label if magnitude == 1 else f"{magnitude}*{label}"
        if not pieces:
            pieces.append(f"-{body}" if coefficient < 0 else body)
        else:
            pieces.append(f" {'-' if coefficient < 0 else '+'} {body}")
    return "".join(pieces) or "0"


def _components(species_ids, stoichiometry) -> tuple[tuple[str, ...], ...]:
    """Connected species components of the stoichiometric bipartite graph."""

    edges = []
    for column in range(stoichiometry.cols):
        participants = [
            species_ids[row] for row in range(stoichiometry.rows) if stoichiometry[row, column]
        ]
        edges.extend((participants[0], species) for species in participants[1:])
    order = {species: index for index, species in enumerate(species_ids)}
    return tuple(
        tuple(sorted(group, key=order.__getitem__))
        for group in connected_components((species_ids, edges))
    )


def run_conserved_quantities(
    context: AnalysisContext, _dependencies: Mapping[str, CheckResult]
) -> tuple[Finding, ...]:
    """Report a primitive integer basis of the left nullspace of S."""

    matrix = context.stoichiometry
    species_ids = context.case.symbols.species_ids
    basis = tuple(primitive_integer_vector(vector) for vector in matrix.T.nullspace())
    rank = matrix.rows - len(basis)
    unchanged = tuple(
        species_ids[row]
        for row in range(matrix.rows)
        if all(matrix[row, column] == 0 for column in range(matrix.cols))
    )
    components = _components(species_ids, matrix)
    summary = Finding(
        context.case.name,
        Verdict.PASS,
        f"Stoichiometric rank {rank}; {len(basis)} independent linear "
        f"conserved quantities in {len(components)} components; "
        f"{len(unchanged)} unchanged species.",
        Evidence(
            "conservation_summary",
            {
                "rank": rank,
                "shape": matrix.shape,
                "connected_components": components,
                "unchanged_species": unchanged,
                "basis_size": len(basis),
            },
        ),
    )
    quantities = []
    for number, vector in enumerate(basis, 1):
        terms = tuple(
            (coefficient, species_id)
            for coefficient, species_id in zip(vector, species_ids)
            if coefficient != 0
        )
        expression = sum(
            (
                coefficient * context.case.symbols.concentration(species_id)
                for coefficient, species_id in terms
            ),
            sp.S.Zero,
        )
        quantities.append(
            Finding(
                f"Q{number}",
                Verdict.PASS,
                f"{_linear_text(terms)} = constant.",
                Evidence(
                    "conserved_quantity",
                    {
                        "coefficients": {
                            species_id: coefficient for coefficient, species_id in terms
                        },
                        "expression": expression,
                    },
                ),
            )
        )
    return (summary, *quantities)


def _insert_minimal(results, candidate) -> None:
    if any(result <= candidate for result in results):
        return
    results[:] = [result for result in results if not candidate < result]
    results.append(candidate)


def _minimal_sets(seeds, violations, species_ids) -> tuple[tuple[frozenset[str], ...], bool]:
    """Branch on one violated support until a bounded minimal set is found."""

    order = {species_id: index for index, species_id in enumerate(species_ids)}
    results: list[frozenset[str]] = []
    visited: set[frozenset[str]] = set()
    truncated = False

    def search(face: frozenset[str]) -> None:
        nonlocal truncated
        if truncated or face in visited or any(result <= face for result in results):
            return
        visited.add(face)
        if len(visited) > _SEARCH_LIMIT:
            truncated = True
            return
        branches = tuple(violations(face))
        if not branches:
            _insert_minimal(results, face)
            truncated = len(results) >= _FACE_LIMIT
            return
        branch = min(branches, key=lambda item: (len(item), tuple(sorted(item))))
        for species_id in sorted(branch, key=order.__getitem__):
            search(face | {species_id})

    for seed in seeds:
        search(seed)
    ordered = tuple(
        sorted(results, key=lambda face: (len(face), tuple(sorted(order[item] for item in face))))
    )
    return ordered, truncated


def _feasible_faces(context, faces) -> tuple[tuple[str, ...], ...]:
    feasible = []
    for face in faces:
        domain = context.physical_domain
        for species_id in face:
            domain = domain.restrict(
                context.case.symbols.concentration(species_id), lower=0, upper=0
            )
        if domain.is_feasible():
            feasible.append(
                tuple(
                    species_id
                    for species_id in context.case.symbols.species_ids
                    if species_id in face
                )
            )
    return tuple(feasible)


def _faces_text(faces) -> str:
    shown = tuple("{" + ", ".join(face) + "}" for face in faces[:6])
    suffix = f", ... ({len(faces)} total)" if len(faces) > len(shown) else ""
    return ", ".join(shown) + suffix if shown else "none"


def run_structural_faces(
    context: AnalysisContext, _dependencies: Mapping[str, CheckResult]
) -> Finding:
    """Report bounded sufficient dead-face and invariant-face certificates."""

    required = tuple(
        frozenset((*reaction.reactants, *reaction.catalysts)) for reaction in context.case.reactions
    )
    production_rules = tuple(
        (
            frozenset(
                species_id
                for species_id, coefficient in reaction.net_stoichiometry.items()
                if coefficient > 0
            ),
            support,
        )
        for reaction, support in zip(context.case.reactions, required)
    )
    species_ids = context.case.symbols.species_ids
    dead, dead_truncated = _minimal_sets(
        (frozenset(),),
        lambda face: tuple(support for support in required if face.isdisjoint(support)),
        species_ids,
    )
    invariant, invariant_truncated = _minimal_sets(
        tuple(frozenset((species_id,)) for species_id in species_ids),
        lambda face: tuple(
            support
            for produced, support in production_rules
            if not face.isdisjoint(produced) and face.isdisjoint(support)
        ),
        species_ids,
    )
    dead_faces = _feasible_faces(context, dead)
    invariant_faces = _feasible_faces(context, invariant)
    truncated = dead_truncated or invariant_truncated
    verdict = Verdict.UNKNOWN if truncated else Verdict.PASS
    qualifier = "Partial" if truncated else "Minimal"
    suffix = " Search limit reached; these certificates are partial." if truncated else ""
    return Finding(
        context.case.name,
        verdict,
        f"{qualifier} structural dead faces: {_faces_text(dead_faces)}; "
        f"invariant faces: {_faces_text(invariant_faces)}.{suffix}",
        Evidence(
            "structural_faces",
            {
                "dead_faces": dead_faces,
                "invariant_faces": invariant_faces,
                "required_supports": {
                    reaction.id: tuple(
                        species_id
                        for species_id in context.case.symbols.species_ids
                        if species_id in support
                    )
                    for reaction, support in zip(context.case.reactions, required)
                },
                "search_truncated": truncated,
                "face_limit": _FACE_LIMIT,
            },
        ),
    )


def _independent_rows(context) -> tuple[int, ...]:
    matrix = context.stoichiometry
    rate_cost = tuple(sp.count_ops(reaction.rate) for reaction in context.case.reactions)
    ordered = sorted(
        range(matrix.rows),
        key=lambda row: (
            sum(matrix[row, column] != 0 for column in range(matrix.cols)),
            sum(rate_cost[column] for column in range(matrix.cols) if matrix[row, column] != 0),
            row,
        ),
    )
    permuted = matrix.extract(ordered, range(matrix.cols))
    _reduced, pivots = permuted.T.rref()
    return tuple(ordered[pivot] for pivot in pivots)


def run_steady_state_equations(
    context: AnalysisContext, _dependencies: Mapping[str, CheckResult]
) -> tuple[Finding, ...]:
    """Report a low-complexity independent set of sparse equations F_i = 0."""

    rows = _independent_rows(context)
    summary = Finding(
        context.case.name,
        Verdict.PASS,
        f"{len(rows)} independent equations span the stoichiometric row space.",
        Evidence("steady_state_summary", {"rank": len(rows)}),
    )
    equations = []
    for row in rows:
        species_id = context.case.symbols.species_ids[row]
        terms = tuple(
            (context.stoichiometry[row, column], reaction)
            for column, reaction in enumerate(context.case.reactions)
            if context.stoichiometry[row, column] != 0
        )
        expression_terms = tuple(coefficient * reaction.rate for coefficient, reaction in terms)
        expression = (
            expression_terms[0]
            if len(expression_terms) == 1
            else sp.Add(*expression_terms, evaluate=False)
        )
        labelled_terms = tuple(
            (coefficient, f"r[{reaction.id}]") for coefficient, reaction in terms
        )
        equations.append(
            Finding(
                species_id,
                Verdict.PASS,
                f"F_{species_id} = {_linear_text(labelled_terms)} = 0.",
                Evidence(
                    "steady_state_equation",
                    {
                        "species": species_id,
                        "coefficients": {
                            reaction.id: coefficient for coefficient, reaction in terms
                        },
                        "expression": expression,
                    },
                ),
            )
        )
    return (summary, *equations)
