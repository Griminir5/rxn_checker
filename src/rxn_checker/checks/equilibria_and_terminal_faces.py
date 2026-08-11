"""Combined equilibrium and concentration-face network check."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import sympy as sp

from ..case import Case
from .equilibria import EquilibriumFamily, equilibrium_family, find_equilibria
from .models import (
    CheckContext,
    CheckDefinition,
    CheckOutcome,
    CheckScope,
    CheckStatus,
    CheckValue,
)
from .terminal_faces import MAX_FACE_TESTS, all_zero, find_terminal_faces


@dataclass(frozen=True)
class EquilibriaAndTerminalFacesResult:
    """Equilibrium families and maximal coordinate faces found for a case."""

    source_terms: Mapping[str, sp.Expr]
    equilibria: tuple[EquilibriumFamily, ...]
    excluded_equilibria: int
    equilibrium_search_complete: bool
    equilibrium_diagnostic: str | None
    terminal_faces: tuple[tuple[str, ...], ...]
    invariant_faces: tuple[tuple[str, ...], ...]
    unresolved_terminal_faces: tuple[tuple[str, ...], ...]
    unresolved_invariant_faces: tuple[tuple[str, ...], ...]
    face_search_complete: bool
    face_tests: int


def _source_terms(case: Case) -> Mapping[str, sp.Expr]:
    """Construct ``F = S r`` in the ordering of the case species."""

    return MappingProxyType(
        {
            species_id: sp.Add(
                *(
                    sp.sympify(reaction.net_stoichiometry.get(species_id, 0))
                    * reaction.rate
                    for reaction in case.reactions
                )
            )
            for species_id in case.states.species_ids
        }
    )


def _terminal_family_is_covered(
    family: EquilibriumFamily,
    point: tuple[sp.Expr, ...],
    substitutions: Mapping[sp.Symbol, sp.Expr],
) -> bool:
    differences = tuple(
        expression.xreplace(substitutions) - coordinate
        for expression, coordinate in zip(
            family.coordinates.values(),
            point,
            strict=True,
        )
    )
    return all_zero(differences) is True


def check_equilibria_and_terminal_faces(
    case: Case,
) -> EquilibriaAndTerminalFacesResult:
    """Locate symbolic equilibria and maximal concentration faces.

    Face identities use bounded, simplification-free proofs. The generic
    nonlinear equilibrium solver runs in a separate process with a hard
    timeout, so an expression outside SymPy's practical range is reported as
    incomplete instead of blocking the full check suite.
    """

    source_terms = _source_terms(case)
    faces = find_terminal_faces(case, source_terms)
    equilibrium_search = find_equilibria(case, source_terms)

    # Every terminal face is an equilibrium family by definition. Include it
    # explicitly even when the general nonlinear solver omits a branch.
    equilibria = list(equilibrium_search.equilibria)
    for face in faces.terminal_faces:
        substitutions = {
            case.states.concentration(species_id): sp.S.Zero for species_id in face
        }
        point = tuple(
            sp.S.Zero if species_id in face else case.states.concentration(species_id)
            for species_id in case.states.species_ids
        )
        if not any(
            _terminal_family_is_covered(family, point, substitutions)
            for family in equilibria
        ):
            equilibria.append(equilibrium_family(point, case))

    return EquilibriaAndTerminalFacesResult(
        source_terms=source_terms,
        equilibria=tuple(equilibria),
        excluded_equilibria=equilibrium_search.excluded,
        equilibrium_search_complete=equilibrium_search.complete,
        equilibrium_diagnostic=equilibrium_search.diagnostic,
        terminal_faces=faces.terminal_faces,
        invariant_faces=faces.invariant_faces,
        unresolved_terminal_faces=faces.unresolved_terminal_faces,
        unresolved_invariant_faces=faces.unresolved_invariant_faces,
        face_search_complete=faces.complete,
        face_tests=faces.tests,
    )


def _format_equilibrium(family: EquilibriumFamily) -> str:
    coordinates = []
    for species_id, expression in family.coordinates.items():
        symbol = sp.Symbol(species_id, real=True)
        rendered = (
            f"{species_id} free"
            if expression == symbol
            else f"{species_id}={expression}"
        )
        coordinates.append(rendered)
    physical = "physical" if family.physical else "conditionally physical"
    return f"{'; '.join(coordinates)} ({physical})."


def _format_face(face: tuple[str, ...], all_species: tuple[str, ...]) -> str:
    if not face:
        return "entire concentration domain"
    equations = ", ".join(f"{species_id}=0" for species_id in face)
    if len(face) == len(all_species):
        return f"origin ({equations})"
    return equations


def _outcome(
    result: EquilibriaAndTerminalFacesResult,
    species_ids: tuple[str, ...],
) -> CheckOutcome:
    details: list[str] = []
    if result.equilibria:
        details.extend(
            f"Equilibrium {index}: {_format_equilibrium(family)}"
            for index, family in enumerate(result.equilibria, start=1)
        )
    else:
        details.append("No physical symbolic equilibrium families were found.")
    if result.excluded_equilibria:
        details.append(
            f"Excluded {result.excluded_equilibria} equilibrium family/families "
            "outside the configured physical bounds."
        )
    if result.equilibrium_diagnostic:
        details.append(
            f"Equilibrium search incomplete: {result.equilibrium_diagnostic}."
        )

    if result.terminal_faces:
        rendered = "; ".join(
            _format_face(face, species_ids) for face in result.terminal_faces
        )
        details.append(f"Maximal terminal faces: {rendered}.")
    else:
        details.append("No terminal concentration faces were found.")

    if result.invariant_faces:
        rendered = "; ".join(
            _format_face(face, species_ids) for face in result.invariant_faces
        )
        details.append(f"Maximal invariant faces: {rendered}.")
    else:
        details.append("No proper invariant concentration faces were found.")

    unresolved = len(result.unresolved_terminal_faces) + len(
        result.unresolved_invariant_faces
    )
    if unresolved:
        details.append(
            "Symbolic zero testing was inconclusive for "
            f"{len(result.unresolved_terminal_faces)} terminal and "
            f"{len(result.unresolved_invariant_faces)} invariant face candidates."
        )
    if not result.face_search_complete:
        details.append(
            f"Face search stopped after {result.face_tests} symbolic tests "
            f"(limit {MAX_FACE_TESTS})."
        )

    complete = (
        result.equilibrium_search_complete
        and result.face_search_complete
        and not unresolved
    )
    status = CheckStatus.PASS if complete else CheckStatus.INDETERMINATE
    return CheckOutcome(
        status=status,
        details=tuple(details),
        values=(
            CheckValue("Equilibrium families", len(result.equilibria)),
            CheckValue("Terminal faces", len(result.terminal_faces)),
            CheckValue("Invariant faces", len(result.invariant_faces)),
        ),
    )


def run(case: Case, context: CheckContext) -> CheckOutcome:
    """Run equilibrium and face discovery once for the complete case."""

    return _outcome(check_equilibria_and_terminal_faces(case), case.states.species_ids)


CHECK = CheckDefinition(
    id="equilibria_and_terminal_faces",
    name="Equilibria and terminal faces",
    group="Network analysis",
    scope=CheckScope.CASE,
    run=run,
)
