"""Symbolic equilibrium and concentration-face discovery."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
import math
from types import MappingProxyType

import sympy as sp

from ..case import Case
from .models import (
    CheckContext,
    CheckDefinition,
    CheckOutcome,
    CheckScope,
    CheckStatus,
    CheckValue,
)
from .network_positivity import network_source_terms

MAX_FACE_TESTS = 4096


@dataclass(frozen=True)
class EquilibriumFamily:
    """One symbolic family of equilibria in concentration coordinates."""

    coordinates: Mapping[str, sp.Expr]
    physical: bool | None


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


def _zero_status(expression: sp.Expr) -> bool | None:
    expression = sp.simplify(expression)
    if expression.is_zero is not None:
        return expression.is_zero
    return expression.equals(sp.S.Zero)


def _all_zero(expressions: tuple[sp.Expr, ...]) -> bool | None:
    conclusions = tuple(_zero_status(expression) for expression in expressions)
    if any(conclusion is False for conclusion in conclusions):
        return False
    if any(conclusion is None for conclusion in conclusions):
        return None
    return True


def _coordinate_is_physical(
    expression: sp.Expr,
    symbol: sp.Symbol,
    case: Case,
) -> bool | None:
    """Classify bounds only when they follow without extra conditions."""

    expression = sp.simplify(expression)
    bounds = case.state_bounds[symbol]
    if expression == symbol:
        return True
    if expression.is_real is False or expression.has(
        sp.nan,
        sp.zoo,
        sp.oo,
        -sp.oo,
    ):
        return False

    if isinstance(expression, sp.Symbol) and expression in case.state_bounds:
        source_bounds = case.state_bounds[expression]
        if (
            source_bounds.physical_lower >= bounds.physical_lower
            and source_bounds.physical_upper <= bounds.physical_upper
        ):
            return True
        if (
            source_bounds.physical_upper < bounds.physical_lower
            or source_bounds.physical_lower > bounds.physical_upper
        ):
            return False
        return None

    if not expression.free_symbols:
        try:
            value = float(expression)
        except (TypeError, ValueError, OverflowError):
            return None
        return (
            math.isfinite(value)
            and bounds.physical_lower <= value <= bounds.physical_upper
        )

    return None


def _equilibrium_family(
    point: tuple[sp.Expr, ...],
    case: Case,
) -> EquilibriumFamily:
    species_ids = case.states.species_ids
    symbols = tuple(case.states.concentration(name) for name in species_ids)
    coordinates = {
        species_id: sp.simplify(expression)
        for species_id, expression in zip(species_ids, point, strict=True)
    }
    physical_coordinates = tuple(
        _coordinate_is_physical(coordinates[species_id], symbol, case)
        for species_id, symbol in zip(species_ids, symbols, strict=True)
    )
    if any(conclusion is False for conclusion in physical_coordinates):
        physical: bool | None = False
    elif any(conclusion is None for conclusion in physical_coordinates):
        physical = None
    else:
        physical = True
    return EquilibriumFamily(MappingProxyType(coordinates), physical)


def _find_equilibria(
    case: Case,
    source_terms: Mapping[str, sp.Expr],
) -> tuple[tuple[EquilibriumFamily, ...], int, bool, str | None]:
    symbols = tuple(case.states.concentration(name) for name in case.states.species_ids)
    equations = tuple(
        dict.fromkeys(
            sp.nsimplify(expression, rational=True)
            for expression in source_terms.values()
            if _zero_status(expression) is not True
        )
    )

    if not equations:
        family = _equilibrium_family(symbols, case)
        return (family,), 0, True, None

    try:
        solution_set = sp.nonlinsolve(equations, symbols)
    except (NotImplementedError, ValueError) as error:
        return (), 0, False, f"{type(error).__name__}: {error}"

    if solution_set is sp.EmptySet:
        return (), 0, True, None
    if not isinstance(solution_set, sp.FiniteSet):
        return (), 0, False, f"Solver returned unresolved set: {solution_set}"

    equilibria: list[EquilibriumFamily] = []
    excluded = 0
    for point in solution_set:
        if not isinstance(point, (tuple, sp.Tuple)) or len(point) != len(symbols):
            return (
                tuple(equilibria),
                excluded,
                False,
                f"Solver returned an unsupported solution: {point}",
            )
        if any(expression is sp.EmptySet for expression in point):
            continue
        if any(isinstance(expression, sp.Set) for expression in point):
            return (
                tuple(equilibria),
                excluded,
                False,
                f"Solver returned an unresolved solution: {point}",
            )
        family = _equilibrium_family(tuple(point), case)
        if family.physical is False:
            excluded += 1
        else:
            equilibria.append(family)
    return tuple(equilibria), excluded, True, None


def _has_known_parent(
    depleted: tuple[str, ...],
    known_faces: list[tuple[str, ...]],
) -> bool:
    depleted_set = set(depleted)
    return any(set(face).issubset(depleted_set) for face in known_faces)


def _find_faces(
    case: Case,
    source_terms: Mapping[str, sp.Expr],
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[str, ...], ...],
    tuple[tuple[str, ...], ...],
    tuple[tuple[str, ...], ...],
    bool,
    int,
]:
    """Enumerate maximal terminal and invariant coordinate faces.

    A face is represented by the species fixed at zero. Enumeration proceeds
    from large geometric faces to their subfaces, so a known parent lets us
    discard all of its descendants from the maximal-face result.
    """

    species_ids = case.states.species_ids
    terminal_faces: list[tuple[str, ...]] = []
    invariant_faces: list[tuple[str, ...]] = []
    unresolved_terminal: list[tuple[str, ...]] = []
    unresolved_invariant: list[tuple[str, ...]] = []
    face_tests = 0

    for depleted_count in range(len(species_ids) + 1):
        for depleted in combinations(species_ids, depleted_count):
            if _has_known_parent(depleted, terminal_faces):
                continue
            if face_tests >= MAX_FACE_TESTS:
                return (
                    tuple(terminal_faces),
                    tuple(invariant_faces),
                    tuple(unresolved_terminal),
                    tuple(unresolved_invariant),
                    False,
                    face_tests,
                )

            substitutions = {
                case.states.concentration(species_id): sp.S.Zero
                for species_id in depleted
            }
            restricted = {
                species_id: sp.simplify(expression.subs(substitutions))
                for species_id, expression in source_terms.items()
            }
            face_tests += 1

            terminal = _all_zero(tuple(restricted.values()))
            if terminal is True:
                terminal_faces.append(depleted)
                if not depleted:
                    return (
                        tuple(terminal_faces),
                        tuple(invariant_faces),
                        tuple(unresolved_terminal),
                        tuple(unresolved_invariant),
                        True,
                        face_tests,
                    )
            elif terminal is None:
                unresolved_terminal.append(depleted)

            if not depleted:
                continue
            invariant = _all_zero(
                tuple(restricted[species_id] for species_id in depleted)
            )
            if invariant is True and not _has_known_parent(
                depleted,
                invariant_faces,
            ):
                invariant_faces.append(depleted)
            elif invariant is None:
                unresolved_invariant.append(depleted)

    return (
        tuple(terminal_faces),
        tuple(invariant_faces),
        tuple(unresolved_terminal),
        tuple(unresolved_invariant),
        True,
        face_tests,
    )


def check_equilibria_and_terminal_faces(
    case: Case,
) -> EquilibriaAndTerminalFacesResult:
    """Locate symbolic equilibria and maximal concentration faces.

    Equilibria solve ``S r = 0`` with temperature and pressure retained as
    parameters. A terminal face makes the complete source vector zero. An
    invariant face only makes source components of its depleted species zero.
    """

    source_terms = network_source_terms(case)
    (
        terminal_faces,
        invariant_faces,
        unresolved_terminal,
        unresolved_invariant,
        face_complete,
        face_tests,
    ) = _find_faces(case, source_terms)
    equilibria, excluded, equilibrium_complete, diagnostic = _find_equilibria(
        case,
        source_terms,
    )

    # Every terminal face is an equilibrium family by definition. Include it
    # explicitly even when the general nonlinear solver omits a branch.
    equilibria = list(equilibria)
    for face in terminal_faces:
        substitutions = {
            case.states.concentration(species_id): sp.S.Zero for species_id in face
        }
        point = tuple(
            sp.S.Zero if species_id in face else case.states.concentration(species_id)
            for species_id in case.states.species_ids
        )
        covered = any(
            _all_zero(
                tuple(
                    expression.subs(substitutions) - coordinate
                    for expression, coordinate in zip(
                        family.coordinates.values(),
                        point,
                        strict=True,
                    )
                )
            )
            is True
            for family in equilibria
        )
        if not covered:
            equilibria.append(_equilibrium_family(point, case))

    return EquilibriaAndTerminalFacesResult(
        source_terms=source_terms,
        equilibria=tuple(equilibria),
        excluded_equilibria=excluded,
        equilibrium_search_complete=equilibrium_complete,
        equilibrium_diagnostic=diagnostic,
        terminal_faces=terminal_faces,
        invariant_faces=invariant_faces,
        unresolved_terminal_faces=unresolved_terminal,
        unresolved_invariant_faces=unresolved_invariant,
        face_search_complete=face_complete,
        face_tests=face_tests,
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
