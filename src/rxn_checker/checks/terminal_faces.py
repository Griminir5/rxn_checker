"""Simplification-free concentration-face discovery."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
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

MAX_FACE_TESTS = 4096
_WITNESS_FRACTIONS = (
    sp.Rational(1, 2),
    sp.Rational(1, 3),
    sp.Rational(2, 3),
)


@dataclass(frozen=True)
class TerminalFaceSearchResult:
    """Maximal coordinate faces certified by bounded symbolic tests."""

    terminal_faces: tuple[tuple[str, ...], ...]
    invariant_faces: tuple[tuple[str, ...], ...]
    unresolved_terminal_faces: tuple[tuple[str, ...], ...]
    unresolved_invariant_faces: tuple[tuple[str, ...], ...]
    complete: bool
    tests: int


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


def zero_status(expression: sp.Expr) -> bool | None:
    """Return cheap zero knowledge without requesting algebraic rewriting."""

    if expression is sp.S.Zero:
        return True
    return expression.is_zero


def _with_relative_interior_assumptions(
    expression: sp.Expr,
    case: Case,
) -> sp.Expr:
    """Give free state symbols the signs they have inside their bounds."""

    replacements: dict[sp.Symbol, sp.Symbol] = {}
    for symbol in expression.free_symbols:
        bounds = case.state_bounds[symbol]
        if bounds.physical_lower >= 0:
            assumptions = {"positive": True}
        elif bounds.physical_upper <= 0:
            assumptions = {"negative": True}
        else:
            assumptions = {"real": True}
        replacements[symbol] = sp.Dummy(symbol.name, **assumptions)
    return expression.xreplace(replacements)


def _witness_values(case: Case, fraction: sp.Rational) -> Mapping[sp.Symbol, sp.Expr]:
    """Return an exact point strictly inside every configured state bound."""

    values: dict[sp.Symbol, sp.Expr] = {}
    for symbol, bounds in case.state_bounds.items():
        lower = sp.Rational(str(bounds.physical_lower))
        upper = sp.Rational(str(bounds.physical_upper))
        values[symbol] = lower + fraction * (upper - lower)
    return values


def _zero_status_on_domain(expression: sp.Expr, case: Case) -> bool | None:
    """Classify an identity using signs and exact nonzero witnesses."""

    conclusion = zero_status(expression)
    if conclusion is not None:
        return conclusion

    physical_expression = _with_relative_interior_assumptions(expression, case)
    if zero_status(physical_expression) is False:
        return False

    for fraction in _WITNESS_FRACTIONS:
        value = expression.xreplace(_witness_values(case, fraction))
        if zero_status(value) is False:
            return False
    return None


def all_zero(
    expressions: tuple[sp.Expr, ...],
    case: Case | None = None,
) -> bool | None:
    """Return whether every expression is identically zero, if provable."""

    unresolved = False
    for expression in expressions:
        conclusion = (
            zero_status(expression)
            if case is None
            else _zero_status_on_domain(expression, case)
        )
        if conclusion is False:
            return False
        if conclusion is None:
            unresolved = True
    return None if unresolved else True


def _has_known_parent(
    depleted: tuple[str, ...],
    known_faces: list[tuple[str, ...]],
) -> bool:
    depleted_set = set(depleted)
    return any(set(face).issubset(depleted_set) for face in known_faces)


def find_terminal_faces(
    case: Case,
    source_terms: Mapping[str, sp.Expr],
) -> TerminalFaceSearchResult:
    """Enumerate maximal terminal and invariant coordinate faces.

    Expressions are restricted with simultaneous atom replacement. Zero
    identities are then classified from existing SymPy assumptions, physical
    signs, and exact witness points. No global simplification or ``equals``
    call is made.
    """

    species_ids = tuple(
        species_id
        for species_id in case.states.species_ids
        if not case.state_bounds[
            case.states.concentration(species_id)
        ].strict_lower
    )
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
                return TerminalFaceSearchResult(
                    terminal_faces=tuple(terminal_faces),
                    invariant_faces=tuple(invariant_faces),
                    unresolved_terminal_faces=tuple(unresolved_terminal),
                    unresolved_invariant_faces=tuple(unresolved_invariant),
                    complete=False,
                    tests=face_tests,
                )

            substitutions = {
                case.states.concentration(species_id): sp.S.Zero
                for species_id in depleted
            }
            restricted = {
                species_id: expression.xreplace(substitutions)
                for species_id, expression in source_terms.items()
            }
            face_tests += 1

            terminal = all_zero(tuple(restricted.values()), case)
            if terminal is True:
                terminal_faces.append(depleted)
                if not depleted:
                    return TerminalFaceSearchResult(
                        terminal_faces=tuple(terminal_faces),
                        invariant_faces=tuple(invariant_faces),
                        unresolved_terminal_faces=tuple(unresolved_terminal),
                        unresolved_invariant_faces=tuple(unresolved_invariant),
                        complete=True,
                        tests=face_tests,
                    )
            elif terminal is None:
                unresolved_terminal.append(depleted)

            if not depleted:
                continue
            invariant = all_zero(
                tuple(restricted[species_id] for species_id in depleted),
                case,
            )
            if invariant is True and not _has_known_parent(
                depleted,
                invariant_faces,
            ):
                invariant_faces.append(depleted)
            elif invariant is None:
                unresolved_invariant.append(depleted)

    return TerminalFaceSearchResult(
        terminal_faces=tuple(terminal_faces),
        invariant_faces=tuple(invariant_faces),
        unresolved_terminal_faces=tuple(unresolved_terminal),
        unresolved_invariant_faces=tuple(unresolved_invariant),
        complete=True,
        tests=face_tests,
    )


def check_terminal_faces(case: Case) -> TerminalFaceSearchResult:
    """Locate maximal terminal and invariant concentration faces."""

    return find_terminal_faces(case, _source_terms(case))


def _format_face(face: tuple[str, ...], all_species: tuple[str, ...]) -> str:
    if not face:
        return "entire concentration domain"
    equations = ", ".join(f"{species_id}=0" for species_id in face)
    if len(face) == len(all_species):
        return f"origin ({equations})"
    return equations


def _outcome(
    result: TerminalFaceSearchResult,
    species_ids: tuple[str, ...],
) -> CheckOutcome:
    details: list[str] = []
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
    if not result.complete:
        details.append(
            f"Face search stopped after {result.tests} symbolic tests "
            f"(limit {MAX_FACE_TESTS})."
        )

    complete = result.complete and not unresolved
    status = CheckStatus.PASS if complete else CheckStatus.INDETERMINATE
    return CheckOutcome(
        status=status,
        details=tuple(details),
        values=(
            CheckValue("Terminal faces", len(result.terminal_faces)),
            CheckValue("Invariant faces", len(result.invariant_faces)),
        ),
    )


def run(case: Case, context: CheckContext) -> CheckOutcome:
    """Run face discovery once for the complete case."""

    return _outcome(check_terminal_faces(case), case.states.species_ids)


CHECK = CheckDefinition(
    id="terminal_faces",
    name="Terminal faces",
    group="Network analysis",
    scope=CheckScope.CASE,
    run=run,
)
