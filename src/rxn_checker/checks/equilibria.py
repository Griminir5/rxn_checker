"""Bounded symbolic equilibrium discovery."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
import multiprocessing as mp
from types import MappingProxyType
from typing import Any

import sympy as sp

from ..case import Case
from .terminal_faces import zero_status

EQUILIBRIUM_SOLVE_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class EquilibriumFamily:
    """One symbolic family of equilibria in concentration coordinates."""

    coordinates: Mapping[str, sp.Expr]
    physical: bool | None


@dataclass(frozen=True)
class EquilibriumSearchResult:
    """Result of a bounded attempt to solve the network source equations."""

    equilibria: tuple[EquilibriumFamily, ...]
    excluded: int
    complete: bool
    diagnostic: str | None


def _coordinate_is_physical(
    expression: sp.Expr,
    symbol: sp.Symbol,
    case: Case,
) -> bool | None:
    """Classify bounds only when they follow without extra conditions."""

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


def equilibrium_family(
    point: tuple[sp.Expr, ...],
    case: Case,
) -> EquilibriumFamily:
    """Build and physically classify one solver result without rewriting it."""

    species_ids = case.states.species_ids
    symbols = tuple(case.states.concentration(name) for name in species_ids)
    coordinates = dict(zip(species_ids, point, strict=True))
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


def _coordinate_zero_faces(
    expression: sp.Expr,
    case: Case,
) -> tuple[tuple[tuple[str, ...], ...], bool]:
    """Find a complete coordinate zero set for simple expression trees.

    The returned Boolean is false when the expression contains a zero locus
    that this structural analyzer cannot represent. It never rewrites the
    expression algebraically.
    """

    conclusion = zero_status(expression)
    if conclusion is True:
        return ((),), True
    if conclusion is False:
        return (), True

    if isinstance(expression, sp.Symbol):
        for species_id, symbol in case.states.concentrations.items():
            if expression == symbol:
                bounds = case.state_bounds[symbol]
                if bounds.physical_lower <= 0 <= bounds.physical_upper:
                    return ((species_id,),), True
                return (), True
        bounds = case.state_bounds[expression]
        if bounds.physical_lower > 0 or bounds.physical_upper < 0:
            return (), True
        return (), False

    if expression.func is sp.exp:
        return (), True

    if expression.func is sp.Abs:
        return _coordinate_zero_faces(expression.args[0], case)

    if expression.func is sp.Max and sp.S.Zero in expression.args:
        other_arguments = tuple(
            arg for arg in expression.args if arg != sp.S.Zero
        )
        if len(other_arguments) == 1 and isinstance(other_arguments[0], sp.Symbol):
            symbol = other_arguments[0]
            species_id = next(
                (
                    name
                    for name, concentration in case.states.concentrations.items()
                    if concentration == symbol
                ),
                None,
            )
            if species_id is not None:
                bounds = case.state_bounds[symbol]
                if bounds.physical_lower >= 0:
                    return ((species_id,),), True
                if bounds.physical_upper <= 0:
                    return ((),), True
        return (), False

    if isinstance(expression, sp.Pow):
        base, exponent = expression.args
        if exponent.is_negative is True:
            return (), True
        if exponent.is_positive is True:
            return _coordinate_zero_faces(base, case)
        return (), False

    if isinstance(expression, sp.Mul):
        faces: list[tuple[str, ...]] = []
        complete = True
        for factor in expression.args:
            factor_faces, factor_complete = _coordinate_zero_faces(factor, case)
            faces.extend(factor_faces)
            complete = complete and factor_complete
        if () in faces:
            return ((),), complete
        return tuple(dict.fromkeys(faces)), complete

    return (), False


def _single_reaction_equilibria(case: Case) -> EquilibriumSearchResult | None:
    """Solve structurally when one rate has a complete coordinate zero set."""

    if len(case.reactions) != 1:
        return None
    reaction = case.reactions[0]
    if not reaction.net_stoichiometry:
        return None

    faces, complete = _coordinate_zero_faces(reaction.rate, case)
    if not complete:
        return None

    families = tuple(
        equilibrium_family(
            tuple(
                sp.S.Zero
                if species_id in face
                else case.states.concentration(species_id)
                for species_id in case.states.species_ids
            ),
            case,
        )
        for face in faces
    )
    return EquilibriumSearchResult(families, 0, True, None)


def _nonlinsolve_worker(
    connection: Any,
    equations: tuple[sp.Expr, ...],
    symbols: tuple[sp.Symbol, ...],
) -> None:
    """Solve in an expendable process because SymPy cannot be interrupted."""

    try:
        solution_set = sp.nonlinsolve(equations, symbols)
        connection.send(("solution", solution_set))
    except BaseException as error:  # pragma: no cover - defensive process boundary
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _process_context() -> mp.context.BaseContext:
    """Prefer fork for low startup cost, with a portable spawn fallback."""

    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    return mp.get_context(method)


def _terminate_process(process: mp.Process) -> None:
    if not process.is_alive():
        process.join()
        return
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():  # pragma: no cover - terminate is sufficient normally
        process.kill()
        process.join()


def solve_with_timeout(
    equations: tuple[sp.Expr, ...],
    symbols: tuple[sp.Symbol, ...],
    timeout_seconds: float,
    *,
    _worker: Any = _nonlinsolve_worker,
) -> tuple[sp.Set | None, str | None]:
    """Run ``nonlinsolve`` with a hard wall-clock limit."""

    context = _process_context()
    receiving_connection, sending_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(sending_connection, equations, symbols),
        daemon=True,
    )
    process.start()
    sending_connection.close()

    try:
        if not receiving_connection.poll(timeout_seconds):
            _terminate_process(process)
            return None, f"Solver timed out after {timeout_seconds:g} seconds"
        try:
            kind, payload = receiving_connection.recv()
        except EOFError:
            _terminate_process(process)
            return None, "Solver process exited without returning a result"
    finally:
        receiving_connection.close()

    process.join(timeout=1.0)
    if process.is_alive():
        _terminate_process(process)
    if kind == "error":
        return None, payload
    return payload, None


def find_equilibria(
    case: Case,
    source_terms: Mapping[str, sp.Expr],
) -> EquilibriumSearchResult:
    """Attempt generic symbolic equilibrium discovery for at most ten seconds."""

    symbols = tuple(case.states.concentration(name) for name in case.states.species_ids)
    equations = tuple(
        dict.fromkeys(
            expression
            for expression in source_terms.values()
            if zero_status(expression) is not True
        )
    )

    if not equations:
        return EquilibriumSearchResult(
            equilibria=(equilibrium_family(symbols, case),),
            excluded=0,
            complete=True,
            diagnostic=None,
        )

    structural_result = _single_reaction_equilibria(case)
    if structural_result is not None:
        return structural_result

    solution_set, diagnostic = solve_with_timeout(
        equations,
        symbols,
        EQUILIBRIUM_SOLVE_TIMEOUT_SECONDS,
    )
    if diagnostic is not None:
        return EquilibriumSearchResult((), 0, False, diagnostic)
    assert solution_set is not None

    if solution_set is sp.EmptySet:
        return EquilibriumSearchResult((), 0, True, None)
    if not isinstance(solution_set, sp.FiniteSet):
        return EquilibriumSearchResult(
            (),
            0,
            False,
            f"Solver returned unresolved set: {solution_set}",
        )

    equilibria: list[EquilibriumFamily] = []
    excluded = 0
    for point in solution_set:
        if not isinstance(point, (tuple, sp.Tuple)) or len(point) != len(symbols):
            return EquilibriumSearchResult(
                tuple(equilibria),
                excluded,
                False,
                f"Solver returned an unsupported solution: {point}",
            )
        if any(expression is sp.EmptySet for expression in point):
            continue
        if any(isinstance(expression, sp.Set) for expression in point):
            return EquilibriumSearchResult(
                tuple(equilibria),
                excluded,
                False,
                f"Solver returned an unresolved solution: {point}",
            )
        family = equilibrium_family(tuple(point), case)
        if family.physical is False:
            excluded += 1
        else:
            equilibria.append(family)
    return EquilibriumSearchResult(tuple(equilibria), excluded, True, None)
