"""Simplification-free concentration-face discovery."""

from collections.abc import Mapping
from dataclasses import dataclass
import heapq

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
from .network import NetworkExpressions, network_expressions, source_terms as _source_terms

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


class _ZeroProver:
    """Reuse physical assumptions, witnesses, and conclusions across faces."""

    def __init__(self, case: Case) -> None:
        self.replacements = {
            symbol: sp.Dummy(
                symbol.name,
                **(
                    {"positive": True}
                    if bounds.physical_lower >= 0
                    else {"negative": True}
                    if bounds.physical_upper <= 0
                    else {"real": True}
                ),
            )
            for symbol, bounds in case.state_bounds.items()
        }
        self.witnesses = tuple(
            _witness_values(case, fraction) for fraction in _WITNESS_FRACTIONS
        )
        self.cache: dict[sp.Expr, bool | None] = {}

    def status(self, expression: sp.Expr) -> bool | None:
        cached = self.cache.get(expression, ...)
        if cached is not ...:
            return cached

        conclusion = zero_status(expression)
        if conclusion is None:
            physical_expression = expression.xreplace(self.replacements)
            if zero_status(physical_expression) is False:
                conclusion = False
            else:
                for witness in self.witnesses:
                    if zero_status(expression.xreplace(witness)) is False:
                        conclusion = False
                        break
        self.cache[expression] = conclusion
        return conclusion


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


class _FaceEvaluator:
    """Restrict shared expressions one symbol at a time and cache every face."""

    def __init__(
        self,
        case: Case,
        source_terms: Mapping[str, sp.Expr],
        *,
        use_reaction_rates: bool,
    ) -> None:
        self.case = case
        self.prover = _ZeroProver(case)
        self.symbols = case.states.concentrations
        self.symbol_order = {
            symbol: index
            for index, symbol in enumerate(case.states.concentrations.values())
        }
        self.original_source = source_terms
        eligible_symbols = {
            case.states.concentration(species_id)
            for species_id in case.states.species_ids
            if not case.state_bounds[
                case.states.concentration(species_id)
            ].strict_lower
        }
        self.proof_order = tuple(
            sorted(
                case.states.species_ids,
                key=lambda species_id: (
                    len(source_terms[species_id].free_symbols & eligible_symbols),
                    source_terms[species_id] is not sp.S.Zero,
                ),
            )
        )
        self.expression_cache: dict[tuple[frozenset[str], str], sp.Expr] = {
            (frozenset(), species_id): expression
            for species_id, expression in source_terms.items()
        }
        self.rate_cache: dict[tuple[frozenset[str], int], sp.Expr] | None = None
        self.rate_zeroers: tuple[frozenset[str], ...] = ()
        if use_reaction_rates:
            self.rate_cache = {
                (frozenset(), index): reaction.rate
                for index, reaction in enumerate(case.reactions)
            }
            self.rate_zeroers = tuple(
                frozenset(
                    species_id
                    for species_id in dict.fromkeys(
                        (*reaction.reactants, *reaction.catalysts)
                    )
                    if not case.state_bounds[
                        case.states.concentration(species_id)
                    ].strict_lower
                    and zero_status(
                        reaction.rate.xreplace(
                            {case.states.concentration(species_id): sp.S.Zero}
                        )
                    )
                    is True
                )
                for reaction in case.reactions
            )

    def _parent(self, depleted: frozenset[str]) -> tuple[frozenset[str], str]:
        species_id = max(
            depleted,
            key=lambda item: self.symbol_order[self.symbols[item]],
        )
        return depleted - {species_id}, species_id

    def _restricted_rate(self, depleted: frozenset[str], index: int) -> sp.Expr:
        assert self.rate_cache is not None
        key = depleted, index
        cached = self.rate_cache.get(key)
        if cached is not None:
            return cached
        parent, species_id = self._parent(depleted)
        rate = self._restricted_rate(parent, index)
        symbol = self.symbols[species_id]
        restricted = (
            sp.S.Zero
            if species_id in self.rate_zeroers[index]
            else rate
            if rate is sp.S.Zero or not rate.has(symbol)
            else rate.xreplace({symbol: sp.S.Zero})
        )
        self.rate_cache[key] = restricted
        return restricted

    def expression(self, depleted: frozenset[str], species_id: str) -> sp.Expr:
        key = depleted, species_id
        cached = self.expression_cache.get(key)
        if cached is not None:
            return cached

        if self.rate_cache is None:
            parent, depleted_species = self._parent(depleted)
            expression = self.expression(parent, species_id)
            symbol = self.symbols[depleted_species]
            restricted = (
                expression
                if expression is sp.S.Zero or not expression.has(symbol)
                else expression.xreplace({symbol: sp.S.Zero})
            )
        else:
            restricted = sp.Add(
                *(
                    sp.Rational(
                        str(
                            reaction.net_stoichiometry.get(
                                species_id,
                                0,
                            )
                        )
                    )
                    * self._restricted_rate(depleted, index)
                    for index, reaction in enumerate(self.case.reactions)
                    if reaction.net_stoichiometry.get(species_id, 0)
                    and self._restricted_rate(depleted, index) is not sp.S.Zero
                )
            )
        self.expression_cache[key] = restricted
        return restricted

    def source(self, depleted: frozenset[str]) -> Mapping[str, sp.Expr]:
        return {
            species_id: self.expression(depleted, species_id)
            for species_id in self.case.states.species_ids
        }


def _face_key(face: frozenset[str], order: Mapping[str, int]) -> tuple[object, ...]:
    indices = tuple(sorted(order[item] for item in face))
    return len(face), indices


def _search_faces(
    evaluator: _FaceEvaluator,
    species_ids: tuple[str, ...],
    *,
    invariant: bool,
    evaluated: set[frozenset[str]],
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]], bool]:
    """Find minimal depleted sets with exact dependency-directed branching."""

    order = {species_id: index for index, species_id in enumerate(species_ids)}
    initial = (
        (frozenset((species_id,)) for species_id in species_ids)
        if invariant
        else (frozenset(),)
    )
    queue: list[tuple[tuple[object, ...], frozenset[str]]] = []
    enqueued: set[frozenset[str]] = set()
    for face in initial:
        heapq.heappush(queue, (_face_key(face, order), face))
        enqueued.add(face)

    found: list[tuple[str, ...]] = []
    unresolved: list[tuple[str, ...]] = []
    while queue:
        _, face = heapq.heappop(queue)
        if _has_known_parent(
            tuple(sorted(face, key=order.__getitem__)),
            found,
        ):
            continue
        if face not in evaluated:
            if len(evaluated) >= MAX_FACE_TESTS:
                return found, unresolved, False
            evaluated.add(face)

        relevant = tuple(
            species_id
            for species_id in evaluator.proof_order
            if not invariant or species_id in face
        )
        unresolved_identity = False
        failed_expression = None
        for species_id in relevant:
            expression = evaluator.expression(face, species_id)
            status = evaluator.prover.status(expression)
            if status is False:
                failed_expression = expression
                break
            if status is None:
                unresolved_identity = True
        conclusion = (
            False
            if failed_expression is not None
            else None
            if unresolved_identity
            else True
        )
        ordered_face = tuple(sorted(face, key=order.__getitem__))
        if conclusion is True:
            found.append(ordered_face)
            continue
        if conclusion is None:
            unresolved.append(ordered_face)

        if failed_expression is not None:
            # Every terminal/invariant descendant must eliminate each failed
            # expression. Its remaining dependencies are therefore a complete
            # set of branches for every possible descendant.
            dependencies = {
                item
                for item in species_ids
                if item not in face
                and evaluator.symbols[item] in failed_expression.free_symbols
            }
        else:
            # An unresolved identity supplies no sound necessary variable.
            # Fall back to every extension so completeness is not weakened.
            dependencies = {item for item in species_ids if item not in face}

        for species_id in sorted(dependencies, key=order.__getitem__):
            child = face | {species_id}
            if child in enqueued:
                continue
            enqueued.add(child)
            heapq.heappush(queue, (_face_key(child, order), child))

    found.sort(key=lambda face: (len(face), tuple(order[item] for item in face)))
    unresolved.sort(key=lambda face: (len(face), tuple(order[item] for item in face)))
    return found, unresolved, True


def find_terminal_faces(
    case: Case,
    source_terms: Mapping[str, sp.Expr],
    *,
    use_reaction_rates: bool = False,
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
    evaluator = _FaceEvaluator(
        case,
        source_terms,
        use_reaction_rates=use_reaction_rates,
    )
    evaluated: set[frozenset[str]] = set()
    terminal_faces, unresolved_terminal, terminal_complete = _search_faces(
        evaluator,
        species_ids,
        invariant=False,
        evaluated=evaluated,
    )
    if terminal_faces == [()] and not unresolved_terminal:
        return TerminalFaceSearchResult(
            terminal_faces=((),),
            invariant_faces=((),),
            unresolved_terminal_faces=(),
            unresolved_invariant_faces=(),
            complete=terminal_complete,
            tests=len(evaluated),
        )
    invariant_faces, unresolved_invariant, invariant_complete = _search_faces(
        evaluator,
        species_ids,
        invariant=True,
        evaluated=evaluated,
    )

    return TerminalFaceSearchResult(
        terminal_faces=tuple(terminal_faces),
        invariant_faces=tuple(invariant_faces),
        unresolved_terminal_faces=tuple(unresolved_terminal),
        unresolved_invariant_faces=tuple(unresolved_invariant),
        complete=terminal_complete and invariant_complete,
        tests=len(evaluated),
    )


def check_terminal_faces(
    case: Case,
    network: NetworkExpressions | None = None,
) -> TerminalFaceSearchResult:
    """Locate maximal terminal and invariant concentration faces."""

    return find_terminal_faces(
        case,
        network.source_terms if network is not None else _source_terms(case),
        use_reaction_rates=True,
    )


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

    network = context.cached(
        case,
        "network",
        lambda: network_expressions(case),
    )
    return _outcome(check_terminal_faces(case, network), case.states.species_ids)


CHECK = CheckDefinition(
    id="terminal_faces",
    name="Terminal faces",
    group="Network analysis",
    scope=CheckScope.CASE,
    run=run,
)
