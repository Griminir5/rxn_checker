"""Certify rate continuity on a neighbourhood of the augmented domain."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType

import sympy as sp
from sympy.core.relational import Relational

from ..case import Case
from ..reaction import Reaction
from ..state import IdealGasClosure, VariableBounds
from .models import (
    CheckContext,
    CheckDefinition,
    CheckOutcome,
    CheckScope,
    CheckStatus,
)
from .symbolic_domain import POSITIVE, LinearDomain, Point, Proof, number

_NONZERO = "nonzero"
_SAFE_FUNCTIONS = frozenset(
    (
        sp.Abs,
        sp.Max,
        sp.Min,
        sp.cos,
        sp.cosh,
        sp.exp,
        sp.sin,
        sp.sinh,
        sp.tanh,
        sp.atan,
        sp.asinh,
    )
)


@dataclass(frozen=True)
class LipschitzContinuityResult:
    """Symbolic conclusion for one rate on its augmented state domain."""

    reaction_id: str
    passed: bool | None
    rate: sp.Expr
    defined: bool | None
    conditions: tuple[Relational, ...] = ()
    unresolved_conditions: tuple[Relational, ...] = ()
    unsupported_functions: tuple[str, ...] = ()
    counterexample: Point | None = None


@cache
def _structure(
    expression: sp.Expr,
) -> tuple[tuple[tuple[sp.Expr, str], ...], tuple[str, ...]]:
    """Return strict domain conditions for supported Lipschitz primitives."""

    requirements: list[tuple[sp.Expr, str]] = []
    unsupported: set[str] = set()
    visited: set[sp.Expr] = set()

    def visit(item: sp.Expr) -> None:
        if item in visited:
            return
        visited.add(item)

        if item.is_Atom:
            if item.is_real is not True or item.is_finite is not True:
                unsupported.add(item.func.__name__)
            return
        if isinstance(item, sp.Pow):
            base, exponent = item.args
            if exponent.is_integer is not True:
                requirements.append((base, POSITIVE))
            elif exponent.is_negative:
                requirements.append((base, _NONZERO))
        elif item.func is sp.log:
            requirements.append((item.args[0], POSITIVE))
        elif (
            not isinstance(item, (sp.Add, sp.Mul))
            and item.func not in _SAFE_FUNCTIONS
        ):
            unsupported.add(item.func.__name__)

        for argument in item.args:
            if isinstance(argument, sp.Expr):
                visit(argument)

    visit(sp.sympify(expression))
    return tuple(dict.fromkeys(requirements)), tuple(sorted(unsupported))


@cache
def _cached_augmented_domain(
    items: tuple[tuple[sp.Symbol, VariableBounds], ...],
    closure_constraints: tuple[Relational, ...],
    open_constraints: tuple[sp.Expr, ...],
) -> tuple[LinearDomain, Mapping[sp.Symbol, sp.Symbol], Point]:
    variables = tuple(symbol for symbol, _ in items)
    weak: list[Relational] = []
    bounds: list[tuple[sp.Expr, sp.Expr]] = []
    assumptions: dict[sp.Symbol, sp.Symbol] = {}

    for symbol, state_bound in items:
        raw_lower, raw_upper = state_bound.interval(include_excursion=True)
        lower, upper = number(raw_lower), number(raw_upper)
        weak.extend(
            (
                sp.Ge(symbol - lower, 0, evaluate=False),
                sp.Ge(upper - symbol, 0, evaluate=False),
            )
        )
        bounds.append((lower, upper))

        if lower > 0:
            kind = "positive"
        elif upper < 0:
            kind = "negative"
        elif lower == 0:
            kind = "nonnegative"
        elif upper == 0:
            kind = "nonpositive"
        else:
            kind = "real"
        assumptions[symbol] = sp.Dummy(symbol.name, **{kind: True})

    # Uniform Lipschitz continuity on an open domain requires a finite margin
    # as its excluded boundary is approached.  Analyze the closure of the
    # chamfer by weakening its strict constraints here.
    weak.extend(closure_constraints)
    weak.extend(
        sp.Ge(expression, 0, evaluate=False) for expression in open_constraints
    )

    domain = LinearDomain(
        variables,
        tuple(weak),
        (),
        tuple(bounds),
    )
    feasible, feasible_point = domain.feasible()
    if not feasible or feasible_point is None:
        raise ValueError("Augmented state domain is empty.")
    return domain, MappingProxyType(assumptions), feasible_point


def _augmented_domain(
    state_bounds: Mapping[sp.Symbol, VariableBounds],
    gas_closure: IdealGasClosure | None,
) -> tuple[LinearDomain, Mapping[sp.Symbol, sp.Symbol], Point]:
    closure_constraints: tuple[Relational, ...] = ()
    open_constraints: tuple[sp.Expr, ...] = ()
    if gas_closure is not None:
        minimum = gas_closure.derived_minimum_total(state_bounds)
        if minimum is None:
            open_constraints = gas_closure.augmented_strict_constraints
        else:
            closure_constraints = (
                sp.Ge(
                    gas_closure.total_concentration - minimum,
                    0,
                    evaluate=False,
                ),
            )
    return _cached_augmented_domain(
        tuple(state_bounds.items()),
        closure_constraints,
        open_constraints,
    )


def _condition(expression: sp.Expr, requirement: str) -> Relational:
    if requirement == POSITIVE:
        return sp.Gt(expression, 0, evaluate=False)
    return sp.Ne(expression, 0, evaluate=False)


def check_lipschitz_continuity(
    reaction: Reaction,
    state_bounds: Mapping[sp.Symbol, VariableBounds],
    gas_closure: IdealGasClosure | None = None,
) -> LipschitzContinuityResult:
    """Certify a rate on an open neighbourhood of its augmented domain.

    The selected gas-closure policy chamfers the augmented concentration box.
    Strict domain conditions are proved on the closure of that domain.  Under
    the ``positive`` policy this includes the limiting zero-total boundary;
    under ``ideal_gas`` it instead includes the derived positive lower face.
    This certifies a uniform margin and rejects expressions whose derivatives
    or values become unbounded as an excluded boundary is approached.
    """

    missing_bounds = reaction.rate.free_symbols - set(state_bounds)
    if missing_bounds:
        names = ", ".join(sorted(map(str, missing_bounds)))
        raise ValueError(f"Reaction rate has no bounds for symbols: {names}.")
    domain_symbols = set(reaction.rate.free_symbols)
    active_closure = None
    if gas_closure is not None and domain_symbols.intersection(
        gas_closure.gas_concentrations
    ):
        active_closure = gas_closure
        domain_symbols.update(gas_closure.gas_concentrations)
        domain_symbols.update((gas_closure.temperature, gas_closure.pressure))
    rate_bounds = {
        symbol: state_bound
        for symbol, state_bound in state_bounds.items()
        if symbol in domain_symbols
    }
    domain, assumptions, point = _augmented_domain(rate_bounds, active_closure)
    requirements, unsupported = _structure(reaction.rate)
    proof = Proof(domain, assumptions, point)
    conditions = tuple(_condition(*requirement) for requirement in requirements)
    invalid_atom = any(
        atom is sp.nan or atom.is_real is False or atom.is_finite is False
        for atom in reaction.rate.atoms()
    )
    if invalid_atom:
        defined, counterexample = False, None
    else:
        defined, counterexample = proof.defined(reaction.rate)

    passed: bool | None
    if defined is False:
        unresolved = ()
        passed = False
    else:
        unresolved_items: list[Relational] = []
        margin_counterexample = None
        for requirement, condition in zip(requirements, conditions):
            if proof.proves(*requirement) is True:
                continue
            witness = proof.violating_point(*requirement)
            if witness is not None:
                margin_counterexample = witness
                break
            unresolved_items.append(condition)
        unresolved = tuple(unresolved_items)
        if margin_counterexample is not None:
            passed = False
            counterexample = margin_counterexample
        elif not unresolved and not unsupported:
            defined, passed = True, True
        else:
            passed = None

    return LipschitzContinuityResult(
        reaction.id,
        passed,
        reaction.rate,
        defined,
        conditions,
        unresolved,
        unsupported,
        counterexample,
    )


def _point_detail(point: Point) -> str:
    values = ", ".join(f"{symbol}={value}" for symbol, value in point.items())
    return f"Exact augmented-domain closure witness: {values}."


def _outcome(result: LipschitzContinuityResult) -> CheckOutcome:
    if result.passed is False:
        if result.defined is False:
            details = [
                "Rate is not real and finite on the closure of the augmented "
                "domain."
            ]
        else:
            details = [
                "A strict domain condition loses its margin on the boundary "
                "of the augmented domain."
            ]
        if result.counterexample is not None:
            details.append(_point_detail(result.counterexample))
        return CheckOutcome(
            status=CheckStatus.FAIL,
            subject=result.reaction_id,
            details=tuple(details),
        )

    if result.passed:
        return CheckOutcome(
            status=CheckStatus.PASS,
            subject=result.reaction_id,
            details=(
                "Rate is Lipschitz continuous on an open neighbourhood of "
                "the augmented domain.",
            ),
        )

    details = []
    if result.defined is None:
        details.append(
            "Could not prove that the rate is real and finite throughout the "
            "augmented domain."
        )
    if result.unresolved_conditions:
        rendered = ", ".join(map(sp.sstr, result.unresolved_conditions))
        details.append(f"Unresolved strict domain conditions: {rendered}.")
    if result.unsupported_functions:
        details.append(
            "Unsupported symbolic functions: "
            + ", ".join(result.unsupported_functions)
            + "."
        )
    return CheckOutcome(
        status=CheckStatus.INDETERMINATE,
        subject=result.reaction_id,
        details=tuple(details),
    )


def run(case: Case, context: CheckContext) -> tuple[CheckOutcome, ...]:
    """Check every rate separately so singularities cannot cancel in ``S r``."""

    return tuple(
        _outcome(
            check_lipschitz_continuity(
                reaction,
                case.state_bounds,
                case.gas_closure,
            )
        )
        for reaction in case.reactions
    )


CHECK = CheckDefinition(
    id="lipschitz_continuity",
    name="Lipschitz continuity",
    group="Numerical robustness",
    scope=CheckScope.REACTION,
    run=run,
)
