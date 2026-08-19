"""Temporary regularity adapter used until the Phase 5 engine."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache

import sympy as sp
from sympy.core.relational import Relational

from ..context import AnalysisContext
from ..domain import ConcentrationDomain
from ..model import Reaction
from ..results import Evidence, Finding, Verdict
from .symbolic_domain import POSITIVE, Point, proof_for_domain

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


def _condition(expression: sp.Expr, requirement: str) -> Relational:
    if requirement == POSITIVE:
        return sp.Gt(expression, 0, evaluate=False)
    return sp.Ne(expression, 0, evaluate=False)


def check_lipschitz_continuity(
    reaction: Reaction,
    domain: ConcentrationDomain,
) -> LipschitzContinuityResult:
    """Certify a rate on an open neighbourhood of its augmented domain.

    The selected gas-closure policy chamfers the augmented concentration box.
    Strict domain conditions are proved on the closure of that domain.  Under
    the ``positive`` policy this includes the limiting zero-total boundary;
    under ``ideal_gas`` it instead includes the derived positive lower face.
    This certifies a uniform margin and rejects expressions whose derivatives
    or values become unbounded as an excluded boundary is approached.
    """

    missing_bounds = reaction.rate.free_symbols - set(domain.all_intervals)
    if missing_bounds:
        names = ", ".join(sorted(map(str, missing_bounds)))
        raise ValueError(f"Reaction rate has no bounds for symbols: {names}.")
    requirements, unsupported = _structure(reaction.rate)
    proof = proof_for_domain(domain)
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


def _evidence(result: LipschitzContinuityResult) -> Evidence | None:
    if result.counterexample is None:
        return None
    return Evidence(
        "exact_counterexample",
        {str(symbol): str(value) for symbol, value in result.counterexample.items()},
    )


def _lipschitz_finding(
    result: LipschitzContinuityResult,
    domain_name: str,
) -> Finding:
    if result.passed is False:
        if result.defined is False:
            summary = f"Rate is not real and finite on the {domain_name} domain."
        else:
            summary = (
                "A strict domain condition loses its margin on the "
                f"{domain_name} boundary."
            )
        return Finding(
            result.reaction_id, Verdict.FAIL, summary, _evidence(result)
        )

    if result.passed:
        return Finding(
            result.reaction_id,
            Verdict.PASS,
            f"Rate is Lipschitz on a neighbourhood of the {domain_name} domain.",
        )

    details: list[str] = []
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
    return Finding(
        result.reaction_id,
        Verdict.UNKNOWN,
        " ".join(details) or "Regularity analysis was inconclusive.",
    )


def _definedness_finding(
    result: LipschitzContinuityResult,
    domain_name: str,
) -> Finding:
    if result.defined is False:
        return Finding(
            result.reaction_id,
            Verdict.FAIL,
            f"Rate is not real and finite on the {domain_name} domain.",
            _evidence(result),
        )
    if result.defined is True:
        return Finding(
            result.reaction_id,
            Verdict.PASS,
            f"Rate is real and finite on the {domain_name} domain.",
        )
    return Finding(
        result.reaction_id,
        Verdict.UNKNOWN,
        f"Rate definedness on the {domain_name} domain was inconclusive.",
    )


def _run(
    context: AnalysisContext,
    domain: ConcentrationDomain,
    domain_name: str,
    *,
    definedness_only: bool,
) -> tuple[Finding, ...]:
    results = tuple(
        context.rate_facts(reaction, domain) for reaction in context.case.reactions
    )
    convert = _definedness_finding if definedness_only else _lipschitz_finding
    return tuple(convert(result, domain_name) for result in results)


def run_physical_definedness(
    context: AnalysisContext,
    _dependencies: Mapping,
) -> tuple[Finding, ...]:
    return _run(context, context.physical_domain, "physical", definedness_only=True)


def run_physical_lipschitz(
    context: AnalysisContext,
    _dependencies: Mapping,
) -> tuple[Finding, ...]:
    return _run(context, context.physical_domain, "physical", definedness_only=False)


def run_augmented_definedness(
    context: AnalysisContext,
    _dependencies: Mapping,
) -> tuple[Finding, ...]:
    return _run(context, context.augmented_domain, "augmented", definedness_only=True)


def run_augmented_lipschitz(
    context: AnalysisContext,
    _dependencies: Mapping,
) -> tuple[Finding, ...]:
    return _run(context, context.augmented_domain, "augmented", definedness_only=False)
