"""Symbolically check recovery from small negative concentrations."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations, islice
from types import MappingProxyType

import sympy as sp
from sympy.core.relational import Relational

from ..case import Case
from .models import (
    CheckContext,
    CheckDefinition,
    CheckOutcome,
    CheckScope,
    CheckStatus,
    CheckValue,
)
from .network import NetworkExpressions, network_expressions
from .stoichiometric_conservation import (
    StoichiometricConservationResult,
    find_conserved_quantities,
)
from .symbolic_domain import (
    NEGATIVE,
    NONNEGATIVE,
    NONPOSITIVE,
    POSITIVE,
    ZERO,
    LinearDomain,
    Point,
    Proof,
    number,
)

MAX_NEGATIVE_REGIONS = 128
MAX_SYMBOLIC_OPERATIONS = 5000
MAX_TOTAL_SYMBOLIC_OPERATIONS = 15000


class RecoveryVerdict(StrEnum):
    """Symbolic conclusion for one negative-species region."""

    STRONGLY_RESTORING = "STRONGLY_RESTORING"
    NET_RESTORING = "NET_RESTORING"
    NON_WORSENING = "NON_WORSENING"
    STUCK = "STUCK"
    WORSENING = "WORSENING"
    UNDEFINED_IN_EXTENSION = "UNDEFINED_IN_EXTENSION"
    STOICHIOMETRICALLY_UNREPAIRABLE = "STOICHIOMETRICALLY_UNREPAIRABLE"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class RecoveryRegionResult:
    """Recovery conclusion on one concentration sign-region."""

    negative_species: tuple[str, ...]
    verdict: RecoveryVerdict
    constraints: tuple[Relational, ...]
    restoration: sp.Expr | None = None
    componentwise: Mapping[str, bool | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    lower_faces: Mapping[str, bool | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    defined: bool | None = None
    decisive_expression: sp.Expr | None = None
    counterexample: Point | None = None

    @property
    def certified(self) -> bool:
        if self.verdict is RecoveryVerdict.STOICHIOMETRICALLY_UNREPAIRABLE:
            return True
        return (
            self.verdict
            in (
                RecoveryVerdict.STRONGLY_RESTORING,
                RecoveryVerdict.NET_RESTORING,
            )
            and self.defined is True
            and all(result is True for result in self.lower_faces.values())
        )


@dataclass(frozen=True)
class NonphysicalRecoveryResult:
    """Network source and all checked negative-species regions."""

    source_terms: Mapping[str, sp.Expr]
    conservation_rays: tuple[sp.Expr, ...]
    regions: tuple[RecoveryRegionResult, ...]
    complete: bool
    tests: int
    diagnostic: str | None = None


def _excursion_size(case: Case, species_id: str) -> sp.Rational:
    symbol = case.states.concentration(species_id)
    lower = case.state_bounds[symbol].excursion_lower
    if lower is None or lower >= 0:
        raise ValueError(f"Species '{species_id}' has no negative excursion.")
    return -number(lower)


def _conservation_rays(
    case: Case,
    conservation: StoichiometricConservationResult,
) -> tuple[sp.Expr, ...]:
    rays = [
        sp.Add(
            *(
                coefficient * case.states.concentration(species_id)
                for species_id, coefficient in quantity.coefficients.items()
            )
        )
        for component in conservation.components
        for quantity in component.extreme_rays
    ]
    rays.extend(
        case.states.concentration(species_id)
        for species_id in conservation.unchanged_species
    )
    return tuple(rays)


def _region_domain(
    case: Case,
    negative_species: tuple[str, ...],
    conservation_rays: tuple[sp.Expr, ...],
) -> LinearDomain:
    negative = frozenset(negative_species)
    species = {
        symbol: species_id
        for species_id, symbol in case.states.concentrations.items()
    }
    weak: list[Relational] = []
    strict: list[sp.Expr] = []
    box: list[tuple[sp.Expr, sp.Expr]] = []

    for symbol, bounds in case.state_bounds.items():
        species_id = species.get(symbol)
        if species_id in negative:
            lower = -_excursion_size(case, species_id)
            box_lower = lower
            weak.append(sp.Ge(symbol - lower, 0, evaluate=False))
            strict.append(-symbol)
        elif bounds.strict_lower:
            box_lower = number(bounds.physical_lower)
            strict.append(symbol - number(bounds.physical_lower))
        else:
            box_lower = number(bounds.physical_lower)
            weak.append(
                sp.Ge(symbol - number(bounds.physical_lower), 0, evaluate=False)
            )
        weak.append(
            sp.Ge(number(bounds.physical_upper) - symbol, 0, evaluate=False)
        )
        box.append((box_lower, number(bounds.physical_upper)))
    weak.extend(sp.Ge(ray, 0, evaluate=False) for ray in conservation_rays)
    return LinearDomain(
        tuple(case.state_bounds),
        tuple(weak),
        tuple(strict),
        tuple(box),
    )


def _negative_sets(
    case: Case,
    conservation: StoichiometricConservationResult,
) -> tuple[tuple[tuple[str, ...], ...], bool]:
    eligible = {
        species_id
        for species_id, symbol in case.states.concentrations.items()
        if (lower := case.state_bounds[symbol].excursion_lower) is not None
        and lower < 0
    }
    groups = [
        tuple(
            species_id
            for species_id in component.species_ids
            if species_id in eligible
        )
        for component in conservation.components
    ]
    groups.extend(
        (species_id,)
        for species_id in conservation.unchanged_species
        if species_id in eligible
    )

    candidates = (
        candidate
        for group in groups
        for size in range(1, len(group) + 1)
        for candidate in combinations(group, size)
    )
    count = sum(2 ** len(group) - 1 for group in groups)
    selected = tuple(islice(candidates, MAX_NEGATIVE_REGIONS))
    return selected, count <= MAX_NEGATIVE_REGIONS


def _assumptions(
    case: Case,
    negative_species: tuple[str, ...],
    domain: LinearDomain,
) -> Mapping[sp.Symbol, sp.Symbol]:
    negative = frozenset(negative_species)
    feasible, point = domain.feasible()
    if not feasible:
        point = None
    replacements = {}
    for symbol, bounds in case.state_bounds.items():
        if symbol.name in negative:
            kind = "negative"
        elif bounds.physical_lower > 0 or bounds.strict_lower:
            kind = "positive"
        elif bounds.physical_lower == 0:
            # The phase-I witness often places optional species exactly at
            # zero. That is already an exact feasibility certificate, avoiding
            # a separate simplex solve for the same question.
            can_be_zero = point is not None and point[symbol] == 0
            if not can_be_zero:
                can_be_zero = domain.feasible(
                    weak=(sp.Le(symbol, 0, evaluate=False),)
                )[0]
            kind = "nonnegative" if can_be_zero else "positive"
        elif bounds.physical_upper <= 0:
            kind = "nonpositive"
        else:
            kind = "real"
        replacements[symbol] = sp.Dummy(symbol.name, **{kind: True})
    return MappingProxyType(replacements)


def _analyse_region(
    case: Case,
    negative_species: tuple[str, ...],
    source_terms: Mapping[str, sp.Expr],
    conservation_rays: tuple[sp.Expr, ...],
) -> RecoveryRegionResult:
    domain = _region_domain(case, negative_species, conservation_rays)
    feasible, raw_point = domain.feasible()
    if not feasible or raw_point is None:
        return RecoveryRegionResult(
            negative_species,
            RecoveryVerdict.STOICHIOMETRICALLY_UNREPAIRABLE,
            domain.constraints,
        )

    point = raw_point
    assumptions = _assumptions(case, negative_species, domain)
    proof = Proof(domain, assumptions, point)
    # A finite real linear combination of finite real rates is necessarily
    # finite and real, so traversing every much-larger source expression would
    # repeat the rate-law proof without adding a condition.
    expressions = tuple(reaction.rate for reaction in case.reactions)
    defined: bool | None = True
    for expression in expressions:
        conclusion, bad_point = proof.defined(expression)
        if conclusion is False:
            return RecoveryRegionResult(
                negative_species,
                RecoveryVerdict.UNDEFINED_IN_EXTENSION,
                domain.constraints,
                defined=False,
                decisive_expression=expression,
                counterexample=bad_point,
            )
        if conclusion is None:
            defined = None

    restoration = sp.Add(
        *(
            -case.states.concentration(species_id)
            * source_terms[species_id]
            / _excursion_size(case, species_id) ** 2
            for species_id in negative_species
        )
    )
    restoration_sign = proof.sign(restoration)
    point_restoration_sign = proof.value_sign(restoration)
    componentwise = MappingProxyType(
        {
            species_id: proof.proves(source_terms[species_id], POSITIVE)
            for species_id in negative_species
        }
    )

    lower_faces: dict[str, bool | None] = {}
    lower_counterexample = None
    lower_expression = None
    for species_id in negative_species:
        symbol = case.states.concentration(species_id)
        lower = -_excursion_size(case, species_id)
        face = domain.equality(symbol - lower)
        face_feasible, raw_face_point = face.feasible()
        if not face_feasible or raw_face_point is None:
            lower_faces[species_id] = True
            continue
        face_point = raw_face_point
        face_proof = Proof(
            face,
            _assumptions(case, negative_species, face),
            face_point,
        )
        expression = source_terms[species_id].subs(symbol, lower)
        lower_faces[species_id] = face_proof.proves(expression, NONNEGATIVE)
        if lower_faces[species_id] is False and lower_counterexample is None:
            lower_counterexample = face_point
            lower_expression = expression

    if defined is None:
        verdict = RecoveryVerdict.INDETERMINATE
        counterexample = None
    elif restoration_sign == POSITIVE:
        verdict = (
            RecoveryVerdict.STRONGLY_RESTORING
            if all(result is True for result in componentwise.values())
            else RecoveryVerdict.NET_RESTORING
        )
        counterexample = None
    elif restoration_sign == NEGATIVE or point_restoration_sign == NEGATIVE:
        verdict = RecoveryVerdict.WORSENING
        counterexample = point
    else:
        source_signs = tuple(
            proof.sign(expression) for expression in source_terms.values()
        )
        steady_point = all(
            proof.value_sign(expression) == ZERO for expression in source_terms.values()
        )
        if all(result == ZERO for result in source_signs) or (
            point_restoration_sign == ZERO and steady_point
        ):
            verdict = RecoveryVerdict.STUCK
            counterexample = point
        elif restoration_sign in (NONNEGATIVE, ZERO):
            verdict = RecoveryVerdict.NON_WORSENING
            counterexample = None
        else:
            verdict = RecoveryVerdict.INDETERMINATE
            counterexample = None

    decisive = restoration
    if lower_counterexample is not None and counterexample is None:
        counterexample = lower_counterexample
        decisive = lower_expression
    return RecoveryRegionResult(
        negative_species,
        verdict,
        domain.constraints,
        restoration,
        componentwise,
        MappingProxyType(lower_faces),
        defined,
        decisive,
        counterexample,
    )


def _limited_result(
    source_terms: Mapping[str, sp.Expr],
    diagnostic: str,
) -> NonphysicalRecoveryResult:
    return NonphysicalRecoveryResult(
        source_terms,
        (),
        (),
        complete=False,
        tests=0,
        diagnostic=diagnostic,
    )


def check_nonphysical_recovery(
    case: Case,
    *,
    stop_on_failure: bool = False,
    network: NetworkExpressions | None = None,
    conservation: StoichiometricConservationResult | None = None,
) -> NonphysicalRecoveryResult:
    """Classify every declared, repairable negative concentration region."""

    network = network or network_expressions(case)
    source_terms = network.source_terms
    expressions = (
        *(reaction.rate for reaction in case.reactions),
        *source_terms.values(),
    )
    operations = tuple(sp.count_ops(expression) for expression in expressions)
    largest = max(operations, default=0)
    if largest > MAX_SYMBOLIC_OPERATIONS:
        return _limited_result(
            source_terms,
            "A rate or source term exceeds the symbolic operation limit "
            f"({largest} > {MAX_SYMBOLIC_OPERATIONS}).",
        )
    total = sum(operations)
    if total > MAX_TOTAL_SYMBOLIC_OPERATIONS:
        return _limited_result(
            source_terms,
            "Network rates and source terms exceed the total symbolic operation "
            f"limit ({total} > {MAX_TOTAL_SYMBOLIC_OPERATIONS}).",
        )

    conservation = conservation or find_conserved_quantities(case, network)
    conservation_rays = _conservation_rays(case, conservation)
    candidates, complete = _negative_sets(case, conservation)
    regions_list: list[RecoveryRegionResult] = []
    stopped_on_failure = False
    for negative in candidates:
        region = _analyse_region(case, negative, source_terms, conservation_rays)
        regions_list.append(region)
        if stop_on_failure and (
            region.verdict
            in (
                RecoveryVerdict.STUCK,
                RecoveryVerdict.WORSENING,
                RecoveryVerdict.UNDEFINED_IN_EXTENSION,
            )
            or any(value is False for value in region.lower_faces.values())
        ):
            stopped_on_failure = True
            break
    regions = tuple(regions_list)
    return NonphysicalRecoveryResult(
        source_terms,
        conservation_rays,
        regions,
        complete and not stopped_on_failure,
        len(regions),
        (
            "Stopped after an exact failure certificate was found."
            if stopped_on_failure
            else None
        ),
    )


def _short(expression: sp.Expr, limit: int = 300) -> str:
    rendered = sp.sstr(expression)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _proof_label(result: bool | None, failed: str) -> str:
    if result is True:
        return "proved"
    return failed if result is False else "unresolved"


def _region_details(case: Case, result: RecoveryRegionResult) -> tuple[str, ...]:
    negative = ", ".join(result.negative_species)
    details = [f"Negative species {negative}: {result.verdict.value}."]
    if result.restoration is not None:
        details.append(f"  Restoration score: {_short(result.restoration)}.")
    for title, values, failed in (
        ("Componentwise inward source", result.componentwise, "failed"),
        ("Lower excursion faces", result.lower_faces, "violated"),
    ):
        if not values:
            continue
        rendered = ", ".join(
            f"{species_id}={_proof_label(value, failed)}"
            for species_id, value in values.items()
        )
        details.append(f"  {title}: {rendered}.")
    if result.restoration is None and result.decisive_expression is not None:
        details.append(
            f"  Decisive expression: {_short(result.decisive_expression)}."
        )
    if result.counterexample is not None:
        values = ", ".join(
            f"{species_id}="
            f"{result.counterexample[case.states.concentration(species_id)]}"
            for species_id in case.states.species_ids
        )
        details.append(f"  Exact counterexample: {values}.")
    return tuple(details)


_FAILURES = {
    RecoveryVerdict.STUCK,
    RecoveryVerdict.WORSENING,
    RecoveryVerdict.UNDEFINED_IN_EXTENSION,
}


def run(case: Case, context: CheckContext) -> CheckOutcome:
    network = context.cached(
        case,
        "network",
        lambda: network_expressions(case),
    )
    conservation = context.cached(
        case,
        "conservation",
        lambda: find_conserved_quantities(case, network),
    )
    result = check_nonphysical_recovery(
        case,
        stop_on_failure=True,
        network=network,
        conservation=conservation,
    )
    if not result.regions:
        status = (
            CheckStatus.INDETERMINATE
            if result.diagnostic is not None
            else CheckStatus.UNAVAILABLE
        )
        detail = result.diagnostic or "No concentration has a negative excursion."
        return CheckOutcome(status=status, details=(detail,))

    failed = any(
        region.verdict in _FAILURES
        or any(value is False for value in region.lower_faces.values())
        for region in result.regions
    )
    unresolved = not result.complete or any(
        not region.certified and region.verdict not in _FAILURES
        for region in result.regions
    )
    status = (
        CheckStatus.FAIL
        if failed
        else CheckStatus.INDETERMINATE if unresolved else CheckStatus.PASS
    )
    details = [
        "Assumptions: declared bounds, each sign-region, and non-negative "
        "conservation rays.",
        "Restoring verdicts do not assert finite-time re-entry.",
    ]
    for region in result.regions:
        details.extend(_region_details(case, region))
    if not result.complete:
        if result.diagnostic:
            details.append(result.diagnostic)
        else:
            details.append(
                f"Stopped after {result.tests} regions "
                f"(limit {MAX_NEGATIVE_REGIONS})."
            )
    return CheckOutcome(
        status=status,
        details=tuple(details),
        values=(CheckValue("Regions checked", result.tests),),
    )


CHECK = CheckDefinition(
    id="nonphysical_recovery",
    name="Recovery from nonphysical concentrations",
    group="Physical checks",
    scope=CheckScope.CASE,
    run=run,
)
