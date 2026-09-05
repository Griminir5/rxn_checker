"""Symbolic differential profile of the concentration-space kinetic system."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache
from itertools import combinations, combinations_with_replacement

import sympy as sp
from sympy.core.relational import (
    Equality,
    GreaterThan,
    LessThan,
    Relational,
    StrictGreaterThan,
    StrictLessThan,
    Unequality,
)
from sympy.utilities.iterables import strongly_connected_components

from ..domain import ConcentrationDomain, DomainKind
from ..model import Reaction
from ..network import source_equivalent_fluxes
from .analysis import ExpressionAnalyzer, ProofVerdict

_MAX_HESSIAN_ENTRIES = 2048
_MAX_SECOND_DERIVATIVE_OPS = 256


class SurfaceLocation(StrEnum):
    EXCLUDED = "excluded"
    BOUNDARY = "boundary"
    INTERIOR = "interior"
    EVERYWHERE = "everywhere"
    POSSIBLE = "possible"
    UNKNOWN = "unknown"


class Regularity(StrEnum):
    C11 = "C1,1"
    C1 = "C1"
    C1_INTERIOR = "C1 on physical interior"
    PIECEWISE_C1 = "piecewise C1"
    LIPSCHITZ = "Lipschitz only"
    CONTINUOUS = "continuous only"
    UNKNOWN = "unknown"


class FeedbackKind(StrEnum):
    DAMPING = "uniformly damping"
    AMPLIFYING = "uniformly amplifying"
    MIXED = "changes sign"
    ZERO = "zero"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BranchReduction:
    expression: sp.Expr
    reductions: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class SurfaceProfile:
    kind: str
    expression: sp.Expr
    source: sp.Expr
    location: SurfaceLocation
    reason: str | None = None


@dataclass(frozen=True)
class DerivativeBound:
    variable: sp.Symbol
    derivative: sp.Expr | None
    lower: sp.Expr | None
    upper: sp.Expr | None
    absolute_upper: sp.Expr | None
    signed: bool
    exact_enclosure: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class RateDifferentialProfile:
    rate_id: str
    regularity: Regularity
    reduced_expression: sp.Expr
    branch_reductions: tuple[Mapping[str, object], ...]
    surfaces: tuple[SurfaceProfile, ...]
    derivatives: tuple[DerivativeBound, ...]
    self_feedback_expression: sp.Expr | None
    self_feedback_lower: sp.Expr | None
    self_feedback_upper: sp.Expr | None
    self_feedback_kind: FeedbackKind
    self_feedback_absolute_upper: sp.Expr | None
    source_jacobian_contribution: sp.Expr | None
    curvature_contribution: sp.Expr | None
    hessian: tuple[Mapping[str, object], ...] = ()
    curvature_status: str = "not_attempted"


@dataclass(frozen=True)
class MatrixEnvelope:
    shape: tuple[int, int]
    structural_nonzeros: int
    entries: tuple[Mapping[str, object], ...]
    infinity_norm_upper: sp.Expr | None
    logarithmic_norm_upper: sp.Expr | None
    spectral_radius_upper: sp.Expr | None
    complete: bool
    reason: str | None = None
    row_labels: tuple[str, ...] = ()
    column_labels: tuple[str, ...] = ()
    metadata: Mapping[str, object] | None = None

    @property
    def density(self) -> sp.Rational:
        size = self.shape[0] * self.shape[1]
        return sp.Rational(self.structural_nonzeros, size) if size else sp.S.Zero


@dataclass(frozen=True)
class DomainDifferentialProfile:
    domain: DomainKind
    rates: tuple[RateDifferentialProfile, ...]
    interaction: MatrixEnvelope
    source_jacobian: MatrixEnvelope
    reduced_jacobian: MatrixEnvelope
    operating_coupling: MatrixEnvelope
    ida_alpha_dominance_threshold: sp.Expr | None
    jacobian_variation_upper: sp.Expr | None
    hessian_truncated: bool
    concentration_scales: tuple[tuple[sp.Symbol, sp.Expr], ...]


@dataclass(frozen=True)
class DifferentialSolverProfile:
    stoichiometric_rank: int
    stoichiometric_basis: tuple[str, ...]
    physical: DomainDifferentialProfile
    augmented: DomainDifferentialProfile
    declared_reactions: int
    source_equivalent_fluxes: int


def _true(value) -> bool:
    return value is True or value is sp.true


def _maximum(values) -> sp.Expr:
    values = tuple(values)
    return sp.S.Zero if not values else values[0] if len(values) == 1 else sp.Max(*values)


def _sum(expressions) -> sp.Expr:
    terms = tuple(expressions)
    return (
        sp.S.Zero if not terms else terms[0] if len(terms) == 1 else sp.Add(*terms, evaluate=False)
    )


def _product(coefficient, expression) -> sp.Expr:
    if coefficient == 1:
        return expression
    return sp.Mul(coefficient, expression, evaluate=False)


def _ordered(symbols) -> tuple[sp.Symbol, ...]:
    return tuple(sorted(set(symbols), key=sp.default_sort_key))


def _condition_truth(condition, domain, analyzer) -> tuple[bool | None, str]:
    if condition in (True, sp.true):
        return True, "unconditional branch"
    if condition in (False, sp.false):
        return False, "false branch"
    if getattr(condition, "func", None) is sp.And:
        results = tuple(_condition_truth(item, domain, analyzer) for item in condition.args)
        if all(value is True for value, _ in results):
            return True, "; ".join(reason for _, reason in results)
        if any(value is False for value, _ in results):
            return False, "; ".join(reason for _, reason in results)
        return None, "conjunction is unresolved"
    if not isinstance(condition, Relational):
        return None, "condition type is unsupported"

    expression = condition.lhs - condition.rhs
    bounds = analyzer.bounds(expression, domain)
    if not bounds.known:
        return None, bounds.reason or "condition could not be bounded"
    lower, upper = bounds.lower, bounds.upper
    if isinstance(condition, GreaterThan):
        truth, falsehood = _true(lower >= 0), _true(upper < 0)
    elif isinstance(condition, StrictGreaterThan):
        truth, falsehood = _true(lower > 0), _true(upper <= 0)
    elif isinstance(condition, LessThan):
        truth, falsehood = _true(upper <= 0), _true(lower > 0)
    elif isinstance(condition, StrictLessThan):
        truth, falsehood = _true(upper < 0), _true(lower >= 0)
    elif isinstance(condition, Equality):
        truth = lower == upper == 0
        falsehood = _true(upper < 0) or _true(lower > 0)
    elif isinstance(condition, Unequality):
        truth = _true(upper < 0) or _true(lower > 0)
        falsehood = lower == upper == 0
    else:
        return None, "relation type is unsupported"
    reason = f"exact interval [{lower}, {upper}] for {expression}"
    return (True if truth else False if falsehood else None), reason


def reduce_branches(
    expression: sp.Expr, domain: ConcentrationDomain, analyzer: ExpressionAnalyzer
) -> BranchReduction:
    """Select only Abs/Min/Max/Piecewise branches proved on ``domain``."""

    reductions: list[Mapping[str, object]] = []

    def record(source: sp.Expr, selected: sp.Expr, reason: str) -> sp.Expr:
        reductions.append(
            {
                "original_subexpression": source,
                "selected_branch": selected,
                "proof_basis": reason,
                "domain": domain.kind,
            }
        )
        return selected

    def visit(value: sp.Expr) -> sp.Expr:
        if value.is_Atom:
            return value
        if isinstance(value, sp.Piecewise):
            prior_false = True
            branches = []
            for branch, condition in value.args:
                branch = visit(branch)
                truth, reason = _condition_truth(condition, domain, analyzer)
                if prior_false and truth is True:
                    return record(value, branch, reason)
                prior_false &= truth is False
                branches.append((branch, condition))
            return sp.Piecewise(*branches, evaluate=False)

        args = tuple(visit(argument) for argument in value.args)
        rebuilt = value.func(*args)
        if rebuilt.func is sp.Abs:
            bounds = analyzer.bounds(args[0], domain)
            if bounds.known and _true(bounds.lower >= 0):
                return record(value, args[0], f"lower bound {bounds.lower} is nonnegative")
            if bounds.known and _true(bounds.upper <= 0):
                return record(value, -args[0], f"upper bound {bounds.upper} is nonpositive")
        if rebuilt.func in {sp.Min, sp.Max}:
            for candidate in args:
                differences = tuple(
                    analyzer.bounds(candidate - other, domain)
                    for other in args
                    if other != candidate
                )
                dominates = all(
                    bound.known
                    and (
                        _true(bound.upper <= 0)
                        if rebuilt.func is sp.Min
                        else _true(bound.lower >= 0)
                    )
                    for bound in differences
                )
                if dominates:
                    relation = "no greater" if rebuilt.func is sp.Min else "no smaller"
                    return record(
                        value, candidate, f"selected argument is {relation} than every alternative"
                    )
        return rebuilt

    return BranchReduction(visit(sp.sympify(expression)), tuple(reductions))


def _surface_location(bounds) -> SurfaceLocation:
    if not bounds.known:
        return SurfaceLocation.UNKNOWN
    lower, upper = bounds.lower, bounds.upper
    if _true(upper < 0) or _true(lower > 0):
        return SurfaceLocation.EXCLUDED
    if bounds.exact:
        if lower == upper == 0:
            return SurfaceLocation.EVERYWHERE
        if _true(lower < 0) and _true(upper > 0):
            return SurfaceLocation.INTERIOR
        if (lower == 0 and _true(upper > 0)) or (upper == 0 and _true(lower < 0)):
            return SurfaceLocation.BOUNDARY
    return SurfaceLocation.POSSIBLE


def _extract_surfaces(expression, domain, analyzer) -> tuple[SurfaceProfile, ...]:
    surfaces: dict[tuple[str, sp.Expr, sp.Expr], SurfaceProfile] = {}

    def add(kind: str, zero: sp.Expr, source: sp.Expr) -> None:
        key = kind, zero, source
        if key not in surfaces:
            bounds = analyzer.bounds(zero, domain)
            surfaces[key] = SurfaceProfile(
                kind, zero, source, _surface_location(bounds), bounds.reason
            )

    for node in sp.preorder_traversal(expression):
        if node.func is sp.Abs:
            add("switch", node.args[0], node)
        elif node.func in {sp.Min, sp.Max}:
            for left, right in combinations(node.args, 2):
                add("switch", left - right, node)
        elif isinstance(node, sp.Piecewise):
            for _branch, condition in node.args:
                if isinstance(condition, Relational):
                    add("switch", condition.lhs - condition.rhs, node)
        elif node.func is sp.log:
            add("rate singularity", node.args[0], node)
        elif isinstance(node, sp.Pow) and node.exp.is_number is True:
            exponent = node.exp
            if exponent.is_integer and exponent.is_negative:
                add("rate singularity", node.base, node)
            elif exponent.is_integer is not True:
                if _true(exponent < 1):
                    kind = "first-derivative singularity"
                elif _true(exponent < 2):
                    kind = "second-derivative singularity"
                else:
                    kind = "noninteger-power boundary"
                add(kind, node.base, node)
    return tuple(surfaces.values())


def _active(surfaces, kinds) -> bool:
    return any(
        surface.kind in kinds and surface.location is not SurfaceLocation.EXCLUDED
        for surface in surfaces
    )


@cache
def symbolic_derivative(expression: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    return sp.diff(expression, variable)


def _feedback_kind(lower, upper) -> FeedbackKind:
    if lower is None or upper is None:
        return FeedbackKind.UNKNOWN
    if lower == upper == 0:
        return FeedbackKind.ZERO
    if _true(upper < 0):
        return FeedbackKind.DAMPING
    if _true(lower > 0):
        return FeedbackKind.AMPLIFYING
    if _true(lower < 0) and _true(upper > 0):
        return FeedbackKind.MIXED
    return FeedbackKind.UNKNOWN


def _signed_interval(
    expression, terms, domain, analyzer
) -> tuple[sp.Expr | None, sp.Expr | None, bool, str | None]:
    if expression is not None and sp.count_ops(expression) <= _MAX_SECOND_DERIVATIVE_OPS:
        bounds = analyzer.bounds(expression, domain)
        if bounds.known:
            return bounds.lower.doit(), bounds.upper.doit(), bounds.exact, None
    if not all(bound.signed for _, bound in terms):
        return None, None, False, "One or more signed derivative bounds are unavailable."
    lower = upper = sp.S.Zero
    for coefficient, bound in terms:
        values = (coefficient * bound.lower, coefficient * bound.upper)
        lower += sp.Min(*values)
        upper += sp.Max(*values)
    return lower.doit(), upper.doit(), False, None


def _linear_derivatives(terms, domain, analyzer) -> dict[str, object]:
    terms = tuple((coefficient, bound) for coefficient, bound in terms if coefficient != 0)
    expression = (
        _sum(_product(coefficient, bound.derivative) for coefficient, bound in terms)
        if all(bound.derivative is not None for _, bound in terms)
        else None
    )
    lower, upper, exact, reason = _signed_interval(expression, terms, domain, analyzer)
    absolute = (
        sum(abs(coefficient) * bound.absolute_upper for coefficient, bound in terms)
        if all(bound.absolute_upper is not None for _, bound in terms)
        else None
    )
    if lower is not None:
        absolute = sp.Max(abs(lower), abs(upper))
    return {
        "expression": expression,
        "lower": lower,
        "upper": upper,
        "absolute_upper": absolute,
        "signed": lower is not None,
        "exact_enclosure": exact,
        "complete": absolute is not None,
        "reason": reason,
    }


def _regularity(defined, gradient, surfaces) -> Regularity:
    reachable = tuple(item for item in surfaces if item.location is not SurfaceLocation.EXCLUDED)
    if defined is not ProofVerdict.PASS:
        return Regularity.UNKNOWN
    if any(item.kind == "switch" for item in reachable):
        interior = any(
            item.location
            in {
                SurfaceLocation.INTERIOR,
                SurfaceLocation.EVERYWHERE,
                SurfaceLocation.POSSIBLE,
                SurfaceLocation.UNKNOWN,
            }
            for item in reachable
            if item.kind == "switch"
        )
        if interior:
            return (
                Regularity.PIECEWISE_C1 if gradient is ProofVerdict.PASS else Regularity.CONTINUOUS
            )
        return Regularity.C1_INTERIOR
    if any(
        "first-derivative" in item.kind or item.kind == "rate singularity" for item in reachable
    ):
        return Regularity.CONTINUOUS
    return Regularity.C1 if gradient is ProofVerdict.PASS else Regularity.UNKNOWN


def _profile_rate(
    analyzer, flux, concentration_symbols, operating_symbols, domain
) -> RateDifferentialProfile:
    reduced = reduce_branches(flux.expression, domain, analyzer)
    surfaces = _extract_surfaces(reduced.expression, domain, analyzer)
    variables = _ordered(
        reduced.expression.free_symbols & set((*concentration_symbols, *operating_symbols))
    )
    gradient = analyzer.gradient_envelope(reduced.expression, domain, variables)
    components = (
        dict(gradient.certificate.gradient_envelope.components)
        if gradient.certificate and gradient.certificate.gradient_envelope
        else {}
    )
    defined = analyzer.defined(reduced.expression, domain).verdict
    regularity = _regularity(defined, gradient.verdict, surfaces)
    nonsmooth = _active(surfaces, {"switch", "rate singularity", "first-derivative singularity"})

    derivatives = []
    for variable in variables:
        derivative = None if nonsmooth else symbolic_derivative(reduced.expression, variable)
        unavailable = derivative is None or derivative.has(sp.Derivative)
        bounds = None if unavailable else analyzer.bounds(derivative, domain)
        signed = bounds is not None and bounds.known
        absolute = (
            sp.Max(abs(bounds.lower), abs(bounds.upper)) if signed else components.get(variable)
        )
        reason = None
        if unavailable:
            reason = "A classical derivative is unavailable on this domain."
        elif not signed:
            reason = bounds.reason
        derivatives.append(
            DerivativeBound(
                variable,
                None if unavailable else derivative,
                bounds.lower if signed else None,
                bounds.upper if signed else None,
                absolute,
                signed,
                bounds.exact if signed else False,
                reason,
            )
        )

    by_variable = {item.variable: item for item in derivatives}
    feedback_terms = tuple(
        (coefficient, by_variable[symbol])
        for symbol, coefficient in zip(concentration_symbols, flux.stoichiometry)
        if coefficient and symbol in by_variable
    )
    feedback = _linear_derivatives(feedback_terms, domain, analyzer)
    feedback_kind = (
        _feedback_kind(feedback["lower"], feedback["upper"])
        if not nonsmooth
        else FeedbackKind.UNKNOWN
    )
    return RateDifferentialProfile(
        flux.id,
        regularity,
        reduced.expression,
        reduced.reductions,
        surfaces,
        tuple(derivatives),
        feedback["expression"].doit() if feedback["expression"] is not None else None,
        feedback["lower"],
        feedback["upper"],
        feedback_kind,
        feedback["absolute_upper"],
        None,
        None,
    )


def _gradient(rates, variables) -> dict[tuple[int, int], DerivativeBound]:
    positions = {symbol: index for index, symbol in enumerate(variables)}
    return {
        (row, positions[bound.variable]): bound
        for row, rate in enumerate(rates)
        for bound in rate.derivatives
        if bound.variable in positions and (bound.derivative != 0 or bound.absolute_upper != 0)
    }


def _sign_label(lower, upper) -> str:
    if lower is None or upper is None:
        return "unknown"
    if lower == upper == 0:
        return "zero"
    if _true(lower > 0):
        return "positive"
    if _true(upper < 0):
        return "negative"
    if _true(lower < 0) and _true(upper > 0):
        return "mixed"
    return "non-strict"


def _entry(row, column, terms, domain, analyzer) -> dict[str, object] | None:
    terms = tuple((coefficient, bound) for coefficient, bound in terms if coefficient)
    if not terms:
        return None
    data = _linear_derivatives(terms, domain, analyzer)
    if data["expression"] == 0 and data["absolute_upper"] == 0:
        return None
    return {"row": row, "column": column, **data, "sign": _sign_label(data["lower"], data["upper"])}


def _numeric(value) -> float:
    if value is None:
        return float("-inf")
    try:
        result = float(sp.N(value, 8))
        return result if result == result else float("-inf")
    except (TypeError, ValueError, OverflowError):
        return float("-inf")


def _ranked(entries, limit=5):
    ranked = sorted(
        entries,
        key=lambda item: (
            -_numeric(item.get("absolute_upper")),
            str(item.get("row")),
            str(item.get("column")),
        ),
    )[:limit]
    return tuple(
        {
            "row": item.get("row"),
            "column": item.get("column"),
            "absolute_upper": item.get("absolute_upper"),
        }
        for item in ranked
    )


def _matrix_entries(left, gradient, right, rows, columns, domain, analyzer):
    """Bound entries of left * gradient * right without expanding rate expressions."""
    entries = []
    for row, row_label in enumerate(rows):
        for column, column_label in enumerate(columns):
            terms = (
                (left[row, reaction] * right[variable, column], bound)
                for (reaction, variable), bound in gradient.items()
                if left[row, reaction] and right[variable, column]
            )
            item = _entry(row_label, column_label, terms, domain, analyzer)
            if item is not None:
                entries.append(item)
    return entries


def _build_interaction(analyzer, network, rates, gradient, domain) -> MatrixEnvelope:
    labels = tuple(flux.id for flux in network.fluxes)
    entries = _matrix_entries(
        sp.eye(len(rates)), gradient, network.stoichiometry, labels, labels, domain, analyzer
    )

    edges = {label: set() for label in labels}
    for item in entries:
        if item["absolute_upper"] != 0:
            edges[item["column"]].add(item["row"])
    fan_in = {label: sum(label in targets for targets in edges.values()) for label in labels}
    fan_out = {label: len(edges[label]) for label in labels}
    graph_edges = [(source, target) for source in labels for target in sorted(edges[source])]
    components = tuple(
        tuple(sorted(group)) for group in strongly_connected_components((labels, graph_edges))
    )
    component_of = {node: index for index, group in enumerate(components) for node in group}
    one_way = sorted(
        {
            (component_of[source], component_of[target])
            for source, targets in edges.items()
            for target in targets
            if component_of[source] != component_of[target]
        }
    )
    mutual = tuple(
        (left, right)
        for left, right in combinations(labels, 2)
        if right in edges[left] and left in edges[right]
    )
    complete = all(item["complete"] for item in entries)
    return MatrixEnvelope(
        (len(rates), len(rates)),
        len(entries),
        tuple(entries),
        None,
        None,
        None,
        complete,
        None if complete else "Some interaction magnitudes are unresolved.",
        labels,
        labels,
        {
            "fan_in": fan_in,
            "fan_out": fan_out,
            "strongly_connected_groups": components,
            "one_way_component_edges": tuple(one_way),
            "mutually_coupled_pairs": mutual,
            "largest_interactions": _ranked(entries),
        },
    )


def _scale_entries(entries, row_scales, column_scales):
    for item in entries:
        factor = column_scales[item["column"]] / row_scales[item["row"]]
        for key in ("lower", "upper", "absolute_upper"):
            item[f"scaled_{key}"] = None if item[key] is None else factor * item[key]


def _matrix_measures(entries, row_labels, *, scaled=False):
    prefix = "scaled_" if scaled else ""
    absolute_key, upper_key = prefix + "absolute_upper", prefix + "upper"
    if any(item[absolute_key] is None for item in entries):
        return None, None, None, False, False
    rows, log_rows = [], []
    coarse = False
    for row in row_labels:
        row_entries = [item for item in entries if item["row"] == row]
        row_sum = sum((item[absolute_key] for item in row_entries), sp.S.Zero)
        diagonal = next((item for item in row_entries if item["column"] == row), None)
        log_sum = row_sum
        if diagonal is not None:
            if diagonal[upper_key] is None:
                coarse = True
            else:
                log_sum += diagonal[upper_key] - diagonal[absolute_key]
        rows.append(row_sum)
        log_rows.append(log_sum)
    infinity = _maximum(rows)
    # Gershgorin's absolute row bound equals the induced infinity norm bound.
    return infinity, _maximum(log_rows), infinity, True, coarse


def _build_source_jacobian(
    analyzer, network, species_ids, concentration_symbols, gradient, scales, domain
) -> MatrixEnvelope:
    columns = tuple(symbol.name for symbol in concentration_symbols)
    entries = _matrix_entries(
        network.stoichiometry,
        gradient,
        sp.eye(len(columns)),
        species_ids,
        columns,
        domain,
        analyzer,
    )
    _scale_entries(entries, dict(zip(species_ids, scales)), dict(zip(columns, scales)))

    infinity, logarithmic, spectral, complete, coarse = _matrix_measures(
        entries, species_ids, scaled=True
    )
    row_widths = Counter(item["row"] for item in entries)
    column_widths = Counter(item["column"] for item in entries)
    return MatrixEnvelope(
        (len(species_ids), len(concentration_symbols)),
        len(entries),
        tuple(entries),
        infinity,
        logarithmic,
        spectral,
        complete,
        None if complete else "Some source-Jacobian entry bounds are unresolved.",
        tuple(species_ids),
        tuple(symbol.name for symbol in concentration_symbols),
        {
            "row_dependency_widths": dict(row_widths),
            "column_influence_widths": dict(column_widths),
            "logarithmic_norm_coarse": coarse,
            "largest_entries": _ranked(
                tuple({**item, "absolute_upper": item["scaled_absolute_upper"]} for item in entries)
            ),
        },
    )


def _build_reduced_jacobian(
    analyzer, network, basis, coordinates, gradient, domain
) -> MatrixEnvelope:
    labels = tuple(f"z:{name}" for name in network.basis_ids)
    entries = _matrix_entries(coordinates, gradient, basis, labels, labels, domain, analyzer)

    infinity, logarithmic, spectral, complete, coarse = _matrix_measures(entries, labels)
    return MatrixEnvelope(
        (basis.cols, basis.cols),
        len(entries),
        tuple(entries),
        infinity,
        logarithmic,
        spectral,
        complete,
        None if complete else "Some active-mode entry bounds are unresolved.",
        labels,
        labels,
        {
            "basis_dependent": True,
            "logarithmic_norm_coarse": coarse,
            "largest_entries": _ranked(entries),
        },
    )


def _build_operating_coupling(
    analyzer, network, rates, species_ids, operating_symbols, domain, concentration_scales
) -> MatrixEnvelope:
    columns = tuple(symbol.name for symbol in operating_symbols)
    entries = _matrix_entries(
        network.stoichiometry,
        _gradient(rates, operating_symbols),
        sp.eye(len(columns)),
        species_ids,
        columns,
        domain,
        analyzer,
    )
    _scale_entries(
        entries,
        dict(zip(species_ids, concentration_scales)),
        dict(zip(columns, _domain_scales(domain, operating_symbols))),
    )

    complete = all(item["scaled_absolute_upper"] is not None for item in entries)
    column_bounds = {}
    for column in columns:
        bounds = [item["scaled_absolute_upper"] for item in entries if item["column"] == column]
        column_bounds[column] = None if None in bounds else _maximum(bounds)
    return MatrixEnvelope(
        (len(species_ids), len(operating_symbols)),
        len(entries),
        tuple(entries),
        _maximum(
            sum(
                (item["scaled_absolute_upper"] for item in entries if item["row"] == row), sp.S.Zero
            )
            for row in species_ids
        )
        if complete
        else None,
        None,
        None,
        complete,
        None if complete else "Some operating-variable coupling bounds are unresolved.",
        tuple(species_ids),
        tuple(symbol.name for symbol in operating_symbols),
        {
            "column_bounds": column_bounds,
            "largest_entries": _ranked(
                tuple({**item, "absolute_upper": item["scaled_absolute_upper"]} for item in entries)
            ),
        },
    )


def _reaction_contributors(
    network, rates, concentration_symbols, scales
) -> tuple[RateDifferentialProfile, ...]:
    profiled = []
    for reaction, rate in enumerate(rates):
        derivatives = {item.variable: item for item in rate.derivatives}
        weighted = tuple(
            derivatives[symbol].absolute_upper * scales[column]
            for column, symbol in enumerate(concentration_symbols)
            if symbol in derivatives and derivatives[symbol].absolute_upper is not None
        )
        complete = all(
            symbol not in derivatives or derivatives[symbol].absolute_upper is not None
            for symbol in concentration_symbols
        )
        direction = _maximum(
            abs(network.stoichiometry[row, reaction]) / scales[row] for row in range(len(scales))
        )
        contribution = direction * sum(weighted, sp.S.Zero) if complete else None
        profiled.append(replace(rate, source_jacobian_contribution=contribution))
    return tuple(profiled)


def _profile_curvature(
    analyzer, network, rates, concentration_symbols, scales, domain
) -> tuple[tuple[RateDifferentialProfile, ...], sp.Expr | None, bool]:
    attempted = 0
    truncated = False
    rate_sums: list[sp.Expr | None] = []
    profiled = []
    concentration_set = set(concentration_symbols)

    for rate in rates:
        nonsmooth = _active(
            rate.surfaces, {"switch", "rate singularity", "first-derivative singularity"}
        )
        if nonsmooth or rate.regularity in {Regularity.CONTINUOUS, Regularity.UNKNOWN}:
            rate_sums.append(None)
            profiled.append(replace(rate, curvature_status="not_applicable_nonsmooth"))
            continue

        dependencies = _ordered(item.variable for item in rate.derivatives)
        entries = []
        complete = True
        for left, right in combinations_with_replacement(dependencies, 2):
            if attempted >= _MAX_HESSIAN_ENTRIES:
                truncated, complete = True, False
                break
            first = symbolic_derivative(rate.reduced_expression, left)
            if first.has(sp.Derivative) or sp.count_ops(first) > _MAX_SECOND_DERIVATIVE_OPS:
                truncated, complete = True, False
                continue
            second = symbolic_derivative(first, right)
            attempted += 1
            if second == 0:
                continue
            if second.has(sp.Derivative) or sp.count_ops(second) > _MAX_SECOND_DERIVATIVE_OPS:
                truncated, complete = True, False
                continue
            bounds = analyzer.bounds(second, domain)
            absolute = bounds.absolute_upper
            if absolute is None:
                complete = False
            entries.append(
                {
                    "left": left,
                    "right": right,
                    "derivative": second,
                    "lower": bounds.lower,
                    "upper": bounds.upper,
                    "absolute_upper": absolute,
                    "exact_enclosure": bounds.exact,
                    "complete": absolute is not None,
                    "reason": bounds.reason,
                }
            )

        concentration_entries = tuple(
            item
            for item in entries
            if item["left"] in concentration_set and item["right"] in concentration_set
        )
        concentration_complete = complete and all(
            item["complete"] for item in concentration_entries
        )
        scale_by_symbol = dict(zip(concentration_symbols, scales))
        curvature_sum = None
        if concentration_complete:
            curvature_sum = sum(
                item["absolute_upper"]
                * scale_by_symbol[item["left"]]
                * scale_by_symbol[item["right"]]
                * (1 if item["left"] == item["right"] else 2)
                for item in concentration_entries
            )
        rate_sums.append(curvature_sum)
        reaction = len(rate_sums) - 1
        direction = _maximum(
            abs(network.stoichiometry[row, reaction]) / scales[row] for row in range(len(scales))
        )
        contribution = direction * curvature_sum if curvature_sum is not None else None
        regularity = (
            Regularity.C11
            if complete
            and rate.regularity is Regularity.C1
            and not _active(rate.surfaces, {"second-derivative singularity"})
            else rate.regularity
        )
        status = "complete" if complete else "partial"
        profiled.append(
            replace(
                rate,
                regularity=regularity,
                curvature_contribution=contribution,
                hessian=tuple(entries),
                curvature_status=status,
            )
        )

    variation = None
    if all(value is not None for value in rate_sums):
        variation = _maximum(
            sum(
                abs(network.stoichiometry[row, reaction]) / scales[row] * rate_sums[reaction]
                for reaction in range(len(rates))
            )
            for row in range(len(scales))
        )
    return tuple(profiled), variation, truncated


def _domain_scales(domain, concentration_symbols) -> tuple[sp.Expr, ...]:
    scales = tuple(
        max(abs(domain.interval(symbol).lower), abs(domain.interval(symbol).upper))
        for symbol in concentration_symbols
    )
    return tuple(value if value != 0 else sp.S.One for value in scales)


def _profile_domain(
    analyzer,
    network,
    species_ids,
    concentration_symbols,
    operating_symbols,
    domain,
    basis,
    coordinates,
) -> DomainDifferentialProfile:
    rates = tuple(
        _profile_rate(analyzer, flux, concentration_symbols, operating_symbols, domain)
        for flux in network.fluxes
    )
    scales = _domain_scales(domain, concentration_symbols)
    rates = _reaction_contributors(network, rates, concentration_symbols, scales)
    gradient = _gradient(rates, concentration_symbols)
    interaction = _build_interaction(analyzer, network, rates, gradient, domain)
    source = _build_source_jacobian(
        analyzer, network, species_ids, concentration_symbols, gradient, scales, domain
    )
    reduced = _build_reduced_jacobian(analyzer, network, basis, coordinates, gradient, domain)
    operating = _build_operating_coupling(
        analyzer, network, rates, species_ids, operating_symbols, domain, scales
    )
    rates, variation, truncated = _profile_curvature(
        analyzer, network, rates, concentration_symbols, scales, domain
    )
    return DomainDifferentialProfile(
        domain.kind,
        rates,
        interaction,
        source,
        reduced,
        operating,
        source.logarithmic_norm_upper,
        variation,
        truncated,
        tuple(zip(concentration_symbols, scales)),
    )


def profile_differential(
    *,
    analyzer: ExpressionAnalyzer,
    reactions: Sequence[Reaction],
    stoichiometry: sp.MatrixBase,
    concentration_symbols: Sequence[sp.Symbol],
    operating_symbols: Sequence[sp.Symbol],
    physical_domain: ConcentrationDomain,
    augmented_domain: ConcentrationDomain,
) -> DifferentialSolverProfile:
    """Profile local kinetic derivatives without sampling or running a solver."""

    reactions = tuple(reactions)
    concentration_symbols = tuple(concentration_symbols)
    operating_symbols = tuple(operating_symbols)
    network = source_equivalent_fluxes(reactions, stoichiometry)
    basis, coordinates = network.rank_factorization
    species_ids = tuple(symbol.name for symbol in concentration_symbols)
    physical, augmented = (
        _profile_domain(
            analyzer,
            network,
            species_ids,
            concentration_symbols,
            operating_symbols,
            domain,
            basis,
            coordinates,
        )
        for domain in (physical_domain, augmented_domain)
    )
    return DifferentialSolverProfile(
        basis.cols, network.basis_ids, physical, augmented, len(reactions), len(network.fluxes)
    )
