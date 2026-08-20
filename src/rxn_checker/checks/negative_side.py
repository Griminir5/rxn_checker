"""Sparse augmented-domain non-repulsion and attraction proofs.

For one concentration coordinate, write its source as

    F_i = sum_j nu_ij r_j.

Each rate is bounded separately.  Terms that already have a sufficient lower
bound are replaced by that bound; only the unresolved residual is assembled for
the small symbolic fallback.  Exact feasible points may disprove a claim, but
candidate points are never used to prove one.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import sympy as sp

from ..context import AnalysisContext
from ..domain import ConcentrationDomain
from ..model import Reaction
from ..proof import ExpressionAnalyzer, ProofVerdict, SignRequirement
from ..proof.analysis import Point
from ..results import CheckResult, Evidence, Finding, Verdict


_RESIDUAL_SIZE_LIMIT = 96
_EXACT_SEARCH_SIZE_LIMIT = 96
_WITNESS_LIMIT = 24


@dataclass(frozen=True)
class ContributionBound:
    reaction_id: str
    coefficient: sp.Expr
    rate_lower: sp.Expr | None
    rate_upper: sp.Expr | None
    source_lower: sp.Expr | None
    source_upper: sp.Expr | None


@dataclass(frozen=True)
class SparseSourceResult:
    verdict: ProofVerdict
    lower_bound: sp.Expr | None
    upper_bound: sp.Expr | None
    contributions: tuple[ContributionBound, ...]
    witness: Point | None = None
    witness_value: sp.Expr | None = None
    reason: str | None = None


def _true(statement: object) -> bool:
    return statement is True or statement is sp.true


def _term_bounds(
    coefficient: sp.Expr,
    lower: sp.Expr,
    upper: sp.Expr,
) -> tuple[sp.Expr, sp.Expr]:
    if coefficient > 0:
        return coefficient * lower, coefficient * upper
    return coefficient * upper, coefficient * lower


def _source_value(
    contributions: Sequence[tuple[sp.Expr, Reaction]],
    point: Point,
) -> sp.Expr:
    return sum(
        (
            coefficient * reaction.rate.subs(point, simultaneous=True)
            for coefficient, reaction in contributions
        ),
        sp.S.Zero,
    )


def _candidate_points(
    domain: ConcentrationDomain,
    target: sp.Symbol,
    expressions: Sequence[sp.Expr],
) -> tuple[Point, ...]:
    """Generate O(number of relevant symbols) directed exact candidates."""

    relevant = tuple(
        sorted(
            set().union(*(expression.free_symbols for expression in expressions))
            & set(domain.all_intervals),
            key=lambda symbol: symbol.name,
        )
    )
    target_interval = domain.interval(target)
    negative = target_interval.lower
    midpoint = (target_interval.lower + target_interval.upper) / 2
    preferences: list[dict[sp.Symbol, sp.Expr]] = [
        {},
        {target: negative},
        {target: midpoint},
    ]

    all_zero = {
        symbol: sp.S.Zero
        for symbol in relevant
        if domain.interval(symbol).contains(0)
    }
    all_zero[target] = negative
    preferences.append(all_zero)
    preferences.append(
        {
            **{symbol: domain.interval(symbol).upper for symbol in relevant},
            target: negative,
        }
    )
    for symbol in relevant:
        interval = domain.interval(symbol)
        values = [interval.lower, interval.upper]
        if interval.contains(0):
            values.insert(1, sp.S.Zero)
        for value in values:
            preferences.append({target: midpoint, symbol: value})

    points: list[Point] = []
    seen: set[tuple[sp.Expr, ...]] = set()
    ordered_symbols = tuple(domain.all_intervals)
    for preference in preferences[:_WITNESS_LIMIT]:
        point = domain.exact_witness(preference)
        if point is None:
            continue
        key = tuple(point[symbol] for symbol in ordered_symbols)
        if key not in seen:
            seen.add(key)
            points.append(point)
    return tuple(points)


def _violates(value: sp.Expr, requirement: SignRequirement) -> bool:
    if value.is_real is not True or value.is_finite is not True:
        return False
    if requirement is SignRequirement.NONNEGATIVE:
        return value.is_negative is True
    return value.is_nonpositive is True


def _prove_source_sign(
    analyzer: ExpressionAnalyzer,
    contributions: Sequence[tuple[sp.Expr, Reaction]],
    domain: ConcentrationDomain,
    target: sp.Symbol,
    requirement: SignRequirement,
    prior_bounds: Sequence[ContributionBound] | None = None,
) -> SparseSourceResult:
    bounds: list[ContributionBound] = []
    unresolved_terms: list[sp.Expr] = []
    safe_margin = sp.S.Zero
    every_bound_known = True

    for index, (coefficient, reaction) in enumerate(contributions):
        prior = prior_bounds[index] if prior_bounds is not None else None
        if prior is None:
            rate_bound = analyzer.bounds(reaction.rate, domain)
            rate_lower, rate_upper = rate_bound.lower, rate_bound.upper
        else:
            rate_lower, rate_upper = prior.rate_lower, prior.rate_upper
        if rate_lower is not None and rate_upper is not None:
            lower, upper = _term_bounds(
                coefficient, rate_lower, rate_upper
            )
            sufficient = (
                _true(lower >= 0)
                if requirement is SignRequirement.NONNEGATIVE
                else _true(lower > 0)
            )
            if sufficient:
                safe_margin += lower
            else:
                unresolved_terms.append(coefficient * reaction.rate)
        else:
            lower = upper = None
            every_bound_known = False
            unresolved_terms.append(coefficient * reaction.rate)
        bounds.append(
            ContributionBound(
                reaction.id,
                coefficient,
                rate_lower,
                rate_upper,
                lower,
                upper,
            )
        )

    known_bounds = tuple(item for item in bounds if item.source_lower is not None)
    total_lower = (
        sum((item.source_lower for item in known_bounds), sp.S.Zero)
        if every_bound_known
        else None
    )
    total_upper = (
        sum((item.source_upper for item in known_bounds), sp.S.Zero)
        if every_bound_known
        else None
    )
    interval_proves = total_lower is not None and (
        _true(total_lower >= 0)
        if requirement is SignRequirement.NONNEGATIVE
        else _true(total_lower > 0)
    )
    if interval_proves:
        return SparseSourceResult(
            ProofVerdict.PASS,
            total_lower,
            total_upper,
            tuple(bounds),
        )

    interval_disproves = total_upper is not None and (
        _true(total_upper < 0)
        if requirement is SignRequirement.NONNEGATIVE
        else _true(total_upper <= 0)
    )
    if interval_disproves:
        return SparseSourceResult(
            ProofVerdict.FAIL,
            total_lower,
            total_upper,
            tuple(bounds),
            reason="The source upper bound violates the sign requirement.",
        )

    residual_size = sum(sp.count_ops(term) for term in unresolved_terms)
    if residual_size <= _RESIDUAL_SIZE_LIMIT:
        # The true source is at least this proved margin plus the exact
        # unresolved terms, so a non-negative residual proves the claim.
        residual = safe_margin + sum(unresolved_terms, sp.S.Zero)
        proof = analyzer.prove_sign(residual, domain, requirement)
        if proof.verdict is ProofVerdict.PASS:
            return SparseSourceResult(
                ProofVerdict.PASS,
                total_lower,
                total_upper,
                tuple(bounds),
            )

    source_size = sum(
        sp.count_ops(coefficient * reaction.rate)
        for coefficient, reaction in contributions
    )
    if source_size <= _EXACT_SEARCH_SIZE_LIMIT:
        expressions = tuple(reaction.rate for _, reaction in contributions)
        for point in _candidate_points(domain, target, expressions):
            value = _source_value(contributions, point)
            if _violates(value, requirement):
                return SparseSourceResult(
                    ProofVerdict.FAIL,
                    total_lower,
                    total_upper,
                    tuple(bounds),
                    point,
                    value,
                    "Exact feasible point violates the source-sign requirement.",
                )

    reason = (
        "The sparse lower bound and bounded residual analysis are inconclusive."
        if residual_size <= _RESIDUAL_SIZE_LIMIT
        and source_size <= _EXACT_SEARCH_SIZE_LIMIT
        else "The unresolved source exceeds a bounded symbolic-search budget."
    )
    return SparseSourceResult(
        ProofVerdict.UNKNOWN,
        total_lower,
        total_upper,
        tuple(bounds),
        reason=reason,
    )


def _result_data(result: SparseSourceResult) -> dict[str, object]:
    data: dict[str, object] = {
        "source_lower_bound": result.lower_bound,
        "source_upper_bound": result.upper_bound,
        "contributions": tuple(
            {
                "reaction": item.reaction_id,
                "stoichiometric_coefficient": item.coefficient,
                "rate_interval": (item.rate_lower, item.rate_upper),
                "source_interval": (item.source_lower, item.source_upper),
            }
            for item in result.contributions
        ),
    }
    if result.witness is not None:
        data["point"] = {
            str(symbol): value for symbol, value in result.witness.items()
        }
        data["source_value"] = result.witness_value
    if result.reason is not None:
        data["diagnostic"] = result.reason
    return data


def _strict_label(result: SparseSourceResult) -> tuple[str, bool]:
    if result.reason == "There are no feasible points with a negative coordinate.":
        return "not applicable", False
    stuck = result.lower_bound == result.upper_bound == 0
    if result.verdict is ProofVerdict.PASS:
        return "proved", False
    if result.verdict is ProofVerdict.FAIL:
        return "disproved", stuck
    return "unknown", False


def _finding(
    species_id: str,
    symbol: sp.Symbol,
    non_repulsion: SparseSourceResult,
    strict: SparseSourceResult,
) -> Finding:
    verdict = {
        ProofVerdict.PASS: Verdict.PASS,
        ProofVerdict.FAIL: Verdict.FAIL,
        ProofVerdict.UNKNOWN: Verdict.UNKNOWN,
    }[non_repulsion.verdict]
    main_label = {
        Verdict.PASS: "proved",
        Verdict.FAIL: "disproved",
        Verdict.UNKNOWN: "unknown",
    }[verdict]
    strict_label, stuck = _strict_label(strict)
    summary = (
        f"Non-repelling for {symbol} <= 0: {main_label}; "
        f"strictly attracting for {symbol} < 0: {strict_label}."
    )
    if stuck:
        summary += " The coordinate is stuck on the negative side."
    return Finding(
        species_id,
        verdict,
        summary,
        Evidence(
            "negative_side_certificate"
            if verdict is Verdict.PASS
            else "negative_side_obstruction",
            {
                "domain": "augmented",
                "coordinate": str(symbol),
                "non_repulsion": {
                    "verdict": main_label,
                    **_result_data(non_repulsion),
                },
                "strict_attraction": {
                    "verdict": strict_label,
                    "stuck": stuck,
                    **_result_data(strict),
                },
            },
        ),
    )


def _summary(case_name: str, findings: Sequence[Finding]) -> Finding:
    strict = [
        finding.evidence.data["strict_attraction"]
        for finding in findings
        if finding.evidence is not None
        and "strict_attraction" in finding.evidence.data
    ]
    counts = {
        label: sum(item["verdict"] == label for item in strict)
        for label in ("proved", "disproved", "unknown")
    }
    counts["stuck"] = sum(bool(item["stuck"]) for item in strict)
    proved = sum(finding.verdict is Verdict.PASS for finding in findings)
    return Finding(
        case_name,
        Verdict.PASS,
        f"Non-repulsion proved for {proved}/{len(findings)} excursion coordinates; "
        f"strict attraction: {counts['proved']} proved, "
        f"{counts['disproved']} disproved, {counts['unknown']} unknown "
        f"({counts['stuck']} stuck).",
        Evidence("negative_side_summary", counts),
    )


def run(
    context: AnalysisContext,
    dependencies: Mapping[str, CheckResult],
) -> tuple[Finding, ...]:
    """Check every feasible coordinate with a configured negative excursion."""

    domain = context.augmented_domain
    regularity = {
        finding.subject: finding.verdict
        for finding in dependencies["augmented_lipschitz"].findings
    }
    findings = []
    for row, species_id in enumerate(context.case.symbols.species_ids):
        symbol = context.case.symbols.concentration(species_id)
        if domain.interval(symbol).lower >= 0:
            continue
        contributions = tuple(
            (coefficient, reaction)
            for coefficient, reaction in zip(
                context.stoichiometry.row(row), context.case.reactions
            )
            if coefficient != 0
        )
        missing = tuple(
            reaction.id
            for _, reaction in contributions
            if regularity.get(reaction.id) is not Verdict.PASS
        )
        if missing:
            findings.append(
                Finding(
                    species_id,
                    Verdict.SKIPPED,
                    "Requires augmented regularity for contributing reactions: "
                    + ", ".join(missing)
                    + ".",
                    Evidence("missing_augmented_regularity", {"reactions": missing}),
                )
            )
            continue

        nonpositive_face = domain.restrict(symbol, upper=0)
        if not nonpositive_face.is_feasible():
            findings.append(
                Finding(
                    species_id,
                    Verdict.PASS,
                    f"No feasible augmented-domain point has {symbol} <= 0.",
                    Evidence(
                        "negative_side_certificate",
                        {"domain": "augmented", "coordinate": str(symbol)},
                    ),
                )
            )
            continue
        non_repulsion = _prove_source_sign(
            context.expression_analyzer,
            contributions,
            nonpositive_face,
            symbol,
            SignRequirement.NONNEGATIVE,
        )

        negative_face = domain.restrict(symbol, upper=0, strict_upper=True)
        if negative_face.is_feasible():
            # Only closure at zero changed, so the previous interval bounds
            # remain valid on this smaller domain.
            strict = _prove_source_sign(
                context.expression_analyzer,
                contributions,
                negative_face,
                symbol,
                SignRequirement.POSITIVE,
                non_repulsion.contributions,
            )
        else:
            strict = SparseSourceResult(
                ProofVerdict.PASS,
                None,
                None,
                (),
                reason="There are no feasible points with a negative coordinate.",
            )
        findings.append(_finding(species_id, symbol, non_repulsion, strict))

    if findings:
        return (*findings, _summary(context.case.name, findings))
    return (
        Finding(
            context.case.name,
            Verdict.PASS,
            "No concentration coordinate permits a negative excursion.",
            Evidence("negative_side_summary", {"coordinates": 0}),
        ),
    )


__all__ = ("ContributionBound", "SparseSourceResult", "run")
