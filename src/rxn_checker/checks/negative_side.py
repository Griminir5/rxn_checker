"""Augmented-domain non-repulsion and attraction checks."""

from collections.abc import Mapping, Sequence

import sympy as sp

from ..context import AnalysisContext
from ..proof import ContributionBound, ProofVerdict, SignRequirement, SumProof
from ..results import CheckResult, Evidence, Finding, Verdict

SparseSourceResult = SumProof


def _data(result: SumProof, reaction_ids: Sequence[str]) -> dict[str, object]:
    data = {
        "source_lower_bound": result.lower,
        "source_upper_bound": result.upper,
        "contributions": tuple(
            {
                "reaction": reaction_id,
                "stoichiometric_coefficient": item.coefficient,
                "rate_interval": (item.lower, item.upper),
                "source_interval": (item.source_lower, item.source_upper),
            }
            for reaction_id, item in zip(reaction_ids, result.contributions)
        ),
    }
    if result.witness is not None:
        data.update(point={str(symbol): value for symbol, value in result.witness.items()},
                    source_value=result.witness_value)
    if result.reason:
        data["diagnostic"] = result.reason
    return data


def _label(result: SumProof) -> tuple[str, bool]:
    if result.reason == "No feasible negative points.":
        return "not applicable", False
    label = {ProofVerdict.PASS: "proved", ProofVerdict.FAIL: "disproved",
             ProofVerdict.UNKNOWN: "unknown"}[result.verdict]
    return label, result.verdict is ProofVerdict.FAIL and result.lower == result.upper == 0


def _finding(species_id, symbol, non_repulsion, attraction, reaction_ids):
    verdict = {ProofVerdict.PASS: Verdict.PASS, ProofVerdict.FAIL: Verdict.FAIL,
               ProofVerdict.UNKNOWN: Verdict.UNKNOWN}[non_repulsion.verdict]
    main, _ = _label(non_repulsion)
    strict, stuck = _label(attraction)
    summary = (f"Non-repelling for {symbol} <= 0: {main}; strictly attracting for "
               f"{symbol} < 0: {strict}.")
    if stuck:
        summary += " The coordinate is stuck on the negative side."
    return Finding(species_id, verdict, summary, Evidence(
        "negative_side_certificate" if verdict is Verdict.PASS else "negative_side_obstruction",
        {"domain": "augmented", "coordinate": str(symbol),
         "non_repulsion": {"verdict": main, **_data(non_repulsion, reaction_ids)},
         "strict_attraction": {"verdict": strict, "stuck": stuck,
                               **_data(attraction, reaction_ids)}}))


def _summary(case_name, findings):
    strict = [finding.evidence.data["strict_attraction"] for finding in findings
              if finding.evidence and "strict_attraction" in finding.evidence.data]
    counts = {label: sum(item["verdict"] == label for item in strict)
              for label in ("proved", "disproved", "unknown")}
    counts["stuck"] = sum(item["stuck"] for item in strict)
    proved = sum(finding.verdict is Verdict.PASS for finding in findings)
    return Finding(case_name, Verdict.PASS,
                   f"Non-repulsion proved for {proved}/{len(findings)} excursion coordinates; "
                   f"strict attraction: {counts['proved']} proved, {counts['disproved']} "
                   f"disproved, {counts['unknown']} unknown ({counts['stuck']} stuck).",
                   Evidence("negative_side_summary", counts))


def run(context: AnalysisContext, dependencies: Mapping[str, CheckResult]) -> tuple[Finding, ...]:
    domain = context.augmented_domain
    regular = {item.subject: item.verdict
               for item in dependencies["augmented_lipschitz"].findings}
    findings = []
    for species_id in context.case.symbols.species_ids:
        symbol = context.case.symbols.concentration(species_id)
        if domain.interval(symbol).lower >= 0:
            continue
        terms = context.source_contributions(species_id)
        reactions = tuple(reaction for coefficient, reaction in zip(
            context.stoichiometry.row(context.case.symbols.species_ids.index(species_id)),
            context.case.reactions) if coefficient != 0)
        missing = tuple(reaction.id for reaction in reactions
                        if regular.get(reaction.id) is not Verdict.PASS)
        if missing:
            findings.append(Finding(species_id, Verdict.SKIPPED,
                "Requires augmented regularity for contributing reactions: " +
                ", ".join(missing) + ".", Evidence("missing_augmented_regularity", {"reactions": missing})))
            continue
        face = domain.restrict(symbol, upper=0)
        if not face.is_feasible():
            findings.append(Finding(species_id, Verdict.PASS,
                f"No feasible augmented-domain point has {symbol} <= 0."))
            continue
        non_repulsion = context.expression_analyzer.prove_sum(
            terms, face, SignRequirement.NONNEGATIVE)
        negative = domain.restrict(symbol, upper=0, strict_upper=True)
        attraction = (context.expression_analyzer.prove_sum(
            terms, negative, SignRequirement.POSITIVE) if negative.is_feasible()
            else SumProof(ProofVerdict.PASS, None, None, (), reason="No feasible negative points."))
        findings.append(_finding(species_id, symbol, non_repulsion, attraction,
                                 tuple(reaction.id for reaction in reactions)))
    if not findings:
        return (Finding(context.case.name, Verdict.PASS,
                        "No concentration coordinate permits a negative excursion.",
                        Evidence("negative_side_summary", {"coordinates": 0})),)
    return (*findings, _summary(context.case.name, findings))


__all__ = ("ContributionBound", "SparseSourceResult", "run")
