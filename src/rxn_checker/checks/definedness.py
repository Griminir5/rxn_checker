"""Physical and augmented rate-definedness checks."""

from collections.abc import Mapping

from ..context import AnalysisContext
from ..domain import ConcentrationDomain
from ..proof import DefinednessResult, ProofVerdict
from ..results import Evidence, Finding, Verdict


def _finding(
    reaction_id: str,
    result: DefinednessResult,
    domain_name: str,
) -> Finding:
    verdict = {
        ProofVerdict.PASS: Verdict.PASS,
        ProofVerdict.FAIL: Verdict.FAIL,
        ProofVerdict.UNKNOWN: Verdict.UNKNOWN,
    }[result.verdict]
    if verdict is Verdict.PASS:
        return Finding(
            reaction_id,
            verdict,
            f"Rate is real and finite on the {domain_name} domain.",
        )

    decisive = result.decisive_subexpression
    summary = (
        f"Rate is not defined throughout the {domain_name} domain."
        if verdict is Verdict.FAIL
        else f"Rate definedness on the {domain_name} domain is inconclusive."
    )
    if decisive is not None:
        rendered = str(decisive)
        if len(rendered) > 72:
            rendered = rendered[:69] + "..."
        summary += f" Decisive expression: {rendered}."

    data: dict[str, object] = {}
    if decisive is not None:
        data["decisive_subexpression"] = str(decisive)
    if result.requirement is not None:
        data["requirement"] = result.requirement.value
    if result.reason is not None:
        data["diagnostic"] = result.reason
    if result.witness is not None:
        data["point"] = {
            str(symbol): str(value) for symbol, value in result.witness.items()
        }
    evidence = Evidence("definedness_guard", data) if data else None
    return Finding(reaction_id, verdict, summary, evidence)


def _run(
    context: AnalysisContext,
    domain: ConcentrationDomain,
) -> tuple[Finding, ...]:
    return tuple(
        _finding(
            reaction.id,
            context.expression_analyzer.defined(reaction.rate, domain),
            domain.kind.value,
        )
        for reaction in context.case.reactions
    )


def run_physical(context: AnalysisContext, _dependencies: Mapping) -> tuple[Finding, ...]:
    return _run(context, context.physical_domain)


def run_augmented(context: AnalysisContext, _dependencies: Mapping) -> tuple[Finding, ...]:
    return _run(context, context.augmented_domain)


__all__ = ("run_augmented", "run_physical")
