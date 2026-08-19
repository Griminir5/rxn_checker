"""Physical rate non-negativity check."""

from collections.abc import Mapping
from dataclasses import dataclass

import sympy as sp

from ..context import AnalysisContext
from ..domain import ConcentrationDomain
from ..model import Reaction
from ..proof import ExpressionAnalyzer, ProofVerdict, SignRequirement
from ..proof.analysis import Point
from ..results import Evidence, Finding, Verdict


@dataclass(frozen=True)
class RateNonnegativityResult:
    """Symbolic conclusion for one reaction rate."""

    reaction_id: str
    passed: bool | None
    rate: sp.Expr
    counterexample: Point | None = None


def check_rate_nonnegativity(
    reaction: Reaction,
    domain: ConcentrationDomain,
    *,
    analyzer: ExpressionAnalyzer | None = None,
) -> RateNonnegativityResult:
    """Prove non-negativity or return an exact violating point."""

    proof = (analyzer or ExpressionAnalyzer()).prove_sign(
        reaction.rate,
        domain,
        SignRequirement.NONNEGATIVE,
    )
    return RateNonnegativityResult(
        reaction.id,
        True
        if proof.verdict is ProofVerdict.PASS
        else False
        if proof.verdict is ProofVerdict.FAIL
        else None,
        reaction.rate,
        proof.witness,
    )


def _finding(result: RateNonnegativityResult) -> Finding:
    if result.passed is False:
        evidence = None
        if result.counterexample is not None:
            evidence = Evidence(
                "exact_counterexample",
                {
                    str(symbol): str(value)
                    for symbol, value in result.counterexample.items()
                },
            )
        return Finding(
            result.reaction_id,
            Verdict.FAIL,
            "Rate is negative at an exact physical-domain point.",
            evidence,
        )
    if result.passed:
        return Finding(
            result.reaction_id,
            Verdict.PASS,
            "Rate is non-negative throughout the physical domain.",
        )
    return Finding(
        result.reaction_id,
        Verdict.UNKNOWN,
        "Symbolic sign analysis was inconclusive.",
    )


def run(context: AnalysisContext, _dependencies: Mapping) -> tuple[Finding, ...]:
    """Run rate non-negativity for every reaction in a case."""

    findings: list[Finding] = []
    domain = context.physical_domain
    for reaction in context.case.reactions:
        try:
            result = check_rate_nonnegativity(
                reaction,
                domain,
                analyzer=context.expression_analyzer,
            )
        except ValueError as error:
            findings.append(
                Finding(reaction.id, Verdict.SKIPPED, str(error.args[0]))
            )
            continue
        findings.append(_finding(result))
    return tuple(findings)
