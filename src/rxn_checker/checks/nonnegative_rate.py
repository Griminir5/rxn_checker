"""Physical rate non-negativity check."""

from collections.abc import Mapping

from ..context import AnalysisContext
from ..domain import ConcentrationDomain
from ..model import Reaction
from ..proof import ExpressionAnalyzer, ProofVerdict, SignProof, SignRequirement
from ..results import Evidence, Finding, Verdict
from .prerequisites import reaction_skip


def check_rate_nonnegativity(
    reaction: Reaction,
    domain: ConcentrationDomain,
    *,
    analyzer: ExpressionAnalyzer | None = None,
) -> SignProof:
    """Prove non-negativity or return an exact violating point."""

    return (analyzer or ExpressionAnalyzer()).prove_sign(
        reaction.rate,
        domain,
        SignRequirement.NONNEGATIVE,
    )


def _finding(reaction_id: str, proof: SignProof) -> Finding:
    if proof.verdict is ProofVerdict.FAIL:
        data: dict[str, object] = {}
        if proof.witness is not None:
            data["point"] = {
                str(symbol): str(value) for symbol, value in proof.witness.items()
            }
        if proof.witness_value is not None:
            data["rate"] = str(proof.witness_value)
        return Finding(
            reaction_id,
            Verdict.FAIL,
            "Rate is negative at an exact physical-domain point.",
            Evidence("exact_counterexample", data),
        )
    if proof.verdict is ProofVerdict.PASS:
        return Finding(
            reaction_id,
            Verdict.PASS,
            "Rate is non-negative throughout the physical domain.",
        )
    decisive = proof.result.decisive_subexpression
    data = {"diagnostic": proof.reason} if proof.reason else {}
    if decisive is not None:
        data["decisive_subexpression"] = str(decisive)
    return Finding(
        reaction_id,
        Verdict.UNKNOWN,
        "Symbolic sign analysis was inconclusive.",
        Evidence("sign_obstruction", data) if data else None,
    )


def run(context: AnalysisContext, dependencies: Mapping) -> tuple[Finding, ...]:
    """Run rate non-negativity for every reaction in a case."""

    domain = context.physical_domain
    findings = []
    for reaction in context.case.reactions:
        skipped = reaction_skip(
            dependencies, "physical_rate_definedness", reaction.id
        )
        findings.append(
            skipped
            or _finding(
                reaction.id,
                check_rate_nonnegativity(
                    reaction,
                    domain,
                    analyzer=context.expression_analyzer,
                ),
            )
        )
    return tuple(findings)
