"""Physical rate non-negativity."""

from ..proof import ExpressionAnalyzer, ProofVerdict, SignRequirement
from ..results import Evidence, Finding, Verdict
from .prerequisites import reaction_skip


def check_rate_nonnegativity(reaction, domain, *, analyzer=None):
    return (analyzer or ExpressionAnalyzer()).prove_sign(
        reaction.rate, domain, SignRequirement.NONNEGATIVE)


def _finding(reaction_id, proof):
    if proof.verdict is ProofVerdict.PASS:
        return Finding(reaction_id, Verdict.PASS,
                       "Rate is non-negative throughout the physical domain.")
    if proof.verdict is ProofVerdict.FAIL:
        data = {}
        if proof.witness:
            data["point"] = {str(key): str(value) for key, value in proof.witness.items()}
        if proof.witness_value is not None:
            data["rate"] = str(proof.witness_value)
        return Finding(reaction_id, Verdict.FAIL,
                       "Rate is negative at an exact physical-domain point.",
                       Evidence("exact_counterexample", data))
    decisive = proof.result.decisive_subexpression
    data = {"diagnostic": proof.reason} if proof.reason else {}
    if decisive is not None:
        data["decisive_subexpression"] = str(decisive)
    return Finding(reaction_id, Verdict.UNKNOWN, "Symbolic sign analysis was inconclusive.",
                   Evidence("sign_obstruction", data) if data else None)


def run(context, dependencies):
    return tuple(reaction_skip(dependencies, "physical_rate_definedness", reaction.id)
        or _finding(reaction.id, check_rate_nonnegativity(
            reaction, context.physical_domain, analyzer=context.expression_analyzer))
        for reaction in context.case.reactions)
