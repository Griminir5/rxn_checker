"""Physical and augmented rate-definedness checks."""

from ..proof import ProofVerdict
from ..results import Evidence, Finding, Verdict

_VERDICT = {ProofVerdict.PASS: Verdict.PASS, ProofVerdict.FAIL: Verdict.FAIL,
            ProofVerdict.UNKNOWN: Verdict.UNKNOWN}


def _finding(reaction_id, result, domain):
    verdict = _VERDICT[result.verdict]
    if verdict is Verdict.PASS:
        return Finding(reaction_id, verdict,
                       f"Rate is real and finite on the {domain} domain.")
    decisive = result.decisive_subexpression
    summary = (f"Rate is not defined throughout the {domain} domain." if verdict is Verdict.FAIL
               else f"Rate definedness on the {domain} domain is inconclusive.")
    if decisive is not None:
        shown = str(decisive)
        summary += f" Decisive expression: {shown[:69] + '...' if len(shown) > 72 else shown}."
    data = {}
    if decisive is not None: data["decisive_subexpression"] = str(decisive)
    if result.requirement is not None: data["requirement"] = result.requirement.value
    if result.reason: data["diagnostic"] = result.reason
    if result.witness: data["point"] = {str(key): str(value) for key, value in result.witness.items()}
    return Finding(reaction_id, verdict, summary,
                   Evidence("definedness_guard", data) if data else None)


def _run(context, domain):
    return tuple(_finding(reaction.id,
        context.expression_analyzer.defined(reaction.rate, domain), domain.kind.value)
        for reaction in context.case.reactions)


def run_physical(context, _dependencies): return _run(context, context.physical_domain)
def run_augmented(context, _dependencies): return _run(context, context.augmented_domain)
