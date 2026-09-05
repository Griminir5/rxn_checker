"""Reaction-level prerequisite reuse."""

from ..results import Finding, Verdict


def reaction_skip(dependencies, check_id, reaction_id):
    result = dependencies.get(check_id)
    if result is None:
        return None
    for finding in result.findings:
        if finding.subject == reaction_id and finding.verdict is not Verdict.PASS:
            return Finding(
                reaction_id,
                Verdict.SKIPPED,
                f"Requires {check_id}=PASS for this reaction ({finding.verdict.value}).",
            )
    return None
