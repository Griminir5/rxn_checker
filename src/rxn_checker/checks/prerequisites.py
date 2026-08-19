"""Small helpers for reaction-wise prerequisite reuse."""

from collections.abc import Mapping

from ..results import CheckResult, Finding, Verdict


def reaction_skip(
    dependencies: Mapping[str, CheckResult],
    check_id: str,
    reaction_id: str,
) -> Finding | None:
    """Return a skip only when this reaction's prerequisite did not pass."""

    if check_id not in dependencies:
        return None
    for finding in dependencies[check_id].findings:
        if finding.subject == reaction_id:
            if finding.verdict is Verdict.PASS:
                return None
            return Finding(
                reaction_id,
                Verdict.SKIPPED,
                f"Requires {check_id}=PASS for this reaction "
                f"({finding.verdict.value}).",
            )
    raise RuntimeError(f"Prerequisite '{check_id}' lacks reaction '{reaction_id}'.")


__all__ = ("reaction_skip",)
