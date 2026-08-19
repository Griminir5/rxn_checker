"""Physical boundary and forward-invariance theorem composition.

No rate expression is analysed here.  The proof uses previously established
rate non-negativity, depletion, and Lipschitz findings together with the signed
stoichiometric coefficients.
"""

from collections.abc import Mapping

from ..context import AnalysisContext
from ..results import CheckResult, Evidence, Finding, Verdict


def _passing_reactions(
    result: CheckResult,
    expected_ids: tuple[str, ...],
) -> frozenset[str]:
    passed = frozenset(
        finding.subject
        for finding in result.findings
        if finding.verdict is Verdict.PASS
    )
    missing = set(expected_ids) - passed
    if missing:
        raise RuntimeError(
            f"Passing prerequisite '{result.check_id}' lacks reactions: "
            + ", ".join(sorted(missing))
            + "."
        )
    return passed


def run_boundary_inward(
    context: AnalysisContext,
    dependencies: Mapping[str, CheckResult],
) -> Finding:
    """Compose the inward-boundary theorem from reaction-level facts."""

    reaction_ids = tuple(reaction.id for reaction in context.case.reactions)
    nonnegative = _passing_reactions(
        dependencies["rate_nonnegativity"], reaction_ids
    )
    depleted = _passing_reactions(
        dependencies["zero_at_depletion"], reaction_ids
    )

    faces: dict[str, object] = {}
    for species_id in context.case.symbols.species_ids:
        vanishing_consumers = []
        nonnegative_contributions = []
        for reaction in context.case.reactions:
            coefficient = reaction.net_stoichiometry.get(species_id)
            if coefficient is None:
                continue
            contribution = {
                "reaction": reaction.id,
                "stoichiometric_coefficient": coefficient,
            }
            if coefficient < 0:
                if reaction.id not in depleted or species_id not in reaction.reactants:
                    raise RuntimeError(
                        f"Missing depletion proof for {reaction.id} at {species_id}=0."
                    )
                vanishing_consumers.append(contribution)
            else:
                if reaction.id not in nonnegative:
                    raise RuntimeError(
                        f"Missing non-negativity proof for {reaction.id}."
                    )
                nonnegative_contributions.append(contribution)
        faces[species_id] = {
            "vanishing_consumers": tuple(vanishing_consumers),
            "nonnegative_contributions": tuple(nonnegative_contributions),
        }

    return Finding(
        context.case.name,
        Verdict.PASS,
        "Every zero-concentration face has a non-negative reaction source.",
        Evidence(
            "physical_boundary_inward_certificate",
            {
                "domain": "physical",
                "faces": faces,
                "argument": (
                    "Negative stoichiometric contributions vanish by depletion; "
                    "all remaining contributions are non-negative."
                ),
            },
        ),
    )


def _certificate(result: CheckResult, kind: str) -> Evidence:
    for finding in result.findings:
        if finding.evidence is not None and finding.evidence.kind == kind:
            return finding.evidence
    raise RuntimeError(f"Passing prerequisite '{result.check_id}' lacks {kind}.")


def run_forward_invariance(
    context: AnalysisContext,
    dependencies: Mapping[str, CheckResult],
) -> Finding:
    """Combine boundary inwardness and source Lipschitz regularity."""

    boundary = _certificate(
        dependencies["physical_boundary_inward"],
        "physical_boundary_inward_certificate",
    )
    regularity = _certificate(
        dependencies["physical_lipschitz"],
        "network_lipschitz_certificate",
    )
    parameters = {
        str(symbol): {
            "lower": interval.lower,
            "upper": interval.upper,
        }
        for symbol, interval in context.physical_domain.parameter_intervals.items()
    }
    return Finding(
        context.case.name,
        Verdict.PASS,
        "The nonnegative concentration orthant is forward invariant throughout "
        "the declared physical domain.",
        Evidence(
            "forward_invariance_certificate",
            {
                "domain": "physical",
                "state_set": "nonnegative concentration orthant",
                "uniform_parameter_ranges": parameters,
                "source_lipschitz_norm": regularity.data["input_norm"],
                "source_lipschitz_constant_bound": regularity.data[
                    "constant_bound"
                ],
                "boundary_face_count": len(boundary.data["faces"]),
                "theorem_inputs": (
                    "physical_boundary_inward",
                    "physical_lipschitz",
                ),
            },
        ),
    )


__all__ = ("run_boundary_inward", "run_forward_invariance")
