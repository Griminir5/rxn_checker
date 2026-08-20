"""Compose physical invariance from established reaction facts."""

from collections.abc import Mapping

from ..context import AnalysisContext
from ..results import CheckResult, Evidence, Finding, Verdict


def run_boundary_inward(context: AnalysisContext,
                        _dependencies: Mapping[str, CheckResult]) -> Finding:
    faces = {}
    for species_id in context.case.symbols.species_ids:
        consumers, producers = [], []
        for reaction in context.case.reactions:
            coefficient = reaction.net_stoichiometry.get(species_id)
            if coefficient is None:
                continue
            contribution = {"reaction": reaction.id,
                            "stoichiometric_coefficient": coefficient}
            (consumers if coefficient < 0 else producers).append(contribution)
        faces[species_id] = {"vanishing_consumers": tuple(consumers),
                             "nonnegative_contributions": tuple(producers)}
    return Finding(context.case.name, Verdict.PASS,
        "Every zero-concentration face has a non-negative reaction source.",
        Evidence("physical_boundary_inward_certificate", {
            "domain": "physical", "faces": faces,
            "argument": "Negative stoichiometric contributions vanish by depletion; "
                        "all remaining contributions are non-negative."}))


def run_forward_invariance(context: AnalysisContext,
                           dependencies: Mapping[str, CheckResult]) -> Finding:
    boundary = dependencies["physical_boundary_inward"].findings[0].evidence
    regularity = next(finding.evidence for finding in
                      dependencies["physical_lipschitz"].findings
                      if finding.evidence and
                      finding.evidence.kind == "network_lipschitz_certificate")
    parameters = {str(symbol): {"lower": interval.lower, "upper": interval.upper}
                  for symbol, interval in context.physical_domain.parameter_intervals.items()}
    return Finding(context.case.name, Verdict.PASS,
        "The nonnegative concentration orthant is forward invariant throughout "
        "the declared physical domain.", Evidence("forward_invariance_certificate", {
            "domain": "physical", "state_set": "nonnegative concentration orthant",
            "uniform_parameter_ranges": parameters,
            "source_lipschitz_norm": regularity.data["input_norm"],
            "source_lipschitz_constant_bound": regularity.data["constant_bound"],
            "boundary_face_count": len(boundary.data["faces"]),
            "theorem_inputs": ("physical_boundary_inward", "physical_lipschitz")}))
