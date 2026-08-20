"""Uniform rate and source-vector Lipschitz certificates."""

import sympy as sp

from ..proof import ProofVerdict, derive_network_lipschitz
from ..results import Evidence, Finding, Verdict
from .prerequisites import reaction_skip

_VERDICT = {ProofVerdict.PASS: Verdict.PASS, ProofVerdict.FAIL: Verdict.FAIL,
            ProofVerdict.UNKNOWN: Verdict.UNKNOWN}


def _approx(value): return str(sp.N(value, 8))


def _display(value):
    exact = str(value)
    return exact if len(exact) <= 64 else f"≈ {_approx(value)} (exact bound in structured evidence)"


def _rate_finding(reaction_id, result):
    verdict = _VERDICT[result.verdict]
    if result.certificate:
        certificate = result.certificate
        guards = tuple({"expression": str(item.expression),
                        "requirement": item.requirement.value, "margin": str(item.margin)}
                       for item in certificate.guard_margins)
        return Finding(reaction_id, Verdict.PASS,
            f"Certified Lipschitz constant (concentration L∞): {_display(certificate.constant_bound)}.",
            Evidence("lipschitz_certificate", {
                "domain": certificate.domain.value, "norm": certificate.norm,
                "constant_bound": certificate.constant_bound,
                "constant_approximate": _approx(certificate.constant_bound),
                "active_variables": tuple(map(str, certificate.active_variables)),
                "uniform_parameters": tuple(map(str, certificate.uniform_parameters)),
                "guard_margins": guards}))
    decisive = result.decisive_subexpression
    shown = "" if decisive is None else str(decisive)
    if len(shown) > 64: shown = shown[:61] + "..."
    summary = ("No open-neighbourhood Lipschitz certificate exists." if verdict is Verdict.FAIL
               else "Lipschitz certification is inconclusive.")
    if shown: summary += f" Decisive expression: {shown}."
    data = {}
    if decisive is not None: data["decisive_subexpression"] = str(decisive)
    if result.reason: data["diagnostic"] = result.reason
    if result.witness: data["point"] = {str(key): str(value) for key, value in result.witness.items()}
    return Finding(reaction_id, verdict, summary,
                   Evidence("lipschitz_obstruction", data) if data else None)


def _network(context, domain, certificates):
    certificate = derive_network_lipschitz(domain, context.case.symbols.species_ids,
                                            context.stoichiometry, certificates)
    return Finding(context.case.name, Verdict.PASS,
        f"Certified source-vector Lipschitz constant (L∞ to L∞): {_display(certificate.constant_bound)}.",
        Evidence("network_lipschitz_certificate", {
            "domain": certificate.domain.value, "input_norm": certificate.norm,
            "output_norm": certificate.norm, "constant_bound": certificate.constant_bound,
            "constant_approximate": _approx(certificate.constant_bound),
            "component_bounds": dict(certificate.component_bounds),
            "active_variables": tuple(map(str, certificate.active_variables)),
            "uniform_parameters": tuple(map(str, certificate.uniform_parameters))}))


def _run(context, domain, dependencies, prerequisite):
    findings, certificates = [], []
    for reaction in context.case.reactions:
        skipped = reaction_skip(dependencies, prerequisite, reaction.id)
        if skipped:
            findings.append(skipped)
            continue
        result = context.expression_analyzer.lipschitz(reaction.rate, domain)
        findings.append(_rate_finding(reaction.id, result))
        if result.certificate: certificates.append(result.certificate)
    if len(certificates) == len(context.case.reactions):
        findings.append(_network(context, domain, tuple(certificates)))
    return tuple(findings)


def run_physical(context, dependencies):
    return _run(context, context.physical_domain, dependencies, "physical_rate_definedness")


def run_augmented(context, dependencies):
    return _run(context, context.augmented_domain, dependencies, "augmented_rate_definedness")
