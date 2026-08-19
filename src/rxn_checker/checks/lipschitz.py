"""Uniform Lipschitz certificates for rates and the network source vector."""

from collections.abc import Mapping

import sympy as sp

from ..context import AnalysisContext
from ..domain import ConcentrationDomain
from ..proof import (
    LipschitzCertificate,
    LipschitzResult,
    ProofVerdict,
    derive_network_lipschitz,
)
from ..results import Evidence, Finding, Verdict


def _short(expression: sp.Expr, limit: int = 64) -> str:
    rendered = str(expression)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _approximate(expression: sp.Expr) -> str:
    numerical = sp.N(expression, 8)
    return str(numerical)


def _display_constant(expression: sp.Expr) -> str:
    exact = str(expression)
    if len(exact) <= 64:
        return exact
    return f"≈ {_approximate(expression)} (exact bound in structured evidence)"


def _rate_finding(reaction_id: str, result: LipschitzResult) -> Finding:
    verdict = {
        ProofVerdict.PASS: Verdict.PASS,
        ProofVerdict.FAIL: Verdict.FAIL,
        ProofVerdict.UNKNOWN: Verdict.UNKNOWN,
    }[result.verdict]
    if result.certificate is not None:
        certificate = result.certificate
        constant = certificate.constant_bound
        guards = tuple(
            {
                "expression": str(guard.expression),
                "requirement": guard.requirement.value,
                "margin": str(guard.margin),
            }
            for guard in certificate.guard_margins
        )
        return Finding(
            reaction_id,
            Verdict.PASS,
            "Certified Lipschitz constant (concentration L∞): "
            f"{_display_constant(constant)}.",
            Evidence(
                "lipschitz_certificate",
                {
                    "domain": certificate.domain.value,
                    "norm": certificate.norm,
                    "constant_bound": constant,
                    "constant_approximate": _approximate(constant),
                    "active_variables": tuple(map(str, certificate.active_variables)),
                    "uniform_parameters": tuple(
                        map(str, certificate.uniform_parameters)
                    ),
                    "guard_margins": guards,
                },
            ),
        )

    decisive = result.decisive_subexpression
    expression_text = (
        f" Decisive expression: {_short(decisive)}."
        if decisive is not None
        else ""
    )
    summary = (
        "No open-neighbourhood Lipschitz certificate exists."
        if verdict is Verdict.FAIL
        else "Lipschitz certification is inconclusive."
    )
    data: dict[str, object] = {}
    if decisive is not None:
        data["decisive_subexpression"] = str(decisive)
    if result.reason is not None:
        data["diagnostic"] = result.reason
    if result.witness is not None:
        data["point"] = {
            str(symbol): str(value) for symbol, value in result.witness.items()
        }
    return Finding(
        reaction_id,
        verdict,
        summary + expression_text,
        Evidence("lipschitz_obstruction", data) if data else None,
    )


def _network_finding(
    context: AnalysisContext,
    domain: ConcentrationDomain,
    certificates: tuple[LipschitzCertificate, ...],
) -> Finding:
    certificate = derive_network_lipschitz(
        domain,
        context.case.symbols.species_ids,
        context.stoichiometry,
        certificates,
    )
    return Finding(
        context.case.name,
        Verdict.PASS,
        "Certified source-vector Lipschitz constant (L∞ to L∞): "
        f"{_display_constant(certificate.constant_bound)}.",
        Evidence(
            "network_lipschitz_certificate",
            {
                "domain": certificate.domain.value,
                "input_norm": certificate.norm,
                "output_norm": certificate.norm,
                "constant_bound": certificate.constant_bound,
                "constant_approximate": _approximate(certificate.constant_bound),
                "component_bounds": dict(certificate.component_bounds),
                "active_variables": tuple(map(str, certificate.active_variables)),
                "uniform_parameters": tuple(
                    map(str, certificate.uniform_parameters)
                ),
            },
        ),
    )


def _run(
    context: AnalysisContext,
    domain: ConcentrationDomain,
) -> tuple[Finding, ...]:
    active = tuple(context.case.symbols.concentrations.values())
    results = tuple(
        context.expression_analyzer.lipschitz(reaction.rate, domain, active)
        for reaction in context.case.reactions
    )
    findings = tuple(
        _rate_finding(reaction.id, result)
        for reaction, result in zip(context.case.reactions, results)
    )
    if any(result.verdict is not ProofVerdict.PASS for result in results):
        return findings
    certificates = tuple(result.certificate for result in results)
    return (*findings, _network_finding(context, domain, certificates))


def run_physical(
    context: AnalysisContext,
    _dependencies: Mapping,
) -> tuple[Finding, ...]:
    return _run(context, context.physical_domain)


def run_augmented(
    context: AnalysisContext,
    _dependencies: Mapping,
) -> tuple[Finding, ...]:
    return _run(context, context.augmented_domain)


__all__ = ("run_augmented", "run_physical")
