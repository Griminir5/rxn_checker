"""Zero-rate checks at reactant and catalyst depletion boundaries."""

from collections.abc import Mapping
from dataclasses import dataclass

import sympy as sp

from ..context import AnalysisContext
from ..domain import ConcentrationDomain
from ..model import CaseSymbols, Reaction
from ..proof import ExpressionAnalyzer, ProofVerdict
from ..proof.analysis import Point
from ..results import Evidence, Finding, Verdict
from .prerequisites import reaction_skip


@dataclass(frozen=True)
class ZeroAtDepletionResult:
    reaction_id: str
    passed: bool | None
    rates_at_depletion: Mapping[str, sp.Expr]
    conclusions: Mapping[str, bool | None]
    counterexamples: Mapping[str, Point]


def check_zero_at_depletion(
    reaction: Reaction, symbols: CaseSymbols, domain: ConcentrationDomain | None = None
) -> ZeroAtDepletionResult:
    required = tuple(dict.fromkeys((*reaction.reactants, *reaction.catalysts)))
    rates, conclusions, counterexamples = {}, {}, {}
    analyzer = ExpressionAnalyzer()
    for species_id in required:
        symbol = symbols.concentration(species_id)
        rate = reaction.rate.subs(symbol, 0, simultaneous=True)
        face = domain.restrict(symbol, lower=0, upper=0) if domain else None
        if face is None:
            if rate.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                conclusion = False
            else:
                numerator, denominator = rate.as_numer_denom()
                conclusion = False if denominator.is_zero else rate.is_zero
                if conclusion is None and sp.count_ops(rate) <= 64:
                    conclusion = sp.factor_terms(rate).is_zero
        else:
            proof = analyzer.prove_zero(rate, face)
            conclusion = {
                ProofVerdict.PASS: True,
                ProofVerdict.FAIL: False,
                ProofVerdict.UNKNOWN: None,
            }[proof.verdict]
            if proof.witness is not None:
                counterexamples[species_id] = proof.witness
        rates[species_id], conclusions[species_id] = rate, conclusion
    passed = (
        False if False in conclusions.values() else (None if None in conclusions.values() else True)
    )
    return ZeroAtDepletionResult(reaction.id, passed, rates, conclusions, counterexamples)


def _finding(result):
    failed = [
        (species, rate)
        for species, rate in result.rates_at_depletion.items()
        if result.conclusions[species] is False
    ]
    unknown = [
        (species, rate)
        for species, rate in result.rates_at_depletion.items()
        if result.conclusions[species] is None
    ]
    if failed:
        summary = " ".join(
            [f"Rate at {species}=0 is {rate}, not zero." for species, rate in failed]
            + [
                f"Could not prove that the rate at {species}=0 is zero: {rate}."
                for species, rate in unknown
            ]
        )
        data = {
            species: {
                "rate": str(rate),
                "point": {
                    str(symbol): str(value)
                    for symbol, value in result.counterexamples.get(species, {}).items()
                },
            }
            for species, rate in failed
        }
        return Finding(result.reaction_id, Verdict.FAIL, summary, Evidence("depletion_rates", data))
    if unknown:
        return Finding(
            result.reaction_id,
            Verdict.UNKNOWN,
            " ".join(
                f"Could not prove that the rate at {species}=0 is zero: {rate}."
                for species, rate in unknown
            ),
        )
    summary = (
        "Rate is exactly zero at every required depletion boundary."
        if result.rates_at_depletion
        else "Reaction has no reactants or catalysts."
    )
    return Finding(result.reaction_id, Verdict.PASS, summary)


def run(context: AnalysisContext, dependencies: Mapping) -> tuple[Finding, ...]:
    return tuple(
        reaction_skip(dependencies, "physical_rate_definedness", reaction.id)
        or _finding(
            check_zero_at_depletion(reaction, context.case.symbols, context.physical_domain)
        )
        for reaction in context.case.reactions
    )
