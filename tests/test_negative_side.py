"""augmented-domain recovery examples."""

import sympy as sp

from rxn_checker import (
    Case,
    CaseSymbols,
    ConcentrationModel,
    DomainSpec,
    Interval,
    Reaction,
    Verdict,
)
from rxn_checker.checks import run_checks
from rxn_checker.reporting import render_text
from rxn_checker.species import PROPERTY_REGISTRY


def _case(rate_builder, *, reactants=None, products=None, negative=("Aye",), inert=()) -> Case:
    species_ids = ("Aye", "Bee", "Cee")
    symbols = CaseSymbols.for_species(species_ids)
    reaction = Reaction(
        "test.reaction", reactants or {"Aye": 1}, products or {"Bee": 1}, (), rate_builder(symbols)
    )
    domain = DomainSpec(
        symbols,
        ConcentrationModel.INDEPENDENT,
        {symbol: 10 for symbol in symbols.concentration_symbols},
        {
            symbols.concentration(species_id): (-1 if species_id in negative else 0)
            for species_id in species_ids
        },
        {symbols.temperature: Interval(300, 1000), symbols.pressure: Interval(100_000, 200_000)},
    )
    return Case(
        "negative_side",
        tuple(PROPERTY_REGISTRY.get_record(item) for item in species_ids),
        symbols,
        (reaction,),
        domain,
        inert,
    )


def _run(case: Case):
    return run_checks(case, only=("negative_side_nonrepulsion",))


def test_linear_consumption_is_nonrepelling_and_strictly_attracting() -> None:
    case = _case(lambda symbols: 2 * symbols.concentration("Aye"))
    result = _run(case)
    finding = result.results["negative_side_nonrepulsion"].findings[0]

    assert result.overall is Verdict.PASS
    assert finding.verdict is Verdict.PASS
    assert finding.evidence.data["non_repulsion"]["verdict"] == "proved"
    assert finding.evidence.data["strict_attraction"]["verdict"] == "proved"
    assert finding.evidence.data["strict_attraction"]["stuck"] is False
    assert "strict attraction: 1 proved" in render_text(result)


def test_clamped_consumption_is_nonrepelling_but_stuck_when_negative() -> None:
    case = _case(lambda symbols: 2 * sp.Max(symbols.concentration("Aye"), 0))
    finding = _run(case).results["negative_side_nonrepulsion"].findings[0]

    assert finding.verdict is Verdict.PASS
    strict = finding.evidence.data["strict_attraction"]
    assert strict["verdict"] == "disproved"
    assert strict["stuck"] is True


def test_absolute_value_rate_worsens_a_negative_reactant() -> None:
    case = _case(lambda symbols: sp.Abs(symbols.concentration("Aye")))
    finding = _run(case).results["negative_side_nonrepulsion"].findings[0]

    assert finding.verdict is Verdict.FAIL
    obstruction = finding.evidence.data["non_repulsion"]
    aye = case.symbols.concentration("Aye")
    assert obstruction["point"]["Aye"] == -1
    assert obstruction["source_value"] == -1
    assert obstruction["point"][str(aye)] == -1


def test_augmented_irregularity_skips_recovery_for_that_source() -> None:
    case = _case(lambda symbols: sp.sqrt(symbols.concentration("Aye")))
    result = _run(case)

    assert result.results["augmented_rate_definedness"].verdict is Verdict.FAIL
    assert result.results["augmented_lipschitz"].verdict is Verdict.SKIPPED
    assert result.results["negative_side_nonrepulsion"].verdict is Verdict.SKIPPED


def test_irregular_unrelated_rate_does_not_block_an_inert_coordinate() -> None:
    case = _case(
        lambda symbols: sp.sqrt(symbols.concentration("Aye")), negative=("Cee",), inert=("Cee",)
    )
    result = _run(case)

    assert result.results["augmented_lipschitz"].verdict is Verdict.FAIL
    recovery = result.results["negative_side_nonrepulsion"]
    assert recovery.verdict is Verdict.PASS
    assert recovery.findings[0].subject == "Cee"
    assert recovery.findings[0].evidence.data["strict_attraction"]["stuck"]


def test_simultaneous_negative_error_is_an_exact_counterexample() -> None:
    case = _case(
        lambda symbols: symbols.concentration("Aye") * symbols.concentration("Bee"),
        reactants={"Aye": 1, "Bee": 1},
        products={"Cee": 1},
        negative=("Aye", "Bee"),
    )
    result = _run(case)
    findings = tuple(
        finding
        for finding in result.results["negative_side_nonrepulsion"].findings
        if finding.subject != case.name
    )

    assert result.results["augmented_lipschitz"].verdict is Verdict.PASS
    assert result.results["negative_side_nonrepulsion"].verdict is Verdict.FAIL
    assert "strict attraction:" in render_text(result)
    for finding in findings:
        point = finding.evidence.data["non_repulsion"]["point"]
        assert point["Aye"] < 0
        assert point["Bee"] < 0
        assert finding.evidence.data["non_repulsion"]["source_value"] < 0


def test_negative_inert_coordinate_is_nonrepelling_but_stuck() -> None:
    case = _case(lambda symbols: symbols.concentration("Aye"), negative=("Cee",), inert=("Cee",))
    finding = _run(case).results["negative_side_nonrepulsion"].findings[0]

    assert finding.subject == "Cee"
    assert finding.verdict is Verdict.PASS
    strict = finding.evidence.data["strict_attraction"]
    assert strict["verdict"] == "disproved"
    assert strict["stuck"] is True
