"""compact nonblocking analyses."""

import json
from pathlib import Path

import sympy as sp

from rxn_checker import (
    AnalysisContext,
    Case,
    CaseSymbols,
    ConcentrationModel,
    DomainSpec,
    Interval,
    Reaction,
    Verdict,
    load_case,
)
from rxn_checker.checks import run_checks
from rxn_checker.reporting import render_json, render_text
from rxn_checker.species import PROPERTY_REGISTRY

ROOT = Path(__file__).parents[1]


def _example():
    return load_case(ROOT / "example_case")


def test_conservation_reports_an_exact_primitive_left_nullspace_basis() -> None:
    case = _example()
    result = run_checks(case, only=("conserved_quantities",))
    summary, quantity = result.results["conserved_quantities"].findings

    assert result.overall is Verdict.PASS
    assert summary.evidence.data == {
        "rank": 2,
        "shape": (3, 2),
        "connected_components": (("Aye", "Bee", "Cee"),),
        "unchanged_species": (),
        "basis_size": 1,
    }
    assert quantity.evidence.data["coefficients"] == {"Aye": 1, "Bee": 1, "Cee": 2}
    vector = sp.Matrix((1, 1, 2))
    assert (vector.T * AnalysisContext(case).stoichiometry).is_zero_matrix
    assert quantity.summary == "Aye + Bee + 2*Cee = constant."


def test_reforming_conservation_separates_unchanged_species_and_components() -> None:
    result = run_checks(load_case(ROOT / "reforming_case"), only=("conserved_quantities",))
    evidence = result.results["conserved_quantities"].findings[0].evidence.data

    assert evidence["rank"] == 4
    assert evidence["basis_size"] == 8
    assert evidence["unchanged_species"] == ("Ar", "He", "N2", "CaAl2O4")
    assert ("Ar",) in evidence["connected_components"]
    assert ("CaAl2O4",) in evidence["connected_components"]


def test_structural_faces_use_required_and_production_supports() -> None:
    result = run_checks(_example(), only=("structural_faces",))
    finding = result.results["structural_faces"].findings[0]
    evidence = finding.evidence.data

    assert finding.verdict is Verdict.PASS
    assert evidence["dead_faces"] == (("Aye",), ("Bee",))
    assert evidence["invariant_faces"] == (("Aye",), ("Bee",))
    assert evidence["required_supports"] == {
        "aye_to_bee.autocatalytic": ("Aye", "Bee"),
        "aye_plus_bee_to_cee.half_order": ("Aye", "Bee"),
    }
    assert evidence["search_truncated"] is False


def test_catalyst_depletion_structurally_disables_a_reaction() -> None:
    symbols = CaseSymbols.for_species(("Aye", "Bee", "Cee"))
    aye = symbols.concentration("Aye")
    cee = symbols.concentration("Cee")
    reaction = Reaction("test.catalysed", {"Aye": 1}, {"Bee": 1}, ("Cee",), aye * cee)
    domain = DomainSpec(
        symbols,
        ConcentrationModel.INDEPENDENT,
        {symbol: 10 for symbol in symbols.concentration_symbols},
        {symbol: -1 for symbol in symbols.concentration_symbols},
        {symbols.temperature: Interval(300, 1000), symbols.pressure: Interval(100_000, 200_000)},
    )
    case = Case(
        "catalysed",
        tuple(PROPERTY_REGISTRY.get_record(item) for item in symbols.species_ids),
        symbols,
        (reaction,),
        domain,
    )

    result = run_checks(case, only=("structural_faces",))
    evidence = result.results["structural_faces"].findings[0].evidence.data

    assert evidence["required_supports"] == {"test.catalysed": ("Aye", "Cee")}
    assert evidence["dead_faces"] == (("Aye",), ("Cee",))
    assert evidence["invariant_faces"] == (("Aye",), ("Cee",))


def test_steady_state_equations_are_sparse_and_row_independent() -> None:
    case = _example()
    context = AnalysisContext(case)
    result = run_checks(case, only=("steady_state_equations",), context=context)
    summary, first, second = result.results["steady_state_equations"].findings

    assert summary.evidence.data["rank"] == 2
    assert (first.subject, second.subject) == ("Cee", "Aye")
    assert first.evidence.data["coefficients"] == {"aye_plus_bee_to_cee.half_order": 1}
    assert second.evidence.data["coefficients"] == {
        "aye_to_bee.autocatalytic": -1,
        "aye_plus_bee_to_cee.half_order": -1,
    }
    selected = context.stoichiometry.extract((2, 0), range(2))
    assert selected.rank() == context.stoichiometry.rank() == 2
    assert "source_vector" not in context.network.__dict__


def test_analysis_text_is_compact_and_json_retains_full_equations() -> None:
    case = _example()
    result = run_checks(case, only=("steady_state_equations",))
    text = render_text(result)
    payload = json.loads(render_json(result))

    assert "F_Cee = r[aye_plus_bee_to_cee.half_order] = 0" in text
    findings = payload["results"]["steady_state_equations"]["findings"]
    assert "expression" in findings[1]["evidence"]["data"]


def test_analyses_continue_after_a_physical_failure() -> None:
    result = run_checks(_example(), profile="analysis")

    assert result.results["physical_lipschitz"].verdict is Verdict.FAIL
    assert result.results["conserved_quantities"].verdict is Verdict.PASS
    assert result.results["structural_faces"].verdict is Verdict.PASS
    assert result.results["steady_state_equations"].verdict is Verdict.PASS
    assert result.overall is Verdict.FAIL
