"""DAETools-oriented structural evaluation profile tests."""

from pathlib import Path

import sympy as sp

from rxn_checker import (
    Case,
    CaseSymbols,
    ConcentrationModel,
    DomainSpec,
    Interval,
    Reaction,
    Verdict,
    load_case,
)
from rxn_checker.checks import plan_checks, run_checks
from rxn_checker.proof import OPERATION_ORDER, profile_evaluation
from rxn_checker.species import PROPERTY_REGISTRY


ROOT = Path(__file__).parents[1]


def _stats(expression, concentrations, operating=()):
    reaction = Reaction("test.rate", {"Aye": 1}, {"Bee": 1}, (), expression)
    profile = profile_evaluation(
        (reaction,), sp.Matrix(((-1,), (1,))), concentrations, operating
    )
    return profile.declared_outputs[0][1]


def test_operation_classification_uses_the_fixed_taxonomy() -> None:
    a, b, c = sp.symbols("a b c", real=True)
    cases = (
        (a + b + c, {"add": 2}),
        (a * b * c, {"multiply": 2}),
        (a / b, {"multiply": 1, "reciprocal": 1}),
        (sp.sqrt(a), {"sqrt": 1}),
        (a**3, {"integer_power": 1}),
        (a ** sp.Rational(3, 5), {"general_power": 1}),
        (sp.exp(a), {"exp": 1}),
        (sp.Max(a, 0), {"max": 1}),
    )

    for expression, expected in cases:
        stats = _stats(expression, (a, b, c))
        operations = dict(stats.operations)
        assert tuple(operations) == OPERATION_ORDER
        assert {name: value for name, value in operations.items() if value} == expected


def test_dependencies_and_ad_work_include_operating_variables() -> None:
    a, temperature, energy = sp.symbols("a temperature energy", real=True)
    stats = _stats(
        sp.exp(-energy / temperature) * a,
        (a,),
        (temperature,),
    )

    assert stats.concentration_dependencies == (a,)
    assert stats.operating_dependencies == (temperature,)
    assert stats.dae_dependencies == (a, temperature)
    assert stats.structural_jacobian_entries == 2
    assert stats.ad_work == 6


def test_local_and_cross_output_cse_measure_exact_reuse() -> None:
    a, b, temperature = sp.symbols("a b temperature", real=True)
    shared = a + b
    local_reaction = Reaction(
        "test.local", {"Aye": 1}, {"Bee": 1}, (), shared**2 + sp.exp(shared)
    )
    local = profile_evaluation(
        (local_reaction,), sp.Matrix(((-1,), (1,))), (a, b), ()
    )
    local_raw = local.declared_outputs[0][1].total_operations
    local_cse = local.declared_local_cse[0][1]

    assert local_cse.temporary_count == 1
    assert local_cse.total_operations < local_raw
    assert local_cse.peak_live_temporaries == 1

    arrhenius = sp.exp(-1 / temperature)
    reactions = (
        Reaction("test.first", {"Aye": 1}, {"Bee": 1}, (), arrhenius * a),
        Reaction("test.second", {"Aye": 1}, {"Bee": 1}, (), arrhenius * b),
    )
    cross = profile_evaluation(
        reactions,
        sp.Matrix(((-1, -1), (1, 1))),
        (a, b),
        (temperature,),
    )
    cross_raw = sum(stats.total_operations for _, stats in cross.declared_outputs)

    assert all(stats.temporary_count == 0 for _, stats in cross.declared_local_cse)
    assert cross.declared_cse.temporary_count >= 1
    assert cross.declared_cse.total_operations < cross_raw


def test_proportional_stoichiometry_forms_deterministic_exact_fluxes() -> None:
    rates = sp.symbols("r0:4", real=True)
    reactions = tuple(
        Reaction(f"test.r{index}", {"Aye": 1}, {"Bee": 1}, (), rate)
        for index, rate in enumerate(rates)
    )
    stoichiometry = sp.Matrix(
        (
            (-1, 1, -2, 0),
            (1, -1, 2, -1),
            (0, 0, 0, 1),
        )
    )

    profile = profile_evaluation(reactions, stoichiometry, rates, ())

    assert len(profile.flux_outputs) == 2
    first_group, second_group = profile.flux_groups
    assert first_group["stoichiometry"] == (-1, 1, 0)
    assert first_group["members"] == (
        ("test.r0", 1),
        ("test.r1", -1),
        ("test.r2", 2),
    )
    assert second_group["members"] == (("test.r3", 1),)
    assert sp.simplify(first_group["expression"] - (rates[0] - rates[1] + 2 * rates[2])) == 0
    assert profile.declared_source_nnz == 8
    assert profile.flux_source_nnz == 4


def _unsupported_case() -> Case:
    symbols = CaseSymbols.for_species(("Aye", "Bee"))
    aye = symbols.concentration("Aye")
    reaction = Reaction(
        "test.unsupported",
        {"Aye": 1},
        {"Bee": 1},
        (),
        sp.Piecewise((sp.sin(aye), aye > 0), (0, True)),
    )
    domain = DomainSpec(
        symbols,
        ConcentrationModel.INDEPENDENT,
        {symbol: 10 for symbol in symbols.concentration_symbols},
        {symbol: -1 for symbol in symbols.concentration_symbols},
        {
            symbols.temperature: Interval(300, 1000),
            symbols.pressure: Interval(100_000, 200_000),
        },
    )
    return Case(
        "unsupported",
        tuple(PROPERTY_REGISTRY.get_record(item) for item in symbols.species_ids),
        symbols,
        (reaction,),
        domain,
    )


def test_unsupported_functions_keep_partial_statistics_nonblocking() -> None:
    result = run_checks(_unsupported_case(), only=("evaluation_profile",))
    finding = result.results["evaluation_profile"].findings[0]
    evidence = finding.evidence.data
    output = evidence["outputs"]["declared"]["test.unsupported"]

    assert finding.verdict is Verdict.UNKNOWN
    assert result.overall is Verdict.PASS
    assert output["raw"]["total_operations"] > 0
    assert set(output["unsupported_functions"]) == {"Piecewise", "sin"}
    assert {item["function_name"] for item in evidence["unsupported_operations"]} == {
        "Piecewise",
        "sin",
    }


def test_reforming_profile_has_expected_structure_and_dependencies() -> None:
    case = load_case(ROOT / "reforming_case")
    result = run_checks(case, only=("evaluation_profile",), debug=True)
    finding = result.results["evaluation_profile"].findings[0]
    evidence = finding.evidence.data
    declared = evidence["declared"]
    fluxes = evidence["source_equivalent"]

    assert finding.verdict is Verdict.PASS
    assert declared["output_count"] == 9
    assert fluxes["output_count"] == 6
    assert declared["global_cse"]["total_operations"] <= declared["raw"]["total_operations"]
    assert fluxes["global_cse"]["total_operations"] <= fluxes["raw"]["total_operations"]
    assert declared["raw"]["operations"]["exp"] > 0
    assert declared["raw"]["operations"]["general_power"] > 0
    assert declared["raw"]["switch_operations"] > 0
    assert not evidence["unsupported_operations"]
    assert any(len(term["outputs"]) > 1 for term in evidence["shared_terms"])

    output = evidence["outputs"]["declared"]["medrano.reduction_h2"]
    dependencies = set(output["dependencies"]["temperature_pressure"])
    assert dependencies == {case.symbols.temperature, case.symbols.pressure}
    residual = fluxes["residual_structure"]
    output_stats = evidence["outputs"]["source_equivalent"]
    assert residual["rate_input_structural_entries"] == sum(
        value["dependencies"]["structural_jacobian_entries"]
        for value in output_stats.values()
    )
    assert residual["total_kinetics_block_structural_entries"] == (
        residual["rate_equations"]
        + residual["rate_input_structural_entries"]
        + residual["stoichiometric_source_links"]
    )


def test_analysis_and_all_profiles_select_evaluation_profile() -> None:
    for name in ("analysis", "all"):
        assert "evaluation_profile" in {
            spec.id for spec in plan_checks(profile=name)
        }
