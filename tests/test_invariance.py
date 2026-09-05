"""chemistry-gate and physical-invariance contracts."""

import sympy as sp

from rxn_checker import (
    AnalysisContext,
    Case,
    CaseSymbols,
    ConcentrationModel,
    DomainKind,
    DomainSpec,
    Interval,
    Phase,
    Reaction,
    Species,
    Verdict,
)
from rxn_checker.checks import run_checks
from rxn_checker.checks.atom_conservation import check_atom_conservation
from rxn_checker.checks.mass_conservation import check_mass_conservation
from rxn_checker.checks.zero_at_depletion import check_zero_at_depletion
from rxn_checker.species import PROPERTY_REGISTRY, PropertyRegistry


def _case(rate_builder, *, registry=PROPERTY_REGISTRY) -> Case:
    symbols = CaseSymbols.for_species(("Aye", "Bee"))
    aye = symbols.concentration("Aye")
    reaction = Reaction("test.forward", {"Aye": 1}, {"Bee": 1}, (), rate_builder(aye))
    domain = DomainSpec(
        symbols,
        ConcentrationModel.INDEPENDENT,
        {symbol: 10 for symbol in symbols.concentration_symbols},
        {symbol: -1 for symbol in symbols.concentration_symbols},
        {symbols.temperature: Interval(300, 1000), symbols.pressure: Interval(100_000, 200_000)},
    )
    return Case(
        "linear_transfer",
        tuple(registry.get_record(item) for item in symbols.species_ids),
        symbols,
        (reaction,),
        domain,
    )


def test_linear_transfer_has_a_forward_invariance_certificate() -> None:
    case = _case(lambda aye: 2 * aye)
    context = AnalysisContext(case)
    result = run_checks(case, only=("forward_invariance",), context=context)

    assert result.overall is Verdict.PASS
    boundary = result.results["physical_boundary_inward"].findings[0]
    assert boundary.verdict is Verdict.PASS
    faces = boundary.evidence.data["faces"]
    assert faces["Aye"]["vanishing_consumers"] == (
        {"reaction": "test.forward", "stoichiometric_coefficient": -1},
    )
    assert faces["Bee"]["nonnegative_contributions"] == (
        {"reaction": "test.forward", "stoichiometric_coefficient": 1},
    )

    certificate = result.results["forward_invariance"].findings[0]
    assert certificate.verdict is Verdict.PASS
    assert certificate.evidence.kind == "forward_invariance_certificate"
    assert certificate.evidence.data["source_lipschitz_constant_bound"] == 2
    assert certificate.evidence.data["boundary_face_count"] == 2
    assert "source_vector" not in context.network.__dict__


def test_consuming_rate_that_does_not_vanish_blocks_invariance() -> None:
    result = run_checks(_case(lambda _aye: sp.S.One), only=("forward_invariance",))

    assert result.results["zero_at_depletion"].verdict is Verdict.FAIL
    assert result.results["physical_boundary_inward"].verdict is Verdict.SKIPPED
    assert result.results["forward_invariance"].verdict is Verdict.SKIPPED
    assert result.overall is Verdict.FAIL


def test_negative_physical_rate_blocks_invariance() -> None:
    result = run_checks(_case(lambda aye: -aye), only=("forward_invariance",))

    assert result.results["rate_nonnegativity"].verdict is Verdict.FAIL
    assert result.results["physical_boundary_inward"].verdict is Verdict.SKIPPED
    assert result.overall is Verdict.FAIL


def test_undefined_rate_skips_only_that_reaction_downstream() -> None:
    case = _case(lambda aye: aye)
    aye = case.symbols.concentration("Aye")
    singular = Reaction("test.singular", {"Aye": 1}, {"Bee": 1}, (), 1 / aye)
    case = Case(
        case.name, case.species, case.symbols, (*case.reactions, singular), case.domain_spec
    )

    result = run_checks(case, only=("physical_lipschitz",))
    findings = result.results["physical_lipschitz"].findings

    assert findings[0].verdict is Verdict.PASS
    assert findings[1].verdict is Verdict.SKIPPED
    assert result.results["physical_rate_definedness"].verdict is Verdict.FAIL


def test_depletion_substitution_uses_the_case_owned_symbol() -> None:
    symbols = CaseSymbols.for_species(("Aye", "Bee"))
    impostor = sp.Symbol("Aye")
    reaction = Reaction("test.impostor", {"Aye": 1}, {"Bee": 1}, (), impostor)

    result = check_zero_at_depletion(reaction, symbols)

    assert result.passed is None
    assert result.rates_at_depletion["Aye"] == impostor


def test_undefined_zero_over_zero_at_depletion_fails() -> None:
    symbols = CaseSymbols.for_species(("Aye", "Bee"))
    aye = symbols.concentration("Aye")
    reciprocal = sp.Pow(aye, -1, evaluate=False)
    rate = sp.Mul(aye, reciprocal, evaluate=False)
    reaction = Reaction("test.undefined", {"Aye": 1}, {"Bee": 1}, (), rate)

    result = check_zero_at_depletion(reaction, symbols)

    assert result.passed is False


def test_exact_physical_witness_disproves_a_nonzero_depletion_rate() -> None:
    case = _case(lambda aye: aye)
    bee = case.symbols.concentration("Bee")
    reaction = Reaction("test.coupled", {"Aye": 1}, {"Bee": 1}, (), bee)

    result = check_zero_at_depletion(
        reaction, case.symbols, case.domain_spec.build(DomainKind.PHYSICAL)
    )

    assert result.passed is False
    assert result.counterexamples["Aye"][bee] == 10


def test_atom_totals_are_exact_and_mass_uses_exact_tolerance_arithmetic() -> None:
    reaction = Reaction(
        "test.fractional", {"Fe0.94O": 1}, {"Fe": "0.94", "O2": "1/2"}, (), sp.S.One
    )

    atoms = check_atom_conservation(reaction, PROPERTY_REGISTRY.records)
    mass = check_mass_conservation(reaction, PROPERTY_REGISTRY.records)
    assert atoms.passed is True
    assert atoms.reactant_totals["Fe"] == sp.Rational(47, 50)
    assert mass.passed is True
    assert mass.reactant_mass.is_Rational
    assert mass.imbalance.is_Rational


def test_exact_atom_and_tolerant_mass_imbalances_fail() -> None:
    reaction = Reaction("test.imbalanced", {"Aye": 1}, {"Cee": 1}, (), sp.S.One)

    atoms = check_atom_conservation(reaction, PROPERTY_REGISTRY.records)
    mass = check_mass_conservation(reaction, PROPERTY_REGISTRY.records)
    assert atoms.passed is False
    assert atoms.imbalances == {"Ex": 1}
    assert mass.passed is False
    assert mass.imbalance == sp.Rational(1, 100)


def test_missing_molar_mass_fails_the_chemistry_gate() -> None:
    records = {
        "Aye": Species("Aye", "Aye", Phase.GAS, {"X": 1}, None),
        "Bee": Species("Bee", "Bee", Phase.GAS, {"X": 1}, 1),
    }
    registry = PropertyRegistry(records)
    case = _case(lambda aye: aye, registry=registry)
    result = run_checks(case, only=("forward_invariance",), context=AnalysisContext(case))

    assert result.results["mass_conservation"].verdict is Verdict.FAIL
    assert result.results["physical_rate_definedness"].verdict is Verdict.SKIPPED
    assert result.overall is Verdict.FAIL
