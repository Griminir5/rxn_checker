"""Mathematical regression tests for unified concentration domains."""

from pathlib import Path

import pytest
from sympy import Rational

from rxn_checker import (
    AnalysisContext,
    CaseSymbols,
    ConcentrationModel,
    DomainKind,
    DomainSpec,
    Interval,
    TotalConstraint,
    affine_form,
    load_case,
)
from rxn_checker.domain import GAS_CONSTANT

ROOT = Path(__file__).parents[1]


def _spec(
    *,
    model: ConcentrationModel = ConcentrationModel.INDEPENDENT,
    constraint: TotalConstraint | None = None,
) -> DomainSpec:
    symbols = CaseSymbols.for_species(("A", "B"))
    return DomainSpec(
        symbols=symbols,
        concentration_model=model,
        upper={symbols.concentration("A"): 10, symbols.concentration("B"): 10},
        excursion_lower={symbols.concentration("A"): -1, symbols.concentration("B"): -2},
        parameter_intervals={
            symbols.temperature: Interval(300, 1000),
            symbols.pressure: Interval(100_000, 200_000),
        },
        total_constraints=() if constraint is None else (constraint,),
    )


def test_independent_spec_generates_physical_and_augmented_domains() -> None:
    spec = _spec()
    aye = spec.symbols.concentration("A")
    temperature = spec.symbols.temperature

    physical = spec.build(DomainKind.PHYSICAL)
    augmented = spec.build(DomainKind.AUGMENTED)

    assert physical.interval(aye) == Interval(0, 10)
    assert augmented.interval(aye) == Interval(-1, 10)
    assert physical.interval(temperature) == augmented.interval(temperature)
    assert physical.total_constraints == augmented.total_constraints == ()


def test_reforming_domain_has_gas_and_selected_solid_chamfers() -> None:
    case = load_case(ROOT / "reforming_case")
    context = AnalysisContext(case)
    physical = context.physical_domain
    augmented = context.augmented_domain
    constraints = {item.name: item for item in physical.total_constraints}

    expected_gas_minimum = Rational(100_000) / (GAS_CONSTANT * Rational("1473.15"))
    assert constraints["gas"].minimum == expected_gas_minimum
    assert tuple(symbol.name for symbol in constraints["solid"].symbols) == ("Ni", "NiO")
    assert constraints["solid"].minimum == Rational("1.0e-8")
    assert physical.total_constraints == augmented.total_constraints
    assert context.physical_domain is physical
    assert context.augmented_domain is augmented


def test_explicit_gas_minimum_loads_exactly(tmp_path: Path) -> None:
    case_path = tmp_path / "case.yaml"
    case_path.write_text(
        """\
schema: 1
species: [Aye, Bee]
reactions: [aye_to_bee.simple]
parameters:
  temperature: [300, 1000]
  pressure: [100000, 200000]
domain:
  concentration_model: chamfered
  upper:
    default: 10
  excursion_lower:
    default: -0.1
  totals:
    gas:
      mode: explicit
      value: "3/2"
""",
        encoding="utf-8",
    )

    case = load_case(case_path)

    constraint = case.domain_spec.total_constraints[0]
    assert constraint.name == "gas"
    assert constraint.minimum == Rational(3, 2)


def test_augmented_witness_can_keep_one_gas_negative() -> None:
    case = load_case(ROOT / "reforming_case")
    domain = AnalysisContext(case).augmented_domain
    argon = case.symbols.concentration("Ar")
    methane = case.symbols.concentration("CH4")

    point = domain.exact_witness({argon: Rational("-0.05"), methane: 1000})

    assert point is not None
    assert point[argon] < 0
    gas = next(item for item in domain.total_constraints if item.name == "gas")
    assert sum(point[symbol] for symbol in gas.symbols) >= gas.minimum


def test_infeasible_total_minimum_is_rejected() -> None:
    symbols = CaseSymbols.for_species(("A", "B"))
    constraint = TotalConstraint(
        "gas", (symbols.concentration("A"), symbols.concentration("B")), 21
    )

    with pytest.raises(ValueError, match="Physical concentration domain is empty"):
        DomainSpec(
            symbols,
            ConcentrationModel.CHAMFERED,
            {symbol: 10 for symbol in symbols.concentration_symbols},
            {symbol: -1 for symbol in symbols.concentration_symbols},
            {
                symbols.temperature: Interval(300, 1000),
                symbols.pressure: Interval(100_000, 200_000),
            },
            (constraint,),
        )


def test_affine_extraction_and_independent_bounds_are_exact() -> None:
    case = load_case(ROOT / "example_case")
    domain = AnalysisContext(case).physical_domain
    aye = case.symbols.concentration("Aye")
    bee = case.symbols.concentration("Bee")
    expression = 3 + 2 * aye - bee

    form = affine_form(expression)
    bounds = domain.affine_bounds(expression)

    assert form is not None
    assert form.constant == 3
    assert form.coefficients == {aye: 2, bee: -1}
    assert affine_form(aye * bee) is None
    assert bounds is not None
    assert (bounds.lower, bounds.upper) == (-997, 203)
    assert expression.subs(bounds.lower_witness) == bounds.lower
    assert expression.subs(bounds.upper_witness) == bounds.upper


def test_chamfered_affine_minimum_uses_fractional_knapsack() -> None:
    symbols = CaseSymbols.for_species(("A", "B"))
    aye = symbols.concentration("A")
    bee = symbols.concentration("B")
    constraint = TotalConstraint("gas", (aye, bee), 5)
    spec = DomainSpec(
        symbols,
        ConcentrationModel.CHAMFERED,
        {aye: 10, bee: 10},
        {aye: -1, bee: -2},
        {symbols.temperature: Interval(300, 1000), symbols.pressure: Interval(100_000, 200_000)},
        (constraint,),
    )

    bounds = spec.build(DomainKind.PHYSICAL).affine_bounds(aye + 2 * bee)

    assert bounds is not None
    assert (bounds.lower, bounds.upper) == (5, 30)
    assert bounds.lower_witness[aye] == 5
    assert bounds.lower_witness[bee] == 0


def test_domain_restriction_preserves_totals_and_detects_empty_regions() -> None:
    symbols = CaseSymbols.for_species(("A", "B"))
    aye = symbols.concentration("A")
    bee = symbols.concentration("B")
    spec = DomainSpec(
        symbols,
        ConcentrationModel.CHAMFERED,
        {aye: 10, bee: 10},
        {aye: -1, bee: -2},
        {symbols.temperature: Interval(300, 1000), symbols.pressure: Interval(100_000, 200_000)},
        (TotalConstraint("gas", (aye, bee), 5),),
    )
    physical = spec.build(DomainKind.PHYSICAL)
    augmented = spec.build(DomainKind.AUGMENTED)

    assert not physical.restrict(aye, upper=0).restrict(bee, upper=4).is_feasible()
    negative = augmented.restrict(aye, upper=0, strict_upper=True)
    assert negative.is_feasible()
    assert negative.total_constraints == augmented.total_constraints
    assert negative.exact_witness()[aye] < 0
