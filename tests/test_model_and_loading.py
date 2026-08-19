"""Regression tests for the exact model and schema-1 loader."""

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
from sympy import Rational

import rxn_checker.loading as loading
from rxn_checker import (
    CaseSymbols,
    Phase,
    Reaction,
    Species,
    load_case,
    parse_rational,
)
from rxn_checker.checks.atom_conservation import check_atom_conservation
from rxn_checker.checks.mass_conservation import check_mass_conservation
from rxn_checker.species import PROPERTY_REGISTRY


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (1, Rational(1)),
        (0.5, Rational(1, 2)),
        ("1/2", Rational(1, 2)),
        ("0.94", Rational(47, 50)),
    ),
)
def test_rational_parser_uses_decimal_spelling(value, expected) -> None:
    assert parse_rational(value) == expected


def test_species_and_reaction_quantities_are_exact() -> None:
    wustite = Species(
        id="Fe0.94O",
        name="Wüstite",
        phase=Phase.SOLID,
        atoms={"Fe": "0.94", "O": 1},
        molar_mass="0.0684957",
    )
    symbols = CaseSymbols.for_species(("O2", "O"))
    reaction = Reaction(
        id="example.oxygen_split",
        reactants={"O2": 0.5},
        products={"O": 1},
        catalysts=(),
        rate=symbols.concentration("O2"),
    )

    assert wustite.atoms["Fe"] == Rational(47, 50)
    assert reaction.reactants["O2"] == Rational(1, 2)
    assert reaction.net_stoichiometry == {"O2": Rational(-1, 2), "O": 1}


@pytest.mark.parametrize(
    ("case_directory", "reaction_count"),
    (("example_case", 2), ("reforming_case", 9)),
)
def test_bundled_cases_load(case_directory: str, reaction_count: int) -> None:
    case = load_case(ROOT / case_directory)

    assert len(case.reactions) == reaction_count
    assert case.symbols.concentration_symbols.isdisjoint(
        case.symbols.parameter_symbols
    )
    assert case.domain_spec.parameter_intervals[
        case.symbols.temperature
    ].lower.is_Rational
    assert case.domain_spec.parameter_intervals[
        case.symbols.pressure
    ].upper.is_Rational


def test_built_in_wustite_composition_is_fractional_and_exact() -> None:
    assert PROPERTY_REGISTRY.get_record("Fe0.94O").atoms["Fe"] == Rational(47, 50)


def test_fractional_wustite_decomposition_balances() -> None:
    reaction = Reaction(
        id="example.wustite_decomposition",
        reactants={"Fe0.94O": 1},
        products={"Fe": "0.94", "O2": "1/2"},
        catalysts=(),
        rate=Rational(1),
    )

    assert check_atom_conservation(reaction).passed
    assert check_mass_conservation(reaction).passed


def test_loading_imports_only_selected_built_in_families() -> None:
    script = """
import sys
from rxn_checker import load_case

load_case('example_case')
assert 'rxn_checker.reactions.aye_to_bee' in sys.modules
assert 'rxn_checker.reactions.aye_plus_bee_to_cee' in sys.modules
assert 'rxn_checker.reactions.medrano' not in sys.modules
assert 'rxn_checker.reactions.xu_froment' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_selected_family_is_built_once(tmp_path: Path, monkeypatch) -> None:
    case_path = tmp_path / "case.yaml"
    case_path.write_text(
        """\
schema: 1
species: [Aye, Bee]
reactions: [probe.forward, probe.backward]
parameters:
  temperature: [300, 1000]
  pressure: [100000, 200000]
domain:
  concentration_model: independent
  upper:
    default: 10
  excursion_lower:
    default: -0.1
""",
        encoding="utf-8",
    )
    calls = 0

    def build_family(symbols: CaseSymbols):
        nonlocal calls
        calls += 1
        aye = symbols.concentration("Aye")
        bee = symbols.concentration("Bee")
        return {
            "forward": Reaction(
                "probe.forward", {"Aye": 1}, {"Bee": 1}, (), aye
            ),
            "backward": Reaction(
                "probe.backward", {"Bee": 1}, {"Aye": 1}, (), bee
            ),
        }

    monkeypatch.setattr(
        loading,
        "_family_module",
        lambda case_directory, family_id: SimpleNamespace(
            build_family=build_family
        ),
    )

    case = load_case(case_path)

    assert calls == 1
    assert tuple(reaction.id for reaction in case.reactions) == (
        "probe.forward",
        "probe.backward",
    )


def test_xu_froment_shared_terms_are_built_once(monkeypatch) -> None:
    from rxn_checker.reactions import xu_froment

    original = xu_froment.xu_froment_terms
    calls = 0

    def counting_terms(symbols: CaseSymbols):
        nonlocal calls
        calls += 1
        return original(symbols)

    monkeypatch.setattr(xu_froment, "xu_froment_terms", counting_terms)

    load_case(ROOT / "reforming_case")

    assert calls == 1


def test_unselected_local_family_is_not_executed(tmp_path: Path) -> None:
    reactions_directory = tmp_path / "reactions"
    reactions_directory.mkdir()
    (reactions_directory / "broken.py").write_text(
        "raise RuntimeError('unselected module was imported')\n",
        encoding="utf-8",
    )
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
  concentration_model: independent
  upper:
    default: 10
  excursion_lower:
    default: -0.1
""",
        encoding="utf-8",
    )

    case = load_case(case_path)

    assert tuple(reaction.id for reaction in case.reactions) == (
        "aye_to_bee.simple",
    )


def test_selected_local_family_loads(tmp_path: Path) -> None:
    reactions_directory = tmp_path / "reactions"
    reactions_directory.mkdir()
    (reactions_directory / "local_decay.py").write_text(
        """\
from rxn_checker import Reaction


def build_family(symbols):
    aye = symbols.concentration("Aye")
    return {
        "simple": Reaction(
            id="local_decay.simple",
            reactants={"Aye": 1},
            products={"Bee": 1},
            catalysts=(),
            rate=aye,
        )
    }
""",
        encoding="utf-8",
    )
    case_path = tmp_path / "case.yaml"
    case_path.write_text(
        """\
schema: 1
species: [Aye, Bee]
reactions: [local_decay]
parameters:
  temperature: [300, 1000]
  pressure: [100000, 200000]
domain:
  concentration_model: independent
  upper:
    default: 10
  excursion_lower:
    default: -0.1
""",
        encoding="utf-8",
    )

    case = load_case(case_path)

    assert tuple(reaction.id for reaction in case.reactions) == (
        "local_decay.simple",
    )
