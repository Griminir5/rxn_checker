from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import sympy as sp

from rxn_checker import StateVariables, load_case
from rxn_checker.state import GAS_CONSTANT_J_PER_MOL_K


BOUNDS_YAML = """\
bounds:
  temperature: [200.0, 1500.0]
  pressure: [10000.0, 10000000.0]
  concentrations:
    default:
      upper: 1000.0
      excursion_lower: -0.1
"""


class StateVariablesTests(unittest.TestCase):
    def test_concentrations_temperature_and_pressure_are_case_owned(self) -> None:
        states = StateVariables(("Aye", "Bee"))
        self.assertEqual(states.concentration("Aye").name, "Aye")
        self.assertEqual(states.concentration("Bee").name, "Bee")
        self.assertEqual(states.temperature.name, "temperature")
        self.assertEqual(states.pressure.name, "pressure")


class CaseLoadingTests(unittest.TestCase):
    def test_case_yaml_loads_reactions_and_state_bounds(self) -> None:
        path = Path(__file__).parents[1] / "example_case" / "case.yaml"
        case = load_case(path)

        self.assertEqual(case.states.species_ids, ("Aye", "Bee", "Cee"))
        self.assertEqual(
            tuple(reaction.id for reaction in case.reactions),
            (
                "aye_to_bee.autocatalytic",
                "aye_plus_bee_to_cee.half_order",
            ),
        )
        self.assertEqual(
            case.state_bounds[case.states.temperature].interval(),
            (200.0, 1500.0),
        )
        self.assertEqual(
            case.state_bounds[case.states.pressure].interval(),
            (10_000.0, 10_000_000.0),
        )
        aye_bounds = case.state_bounds[case.states.concentration("Aye")]
        bee_bounds = case.state_bounds[case.states.concentration("Bee")]
        self.assertEqual(aye_bounds.interval(), (0.0, 100.0))
        self.assertEqual(aye_bounds.interval(include_excursion=True), (-0.1, 100.0))
        self.assertEqual(bee_bounds.interval(), (0.0, 1000.0))
        self.assertIsNotNone(case.gas_closure)
        closure = case.gas_closure
        self.assertEqual(closure.minimum_total, "positive")
        self.assertEqual(
            closure.gas_concentrations,
            tuple(case.states.concentrations.values()),
        )
        self.assertEqual(closure.equation.lhs, case.states.pressure)
        self.assertEqual(
            sp.simplify(
                closure.equation.rhs
                - GAS_CONSTANT_J_PER_MOL_K
                * case.states.temperature
                * sp.Add(*case.states.concentrations.values())
            ),
            0,
        )

    def test_temperature_must_have_a_positive_lower_bound(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "case.yaml"
            path.write_text(
                "species:\n  - Aye\n  - Bee\n"
                "reactions:\n  - aye_to_bee.simple\n"
                + BOUNDS_YAML.replace(
                    "temperature: [200.0, 1500.0]",
                    "temperature: [0.0, 1500.0]",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be positive"):
                load_case(path)

    def test_pressure_must_have_a_positive_lower_bound(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "case.yaml"
            path.write_text(
                "species:\n  - Aye\n  - Bee\n"
                "reactions:\n  - aye_to_bee.simple\n"
                + BOUNDS_YAML.replace(
                    "pressure: [10000.0, 10000000.0]",
                    "pressure: [0.0, 10000000.0]",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be positive"):
                load_case(path)

    def test_ideal_gas_minimum_is_derived_from_pressure_and_temperature(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "case.yaml"
            path.write_text(
                "species:\n  - Aye\n  - Bee\n"
                "reactions:\n  - aye_to_bee.simple\n"
                "gas_closure:\n  minimum_total: ideal_gas\n"
                + BOUNDS_YAML,
                encoding="utf-8",
            )

            case = load_case(path)

        expected = sp.Rational(10_000) / (
            sp.Rational(str(GAS_CONSTANT_J_PER_MOL_K)) * 1500
        )
        self.assertEqual(case.gas_closure.minimum_total, "ideal_gas")
        self.assertEqual(
            case.gas_closure.derived_minimum_total(case.state_bounds),
            expected,
        )

    def test_unknown_gas_closure_minimum_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "case.yaml"
            path.write_text(
                "species:\n  - Aye\n  - Bee\n"
                "reactions:\n  - aye_to_bee.simple\n"
                "gas_closure:\n  minimum_total: guessed\n"
                + BOUNDS_YAML,
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "'positive' or 'ideal_gas'"):
                load_case(path)

    def test_bounds_are_required(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "case.yaml"
            path.write_text(
                "species:\n  - Aye\n  - Bee\n"
                "reactions:\n  - aye_to_bee.simple\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "'bounds' must be"):
                load_case(path)

    def test_concentration_override_must_name_a_case_species(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "case.yaml"
            path.write_text(
                "species:\n  - Aye\n  - Bee\n"
                "reactions:\n  - aye_to_bee.simple\n"
                + BOUNDS_YAML
                + "    overrides:\n      Missing:\n        upper: 10.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown species: Missing"):
                load_case(path)

    def test_negative_excursion_does_not_change_physical_lower_bound(self) -> None:
        case = self.load_selectors(("aye_to_bee.simple",))
        aye_bounds = case.state_bounds[case.states.concentration("Aye")]

        self.assertEqual(aye_bounds.interval(), (0.0, 1000.0))
        self.assertEqual(
            aye_bounds.interval(include_excursion=True),
            (-0.1, 1000.0),
        )

    def test_inert_species_are_loaded_as_strictly_positive(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "case.yaml"
            path.write_text(
                "species:\n  - Aye\n  - Bee\n  - Cee\n"
                "inerts:\n  - Cee\n"
                "reactions:\n  - aye_to_bee.simple\n"
                + BOUNDS_YAML,
                encoding="utf-8",
            )

            case = load_case(path)

        cee = case.states.concentration("Cee")
        self.assertEqual(case.inert_species, ("Cee",))
        self.assertTrue(case.state_bounds[cee].strict_lower)
        self.assertEqual(case.state_bounds[cee].physical_lower, 0.0)

    def test_ideal_gas_closure_excludes_solid_concentrations(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "case.yaml"
            path.write_text(
                "species:\n  - Aye\n  - Bee\n  - Ni\n"
                "reactions:\n  - aye_to_bee.simple\n"
                + BOUNDS_YAML,
                encoding="utf-8",
            )

            case = load_case(path)

        self.assertEqual(
            case.gas_closure.gas_concentrations,
            (
                case.states.concentration("Aye"),
                case.states.concentration("Bee"),
            ),
        )

    def test_inert_species_must_be_known_and_nonparticipating(self) -> None:
        for inert, message in (
            ("Missing", "unknown inert species: Missing"),
            ("Aye", "participate in reaction 'aye_to_bee.simple': Aye"),
        ):
            with self.subTest(inert=inert), TemporaryDirectory() as directory:
                path = Path(directory) / "case.yaml"
                path.write_text(
                    "species:\n  - Aye\n  - Bee\n"
                    f"inerts:\n  - {inert}\n"
                    "reactions:\n  - aye_to_bee.simple\n"
                    + BOUNDS_YAML,
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    load_case(path)

    def load_selectors(self, selectors: tuple[str, ...]):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "case.yaml"
            reaction_lines = (
                "\n".join(f"  - {selector}" for selector in selectors) or "  []"
            )
            path.write_text(
                "species:\n  - Aye\n  - Bee\nreactions:\n"
                + reaction_lines
                + "\n"
                + BOUNDS_YAML,
                encoding="utf-8",
            )
            return load_case(path)

    def test_bare_family_selects_every_reaction_in_that_family(self) -> None:
        case = self.load_selectors(("aye_to_bee",))
        self.assertEqual(
            tuple(reaction.id for reaction in case.reactions),
            ("aye_to_bee.autocatalytic", "aye_to_bee.simple"),
        )

    def test_overlapping_selectors_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "selected more than once"):
            self.load_selectors(("aye_to_bee", "aye_to_bee.simple"))

    def test_unknown_reaction_names_are_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown reaction"):
            self.load_selectors(("aye_to_bee.missing",))

    def test_empty_reaction_lists_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one reaction"):
            self.load_selectors(())
