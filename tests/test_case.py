from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rxn_checker import StateVariables, load_case


class StateVariablesTests(unittest.TestCase):
    def test_concentrations_temperature_and_pressure_are_case_owned(self) -> None:
        states = StateVariables(("Aye", "Bee"))
        self.assertEqual(states.concentration("Aye").name, "Aye")
        self.assertEqual(states.concentration("Bee").name, "Bee")
        self.assertEqual(states.temperature.name, "temperature")
        self.assertEqual(states.pressure.name, "pressure")


class CaseLoadingTests(unittest.TestCase):
    def test_case_yaml_selects_one_reaction_from_a_family(self) -> None:
        path = Path(__file__).parents[1] / "example_case" / "case.yaml"
        case = load_case(path)

        self.assertEqual(case.states.species_ids, ("Aye", "Bee"))
        self.assertEqual(case.reaction_ids, ("aye_to_bee.simple",))
        self.assertEqual(
            tuple(reaction.id for reaction in case.reactions),
            ("aye_to_bee.simple",),
        )
        for reaction in case.reactions:
            self.assertLessEqual(reaction.rate.free_symbols, case.states.symbols)

    def load_selectors(self, selectors: tuple[str, ...]):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "case.yaml"
            reaction_lines = "\n".join(f"  - {selector}" for selector in selectors)
            path.write_text(
                "species:\n  - Aye\n  - Bee\nreactions:\n" + reaction_lines + "\n",
                encoding="utf-8",
            )
            return load_case(path)

    def test_bare_family_selects_every_reaction_in_that_family(self) -> None:
        case = self.load_selectors(("aye_to_bee",))
        self.assertEqual(
            case.reaction_ids,
            ("aye_to_bee.autocatalytic", "aye_to_bee.simple"),
        )

    def test_overlapping_selectors_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "selected more than once"):
            self.load_selectors(("aye_to_bee", "aye_to_bee.simple"))

    def test_unknown_reaction_names_are_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown reaction"):
            self.load_selectors(("aye_to_bee.missing",))


if __name__ == "__main__":
    unittest.main()
