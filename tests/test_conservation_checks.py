import unittest

from rxn_checker import Reaction
from rxn_checker.checks import (
    check_atom_conservation,
    check_mass_conservation,
)
from rxn_checker.species import PropertyRegistry, SpeciesProperties


class ConservationCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.combustion = Reaction(
            name="combustion",
            family="methane",
            reactants={"CH4": 1, "O2": 2},
            products={"CO2": 1, "H2O": 2},
            rate=1,
        )

    def test_atom_conservation_reports_side_totals_and_imbalances(self) -> None:
        result = check_atom_conservation(self.combustion)

        self.assertTrue(result.passed)
        self.assertEqual(result.reactant_totals, {"C": 1, "H": 4, "O": 4})
        self.assertEqual(result.product_totals, {"C": 1, "O": 4, "H": 4})
        self.assertEqual(result.imbalances, {"C": 0, "H": 0, "O": 0})

    def test_atom_imbalance_identifies_each_unbalanced_element(self) -> None:
        reaction = Reaction(
            name="unbalanced",
            family="methane",
            reactants={"CH4": 1, "O2": 1},
            products={"CO2": 1, "H2O": 1},
            rate=1,
        )

        result = check_atom_conservation(reaction)

        self.assertFalse(result.passed)
        self.assertEqual(result.imbalances, {"C": 0, "H": -2, "O": 1})

    def test_atom_check_ignores_non_consumed_catalysts(self) -> None:
        reaction = Reaction(
            name="catalysed",
            family="methane",
            reactants={"CH4": 1, "O2": 2},
            products={"CO2": 1, "H2O": 2},
            catalysts=("Ni",),
            rate=1,
        )

        result = check_atom_conservation(reaction)

        self.assertTrue(result.passed)
        self.assertNotIn("Ni", result.reactant_totals)
        self.assertNotIn("Ni", result.product_totals)

    def test_mass_conservation_allows_for_tabulated_weight_rounding(self) -> None:
        result = check_mass_conservation(self.combustion)

        self.assertTrue(result.passed)
        self.assertAlmostEqual(
            result.imbalance,
            result.product_mass - result.reactant_mass,
        )

    def test_mass_tolerance_handles_lower_precision_registry_entries(self) -> None:
        reaction = Reaction(
            name="oxidation",
            family="copper",
            reactants={"Cu": 2, "O2": 0.5},
            products={"Cu2O": 1},
            rate=1,
        )

        self.assertTrue(check_atom_conservation(reaction).passed)
        self.assertTrue(check_mass_conservation(reaction).passed)

    def test_mass_imbalance_is_reported_in_kg_per_mol(self) -> None:
        reaction = Reaction(
            name="unbalanced",
            family="methane",
            reactants={"CH4": 1},
            products={"CO2": 1},
            rate=1,
        )

        result = check_mass_conservation(reaction)

        self.assertFalse(result.passed)
        self.assertAlmostEqual(result.reactant_mass, 16.043e-3)
        self.assertAlmostEqual(result.product_mass, 44.0095e-3)
        self.assertAlmostEqual(result.imbalance, 27.9665e-3)

    def test_mass_check_requires_every_consumed_species_weight(self) -> None:
        registry = PropertyRegistry(
            {
                "A": SpeciesProperties("A", "gas", {"X": 1}),
                "B": SpeciesProperties("B", "gas", {"X": 1}),
            }
        )
        reaction = Reaction(
            name="conversion",
            family="example",
            reactants={"A": 1},
            products={"B": 1},
            rate=1,
        )

        self.assertTrue(check_atom_conservation(reaction, registry).passed)
        with self.assertRaisesRegex(
            ValueError,
            "molecular weight is missing for species: A, B",
        ):
            check_mass_conservation(reaction, registry)

    def test_callers_can_tighten_mass_tolerance(self) -> None:
        registry = PropertyRegistry(
            {
                "A": SpeciesProperties("A", "gas", {"X": 1}, 1.0),
                "B": SpeciesProperties("B", "gas", {"X": 1}, 1.000005),
            }
        )
        reaction = Reaction(
            name="conversion",
            family="example",
            reactants={"A": 1},
            products={"B": 1},
            rate=1,
        )

        self.assertTrue(check_mass_conservation(reaction, registry).passed)
        self.assertFalse(
            check_mass_conservation(reaction, registry, rel_tol=0, abs_tol=0).passed
        )

    def test_tolerances_must_be_finite_and_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "rel_tol"):
            check_atom_conservation(self.combustion, rel_tol=-1)
        with self.assertRaisesRegex(ValueError, "abs_tol"):
            check_mass_conservation(self.combustion, abs_tol=float("inf"))
