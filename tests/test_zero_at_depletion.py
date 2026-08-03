import unittest

import sympy as sp

from rxn_checker import Case, Reaction, StateVariables
from rxn_checker.checks import CheckStatus, check_zero_at_depletion
from rxn_checker.checks.models import CheckContext
from rxn_checker.checks.zero_at_depletion import run


class ZeroAtDepletionCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = StateVariables(("Aye", "Bee", "Cat", "Product"))
        self.aye = self.states.concentration("Aye")
        self.bee = self.states.concentration("Bee")
        self.catalyst = self.states.concentration("Cat")
        self.product = self.states.concentration("Product")

    def reaction(
        self,
        rate,
        *,
        reactants=("Aye",),
        catalysts=(),
    ) -> Reaction:
        return Reaction(
            name="conversion",
            family="example",
            reactants={species_id: 1 for species_id in reactants},
            products={"Product": 1},
            catalysts=catalysts,
            rate=rate,
        )

    def test_rate_must_be_zero_at_each_reactant_boundary(self) -> None:
        reaction = self.reaction(
            2 * self.aye * self.bee,
            reactants=("Aye", "Bee"),
        )

        result = check_zero_at_depletion(reaction)

        self.assertTrue(result.passed)
        self.assertEqual(dict(result.rates_at_depletion), {"Aye": 0, "Bee": 0})

    def test_missing_reactant_factor_fails(self) -> None:
        reaction = self.reaction(self.aye, reactants=("Aye", "Bee"))

        result = check_zero_at_depletion(reaction)

        self.assertFalse(result.passed)
        self.assertEqual(result.rates_at_depletion["Bee"], self.aye)
        case = Case("missing-factor", self.states, (reaction,))
        outcome = run(case, CheckContext())[0]
        self.assertEqual(outcome.status, CheckStatus.FAIL)
        self.assertIn("Bee=0", outcome.details[0])

    def test_catalyst_is_a_required_depletion_boundary(self) -> None:
        passing = self.reaction(self.aye * self.catalyst, catalysts=("Cat",))
        failing = self.reaction(self.aye, catalysts=("Cat",))

        self.assertTrue(check_zero_at_depletion(passing).passed)
        self.assertFalse(check_zero_at_depletion(failing).passed)

    def test_product_only_species_is_not_a_depletion_boundary(self) -> None:
        reaction = self.reaction(self.aye * (1 + self.product))

        result = check_zero_at_depletion(reaction)

        self.assertTrue(result.passed)
        self.assertEqual(tuple(result.rates_at_depletion), ("Aye",))

    def test_source_reaction_passes_vacuously(self) -> None:
        reaction = self.reaction(1, reactants=())

        result = check_zero_at_depletion(reaction)

        self.assertTrue(result.passed)
        self.assertEqual(dict(result.rates_at_depletion), {})

    def test_singular_rate_at_depletion_fails(self) -> None:
        reaction = self.reaction(1 / self.aye)

        result = check_zero_at_depletion(reaction)

        self.assertFalse(result.passed)
        self.assertEqual(result.rates_at_depletion["Aye"], sp.zoo)


if __name__ == "__main__":
    unittest.main()
