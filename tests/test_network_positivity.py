import unittest

from rxn_checker import Case, Reaction, StateVariables
from rxn_checker.checks import CheckScope, CheckStatus, check_network_positivity
from rxn_checker.checks.models import CheckContext
from rxn_checker.checks.network_positivity import CHECK, run
from tests import make_state_bounds


class NetworkPositivityCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = StateVariables(("Aye", "Bee"))
        self.aye = self.states.concentration("Aye")
        self.bee = self.states.concentration("Bee")
        self.bounds = make_state_bounds(self.states)

    def reaction(self, name, reactants, products, rate) -> Reaction:
        return Reaction(
            name=name,
            family="example",
            reactants=reactants,
            products=products,
            rate=rate,
        )

    def case(self, *reactions: Reaction) -> Case:
        return Case("network", self.states, reactions, self.bounds)

    def test_standard_conversion_is_positive(self) -> None:
        case = self.case(
            self.reaction("forward", {"Aye": 1}, {"Bee": 1}, 2 * self.aye)
        )

        result = check_network_positivity(case)

        self.assertTrue(result.passed)
        self.assertEqual(
            dict(result.source_terms),
            {"Aye": -2 * self.aye, "Bee": 2 * self.aye},
        )
        self.assertEqual(
            dict(result.boundary_sources),
            {"Aye": 0, "Bee": 2 * self.aye},
        )
        self.assertEqual(dict(result.conclusions), {"Aye": True, "Bee": True})

    def test_consumption_on_a_depleted_boundary_fails(self) -> None:
        case = self.case(
            self.reaction("forward", {"Aye": 1}, {"Bee": 1}, self.bee)
        )

        result = check_network_positivity(case)

        self.assertFalse(result.passed)
        self.assertEqual(result.boundary_sources["Aye"], -self.bee)
        self.assertFalse(result.conclusions["Aye"])
        outcome = run(case, CheckContext())
        self.assertEqual(outcome.status, CheckStatus.FAIL)
        self.assertIn("Aye=0", outcome.details[0])

    def test_reactions_are_summed_before_boundary_sign_is_checked(self) -> None:
        case = self.case(
            self.reaction("forward", {"Aye": 1}, {"Bee": 1}, self.bee),
            self.reaction("reverse", {"Bee": 1}, {"Aye": 1}, self.bee),
        )

        result = check_network_positivity(case)

        self.assertTrue(result.passed)
        self.assertEqual(dict(result.source_terms), {"Aye": 0, "Bee": 0})

    def test_sign_changing_boundary_source_is_indeterminate(self) -> None:
        case = self.case(
            self.reaction(
                "forward",
                {"Aye": 1},
                {"Bee": 1},
                self.aye * (1 - self.aye),
            )
        )

        result = check_network_positivity(case)

        self.assertIsNone(result.passed)
        self.assertIsNone(result.conclusions["Bee"])
        self.assertEqual(run(case, CheckContext()).status, CheckStatus.INDETERMINATE)

    def test_registered_check_has_case_scope(self) -> None:
        self.assertEqual(CHECK.scope, CheckScope.CASE)


if __name__ == "__main__":
    unittest.main()
