import unittest
from unittest.mock import patch

from rxn_checker import Case, Reaction, StateVariables
from rxn_checker.checks import CheckStatus, check_rate_nonnegativity
from rxn_checker.checks.nonnegative_rate import run
from rxn_checker.checks.models import CheckContext
from rxn_checker.reactions.aye_to_bee import build_simple
from tests import make_state_bounds


class RateNonnegativityCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = StateVariables(("Aye", "Bee"))
        self.aye = self.states.concentration("Aye")
        self.bee = self.states.concentration("Bee")
        self.state_bounds = make_state_bounds(self.states)

    def reaction(self, rate) -> Reaction:
        return Reaction(
            name="conversion",
            family="example",
            reactants={"Aye": 1},
            products={"Bee": 1},
            rate=rate,
        )

    def test_standard_kinetic_rate_is_symbolically_proven(self) -> None:
        result = check_rate_nonnegativity(
            self.reaction(2 * self.aye * self.bee),
            self.state_bounds,
        )

        self.assertTrue(result.passed)

    def test_expensive_rewrites_are_lazy_when_raw_sign_is_known(self) -> None:
        with (
            patch("sympy.factor_terms", side_effect=AssertionError("factor_terms")),
            patch("sympy.factor", side_effect=AssertionError("factor")),
        ):
            result = check_rate_nonnegativity(
                self.reaction(2 * self.aye * self.bee),
                self.state_bounds,
            )

        self.assertTrue(result.passed)

    def test_arrhenius_rate_uses_strictly_positive_temperature_bound(self) -> None:
        result = check_rate_nonnegativity(
            build_simple(self.states),
            self.state_bounds,
        )

        self.assertTrue(result.passed)

    def test_symbolically_negative_rate_fails(self) -> None:
        reaction = self.reaction(-self.aye * self.bee)
        result = check_rate_nonnegativity(reaction, self.state_bounds)

        self.assertFalse(result.passed)

        case = Case(
            "negative",
            self.states,
            (reaction,),
            self.state_bounds,
        )
        outcome = run(case, CheckContext())[0]
        self.assertEqual(outcome.status, CheckStatus.FAIL)

    def test_unproved_rate_is_indeterminate(self) -> None:
        rate = self.aye**4 - self.aye + 1
        result = check_rate_nonnegativity(
            self.reaction(rate),
            self.state_bounds,
        )

        self.assertIsNone(result.passed)

        case = Case(
            "sampled",
            self.states,
            (self.reaction(rate),),
            self.state_bounds,
        )
        self.assertEqual(run(case, CheckContext())[0].status, CheckStatus.INDETERMINATE)

    def test_sign_changing_rate_is_indeterminate_without_interval_analysis(
        self,
    ) -> None:
        result = check_rate_nonnegativity(
            self.reaction(self.aye * (1 - self.aye)),
            self.state_bounds,
        )

        self.assertIsNone(result.passed)


if __name__ == "__main__":
    unittest.main()
