import unittest
from unittest.mock import patch

import sympy as sp

from rxn_checker import (
    Case,
    IdealGasClosure,
    Reaction,
    StateVariables,
    VariableBounds,
)
from rxn_checker.checks import CheckStatus
from rxn_checker.checks.models import CheckContext
from rxn_checker.checks.negative_side_recovery import (
    CHECK,
    check_negative_side_recovery,
    run,
)
from tests import make_state_bounds


class NegativeSideRecoveryTests(unittest.TestCase):
    def case(self, states, reactions, *, bounds=None, gas_closure=None):
        return Case(
            "negative-side",
            states,
            reactions,
            make_state_bounds(states) if bounds is None else bounds,
            gas_closure=gas_closure,
        )

    def test_proves_nonrepulsion_and_strict_attraction(self) -> None:
        states = StateVariables(("A",))
        aye = states.concentration("A")
        reaction = Reaction("restore", "test", {"A": 1}, {}, aye)

        result = check_negative_side_recovery(self.case(states, (reaction,)))

        self.assertTrue(result.nonrepelling)
        self.assertTrue(result.attracting)
        self.assertEqual(len(result.species), 1)
        self.assertTrue(result.species[0].nonrepelling)
        self.assertTrue(result.species[0].attracting)
        self.assertEqual(result.species[0].source, -aye)

    def test_nonrepulsion_passes_when_strict_attraction_is_disproved(self) -> None:
        states = StateVariables(("A", "B"))
        reaction = Reaction("stopped", "test", {"A": 1}, {"B": 1}, 0)
        case = self.case(states, (reaction,))

        result = check_negative_side_recovery(case)
        outcome = run(case, CheckContext())

        self.assertTrue(result.nonrepelling)
        self.assertFalse(result.attracting)
        self.assertTrue(all(item.nonrepelling for item in result.species))
        self.assertTrue(all(item.attracting is False for item in result.species))
        self.assertEqual(outcome.status, CheckStatus.PASS)
        self.assertTrue(
            any("strict attraction disproved" in line for line in outcome.details)
        )

    def test_finds_exact_nonrepulsion_counterexample(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = Reaction("worsen", "test", {"A": 1}, {"B": 1}, -aye)
        case = self.case(states, (reaction,))

        result = check_negative_side_recovery(case)
        aye_result = result.species[0]
        outcome = run(case, CheckContext())

        self.assertFalse(result.nonrepelling)
        self.assertFalse(aye_result.nonrepelling)
        self.assertIsNotNone(aye_result.nonrepulsion_counterexample)
        self.assertLess(aye_result.nonrepulsion_counterexample[aye], 0)
        self.assertEqual(outcome.status, CheckStatus.FAIL)

    def test_other_species_keep_their_negative_excursions(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = Reaction("convert", "test", {"A": 1}, {"B": 1}, aye)

        result = check_negative_side_recovery(self.case(states, (reaction,)))
        bee_result = result.species[1]

        self.assertEqual(bee_result.species_id, "B")
        self.assertFalse(bee_result.nonrepelling)
        self.assertIsNotNone(bee_result.nonrepulsion_counterexample)
        self.assertLess(bee_result.nonrepulsion_counterexample[aye], 0)

    def test_strict_gas_total_is_not_weakened_to_its_boundary(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        bee = states.concentration("B")
        total = aye + bee
        reaction = Reaction("restore", "test", {}, {"A": 1}, total)
        closure = IdealGasClosure(
            (aye, bee),
            states.temperature,
            states.pressure,
        )

        result = check_negative_side_recovery(
            self.case(states, (reaction,), gas_closure=closure)
        )
        aye_result = result.species[0]

        self.assertTrue(aye_result.nonrepelling)
        self.assertTrue(aye_result.attracting)

    def test_no_declared_negative_excursion_is_unavailable(self) -> None:
        states = StateVariables(("A",))
        aye = states.concentration("A")
        reaction = Reaction("sink", "test", {"A": 1}, {}, aye)
        bounds = make_state_bounds(states)
        bounds[aye] = VariableBounds(0, 10)
        case = self.case(states, (reaction,), bounds=bounds)

        result = check_negative_side_recovery(case)
        outcome = run(case, CheckContext())

        self.assertEqual(result.species, ())
        self.assertEqual(outcome.status, CheckStatus.UNAVAILABLE)

    def test_source_operation_limit_is_indeterminate(self) -> None:
        states = StateVariables(("A",))
        aye = states.concentration("A")
        reaction = Reaction("sink", "test", {"A": 1}, {}, aye)
        case = self.case(states, (reaction,))

        with patch(
            "rxn_checker.checks.negative_side_recovery.MAX_SOURCE_OPERATIONS",
            -1,
        ):
            result = check_negative_side_recovery(case)
            outcome = run(case, CheckContext())

        self.assertIsNone(result.nonrepelling)
        self.assertEqual(outcome.status, CheckStatus.INDETERMINATE)
        self.assertTrue(
            any("operation limit" in line for line in outcome.details)
        )

    def test_check_is_registered_at_case_scope(self) -> None:
        self.assertEqual(CHECK.id, "negative_side_recovery")
        self.assertEqual(CHECK.scope.value, "case")


if __name__ == "__main__":
    unittest.main()
