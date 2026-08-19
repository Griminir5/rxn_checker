import unittest

import sympy as sp

from rxn_checker import IdealGasClosure, Reaction, StateVariables, VariableBounds
from rxn_checker.checks.lipschitz_continuity import (
    check_lipschitz_continuity,
)


class LipschitzContinuityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = StateVariables(("A", "B"))
        self.a = self.states.concentration("A")
        self.b = self.states.concentration("B")
        self.bounds = {
            self.a: VariableBounds(0.0, 2.0, -0.1),
            self.b: VariableBounds(0.0, 2.0, -0.1),
            self.states.temperature: VariableBounds(200.0, 1500.0),
            self.states.pressure: VariableBounds(10_000.0, 10_000_000.0),
        }
        self.gas_closure = IdealGasClosure(
            (self.a, self.b),
            self.states.temperature,
            self.states.pressure,
        )
        self.derived_gas_closure = IdealGasClosure(
            (self.a, self.b),
            self.states.temperature,
            self.states.pressure,
            minimum_total="ideal_gas",
        )

    def reaction(self, rate: sp.Expr) -> Reaction:
        return Reaction("rate", "test", {"A": 1}, {}, rate)

    def check(self, rate: sp.Expr, bounds=None, gas_closure=None):
        return check_lipschitz_continuity(
            self.reaction(rate),
            self.bounds if bounds is None else bounds,
            gas_closure,
        )

    def test_certifies_supported_expression_with_strict_safety_margins(
        self,
    ) -> None:
        rate = (
            sp.exp(-1 / self.states.temperature)
            * sp.log(self.a + 2)
            / (self.a**2 + 1)
        )

        result = self.check(rate)

        self.assertTrue(result.passed)
        self.assertTrue(result.defined)
        self.assertFalse(result.unresolved_conditions)

    def test_abs_min_and_max_are_valid_at_their_kinks(self) -> None:
        rate = sp.Abs(self.a) + sp.Min(self.a, 1) + sp.Max(self.a, 0)

        self.assertTrue(self.check(rate).passed)

    def test_undefined_point_in_augmented_domain_is_a_failure(self) -> None:
        result = self.check(1 / self.a)

        self.assertFalse(result.passed)
        self.assertFalse(result.defined)
        self.assertIsNotNone(result.counterexample)
        self.assertEqual(result.counterexample[self.a], 0)

    def test_gas_closure_retains_zero_total_as_a_uniform_margin_boundary(
        self,
    ) -> None:
        result = self.check(1 / (self.a + self.b), gas_closure=self.gas_closure)

        self.assertFalse(result.passed)
        self.assertFalse(result.defined)
        self.assertIsNotNone(result.counterexample)
        self.assertEqual(
            result.counterexample[self.a] + result.counterexample[self.b],
            0,
        )

    def test_ideal_gas_minimum_gives_inverse_total_a_uniform_margin(self) -> None:
        result = self.check(
            1 / (self.a + self.b),
            gas_closure=self.derived_gas_closure,
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.defined)

    def test_fractional_power_loses_lipschitz_margin_at_chamfer_boundary(
        self,
    ) -> None:
        result = self.check(
            sp.sqrt(self.a + self.b),
            gas_closure=self.gas_closure,
        )

        self.assertFalse(result.passed)
        self.assertIsNotNone(result.counterexample)
        self.assertEqual(
            result.counterexample[self.a] + result.counterexample[self.b],
            0,
        )

    def test_gas_closure_does_not_require_each_component_to_be_positive(
        self,
    ) -> None:
        result = self.check(1 / self.a, gas_closure=self.gas_closure)

        self.assertFalse(result.passed)
        self.assertFalse(result.defined)
        self.assertIsNotNone(result.counterexample)
        self.assertEqual(result.counterexample[self.a], 0)
        self.assertGreaterEqual(result.counterexample[self.b], 0)

    def test_gas_closure_retains_individual_negative_excursions(self) -> None:
        result = self.check(sp.sqrt(self.a), gas_closure=self.gas_closure)

        self.assertFalse(result.passed)
        self.assertIsNotNone(result.counterexample)
        self.assertLess(result.counterexample[self.a], 0)
        self.assertGreaterEqual(
            result.counterexample[self.a] + result.counterexample[self.b],
            0,
        )

    def test_fractional_power_over_negative_excursion_is_a_failure(self) -> None:
        result = self.check(sp.sqrt(self.a))

        self.assertFalse(result.passed)
        self.assertFalse(result.defined)

    def test_fractional_power_touching_boundary_has_no_uniform_margin(self) -> None:
        bounds = dict(self.bounds)
        bounds[self.a] = VariableBounds(0.0, 2.0)

        result = self.check(sp.sqrt(self.a), bounds)

        self.assertFalse(result.passed)
        self.assertTrue(result.defined)
        self.assertIsNotNone(result.counterexample)
        self.assertEqual(result.counterexample[self.a], 0)

    def test_unsupported_function_is_indeterminate(self) -> None:
        result = self.check(sp.floor(self.a))

        self.assertIsNone(result.passed)
        self.assertEqual(result.unsupported_functions, ("floor",))

    def test_nonreal_rate_is_a_failure(self) -> None:
        result = self.check(sp.I * self.a)

        self.assertFalse(result.passed)
        self.assertFalse(result.defined)

    def test_rates_are_checked_before_source_term_cancellation(self) -> None:
        rates = (1 / self.a, -1 / self.a)

        results = tuple(self.check(rate) for rate in rates)

        self.assertEqual(sp.Add(*rates), 0)
        self.assertTrue(all(result.passed is False for result in results))


if __name__ == "__main__":
    unittest.main()
