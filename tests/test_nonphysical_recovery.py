import unittest
from unittest.mock import patch

import sympy as sp

from rxn_checker import Case, Reaction, StateVariables, VariableBounds
from rxn_checker.checks import CheckStatus
from rxn_checker.checks.models import CheckContext
from rxn_checker.checks.nonphysical_recovery import (
    CHECK,
    RecoveryVerdict,
    check_nonphysical_recovery,
    run,
)
from tests import make_state_bounds


class NonphysicalRecoveryTests(unittest.TestCase):
    def case(self, species, reactants, products, rate, *, inerts=()):
        states = StateVariables(species)
        reaction = Reaction(
            name="forward",
            family="example",
            reactants=reactants,
            products=products,
            rate=rate(states),
        )
        return Case(
            "recovery",
            states,
            (reaction,),
            make_state_bounds(states),
            inert_species=inerts,
        )

    def test_linear_conversion_restores_each_repairable_singleton(self) -> None:
        case = self.case(
            ("A", "B"),
            {"A": 1},
            {"B": 1},
            lambda states: states.concentration("A"),
        )

        result = check_nonphysical_recovery(case)

        self.assertEqual(
            tuple(
                (region.negative_species, region.verdict)
                for region in result.regions
            ),
            (
                (("A",), RecoveryVerdict.STRONGLY_RESTORING),
                (("B",), RecoveryVerdict.STRONGLY_RESTORING),
                (
                    ("A", "B"),
                    RecoveryVerdict.STOICHIOMETRICALLY_UNREPAIRABLE,
                ),
            ),
        )
        self.assertTrue(all(region.certified for region in result.regions))

    def test_simultaneous_negative_reactants_are_worsening(self) -> None:
        case = self.case(
            ("A", "B", "C"),
            {"A": 1, "B": 1},
            {"C": 1},
            lambda states: (
                states.concentration("A") * states.concentration("B")
            ),
        )

        result = check_nonphysical_recovery(case)
        simultaneous = next(
            region
            for region in result.regions
            if region.negative_species == ("A", "B")
        )

        self.assertEqual(simultaneous.verdict, RecoveryVerdict.WORSENING)
        self.assertEqual(dict(simultaneous.componentwise), {"A": False, "B": False})
        self.assertEqual(dict(simultaneous.lower_faces), {"A": False, "B": False})
        self.assertIsNotNone(simultaneous.counterexample)

    def test_net_recovery_does_not_require_every_component_to_improve(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        bee = states.concentration("B")
        reactions = (
            Reaction(
                "source",
                "example",
                {},
                {"A": 1, "B": 1},
                (-3 * aye + bee) / 2,
            ),
            Reaction(
                "exchange",
                "example",
                {"B": 1},
                {"A": 1},
                (aye + 3 * bee) / 2,
            ),
        )
        case = Case("net", states, reactions, make_state_bounds(states))

        simultaneous = next(
            region
            for region in check_nonphysical_recovery(case).regions
            if region.negative_species == ("A", "B")
        )

        self.assertEqual(simultaneous.verdict, RecoveryVerdict.NET_RESTORING)
        self.assertEqual(dict(simultaneous.componentwise), {"A": False, "B": True})
        self.assertEqual(
            sp.expand(simultaneous.restoration),
            100 * aye**2 + 100 * bee**2,
        )

    def test_non_worsening_is_not_mistaken_for_strict_recovery(self) -> None:
        states = StateVariables(("A", "B"))
        bee = states.concentration("B")
        reactions = (
            Reaction("a_source", "example", {}, {"A": 1}, bee),
            Reaction("b_source", "example", {}, {"B": 1}, 1),
        )
        case = Case("non-worsening", states, reactions, make_state_bounds(states))

        aye_negative = next(
            region
            for region in check_nonphysical_recovery(case).regions
            if region.negative_species == ("A",)
        )

        self.assertEqual(aye_negative.verdict, RecoveryVerdict.NON_WORSENING)
        self.assertFalse(aye_negative.certified)

    def test_clamped_absolute_and_fractional_extensions_are_distinguished(
        self,
    ) -> None:
        builders = (
            (lambda aye: sp.Max(aye, 0), RecoveryVerdict.STUCK),
            (sp.Abs, RecoveryVerdict.WORSENING),
            (sp.sqrt, RecoveryVerdict.UNDEFINED_IN_EXTENSION),
        )
        for extension, expected in builders:
            with self.subTest(verdict=expected):
                case = self.case(
                    ("A", "B"),
                    {"A": 1},
                    {"B": 1},
                    lambda states: extension(states.concentration("A")),
                )
                region = check_nonphysical_recovery(case).regions[0]
                self.assertEqual(region.negative_species, ("A",))
                self.assertEqual(region.verdict, expected)

    def test_negative_inert_is_stoichiometrically_unrepairable(self) -> None:
        case = self.case(
            ("A", "B", "I"),
            {"A": 1},
            {"B": 1},
            lambda states: states.concentration("A"),
            inerts=("I",),
        )

        inert = next(
            region
            for region in check_nonphysical_recovery(case).regions
            if region.negative_species == ("I",)
        )

        self.assertEqual(
            inert.verdict,
            RecoveryVerdict.STOICHIOMETRICALLY_UNREPAIRABLE,
        )
        self.assertTrue(inert.certified)

    def test_excursion_sizes_weight_the_restoration_score(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        bee = states.concentration("B")
        bounds = make_state_bounds(states)
        bounds[aye] = VariableBounds(0, 10, -sp.Rational(1, 2))
        bounds[bee] = VariableBounds(0, 10, -sp.Rational(1, 4))
        reactions = (Reaction("source", "example", {}, {"A": 1, "B": 1}, 1),)
        case = Case("weighted", states, reactions, bounds)

        simultaneous = next(
            region
            for region in check_nonphysical_recovery(case).regions
            if region.negative_species == ("A", "B")
        )

        self.assertEqual(
            sp.expand(simultaneous.restoration),
            -4 * aye - 16 * bee,
        )
        self.assertEqual(simultaneous.verdict, RecoveryVerdict.STRONGLY_RESTORING)

    def test_runner_maps_certificate_and_failure_to_framework_statuses(self) -> None:
        case = self.case(
            ("A", "B"),
            {"A": 1},
            {"B": 1},
            lambda states: sp.Abs(states.concentration("A")),
        )

        outcome = run(case, CheckContext())

        self.assertEqual(CHECK.scope.value, "case")
        self.assertEqual(outcome.status, CheckStatus.FAIL)
        self.assertTrue(
            any("Negative species A: WORSENING." in line for line in outcome.details)
        )
        self.assertEqual(outcome.values[0].value, 1)
        self.assertIn("exact failure certificate", outcome.details[-1])

    def test_complex_network_stops_before_region_analysis(self) -> None:
        case = self.case(
            ("A", "B"),
            {"A": 1},
            {"B": 1},
            lambda states: states.concentration("A"),
        )

        with patch(
            "rxn_checker.checks.nonphysical_recovery.MAX_SYMBOLIC_OPERATIONS",
            0,
        ):
            result = check_nonphysical_recovery(case)
            outcome = run(case, CheckContext())

        self.assertFalse(result.complete)
        self.assertEqual(result.regions, ())
        self.assertIn("symbolic operation limit", result.diagnostic)
        self.assertEqual(outcome.status, CheckStatus.INDETERMINATE)


if __name__ == "__main__":
    unittest.main()
