import unittest

from rxn_checker import Case, Reaction, StateVariables
from rxn_checker.checks import (
    CheckScope,
    CheckStatus,
    check_equilibria_and_terminal_faces,
)
from rxn_checker.checks.equilibria_and_terminal_faces import CHECK, run
from rxn_checker.checks.models import CheckContext
from tests import make_state_bounds


class EquilibriaAndTerminalFacesTests(unittest.TestCase):
    def reaction(self, name, reactants, products, rate):
        return Reaction(
            name=name,
            family="example",
            reactants=reactants,
            products=products,
            rate=rate,
        )

    def test_irreversible_conversion_has_one_equilibrium_face(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = self.reaction("forward", {"A": 1}, {"B": 1}, 2 * aye)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria_and_terminal_faces(case)

        self.assertTrue(result.equilibrium_search_complete)
        self.assertEqual(len(result.equilibria), 1)
        self.assertEqual(
            dict(result.equilibria[0].coordinates),
            {"A": 0, "B": states.concentration("B")},
        )
        self.assertTrue(result.equilibria[0].physical)
        self.assertEqual(result.terminal_faces, (("A",),))
        self.assertEqual(result.invariant_faces, (("A",),))

    def test_reversible_conversion_has_an_interior_family_and_origin_face(
        self,
    ) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        bee = states.concentration("B")
        reactions = (
            self.reaction("forward", {"A": 1}, {"B": 1}, 2 * aye),
            self.reaction("reverse", {"B": 1}, {"A": 1}, 3 * bee),
        )
        case = Case("network", states, reactions, make_state_bounds(states))

        result = check_equilibria_and_terminal_faces(case)

        self.assertEqual(len(result.equilibria), 1)
        self.assertEqual(
            dict(result.equilibria[0].coordinates),
            {"A": 3 * bee / 2, "B": bee},
        )
        self.assertIsNone(result.equilibria[0].physical)
        self.assertEqual(result.terminal_faces, (("A", "B"),))
        self.assertEqual(result.invariant_faces, (("A", "B"),))

    def test_invariant_nonterminal_face_is_reported_separately(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = self.reaction("decay", {"A": 1}, {}, aye)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria_and_terminal_faces(case)

        self.assertEqual(result.terminal_faces, (("A",),))
        self.assertEqual(result.invariant_faces, (("A",), ("B",)))

    def test_float_fractional_power_keeps_all_terminal_equilibria(self) -> None:
        states = StateVariables(("A", "B", "C"))
        aye = states.concentration("A")
        bee = states.concentration("B")
        cee = states.concentration("C")
        reaction = self.reaction(
            "half_order",
            {"A": 1, "B": 1},
            {"C": 1},
            aye * bee**0.5,
        )
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria_and_terminal_faces(case)

        self.assertEqual(result.terminal_faces, (("A",), ("B",)))
        self.assertEqual(len(result.equilibria), 2)
        self.assertEqual(
            {tuple(family.coordinates.values()) for family in result.equilibria},
            {(0, bee, cee), (aye, 0, cee)},
        )

    def test_definitely_out_of_bounds_equilibrium_is_excluded(self) -> None:
        states = StateVariables(("A",))
        aye = states.concentration("A")
        reaction = self.reaction(
            "autocatalytic",
            {"A": 1},
            {"A": 2},
            aye * (aye - 2000),
        )
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria_and_terminal_faces(case)

        self.assertEqual(len(result.equilibria), 1)
        self.assertEqual(dict(result.equilibria[0].coordinates), {"A": 0})
        self.assertEqual(result.excluded_equilibria, 1)

    def test_constant_source_has_no_equilibrium_or_terminal_face(self) -> None:
        states = StateVariables(("A",))
        reaction = self.reaction("source", {}, {"A": 1}, 1)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria_and_terminal_faces(case)

        self.assertEqual(result.equilibria, ())
        self.assertEqual(result.terminal_faces, ())
        self.assertEqual(result.invariant_faces, ())
        self.assertTrue(result.equilibrium_search_complete)

    def test_identically_zero_network_reports_the_whole_domain(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reactions = (
            self.reaction("forward", {"A": 1}, {"B": 1}, aye),
            self.reaction("reverse", {"B": 1}, {"A": 1}, aye),
        )
        case = Case("network", states, reactions, make_state_bounds(states))

        result = check_equilibria_and_terminal_faces(case)

        self.assertEqual(result.terminal_faces, ((),))
        self.assertEqual(len(result.equilibria), 1)
        self.assertEqual(result.face_tests, 1)

    def test_registered_runner_reports_counts(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = self.reaction("forward", {"A": 1}, {"B": 1}, aye)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        outcome = run(case, CheckContext())

        self.assertEqual(CHECK.scope, CheckScope.CASE)
        self.assertEqual(outcome.status, CheckStatus.PASS)
        self.assertIn("Maximal terminal faces: A=0.", outcome.details)
        self.assertEqual(tuple(value.value for value in outcome.values), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
