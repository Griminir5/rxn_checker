import unittest
from unittest.mock import patch

import sympy as sp

from rxn_checker import Case, Reaction, StateVariables
from rxn_checker.checks import CheckScope, CheckStatus, check_terminal_faces
from rxn_checker.checks.models import CheckContext
from rxn_checker.checks.terminal_faces import (
    CHECK,
    _source_terms,
    find_terminal_faces,
    run,
)
from tests import make_state_bounds


class TerminalFacesTests(unittest.TestCase):
    def reaction(self, name, reactants, products, rate):
        return Reaction(
            name=name,
            family="example",
            reactants=reactants,
            products=products,
            rate=rate,
        )

    def test_irreversible_conversion_has_one_terminal_face(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = self.reaction("forward", {"A": 1}, {"B": 1}, 2 * aye)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_terminal_faces(case)

        self.assertEqual(result.terminal_faces, (("A",),))
        self.assertEqual(result.invariant_faces, (("A",),))

    def test_reversible_conversion_has_origin_face(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        bee = states.concentration("B")
        reactions = (
            self.reaction("forward", {"A": 1}, {"B": 1}, 2 * aye),
            self.reaction("reverse", {"B": 1}, {"A": 1}, 3 * bee),
        )
        case = Case("network", states, reactions, make_state_bounds(states))

        result = check_terminal_faces(case)

        self.assertEqual(result.terminal_faces, (("A", "B"),))
        self.assertEqual(result.invariant_faces, (("A", "B"),))

    def test_invariant_nonterminal_face_is_reported_separately(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = self.reaction("decay", {"A": 1}, {}, aye)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_terminal_faces(case)

        self.assertEqual(result.terminal_faces, (("A",),))
        self.assertEqual(result.invariant_faces, (("A",), ("B",)))

    def test_constant_source_has_no_terminal_face(self) -> None:
        states = StateVariables(("A",))
        reaction = self.reaction("source", {}, {"A": 1}, 1)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_terminal_faces(case)

        self.assertEqual(result.terminal_faces, ())
        self.assertEqual(result.invariant_faces, ())

    def test_strictly_positive_inerts_are_not_face_coordinates(self) -> None:
        states = StateVariables(("A", "B", "I"))
        aye = states.concentration("A")
        reaction = self.reaction("forward", {"A": 1}, {"B": 1}, aye)
        case = Case(
            "network",
            states,
            (reaction,),
            make_state_bounds(states),
            inert_species=("I",),
        )

        result = check_terminal_faces(case)

        self.assertEqual(result.terminal_faces, (("A",),))
        self.assertNotIn("I", {item for face in result.invariant_faces for item in face})

    def test_identically_zero_network_reports_the_whole_domain(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reactions = (
            self.reaction("forward", {"A": 1}, {"B": 1}, aye),
            self.reaction("reverse", {"B": 1}, {"A": 1}, aye),
        )
        case = Case("network", states, reactions, make_state_bounds(states))

        result = check_terminal_faces(case)

        self.assertEqual(result.terminal_faces, ((),))
        self.assertEqual(result.tests, 1)

    def test_registered_runner_reports_face_counts(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = self.reaction("forward", {"A": 1}, {"B": 1}, aye)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        outcome = run(case, CheckContext())

        self.assertEqual(CHECK.scope, CheckScope.CASE)
        self.assertEqual(outcome.status, CheckStatus.PASS)
        self.assertIn("Maximal terminal faces: A=0.", outcome.details)
        self.assertEqual(tuple(value.value for value in outcome.values), (1, 1))

    def test_face_search_does_not_simplify_or_call_equals(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = self.reaction("forward", {"A": 1}, {"B": 1}, 2 * aye)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        with (
            patch("sympy.simplify", side_effect=AssertionError("simplify called")),
            patch.object(
                sp.Expr,
                "equals",
                side_effect=AssertionError("equals called"),
            ),
        ):
            result = find_terminal_faces(case, _source_terms(case))

        self.assertEqual(result.terminal_faces, (("A",),))
        self.assertEqual(result.invariant_faces, (("A",),))


if __name__ == "__main__":
    unittest.main()
