import unittest

import sympy as sp

from rxn_checker import (
    Case,
    Reaction,
    StateVariables,
    build_check_report,
    find_conserved_quantities,
)
from rxn_checker.checks.stoichiometric_conservation import CHECK
from tests import make_state_bounds


class StoichiometricConservationTests(unittest.TestCase):
    def test_split_reaction_has_two_nonnegative_extreme_rays(self) -> None:
        states = StateVariables(("A", "B", "C"))
        reaction = Reaction(
            name="split",
            family="example",
            reactants={"A": 1},
            products={"B": 1, "C": 1},
            rate=states.concentration("A"),
        )
        case = Case("split", states, (reaction,), make_state_bounds(states))

        result = find_conserved_quantities(case)

        self.assertEqual(result.rank, 1)
        self.assertEqual(result.dimension, 2)
        self.assertEqual(len(result.components), 1)
        component = result.components[0]
        self.assertEqual(
            tuple(dict(quantity.coefficients) for quantity in component.extreme_rays),
            ({"A": 1, "B": 1}, {"A": 1, "C": 1}),
        )
        self.assertEqual(component.signed_basis, ())

        report = build_check_report(case, checks=(CHECK,))
        self.assertIsNone(report.overall_status)
        self.assertIn("Non-negative extreme rays:", report.text)
        self.assertIn("Q1 = [A] + [B]", report.text)
        self.assertIn("Q2 = [A] + [C]", report.text)

    def test_signed_relation_is_reported_when_nonnegative_cone_is_empty(
        self,
    ) -> None:
        states = StateVariables(("A", "B"))
        reaction = Reaction(
            name="source",
            family="example",
            reactants={},
            products={"A": 1, "B": 1},
            rate=sp.S.One,
        )
        case = Case("source", states, (reaction,), make_state_bounds(states))

        result = find_conserved_quantities(case)

        component = result.components[0]
        self.assertEqual(component.extreme_rays, ())
        self.assertEqual(
            tuple(dict(quantity.coefficients) for quantity in component.signed_basis),
            ({"A": 1, "B": -1},),
        )
        report = build_check_report(case, checks=(CHECK,))
        self.assertIn("Signed basis relations:", report.text)
        self.assertIn("Q1 = [A] - [B]", report.text)

    def test_signed_relations_complete_a_partial_extreme_ray_span(self) -> None:
        states = StateVariables(("A", "B", "C", "D"))
        reactions = (
            Reaction(
                name="first",
                family="example",
                reactants={"B": 1},
                products={"A": 1, "C": 1, "D": 1},
                rate=sp.S.One,
            ),
            Reaction(
                name="second",
                family="example",
                reactants={"B": 1, "C": 1, "D": 1},
                products={"A": 1},
                rate=sp.S.One,
            ),
        )
        case = Case("partial-cone", states, reactions, make_state_bounds(states))

        result = find_conserved_quantities(case)

        component = result.components[0]
        self.assertEqual(
            tuple(dict(quantity.coefficients) for quantity in component.extreme_rays),
            ({"A": 1, "B": 1},),
        )
        self.assertEqual(
            tuple(dict(quantity.coefficients) for quantity in component.signed_basis),
            ({"C": 1, "D": -1},),
        )
        report = build_check_report(case, checks=(CHECK,))
        self.assertIn("Additional signed basis relations:", report.text)
        self.assertIn("Q2 = [C] - [D]", report.text)

    def test_decimal_coefficients_are_interpreted_as_exact_rationals(self) -> None:
        states = StateVariables(("A", "B"))
        reaction = Reaction(
            name="fractional",
            family="example",
            reactants={"A": 0.1},
            products={"B": 0.2},
            rate=states.concentration("A"),
        )
        case = Case("fractional", states, (reaction,), make_state_bounds(states))

        result = find_conserved_quantities(case)

        self.assertEqual(
            result.stoichiometric_matrix,
            sp.ImmutableMatrix([[-sp.Rational(1, 10)], [sp.Rational(1, 5)]]),
        )
        self.assertEqual(
            dict(result.components[0].extreme_rays[0].coefficients),
            {"A": 2, "B": 1},
        )

    def test_disconnected_components_and_unchanged_species_are_separate(self) -> None:
        states = StateVariables(("A", "B", "C", "D", "Ar"))
        reactions = (
            Reaction(
                name="ab",
                family="example",
                reactants={"A": 1},
                products={"B": 1},
                rate=states.concentration("A"),
            ),
            Reaction(
                name="cd",
                family="example",
                reactants={"C": 1},
                products={"D": 1},
                rate=states.concentration("C"),
            ),
        )
        case = Case("disconnected", states, reactions, make_state_bounds(states))

        result = find_conserved_quantities(case)

        self.assertEqual(result.rank, 2)
        self.assertEqual(result.dimension, 3)
        self.assertEqual(result.unchanged_species, ("Ar",))
        self.assertEqual(
            tuple(component.species_ids for component in result.components),
            (("A", "B"), ("C", "D")),
        )
        report = build_check_report(case, checks=(CHECK,))
        self.assertIn("Individually unchanged species: Ar.", report.text)
        self.assertIn("Component 1 (A, B):", report.text)
        self.assertIn("Component 2 (C, D):", report.text)

    def test_zero_net_reaction_leaves_each_species_individually_unchanged(
        self,
    ) -> None:
        states = StateVariables(("A", "B"))
        reaction = Reaction(
            name="identity",
            family="example",
            reactants={"A": 1},
            products={"A": 1},
            rate=states.concentration("A"),
        )
        case = Case("identity", states, (reaction,), make_state_bounds(states))

        result = find_conserved_quantities(case)

        self.assertEqual(result.rank, 0)
        self.assertEqual(result.dimension, 2)
        self.assertEqual(result.unchanged_species, ("A", "B"))
        self.assertEqual(result.components, ())

    def test_full_row_rank_network_has_no_conserved_quantity(self) -> None:
        states = StateVariables(("A",))
        reaction = Reaction(
            name="source",
            family="example",
            reactants={},
            products={"A": 1},
            rate=sp.S.One,
        )
        case = Case("full-rank", states, (reaction,), make_state_bounds(states))

        result = find_conserved_quantities(case)

        self.assertEqual(result.dimension, 0)
        report = build_check_report(case, checks=(CHECK,))
        self.assertIn(
            "No non-zero linear concentration invariant exists.",
            report.text,
        )

    def test_registered_definition_is_status_free_and_case_scoped(self) -> None:
        self.assertEqual(CHECK.id, "stoichiometric_conservation")
        self.assertEqual(CHECK.scope.value, "case")


if __name__ == "__main__":
    unittest.main()
