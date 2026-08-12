import unittest

import sympy as sp

from rxn_checker import Case, Reaction, StateVariables, load_case
from rxn_checker.checks import CheckScope, CheckStatus, check_equilibria
from rxn_checker.checks.equilibria import CHECK, run
from rxn_checker.checks.models import CheckContext
from tests import make_state_bounds


class EquilibriaTests(unittest.TestCase):
    def reaction(self, name, reactants, products, rate, *, catalysts=()):
        return Reaction(
            name=name,
            family="example",
            reactants=reactants,
            products=products,
            catalysts=catalysts,
            rate=rate,
        )

    def test_irreversible_conversion_has_the_complete_zero_branch(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = self.reaction("forward", {"A": 1}, {"B": 1}, 2 * aye)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria(case)

        self.assertTrue(result.algebraic)
        self.assertEqual(len(result.branches), 1)
        self.assertEqual(result.branches[0].balances, (aye,))
        self.assertEqual(result.branches[0].helpers, ())

    def test_reversible_conversion_is_kept_as_a_readable_relation(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        bee = states.concentration("B")
        reactions = (
            self.reaction("forward", {"A": 1}, {"B": 1}, 2 * aye),
            self.reaction("reverse", {"B": 1}, {"A": 1}, 3 * bee),
        )
        case = Case("network", states, reactions, make_state_bounds(states))

        result = check_equilibria(case)

        self.assertEqual(len(result.branches), 1)
        self.assertEqual(len(result.branches[0].balances), 1)
        self.assertEqual(
            sp.solve(result.branches[0].balances[0], aye),
            [3 * bee / 2],
        )

    def test_product_zero_set_is_returned_as_separate_branches(self) -> None:
        states = StateVariables(("A", "B", "C"))
        aye = states.concentration("A")
        cee = states.concentration("C")
        reaction = self.reaction(
            "catalysed",
            {"A": 1},
            {"B": 1},
            aye * cee,
            catalysts=("C",),
        )
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria(case)

        self.assertEqual(
            {branch.balances[-1] for branch in result.branches},
            {aye, cee},
        )

    def test_radicals_become_ordered_polynomial_helpers(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = self.reaction(
            "half_order",
            {"A": 1},
            {"B": 1},
            sp.sqrt(aye),
        )
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria(case)

        branch = result.branches[0]
        self.assertTrue(result.algebraic)
        self.assertEqual(tuple(helper.kind for helper in branch.helpers), ("root",))
        variables = result.species + tuple(helper.symbol for helper in branch.helpers)
        self.assertTrue(
            all(equation.is_polynomial(*variables) for equation in branch.equations)
        )
        self.assertTrue(branch.conditions)

    def test_exact_binary_float_half_power_is_lifted(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = self.reaction("half_order", {"A": 1}, {"B": 1}, aye**0.5)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria(case)

        self.assertTrue(result.algebraic)
        self.assertEqual(tuple(helper.kind for helper in result.helpers), ("root",))

    def test_helper_conditions_are_kept_on_the_branch_that_uses_them(self) -> None:
        states = StateVariables(("A", "B", "C"))
        aye = states.concentration("A")
        bee = states.concentration("B")
        reaction = self.reaction(
            "half_order",
            {"A": 1},
            {"C": 1},
            aye * sp.sqrt(bee),
        )
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria(case)

        plain_branch = next(
            branch for branch in result.branches if aye in branch.balances
        )
        root_branch = next(branch for branch in result.branches if branch.helpers)
        self.assertEqual(plain_branch.conditions, ())
        self.assertEqual(plain_branch.helpers, ())
        self.assertTrue(root_branch.conditions)

        details = run(case, CheckContext()).details
        helper_line = next(
            i for i, line in enumerate(details) if "u = sqrt" in line
        )
        balance_line = next(
            i for i, line in enumerate(details) if "Balance equations:" in line
        )
        condition_line = next(
            i for i, line in enumerate(details) if "u >= 0" in line
        )
        self.assertLess(helper_line, balance_line)
        self.assertLess(balance_line, condition_line)

    def test_state_transcendental_is_exact_but_marked_mixed(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = self.reaction(
            "transcendental",
            {"A": 1},
            {"B": 1},
            sp.exp(aye) - 1,
        )
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria(case)

        self.assertFalse(result.algebraic)
        self.assertIn("nonalgebraic", result.diagnostic)
        self.assertEqual(
            tuple(helper.kind for helper in result.helpers),
            ("state function",),
        )

    def test_ordered_and_absolute_values_use_exact_helpers(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        bee = states.concentration("B")

        for kind, rate in (
            ("maximum", sp.Max(aye - bee, 0)),
            ("absolute value", sp.Abs(aye - bee)),
        ):
            with self.subTest(kind=kind):
                reaction = self.reaction("special", {"A": 1}, {"B": 1}, rate)
                case = Case(
                    "network",
                    states,
                    (reaction,),
                    make_state_bounds(states),
                )

                result = check_equilibria(case)

                self.assertTrue(result.algebraic)
                self.assertEqual(
                    tuple(helper.kind for helper in result.helpers),
                    (kind,),
                )

    def test_repeated_ordered_values_share_one_helper_definition(self) -> None:
        states = StateVariables(("A", "B", "C"))
        aye = states.concentration("A")
        bee = states.concentration("B")
        maximum = sp.Max(aye - bee, 0)
        minimum = sp.Min(aye + bee, 1)
        rate = maximum / (minimum + 1) + (maximum + 1) / (minimum + 2)
        reaction = self.reaction("reused", {"A": 1}, {"C": 1}, rate)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria(case)

        self.assertEqual(len(result.helpers), 2)
        self.assertEqual(
            {helper.kind: helper.expression for helper in result.helpers},
            {"maximum": maximum, "minimum": minimum},
        )

    def test_physical_clamp_is_removed_without_a_helper(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = self.reaction(
            "clamped",
            {"A": 1},
            {"B": 1},
            sp.Max(aye, 0),
        )
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria(case)

        self.assertEqual(result.helpers, ())
        self.assertEqual(result.branches[0].balances, (aye,))

    def test_constant_and_zero_sources_return_empty_and_full_sets(self) -> None:
        states = StateVariables(("A",))
        source = self.reaction("source", {}, {"A": 1}, 1)
        sink = self.reaction("sink", {"A": 1}, {}, 1)

        empty = check_equilibria(
            Case("constant", states, (source,), make_state_bounds(states))
        )
        full = check_equilibria(
            Case("zero", states, (source, sink), make_state_bounds(states))
        )

        self.assertEqual(empty.branches, ())
        self.assertEqual(len(full.branches), 1)
        self.assertEqual(full.branches[0].equations, ())

    def test_denominator_is_retained_as_a_nonzero_condition(self) -> None:
        states = StateVariables(("A", "B", "C"))
        aye = states.concentration("A")
        bee = states.concentration("B")
        reaction = self.reaction("rational", {"A": 1}, {"C": 1}, aye / bee)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        result = check_equilibria(case)

        self.assertEqual(result.branches[0].balances, (aye,))
        self.assertEqual(result.branches[0].nonzero, (bee,))

    def test_positive_inert_proves_a_total_concentration_nonzero(self) -> None:
        states = StateVariables(("A", "B", "I"))
        aye = states.concentration("A")
        inert = states.concentration("I")
        reaction = self.reaction(
            "diluted",
            {"A": 1},
            {"B": 1},
            aye / (aye + inert),
        )
        case = Case(
            "network",
            states,
            (reaction,),
            make_state_bounds(states),
            inert_species=("I",),
        )

        result = check_equilibria(case)

        self.assertEqual(result.branches[0].nonzero, ())
        self.assertIn(sp.Gt(inert, 0), result.physical_domain)

    def test_xu_froment_has_two_complete_compact_branches(self) -> None:
        result = check_equilibria(load_case("xu_froment_case/case.yaml"))

        self.assertTrue(result.algebraic)
        self.assertEqual(result.source_species, ("CO", "CO2"))
        self.assertEqual(len(result.branches), 2)
        self.assertEqual(result.branches[0].balances, (result.species[-1],))
        interior = result.branches[1]
        self.assertEqual(interior.nonzero[-1], result.species[-1])
        self.assertEqual(sum(helper.kind == "root" for helper in interior.helpers), 2)
        self.assertEqual(
            sum(helper.kind == "parameter coefficient" for helper in interior.helpers),
            6,
        )
        self.assertLess(result.relation_operations, result.source_operations)

    def test_medrano_reuses_its_distinct_minimum_and_maximum_helpers(self) -> None:
        result = check_equilibria(load_case("medrano_case/case.yaml"))

        for branch in result.branches:
            with self.subTest(branch=branch.label):
                self.assertEqual(
                    sum(helper.kind == "minimum" for helper in branch.helpers),
                    2,
                )
                self.assertEqual(
                    sum(helper.kind == "maximum" for helper in branch.helpers),
                    6,
                )
        self.assertLess(result.relation_operations, result.source_operations)

    def test_registered_report_is_short_and_sequential(self) -> None:
        states = StateVariables(("A", "B"))
        aye = states.concentration("A")
        reaction = self.reaction("forward", {"A": 1}, {"B": 1}, aye)
        case = Case("network", states, (reaction,), make_state_bounds(states))

        outcome = run(case, CheckContext())

        self.assertEqual(CHECK.scope, CheckScope.CASE)
        self.assertEqual(outcome.status, CheckStatus.PASS)
        self.assertIn("Branch 1: A = 0.", outcome.details)
        self.assertIn(
            "The complete physical steady-state relationship has 1 branch.",
            outcome.details,
        )
        self.assertEqual(outcome.values, ())

    def test_xu_froment_report_contains_the_readable_equations(self) -> None:
        outcome = run(
            load_case("xu_froment_case/case.yaml"),
            CheckContext(),
        )
        text = "\n".join(outcome.details)

        self.assertIn("Branch 1: Ni = 0.", text)
        self.assertIn("Branch 2: Ni > 0.", text)
        self.assertIn("s1 = CH4 + CO + CO2 + H2 + H2O + N2", text)
        self.assertIn("u = sqrt", text)
        self.assertIn("k6 = exp(", text)
        self.assertIn("Balance equations:", text)
        self.assertIn("4.77482391289570e+53", text)
        self.assertNotIn("Source operations", text)
        self.assertNotIn("programmatic result", text)
        self.assertLessEqual(max(map(len, outcome.details)), 100)


if __name__ == "__main__":
    unittest.main()
