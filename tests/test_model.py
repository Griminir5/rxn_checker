from __future__ import annotations

import unittest

from sympy import Symbol

from rxn_checker import Case, Reaction, ReactionContext, parameter, state


def quadratic_loss(context: ReactionContext):
    return context.parameter("k") * context.state("A") ** 2


class HookableReactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = state("A")
        self.b = state("B")
        self.k = parameter("k", positive=True)
        self.reaction = Reaction(
            id="quadratic_loss",
            name="Quadratic loss of A",
            stoichiometry={"A": -1, "B": 1},
            parameter_dependencies=("k",),
            implementation=quadratic_loss,
            source_reference="example",
        )

    def test_case_invokes_hook_with_registered_objects(self) -> None:
        case = Case(
            name="symbolic",
            states={"A": self.a, "B": self.b},
            parameters={"k": self.k},
            reactions=(self.reaction,),
        )

        evaluated = case.evaluate_reaction("quadratic_loss")

        self.assertEqual(evaluated.expression, self.k * self.a**2)
        self.assertEqual(evaluated.stoichiometry, {"A": -1, "B": 1})
        self.assertEqual(evaluated.contribution("A"), -self.k * self.a**2)
        self.assertEqual(evaluated.contribution("B"), self.k * self.a**2)
        self.assertEqual(evaluated.contribution("unused"), 0)

    def test_case_may_supply_concrete_parameter_values(self) -> None:
        case = Case(
            states={"A": self.a, "B": self.b},
            parameters={"k": 2.0},
            reactions=(self.reaction,),
        )
        self.assertEqual(case.expressions["quadratic_loss"], 2.0 * self.a**2)

    def test_negative_region_can_be_inspected_after_evaluation(self) -> None:
        case = Case(
            states={"A": self.a, "B": self.b},
            parameters={"k": 2},
            reactions=(self.reaction,),
        )
        evaluated = case.evaluate_reaction("quadratic_loss")
        contribution = evaluated.contribution("A")
        self.assertLess(contribution.subs({self.a: -0.001}), 0)

    def test_non_stoichiometric_state_dependency_is_available(self) -> None:
        catalyst = state("catalyst")

        def catalysed(context: ReactionContext):
            return context.parameter("k") * context.state("A") * context.state("catalyst")

        reaction = Reaction(
            id="catalysed",
            stoichiometry={"A": -1},
            state_dependencies=("catalyst",),
            parameter_dependencies=("k",),
            implementation=catalysed,
        )
        case = Case(
            states={"A": self.a, "catalyst": catalyst},
            parameters={"k": 2},
            reactions=(reaction,),
        )
        self.assertEqual(case.expressions["catalysed"], 2 * self.a * catalyst)

    def test_callable_object_is_a_valid_complete_implementation(self) -> None:
        class LinearLoss:
            def __call__(self, context: ReactionContext):
                return context.parameter("k") * context.state("A")

        reaction = Reaction(
            id="linear_loss",
            stoichiometry={"A": -1},
            parameter_dependencies=("k",),
            implementation=LinearLoss(),
        )
        case = Case(
            states={"A": self.a},
            parameters={"k": self.k},
            reactions=(reaction,),
        )
        self.assertEqual(case.expressions["linear_loss"], self.k * self.a)

    def test_only_declared_dependencies_are_visible_to_hook(self) -> None:
        def undeclared_access(context: ReactionContext):
            return context.state("B")

        reaction = Reaction(
            id="bad",
            stoichiometry={"A": -1},
            implementation=undeclared_access,
        )
        case = Case(
            states={"A": self.a, "B": self.b},
            parameters={},
            reactions=(reaction,),
        )
        with self.assertRaisesRegex(KeyError, "not available to this reaction"):
            case.evaluate_reactions()

    def test_closed_over_undeclared_symbols_are_rejected(self) -> None:
        hidden = Symbol("hidden", real=True)

        def closes_over_hidden(context: ReactionContext):
            return context.state("A") * hidden

        reaction = Reaction(
            id="bad",
            stoichiometry={"A": -1},
            implementation=closes_over_hidden,
        )
        case = Case(
            states={"A": self.a},
            parameters={},
            reactions=(reaction,),
        )
        with self.assertRaisesRegex(ValueError, "outside its declared dependencies: hidden"):
            case.evaluate_reactions()


class ValidationTests(unittest.TestCase):
    def test_case_rejects_missing_hook_dependencies(self) -> None:
        a = state("A")
        reaction = Reaction(
            id="loss",
            stoichiometry={"A": -1},
            parameter_dependencies=("k",),
            implementation=quadratic_loss,
        )
        with self.assertRaisesRegex(ValueError, "missing parameters k"):
            Case(states={"A": a}, parameters={}, reactions=(reaction,))

    def test_case_rejects_missing_stoichiometric_states(self) -> None:
        a = state("A")
        reaction = Reaction(
            id="conversion",
            stoichiometry={"A": -1, "B": 1},
            implementation=lambda context: context.state("A"),
        )
        with self.assertRaisesRegex(ValueError, "missing states B"):
            Case(states={"A": a}, parameters={}, reactions=(reaction,))

    def test_case_reaction_ids_are_unique(self) -> None:
        a = state("A")
        first = Reaction("loss", {"A": -1}, lambda context: context.state("A"))
        second = Reaction("loss", {"A": -2}, lambda context: context.state("A"))
        with self.assertRaisesRegex(ValueError, "reaction ids must be unique"):
            Case(states={"A": a}, parameters={}, reactions=(first, second))

    def test_symbolic_stoichiometry_is_rejected(self) -> None:
        coefficient = parameter("coefficient")
        with self.assertRaisesRegex(ValueError, "finite real number"):
            Reaction("bad", {"A": coefficient}, lambda context: 1)

    def test_zero_stoichiometry_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-zero"):
            Reaction("bad", {"A": 0}, lambda context: 1)

    def test_one_string_is_not_mistaken_for_many_dependencies(self) -> None:
        with self.assertRaisesRegex(TypeError, "sequence of strings"):
            Reaction(
                "bad",
                {"A": -1},
                lambda context: 1,
                parameter_dependencies="rate_constant",
            )

    def test_state_symbols_do_not_assume_physical_nonnegativity(self) -> None:
        concentration = state("concentration")
        self.assertTrue(concentration.is_real)
        self.assertIsNone(concentration.is_nonnegative)


if __name__ == "__main__":
    unittest.main()
