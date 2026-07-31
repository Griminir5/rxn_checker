from __future__ import annotations

import unittest

from sympy import Symbol

from case import Case, StateVariables
from reactions import Reaction
from reactions.aye_to_bee import (
    build_autocatalytic as build_autocatalytic_reaction,
)
from reactions.aye_to_bee import build_simple as build_simple_reaction


class ReactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = StateVariables(("Aye", "Bee"))

    def test_module_uses_case_owned_symbols(self) -> None:
        reactions = (
            build_autocatalytic_reaction(self.states),
            build_simple_reaction(self.states),
        )
        for reaction in reactions:
            self.assertLessEqual(reaction.rate.free_symbols, self.states.symbols)

    def test_full_sides_are_preserved_while_net_is_derived(self) -> None:
        autocatalytic = build_autocatalytic_reaction(self.states)
        simple = build_simple_reaction(self.states)

        self.assertEqual(autocatalytic.reactants, {"Aye": 1, "Bee": 1})
        self.assertEqual(autocatalytic.products, {"Bee": 2})
        self.assertEqual(simple.reactants, {"Aye": 1})
        self.assertEqual(simple.products, {"Bee": 1})
        self.assertEqual(
            dict(autocatalytic.net_stoichiometry),
            dict(simple.net_stoichiometry),
        )

    def test_rates_contain_only_state_symbols_and_numeric_constants(self) -> None:
        aye = self.states.concentration("Aye")
        bee = self.states.concentration("Bee")
        autocatalytic = build_autocatalytic_reaction(self.states)
        simple = build_simple_reaction(self.states)

        self.assertEqual(autocatalytic.rate, 2.0 * aye * bee)
        self.assertEqual(simple.rate, 2.0 * aye)

    def test_catalysts_are_species_but_have_no_net_coefficient(self) -> None:
        states = StateVariables(("Aye", "Bee", "catalyst"))
        a = states.concentration("Aye")
        catalyst = states.concentration("catalyst")
        reaction = Reaction(
            id="catalysed",
            family="conversion",
            reactants={"Aye": 1},
            products={"Bee": 1},
            catalysts=("catalyst",),
            rate=a * catalyst,
        )
        self.assertEqual(reaction.species_ids, ("Aye", "Bee", "catalyst"))
        self.assertNotIn("catalyst", reaction.net_stoichiometry)

    def test_undeclared_state_symbols_are_rejected(self) -> None:
        unknown = Symbol("unknown", real=True)
        reaction = Reaction(
            id="bad",
            family="bad",
            reactants={"Aye": 1},
            products={"Bee": 1},
            rate=unknown,
        )
        with self.assertRaisesRegex(
            ValueError,
            "does not use this case's state symbols",
        ):
            Case("bad", (), self.states, (reaction,))

    def test_each_reaction_is_one_way(self) -> None:
        aye = self.states.concentration("Aye")
        bee = self.states.concentration("Bee")
        forward = Reaction(
            id="Aye_to_Bee",
            family="forward",
            reactants={"Aye": 1},
            products={"Bee": 1},
            rate=aye,
        )
        reverse = Reaction(
            id="Bee_to_Aye",
            family="reverse",
            reactants={"Bee": 1},
            products={"Aye": 1},
            rate=bee,
        )
        self.assertEqual(dict(forward.net_stoichiometry), {"Aye": -1, "Bee": 1})
        self.assertEqual(dict(reverse.net_stoichiometry), {"Bee": -1, "Aye": 1})


if __name__ == "__main__":
    unittest.main()
