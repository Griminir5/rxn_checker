import unittest

from sympy import Symbol, exp

from rxn_checker import Case, Reaction, StateVariables
from rxn_checker.reactions import FAMILY_REGISTRY, REACTION_REGISTRY
from rxn_checker.reactions.aye_plus_bee_to_cee import (
    ACTIVATION_ENERGY as HALF_ORDER_ACTIVATION_ENERGY,
    GAS_CONSTANT as HALF_ORDER_GAS_CONSTANT,
    build_simple as build_half_order_reaction,
)
from rxn_checker.reactions.aye_to_bee import (
    ACTIVATION_ENERGY,
    GAS_CONSTANT,
    build_autocatalytic as build_autocatalytic_reaction,
    build_simple as build_simple_reaction,
)
from tests import make_state_bounds


class ReactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = StateVariables(("Aye", "Bee"))

    def test_qualified_id_is_derived_from_family_and_local_name(self) -> None:
        reaction = build_simple_reaction(self.states)
        self.assertEqual(reaction.name, "simple")
        self.assertEqual(reaction.id, "aye_to_bee.simple")

    def test_family_files_are_registered_automatically(self) -> None:
        self.assertEqual(
            FAMILY_REGISTRY["aye_to_bee"],
            ("aye_to_bee.autocatalytic", "aye_to_bee.simple"),
        )
        self.assertIn("aye_to_bee.simple", REACTION_REGISTRY)

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

    def test_rates_include_case_owned_temperature_dependence(self) -> None:
        aye = self.states.concentration("Aye")
        bee = self.states.concentration("Bee")
        temperature = self.states.temperature
        autocatalytic = build_autocatalytic_reaction(self.states)
        simple = build_simple_reaction(self.states)
        half_order = build_half_order_reaction(self.states)

        activation_factor = exp(
            -ACTIVATION_ENERGY / (GAS_CONSTANT * temperature)
        )
        half_order_activation_factor = exp(
            -HALF_ORDER_ACTIVATION_ENERGY
            / (HALF_ORDER_GAS_CONSTANT * temperature)
        )

        self.assertEqual(autocatalytic.rate, 2.0 * activation_factor * aye * bee)
        self.assertEqual(simple.rate, 2.0 * activation_factor * aye)
        self.assertEqual(
            half_order.rate,
            2.0 * half_order_activation_factor * aye * bee**0.5,
        )
        for reaction in (autocatalytic, simple, half_order):
            self.assertIn(temperature, reaction.rate.free_symbols)
            self.assertLessEqual(reaction.rate.free_symbols, self.states.symbols)

    def test_catalysts_are_species_but_have_no_net_coefficient(self) -> None:
        states = StateVariables(("Aye", "Bee", "catalyst"))
        a = states.concentration("Aye")
        catalyst = states.concentration("catalyst")
        reaction = Reaction(
            name="catalysed",
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
            name="bad",
            family="bad",
            reactants={"Aye": 1},
            products={"Bee": 1},
            rate=unknown,
        )
        with self.assertRaisesRegex(
            ValueError,
            "uses symbols not owned by this case",
        ):
            Case(
                "bad",
                self.states,
                (reaction,),
                make_state_bounds(self.states),
            )

    def test_each_reaction_is_one_way(self) -> None:
        aye = self.states.concentration("Aye")
        bee = self.states.concentration("Bee")
        forward = Reaction(
            name="Aye_to_Bee",
            family="forward",
            reactants={"Aye": 1},
            products={"Bee": 1},
            rate=aye,
        )
        reverse = Reaction(
            name="Bee_to_Aye",
            family="reverse",
            reactants={"Bee": 1},
            products={"Aye": 1},
            rate=bee,
        )
        self.assertEqual(dict(forward.net_stoichiometry), {"Aye": -1, "Bee": 1})
        self.assertEqual(dict(reverse.net_stoichiometry), {"Bee": -1, "Aye": 1})
