import unittest

from sympy import Expr, Max, Min

from rxn_checker import StateVariables
from rxn_checker.checks import check_mass_conservation
from rxn_checker.checks.zero_at_depletion import check_zero_at_depletion
from rxn_checker.reactions.medrano import (
    POS_EPS,
    REACTIONS,
    _medrano_reaction_rate_expr,
    _medrano_reaction_state_expr,
    _rational_power_expr,
    _total_gas_concentration,
    build_oxidation_o2,
    build_reduction_co,
    build_reduction_h2,
    medrano_terms,
    oxidation_o2_rate,
    reduction_co_rate,
    reduction_h2_rate,
)


class MedranoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = StateVariables(
            ("H2", "H2O", "CO", "CO2", "O2", "N2", "Ni", "NiO")
        )

    def test_gas_mole_fraction_excludes_solids_and_includes_inerts(self) -> None:
        expected_total = sum(
            self.states.concentration(species_id)
            for species_id in ("H2", "H2O", "CO", "CO2", "O2", "N2")
        )

        total = _total_gas_concentration(self.states)
        terms = medrano_terms(self.states, "H2")

        self.assertEqual(total, expected_total)
        self.assertEqual(
            terms.gas_mole_fraction,
            self.states.concentration("H2") / expected_total,
        )
        self.assertNotIn(self.states.concentration("Ni"), total.free_symbols)
        self.assertNotIn(self.states.concentration("NiO"), total.free_symbols)

    def test_solid_fractions_match_the_reference_regularization(self) -> None:
        ni = self.states.concentration("Ni")
        nio = self.states.concentration("NiO")
        ni_available = Max(ni, 0)
        nio_available = Max(nio, 0)
        total = ni_available + nio_available

        terms = medrano_terms(self.states, "H2")

        self.assertEqual(terms.ni_conc_molm3, ni_available)
        self.assertEqual(terms.nio_conc_molm3, nio_available)
        self.assertEqual(terms.total_solid_inventory_molm3, total)
        self.assertEqual(terms.frac_reduced, ni_available / (total + POS_EPS))
        self.assertEqual(terms.frac_oxidised, nio_available / (total + POS_EPS))

    def test_reaction_states_select_the_unreacted_solid(self) -> None:
        reduction_terms = medrano_terms(self.states, "H2")
        oxidation_terms = medrano_terms(self.states, "O2")

        reduction = _medrano_reaction_state_expr("H2", reduction_terms)
        oxidation = _medrano_reaction_state_expr("O2", oxidation_terms)

        self.assertEqual(reduction.conversion, reduction_terms.frac_reduced)
        self.assertEqual(
            reduction.unreacted_fraction,
            reduction_terms.frac_oxidised,
        )
        self.assertEqual(oxidation.conversion, oxidation_terms.frac_oxidised)
        self.assertEqual(
            oxidation.unreacted_fraction,
            oxidation_terms.frac_reduced,
        )

    def test_rational_fractional_powers_are_zero_and_normalized_at_one(self) -> None:
        for power in (1.0 / 3.0, 2.0 / 3.0, 0.60, 0.65, 0.90):
            with self.subTest(power=power):
                self.assertEqual(_rational_power_expr(power, 0), 0)
                self.assertAlmostEqual(float(_rational_power_expr(power, 1)), 1.0)

    def test_reactions_preserve_the_reference_stoichiometry(self) -> None:
        expected = (
            (
                build_reduction_h2(self.states),
                {"H2": 1, "NiO": 1},
                {"Ni": 1, "H2O": 1},
            ),
            (
                build_reduction_co(self.states),
                {"CO": 1, "NiO": 1},
                {"Ni": 1, "CO2": 1},
            ),
            (
                build_oxidation_o2(self.states),
                {"O2": 0.5, "Ni": 1},
                {"NiO": 1},
            ),
        )

        for reaction, reactants, products in expected:
            with self.subTest(reaction=reaction.id):
                self.assertEqual(reaction.reactants, reactants)
                self.assertEqual(reaction.products, products)
                self.assertEqual(reaction.catalysts, ())
                self.assertTrue(check_mass_conservation(reaction).passed)

    def test_rates_match_the_reference_medrano_composition(self) -> None:
        for comp_key, rate_builder in (
            ("H2", reduction_h2_rate),
            ("CO", reduction_co_rate),
            ("O2", oxidation_o2_rate),
        ):
            terms = medrano_terms(self.states, comp_key)
            state = _medrano_reaction_state_expr(comp_key, terms)
            expected = _medrano_reaction_rate_expr(
                comp_key,
                temperature_k=terms.temperature_k,
                total_gas_concentration_molm3=terms.total_gas_conc_molm3,
                gas_mole_fraction=terms.gas_mole_fraction,
                conversion=state.conversion,
                unreacted_fraction=state.unreacted_fraction,
                total_solid_inventory_molm3=(
                    state.total_solid_inventory_molm3
                ),
            )

            with self.subTest(component=comp_key):
                actual = rate_builder(self.states)
                self.assertEqual(actual, expected)
                self.assertIsInstance(actual, Expr)

    def test_rates_bound_conversion_and_unreacted_fraction(self) -> None:
        for comp_key in ("H2", "CO", "O2"):
            terms = medrano_terms(self.states, comp_key)
            state = _medrano_reaction_state_expr(comp_key, terms)
            rate = _medrano_reaction_rate_expr(
                comp_key,
                temperature_k=terms.temperature_k,
                total_gas_concentration_molm3=terms.total_gas_conc_molm3,
                gas_mole_fraction=terms.gas_mole_fraction,
                conversion=state.conversion,
                unreacted_fraction=state.unreacted_fraction,
                total_solid_inventory_molm3=(
                    state.total_solid_inventory_molm3
                ),
            )

            with self.subTest(component=comp_key):
                self.assertTrue(rate.has(Min))
                self.assertTrue(rate.has(Max))

    def test_each_rate_stops_at_reactant_depletion(self) -> None:
        for builder in REACTIONS.values():
            reaction = builder(self.states)
            with self.subTest(reaction=reaction.id):
                self.assertTrue(check_zero_at_depletion(reaction).passed)

    def test_all_reactions_are_registered(self) -> None:
        self.assertEqual(
            tuple(REACTIONS),
            ("reduction_h2", "reduction_co", "oxidation_o2"),
        )


if __name__ == "__main__":
    unittest.main()
