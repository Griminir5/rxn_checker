import unittest
from pathlib import Path

from sympy import Expr, Pow, Rational, sqrt

from rxn_checker import StateVariables, load_case
from rxn_checker.checks import check_mass_conservation
from rxn_checker.checks.zero_at_depletion import check_zero_at_depletion
from rxn_checker.reactions.xu_froment import (
    H2_MOLE_FRACTION_SMOOTH_EPS_SQUARED,
    MIN_H2_MOLE_FRACTION,
    REACTIONS,
    _eq_const_overall,
    _eq_const_smr,
    _eq_const_wgs,
    _rate_constant_expression,
    _total_gas_concentration,
    build_overall_bw,
    build_overall_fw,
    build_smr_bw,
    build_smr_fw,
    build_wgs_bw,
    build_wgs_fw,
    overall_bw_rate,
    overall_fw_rate,
    smr_bw_rate,
    smr_fw_rate,
    wgs_bw_rate,
    wgs_fw_rate,
    xu_froment_terms,
)


class XuFromentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = StateVariables(
            ("CH4", "H2O", "CO", "CO2", "H2", "N2", "Ni", "NiO")
        )

    def test_total_gas_concentration_excludes_solids_and_includes_inerts(
        self,
    ) -> None:
        expected = sum(
            self.states.concentration(species_id)
            for species_id in ("CH4", "H2O", "CO", "CO2", "H2", "N2")
        )

        total = _total_gas_concentration(self.states)

        self.assertEqual(total, expected)
        self.assertNotIn(self.states.concentration("Ni"), total.free_symbols)
        self.assertNotIn(self.states.concentration("NiO"), total.free_symbols)

    def test_inverse_hydrogen_pressure_uses_reference_regularization(self) -> None:
        total = _total_gas_concentration(self.states)
        h2_mole_fraction = self.states.concentration("H2") / total
        controlled_h2_mole_fraction = MIN_H2_MOLE_FRACTION + Rational(1, 2) * (
            h2_mole_fraction
            + sqrt(
                h2_mole_fraction**2
                + H2_MOLE_FRACTION_SMOOTH_EPS_SQUARED
            )
        )
        expected = 1 / (
            self.states.pressure * controlled_h2_mole_fraction
        )

        self.assertEqual(xu_froment_terms(self.states).p_inv_h2_pa_inv, expected)

    def test_split_reactions_have_reverse_stoichiometry_and_nickel_catalyst(
        self,
    ) -> None:
        reaction_pairs = (
            (
                build_smr_fw(self.states),
                build_smr_bw(self.states),
                {"CH4": 1, "H2O": 1},
                {"CO": 1, "H2": 3},
            ),
            (
                build_wgs_fw(self.states),
                build_wgs_bw(self.states),
                {"CO": 1, "H2O": 1},
                {"CO2": 1, "H2": 1},
            ),
            (
                build_overall_fw(self.states),
                build_overall_bw(self.states),
                {"CH4": 1, "H2O": 2},
                {"CO2": 1, "H2": 4},
            ),
        )

        for forward, backward, reactants, products in reaction_pairs:
            with self.subTest(reaction=forward.name):
                self.assertEqual(forward.reactants, reactants)
                self.assertEqual(forward.products, products)
                self.assertEqual(backward.reactants, forward.products)
                self.assertEqual(backward.products, forward.reactants)
                self.assertEqual(forward.catalysts, ("Ni",))
                self.assertEqual(backward.catalysts, ("Ni",))

    def test_split_reactions_conserve_mass(self) -> None:
        for builder in REACTIONS.values():
            reaction = builder(self.states)
            with self.subTest(reaction=reaction.id):
                self.assertTrue(check_mass_conservation(reaction).passed)

    def test_split_rates_match_the_reference_driving_force_terms(self) -> None:
        terms = xu_froment_terms(self.states)

        smr_common_factor = (
            _rate_constant_expression(
                "smr",
                terms.temperature_k,
                terms.catalyst_mass_density_kg_per_m3,
            )
            * Pow(terms.p_inv_h2_pa_inv, Rational(5, 2))
            / (10.0**-2.5)
            / Pow(terms.denominator, 2)
        )
        expected_smr_forward = (
            smr_common_factor * terms.p_ch4_pa * terms.p_h2o_pa
        )
        expected_smr_backward = (
            smr_common_factor
            * Pow(terms.p_h2_pa, 3)
            * terms.p_co_pa
            / (1e10 * _eq_const_smr(terms.temperature_k))
        )

        wgs_common_factor = (
            _rate_constant_expression(
                "wgs",
                terms.temperature_k,
                terms.catalyst_mass_density_kg_per_m3,
            )
            * terms.p_inv_h2_pa_inv
            / 1.0e5
            / Pow(terms.denominator, 2)
        )
        expected_wgs_forward = wgs_common_factor * terms.p_co_pa * terms.p_h2o_pa
        expected_wgs_backward = (
            wgs_common_factor
            * terms.p_h2_pa
            * terms.p_co2_pa
            / _eq_const_wgs(terms.temperature_k)
        )

        overall_common_factor = (
            _rate_constant_expression(
                "overall",
                terms.temperature_k,
                terms.catalyst_mass_density_kg_per_m3,
            )
            * Pow(terms.p_inv_h2_pa_inv, Rational(7, 2))
            / (10.0**-2.5)
            / Pow(terms.denominator, 2)
        )
        expected_overall_forward = (
            overall_common_factor * terms.p_ch4_pa * Pow(terms.p_h2o_pa, 2)
        )
        expected_overall_backward = (
            overall_common_factor
            * Pow(terms.p_h2_pa, 4)
            * terms.p_co2_pa
            / (1e10 * _eq_const_overall(terms.temperature_k))
        )

        expected_rates = (
            (smr_fw_rate(self.states), expected_smr_forward),
            (smr_bw_rate(self.states), expected_smr_backward),
            (wgs_fw_rate(self.states), expected_wgs_forward),
            (wgs_bw_rate(self.states), expected_wgs_backward),
            (overall_fw_rate(self.states), expected_overall_forward),
            (overall_bw_rate(self.states), expected_overall_backward),
        )
        for actual, expected in expected_rates:
            with self.subTest(rate=actual):
                self.assertEqual(actual, expected)
                self.assertIsInstance(actual, Expr)

    def test_each_split_rate_stops_at_reactant_and_catalyst_depletion(self) -> None:
        for reaction in (
            build_smr_fw(self.states),
            build_smr_bw(self.states),
            build_wgs_fw(self.states),
            build_wgs_bw(self.states),
            build_overall_fw(self.states),
            build_overall_bw(self.states),
        ):
            with self.subTest(reaction=reaction.id):
                self.assertTrue(check_zero_at_depletion(reaction).passed)

    def test_all_split_reactions_are_registered(self) -> None:
        self.assertEqual(
            tuple(REACTIONS),
            (
                "wgs_fw",
                "smr_fw",
                "overall_fw",
                "wgs_bw",
                "smr_bw",
                "overall_bw",
            ),
        )

    def test_example_case_loads_the_complete_family(self) -> None:
        case_path = Path(__file__).parents[1] / "xu_froment_case" / "case.yaml"

        case = load_case(case_path)

        self.assertEqual(
            tuple(reaction.id for reaction in case.reactions),
            tuple(f"xu_froment.{reaction_name}" for reaction_name in REACTIONS),
        )


if __name__ == "__main__":
    unittest.main()
