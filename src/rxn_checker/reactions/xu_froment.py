from collections.abc import Mapping
from dataclasses import dataclass

from sympy import Add, Expr, Pow, Rational, exp, sqrt
from sympy.functions.elementary.miscellaneous import Max

from ..model import CaseSymbols, Reaction
from ..species.registry import PROPERTY_REGISTRY

GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
MIN_H2_MOLE_FRACTION = 0.001
H2_MOLE_FRACTION_SMOOTH_EPS_SQUARED = 1.0e-4
NI_MW_KG_PER_MOL = PROPERTY_REGISTRY.get_record("Ni").mw

if NI_MW_KG_PER_MOL is None:
    raise ValueError("Nickel molecular weight must be available for Xu-Froment kinetics.")

XU_FROMENT_RATE_COEFFICIENTS = {
    "smr": 1.17e15,
    "wgs": 5.43e5,
    "overall": 2.81e14,
}
XU_FROMENT_ACTIVATION_ENERGIES_J_PER_MOL = {
    "smr": 240100.0,
    "wgs": 67130.0,
    "overall": 243900.0,
}
XU_FROMENT_ADSORPTION_COEFFICIENTS = {
    "CO": 8.23e-10,
    "H2": 6.12e-14,
    "CH4": 6.65e-9,
    "H2O": 1.77e5,
}
XU_FROMENT_ADSORPTION_ENERGIES_J_PER_MOL = {
    "CO": 70650.0,
    "H2": 82900.0,
    "CH4": 38280.0,
    "H2O": -88680.0,
}


@dataclass(frozen=True)
class XuFromentTerms:
    temperature_k: Expr
    p_ch4_pa: Expr
    p_co_pa: Expr
    p_co2_pa: Expr
    p_h2_pa: Expr
    p_h2o_pa: Expr
    p_inv_h2_pa_inv: Expr
    denominator: Expr
    catalyst_mass_density_kg_per_m3: Expr


def _total_gas_concentration(symbols: CaseSymbols) -> Expr:
    gas_concentrations = tuple(
        concentration
        for species_id, concentration in symbols.concentrations.items()
        if PROPERTY_REGISTRY.get_record(species_id).phase == "gas"
    )

    if not gas_concentrations:
        raise ValueError("Xu-Froment kinetics require at least one gas species.")

    return Add(*gas_concentrations)


def _adsorption_constant_expression(species_id: str, temperature_k: Expr) -> Expr:
    coefficient = XU_FROMENT_ADSORPTION_COEFFICIENTS[species_id]
    adsorption_energy = XU_FROMENT_ADSORPTION_ENERGIES_J_PER_MOL[species_id]
    return coefficient * exp(
        adsorption_energy / (GAS_CONSTANT_J_PER_MOL_K * temperature_k)
    )


def _temperature_polynomial_expression(
    temperature_k, coefficients: tuple[float, float, float, float, float]
) -> Expr:
    temperature_kilo = temperature_k / 1000.0
    c0, c1, c2, c3, c4 = coefficients
    return (
        c0
        + c1 * temperature_kilo
        + c2 * Pow(temperature_kilo, 2)
        + c3 * Pow(temperature_kilo, 3)
        + c4 * Pow(temperature_kilo, 4)
    )


def _eq_const_smr(temperature_k) -> Expr:
    return exp(
        _temperature_polynomial_expression(
            temperature_k,
            (
                -109.009403073377,
                282.095558262226,
                -280.329186487247,
                136.639283940245,
                -26.1188462628166,
            ),
        )
    )


def _eq_const_wgs(temperature_k) -> Expr:
    return exp(
        _temperature_polynomial_expression(
            temperature_k,
            (
                15.3296851277332,
                -30.4525115685455,
                19.943075765819,
                -4.24085991611101,
                -0.232844426047115,
            ),
        )
    )


def _eq_const_overall(temperature_k) -> Expr:
    return exp(
        _temperature_polynomial_expression(
            temperature_k,
            (
                -88.3330905278377,
                229.002065362964,
                -225.776305234964,
                109.686395839527,
                -20.9330826181798,
            ),
        )
    )


def _rate_constant_expression(
    rate_key: str, temperature_k, catalyst_mass_density_kg_per_m3
) -> Expr:
    coefficient = XU_FROMENT_RATE_COEFFICIENTS[rate_key]
    activation_energy = XU_FROMENT_ACTIVATION_ENERGIES_J_PER_MOL[rate_key]
    return (
        catalyst_mass_density_kg_per_m3
        * coefficient
        * exp(-activation_energy / (GAS_CONSTANT_J_PER_MOL_K * temperature_k))
    )


def xu_froment_terms(symbols: CaseSymbols) -> XuFromentTerms:
    temperature_k = symbols.temperature
    pressure_pa = symbols.pressure
    total_gas_concentration = _total_gas_concentration(symbols)

    def partial_pressure(species_id: str) -> Expr:
        return (
            pressure_pa
            * symbols.concentration(species_id)
            / total_gas_concentration
        )

    p_ch4_pa = partial_pressure("CH4")
    p_co_pa = partial_pressure("CO")
    p_co2_pa = partial_pressure("CO2")
    p_h2_pa = partial_pressure("H2")
    p_h2o_pa = partial_pressure("H2O")

    h2_mole_fraction = symbols.concentration("H2") / total_gas_concentration
    controlled_h2_mole_fraction = MIN_H2_MOLE_FRACTION + Rational(1, 2) * (
        h2_mole_fraction
        + sqrt(
            Pow(h2_mole_fraction, 2)
            + H2_MOLE_FRACTION_SMOOTH_EPS_SQUARED
        )
    )
    p_inv_h2_pa_inv = 1 / (pressure_pa * controlled_h2_mole_fraction)
    denominator = (
        1
        + _adsorption_constant_expression("CO", temperature_k) * p_co_pa
        + _adsorption_constant_expression("H2", temperature_k) * p_h2_pa
        + _adsorption_constant_expression("CH4", temperature_k) * p_ch4_pa
        + _adsorption_constant_expression("H2O", temperature_k)
        * p_h2o_pa
        * p_inv_h2_pa_inv
    )

    return XuFromentTerms(
        temperature_k=temperature_k,
        p_ch4_pa=p_ch4_pa,
        p_co_pa=p_co_pa,
        p_co2_pa=p_co2_pa,
        p_h2_pa=p_h2_pa,
        p_h2o_pa=p_h2o_pa,
        p_inv_h2_pa_inv=p_inv_h2_pa_inv,
        denominator=denominator,
        catalyst_mass_density_kg_per_m3=(
            Max(symbols.concentration("Ni"), 0) * NI_MW_KG_PER_MOL
        ),
    )


def _common_rate_factor(
    terms: XuFromentTerms,
    rate_key: str,
    inverse_hydrogen_pressure_exponent: Expr,
    pressure_scale: float,
) -> Expr:
    return (
        _rate_constant_expression(
            rate_key, terms.temperature_k, terms.catalyst_mass_density_kg_per_m3
        )
        * Pow(terms.p_inv_h2_pa_inv, inverse_hydrogen_pressure_exponent)
        / pressure_scale
        / Pow(terms.denominator, 2)
    )


def _smr_fw_rate(terms: XuFromentTerms) -> Expr:
    driving_force = terms.p_ch4_pa * terms.p_h2o_pa
    return (
        _common_rate_factor(terms, "smr", Rational(5, 2), 10.0**-2.5)
        * driving_force
    )


def smr_fw_rate(symbols: CaseSymbols) -> Expr:
    return _smr_fw_rate(xu_froment_terms(symbols))


def _smr_bw_rate(terms: XuFromentTerms) -> Expr:
    driving_force = (
        Pow(terms.p_h2_pa, 3)
        * terms.p_co_pa
        / (1e10 * _eq_const_smr(terms.temperature_k))
    )
    return (
        _common_rate_factor(terms, "smr", Rational(5, 2), 10.0**-2.5)
        * driving_force
    )


def smr_bw_rate(symbols: CaseSymbols) -> Expr:
    return _smr_bw_rate(xu_froment_terms(symbols))


def _wgs_fw_rate(terms: XuFromentTerms) -> Expr:
    driving_force = terms.p_co_pa * terms.p_h2o_pa
    return _common_rate_factor(terms, "wgs", Rational(1), 1.0e5) * driving_force


def wgs_fw_rate(symbols: CaseSymbols) -> Expr:
    return _wgs_fw_rate(xu_froment_terms(symbols))


def _wgs_bw_rate(terms: XuFromentTerms) -> Expr:
    driving_force = (
        terms.p_h2_pa
        * terms.p_co2_pa
        / _eq_const_wgs(terms.temperature_k)
    )
    return _common_rate_factor(terms, "wgs", Rational(1), 1.0e5) * driving_force


def wgs_bw_rate(symbols: CaseSymbols) -> Expr:
    return _wgs_bw_rate(xu_froment_terms(symbols))


def _overall_fw_rate(terms: XuFromentTerms) -> Expr:
    driving_force = terms.p_ch4_pa * Pow(terms.p_h2o_pa, 2)
    return (
        _common_rate_factor(terms, "overall", Rational(7, 2), 10.0**-2.5)
        * driving_force
    )


def overall_fw_rate(symbols: CaseSymbols) -> Expr:
    return _overall_fw_rate(xu_froment_terms(symbols))


def _overall_bw_rate(terms: XuFromentTerms) -> Expr:
    driving_force = (
        Pow(terms.p_h2_pa, 4)
        * terms.p_co2_pa
        / (1e10 * _eq_const_overall(terms.temperature_k))
    )
    return (
        _common_rate_factor(terms, "overall", Rational(7, 2), 10.0**-2.5)
        * driving_force
    )


def overall_bw_rate(symbols: CaseSymbols) -> Expr:
    return _overall_bw_rate(xu_froment_terms(symbols))


def build_family(symbols: CaseSymbols) -> Mapping[str, Reaction]:
    terms = xu_froment_terms(symbols)
    return {
        "wgs_fw": Reaction(
            id="xu_froment.wgs_fw",
            reactants={"CO": 1, "H2O": 1},
            products={"CO2": 1, "H2": 1},
            catalysts=("Ni",),
            rate=_wgs_fw_rate(terms),
        ),
        "smr_fw": Reaction(
            id="xu_froment.smr_fw",
            reactants={"CH4": 1, "H2O": 1},
            products={"CO": 1, "H2": 3},
            catalysts=("Ni",),
            rate=_smr_fw_rate(terms),
        ),
        "overall_fw": Reaction(
            id="xu_froment.overall_fw",
            reactants={"CH4": 1, "H2O": 2},
            products={"CO2": 1, "H2": 4},
            catalysts=("Ni",),
            rate=_overall_fw_rate(terms),
        ),
        "wgs_bw": Reaction(
            id="xu_froment.wgs_bw",
            reactants={"CO2": 1, "H2": 1},
            products={"CO": 1, "H2O": 1},
            catalysts=("Ni",),
            rate=_wgs_bw_rate(terms),
        ),
        "smr_bw": Reaction(
            id="xu_froment.smr_bw",
            reactants={"CO": 1, "H2": 3},
            products={"CH4": 1, "H2O": 1},
            catalysts=("Ni",),
            rate=_smr_bw_rate(terms),
        ),
        "overall_bw": Reaction(
            id="xu_froment.overall_bw",
            reactants={"CO2": 1, "H2": 4},
            products={"CH4": 1, "H2O": 2},
            catalysts=("Ni",),
            rate=_overall_bw_rate(terms),
        ),
    }
