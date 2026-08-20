"""Xu-Froment methane-reforming reaction family."""

from collections.abc import Mapping
from dataclasses import dataclass

import sympy as sp

from ..model import CaseSymbols, Reaction
from ..species.registry import PROPERTY_REGISTRY

GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324
MIN_H2_MOLE_FRACTION = 0.001
H2_MOLE_FRACTION_SMOOTH_EPS_SQUARED = 1.0e-4
NI_MOLAR_MASS_KG_PER_MOL = PROPERTY_REGISTRY.get_record("Ni").molar_mass
XU_FROMENT_RATE_COEFFICIENTS = {"smr": 1.17e15, "wgs": 5.43e5, "overall": 2.81e14}
XU_FROMENT_ACTIVATION_ENERGIES_J_PER_MOL = {
    "smr": 240100.0, "wgs": 67130.0, "overall": 243900.0}
XU_FROMENT_ADSORPTION_COEFFICIENTS = {
    "CO": 8.23e-10, "H2": 6.12e-14, "CH4": 6.65e-9, "H2O": 1.77e5}
XU_FROMENT_ADSORPTION_ENERGIES_J_PER_MOL = {
    "CO": 70650.0, "H2": 82900.0, "CH4": 38280.0, "H2O": -88680.0}
_EQUILIBRIUM = {
    "smr": (-109.009403073377, 282.095558262226, -280.329186487247,
            136.639283940245, -26.1188462628166),
    "wgs": (15.3296851277332, -30.4525115685455, 19.943075765819,
            -4.24085991611101, -0.232844426047115),
    "overall": (-88.3330905278377, 229.002065362964, -225.776305234964,
                109.686395839527, -20.9330826181798),
}


@dataclass(frozen=True)
class XuFromentTerms:
    temperature_k: sp.Expr
    p_ch4_pa: sp.Expr
    p_co_pa: sp.Expr
    p_co2_pa: sp.Expr
    p_h2_pa: sp.Expr
    p_h2o_pa: sp.Expr
    p_inv_h2_pa_inv: sp.Expr
    denominator: sp.Expr
    catalyst_mass_density_kg_per_m3: sp.Expr


def _total_gas_concentration(symbols):
    gases = tuple(value for species_id, value in symbols.concentrations.items()
                   if PROPERTY_REGISTRY.get_record(species_id).phase == "gas")
    if not gases:
        raise ValueError("Xu-Froment kinetics require at least one gas species.")
    return sp.Add(*gases)


def _adsorption(species_id, temperature):
    return XU_FROMENT_ADSORPTION_COEFFICIENTS[species_id] * sp.exp(
        XU_FROMENT_ADSORPTION_ENERGIES_J_PER_MOL[species_id] /
        (GAS_CONSTANT_J_PER_MOL_K * temperature))


def _equilibrium(kind, temperature):
    scaled = temperature / 1000.0
    return sp.exp(sum(coefficient * scaled**power
                      for power, coefficient in enumerate(_EQUILIBRIUM[kind])))


def _eq_const_smr(temperature):
    return _equilibrium("smr", temperature)


def _eq_const_wgs(temperature):
    return _equilibrium("wgs", temperature)


def _eq_const_overall(temperature):
    return _equilibrium("overall", temperature)


def xu_froment_terms(symbols: CaseSymbols) -> XuFromentTerms:
    temperature, pressure = symbols.temperature, symbols.pressure
    total = _total_gas_concentration(symbols)
    partial = lambda species: pressure * symbols.concentration(species) / total
    p_ch4, p_co, p_co2, p_h2, p_h2o = map(
        partial, ("CH4", "CO", "CO2", "H2", "H2O"))
    fraction = symbols.concentration("H2") / total
    controlled = MIN_H2_MOLE_FRACTION + sp.Rational(1, 2) * (
        fraction + sp.sqrt(fraction**2 + H2_MOLE_FRACTION_SMOOTH_EPS_SQUARED))
    inverse_h2 = 1 / (pressure * controlled)
    denominator = (1 + _adsorption("CO", temperature) * p_co
        + _adsorption("H2", temperature) * p_h2
        + _adsorption("CH4", temperature) * p_ch4
        + _adsorption("H2O", temperature) * p_h2o * inverse_h2)
    return XuFromentTerms(temperature, p_ch4, p_co, p_co2, p_h2, p_h2o,
        inverse_h2, denominator,
        sp.Max(symbols.concentration("Ni"), 0) * NI_MOLAR_MASS_KG_PER_MOL)


def _common(terms, kind, inverse_exponent, pressure_scale):
    rate = (terms.catalyst_mass_density_kg_per_m3 *
        XU_FROMENT_RATE_COEFFICIENTS[kind] * sp.exp(
        -XU_FROMENT_ACTIVATION_ENERGIES_J_PER_MOL[kind] /
        (GAS_CONSTANT_J_PER_MOL_K * terms.temperature_k)))
    return rate * terms.p_inv_h2_pa_inv**inverse_exponent / (
        pressure_scale * terms.denominator**2)


def _rate(terms, kind, backward=False):
    if kind == "smr":
        drive = (terms.p_h2_pa**3 * terms.p_co_pa /
                 (1e10 * _equilibrium(kind, terms.temperature_k)) if backward else
                 terms.p_ch4_pa * terms.p_h2o_pa)
        return _common(terms, kind, sp.Rational(5, 2), 10.0**-2.5) * drive
    if kind == "wgs":
        drive = (terms.p_h2_pa * terms.p_co2_pa /
                 _equilibrium(kind, terms.temperature_k) if backward else
                 terms.p_co_pa * terms.p_h2o_pa)
        return _common(terms, kind, sp.S.One, 1.0e5) * drive
    drive = (terms.p_h2_pa**4 * terms.p_co2_pa /
             (1e10 * _equilibrium(kind, terms.temperature_k)) if backward else
             terms.p_ch4_pa * terms.p_h2o_pa**2)
    return _common(terms, kind, sp.Rational(7, 2), 10.0**-2.5) * drive


def _smr_fw_rate(terms): return _rate(terms, "smr")
def _smr_bw_rate(terms): return _rate(terms, "smr", True)
def _wgs_fw_rate(terms): return _rate(terms, "wgs")
def _wgs_bw_rate(terms): return _rate(terms, "wgs", True)
def _overall_fw_rate(terms): return _rate(terms, "overall")
def _overall_bw_rate(terms): return _rate(terms, "overall", True)
def smr_fw_rate(symbols): return _smr_fw_rate(xu_froment_terms(symbols))
def smr_bw_rate(symbols): return _smr_bw_rate(xu_froment_terms(symbols))
def wgs_fw_rate(symbols): return _wgs_fw_rate(xu_froment_terms(symbols))
def wgs_bw_rate(symbols): return _wgs_bw_rate(xu_froment_terms(symbols))
def overall_fw_rate(symbols): return _overall_fw_rate(xu_froment_terms(symbols))
def overall_bw_rate(symbols): return _overall_bw_rate(xu_froment_terms(symbols))


def build_family(symbols: CaseSymbols) -> Mapping[str, Reaction]:
    terms = xu_froment_terms(symbols)
    specs = {
        "wgs_fw": ({"CO": 1, "H2O": 1}, {"CO2": 1, "H2": 1}, _wgs_fw_rate),
        "smr_fw": ({"CH4": 1, "H2O": 1}, {"CO": 1, "H2": 3}, _smr_fw_rate),
        "overall_fw": ({"CH4": 1, "H2O": 2}, {"CO2": 1, "H2": 4}, _overall_fw_rate),
        "wgs_bw": ({"CO2": 1, "H2": 1}, {"CO": 1, "H2O": 1}, _wgs_bw_rate),
        "smr_bw": ({"CO": 1, "H2": 3}, {"CH4": 1, "H2O": 1}, _smr_bw_rate),
        "overall_bw": ({"CO2": 1, "H2": 4}, {"CH4": 1, "H2O": 2}, _overall_bw_rate),
    }
    return {name: Reaction(f"xu_froment.{name}", reactants, products, ("Ni",), rate(terms))
            for name, (reactants, products, rate) in specs.items()}
